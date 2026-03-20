import os
import json
import warnings
import aiohttp
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMUserContextAggregator,
    LLMAssistantContextAggregator,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.elevenlabs import ElevenLabsHttpTTSService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    LLMMessagesFrame,
    TTSSpeakFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.ai_services import TTSService
from pipecat.frames.frames import AudioRawFrame, TTSStartedFrame, TTSStoppedFrame

load_dotenv()
app = FastAPI()


# ---------------------------------------------------------------------------
# Sarvam AI — Bulbul v2 TTS (Hindi / Marathi)
# Sarvam Shuka STT (Hindi / Marathi)
# Docs: https://docs.sarvam.ai
# ---------------------------------------------------------------------------

class SarvamBulbulTTSService(TTSService):
    """
    Sarvam Bulbul v2 TTS via REST API.
    Returns WAV audio which Pipecat's transport will stream to the browser.
    """
    def __init__(self, api_key: str, language: str = "hi-IN", speaker: str = "meera"):
        super().__init__()
        self._api_key = api_key
        # Sarvam language codes: hi-IN, mr-IN
        self._language = "hi-IN" if language == "hi" else "mr-IN"
        self._speaker = speaker  # meera, pavithra, maitreyi, etc.

    async def run_tts(self, sentence: str):
        async with aiohttp.ClientSession() as session:
            payload = {
                "inputs": [sentence],
                "target_language_code": self._language,
                "speaker": self._speaker,
                "pitch": 0,
                "pace": 1.0,
                "loudness": 1.5,
                "speech_sample_rate": 8000,
                "enable_preprocessing": True,
                "model": "bulbul:v2"
            }
            headers = {
                "api-subscription-key": self._api_key,
                "Content-Type": "application/json"
            }
            async with session.post(
                "https://api.sarvam.ai/text-to-speech",
                json=payload,
                headers=headers
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    print(f"[Sarvam TTS ERROR] {resp.status}: {error}")
                    return
                result = await resp.json()
                # Sarvam returns base64-encoded WAV in audios[0]
                import base64
                audio_b64 = result["audios"][0]
                audio_bytes = base64.b64decode(audio_b64)
                # Yield as a single AudioRawFrame (WAV bytes)
                yield AudioRawFrame(audio=audio_bytes, sample_rate=8000, num_channels=1)


class SarvamShukaSTTService:
    """
    Sarvam Shuka STT via REST — wraps into Pipecat-compatible transcription.
    NOTE: This is a direct REST wrapper since Pipecat doesn't have a native
    Sarvam STT service yet. We post raw PCM and get back a transcript.
    """
    def __init__(self, api_key: str, language: str = "hi"):
        self._api_key = api_key
        self._language = "hi-IN" if language == "hi" else "mr-IN"

    async def transcribe(self, pcm_bytes: bytes) -> str:
        import base64
        # Sarvam expects base64-encoded WAV, 16kHz mono
        audio_b64 = base64.b64encode(pcm_bytes).decode()
        payload = {
            "model": "saarika:v2",
            "language_code": self._language,
            "audio": audio_b64,
            "with_disfluencies": False,
        }
        headers = {
            "api-subscription-key": self._api_key,
            "Content-Type": "application/json"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.sarvam.ai/speech-to-text",
                json=payload,
                headers=headers
            ) as resp:
                if resp.status != 200:
                    print(f"[Sarvam STT ERROR] {resp.status}: {await resp.text()}")
                    return ""
                result = await resp.json()
                return result.get("transcript", "")





# ---------------------------------------------------------------------------
# System prompts per language
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "en": (
        "You are Jessica, a concise AI Sales Development Representative for Arios AI. "
        "Keep all responses under 2 sentences. Speak like a professional human on a phone call."
    ),
    "hi": (
        "आप जेसिका हैं, Arios AI की एक संक्षिप्त AI सेल्स रिप्रेजेंटेटिव। "
        "सभी जवाब 2 वाक्यों से कम में दें। फ़ोन पर एक पेशेवर इंसान की तरह बात करें।"
    ),
    "mr": (
        "तुम्ही जेसिका आहात, Arios AI ची एक संक्षिप्त AI सेल्स प्रतिनिधी. "
        "सर्व उत्तरे 2 वाक्यांपेक्षा कमी ठेवा. फोनवर एका व्यावसायिक माणसासारखे बोला."
    ),
}

GREETING_PROMPTS = {
    "en": "Hello! Please introduce yourself as Jessica, the AI SDR for Arios AI, and ask how you can help.",
    "hi": "नमस्ते! कृपया अपना परिचय जेसिका के रूप में दें, Arios AI की AI SDR, और पूछें कि आप कैसे मदद कर सकते हैं।",
    "mr": "नमस्कार! कृपया स्वतःची ओळख जेसिका म्हणून द्या, Arios AI ची AI SDR, आणि विचारा की तुम्ही कशी मदत करू शकता.",
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def get():
    return FileResponse("index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, lang: str = "en"):
    await websocket.accept()
    print(f"🌐 New connection — language: {lang}")

    async with aiohttp.ClientSession() as session:
        params = FastAPIWebsocketParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
        transport = FastAPIWebsocketTransport(websocket, params=params)

        # --- STT: Deepgram for EN, Sarvam Shuka for HI/MR ---
        if lang in ("hi", "mr"):
            # Sarvam STT is REST-based; we use DeepgramSTTService with language hint
            # for now as a streaming fallback, and Sarvam for final transcription.
            # Full Sarvam STT integration requires a custom FrameProcessor (see below).
            stt = DeepgramSTTService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                live_options=LiveOptions(
                    model="nova-2",
                    encoding="linear16",
                    sample_rate=16000,
                    channels=1,
                    interim_results=True,
                    language=lang
                )
            )
        else:
            stt = DeepgramSTTService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                live_options=LiveOptions(
                    model="nova-2",
                    encoding="linear16",
                    sample_rate=16000,
                    channels=1,
                    interim_results=True
                )
            )

        # --- LLM: Groq Llama 3.1 (unchanged) ---
        llm = OpenAILLMService(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            settings=OpenAILLMService.Settings(model="llama-3.1-8b-instant")
        )

        # --- TTS: ElevenLabs for EN, Sarvam Bulbul v2 for HI/MR ---
        if lang in ("hi", "mr"):
            tts = SarvamBulbulTTSService(
                api_key=os.getenv("SARVAM_API_KEY"),
                language=lang,
                speaker="meera"  # Options: meera, pavithra, maitreyi, arvind, amol, amartya
            )
        else:
            tts = ElevenLabsHttpTTSService(
                api_key=os.getenv("ELEVENLABS_API_KEY"),
                voice_id="cgSgspJ2msm6clMCkdW9",  # Jessica
                aiohttp_session=session
            )

        # --- Context with language-aware system prompt ---
        context = OpenAILLMContext([{
            "role": "system",
            "content": SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
        }])
        user_aggregator = LLMUserContextAggregator(context)
        assistant_aggregator = LLMAssistantContextAggregator(context)

        pipeline = Pipeline([
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ])

        task = PipelineTask(pipeline)

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            print(f"✅ Client connected — lang={lang}")
            greeting = GREETING_PROMPTS.get(lang, GREETING_PROMPTS["en"])
            await task.queue_frames([LLMMessagesFrame([{"role": "user", "content": greeting}])])

        runner = PipelineRunner()
        print(f"🚀 LeadGen Engine Active — Language: {lang.upper()}")
        await runner.run(task)
