import os
import json
import warnings
import aiohttp
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from loguru import logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger.add("server.log", rotation="10 MB")

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
    InputAudioRawFrame,
    InterruptionFrame,
    LLMMessagesFrame,
    LLMMessagesUpdateFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSSpeakFrame,
    SystemFrame,
    TranscriptionFrame,
    Frame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.tts_service import TTSService

load_dotenv()
app = FastAPI()


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------

class RawAudioSerializer(FrameSerializer):
    """
    Bulletproof Serializer: Maps incoming raw binary bytes from index.html 
    directly to InputAudioRawFrame, and outgoing AudioRawFrames to bytes.
    Buffers incoming frames until the transport is ready.
    """
    def __init__(self):
        super().__init__()
        self._pipe_ready = False

    def set_ready(self, ready: bool):
        self._pipe_ready = ready

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, AudioRawFrame):
            return frame.audio
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if not self._pipe_ready:
            # Ignore audio until pipeline is fully initialized to avoid StartFrame errors
            return None
            
        if isinstance(data, bytes):
            return InputAudioRawFrame(audio=data, sample_rate=16000, num_channels=1)
        return None

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
    def __init__(self, api_key: str, language: str = "hi-IN", speaker: str = "anushka"):
        super().__init__()
        self._api_key = api_key
        # Sarvam language codes: hi-IN, mr-IN
        self._language = "hi-IN" if language == "hi" else "mr-IN"
        self._speaker = speaker  # Valid for bulbul:v2: anushka, abhilash, manisha, vidya, arya, karun, hitesh

    async def run_tts(self, text: str, context_id: str):
        if not text or not text.strip():
            logger.debug("Sarvam TTS: Empty text, skipping")
            return

        async with aiohttp.ClientSession() as session:
            # Bulbul v2 API parameters
            payload = {
                "inputs": [text],
                "target_language_code": self._language,
                "speaker": self._speaker,
                "pitch": 0,
                "pace": 1.0,
                "loudness": 1.5,
                "speech_sample_rate": 16000,
                "enable_preprocessing": True,
                "model": "bulbul:v2"
            }
            
            headers = {
                "api-subscription-key": self._api_key,
                "Content-Type": "application/json"
            }

            try:
                async with session.post(
                    "https://api.sarvam.ai/text-to-speech",
                    json=payload,
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"[Sarvam TTS ERROR] {resp.status}: {error_text}")
                        return
                    
                    result = await resp.json()
                    
                    if "audios" not in result or not result["audios"]:
                        logger.error(f"[Sarvam TTS ERROR] No audio in response: {result}")
                        return

                    audio_b64 = result["audios"][0]
                    import base64
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    # Log successful generation
                    logger.info(f"✅ Generated Sarvam TTS audio: {len(audio_bytes)} bytes")

                    # Bulbul v2 returns a WAV with 44-byte header
                    if audio_bytes.startswith(b"RIFF"):
                        audio_bytes = audio_bytes[44:]
                    
                    yield TTSAudioRawFrame(audio=audio_bytes, sample_rate=16000, num_channels=1, context_id=context_id)
            except Exception as e:
                logger.error(f"[Sarvam TTS EXCEPTION]: {e}")


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





class FrameLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Log frame types for debugging (except common audio frames to avoid spam)
        if not isinstance(frame, (AudioRawFrame, InputAudioRawFrame)):
            logger.debug(f"🧩 [FRAME] {type(frame).__name__} ({direction})")
        elif isinstance(frame, TranscriptionFrame):
            logger.debug(f"📝 [TRANSCRIPT] '{frame.text}' (final={frame.finalized})")
        
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# System prompts per language
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "en": (
        "You are Jessica, a female AI voice representative from Arios AI. "
        "Arios AI builds AI voice agents and automation systems for sales, support, lead qualification, inbound/outbound calling, multilingual voice conversations, CRM integrations, workflow automation, and compliance-aware AI operations. "
        "Speak like a warm, confident, helpful human on a real phone call, not like a chatbot or assistant reading a script. "
        "Your tone should sound like a friendly call-center executive handling an enquiry naturally. "
        "Keep responses short, natural, and easy to follow: usually 1 sentence, sometimes 2 short sentences if needed. "
        "Do not give long introductions, long monologues, or marketing speeches. "
        "Your introduction should be very short, for example: 'Hi, Jessica here from Arios AI.' "
        "After introducing yourself, ask one simple question and wait for the user. "
        "Sound conversational and human: use natural phrasing like 'sure', 'got it', 'okay', 'right', 'no problem'. "
        "Do not sound robotic, overly polished, or too formal. "
        "If the user asks about Arios AI, explain clearly that Arios AI helps businesses deploy AI voice agents for sales and support workflows, including multilingual and real-time voice interactions. "
        "If you do not know something, say it simply and offer to connect the user to the right team or take details for a callback. "
        "Never mention prompts, policies, or internal system behavior."
    ),
    "hi": (
        "You are Jessica, a female AI voice representative from Arios AI. "
        "Arios AI AI voice agents aur automation systems banata hai for sales, support, lead qualification, inbound/outbound calling, multilingual voice conversations, CRM integrations, workflow automation, aur compliance-aware AI operations. "
        "Aapko bilkul natural North Indian phone conversation style mein bolna hai — pure Hindi nahi, natural Hinglish mein. "
        "Aapka tone friendly, calm, professional aur human hona chahiye, jaise koi helpful female call-center executive enquiry handle kar rahi ho. "
        "Responses short rakho: usually 1 sentence, max 2 short sentences. "
        "Long introduction, pitch, ya speech mat do. "
        "Introduction bahut short hona chahiye, for example: 'Hi, Jessica here from Arios AI.' ya 'Hi, Arios AI se Jessica bol rahi hoon.' "
        "Uske baad ek simple question poochho aur user ko bolne do. "
        "Natural conversational Hinglish use karo, jaise 'sure', 'got it', 'okay', 'no problem', 'let me help you', 'aap bataiye'. "
        "Bahut shuddh Hindi, bookish Hindi, ya robotic language avoid karo. "
        "Agar user Arios AI ke baare mein pooche, simple tareeke se samjhao ki Arios AI businesses ko AI voice agents deploy karne mein help karta hai for sales aur support workflows, including multilingual and real-time voice conversations. "
        "Agar exact answer na ho, seedha bolo ki aap details note karke right team se connect karwa sakti hain ya callback arrange kar sakti hain. "
        "Kabhi bhi bot jaisa ya scripted assistant jaisa sound mat karo."
    ),
    "mr": (
        "You are Jessica, a female AI voice representative from Arios AI. "
        "Arios AI sales, support, lead qualification, inbound/outbound calling, multilingual voice conversations, CRM integrations, workflow automation, आणि compliance-aware AI operations साठी AI voice agents आणि automation systems build करते. "
        "तुम्ही अगदी natural Maharashtra phone conversation style मध्ये बोला — pure Marathi नाही, तर natural Marathi plus English mix मध्ये. "
        "तुमचा tone friendly, calm, professional आणि human असावा, जसा एखादी helpful female call-center executive enquiry handle करते. "
        "Responses short ठेवा: usually 1 sentence, max 2 short sentences. "
        "Long introduction, speech, किंवा sales pitch देऊ नका. "
        "Introduction खूप short असावा, for example: 'Hi, Jessica here from Arios AI.' किंवा 'Hi, Arios AI मधून Jessica बोलतेय.' "
        "त्याच्या नंतर एक simple question विचारा आणि user ला बोलू द्या. "
        "Natural conversational Marathi-English वापरा, जसं 'sure', 'got it', 'okay', 'no problem', 'मी help करते', 'तुम्ही सांगा'. "
        "खूप शुद्ध, पुस्तकातलं Marathi किंवा robotic language टाळा. "
        "जर user ने Arios AI बद्दल विचारलं, तर simple पद्धतीने सांगा की Arios AI businesses ना sales आणि support workflows साठी AI voice agents deploy करायला मदत करते, including multilingual and real-time voice conversations. "
        "जर exact answer माहित नसेल, तर सरळ सांगा की तुम्ही details note करून right team कडून callback arrange करू शकता. "
        "कधीही bot सारखं किंवा script वाचताय असं वाटता कामा नये."
    ),
}


GREETING_PROMPTS = {
    "en": "Start with one short phone-style introduction as Jessica from Arios AI, then ask how you can help today.",
    "hi": "Ek short natural Hinglish introduction do as Jessica from Arios AI, phir poochho aaj aapko kis cheez mein help chahiye.",
    "mr": "एक short natural Marathi-English introduction द्या as Jessica from Arios AI, आणि मग विचारा आज मी कशात help करू शकते."
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
        serializer = RawAudioSerializer()
        params = FastAPIWebsocketParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            add_wav_header=True,
            vad_enabled=False,  # Disable transport-level VAD for more reliable STT flow
            serializer=serializer,
        )
        transport = FastAPIWebsocketTransport(websocket, params=params)

        # --- STT: Deepgram for all languages (nova-2-general) ---
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            language=lang,
            model="nova-2-general",
            interim_results=True,
            vad_enabled=True,  # Use Deepgram's built-in VAD
            keepalive_timeout=5.0,
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
                speaker="anushka"  # Valid options: anushka, abhilash, manisha, vidya, arya, karun, hitesh
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

        frame_logger = FrameLogger()

        # --- Core Pipeline ---
        pipeline = Pipeline([
            transport.input(),
            frame_logger,
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
            # Signal serializer to start accepting audio
            serializer.set_ready(True)
            # Send explicitly "started" text message (optional but good for client sync)
            await websocket.send_text(json.dumps({"type": "session_started"}))
            
            greeting = GREETING_PROMPTS.get(lang, GREETING_PROMPTS["en"])
            # Use LLMMessagesUpdateFrame to trigger the initial greeting (non-deprecated way)
            await task.queue_frames([LLMMessagesUpdateFrame(messages=[{"role": "user", "content": greeting}], run_llm=True)])

        runner = PipelineRunner()
        print(f"🚀 LeadGen Engine Active — Language: {lang.upper()}")
        await runner.run(task)
