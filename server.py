import os
import warnings
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# Silence the legacy warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMUserContextAggregator, 
    LLMAssistantContextAggregator
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs import ElevenLabsTTSService, ElevenLabsHttpTTSService
import aiohttp
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMMessagesFrame,
    OutputAudioRawFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.serializers.base_serializer import FrameSerializer

class FrameLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

load_dotenv()
app = FastAPI()

class RawAudioSerializer(FrameSerializer):
    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, AudioRawFrame):
            return frame.audio
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            return InputAudioRawFrame(audio=data, sample_rate=16000, num_channels=1)
        # Handle text input for debugging
        from pipecat.frames.frames import TranscriptionFrame
        return TranscriptionFrame(text=str(data), user_id="user", timestamp="")

@app.get("/")
async def get():
    return FileResponse("index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async with aiohttp.ClientSession() as session:
        # 1. Transport Params
        params = FastAPIWebsocketParams(
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=RawAudioSerializer()
        )
        
        # 2. THE CRITICAL FIX: Use FastAPIWebsocketTransport
        transport = FastAPIWebsocketTransport(websocket, params=params)

        # 3. Services
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        llm = OpenAILLMService(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            settings=OpenAILLMService.Settings(model="llama-3.1-8b-instant")
        )
        tts = ElevenLabsHttpTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="cgSgspJ2msm6clMCkdW9", # Jessica
            aiohttp_session=session
        )

        # 4. Context & Aggregators
        context = OpenAILLMContext([{"role": "system", "content": "You are a concise AI SDR. Keep responses under 2 sentences."}])
        user_aggregator = LLMUserContextAggregator(context)
        assistant_aggregator = LLMAssistantContextAggregator(context)

        # 5. Pipeline
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
            # Kick-off the greeting when a client connects
            print(f"!!! DRIVING GREETING: {client} !!!")
            content = [{"role": "user", "content": "Hello! Please introduce yourself as Jessica, the AI SDR, and ask how you can help."}]
            await task.queue_frames([LLMMessagesFrame(content)])

        runner = PipelineRunner()
        
        print("🚀 LeadGen Engine Brain Active. Waiting for audio...")
        await runner.run(task)