import asyncio
import os
import sys
import warnings
from dotenv import load_dotenv

# Suppress warnings for a clean console
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv()

async def main():
    print("Initializing LeadGen WebRTC Pipeline...")

    room_url = os.getenv("DAILY_ROOM_URL")
    if not room_url:
        print("ERROR: Please set DAILY_ROOM_URL in your .env file")
        sys.exit(1)

    # 1. WebRTC Transport (Bypasses Mac Audio hardware entirely)
    transport = DailyTransport(
        room_url=room_url,
        token="",
        bot_name="Rachel (AI SDR)",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        )
    )

    # 2. API Services
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    
    llm = OpenAILLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        settings=OpenAILLMService.Settings(model="llama-3.1-8b-instant")
    )
    
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        settings=ElevenLabsTTSService.Settings(voice="21m00Tcm4TlvDq8ikWAM")
    )

    # 3. Context
    system_prompt = (
        "You are an AI sales development representative for LeadGen Engine. "
        "Keep your responses extremely concise, conversational, and under 2 sentences. "
        "Do not act like a typical AI. Speak like a human on a phone call."
    )
    context = OpenAILLMContext([{"role": "system", "content": system_prompt}])
    context_aggregator = llm.create_context_aggregator(context)

    # 4. Construct Pipeline
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    # Graceful start and stop
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        print(f"\n✅ You connected to the room! Start talking.")

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        print(f"\nUser left the room. Shutting down pipeline.")
        await task.queue_frames([EndFrame()])

    print(f"\n🌐 Waiting for you to join the WebRTC Room: {room_url}")
    
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())