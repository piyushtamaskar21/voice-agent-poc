"""
Voice Agent POC — Bulletproof Architecture
==========================================
Instead of fighting Pipecat's LocalAudioTransport (which has broken sample
rate handling in 0.0.106), we own the audio layer completely:

  - PyAudio input thread  → feeds raw PCM into an asyncio queue
  - Silero VAD            → detects speech start/stop on our terms  
  - Deepgram STT REST     → transcribes completed utterances
  - Groq LLM              → generates response
  - Deepgram TTS REST     → synthesizes speech as raw PCM
  - PyAudio output        → plays at exactly the right rate/channels

No Pipecat transport. No sample rate guessing. No chipmunk.
"""

import asyncio
import os
import queue
import struct
import threading
import warnings
warnings.filterwarnings("ignore")

import pyaudio
import requests
import torch
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------------------
# Hardware constants — confirmed via PyAudio probe
# ---------------------------------------------------------------------------
INPUT_DEVICE   = 0       # AirPods Pro mic
OUTPUT_DEVICE  = 1       # AirPods Pro speakers
MIC_RATE       = 16000   # Silero VAD requires 8000 or 16000
MIC_CHANNELS   = 1       # mono mic
OUT_RATE       = 48000   # AirPods output native
OUT_CHANNELS   = 2       # stereo output
CHUNK          = 512     # ~32ms at 16kHz — good VAD granularity

# ---------------------------------------------------------------------------
# Silero VAD — load once
# ---------------------------------------------------------------------------
print("Loading Silero VAD...")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
)
(get_speech_timestamps, _, read_audio, *_) = vad_utils
print("VAD loaded.")

# ---------------------------------------------------------------------------
# Deepgram STT — simple REST call
# ---------------------------------------------------------------------------
def transcribe(pcm_bytes: bytes) -> str:
    r = requests.post(
        f"https://api.deepgram.com/v1/listen?model=nova-2&language=en&encoding=linear16&sample_rate={MIC_RATE}&channels={MIC_CHANNELS}",
        headers={
            "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}",
            "Content-Type": "audio/raw",
        },
        data=pcm_bytes,
        timeout=10,
    )
    result = r.json()
    try:
        return result["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Groq LLM
# ---------------------------------------------------------------------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
conversation = [
    {
        "role": "system",
        "content": (
            "You are an AI sales development representative for LeadGen Engine. "
            "Keep your responses extremely concise, conversational, and under 2 sentences. "
            "Do not act like a typical AI. Speak like a human on a phone call."
        ),
    }
]

def get_llm_response(user_text: str) -> str:
    conversation.append({"role": "user", "content": user_text})
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=conversation,
    )
    reply = resp.choices[0].message.content.strip()
    conversation.append({"role": "assistant", "content": reply})
    return reply

# ---------------------------------------------------------------------------
# Deepgram TTS — REST, returns raw PCM at OUT_RATE mono, we upsample to stereo
# ---------------------------------------------------------------------------
def synthesize(text: str) -> bytes:
    r = requests.post(
        f"https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate={OUT_RATE}",
        headers={
            "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={"text": text},
        timeout=15,
    )
    mono_pcm = r.content
    # Convert mono → stereo by duplicating each sample
    samples = struct.unpack(f"<{len(mono_pcm)//2}h", mono_pcm)
    stereo = struct.pack(f"<{len(samples)*2}h", *[s for sample in samples for s in (sample, sample)])
    return stereo

# ---------------------------------------------------------------------------
# PyAudio output — plays stereo PCM at OUT_RATE
# ---------------------------------------------------------------------------
_pa = pyaudio.PyAudio()
_out_stream = _pa.open(
    format=pyaudio.paInt16,
    channels=OUT_CHANNELS,
    rate=OUT_RATE,
    output=True,
    output_device_index=OUTPUT_DEVICE,
    frames_per_buffer=2048,
)

def play_audio(stereo_pcm: bytes):
    _out_stream.write(stereo_pcm)

# ---------------------------------------------------------------------------
# VAD state machine
# ---------------------------------------------------------------------------
SILENCE_CHUNKS_NEEDED = int(0.8 * MIC_RATE / CHUNK)  # 800ms of silence = end of speech

def is_speech(chunk_bytes: bytes) -> bool:
    samples = struct.unpack(f"<{len(chunk_bytes)//2}h", chunk_bytes)
    tensor = torch.FloatTensor(samples) / 32768.0
    confidence = vad_model(tensor, MIC_RATE).item()
    return confidence > 0.5

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def main():
    print("\nReady! Speak into your AirPods mic. Press Ctrl+C to exit.\n")

    audio_queue: queue.Queue = queue.Queue()
    bot_speaking = threading.Event()

    # --- Mic capture thread ---
    def mic_thread():
        stream = _pa.open(
            format=pyaudio.paInt16,
            channels=MIC_CHANNELS,
            rate=MIC_RATE,
            input=True,
            input_device_index=INPUT_DEVICE,
            frames_per_buffer=CHUNK,
        )
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if not bot_speaking.is_set():
                audio_queue.put(data)

    t = threading.Thread(target=mic_thread, daemon=True)
    t.start()

    # --- Main VAD + pipeline loop ---
    speech_buffer = []
    silence_count = 0
    in_speech = False

    while True:
        try:
            chunk = audio_queue.get(timeout=0.05)
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue

        speaking = is_speech(chunk)

        if speaking:
            if not in_speech:
                print("[VAD] Speech detected")
                in_speech = True
            speech_buffer.append(chunk)
            silence_count = 0

        elif in_speech:
            speech_buffer.append(chunk)
            silence_count += 1

            if silence_count >= SILENCE_CHUNKS_NEEDED:
                print("[VAD] Speech ended — processing...")
                in_speech = False
                silence_count = 0

                pcm = b"".join(speech_buffer)
                speech_buffer = []

                # Run STT → LLM → TTS in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()

                transcript = await loop.run_in_executor(None, transcribe, pcm)
                if not transcript:
                    print("[STT] No transcript, skipping")
                    continue

                print(f"[YOU] {transcript}")

                reply = await loop.run_in_executor(None, get_llm_response, transcript)
                print(f"[BOT] {reply}")

                audio = await loop.run_in_executor(None, synthesize, reply)

                bot_speaking.set()     # mute mic while speaking
                await loop.run_in_executor(None, play_audio, audio)
                bot_speaking.clear()   # re-open mic

                print()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        _out_stream.close()
        _pa.terminate()
