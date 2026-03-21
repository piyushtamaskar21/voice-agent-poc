#!/usr/bin/env python3
"""
daily_agent.py — Arios AI Daily.co Voice Agent
Hybrid STT: Deepgram (EN/HI) + Sarvam (MR)
Usage:
    python3 daily_agent.py               # English
    python3 daily_agent.py --lang hi     # Hindi
    python3 daily_agent.py --lang mr     # Marathi
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import struct
import subprocess
import threading
import time
import wave

import httpx
import websockets
from daily import CallClient, Daily, EventHandler
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("daily-agent")

# ── Config ────────────────────────────────────────────────────────────────────
DAILY_ROOM_URL   = os.getenv("DAILY_ROOM_URL", "https://ariosai.daily.co/ariosai")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
SARVAM_API_KEY   = os.getenv("SARVAM_API_KEY", "")

INPUT_SAMPLE_RATE  = 16000
OUTPUT_SAMPLE_RATE = 16000
CHANNELS           = 1

# STT tuning
VAD_RMS_THRESHOLD  = 100   # chunks below this RMS are silence
MIN_SPEECH_SECS    = 1.5   # minimum buffered speech before flush
MAX_BUFFER_SECS    = 6.0   # force flush after this long
SILENCE_SECS       = 0.8   # flush after this much silence post-speech

SYSTEM_PROMPTS = {
    "en": (
        "You are Jessica, a female AI SDR and receptionist from Arios AI. "
        "Speak like a real human on a phone call — warm, confident, polite, slightly enthusiastic, and patient. "
        "Never sound robotic or scripted. "

        "Keep responses short and natural: usually 1 sentence, max 2 short sentences. "
        "Do not give long explanations unless the user explicitly asks. "

        "Conversation style rules: "
        "Introduce yourself briefly only once at the beginning. "
        "Always ask one simple, clear question and then wait. "
        "Acknowledge naturally like 'Got it', 'Makes sense', 'Understood'. "
        "If the user is confused, simplify — don’t over-explain. "
        "Be patient, calm, and never rush. "

        "Your role: "
        "Handle customer queries like a trained support and sales executive. "
        "Understand user needs, qualify them, and guide toward a demo or next step. "
        "Collect basic details if needed. "
        "If something is complex, offer to connect with a human expert. "

        "About Arios AI: "
        "We build enterprise AI systems end-to-end. "
        "Core areas include LLM apps, RAG systems, AI agents, and workflow automation. "
        "We build voice agents, support bots, sales copilots, and knowledge assistants. "
        "Focus is always ROI — reducing cost, improving efficiency, automating work. "

        "Important: "
        "Do not assume or hallucinate. "
        "If unsure, say you will confirm with the team. "

        "Goal: "
        "Keep the conversation natural, helpful, and move toward understanding the user need and next step."
    ),

    "hi": (
        "You are Jessica, a female AI SDR and receptionist from Arios AI. "
        "Natural Hinglish mein baat karo — jaise real India phone call hoti hai. "
        "Tone warm, polite, thodi enthusiastic aur patient hona chahiye. "
        "Robot jaisa bilkul nahi lagna chahiye. "

        "Responses short rakho: usually 1 sentence, max 2 short sentences. "
        "Long explanation mat do jab tak user specifically na puche. "

        "Conversation style: "
        "Sirf start mein short intro do. "
        "Har response ke baad ek simple question pucho aur ruk jao. "
        "Natural fillers use karo jaise 'Got it', 'samajh gayi', 'makes sense'. "
        "User confused ho toh simple explain karo, over-explain mat karo. "
        "Calm aur patient raho. "

        "Tumhara role: "
        "Customer queries handle karna like a professional call center executive. "
        "User ka need samajhna aur qualify karna. "
        "Demo ya next step ke liye guide karna. "
        "Basic details politely collect kar sakti ho. "

        "Arios AI kya karta hai: "
        "AI systems banata hai jo business ka manual kaam automate karte hain. "
        "LLM apps, RAG systems, AI agents, automation workflows. "
        "Voice AI agents, support bots, sales copilots, knowledge assistants. "
        "Focus ROI pe hota hai — cost kam, efficiency zyada. "

        "Important: "
        "Kuch bhi assume mat karo. "
        "Agar unsure ho toh bolo ki main team se confirm kar leti hoon. "

        "Goal: "
        "Conversation natural, helpful aur smooth rakhna aur user ko next step tak le jaana."
    ),

    "mr": (
        "You are Jessica, a female AI SDR and receptionist from Arios AI. "
        "Marathi + English mix madhe natural bola — jaise Maharashtra madhe real phone conversation hote. "
        "Tone warm, polite, thodi enthusiastic ani patient hava. "
        "Robot sarkha bilkul vatayla nako. "

        "Responses short theva: usually 1 sentence, max 2 short sentences. "
        "User ne specifically vicharla tarach detail madhe jaa. "

        "Conversation style: "
        "Suruvatila ek short intro dya. "
        "Pratyek response nantar ek simple prashna vichara ani thamba. "
        "Natural phrases use kara jaise 'Got it', 'samajhla', 'makes sense'. "
        "User confused asel tar simple explain kara, over-explain nako. "
        "Calm ani patient raha. "

        "Tumcha role: "
        "Customer queries handle karne like professional call center executive. "
        "User cha requirement samjun gheu shakte ani qualify karu shakte. "
        "Demo kiwa next step sathi guide karu shakte. "
        "Basic details politely collect karu shakte. "

        "Arios AI kay karto: "
        "AI systems banavte je business madhla manual kaam automate kartat. "
        "LLM apps, RAG systems, AI agents, workflow automation. "
        "Voice AI agents, support bots, sales copilots, knowledge assistants. "
        "Focus ROI var — cost kami, efficiency jast. "

        "Important: "
        "Kahi hi assume karu naka. "
        "Jar unsure asal tar sanga ki mi team sobat confirm karte. "

        "Goal: "
        "Conversation natural, helpful ani smooth thevne ani user la next step kade gheun jane."
    ),
}

GREETINGS = {
    "en": "Hi, Jessica here from Arios AI. How can I help you today?",

    "hi": "Hi, Arios AI se Jessica bol rahi hoon. Aaj main aapki kaise help kar sakti hoon?",

    "mr": "Namaste! Mi Jessica, Arios AI madhun bolte. Aaj mi tumhala kashi madad karu shakte?",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def mp3_to_pcm16k(mp3_bytes: bytes) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "mp3", "-i", "pipe:0",
         "-f", "s16le", "-ar", str(OUTPUT_SAMPLE_RATE), "-ac", "1", "pipe:1"],
        input=mp3_bytes, capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg: {result.stderr.decode()[:300]}")
    return result.stdout


def wav_to_pcm(wav_bytes: bytes) -> bytes:
    return wav_bytes[44:]


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = INPUT_SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def get_rms(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    samples = struct.unpack(f"{len(pcm)//2}h", pcm)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


def is_hallucination(transcript: str) -> bool:
    """Detect repeated word hallucinations like 'बरं बरं बरं बरं...'"""
    words = transcript.split()
    if len(words) < 6:
        return False
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < 0.25


# ── LLM ───────────────────────────────────────────────────────────────────────

class LLM:
    def __init__(self, lang: str):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.system_prompt = SYSTEM_PROMPTS[lang]
        self.history = []

    def generate(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": self.system_prompt}] + self.history[-12:]
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=messages,
            temperature=0.3, max_tokens=90,
        )
        reply = (resp.choices[0].message.content or "").strip()
        self.history.append({"role": "assistant", "content": reply})
        logger.info(f"[LLM] {reply}")
        return reply


# ── TTS ───────────────────────────────────────────────────────────────────────

class TTS:
    def __init__(self, lang: str):
        self.lang = lang
        self.http = httpx.Client(timeout=30)

    def synthesize(self, text: str) -> bytes:
        return self._openai_tts(text) if self.lang == "en" else self._sarvam_tts(text)

    def _openai_tts(self, text: str) -> bytes:
        logger.info(f"[TTS-EN] {text[:80]}")
        resp = self.http.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": "tts-1", "input": text,
                  "voice": "nova", "response_format": "mp3"},
        )
        resp.raise_for_status()
        return mp3_to_pcm16k(resp.content)

    def _sarvam_tts(self, text: str) -> bytes:
        lang_code = "hi-IN" if self.lang == "hi" else "mr-IN"
        logger.info(f"[TTS-{self.lang.upper()}] {text[:80]}")
        resp = self.http.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": SARVAM_API_KEY,
                     "content-type": "application/json"},
            json={
                "target_language_code": lang_code,
                "speaker": "ritu",
                "pace": 1.0,
                "speech_sample_rate": 16000,
                "enable_preprocessing": True,
                "model": "bulbul:v3",   # ← v3: no pitch/loudness params!
                "inputs": [text],
            },
        )
        if not resp.is_success:
            logger.error(f"[TTS-Sarvam] {resp.status_code}: {resp.text}")
            raise RuntimeError(f"Sarvam TTS failed: {resp.status_code}")
        audios = resp.json().get("audios") or []
        if not audios:
            raise RuntimeError("Sarvam TTS returned no audio")
        return wav_to_pcm(base64.b64decode(audios[0]))

    def close(self):
        self.http.close()


# ── STT: Deepgram (EN + HI streaming) ────────────────────────────────────────

class DeepgramSTT:
    def __init__(self, lang: str, on_final_callback):
        self.lang = lang
        self.on_final_callback = on_final_callback
        self.ws = None
        self._recv_task = None
        self._connected = False

    def _build_url(self):
        lang_code = "hi" if self.lang == "hi" else "en"
        return (
            "wss://api.deepgram.com/v1/listen"
            "?model=nova-2"
            "&encoding=linear16"
            f"&sample_rate={INPUT_SAMPLE_RATE}"
            "&channels=1"
            "&interim_results=false"
            "&punctuate=true"
            "&smart_format=true"
            "&endpointing=400"
            f"&language={lang_code}"
        )

    async def connect(self):
        self.ws = await websockets.connect(
            self._build_url(),
            additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            max_size=None, ping_interval=20, ping_timeout=20,
        )
        self._connected = True
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info(f"[STT-Deepgram] Connected — lang={self.lang}")

    async def _recv_loop(self):
        try:
            async for msg in self.ws:
                if isinstance(msg, bytes):
                    continue
                data = json.loads(msg)
                if data.get("type") != "Results":
                    continue
                alts = data.get("channel", {}).get("alternatives", [])
                if not alts:
                    continue
                text = (alts[0].get("transcript") or "").strip()
                if text and data.get("is_final"):
                    logger.info(f"[STT-Deepgram] Final: {text}")
                    await self.on_final_callback(text)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"[STT-Deepgram] Disconnected: {e}")
            self._connected = False

    async def send(self, pcm: bytes):
        if not self._connected:
            try:
                await self.connect()
            except Exception as e:
                logger.error(f"[STT-Deepgram] Reconnect failed: {e}")
                return
        try:
            await self.ws.send(pcm)
        except Exception:
            self._connected = False

    async def close(self):
        self._connected = False
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except Exception:
                pass
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await self.ws.close()
            except Exception:
                pass


# ── STT: Sarvam (MR — buffered multipart REST) ───────────────────────────────

class SarvamSTT:
    def __init__(self, on_final_callback):
        self.on_final_callback = on_final_callback
        self.http = httpx.AsyncClient(timeout=30)

        # Speech buffer — only real speech chunks go here
        self._speech_buf: list[bytes] = []
        # Pre-speech buffer — last N silence chunks kept for context
        self._pre_buf: list[bytes] = []
        self._pre_buf_max = 8  # ~320ms of pre-speech context

        self._lock = asyncio.Lock()
        self._flush_task = None
        self._running = False
        self._last_speech_time = 0.0
        self._chunk_count = 0
        self._in_speech = False

    async def start(self):
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("[STT-Sarvam] Started — lang=mr")

    async def _flush_loop(self):
        while self._running:
            await asyncio.sleep(0.3)
            now = time.time()
            async with self._lock:
                speech_secs = sum(len(c) for c in self._speech_buf) / INPUT_SAMPLE_RATE / 2
                silence_gap = self._in_speech and (now - self._last_speech_time) > SILENCE_SECS
                max_hit     = speech_secs >= MAX_BUFFER_SECS

            if speech_secs >= MIN_SPEECH_SECS and (silence_gap or max_hit):
                reason = "silence" if silence_gap else "max_buffer"
                logger.info(f"[STT-Sarvam] Flush — reason={reason} speech={speech_secs:.1f}s")
                await self._flush()

    async def _flush(self):
        async with self._lock:
            if not self._speech_buf:
                return
            # Prepend pre-speech context for better transcription
            pcm = b"".join(self._pre_buf + self._speech_buf)
            self._speech_buf.clear()
            self._pre_buf.clear()
            self._in_speech = False

        duration = len(pcm) / INPUT_SAMPLE_RATE / 2
        logger.info(f"[STT-Sarvam] Sending {duration:.1f}s to Sarvam API...")
        wav = pcm_to_wav(pcm)

        try:
            resp = await self.http.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": SARVAM_API_KEY},
                files={"file": ("audio.wav", io.BytesIO(wav), "audio/wav")},
                data={
                    "model": "saarika:v2.5",
                    "language_code": "mr-IN",
                    "with_timestamps": "false",
                },
            )
            if not resp.is_success:
                logger.error(f"[STT-Sarvam] {resp.status_code}: {resp.text}")
                return
            transcript = (resp.json().get("transcript") or "").strip()
            if not transcript:
                logger.info("[STT-Sarvam] Empty transcript")
                return
            if is_hallucination(transcript):
                logger.info(f"[STT-Sarvam] Hallucination filtered: {transcript[:60]}...")
                return
            logger.info(f"[STT-Sarvam] Final: {transcript}")
            await self.on_final_callback(transcript)
        except Exception as e:
            logger.error(f"[STT-Sarvam] Exception: {e}")

    async def send(self, pcm: bytes):
        rms = get_rms(pcm)
        async with self._lock:
            self._chunk_count += 1
            n = self._chunk_count

            if n == 1:
                logger.info(f"[Audio] First chunk — RMS={rms:.1f} size={len(pcm)}")

            if rms > VAD_RMS_THRESHOLD:
                # Real speech
                if not self._in_speech:
                    logger.info(f"[STT-Sarvam] Speech started — RMS={rms:.0f}")
                    self._in_speech = True
                self._speech_buf.append(pcm)
                self._last_speech_time = time.time()
            else:
                # Silence — keep rolling pre-speech context
                self._pre_buf.append(pcm)
                if len(self._pre_buf) > self._pre_buf_max:
                    self._pre_buf.pop(0)

            speech_secs = sum(len(c) for c in self._speech_buf) / INPUT_SAMPLE_RATE / 2
            if n % 100 == 0:
                logger.info(f"[STT-Sarvam] chunk #{n} RMS={rms:.0f} speech_buf={speech_secs:.1f}s in_speech={self._in_speech}")

    async def close(self):
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except Exception:
                pass
        await self._flush()
        await self.http.aclose()


# ── Daily EventHandler ────────────────────────────────────────────────────────

class _DailyHandler(EventHandler):
    def __init__(self):
        EventHandler.__init__(self)
        self._agent = None

    def on_participant_joined(self, participant):
        name     = participant.get("info", {}).get("userName", "?")
        pid      = participant.get("id", "")
        is_local = participant.get("info", {}).get("isLocal", False)
        logger.info(f"[Daily] Participant joined: {name} (local={is_local})")

        if is_local or not pid or not self._agent:
            return

        agent = self._agent
        counter = {"n": 0}

        def on_audio_data(participant_id, audio_data, *args):
            try:
                pcm = bytes(audio_data.audio_frames)
            except AttributeError:
                pcm = bytes(audio_data)

            counter["n"] += 1
            n = counter["n"]

            if n == 1:
                logger.info(f"[Audio] First chunk! size={len(pcm)} is_speaking={agent._is_speaking}")
            if n % 200 == 0:
                logger.info(f"[Audio] Flowing — chunk #{n} size={len(pcm)} is_speaking={agent._is_speaking}")

            if not pcm or agent._is_speaking:
                return

            if agent._loop:
                asyncio.run_coroutine_threadsafe(
                    agent.stt.send(pcm), agent._loop
                )

        try:
            agent.client.set_audio_renderer(
                pid, on_audio_data,
                audio_source="microphone",
                sample_rate=INPUT_SAMPLE_RATE,
            )
            logger.info(f"[Daily] Audio renderer set for: {name} ✓")
        except Exception as e:
            logger.error(f"[Daily] set_audio_renderer failed: {e}")

        try:
            agent.client.update_subscriptions(
                participant_settings={pid: {"media": "subscribed"}}
            )
            logger.info(f"[Daily] Subscriptions updated for: {name} ✓")
        except Exception as e:
            logger.error(f"[Daily] update_subscriptions failed: {e}")

    def on_participant_left(self, participant, reason):
        logger.info(f"[Daily] Left: {participant.get('info',{}).get('userName','?')} ({reason})")

    def on_error(self, message):
        logger.error(f"[Daily] Error: {message}")


# ── Main Agent ────────────────────────────────────────────────────────────────

class DailyAgent:
    def __init__(self, lang: str):
        self.lang          = lang
        self.llm           = LLM(lang)
        self.tts           = TTS(lang)
        self.stt           = None
        self.client        = None
        self.mic           = None
        self._loop         = None
        self._turn_lock    = None
        self._is_speaking  = False
        self._joined_event = threading.Event()

    def _build_stt(self, callback):
        if self.lang == "mr":
            logger.info("[STT] Using Sarvam STT — lang=mr")
            return SarvamSTT(on_final_callback=callback)
        logger.info(f"[STT] Using Deepgram STT — lang={self.lang}")
        return DeepgramSTT(lang=self.lang, on_final_callback=callback)

    async def run(self):
        self._loop      = asyncio.get_event_loop()
        self._turn_lock = asyncio.Lock()

        async def on_transcript(text: str):
            await self._handle_transcript(text)

        self.stt = self._build_stt(on_transcript)
        if isinstance(self.stt, SarvamSTT):
            await self.stt.start()
        else:
            await self.stt.connect()

        Daily.init()
        self.mic = Daily.create_microphone_device(
            "agent-mic",
            sample_rate=OUTPUT_SAMPLE_RATE,
            channels=CHANNELS,
            non_blocking=True,
        )

        handler        = _DailyHandler()
        handler._agent = self
        self.client    = CallClient(event_handler=handler)

        self.client.update_inputs({
            "camera":     {"isEnabled": False},
            "microphone": {"isEnabled": True,
                           "settings": {"deviceId": "agent-mic"}},
        })
        self.client.update_subscription_profiles({
            "base": {"camera": "unsubscribed", "microphone": "subscribed"}
        })

        def on_joined(data, error):
            if error:
                logger.error(f"[Daily] Join failed: {error}")
            else:
                logger.info("[Daily] Joined room ✓")
                self._joined_event.set()

        logger.info(f"[Daily] Joining {DAILY_ROOM_URL} — lang={self.lang}")
        self.client.join(DAILY_ROOM_URL, completion=on_joined)

        for _ in range(200):
            if self._joined_event.is_set():
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError("Timed out waiting to join Daily room")

        await asyncio.sleep(1.5)
        await self._speak(GREETINGS[self.lang])

        logger.info("[Daily] Agent live — press Ctrl+C to stop")
        try:
            while True:
                await asyncio.sleep(1)
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            await self._cleanup()

    async def _handle_transcript(self, text: str):
        async with self._turn_lock:
            logger.info(f"[Turn] User: {text}")
            reply = await self._loop.run_in_executor(None, self.llm.generate, text)
            await self._speak(reply)

    async def _speak(self, text: str):
        self._is_speaking = True
        try:
            logger.info(f"[Speak] {text}")
            pcm = await self._loop.run_in_executor(None, self.tts.synthesize, text)
            chunk_size = int(OUTPUT_SAMPLE_RATE * 0.02) * 2
            offset = 0
            while offset < len(pcm):
                self.mic.write_frames(pcm[offset: offset + chunk_size])
                offset += chunk_size
                await asyncio.sleep(0.018)
        except Exception as e:
            logger.error(f"[Speak] {e}")
        finally:
            self._is_speaking = False

    async def _cleanup(self):
        logger.info("[Daily] Shutting down...")
        await self.stt.close()
        self.tts.close()
        if self.client:
            self.client.leave()
            self.client.release()
        Daily.deinit()
        logger.info("[Daily] Done.")


# ── Entry ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Arios AI — Daily.co Voice Agent")
    parser.add_argument("--lang", choices=["en", "hi", "mr"], default="en")
    args = parser.parse_args()
    logger.info(f"Starting — lang={args.lang}")
    await DailyAgent(lang=args.lang).run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped.")
