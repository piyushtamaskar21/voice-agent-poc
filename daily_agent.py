#!/usr/bin/env python3
"""
daily_agent.py — Standalone Daily.co AI voice agent
Usage:
    python3 daily_agent.py               # English (default)
    python3 daily_agent.py --lang hi     # Hindi
    python3 daily_agent.py --lang mr     # Marathi
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import subprocess
import threading

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
        "Tone warm, polite, thoda enthusiastic aur patient hona chahiye. "
        "Robot jaisa bilkul nahi lagna chahiye. "

        "Responses short rakho: usually 1 sentence, max 2 short sentences. "
        "Long explanation mat do jab tak user specifically na puche. "

        "Conversation style: "
        "Sirf start mein short intro do. "
        "Har response ke baad ek simple question pucho aur ruk jao. "
        "Natural fillers use karo jaise 'Got it', 'samajh gaya', 'makes sense'. "
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
        "Agar unsure ho toh bolo ki team se confirm karogi. "

        "Goal: "
        "Conversation natural, helpful aur smooth rakhna aur user ko next step tak le jaana."
    ),

    "mr": (
        "You are Jessica, a female AI SDR and receptionist from Arios AI. "
        "Marathi + English mix madhe natural bolaa — jaise Maharashtra madhe real phone conversation hote. "
        "Tone warm, polite, thoda enthusiastic ani patient hava. "
        "Robot sarkha bilkul vatla nahi pahije. "

        "Responses short theva: usually 1 sentence, max 2 short sentences. "
        "User ne specifically vicharla tarach detail madhe jaa. "

        "Conversation style: "
        "Suruvatila ek short intro dya. "
        "Pratyek response nantar ek simple prashna vichara ani thamba. "
        "Natural phrases use kara jaise 'Got it', 'samajhla', 'makes sense'. "
        "User confused asel tar simple explain kara, over-explain nako. "
        "Calm ani patient raha. "

        "Tumcha role: "
        "Customer queries handle karna like professional call center executive. "
        "User cha requirement samajhne ani qualify karne. "
        "Demo kiwa next step sathi guide karne. "
        "Basic details politely collect karu shakta. "

        "Arios AI kay karto: "
        "AI systems banavto je business madhla manual kaam automate kartat. "
        "LLM apps, RAG systems, AI agents, workflow automation. "
        "Voice AI agents, support bots, sales copilots, knowledge assistants. "
        "Focus ROI var — cost kami, efficiency jasta. "

        "Important: "
        "Kahi hi assume karu naka. "
        "Jar unsure asal tar sanga ki team sobat confirm karal. "

        "Goal: "
        "Conversation natural, helpful ani smooth thevne ani user la next step kade gheun jaan."
    ),
}

GREETINGS = {
    "en": "Hi, Jessica here from Arios AI. How can I help you today?",
    "hi": "Hi, Arios AI se Jessica bol rahi hoon. Aaj main aapki kaise help kar sakti hoon?",
    "mr": "Hi, Arios AI मधून Jessica बोलतेय. आज मी तुम्हाला कशात help करू शकते?",
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
        pcm = mp3_to_pcm16k(resp.content)
        logger.info(f"[TTS-EN] {len(pcm)} PCM bytes")
        return pcm

    def _sarvam_tts(self, text: str) -> bytes:
        lang_code = "hi-IN" if self.lang == "hi" else "mr-IN"
        logger.info(f"[TTS-{self.lang.upper()}] {text[:80]}")
        resp = self.http.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": SARVAM_API_KEY,
                     "content-type": "application/json"},
            json={"target_language_code": lang_code, "speaker": "anushka",
                  "pitch": 0, "pace": 1.0, "loudness": 1.0,
                  "speech_sample_rate": OUTPUT_SAMPLE_RATE,
                  "enable_preprocessing": True, "model": "bulbul:v2",
                  "inputs": [text]},
        )
        resp.raise_for_status()
        audios = resp.json().get("audios") or []
        if not audios:
            raise RuntimeError("Sarvam returned no audio")
        return wav_to_pcm(base64.b64decode(audios[0]))

    def close(self):
        self.http.close()


# ── STT ───────────────────────────────────────────────────────────────────────

class DeepgramSTT:
    def __init__(self, lang: str, on_final_callback):
        self.lang = lang
        self.on_final_callback = on_final_callback
        self.ws = None
        self._recv_task = None
        self._connected = False

    def _lang_code(self):
        # Deepgram doesn't support 'multi' on streaming — use detect_language instead
        return {"hi": "hi", "mr": "hi"}.get(self.lang, "en")

    async def connect(self):
        lang_code = self._lang_code()
        detect = "&detect_language=true" if self.lang == "mr" else ""
        url = (
            "wss://api.deepgram.com/v1/listen"
            "?encoding=linear16"
            f"&sample_rate={INPUT_SAMPLE_RATE}"
            "&channels=1&interim_results=false"
            "&punctuate=true&smart_format=true&endpointing=400"
            f"&language={lang_code}{detect}"
        )
        self.ws = await websockets.connect(
            url,
            additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            max_size=None, ping_interval=20, ping_timeout=20,
        )
        self._connected = True
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info("[STT] Deepgram connected")

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
                    logger.info(f"[STT] Final: {text}")
                    await self.on_final_callback(text)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"[STT] Disconnected: {e} — will reconnect")
            self._connected = False

    async def send(self, pcm: bytes):
        if not self._connected:
            try:
                await self.connect()
            except Exception as e:
                logger.error(f"[STT] Reconnect failed: {e}")
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


# ── Daily EventHandler ────────────────────────────────────────────────────────

class _DailyHandler(EventHandler):
    def __init__(self):
        EventHandler.__init__(self)
        self._agent = None   # injected by DailyAgent

    def on_participant_joined(self, participant):
        name = participant.get("info", {}).get("userName", "?")
        pid  = participant.get("id", "")
        is_local = participant.get("info", {}).get("isLocal", False)
        logger.info(f"[Daily] Participant joined: {name} (local={is_local})")

        # Skip the bot's own audio — only subscribe to remote humans
        if is_local or not pid or not self._agent:
            return

        agent = self._agent

        # ── Key fix: use set_audio_renderer per participant ──
        def on_audio_data(participant_id, audio_data, *args):
            if agent._is_speaking:
                return
            try:
                pcm = audio_data.audio_frames   # daily.AudioData object
            except AttributeError:
                pcm = bytes(audio_data)
            if pcm and agent._loop:
                asyncio.run_coroutine_threadsafe(
                    agent.stt.send(bytes(pcm)), agent._loop
                )

        try:
            agent.client.set_audio_renderer(
                pid,
                on_audio_data,
                audio_source="microphone",
                sample_rate=INPUT_SAMPLE_RATE,
            )
            logger.info(f"[Daily] Audio renderer set for: {name}")
        except Exception as e:
            logger.error(f"[Daily] set_audio_renderer failed: {e}")

        # Also update subscription
        try:
            agent.client.update_subscriptions(
                participant_settings={pid: {"media": "subscribed"}}
            )
        except Exception as e:
            logger.error(f"[Daily] update_subscriptions failed: {e}")

    def on_participant_left(self, participant, reason):
        name = participant.get("info", {}).get("userName", "?")
        logger.info(f"[Daily] Participant left: {name} ({reason})")

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

    async def run(self):
        self._loop      = asyncio.get_event_loop()
        self._turn_lock = asyncio.Lock()

        async def on_transcript(text: str):
            await self._handle_transcript(text)

        self.stt = DeepgramSTT(self.lang, on_transcript)
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
