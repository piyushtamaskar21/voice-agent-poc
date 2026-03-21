import os
import json
import time
import uuid
import base64
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Awaitable

import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from openai import AsyncOpenAI
from sarvamai import AsyncSarvamAI
from sarvamai.core.events import EventType

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("arios-voice-server")

app = FastAPI(title="Arios AI Voice Server")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

INPUT_SAMPLE_RATE = int(os.getenv("INPUT_SAMPLE_RATE", "16000"))
OUTPUT_SAMPLE_RATE = int(os.getenv("OUTPUT_SAMPLE_RATE", "24000"))

ENGLISH_ELEVENLABS_VOICE_ID = "cgSgspJ2msm6clMCkdW9"
ELEVENLABS_MODEL_ID = "eleven_flash_v2_5"


def normalize_lang(locale: str) -> str:
    locale = (locale or "en-IN").lower()
    if locale.startswith("hi"):
        return "hi"
    if locale.startswith("mr"):
        return "mr"
    return "en"


SYSTEM_PROMPTS = {
    "en": (
        "You are Jessica, a female AI SDR from Arios AI. "
        "Speak like a real, warm, confident human on a phone call. "
        "Keep responses short and natural: usually 1 sentence, max 2 short sentences. "
        "Do not give long speeches. "
        "Introduce yourself briefly, then ask one simple question and wait."
    ),
    "hi": (
        "You are Jessica, a female AI SDR from Arios AI. "
        "Speak in natural Hinglish suitable for a real India phone conversation. "
        "Be warm, concise, and professional. "
        "Keep responses short: usually 1 sentence, max 2 short sentences. "
        "No long speeches. Introduce yourself briefly, ask one simple question, then wait."
    ),
    "mr": (
        "You are Jessica, a female AI SDR from Arios AI. "
        "Speak in natural Marathi mixed with English, like a real Maharashtra phone conversation. "
        "Be warm, concise, and professional. "
        "Keep responses short: usually 1 sentence, max 2 short sentences. "
        "No long speeches. Introduce yourself briefly, ask one simple question, then wait."
    ),
}

GREETING_MESSAGES = {
    "en": "Hi, Jessica here from Arios AI. How can I help you today?",
    "hi": "Hi, Arios AI se Jessica bol rahi hoon. Aaj main aapki kis cheez mein help kar sakti hoon?",
    "mr": "Hi, Arios AI मधून Jessica बोलतेय. आज मी तुम्हाला कशात help करू शकते?",
}


def add_wav_header(pcm_data: bytes, sample_rate: int, channels: int = 1, bit_depth: int = 16) -> bytes:
    """Adds a standard WAV header to raw 16-bit mono PCM bytes."""
    header = bytearray()
    header.extend(b'RIFF')
    header.extend((36 + len(pcm_data)).to_bytes(4, 'little'))
    header.extend(b'WAVE')
    header.extend(b'fmt ')
    header.extend((16).to_bytes(4, 'little'))  # Subchunk1Size
    header.extend((1).to_bytes(2, 'little'))   # AudioFormat (PCM=1)
    header.extend((channels).to_bytes(2, 'little'))
    header.extend((sample_rate).to_bytes(4, 'little'))
    header.extend((sample_rate * channels * bit_depth // 8).to_bytes(4, 'little'))
    header.extend((channels * bit_depth // 8).to_bytes(2, 'little'))
    header.extend((bit_depth).to_bytes(2, 'little'))
    header.extend(b'data')
    header.extend((len(pcm_data)).to_bytes(4, 'little'))
    return bytes(header) + pcm_data


def get_system_prompt(locale: str) -> str:
    return SYSTEM_PROMPTS[normalize_lang(locale)]


def get_greeting(locale: str) -> str:
    return GREETING_MESSAGES[normalize_lang(locale)]


def trim_history(messages: List[Dict[str, str]], max_messages: int = 12) -> List[Dict[str, str]]:
    return messages[-max_messages:]


def enforce_short_reply(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Could you please repeat that?"

    cleaned = " ".join(text.replace("\n", " ").split()).strip()
    if not cleaned:
        return "Could you please repeat that?"

    normalized = cleaned.replace("?", "?.").replace("!", "!.")
    parts = [p.strip() for p in normalized.split(".") if p.strip()]
    if len(parts) <= 2:
        return cleaned

    result = ". ".join(parts[:2]).strip()
    if not result.endswith((".", "?", "!")):
        result += "."
    return result


@dataclass
class TenantConfig:
    tenant_id: str
    locale: str
    system_prompt: str
    greeting: str
    temperature: float = 0.3
    max_tokens: int = 90


@dataclass
class SessionState:
    session_id: str
    websocket: WebSocket
    tenant: TenantConfig
    conversation: List[Dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stop_tts_event: asyncio.Event = field(default_factory=asyncio.Event)
    speaking_task: Optional[asyncio.Task] = None
    stt: Optional["BaseSTTAdapter"] = None
    llm: Optional["LLMResponder"] = None
    tts: Optional["TTSRouter"] = None
    closed: bool = False
    is_speaking: bool = False
    last_stop_time: float = 0.0  # Track when we last stopped for grace period

    async def send_audio_chunk(self, audio_bytes: bytes, fmt: str = "pcm_s16le", sample_rate: int = OUTPUT_SAMPLE_RATE):
        if self.closed or not audio_bytes:
            return
        
        # For compatibility with browser's decodeAudioData, wrap PCM in WAV header
        if fmt.startswith("pcm"):
            audio_bytes = add_wav_header(audio_bytes, sample_rate)
            
        try:
            # Re-enable lock for sanity, but keep it tight
            async with self.send_lock:
                await self.websocket.send_bytes(audio_bytes)
        except Exception as e:
            logger.error(f"Failed to send audio bytes: {e}")

    async def send_json(self, payload: Dict):
        if self.closed:
            return
        print(f"DEBUG: Session {self.session_id} - send_json: {payload.get('type')}")
        try:
            # Temporarily bypass send_lock to rule out deadlocks
            await self.websocket.send_text(json.dumps(payload))
        except Exception as e:
            print(f"DEBUG: Session {self.session_id} - FAILED to send JSON: {e}")

    async def interrupt_speech(self):
        self.stop_tts_event.set()
        if self.speaking_task and not self.speaking_task.done():
            self.speaking_task.cancel()
            try:
                await self.speaking_task
            except (Exception, asyncio.CancelledError):
                pass
        self.is_speaking = False
        self.last_stop_time = time.time()
        # Note: we still send a stop signal here to ensure the client is in a valid state
        await self.send_json({"type": "assistant_audio_stop"})

    async def close(self):
        self.closed = True
        self.stop_tts_event.set()
        if self.speaking_task and not self.speaking_task.done():
            self.speaking_task.cancel()
            try:
                await self.speaking_task
            except (Exception, asyncio.CancelledError):
                pass
        if self.stt:
            await self.stt.close()


class BaseSTTAdapter:
    provider: str = "base"

    async def connect(self):
        raise NotImplementedError

    async def send_audio(self, pcm: bytes):
        raise NotImplementedError

    async def close(self):
        raise NotImplementedError


class DeepgramSTTAdapter(BaseSTTAdapter):
    provider = "deepgram"

    def __init__(
        self,
        locale: str,
        on_interim: Callable[[str], Awaitable[None]],
        on_final: Callable[[str], Awaitable[None]],
    ):
        self.locale = locale
        self.on_interim = on_interim
        self.on_final = on_final
        self.ws = None
        self.recv_task: Optional[asyncio.Task] = None
        self.closed = False

    def _map_locale(self) -> str:
        lang = normalize_lang(self.locale)
        if lang == "hi":
            return "hi"
        if lang == "mr":
            return "multi"
        return "en"

    async def connect(self):
        if not DEEPGRAM_API_KEY:
            raise RuntimeError("Missing DEEPGRAM_API_KEY")

        url = (
            "wss://api.deepgram.com/v1/listen"
            "?encoding=linear16"
            f"&sample_rate={INPUT_SAMPLE_RATE}"
            "&channels=1"
            "&interim_results=true"
            "&punctuate=true"
            "&no_delay=true"
            "&smart_format=true"
            "&endpointing=300"
            f"&language={self._map_locale()}"
        )
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        self.ws = await websockets.connect(
            url,
            additional_headers=headers,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        )
        self.recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self):
        try:
            async for message in self.ws:
                if isinstance(message, bytes):
                    continue

                data = json.loads(message)
                if data.get("type") != "Results":
                    continue

                channel = data.get("channel", {})
                alternatives = channel.get("alternatives", [])
                if not alternatives:
                    continue

                transcript = (alternatives[0].get("transcript") or "").strip()
                if not transcript:
                    continue

                if data.get("is_final", False):
                    await self.on_final(transcript)
                else:
                    await self.on_interim(transcript)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Deepgram STT failed: %s", e)

    async def send_audio(self, pcm: bytes):
        if self.closed or not self.ws or not pcm:
            return
        await self.ws.send(pcm)

    async def close(self):
        self.closed = True
        if self.recv_task:
            self.recv_task.cancel()
            try:
                await self.recv_task
            except (Exception, asyncio.CancelledError):
                pass
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            try:
                await self.ws.close()
            except Exception:
                pass


class SarvamSTTAdapter(BaseSTTAdapter):
    provider = "sarvam"

    def __init__(
        self,
        locale: str,
        on_interim: Callable[[str], Awaitable[None]],
        on_final: Callable[[str], Awaitable[None]],
    ):
        self.locale = locale
        self.on_interim = on_interim
        self.on_final = on_final
        self.client = None
        self.ws_context = None
        self.ws = None
        self.recv_task: Optional[asyncio.Task] = None
        self.closed = False

    async def connect(self):
        if not SARVAM_API_KEY:
            raise RuntimeError("Missing SARVAM_API_KEY")

        self.client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
        self.ws_context = self.client.speech_to_text_streaming.connect(
            model="saaras:v3",
            sample_rate=str(INPUT_SAMPLE_RATE),
            language_code=self._language_code(),
            mode="codemix",
            # Add flush_signal for better responsiveness if supported
        )
        self.ws = await self.ws_context.__aenter__()

        def _message_handler(message):
            if self.closed:
                return
            if message.type == "data":
                transcript = (message.data.transcript or "").strip()
                if transcript:
                    # Sarvam saaras:v3 typically returns final utterances in streaming
                    asyncio.create_task(self.on_final(transcript))

        self.ws.on(EventType.MESSAGE, _message_handler)
        self.recv_task = asyncio.create_task(self.ws.start_listening())
        logger.info("Sarvam AI SDK connected")

    def _language_code(self) -> str:
        lang = normalize_lang(self.locale)
        return "mr-IN" if lang == "mr" else "hi-IN"

    async def send_audio(self, pcm: bytes):
        if self.closed or not self.ws or not pcm:
            return
        try:
            # SDK expects base64 encoded audio for the transcribe method
            audio_b64 = base64.b64encode(pcm).decode("utf-8")
            await self.ws.transcribe(
                audio=audio_b64,
                encoding="audio/wav",  # SDK explicitly requires audio/wav (Pydantic validation)
                sample_rate=INPUT_SAMPLE_RATE,
            )
        except Exception as e:
            if not self.closed:
                logger.error("Error sending audio to Sarvam: %s", e)

    async def close(self):
        self.closed = True
        if self.recv_task:
            self.recv_task.cancel()
            try:
                await self.recv_task
            except (Exception, asyncio.CancelledError):
                pass
        if self.ws_context:
            try:
                await self.ws_context.__aexit__(None, None, None)
            except Exception:
                pass
        logger.info("Sarvam AI SDK disconnected")


class LLMResponder:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def generate(self, session: SessionState, user_text: str) -> str:
        messages = [{"role": "system", "content": session.tenant.system_prompt}]
        messages.extend(trim_history(session.conversation))
        messages.append({"role": "user", "content": user_text})

        resp = await self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=session.tenant.temperature,
            max_tokens=session.tenant.max_tokens,
        )

        content = resp.choices[0].message.content or ""
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            content = " ".join(parts)

        return enforce_short_reply(str(content).strip())


class DisabledLLMResponder:
    async def generate(self, session: SessionState, user_text: str) -> str:
        return "Sorry, the response engine is not configured."


class TTSRouter:
    def __init__(self, elevenlabs_api_key: str, sarvam_api_key: str):
        self.elevenlabs_api_key = elevenlabs_api_key
        self.sarvam_api_key = sarvam_api_key
        self.http = httpx.AsyncClient(timeout=60.0)

    async def close(self):
        await self.http.aclose()

    async def speak(self, session: SessionState, text: str):
        if not text or not text.strip():
            return
        start_time = time.time()
        total_bytes = 0
        try:
            session.stop_tts_event.clear()
            if not session.is_speaking:
                session.is_speaking = True
                await session.send_json({"type": "assistant_audio_start"})
            
            # Wrap the actual sending to track bytes
            orig_send = session.send_audio_chunk
            async def tracked_send(audio_bytes, fmt="pcm_s16le", sample_rate=OUTPUT_SAMPLE_RATE):
                nonlocal total_bytes
                total_bytes += len(audio_bytes)
                await orig_send(audio_bytes, fmt, sample_rate)
            
            # Monkeypatch temporarily for this task (simple way to track across adapters)
            session.send_audio_chunk = tracked_send
            
            lang = normalize_lang(session.tenant.locale)
            if lang.startswith("en"):
                await self._speak_elevenlabs(session, text)
            else:
                await self._speak_sarvam(session, text, lang)
            
            # Calculate duration. 2 channels * 2 bytes/sample for WAV header impact if added
            # But we use Mono (1 channel). 2 bytes/sample.
            # PCM part is Rate * 2 bytes/sec.
            duration = total_bytes / (OUTPUT_SAMPLE_RATE * 2)
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            if remaining > 0 and not session.stop_tts_event.is_set():
                # Allow some overlapping (e.g. 0.5s) to feel faster but prevent early mic
                await asyncio.sleep(remaining + 0.5) 
                
        finally:
            session.send_audio_chunk = orig_send # Restore
            if session.is_speaking:
                session.is_speaking = False
                session.last_stop_time = time.time()
                await session.send_json({"type": "assistant_audio_stop"})

    async def _speak_elevenlabs(self, session: SessionState, text: str):
        if not self.elevenlabs_api_key:
            await session.send_json({
                "type": "server_error",
                "message": "ELEVENLABS_API_KEY is not configured",
            })
            return

        await session.send_json({"type": "assistant_text", "text": text})
        await session.send_json({
            "type": "assistant_audio_start",
            "provider": "elevenlabs",
            "voice_id": ENGLISH_ELEVENLABS_VOICE_ID,
            "format": "pcm_s16le",
            "sample_rate": OUTPUT_SAMPLE_RATE,
        })

        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/"
            f"{ENGLISH_ELEVENLABS_VOICE_ID}/stream?output_format=pcm_{OUTPUT_SAMPLE_RATE}"
        )
        headers = {
            "xi-api-key": self.elevenlabs_api_key,
            "accept": "audio/pcm",
            "content-type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": ELEVENLABS_MODEL_ID,
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.75,
                "style": 0.10,
                "use_speaker_boost": True,
            },
        }

        try:
            buffer = bytearray()
            # Buffer size: roughly 0.1s of audio (24000 Hz * 2 bytes * 0.1s = 4800 bytes)
            MIN_CHUNK_SIZE = 4800 

            async with self.http.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    if session.stop_tts_event.is_set():
                        break
                    if chunk:
                        buffer.extend(chunk)
                        if len(buffer) >= MIN_CHUNK_SIZE:
                            await session.send_audio_chunk(
                                audio_bytes=bytes(buffer),
                                fmt="pcm_s16le",
                                sample_rate=OUTPUT_SAMPLE_RATE,
                            )
                            buffer.clear()
                
                # Send remaining buffer
                if buffer and not session.stop_tts_event.is_set():
                    await session.send_audio_chunk(
                        audio_bytes=bytes(buffer),
                        fmt="pcm_s16le",
                        sample_rate=OUTPUT_SAMPLE_RATE,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("ElevenLabs TTS failed: %s", e)
            await session.send_json({
                "type": "server_error",
                "message": f"ElevenLabs TTS failed: {str(e)}",
            })

    async def _speak_sarvam(self, session: SessionState, text: str, lang: str):
        if not self.sarvam_api_key:
            await session.send_json({
                "type": "server_error",
                "message": "SARVAM_API_KEY is not configured",
            })
            return

        language_code = "hi-IN" if lang == "hi" else "mr-IN"

        await session.send_json({"type": "assistant_text", "text": text})
        await session.send_json({
            "type": "assistant_audio_start",
            "provider": "sarvam",
            "language_code": language_code,
            "format": "wav",
            "sample_rate": OUTPUT_SAMPLE_RATE,
        })

        url = "https://api.sarvam.ai/text-to-speech"
        headers = {
            "api-subscription-key": self.sarvam_api_key,
            "content-type": "application/json",
        }
        payload = {
            "target_language_code": language_code,
            "speaker": "anushka",
            "pitch": 0,
            "pace": 1.0,
            "loudness": 1.0,
            "speech_sample_rate": OUTPUT_SAMPLE_RATE,
            "enable_preprocessing": True,
            "model": "bulbul:v2",
            "inputs": [text],
        }

        try:
            resp = await self.http.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            audios = data.get("audios") or []
            if not audios:
                raise RuntimeError("Sarvam returned no audio")

            audio_bytes = base64.b64decode(audios[0])

            if not session.stop_tts_event.is_set():
                await session.send_audio_chunk(
                    audio_bytes=audio_bytes,
                    fmt="wav",
                    sample_rate=OUTPUT_SAMPLE_RATE,
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Sarvam TTS failed: %s", e)
            await session.send_json({
                "type": "server_error",
                "message": f"Sarvam TTS failed: {str(e)}",
            })


TTS_ADAPTER: Optional[TTSRouter] = None
LLM_ADAPTER: Optional[object] = None


def choose_stt_provider(locale: str) -> str:
    lang = normalize_lang(locale)
    if lang in {"hi", "mr"} and SARVAM_API_KEY:
        return "sarvam"
    return "deepgram"


async def build_stt(
    locale: str,
    on_interim: Callable[[str], Awaitable[None]],
    on_final: Callable[[str], Awaitable[None]],
) -> BaseSTTAdapter:
    provider = choose_stt_provider(locale)

    if provider == "sarvam":
        stt = SarvamSTTAdapter(locale, on_interim, on_final)
    else:
        stt = DeepgramSTTAdapter(locale, on_interim, on_final)

    await stt.connect()
    return stt


def tenant_from_websocket(ws: WebSocket) -> TenantConfig:
    tenant_id = ws.query_params.get("tenant_id", "default")
    locale = ws.query_params.get("locale", "en-IN")
    return TenantConfig(
        tenant_id=tenant_id,
        locale=locale,
        system_prompt=get_system_prompt(locale),
        greeting=get_greeting(locale),
    )


async def handle_interim_transcript(session: SessionState, transcript: str):
    transcript = (transcript or "").strip()
    if not transcript:
        return

    await session.send_json({
        "type": "user_transcript",
        "text": transcript,
        "final": False,
    })


async def handle_final_transcript(session: SessionState, transcript: str):
    transcript = (transcript or "").strip()

    # Grace period: ignore STT for 1.5s after finishing speaking to avoid echo
    if time.time() - session.last_stop_time < 1.5:
        logger.debug(f"Ignoring STT during grace period. Last stop: {session.last_stop_time:.2f}, now: {time.time():.2f}")
        return
    
    if session.is_speaking:
        # Ignore transcripts if they are too short to be meaningful interruptions
        if len(transcript) < 5:
            logger.debug(f"Ignoring short STT while speaking: '{transcript}'")
            return
    
    if not transcript:
        return

    async with session.turn_lock:
        session.last_activity = time.time()

        await session.send_json({
            "type": "user_transcript",
            "text": transcript,
            "final": True,
        })

        session.conversation.append({"role": "user", "content": transcript})
        session.conversation = trim_history(session.conversation)

        reply = await session.llm.generate(session, transcript)
        reply = enforce_short_reply(reply)

        session.conversation.append({"role": "assistant", "content": reply})
        session.conversation = trim_history(session.conversation)

        await session.interrupt_speech()
        session.speaking_task = asyncio.create_task(session.tts.speak(session, reply))


async def send_opening_greeting(session: SessionState):
    greeting = session.tenant.greeting
    session.conversation.append({"role": "assistant", "content": greeting})
    session.conversation = trim_history(session.conversation)
    session.speaking_task = asyncio.create_task(session.tts.speak(session, greeting))


@app.on_event("startup")
async def startup():
    global TTS_ADAPTER, LLM_ADAPTER

    if OPENAI_API_KEY:
        LLM_ADAPTER = LLMResponder()
        logger.info("LLM enabled with model=%s", OPENAI_MODEL)
    else:
        LLM_ADAPTER = DisabledLLMResponder()
        logger.warning("OPENAI_API_KEY missing; LLM disabled")

    TTS_ADAPTER = TTSRouter(
        elevenlabs_api_key=ELEVENLABS_API_KEY,
        sarvam_api_key=SARVAM_API_KEY,
    )

    logger.info("Startup complete")


@app.on_event("shutdown")
async def shutdown():
    global TTS_ADAPTER
    if TTS_ADAPTER:
        await TTS_ADAPTER.close()


@app.get("/")
async def get_index():
    return FileResponse("index.html")


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "openai_configured": bool(OPENAI_API_KEY),
        "deepgram_configured": bool(DEEPGRAM_API_KEY),
        "sarvam_configured": bool(SARVAM_API_KEY),
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY),
        "english_elevenlabs_voice_id": ENGLISH_ELEVENLABS_VOICE_ID,
        "tts_routing": {
            "en": "elevenlabs",
            "hi": "sarvam",
            "mr": "sarvam",
        },
    }


@app.websocket("/ws")
async def websocket_entry(ws: WebSocket):
    await ws.accept()

    tenant = tenant_from_websocket(ws)

    session = SessionState(
        session_id=str(uuid.uuid4()),
        websocket=ws,
        tenant=tenant,
        llm=LLM_ADAPTER,
        tts=TTS_ADAPTER,
    )

    async def on_interim(text: str):
        await handle_interim_transcript(session, text)

    async def on_final(text: str):
        await handle_final_transcript(session, text)

    try:
        session.stt = await build_stt(tenant.locale, on_interim, on_final)

        await session.send_json({
            "type": "session_ready",
            "session_id": session.session_id,
            "tenant_id": tenant.tenant_id,
            "locale": tenant.locale,
            "stt_provider": session.stt.provider,
            "tts_provider": "sarvam" if normalize_lang(tenant.locale) in {"hi", "mr"} else "elevenlabs",
            "english_voice_id": ENGLISH_ELEVENLABS_VOICE_ID,
            "input_sample_rate": INPUT_SAMPLE_RATE,
            "output_sample_rate": OUTPUT_SAMPLE_RATE,
        })

        # Diagnostic: Send a small 0.5s beep at 16k directly to test audio path
        print(f"DEBUG: Session {session.session_id} - Sending diagnostic beep")
        diagnostic_pcm = (b'\x00\x00\x3f\x3f' * 2000) # Simple noise/square wave
        await session.send_audio_chunk(diagnostic_pcm, fmt="pcm_s16le", sample_rate=16000)

        await send_opening_greeting(session)

        while True:
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected normally")
                break
            except Exception as e:
                logger.error("WebSocket receive error: %s", e)
                break

            if msg["type"] == "websocket.disconnect":
                logger.info("Client sent disconnect message")
                break

            session.last_activity = time.time()

            if "bytes" in msg and msg["bytes"] is not None:
                audio = msg["bytes"]
                if audio:
                    await session.interrupt_speech()
                    await session.stt.send_audio(audio)
                continue

            if "text" in msg and msg["text"] is not None:
                raw = msg["text"]

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await session.send_json({
                        "type": "server_error",
                        "message": "Invalid JSON message",
                    })
                    continue

                message_type = data.get("type")

                if message_type == "ping":
                    await session.send_json({"type": "pong", "ts": time.time()})
                    continue

                if message_type == "audio":
                    audio_b64 = data.get("audio_b64", "")
                    if audio_b64:
                        pcm = base64.b64decode(audio_b64)
                        await session.interrupt_speech()
                        await session.stt.send_audio(pcm)
                    continue

                if message_type == "user_speaking_start":
                    await session.interrupt_speech()
                    continue

                if message_type == "text_input":
                    text = (data.get("text") or "").strip()
                    if text:
                        await handle_final_transcript(session, text)
                    continue

                if message_type == "config_update":
                    new_locale = data.get("locale")
                    if new_locale:
                        tenant.locale = new_locale
                        tenant.system_prompt = get_system_prompt(new_locale)
                        tenant.greeting = get_greeting(new_locale)

                        if session.stt:
                            await session.stt.close()
                        session.stt = await build_stt(tenant.locale, on_interim, on_final)

                        await session.send_json({
                            "type": "config_updated",
                            "locale": tenant.locale,
                            "stt_provider": session.stt.provider,
                            "tts_provider": "sarvam" if normalize_lang(tenant.locale) in {"hi", "mr"} else "elevenlabs",
                            "english_voice_id": ENGLISH_ELEVENLABS_VOICE_ID,
                        })
                    continue

                if message_type == "disconnect":
                    break

                await session.send_json({
                    "type": "server_error",
                    "message": f"Unsupported message type: {message_type}",
                })

    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", session.session_id)
    except Exception as e:
        logger.exception("Session failed: %s", e)
        try:
            await session.send_json({
                "type": "server_error",
                "message": str(e),
            })
        except Exception:
            pass
    finally:
        await session.close()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
