# LeadGen Engine: Jessica AI SDR 🎙️
> **The future of automated, low-latency voice lead generation.**

LeadGen Engine is a state-of-the-art voice-to-voice AI platform designed to automate initial SDR interactions. Featuring "Jessica," an ultra-low-latency AI Voice Agent, the project demonstrates seamless real-time conversation using modern AI primitives.

---

## 📖 Overview

LeadGen Engine solves the problem of friction in early-stage lead qualification. Instead of static forms or slow text-based bots, it provides a human-like voice interface that can instantly greet, qualify, and assist potential leads.

### Key Value Propositions:
- **Zero Latency (Perceived):** Optimized pipeline for near-instant response times.
- **Natural Interaction:** Advanced VAD (Voice Activity Detection) for fluid turn-taking.
- **Production-Ready Core:** Built on Pipecat AI and FastAPI for high scalability.

---

## ✨ Features

- **Real-Time Voice-to-Voice:** Intelligent conversation with sub-second response times.
- **Automated AI Greeting:** Instant engagement upon connection.
- **Multi-Service Integration:** Pluggable architecture supports Deepgram, ElevenLabs, and Groq/OpenAI.
- **High-Fidelity Audio:** 16kHz linear PCM audio delivery via WebSockets.
- **Responsive Web Client:** Clean, intuitive UI for seamless browser-based interaction.

---

## 🛠️ Tech Stack

- **Frameworks:** [Pipecat AI](https://github.com/pipecat-ai/pipecat), [FastAPI](https://fastapi.tiangolo.com/)
- **Large Language Model:** Groq (Llama 3 / Mixtral) for high-speed inference.
- **Speech-to-Text (STT):** [Deepgram](https://www.deepgram.com/)
- **Text-to-Speech (TTS):** [ElevenLabs](https://elevenlabs.io/) (Turbo v2.5)
- **Voice Activity Detection:** [Silero VAD](https://github.com/snakers4/silero-vad)
- **Frontend:** Vanilla HTML5 / JavaScript (WebSocket-based)

---

## 🏗️ Architecture (High-Level)

The system utilizes a linear pipeline architecture:
1. **Input Stage:** Browser captures audio -> WebSocket -> `FastAPIWebsocketTransport`.
2. **Analysis Stage:** `SileroVADAnalyzer` detects speech; `DeepgramSTT` transcribes audio in real-time.
3. **Logic Stage:** `OpenAILLMService` (via Groq) processes intent and generates conversational text.
4. **Synthesis Stage:** `ElevenLabsHttpTTSService` converts text to high-fidelity binary audio.
5. **Output Stage:** `RawAudioSerializer` packages audio for delivery -> Client plays audio via `AudioContext`.

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Node.js (for optional development tooling)
- API Keys for: Deepgram, ElevenLabs, and Groq (or OpenAI).

### Step-by-Step Setup
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd voice-poc
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration:**
   Create a `.env` file in the root directory:
   ```env
   DEEPGRAM_API_KEY=your_key_here
   ELEVENLABS_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   ```

---

## 📖 Usage

### Running Locally
Start the lead generation server:
```bash
PYTHONUNBUFFERED=1 ./venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### Accessing the Interface
1. Open your browser and navigate to `http://localhost:8000`.
2. Click **"Start Conversation"**.
3. Allow microphone access.
4. Jessica will greet you immediately. Start speaking to interact!

---

## 📂 Project Structure

```text
├── server.py              # Main FastAPI application & Pipecat pipeline
├── index.html             # Frontend client interface
├── .env                   # Environment variables (secret)
├── venv/                  # Python virtual environment
└── docs/                  # Technical documentation & post-mortems
```

---

## 🔌 API Documentation

### WebSockets
- **Endpoint:** `/ws`
- **Protocol:** Binary WebSocket (PCM Audio Chunks)
- **Initialization:** Client sends `ArrayBuffer` audio data; Server responds with binary audio chunks.

---

## 🧪 Testing

### Audio Verification
Run the standalone verification script to confirm server-side audio generation:
```bash
python scripts/verify_audio_output.py
```

---

## 🛡️ Security Considerations

- **API Security:** Environment variables are used for all sensitive credentials.
- **WebSocket Safety:** Implements proper connection lifecycle management to prevent orphan sessions.
- **Data Privacy:** Local VAD analysis minimizes unnecessary audio transmission to cloud providers.

---

## 🗺️ Roadmap

- [ ] Support for multiple languages (Spanish, French, German).
- [ ] Direct CRM integration (Salesforce/HubSpot) for lead capture.
- [ ] Visual Avatar integration.
- [ ] Enterprise-grade Auth (OAuth2/JWT).

---

## 🛠️ Troubleshooting

**Common Issue:** *No audio playback in browser.*
- **Fix:** Ensure `RawAudioSerializer` is active in `server.py` and that the browser hasn't blocked auto-playing audio (click 'Start' first).
- **Check Logs:** Monitor `server.log` for any Deepgram or ElevenLabs API errors.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgements

- The Pipecat AI team for the incredible framework.
- ElevenLabs and Deepgram for best-in-class audio primitives.
