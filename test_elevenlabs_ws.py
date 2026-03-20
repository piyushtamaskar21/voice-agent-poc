import asyncio
import os
import base64
import json
import websockets
from dotenv import load_dotenv

async def test_elevenlabs_ws():
    load_dotenv()
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = "cgSgspJ2msm6clMCkdW9" # Jessica
    model = "eleven_turbo_v2_5"
    
    url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/multi-stream-input?model_id={model}"
    context_id = "test-context-123"
    
    async with websockets.connect(url) as ws:
        # Send initial config
        bos_message = {
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
            "xi_api_key": api_key,
            "context_id": context_id
        }
        await ws.send(json.dumps(bos_message))
        
        # Send text
        text_message = {
            "text": "Hello, this is a multi-stream websocket test. Please respond.",
            "try_trigger_generation": True,
            "context_id": context_id
        }
        await ws.send(json.dumps(text_message))
        
        # EOS
        eos_message = {"text": "", "context_id": context_id}
        await ws.send(json.dumps(eos_message))
        
        print("Sent messages to ElevenLabs WS.")
        
        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=10)
                data = json.loads(response)
                if data.get("audio"):
                    print(f"✅ Received audio chunk: {len(base64.b64decode(data['audio']))} bytes")
                elif data.get("isFinal"):
                    print("🏁 Received final message.")
                    break
                else:
                    print(f"Received other message: {data}")
        except asyncio.TimeoutError:
            print("Timeout waiting for ElevenLabs WS response.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_elevenlabs_ws())
