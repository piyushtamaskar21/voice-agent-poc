import asyncio
import os
import base64
import json
import aiohttp
from dotenv import load_dotenv

async def test_elevenlabs():
    load_dotenv()
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = "cgSgspJ2msm6clMCkdW9" # Jessica
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": "Hello, this is a test. If you hear this, ElevenLabs is working correctly.",
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            if response.status == 200:
                print(f"✅ Success! Received audio stream. Status: {response.status}")
                audio_content = await response.read()
                print(f"Received {len(audio_content)} bytes of audio.")
                with open("test_audio.mp3", "wb") as f:
                    f.write(audio_content)
                print("Saved to test_audio.mp3")
            else:
                error_body = await response.text()
                print(f"❌ Failed! Status: {response.status}")
                print(f"Error: {error_body}")

if __name__ == "__main__":
    asyncio.run(test_elevenlabs())
