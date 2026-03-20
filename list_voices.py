import asyncio
import os
import aiohttp
from dotenv import load_dotenv

async def list_voices():
    load_dotenv()
    api_key = os.getenv("ELEVENLABS_API_KEY")
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": api_key}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                voices = await response.json()
                print("Available Voices:")
                for voice in voices["voices"]:
                    print(f"- {voice['name']} (ID: {voice['voice_id']}) - Category: {voice['category']}")
            else:
                print(f"Failed to list voices. Status: {response.status}")
                print(await response.text())

if __name__ == "__main__":
    asyncio.run(list_voices())
