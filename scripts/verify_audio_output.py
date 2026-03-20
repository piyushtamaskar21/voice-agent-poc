import asyncio
import websockets
import time

async def test_audio_output():
    uri = "ws://localhost:8000/ws"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Waiting for greeting audio...")
            start_time = time.time()
            chunks_received = 0
            while time.time() - start_time < 15:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    if isinstance(message, bytes):
                        chunks_received += 1
                        print(f"Received audio chunk #{chunks_received}: {len(message)} bytes")
                        if chunks_received >= 5:
                            print("\nSUCCESS: Received multiple audio chunks!")
                            return True
                except asyncio.TimeoutError:
                    print("Timeout waiting for message...")
                    break
            
            if chunks_received > 0:
                print(f"\nSUCCESS: Received {chunks_received} audio chunks total.")
                return True
            else:
                print("\nFAILURE: No audio chunks received.")
                return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_audio_output())
