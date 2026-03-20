import asyncio
import websockets

async def test_ws():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket.")
        # Send a text trigger
        await websocket.send("Hello Jessica, this is a test. Please speak.")
        print("Message sent.")
        
        try:
            # Wait for messages
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=15)
                if isinstance(response, bytes):
                    print(f"Received binary data: {len(response)} bytes")
                else:
                    print(f"Received text data: {response}")
        except asyncio.TimeoutError:
            print("No more data received after 15 seconds.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ws())
