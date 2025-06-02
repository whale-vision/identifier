import asyncio
from featureExtraction.featureExtraction import getDevice, getPath
import websockets
import json
from fileHandler import saveWhaleIdentities 
from main import extractWhales, identifyWhales

async def handle(websocket):
    async for message in websocket:
        try: 
            content = json.loads(message)
            print(f"received: {content["type"]}")

            if (content["type"] == "extract"):
                await websocket.send(f"starting on {getDevice()} with {getPath()}")
                
                fileNames = content["fileNames"]
                await extractWhales(fileNames, websocket)

            elif (content["type"] == "identify"):
                await websocket.send(f"starting on {getDevice()} with {getPath()}")

                data = content["data"]
                await identifyWhales(data, websocket)

            elif (content["type"] == "save"):
                whaleIdentities = content["whaleIdentities"]
                await saveWhaleIdentities(whaleIdentities, websocket)
            

            await websocket.send(f"complete!")
        except Exception as e:
            print(e)
            await websocket.send(f"error: {e}")

async def main():
    async with websockets.serve(handle, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
