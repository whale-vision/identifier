import asyncio
from featureExtraction.featureExtraction import getDevice
import websockets
import json
from fileHandler import saveWhaleIdentities 
from main import identifyWhales

async def handle(websocket, _):
    async for message in websocket:
        try: 
            content = json.loads(message)

            if (content["type"] == "identify"):
                await websocket.send(f"starting on {getDevice()}")
                
                fileNames = content["fileNames"]
                await identifyWhales(fileNames, websocket)

            elif (content["type"] == "save"):
                whaleIdentities = content["whaleIdentities"]
                await saveWhaleIdentities(whaleIdentities, websocket)
            

            await websocket.send(f"complete!")
        except Exception as e:
            print(e)
            await websocket.send(f"error: {e}")

start_server = websockets.serve(handle, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
