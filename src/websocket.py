import asyncio
import websockets
import json
from fileHandler import saveWhaleIdentities 
from main import extractWhales, identifyWhales

from featureExtraction.featureExtraction import extractor, getDevice, getPath
from identification.identification import identifier
from segmentation.segmentation import segmentor

def curriedHandle(imageSegmentor, imageExtractor, imageIdentifier):
    async def handle(websocket):
        async for message in websocket:
            try: 
                content = json.loads(message)
                print(f"received: {content["type"]}")

                if (content["type"] == "extract"):
                    await websocket.send(f"starting on {getDevice()} with {getPath()}")
                    
                    fileNames = content["fileNames"]
                    await extractWhales(fileNames, websocket, imageSegmentor, imageExtractor)

                elif (content["type"] == "identify"):
                    await websocket.send(f"starting on {getDevice()} with {getPath()}")

                    data = content["data"]
                    await identifyWhales(data, websocket, imageIdentifier)

                elif (content["type"] == "save"):
                    whaleIdentities = content["whaleIdentities"]
                    await saveWhaleIdentities(whaleIdentities, websocket)
                

                await websocket.send(f"complete!")
            except Exception as e:
                print(e)
                await websocket.send(f"error: {e}")

    return handle

async def main():
    imageSegmentor = segmentor()
    imageExtractor = extractor()
    imageIdentifier = identifier()

    print(f"Starting WebSocket server on ws://localhost:8765")
    
    async with websockets.serve(curriedHandle(imageSegmentor, imageExtractor, imageIdentifier), "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
