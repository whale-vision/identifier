import asyncio
import os
import json

from featureExtraction.featureExtraction import extractor
from identification.identification import identifier, identityCreator
from segmentation.segmentation import segmentor


def getListOfFiles(directory):
    whaleDict = {}

    # Walk through the directory tree
    for root, _, files in os.walk(directory):
        # Get the relative path from the root directory
        relativePath = os.path.relpath(root, directory)
        # Split the relative path to find the first subdirectory (whale name)
        pathParts = relativePath.split(os.sep)

        if len(pathParts) > 0 and pathParts[0] != ".":
            whaleName = pathParts[0]  # First subdirectory is the whale name

            # Initialize the list for this whale if it doesn't exist
            if whaleName not in whaleDict:
                whaleDict[whaleName] = []

            # Add each file to the whale's list
            for file in files:
                fullPath = os.path.join(root, file)
                whaleDict[whaleName].append((whaleName, fullPath))

    return whaleDict


async def mapProgress(func, iterable, processName, websocket):
    i = 0
    newIterable = []
    for item in iterable:
        print(f"{processName}: {i}/{len(iterable)}")

        if (item == None):
            continue

        newIterable.append(func(item))

        i += 1 
        await websocket.send(f"{processName}: {i}/{len(iterable)}")
        await asyncio.sleep(0)

    return newIterable

async def segmentImages(imagePaths: list, websocket):
    imageSegmentor = segmentor()
    return await mapProgress(imageSegmentor.crop, imagePaths, "segmenting", websocket)

async def extractImages(images: list, websocket):
    imageExtractor = extractor()
    return await mapProgress(imageExtractor.extract, images, "extracting", websocket)

async def identifyImages(images: list, websocket):
    imageIdentifier = identifier()

    [imageIdentifier.addIdentity(image) for image in images if "identity" in image]
    imageIdentifier.calculateAverages()

    otherImages = [image for image in images if not "identity" in image]

    return await mapProgress(imageIdentifier.identify, otherImages, "identifying", websocket)


def generateIdentificationFile(path: str):
    whales = getListOfFiles(path)

    whales = segmentImages(whales)
    whales = filter(lambda whale: whale != None, whales)

    whaleFlukes = filter(lambda whale: whale["type"] == "fluke", whales)
    whaleFlukes = list(whaleFlukes)

    whaleFlukes = extractImages(whaleFlukes)
    whaleFlukes = filter(lambda whale: whale != None, whaleFlukes)
    whaleFlukes = list(whaleFlukes)

    flukeIdentifier = identityCreator()
    mapProgress(flukeIdentifier.addIdentity, whaleFlukes, "creating fluke identity")

    # flukeIdentifier.calculateMean("identification_flank.csv")



async def extractWhales(whales: list[str], websocket):
    whaleSegmented = await segmentImages(whales, websocket)
    whaleExtracted = await extractImages(whaleSegmented, websocket)

    await websocket.send(json.dumps([{
            "path": whale["path"],
            "type": whale["type"],
            "embedding": whale["embedding"]
        } for whale in whaleExtracted]))


async def identifyWhales(whales, websocket):
    whaleIdentified = await identifyImages(whales, websocket)


    await websocket.send(json.dumps([{
            "path": whale["path"],
            "identities": whale["distances"]
        } for whale in whaleIdentified]))


