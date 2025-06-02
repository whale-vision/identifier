import csv
from functools import partial
import os
import torch
from PIL import Image
from main import getListOfFiles
from fileHandler import saveWhaleIdentities
from featureExtraction.featureExtraction import extractor, getDevice
from identification.identification import identityCreator
from segmentation.segmentation import segmentor
from torchvision import models, transforms

import asyncio
from tqdm import tqdm
import concurrent.futures

def getAllWhales(directory):
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

def segment_whale(whale, imageSegmentor):
    segmentedWhale = imageSegmentor.crop(whale[1])
    if segmentedWhale is None: return None

    segmentedWhale["identity"] = whale[0]
    return segmentedWhale


def extract_whale(whale, imageExtractor):
    return imageExtractor.extract(whale)



def createIdentity(whales, species):
    imageSegmentor = segmentor()
    segmented = [segment_whale(whale, imageSegmentor) for whale in whales]
    segmented = [whale for whale in segmented if whale is not None]

    if species != "pm":
        [whale.update({"type": "flank"}) for whale in segmented]

        imageExtractor = extractor(
            "../Identifier/src/reallyOld/" + species + ".pth",
            "../Identifier/src/reallyOld/" + species + ".pth" 
        )

    else:
        imageExtractor = extractor(
            "G:\Whale Stuff\Identifier\src\\new_old_models\dd_flank.pth" ,
            "G:\Whale Stuff\Identifier\src\\new_old_models\dd_flank.pth",
        )



    extracted = [extract_whale(whale, imageExtractor) for whale in segmented]



    # save embeddings
    for whale in extracted:
        if whale is None:
            continue

        fileName = "../Identifier/src/new_old_models/" + species + "_identification_" + whale["type"] + "_ind.csv"

        with open(fileName, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([whale["identity"], whale["path"], whale["features"]])

    # ((name, path), (features))




def processNetwork(networkInfo):
    # get all whales in file
    whalesByName = getListOfFiles(networkInfo["datasetDirectory"])

    for whaleName in tqdm(whalesByName):
        # print(whaleName)
        createIdentity(whalesByName[whaleName], networkInfo["species"])


if __name__ == "__main__":

    allData = [
        # {
        #     "species": "dd",
        #     "datasetDirectory": "G:\Whale Stuff\_data\cetaceans\Dd\Dd_INDIVIDUALS",
        # },
        # {
        #     "species": "gg",
        #     "datasetDirectory": "G:\Whale Stuff\_data\cetaceans\Gg\Gg_Individuals",
        # },
        {
            "species": "pm",
            "datasetDirectory": "G:\\Whale Stuff\\_data\\cetaceans\\curated\\pm\\all",
        },
        # {
        #     "species": "tt",
        #     "datasetDirectory": "G:\Whale Stuff\_data\cetaceans\Tt\Tt_INDIVIDUALS",
        # },
    ]

    for data in allData:
        print("PROCESSING", data["species"])
        processNetwork(data)
