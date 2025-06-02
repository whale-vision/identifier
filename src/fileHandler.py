import json
import os
import shutil
from tkinter import filedialog

async def saveWhaleIdentities(whaleIdentities, websocket):
    #create dialog to select directory
    directory = filedialog.askdirectory()

    if directory == "":
        return
    
    for identity in whaleIdentities:
        await saveWhaleIdentity(identity, directory, websocket)

async def saveWhaleIdentity(identity, directory, websocket):
    originalImage = identity["file"]
    newPath = os.path.join(directory, identity["selectedIdentity"])
    newFile = os.path.join(newPath, os.path.basename(originalImage))

    if websocket: await websocket.send(f"saving {originalImage} to {newFile}...")

    if not os.path.exists(newPath):
        os.mkdir(newPath)

    shutil.copyfile(originalImage, newFile)

    # save identity to json file
    identityFile = newFile + ".json"
        
    with open(identityFile, "w") as f:
        json.dump(identity, f, indent=4)


