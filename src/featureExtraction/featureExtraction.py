from torchvision import models, transforms
from PIL import Image
import torch
from torch import nn
import os

def getDevice():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def getPath():
    if os.path.exists("./assets/identifier"):
        return "./assets/identifier"
    
    if os.path.exists("../identifier"):
        return "../identifier"
    
    return "./resources/assets/identifier"

class L2_norm(nn.Module):
    def __init__(self):
        super(L2_norm, self).__init__()

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=1)


def loadModel(modelPath):
    model = models.resnet101(weights=None)

    model.fc = nn.Sequential(nn.Linear(
        in_features=model.fc.in_features, out_features=128, bias=False), L2_norm())
    model = nn.DataParallel(model)

    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint)

    model = model.to(getDevice())
    model.eval()

    return model


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocessImage(image):
    image_preprocessed = preprocess(image)
    image_tensor = torch.unsqueeze(image_preprocessed, 0)

    return image_tensor


class extractor:
    def __init__(self,
                 flukePath=getPath()+"/src/models/pm_fluke.pth",
                 flankPath=getPath()+"/src/models/pm_flank.pth"
            ):
        flukeModel = loadModel(flukePath)
        flankModel = loadModel(flankPath)
        self.models = {
            "fluke": flukeModel,
            "flank": flankModel,
        }


    def extract(self, whale):
        try:
            image = Image.open(whale["path"])
            box = whale["croppingDimensions"]
            croppedImage = image.crop((box[0].item(), box[1].item(), box[2].item(), box[3].item()))

            image_tensor = preprocessImage(croppedImage).to(getDevice())

            model = self.models[whale["type"]]
            out = model(image_tensor).cpu().tolist()[0]

            whale.update({"embedding": out})
            return whale
        
        except Exception as e:
            print("WARNING: Failed to extract image.")
            print(e)
