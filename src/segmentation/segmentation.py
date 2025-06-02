import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from featureExtraction.featureExtraction import getDevice, getPath

NUM_CLASSES = 3

TARGET_WIDTH = 224
TARGET_HEIGHT = 224

def readImage(imagePath: str):
    img = Image.open(imagePath)
    width = img.size[0]
    height = img.size[1]

    img.convert('RGB')
    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT))
    img = np.array(img).astype(np.float32)
    img /= 255.0

    # reshape from (h, w, c) to (c, h, w)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)

    return img, width, height


def cropImage(imagePath: str, predictor):
    try:
        predictionImage, orgWidth, orgHeight = readImage(imagePath)
        tensorImage = torch.from_numpy(predictionImage)
        
        tensorImage = tensorImage.to(getDevice())
        outputs = predictor([tensorImage])
        
        if len(outputs[0]['boxes']) == 0:
            return

        box = outputs[0]["boxes"][0]
        predictedClass = outputs[0]["labels"][0]
        type = "fluke" if predictedClass==1 else "flank"

        # Inflate the boxes back up to the original image size
        box[0] = box[0] * (orgWidth / TARGET_WIDTH)
        box[1] = box[1] * (orgHeight / TARGET_HEIGHT)
        box[2] = box[2] * (orgWidth / TARGET_WIDTH)
        box[3] = box[3] * (orgHeight / TARGET_HEIGHT)

        # Inflate the boxes by 1% to help with predictions segmented slightly too tight.
        inflationX = (box[2] - box[0]) * 0.01
        inflationY = (box[3] - box[1]) * 0.01

        box[0] = max(0, box[0] - inflationX)
        box[1] = max(0, box[1] - inflationY)
        box[2] = min(orgWidth, box[2] + inflationX)
        box[3] = min(orgHeight, box[3] + inflationY)


        return {
            "path": imagePath,
            "croppingDimensions": box.detach().cpu(),
            "type": type,
        }

    except Exception as e:
        print("WARNING: Failed to segment image.")
        print(e)


class segmentor:
    def __init__(self):
        # load faster rcnn model segmentation.pth
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES) 

        model.load_state_dict(torch.load(getPath()+"/src/models/segmentation.pth", map_location=torch.device(getDevice()), weights_only=True))
        model.to(getDevice())
        model.eval()

        self.predictor = model


    def crop(self, imagePath: str):
        return cropImage(imagePath, self.predictor)
