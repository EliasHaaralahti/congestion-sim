"""
Class for using pretrained pytorch yolov5
https://pytorch.org/hub/ultralytics_yolov5/
"""
import numpy as np
import torch


# TODO: To fix the current issue where the input image is of wrong type
# https://github.com/ultralytics/yolov5/issues/36 detailed version could work.

models = {
    "nano": "yolov5x",
    "small": "yolov5s",
    "medium": "yolov5m",
    "large": "yolov5l",
    "xlarge": "yolov5x"
}

class Model:
    def __init__(self, model_name):
        model = models[model_name]
        self.model = torch.hub.load(
            'ultralytics/yolov5', model, pretrained=True)

    def forward(self, image: np.ndarray) -> object:
        return self.model(image)
