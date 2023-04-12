"""
Class for using pretrained pytorch yolov5
https://pytorch.org/hub/ultralytics_yolov5/
"""
import numpy as np
import torch

# Relevant documentation: https://github.com/ultralytics/yolov5/issues/36
# Such as running on cpu/cuda with model.cpu() / model.cuda()

models = {
    "nano": "yolov5n",
    "small": "yolov5s",
    "medium": "yolov5m",
    "large": "yolov5l",
    "xlarge": "yolov5x"
}

class Model:
    def __init__(self, model_name):
        model_actual_name = models[model_name]
        self.model = torch.hub.load(
            'ultralytics/yolov5', model_actual_name, pretrained=True)
        device = self.model.parameters().__next__().device
        print(f"Initialized model {model_actual_name} on device {device}")

    def forward(self, image: np.ndarray) -> object:
        return self.model(image)
