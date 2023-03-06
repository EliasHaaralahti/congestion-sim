"""
Class for using pretrained pytorch yolov5
https://pytorch.org/hub/ultralytics_yolov5/
"""
import numpy as np
import torch


# TODO: To fix the current issue where the input image is of wrong type
# https://github.com/ultralytics/yolov5/issues/36 detailed version could work.


class Model:
    def __init__(self):
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'yolov5x', pretrained=True)

    def forward(self, image: np.ndarray) -> object:
        return self.model(image)
