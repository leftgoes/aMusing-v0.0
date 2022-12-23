import cv2
import numpy as np
import os

from .common import StrPath


class Edit:
    def __init__(self, source: StrPath = 'frames', target: StrPath = 'frames_new') -> None:
        self.source = source
        self.target = target
    
    def process(self, ext: str = '.png') -> None:
        filepaths = [path for path in os.listdir(self.source) if path.endswith(ext)]
        
        if not os.path.exists(self.target):
            os.mkdir(self.target)

        for path in filepaths:
            alpha = cv2.imread(os.path.join(self.source, path), -1)[:, :, 3]

            cv2.imwrite(os.path.join(self.target, path), alpha)
