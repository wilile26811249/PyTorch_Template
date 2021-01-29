import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset:
    def __init__(self,
                 image_paths,
                 targets,
                 resize = None,
                 backend = 'pil',
                 channel_first = True):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.backend = backend
        self.channel_first = channel_first


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        targets = self.targets[index]
        if self.backend == 'pil':
            image = Image.open(self.image_paths)
            if self.resize is not None:
                image = image.resize(self.resize[1], self.resize[0], resample = Image.BILINEAR)
                image = np.array(image)
            elif self.backend == "cv2":
                image = cv2.imread(self.image_paths[index])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if self.resize is not None:
                    image = cv2.resize(image, (self.resize[1], self.resize[0]), interpolation = cv2.INTER_CUBIC)
            else:
                raise Exception("Backend not implemented")

        return {
            "image" : torch.tensor(image),
            "targets" : torch.tensor(targets),
        }
