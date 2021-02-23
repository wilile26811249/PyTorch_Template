import os
import cv2
import torch
import numpy as np
from typing import Callable, Optional, List, Dict, Tuple
from PIL import Image, ImageFile
from torch._C import dtype
from torchvision.transforms.transforms import ToTensor

from .transformation import val_transform
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset:
    def __init__(self,
        root_dir: str,
        targets: Optional[Callable] = None,
        resize: Optional[List[int]] = None,
        backend: Optional[str]= 'pil',
        channel_first: Optional[bool]= True,
        transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.targets = targets
        self.resize = resize
        self.backend = backend
        self.channel_first = channel_first
        self.transform = transform
        self.image_paths = []
        self.targets = []
        self.classes, self.class_to_index = self._find_classes(self.root_dir)

        for target_class in sorted(self.class_to_index.keys()):
            class_index = self.class_to_index[target_class]
            target_dir = os.path.join(self.root_dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, file_names in os.walk(target_dir, followlinks = True):
                for fname in sorted(file_names):
                    self.image_paths.append(os.path.join(root, fname))
                    self.targets.append(class_index)


    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_index = {cls_name : i for i, cls_name in enumerate(classes)}
        return classes, class_to_index


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index):
        targets = self.targets[index]
        if self.backend == 'pil':
            image = Image.open(self.image_paths[index])
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

        if self.transform:
            image = self.transform(image)
        else:
            image = val_transform(image)

        return {
            "image" : torch.as_tensor(image, dtype = torch.float),
            "targets" : torch.as_tensor(targets, dtype = torch.long)
        }