import numpy as np
from numpy.core.fromnumeric import resize
import torchvision.transforms as T
from imgaug import augmenters as iaa
from torchvision.transforms.transforms import RandomCrop

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Sometimes(0.8, iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ])),
            iaa.Sometimes(0.5, iaa.Sequential([
                iaa.Crop(percent=(0.1, 0.2))
            ])),
            iaa.LinearContrast((0.75, 1.5)),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.8,
                          iaa.Affine(
                              scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                              translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                              rotate=(-25, 25),
                              shear=(-8, 8)
                          )),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img

basic_transform_rgb = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

basic_transform_gray = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5), (0.5))
])

val_transform = T.Compose([
    T.RandomCrop(3, 32, 32),
    T.ToTensor()
])

train_transform = T.Compose([
    ImgAugTransform(),
    T.RandomCrop(3, 32, 32),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])