import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from .image_dataset import ImageDataset


class BasicDataset(Dataset):
    def __init__(self, subset, transform = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]['image'], self.subset[index]['targets']
        if self.transform:
            x = self.transform(x)
        return {
            "image" : x,
            "targets" : y
        }

    def __len__(self):
        return len(self.subset)

def get_dataloaders(
        train_dir,
        test_dir,
        train_transform = None,
        test_transform = None,
        split = (0.8, 0.2),
        **kwargs
    ):
    """
    This function returns the train, val and test dataloaders.
    """
    # Generate the datasets
    train_dataset = ImageDataset(train_dir, train_transform)
    test_dataset = ImageDataset(test_dir, test_transform)

    # Split the train_dataset in train and val
    lengths = np.array(split) * len(train_dataset)
    lengths = lengths.astype(int)
    left = len(train_dataset) - lengths.sum()
    lengths[-1] += left

    train_dataset, val_dataset = random_split(train_dataset, lengths.tolist())
    train_dataset = BasicDataset(train_dataset, transform = train_transform)
    val_dataset = BasicDataset(val_dataset, transform = test_transform)

    train_dl = DataLoader(train_dataset, **kwargs)
    val_dl = DataLoader(val_dataset, **kwargs)
    test_dl = DataLoader(test_dataset, **kwargs)

    return train_dl, val_dl, test_dl