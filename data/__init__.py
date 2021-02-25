import numpy as np
from .image_dataset import ImageDataset
from torch.utils.data import DataLoader, random_split

def get_dataloaders(
        train_dir,
        test_dir,
        train_transform = None,
        test_transform = None,
        split = (0.7, 0.3),
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

    train_dl = DataLoader(train_dataset, **kwargs)
    val_dl = DataLoader(val_dataset, **kwargs)
    test_dl = DataLoader(test_dataset, **kwargs)

    return train_dl, val_dl, test_dl