import collections
import numpy as np
from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from configs import config
from multiclass.augmentations import compose, resize_transforms, hard_transforms, post_transforms, pre_transforms
from multiclass.segmentationDataset import SegmentationDataset


def get_loaders(
        images: List[Path],
        masks: List[Path],
        random_state: int,
        valid_size: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transforms_fn=None,
        valid_transforms_fn=None,
) -> dict:
    indices = np.arange(len(images))

    # Let's divide the data set into train and valid parts.
    train_indices, valid_indices = train_test_split(
        indices, test_size=valid_size, random_state=random_state, shuffle=True
    )

    np_images = np.array(images)
    np_masks = np.array(masks)

    # Creates our train dataset
    train_dataset = SegmentationDataset(
        images=np_images[train_indices].tolist(),
        masks=np_masks[train_indices].tolist(),
        transforms=train_transforms_fn
    )

    # Creates our valid dataset
    valid_dataset = SegmentationDataset(
        images=np_images[valid_indices].tolist(),
        masks=np_masks[valid_indices].tolist(),
        transforms=valid_transforms_fn
    )

    # Catalyst uses normal torch.data.DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    # And excpect to get an OrderedDict of loaders
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders


if __name__ == "__main__":
    ROOT = Path(config.DATA_PATH)

    train_image_path = ROOT / "train"
    train_mask_path = ROOT / "trainannot"
    test_image_path = ROOT / "test"

    ALL_IMAGES = sorted(train_image_path.glob("*.png"))
    ALL_MASKS = sorted(train_mask_path.glob("*.png"))

    train_transforms = compose([
        resize_transforms(),
        hard_transforms(),
        post_transforms()
    ])
    valid_transforms = compose([pre_transforms(), post_transforms()])

    loaders = get_loaders(
        images=ALL_IMAGES,
        masks=ALL_MASKS,
        random_state=config.SEED,
        train_transforms_fn=train_transforms,
        valid_transforms_fn=valid_transforms,
        batch_size=config.BATCH_SIZE
    )
