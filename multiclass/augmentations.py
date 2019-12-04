from pathlib import Path

import albumentations as albu
from albumentations.pytorch import ToTensor

from configs import config
from multiclass.show import show_random


def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]


def hard_transforms():
    result = [
        albu.RandomRotate90(),
        albu.Cutout(),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        albu.GridDistortion(p=0.3),
        albu.HueSaturationValue(p=0.3)
    ]

    return result


def resize_transforms(image_size=224):
    BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose([
        albu.SmallestMaxSize(pre_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

    ])

    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    random_crop_big = albu.Compose([
        albu.LongestMaxSize(pre_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

    ])

    # Converts the image to a square of size image_size x image_size
    result = [
        albu.OneOf([
            random_crop,
            rescale,
            random_crop_big
        ], p=1)
    ]

    return result


def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensor()]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result


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

    show_transforms = compose([resize_transforms(), hard_transforms()])

    show_random(ALL_IMAGES, ALL_MASKS, transforms=show_transforms)
