import random
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread as gif_imread
from catalyst import utils

from configs import config


def show_examples(name: str, image: np.ndarray, mask: np.ndarray):
    plt.figure(figsize=(10, 14))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Image: {name}")

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title(f"Mask: {name}")
    plt.show()


def show(index: int, images: List[Path], masks: List[Path], transforms=None) -> None:
    image_path = images[index]
    name = image_path.name

    image = utils.imread(image_path)
    mask = utils.imread(masks[index])

    if transforms is not None:
        temp = transforms(image=image, mask=mask)
        image = temp["image"]
        mask = temp["mask"]

    show_examples(name, image, mask)


def show_random(images: List[Path], masks: List[Path], transforms=None) -> None:
    length = len(images)
    index = random.randint(0, length - 1)
    show(index, images, masks, transforms)


if __name__ == '__main__':
    ROOT = Path(config.DATA_PATH)

    train_image_path = ROOT / "train"
    train_mask_path = ROOT / "trainannot"
    test_image_path = ROOT / "test"

    ALL_IMAGES = sorted(train_image_path.glob("*.png"))
    ALL_MASKS = sorted(train_mask_path.glob("*.png"))

    show_random(ALL_IMAGES, ALL_MASKS)
