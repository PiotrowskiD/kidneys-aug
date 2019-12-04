import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import matplotlib.pyplot as plt

from configs import config


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()



class Dataset(BaseDataset):

    CLASSES = ['kidney', 'tumor']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_ids = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_ids = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_ids[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_ids[i])

        if np.shape(image)[1] != 512:
            image = cv2.resize(image, (512, 512))
            mask = cv2.resize(mask, (512, 512))

        # extract certain classes from mask (e.g. cars)
        mask_kidney = mask[:, :, 0] == 255
        mask_tumor = mask[:, :, 1] == 255
        masks = [mask_kidney, mask_tumor]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)

            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    DATA_DIR = Path(config.DATA_PATH)
    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')

    dataset = Dataset(x_train_dir, y_train_dir, classes=['kidney'])

    image, mask = dataset[127]  # get some sample
    visualize(
        image=image,
        cars_mask=mask.squeeze(),
    )
