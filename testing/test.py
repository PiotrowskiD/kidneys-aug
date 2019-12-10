import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs import config
from dataset.base_augs import get_preprocessing

from dataset.dataset import Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch
import matplotlib.pyplot as plt


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        if np.shape(image)[0] == 2:
            plt.imshow(image[1, :, :])
        else:
            plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    show_images = "false"
    DATA_DIR = Path(config.DATA_PATH)
    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')

    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['kidney', 'tumor']
    ACTIVATION = 'sigmoid'
    DEVICE = 'cuda'

    for file in os.listdir(config.MODELS):
        if file.endswith(".pth"):
            best_model = torch.load(file)
            preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

            test_dataset = Dataset(
                x_test_dir,
                y_test_dir,
                preprocessing=get_preprocessing(preprocessing_fn),
                classes=CLASSES,
            )

            test_dataloader = DataLoader(test_dataset)

            loss = smp.utils.losses.DiceLoss()
            metrics = [
                smp.utils.metrics.IoUMetric(eps=1., threshold=0.5),
            ]

            test_epoch = smp.utils.train.ValidEpoch(
                model=best_model,
                loss=loss,
                metrics=metrics,
                device=DEVICE,
            )

            logs = test_epoch.run(test_dataloader)
            if show_images:

                test_dataset_vis = Dataset(
                    x_test_dir, y_test_dir,
                    classes=CLASSES,
                )

                for i in range(5):
                    n = np.random.choice(len(test_dataset))
                    image_vis = test_dataset_vis[n][0].astype('uint8')
                    image, gt_mask = test_dataset[n]

                    gt_mask = gt_mask.squeeze()

                    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
                    pr_mask = best_model.predict(x_tensor)
                    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

                    visualize(
                        image=image_vis,
                        ground_truth_mask=gt_mask,
                        predicted_mask=pr_mask
                    )
