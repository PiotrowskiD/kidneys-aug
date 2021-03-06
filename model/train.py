import os
from pathlib import Path

from torch.utils.data import DataLoader
import torch
import numpy as np
import segmentation_models_pytorch as smp

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs import config
from dataset.base_augs import get_training_augmentation, get_preprocessing, get_validation_augmentation
from dataset.dataset import Dataset

from torch import nn

DATA_DIR = Path(config.DATA_PATH)
x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['kidney', 'tumor']
ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

if __name__ == "__main__":

    aug_list = config.AUGS
    for aug_prob in [0.5, 0.7]:
        for aug_name, aug_func in aug_list.items():
            # create segmentation model with pretrained encoder
            model = smp.Unet(
                encoder_name=ENCODER,
                encoder_weights=ENCODER_WEIGHTS,
                classes=len(CLASSES),
                activation=ACTIVATION,
            )
            model = nn.DataParallel(model)

            preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

            train_dataset = Dataset(
                x_train_dir,
                y_train_dir,
                augmentation=aug_func,
                aug_prob=aug_prob,
                preprocessing=get_preprocessing(preprocessing_fn),
                classes=CLASSES,
            )

            valid_dataset = Dataset(
                x_valid_dir,
                y_valid_dir,
                augmentation=aug_func,
                preprocessing=get_preprocessing(preprocessing_fn),
                classes=CLASSES,
            )

            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
            valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

            # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
            # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

            loss = smp.utils.losses.DiceLoss()
            metrics = [
                smp.utils.metrics.IoUMetric(threshold=0.5, eps=1.),
            ]

            optimizer = torch.optim.Adam([
                dict(params=model.parameters(), lr=0.0001),
            ])

            # create epoch runners
            # it is a simple loop of iterating over dataloader`s samples
            train_epoch = smp.utils.train.TrainEpoch(
                model,
                loss=loss,
                metrics=metrics,
                optimizer=optimizer,
                device=DEVICE,
                verbose=True,
            )

            valid_epoch = smp.utils.train.ValidEpoch(
                model,
                loss=loss,
                metrics=metrics,
                device=DEVICE,
                verbose=True,
            )

            # train model for 40 epochs

            max_score = 0

            for i in range(0, 40):

                print('\nEpoch: {}'.format(i))
                train_logs = train_epoch.run(train_loader)
                valid_logs = valid_epoch.run(valid_loader)

                # do something (save model, change lr, etc.)
                if max_score < valid_logs['iou']:
                    max_score = valid_logs['iou']
                    torch.save(model, os.path.join(config.MODELS, aug_name + "_" + str(aug_prob) + '_model.pth'))
                    print('Model saved!')

                if i == 25:
                    optimizer.param_groups[0]['lr'] = 1e-5
                    print('Decrease decoder learning rate to 1e-5!')
