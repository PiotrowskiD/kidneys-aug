import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.dataset import Dataset
from torch.utils.data import DataLoader

import torch


model_path = ('saved_models/base.pth')
best_model = torch.load(model_path)

test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)

test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)
