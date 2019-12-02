import os
import sys
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs import config
from dataset.base_augs import get_preprocessing


from dataset.dataset import Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch


class InterpolateWrapper(torch.nn.Module):
    def __init__(self, model, step=32):
        super().__init__()

        self.model = model
        self.step = step

    def forward(self, x):
        initial_size = list(x.size()[-2:])
        interpolated_size = [(d // self.step) * self.step for d in initial_size]

        x = torch.nn.functional.interpolate(x, interpolated_size)
        x = self.model(x)
        x = torch.nn.functional.interpolate(x, initial_size)

        return x


model_path = os.path.join(config.MODELS, "base.pth")
best_model = torch.load(model_path)



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
ACTIVATION = 'softmax' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

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
    smp.utils.metrics.IoUMetric(threshold=0.5),
]


test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)
print("test")
logs = test_epoch.run(test_dataloader)
