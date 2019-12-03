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
import cv2

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

    def predict(self, x):
        initial_size = list(x.size()[-2:])
        interpolated_size = [(d // self.step) * self.step for d in initial_size]

        x = torch.nn.functional.interpolate(x, interpolated_size)
        x = self.model.predict(x)
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
ACTIVATION = 'softmax'  # could be None for logits or 'softmax2d' for multicalss segmentation
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

#logs = test_epoch.run(test_dataloader)

test_dataset_vis = Dataset(
    x_test_dir, y_test_dir,
    classes=CLASSES,
)


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
<<<<<<< Updated upstream
    plt.show()
=======
    plt.savefig('test_imgs/' + prefix + "viz.png")
>>>>>>> Stashed changes


for i in range(5):
    n = np.random.choice(len(test_dataset))

    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]

    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
<<<<<<< Updated upstream
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    for k in range(512):
        for j in range(512):
            if gt_mask[0,i,j] == 0:
                print(i)
                print(j)
    # visualize(
    #     image=image_vis,
    #     ground_truth_mask=gt_mask,
    #     predicted_mask=pr_mask
    # )
=======
    pr_mask = best_model.model.predict(x_tensor)
   
    #pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    pr_kidney = pr_mask[0,:,:]
    norm_mask = cv2.normalize(pr_kidney, None)
    norm_mask = norm_mask.astype(np.uint8)    
    for i in range(1,255):
        if i in pr_mask[0,:,:]:
            print(i)

>>>>>>> Stashed changes
