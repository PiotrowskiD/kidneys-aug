from pathlib import Path

import segmentation_models_pytorch as smp
from catalyst import utils

from torch import nn
from torch import optim

from catalyst.contrib.optimizers import RAdam, Lookahead
from catalyst.contrib.criterion import DiceLoss, IoULoss
from catalyst.dl.callbacks import DiceCallback, IouCallback, \
    CriterionCallback, CriterionAggregatorCallback

from catalyst.dl import SupervisedRunner

# We will use Feature Pyramid Network with pre-trained ResNeXt50 backbone
from configs import config
from multiclass.augmentations import compose, resize_transforms, hard_transforms, post_transforms, pre_transforms
from multiclass.loaders import get_loaders

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

model = smp.Unet(encoder_name="resnext50_32x4d", classes=2, activation='sigmoid')

# we have multiple criterions
criterion = {
    "dice": DiceLoss(),
    "iou": IoULoss(),
    "bce": nn.BCEWithLogitsLoss()
}

learning_rate = 0.001
encoder_learning_rate = 0.0005

# Since we use a pre-trained encoder, we will reduce the learning rate on it.
layerwise_params = {"encoder*": dict(lr=encoder_learning_rate, weight_decay=0.00003)}

# This function removes weight_decay for biases and applies our layerwise_params
model_params = utils.process_model_params(model, layerwise_params=layerwise_params)

# Catalyst has new SOTA optimizers out of box
base_optimizer = RAdam(model_params, lr=learning_rate, weight_decay=0.0003)
optimizer = Lookahead(base_optimizer)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

num_epochs = 3
logdir = "./logs/segmentation"

device = utils.get_device()
print(f"device: {device}")

# by default SupervisedRunner uses "features" and "targets",
# in our case we get "image" and "mask" keys in dataset __getitem__
runner = SupervisedRunner(device=device, input_key="image", input_target_key="mask")

SEED = config.SEED
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,

    # our dataloaders
    loaders=get_loaders(
        images=ALL_IMAGES,
        masks=ALL_MASKS,
        random_state=SEED,
        train_transforms_fn=train_transforms,
        valid_transforms_fn=valid_transforms,
        batch_size=config.BATCH_SIZE
    ),

    callbacks=[
        # Each criterion is calculated separately.
        CriterionCallback(
            input_key="mask",
            prefix="loss_dice",
            criterion_key="dice"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_iou",
            criterion_key="iou"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_bce",
            criterion_key="bce",
            multiplier=0.8
        ),

        # And only then we aggregate everything into one loss.
        CriterionAggregatorCallback(
            prefix="loss",
            loss_keys=["loss_dice", "loss_iou", "loss_bce"],
            loss_aggregate_fn="sum"  # or "mean"
        ),

        # metrics
        DiceCallback(input_key="mask"),
        IouCallback(input_key="mask"),
    ],
    # path to save logs
    logdir=logdir,

    num_epochs=num_epochs,

    # save our best checkpoint by IoU metric
    main_metric="iou",
    # IoU needs to be maximized.
    minimize_metric=False,

    # prints train logs
    verbose=False
)
