import os
from random import randrange, uniform, random
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFilter, ImageEnhance, ImageOps
from albumentations import (
    GridDistortion, ElasticTransform,
    IAAAffine)

from configs import config


def warp(image, mask, alpha=40, sigma=5):
    aug = ElasticTransform(alpha=alpha, sigma=sigma, p=1.)
    augmented = aug(image=image, mask=mask)
    return augmented['image'], augmented['mask']


def flip_lr(image, mask):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    msk = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_pil = Image.fromarray(msk)
    return np.asarray(im_pil.transpose(Image.FLIP_LEFT_RIGHT)), np.asarray(mask_pil.transpose(Image.FLIP_LEFT_RIGHT))


def flip_tb(image, mask):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    msk = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_pil = Image.fromarray(msk)
    return np.asarray(im_pil.transpose(Image.FLIP_LEFT_RIGHT)), np.asarray(mask_pil.transpose(Image.FLIP_TOP_BOTTOM))


def rotate(image, mask, angle=None):
    if not angle:
        angles = [90, 180, 270]
        angle = random.choice(angles)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    msk = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_pil = Image.fromarray(msk)
    if angle == 270:
        return np.asarray(im_pil.transpose(Image.ROTATE_270)), np.asarray(mask_pil.transpose(Image.ROTATE_270))
    if angle == 180:
        return np.asarray(im_pil.transpose(Image.ROTATE_180)), np.asarray(mask_pil.transpose(Image.ROTATE_180))
    return np.asarray(im_pil.transpose(Image.ROTATE_90)), np.asarray(mask_pil.transpose(Image.ROTATE_90))


def rotate_rnd(image, mask):
    angle = randrange(0, 360)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    msk = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_pil = Image.fromarray(msk)
    return np.asarray(im_pil.rotate(angle)), np.asarray(mask_pil.rotate(angle))


def blur(image, mask, radius=None):
    if not radius:
        radius = randrange(0, 3)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    msk = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_pil = Image.fromarray(msk)
    return np.asarray(im_pil.filter(ImageFilter.GaussianBlur(radius))), np.asarray(
        mask_pil.filter(ImageFilter.GaussianBlur(radius)))


def sharp(image, mask):
    factor = uniform(1.1, 1.9)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    msk = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_pil = Image.fromarray(msk)

    enhancer = ImageEnhance.Sharpness(im_pil)
    enhancer_msk = ImageEnhance.Sharpness(mask_pil)

    return np.asarray(enhancer.enhance(factor)), np.asarray(enhancer_msk.enhance(factor))


def smooth(image, mask):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    msk = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_pil = Image.fromarray(msk)
    return np.asarray(im_pil.filter(ImageFilter.SMOOTH)), np.asarray(mask_pil.filter(ImageFilter.SMOOTH))


def posterize(image, mask, bits=None):
    if not bits:
        bits = randrange(5, 8)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    msk = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_pil = Image.fromarray(msk)
    return np.asarray(ImageOps.posterize(im_pil, bits)), np.asarray(ImageOps.posterize(mask_pil, bits))


def colour_adj(image, mask):
    factor = uniform(0.65, 0.99)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    msk = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_pil = Image.fromarray(msk)

    enhancer = ImageEnhance.Color(im_pil)
    enhancer_msk = ImageEnhance.Color(mask_pil)

    return np.asarray(enhancer.enhance(factor)), np.asarray(enhancer_msk.enhance(factor))


def brightness_adj(image, mask):
    factor = uniform(0.65, 0.99)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    msk = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_pil = Image.fromarray(msk)

    enhancer = ImageEnhance.Brightness(im_pil)
    enhancer_msk = ImageEnhance.Brightness(mask_pil)

    return np.asarray(enhancer.enhance(factor)), np.asarray(enhancer_msk.enhance(factor))


def normalize_contrast(image, mask):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    msk = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_pil = Image.fromarray(msk)
    return np.asarray(ImageOps.autocontrast(im_pil)), np.asarray(ImageOps.autocontrast(mask_pil))


def contrast_adj(image, mask):
    factor = uniform(0.65, 0.99)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    msk = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_pil = Image.fromarray(msk)

    enhancer = ImageEnhance.Contrast(im_pil)
    enhancer_msk = ImageEnhance.Contrast(mask_pil)

    return np.asarray(enhancer.enhance(factor)), np.asarray(enhancer_msk.enhance(factor))


def normalize_contrast(image, mask):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    msk = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_pil = Image.fromarray(msk)
    return np.asarray(ImageOps.equalize(im_pil)), np.asarray(ImageOps.equalize(mask_pil))


def grid_distortion(image, mask, num_steps=5, limit=0.3):
    aug = GridDistortion(num_steps=num_steps, distort_limit=limit, p=1.)
    augmented = aug(image=image, mask=mask)
    return augmented['image'], augmented['mask']


def affine(image, mask):
    aug = IAAAffine(mode="constant", p=1.)
    augmented = aug(image=image, mask=mask)
    return augmented['image'], augmented['mask']


if __name__ == '__main__':
    aug_list = config.AUGS

    for key, func in aug_list.items():
        print(key)
        img = cv2.imread(os.path.join(config.DATA_PATH, 'test/img_case_00000_00291.png'))
        mask = cv2.imread(os.path.join(config.DATA_PATH, 'testannot/img_case_00000_00291.png'))
        plt.imshow(img)
        plt.show()
        img, mask = func(img, mask)
        plt.imshow(img)
        plt.show()
