import os
import random
from glob import glob
from pathlib import Path
import shutil
import cv2
from tqdm import tqdm
import numpy as np
from configs import config

DATA_DEST = config.DATA_PATH
paths = glob(config.ORIGINAL_DATA + '/visualization/case_*/*')

for img_path in tqdm(paths):

    if not os.path.basename(img_path).startswith("img"):
        continue
    seg_img = cv2.imread(img_path)
    if np.shape(seg_img)[2]> 512:
        pass