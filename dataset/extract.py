from pathlib import Path
import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs import config
from tqdm import tqdm

from starter_code import utils
from starter_code import visualize

DATAPATH = config.ORIGINAL_DATA

if __name__ == '__main__':

    for idx in tqdm(range(210)):
        visualize.visualize(
            idx, DATAPATH + 'visualization-perspectives/case_{:05d}'.format(idx),
            data_path=Path(DATAPATH),
            separate=True
        )
