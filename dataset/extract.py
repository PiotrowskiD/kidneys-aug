from pathlib import Path

from tqdm import tqdm


from starter_code import utils
from starter_code import visualize

DATAPATH = Path('../data/original/')

if __name__ == '__main__':

    for idx in tqdm(range(210)):
        visualize.visualize(
            idx, '../data/visualization/case_{:05d}'.format(idx),
            data_path=DATAPATH,
            separate=True
        )
