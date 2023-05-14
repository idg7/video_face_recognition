import os
from glob import glob
from tqdm import tqdm
import numpy as np
from typing import List
from argparse import ArgumentParser, Namespace
import shutil


def transfer_datapoints(dest_dataset_loc: str, source_path: str, data_points: List[str]) -> None:
    """
    Create a symlink (or copy, if symlink cannot be made) for all data_points in source_path to the new dest_dataset_loc

    :dest_dataset_loc: The destination directory, (where to move the files to)
    :source_path: The source directory containing the files
    :data_points: The absolute path of all files to copy
    """
    # pbar = tqdm(data_points)
    for point in data_points:
        dest_point = os.path.join(dest_dataset_loc, os.path.relpath(point, source_path))
        os.makedirs(os.path.dirname(dest_point), exist_ok=True)
        try:
            os.symlink(os.path.abspath(point), dest_point)
        except OSError:
            # pbar.set_description("Encountered error on point. Copying...")
            shutil.copyfile(point, dest_point)


def globall(path: str) -> List[str]:
    return glob(os.path.join(path, '*'))


def n_to_one_split(src_dir: str, dest_dir: str) -> None:
    classes = globall(src_dir)

    for cls in tqdm(classes):
        vids = globall(cls)
        
        val_vid = np.random.choice(vids, 1, replace=False)
        train_vids = np.setdiff1d(vids, val_vid)
        print(train_vids)
        if len(train_vids) > 0:
            for n in range(len(train_vids), 0, -1):
                n_dir = os.path.join(dest_dir, f'{n}_train_1_val')
                print(n_dir)
                
                val_dir = os.path.join(n_dir, 'val')
                train_dir = os.path.join(n_dir, 'train')

                os.makedirs(os.path.dirname(train_dir), exist_ok=True)
                os.makedirs(os.path.dirname(val_dir), exist_ok=True)
                
                transfer_datapoints(val_dir, src_dir, val_vid)
                train_vids = np.random.choice(train_vids, n, replace=False)
                transfer_datapoints(train_dir, src_dir, train_vids)


def over_n_to_one_split(src_dir: str, dest_dir: str) -> None:
    classes = globall(src_dir)

    for cls in tqdm(classes):
        vids = globall(cls)
        
        val_vid = np.random.choice(vids, 1, replace=False)
        train_vids = np.setdiff1d(vids, val_vid)
        print(train_vids)
        if len(train_vids) > 0:
            for n in range(len(train_vids), 0, -1):
                n_dir = os.path.join(dest_dir, f'over_{n}_train_1_val')
                print(n_dir)
                
                val_dir = os.path.join(n_dir, 'val')
                train_dir = os.path.join(n_dir, 'train')

                os.makedirs(os.path.dirname(train_dir), exist_ok=True)
                os.makedirs(os.path.dirname(val_dir), exist_ok=True)
                
                transfer_datapoints(val_dir, src_dir, val_vid)
                # train_vids = np.random.choice(train_vids, n, replace=False)
                transfer_datapoints(train_dir, src_dir, train_vids)

def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('dataset_src', type=str, help='Directory containing the raw dataset')
    parser.add_argument('dataset_dest', type=str, help='Directory containing processed dataset')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print(args)
    over_n_to_one_split(args.dataset_src, args.dataset_dest)