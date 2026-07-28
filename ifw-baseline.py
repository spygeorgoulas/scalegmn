'''
cd /home/intern/spygeorgoulas/thesis-metanets/scalegmn
python ifw-baseline.py
'''

from pathlib import Path
import random

import numpy as np
import torch


DATASET_DIR = Path("/home/intern/spygeorgoulas/thesis-metanets/neural-fields-3d/data/ifw")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 0
NUM_T_OUT = 5
EPS = 1e-12


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    diff = (pred - target).reshape(-1)
    target = target.reshape(-1)
    return (diff.norm() / (target.norm() + eps)).item()


def split_train_val_test(files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    return train_files, val_files, test_files


def main():
    files = sorted(DATASET_DIR.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in: {DATASET_DIR}")

    train_files, val_files, test_files = split_train_val_test(
        files,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    total_last = 0.0
    total_identity = 0.0
    total_zero = 0.0

    for npz_path in val_files:
        with np.load(npz_path) as data:
            velocity_in = torch.tensor(data["velocity_in"], dtype=torch.float32)
            velocity_out = torch.tensor(data["velocity_out"], dtype=torch.float32)

        pred_last = velocity_in[-1:].repeat(NUM_T_OUT, 1, 1)
        pred_identity = velocity_in[:NUM_T_OUT]
        pred_zero = torch.zeros_like(velocity_out)

        total_last += relative_l2(pred_last, velocity_out, eps=EPS)
        total_identity += relative_l2(pred_identity, velocity_out, eps=EPS)
        total_zero += relative_l2(pred_zero, velocity_out, eps=EPS)

    n_val = len(val_files)

    print(f"Total files found: {len(files)}")
    print(f"Train/Val/Test: {len(train_files)}/{len(val_files)}/{len(test_files)}")
    print(f"Seed: {SEED}")
    print()
    print(f"RL2 baseline | repeat last input frame: {total_last / n_val:.6f}")
    print(f"RL2 baseline | copy input window:       {total_identity / n_val:.6f}")
    print(f"RL2 baseline | zeros:                   {total_zero / n_val:.6f}")


if __name__ == "__main__":
    main()