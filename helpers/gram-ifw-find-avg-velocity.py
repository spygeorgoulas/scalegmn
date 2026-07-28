'''
python /home/intern/spygeorgoulas/thesis-metanets/scalegmn/helpers/gram-ifw-find-avg-velocity.py \
  --data_dir /home/intern/spygeorgoulas/thesis-metanets/neural-fields-3d/data/ifw
'''

'''
===== Velocity Statistics =====
Files processed     : 810
Mean velocity       : 37.756039
Std velocity        : 19.710636
Min velocity        : 0.000000
Max velocity        : 194.266403
'''

#!/usr/bin/env python3
"""
Compute average velocity magnitude for GRaM / IFW dataset.

Usage:
python compute_avg_velocity.py \
    --data_dir /path/to/ifw \
    --use velocity_out \
    --max_files 100 \
    --sample_points 20000
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


def compute_velocity_stats(data_dir, use="velocity_out", max_files=None, sample_points=None):
    data_dir = Path(data_dir)

    npz_files = sorted(list(data_dir.glob("**/*.npz")))
    if len(npz_files) == 0:
        raise RuntimeError(f"No .npz files found in {data_dir}")

    if max_files is not None:
        npz_files = npz_files[:max_files]

    all_means = []
    all_vals = []

    for f in tqdm(npz_files, desc="Processing files"):
        data = np.load(f)

        if use not in data:
            raise KeyError(f"{use} not found in {f}")

        vel = data[use]  # shape: (T, N, 3)

        T, N, _ = vel.shape
        vel = vel.reshape(T * N, 3)

        # Optional subsampling for speed
        if sample_points is not None and sample_points < vel.shape[0]:
            idx = np.random.choice(vel.shape[0], sample_points, replace=False)
            vel = vel[idx]

        # Compute magnitude
        mag = np.linalg.norm(vel, axis=1)

        all_means.append(mag.mean())
        all_vals.append(mag)

    all_vals = np.concatenate(all_vals)

    stats = {
        "num_files": len(npz_files),
        "mean_velocity": float(all_vals.mean()),
        "std_velocity": float(all_vals.std()),
        "min_velocity": float(all_vals.min()),
        "max_velocity": float(all_vals.max()),
    }

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--use", type=str, default="velocity_out", choices=["velocity_out", "velocity_in"])
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of files")
    parser.add_argument("--sample_points", type=int, default=None, help="Sample points per file")

    args = parser.parse_args()

    stats = compute_velocity_stats(
        data_dir=args.data_dir,
        use=args.use,
        max_files=args.max_files,
        sample_points=args.sample_points,
    )

    print("\n===== Velocity Statistics =====")
    print(f"Files processed     : {stats['num_files']}")
    print(f"Mean velocity       : {stats['mean_velocity']:.6f}")
    print(f"Std velocity        : {stats['std_velocity']:.6f}")
    print(f"Min velocity        : {stats['min_velocity']:.6f}")
    print(f"Max velocity        : {stats['max_velocity']:.6f}")
    print("================================\n")


if __name__ == "__main__":
    main()