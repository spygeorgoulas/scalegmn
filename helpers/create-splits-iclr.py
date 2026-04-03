'''
python /home/intern/spygeorgoulas/thesis-metanets/scalegmn/helpers/create-splits-iclr.py \
  --input_dir /home/intern/spygeorgoulas/thesis-metanets/scalegmn/data/gram_ifw_4x64_300k_rwi_pth \
  --output_dir /home/intern/spygeorgoulas/thesis-metanets/scalegmn/data/gram_ifw_4x64_300k_rwi_pth_splitted \
  --train_ratio 0.8 \
  --val_ratio 0.10
'''

#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path
import random
from collections import defaultdict


def extract_group_id(filename: str):
    """
    Example:
        1021_1-0.pth -> 1021_1
    """
    stem = Path(filename).stem
    return stem.rsplit("-", 1)[0]


def create_split(groups, train_ratio, val_ratio, seed):
    random.seed(seed)
    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    n_total = len(group_keys)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_keys = group_keys[:n_train]
    val_keys = group_keys[n_train:n_train + n_val]
    test_keys = group_keys[n_train + n_val:]

    return train_keys, val_keys, test_keys


def copy_files(groups, split_keys, split_name, output_dir):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for key in split_keys:
        for file_path in groups[key]:
            dst = split_dir / file_path.name
            shutil.copy2(file_path, dst)
            count += 1

    print(f"{split_name}: {count} files")


def main():
    parser = argparse.ArgumentParser(description="Split INR dataset into train/val/test (grouped).")
    parser.add_argument("--input_dir", required=True, help="Path to folder with all INR .pth files")
    parser.add_argument("--output_dir", required=True, help="Path to save split dataset")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    assert input_dir.exists(), f"Input dir not found: {input_dir}"

    all_files = sorted(input_dir.glob("*.pth"))
    assert len(all_files) > 0, "No .pth files found"

    print(f"Found {len(all_files)} INR files")

    # ============================================================
    # GROUP FILES
    # ============================================================
    groups = defaultdict(list)

    for file_path in all_files:
        group_id = extract_group_id(file_path.name)
        groups[group_id].append(file_path)

    print(f"Number of groups (simulations): {len(groups)}")

    # ============================================================
    # CREATE SPLITS
    # ============================================================
    train_keys, val_keys, test_keys = create_split(
        groups,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    print(f"Train groups: {len(train_keys)}")
    print(f"Val groups: {len(val_keys)}")
    print(f"Test groups: {len(test_keys)}")

    # ============================================================
    # COPY FILES
    # ============================================================
    copy_files(groups, train_keys, "train", output_dir)
    copy_files(groups, val_keys, "val", output_dir)
    copy_files(groups, test_keys, "test", output_dir)

    print("\nDone. Dataset split created at:")
    print(output_dir)


if __name__ == "__main__":
    main()