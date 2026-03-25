from pathlib import Path
import shutil

# ====== SET THESE PATHS ======
src_dir = Path("/home/intern/spygeorgoulas/thesis-metanets/scalegmn/data/navier_h_velocity_2x64_2k_rwi_pth_parallel_linearonly")
dst_dir = Path("/home/intern/spygeorgoulas/thesis-metanets/scalegmn/data/navier_h_velocity_2x64_2k_rwi_pth_parallel_linearonly_splitted_all")
# =============================

# get all .pth files in sorted order
pth_files = sorted(src_dir.glob("*.pth"))

# keep only the first 1200
pth_files = pth_files[:2310]

# sanity check
if len(pth_files) < 2310:
    raise ValueError(f"Found only {len(pth_files)} .pth files, but need at least 2310.")

# define splits
train_files = pth_files[:1900]
val_files = pth_files[1900:2160]
test_files = pth_files[2160:2310]

# create folders
train_dir = dst_dir / "train"
val_dir = dst_dir / "val"
test_dir = dst_dir / "test"

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

# copy files
for f in train_files:
    shutil.copy2(f, train_dir / f.name)

for f in val_files:
    shutil.copy2(f, val_dir / f.name)

for f in test_files:
    shutil.copy2(f, test_dir / f.name)

print("Done.")
print(f"Train: {len(train_files)} files")
print(f"Val:   {len(val_files)} files")
print(f"Test:  {len(test_files)} files")