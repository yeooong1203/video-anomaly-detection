import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--list_path", default=r"ucf-i3d-test.list")
parser.add_argument("--feature_dir", required=True)   #UCF_Test_ten_i3d 피처가 있는 경로
parser.add_argument("--out", default=r"ucf-i3d_test_fixed_local.list")
args = parser.parse_args()

LIST_PATH = args.list_path
FEATURE_DIR = args.feature_dir
OUT_FIXED_LIST = args.out

folder_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith(".npy")])
folder_set = set(folder_files)

fixed_paths = []
missing = []

with open(LIST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        p = line.split()[0]
        base = os.path.basename(p)

        if not base.endswith(".npy"):
            base = base + ".npy"

        if base not in folder_set:
            missing.append(base)

        fixed_paths.append(os.path.join(FEATURE_DIR, base))

print("folder npy:", len(folder_files))
print("list entries:", len(fixed_paths))
print("missing_in_folder:", len(missing))

if missing:
    print("example missing:", missing[:10])

os.makedirs(os.path.dirname(OUT_FIXED_LIST), exist_ok=True)

with open(OUT_FIXED_LIST, "w", encoding="utf-8") as f:
    for p in fixed_paths:
        f.write(p + "\n")

print("Wrote:", OUT_FIXED_LIST)