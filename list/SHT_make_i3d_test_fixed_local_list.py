import numpy as np
from pathlib import Path

names_path = Path(r"C:\Users\jplabuser\data\shanghaitech\processed\ShanghaiTech_GCN_test_video_names.npy")
out_path = Path(r"C:\Users\jplabuser\data\shanghaitech\processed\ShanghaiTech_GCN_test_video_names.list")

names = np.load(names_path, allow_pickle=True)

with open(out_path, "w", encoding="utf-8") as f:
    for name in names:
        f.write(str(name) + "\n")

print("saved:", out_path)
print("num videos:", len(names))