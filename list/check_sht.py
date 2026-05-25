import numpy as np
from pathlib import Path
import csv
import math

processed = Path("../data/shanghaitech/processed")

nalist_path = processed / "ShanghaiTech_GCN_train_nalist.npy"
names_path = processed / "ShanghaiTech_GCN_train_video_names.npy"

nalist = np.load(nalist_path)
names = np.load(names_path, allow_pickle=True)

lengths = nalist[:, 1] - nalist[:, 0]

print("nalist shape:", nalist.shape)
print("names shape:", names.shape)
print("total segments:", int(lengths.sum()))
print()

print("[Segment length summary]")
print("min   :", int(lengths.min()))
print("max   :", int(lengths.max()))
print("mean  :", float(lengths.mean()))
print("median:", float(np.median(lengths)))
print("p25   :", float(np.percentile(lengths, 25)))
print("p75   :", float(np.percentile(lengths, 75)))
print("p90   :", float(np.percentile(lengths, 90)))
print()

for th in [32, 64, 128, 256, 512, 1000, 2000]:
    print(f"videos <= {th:4d} segments:", int((lengths <= th).sum()),
          f"/ videos > {th:4d}:", int((lengths > th).sum()))

print()

def count_windows(T, window_size, stride):
    if T <= window_size:
        return 1

    count = 0
    for _ in range(0, T - window_size + 1, stride):
        count += 1

    remainder = (T - window_size) % stride
    if remainder > 0:
        count += 1

    return count

configs = [
    (2000, 2000),
    (128, 64),
    (64, 32),
    (32, 16),
]

print("[Expected number of training windows]")
for ws, st in configs:
    total_windows = sum(count_windows(int(T), ws, st) for T in lengths)
    print(f"window_size={ws:4d}, stride={st:4d} -> total windows: {total_windows}")

print()

print("[Shortest 10 videos]")
idxs = np.argsort(lengths)[:10]
for i in idxs:
    print(str(names[i]), int(lengths[i]))

print()

print("[Longest 10 videos]")
idxs = np.argsort(lengths)[-10:][::-1]
for i in idxs:
    print(str(names[i]), int(lengths[i]))

# 전체 비디오별 segment 수 CSV 저장
out_csv = processed / "ShanghaiTech_GCN_train_segment_counts.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["video_name", "start", "end", "num_segments"])
    for name, (s, e), T in zip(names, nalist, lengths):
        writer.writerow([str(name), int(s), int(e), int(T)])

print()
print("saved csv:", out_csv)