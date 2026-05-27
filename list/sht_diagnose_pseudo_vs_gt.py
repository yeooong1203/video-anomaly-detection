import ast
import csv
from pathlib import Path

import numpy as np


# ====== 경로만 네 환경에 맞게 확인 ======
pseudo_path = Path("../Unsup_labels/SHT_pseudo_labels_fixedload.npy")

train_nalist_path = Path("../../data/shanghaitech/processed/ShanghaiTech_GCN_train_nalist.npy")
train_names_path = Path("../../data/shanghaitech/processed/ShanghaiTech_GCN_train_video_names.npy")

# abnormal interval annotation txt
ann_path = Path("../../data/shanghaitech/annotations/shanghaitech_nalist.txt")

out_csv = Path("diagnostic_pseudo_vs_train_gt.csv")

frame_repeat = 16


def normalize_name(x):
    return Path(str(x)).stem


def load_annotation_txt(path):
    """
    line format:
    video_name    global_start    global_end    [(a,b), ...]
    여기서 (a,b)는 해당 abnormal video 내부 frame index 구간으로 처리
    """
    ann = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=3)
            if len(parts) < 4:
                continue

            name = normalize_name(parts[0])
            global_start = int(parts[1])
            global_end = int(parts[2])
            intervals = ast.literal_eval(parts[3])

            n_frames = global_end - global_start + 1
            ann[name] = {
                "n_frames": n_frames,
                "intervals": intervals,
            }

    return ann


def frame_gt_to_segment_gt(n_frames, intervals, T, frame_repeat=16):
    frame_gt = np.zeros(n_frames, dtype=np.int32)

    for a, b in intervals:
        a = max(0, int(a))
        b = min(n_frames - 1, int(b))
        if a <= b:
            frame_gt[a:b + 1] = 1

    seg_gt = np.zeros(T, dtype=np.int32)

    for i in range(T):
        fs = i * frame_repeat
        fe = min((i + 1) * frame_repeat, n_frames)

        if fs < n_frames and frame_gt[fs:fe].max() > 0:
            seg_gt[i] = 1

    return seg_gt


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def main():
    pseudo = np.load(pseudo_path).astype(np.int32)
    nalist = np.load(train_nalist_path)
    names = np.load(train_names_path, allow_pickle=True)
    ann = load_annotation_txt(ann_path)

    assert len(names) == len(nalist), f"names={len(names)}, nalist={len(nalist)}"
    assert len(pseudo) == int(nalist[-1, 1]), (
        f"pseudo len={len(pseudo)}, nalist total_T={int(nalist[-1, 1])}"
    )

    gt_all = np.zeros_like(pseudo, dtype=np.int32)

    rows = []

    for vid_idx, raw_name in enumerate(names):
        name = normalize_name(raw_name)
        s, e = map(int, nalist[vid_idx])
        T = e - s

        pred_v = pseudo[s:e].astype(np.int32)
        gt_v = np.zeros(T, dtype=np.int32)

        has_ann = name in ann

        if has_ann:
            info = ann[name]
            gt_v = frame_gt_to_segment_gt(
                n_frames=info["n_frames"],
                intervals=info["intervals"],
                T=T,
                frame_repeat=frame_repeat,
            )

        gt_all[s:e] = gt_v

        tp = int(((pred_v == 1) & (gt_v == 1)).sum())
        fp = int(((pred_v == 1) & (gt_v == 0)).sum())
        fn = int(((pred_v == 0) & (gt_v == 1)).sum())
        tn = int(((pred_v == 0) & (gt_v == 0)).sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        rows.append({
            "video": name,
            "has_annotation": int(has_ann),
            "T_segments": T,
            "gt_abn": int(gt_v.sum()),
            "pseudo_abn": int(pred_v.sum()),
            "pseudo_ratio": float(pred_v.mean()),
            "gt_ratio": float(gt_v.mean()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    # 전체 segment-level confusion
    tp = int(((pseudo == 1) & (gt_all == 1)).sum())
    fp = int(((pseudo == 1) & (gt_all == 0)).sum())
    fn = int(((pseudo == 0) & (gt_all == 1)).sum())
    tn = int(((pseudo == 0) & (gt_all == 0)).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    print("\n[Global segment-level diagnostic]")
    print("total segments      :", len(pseudo))
    print("GT abnormal segments:", int(gt_all.sum()), f"({gt_all.mean():.4f})")
    print("Pseudo abnormal segs:", int(pseudo.sum()), f"({pseudo.mean():.4f})")
    print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
    print("precision:", round(precision, 4))
    print("recall   :", round(recall, 4))
    print("f1       :", round(f1, 4))

    # annotation 있는 abnormal 영상만 따로 보기
    rows_ann = [r for r in rows if r["has_annotation"] == 1]
    if rows_ann:
        gt_abn_videos = sum(1 for r in rows_ann if r["gt_abn"] > 0)
        detected_videos = sum(1 for r in rows_ann if r["pseudo_abn"] > 0)
        missed_videos = sum(1 for r in rows_ann if r["gt_abn"] > 0 and r["pseudo_abn"] == 0)

        print("\n[Abnormal annotated video diagnostic]")
        print("annotated videos:", len(rows_ann))
        print("gt abnormal videos:", gt_abn_videos)
        print("videos with any pseudo abnormal:", detected_videos)
        print("completely missed abnormal videos:", missed_videos)

    # FP 많이 나는 영상
    rows_sorted_fp = sorted(rows, key=lambda x: x["fp"], reverse=True)[:10]
    print("\n[Top 10 videos by FP]")
    for r in rows_sorted_fp:
        print(
            r["video"],
            "T=", r["T_segments"],
            "gt_abn=", r["gt_abn"],
            "pseudo_abn=", r["pseudo_abn"],
            "FP=", r["fp"],
            "precision=", round(r["precision"], 3),
            "recall=", round(r["recall"], 3),
        )

    # FN 많이 나는 영상
    rows_sorted_fn = sorted(rows, key=lambda x: x["fn"], reverse=True)[:10]
    print("\n[Top 10 videos by FN]")
    for r in rows_sorted_fn:
        print(
            r["video"],
            "T=", r["T_segments"],
            "gt_abn=", r["gt_abn"],
            "pseudo_abn=", r["pseudo_abn"],
            "FN=", r["fn"],
            "precision=", round(r["precision"], 3),
            "recall=", round(r["recall"], 3),
        )

    # CSV 저장
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved per-video diagnostic:", out_csv)


if __name__ == "__main__":
    main()