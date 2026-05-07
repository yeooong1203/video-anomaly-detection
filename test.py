import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import numpy as np
from dataset import UCFTestVideoDataset
from torch.utils.data import DataLoader
import option
import torch
from tqdm import tqdm
import csv
import copy
from model import Model_V2_AllCNN
from adapter import ResidualAdapter2048
import torch.nn.functional as F
import json
from pathlib import Path


def _segment_gt_from_gt(gt, total_T, frame_repeat=16):
    gt = np.asarray(gt).astype(np.int64).reshape(-1)

    if len(gt) == total_T:
        seg_gt = gt
        gt_mode = "segment"
    elif len(gt) == total_T * frame_repeat:
        seg_gt = gt.reshape(total_T, frame_repeat).max(axis=1)
        gt_mode = "frame"
    else:
        raise ValueError(
            f"GT length mismatch: len(gt)={len(gt)}, total_T={total_T}, "
            f"expected {total_T} or {total_T * frame_repeat}"
        )
    return seg_gt, gt_mode


def _normalize_video_feature_shape(x_video_np):
    """
    입력 feature를 최종적으로 (T, D)로 맞춘다.
    """
    x = np.asarray(x_video_np)

    # case 1: (T, D)
    if x.ndim == 2:
        return x

    # case 2: (T, C, D)  -> crop 평균
    if x.ndim == 3:
        return x.mean(axis=1)

    # case 3: (T, 1, C, D) or (T, C, 1, D)
    if x.ndim == 4:
        x = np.squeeze(x)

        if x.ndim == 2:
            return x
        elif x.ndim == 3:
            return x.mean(axis=1)
        else:
            raise ValueError(f"Unexpected 4D->squeezed shape: {x.shape}")

    raise ValueError(f"Unsupported feature shape: {x.shape}")


# prefix warm-up 실험을 위한 helper 함수들 (앞 5개 segment로 적응, 평가에서는 제외) ------------------------------------------------------------------------
def _split_prefix_suffix_video(x_video, warmup_segments=5):
    """
    x_video: torch tensor, (T, D)
    return:
      x_prefix: adaptation용
      prefix_len: 실제 prefix 길이
    """
    T = x_video.shape[0]
    prefix_len = min(warmup_segments, T)
    x_prefix = x_video[:prefix_len]
    return x_prefix, prefix_len


def _build_eval_mask_from_nalist(total_T, nalist, warmup_segments=5):
    """
    비디오별 prefix는 False, suffix는 True
    """
    eval_mask_seg = np.zeros(total_T, dtype=bool)

    for i in range(len(nalist)):
        s, e = nalist[i]
        s, e = int(s), int(e)
        prefix_len = min(warmup_segments, e - s)

        if e - s <= prefix_len:
            continue

        eval_mask_seg[s + prefix_len : e] = True

    return eval_mask_seg


def normalize_name(x):
    x = x.strip()

    x = x.replace("\\", "/")
    x = x.split("/")[-1]

    if x.endswith(".npy"):
        x = x[:-4]
    elif x.endswith(".mp4"):
        x = x[:-4]

    # annotation 쪽에 붙어 있는 v= 제거
    if x.startswith("v="):
        x = x[2:]

    return x 


def load_video_names(list_path):
    names = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            name = normalize_name(line)
            names.append(name)

    return names


# CORE---------------------------------------------------------------------------------
def _tea_update_one_video(
    x_video,          # torch tensor, (T_i, 1024)
    adapter_episode,
    model,
    q=0.2,
    min_keep=8,
    min_run=2,
    tea_lr=1e-3,
    tea_steps_per_video=1,

    n_reference=5,
    proto_l2_normalize=False,

    # prefix only adaptation & suffix evaluation 을 위한 인자들
    adapt_prefix_only=False,
    warmup_segments=5,
):
    """
    비디오 하나에 대해 adapter_episode.ln만 업데이트.
    return: adapter_episode (updated), debug_info
    """

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if not hasattr(adapter_episode, "ln"):
        raise ValueError("adapter_episode must have .ln for LN-only TEA")

    # adapter 전체 freeze, LN만 update
    adapter_episode.train()
    for p in adapter_episode.parameters():
        p.requires_grad_(False)
    adapter_episode.ln.weight.requires_grad_(True)
    adapter_episode.ln.bias.requires_grad_(True)

    ln_weight_init = adapter_episode.ln.weight.detach().clone()
    ln_bias_init = adapter_episode.ln.bias.detach().clone()

    optimizer = torch.optim.Adam(
        [adapter_episode.ln.weight, adapter_episode.ln.bias],
        lr=tea_lr
    )

    debug = []
    with torch.no_grad():
        if adapt_prefix_only:
            x_adapt, prefix_len = _split_prefix_suffix_video(
                x_video, warmup_segments=warmup_segments
            )
        else:
            x_adapt = x_video
            prefix_len = x_video.shape[0]

        x_ref_in = x_adapt.unsqueeze(0)                          # (1, T, 1024)
        x_ref_2048 = adapter_episode(x_ref_in)                   # (1, T, 2048)
        _, logit_ref = model(x_ref_2048, return_logits=True)     # (1, T, 1)
        #E_ref_init = F.softplus(logit_ref[0, :, 0]).mean().item() 


    for step_idx in range(tea_steps_per_video):
        # adaptation pool 결정
        if adapt_prefix_only:
            x_adapt, prefix_len = _split_prefix_suffix_video(
                x_video, warmup_segments=warmup_segments
            )
        else:
            x_adapt = x_video
            prefix_len = x_video.shape[0]

        if x_adapt.shape[0] == 0:
            debug.append({
                "step": step_idx,
                "skipped": True,
                "reason": "empty_adaptation_pool",
            })
            break


        # 1) 현재 비디오 baseline score
        with torch.no_grad():
            x_adapt_in = x_adapt.unsqueeze(0)   #(1, T, 1024)

            x_2048 = adapter_episode(x_adapt_in)    #(1, T, 2048)

            prob, logit = model(x_2048, return_logits=True)     # (1, T, 1)

        prob = prob[0, :, 0]
        logit = logit[0, :, 0]
        
        # 1) normal 후보 선택
        thresh = torch.quantile(prob.detach(), q)
        mask = prob.detach() <= thresh
        
        num_selected = int(mask.sum().item())
        if num_selected < min_keep:
            k = min(min_keep, prob.numel())
            topk_idx = torch.argsort(prob.detach())[:k]
            mask = torch.zeros_like(prob, dtype=torch.bool)
            mask[topk_idx] = True
            num_selected = int(mask.sum().item())

        proto_dists = None

        x_sel = x_adapt[mask].detach()      # (N,1024)  # 정상 후보로 선택된 샘플
        prob_sel = prob[mask].detach()
        logit_sel = logit[mask].detach()

        # 2) real energy
        x_sel_2048 = adapter_episode(x_sel)
        _, logit_real = model(x_sel_2048, return_logits=True)
        logit_real = logit_real.squeeze(-1)
        tau = 0.3    
        with torch.no_grad():
            w = torch.softmax(-prob_sel / tau, dim=0)  # (N,)  lower prob_sel = higher weight
        E_real = (w * F.softplus(logit_real)).sum()       
        #E_real = F.softplus(logit_real).mean()

        # 4) TEA loss 
        #loss = F.relu(E_real - E_fake)
        loss = E_real

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        debug.append({
            "step": step_idx,
            "skipped": False,
            "num_selected": num_selected,
            "threshold": float(thresh.item()),
            "E_real": float(E_real.item()),
            "loss": float(loss.item()),
            "prob_mean": float(prob.mean().item()),
            "prob_sel_mean": float(prob_sel.mean().item()),
            "logit_sel_mean": float(logit_sel.mean().item()),
            "min_run": int(min_run),
            "proto_dist_sel_mean": (
                float(proto_dists[mask].mean().item())
                if (proto_dists is not None and int(mask.sum().item()) > 0)
                else None
            ),
            "adapt_prefix_only": adapt_prefix_only,
            "prefix_len": int(prefix_len),
            "adapt_pool_size": int(x_adapt.shape[0]),
        })

    return adapter_episode, debug


# 전체 evaluation용 함수
def eval_xd_with_episodic_tea(
    X_flat,              # numpy, (total_T, 1024)
    nalist,              # numpy, (num_videos, 2)
    gt,                  # numpy, (total_T,) or (total_T*16,)
    adapter,
    model,
    device,
    frame_repeat=16,

    use_tea=True,
    q=0.2,
    min_keep=8,
    min_run = 2,
    tea_lr=1e-3,
    tea_steps_per_video=1,

    n_reference=5,
    proto_l2_normalize=False,

    exclude_prefix_from_eval=False,
    adapt_prefix_only=False,
    warmup_segments=5,

    verbose_every=100,
):

    total_T = X_flat.shape[0]
    seg_gt, gt_mode = _segment_gt_from_gt(gt, total_T, frame_repeat=frame_repeat)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    seg_scores_all = np.zeros(total_T, dtype=np.float32)
    tea_logs = []

    for vid_idx in range(len(nalist)):
        s, e = nalist[vid_idx]
        x_video_np = X_flat[s:e]                            # (T_i,1024)
        x_video_np = _normalize_video_feature_shape(x_video_np)
        x_video = torch.from_numpy(x_video_np).float().to(device)

        # 비디오마다 adapter 리셋
        adapter_episode = copy.deepcopy(adapter).to(device)
        adapter_episode.eval()
        
        # 1) optional TEA
        if use_tea:
            adapter_episode, debug = _tea_update_one_video(
                x_video=x_video,
                adapter_episode=adapter_episode,
                model=model,
                q=q,
                min_keep=min_keep,
                min_run=min_run,
                tea_lr=tea_lr,
                tea_steps_per_video=tea_steps_per_video,
                n_reference=n_reference,
                proto_l2_normalize=proto_l2_normalize,
                adapt_prefix_only=adapt_prefix_only,
                warmup_segments=warmup_segments,
            )
        else:
            debug = None
        

        # 2) adaptation 후 최종 inference
        adapter_episode.eval()
        with torch.no_grad():
            x_video_in = x_video.unsqueeze(0)                 # (1,T,1024)
            x_2048 = adapter_episode(x_video_in)              # (1,T,2048)
            #x_2048 = adapter_episode(x_video)
            prob, _ = model(x_2048, return_logits=True)

        #prob = prob.squeeze(-1).detach().cpu().numpy()     # (T_i,)
        prob = prob[0, :, 0].detach().cpu().numpy()

        seg_scores_all[s:e] = prob

        if debug is not None:
            tea_logs.append({
                "vid_idx": vid_idx,
                "start": int(s),
                "end": int(e),
                "T": int(e - s),
                "debug": debug,
            })

        if (vid_idx % verbose_every == 0) or (vid_idx == len(nalist) - 1):
            print(f"[{vid_idx+1}/{len(nalist)}] done, T={e-s}")

    # 3) metric 계산
    if exclude_prefix_from_eval:
        eval_mask_seg = _build_eval_mask_from_nalist(
            total_T=total_T,
            nalist=nalist,
            warmup_segments=warmup_segments,
        )
    else:
        eval_mask_seg = np.ones(total_T, dtype=bool)

    if gt_mode == "segment":
        y_true = seg_gt[eval_mask_seg]
        y_score = seg_scores_all[eval_mask_seg]
    else:
        # segment score를 16배 반복해서 frame-level score로 맞춤
        y_true = np.asarray(gt).reshape(-1)
        y_score = np.repeat(seg_scores_all, frame_repeat)
        eval_mask_frame = np.repeat(eval_mask_seg, frame_repeat)

        min_len = min(len(y_true), len(y_score), len(eval_mask_frame))
        y_true = y_true[:min_len]
        y_score = y_score[:min_len]
        eval_mask_frame = eval_mask_frame[:min_len]

        y_true = y_true[eval_mask_frame]
        y_score = y_score[eval_mask_frame]

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return {
        "auc": float(auc),
        "ap": float(ap),
        "seg_scores_all": seg_scores_all,
        "tea_logs": tea_logs,
        "eval_mask_seg": eval_mask_seg,
    }


# Analysis ---------------------------------------------------------------------------------
def bootstrap_video_ci(
    seg_scores_base,     # (total_T,)
    seg_scores_tea,      # (total_T,)
    nalist,              # (num_videos, 2)
    gt,                  # (total_T,) or (total_T*16,)
    frame_repeat=16,
    n_boot=1000,
    seed=42,
    exclude_prefix_from_eval=False,
    warmup_segments=5,
):
    rng = np.random.default_rng(seed)

    total_T = len(seg_scores_base)
    gt = np.asarray(gt).reshape(-1)

    # GT 모드
    if len(gt) == total_T:
        gt_mode = "segment"
        seg_gt = gt.astype(np.int64)
    elif len(gt) == total_T * frame_repeat:
        gt_mode = "frame"
        seg_gt = gt.reshape(total_T, frame_repeat).max(axis=1).astype(np.int64)
    else:
        raise ValueError(
            f"GT length mismatch: len(gt)={len(gt)}, total_T={total_T}, "
            f"expected {total_T} or {total_T*frame_repeat}"
        )

    num_videos = len(nalist)

    delta_auc_list = []
    delta_ap_list = []

    for _ in range(n_boot):
        # 비디오 인덱스 복원추출
        boot_vids = rng.integers(0, num_videos, size=num_videos)

        y_true_parts = []
        y_base_parts = []
        y_tea_parts = []

        for vid_idx in boot_vids:
            s, e = nalist[vid_idx]
            s, e = int(s), int(e)

            if exclude_prefix_from_eval:
                prefix_len = min(warmup_segments, e - s)
                s_eval = s + prefix_len
                e_eval = e
            else:
                s_eval = s
                e_eval = e

            if s_eval >= e_eval:
                continue

            if gt_mode == "segment":
                y_true_parts.append(seg_gt[s_eval:e_eval])
                y_base_parts.append(seg_scores_base[s_eval:e_eval])
                y_tea_parts.append(seg_scores_tea[s_eval:e_eval])
            else:
                # frame-level metric과 맞추기 위해 segment score를 16번 반복
                y_true_parts.append(gt[s_eval*frame_repeat:e_eval*frame_repeat])
                y_base_parts.append(np.repeat(seg_scores_base[s_eval:e_eval], frame_repeat))
                y_tea_parts.append(np.repeat(seg_scores_tea[s_eval:e_eval], frame_repeat))

        y_true = np.concatenate(y_true_parts)
        y_base = np.concatenate(y_base_parts)
        y_tea = np.concatenate(y_tea_parts)

        auc_base = roc_auc_score(y_true, y_base)
        ap_base = average_precision_score(y_true, y_base)

        auc_tea = roc_auc_score(y_true, y_tea)
        ap_tea = average_precision_score(y_true, y_tea)

        delta_auc_list.append(auc_tea - auc_base)
        delta_ap_list.append(ap_tea - ap_base)

    delta_auc = np.array(delta_auc_list)
    delta_ap = np.array(delta_ap_list)

    def ci95(x):
        return np.percentile(x, [2.5, 50, 97.5])

    auc_ci = ci95(delta_auc)
    ap_ci = ci95(delta_ap)

    print("[Bootstrap Δ = TEA - Baseline]")
    print(f"ΔAUC median={auc_ci[1]:.6f}, 95% CI=({auc_ci[0]:.6f}, {auc_ci[2]:.6f})")
    print(f"ΔAP  median={ap_ci[1]:.6f}, 95% CI=({ap_ci[0]:.6f}, {ap_ci[2]:.6f})")

    return {
        "delta_auc": delta_auc,
        "delta_ap": delta_ap,
        "auc_ci95": auc_ci,
        "ap_ci95": ap_ci,
    }


def summarize_demo_candidates(seg_scores_all, nalist, out_csv_path, video_names=None, top_k_mean=5):
    """
    비디오별 점수 요약 csv 저장:
    - max score
    - mean score
    - top-k 평균
    - 길이(T)
    """
    rows = []

    for vid_idx in range(len(nalist)):
        s, e = map(int, nalist[vid_idx])
        scores = np.asarray(seg_scores_all[s:e], dtype=np.float32)

        if len(scores) == 0:
            continue

        name = video_names[vid_idx] if video_names is not None and vid_idx < len(video_names) else f"video_{vid_idx}"
        k = min(top_k_mean, len(scores))
        topk_mean = float(np.sort(scores)[-k:].mean())

        rows.append({
            "vid_idx": vid_idx,
            "video_name": name,
            "start": s,
            "end": e,
            "T": e - s,
            "score_max": float(scores.max()),
            "score_mean": float(scores.mean()),
            "score_topk_mean": topk_mean,
        })

    rows = sorted(rows, key=lambda x: (-x["score_max"], -x["score_topk_mean"], -x["score_mean"]))

    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[saved] candidate summary csv -> {out_csv_path}")
    return rows


def save_video_score_plots(
    seg_scores_all,
    nalist,
    out_dir,
    video_names=None,
    threshold=None,
    top_n=None,
    gt=None,
    frame_repeat=16,
    show_gt=True,
    show_prefix=False,
    warmup_segments=5,
):
    """
    비디오별 score timeline plot 저장
    - gt가 주어지면 실제 anomaly GT 구간을 빨간 음영으로 표시
    - show_prefix=True면 warm-up prefix 구간을 회색 음영으로 표시
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_videos = len(nalist) if top_n is None else min(top_n, len(nalist))

    # 전체 GT를 segment-level로 통일
    seg_gt_all = None
    if gt is not None:
        total_T = int(nalist[-1, 1])
        seg_gt_all, gt_mode = _segment_gt_from_gt(gt, total_T, frame_repeat=frame_repeat)
        print(f"[plot] GT mode: {gt_mode} -> segment-level GT for plotting")

    for vid_idx in range(num_videos):
        s, e = map(int, nalist[vid_idx])
        scores = np.asarray(seg_scores_all[s:e], dtype=np.float32)

        if len(scores) == 0:
            continue

        name = video_names[vid_idx] if video_names is not None and vid_idx < len(video_names) else f"video_{vid_idx}"
        x = np.arange(len(scores))

        plt.figure(figsize=(10, 3.5))
        plt.plot(x, scores, linewidth=1.5, label="score")

        if threshold is not None:
            plt.axhline(threshold, linestyle="--", label=f"threshold={threshold:.2f}")

        # prefix warm-up 구간 표시
        if show_prefix and warmup_segments > 0:
            prefix_len = min(warmup_segments, len(scores))
            if prefix_len > 0:
                plt.axvspan(-0.5, prefix_len - 0.5, alpha=0.12, color="gray", label="warm-up prefix")

        # 실제 GT anomaly 구간 표시
        if show_gt and seg_gt_all is not None:
            seg_gt_video = np.asarray(seg_gt_all[s:e], dtype=np.int64)

            in_run = False
            run_start = None
            for i, g in enumerate(seg_gt_video):
                if g == 1 and not in_run:
                    in_run = True
                    run_start = i
                elif g == 0 and in_run:
                    plt.axvspan(run_start - 0.5, i - 0.5, alpha=0.22, color="red")
                    in_run = False
                    run_start = None

            if in_run and run_start is not None:
                plt.axvspan(run_start - 0.5, len(seg_gt_video) - 0.5, alpha=0.22, color="red")

        plt.title(f"{name} | vid_idx={vid_idx} | T={e-s}")
        plt.xlabel("segment index")
        plt.ylabel("anomaly score")
        plt.ylim(0.0, 0.3)
        plt.tight_layout()

        handles, labels = plt.gca().get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        if len(uniq) > 0:
            plt.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)

        plt.tight_layout()

        safe_name = str(name).replace("/", "_").replace("\\", "_")
        plt.savefig(out_dir / f"{vid_idx:03d}_{safe_name}.png", dpi=150)
        plt.close()

    print(f"[saved] score plots -> {out_dir}")


def export_demo_jsons(
    seg_scores_adapted,
    nalist,
    out_dir,
    video_names=None,
    fps=30,
    frames_per_seg=16,
    warmup_segments=5,
    display_reference=0.10,
    selected_vid_indices=None,
    actual_video_duration_map=None,
    seg_scores_baseline=None,   # 추가
):
    """
    데모 앱용 JSON export
    - manifest.json
    - camXX_scores.json 여러 개
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if selected_vid_indices is None:
        selected_vid_indices = list(range(len(nalist)))

    manifest = {
        "projectTitle": "Anomaly Detection in Surveillance Video",
        "warmupSegments": warmup_segments,
        "videos": []
    }

    cam_counter = 1

    for vid_idx in selected_vid_indices:
        s, e = map(int, nalist[vid_idx])

        scores_adapted = np.asarray(seg_scores_adapted[s:e], dtype=np.float32)
        if len(scores_adapted) == 0:
            continue

        scores_baseline = None
        if seg_scores_baseline is not None:
            scores_baseline = np.asarray(seg_scores_baseline[s:e], dtype=np.float32)

        video_name = (
            video_names[vid_idx]
            if video_names is not None and vid_idx < len(video_names)
            else f"video_{vid_idx}"
        )
        video_id = f"cam{cam_counter:02d}"

        segment_duration_sec = frames_per_seg / fps
        duration_sec = len(scores_adapted) * segment_duration_sec

        if actual_video_duration_map is not None:
            if vid_idx in actual_video_duration_map:
                duration_sec = float(actual_video_duration_map[vid_idx])
                segment_duration_sec = duration_sec / max(len(scores_adapted), 1)
            elif video_name in actual_video_duration_map:
                duration_sec = float(actual_video_duration_map[video_name])
                segment_duration_sec = duration_sec / max(len(scores_adapted), 1)

        data = {
            "videoId": video_id,
            "sourceVideoName": video_name,
            "fps": fps,
            "durationSec": float(duration_sec),
            "scoreType": "segment",
            "segmentDurationSec": float(segment_duration_sec),
            "warmupSegments": warmup_segments,
            "stateMode": "adaptive_from_warmup",
            "displayReference": float(display_reference),
            "scoresAdapted": scores_adapted.astype(float).tolist(),
        }

        if scores_baseline is not None:
            data["scoresBaseline"] = scores_baseline.astype(float).tolist()

        with open(out_dir / f"{video_id}_scores.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        manifest["videos"].append({
            "id": video_id,
            "cameraName": f"Camera {cam_counter:02d}",
            "location": video_name,
            "videoPath": f"/videos/{video_id}.mp4",
            "scorePath": f"/data/{video_id}_scores.json"
        })

        cam_counter += 1

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[saved] demo jsons -> {out_dir}")



# ---------------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------------

if __name__ == '__main__':
    args = option.parser.parse_args()
    device = torch.device("cuda")

    # 1. GT / dataset
    gt = np.load(args.gt)
    ucf_test_dataset = UCFTestVideoDataset(
        conall_path=args.test_conall_path,
        nalist_path=args.test_nalist_path,
    )
    X_flat = ucf_test_dataset.con_all
    nalist = ucf_test_dataset.nalist
    #nalist = np.load(args.nalist_path)
    print("X_flat shape:", X_flat.shape)
    print("nalist shape:", nalist.shape)

    # 2. adapter
    adapter = ResidualAdapter2048(d=2048, use_ln = True).to(device)
    torch.save(adapter.state_dict(), "adapter_init.pt") #baseline adapter 고정(저장) - 초기 1번만
    adapter.load_state_dict(torch.load("adapter_init.pt",  map_location=device))
    adapter.eval()

    # 3. model
    model = Model_V2_AllCNN(args.feature_size).to(device)
    model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('unsupervised_ckpt/UCF_all_cnn_final_20260331_020353_wv5ldb2h.pkl').items()})

    model.eval()
    

    # baseline (TEA 없음)
    res_base = eval_xd_with_episodic_tea(
        X_flat=X_flat,
        nalist=nalist,
        gt=gt,
        adapter=adapter,
        model=model,
        device=device,
        frame_repeat=16,

        use_tea=False,
        verbose_every=100,
    )
    print("\n[BASELINE]")
    print("AUC:", res_base["auc"])
    print("AP :", res_base["ap"])

    
    # warm up baseline (prefix 적응은 안하지만 suffix만 평가)
    res_base_warm = eval_xd_with_episodic_tea(
        X_flat=X_flat,
        nalist=nalist,
        gt=gt,
        adapter=adapter,
        model=model,
        device=device,
        frame_repeat=16,

        use_tea=False,          

        adapt_prefix_only=False,          # baseline -> 적응 안 함
        exclude_prefix_from_eval=True,    # suffix만 평가
        warmup_segments=5,
    )

    print("\n[BASELINE - SUFFIX ONLY]")
    print("AUC:", res_base_warm["auc"])
    print("AP :", res_base_warm["ap"])

    
    # warm up (prefix 적응, suffix만 평가)
    res_tea_warm = eval_xd_with_episodic_tea(
    X_flat=X_flat,
    nalist=nalist,
    gt=gt,
    adapter=adapter,
    model=model,
    device=device,
    frame_repeat=16,

    use_tea=True,

    q=1.0,   
    min_keep=8,
    tea_lr=1e-2,
    tea_steps_per_video=30,
    adapt_prefix_only=True,           # prefix 안에서만 selection/update
    exclude_prefix_from_eval=True,    # suffix만 평가
    warmup_segments=5,                # adaptation pool 지정 (prefix)
    )

    print("\n[PREFIX WARM-UP TEA]")
    print("AUC:", res_tea_warm["auc"])
    print("AP :", res_tea_warm["ap"])
    

    # --------------------------------------------------
    # Demo candidate analysis / export
    # --------------------------------------------------
    # 비디오 이름 로드
    video_names = load_video_names("list/ucf-i3d_test_fixed_local.list")
    
    # 1) baseline 후보 영상 요약 csv
    base_rows = summarize_demo_candidates(
        seg_scores_all=res_base["seg_scores_all"],
        nalist=nalist,
        out_csv_path="demo_exports/base_candidate_summary.csv",
        video_names=video_names,
        top_k_mean=5,
    )

    # 2) baseline 전체 score plot 저장
    save_video_score_plots(
        seg_scores_all=res_base["seg_scores_all"],
        nalist=nalist,
        out_dir="demo_exports/base_plots",
        video_names=video_names,
        threshold=0.2,
        gt=gt,
        frame_repeat=16,
        show_gt=True,
        show_prefix=False,
    )

    # 3) warm-up TEA 쪽도 같이 보고 싶으면 저장
    tea_rows = summarize_demo_candidates(
        seg_scores_all=res_tea_warm["seg_scores_all"],
        nalist=nalist,
        out_csv_path="demo_exports/tea_warm_candidate_summary.csv",
        video_names=video_names,
        top_k_mean=5,
    )

    save_video_score_plots(
        seg_scores_all=res_tea_warm["seg_scores_all"],
        nalist=nalist,
        out_dir="demo_exports/tea_warm_plots",
        video_names=video_names,
        threshold=0.2,
        gt=gt,
        frame_repeat=16,
        show_gt=True,
        show_prefix=True,
        warmup_segments=5,
    )

    # 4) vid_idx를 넣어서 JSON export
    selected_vid_indices = [17, 30, 97, 230]

    export_demo_jsons(
        seg_scores_adapted=res_tea_warm["seg_scores_all"],
        seg_scores_baseline=res_base["seg_scores_all"],   # 추가
        nalist=nalist,
        out_dir="demo_exports/demo_json_base",
        video_names=video_names,
        fps=30,
        frames_per_seg=16,
        warmup_segments=5,
        display_reference=0.10,
        selected_vid_indices=selected_vid_indices,
        actual_video_duration_map=None,
    )
    
    #부트스트랩으로 확인
    boot_res = bootstrap_video_ci(
        seg_scores_base=res_base_warm["seg_scores_all"],
        seg_scores_tea=res_tea_warm["seg_scores_all"],
        nalist=nalist,
        gt=gt,
        frame_repeat=16,
        n_boot=1000,
        seed=42,
    )