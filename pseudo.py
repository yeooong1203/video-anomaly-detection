import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import warnings


def find_optimal_threshold(scores):
    valid_scores = scores
    
    return np.percentile(valid_scores, 90)


def temporal_attraction(video_scores, attraction_strength=0.4, iterations=3):

    scores = video_scores.copy()
    
    for _ in range(iterations):
        attracted = scores.copy()
        
        for i in range(len(scores)):
            
            # Window (i-10 ~ i+10)
            window_start = max(0, i - 10)
            window_end = min(len(scores), i + 11)
            window = scores[window_start:window_end]
            
            # 주변에 더 높은 score 있으면 끌어올림
            if window.max() > scores[i]:
                max_idx = window.argmax() + window_start
                distance = abs(max_idx - i)
                
                force = attraction_strength * window.max() * np.exp(-distance / 2.0)
                attracted[i] += force
        
        max_val = attracted.max()
        if max_val > 0:
            scores = attracted / max_val
        else:
            scores = attracted
    
    return scores


def remove_isolated_abnormal(binary_labels, min_length=1):
    cleaned = binary_labels.copy()
    T = len(binary_labels)
    
    i = 0
    while i < T:
        if cleaned[i] == 1:
            run_start = i
            run_end = i
            
            while run_end < T and cleaned[run_end] == 1:
                run_end += 1
            
            run_length = run_end - run_start

            if run_length <= min_length:
                cleaned[run_start:run_end] = 0
            
            i = run_end
        else:
            i += 1
    
    return cleaned


def fill_isolated_normal(binary_labels, max_gap=1):
    filled = binary_labels.copy()
    T = len(binary_labels)
    
    i = 0
    while i < T:
        if filled[i] == 1:
            j = i + 1
            
            while j < T and filled[j] == 0:
                j += 1
            
            gap_length = j - i - 1
            
            if j < T and filled[j] == 1 and gap_length <= max_gap:
                filled[i+1:j] = 1
                i = j
            else:
                i += 1
        else:
            i += 1
    
    return filled


def check_prototype_swap(video_binary_labels, abnormal_ratio_threshold=0.8):

    abnormal_ratio = video_binary_labels.mean()
    should_swap = abnormal_ratio >= abnormal_ratio_threshold
    
    return should_swap, abnormal_ratio


def apply_prototype_swap(video_binary_labels):
    return 1 - video_binary_labels

def generate_improved_pseudo_labels(train_data, nalist,
                                    feature_normalization='standard',
                                    threshold_method='none',
                                    score_normalization='zscore',
                                    prototype_method='none',
                                    use_attraction=True,
                                    attraction_strength=0.4,
                                    attraction_iterations=3,
                                    remove_isolated_abn=True,
                                    isolated_abn_min_length=1,
                                    fill_isolated_norm=True,
                                    isolated_norm_max_gap=2,
                                    use_prototype_swap=True,
                                    swap_threshold=0.8):

    total_T = int(nalist[-1, 1])
    
    # Feature Normalization
    all_features = []
    
    for info in tqdm(nalist, desc="Loading"):
        start, end = int(info[0]), int(info[1])
        video_feat = np.mean(train_data[start:end], axis=1)
        all_features.append(video_feat)
    
    # Normalize per video
    if feature_normalization != 'none':
        for i, video_feat in enumerate(tqdm(all_features, desc="Normalizing")):
            if feature_normalization == 'standard':
                mean = video_feat.mean(axis=0)
                std = video_feat.std(axis=0) + 1e-8
                all_features[i] = (video_feat - mean) / std
            elif feature_normalization == 'l2':
                norms = np.linalg.norm(video_feat, axis=1, keepdims=True) + 1e-8
                all_features[i] = video_feat / norms
    
    # Compute Distance Scores
    all_scores = []
    
    for video_feat in tqdm(all_features, desc="Scoring"):


        if len(video_feat) < 8:
            all_scores.append(np.zeros(len(video_feat)))
            continue
        
        # Prototype
        prototype = video_feat[:5].mean(axis=0)
        proto_indices = range(5)
        
        # Distance
        distances = np.linalg.norm(video_feat - prototype, axis=1)
        distances[proto_indices] = 0
        
        all_scores.append(distances)
    
    # Score Normalization
    if score_normalization != 'none':
        for i, video_scores in enumerate(tqdm(all_scores, desc="Normalizing scores")):
            valid_mask = video_scores > 0
            
            if valid_mask.sum() < 2:
                continue
            
            valid_scores = video_scores[valid_mask]
            
            if score_normalization == 'zscore':
                mean = valid_scores.mean()
                std = valid_scores.std()
                if std > 1e-6:
                    normalized = np.zeros_like(video_scores)
                    normalized[valid_mask] = (valid_scores - mean) / std
                    all_scores[i] = normalized

    
    # Temporal Attraction
    if use_attraction:
        for i, video_scores in enumerate(tqdm(all_scores, desc="Attraction")):
            if len(video_scores) >= 3:
                all_scores[i] = temporal_attraction(
                    video_scores,
                    attraction_strength=attraction_strength,
                    iterations=attraction_iterations
                )

            from scipy.ndimage import gaussian_filter1d
            all_scores[i] = gaussian_filter1d(all_scores[i], sigma=2.0)
    
    # Threshold
    all_scores_flat = np.concatenate(all_scores)

    threshold = find_optimal_threshold(all_scores_flat)
    
    all_binary_labels = []
    
    for video_scores in all_scores:
        binary = (video_scores >= threshold).astype(int)
        all_binary_labels.append(binary)
    
    
    if remove_isolated_abn:
        for i, binary in enumerate(tqdm(all_binary_labels, desc="Remove isolated abn")):
            all_binary_labels[i] = remove_isolated_abnormal(binary, min_length=isolated_abn_min_length)

    if fill_isolated_norm:
        for i, binary in enumerate(tqdm(all_binary_labels, desc="Fill isolated norm")):
            all_binary_labels[i] = fill_isolated_normal(binary, max_gap=isolated_norm_max_gap)
    
    swap_count = 0
    swap_ratios = []
    
    if use_prototype_swap:
        for i, binary in enumerate(tqdm(all_binary_labels, desc="Prototype swap")):
            should_swap, abn_ratio = check_prototype_swap(binary, abnormal_ratio_threshold=swap_threshold)
            
            if should_swap:
                all_binary_labels[i] = apply_prototype_swap(binary)
                swap_count += 1
                swap_ratios.append(abn_ratio)
        
    
    return all_binary_labels


def main():
    train_nalist_path = r".\list\nalist_i3d.npy"
    train_data_path = r"..\C2FPL\concat_UCF.npy"
    
    nalist = np.load(train_nalist_path)
    total_T = int(nalist[-1, 1])
    
    train_data = np.memmap(
        train_data_path,
        dtype="float32",
        mode="r",
        shape=(total_T, 10, 2048)
    )
    
    print(f"  Segments: {total_T:,}")
    print(f"  Videos: {len(nalist)}")
    
    pseudo_labels_list = generate_improved_pseudo_labels(
        train_data, nalist,
        feature_normalization='standard',
        threshold_method='p_value',
        score_normalization='zscore',
        prototype_method='none',
        use_attraction=True,
        attraction_strength=0.4,
        attraction_iterations=3,
        remove_isolated_abn=True,  
        isolated_abn_min_length=2,  # N-A-N 제거
        fill_isolated_norm=True,  
        isolated_norm_max_gap=2,  # A-N-A 제거
        use_prototype_swap=True,  
        swap_threshold=0.7  # 70% 이상이면 swap
    )
    
    all_labels_flat = np.concatenate(pseudo_labels_list)
    np.save("pseudo_labels_swap.npy", all_labels_flat)
    
    print(f"  Saved: pseudo_labels_swap.npy")


if __name__ == "__main__":
    main()