"""
confidence.py - Temporal Consistency based Confidence
"""

import numpy as np
from tqdm import tqdm


class BidirectionalTemporalConsistency:
    """
    양방향 temporal consistency로 confidence 계산
    
    핵심:
    - 앞뒤 segment와 feature & label consistency 체크
    - 더 가까운 쪽 선택
    - Consistent하면 high confidence
    """
    
    def __init__(self):
        pass
    
    
    def compute_consistency_score(self, features, labels, t):
        """
        t번째 segment의 confidence 계산
        
        Args:
            features: (T, 2048) numpy array
            labels: (T,) numpy array - binary (0 or 1)
            t: int - current position
        
        Returns:
            confidence: float (0~1)
        """
        T = len(features)
        
        # ⭐ 처음/끝은 낮은 confidence
        if t == 0 or t == T - 1:
            return 0.5
        
        current_feat = features[t]
        current_label = labels[t]
        
        # ====== 앞 segment
        prev_feat = features[t-1]
        prev_label = labels[t-1]
        
        # Feature distance (L2)
        prev_feat_dist = np.linalg.norm(current_feat - prev_feat)
        
        # Label difference (0 or 1)
        prev_label_diff = abs(current_label - prev_label)
        
        # ===== 뒤 segment
        next_feat = features[t+1]
        next_label = labels[t+1]
        
        next_feat_dist = np.linalg.norm(current_feat - next_feat)
        next_label_diff = abs(current_label - next_label)
        
        # 더 가까운 쪽 선택
        if prev_feat_dist < next_feat_dist:
            # 앞과 더 가까움
            feat_dist = prev_feat_dist
            label_diff = prev_label_diff
        else:
            # 뒤와 더 가까움
            feat_dist = next_feat_dist
            label_diff = next_label_diff
        
        # 1. Feature similarity (0~1)
        #    가까우면 1, 멀면 0
        feat_similarity = np.exp(-feat_dist / 15.0)
        
        # 2. Label consistency (0 or 1)
        #    같으면 1, 다르면 0
        label_consistency = 1.0 - label_diff
        
        # 3. Combined (평균)
        confidence = (feat_similarity) * (label_consistency) + (1 - feat_similarity) * (1 - label_consistency)
        #confidence = 0.5 * feat_similarity + 0.5 * label_consistency
        #confidence = 1 - np.abs(feat_similarity - label_consistency)
        #confidence = 1 - (feat_similarity - label_consistency) ** 2

        return confidence
    

    def compute_video_confidences(self, video_features, video_labels):
        """
        전체 video의 confidence 계산
        
        Args:
            video_features: (T, 2048) numpy array
            video_labels: (T,) numpy array - binary
        
        Returns:
            confidences: (T,) numpy array
        """
        T = len(video_features)
        confidences = np.zeros(T)
        
        for t in range(T):
            confidence = self.compute_consistency_score(
                video_features, video_labels, t
            )
            confidences[t] = confidence
        
        return confidences


def generate_confidence_scores(train_data, nalist, pseudo_labels):
    """
    전체 dataset의 confidence 생성
    
    Args:
        train_data: memmap of I3D features
        nalist: video boundaries
        pseudo_labels: binary labels (flat array)
    
    Returns:
        all_confidences: list of numpy arrays
    """
    print("\n" + "="*80)
    print("Confidence Score Generation")
    print("="*80)
    
    calculator = BidirectionalTemporalConsistency()
    
    all_confidences = []
    
    # Video별로 처리
    print("\n[Computing Confidences]")
    for info in tqdm(nalist, desc="Processing videos"):
        start, end = int(info[0]), int(info[1])
        
        # Load features (평균)
        video_feat = train_data[start:end].mean(axis=1)  # (T, 2048)
        
        # Load labels
        video_labels = pseudo_labels[start:end]  # (T,)
        
        # Compute confidence
        video_confidences = calculator.compute_video_confidences(
            video_feat, video_labels
        )
        
        all_confidences.append(video_confidences)
    
    # Statistics
    print(f"\n{'='*80}")
    print("Confidence Statistics")
    print(f"{'='*80}")
    
    all_conf_flat = np.concatenate(all_confidences)
    
    print(f"\nOverall:")
    print(f"  Mean: {all_conf_flat.mean():.3f}")
    print(f"  Std:  {all_conf_flat.std():.3f}")
    print(f"  Min:  {all_conf_flat.min():.3f}")
    print(f"  Max:  {all_conf_flat.max():.3f}")
    
    print(f"\nDistribution:")
    print(f"  High (>0.8):   {(all_conf_flat > 0.8).sum():,} ({100*(all_conf_flat>0.8).mean():.1f}%)")
    print(f"  Medium (0.5-0.8): {((all_conf_flat >= 0.5) & (all_conf_flat <= 0.8)).sum():,} ({100*((all_conf_flat>=0.5) & (all_conf_flat<=0.8)).mean():.1f}%)")
    print(f"  Low (<0.5):    {(all_conf_flat < 0.5).sum():,} ({100*(all_conf_flat<0.5).mean():.1f}%)")
    
    # Per-video analysis
    video_conf_means = [conf.mean() for conf in all_confidences]
    video_conf_means = np.array(video_conf_means)
    
    print(f"\nPer-Video:")
    print(f"  Mean confidence: {video_conf_means.mean():.3f} ± {video_conf_means.std():.3f}")
    print(f"  Min: {video_conf_means.min():.3f}")
    print(f"  Max: {video_conf_means.max():.3f}")
    
    print(f"{'='*80}")
    
    return all_confidences