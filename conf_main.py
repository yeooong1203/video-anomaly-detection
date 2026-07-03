import numpy as np
from confidence import generate_confidence_scores
import option
args = option.parser.parse_args()

def main():

    print("="*80)
    print("Generate Confidence Scores")
    print("="*80)

    # ===== 1. load data =====
    print("\n[Step 1: Loading Data]")

    train_nalist_path = args.train_nalist_path
    train_conall_path = args.train_conall_path
    pseudo_labels_path = args.pseudofile
    conf_save_path = args.confidence_path

    nalist = np.load(train_nalist_path)
    total_T = int(nalist[-1, 1])
    print(f"  Total segments: {total_T:,}")
    print(f"  Total videos: {len(nalist)}")
    
    '''
    train_data = np.memmap(
        train_conall_path,
        dtype="float32",
        mode="r",
        shape=(total_T, 10, args.feature_size)
    )
    '''    
    train_data = np.load(train_conall_path, mmap_mode="r")
    
    print(f"  Features loaded: {train_data.shape}")

    
    # ⭐ Pseudo labels (pseudo.py를 돌려서 얻은 pseudo label)
    pseudo_labels = np.load(pseudo_labels_path)
    print(f"  Pseudo labels loaded: {pseudo_labels.shape}")

    # Validation
    assert len(pseudo_labels) == total_T, f"Label length mismatch: {len(pseudo_labels)} vs {total_T}"

    print(f"  Normal (0): {(pseudo_labels == 0).sum():,} ({100*(pseudo_labels==0).mean():.1f}%)")
    print(f"  Abnormal (1): {(pseudo_labels == 1).sum():,} ({100*(pseudo_labels==1).mean():.1f}%)")
    

    # ===== 2. Generate confidences =====
    print("\n[Step 2: Generating Confidences]")
    
    confidence_list = generate_confidence_scores(
        train_data, 
        nalist, 
        pseudo_labels
    )
    
    # ===== 3. Save =====
    print("\n[Step 3: Saving Results]")
    
    # Flat array로 저장
    all_confidences_flat = np.concatenate(confidence_list)
    np.save(conf_save_path, all_confidences_flat)
    print(f"  Saved: {conf_save_path}")
    print(f"  Shape: {all_confidences_flat.shape}")
    
    # ===== 4. Verification =====
    print("\n[Step 4: Verification]")
    
    # Reload (결과 확인)
    loaded = np.load(conf_save_path)
    print(f"  Reloaded shape: {loaded.shape}")
    print(f"  Mean: {loaded.mean():.3f}")
    print(f"  Std: {loaded.std():.3f}")
    
    # Check alignment with labels
    print(f"\n  Label-Confidence alignment:")
    print(f"    Normal segments confidence: {loaded[pseudo_labels == 0].mean():.3f}")
    print(f"    Abnormal segments confidence: {loaded[pseudo_labels == 1].mean():.3f}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()