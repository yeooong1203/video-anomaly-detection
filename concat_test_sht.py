import os
import numpy as np
import torch
import torch.nn.functional as F

output_file = 'Concat_test_sht.npy'

feature_dir = "C:\\Users\\jplabuser\\Downloads\\ShanghaiTech_features\\ShanghaiTech_features\\SH_Test_ten_crop_i3d\\SH_Test_ten_crop_i3d"
video_files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.npy')])

num_videos = len(video_files)
target_segments = 32
num_crops = 10
feature_dim = 2048


# 테스트용은 10-crop 차원이 추가되어 용량이 크므로 memmap 권장
fp = np.memmap(output_file, dtype='float32', mode='w+', shape=(num_videos, target_segments, num_crops, feature_dim))

for i, video_name in enumerate(video_files):
    file_path = os.path.join(feature_dir, video_name)
    
    if not os.path.exists(file_path):
        print(f"파일 없음: {video_name}")
        continue
        
    # 데이터 로드: (T, 10, 2048)
    feat = np.load(file_path).astype(np.float32)
    
    # 32 세그먼트 보간 (10개 크롭을 유지하며 수행)
    # torch interpolate를 위해 (Crops, Dim, Time)으로 변경
    feat_tensor = torch.from_numpy(feat).permute(1, 2, 0) # (10, 2048, T)
    
    # 보간 수행: (10, 2048, T) -> (10, 2048, 32)
    resampled_feat = F.interpolate(feat_tensor, size=target_segments, mode='linear', align_corners=False)
    
    # 다시 (32, 10, 2048) 순서로 복구
    resampled_feat = resampled_feat.permute(2, 0, 1).numpy()

    fp[i, :, :, :] = resampled_feat
    
    if i % 50 == 0:
        print(f"테스트 데이터 진행중: {i}/{num_videos}")

fp.flush()
print("🎉 Concat_test_sht.npy 생성 완료!")