# FRD 1crop ShanghaiTech Dataset Audit Summary

- Audit status: `Compatible with FRD 1crop`
- Requested project root exists: False
- Reports written under: `C:\Users\jplabuser\vad_merge\runs\frd_1crop_sht_dataset_audit`
- Downloaded dataset root: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\dataset\shanghaitech\shanghaitech`
- Local FRD root: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD`

## Feature Summary
- `combine` (T, 2048), 1crop_no_crop_dim, dtype=float32, dim=2048: count=437, total_T=19515, arranged_all=True, examples=['features_video\\i3d\\combine\\01_001\\feature.npy', 'features_video\\i3d\\combine\\01_0014\\feature.npy', 'features_video\\i3d\\combine\\01_0015\\feature.npy', 'features_video\\i3d\\combine\\01_0016\\feature.npy', 'features_video\\i3d\\combine\\01_002\\feature.npy']
- `flow` (T, 1024), 1crop_no_crop_dim, dtype=float32, dim=1024: count=437, total_T=19515, arranged_all=True, examples=['features_video\\i3d\\flow\\01_001\\feature.npy', 'features_video\\i3d\\flow\\01_0014\\feature.npy', 'features_video\\i3d\\flow\\01_0015\\feature.npy', 'features_video\\i3d\\flow\\01_0016\\feature.npy', 'features_video\\i3d\\flow\\01_002\\feature.npy']
- `rgb` (T, 1024), 1crop_no_crop_dim, dtype=float32, dim=1024: count=437, total_T=19515, arranged_all=True, examples=['features_video\\i3d\\rgb\\01_001\\feature.npy', 'features_video\\i3d\\rgb\\01_0014\\feature.npy', 'features_video\\i3d\\rgb\\01_0015\\feature.npy', 'features_video\\i3d\\rgb\\01_0016\\feature.npy', 'features_video\\i3d\\rgb\\01_002\\feature.npy']

## FRD Compatibility Decision
- `train_split.txt` present: True
- `test_split.txt` present: True
- `GT/frame_label.pickle` present: True
- `GT/video_label.pickle` present: True
- `best_ckpt/ST.pth` present: True
- Combine features are arranged as `<video_name>/feature.npy`: True
- Combine feature shape is consistently `(T, 2048)`: True
- Looks like 10crop: False
- Split names match combine feature folders: True

## Reproducibility Note
- FRD own inference appears runnable in principle because code, `ST.pth`, splits, GT pickles, and combine 1crop features are present. This audit did not run inference.
- The reference command expects the dataset to be visible at `./dataset/shanghaitech` from the FRD repo root. The downloaded folder currently has an extra nested `dataset/shanghaitech/shanghaitech` component, so running as-is may require placing/symlinking that inner folder to the expected path.

## Reports
- `C:\Users\jplabuser\vad_merge\runs\frd_1crop_sht_dataset_audit\frd_1crop_feature_shape_report.csv`
- `C:\Users\jplabuser\vad_merge\runs\frd_1crop_sht_dataset_audit\frd_1crop_required_files_check.md`
- `C:\Users\jplabuser\vad_merge\runs\frd_1crop_sht_dataset_audit\frd_1crop_code_assumption_report.md`
- `C:\Users\jplabuser\vad_merge\runs\frd_1crop_sht_dataset_audit\frd_1crop_audit_summary.md`
