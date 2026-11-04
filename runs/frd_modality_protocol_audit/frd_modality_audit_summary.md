# FRD Modality Audit Summary

- FRD root: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD`
- ShanghaiTech dataset root inspected: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\dataset\shanghaitech\shanghaitech`
- Reports dir: `C:\Users\jplabuser\vad_merge\runs\frd_modality_protocol_audit`

## Official Modality From run_test.sh
- ShanghaiTech uses `feature_modal=combine`, `feature_pretrain_model=i3d`, `ckpt_path=best_ckpt/ST.pth`, `Vitblock_num=2`, `cross_clip=4`.
- UCF-Crime uses `feature_modal=rgb`, `feature_pretrain_model=i3d`, `ckpt_path=best_ckpt/UCF.pth`, `Vitblock_num=8`, `cross_clip=4`.
- CUHK Avenue command present: True; feature_modal=combine, ckpt_path=best_ckpt/avenue_91.23.pkl, Vitblock_num=2, cross_clip=(default 4).

## Local ShanghaiTech Feature Folders
- Existing checked folders under `features_video/i3d`: ['RGB', 'Flow', 'flow', 'rgb', 'combine']
- Uppercase `RGB` exists: True
- Uppercase `Flow` exists: True
- Lowercase `rgb` exists: True
- Lowercase `flow` exists: True
- `combine` exists: True

### `RGB`
- videos: 437
- split names match train/test splits: True
- shape examples: ['(47, 1024)', '(16, 1024)', '(27, 1024)', '(21, 1024)', '(30, 1024)', '(37, 1024)', '(31, 1024)', '(25, 1024)', '(28, 1024)', '(19, 1024)']
- shape categories: ['(T, 1024)']
- feature dimensions: [1024]
- total train T: 10734
- total test T: 8781
- looks like: RGB-only
- arranged as `<video_name>/feature.npy`: True

### `Flow`
- videos: 437
- split names match train/test splits: True
- shape examples: ['(47, 1024)', '(16, 1024)', '(27, 1024)', '(21, 1024)', '(30, 1024)', '(37, 1024)', '(31, 1024)', '(25, 1024)', '(28, 1024)', '(19, 1024)']
- shape categories: ['(T, 1024)']
- feature dimensions: [1024]
- total train T: 10734
- total test T: 8781
- looks like: Flow-only
- arranged as `<video_name>/feature.npy`: True

### `flow`
- videos: 437
- split names match train/test splits: True
- shape examples: ['(47, 1024)', '(16, 1024)', '(27, 1024)', '(21, 1024)', '(30, 1024)', '(37, 1024)', '(31, 1024)', '(25, 1024)', '(28, 1024)', '(19, 1024)']
- shape categories: ['(T, 1024)']
- feature dimensions: [1024]
- total train T: 10734
- total test T: 8781
- looks like: Flow-only
- arranged as `<video_name>/feature.npy`: True

### `rgb`
- videos: 437
- split names match train/test splits: True
- shape examples: ['(47, 1024)', '(16, 1024)', '(27, 1024)', '(21, 1024)', '(30, 1024)', '(37, 1024)', '(31, 1024)', '(25, 1024)', '(28, 1024)', '(19, 1024)']
- shape categories: ['(T, 1024)']
- feature dimensions: [1024]
- total train T: 10734
- total test T: 8781
- looks like: RGB-only
- arranged as `<video_name>/feature.npy`: True

### `combine`
- videos: 437
- split names match train/test splits: True
- shape examples: ['(47, 2048)', '(16, 2048)', '(27, 2048)', '(21, 2048)', '(30, 2048)', '(37, 2048)', '(31, 2048)', '(25, 2048)', '(28, 2048)', '(19, 2048)']
- shape categories: ['(T, 2048)']
- feature dimensions: [2048]
- total train T: 10734
- total test T: 8781
- looks like: RGB+Flow combine
- arranged as `<video_name>/feature.npy`: True

## Specific Verifications
- ShanghaiTech official FRD inference is supposed to use `feature_modal=combine`: True
- UCF-Crime official FRD inference is supposed to use `feature_modal=rgb`: True
- Local ShanghaiTech `combine` contains `(T, 2048)` features: True
- Local ShanghaiTech lowercase `rgb` contains separate `(T, 1024)` features: True
- Local ShanghaiTech lowercase `flow` contains separate `(T, 1024)` features: True
- Path using `features_video/i3d/RGB/<video_name>/feature.npy` is wrong for official SHT reproduction: True

## Run Readiness
- `ST.pth` exists: True
- Official expected path exists for `--dataset_name shanghaitech`: False
- Nested downloaded path exists: True
- No inference was run.

## Correct Command
- Readiness: safe only with nested dataset_name workaround, or after placing the inner shanghaitech folder at dataset/shanghaitech
- Working directory: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD`
- Command for current nested layout: `python run_test.py --dataset_name shanghaitech/shanghaitech --ckpt_path best_ckpt/ST.pth --feature_pretrain_model i3d --feature_modal combine --Vitblock_num 2 --cross_clip 4`

## Reports
- `C:\Users\jplabuser\vad_merge\runs\frd_modality_protocol_audit\frd_run_test_protocol_summary.md`
- `C:\Users\jplabuser\vad_merge\runs\frd_modality_protocol_audit\frd_sht_feature_modality_shapes.csv`
- `C:\Users\jplabuser\vad_merge\runs\frd_modality_protocol_audit\frd_modality_audit_summary.md`
