# FRD Run-Test Protocol Summary

- FRD root inspected: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD`
- `run_test.sh`: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\run_test.sh`
- `run_test.py`: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\run_test.py`
- Dataset loader: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\video_dataset_anomaly_balance_uni_sample.py`

## Commands Found In run_test.sh
### Avenue
- Command: `python run_test.py --dataset_name Avenue --ckpt_path best_ckpt/avenue_91.23.pkl --feature_pretrain_model i3d --feature_modal combine --Vitblock_num 2`
- `dataset_name`: `Avenue`
- `feature_pretrain_model`: `i3d`
- `feature_modal`: `combine`
- `ckpt_path`: `best_ckpt/avenue_91.23.pkl`
- `Vitblock_num`: `2`
- `cross_clip`: `(default 4)`

### shanghaitech
- Command: `python run_test.py --dataset_name shanghaitech --ckpt_path best_ckpt/ST.pth --feature_pretrain_model i3d --feature_modal combine --Vitblock_num 2 --cross_clip 4`
- `dataset_name`: `shanghaitech`
- `feature_pretrain_model`: `i3d`
- `feature_modal`: `combine`
- `ckpt_path`: `best_ckpt/ST.pth`
- `Vitblock_num`: `2`
- `cross_clip`: `4`

### UCF-Crime
- Command: `python run_test.py --dataset_name ucfcrime  --ckpt_path best_ckpt/UCF.pth --feature_pretrain_model i3d --feature_modal rgb --Vitblock_num 8 --cross_clip 4`
- `dataset_name`: `ucfcrime`
- `feature_pretrain_model`: `i3d`
- `feature_modal`: `rgb`
- `ckpt_path`: `best_ckpt/UCF.pth`
- `Vitblock_num`: `8`
- `cross_clip`: `4`

## Code-Level Assumptions
- `run_test.py` default `feature_modal` is `combine`, and it constructs the inference model with `feature_size=2048`.
- `video_dataset_anomaly_balance_uni_sample.py` resolves feature path as `./dataset/<dataset_name>/features_video/<feature_pretrain_model>/<feature_modal>/<video_name>/feature.npy`.
- The loader squeezes only singleton 3D feature layouts `(T, 1, D)` and `(1, T, D)`. It does not collapse `(T, 10, D)` 10crop features.
- `feature_modal=combine` must therefore provide 2048D per-snippet features for the local checkpoint/model path.

