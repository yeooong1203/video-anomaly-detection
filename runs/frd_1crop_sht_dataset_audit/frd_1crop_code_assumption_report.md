# FRD 1crop Code Assumption Report

- Local code root inspected: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD`
- Public upstream checked: `https://github.com/tcc-power/FRD-unsupervised-video-anomaly-detection-1crop`

## Expected Feature Path
- `video_dataset_anomaly_balance_uni_sample.py` builds `./dataset/<dataset_name>/features_video/<feature_pretrain_model>/<feature_modal>/<video_name>/feature.npy`.
- With the reference command, this resolves to `dataset/shanghaitech/features_video/i3d/combine/<video_name>/feature.npy`.

## Expected Feature Shape
- `run_test.py` constructs the test model with `feature_size=2048` for `feature_modal=combine`.
- The dataset loader accepts normal 2D features and explicitly squeezes only two singleton 3D layouts: `(T, 1, D)` via `squeeze(axis=1)` and `(1, T, D)` via `squeeze(axis=0)`.
- Therefore `(T, 2048)`, `(T, 1, 2048)`, and `(1, T, 2048)` are compatible for combine features.
- `(T, 10, 2048)` is not handled as 10crop; it would not be squeezed and would add an extra dimension into the DataLoader/model path, so it should not be treated as accepted without modification.

## Frame-Level AUC
- `eval.py` loads `GT/frame_label.pickle` and `GT/video_label.pickle`.
- For each video prediction vector `v`, it sets `new_len = int(len(frame_labels) / 16)`, truncates `v` to `new_len`, repeats each score 16 times, truncates frame labels to the repeated prediction length, then computes `roc_auc_score` over concatenated frame labels and repeated scores.

## Local Downloaded Dataset Fit
- Downloaded `combine` features are `(T, 2048)` with no crop dimension.
- Downloaded `rgb` and `flow` features are `(T, 1024)` each, also no crop dimension.
- This matches FRD 1crop layout for `feature_modal=combine` and does not look like 10crop.
