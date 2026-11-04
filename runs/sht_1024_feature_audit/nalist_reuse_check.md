# SHT 1024D NALIST Reuse Check

- Project root: `C:\Users\jplabuser\vad_merge`
- Public feature root: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\dataset\shanghaitech`
- Public 1024D primary group used for comparison: `rgb`

## Public Feature Structure
- `rgb`: 437 files, dims=['1024'], example_shapes=['(103, 1024)', '(106, 1024)', '(112, 1024)', '(114, 1024)', '(115, 1024)', '(12, 1024)', '(13, 1024)', '(14, 1024)'], any_crop_dim=False
- `flow`: 437 files, dims=['1024'], example_shapes=['(103, 1024)', '(106, 1024)', '(112, 1024)', '(114, 1024)', '(115, 1024)', '(12, 1024)', '(13, 1024)', '(14, 1024)'], any_crop_dim=False
- `combine`: 437 files, dims=['2048'], example_shapes=['(103, 2048)', '(106, 2048)', '(112, 2048)', '(114, 2048)', '(115, 2048)', '(12, 2048)', '(13, 2048)', '(14, 2048)'], any_crop_dim=False

## Existing 2048D Protocol
- Existing train nalist path: `C:\Users\jplabuser\vad_merge\list\SHT_nalist_train.npy`
- Existing test nalist path: `C:\Users\jplabuser\vad_merge\list\SHT_nalist_test.npy`
- Existing train total T from current nalist: 10762
- Existing test total T from current nalist: 8803
- Existing train feature shape: (10762, 10, 2048), dtype: float32
- Existing test feature shape: (8803, 10, 2048), dtype: float32

## Public 1024D Per-Group Totals
- `flow`: train total T=10734, test total T=8781, train missing=0, test missing=0
- `rgb`: train total T=10734, test total T=8781, train missing=0, test missing=0

## Shape Reuse Decision
- Public 1024D train total T: 10734
- Public 1024D test total T: 8781
- Train total T matches existing nalist: False
- Test total T matches existing nalist: False
- Public feature can use existing nalist purely by shape: False
- Existing nalist reusable: False

## Video Order Evidence
- Public train split path: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\dataset\shanghaitech\shanghaitech\train_split.txt`; entries=238
- Public test split path: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\dataset\shanghaitech\shanghaitech\test_split.txt`; entries=199
- Project test order path: `C:\Users\jplabuser\vad_merge\list\sht-test_video.list`; entries=199
- Project test order matches public test order: True
- Project train video-order list was not found, so current train order cannot be verified against public order.

## Decision Notes
- Existing nalist cannot be reused because total T does not match the public 1024D feature totals.
- Public per-video features can support a proposed new nalist from public split order and per-video lengths, but it was not written.
