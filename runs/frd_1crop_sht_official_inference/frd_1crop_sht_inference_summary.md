# FRD 1crop ShanghaiTech Official Inference Summary

## Run
- Exact command used: `python run_test.py --dataset_name shanghaitech/shanghaitech --ckpt_path best_ckpt/ST.pth --feature_pretrain_model i3d --feature_modal combine --Vitblock_num 2 --cross_clip 4`
- Working directory: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD`
- Dataset path used by FRD code: `.\dataset\shanghaitech\shanghaitech`
- Checkpoint path used: `best_ckpt/ST.pth`
- Feature modality used: `combine`
- First console log: `C:\Users\jplabuser\vad_merge\runs\frd_1crop_sht_official_inference\frd_1crop_sht_console_log.txt`
- Retry console log: `C:\Users\jplabuser\vad_merge\runs\frd_1crop_sht_official_inference\frd_1crop_sht_console_log_retry1.txt`

## Result
- Exit code: `1`
- Final frame-level AUC: not produced
- Reproduces approximately 80.14: no, inference did not start successfully
- FRD result directory created: no

## Attempts
### Attempt 1
The first run failed during Python imports before model loading or dataset iteration:

```text
ModuleNotFoundError: No module named 'utils'
```

Trace path:

```text
run_test.py -> from eval import eval_p
eval.py -> from utils import scorebinary, anomap
```

`utils.py` is missing from `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD`; a recursive search under that checkout did not find it.

### Retry 1
After `utils.py` was added, the retry got past `eval.py` and failed at the next missing module:

```text
ModuleNotFoundError: No module named 'losses'
```

Trace path:

```text
run_test.py -> from models_test import model
models_test.py -> from losses import *
```

`losses.py` is missing from `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD`; a recursive search under that checkout did not find it.

## Safety
- No training was run.
- No model evaluation completed.
- No FRD `result` output directory was produced.
- The dataset/checkpoint/protocol is still the correct FRD-like ShanghaiTech setting by modality and file layout, but this local FRD checkout is incomplete for official inference until the missing `losses.py` dependency is restored.
