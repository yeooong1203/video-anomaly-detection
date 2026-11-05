# 학교 PC에서 XD-Violence 실행

이 폴더는 해당 checkout의 원본 모델·학습 함수·pseudo 생성기·ITA를 연결한다. 기존 UCF/SHT 파일을 수정하지 않는다. **원본 저장소와 별도 경로에 준비 폴더를 복사**하고, 공식 배포 feature도 학교 PC로 옮긴 뒤 실행한다. 추가 I3D 추출은 필요 없다.

## 환경

Python **3.11 이상**, CUDA를 사용할 수 있는 PyTorch **2.1 이상**이 필요하다. 기존 연구 환경과 분리한 XD 전용 환경을 권장한다. PyTorch는 학교 PC 드라이버/CUDA 환경에 맞춰 별도로 설치하고, 이미 작동하는 GPU용 PyTorch를 일반 requirements 설치로 교체하지 않는다.

환경을 활성화한 뒤 확인한다.

```powershell
python -c "import sys,torch; print(sys.version); print(torch.__version__,torch.version.cuda); print(torch.cuda.is_available()); assert sys.version_info >= (3,11); assert torch.cuda.is_available()"
```

나머지 필요한 패키지만 설치한다(`--upgrade`를 사용하지 않는다). `requirements.txt`는 최소 요구사항 참고용이며 PyTorch 항목도 들어 있다.

```powershell
python -m pip install "numpy>=1.24" "scipy>=1.10" "scikit-learn>=1.3" "matplotlib>=3.7" "tqdm>=4.65"
python -m pip check
```

## 경로와 공통 설정

아래 예시 경로는 학교 PC의 **새 작업 폴더**에 맞춰 바꾼다. `$xdFeatures`에는 `RGB`와 `RGBTest`가 들어 있는 폴더를 지정한다. 상위 `xd-i3d-features` 폴더도 자동 인식한다. Flow·FlowTest와 audio는 사용하지 않는다.

```powershell
$xdRoot = 'C:\Users\jplabuser\XD-Violence-Preparation'
$xdBranch = 'APWCS'  # 변형 모델 실행 시 'UCFSHT'
$xdFeatures = 'C:\Users\jplabuser\xd-i3d-features\i3d-features'
$xdCache = Join-Path $xdRoot 'cache\rgb-five-crop'
$xdOutput = Join-Path $xdRoot ("runs\{0}_seed42" -f $xdBranch)
Set-Location -LiteralPath (Join-Path $xdRoot $xdBranch)

$xdArgs = @(
    '--features', $xdFeatures,
    '--cache', $xdCache,
    '--output', $xdOutput,
    '--device', 'cuda',
    '--seed', '42',
    '--epochs', '30',
    '--lr', '0.001',
    '--batch-size', '8',
    '--window', '500',
    '--stride', '500',
    '--ita',
    '--k', '5',
    '--q', '1.0',
    '--min-keep', '3',
    '--ita-lr', '0.01',
    '--ita-steps', '30'
)
```

두 브랜치는 **같은 RGB cache**를 사용하며 output만 다르게 지정한다. cache와 run output을 원본 feature 폴더나 원본 연구 저장소 안에 지정하지 않는다. seed를 바꾸면 output 폴더 이름도 바꾼다. 기존 run에 설정을 바꿔 덮어쓰거나 암묵적으로 이어 학습하지 않는다.

## 실행

선택 사항으로, 아래 CPU 준비를 학교 PC에서 **한 번만** 수행할 수 있다. 모든 원본 RGB payload의 유한 값과 crop 길이를 검사하고 실제 다섯 crop의 평균 cache를 만든다. 두 브랜치용으로 별도 추출·복제할 필요가 없다.

```powershell
python -m xd_experiment.run --stage prepare --features $xdFeatures --cache $xdCache --device cpu
```

준비·pseudo 생성·confidence 생성·학습·평가를 연결한다. cache가 이미 있으면 재사용한다. 아직 없다면 `all`이 준비부터 수행하므로 위 `prepare`를 생략해도 된다.

```powershell
python -m xd_experiment.run --stage all @xdArgs
```

단계별 실행이 필요하면 동일한 `$xdArgs`를 유지한다.

```powershell
python -m xd_experiment.run --stage pseudo @xdArgs
python -m xd_experiment.run --stage train @xdArgs
python -m xd_experiment.run --stage evaluate @xdArgs
```

`all`과 단계별 명령 중 한 방식을 선택한다. 완료된 stage를 같은 output에서 다시 실행하면 오류로 중단한다. `--ita` 등 평가 옵션도 run identity에 포함되므로 단계 사이에서 빼거나 바꾸지 않는다. CUDA를 요청했는데 사용할 수 없으면 CPU 장시간 학습으로 전환하지 않고 중단한다.

이후 `$xdBranch = 'UCFSHT'`와 해당 output으로 경로·인자 배열을 다시 설정하고 같은 명령을 실행한다. 학습은 각 브랜치의 새 detector로 시작하며 UCF/SHT checkpoint를 불러오지 않는다. 학습 정답 video label과 test GT를 pseudo 생성·loss·checkpoint 선택에 사용하지 않는다. 사전에 정한 30 epoch의 final을 평가한다.

## Feature와 평가 protocol

- 공식 split: train 3,954 / test 800 영상. 파일 하나는 `(T,1024)`이고 실제 crop 파일 `__0`~`__4` 다섯 개를 평균하여 `(T,1024)`로 사용한다. 시간축 보간·D padding·가짜 crop 복제는 없다. 미니배치 안의 가변 길이 zero padding은 기존 학습 mask와 함께 사용한다.
- 출력 score 한 개는 공식 feature의 16frame에 대응한다. 공개 annotation `[start:end]`로 재구성한 GT는 원본 공식 GT **2,330,384frame과 완전히 일치**했다. 원본 영상 마지막 불완전 구간을 추가한 평가는 하지 않는다.
- 주 지표 `ap`: 전체 frame을 연결한 `sklearn.average_precision_score`. `official_pr_auc`는 원저자 `auc(recall,precision)`로 따로 저장한다. 두 값은 계산 방식이 다르다. `roc_auc`는 보조 지표다.
- 지표 기본값은 **0~1**, `_percent` 필드는 해당 값×100이다. per-video AP 평균을 전체 AP로 보고하지 않는다.
- `raw_full`: test **800개 전체**의 detector 결과. `ln_suffix`와 `ita_suffix`는 초기 5clip(80frame)을 동일하게 제외한 비교다. suffix는 **799개 영상 / 2,266,384frame**, 제외량은 64,000frame이다. 5clip짜리 영상 하나도 raw 전체 평가에는 포함된다. 다른 논문의 full-test 수치와 suffix 수치를 같은 조건으로 비교하지 않는다.
- 원본 APWCS/UCFSHT의 CNN·adapter 구조는 동일하지만 pseudo 생성·학습 loss·confidence 적용은 다르다. 각 checkout의 원본 구현을 독립적으로 로드한다. 공통 window500/stride500/batch8/SGD lr0.001/scheduler milestones20·30은 **새 XD 통제 설정**이며 과거 논문 hyperparameter를 그대로 재현했다고 주장하지 않는다.

## 결과와 검토 항목

`metrics.json`, `coverage.json`, 영상별 `scores/*.npz`, `final.pt`, `checkpoint.json`, `train.csv`, `identity.json`, `environment.json` 및 pseudo/confidence metadata를 보관한다. 원본 코드 SHA-256은 CRLF→LF만 정규화해서 검사하며 다른 byte 변경은 거절한다. 설정·code·cache hash가 다르면 새 output을 사용한다.

검토할 사항:

- APWCS는 `train.sh`의 `*_ng.npy` 이름과 frozen generator 주석을 근거로 `pseudo_proto_dist_swap.py`의 main 설정을 선택했다. README의 일반 `pseudo.py`와 다르며 과거 실제 실행 이력은 입증되지 않았다. APWCS confidence refinement는 활성, UCFSHT는 원본대로 비활성이다.
- 원본 ITA의 `(N,D)` 입력 시간축 처리와 `(N,)×(N,1)` broadcasting을 **보존**했다. 이를 수정한 성능은 별도 실험으로 구분해야 한다. 계산만 하는 margin/reg를 새 loss에 추가하지 않았다.
- 이 패키지는 로컬 CPU에서 데이터 구조·GT 정렬 및 합성 model/pseudo/train/ITA 연결을 검증했다. **실제 XD GPU 학습·성능·RTX 3060 peak memory는 미검증**이다. 학교 PC에서 환경 점검과 작은 실행을 거친 뒤 고정된 본 실험을 진행한다.

공식 출처: [feature·split·dataset](https://roc-ng.github.io/XD-Violence/), [원저자 평가 및 GT 생성 코드](https://github.com/Roc-Ng/XDVioDet/tree/f846d8cfc454943ec165bd2489b71e7dc0a1064d), [annotation 원본](https://github.com/Roc-Ng/XD-Violence/blob/3d4db78b15db767d31b1799be08b33a7bead9219/images/annotations.txt). Source/GT provenance는 상위 작업 폴더의 `protocol-audit/`에서 검토할 수 있다.
