## 각 데이터셋 별 전체 파이프라인 실행 방법

### UCF
1. pseudo label 생성
    - `python pseudo.py --pseudo-output Unsup_labels/UCF_pseudo_labels.npy`

2. 초기 pseudo label 기반 confidence score 뽑기 (만든 pseudo 파일명에 따라 뒤에 pseudofile 인자로 다른거 주어야 제대로 적용됨!)
    - `python conf_main.py --psuedofile Unsup_labels/UCF_pseudo_labels.npy --confidence-path list/UCF_confidence_scores.npy`

3. confidence score 이용한 label refinement
4. 최종 psuedo label을 가지고 모델 학습 + 평가
    - `sh train.sh`
    - 현재 sh train.sh 돌리면 3,4 과정이 한번에 진행되고 있음. 분리해서 보거나 confidence 모듈과 이 부분 합체해보는 것도 좋을 듯! 

5. TTA - 주어진 모델을 받아와 tta 적용
    - `sh test.sh`

### SHT
1. pseudo label 생성
    - `python pseudo_SHT.py --datasetname ShanghaiTech --train-conall-path SHT_concat_train.npy --train-nalist-path list/SHT_nalist_train.npy  --pseudo-output Unsup_labels/SHT_pseudo_labels.npy`

2. 초기 pseudo label 기반 confidence score 뽑기 (SHT는 인자 직접 주어야 함)
    - `python conf_main.py --train-conall-path SHT_concat_train.npy --train-nalist-path list/SHT_nalist_train.npy --pseudofile Unsup_labels/SHT_pseudo_labels.npy --confidence-path list/SHT_confidence_scores.npy`

3. confidence score 이용한 label refinement
4. 최종 psuedo label을 가지고 모델 학습 + 평가
    - `sh train_SHT.sh`

5. TTA - 주어진 모델을 받아와 tta 적용
    - `sh.test_SHT.sh`


## 데이터셋
UCF_train_feature.zip 다운로드 (train feature - RTFM)
https://drive.google.com/file/d/1i2P9Nn62i0cVil_WS24HKzbzmyUxA9vX/view?usp=sharing

UCF_Test_ten_i3d 다운로드 (test feature - MGFN)
https://github.com/carolchenyx/MGFN.?tab=readme-ov-file,It#dataset-prepare-ucf-crime-ten-crop-i3d-rename-the-data-path-in-ucf-i3dlist-and-ucf-i3d-testlist-based-on-your-data-path

Concat_test_10.npy 파일 다운로드 (test feature concat)
https://drive.google.com/file/d/1T2mKd9_g3PIcpBGbIlXsFvr0QvPhsdZX/view?usp=sharing


