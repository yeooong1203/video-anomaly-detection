# FRD 1crop Required Files Check

- Requested project root: `C:\Users\EZ\Dev\vad_merge`
- Requested project root exists: False
- Reports written under: `C:\Users\jplabuser\vad_merge\runs\frd_1crop_sht_dataset_audit`
- Local FRD repo/root inspected: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD`
- Local FRD git commit: `83704ecd6caab45481acf2f3143b777494cad9f2`
- Local FRD remotes: `origin	https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020.git (fetch)
origin	https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020.git (push)`
- Downloaded dataset root inspected: `C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\dataset\shanghaitech\shanghaitech`

## Required Files
- `train_split.txt`: exists=True, path=`C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\dataset\shanghaitech\shanghaitech\train_split.txt`, size=1728
- `test_split.txt`: exists=True, path=`C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\dataset\shanghaitech\shanghaitech\test_split.txt`, size=1436
- `GT/frame_label.pickle`: exists=True, path=`C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\dataset\shanghaitech\shanghaitech\GT\frame_label.pickle`, size=2236192
- `GT/video_label.pickle`: exists=True, path=`C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\dataset\shanghaitech\shanghaitech\GT\video_label.pickle`, size=9300
- `features_video/i3d/combine`: exists=True, path=`C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\dataset\shanghaitech\shanghaitech\features_video\i3d\combine`, size=n/a
- `best_ckpt/ST.pth`: exists=True, path=`C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\best_ckpt\ST.pth`, size=8920357
- `run_test.py`: exists=True, path=`C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\run_test.py`, size=2780
- `video_dataset_anomaly_balance_uni_sample.py`: exists=True, path=`C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\video_dataset_anomaly_balance_uni_sample.py`, size=10422
- `eval.py`: exists=True, path=`C:\Users\jplabuser\Downloads\FRD-UVAD\CRD-UVAD\eval.py`, size=6355

## Split And Feature Coverage
- train_split entries: 238
- test_split entries: 199
- combine feature video count: 437
- split names equal combine feature names: True
- split names missing combine features: []
- combine features not in splits: []

## GT Pickle Summary
- `frame_label.pickle`: type=dict, entries=437, sample_keys=['07_001', '06_008', '05_0017', '01_067', '04_012'], sample_value_lengths=[640, 1104, 433, 1376, 944, 736, 320, 736, 1360, 752]
- `video_label.pickle`: type=dict, entries=437, sample_keys=['07_001', '06_008', '05_0017', '01_067', '04_012'], sample_value_lengths=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
