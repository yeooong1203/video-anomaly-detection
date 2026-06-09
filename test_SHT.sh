python test.py \
  --datasetname ShanghaiTech \
  --ckpt-path unsupervised_ckpt/ShanghaiTech_all_cnn_final_20260609_173515_58ykf0dy.pkl \
  --test-conall-path SHT_concat_test.npy \
  --test-nalist-path list/SHT_nalist_test.npy \
  --gt list/gt-SHT-segment.npy\
  --video-list-path list/sht-test_video.list \
  --feature-size 2048 \
  --frame-repeat 16 \
  --test-batch-size 1\
  --warmup-segments 3\
  --tta-q 1.0 \
  --tta-min-keep 3 \
  --tta-lr 1e-2\
  --tta-steps-per-video 30 \
  --output-dir demo_exports/SHT \
  --plot-y 0.6 \


#MODEL note
#SHT ABLATION
#ShanghaiTech_all_cnn_final_20260528_131928_ey7x35aw -> w/o TempAtt, w/o Conf
#ShanghaiTech_all_cnn_final_20260528_131708_k31nfzbi -> w/ TempAtt, w/o Conf
#ShanghaiTech_all_cnn_final_20260529_182252_n4q3m64v -> w/ TempAtt, w/ Conf
