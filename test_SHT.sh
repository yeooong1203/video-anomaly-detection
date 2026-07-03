python test.py \
  --datasetname ShanghaiTech \
  --ckpt-path unsupervised_ckpt/ShanghaiTech_all_cnn_final.pkl \
  --test-conall-path SHT_concat_test.npy \
  --test-nalist-path list/SHT_nalist_test.npy \
  --gt list/gt-SHT-frame.npy\
  --video-list-path list/sht-test_video.list \
  --feature-size 2048 \
  --frame-repeat 16 \
  --test-batch-size 1\
  --warmup-segments 5\
  --tta-q 1.0 \
  --tta-min-keep 1 \
  --tta-lr 1e-2\
  --tta-steps-per-video 30 \
  --output-dir demo_exports/SHT \
  --plot-y 0.1 \

