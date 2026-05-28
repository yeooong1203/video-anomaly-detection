python test.py \
  --datasetname ShanghaiTech \
  --ckpt-path unsupervised_ckpt/ShanghaiTech_all_cnn_final_20260528_131708_k31nfzbi.pkl \
  --test-conall-path ../data/shanghaitech/processed/ShanghaiTech_GCN_test_features.npy \
  --test-nalist-path ../data/shanghaitech/processed/ShanghaiTech_GCN_test_nalist.npy \
  --gt ../data/shanghaitech/processed/ShanghaiTech_GCN_test_gt_segment.npy\
  --video-list-path ../data/shanghaitech/processed/ShanghaiTech_GCN_test_video_names.list \
  --feature-size 2048 \
  --frame-repeat 16 \
  --test-batch-size 1\
  --warmup-segments 5\
  --tta-q 1.0 \
  --tta-min-keep 3 \
  --tta-lr 1e-2\
  --tta-steps-per-video 5 \
  --output-dir demo_exports/SHT \
  --plot-y 0.75 \


#MODEL note
#ShanghaiTech_all_cnn_final_20260528_131928_ey7x35aw -> w/o TA,CR
#ShanghaiTech_all_cnn_final_20260528_131708_k31nfzbi -> w/ TA, w/o CR
