python test.py \
  --datasetname UCF \
  --ckpt-path  unsupervised_ckpt/UCF_all_cnn_final_20260609_035150_tco5tx0u.pkl \
  --test-conall-path Concat_test_10.npy \
  --test-nalist-path list/nalist_test_i3d.npy \
  --gt list/gt-ucf-RTFM.npy \
  --video-list-path list/ucf-i3d_test_fixed_local.list \
  --feature-size 2048 \
  --frame-repeat 16 \
  --test-batch-size 1\
  --warmup-segments 5 \
  --tta-q 1.0 \
  --tta-min-keep 3 \
  --tta-lr 1e-2 \
  --tta-steps-per-video 30 \
  --output-dir demo_exports/UCF \
  --plot-y 0.35 \


#MODEL note 
#UCF ABLATION
#ShanghaiTech_all_cnn_final_20260528_131928_ey7x35aw -> w/o TempAtt, w/o Conf
#ShanghaiTech_all_cnn_final_20260528_131708_k31nfzbi -> w/ TempAtt, w/o Conf
#ShanghaiTech_all_cnn_final_20260529_182252_n4q3m64v -> w/ TempAtt, w/ Conf
