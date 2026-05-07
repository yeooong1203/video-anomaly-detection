python mainv2.py \
    --use_variable_length \
    --model_type all_cnn \
    --batch_size_video 32 \
    --temporal_kernel 5 \
    --pseudofile Unsup_labels/pseudo_labels_swap_90.npy \
    --train-conall-path concat_UCF.npy \
    --test-conall-path Concat_test_10.npy \
    --max-epoch 30 \
    --lr 0.001