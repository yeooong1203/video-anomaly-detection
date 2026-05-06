python mainv2.py \
    --use_variable_length \
    --model_type all_cnn \
    --batch_size_video 32 \
    --temporal_kernel 5 \
    --pseudofile Unsup_labels/pseudo_labels_swap_90.npy \
    --max-epoch 30