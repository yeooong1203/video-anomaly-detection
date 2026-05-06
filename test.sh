python test.py \
    --lossfn BCE \
    --optimizer SGD \
    --max-epoch 20 \
    --outer-epochs 1\
    --batch-size 1\
    --lr 0.01\
    --feature-size 2048\
    --gt list/gt-ucf-RTFM.npy\
    --datasetname UCF \
    --windowsize 0.15\
    --eps 0.225 \
    --eps2 1.32475  \
    --pseudofile Unsup_labels/pseudo_labels_swap_0401_max_2.npy\
    --conall concat_UCF\
    #--create True\