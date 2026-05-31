import argparse

parser = argparse.ArgumentParser(description='C2FPL')
parser.add_argument('--wandb-mode', type=str, default='online', choices=['online', 'offline', 'disabled'], help='Weights & Biases logging mode')

parser.add_argument('--datasetname', default='UCF', help='dataset to train on')
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature')
parser.add_argument('--workers', type=int, default=0, help='number of workers in dataloader')
parser.add_argument('--gt', default='list/gt-ucf-RTFM.npy', help='file of ground truth ')
parser.add_argument('--lr', type=float, default=0.001, help='train learning rate')
parser.add_argument('--train-batch-size', type=int, default=32, help='Batch size for video-level training')
parser.add_argument('--window-size', type=int, default=500)
parser.add_argument('--stride', type=int, default=500)
parser.add_argument('--max-epoch', type=int, default=50, help='maximum iteration to train')

parser.add_argument('--use_variable_length', action='store_true', help='Use variable-length video training')
parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'temporal', 'lstm', 'all_cnn', 'all_lstm', 'hybrid_cnn'], help='Model type: mlp (no temporal), temporal (Conv1d), lstm (LSTM)')
parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--temporal_kernel', type=int, default=5, help='Kernel size for temporal conv')

parser.add_argument('--pseudo-output',type=str,  default='UCF_pseudo_labels.npy', help='Output path for generated pseudo labels')
parser.add_argument('--pseudofile',type=str,  default='Unsup_labels/UCF_pseudo_labels.npy', help='ground truth file')
parser.add_argument('--train-conall-path', type=str, default='../C2FPL/concat_UCF.npy', help='Path to concatenated train feature memmap')
parser.add_argument('--train-nalist-path', type=str, default='list/nalist_i3d.npy', help='Path to train video start/end index list')
parser.add_argument('--test-conall-path', type=str, default='Concat_test_10.npy', help='Path to concatenated test feature memmap')
parser.add_argument('--test-nalist-path', type=str, default='list/nalist_test_i3d.npy', help='Path to test video start/end index list')
parser.add_argument('--confidence-path', type=str, default='list/UCF_confidence_scores.npy')
parser.add_argument('--ckpt-dir', type=str, default='unsupervised_ckpt', help='Directory to save training checkpoints')
parser.add_argument('--ckpt-path', type=str, default='unsupervised_ckpt/UCF_all_cnn_best_20260514_030319_0k0lg0eg.pkl', help='Path to model checkpoint for test/TTA')
parser.add_argument('--video-list-path', type=str, default='list/ucf-i3d_test_fixed_local.list', help='Path to model checkpoint for test/TTA')

parser.add_argument('--test-batch-size', type=int, default=1)
parser.add_argument('--warmup-segments', type=int, default=5)
parser.add_argument('--tta-q', type=float, default=1.0)
parser.add_argument('--tta-min-keep', type=int, default=8)
parser.add_argument('--tta-lr', type=float, default=0.01)
parser.add_argument('--tta-steps-per-video', type=int, default=30)
parser.add_argument('--plot-y', type=float, default=1.0)
parser.add_argument('--frame-repeat', type=int, default=16)
parser.add_argument('--selected-vid-indices', type=str, default='17,30,97,230')
parser.add_argument('--output-dir', type=str, default='demo_exports')





