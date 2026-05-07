import argparse

parser = argparse.ArgumentParser(description='C2FPL')
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')
parser.add_argument('--gt', default='list/gt-ucf-RTFM.npy', help='file of ground truth ')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=128, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', type=int, default=0, help='number of workers in dataloader')
parser.add_argument('--datasetname', default='UCF', help='dataset to train on (default: )')
parser.add_argument('--max-epoch', type=int, default=100, help='maximum iteration to train (default: 100)')

parser.add_argument('--optimizer', default='SGD', help='Number of segments of each video')
parser.add_argument('--lossfn', default='BCE', help='Number of segments of each video')
parser.add_argument('--stepsize',type=int,  default=5, help='lr_scheduler stepsize')

parser.add_argument('--windowsize',type=float,  default=0.09, help='lr_scheduler stepsize')
parser.add_argument('--modelversion',type=str,  default='Model_V2', help='Model version')
parser.add_argument('--pseudofile',type=str,  default='Unsup_labels/pseudo_labels_swap_90.npy', help='ground truth file')
parser.add_argument('--conall', type=str, default='concat_UCF', help='Legacy dataset name/prefix option')
parser.add_argument('--train-conall-path', type=str, default='../C2FPL/concat_UCF.npy', help='Path to concatenated train feature memmap')
parser.add_argument('--train-nalist-path', type=str, default='list/nalist_i3d.npy', help='Path to train video start/end index list')
parser.add_argument('--test-conall-path', type=str, default='Concat_test_10.npy', help='Path to concatenated test feature memmap')
parser.add_argument('--test-nalist-path', type=str, default='list/nalist_test_i3d.npy', help='Path to test video start/end index list')

parser.add_argument('--use_variable_length', action='store_true',
                   help='Use variable-length video training')
parser.add_argument('--model_type', type=str, default='mlp',
                   choices=['mlp', 'temporal', 'lstm', 'all_cnn', 'all_lstm', 'hybrid_cnn'],
                   help='Model type: mlp (no temporal), temporal (Conv1d), lstm (LSTM)')
parser.add_argument('--batch_size_video', type=int, default=8,
                   help='Batch size for video-level training')
parser.add_argument('--accumulation_steps', type=int, default=1,
                   help='Gradient accumulation steps')
parser.add_argument('--temporal_kernel', type=int, default=5,
                   help='Kernel size for temporal conv')
parser.add_argument('--lstm_hidden', type=int, default=64,
                   help='LSTM hidden size')

parser.add_argument('--ckpt-path', type=str, default='unsupervised_ckpt/UCF_all_cnn_final_20260331_020353_wv5ldb2h.pkl', help='Path to model checkpoint for test/TTA')
parser.add_argument('--wandb-mode', type=str, default='online', choices=['online', 'offline', 'disabled'], help='Weights & Biases logging mode')
parser.add_argument('--ckpt-dir', type=str, default='unsupervised_ckpt', help='Directory to save training checkpoints')
parser.add_argument('--pseudo-output', type=str, default='pseudo_labels_swap.npy', help='Output path for generated pseudo labels')