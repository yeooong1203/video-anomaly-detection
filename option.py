import argparse

parser = argparse.ArgumentParser(description='C2FPL')
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')
parser.add_argument('--gt', default='list/gt-ucf-RTFM.npy', help='file of ground truth ')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=128, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=0, help='number of workers in dataloader')
parser.add_argument('--datasetname', default='UCF', help='dataset to train on (default: )')
parser.add_argument('--max-epoch', type=int, default=100, help='maximum iteration to train (default: 100)')

parser.add_argument('--optimizer', default='SGD', help='Number of segments of each video')
parser.add_argument('--lossfn', default='BCE', help='Number of segments of each video')
parser.add_argument('--stepsize',type=int,  default=5, help='lr_scheduler stepsize')

parser.add_argument('--windowsize',type=float,  default=0.09, help='lr_scheduler stepsize')
parser.add_argument('--modelversion',type=str,  default='Model_V2', help='Model version')
parser.add_argument('--pseudofile',type=str,  default='Unsup_labels/pseudo_labels_swap_90.npy', help='ground truth file')
parser.add_argument('--conall',type=str,  default='concat_UCF', help='ground truth file')

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