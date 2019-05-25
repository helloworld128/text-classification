import torch
import train
import cnn
import lstm
import mlp
import argparse
from parse_data import MyDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='CNN text classifier')
# learning
parser.add_argument('-lr', type=float, default=0.005, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=30, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.4, help='the probability for dropout [default: 0.5]')
# parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 300]')
parser.add_argument('-kernel-num', type=int, default=64, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5,7', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=True, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')



args = parser.parse_args()
args.label_num = 8
args.cuda = torch.cuda.is_available() and not args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
tmp0 = torch.load('data/train.pt')
tmp1 = torch.load('data/test.pt')

args.model = 'mlp'

if args.model == 'cnn':
    train_data = DataLoader(tmp0, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_data = DataLoader(tmp1, batch_size=args.batch_size, shuffle=True, num_workers=8)
    mlpModel = cnn.TextCNN(args)
    try:
        train.train(train_data, test_data, mlpModel, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')
elif args.model == 'rnn':
    rnnModel = lstm.RNN('GRU', 300, 64, 8, 2, True, 0.3)
    train_data = DataLoader(tmp0, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_data = DataLoader(tmp1, batch_size=args.batch_size, shuffle=True, num_workers=8)
    try:
        train.train(train_data, test_data, rnnModel, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')
elif args.model == 'mlp':
    train_data = DataLoader(tmp0, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_data = DataLoader(tmp1, batch_size=args.batch_size, shuffle=True, num_workers=8)
    mlpModel = mlp.MLP(args)
    try:
        train.train(train_data, test_data, mlpModel, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')



