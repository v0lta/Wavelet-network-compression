import torch
from RNN_compression.sequential_mnist import data_generator

import torch
import numpy as np
import pywt
import matplotlib.pyplot as plt
from RNN_compression.cells import GRUCell, FastFoodGRU, WaveletGRU
from RNN_compression.sequential_mnist import data_generator
from util import pd_to_string, compute_parameter_total
import pickle
import collections
import argparse

CustomWavelet = collections.namedtuple('Wavelet', ['dec_lo', 'dec_hi',
                                                   'rec_lo', 'rec_hi', 'name'])

parser = argparse.ArgumentParser(description='Sequence Modeling - Sequential cifar/mnist problems')
parser.add_argument('--problem', type=str, default='MNIST',
                    help='choose MNIST or CIFAR')
parser.add_argument('--cell', type=str, default='WaveletGRU',
                    help='Cell type: Choose GRU or WaveletGRU or FastFoodGRU.')
parser.add_argument('--hidden', type=int, default=64,
                    help='Cell size. Default 512.')
parser.add_argument('--compression_mode', type=str, default='full',
                    help='How to compress the cell options:'
                         'gates, state, reset, update, state_reset, state_update, full')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Choose the number of samples used to during each update step.')
parser.add_argument('--lr', type=float, default=1.0,
                    help='The learning rate.')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: 0.15)')
parser.add_argument('--epochs', type=int, default=20,
                    help='Passes over the entire data set default: 30')
parser.add_argument('--wave_dropout', type=float, default=0.0,
                    help='Dropout within the wavelet layer.')

args = parser.parse_args()


print(args)
pd = vars(args)

if args.problem == 'MNIST':
    root = './data_sets/mnist/'
    input_size = 1
    output_size = 10
    train_loader, test_loader = data_generator(root, args.batch_size)
elif args.problem == 'CIFAR':
    input_size = 3
    root = './data_sets/cifar/CIFAR'
    train_loader, test_loader = data_generator(root, args.batch_size)
else:
    output_size = None
    raise NotImplementedError

if args.cell == 'GRU':
    cell = GRUCell(input_size, args.hidden, output_size).cuda()
elif args.cell == 'WaveletGRU':
    init_wavelet = CustomWavelet(dec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                 dec_hi=[0, 0, -0.7071067811865476, 0.7071067811865476, 0, 0],
                                 rec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                 rec_hi=[0, 0, 0.7071067811865476, -0.7071067811865476, 0, 0],
                                 name='custom')
    cell = WaveletGRU(input_size, args.hidden, output_size,
                      mode=args.compression_mode,  p_drop=args.wave_dropout).cuda()
elif args.cell == 'FastFoodGRU':
    cell = FastFoodGRU(input_size, args.hidden, output_size,  p_drop=args.wave_dropout).cuda()
else:
    raise NotImplementedError()

pt = compute_parameter_total(cell)
print('parameter total', pt)
# optimizer = torch.optim.RMSprop(cell.parameters(), args.lr)
optimizer = torch.optim.Adadelta(cell.parameters(), args.lr)
loss_fun = torch.nn.CrossEntropyLoss()


def train_test(x, y, iteration_no, e_no, train=False):
    """
    Run the network.
        x: image tensors, [batch_size, 1, 28, 28]
        y: ground truth, [batch_size]
        iteration_no: iteration count
        e_no: epoch count
        train: if true turn on gradient descent.

    Returns:
        cpu_loss: loss on the current batch
        sum_correct: The total of correctly identified digits.
    """
    # reshape into batch, time, channel
    x_shape = x.shape
    x_flat = torch.reshape(x, list(x_shape[:2]) + [-1])
    x_flat = x_flat.permute(0, 2, 1)

    if train:
        optimizer.zero_grad()
        cell.train()
    else:
        cell.eval()

    time_steps = x_flat.shape[1]
    # run the RNN
    hc = None
    for t in range(time_steps):
        # batch_major format [b,t,d]
        yc, hc = cell(x_flat[:, t, :].type(torch.float32), hc)

    # only the last output is interesting
    loss = loss_fun(yc, y)
    sum_correct = torch.sum(torch.max(yc, dim=-1)[1] == y).type(torch.float32).detach().cpu().numpy()
    acc = sum_correct/(args.batch_size*1.0)

    cpu_loss = loss.detach().cpu().numpy()
    # compute gradients
    if args.cell == 'WaveletGRU':
        loss_wave = cell.get_wavelet_loss()
        loss_full = loss + loss_wave
        loss_wave_cpu = loss_wave.detach().cpu().numpy()
    else:
        loss_wave_cpu = 0
        loss_full = loss

    if train:
        loss_full.backward()

        # clip gradients.
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(cell.parameters(), args.clip)

        # apply gradients
        optimizer.step()
    if iteration_no % 50 == 0:
        print('e', e_no, 'step', iteration_no,
              'loss', cpu_loss, 'acc', acc, 'wl',
              loss_wave_cpu, 'train', train)
    return cpu_loss, sum_correct


train_it = 0
train_loss_lst = []
train_acc_lst = []
for e_num in range(args.epochs):
    epoch_true_total = 0
    epoch_element_total = 0
    for train_x, train_y in train_loader:
        train_loss, sum_correct = train_test(train_x.cuda(), train_y.cuda(),
                                             train_it, e_num, train=True)
        epoch_true_total += sum_correct
        epoch_element_total += train_y.shape[0]
        train_it += 1
        train_loss_lst.append(train_loss)
        train_acc_lst.append(epoch_true_total/epoch_element_total)


print('training done... testing ...')
test_loss_lst = []
test_acc_lst = []
test_it = 0
test_true_total = 0
test_elements_total = 0
for test_x, test_y in test_loader:
    with torch.no_grad():
        test_loss, test_sum_correct = train_test(test_x.cuda(), test_y.cuda(),
                                                 test_it, -1, train=False)
        test_it += 1
        test_true_total += test_sum_correct
        test_elements_total += test_y.shape[0]
        test_loss_lst.append(test_loss)
print('test_true_total', test_true_total,
      'test_elements_total', test_elements_total)
test_acc = test_true_total/(test_elements_total*1.0)

# pickle the results
print(pd)
print('test loss mean', np.mean(test_loss_lst), 'test acc', test_acc, 'pt', pt)
# pd_str = pd_to_string(current_run_pd)
# store_lst = [train_loss_lst, train_acc_lst, test_loss_lst, test_acc_lst, pt]
# pickle.dump(store_lst, open('./runs/amp' + pd_str + '.pkl', 'wb'))


