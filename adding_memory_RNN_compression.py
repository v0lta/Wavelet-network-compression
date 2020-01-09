'''
Following https://github.com/v0lta/Complex-gated-recurrent-neural-networks/blob/master/synthetic_experiments.py
'''

import time
import datetime
import argparse
import torch
import numpy as np
from RNN_compression.cells import GRUCell, FastFoodGRU, WaveletGRU
from RNN_compression.adding_memory import generate_data_adding, generate_data_memory
from util import pd_to_string, compute_parameter_total
import pickle
import collections
CustomWavelet = collections.namedtuple('Wavelet', ['dec_lo', 'dec_hi',
                                                   'rec_lo', 'rec_hi', 'name'])


def train_test_loop(args, in_x, in_y_gt, iteration_no, cell, loss_fun,
                    train=False, optimizer=None,
                    baseline=None):
    """
    Run the network on the adding or copy memory problems.
    train: if true turns backpropagation on.
    """
    if train:
        optimizer.zero_grad()
        cell.train()
    else:
        cell.eval()

    time_steps = in_x.shape[1]
    # run the RNN
    y_cell_lst = []
    h = None
    for t in range(time_steps):
        # batch_major format [b,t,d]
        y, h = cell(in_x[:, t, :].type(torch.float32), h)
        y_cell_lst.append(y)

    if args.problem == 'memory':
        el = np.prod(in_y_gt[:, -10:].shape).astype(np.float32)
        y_tensor = torch.stack(y_cell_lst, dim=-1)
        loss = loss_fun(y_tensor, in_y_gt)
        mem_res = torch.max(y_tensor[:, :, -10:], dim=1)[1]
        acc_sum = torch.sum(mem_res == in_y_gt[:, -10:]).type(torch.float32).detach().cpu().numpy()
        acc = acc_sum/(el*1.0)
    else:
        # only the last output is interesting
        el = in_y_gt.shape[0]
        train_y_gt = in_y_gt.type(torch.float32)
        loss = loss_fun(y, train_y_gt)
        acc_sum = torch.sum(torch.abs(y - train_y_gt) < 0.05).type(torch.float32).detach().cpu().numpy()
        acc = acc_sum/(el*1.0)

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
        # apply gradients
        optimizer.step()
    if iteration_no % 50 == 0:
        print('step', iteration_no, 'loss', cpu_loss, 'baseline:', baseline, 'acc', acc, 'wl',
              loss_wave_cpu, 'train', train)
    return cpu_loss, acc_sum


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequence Modeling - Adding and Memory Problems')
    parser.add_argument('--problem', type=str, default='adding',
                        help='choose adding or memory')
    parser.add_argument('--cell', type=str, default='WaveletGRU',
                        help='Cell type: Choose GRU or WaveletGRU or FastFoodGRU.')
    parser.add_argument('--hidden', type=int, default=512,
                        help='Cell size: Default 512.')
    parser.add_argument('--time_steps', type=int, default=150,
                        help='The number of time steps in the problem, default 150.')
    parser.add_argument('--compression_mode', type=str, default='full',
                        help='How to compress the cell.')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='The size of the training batches. default 50')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='The size of the training batches.')
    parser.add_argument('--n_train', type=int, default=int(6e5),
                        help='The size of the training batches. Default 6e5')
    parser.add_argument('--n_test', type=int, default=int(1e4),
                        help='The size of the training batches. Default 1e4')
    parser.add_argument('--wave_dropout', type=float, default=0.0,
                        help='Compression layer dropout probability')
    args = parser.parse_args()

    train_iterations = int(args.n_train/args.batch_size)
    test_iterations = int(args.n_test/args.batch_size)
    time_start = time.time()

    print(args)
    pd = vars(args)

    if args.problem == 'memory':
        # following https://github.com/amarshah/complex_RNN/blob/master/memory_problem.py
        input_size = 10
        output_size = 10
        x_train, y_train = generate_data_memory(args.time_steps, args.n_train)
        x_test, y_test = generate_data_memory(args.time_steps, args.n_test)
        # --- baseline ----------------------
        baseline = np.log(8) * 10/(args.time_steps + 20)
        print("Baseline is " + str(baseline))
        loss_fun = torch.nn.CrossEntropyLoss()
    elif args.problem == 'adding':
        input_size = 2
        output_size = 1
        x_train, y_train = generate_data_adding(args.time_steps, args.n_train)
        x_test, y_test = generate_data_adding(args.time_steps, args.n_test)
        baseline = 0.167
        loss_fun = torch.nn.MSELoss()
    else:
        raise NotImplementedError()

    if args.cell == 'GRU':
        cell = GRUCell(input_size, args.hidden, output_size).cuda()
    elif args.cell == 'WaveletGRU':
        init_wavelet = CustomWavelet(dec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                     dec_hi=[0, 0, -0.7071067811865476, 0.7071067811865476, 0, 0],
                                     rec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                     rec_hi=[0, 0, 0.7071067811865476, -0.7071067811865476, 0, 0],
                                     name='custom')
        cell = WaveletGRU(input_size, args.hidden, output_size, mode=args.compression_mode,
                          init_wavelet=init_wavelet, p_drop=args.wave_dropout).cuda()
    elif args.cell == 'FastFoodGRU':
        cell = FastFoodGRU(input_size, args.hidden, output_size, p_drop=args.dropout_prob).cuda()
    else:
        raise NotImplementedError()

    pt = compute_parameter_total(cell)
    print('parameter total', pt)
    optimizer = torch.optim.RMSprop(cell.parameters(), args.lr)

    x_train_lst = torch.split(x_train.cuda(), args.batch_size, dim=0)
    y_train_lst = torch.split(y_train.cuda(), args.batch_size, dim=0)
    x_test_lst = torch.split(x_test.cuda(), args.batch_size, dim=0)
    y_test_lst = torch.split(y_test.cuda(), args.batch_size, dim=0)

    train_loss_lst = []
    for train_iteration_no in range(train_iterations):
        x_train_batch = x_train_lst[train_iteration_no]
        y_train_batch = y_train_lst[train_iteration_no]
        if args.problem == 'memory':
            # --- one hot encoding -------------
            x_train_batch = torch.nn.functional.one_hot(x_train_batch.type(torch.int64))
            y_train_batch = y_train_batch.type(torch.int64)
        train_loss, _ = train_test_loop(args, x_train_batch, y_train_batch, train_iteration_no, cell, loss_fun, train=True,
                                        optimizer=optimizer, baseline=baseline)
        train_loss_lst.append(train_loss)

    print('training done... testing ...')
    test_loss_lst = []
    test_acc_sum = 0
    test_el_total = 0
    for test_iteration_no in range(test_iterations):
        with torch.no_grad():
            x_test_batch = x_test_lst[test_iteration_no]
            y_test_batch = y_test_lst[test_iteration_no]
            if args.problem == 'memory':
                # --- one hot encoding -------------
                x_test_batch = torch.nn.functional.one_hot(x_test_batch.type(torch.int64))
                y_test_batch = y_test_batch.type(torch.int64)
            test_loss, test_true_sum = train_test_loop(args, x_test_batch, y_test_batch, test_iteration_no, cell,
                                                       loss_fun, baseline=baseline)
            test_acc_sum += test_true_sum
            if args.problem == 'memory':
                test_el_total += np.prod(y_test_batch[:, -10:].shape).astype(np.float32)
            else:
                test_el_total += y_test_batch.shape[0]
            test_loss_lst.append(test_loss)
    # assert test_el_total == args.n_test
    print('test_el_total', test_el_total, 'test_acc_sum', test_acc_sum)
    test_acc = test_acc_sum/(test_el_total*1.0)

    print('test loss mean', np.mean(test_loss_lst), 'test acc', test_acc, 'pt', pt)
    store_lst = [train_loss_lst, test_loss_lst, test_acc, pt]
    pd_str = pd_to_string(pd)
    time_str = str(datetime.datetime.today())
    print('time:', time_str, 'experiment took', time.time() - time_start, '[s]')
    # pickle.dump(store_lst, open('./runs/' + time_str + pd_str + '.pkl', 'wb'))
