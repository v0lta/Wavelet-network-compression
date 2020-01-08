# Created by moritz (wolter@cs.uni-bonn.de) at 18/12/2019

import pickle
import time
import datetime
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from RNN_compression.cells import GRUCell, FastFoodGRU, WaveletGRU
from RNN_compression.adding_memory import generate_data_adding, generate_data_memory
from util import pd_to_string, compute_parameter_total
from adding_memory_RNN_compression import train_test_loop
import collections
CustomWavelet = collections.namedtuple('Wavelet', ['dec_lo', 'dec_hi',
                                                   'rec_lo', 'rec_hi', 'name'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequence Modeling - Adding and Memory Problems')
    parser.add_argument('--problem', type=str, default='memory',
                        help='choose adding or memory')
    parser.add_argument('--cell', type=str, default='WaveletGRU',
                        help='Cell type: Choose GRU or WaveletGRU or FastFoodGRU.')
    parser.add_argument('--hidden_min', type=int, default=8,
                        help='Cell size: Default 512.')
    parser.add_argument('--hidden_max', type=int, default=136,
                        help='Cell size: Default 512.')
    parser.add_argument('--hidden_step', type=int, default=16,
                        help='State resolution on the grid.')
    parser.add_argument('--time_min', type=int, default=10,
                        help='The number of time steps in the problem, default 10.')
    parser.add_argument('--time_max', type=int, default=60,
                        help='The number of time steps in the problem, default 60.')
    parser.add_argument('--time_step', type=int, default=10,
                        help='Time step resolution on the grid.')
    parser.add_argument('--compression_mode', type=str, default='state',
                        help='How to compress the cell.')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='The size of the training batches.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='The size of the training batches.')
    parser.add_argument('--n_train', type=int, default=int(6e5),
                        help='The size of the training batches. Default 6e5')
    parser.add_argument('--n_test', type=int, default=int(1e4),
                        help='The size of the training batches. Default 1e4')
    args = parser.parse_args()

    train_iterations = int(args.n_train/args.batch_size)
    test_iterations = int(args.n_test/args.batch_size)

    print(args)
    pd = vars(args)

    hidden_array = np.arange(args.hidden_min, args.hidden_max+1, args.hidden_step)
    time_array = np.arange(args.time_min, args.time_max+1, args.time_step)
    result_list = []

    for state_size in hidden_array:
        for time_steps in time_array:
            time_start = time.time()
            print('time', datetime.datetime.today())
            print('experiment:', 'state', state_size, 'time', time_steps, args.cell, args.problem)
            if args.problem == 'memory':
                # following https://github.com/amarshah/complex_RNN/blob/master/memory_problem.py
                input_size = 10
                output_size = 10
                x_train, y_train = generate_data_memory(time_steps, args.n_train)
                x_test, y_test = generate_data_memory(time_steps, args.n_test)
                # --- baseline ----------------------
                baseline = np.log(8) * 10/(time_steps + 20)
                print("Baseline is " + str(baseline))
                loss_fun = torch.nn.CrossEntropyLoss()
            elif args.problem == 'adding':
                input_size = 2
                output_size = 1
                x_train, y_train = generate_data_adding(time_steps, args.n_train)
                x_test, y_test = generate_data_adding(time_steps, args.n_test)
                baseline = 0.167
                loss_fun = torch.nn.MSELoss()
            else:
                raise NotImplementedError()

            if args.cell == 'GRU':
                cell = GRUCell(input_size, state_size, output_size).cuda()
            elif args.cell == 'WaveletGRU':
                init_wavelet = CustomWavelet(dec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                             dec_hi=[0, 0, -0.7071067811865476, 0.7071067811865476, 0, 0],
                                             rec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                             rec_hi=[0, 0, 0.7071067811865476, -0.7071067811865476, 0, 0],
                                             name='custom')
                cell = WaveletGRU(input_size, state_size, output_size, mode=args.compression_mode,
                                  init_wavelet=init_wavelet).cuda()
            elif args.cell == 'FastFoodGRU':
                cell = FastFoodGRU(input_size, state_size, output_size).cuda()
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
                train_loss, _ = train_test_loop(args, x_train_batch, y_train_batch, train_iteration_no, train=True,
                                                optimizer=optimizer, cell=cell, loss_fun=loss_fun, baseline=baseline)
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
                    test_loss, test_true_sum = train_test_loop(args, x_test_batch, y_test_batch, test_iteration_no,
                                                               cell=cell, loss_fun=loss_fun, baseline=baseline)
                    test_acc_sum += test_true_sum
                    if args.problem == 'memory':
                        test_el_total += np.prod(y_test_batch[:, -10:].shape).astype(np.float32)
                    else:
                        test_el_total += y_test_batch.shape[0]
                    test_loss_lst.append(test_loss)
            # assert test_el_total == args.n_test
            print('test_el_total', test_el_total, 'test_acc_sum', test_acc_sum)
            test_acc = test_acc_sum/(test_el_total*1.0)

            print('test loss mean', np.mean(test_loss_lst), 'test acc mean', test_acc, 'pt', pt)
            store_lst = [state_size, time_steps, test_loss_lst, test_acc, pt, test_acc_sum, test_el_total, baseline]
            pd_str = pd_to_string(pd)
            time_str = str(datetime.datetime.today())
            print('time:', time_str, 'experiment took', time.time() - time_start, '[s]')
            result_list.append(store_lst)

    pickle.dump(store_lst, open('./runs/grid_' + pd_str + '.pkl', 'wb'))
    print('done')
    # do the plotting.
    test_acc_lst = []
    hidden_lst = []
    pt_lst = []
    time_lst = []
    for exp in result_list:
        test_acc_lst.append(exp[3])
        time_lst.append(exp[1])
        hidden_lst.append(exp[0])
        pt_lst.append(exp[4])
    pt_mat = np.array(pt_lst).reshape(hidden_array.shape[0], time_array.shape[0])
    time_mat = np.array(time_lst).reshape(hidden_array.shape[0], time_array.shape[0])
    hidden_mat = np.array(hidden_lst).reshape(hidden_array.shape[0], time_array.shape[0])
    test_acc_mat = np.array(test_acc_lst).reshape(hidden_array.shape[0], time_array.shape[0])

    # plt.imshow(test_acc_mat.transpose()); plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(time_mat, pt_mat, test_acc_mat, cmap='viridis')
    ax.set_title('WaveletGRU')
    ax.set_xlabel('time')
    ax.set_ylabel('parameters')
    ax.set_zlabel('accuracy')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('gru_surf_wave_pt.pdf')
    plt.show()
