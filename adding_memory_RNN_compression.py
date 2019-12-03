'''
Following https://github.com/v0lta/Complex-gated-recurrent-neural-networks/blob/master/synthetic_experiments.py
'''

import torch
import numpy as np
import pywt
import matplotlib.pyplot as plt
from RNN_compression.cells import GRUCell, FastFoodGRU, WaveletGRU
from RNN_compression.adding_memory import generate_data_adding, generate_data_memory
from util import pd_to_string, compute_parameter_total
import pickle
import collections
CustomWavelet = collections.namedtuple('Wavelet', ['dec_lo', 'dec_hi',
                                                   'rec_lo', 'rec_hi', 'name'])

pd = {}
pd['problem'] = 'adding'
pd['cell'] = 'GRU'  # 'GRU'  'WaveletGRU' 'FastFoodGRU'
pd['hidden'] = 512
pd['time_steps'] = 150
pd['compression_mode'] = 'full'  # gates, state, state_reset, state_update, full
pd['batch_size'] = 50
pd['n_train'] = int(9e5)  # int(9e5)
pd['n_test'] = int(1e4)
pd['lr'] = 1e-3
if pd['cell'] == 'WaveletGRU':
    # pd['init_wavelet'] = pywt.Wavelet('db6')
    pd['init_wavelet'] = CustomWavelet(dec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                       dec_hi=[0, 0, -0.7071067811865476, 0.7071067811865476, 0, 0],
                                       rec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                       rec_hi=[0, 0, 0.7071067811865476, -0.7071067811865476, 0, 0],
                                       name='custom')

else:
    pd['init_wavelet'] = None

train_iterations = int(pd['n_train']/pd['batch_size'])
test_iterations = int(pd['n_test']/pd['batch_size'])
pd_lst = [pd]

print(pd)

for current_run_pd in pd_lst:

    if current_run_pd['problem'] == 'memory':
        # following https://github.com/amarshah/complex_RNN/blob/master/memory_problem.py
        input_size = 10
        output_size = 10
        x_train, y_train = generate_data_memory(current_run_pd['time_steps'], current_run_pd['n_train'])
        x_test, y_test = generate_data_memory(current_run_pd['time_steps'], current_run_pd['n_test'])
        # --- baseline ----------------------
        baseline = np.log(8) * 10/(current_run_pd['time_steps'] + 20)
        print("Baseline is " + str(baseline))
        loss_fun = torch.nn.CrossEntropyLoss()
    elif current_run_pd['problem'] == 'adding':
        input_size = 2
        output_size = 1
        x_train, y_train = generate_data_adding(current_run_pd['time_steps'], current_run_pd['n_train'])
        x_test, y_test = generate_data_adding(current_run_pd['time_steps'], current_run_pd['n_test'])
        baseline = 0.167
        loss_fun = torch.nn.MSELoss()
    else:
        raise NotImplementedError()

    if current_run_pd['cell'] == 'GRU':
        cell = GRUCell(input_size, current_run_pd['hidden'], output_size).cuda()
    elif current_run_pd['cell'] == 'WaveletGRU':
        cell = WaveletGRU(input_size, current_run_pd['hidden'], output_size, mode=pd['compression_mode']).cuda()
    elif current_run_pd['cell'] == 'FastFoodGRU':
        cell = FastFoodGRU(input_size, current_run_pd['hidden'], output_size).cuda()
    else:
        raise NotImplementedError()

    pt = compute_parameter_total(cell)
    print('parameter total', pt)
    optimizer = torch.optim.RMSprop(cell.parameters(), current_run_pd['lr'])

    x_train_lst = torch.split(x_train.cuda(), current_run_pd['batch_size'], dim=0)
    y_train_lst = torch.split(y_train.cuda(), current_run_pd['batch_size'], dim=0)
    x_test_lst = torch.split(x_test.cuda(), current_run_pd['batch_size'], dim=0)
    y_test_lst = torch.split(y_test.cuda(), current_run_pd['batch_size'], dim=0)

    def train_test_loop(train_iteration_no, train):
        if train:
            optimizer.zero_grad()
        x_train_batch = x_train_lst[train_iteration_no]
        y_train_batch = y_train_lst[train_iteration_no]

        if current_run_pd['problem'] == 'memory':
            # --- one hot encoding -------------
            x_train_batch = torch.nn.functional.one_hot(x_train_batch.type(torch.int64))
            y_train_batch = y_train_batch.type(torch.int64)

        time_steps = x_train_batch.shape[1]
        # run the RNN
        y_cell_lst = []
        h = None
        for t in range(time_steps):
            # batch_major format [b,t,d]
            y, h = cell(x_train_batch[:, t, :].type(torch.float32), h)
            y_cell_lst.append(y)

        el = np.prod(y_train_batch[:, -10:].shape).astype(np.float32)
        if current_run_pd['problem'] == 'memory':
            assert time_steps == current_run_pd['time_steps'] + 20
            y_tensor = torch.stack(y_cell_lst, dim=-1)
            loss = loss_fun(y_tensor, y_train_batch)
            mem_res = torch.max(y_tensor[:, :, -10:], dim=1)[1]
            acc = torch.sum(mem_res == y_train_batch[:, -10:]).type(torch.float32).detach().cpu().numpy()
            acc = acc/el
        else:
            assert time_steps == current_run_pd['time_steps']
            # only the last output is interesting
            y_train_batch = y_train_batch.type(torch.float32)
            loss = loss_fun(y, y_train_batch)
            acc = torch.sum(torch.abs(y - y_train_batch) < 0.05).type(torch.float32).detach().cpu().numpy()
            acc = acc/el

        cpu_loss = loss.detach().cpu().numpy()
        # compute gradients
        if pd['cell'] == 'WaveletGRU':
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
        if train_iteration_no % 50 == 0:
            print('step', train_iteration_no, 'loss', cpu_loss, 'baseline:', baseline, 'acc', acc, 'wl',
                  loss_wave_cpu, 'train', train)
        return cpu_loss, acc

    train_loss_lst = []
    train_acc_lst = []
    for train_iteration_no in range(train_iterations):
        train_loss, train_acc = train_test_loop(train_iteration_no, train=True)
        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)

    print('training done... testing ...')
    test_loss_lst = []
    test_acc_lst = []
    for test_iteration_no in range(test_iterations):
        with torch.no_grad():
            test_loss, test_acc = train_test_loop(test_iteration_no, train=False)
            test_loss_lst.append(test_loss)
            test_acc_lst.append(test_acc)


# pickle the results
print(pd)
print('test loss mean', np.mean(test_loss_lst), 'test acc mean', np.mean(test_acc), 'pt', pt)
pd_str = pd_to_string(current_run_pd)
store_lst = [train_loss_lst, train_acc_lst, test_loss_lst, test_acc_lst, pt]
pickle.dump(store_lst, open('./runs/amp' + pd_str + '.pkl', 'wb'))
