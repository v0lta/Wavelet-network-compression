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


pd = {}
pd['problem'] = 'MNIST'
pd['cell'] = 'WaveletGRU'  # 'GRU'  'WaveletGRU' 'FastFoodGRU'
pd['hidden'] = 512
pd['batch_size'] = 50
pd['epochs'] = 10
pd['lr'] = 1e-3
if pd['cell'] == 'WaveletGRU':
    pd['init_wavelet'] = pywt.Wavelet('db6')
else:
    pd['init_wavelet'] = None

pd_lst = [pd]
print(pd)

for current_run_pd in pd_lst:

    if current_run_pd['problem'] == 'MNIST':
        root = './data_sets/mnist/'
        input_size = 1
        output_size = 10
        train_loader, test_loader = data_generator(root, pd['batch_size'])
    elif current_run_pd['problem'] == 'CIFAR':
        input_size = 3
        root = './data_sets/cifar/CIFAR'
        train_loader, test_loader = data_generator(root, pd['batch_size'])
    else:
        raise NotImplementedError

    if current_run_pd['cell'] == 'GRU':
        cell = GRUCell(input_size, current_run_pd['hidden'], output_size).cuda()
    elif current_run_pd['cell'] == 'WaveletGRU':
        cell = WaveletGRU(input_size, current_run_pd['hidden'], output_size).cuda()
    elif current_run_pd['cell'] == 'FastFoodGRU':
        cell = FastFoodGRU(input_size, current_run_pd['hidden'], output_size).cuda()
    else:
        raise NotImplementedError()

    pt = compute_parameter_total(cell)
    print('parameter total', pt)
    optimizer = torch.optim.RMSprop(cell.parameters(), current_run_pd['lr'])
    loss_fun = torch.nn.CrossEntropyLoss()

    def train_test_loop(x, y, iteration_no, e_no, train):
        # reshape into batch, time, channel
        x_shape = x.shape
        x_flat = torch.reshape(x, list(x_shape[:2]) + [-1])
        x_flat = x_flat.permute(0, 2, 1)

        if train:
            optimizer.zero_grad()

        time_steps = x_flat.shape[1]
        # run the RNN
        hc = None
        for t in range(time_steps):
            # batch_major format [b,t,d]
            yc, hc = cell(x_flat[:, t, :].type(torch.float32), hc)

        # only the last output is interesting
        loss = loss_fun(yc, y)
        acc = torch.sum(torch.max(yc, dim=-1)[1] == y).type(torch.float32).detach().cpu().numpy()
        acc = acc/current_run_pd['batch_size']

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
        if iteration_no % 50 == 0:
            print('e', e_no, 'step', iteration_no, 'loss', cpu_loss, 'acc', acc, 'wl',
                  loss_wave_cpu, 'train', train)
        return cpu_loss, acc

    train_loss_lst = []
    train_acc_lst = []
    train_it = 0

    for e_num in range(pd['epochs']):
        for train_x, train_y in train_loader:
            train_loss, train_acc = train_test_loop(train_x.cuda(), train_y.cuda(), train_it, e_num, train=True)
            train_it += 1
            train_loss_lst.append(train_loss)
            train_acc_lst.append(train_acc)

    print('training done... testing ...')
    test_loss_lst = []
    test_acc_lst = []
    test_it = 0
    for test_x, test_y in test_loader:
        with torch.no_grad():
            test_loss, test_acc = train_test_loop(test_x.cuda(), test_y.cuda(), test_it, -1, train=False)
            test_it += 1
            test_loss_lst.append(test_loss)
            test_acc_lst.append(test_acc)


# pickle the results
print(pd)
print('test loss mean', np.mean(test_loss_lst), 'test acc mean', np.mean(test_acc), 'pt', pt)
# pd_str = pd_to_string(current_run_pd)
# store_lst = [train_loss_lst, train_acc_lst, test_loss_lst, test_acc_lst, pt]
# pickle.dump(store_lst, open('./runs/amp' + pd_str + '.pkl', 'wb'))


