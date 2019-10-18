import time
import torch
import numpy as np
import pywt
import matplotlib.pyplot as plt
from fourier.frequency_dilation_networks import FDN
from temporal_convolutions.kernel_dilation import TemporalConvNet
from data_loading.sequential_mnist import data_generator
from util import pd_to_string, compute_parameter_total
import pickle


pd = {}
pd['problem'] = 'MNIST'
pd['architecture'] = 'FDN'  # 'TCN', 'FDN'
pd['num_channels'] = [25, 25, 25, 25, 25, 25, 25, 25, 25]
pd['batch_size'] = 56
pd['epochs'] = 20
pd['lr'] = 1e-4

pd_lst = [pd]
print(pd)

for current_run_pd in pd_lst:
    output_size = None
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

    if current_run_pd['architecture'] == 'FDN':
        net = FDN(in_channels=input_size,
                  output_channels=output_size,
                  num_channels=current_run_pd['num_channels'],
                  initial_kernel_size=8).cuda()
    elif current_run_pd['architecture'] == 'TCN':
        net = TemporalConvNet(num_inputs=input_size,
                              num_channels=current_run_pd['num_channels'],
                              kernel_size=7, dropout=0.2,
                              output_size=output_size).cuda()
    else:
        raise NotImplementedError()

    pt = compute_parameter_total(net)
    print('parameter total', pt)
    optimizer = torch.optim.Adam(net.parameters(), current_run_pd['lr'])
    loss_fun = torch.nn.CrossEntropyLoss()

    def train_test_loop(x, y, iteration_no, e_no, train):
        # reshape into batch, time, channel
        x_shape = x.shape
        x_flat = torch.reshape(x, list(x_shape[:2]) + [-1])
        # x_flat = x_flat.permute(0, 2, 1)

        if train:
            net.train()
            optimizer.zero_grad()
        else:
            net.eval()

        time_start = time.time()
        yc = net(x_flat)
        time_stop = time.time() - time_start
        loss = loss_fun(yc, y)
        acc = torch.sum(torch.max(yc, dim=-1)[1] == y).type(torch.float32).detach().cpu().numpy()
        acc = acc/current_run_pd['batch_size']

        cpu_loss = loss.detach().cpu().numpy()
        # compute gradients
        if train:
            loss.backward()

            total_norm = 0
            for p in net.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            # apply gradients
            optimizer.step()
        else:
            total_norm = 0
        if iteration_no % 50 == 0:
            print('e', e_no, 'step', iteration_no, 'loss %1.5f' % cpu_loss, 'grad %5.5f' % total_norm,
                  'acc %1.5f' % acc, 'train', train, 'time %1.5f' % time_stop)
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
print('test loss mean', np.mean(test_loss_lst), 'test acc mean', np.mean(test_acc_lst), 'pt', pt)
