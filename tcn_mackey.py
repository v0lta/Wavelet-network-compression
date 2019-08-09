import time
import numpy as np
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.utils import weight_norm
from torch.utils.tensorboard import SummaryWriter
from generators.mackey_glass import MackeyGenerator
import ipdb


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        '''
        Removes comp_size elements from the end of the last dimension.
        '''
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size,
                 stride, dilation, padding, dropout=0.2):
        '''
        Defines a temporal convolution block.
            n_inputs: The number of input channels.
            n_ouptuts: The number of convolution filters and output dimension.
            kernel_size: size of the convolution kernel.
            stride: convolution stride.
            dilation: The skip factor used to dilate the strides.
            padding: The number of zeros added to each side of each data chunk.
            dropout: Probability of an element to be zeroed.
        '''
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) \
            if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        '''
        Create a dilated multi-layer single stride temporal-CNN
        '''
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


generator = MackeyGenerator(batch_size=10,
                            tmax=1200,
                            delta_t=1.0)
bpd = {}
bpd['iterations'] = 25000
bpd['pred_samples'] = 200
bpd['window_size'] = 1
bpd['lr'] = 0.001

pd_lst = [bpd]
for window_size in [1, 10, 25, 50, 75, 100]:
    for lr in [0.01, 0.001, 0.0001, 0.00001]:
        new_pd = copy.deepcopy(bpd)
        new_pd['lr'] = lr
        new_pd['window_size'] = window_size
        pd_lst.append(new_pd)

for pd in pd_lst:
    tcn = TemporalConvNet(num_inputs=1, num_channels=[125, 100, pd['window_size']]).cuda()
    model_parameters = filter(lambda p: p.requires_grad, tcn.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('model parameters', params)
    pd['params'] = params

    pd_str = ''
    for key in pd.keys():
        pd_str += '_' + key + '_' + str(pd[key])
    summary = SummaryWriter(comment=pd_str)
    # summary.add_graph(tcn)

    optimizer = torch.optim.Adam(tcn.parameters(), lr=pd['lr'])
    critereon = torch.nn.MSELoss()
    loss_lst = []

    for i in range(pd['iterations']):
        steps = int(pd['pred_samples']//pd['window_size'])
        tcn.train()
        mackey_data = torch.squeeze(generator())
        total_time = mackey_data.shape[-1]
        x, y = torch.split(mackey_data, [total_time - pd['pred_samples'],
                                         pd['pred_samples']], dim=-1)
        optimizer.zero_grad()

        x_in = x
        net_out = []
        for j in range(steps):
            y_pred = tcn(x_in.unsqueeze(1)).squeeze(0)
            y_pred = y_pred[:, :, -1]
            net_out.append(y_pred)
            x_in = torch.cat([x_in, y_pred], -1)

        prediction = torch.cat(net_out, -1)
        # loss = -torch.trace(torch.matmul(y, torch.log(prediction).float().t()) +
        #                     torch.matmul((1 - y), torch.log(1 - prediction).float().t()))
        loss = critereon(y, prediction)
        loss.backward()
        optimizer.step()

        rec_loss = loss.detach().cpu().numpy()
        loss_lst.append(rec_loss)
        print('iteration', i, 'loss', loss_lst[-1])
        summary.add_scalar('mse', loss_lst[-1], global_step=i)

        if i % 100 == 0:
            fig = plt.figure()
            plt.plot(prediction.detach().cpu().numpy()[0, :])
            plt.plot(y.detach().cpu().numpy()[0, :])
            summary.add_figure('predictions', fig, global_step=i)
            plt.clf()
            plt.close()

    fig = plt.figure()
    plt.plot(prediction.detach().cpu().numpy()[0, :])
    plt.plot(y.detach().cpu().numpy()[0, :])
    summary.add_figure('predictions', fig, global_step=i)
    plt.clf()
    plt.close()
