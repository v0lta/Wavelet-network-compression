import time
import numpy as np
import copy
import torch

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from data_loading.mackey_glass import MackeyGenerator
from temporal_convolutions.kernel_dilation import TemporalConvNet
import ipdb

bpd = {}
bpd['iterations'] = 8000
bpd['tmax'] = 512
bpd['delta_t'] = 1.0
bpd['pred_samples'] = 256
bpd['window_size'] = 1
bpd['lr'] = 0.001
bpd['batch_size'] = 32
bpd['dropout'] = 0.0
bpd['channels'] = [30, 30, 30, 30, 30, 30]

generator = MackeyGenerator(batch_size=bpd['batch_size'],
                            tmax=bpd['tmax'],
                            delta_t=bpd['delta_t'])


pd_lst = [bpd]
for lr in [0.004]:
    for window_size in [10, 25, 50, 100]:
        new_pd = copy.deepcopy(bpd)
        new_pd['lr'] = lr
        new_pd['window_size'] = window_size
        pd_lst.append(new_pd)

for pd in pd_lst:
    tcn = TemporalConvNet(num_inputs=1,
                          num_channels=bpd['channels'] + [pd['window_size']],
                          dropout=bpd['dropout']).cuda()
    model_parameters = filter(lambda p: p.requires_grad, tcn.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('model parameters', params)
    pd['params'] = params

    pd_str = ''
    for key in pd.keys():
        pd_str += '_' + key + '_' + str(pd[key])
    print('experiemnt params:', pd_str)
    summary = SummaryWriter(comment=pd_str)
    # summary.add_graph(tcn)

    optimizer = torch.optim.Adam(tcn.parameters(), lr=pd['lr'])
    critereon = torch.nn.MSELoss()
    loss_lst = []

    for i in range(pd['iterations']):
        steps = int(pd['pred_samples']//pd['window_size'])
        tcn.train()
        start = time.time()
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
            x_in = torch.cat([x_in[:, bpd['window_size']:], y_pred], -1)

        prediction = torch.cat(net_out, -1)
        # loss = -torch.trace(torch.matmul(y, torch.log(prediction).float().t()) +
        #                     torch.matmul((1 - y), torch.log(1 - prediction).float().t()))
        loss = critereon(y, prediction)
        loss.backward()

        total_norm = 0
        for p in tcn.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        optimizer.step()

        rec_loss = loss.detach().cpu().numpy()
        loss_lst.append(rec_loss)
        runtime = time.time() - start
        print('iteration', i, 'loss', loss_lst[-1], 'runtime', runtime)
        summary.add_scalar('mse', loss_lst[-1], global_step=i)
        summary.add_scalar('gradient-norm', total_norm, global_step=i)

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
