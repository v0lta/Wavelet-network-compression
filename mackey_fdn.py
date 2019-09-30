import time
import numpy as np
import copy
import torch
import pywt
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from generators.mackey_glass import MackeyGenerator
from temporal_convolutions.kernel_dilation import TemporalConvNet
from temporal_convolutions.wavelet_dilation import FrequencyDilationNetwork

bpd = {}
bpd['iterations'] = 8000
bpd['tmax'] = 200
bpd['delta_t'] = 1.0
bpd['pred_samples'] = 100
bpd['window_size'] = 1
bpd['lr'] = 0.004
bpd['batch_size'] = 32
bpd['dropout'] = 0.0


generator = MackeyGenerator(batch_size=bpd['batch_size'],
                            tmax=bpd['tmax'],
                            delta_t=bpd['delta_t'])

pd_lst = [bpd]


for pd in pd_lst:
    init_wavelet = pywt.Wavelet('db1')
    # init_wavelet, scales, threshold, in_dim, depth, out_dim
    fdn = FrequencyDilationNetwork(init_wavelet=init_wavelet, scales=8, threshold=0.25,
                                   in_dim=pd['pred_samples'], depth=800, out_dim=1).cuda()
    fdn.block1.init_weights()

    model_parameters = filter(lambda p: p.requires_grad, fdn.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('model parameters', params)
    pd['params'] = params

    pd_str = ''
    for key in pd.keys():
        pd_str += '_' + key + '_' + str(pd[key])
    print('experiemnt params:', pd_str)
    summary = SummaryWriter(comment='_fdn_' + pd_str)
    # summary.add_graph(fdn)

    optimizer = torch.optim.Adam(fdn.parameters(), lr=pd['lr'])
    critereon = torch.nn.MSELoss()
    loss_lst = []

    for i in range(pd['iterations']):
        steps = int(pd['pred_samples'])
        fdn.train()
        start = time.time()
        mackey_data = torch.squeeze(generator())
        total_time = mackey_data.shape[-1]
        x, y = torch.split(mackey_data, [total_time - pd['pred_samples'],
                                         pd['pred_samples']], dim=-1)
        optimizer.zero_grad()

        x_in = x
        net_out = []
        for j in range(steps):
            y_pred = fdn(x_in[:, -bpd['pred_samples']:])
            net_out.append(y_pred)
            x_in = torch.cat([x_in, y_pred], -1)

        prediction = torch.cat(net_out, -1)
        # loss = -torch.trace(torch.matmul(y, torch.log(prediction).float().t()) +
        #                     torch.matmul((1 - y), torch.log(1 - prediction).float().t()))
        wl = fdn.wavelet_loss()
        loss = critereon(y, prediction) + wl
        loss.backward()
        optimizer.step()

        rec_loss = loss.detach().cpu().numpy()
        loss_lst.append(rec_loss)
        runtime = time.time() - start
        print('iteration', i, 'loss', loss_lst[-1], 'wl', wl.detach().cpu().numpy(), 'runtime', runtime)
        summary.add_scalar('mse', loss_lst[-1], global_step=i)

        if i % 100 == 0:
            fig = plt.figure()
            plt.plot(prediction.detach().cpu().numpy()[0, :])
            plt.plot(y.detach().cpu().numpy()[0, :])
            summary.add_figure('predictions', fig, global_step=i)
            plt.clf()
            plt.close()

            fdn.block1.summary_to_tensorboard(summary, i)

    fig = plt.figure()
    plt.plot(prediction.detach().cpu().numpy()[0, :])
    plt.plot(y.detach().cpu().numpy()[0, :])
    summary.add_figure('predictions', fig, global_step=i)
    plt.clf()
    plt.close()
