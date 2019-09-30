import time
import numpy as np
import copy
import torch
import scipy.signal as scisig

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from generators.mackey_glass import MackeyGenerator
from temporal_convolutions.kernel_dilation import TemporalConvNet
from fourier.short_time_fourier_pytorch import stft, istft

bpd = {}
bpd['iterations'] = 8000
bpd['tmax'] = 1024
bpd['delta_t'] = 1.0
bpd['pred_samples'] = 512
bpd['window_size'] = 64
bpd['lr'] = 0.004
bpd['batch_size'] = 32
bpd['dropout'] = 0.0
bpd['channels'] = [30, 30, 30, 30, 30, 30]
bpd['overlap'] = int(bpd['window_size']*0.5)
bpd['step_size'] = bpd['window_size'] - bpd['overlap']
bpd['fft_freq_no'] = int(bpd['window_size']//2 + 1)*2  # *2 for C
bpd['window_fun'] = 'hamming'

generator = MackeyGenerator(batch_size=bpd['batch_size'],
                            tmax=bpd['tmax'],
                            delta_t=bpd['delta_t'])

pd_lst = [bpd]
for lr in [0.004]:
    for window_size in [10, 25, 50, 100]:
        new_pd = copy.deepcopy(bpd)
        new_pd['lr'] = lr
        new_pd['window_size'] = window_size
        new_pd['overlap'] = int(new_pd['window_size']*0.5)
        new_pd['step_size'] = new_pd['window_size'] - new_pd['overlap']
        new_pd['fft_freq_no'] = int(new_pd['window_size']//2 + 1)*2  # *2 for C
        pd_lst.append(new_pd)

for pd in pd_lst:

    window = torch.from_numpy(scisig.get_window(pd['window_fun'], Nx=pd['window_size']))
    window = window.type(torch.float32).cuda()
    tcn = TemporalConvNet(num_inputs=pd['fft_freq_no'],
                          num_channels=pd['channels'] + [pd['fft_freq_no']],
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
        tcn.train()
        start = time.time()
        mackey_data = torch.squeeze(generator())
        total_time = mackey_data.shape[-1]
        x, y = torch.split(mackey_data, [total_time - pd['pred_samples'],
                                         pd['pred_samples']], dim=-1)

        # run the fft
        freq_x = stft(x, window, nperseg=pd['window_size'], noverlap=pd['overlap'],
                      boundary='zeros', padded=False)
        freq_y = stft(y, window, nperseg=pd['window_size'],
                      noverlap=pd['overlap'], boundary='zeros',
                      padded=False)
        steps = freq_y.shape[-2]

        # cat into [batch, freq*2, time]
        freq_x_cat = torch.cat([freq_x[..., 0], freq_x[..., 1]], -2)
        freq_y_cat = torch.cat([freq_y[..., 0], freq_y[..., 1]], -2)

        optimizer.zero_grad()

        x_in = freq_x_cat
        net_out = []
        for j in range(steps):
            y_pred = tcn(x_in)
            y_pred = torch.unsqueeze(y_pred[:, :, -1], -1)
            net_out.append(y_pred)
            x_in = torch.cat([x_in, y_pred], -1)

        freq_prediction = torch.cat(net_out, -1)
        freq_prediction = torch.stack([freq_prediction[:, :(pd['fft_freq_no']//2), :],
                                       freq_prediction[:, (pd['fft_freq_no']//2):, :]],
                                      -1)
        prediction = istft(freq_prediction, window, nperseg=pd['window_size'],
                           noverlap=pd['overlap'], boundary=True)
        loss = critereon(y, prediction)
        loss.backward()
        optimizer.step()

        rec_loss = loss.detach().cpu().numpy()
        loss_lst.append(rec_loss)
        runtime = time.time() - start
        print('iteration', i, 'loss', loss_lst[-1], 'runtime', runtime)
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
