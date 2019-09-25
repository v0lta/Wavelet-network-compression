import time
import torch
import pywt
import numpy as np
import matplotlib.pyplot as plt
from generators.mackey_glass import MackeyGenerator
from wave.wavelet_linear import WaveletLayer
from fastfood.fastfood import FastFoodLayer
from torch.utils.tensorboard.writer import SummaryWriter

bpd = {}
bpd['iterations'] = 30000
bpd['tmax'] = 100
bpd['delta_t'] = 1.0
bpd['pred_samples'] = 50
bpd['window_size'] = 1
bpd['hidden_size'] = 1024
bpd['lr'] = 0.001
bpd['batch_size'] = 32
bpd['wavelet'] = True
pd_list = [bpd]


generator = MackeyGenerator(batch_size=bpd['batch_size'],
                            tmax=bpd['tmax'],
                            delta_t=bpd['delta_t'])


# Define Baseline and compressed GRU-Cells.
class GRUCell(torch.nn.Module):
    """ A LSTM-Cell reference implementation as outlined in https://arxiv.org/abs/1503.04069 """

    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size

        # update gate weights.
        self.Whz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wxz = torch.nn.Linear(input_size, hidden_size, bias=True)

        # reset gate weights.
        self.Whr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wxr = torch.nn.Linear(input_size, hidden_size, bias=True)

        # state candidate mapping
        self.Whh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wxh = torch.nn.Linear(input_size, hidden_size, bias=True)

        self.W_proj = torch.nn.Linear(hidden_size, out_size, bias=True)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(x.size(0), self._hidden_size, dtype=x.dtype, device=x.device)
        z = torch.nn.functional.sigmoid(self.Whz(h) + self.Wxz(x))
        r = torch.nn.functional.sigmoid(self.Whr(h) + self.Wxr(x))
        hc = torch.nn.functional.tanh(self.Whh(r*h) + self.Wxh(x))
        hn = (1 - z)*hc + z*hc
        y = self.W_proj(hn)
        return y, hn


class WaveletGRU(GRUCell):
    """ A compressed cell using a wavelet basis in the gates."""

    def __init__(self, input_size, hidden_size, out_size):
        super().__init__(input_size, hidden_size, out_size)
        init_wavelet = pywt.Wavelet('db6')
        scales = 8
        self.Whh = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales)
        self.Whu = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales)
        self.Whr = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales)


class FastFoodGRU(GRUCell):
    """ A compressed cell using a wavelet basis in the gates."""

    def __init__(self, input_size, hidden_size, out_size):
        super().__init__(input_size, hidden_size, out_size)

        self.Whh = FastFoodLayer(hidden_size)
        self.Whu = FastFoodLayer(hidden_size)
        self.Whr = FastFoodLayer(hidden_size)


for pd in pd_list:
    if pd['wavelet']:
        cell = WaveletGRU(input_size=pd['window_size'], hidden_size=pd['hidden_size'],
                          out_size=pd['window_size']).cuda()
    else:
        cell = GRUCell(input_size=pd['window_size'], hidden_size=pd['hidden_size'],
                       out_size=pd['window_size']).cuda()
        # torch.nn.GRUCell(input_size=pd['window_size'], hidden_size, bias=True)

    model_parameters = filter(lambda p: p.requires_grad, cell.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('model parameters', params)
    pd['params'] = params

    pd_str = ''
    for key in pd.keys():
        pd_str += '_' + key + '_' + str(pd[key])
    print('experiemnt params:', pd_str)
    summary = SummaryWriter(comment=pd_str)
    # summary.add_graph(tcn)

    optimizer = torch.optim.Adam(cell.parameters(), lr=pd['lr'])
    critereon = torch.nn.MSELoss()
    loss_lst = []

    for i in range(pd['iterations']):
        steps = int(pd['pred_samples']//pd['window_size'])
        cell.train()
        start = time.time()
        mackey_data = torch.squeeze(generator())
        total_time = mackey_data.shape[-1]
        x, y = torch.split(mackey_data, [total_time - pd['pred_samples'],
                                         pd['pred_samples']], dim=-1)
        optimizer.zero_grad()

        x_in = x
        pred_encode_lst = []
        for j in range(x.shape[-1]):
            yti, hi = cell(x[:, j].unsqueeze(-1))
            pred_encode_lst.append(yti)

        pred_lst = []
        yt = x[:, -1].unsqueeze(-1)
        for k in range(y.shape[-1]):
            yt, hi = cell(yt, hi)
            pred_lst.append(yt)

        prediction = torch.cat(pred_lst, -1)
        # loss = -torch.trace(torch.matmul(y, torch.log(prediction).float().t()) +
        #                     torch.matmul((1 - y), torch.log(1 - prediction).float().t()))
        loss = critereon(y, prediction)
        loss.backward()
        torch.nn.utils.clip_grad_value_(cell.parameters(), 1.)
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
