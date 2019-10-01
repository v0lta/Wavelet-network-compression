import torch
import numpy as np
from wave.wavelet_linear import WaveletLayer
from fastfood.fastfood import FastFoodLayer
import pywt


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
        z = torch.sigmoid(self.Whz(h) + self.Wxz(x))
        r = torch.sigmoid(self.Whr(h) + self.Wxr(x))
        hc = torch.tanh(self.Whh(r*h) + self.Wxh(x))
        hn = (1 - z)*h + z*hc
        y = self.W_proj(hn)
        return y, hn


class WaveletGRU(GRUCell):
    """ A compressed cell using a wavelet basis in the gates."""

    def __init__(self, input_size, hidden_size, out_size, init_wavelet=pywt.Wavelet('db6')):
        super().__init__(input_size, hidden_size, out_size)
        self.init_wavelet = init_wavelet
        scales = 8
        self.Whh = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales)
        self.Whu = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales)
        self.Whr = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales)
        print('Creating a Wavelet GRU, do not forget to add the wavelet-loss.')

    def get_wavelet_loss(self):
        return self.Whh.get_wavelet_loss() + self.Whu.get_wavelet_loss() + self.Whr.get_wavelet_loss()


class FastFoodGRU(GRUCell):
    """ A compressed cell using a wavelet basis in the gates."""

    def __init__(self, input_size, hidden_size, out_size):
        super().__init__(input_size, hidden_size, out_size)
        self.Whh = FastFoodLayer(hidden_size)
        self.Whu = FastFoodLayer(hidden_size)
        self.Whr = FastFoodLayer(hidden_size)