import torch
from src.wavelet_learning.wavelet_linear import WaveletLayer
from src.fastfood.fastfood import FastFoodLayer
import pywt


# Define Baseline and compressed GRU-Cells.
class GRUCell(torch.nn.Module):
    """ A GRU-Cell reference implementation as outlined in https://arxiv.org/abs/1503.04069 """

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

    def __init__(self, input_size, hidden_size, out_size, init_wavelet=pywt.Wavelet('db6'), mode='full',
                 p_drop=0.5):
        super().__init__(input_size, hidden_size, out_size)
        self.init_wavelet = init_wavelet
        self.mode = mode
        self.drop_prob = p_drop
        scales = 4

        if mode == 'gates':
            print('gates compression')
            self.Whz = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales, p_drop=self.drop_prob)
            self.Whr = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales, p_drop=self.drop_prob)
        elif mode == 'reset':
            print('reset compression')
            self.Whr = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales, p_drop=self.drop_prob)
        elif mode == 'update':
            print('update compression')
            self.Whz = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales, p_drop=self.drop_prob)
        elif mode == 'state':
            print('state compression')
            self.Whh = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales, p_drop=self.drop_prob)
        elif mode == 'state_reset':
            print('state+reset gate compression')
            self.Whh = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales, p_drop=self.drop_prob)
            self.Whr = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales, p_drop=self.drop_prob)
        elif mode == 'state_update':
            print('state+update gate compression')
            self.Whh = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales, p_drop=self.drop_prob)
            self.Whz = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales, p_drop=self.drop_prob)
        else:
            print('full compression')
            self.Whz = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales, p_drop=self.drop_prob)
            self.Whr = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales, p_drop=self.drop_prob)
            self.Whh = WaveletLayer(hidden_size, init_wavelet=init_wavelet, scales=scales, p_drop=self.drop_prob)
        print('Creating a Wavelet GRU, do not forget to add the wavelet-loss.')

    def get_wavelet_loss(self):
        if self.mode == 'gates':
            return self.Whz.get_wavelet_loss() + self.Whr.get_wavelet_loss()
        elif self.mode == 'state':
            return self.Whh.get_wavelet_loss()
        elif self.mode == 'reset':
            return self.Whr.get_wavelet_loss()
        elif self.mode == 'update':
            return self.Whz.get_wavelet_loss()
        elif self.mode == 'state_reset':
            return self.Whh.get_wavelet_loss() + self.Whr.get_wavelet_loss()
        elif self.mode == 'state_update':
            return self.Whh.get_wavelet_loss() + self.Whz.get_wavelet_loss()
        else:
            return self.Whh.get_wavelet_loss() + self.Whz.get_wavelet_loss() + self.Whr.get_wavelet_loss()


class FastFoodGRU(GRUCell):
    """ A compressed cell using a wavelet basis in the gates."""

    def __init__(self, input_size, hidden_size, out_size, p_drop=0.5):
        super().__init__(input_size, hidden_size, out_size)
        self.drop_prob = p_drop
        self.Whz = FastFoodLayer(hidden_size, p_drop=self.drop_prob)
        self.Whr = FastFoodLayer(hidden_size, p_drop=self.drop_prob)
        self.Whh = FastFoodLayer(hidden_size, p_drop=self.drop_prob)
