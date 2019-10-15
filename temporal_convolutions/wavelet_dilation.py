import torch
from wave.learn_wave import Wave1D
import numpy as np
import pywt
import matplotlib.pyplot as plt


class WaveletDilationBlock(torch.nn.Module):
    def __init__(self, init_wavelet, scales, std_factor, in_dim, out_dim, name):
        '''
        Set up a frequency-time 1D temporal convnet.
        '''
        super().__init__()
        self.init_wavelet = init_wavelet
        self.scales = scales
        self.std_factor = std_factor
        self.wavelet = Wave1D(init_wavelet,  scales=scales)
        self.out_dim = out_dim
        # compute wavlet coefficient no.
        scales = self.wavelet.compute_coeff_no(in_dim)
        in_dim_wave = np.sum(scales)
        self.block_mat = torch.nn.Parameter(torch.Tensor(in_dim_wave, out_dim))
        self.activation = torch.nn.ReLU()
        self.name = name

        self.x_freq_mat = None
        self.mask = None
        self.threshold = None

    def init_weights(self):
        torch.nn.init.kaiming_uniform_(self.block_mat, a=0)

    def forward(self, x):
        x_freq = self.wavelet.analysis(x)
        x_freq_cat = torch.cat(x_freq, -1)
        # torch.sparse only does matrices, create one.
        c_no = x_freq_cat.shape[-1]
        merge_dim = np.prod(x_freq_cat.shape[:-1])
        x_freq_mat = torch.reshape(x_freq_cat, [merge_dim, c_no])
        self.threshold = torch.mean(torch.abs(x_freq_mat)) + torch.std(torch.abs(x_freq_mat))*self.std_factor
        mask = torch.abs(x_freq_mat) > self.threshold
        v_freq = torch.masked_select(x_freq_mat, mask)
        i_freq = mask.nonzero()

        x_freq_sparse = torch.sparse.FloatTensor(i_freq.t(), v_freq, x_freq_mat.shape)

        self.x_freq_mat = x_freq_mat
        self.mask = mask

        out = torch.sparse.mm(x_freq_sparse, self.block_mat)
        # restore the leading dimensions.
        out = self.activation(out)
        out = torch.reshape(out, list(x_freq_cat.shape[:-1]) + [self.out_dim])
        # print('x_freq', x_freq_sparse.shape)
        return out

    def wavelet_loss(self):
        acl, _, _ = self.wavelet.alias_cancellation_loss()
        prl, _, _ = self.wavelet.perfect_reconstruction_loss()
        return acl + prl

    def summary_to_tensorboard(self, tensorboard_writer, step):
        self.wavelet.add_wavelet_summary(self.name, tensorboard_writer, step)
        fig = plt.figure()
        plt.semilogy(self.x_freq_mat[0, :].detach().cpu().numpy())
        selected = self.x_freq_mat*self.mask.type(torch.float32)
        plt.semilogy(selected[0, :].detach().cpu().numpy(), '.')
        plt.legend(['full, selected'])
        tensorboard_writer.add_figure(self.name + '/wavelet/coefficients', fig, step, close=True)
        plt.close()


class WaveletDilationNetwork(torch.nn.Module):
    """ Create a WDN """
    def __init__(self, init_wavelet, scales, std_factor, in_dim, depth, out_dim):
        super().__init__()
        self.block1 = WaveletDilationBlock(init_wavelet, scales, std_factor, in_dim, depth,
                                             name='block_1')
        self.block2 = WaveletDilationBlock(init_wavelet, scales, std_factor, in_dim=depth, out_dim=depth,
                                             name='block_2')
        self.output_projection = torch.nn.Linear(depth, out_dim, bias=True)

    def forward(self, x):
        step1 = self.block1(x.unsqueeze(1).unsqueeze(1))
        step2 = self.block2(step1)
        return self.output_projection(step2).squeeze(1).squeeze(1)

    def wavelet_loss(self):
        return self.block1.wavelet_loss() + self.block2.wavelet_loss()
