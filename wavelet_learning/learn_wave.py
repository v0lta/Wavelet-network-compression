import torch
import pywt
import numpy as np
from wavelet_learning.lowlevel import afb1d
from wavelet_learning.lowlevel import sfb1d
import matplotlib.pyplot as plt
'''A learnable filter wavelet transform in 1D.'''


class Wave1D(torch.nn.Module):
    def __init__(self, init_wavelet, scales=1,
                 mode='zero'):
        super().__init__()

        def to_tensor(filt_lst: list):
                tensor = torch.tensor(np.array(filt_lst).ravel(),
                                      dtype=torch.float)
                return torch.nn.Parameter(tensor)

        self.init_wavelet = init_wavelet
        self.dec_lo = to_tensor(init_wavelet.dec_lo)
        self.dec_hi = to_tensor(init_wavelet.dec_hi)
        self.rec_lo = to_tensor(init_wavelet.rec_lo)
        self.rec_hi = to_tensor(init_wavelet.rec_hi)

        # all filter shapes must be the same.
        assert self.dec_lo.shape == self.dec_hi.shape, 'filters must have the same sizes'
        assert self.rec_lo.shape == self.rec_hi.shape, 'filters must have the same sizes'
        assert self.dec_lo.shape == self.rec_lo.shape, 'filters must have the same sizes'

        self.scales = scales
        self.mode = mode

    def coeff_len(self, input_length):
            return pywt.dwt_coeff_len(input_length, self.dec_lo.shape[0], self.mode)

    @staticmethod
    def check_sym(in_filter) -> bool:
        # 1d in_filterer arrays.
        assert len(in_filter.shape) == 1
        length = in_filter.shape[0]
        sym = True
        for i in range(length//2):
            if in_filter[i] != in_filter[length-1]:
                sym = False
        return sym

    @staticmethod
    def check_anti_sym(in_filter) -> bool:
        assert len(in_filter.shape) == 1
        length = in_filter.shape[0]
        anti_sym = True
        for i in range(length//2):
            if in_filter[i] != -in_filter[length-1]:
                anti_sym = False
        return anti_sym

    def alias_cancellation_loss(self) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Strang+Nguyen 105: F0(z) = H1(-z); F1(z) = -H0(-z)
        Alternating sign convention from 0 to N see strang overview on the back of the cover.
        '''
        m1 = torch.tensor([-1], device=self.dec_hi.device, dtype=self.dec_hi.dtype)
        length = self.rec_lo.shape[0]
        mask = torch.tensor([torch.pow(m1, n) for n in range(length)][::-1],
                            device=self.dec_hi.device, dtype=self.dec_hi.dtype)
        err1 = self.rec_lo - mask*self.dec_hi
        err1s = torch.sum(err1*err1)

        length = self.rec_hi.shape[0]
        mask = torch.tensor([torch.pow(m1, n) for n in range(length)][::-1],
                            device=self.dec_lo.device, dtype=self.dec_lo.dtype)
        err2 = self.rec_hi - m1*mask*self.dec_lo
        err2s = torch.sum(err2*err2)
        return err1s + err2s, err1, err2

    def perfect_reconstruction_loss(self) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Strang 107:
        Assuming alias cancellation holds:
        P(z) = F(z)H(z)
        Product filter P(z) + P(-z) = 2.
        However since alias cancellation is implemented as soft constraint:
        P_0 + P_1 = 2

        Somehow numpy and torch implement convolution differently.
        For some reason the machine learning people call cross-correlation convolution.
        https://discuss.pytorch.org/t/numpy-convolve-and-conv1d-in-pytorch/12172
        Therefore for true convolution one element needs to be flipped.
        '''
        # polynomial multiplication is convolution, compute p(z):
        pad = self.dec_lo.shape[0]-1
        p_lo = torch.nn.functional.conv1d(
            self.dec_lo.unsqueeze(0).unsqueeze(0),
            torch.flip(self.rec_lo, [-1]).unsqueeze(0).unsqueeze(0),
            padding=pad)

        pad = self.dec_hi.shape[0]-1
        p_hi = torch.nn.functional.conv1d(
            self.dec_hi.unsqueeze(0).unsqueeze(0),
            torch.flip(self.rec_hi, [-1]).unsqueeze(0).unsqueeze(0),
            padding=pad)

        p_test = p_lo + p_hi
        two_at_power_zero = torch.zeros(p_test.shape, device=p_test.device,
                                        dtype=p_test.dtype)
        # numpy comparison for debugging.
        # np.convolve(self.init_wavelet.filter_bank[0], self.init_wavelet.filter_bank[2])
        # np.convolve(self.init_wavelet.filter_bank[1], self.init_wavelet.filter_bank[3])
        two_at_power_zero[..., p_test.shape[-1]//2] = 2
        return torch.sum((p_test - two_at_power_zero)*(p_test - two_at_power_zero)), p_test, two_at_power_zero

    def orthogonal_filter_cond(self):
        '''
        Check the biorthogonality conditions as described in Strang and Nguyen; Wavelets
        and filter banks; Page 111.
        '''
        # if even:
        if self.dec_lo.shape[0] % 2 == 0:
            pass

        # if odd
        if self.dec_lo.shape[0] % 2 != 0:
            # filters must symmetric and antisymmetric
            for no, dec_filt in enumerate(self.dec_lo):
                pass

    def compute_coeff_no(self, init_length):
        """
        Compute the number of resulting wavelet coefficients.
        @param init_length: length of the input signal vector.
        """
        lengths = [init_length]
        for J in range(self.scales):
            lengths.append(pywt.dwt_coeff_len(
                lengths[-1], self.dec_lo.shape[-1], self.mode))
        lengths.append(lengths[-1])
        return lengths[1:]

    def analysis(self, x, mode='zero'):
        yh = []
        lo = x
        # flip filters
        flip_dec_lo = torch.flip(self.dec_lo, [-1])
        flip_dec_hi = torch.flip(self.dec_hi, [-1])
        for s in range(self.scales):
            lohi = afb1d(lo, flip_dec_lo, flip_dec_hi, mode=self.mode, dim=-1)
            lo, hi = torch.split(lohi, split_size_or_sections=[1, 1], dim=1)
            yh.append(hi)
        yh.append(lo)
        return yh

    def reconstruction(self, X):
        X_inv = X[::-1]
        lo = X_inv[0]

        for hi in X_inv[1:]:
            # 'Unpad' added dimensions
            if lo.shape[-2] > hi.shape[-2]:
                lo = lo[..., :-1, :]
            if lo.shape[-1] > hi.shape[-1]:
                lo = lo[..., :-1]
            lo = sfb1d(lo, hi, self.rec_lo, self.rec_hi, mode=self.mode, dim=-1)
        return lo

    def add_wavelet_summary(self, name, tensorboard_writer, step):
        fig = plt.figure()
        plt.plot(self.dec_lo.detach().cpu().numpy())
        plt.plot(self.dec_hi.detach().cpu().numpy())
        plt.plot(self.rec_lo.detach().cpu().numpy())
        plt.plot(self.rec_hi.detach().cpu().numpy())
        plt.legend(['dec_lo,', 'dec_hi', 'rec_lo', 'rec_hi'])
        tensorboard_writer.add_figure(name + '/wavelet/filters', fig, step, close=True)
        plt.close()
        acl, err1, err2 = self.alias_cancellation_loss()
        prl, p_test, two_at_power_zero = self.perfect_reconstruction_loss()
        tensorboard_writer.add_scalar(name + '/wavelet/prl', prl.detach().cpu().numpy(), step)
        tensorboard_writer.add_scalar(name + '/wavelet/acl', acl.detach().cpu().numpy(), step)

        fig = plt.figure()
        plt.plot(err1.detach().cpu().numpy())
        plt.plot(err2.detach().cpu().numpy())
        plt.legend(['e1,', 'e2'])
        tensorboard_writer.add_figure(name + '/wavelet/filters-acl', fig, step, close=True)
        plt.close()

        fig = plt.figure()
        plt.plot(p_test.squeeze().detach().cpu().numpy())
        plt.plot(np.abs((p_test - two_at_power_zero).squeeze().detach().cpu().numpy()))
        plt.legend(['p_test,', 'err'])
        tensorboard_writer.add_figure(name + '/wavelet/filters-prl', fig, step, close=True)
        plt.close()


if __name__ == "__main__":
    print('haar wavelet')
    w = Wave1D(init_wavelet=pywt.Wavelet('haar'))
    print('acl', w.alias_cancellation_loss()[0])
    print('prl', w.perfect_reconstruction_loss()[0])

    print('db6 wavelet')
    w = Wave1D(init_wavelet=pywt.Wavelet('db6'))
    print('acl', w.alias_cancellation_loss()[0])
    print('prl', w.perfect_reconstruction_loss()[0])

    print('sym3 wavelet')
    w = Wave1D(init_wavelet=pywt.Wavelet('sym3'))
    print('acl', w.alias_cancellation_loss()[0])
    print('prl', w.perfect_reconstruction_loss()[0])