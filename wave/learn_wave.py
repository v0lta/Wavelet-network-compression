import torch
import numpy as np
from wave.lowlevel import afb1d
import ipdb
'''A learnable filter wavelet transform in 1D.'''


class Wave1D(torch.nn.Module):
    def __init__(self, init_wavelet, scales=1,
                 mode='zero'):
        super().__init__()

        def to_tensor(filt_lst: list):
            return torch.tensor(np.copy(np.array(filt_lst).ravel()[::-1]),
                                dtype=torch.float)  # , device=x.device

        self.init_wavelet = init_wavelet
        self.dec_lo = to_tensor(init_wavelet.dec_lo)
        self.dec_hi = to_tensor(init_wavelet.dec_hi)
        self.rec_lo = to_tensor(init_wavelet.rec_lo)
        self.rec_hi = to_tensor(init_wavelet.rec_hi)

        self.scales = scales
        self.mode = mode

    def check_sym(self, filt) -> bool:
        # 1d filter arrays.
        assert len(filt.shape) == 1
        length = filt.shape[0]
        sym = True
        for i in range(length//2):
            if filt[i] != filt[length-1]:
                sym = False
        return sym

    def check_antisym(self, filt) -> bool:
        assert len(filt.shape) == 1
        length = filt.shape[0]
        anti_sym = True
        for i in range(length//2):
            if filt[i] != -filt[length-1]:
                anti_sym = False
        return anti_sym

    def alias_cancellation_loss(self) -> torch.Tensor:
        '''
        Strang 105: F0(z) = H1(-z); F1(z) = -H0(-z)
        '''
        assert self.rec_lo.shape == self.dec_hi.shape, 'filters must have the same sizes'
        assert self.rec_hi.shape == self.dec_lo.shape, 'filters must have the same sizes'

        m1 = torch.Tensor([-1])
        length = self.rec_lo.shape[0]
        mask = torch.Tensor([torch.pow(m1, n) for n in range(length)])
        err1 = self.rec_lo - mask*self.dec_hi
        err1 = torch.sum(err1*err1)

        length = self.rec_hi.shape[0]
        mask = torch.Tensor([torch.pow(m1, n) for n in range(length)])
        err2 = self.rec_hi - m1*mask*self.dec_lo
        err2 = torch.sum(err2*err2)
        return err1 + err2

    def perfect_reconstruction_loss(self) -> torch.Tensor:
        '''
        Strang 107:
        Assuming alias cancellation holds:
        P(z) = F(z)H(z)
        Product filter P(z) + P(-z) = 2.
        However since alias cancellation is implemented as soft constraint:
        P_0 + P_1 = 2
        '''
        assert self.dec_lo.shape == self.rec_lo.shape, 'filters must have the same sizes'
        assert self.dec_hi.shape == self.rec_hi.shape, 'filters must have the same sizes'
        assert self.dec_lo.shape == self.dec_hi.shape, 'filters must have the same sizes'

        # polynomial multiplication is convolution, compute p(z):
        pad = self.dec_lo.shape[0]-1  # TODO: len(input/2) ?
        p_lo = torch.nn.functional.conv1d(
            self.dec_lo.unsqueeze(0).unsqueeze(0),
            self.rec_lo.unsqueeze(0).unsqueeze(0),
            padding=pad)

        pad = self.dec_hi.shape[0]-1
        p_hi = torch.nn.functional.conv1d(
            self.dec_hi.unsqueeze(0).unsqueeze(0),
            self.rec_hi.unsqueeze(0).unsqueeze(0),
            padding=pad)

        p_test = p_lo + p_hi
        two_at_power_zero = torch.zeros(p_test.shape)
        ipdb.set_trace()
        two_at_power_zero[..., p_test.shape[-1]//2] = 2
        return torch.sum((p_test - two_at_power_zero)*(p_test - two_at_power_zero))

    def orthogonal_filter_cond(self):
        '''
        Check the biorthogonality conditions as described in Strang and Nguyen; Wavelets
        and filter banks; Page 111.
        '''
        # all filter shapes must be the same.
        assert self.dec_lo.shape == self.dec_hi.shape, 'filters must have the same sizes'
        assert self.rec_lo.shape == self.rec_hi.shape, 'filters must have the same sizes'
        assert self.dec_lo.shape == self.rec_lo.shape, 'filters must have the same sizes'

        # if even:
        if self.dec_lo.shape[0] % 2 == 0:
            pass

        # if odd
        if self.dec_lo.shape[0] % 2 != 0:
            # filters must symmetric and antisymmetric
            for no, dec_filt in enumerate(self.dec_lo):
                pass

    def analysis(self, x, mode='zero'):
        yh = []
        lo = x
        for s in range(self.scales):
            lohi = afb1d(lo, self.dec_lo, self.dec_hi, mode=self.mode, dim=-1)
            lo, hi = torch.split(lohi, split_size_or_sections=[1, 1], dim=1)
            yh.append(hi)
        # yh.insert(-1, lo)
        yh.append(lo)
        return yh

    def reconstruction(self, X):
        pass
