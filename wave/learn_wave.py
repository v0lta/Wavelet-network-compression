import torch
from wave.lowlevel import afb1d
'''A learnable filter wavelet transform in 1D.'''


class Wave1D(torch.nn.Module):
    def __init__(self, dec_lo, dec_hi, rec_lo, rec_hi, scales=1):
        super().__init__()
        self.dec_lo = torch.Tensor(dec_lo)
        self.dec_hi = torch.Tensor(dec_hi)
        self.rec_lo = torch.Tensor(rec_lo)
        self.rec_hi = torch.Tensor(rec_hi)

        self.scales = scales

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

    def orthogonal_filter_cond(self):
        '''
        Check the biorthogonality conditions as described in Strang and Nguyen; Wavelets
        and filter banks; Page 111.
        '''
        # all filter shapes must be the same.
        assert self.dec_lo.shape == self.dec_hi.shape
        assert self.rec_lo.shape == self.rec_hi.shape
        assert self.dec_lo.shape == self.rec_lo.shape

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
            lohi = afb1d(lo, self.dec_lo, self.dec_hi, mode='zero', dim=-1)
            lo, hi = torch.split(lohi, split_size_or_sections=[1, 1], dim=1)
            yh.append(hi)
        # yh.insert(-1, lo)
        yh.append(lo)
        return yh

    def reconstruction(self, X):
        pass
