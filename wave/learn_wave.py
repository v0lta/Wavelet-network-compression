import torch
from wave.lowlevel import afb1d
'''A learnable filter wavelet transform in 1D.'''


class Wave1D(torch.nn.Module):
    def __init__(self, dec_lo, dec_hi, rec_lo, rec_hi, scales=1):
        self.dec_lo = dec_lo
        self.dec_hi = dec_hi
        self.rec_lo = rec_lo
        self.rec_hi = rec_hi
        self.scales = scales

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
