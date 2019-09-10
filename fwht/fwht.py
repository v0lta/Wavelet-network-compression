import torch
import numpy as np


def walsh_hadamard_transform(seq_in, inverse=False, scale=True):
    """Utility function for the Walsh Hadamard Transform,
       produces Hadamard ordered coefficients.
       Found at: https://docs.sympy.org/latest/_modules/sympy/discrete/transforms.html#fwht"""
    assert seq_in.dtype == torch.float32, 'float tensor input required.'

    a = seq_in.clone()

    if inverse and scale:
        a *= len(a)

    n = len(a)
    if n < 2:
        return a

    if n&(n - 1):
        n = 2**n.bit_length()

    # a += [S.Zero]*(n - len(a))
    h = 2
    while h <= n:
        hf, ut = h // 2, n // h
        for i in range(0, n, h):
            for j in range(hf):
                u, v = a[i + j], a[i + j + hf]
                a[i + j], a[i + j + hf] = u + v, u - v
        h *= 2

    if inverse:
        a = a/n
    else:
        # scale if desired
        if scale:
            a = a/(len(a)*1.0)
    return a


if __name__ == '__main__':
    import sympy
    seq = torch.tensor([1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0.])
    print('len', len(seq))
    seq_freq = walsh_hadamard_transform(seq)
    print('freq', seq_freq)
    seq_freq_scl = walsh_hadamard_transform(seq, scale=False)
    print('freq scl', seq_freq_scl)

    seq_rec = walsh_hadamard_transform(seq_freq, inverse=True)
    print(seq_rec.numpy(), seq - seq_rec)
    seq_rec_scl = walsh_hadamard_transform(seq_freq_scl, inverse=True, scale=False)
    print(seq_rec_scl.numpy(), seq - seq_rec_scl)

