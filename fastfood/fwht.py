import torch
import numpy as np
from scipy.linalg import hadamard


def matmul_wht(x, h_mat=None, inverse=False):
    """
    Welsh-Hadamard transform by matrix multiplication.
    @ param x: The sequence to be transformed [batchsize, seq_len].
    @ param inverse: If true computes the inverse transform.
    """
    n = x.shape[-1]

    if h_mat is None:
        h_mat = torch.from_numpy(hadamard(n).astype(np.float32))
        if x.device.type == 'cuda':
            h_mat = h_mat.cuda()
    y = torch.nn.functional.linear(x.unsqueeze(0), h_mat, bias=None)
    if not inverse:
        y = y/n
    return y[0]


def fwht(x, inverse=False):
    """
    Matlab inspired fast welsh-hadamard transform.
    :param inverse: If true the ifwht is computed.
    :param x: The tensor to be transformed
    :return: The welsh hadamard coefficients.
    """

    x = x.clone()

    n = x.shape[-1]
    if n < 2:
        return x

    if n % 2 != 0:
        raise AssertionError("Input feature dimension must be a power of two.")

    for i in range(0, n, 2):
        x[..., i] = x[..., i] + x[..., i+1]
        x[..., i+1] = x[..., i] - 2 * x[..., i+1]

    l = 1
    y = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    for nStage in range(2, int(np.log2(n) + 1)):  # np.log2(n) = number of stages in the flow diagram
        # calculate coefficients for the ith stage specified by nStage
        m = int(np.power(2, l))
        jb = 0
        k = 0
        while k < n:
            # print('jb, jb+m, k, n, m', jb, jb+m, k, n, m)
            for j in range(jb, jb+m, 2):
                y[..., k] = x[..., j] + x[..., j+m]
                y[..., k+1] = x[..., j] - x[..., j+m]
                y[..., k+2] = x[..., j+1] - x[..., j+1+m]
                y[..., k+3] = x[..., j+1] + x[..., j+1+m]
                k = k + 4
            jb = jb + 2*m

        # store coefficients in x at the end of each stage
        x = y.clone()
        l = l + 1
    # perform scaling of coefficients
    if not inverse:
         y = x / n
    return y


def walsh_hadamard_transform(seq_in, inverse=False, scale=True):
    """Utility function for the Walsh Hadamard Transform,
       produces Hadamard ordered coefficients.
       Based on: https://docs.sympy.org/latest/_modules/sympy/discrete/transforms.html#fwht"""
    assert seq_in.dtype == torch.float32, 'float tensor input required.'

    a = seq_in.clone()

    if inverse and scale:
        a *= len(a)

    n = a.shape[-1]
    if n < 2:
        return a

    if n % 2 != 0:
        raise AssertionError("Input feature dimension must be a power of two.")

    # zero padding
    # a += [S.Zero]*(n - len(a))
    h = 2
    while h <= n:
        hf, ut = h // 2, n // h
        for i in range(0, n, h):
            for j in range(hf):
                u, v = a[..., i + j], a[..., i + j + hf]
                a[..., i + j], a[..., i + j + hf] = u + v, u - v
        h *= 2

    if inverse:
        a = a/n
    else:
        # scale if desired
        if scale:
            a = a/(len(a)*1.0)
    return a


if __name__ == '__main__':
    seq = torch.tensor([1., 1., 1., 1., 0, 0, 0, 1.])
    print('len', len(seq))
    seq_freq = walsh_hadamard_transform(seq)
    print('freq', seq_freq)
    seq_freq_scl = walsh_hadamard_transform(seq, scale=False)
    print('freq scl', seq_freq_scl)
    seq_rec = walsh_hadamard_transform(seq_freq, inverse=True)
    print(seq_rec.numpy(), seq - seq_rec)
    seq_rec_scl = walsh_hadamard_transform(seq_freq_scl, inverse=True, scale=False)
    print(seq_rec_scl.numpy(), seq - seq_rec_scl)

    fwht_seq = fwht(seq)
    print(fwht_seq)

    # haramard
    res = matmul_wht(seq.unsqueeze(0), inverse=False)
    print('res', res)
    inv = matmul_wht(res, inverse=True)
    print('inv', inv)
