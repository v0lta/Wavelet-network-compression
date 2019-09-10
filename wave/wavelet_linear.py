import torch
import numpy as np
from torch.nn.parameter import Parameter
from wave.learn_wave import Wave1D


class WaveletLinear(torch.nn.Module):
    def __init__(self, init_wavelet, scales, cut_off,
                 in_features, out_features, bias=False):
        super().__init__()
        self.scales = scales
        self.cut_off = cut_off
        self.wavelet = Wave1D(init_wavelet,  scales=scales)
        self.in_features = in_features
        self.out_features = out_features
        scales = self.wavelet.compute_coeff_no(out_features)

        self.scale_list = []
        for no, s in enumerate(scales):
            if no > cut_off:
                self.scale_list.append(
                    torch.nn.Parameter(torch.Tensor(in_features, s).uniform_(-.2, .2).unsqueeze(1).unsqueeze(1)))
            else:
                self.scale_list.append(torch.zeros(in_features, s).unsqueeze(1).unsqueeze(1))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        print('wavlet linear layer.')

    def forward(self, x):
        mat = self.wavelet.reconstruction(self.scale_list)
        return torch.mm(x, mat.squeeze(1).squeeze(1))


class WaveletLayer(torch.nn.Module):
    """
    Create a learn-able Fast-food layer as described in
    https://arxiv.org/abs/1412.7149
    The weights are parametrized by S*H*G*P*H*B
    With S,G,B diagonal matrices, P a random permutation and H the Walsh-Hadamard transform.
    """
    def __init__(self, depth, init_wavelet, scales):
        super().__init__()
        ones = np.ones(depth, np.float32)
        self.diag_vec_s = Parameter(torch.from_numpy(ones))
        self.diag_vec_g = Parameter(torch.from_numpy(ones))
        self.diag_vec_b = Parameter(torch.from_numpy(ones))
        perm = np.random.permutation(np.eye(depth, dtype=np.float32))
        self.perm = torch.from_numpy(perm)
        self.wavelet = Wave1D(init_wavelet=init_wavelet, scales=scales)

    def mul_s(self, x):
        return torch.mm(x, torch.diag(self.diag_vec_s))

    def mul_g(self, x):
        return torch.mm(x, torch.diag(self.diag_vec_g))

    def mul_b(self, x):
        return torch.mm(x, torch.diag(self.diag_vec_b))

    def mul_p(self, x):
        return torch.mm(x, self.perm)

    def wavelet_analysis(self, x):
        c_lst = self.wavelet.analysis(x.unsqueeze(0).unsqueeze(0))
        c_tensor = torch.cat([c.squeeze(0).squeeze(0) for c in c_lst], -1)
        x_len = x.shape[-1]
        start_c = c_tensor.shape[-1] - x_len
        return c_tensor[:, start_c:]

    def forward(self, x):
        # test = self.wavelet_analysis(x)
        return self.mul_s(self.wavelet_analysis(self.mul_g(self.mul_p(self.wavelet_analysis(self.mul_b(x))))))