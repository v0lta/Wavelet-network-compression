import torch
import numpy as np
from torch.nn.parameter import Parameter
from wavelet_learning.learn_wave import Wave1D
import pywt


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
    Create a learn-able Wavelet layer to be published in upcoming paper.
    The weights are parametrized by S*W*G*P*W*B
    With S,G,B diagonal matrices, P a random permutation and W a learnable-wavelet transform.
    """
    def __init__(self, depth, init_wavelet, scales, p_drop=0.5):
        super().__init__()
        self.scales = scales
        self.wavelet = Wave1D(init_wavelet=init_wavelet, scales=scales)
        self.coefficient_len_lst = [depth]
        for _ in range(scales):
            self.coefficient_len_lst.append(self.wavelet.coeff_len(self.coefficient_len_lst[-1]))
        self.coefficient_len_lst = self.coefficient_len_lst[1:]
        self.coefficient_len_lst.append(self.coefficient_len_lst[-1])

        wave_depth = np.sum(self.coefficient_len_lst)
        self.depth = depth
        self.diag_vec_s = Parameter(torch.from_numpy(np.ones(depth, np.float32)))
        self.diag_vec_g = Parameter(torch.from_numpy(np.ones(wave_depth, np.float32)))
        self.diag_vec_b = Parameter(torch.from_numpy(np.ones(depth, np.float32)))
        perm = np.random.permutation(np.eye(wave_depth, dtype=np.float32))
        self.perm = Parameter(torch.from_numpy(perm), requires_grad=False)

        self.drop_s = torch.nn.Dropout(p=p_drop)
        self.drop_g = torch.nn.Dropout(p=p_drop)
        self.drop_b = torch.nn.Dropout(p=p_drop)

    def mul_s(self, x):
        return torch.mm(x, self.drop_s(torch.diag(self.diag_vec_s)))

    def mul_g(self, x):
        return torch.mm(x, self.drop_g(torch.diag(self.diag_vec_g)))

    def mul_b(self, x):
        return torch.mm(x, self.drop_b(torch.diag(self.diag_vec_b)))

    def mul_p(self, x):
        return torch.mm(x, self.perm)

    def wavelet_analysis(self, x):
        c_lst = self.wavelet.analysis(x.unsqueeze(0).unsqueeze(0))
        shape_lst = [c_el.shape[-1] for c_el in c_lst]
        c_tensor = torch.cat([c.squeeze(0).squeeze(0) for c in c_lst], -1)
        assert shape_lst == self.coefficient_len_lst, 'Wavelet shape assumptions false. This is a bug.'
        return c_tensor

    def wavelet_reconstruction(self, x):
        coeff_lst = []
        start = 0
        for s in range(self.scales + 1):
            stop = start + self.coefficient_len_lst[s]
            coeff_lst.append(x[..., start:stop].unsqueeze(0).unsqueeze(0))
            start = self.coefficient_len_lst[s]
        # turn into list
        y = self.wavelet.reconstruction(coeff_lst)
        return y.squeeze(0).squeeze(0)

    def forward(self, x):
        # test = self.wavelet_analysis(x)
        step1 = self.mul_b(x)
        step2 = self.wavelet_analysis(step1)
        step3 = self.mul_p(step2)
        step4 = self.mul_g(step3)
        step5 = self.wavelet_reconstruction(step4)
        step6 = self.mul_s(step5)

        return step6

    def extra_repr(self):
        return 'depth={}'.format(self.depth)

    def get_wavelet_loss(self) -> torch.Tensor:
        prl, _, _ = self.wavelet.perfect_reconstruction_loss()
        acl, _, _ = self.wavelet.perfect_reconstruction_loss()
        return prl + acl