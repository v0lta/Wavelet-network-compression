import torch

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
