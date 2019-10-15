import torch
import time
import numpy as np
import matplotlib.pyplot as plt


def complex_hadamard(ci1, ci2):
    assert ci1.shape[-1] == 2, 'we require real and imaginary part.'
    assert ci2.shape[-1] == 2, 'we require real and imaginary part.'
    x1 = ci1[..., 0]
    y1 = ci1[..., 1]
    x2 = ci2[..., 0]
    y2 = ci2[..., 1]

    # multiplication in polar form is slow and numerically unstable in the backward pass.
    # r1 = torch.sqrt(x1*x1 + y1*y1)
    # phi1 = torch.atan2(y1, x1)
    # r2 = torch.sqrt(x2*x2 + y2*y2)
    # phi2 = torch.atan2(y2, x2)
    # r = r1*r2
    # phi = phi1 + phi2
    # x = torch.cos(phi)*r
    # y = torch.sin(phi)*r

    x = x1*x2 - y1*y2
    y = x1*y2 + y1*x2
    return torch.stack([x, y], -1)


def freq_convolution_1d(input_tensor, weight_tensor, frequency_dilation=1):
    '''
    Compute a frequency domain convolution using frequency dilation.
    :param input_tensor: The input data tensor [batch_size, dim, time]
    :param weight_tensor: The kernel [out_channels, in_channels, kernel_size].
    :param frequency_dilation: The ration of frequency domain coefficients.
    :return: The kernel input convolution result [batch_size, time].
    '''
    # add the out_channels dimension
    input_tensor = input_tensor.unsqueeze(1)
    # add the batch_dimension.
    weight_tensor = weight_tensor.unsqueeze(0)

    time_steps = input_tensor.shape[-1]
    to_pad = (input_tensor.shape[-1] - weight_tensor.shape[-1])
    pad_weight_tensor = torch.nn.functional.pad(weight_tensor, [to_pad, 0])
    fft_inp = torch.rfft(input_tensor, 1)
    fft_ker = torch.rfft(pad_weight_tensor, 1)
    frequencies = fft_inp.shape[-2]//frequency_dilation
    coefficients = complex_hadamard(fft_ker[..., :frequencies, :],
                                    fft_inp[..., :frequencies, :])
    # sum the input dimension away.
    coefficients = torch.sum(coefficients, 2)
    output = torch.irfft(coefficients, 1)
    return output


def convolution_1d(input_tensor, weight_tensor, pad, freq_dilation=1):
    input_tensor_pad = torch.nn.functional.pad(input_tensor, [pad, pad])
    time_conv = time.time()
    out = torch.nn.functional.conv1d(input_tensor_pad, weight_tensor)
    time_conv = time.time() - time_conv

    # pad the kernel to the same length as the input.
    time_fft = time.time()
    out_pad = freq_convolution_1d(input_tensor_pad, weight_tensor, freq_dilation)
    time_fft = time.time() - time_fft
    print('time conv', time_conv, 'time fft', time_fft)
    return out, out_pad


class FreqConv1d(torch.nn.Module):
    # TODO: Add frequency domain weights to save on the initial kernel fft.
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, freq_dilation=1,
                 time_weights=True, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.freq_dilation = freq_dilation
        self.time_weights = time_weights
        self.bias_add = bias
        if time_weights:
            self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        else:
            pass
            # TODO: explore complex freq-domain weight formulations.
        if self.bias_add:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.1)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -0.01, 0.01)

    def forward(self, input_tensor):
        pad = self.padding // 2
        input_tensor_pad = torch.nn.functional.pad(input_tensor, [pad, pad])
        conv_out = freq_convolution_1d(input_tensor_pad, self.weight, self.freq_dilation)
        if self.bias_add:
            # unsqueeze the batch and time dimensions.
            conv_out += self.bias.unsqueeze(0).unsqueeze(-1)

        unpad = int(np.ceil(pad/self.freq_dilation))
        # if unpad > 0:
        #     conv_out = conv_out[..., unpad:-unpad]
        # print('input_shape', input_tensor.shape, 'output_shape', conv_out.shape)
        return conv_out


if __name__ == '__main__':
    length = 1000
    freq_dilation = 16
    t = np.linspace(0, 2*np.pi, length)
    sine = np.cos(t)
    zeroes = np.zeros([length])
    input_signal = np.concatenate([zeroes, sine, zeroes, sine, zeroes])
    input_signal_t = torch.from_numpy(input_signal).unsqueeze(0).unsqueeze(0)
    kernel_t = torch.from_numpy(sine).unsqueeze(0).unsqueeze(0)
    out, out_pad = convolution_1d(input_signal_t, kernel_t, pad=length//2, freq_dilation=freq_dilation)
    freq_conv = FreqConv1d(1, 1, kernel_size=3, padding=length//2, freq_dilation=freq_dilation)
    out_pad2 = freq_conv(input_signal_t)

    plt.plot(input_signal)
    plt.plot(out[0, 0, :(length*5)].numpy()/np.max(out.numpy()))
    if freq_dilation != 1:
        freq_conv = out_pad[0, 0, :(length*5)].numpy()
        max = np.max(np.abs(freq_conv))
        for pos, fcv in enumerate(freq_conv):
            plt.plot(pos*freq_dilation, fcv/max, 'g.')
            if pos*freq_dilation > (length*5):
                break
    else:
        plt.plot(out_pad[0, 0, :(length*5)].numpy()/np.max(out_pad.numpy()))
    plt.show()

