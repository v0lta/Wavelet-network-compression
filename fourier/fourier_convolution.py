import torch
import time
import numpy as np
import matplotlib.pyplot as plt


def complex_hadamard(ci1, ci2):
    assert ci1.shape[-1] == 2, 'we require real and imaginary part.'
    assert ci2.shape[-1] == 2, 'we require real and imaginary part.'
    x1 = ci1[..., 0]
    y1 = ci1[..., 1]
    r1 = torch.sqrt(x1*x1 + y1*y1)
    phi1 = torch.atan2(y1, x1)

    x2 = ci2[..., 0]
    y2 = ci2[..., 1]
    r2 = torch.sqrt(x2*x2 + y2*y2)
    phi2 = torch.atan2(y2, x2)

    r = r1*r2
    phi = phi1 + phi2

    x = torch.cos(phi)*r
    y = torch.sin(phi)*r
    return torch.stack([x, y], -1)


def freq_convolution_1d(input_tensor, weight_tensor, freq_dilation=1):
    to_pad = (input_tensor.shape[-1] - weight_tensor.shape[-1])
    pad_weight_tensor = torch.nn.functional.pad(weight_tensor, (to_pad, 0))
    fft_inp = torch.rfft(input_tensor, 1)
    fft_ker = torch.rfft(pad_weight_tensor, 1)
    freqs = fft_inp.shape[-2]//freq_dilation
    out = torch.irfft(complex_hadamard(fft_ker[:, :, :freqs, :], fft_inp[:, :, :freqs, :]), 1)
    return out


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


class FreqConv1d(torch.nn.module):
    # TODO: Write me!
    pass


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
