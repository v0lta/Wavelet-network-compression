import numpy as np
import matplotlib.pyplot as plt

# kernel_lst = [1., 2., 3., 4., 5., 4., 3., 2., 1.]
# kernel = [1., 2., 1.]
kernel_lst = np.random.uniform(0, 10, 10)


def dilation(kernel, factor):
    new_kernel = []
    for kernel_el in kernel:
        new_kernel.append(kernel_el)
        for _ in range(factor):
            new_kernel.append(0.0)
    return new_kernel


pad_to = 10000

for dilation_factor in range(6):
    kernel = np.array(dilation(kernel_lst, dilation_factor))
    print(dilation_factor, kernel.shape)
    to_pad = (pad_to - kernel.shape[-1])
    pad_weight_tensor = np.pad(kernel, [to_pad, 0])
    fft_res = np.fft.rfft(pad_weight_tensor)
    # plt.semilogy(np.abs(fft_res))
    plt.plot(np.abs(fft_res), label=str(dilation_factor))
plt.legend()
plt.show()