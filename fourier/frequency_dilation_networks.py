import torch
from fourier.fourier_convolution import FreqConv1d


class FDN(torch.nn.Module):
    def __init__(self,  in_channels, num_channels, output_channels,
                 initial_kernel_size=2, dropout=0.2):
        '''

        :param num_inputs: The input dimensions.
        :param num_channels: A list with the depth of each FDN layer.
        :param initial_kernel_size: The initial kernel size.
        :param dropout: The neuron dropout probability.

        # TODO: Add dropout.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.initial_kernel_size = initial_kernel_size
        self.dropout = dropout
        layers = []
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(num_channels[-1], output_channels)

        for i, out_channels in enumerate(num_channels):
            dilation_factor = 2
            kernel_size = self.initial_kernel_size*(i+1)
            in_channels = in_channels if i == 0 else num_channels[i-1]
            print('layer', i, 'dilation', dilation_factor, 'kernel', kernel_size, 'channels', in_channels)
            layers += [FreqConv1d(in_channels, out_channels, kernel_size,
                                  freq_dilation=dilation_factor,
                                  padding=kernel_size//2, bias=True)]
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        fdn_out = self.network(x)
        fdn_out_map = self.linear(fdn_out[:, :, -1])
        return fdn_out_map

