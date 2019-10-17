import torch
from torch.nn.utils import weight_norm
from fourier.fourier_convolution import FreqConv1d


class FreqBlock(torch.nn.Module):
    '''
    TODO: Write me!
    '''
    def __init__(self, in_channels, out_channels, kernel_size, dilation_factor, dropout):
        super().__init__()
        # self.conv1 = weight_norm(FreqConv1d(in_channels, out_channels, kernel_size,
        #                          freq_dilation=dilation_factor,
        #                          padding=kernel_size//2, bias=True))
        # self.relu1 = torch.nn.ReLU()
        # self.dropout1 = nn.Dropout()
        pass

    def forward(self, x):
        pass


class FDN(torch.nn.Module):
    def __init__(self,  in_channels, num_channels, output_channels=None,
                 initial_kernel_size=4, dropout=0.2):
        '''

        :param num_inputs: The input dimensions.
        :param num_channels: A list with the depth of each FDN layer.
        :param initial_kernel_size: The initial kernel size.
        :param dropout: The neuron dropout probability.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.initial_kernel_size = initial_kernel_size
        self.dropout = dropout
        layers = []
        self.relu = torch.nn.ReLU()
        self.output_channels = output_channels
        if self.output_channels is not None:
            self.linear = torch.nn.Linear(num_channels[-1], output_channels)

        for i, out_channels in enumerate(num_channels):
            dilation_factor = 2 - (1./len(num_channels))*i
            kernel_size = self.initial_kernel_size  # *(i+1)
            in_channels = in_channels if i == 0 else num_channels[i-1]
            print('layer', i, 'dilation', dilation_factor, 'kernel', kernel_size, 'channels', in_channels)
            layers += [weight_norm(FreqConv1d(in_channels, out_channels, kernel_size,
                                   freq_dilation=dilation_factor,
                                   padding=kernel_size//2, bias=True)),
                       torch.nn.Dropout(self.dropout)]
        self.network = torch.nn.Sequential(*layers)
        self._print = True

    def forward(self, x):
        '''
        Compute the forward pass.
        :param x: Input tensor of shape [batch_size, in_channels, time]
        :return: TODO
        '''
        fdn_out = self.network(x)
        if self.output_channels is not None:
            batch_size = fdn_out.shape[0]
            if self._print:
                print('batch_size', batch_size, ' fdn_out.shape', fdn_out.shape)
                self._print = False
            # fdn_out = fdn_out.reshape([batch_size, -1])
            fdn_out = fdn_out[:, :, -1]
            fdn_out = self.linear(fdn_out)
        return fdn_out

