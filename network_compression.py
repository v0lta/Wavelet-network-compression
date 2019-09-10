import torch
import collections
import numpy as np
import torchvision
import pywt
import matplotlib.pyplot as plt
from wave.learn_wave import Wave1D
from wave.wavelet_linear import WaveletLinear
from fastfood.fastfood import FastFoodLayer

epochs = 1
batch_size = 64
learning_rate = 0.001

mnist_data_set = torchvision.datasets.MNIST(root='./mnist/', download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,))
                                            ]), train=True)

mnist_test_set = torchvision.datasets.MNIST(root='./mnist/', download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                            ]), train=False)

mnist_loader = torch.utils.data.DataLoader(mnist_data_set, batch_size=batch_size,
                                           shuffle=True)
test_mnist_loader = torch.utils.data.DataLoader(mnist_test_set, batch_size=batch_size,
                                                shuffle=False)
loss = torch.nn.CrossEntropyLoss()


CustomWavelet = collections.namedtuple('Wavelet', ['dec_lo', 'dec_hi',
                                                   'rec_lo', 'rec_hi', 'name'])



class Net(torch.nn.Module):
    def __init__(self, wavelet=False, fastfood=False):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.wavelet = wavelet
        self.fastfood = fastfood

        if wavelet is False and fastfood is False:
            self.fc1 = torch.nn.Linear(4*4*50, 800)
            self.fc2 = torch.nn.Linear(800, 10)
        elif fastfood is True:
            self.fc1 = FastFoodLayer(800)
            self.fc2 = torch.nn.Linear(800, 10)
        else:
            # TODO: Start from noise here! Or perhaps a haar wavelet which turns into something else.
            wavelet = CustomWavelet(dec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                    dec_hi=[0, 0, -0.7071067811865476, 0.7071067811865476, 0, 0],
                                    rec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                    rec_hi=[0, 0, 0.7071067811865476, -0.7071067811865476, 0, 0],
                                    name='custom')
            self.fc1 = WaveletLinear(init_wavelet=wavelet, scales=8, cut_off=0, in_features=4*4*50,
                                     out_features=500)
            self.fc2 = WaveletLinear(init_wavelet=wavelet, scales=8, cut_off=0, in_features=500,
                                     out_features=10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        # return torch.nn.functional.log_softmax(x, dim=1)
        return x

    def wavelet_loss(self):
        if self.wavelet is False:
            return torch.tensor(0.0)
        else:
            return self.fc1.wavelet.alias_cancellation_loss() + self.fc1.wavelet.perfect_reconstruction_loss() \
                 + self.fc2.wavelet.alias_cancellation_loss() + self.fc2.wavelet.perfect_reconstruction_loss()


net = Net(wavelet=False, fastfood=True)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_fun = torch.nn.CrossEntropyLoss()

train_loss = []
for e in range(0, epochs):
    print(e)
    for i, (input, target) in enumerate(mnist_loader):
        optimizer.zero_grad()
        out = net(input)
        cel = loss_fun(out, target)
        wvl = net.wavelet_loss()
        loss = cel + wvl
        loss.backward()
        optimizer.step()
        print(e, i, cel.detach().numpy(), wvl.detach().numpy())


# full network acc
acc_lst = []
with torch.no_grad():
    for i, (input, target) in enumerate(test_mnist_loader):
        out = net(input)
        out = torch.nn.functional.log_softmax(out, dim=1)
        acc = torch.sum((torch.max(out, dim=1)[1] == target).type(torch.float32))/batch_size * 100
        acc_lst.append(acc)


def compute_parameter_total(net):
    total = 0
    for p in net.parameters():
        print(p.shape)
        total += np.prod(p.shape)
    return total

# compressed network acc.
print('test acc:', np.mean(acc_lst))
print('param_total', compute_parameter_total(net))

# take a look at the wavelet coefficients.
# wavelet = pywt.Wavelet('db6')
# wave1d = Wave1D(init_wavelet=wavelet, scales=5)
# coeff = wave1d.analysis(net.fc1.weight.unsqueeze(1).unsqueeze(1))

# coeff_cat = torch.cat(coeff, -1).squeeze(1).squeeze(1).detach().numpy()
# plt.plot(coeff_cat.flatten())
# plt.show()

# coefficient cutoff approach.