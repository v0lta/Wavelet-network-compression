import torch
import collections
import numpy as np
import torchvision
import pywt
import matplotlib.pyplot as plt
from wave.wavelet_linear import WaveletLayer
from fastfood.fastfood import FastFoodLayer
from torch.utils.tensorboard.writer import SummaryWriter

epochs = 2
batch_size = 64
learning_rate = 0.001
runs = 20
wavelet = False
fastfood = True

mnist_data_set = torchvision.datasets.MNIST(root='./data_sets/mnist/', download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,))
                                            ]), train=True)

mnist_test_set = torchvision.datasets.MNIST(root='./data_sets/mnist/', download=True,
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
        self.conv1 = torch.nn.Conv2d(1, 8, 5, 1)
        self.conv2 = torch.nn.Conv2d(8, 16, 5, 1)
        self.wavelet = wavelet
        self.fastfood = fastfood

        if wavelet is False and fastfood is False:
            self.fc1 = torch.nn.Linear(4*4*16, 256)
            self.fc2 = torch.nn.Linear(256, 10)
        elif fastfood is True:
            assert wavelet is False
            self.fc1 = FastFoodLayer(256)
            self.fc2 = torch.nn.Linear(256, 10)
        else:
            assert wavelet is True
            wavelet = CustomWavelet(dec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                    dec_hi=[0, 0, -0.7071067811865476, 0.7071067811865476, 0, 0],
                                    rec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                    rec_hi=[0, 0, 0.7071067811865476, -0.7071067811865476, 0, 0],
                                    name='customHaar')
            self.fc1 = WaveletLayer(init_wavelet=wavelet, scales=8, depth=256)
            self.fc2 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*16)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        # return torch.nn.functional.log_softmax(x, dim=1)
        return x

    def wavelet_loss(self):
        if self.wavelet is False:
            return torch.tensor(0.0)
        else:
            return self.fc1.wavelet.alias_cancellation_loss() + self.fc1.wavelet.perfect_reconstruction_loss()


test_accs = []
for run_no in range(runs):
    net = Net(wavelet=wavelet, fastfood=fastfood).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_fun = torch.nn.CrossEntropyLoss()
    if wavelet:
        comment_str = '_wavelet_run_' + str(run_no)
    elif fastfood:
        comment_str = '_fastfood_run_' + str(run_no)
    else:
        comment_str = '_run_' + str(run_no)
    writer = SummaryWriter(comment=comment_str)
    train_loss = []
    train_iters = 0

    def test():
        # full network acc
        acc_lst_test = []
        with torch.no_grad():
            for _, (input_test, target_test) in enumerate(test_mnist_loader):
                out_test = net(input_test.cuda())
                out_test = torch.nn.functional.log_softmax(out_test, dim=1)
                acc = torch.sum((torch.max(out_test, dim=1)[1] == target_test.cuda()).type(torch.float32))/batch_size * 100
                acc_lst_test.append(acc.cpu().numpy())
        writer.add_scalar('Loss/acc', np.mean(acc_lst_test), train_iters)
        return acc_lst_test


    for e in range(0, epochs):
        print(e)
        for i, (input, target) in enumerate(mnist_loader):
            optimizer.zero_grad()
            out = net(input.cuda())
            cel = loss_fun(out, target.cuda())
            wvl = net.wavelet_loss()
            loss = cel + wvl
            loss.backward()
            optimizer.step()
            print('e', e, 'i', i, 'l', cel.detach().cpu().numpy(), wvl.detach().cpu().numpy())
            train_iters += 1
            writer.add_scalar('Loss/train', cel.detach().cpu().numpy(), train_iters)

            if i % 300 == 0:
                acc_lst = test()
                print('test accs:', np.mean(acc_lst))

        acc_lst = test()

    test_accs.append(np.mean(acc_lst))
    print('test accs:', test_accs)

def compute_parameter_total(net):
    total = 0
    for p in net.parameters():
        if p.requires_grad:
            print(p.shape)
            total += np.prod(p.shape)
    return total


# compressed network acc.
print('mean accs:', test_accs)
print('param_total', compute_parameter_total(net))
print('mean acc over ', runs, 'runs ', np.mean(test_accs), np.std(test_accs))
print(net)
