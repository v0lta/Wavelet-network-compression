import torch
import collections
import numpy as np
import torchvision
import time
import pywt
import matplotlib.pyplot as plt
from wave.wavelet_linear import WaveletLayer
from fastfood.fastfood import FastFoodLayer
from torch.utils.tensorboard.writer import SummaryWriter


epochs = 30
batch_size = 64
learning_rate_init = 0.001
# learning_rate_init = 0.01
milestones = [10, 20, 30]
gamma = 0.6
runs = 1
wavelet = True
fastfood = False

mnist_data_set = torchvision.datasets.MNIST(root='./data_sets/mnist/', download=True,
                                            transform=torchvision.transforms.Compose([
                                                # torchvision.transforms.Resize([32, 32]),
                                                torchvision.transforms.ToTensor()]), train=True)

mnist_test_set = torchvision.datasets.MNIST(root='./data_sets/mnist/', download=True,
                                            transform=torchvision.transforms.Compose([
                                                # torchvision.transforms.Resize([32, 32]),
                                                torchvision.transforms.ToTensor()]), train=False)

mnist_loader = torch.utils.data.DataLoader(mnist_data_set, batch_size=batch_size,
                                           shuffle=True)
test_mnist_loader = torch.utils.data.DataLoader(mnist_test_set, batch_size=batch_size,
                                                shuffle=False)

means = []
std = []
for input_img, _ in mnist_loader:
    means.append(torch.mean(input_img))
    std.append((torch.std(input_img)))

mean = torch.mean(torch.stack(means)).cuda()
std = torch.mean(torch.stack(std)).cuda()
loss = torch.nn.CrossEntropyLoss()

CustomWavelet = collections.namedtuple('Wavelet', ['dec_lo', 'dec_hi',
                                                   'rec_lo', 'rec_hi', 'name'])


class LeNet5(torch.nn.Module):
    def __init__(self, wavelet=None, fastfood=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 5)
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        self.wavelet = wavelet
        self.fastfood = fastfood

        if wavelet is None and fastfood is False:
            self.fc1 = torch.nn.Linear(4*4*64, 256)
            self.fc2 = torch.nn.Linear(256, 10)
        elif fastfood is True:
            assert wavelet is None
            self.fc1 = FastFoodLayer(1024)
            self.fc2 = torch.nn.Linear(1024, 10)
        else:
            assert wavelet is not None, 'initial wavelet must be set.'

            self.fc1 = WaveletLayer(init_wavelet=wavelet, scales=8, depth=1024)
            self.fc2 = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        # return torch.nn.functional.log_softmax(x, dim=1)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def wavelet_loss(self):
        if self.wavelet is None:
            return torch.tensor(0.0)
        else:
            acl, _, _ = self.fc1.wavelet.alias_cancellation_loss()
            prl, _, _ = self.fc1.wavelet.perfect_reconstruction_loss()
            return acl, prl


test_accs = []
test_max = []
time_per_update = []
for run_no in range(runs):

    if wavelet:
        # init_wavelet = CustomWavelet(
        #    dec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
        #    dec_hi=[0, 0, -0.7071067811865476, 0.7071067811865476, 0, 0],
        #    rec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
        #    rec_hi=[0, 0, 0.7071067811865476, -0.7071067811865476, 0, 0],
        #    name='customHaar')
        # init_wavelet = pywt.Wavelet('db12')
        # init_wavelet = pywt.Wavelet(pywt.wavelist(kind='discrete')[run_no])
        init_wavelet = CustomWavelet(
            dec_lo=np.random.normal(size=12),
            dec_hi=np.random.normal(size=12),
            rec_lo=np.random.normal(size=12),
            rec_hi=np.random.normal(size=12),
            name='random_init')
    else:
        init_wavelet = None

    net = LeNet5(init_wavelet, fastfood).cuda()
    net.eval()
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate_lst[0])
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate_init)
    # optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=learning_rate_init,
    #                             weight_decay=0.0005, nesterov=False)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 12, 18], gamma=gamma)
    loss_fun = torch.nn.CrossEntropyLoss()
    if wavelet:
        comment_str = '_' + init_wavelet.name + '_' + str(run_no) + '_expl'
    elif fastfood:
        comment_str = '_fastfood_run_' + str(run_no)
    else:
        comment_str = '_e_' + str(epochs) + '_bs_' +  \
            str(batch_size) + '_lr_' + str(learning_rate_init) + '_run_' + str(run_no)
    writer = SummaryWriter(comment=comment_str)
    train_loss = []
    train_acl = []
    train_prl = []
    train_iters = 0

    def test():
        # full network acc
        acc_lst_test = []
        with torch.no_grad():
            for _, (input_test, target_test) in enumerate(test_mnist_loader):
                in_norm_test = (input_test.cuda() - mean)/std
                out_test = net(in_norm_test)
                out_test = torch.nn.functional.log_softmax(out_test, dim=1)
                acc = torch.sum((torch.max(out_test, dim=1)[1] == target_test.cuda()).type(torch.float32))/batch_size * 100
                acc_lst_test.append(acc.cpu().numpy())
        writer.add_scalar('test/acc', np.mean(acc_lst_test), train_iters)
        if wavelet:
            net.fc1.wavelet.add_wavelet_summary('wavelet_layer', writer, train_iters)
        # print('accs', acc_lst_test)
        return np.mean(acc_lst_test)


    for e in range(0, epochs):
        for i, (train_input, target) in enumerate(mnist_loader):
            time_start = time.time()
            optimizer.zero_grad()
            in_norm = (train_input.cuda() - mean)/std
            out = net(train_input.cuda())
            cel = loss_fun(out, target.cuda())
            acl, prl = net.wavelet_loss()
            wvl = acl + prl
            loss = cel + wvl
            loss.backward()
            optimizer.step()
            time_per_update.append((time.time() - time_start))
            if i % 15 == 0:
                print('e', e, 'i', i, 'l', cel.detach().cpu().numpy(), 'wvl', wvl.detach().cpu().numpy())
            train_iters += 1
            writer.add_scalar('train/loss', cel.detach().cpu().numpy(), train_iters)
            # writer.add_scalar('train/wavelet-loss', wvl.detach().cpu().numpy(), train_iters)
            writer.add_scalar('train/lr', optimizer.param_groups[-1]['lr'], train_iters)
            train_loss.append(cel.detach().cpu().numpy())
            train_acl.append(acl.detach().cpu().numpy())
            train_prl.append(prl.detach().cpu().numpy())

            if (i % 465 == 0 and i > 0) or (i == 0 and e == 0):
                acc_lst = test()
                print('test acc:', np.mean(acc_lst), 'test error', 100.0-np.mean(acc_lst))
                print('mean time per update', np.mean(time_per_update), 's')
                test_max.append(np.mean(acc_lst))
        # scheduler.step()
        acc = test()

    test_accs.append(acc)
    print('test accs:', test_accs)


def compute_parameter_total(net):
    total = 0
    for p in net.parameters():
        if p.requires_grad:
            print(p.shape)
            total += np.prod(p.shape)
    return total


# compressed network acc.
print('param_total', compute_parameter_total(net))
print('mean acc over ', runs, 'runs ', np.mean(test_accs), np.std(test_accs))
print('max', np.max(test_max))
print('time_per_update', np.mean(time_per_update))
print(net)
print(epochs, batch_size, learning_rate_init)

plot = False
if plot:
    acc_lst = []
    for acc_no, acc in enumerate(test_max):
        acc_lst.append((acc_no*465, acc))
    acc_array = np.array(acc_lst)

    p_lo = np.convolve(init_wavelet.dec_lo, init_wavelet.rec_lo)
    p_hi = np.convolve(init_wavelet.dec_hi, init_wavelet.rec_hi)
    p_test = p_lo + p_hi

    two_at_power_zero = np.zeros(p_test.shape)
    two_at_power_zero[..., p_test.shape[-1]//2] = 2

    length = init_wavelet.rec_lo.shape[0]
    mask = np.array([np.power(-1, n) for n in range(length)][::-1])
    err1 = init_wavelet.rec_lo - mask*init_wavelet.dec_hi
    err1s = np.sum(err1*err1)

    length = init_wavelet.rec_hi.shape[0]
    mask = np.array([np.power(-1, n) for n in range(length)][::-1])
    err2 = init_wavelet.rec_hi - -1*mask*init_wavelet.dec_lo
    err2s = np.sum(err2*err2)

    _, p_test_final, _ = net.fc1.wavelet.perfect_reconstruction_loss()
    p_test_final_np = p_test_final.squeeze().detach().cpu().numpy()
    ac_final, err1_final, err2_final = net.fc1.wavelet.alias_cancellation_loss()
    err1_final_np = err1_final.squeeze().detach().cpu().numpy()
    err2_final_np = err2_final.squeeze().detach().cpu().numpy()

    plt.plot(p_test)
    plt.plot(p_test_final_np)
    plt.show()

    plt.plot(err1 + err2)
    plt.plot(err1_final_np + err2_final_np)
    plt.show()

    plt.plot(init_wavelet.dec_lo)
    plt.plot(init_wavelet.dec_hi)
    plt.plot(init_wavelet.rec_lo)
    plt.plot(init_wavelet.rec_hi)
    plt.legend(['dec_lo', 'dec_hi', 'rec_lo', 'rec_hi'])
    plt.show()

    plt.plot(net.fc1.wavelet.dec_lo.detach().cpu().numpy())
    plt.plot(net.fc1.wavelet.dec_hi.detach().cpu().numpy())
    plt.plot(net.fc1.wavelet.rec_lo.detach().cpu().numpy())
    plt.plot(net.fc1.wavelet.rec_hi.detach().cpu().numpy())
    plt.legend(['dec_lo', 'dec_hi', 'rec_lo', 'rec_hi'])
    plt.show()
