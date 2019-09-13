import torch
# from torch.utils import load_state_dict_from_url
import torchvision.models as models
import torch.nn as nn
from wave.wavelet_linear import WaveletLayer
from fastfood.fastfood import FastFoodLayer


class DeepFriedAlexNet(nn.Module):
    '''
    A re-implementation of the modified Alexnet proposed in
    https://arxiv.org/pdf/1412.7149.pdf
    '''

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # nn.Linear(256 * 6 * 6, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            FastFoodLayer(256*6*6),  # TODO: Dropout
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class WaveletAlexNet(nn.Module):
    '''
    A re-implementation of the modified Alexnet proposed in
    https://arxiv.org/pdf/1412.7149.pdf
    '''

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # nn.Linear(256 * 6 * 6, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            WaveletLayer(256*6*6),  # TODO: Dropout, after the diagonal layers.
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def convert_states(alexnet_states):
    reduced_dict = {}
    for key in alexnet_states.keys():
        if key.split('.')[0] == 'classifier':
            if key.split('.')[1] == '6':
                new_key = 'classifier.3.' + key.split('.')[-1]
                reduced_dict[new_key] = alexnet_states[key]
        else:
            reduced_dict[key] = alexnet_states[key]
    return reduced_dict


def freeze_alexnet_features(alex_type_model):
    for name, param in alex_type_model.named_parameters():
        if name.split('.')[0] == 'features':
            print('turn of optimization for', name)
            param.requires_grad = False


if __name__ == '__main__':
    alexnet = models.alexnet(pretrained=True)
    fried_alex = DeepFriedAlexNet()

    fried_dict = convert_states(alexnet.state_dict())
    missing_keys, unexpected_keys = fried_alex.load_state_dict(fried_dict, strict=False)
    print(missing_keys, unexpected_keys)

    freeze_alexnet_features(fried_alex)
