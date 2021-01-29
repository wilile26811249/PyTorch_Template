import torch
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from collections import OrderedDict

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = x.mm(self.w)
        return x + self.b.expand_as(x)

class Perceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(Perceptron, self).__init__()
        self.layer1 = Linear(in_features, hidden_features)
        self.layer2 = Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        return self.layer2(x)

to_tensor = ToTensor()
to_pil = ToPILImage()

net1 = nn.Sequential()
net1.add_module('conv', nn.Conv2d(3, 3, 3))
net1.add_module('bachnorm', nn.BatchNorm2d(3))
net1.add_module('activation_layer', nn.ReLU())

net2 = nn.Sequential(
    nn.Conv2d(3, 3, 3),
    nn.BatchNorm2d(3),
    nn.ReLU()
)

net3 = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(3, 3, 3)),
    ('bn1', nn.BatchNorm2d(3)),
    ('relu1', nn.ReLU())
]))

print('Net1: ', net1)
print('Net2: ', net2)
print('Net3: ', net3)
