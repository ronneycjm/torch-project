import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # Define layers here
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        # Define forward pass here
        x = self.conv1(x)
        return x


myModule = MyModule()

for data in dataloader:
    imgs, targets = data
    output = myModule(imgs)
    print(output.shape)
    print(imgs.shape)
