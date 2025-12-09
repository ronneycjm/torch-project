import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

input_matrix = torch.tensor([[1, 2, 3, 4, 5],
                             [6, 7, 8, 9, 10],
                             [11, 12, 13, 14, 15]], dtype=torch.float32)

input_matrix = torch.reshape(input_matrix, (-1, 1, 3, 5))


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, ceil_mode=False)

    def forward(self, x):
        return self.maxpool1(x)


class MyModule2(nn.Module):
    def __init__(self):
        super(MyModule2, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        return self.maxpool1(x)


model = MyModule()
output = model(input_matrix)
print(output)

model2 = MyModule2()
output2 = model2(input_matrix)
print(output2)

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64, shuffle=False,drop_last=True)

for idx, imgs in enumerate(dataloader):
    print(imgs[0].shape)
    print(model(imgs[0]).shape)
    print("******")

