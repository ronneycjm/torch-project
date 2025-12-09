import torch
from torch import nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers here

    def forward(self, x):
        # Define forward pass here
        output = x + 1
        return output


test_Module = MyModule()
x = torch.tensor(1.0)

output = test_Module(x)
print(output)



