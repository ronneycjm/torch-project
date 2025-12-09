import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Transform the data
data_transform = transforms.Compose([transforms.ToTensor()])

# Download the CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root="data/", train=True, transform=data_transform, download=True
)
test_dataset = torchvision.datasets.CIFAR10(
    root="data/", train=False, transform=data_transform, download=True
)

img, target = test_dataset[0]
print(img.shape, target)


# Create the data loaders
myDataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

for data in myDataloader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)