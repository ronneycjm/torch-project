import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from model import vgg16_cifar10

# Load CIFAR-10 dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load test dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load the VGG16 model adapted for CIFAR-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vgg16_cifar10().to(device)

print("Model loaded successfully!")
print(f"Model will run on: {device}")
print(f"Model architecture:\n{model}")

# Test with a batch from the dataset
data_iter = iter(test_loader)
images, labels = next(data_iter)
print(f"\nSample batch shape: {images.shape}")
print(f"Sample labels: {labels}")

# Run a forward pass
with torch.no_grad():
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs, 1)
    
print(f"Model outputs shape: {outputs.shape}")
print(f"Predicted classes: {predicted}")
print(f"Actual labels: {labels}")

# Print some information about the model
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")