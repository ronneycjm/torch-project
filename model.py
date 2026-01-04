import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    """
    VGG16 model implementation from the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    """
    
    def __init__(self, num_classes=1000, init_weights=True):
        """
        Initialize the VGG16 model
        
        Args:
            num_classes (int): Number of output classes for classification
            init_weights (bool): Whether to initialize weights using Xavier initialization
        """
        super(VGG16, self).__init__()
        
        # Define the convolutional layers based on VGG16 architecture
        # Configuration: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fifth conv block
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Define the classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Assuming input is 224x224
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        """
        Forward pass of the VGG16 model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """
        Initialize weights using Xavier initialization for linear layers
        and normal initialization for convolutional layers
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg16(pretrained=False, **kwargs):
    """
    Create a VGG16 model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        **kwargs: Additional arguments to pass to the model
    
    Returns:
        VGG16: A VGG16 model instance
    """
    model = VGG16(**kwargs)
    if pretrained:
        # Load pretrained weights here if needed
        # This would require downloading the pretrained model
        pass
    return model


def vgg16_cifar10():
    """
    Create a VGG16 model adapted for CIFAR-10 dataset (32x32 images, 10 classes)
    
    Returns:
        VGG16: A VGG16 model instance adapted for CIFAR-10
    """
    # For CIFAR-10, we need to adjust the classifier since input size is 32x32
    # Calculate the correct flattened size after all conv and pooling layers
    # Input: 32x32
    # After each conv layer with padding=1 and kernel=3: size remains same
    # After each maxpool with stride=2: size halves
    # So: 32 -> 16 -> 8 -> 4 -> 2 -> 1 (after 5 pooling layers)
    # Final feature map size: 1x1x512
    # So flattened size is 512*1*1=512
    
    model = VGG16(num_classes=10)
    
    # Adjust the first linear layer to match the flattened size for 32x32 input
    model.classifier = nn.Sequential(
        nn.Linear(512, 4096),  # Changed from 512*7*7 to 512 for CIFAR-10
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 10),  # 10 classes for CIFAR-10
    )
    return model


def explain_flatten():
    """
    Explanation and examples of torch.flatten() method
    """
    print("=== EXPLANATION OF torch.flatten() ===\n")
    
    print("SYNTAX:")
    print("torch.flatten(input, start_dim=0, end_dim=-1)")
    print()
    
    print("PARAMETERS:")
    print("- input: The tensor to flatten")
    print("- start_dim (optional): The first dimension to flatten. Default is 0")
    print("- end_dim (optional): The last dimension to flatten. Default is -1 (last dimension)")
    print()
    
    print("WHAT IT DOES:")
    print("Flattens a contiguous range of dimensions in a tensor.")
    print("It reshapes the tensor by combining the specified dimensions into a single dimension.")
    print()
    
    print("EXAMPLES:")
    
    # Example 1: Basic flattening
    print("\n1. Basic flattening (flatten all dimensions):")
    x1 = torch.randn(2, 3, 4)  # Shape: [2, 3, 4]
    print(f"Original tensor shape: {x1.shape}")
    flat1 = torch.flatten(x1)  # Flatten all dimensions
    print(f"After torch.flatten(x1): {flat1.shape}")
    print(f"Element count: {x1.numel()} (should equal {flat1.numel()})")
    
    # Example 2: Flatten from a specific start dimension
    print("\n2. Flatten from start_dim=1 (flatten all except batch):")
    x2 = torch.randn(2, 3, 4)  # Shape: [2, 3, 4]
    print(f"Original tensor shape: {x2.shape}")
    flat2 = torch.flatten(x2, 1)  # Flatten from dimension 1 onwards
    print(f"After torch.flatten(x2, 1): {flat2.shape}")
    print("This keeps the batch dimension (dim 0) separate and flattens the rest")
    print("Useful for neural networks to preserve batch dimension")
    
    # Example 3: Flattening with specific start and end dimensions
    print("\n3. Flatten specific dimensions (start_dim=1, end_dim=2):")
    x3 = torch.randn(2, 3, 4, 5)  # Shape: [2, 3, 4, 5]
    print(f"Original tensor shape: {x3.shape}")
    flat3 = torch.flatten(x3, 1, 2)  # Flatten dimensions 1 and 2
    print(f"After torch.flatten(x3, 1, 2): {flat3.shape}")
    print("This flattens dimensions 1 and 2 (3*4=12), keeping dimensions 0 and 3 separate")
    
    # Example 4: How it's used in VGG16
    print("\n4. How it's used in VGG16 (with typical CNN output):")
    print("After convolutional layers, we typically have:")
    conv_output = torch.randn(4, 512, 7, 7)  # Batch=4, Channels=512, Height=7, Width=7
    print(f"Convolutional output shape: {conv_output.shape}")
    
    flattened = torch.flatten(conv_output, 1)  # Flatten from dimension 1 (keep batch separate)
    print(f"After torch.flatten(conv_output, 1): {flattened.shape}")
    print("This prepares the tensor for the fully connected (linear) layers")
    print("The batch dimension (4) is preserved, and the rest [512, 7, 7] becomes [512*7*7]")
    
    print("\n5. Why use start_dim=1 in VGG16?")
    print("- We want to preserve the batch dimension (dimension 0)")
    print("- We want to flatten the channel, height, and width dimensions")
    print("- This allows processing multiple samples in the batch simultaneously")
    print("- The classifier expects input of shape (batch_size, flattened_features)")


if __name__ == "__main__":
    print("Testing VGG16 model for ImageNet (1000 classes):")
    model_imagenet = vgg16(num_classes=1000)
    sample_input_imagenet = torch.randn(1, 3, 224, 224)
    output_imagenet = model_imagenet(sample_input_imagenet)
    print(f"Input shape: {sample_input_imagenet.shape}")
    print(f"Output shape: {output_imagenet.shape}")
    print()
    
    print("Testing VGG16 model for CIFAR-10 (10 classes):")
    model_cifar = vgg16_cifar10()
    sample_input_cifar = torch.randn(1, 3, 32, 32)
    output_cifar = model_cifar(sample_input_cifar)
    print(f"Input shape: {sample_input_cifar.shape}")
    print(f"Output shape: {output_cifar.shape}")
    print()
    
    print("Model architecture:")
    print(model_imagenet)
    
    # Count parameters
    total_params = sum(p.numel() for p in model_imagenet.parameters())
    trainable_params = sum(p.numel() for p in model_imagenet.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Explain flatten method
    explain_flatten()