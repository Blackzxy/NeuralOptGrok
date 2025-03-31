import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class AdaptiveModelPerturbation(nn.Module):
    """
    Adaptive Model Perturbation (AMP) module for gradient transformation
    """
    def __init__(self, model):
        super().__init__()
        # Store references to original model modules
        self.model = model
        
        # Create learnable parameters for gradient scaling/transformation
        self.module_scales = nn.ParameterDict()
        self.module_shifts = nn.ParameterDict()
        
        # Initialize AMP parameters for each module
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Create learnable scale and shift for each module's gradient
                self.module_scales[name] = nn.Parameter(torch.ones(1))
                self.module_shifts[name] = nn.Parameter(torch.zeros(1))
    
    def transform_gradient(self, module_name, gradient):
        """
        Transform gradient for a specific module
        
        Args:
            module_name (str): Name of the module
            gradient (torch.Tensor): Original gradient
        
        Returns:
            torch.Tensor: Transformed gradient
        """
        if module_name in self.module_scales and module_name in self.module_shifts:
            scale = self.module_scales[module_name]
            shift = self.module_shifts[module_name]
            
            # Gradient transformation: scale * gradient + shift
            transformed_gradient = scale * gradient + shift
            
            return transformed_gradient
        return gradient
    
    def forward(self, x):
        """
        Forward pass through the original model
        """
        return self.model(x)

def train_with_amp(model, amp, train_loader, test_loader, lr=0.01, epochs=5):
    """
    Training loop with AMP gradient transformation
    
    Args:
        model (nn.Module): Original model
        amp (AdaptiveModelPerturbation): AMP module
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        lr (float): Learning rate
        epochs (int): Number of training epochs
    """
    # Optimizer for both model and AMP parameters
    optimizer = optim.SGD(list(model.parameters()) + list(amp.parameters()), lr=lr)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output = amp(data)
            loss = F.cross_entropy(output, target)
            
            # Backward pass with gradient transformation
            loss.backward()
            
            # Custom gradient transformation for each module
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and module.weight.grad is not None:
                    # Transform gradient for module
                    module.weight.grad = amp.transform_gradient(name, module.weight.grad)
            
            # Update model parameters
            optimizer.step()
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = amp(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Backpropagate test loss to update AMP parameters
        test_loss /= len(test_loader.dataset)
        test_loss_tensor = torch.tensor(test_loss, requires_grad=True)
        
        # Compute gradients for AMP parameters based on test loss
        test_loss_tensor.backward()
        
        print(f'Epoch {epoch+1}: Test Loss: {test_loss:.4f}, '
              f'Accuracy: {100. * correct / len(test_loader.dataset):.2f}%')
    
    return model, amp

# Example usage (placeholder implementation)
class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Simulated data loaders (replace with actual data loaders)
class MockDataLoader:
    def __init__(self, dataset_size=1000, input_size=784, num_classes=10):
        self.dataset_size = dataset_size
        self.data = torch.randn(dataset_size, input_size)
        self.targets = torch.randint(0, num_classes, (dataset_size,))
    
    def __iter__(self):
        for i in range(0, self.dataset_size, 32):
            batch_data = self.data[i:i+32]
            batch_targets = self.targets[i:i+32]
            yield batch_data, batch_targets
    
    def __len__(self):
        return self.dataset_size

# Create mock data loaders
train_loader = MockDataLoader()
test_loader = MockDataLoader()

# Initialize model and AMP
model = ExampleModel()
amp_model = AdaptiveModelPerturbation(model)

# Train the model with AMP
trained_model, amp = train_with_amp(model, amp_model, train_loader, test_loader)

# Example of accessing AMP module scales and shifts
print("\nAMP Module Scales:")
for name, scale in amp.module_scales.items():
    print(f"{name}: {scale.item()}")

print("\nAMP Module Shifts:")
for name, shift in amp.module_shifts.items():
    print(f"{name}: {shift.item()}")