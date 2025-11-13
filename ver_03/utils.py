# ver_03/utils.py
# 'rl/ver_02'의 utils.py와 동일. (ver_03용으로 복사)
# PPO 클래스에서 save/load 로직이 이미 구현되었으므로,
# 이 파일은 현재 사용되지 않을 수 있으나 호환성을 위해 유지.

import torch
import torch.nn as nn
import os

# Function to save model parameters
def save_model(model, filename):
    """
    Saves the state dictionary of a PyTorch model to a file.
    
    Args:
        model (torch.nn.Module): The model to save.
        filename (str): The path to the file where the model should be saved.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

# Function to load model parameters
def load_model(model, filename, device='cpu'):
    """
    Loads the state dictionary from a file into a PyTorch model.
    
    Args:
        model (torch.nn.Module): The model to load parameters into.
        filename (str): The path to the file from which to load parameters.
        device (str or torch.device): The device to load the model onto.
    """
    try:
        model.load_state_dict(torch.load(filename, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode after loading
        print(f"Model loaded from {filename}")
    except FileNotFoundError:
        print(f"Error: No model file found at {filename}")
    except Exception as e:
        print(f"Error loading model: {e}")

# Function to initialize weights
def init_weights(m):
    """
    Initializes weights of linear layers using orthogonal initialization
    and biases to zero.
    
    Args:
        m (torch.nn.Module): The module to initialize.
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=nn.init.calculate_gain('tanh'))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# Example usage (optional, for testing)
if __name__ == "__main__":
    # Example model
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.layer = nn.Linear(10, 5)
            self.apply(init_weights) # Apply weight initialization

        def forward(self, x):
            return self.layer(x)

    net = SimpleNet()
    print("Model architecture:")
    print(net)
    print("\nModel parameters (initialized):")
    for name, param in net.named_parameters():
        print(f"{name}:\n{param.data}")

    # Test save and load
    test_filename = "temp_model_test.pth"
    save_model(net, test_filename)
    
    net_loaded = SimpleNet()
    load_model(net_loaded, test_filename)
    
    print("\nLoaded model parameters:")
    for name, param in net_loaded.named_parameters():
        print(f"{name}:\n{param.data}")
        
    # Clean up
    if os.path.exists(test_filename):
        os.remove(test_filename)
    print(f"\nCleaned up {test_filename}")