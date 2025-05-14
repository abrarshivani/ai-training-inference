"""
PyTorch model for house price prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HousePriceModel(nn.Module):
    """
    Neural network model for predicting house prices.
    Uses multiple fully connected layers with ReLU activations.
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        """
        Initialize the model with specified layer dimensions.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dims : list
            List of hidden layer dimensions (default: [64, 32])
        """
        super(HousePriceModel, self).__init__()
        
        # Build network layers dynamically based on hidden_dims
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # Final output layer (regression)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [batch_size, input_dim]
        
        Returns:
        --------
        torch.Tensor
            Output tensor of shape [batch_size, 1]
        """
        return self.model(x)
    
    def save_model(self, path):
        """
        Save the model state to a file.
        
        Parameters:
        -----------
        path : str
            Path to save the model
        """
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load_model(cls, path, input_dim, hidden_dims=[64, 32]):
        """
        Load a model from file.
        
        Parameters:
        -----------
        path : str
            Path to the saved model
        input_dim : int
            Number of input features
        hidden_dims : list
            List of hidden layer dimensions
            
        Returns:
        --------
        HousePriceModel
            Loaded model
        """
        model = cls(input_dim, hidden_dims)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model