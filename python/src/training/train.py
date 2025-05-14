"""
Training script for house price prediction model.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import pickle

# Add project root to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.data_utils import load_housing_data, preprocess_data, create_synthetic_housing_data
from src.training.model import HousePriceModel


def train_model(X_train, y_train, input_dim, batch_size=32, epochs=100, 
                lr=0.001, hidden_dims=[64, 32], val_data=None):
    """
    Train the house price prediction model.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training target
    input_dim : int
        Number of input features
    batch_size : int
        Batch size (default: 32)
    epochs : int
        Number of training epochs (default: 100)
    lr : float
        Learning rate (default: 0.001)
    hidden_dims : list
        List of hidden layer dimensions (default: [64, 32])
    val_data : tuple or None
        Validation data as (X_val, y_val) (default: None)
        
    Returns:
    --------
    tuple
        (trained model, training history)
    """
    # Convert numpy arrays to torch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create validation dataloader if validation data is provided
    if val_data is not None:
        X_val, y_val = val_data
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model, loss function, and optimizer
    model = HousePriceModel(input_dim, hidden_dims)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [] if val_data is not None else None
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        if val_data is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            print('Epoch {}/{} - Loss: {:.4f} - Val Loss: {:.4f}'.format(
                epoch+1, epochs, avg_train_loss, avg_val_loss))
        else:
            print('Epoch {}/{} - Loss: {:.4f}'.format(
                epoch+1, epochs, avg_train_loss))
    
    return model, history


def export_to_onnx(model, input_dim, output_path, metadata=None):
    """
    Export PyTorch model to ONNX format.
    
    Parameters:
    -----------
    model : HousePriceModel
        Trained PyTorch model
    input_dim : int
        Number of input features
    output_path : str
        Path to save the ONNX model
    metadata : dict or None
        Additional metadata to save with the model (default: None)
    """
    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, input_dim)
    
    # Set model to evaluation mode
    model.eval()
    
    # Export to ONNX format
    torch.onnx.export(model,                  # model being run
                      dummy_input,            # model input
                      output_path,            # where to save the model
                      export_params=True,     # store the trained parameter weights
                      opset_version=12,       # the ONNX version
                      do_constant_folding=True,  # whether to optimize
                      input_names=['input'],  # input names
                      output_names=['output'],  # output names
                      dynamic_axes={'input': {0: 'batch_size'},  # dynamic batch size
                                   'output': {0: 'batch_size'}})
    
    print("Model exported to ONNX format at: {}".format(output_path))
    
    # Save additional metadata if provided
    if metadata:
        metadata_path = output_path.replace('.onnx', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print("Model metadata saved at: {}".format(metadata_path))


def main():
    """
    Main function to run the training process.
    """
    parser = argparse.ArgumentParser(description='Train a house price prediction model.')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to housing data CSV file.')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data instead of loading from file.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--hidden_dims', type=str, default='64,32',
                        help='Comma-separated list of hidden layer dimensions.')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save the model.')
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data or create synthetic data
    if args.synthetic:
        print("Using synthetic housing data")
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                 'data', 'synthetic_housing_data.csv')
        df = create_synthetic_housing_data(n_samples=10000, output_path=data_path)
    elif args.data_path:
        print("Loading housing data from: {}".format(args.data_path))
        df = load_housing_data(args.data_path)
    else:
        print("No data provided. Using synthetic data.")
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                 'data', 'synthetic_housing_data.csv')
        df = create_synthetic_housing_data(n_samples=10000, output_path=data_path)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_names = preprocess_data(df)
    
    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    print("Training with hidden dimensions: {}".format(hidden_dims))
    print("Input dimension: {}".format(X_train.shape[1]))
    
    # Train the model
    model, history = train_model(
        X_train, y_train,
        input_dim=X_train.shape[1],
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dims=hidden_dims,
        val_data=(X_test, y_test)
    )
    
    # Save the PyTorch model
    pytorch_model_path = os.path.join(model_dir, 'house_price_model.pth')
    model.save_model(pytorch_model_path)
    print("PyTorch model saved at: {}".format(pytorch_model_path))
    
    # Save the scalers
    with open(os.path.join(model_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump({'X': scaler_X, 'y': scaler_y, 'feature_names': feature_names}, f)
    
    # Export to ONNX
    onnx_model_path = os.path.join(model_dir, 'house_price_model.onnx')
    
    # Metadata to help with inference in Rust
    metadata = {
        'feature_names': feature_names,
        'input_dim': X_train.shape[1],
        'hidden_dims': hidden_dims
    }
    
    export_to_onnx(model, X_train.shape[1], onnx_model_path, metadata)


if __name__ == '__main__':
    main()
