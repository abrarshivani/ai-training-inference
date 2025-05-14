"""
Utility functions for loading and preprocessing house price data.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_housing_data(data_path):
    """
    Load housing data from a CSV file.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing housing data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the housing data
    """
    return pd.read_csv(data_path)

def preprocess_data(df, target_col='price', test_size=0.2, random_state=42):
    """
    Preprocess housing data for model training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the housing data
    target_col : str
        Name of the target column (default: 'price')
    test_size : float
        Fraction of data to use for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, scaler_X, scaler_y)
    """
    # Handle missing values
    df = df.dropna()
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[[target_col]]
    
    # Convert categorical variables to one-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Save feature names for later use with the model
    feature_names = X.columns.tolist()
    
    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, 
            scaler_X, scaler_y, feature_names)

def create_synthetic_housing_data(n_samples=1000, output_path=None):
    """
    Create synthetic housing data for demonstration purposes.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate (default: 1000)
    output_path : str or None
        If provided, save the data to this path (default: None)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic housing data
    """
    np.random.seed(42)
    
    # Generate synthetic features
    sqft = np.random.randint(500, 5000, size=n_samples)
    bedrooms = np.random.randint(1, 6, size=n_samples)
    bathrooms = np.random.randint(1, 5, size=n_samples)
    age = np.random.randint(0, 100, size=n_samples)
    
    # Location as categorical feature
    locations = ['urban', 'suburban', 'rural']
    location = np.random.choice(locations, size=n_samples)
    
    # Generate price based on features with some noise
    price = (
        100 * sqft + 
        15000 * bedrooms + 
        20000 * bathrooms - 
        1000 * age +
        np.where(location == 'urban', 50000, 
                np.where(location == 'suburban', 25000, 0))
    )
    
    # Add some noise to the price
    price = price + np.random.normal(0, 25000, size=n_samples)
    price = np.maximum(50000, price)  # Ensure minimum price
    
    # Create DataFrame
    data = pd.DataFrame({
        'sqft': sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'location': location,
        'price': price
    })
    
    # Save data if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
    
    return data