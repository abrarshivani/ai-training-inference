# House Price Prediction with PyTorch and Rust

This project demonstrates how to create a machine learning pipeline for house price prediction using PyTorch for model training and Rust for inference. The workflow includes:

1. Training a neural network model in PyTorch
2. Exporting the trained model to ONNX format
3. Loading and running inference on the model using Rust

## Project Structure

```
ai-training-inference/
├── python/            # PyTorch model training code
│   ├── data/          # Housing datasets
│   ├── models/        # Saved models
│   ├── src/           # Source code
│   │   ├── training/       # Model training code
│   │   └── utils/          # Utility functions
│   └── requirements.txt    # Python dependencies
├── rust/             # Rust inference code
│   ├── src/          # Rust source code
│   └── Cargo.toml    # Rust dependencies
```

## Setup

### Python Setup

1. Create a Python virtual environment (optional but recommended):

```bash
cd python
python -m venv venv
source venv/bin/activate 
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Rust Setup

1. Install Rust if you haven't already:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Build the Rust project:

```bash
cd rust
cargo build --release
```

## Training the Model

The project supports both using real housing data or generating synthetic data for demonstration purposes.

### Using Synthetic Data

```bash
cd python
python -m src.training.train --synthetic --epochs 50
```

### Using Your Own Data

```bash
cd python
python -m src.training.train --data_path /path/to/your/housing_data.csv --epochs 100
```

This will:
1. Load and preprocess the housing data
2. Train a neural network model
3. Save the PyTorch model to `models/house_price_model.pth`
4. Export the model to ONNX format in `models/house_price_model.onnx`
5. Save model metadata in `models/house_price_model_metadata.json`

## Running Inference with Rust

After training the model, you can run inference using the Rust application:

```bash
cd rust
cargo run --release -- \
  --model ../python/models/house_price_model.onnx \
  --metadata ../python/models/house_price_model_metadata.json \
  --input "1500,3,2,15,1,0"
```

Replace the input values with your own housing features.

## Features

- **PyTorch Model**: Multi-layer neural network for regression
- **Data Processing**: Handles categorical variables and scaling
- **ONNX Export**: Platform-independent model format
- **Rust Inference**: Fast, safe inference in production
- **CLI Interface**: Easy-to-use command line tools

## Dependencies

### Python
- PyTorch
- NumPy
- Pandas
- scikit-learn
- ONNX
- ONNXRuntime

### Rust
- tract-onnx (ONNX runtime)
- ndarray (N-dimensional arrays)
- clap (Command line parsing)
- serde (Serialization/Deserialization)

## License

MIT