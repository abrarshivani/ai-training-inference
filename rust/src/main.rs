use anyhow::{Context, Result};
use clap::Parser;
use ndarray::{Array, Axis, IxDyn};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use tract_onnx::prelude::*;

/// House price prediction using ONNX model
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the ONNX model file
    #[arg(short, long)]
    model: PathBuf,

    /// Path to the model metadata JSON file
    #[arg(short = 'd', long)]
    metadata: PathBuf,

    /// Input features as a comma-separated list (order matters)
    #[arg(short, long)]
    input: String,

    /// Input feature names as a comma-separated list (optional)
    #[arg(long)]
    feature_names: Option<String>,
}

/// Model metadata
#[derive(Debug, Deserialize, Serialize)]
struct ModelMetadata {
    feature_names: Vec<String>,
    input_dim: usize,
    hidden_dims: Vec<usize>,
}

/// House Feature Input with standardized values
#[derive(Debug)]
struct HouseFeatures {
    values: Vec<f32>,
}

impl HouseFeatures {
    fn new(values: Vec<f32>) -> Self {
        HouseFeatures { values }
    }
}

/// House Price Model for inference
struct HousePriceModel {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    metadata: ModelMetadata,
}

impl HousePriceModel {
    /// Load a model from ONNX file
    fn load(model_path: &Path, metadata_path: &Path) -> Result<Self> {
        // Load metadata
        let metadata_file = File::open(metadata_path)
            .with_context(|| format!("Failed to open metadata file: {:?}", metadata_path))?;
        let metadata: ModelMetadata = serde_json::from_reader(BufReader::new(metadata_file))
            .with_context(|| format!("Failed to parse metadata file: {:?}", metadata_path))?;

        // Load model
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .with_context(|| format!("Failed to load ONNX model: {:?}", model_path))?
            .into_optimized()?
            .into_runnable()?;

        Ok(HousePriceModel { model, metadata })
    }

    /// Run inference on input features
    fn predict(&self, features: &HouseFeatures) -> Result<f32> {
        // Check input dimension
        if features.values.len() != self.metadata.input_dim {
            return Err(anyhow::anyhow!(
                "Input dimension mismatch: expected {}, got {}",
                self.metadata.input_dim,
                features.values.len()
            ));
        }

        // Create input tensor
        let input_shape = [1, features.values.len()];
        let input_tensor =
            Array::from_shape_vec(input_shape, features.values.iter().copied().collect())?
                .into_dyn();

        // Run inference
        let result = self.model.run(tvec!(input_tensor.into_tensor().into()))?;

        // Get output value
        let output_tensor = result[0]
            .to_array_view::<f32>()?
            .into_dimensionality::<IxDyn>()?;

        // Extract the single prediction value
        let prediction = output_tensor
            .index_axis(Axis(0), 0)
            .first()
            .copied()
            .unwrap();

        Ok(prediction)
    }
}

fn parse_input_features(input: &str, feature_names: &[String]) -> Result<HouseFeatures> {
    let values: Vec<f32> = input
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f32>()
                .map_err(|_| anyhow::anyhow!("Failed to parse input value: {}", s))
        })
        .collect::<Result<Vec<f32>>>()?;

    // Validate input length
    if !feature_names.is_empty() && values.len() != feature_names.len() {
        return Err(anyhow::anyhow!(
            "Number of input values ({}) does not match number of features ({})",
            values.len(),
            feature_names.len()
        ));
    }

    Ok(HouseFeatures::new(values))
}

fn main() -> Result<()> {
    // Parse command-line arguments
    let args = Args::parse();

    // Attempt to find metadata file if not specified
    let metadata_path = if args.metadata.exists() {
        args.metadata.clone()
    } else {
        let default_metadata_path = args
            .model
            .with_extension("")
            .with_extension("_metadata.json");
        if default_metadata_path.exists() {
            default_metadata_path
        } else {
            return Err(anyhow::anyhow!(
                "Metadata file not found: {:?}",
                args.metadata
            ));
        }
    };

    println!("Loading model from: {:?}", args.model);
    println!("Loading metadata from: {:?}", metadata_path);

    // Load model
    let model = HousePriceModel::load(&args.model, &metadata_path)?;

    // Display model information
    println!("Model loaded successfully");
    println!("Input dimension: {}", model.metadata.input_dim);
    println!("Feature names: {:?}", model.metadata.feature_names);

    // Parse input features
    let features = parse_input_features(&args.input, &model.metadata.feature_names)?;

    // Display input
    println!("Input features: {:?}", features.values);

    // Perform prediction
    let prediction = model.predict(&features)?;

    println!("Predicted house price: {}", prediction);

    Ok(())
}
