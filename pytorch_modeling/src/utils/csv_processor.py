import pandas as pd
import yaml
import os
from typing import Dict, List
from datetime import datetime

def ensure_config_exists(config_path: str) -> None:
    """
    Ensure config file exists, create with default structure if it doesn't
    """
    if not os.path.exists(config_path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Default config structure
        default_config = {
            'categorical_features': {
                'symbol': {'num_classes': 5},
                'market_type': {'num_classes': 2},
                'trade_type': {'num_classes': 3},
                'sector': {'num_classes': 4}
            },
            'numerical_features': [
                'price',
                'volume',
                'volatility',
                'market_cap'
            ]
        }
        
        # Save default config
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        print(f"Created default config file at: {config_path}")

def extract_categorical_features(
    csv_path: str,
    config_path: str,
    output_dir: str,
    force_update: bool = False
) -> None:
    """
    Extract unique values for each categorical feature and save to separate YAML files
    
    Args:
        csv_path: Path to the features CSV file
        config_path: Path to the preprocessing config YAML
        output_dir: Directory to save feature YAML files
        force_update: If True, overwrites existing YAML files
    """
    # Ensure config file exists
    ensure_config_exists(config_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get categorical columns from config
    categorical_cols = list(config['categorical_features'].keys())
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Process each categorical column
    for col in categorical_cols:
        output_file = os.path.join(output_dir, f'{col}_features.yml')
        
        # Check if file exists and force_update is False
        if os.path.exists(output_file) and not force_update:
            print(f"Skipping {col}: Feature file already exists")
            continue
            
        # Get unique values
        unique_values = df[col].unique().tolist()
        unique_values.sort()  # Sort for consistency
        
        # Create feature info
        feature_info = {
            'name': col,
            'unique_values': unique_values,
            'count': len(unique_values),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to YAML
        with open(output_file, 'w') as f:
            yaml.dump(feature_info, f, default_flow_style=False)
        
        print(f"Processed {col}: Found {len(unique_values)} unique values")

if __name__ == "__main__":
    # Example usage
    PROJECT_PATH = os.environ['TRADE_SMART_PROJECT_PATH']
    
    csv_path = os.path.join(PROJECT_PATH, 'data', 'raw', 'features.csv')
    config_path = os.path.join(PROJECT_PATH, 'src', 'config', 'preprocessing_config.yml')
    output_dir = os.path.join(PROJECT_PATH, 'src', 'config', 'features')
    
    # Set force_update=True to regenerate all feature files
    extract_categorical_features(csv_path, config_path, output_dir, force_update=False)