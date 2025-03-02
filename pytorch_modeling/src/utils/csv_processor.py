import os
import yaml
import pandas as pd
from typing import Dict
from datetime import datetime

def ensure_directory_exists(file_path: str) -> None:
    """
    Ensures that the directory for the given file_path exists.
    Creates it if it doesn't.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def load_yaml_config(file_path: str) -> Dict:
    """
    Loads a YAML config from file_path if it exists, otherwise returns an empty dict.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}

def save_yaml_config(data: Dict, file_path: str) -> None:
    """
    Saves a dictionary as a YAML file at file_path.
    """
    ensure_directory_exists(file_path)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def load_or_infer_column_definitions(
    df: pd.DataFrame,
    column_definition_path: str
) -> Dict[str, str]:
    """
    Loads column-to-encoder definitions from 'column_definition.yaml' if it exists.
    If it does not exist, infer based on df dtypes and create it.

    Inference rules:
      - Object / String columns → Onehotencoder
      - Numeric columns → StandardScaler

    Returns:
      A dictionary mapping column_name -> encoder_type
    """
    # 1. Try to load existing definitions
    column_definitions = load_yaml_config(column_definition_path)
    if column_definitions:
        print(f"Loaded existing column definitions from {column_definition_path}")
        return column_definitions
    
    # 2. If no file or empty definitions, infer from dtypes
    inferred_definitions = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            inferred_definitions[col] = 'StandardScaler'
        else:
            inferred_definitions[col] = 'Onehotencoder'
    # Save the inferred definitions to column_definition.yaml
    save_yaml_config(inferred_definitions, column_definition_path)
    print(f"No column_definition.yaml found. Inferred definitions saved to {column_definition_path}")
    return inferred_definitions

def init_preprocessing_structure() -> Dict:
    """
    Returns the initial structure for the main preprocessing config file.
    """
    return {
        'Onehotencoder': {},
        'Labelencoder': {},
        'StandardScaler': {}
    }

def generate_preprocessing_config(
    df: pd.DataFrame,
    column_definitions: Dict[str, str],
    preprocessing_config_path: str,
    force_update: bool = False
) -> None:
    """
    Generates or updates the YAML config file (preprocessing_config.yaml) with
    metadata for each column based on its assigned category.

    Args:
        df: Pandas DataFrame to analyze.
        column_definitions: Mapping from column name to encoder type.
        preprocessing_config_path: Destination path of the main config file.
        force_update: If True, overwrite existing entries.
    """
    # Load existing preprocessing config (if any)
    existing_config = load_yaml_config(preprocessing_config_path)

    # If empty or force_update is True, start fresh
    if not existing_config or force_update:
        config = init_preprocessing_structure()
    else:
        # Ensure required top-level keys exist
        config = existing_config
        for cat in ['Onehotencoder', 'Labelencoder', 'StandardScaler']:
            if cat not in config:
                config[cat] = {}

    # Populate config with metadata
    for col_name, col_type in column_definitions.items():
        # Skip if column doesn't exist in DataFrame
        if col_name not in df.columns:
            print(f"Column '{col_name}' not in CSV; skipping.")
            continue

        # Skip if invalid encoder type
        if col_type not in ['Onehotencoder', 'Labelencoder', 'StandardScaler']:
            print(f"Invalid encoder type '{col_type}' for column '{col_name}'; skipping.")
            continue

        # If already in config and not forcing update, skip
        if (col_name in config[col_type]) and not force_update:
            print(f"Column '{col_name}' already in '{col_type}' and force_update=False; skipping.")
            continue

        # Gather metadata
        if col_type in ['Onehotencoder', 'Labelencoder']:
            unique_vals = df[col_name].dropna().unique().tolist()
            # Ensure "UNKNOWN" is included
            if "UNKNOWN" not in unique_vals:
                unique_vals.append("UNKNOWN")
            unique_vals.sort()

            col_metadata = {
                'unique_values': unique_vals,
                'number_of_unique_classes': len(unique_vals),
                'date_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        elif col_type == 'StandardScaler':
            series = df[col_name].dropna()
            col_metadata = {
                'mean': float(series.mean()) if not series.empty else 0.0,
                'std': float(series.std()) if not series.empty else 0.0,
                'date_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        else:
            # In case there's some other category in the future
            col_metadata = {
                'date_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        # Update config
        config[col_type][col_name] = col_metadata
        print(f"Processed column '{col_name}' with encoder '{col_type}'.")

    # Save the updated preprocessing config
    save_yaml_config(config, preprocessing_config_path)
    print(f"Preprocessing config saved at: {preprocessing_config_path}")

def main():
    """
    Full pipeline:
      1) Read CSV
      2) Load or infer column definitions (column_definition.yaml)
      3) Generate preprocessing config (preprocessing_config.yaml)
    """
    # Example paths (modify accordingly)
    PROJECT_PATH = os.environ.get('TRADE_SMART_PROJECT_PATH', '/path/to/project')
    csv_path = os.path.join(PROJECT_PATH, 'data', 'raw', 'features.csv')
    column_definition_path = os.path.join(PROJECT_PATH, 'src', 'config', 'column_definition.yaml')
    preprocessing_config_path = os.path.join(PROJECT_PATH, 'src', 'config', 'preprocessing_config.yaml')

    # 1) Load CSV
    df = pd.read_csv(csv_path)

    # 2) Load (or infer) column definitions
    column_definitions = load_or_infer_column_definitions(df, column_definition_path)

    # 3) Generate preprocessing config
    generate_preprocessing_config(
        df=df,
        column_definitions=column_definitions,
        preprocessing_config_path=preprocessing_config_path,
        force_update=True  # set to True to overwrite existing metadata
    )

if __name__ == '__main__':
    main()
