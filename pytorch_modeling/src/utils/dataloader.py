import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict
import yaml
import datetime

# Change these as needed for your environment
CACHE_FILE_PATH = os.environ.get('TRADE_SMART_CACHE_FILE_PATH', '/tmp')
PROJECT_LOCATION = os.environ.get('TRADE_SMART_PROJECT_PATH', '/path/to/project')

# Paths for your config files
COLUMN_DEFINITION_PATH = os.path.join(PROJECT_LOCATION, 'src', 'config', 'column_definition.yaml')
PREPROCESSING_CONFIG_PATH = os.path.join(PROJECT_LOCATION, 'src', 'config', 'preprocessing_config.yaml')

# Where to write warnings
LOG_FILE_PATH = os.path.join(PROJECT_LOCATION, 'logs', 'preprocessing_warnings.log')

def _log_warning(message: str) -> None:
    """
    Print a warning to console and also append it to a log file.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp} WARNING] {message}"
    print(full_message)
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(full_message + "\n")

def load_config(path: str) -> Dict:
    """
    Load a YAML configuration from the specified path. Returns an empty dict if file not found.
    """
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}

class DataPreprocessor:
    """
    A class that:
      1) Reads column definitions from column_definition.yaml to determine which
         preprocessing method (LabelEncoder, Onehotencoder, StandardScaler) to apply.
      2) Uses preprocessing_config.yaml to reference known categories, handle unknown categories, etc.
    """
    def __init__(self):
        # Load user-specified column definitions
        # (keys = column names, values = 'Labelencoder'|'Onehotencoder'|'StandardScaler')
        self.column_definitions = load_config(COLUMN_DEFINITION_PATH)

        # Load the main preprocessing config for reference
        # e.g. known categories or mean/std from a prior run
        self.config = load_config(PREPROCESSING_CONFIG_PATH)

        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # For convenience, keep track of columns by type
        self.label_encode_cols: List[str] = []
        self.onehot_encode_cols: List[str] = []
        self.numeric_cols: List[str] = []
        
        # Cache file to store fitted encoders/scalers
        self.cache_file = os.path.join(CACHE_FILE_PATH, "preprocessor_state.pkl")

    def _determine_column_lists(self, df: pd.DataFrame) -> None:
        """
        Based on the column_definition.yaml, split columns into label_encode_cols, onehot_encode_cols, and numeric_cols.
        If a column is not in the DataFrame, skip it.
        """
        for col, encoder_type in self.column_definitions.items():
            if col not in df.columns:
                _log_warning(f"Column '{col}' is defined in column_definition.yaml but not found in the DataFrame.")
                continue

            if encoder_type == 'Labelencoder':
                self.label_encode_cols.append(col)
            elif encoder_type == 'Onehotencoder':
                self.onehot_encode_cols.append(col)
            elif encoder_type == 'StandardScaler':
                self.numeric_cols.append(col)
            else:
                _log_warning(f"Unknown encoder_type '{encoder_type}' for column '{col}'. Skipping.")

    def save_state(self) -> None:
        """
        Save fitted state (label_encoders, scalers) to disk.
        """
        state = {
            'label_encoders': self.label_encoders,
            'scalers': self.scalers,
            'label_encode_cols': self.label_encode_cols,
            'onehot_encode_cols': self.onehot_encode_cols,
            'numeric_cols': self.numeric_cols
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(state, f)
        print(f"Preprocessor state saved to {self.cache_file}")

    def load_state(self) -> None:
        """
        Load previously fitted state (label_encoders, scalers) from disk, if available.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                state = pickle.load(f)
            self.label_encoders = state['label_encoders']
            self.scalers = state['scalers']
            self.label_encode_cols = state['label_encode_cols']
            self.onehot_encode_cols = state['onehot_encode_cols']
            self.numeric_cols = state['numeric_cols']
            print(f"Preprocessor state loaded from {self.cache_file}")
        else:
            _log_warning("No preprocessor_state.pkl found; cannot load pre-existing state.")

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the encoders/scalers based on the DataFrame.
         1) Identify columns by type via column_definition.yaml
         2) Fit LabelEncoder for label-encoded columns
         3) Fit LabelEncoder for onehot-encoded columns (we'll expand them at transform-time)
         4) Fit StandardScaler for numeric columns
         5) Save state
        """
        # 1) Determine which columns apply for each category
        self._determine_column_lists(df)

        # 2) Fit label encoders for label_encode_cols
        for col in self.label_encode_cols:
            series = df[col].astype(str).dropna().unique().tolist()

            # We must ensure "UNKNOWN" is included to handle future unknown categories
            if "UNKNOWN" not in series:
                series.append("UNKNOWN")
            series = sorted(set(series))

            le = LabelEncoder()
            le.fit(series)
            self.label_encoders[col] = le

        # 3) Fit label encoders for onehot_encode_cols
        #    We'll do exactly the same approach, but at transform-time we use F.one_hot
        for col in self.onehot_encode_cols:
            series = df[col].astype(str).dropna().unique().tolist()
            if "UNKNOWN" not in series:
                series.append("UNKNOWN")
            series = sorted(set(series))

            le = LabelEncoder()
            le.fit(series)
            self.label_encoders[col] = le

        # 4) Fit standard scalers for numeric columns
        for col in self.numeric_cols:
            ss = StandardScaler()
            # If column is missing entirely or has no rows, skip
            if col not in df.columns or df[col].dropna().empty:
                _log_warning(f"Numeric column '{col}' is empty or missing; skipping StandardScaler fit.")
                continue
            ss.fit(df[[col]].values)  # Fit on single column
            self.scalers[col] = ss

        # 5) Save the entire state so we can reuse
        self.save_state()

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Transform the DataFrame into a PyTorch tensor:
          - Replace unknown categories with 'UNKNOWN' if they're not in preprocessing_config.yaml
          - Label-encode label_encode_cols
          - One-hot-encode onehot_encode_cols
          - Scale numeric_cols
          - Concatenate into a single PyTorch tensor
        """
        df_processed = df.copy()
        all_features = []

        # We expect the config to have top-level keys: 'Onehotencoder', 'Labelencoder', 'StandardScaler'
        # Each category has a dict of columns -> metadata (including unique_values)
        # We'll leverage that to detect unknown categories.

        # 1) Transform label-encoded columns
        for col in self.label_encode_cols:
            if col not in df_processed.columns:
                _log_warning(f"Column '{col}' missing at transform-time; filling with 'UNKNOWN'.")
                df_processed[col] = "UNKNOWN"

            # read the series
            series = df_processed[col].astype(str)

            # If the col is not in config or has no known categories, skip
            known_categories = set()
            if 'Labelencoder' in self.config and col in self.config['Labelencoder']:
                known_categories = set(self.config['Labelencoder'][col].get('unique_values', []))

            # Replace categories not in 'known_categories' with 'UNKNOWN'
            mask_unknown = ~series.isin(known_categories)
            if mask_unknown.any():
                for unk_val in series[mask_unknown].unique():
                    _log_warning(f"Column '{col}' has unknown category '{unk_val}'; substituting 'UNKNOWN'.")
                series[mask_unknown] = "UNKNOWN"

            # Now transform with the fitted LabelEncoder
            # but first ensure that 'UNKNOWN' is in the label encoder classes
            if 'UNKNOWN' not in self.label_encoders[col].classes_:
                # This can happen if the config was re-generated.
                # You can decide how to handle it: re-fit or raise an error.
                # Here we'll re-fit quickly to add 'UNKNOWN':
                all_classes = list(self.label_encoders[col].classes_)
                all_classes.append('UNKNOWN')
                self.label_encoders[col].fit(sorted(set(all_classes)))

            encoded_vals = self.label_encoders[col].transform(series)
            tensor_encoded = torch.tensor(encoded_vals, dtype=torch.long).unsqueeze(1)
            all_features.append(tensor_encoded)

        # 2) Transform one-hot-encoded columns
        for col in self.onehot_encode_cols:
            if col not in df_processed.columns:
                _log_warning(f"Column '{col}' missing at transform-time; filling with 'UNKNOWN'.")
                df_processed[col] = "UNKNOWN"

            series = df_processed[col].astype(str)

            known_categories = set()
            if 'Onehotencoder' in self.config and col in self.config['Onehotencoder']:
                known_categories = set(self.config['Onehotencoder'][col].get('unique_values', []))

            mask_unknown = ~series.isin(known_categories)
            if mask_unknown.any():
                for unk_val in series[mask_unknown].unique():
                    _log_warning(f"Column '{col}' has unknown category '{unk_val}'; substituting 'UNKNOWN'.")
                series[mask_unknown] = "UNKNOWN"

            # Ensure 'UNKNOWN' in label encoder classes
            if 'UNKNOWN' not in self.label_encoders[col].classes_:
                all_classes = list(self.label_encoders[col].classes_)
                all_classes.append('UNKNOWN')
                self.label_encoders[col].fit(sorted(set(all_classes)))

            label_encoded = self.label_encoders[col].transform(series)

            # one-hot dimension = number_of_unique_classes from config
            if 'Onehotencoder' in self.config and col in self.config['Onehotencoder']:
                num_classes = self.config['Onehotencoder'][col].get('number_of_unique_classes', 0)
            else:
                # Fallback if missing in config
                num_classes = len(self.label_encoders[col].classes_)

            tensor_encoded = torch.tensor(label_encoded, dtype=torch.long)
            one_hot = F.one_hot(tensor_encoded, num_classes=num_classes)
            all_features.append(one_hot)

        # 3) Transform numeric columns
        for col in self.numeric_cols:
            if col not in df_processed.columns:
                # If missing, fill with zero (or some default)
                _log_warning(f"Numeric column '{col}' missing at transform-time; filling with 0.")
                df_processed[col] = 0

            values = df_processed[col].values.reshape(-1, 1).astype(float)
            if col in self.scalers:
                scaled_vals = self.scalers[col].transform(values)
            else:
                # If we never fit a scaler for this col, just pass values through
                _log_warning(f"No fitted scaler found for numeric column '{col}'; using raw values.")
                scaled_vals = values

            numerical_tensor = torch.tensor(scaled_vals, dtype=torch.float32)
            all_features.append(numerical_tensor)

        # 4) Concatenate all features into a single tensor
        if not all_features:
            _log_warning("No features found to transform! Returning empty tensor.")
            return torch.empty((len(df_processed), 0))

        return torch.cat(all_features, dim=1)

    def fit_transform(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Convenience method to do both fit and transform in one call.
        """
        self.fit(df)
        return self.transform(df)

# Example usage
if __name__ == "__main__":
    # Suppose we have a DataFrame to fit/transform
    df = pd.read_csv(r"C:\programming\trade_smart\pytorch_modeling\data\raw\features.csv")

    preprocessor = DataPreprocessor()
    preprocessor.fit(df)          # Fit all encoders/scalers
    transformed_tensor = preprocessor.transform(df)  # Transform into a PyTorch tensor
    print("Transformed tensor shape:", transformed_tensor.shape)
