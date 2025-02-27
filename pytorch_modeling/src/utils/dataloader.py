import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict
import yaml

CACHE_FILE_PATH = os.environ['TRADE_SMART_CACHE_FILE_PATH']
PROJECT_LOCATION = os.environ['TRADE_SMART_PROJECT_PATH']
CONFIG_PATH = os.path.join(PROJECT_LOCATION, 'src', 'config', 'preprocessing_config.yml')

def load_config():
    """Load preprocessing configuration from YAML file"""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

class DataPreprocessor:
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.cache_file = "preprocessor_state.pkl"
        self.config = load_config()
        
    def fit(self, df: pd.DataFrame, categorical_cols: List[str], numerical_cols: List[str]):
        """
        Fit the preprocessor using one-hot encoding for categorical variables
        """
        # Fit label encoders first (we need this for initial encoding)
        for col in categorical_cols:
            if col not in self.config["categorical_features"]:
                raise ValueError(f"Column {col} not found in config")
                
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(df[col].astype(str))
            
            # Verify the number of classes matches config
            num_classes = self.config["categorical_features"][col]["num_classes"]
            if len(self.label_encoders[col].classes_) > num_classes:
                raise ValueError(f"More classes found in {col} than specified in config")
        
        # Fit scaler for numerical columns
        self.scaler.fit(df[numerical_cols])
        
        # Save state
        self.save_state()
    
    def transform(self, df: pd.DataFrame, categorical_cols: List[str], numerical_cols: List[str]) -> torch.Tensor:
        """
        Transform the data using one-hot encoding for categorical variables
        """
        df_processed = df.copy()
        all_features = []
        
        # Transform categorical columns to one-hot
        for col in categorical_cols:
            # First apply label encoding
            label_encoded = self.label_encoders[col].transform(df_processed[col].astype(str))
            
            # Convert to tensor and apply one-hot encoding
            tensor_encoded = torch.tensor(label_encoded, dtype=torch.long)
            num_classes = self.config["categorical_features"][col]["num_classes"]
            one_hot = F.one_hot(tensor_encoded, num_classes=num_classes)
            
            all_features.append(one_hot)
        
        # Transform numerical
        numerical_data = self.scaler.transform(df_processed[numerical_cols])
        numerical_tensor = torch.tensor(numerical_data, dtype=torch.float32)
        all_features.append(numerical_tensor)
        
        # Concatenate all features
        return torch.cat(all_features, dim=1)