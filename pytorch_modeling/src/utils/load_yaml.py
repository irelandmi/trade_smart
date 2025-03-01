import os
import pandas as pd

import pickle

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
    
config = load_config()
print(config)
print(type(config))

for i in config["categorical_features"]:
    print(i)