import pandas as pd
import numpy as np
import os

def generate_sample_features(
    output_path: str,
    num_rows: int = 1000
) -> None:
    """
    Generate sample features CSV file with both categorical and numerical columns
    
    Args:
        output_path: Path to save the CSV file
        num_rows: Number of rows to generate
    """
    # Create sample data
    data = {
        # Categorical features
        'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'], num_rows),
        'market_type': np.random.choice(['NYSE', 'NASDAQ'], num_rows),
        'trade_type': np.random.choice(['BUY', 'SELL', 'HOLD'], num_rows),
        'sector': np.random.choice(['TECH', 'FINANCE', 'HEALTH', 'ENERGY'], num_rows),
        
        # Numerical features
        'price': np.random.uniform(10, 1000, num_rows),
        'volume': np.random.uniform(1000, 1000000, num_rows),
        'volatility': np.random.uniform(0.01, 0.5, num_rows),
        'market_cap': np.random.uniform(1e6, 1e12, num_rows)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated sample features file at: {output_path}")
    
    # Print summary
    print("\nDataset Summary:")
    print(f"Total rows: {len(df)}")
    print("\nUnique values per categorical column:")
    categorical_cols = ['symbol', 'market_type', 'trade_type', 'sector']
    for col in categorical_cols:
        print(f"{col}: {df[col].nunique()} unique values")

if __name__ == "__main__":
    PROJECT_PATH = os.environ['TRADE_SMART_PROJECT_PATH']
    output_path = os.path.join(PROJECT_PATH, 'data', 'raw', 'features.csv')
    
    generate_sample_features(output_path)