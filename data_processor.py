import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import os
import crypto  # Import our cryptocurrency data collection module

def load_and_preprocess_data(crypto_id, days=365, output_dir='processed_data'):
    """
    Load cryptocurrency data and preprocess it for ML models
    
    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., 'bitcoin')
        days (int): Number of days of historical data to collect
        output_dir (str): Directory to save processed data files
        
    Returns:
        dict: Dictionary containing processed datasets and preprocessing objects
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Collecting data for {crypto_id}...")
    # Collect comprehensive data
    raw_data = crypto.collect_comprehensive_data(crypto_id, days)
    
    if raw_data is None or raw_data.get('price_data') is None:
        print(f"Failed to collect data for {crypto_id}")
        return None
    
    # Extract price data with technical indicators
    df = raw_data['price_data']
    
    # 1. Handle missing values
    df = handle_missing_values(df)
    
    # 2. Feature engineering
    df = engineer_features(df)
    
    # 3. Create target variables (future price changes)
    df = create_target_variables(df)
    
    # 4. Split data into features and targets
    features, targets = extract_features_and_targets(df)
    
    # 5. Scale/normalize features
    features_scaled, scalers = scale_features(features)
    
    # 6. Split into training and testing sets
    X_train, X_test, y_train, y_test = split_data(features_scaled, targets)
    
    # Save processed data
    save_processed_data(
        output_dir, 
        crypto_id, 
        {
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test
        },
        scalers
    )
    
    # Return the processed data and preprocessing objects
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scalers': scalers,
        'feature_names': features.columns.tolist(),
        'full_data': df,
        'raw_data': raw_data
    }

def handle_missing_values(df):
    """
    Handle missing values in the dataframe
    
    Args:
        df (DataFrame): DataFrame to process
        
    Returns:
        DataFrame: DataFrame with missing values handled
    """
    print("Handling missing values...")
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Drop rows with missing timestamps (critical)
    df_clean.dropna(subset=['timestamp'], inplace=True)
    
    # For technical indicators and other features, use imputation
    # First separate datetime column
    datetime_col = df_clean['timestamp']
    df_numeric = df_clean.drop('timestamp', axis=1)
    
    # Use median imputation for numeric data
    imputer = SimpleImputer(strategy='median')
    df_numeric_imputed = pd.DataFrame(
        imputer.fit_transform(df_numeric),
        columns=df_numeric.columns
    )
    
    # Reattach datetime column
    df_clean = pd.concat([datetime_col.reset_index(drop=True), 
                          df_numeric_imputed.reset_index(drop=True)], axis=1)
    
    return df_clean

def engineer_features(df):
    """
    Engineer additional features from existing data
    
    Args:
        df (DataFrame): DataFrame to process
        
    Returns:
        DataFrame: DataFrame with additional engineered features
    """
    print("Engineering additional features...")
    
    # Make a copy to avoid modifying the original
    df_featured = df.copy()
    
    # Add day of week (0=Monday, 6=Sunday)
    df_featured['day_of_week'] = df_featured['timestamp'].dt.dayofweek
    
    # Add month
    df_featured['month'] = df_featured['timestamp'].dt.month
    
    # Add quarter
    df_featured['quarter'] = df_featured['timestamp'].dt.quarter
    
    # Add is_weekend flag
    df_featured['is_weekend'] = df_featured['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Add price distance from moving averages (%)
    # Percentage difference from 7-day MA
    if 'price_ma7' in df_featured.columns:
        df_featured['price_ma7_diff'] = ((df_featured['price'] - df_featured['price_ma7']) 
                                         / df_featured['price_ma7'] * 100)
    
    # Percentage difference from 20-day MA
    if 'ma20' in df_featured.columns:
        df_featured['ma20_diff'] = ((df_featured['price'] - df_featured['ma20']) 
                                    / df_featured['ma20'] * 100)
    
    # Bollinger Band position (%)
    if all(col in df_featured.columns for col in ['bollinger_upper', 'bollinger_lower']):
        bb_range = df_featured['bollinger_upper'] - df_featured['bollinger_lower']
        df_featured['bb_position'] = ((df_featured['price'] - df_featured['bollinger_lower']) 
                                      / bb_range * 100)
    
    # Add lagged features (t-1, t-3, t-7)
    price_lags = [1, 3, 7]
    for lag in price_lags:
        df_featured[f'price_lag_{lag}'] = df_featured['price'].shift(lag)
        df_featured[f'price_change_pct_lag_{lag}'] = df_featured['price_change_pct'].shift(lag)
        
        if 'volume' in df_featured.columns:
            df_featured[f'volume_lag_{lag}'] = df_featured['volume'].shift(lag)
    
    # Add rolling statistics
    df_featured['price_rolling_mean_7'] = df_featured['price'].rolling(window=7).mean()
    df_featured['price_rolling_std_7'] = df_featured['price'].rolling(window=7).std()
    df_featured['price_rolling_min_7'] = df_featured['price'].rolling(window=7).min()
    df_featured['price_rolling_max_7'] = df_featured['price'].rolling(window=7).max()
    
    return df_featured

def create_target_variables(df):
    """
    Create target variables for ML prediction
    
    Args:
        df (DataFrame): DataFrame to process
        
    Returns:
        DataFrame: DataFrame with target variables added
    """
    print("Creating target variables...")
    
    # Make a copy to avoid modifying the original
    df_targets = df.copy()
    
    # Next day price (t+1)
    df_targets['next_day_price'] = df_targets['price'].shift(-1)
    
    # Price change next day (%)
    df_targets['price_change_next_day_pct'] = ((df_targets['next_day_price'] - df_targets['price']) 
                                              / df_targets['price'] * 100)
    
    # Price changes for different time horizons (%)
    prediction_horizons = [3, 7, 14, 30]
    for horizon in prediction_horizons:
        future_price = df_targets['price'].shift(-horizon)
        df_targets[f'price_change_{horizon}d_pct'] = ((future_price - df_targets['price']) 
                                                     / df_targets['price'] * 100)
    
    # Binary targets (price up/down)
    df_targets['price_up_next_day'] = (df_targets['price_change_next_day_pct'] > 0).astype(int)
    
    for horizon in prediction_horizons:
        df_targets[f'price_up_{horizon}d'] = (df_targets[f'price_change_{horizon}d_pct'] > 0).astype(int)
    
    return df_targets

def extract_features_and_targets(df):
    """
    Extract features and targets from the processed dataframe
    
    Args:
        df (DataFrame): Processed DataFrame
        
    Returns:
        tuple: (features_df, targets_df)
    """
    print("Extracting features and targets...")
    
    # Drop rows with NaN in target variables
    df_valid = df.dropna(subset=['price_change_next_day_pct', 'price_up_next_day'])
    
    # Define target columns
    target_columns = ['price_change_next_day_pct', 'price_up_next_day', 
                      'price_change_3d_pct', 'price_up_3d',
                      'price_change_7d_pct', 'price_up_7d',
                      'price_change_14d_pct', 'price_up_14d',
                      'price_change_30d_pct', 'price_up_30d']
    
    # Drop non-feature columns and target columns to get features
    exclude_columns = target_columns + ['timestamp', 'next_day_price']
    feature_columns = [col for col in df_valid.columns if col not in exclude_columns]
    
    # Extract features and targets
    features = df_valid[feature_columns]
    targets = df_valid[target_columns]
    
    return features, targets

def scale_features(features):
    """
    Scale features using normalization or standardization
    
    Args:
        features (DataFrame): Features DataFrame
        
    Returns:
        tuple: (scaled_features_df, scalers_dict)
    """
    print("Scaling features...")
    
    # Initialize empty DataFrame for scaled features
    scaled_features = pd.DataFrame(index=features.index)
    
    # Dictionary to store scalers
    scalers = {}
    
    # For price-related features, use MinMaxScaler (0-1 range)
    price_columns = [col for col in features.columns if 'price' in col or 'ma' in col or 
                     'bollinger' in col or 'market_cap' in col]
    
    if price_columns:
        price_scaler = MinMaxScaler()
        scaled_features[price_columns] = price_scaler.fit_transform(features[price_columns])
        scalers['price_scaler'] = price_scaler
    
    # For indicators, use StandardScaler (mean=0, std=1)
    indicator_columns = [col for col in features.columns if col not in price_columns and 
                         col not in ['day_of_week', 'month', 'quarter', 'is_weekend']]
    
    if indicator_columns:
        indicator_scaler = StandardScaler()
        scaled_features[indicator_columns] = indicator_scaler.fit_transform(features[indicator_columns])
        scalers['indicator_scaler'] = indicator_scaler
    
    # For categorical features, just copy them (no scaling needed)
    categorical_columns = ['day_of_week', 'month', 'quarter', 'is_weekend']
    categorical_columns = [col for col in categorical_columns if col in features.columns]
    
    if categorical_columns:
        scaled_features[categorical_columns] = features[categorical_columns]
    
    return scaled_features, scalers

def split_data(features, targets, test_size=0.2, random_state=42):
    """
    Split data into training and test sets
    
    Args:
        features (DataFrame): Features DataFrame
        targets (DataFrame): Targets DataFrame
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("Splitting data into train and test sets...")
    
    # First, do a chronological split (cryptocurrency data is time series)
    split_idx = int(len(features) * (1 - test_size))
    
    # Training data (earlier dates)
    X_train = features.iloc[:split_idx]
    y_train = targets.iloc[:split_idx]
    
    # Test data (later dates)
    X_test = features.iloc[split_idx:]
    y_test = targets.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test

def save_processed_data(output_dir, crypto_id, data_dict, scalers):
    """
    Save processed data to files
    
    Args:
        output_dir (str): Output directory
        crypto_id (str): Cryptocurrency ID
        data_dict (dict): Dictionary of DataFrames to save
        scalers (dict): Dictionary of scalers to save
    """
    print(f"Saving processed data to {output_dir}...")
    
    # Create subdirectory for this cryptocurrency
    crypto_dir = os.path.join(output_dir, crypto_id)
    if not os.path.exists(crypto_dir):
        os.makedirs(crypto_dir)
    
    # Save each DataFrame
    for name, df in data_dict.items():
        file_path = os.path.join(crypto_dir, f"{name}.csv")
        df.to_csv(file_path, index=False)
    
    # Save scalers
    scalers_path = os.path.join(crypto_dir, "scalers.joblib")
    joblib.dump(scalers, scalers_path)
    
    print(f"Data processing complete. Files saved in {crypto_dir}")

if __name__ == "__main__":
    # Process data for Bitcoin and Ethereum
    cryptos = ['bitcoin', 'ethereum']
    days_to_collect = 365  # 1 year of data
    
    for crypto_id in cryptos:
        result = load_and_preprocess_data(crypto_id, days=days_to_collect)
        
        if result:
            print(f"\nPreprocessed data summary for {crypto_id}:")
            print(f"Training data shape: {result['X_train'].shape}")
            print(f"Testing data shape: {result['X_test'].shape}")
            print(f"Number of features: {len(result['feature_names'])}")
            
            # Show first 5 features
            preview_features = result['feature_names'][:5]
            print(f"Feature preview: {', '.join(preview_features)}")
            
            # Show target variables
            target_preview = list(result['y_train'].columns)
            print(f"Target variables: {', '.join(target_preview)}") 