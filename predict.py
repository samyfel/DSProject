import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import glob
import joblib
import json
import sys

# Import custom modules
import crypto
import data_processor

def load_models(crypto_id, models_dir='models'):
    """
    Load trained models for a cryptocurrency
    
    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., 'bitcoin')
        models_dir (str): Directory containing model files
        
    Returns:
        dict: Dictionary of trained models
    """
    print(f"Loading models for {crypto_id}...")
    models_path = os.path.join(models_dir, crypto_id)
    
    if not os.path.exists(models_path):
        print(f"No models found for {crypto_id} at {models_path}")
        return None
    
    models = {'regression': {}, 'classification': {}}
    model_count = 0
    
    # Load regression models
    for model_file in glob.glob(os.path.join(models_path, '*.joblib')):
        try:
            # Skip scaler files
            if 'scaler' in model_file:
                continue
                
            model_name = os.path.basename(model_file).replace('.joblib', '')
            model = joblib.load(model_file)
            
            # Determine model type
            if model_name.startswith('rf_price_change_') or model_name.startswith('linear_price_change_'):
                models['regression'][model_name] = model
                model_count += 1
            elif model_name.startswith('rf_price_up_') or model_name.startswith('logistic_price_up_'):
                models['classification'][model_name] = model
                model_count += 1
                
        except Exception as e:
            print(f"Error loading model {model_file}: {e}")
    
    print(f"Loaded {model_count} models for {crypto_id}")
    return models

def get_current_data(crypto_id, days_history=60):
    """
    Get current data for prediction
    
    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., 'bitcoin')
        days_history (int): Number of days of historical data to retrieve
        
    Returns:
        DataFrame: DataFrame with current data
    """
    print(f"Fetching current data for {crypto_id}...")
    
    # Get historical data with indicators
    data = crypto.get_historical_data_with_indicators(crypto_id, days_history)
    
    if data is None:
        print(f"Failed to fetch data for {crypto_id}")
        return None
    
    # Process data
    print("Processing data...")
    
    # Handle missing values
    data = data_processor.handle_missing_values(data)
    
    # Engineer features
    try:
        # Try to use data_processor's engineer_features function
        data = data_processor.engineer_features(data)
    except Exception as e:
        print(f"Warning: Error using data_processor.engineer_features: {e}")
        print("Using fallback feature engineering")
        # Fallback feature engineering
        data = engineer_features_fallback(data)
    
    # Get most recent data point
    current_data = data.iloc[-1].copy()
    
    return current_data, data

def engineer_features_fallback(df):
    """
    Fallback function to engineer features for prediction
    
    Args:
        df (DataFrame): DataFrame to process
        
    Returns:
        DataFrame: DataFrame with additional engineered features
    """
    print("Using fallback feature engineering...")
    
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
    
    # Ensure all the required features exist
    required_features = [
        'price_change_pct', 'price_change_pct_lag_1', 'price_change_pct_lag_3', 'price_change_pct_lag_7',
        'price_ma7_diff', 'ma20_diff', 'bb_position', 'volume_lag_1', 'volume_lag_3', 'volume_lag_7'
    ]
    
    # Add price change percentage if not present
    if 'price_change_pct' not in df_featured.columns and 'price' in df_featured.columns:
        df_featured['price_change_pct'] = df_featured['price'].pct_change() * 100
    
    # Add price distance from moving averages (%)
    if 'price_ma7' in df_featured.columns and 'price_ma7_diff' not in df_featured.columns:
        df_featured['price_ma7_diff'] = ((df_featured['price'] - df_featured['price_ma7']) 
                                        / df_featured['price_ma7'] * 100)
    
    if 'ma20' in df_featured.columns and 'ma20_diff' not in df_featured.columns:
        df_featured['ma20_diff'] = ((df_featured['price'] - df_featured['ma20']) 
                                   / df_featured['ma20'] * 100)
    
    # Bollinger Band position (%)
    if all(col in df_featured.columns for col in ['bollinger_upper', 'bollinger_lower']) and 'bb_position' not in df_featured.columns:
        bb_range = df_featured['bollinger_upper'] - df_featured['bollinger_lower']
        df_featured['bb_position'] = ((df_featured['price'] - df_featured['bollinger_lower']) 
                                     / bb_range * 100)
    
    # Add lagged features
    if 'price' in df_featured.columns:
        price_lags = [1, 3, 7]
        for lag in price_lags:
            lag_col = f'price_lag_{lag}'
            if lag_col not in df_featured.columns:
                df_featured[lag_col] = df_featured['price'].shift(lag)
    
    if 'price_change_pct' in df_featured.columns:
        pct_change_lags = [1, 3, 7]
        for lag in pct_change_lags:
            lag_col = f'price_change_pct_lag_{lag}'
            if lag_col not in df_featured.columns:
                df_featured[lag_col] = df_featured['price_change_pct'].shift(lag)
    
    if 'volume' in df_featured.columns:
        volume_lags = [1, 3, 7]
        for lag in volume_lags:
            lag_col = f'volume_lag_{lag}'
            if lag_col not in df_featured.columns:
                df_featured[lag_col] = df_featured['volume'].shift(lag)
    
    # Fill NaN values in all columns with median for each column
    for col in df_featured.columns:
        if col != 'timestamp' and df_featured[col].dtype != 'object' and df_featured[col].dtype != 'datetime64[ns]':
            if df_featured[col].isna().any():
                df_featured[col] = df_featured[col].fillna(df_featured[col].median())
    
    return df_featured

def get_training_feature_order(data_dir, crypto_id):
    """
    Get the order of features from the training data
    
    Args:
        data_dir (str): Directory containing processed data
        crypto_id (str): Cryptocurrency ID
        
    Returns:
        list: Feature names in the correct order
    """
    try:
        x_train_path = os.path.join(data_dir, crypto_id, 'X_train.csv')
        if os.path.exists(x_train_path):
            # Read just the header row
            df_header = pd.read_csv(x_train_path, nrows=0)
            return list(df_header.columns)
        else:
            print(f"Training data file not found at {x_train_path}")
            return None
    except Exception as e:
        print(f"Error reading training feature order: {e}")
        return None

def prepare_for_prediction(current_data, scalers_path, training_features=None, data_dir='processed_data', crypto_id=None):
    """
    Prepare the current data for prediction
    
    Args:
        current_data: Current data point for prediction
        scalers_path (str): Path to the scalers file
        training_features (list): List of feature names in the order used for training
        data_dir (str): Directory containing processed data
        crypto_id (str): Cryptocurrency ID
        
    Returns:
        DataFrame: Prepared data for prediction
    """
    # Load scalers
    scalers = joblib.load(scalers_path)
    
    # Get the scalers
    price_scaler = scalers.get('price_scaler')
    indicator_scaler = scalers.get('indicator_scaler')
    
    if price_scaler is None or indicator_scaler is None:
        print("Warning: Could not find required scalers in scalers file")
        return None
    
    # Convert current_data to DataFrame if it's a Series
    if isinstance(current_data, pd.Series):
        current_data = pd.DataFrame([current_data])
    
    # Make a copy of current data to avoid modifying the original
    features_df = current_data.copy()
    
    # Add date-related features if timestamp is available
    if 'timestamp' in features_df.columns:
        timestamp = features_df['timestamp'].iloc[0]
        
        # Add day of week (0=Monday, 6=Sunday)
        features_df['day_of_week'] = timestamp.dayofweek
        
        # Add month
        features_df['month'] = timestamp.month
        
        # Add quarter
        features_df['quarter'] = timestamp.quarter
        
        # Add is_weekend flag
        features_df['is_weekend'] = 1 if timestamp.dayofweek >= 5 else 0
    
    # Remove timestamp and target columns
    exclude_cols = ['timestamp']
    for col in features_df.columns:
        if col.startswith('next_day_price') or col.startswith('price_change_') or col.startswith('price_up_'):
            if not col.endswith('_lag_1') and not col.endswith('_lag_3') and not col.endswith('_lag_7'):
                exclude_cols.append(col)
    
    features_df = features_df.drop(exclude_cols, axis=1, errors='ignore')
    
    # Extract feature names from scalers
    try:
        price_scaler_features = price_scaler.get_feature_names_out()
    except:
        # For older scikit-learn versions or custom scalers
        price_scaler_features = price_scaler.feature_names_in_ if hasattr(price_scaler, 'feature_names_in_') else []
    
    try:
        indicator_scaler_features = indicator_scaler.get_feature_names_out()
    except:
        # For older scikit-learn versions or custom scalers
        indicator_scaler_features = indicator_scaler.feature_names_in_ if hasattr(indicator_scaler, 'feature_names_in_') else []
    
    # Get training feature order if not provided
    if training_features is None and crypto_id is not None:
        training_features = get_training_feature_order(data_dir, crypto_id)
    
    # List of all features expected by the scalers
    all_expected_features = list(price_scaler_features) + list(indicator_scaler_features)
    all_expected_features = list(set(all_expected_features))  # Remove duplicates
    
    # Add date-related features if they're not already in the expected features
    for date_feat in ['day_of_week', 'month', 'quarter', 'is_weekend']:
        if date_feat not in all_expected_features:
            all_expected_features.append(date_feat)
    
    # If we have training features, use them to define the order
    if training_features:
        # Keep only the expected features
        ordered_features = [f for f in training_features if f in all_expected_features]
        
        # Add any missing expected features (that weren't in training data)
        for feat in all_expected_features:
            if feat not in ordered_features:
                ordered_features.append(feat)
    else:
        # Sort alphabetically as a fallback
        ordered_features = sorted(all_expected_features)
    
    # Check if any expected features are missing
    missing_features = [feat for feat in ordered_features if feat not in features_df.columns]
    if missing_features:
        print(f"Warning: Missing {len(missing_features)} features: {missing_features[:5]}...")
    
    # Create DataFrame with all required features in the correct order
    prediction_df = pd.DataFrame(0, index=features_df.index, columns=ordered_features)
    
    # Fill with available data
    for col in features_df.columns:
        if col in ordered_features:
            prediction_df[col] = features_df[col].values
    
    # For missing lagged features, try to calculate them if possible
    if 'price' in features_df.columns:
        if 'price_change_pct' in missing_features and 'price_change_pct' not in features_df.columns:
            # Use the last data point
            prediction_df['price_change_pct'] = 0  # Default to 0 for most recent point
    
    # Get columns for price features and indicator features
    price_features = [col for col in prediction_df.columns if col in price_scaler_features]
    indicator_features = [col for col in prediction_df.columns if col in indicator_scaler_features]
    
    # Scale price features
    if price_features:
        prediction_df[price_features] = price_scaler.transform(prediction_df[price_features])
    
    # Scale indicator features
    if indicator_features:
        prediction_df[indicator_features] = indicator_scaler.transform(prediction_df[indicator_features])
    
    return prediction_df

def make_predictions(models, prepared_data, current_price):
    """
    Make predictions using the trained models
    
    Args:
        models (dict): Dictionary of trained models
        prepared_data (DataFrame): Prepared data for prediction
        current_price (float): The current price of the cryptocurrency
        
    Returns:
        dict: Dictionary of predictions
    """
    predictions = {}
    
    # Check for each model type
    for model_type in ['regression', 'classification']:
        if model_type in models:
            # Get models for this type
            type_models = models[model_type]
            
            for model_name, model in type_models.items():
                try:
                    # Get the target name from the model name
                    target_name = model_name.split('_', 1)[1]  # Remove model type prefix
                    
                    # Make prediction
                    if model_type == 'regression':
                        # For regression models, predict price change percentage
                        pred = model.predict(prepared_data)[0]
                        predictions[target_name] = pred
                        
                        # If this is a price change prediction, calculate the actual price
                        if 'price_change' in target_name:
                            # Extract time frame (next_day, 3d, 7d, etc.)
                            time_frame = target_name.replace('price_change_', '').replace('_pct', '')
                            
                            # Calculate predicted price
                            predicted_price = current_price * (1 + pred/100)
                            predictions[f'predicted_price_{time_frame}'] = predicted_price
                    
                    elif model_type == 'classification':
                        # For classification models, predict probability of price going up
                        try:
                            # Try to get probability of class 1 (price up)
                            pred_proba = model.predict_proba(prepared_data)[0][1]
                            predictions[f'{target_name}_probability'] = pred_proba
                        except:
                            # If predict_proba is not available, just use the class prediction
                            pred = model.predict(prepared_data)[0]
                            predictions[target_name] = pred
                
                except Exception as e:
                    print(f"Error making prediction with model {model_name}: {e}")
    
    return predictions

def format_predictions(predictions, current_price, crypto_id):
    """
    Format predictions for display
    
    Args:
        predictions (dict): Dictionary of predictions
        current_price (float): The current price of the cryptocurrency
        crypto_id (str): The ID of the cryptocurrency
        
    Returns:
        dict: Formatted predictions
    """
    formatted = {
        'crypto_id': crypto_id,
        'current_price': current_price,
        'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'price_predictions': {},
        'direction_predictions': {}
    }
    
    # Price change predictions
    time_frames = ['next_day', '3d', '7d', '14d', '30d']
    for time_frame in time_frames:
        # Check for percentage change prediction
        pct_key = f'price_change_{time_frame}_pct'
        price_key = f'predicted_price_{time_frame}'
        
        if pct_key in predictions:
            formatted['price_predictions'][time_frame] = {
                'percent_change': round(predictions[pct_key], 2),
                'predicted_price': round(predictions.get(price_key, 0), 2)
            }
        
        # Check for direction prediction
        direction_key = f'price_up_{time_frame}'
        prob_key = f'price_up_{time_frame}_probability'
        
        if prob_key in predictions:
            formatted['direction_predictions'][time_frame] = {
                'probability_up': round(predictions[prob_key] * 100, 2)
            }
    
    return formatted

def visualize_predictions(predictions, historical_data, output_dir):
    """
    Visualize price predictions and save the plot
    
    Args:
        predictions (dict): Formatted predictions
        historical_data (DataFrame): Historical price data
        output_dir (str): Directory to save the plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare plot data
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    dates = historical_data['timestamp'].iloc[-30:]  # Last 30 days
    prices = historical_data['price'].iloc[-30:]
    plt.plot(dates, prices, 'b-', label='Historical Price')
    
    # Get current date and price
    current_date = historical_data['timestamp'].iloc[-1]
    current_price = predictions['current_price']
    
    # Plot current price
    plt.scatter([current_date], [current_price], color='blue', s=50, zorder=5)
    
    # Plot predictions
    future_dates = []
    predicted_prices = []
    confidence_values = []
    
    time_frames = ['next_day', '3d', '7d', '14d', '30d']
    for time_frame in time_frames:
        if time_frame in predictions['price_predictions']:
            # Convert time frame to days
            if time_frame == 'next_day':
                days = 1
            else:
                days = int(time_frame.replace('d', ''))
            
            # Get prediction date
            pred_date = current_date + pd.Timedelta(days=days)
            future_dates.append(pred_date)
            
            # Get predicted price
            pred_price = predictions['price_predictions'][time_frame]['predicted_price']
            predicted_prices.append(pred_price)
            
            # Get confidence
            if time_frame in predictions['direction_predictions']:
                confidence = predictions['direction_predictions'][time_frame]['probability_up']
                confidence_values.append(confidence)
            else:
                confidence_values.append(50)  # Default confidence
    
    # Plot predicted prices
    plt.plot(future_dates, predicted_prices, 'r--', label='Predicted Price')
    
    # Add confidence as scatter points with color gradient
    for i, (date, price, confidence) in enumerate(zip(future_dates, predicted_prices, confidence_values)):
        # Color based on confidence (green for high confidence, red for low)
        if confidence > 50:
            confidence_color = (0, min(1, confidence/100), 0)  # Green with intensity based on confidence
        else:
            confidence_color = (min(1, (100-confidence)/100), 0, 0)  # Red with intensity based on confidence
        
        plt.scatter([date], [price], color=confidence_color, s=50, zorder=5)
    
    # Add annotations
    for i, (date, price, confidence) in enumerate(zip(future_dates, predicted_prices, confidence_values)):
        direction = "↑" if confidence > 50 else "↓"
        plt.annotate(f"{direction} {confidence:.1f}%", 
                    (date, price), 
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center')
    
    # Format plot
    crypto_name = predictions['crypto_id'].capitalize()
    plt.title(f"{crypto_name} Price Prediction ({predictions['prediction_time']})")
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Prediction visualization saved to {plot_path}")

def predict_crypto(crypto_id, models_dir='models', data_dir='processed_data'):
    """
    Make predictions for a cryptocurrency
    
    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., 'bitcoin')
        models_dir (str): Directory containing model files
        data_dir (str): Directory containing processed data
        
    Returns:
        dict: Formatted predictions
    """
    # Load models
    models = load_models(crypto_id, models_dir)
    if models is None:
        print(f"Failed to load models for {crypto_id}")
        return None
    
    # Get current data
    current_data, historical_data = get_current_data(crypto_id)
    if current_data is None:
        print(f"Failed to get current data for {crypto_id}")
        return None
    
    # Current price
    current_price = current_data['price']
    
    # Path to scalers
    scalers_path = os.path.join(data_dir, crypto_id, 'scalers.joblib')
    if not os.path.exists(scalers_path):
        print(f"Failed to find scalers at {scalers_path}")
        return None
    
    # Prepare data for prediction
    features_df = prepare_for_prediction(current_data, scalers_path, crypto_id=crypto_id, data_dir=data_dir)
    if features_df is None:
        print(f"Failed to prepare data for prediction")
        return None
        
    # Make predictions
    raw_predictions = make_predictions(models, features_df, current_price)
    
    # Format predictions
    formatted_predictions = format_predictions(raw_predictions, current_price, crypto_id)
    
    # Visualize predictions
    try:
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join('plots', crypto_id, 'predictions')
        os.makedirs(plots_dir, exist_ok=True)
        
        visualize_predictions(formatted_predictions, historical_data, plots_dir)
    except Exception as e:
        print(f"Warning: Could not visualize predictions: {e}")
    
    return formatted_predictions

def display_predictions(predictions):
    """
    Display the formatted predictions
    
    Args:
        predictions (dict): Formatted predictions
    """
    crypto_name = predictions['crypto_id'].capitalize()
    current_price = predictions['current_price']
    pred_time = predictions['prediction_time']
    
    print(f"{'-'*50}")
    print(f"{crypto_name} PRICE PREDICTIONS ({pred_time})")
    print(f"Current price: ${current_price:.2f}")
    print(f"{'-'*50}")
    
    # Short-term predictions
    print("SHORT-TERM PREDICTIONS:")
    try:
        print(f"Tomorrow: ${predictions['price_predictions']['next_day']['predicted_price']:.2f} ({predictions['price_predictions']['next_day']['percent_change']:.2f}%) - Direction: {'up' if predictions['direction_predictions']['next_day']['probability_up'] > 50 else 'down'} ({predictions['direction_predictions']['next_day']['probability_up']:.1f}% confidence)")
        print(f"3 days: ${predictions['price_predictions']['3d']['predicted_price']:.2f} ({predictions['price_predictions']['3d']['percent_change']:.2f}%) - Direction: {'up' if predictions['direction_predictions']['3d']['probability_up'] > 50 else 'down'} ({predictions['direction_predictions']['3d']['probability_up']:.1f}% confidence)")
        print(f"7 days: ${predictions['price_predictions']['7d']['predicted_price']:.2f} ({predictions['price_predictions']['7d']['percent_change']:.2f}%) - Direction: {'up' if predictions['direction_predictions']['7d']['probability_up'] > 50 else 'down'} ({predictions['direction_predictions']['7d']['probability_up']:.1f}% confidence)")
    except KeyError as e:
        print(f"Warning: Missing prediction data - {e}")
    print(f"{'-'*50}")
    
    # Medium-term predictions
    print("MEDIUM-TERM PREDICTIONS:")
    try:
        print(f"14 days: ${predictions['price_predictions']['14d']['predicted_price']:.2f} ({predictions['price_predictions']['14d']['percent_change']:.2f}%) - Direction: {'up' if predictions['direction_predictions']['14d']['probability_up'] > 50 else 'down'} ({predictions['direction_predictions']['14d']['probability_up']:.1f}% confidence)")
        print(f"30 days: ${predictions['price_predictions']['30d']['predicted_price']:.2f} ({predictions['price_predictions']['30d']['percent_change']:.2f}%) - Direction: {'up' if predictions['direction_predictions']['30d']['probability_up'] > 50 else 'down'} ({predictions['direction_predictions']['30d']['probability_up']:.1f}% confidence)")
    except KeyError as e:
        print(f"Warning: Missing prediction data - {e}")
    print(f"{'-'*50}")
    
    # Save predictions to JSON
    output_dir = os.path.join('predictions', predictions['crypto_id'])
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to {output_file}")

def main():
    """
    Main function to run predictions
    """
    # Parse command line arguments
    cryptos = ['bitcoin', 'ethereum']
    if len(sys.argv) > 1:
        cryptos = sys.argv[1:]
    
    # Ensure the output directories exist
    os.makedirs('predictions', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Make predictions for each cryptocurrency
    for crypto_id in cryptos:
        print(f"\nPredicting for {crypto_id.capitalize()}...")
        try:
            predictions = predict_crypto(crypto_id)
            if predictions:
                display_predictions(predictions)
            else:
                print(f"Failed to generate predictions for {crypto_id}")
        except Exception as e:
            print(f"Error predicting {crypto_id}: {e}")

if __name__ == "__main__":
    main() 