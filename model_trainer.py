import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import data_processor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def load_processed_data(crypto_id, data_dir='processed_data'):
    """
    Load processed data for a cryptocurrency
    
    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., 'bitcoin')
        data_dir (str): Directory where processed data is stored
        
    Returns:
        dict: Dictionary containing loaded data
    """
    crypto_dir = os.path.join(data_dir, crypto_id)
    
    # Check if directory exists
    if not os.path.exists(crypto_dir):
        print(f"No processed data found for {crypto_id}")
        return None
    
    # Load data from CSV files
    X_train = pd.read_csv(os.path.join(crypto_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(crypto_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(crypto_dir, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(crypto_dir, 'y_test.csv'))
    
    # Check for and handle NaN values
    print("Checking for NaN values...")
    
    # Handle NaNs in training data
    if X_train.isna().any().any():
        print(f"Found NaN values in X_train. Handling them with median imputation.")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns
        )
    
    # Handle NaNs in test data
    if X_test.isna().any().any():
        print(f"Found NaN values in X_test. Handling them with median imputation.")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_test = pd.DataFrame(
            imputer.fit_transform(X_test),
            columns=X_test.columns
        )
    
    # Handle NaNs in target data
    for df, name in [(y_train, 'y_train'), (y_test, 'y_test')]:
        if df.isna().any().any():
            print(f"Found NaN values in {name}. Filling regression targets with 0 and classification targets with most frequent value.")
            
            # For regression targets (percentage changes)
            reg_cols = [col for col in df.columns if '_pct' in col]
            for col in reg_cols:
                df[col] = df[col].fillna(0)
                
            # For classification targets (binary)
            class_cols = [col for col in df.columns if '_up_' in col]
            for col in class_cols:
                # Fill with most frequent value (0 or 1)
                most_frequent = df[col].mode()[0]
                df[col] = df[col].fillna(most_frequent)
    
    # Load scalers
    scalers_path = os.path.join(crypto_dir, 'scalers.joblib')
    scalers = joblib.load(scalers_path)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scalers': scalers
    }

def train_price_change_models(X_train, y_train, output_dir='models'):
    """
    Train regression models to predict price changes
    
    Args:
        X_train (DataFrame): Training features
        y_train (DataFrame): Training targets
        output_dir (str): Directory to save models
        
    Returns:
        dict: Dictionary of trained models
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    models = {}
    
    # Target columns for regression (percentage price changes)
    regression_targets = [col for col in y_train.columns if '_pct' in col]
    
    for target in regression_targets:
        print(f"\nTraining regression models for {target}")
        
        # Get target data
        y = y_train[target]
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y)
        models[f'linear_{target}'] = lr
        print(f"Linear Regression trained for {target}")
        
        # Random Forest Regression
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y)
        models[f'rf_{target}'] = rf
        print(f"Random Forest trained for {target}")
        
        # MLP Regression
        mlp = train_mlp_regression(X_train, y, target)
        models[f'mlp_{target}'] = mlp
        print(f"MLP trained for {target}")
        
        # LSTM Regression (reshape data for LSTM)
        lstm = train_lstm_regression(X_train, y, target)
        models[f'lstm_{target}'] = lstm
        print(f"LSTM trained for {target}")
        
        # Save models
        joblib.dump(lr, os.path.join(output_dir, f'linear_{target}.joblib'))
        joblib.dump(rf, os.path.join(output_dir, f'rf_{target}.joblib'))
        mlp.save(os.path.join(output_dir, f'mlp_{target}.keras'))
        lstm.save(os.path.join(output_dir, f'lstm_{target}.keras'))
    
    return models

def train_price_direction_models(X_train, y_train, output_dir='models'):
    """
    Train classification models to predict price direction (up/down)
    
    Args:
        X_train (DataFrame): Training features
        y_train (DataFrame): Training targets
        output_dir (str): Directory to save models
        
    Returns:
        dict: Dictionary of trained models
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    models = {}
    
    # Target columns for classification (binary price direction)
    classification_targets = [col for col in y_train.columns if '_up_' in col]
    
    for target in classification_targets:
        print(f"\nTraining classification models for {target}")
        
        # Get target data
        y = y_train[target]
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y)
        models[f'logistic_{target}'] = lr
        print(f"Logistic Regression trained for {target}")
        
        # Random Forest Classification
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y)
        models[f'rf_{target}'] = rf
        print(f"Random Forest trained for {target}")
        
        # MLP Classification
        mlp = train_mlp_classification(X_train, y, target)
        models[f'mlp_{target}'] = mlp
        print(f"MLP trained for {target}")
        
        # LSTM Classification
        lstm = train_lstm_classification(X_train, y, target)
        models[f'lstm_{target}'] = lstm
        print(f"LSTM trained for {target}")
        
        # Save models
        joblib.dump(lr, os.path.join(output_dir, f'logistic_{target}.joblib'))
        joblib.dump(rf, os.path.join(output_dir, f'rf_{target}.joblib'))
        mlp.save(os.path.join(output_dir, f'mlp_{target}.keras'))
        lstm.save(os.path.join(output_dir, f'lstm_{target}.keras'))
    
    return models

def train_mlp_regression(X_train, y_train, target):
    """
    Train MLP regression model using Keras
    
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        target (str): Name of target variable
        
    Returns:
        Model: Trained Keras MLP model
    """
    # Convert to numpy arrays if they're DataFrames/Series
    X = X_train.values if hasattr(X_train, 'values') else X_train
    y = y_train.values if hasattr(y_train, 'values') else y_train
    
    # Create model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    model.fit(
        X, y,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    return model

def train_mlp_classification(X_train, y_train, target):
    """
    Train MLP classification model using Keras
    
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        target (str): Name of target variable
        
    Returns:
        Model: Trained Keras MLP model
    """
    # Convert to numpy arrays if they're DataFrames/Series
    X = X_train.values if hasattr(X_train, 'values') else X_train
    y = y_train.values if hasattr(y_train, 'values') else y_train
    
    # Create model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    model.fit(
        X, y,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    return model

def prepare_data_for_lstm(X, y=None, time_steps=10):
    """
    Prepare data for LSTM model (convert to 3D format)
    
    Args:
        X (DataFrame/array): Features
        y (Series/array): Target (optional)
        time_steps (int): Number of time steps to use
        
    Returns:
        tuple: (X_lstm, y_lstm) reshaped for LSTM
    """
    # Convert to numpy arrays if they're DataFrames/Series
    X_array = X.values if hasattr(X, 'values') else X
    
    # Create 3D data for LSTM [samples, time_steps, features]
    X_lstm = []
    y_lstm = []
    
    for i in range(len(X_array) - time_steps):
        X_lstm.append(X_array[i:i + time_steps])
        if y is not None:
            y_array = y.values if hasattr(y, 'values') else y
            y_lstm.append(y_array[i + time_steps])
    
    X_lstm = np.array(X_lstm)
    
    if y is not None:
        y_lstm = np.array(y_lstm)
        return X_lstm, y_lstm
    else:
        return X_lstm

def train_lstm_regression(X_train, y_train, target, time_steps=10):
    """
    Train LSTM regression model using Keras
    
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        target (str): Name of target variable
        time_steps (int): Number of time steps to use
        
    Returns:
        Model: Trained Keras LSTM model
    """
    # Prepare data for LSTM
    try:
        X_lstm, y_lstm = prepare_data_for_lstm(X_train, y_train, time_steps)
        
        # Create model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        model.fit(
            X_lstm, y_lstm,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return model
    except Exception as e:
        print(f"Error training LSTM regression model: {e}")
        # Return a simpler model if LSTM fails
        return train_mlp_regression(X_train, y_train, target)

def train_lstm_classification(X_train, y_train, target, time_steps=10):
    """
    Train LSTM classification model using Keras
    
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        target (str): Name of target variable
        time_steps (int): Number of time steps to use
        
    Returns:
        Model: Trained Keras LSTM model
    """
    # Prepare data for LSTM
    try:
        X_lstm, y_lstm = prepare_data_for_lstm(X_train, y_train, time_steps)
        
        # Create model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        model.fit(
            X_lstm, y_lstm,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return model
    except Exception as e:
        print(f"Error training LSTM classification model: {e}")
        # Return a simpler model if LSTM fails
        return train_mlp_classification(X_train, y_train, target)

def evaluate_regression_models(models, X_test, y_test):
    """
    Evaluate regression models on test data
    
    Args:
        models (dict): Dictionary of trained models
        X_test (DataFrame): Test features
        y_test (DataFrame): Test targets
        
    Returns:
        DataFrame: Evaluation metrics for each model
    """
    print("\nEvaluating regression models...")
    
    results = []
    
    # Evaluate each regression model
    for model_name, model in models.items():
        # Skip if not a regression model
        if not (model_name.startswith('linear_') or 
                model_name.startswith('rf_') or 
                model_name.startswith('mlp_') or 
                model_name.startswith('lstm_')) or '_up_' in model_name:
            continue
        
        # Extract target name from model name
        if model_name.startswith('linear_'):
            target = model_name[7:]
        elif model_name.startswith('rf_'):
            target = model_name[3:]
        elif model_name.startswith('mlp_'):
            target = model_name[4:]
        elif model_name.startswith('lstm_'):
            target = model_name[5:]
        
        # Make predictions
        if model_name.startswith('mlp_') or model_name.startswith('lstm_'):
            # For Keras models
            if model_name.startswith('lstm_'):
                # Prepare data for LSTM
                X_lstm = prepare_data_for_lstm(X_test, time_steps=10)
                # Get target values excluding the first time_steps rows
                y_true = y_test[target].values[10:]
                try:
                    y_pred = model.predict(X_lstm, verbose=0).flatten()
                except Exception as e:
                    print(f"Error predicting with LSTM model: {e}")
                    continue
            else:
                # For MLP models
                y_true = y_test[target]
                y_pred = model.predict(X_test, verbose=0).flatten()
        else:
            # For sklearn models
            y_true = y_test[target]
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Save results
        results.append({
            'model': model_name,
            'target': target,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
        
        print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    return pd.DataFrame(results)

def evaluate_classification_models(models, X_test, y_test):
    """
    Evaluate classification models on test data
    
    Args:
        models (dict): Dictionary of trained models
        X_test (DataFrame): Test features
        y_test (DataFrame): Test targets
        
    Returns:
        DataFrame: Evaluation metrics for each model
    """
    print("\nEvaluating classification models...")
    
    results = []
    
    # Evaluate each classification model
    for model_name, model in models.items():
        # Skip if not a classification model
        if not (model_name.startswith('logistic_') or 
                model_name.startswith('rf_') or 
                model_name.startswith('mlp_') or 
                model_name.startswith('lstm_')) or '_pct' in model_name:
            continue
        
        # Extract target name from model name
        if model_name.startswith('logistic_'):
            target = model_name[9:]
        elif model_name.startswith('rf_'):
            target = model_name[3:]
        elif model_name.startswith('mlp_'):
            target = model_name[4:]
        elif model_name.startswith('lstm_'):
            target = model_name[5:]
        
        # Make predictions
        if model_name.startswith('mlp_') or model_name.startswith('lstm_'):
            # For Keras models
            if model_name.startswith('lstm_'):
                # Prepare data for LSTM
                X_lstm = prepare_data_for_lstm(X_test, time_steps=10)
                # Get target values excluding the first time_steps rows
                y_true = y_test[target].values[10:]
                try:
                    y_pred_proba = model.predict(X_lstm, verbose=0).flatten()
                    y_pred = (y_pred_proba > 0.5).astype(int)
                except Exception as e:
                    print(f"Error predicting with LSTM model: {e}")
                    continue
            else:
                # For MLP models
                y_true = y_test[target]
                y_pred_proba = model.predict(X_test, verbose=0).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            # For sklearn models
            y_true = y_test[target]
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Save results
        results.append({
            'model': model_name,
            'target': target,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return pd.DataFrame(results)

def plot_feature_importance(models, feature_names, output_dir='plots'):
    """
    Plot feature importance for Random Forest models
    
    Args:
        models (dict): Dictionary of trained models
        feature_names (list): List of feature names
        output_dir (str): Directory to save plots
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot feature importance for each Random Forest model
    for model_name, model in models.items():
        # Skip if not a Random Forest model
        if not model_name.startswith('rf_'):
            continue
        
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            continue
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Take top 15 features
        top_n = 15
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature importance for {model_name}')
        plt.barh(range(top_n), top_importances, align='center')
        plt.yticks(range(top_n), top_features)
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'{model_name}_importance.png'))
        plt.close()

def plot_model_comparison(models, X_test, y_test, target, dates=None, plot_dir='plots'):
    """
    Plot true vs predicted values for multiple models
    
    Args:
        models (dict): Dictionary of trained models
        X_test (DataFrame): Test features
        y_test (DataFrame): Test targets
        target (str): Target variable name
        dates (Series, optional): Dates for x-axis
        plot_dir (str): Directory to save plots
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    plt.figure(figsize=(12, 6))
    
    # Plot true values
    y_true = y_test[target]
    
    if dates is not None:
        plt.plot(dates, y_true, label='True Values', color='black', linestyle='--')
    else:
        plt.plot(y_true.values, label='True Values', color='black', linestyle='--')
    
    # Plot predicted values for each model
    model_types_to_plot = ['linear', 'rf', 'mlp', 'lstm']
    model_colors = ['blue', 'green', 'red', 'purple']
    
    for model_type, color in zip(model_types_to_plot, model_colors):
        model_name = f"{model_type}_{target}"
        
        if model_name not in models:
            continue
            
        model = models[model_name]
        
        # Get predictions
        if model_type == 'lstm':
            # Prepare data for LSTM
            X_lstm = prepare_data_for_lstm(X_test, time_steps=10)
            try:
                y_pred = model.predict(X_lstm, verbose=0).flatten()
                # Adjust the length to match the original data (LSTM predictions are shorter)
                if dates is not None:
                    # Crop the first 10 points from dates and true values for proper alignment
                    plt.plot(dates[10:], y_pred, label=f'{model_type.upper()} Predictions', color=color)
                else:
                    plt.plot(range(10, 10 + len(y_pred)), y_pred, label=f'{model_type.upper()} Predictions', color=color)
            except Exception as e:
                print(f"Error plotting LSTM predictions: {e}")
        elif model_type == 'mlp':
            try:
                y_pred = model.predict(X_test, verbose=0).flatten()
                if dates is not None:
                    plt.plot(dates, y_pred, label=f'{model_type.upper()} Predictions', color=color)
                else:
                    plt.plot(y_pred, label=f'{model_type.upper()} Predictions', color=color)
            except Exception as e:
                print(f"Error plotting MLP predictions: {e}")
        else:
            y_pred = model.predict(X_test)
            if dates is not None:
                plt.plot(dates, y_pred, label=f'{model_type.upper()} Predictions', color=color)
            else:
                plt.plot(y_pred, label=f'{model_type.upper()} Predictions', color=color)
    
    plt.title(f'True vs Predicted Values for {target}')
    plt.xlabel('Time' if dates is not None else 'Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(plot_dir, f'{target}_model_comparison.png'))
    plt.close()

def plot_error_metrics(regression_metrics, classification_metrics, plot_dir='plots'):
    """
    Plot error metrics comparison for different models
    
    Args:
        regression_metrics (DataFrame): Metrics for regression models
        classification_metrics (DataFrame): Metrics for classification models
        plot_dir (str): Directory to save plots
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    # Regression metrics
    if regression_metrics is not None and not regression_metrics.empty:
        # Plot RMSE
        plt.figure(figsize=(12, 6))
        
        for target in regression_metrics['target'].unique():
            target_df = regression_metrics[regression_metrics['target'] == target]
            
            models = target_df['model'].apply(lambda x: x.split('_')[0].upper())
            rmse_values = target_df['RMSE']
            
            plt.bar(models, rmse_values, label=target)
        
        plt.title('RMSE Comparison Across Models')
        plt.xlabel('Model')
        plt.ylabel('RMSE')
        if len(regression_metrics['target'].unique()) > 1:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'rmse_comparison.png'))
        plt.close()
        
        # Plot R²
        plt.figure(figsize=(12, 6))
        
        for target in regression_metrics['target'].unique():
            target_df = regression_metrics[regression_metrics['target'] == target]
            
            models = target_df['model'].apply(lambda x: x.split('_')[0].upper())
            r2_values = target_df['R2']
            
            plt.bar(models, r2_values, label=target)
        
        plt.title('R² Comparison Across Models')
        plt.xlabel('Model')
        plt.ylabel('R²')
        if len(regression_metrics['target'].unique()) > 1:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'r2_comparison.png'))
        plt.close()
    
    # Classification metrics
    if classification_metrics is not None and not classification_metrics.empty:
        # Plot F1 Score
        plt.figure(figsize=(12, 6))
        
        for target in classification_metrics['target'].unique():
            target_df = classification_metrics[classification_metrics['target'] == target]
            
            models = target_df['model'].apply(lambda x: x.split('_')[0].upper())
            f1_values = target_df['f1_score']
            
            plt.bar(models, f1_values, label=target)
        
        plt.title('F1 Score Comparison Across Models')
        plt.xlabel('Model')
        plt.ylabel('F1 Score')
        if len(classification_metrics['target'].unique()) > 1:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'f1_comparison.png'))
        plt.close()

def plot_training_history(history, model_name, plot_dir='plots'):
    """
    Plot training and validation loss curves for Keras models
    
    Args:
        history (History): Keras training history
        model_name (str): Name of the model
        plot_dir (str): Directory to save plots
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    plt.figure(figsize=(10, 6))
    
    # Plot training & validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    
    plt.title(f'Training and Validation Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(plot_dir, f'{model_name}_training_history.png'))
    plt.close()

def plot_moving_averages(y_true, y_pred, dates=None, window_sizes=[7, 30], plot_dir='plots', title='Price with Moving Averages'):
    """
    Plot actual values, predicted values, and moving averages
    
    Args:
        y_true (Series): True values
        y_pred (Series/array): Predicted values
        dates (Series, optional): Dates for x-axis
        window_sizes (list): Window sizes for moving averages
        plot_dir (str): Directory to save plots
        title (str): Plot title
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    plt.figure(figsize=(12, 6))
    
    # Plot true and predicted values
    if dates is not None:
        plt.plot(dates, y_true, label='Actual', color='black')
        plt.plot(dates, y_pred, label='Predicted', color='blue')
    else:
        plt.plot(y_true, label='Actual', color='black')
        plt.plot(y_pred, label='Predicted', color='blue')
    
    # Calculate and plot moving averages
    colors = ['green', 'red', 'purple', 'orange']
    
    for i, window in enumerate(window_sizes):
        if i < len(colors):
            ma = y_true.rolling(window=window).mean()
            
            if dates is not None:
                plt.plot(dates, ma, label=f'{window}-day MA', color=colors[i])
            else:
                plt.plot(ma, label=f'{window}-day MA', color=colors[i])
    
    plt.title(title)
    plt.xlabel('Time' if dates is not None else 'Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(plot_dir, 'moving_averages.png'))
    plt.close()

def backtest_models(models, X_test, y_test, scalers=None, test_dates=None, output_dir='backtest_results'):
    """
    Backtest ML models on test data and generate comprehensive evaluation
    
    Args:
        models (dict): Dictionary of trained models
        X_test (DataFrame): Test features
        y_test (DataFrame): Test targets
        scalers (dict, optional): Dictionary of scalers for inverse transformation
        test_dates (Series, optional): Dates corresponding to test data
        output_dir (str): Directory to save backtest results
        
    Returns:
        tuple: (regression_metrics, classification_metrics)
    """
    print("\nStarting model backtesting and evaluation...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Separate regression and classification targets
    regression_targets = [col for col in y_test.columns if '_pct' in col]
    classification_targets = [col for col in y_test.columns if '_up_' in col]
    
    # Initialize results
    regression_results = []
    classification_results = []
    
    # Create subdirectories for plots
    plots_dir = os.path.join(output_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Evaluate regression models
    for target in regression_targets:
        print(f"\nEvaluating models for target: {target}")
        
        # Plot model comparison
        print(f"Plotting model comparison for {target}")
        plot_model_comparison(
            models, 
            X_test, 
            y_test, 
            target, 
            dates=test_dates, 
            plot_dir=plots_dir
        )
        
        # For each model type
        for model_type in ['linear', 'rf', 'mlp', 'lstm']:
            model_name = f"{model_type}_{target}"
            
            if model_name not in models:
                continue
                
            model = models[model_name]
            y_true = y_test[target]
            
            # Get predictions
            if model_type == 'lstm':
                # Prepare data for LSTM
                X_lstm = prepare_data_for_lstm(X_test, time_steps=10)
                try:
                    y_pred = model.predict(X_lstm, verbose=0).flatten()
                    y_true_lstm = y_true.values[10:]  # Adjust for LSTM time steps
                    # Calculate metrics
                    mse = mean_squared_error(y_true_lstm, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_true_lstm, y_pred)
                    r2 = r2_score(y_true_lstm, y_pred)
                    
                    # Plot moving averages (only for LSTM model as an example)
                    if len(y_pred) > 30:  # Need at least 30 points for 30-day MA
                        # Convert to Series for rolling function
                        y_pred_series = pd.Series(y_pred)
                        y_true_series = pd.Series(y_true_lstm)
                        
                        plot_moving_averages(
                            y_true_series,
                            y_pred_series,
                            dates=test_dates[10:] if test_dates is not None else None,
                            plot_dir=plots_dir,
                            title=f'{model_type.upper()} Predictions with Moving Averages - {target}'
                        )
                except Exception as e:
                    print(f"Error evaluating LSTM model: {e}")
                    continue
            elif model_type == 'mlp':
                try:
                    y_pred = model.predict(X_test, verbose=0).flatten()
                    # Calculate metrics
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                except Exception as e:
                    print(f"Error evaluating MLP model: {e}")
                    continue
            else:
                y_pred = model.predict(X_test)
                # Calculate metrics
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
            
            # Store results
            regression_results.append({
                'model': model_name,
                'target': target,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })
            
            print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    # Evaluate classification models
    for target in classification_targets:
        print(f"\nEvaluating models for target: {target}")
        
        # For each model type
        for model_type in ['logistic', 'rf', 'mlp', 'lstm']:
            model_name = f"{model_type}_{target}"
            
            if model_name not in models:
                continue
                
            model = models[model_name]
            y_true = y_test[target]
            
            # Get predictions
            if model_type == 'lstm':
                # Prepare data for LSTM
                X_lstm = prepare_data_for_lstm(X_test, time_steps=10)
                try:
                    y_pred_proba = model.predict(X_lstm, verbose=0).flatten()
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    y_true_lstm = y_true.values[10:]  # Adjust for LSTM time steps
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_true_lstm, y_pred)
                    precision = precision_score(y_true_lstm, y_pred)
                    recall = recall_score(y_true_lstm, y_pred)
                    f1 = f1_score(y_true_lstm, y_pred)
                except Exception as e:
                    print(f"Error evaluating LSTM model: {e}")
                    continue
            elif model_type == 'mlp':
                try:
                    y_pred_proba = model.predict(X_test, verbose=0).flatten()
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred)
                    recall = recall_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred)
                except Exception as e:
                    print(f"Error evaluating MLP model: {e}")
                    continue
            else:
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
            
            # Store results
            classification_results.append({
                'model': model_name,
                'target': target,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Convert results to DataFrames
    regression_metrics = pd.DataFrame(regression_results)
    classification_metrics = pd.DataFrame(classification_results)
    
    # Save metrics to CSV
    regression_metrics.to_csv(os.path.join(output_dir, 'regression_metrics.csv'), index=False)
    classification_metrics.to_csv(os.path.join(output_dir, 'classification_metrics.csv'), index=False)
    
    # Plot error metrics comparison
    plot_error_metrics(regression_metrics, classification_metrics, plot_dir=plots_dir)
    
    # Determine best models
    if not regression_metrics.empty:
        best_reg_model = regression_metrics.loc[regression_metrics['R2'].idxmax()]
        print(f"\nBest regression model by R²: {best_reg_model['model']} with R² = {best_reg_model['R2']:.4f}")
    
    if not classification_metrics.empty:
        best_class_model = classification_metrics.loc[classification_metrics['f1_score'].idxmax()]
        print(f"Best classification model by F1: {best_class_model['model']} with F1 = {best_class_model['f1_score']:.4f}")
    
    return regression_metrics, classification_metrics

def train_and_evaluate_models(crypto_id, data_dir='processed_data', models_dir='models', plots_dir='plots'):
    """
    Train and evaluate ML models for a cryptocurrency
    
    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., 'bitcoin')
        data_dir (str): Directory where processed data is stored
        models_dir (str): Directory to save trained models
        plots_dir (str): Directory to save plots
        
    Returns:
        tuple: (regression_metrics, classification_metrics, models)
    """
    # Load processed data
    data = load_processed_data(crypto_id, data_dir)
    
    if data is None:
        return None
    
    # Create crypto-specific directories
    crypto_models_dir = os.path.join(models_dir, crypto_id)
    crypto_plots_dir = os.path.join(plots_dir, crypto_id)
    backtest_dir = os.path.join('backtest_results', crypto_id)
    
    for directory in [crypto_models_dir, crypto_plots_dir, backtest_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Train regression models
    regression_models = train_price_change_models(data['X_train'], data['y_train'], crypto_models_dir)
    
    # Train classification models
    classification_models = train_price_direction_models(data['X_train'], data['y_train'], crypto_models_dir)
    
    # Combine all models
    all_models = {**regression_models, **classification_models}
    
    # Evaluate models
    regression_metrics = evaluate_regression_models(all_models, data['X_test'], data['y_test'])
    classification_metrics = evaluate_classification_models(all_models, data['X_test'], data['y_test'])
    
    # Plot feature importance
    plot_feature_importance(all_models, data['X_train'].columns, crypto_plots_dir)
    
    # Backtest models and generate comprehensive evaluation
    backtest_metrics = backtest_models(
        all_models,
        data['X_test'],
        data['y_test'],
        scalers=data.get('scalers'),
        output_dir=backtest_dir
    )
    
    return regression_metrics, classification_metrics, all_models

if __name__ == "__main__":
    # Train and evaluate models for Bitcoin and Ethereum
    cryptos = ['bitcoin', 'ethereum']
    
    for crypto_id in cryptos:
        print(f"\n{'='*50}")
        print(f"Training models for {crypto_id}")
        print(f"{'='*50}")
        
        results = train_and_evaluate_models(crypto_id)
        
        if results:
            reg_metrics, class_metrics, _ = results
            
            # Save metrics to CSV
            reg_metrics.to_csv(f'models/{crypto_id}/regression_metrics.csv', index=False)
            class_metrics.to_csv(f'models/{crypto_id}/classification_metrics.csv', index=False)
            
            print(f"\nModel training and evaluation complete for {crypto_id}")
            print(f"Best regression model: {reg_metrics.loc[reg_metrics['R2'].idxmax()]['model']}")
            print(f"Best classification model: {class_metrics.loc[class_metrics['f1_score'].idxmax()]['model']}") 