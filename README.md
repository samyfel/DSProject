# Cryptocurrency Price Prediction System

## 🚀 Quick Start - Frontend Setup

### Prerequisites
- Node.js (v14 or higher)
- npm (Node Package Manager)

### Frontend Setup Instructions
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open your browser and visit:
   ```
   http://localhost:3000
   ```

The frontend application should now be running! If you encounter any issues:
- Make sure all dependencies are installed correctly
- Check if port 3000 is available
- Clear npm cache if needed: `npm cache clean --force`
- Delete node_modules and reinstall: `rm -rf node_modules && npm install`

### Troubleshooting Common Issues
- **"Module not found" errors**: Run `npm install` again
- **Port 3000 already in use**: Kill the process using the port or use a different port:
  ```bash
  npm start -- --port 3001
  ```
- **Compilation errors**: Clear the cache and restart:
  ```bash
  rm -rf node_modules/.cache
  npm start
  ```

## Overview

This project implements a comprehensive cryptocurrency price prediction system that uses historical data and machine learning techniques to forecast future price movements of Bitcoin and Ethereum. The system collects data from CoinGecko's API, processes it, trains various machine learning models, and generates predictions for different time horizons.

## Project Structure

```
├── crypto.py              # Data collection module using CoinGecko API
├── data_processor.py      # Data processing and feature engineering
├── model_trainer.py       # Training ML models and evaluating performance
├── predict.py             # Making predictions with trained models
├── models/                # Directory for storing trained models
│   ├── bitcoin/
│   └── ethereum/
├── processed_data/        # Directory for processed datasets
│   ├── bitcoin/
│   └── ethereum/
├── predictions/           # Output predictions in JSON format
│   ├── bitcoin/
│   └── ethereum/
├── plots/                 # Visualizations of predictions and model performance
│   ├── bitcoin/
│   └── ethereum/
└── backtest_results/      # Results from backtesting models
    ├── bitcoin/
    └── ethereum/
```

## Prerequisites

This project requires the following Python packages:
- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- requests
- joblib

You can install these packages using pip:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib requests joblib
```

## How to Run the System

The system must be run in a specific order. Follow these steps to execute the complete pipeline:

### 1. Data Collection (`crypto.py`)

This script collects historical cryptocurrency data from the CoinGecko API.

```bash
python crypto.py
```

**What happens:**
- Connects to CoinGecko's Premium API
- Fetches historical price data for Bitcoin and Ethereum
- Retrieves additional information like market cap, volume, and community metrics
- Calculates basic technical indicators
- Outputs raw data for use in subsequent processing

**Output:**
- Displays sample data and confirmation of successful API connections
- No files are created at this stage; data is passed to the next step

### 2. Data Processing (`data_processor.py`)

This script processes the raw data, engineers features, and prepares datasets for model training.

```bash
python data_processor.py
```

**What happens:**
- Handles missing values
- Creates technical indicators and additional features
- Generates target variables for different prediction horizons
- Normalizes/scales features
- Splits data into training and testing sets
- Saves processed datasets to disk

**Output:**
- Creates directories and files in `processed_data/` 
- Summary of preprocessed data dimensions and features

### 3. Model Training (`model_trainer.py`)

This script trains multiple machine learning models and evaluates their performance.

```bash
python model_trainer.py
```

**What happens:**
- Trains regression models (Linear Regression, Random Forest, MLP, LSTM) for price changes
- Trains classification models (Logistic Regression, Random Forest, MLP, LSTM) for direction prediction
- Performs model evaluation using metrics like RMSE, MAE, R², accuracy, F1 score
- Creates performance visualizations
- Performs backtesting on historical data
- Selects the best performing models
- Saves trained models to disk

**Output:**
- Creates `models/` directory with trained model files
- Creates performance plots in `plots/` directory
- Creates backtesting results in `backtest_results/` directory
- Detailed performance metrics in the console output

### 4. Making Predictions (`predict.py`)

This script loads the trained models and generates predictions for future price movements.

```bash
python predict.py
```

**What happens:**
- Loads the trained models
- Fetches the most recent data for Bitcoin and Ethereum
- Preprocesses the current data
- Generates predictions for multiple time horizons (1, 3, 7, 14, and 30 days)
- Creates visualizations of predictions
- Saves prediction results in JSON format

**Output:**
- Creates prediction visualization in `plots/[coin]/predictions/`
- Saves prediction data to `predictions/[coin]/` in JSON format
- Displays formatted prediction results in the console

## Understanding the Predictions

The system provides two types of predictions:

1. **Price Change Predictions**: Percentage and absolute price changes for different time horizons
2. **Directional Predictions**: Whether the price will go up or down, with a confidence score

For example, a typical prediction output looks like:

```
Bitcoin PRICE PREDICTIONS (2025-04-02 17:34:43)
Current price: $84502.46
--------------------------------------------------
SHORT-TERM PREDICTIONS:
Tomorrow: $83713.85 (-0.93%) - Direction: up (57.0% confidence)
3 days: $87317.30 (3.33%) - Direction: up (53.0% confidence)
7 days: $82710.41 (-2.12%) - Direction: up (65.8% confidence)
--------------------------------------------------
MEDIUM-TERM PREDICTIONS:
14 days: $87783.23 (3.88%) - Direction: up (77.1% confidence)
30 days: $87154.74 (3.14%) - Direction: up (71.1% confidence)
```

## Model Performance and Limitations

The system evaluates models using several metrics:

- **Regression Models**: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), R² score
- **Classification Models**: Accuracy, Precision, Recall, F1 score

It's important to note that cryptocurrency price prediction is inherently difficult due to market volatility and external factors. The models aim to identify patterns, but predictions should not be used as financial advice.

## Customization Options

You can modify various aspects of the system:

- **Change cryptocurrencies**: Edit the `CRYPTO_IDS` list in each script
- **Adjust time horizons**: Modify the `TARGET_HORIZONS` in `data_processor.py`
- **Fine-tune models**: Edit hyperparameters in the model creation functions in `model_trainer.py`
- **Feature engineering**: Add or remove features in `calculate_technical_indicators()` function in `crypto.py`

## Starting Over

To reset the system and start from scratch, delete the generated directories:

```bash
rm -rf predictions/ plots/ backtest_results/ models/ processed_data/ __pycache__/
```

Then run the scripts in order as described above.

## Troubleshooting

- **API errors**: Check your CoinGecko API key in `crypto.py`
- **Missing dependencies**: Ensure all required Python packages are installed
- **Memory issues**: Reduce the amount of historical data or simplify models if facing memory constraints

## Extending the System

The modular design allows for easy extension:
- Add new cryptocurrencies by updating the `CRYPTO_IDS` list
- Implement new machine learning models in `model_trainer.py`
- Add more technical indicators and features in `crypto.py` and `data_processor.py`
- Create additional visualization types in `predict.py`

## Disclaimer

This system is for educational purposes only. Cryptocurrency markets are highly volatile, and predictions should not be considered financial advice. Always do your own research before making investment decisions. 