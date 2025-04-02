import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import random
import numpy as np

# CoinGecko Premium API key
API_KEY = "CG-fzbWUrBfnpBp5RMBZrJvWoMQ"
# Base URL for CoinGecko Pro API
PRO_API_BASE_URL = "https://pro-api.coingecko.com/api/v3"

def make_api_request(endpoint, params=None, max_retries=3, delay=1.0):
    """
    Make API request with retry mechanism and rate limiting
    
    Args:
        endpoint (str): API endpoint path (without base URL)
        params (dict, optional): Request parameters. Defaults to None.
        max_retries (int, optional): Maximum number of retries. Defaults to 3.
        delay (float, optional): Base delay between retries in seconds. Defaults to 1.0.
        
    Returns:
        dict: JSON response from the API
    """
    # Construct full URL
    url = f"{PRO_API_BASE_URL}/{endpoint}"
    
    # Add API key to header
    headers = {"x-cg-pro-api-key": API_KEY}
    
    # Initialize parameters if None
    if params is None:
        params = {}
    
    # Try up to max_retries times
    for attempt in range(max_retries):
        try:
            # Add a small random delay to avoid hitting rate limits
            if attempt > 0:
                # Exponential backoff with jitter
                sleep_time = delay * (2 ** attempt) + random.uniform(0, 0.5)
                print(f"Retrying in {sleep_time:.2f} seconds (attempt {attempt+1}/{max_retries})...")
                time.sleep(sleep_time)
                
            # Make the API request
            response = requests.get(url, params=params, headers=headers)
            
            # If successful, return the JSON response
            if response.status_code == 200:
                return response.json()
                
            # If rate limited, wait and retry
            elif response.status_code == 429:
                print(f"Rate limited. Waiting before retry...")
                # Wait longer for rate limit errors
                time.sleep(delay * 3)
                continue
                
            # Handle other errors
            else:
                print(f"Error: {response.status_code} - {response.text}")
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Error accessing CoinGecko API after {max_retries} attempts: {e}")
                return None
            print(f"Request failed: {e}")
    
    return None  # If we get here, all retries failed

def get_historical_data(crypto_id, days):
    """
    Collect historical cryptocurrency price data from CoinGecko API
    
    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum')
        days (int): Number of days of historical data to retrieve
        
    Returns:
        DataFrame: Pandas DataFrame with price and market data
    """
    # CoinGecko API endpoint for market chart data
    endpoint = f"coins/{crypto_id}/market_chart"
    
    # Parameters for the API request
    params = {
        'vs_currency': 'usd',  # Price in USD
        'days': days,          # Number of days of data
        'interval': 'daily'    # Daily data points
    }
    
    # Make API request with retry mechanism
    data = make_api_request(endpoint, params)
    
    if data is None:
        print(f"Failed to fetch historical data for {crypto_id}")
        return None
    
    try:
        # Get price, market cap, and volume data
        price_data = data['prices']
        market_cap_data = data['market_caps']
        volume_data = data['total_volumes']
        
        # Create DataFrames for each data type
        df_price = pd.DataFrame(price_data, columns=['timestamp', 'price'])
        df_market_cap = pd.DataFrame(market_cap_data, columns=['timestamp', 'market_cap'])
        df_volume = pd.DataFrame(volume_data, columns=['timestamp', 'volume'])
        
        # Convert timestamps to datetime
        df_price['timestamp'] = pd.to_datetime(df_price['timestamp'], unit='ms')
        
        # Merge all DataFrames on timestamp
        # First merge price and market cap
        df = pd.merge(df_price, df_market_cap.drop('timestamp', axis=1), 
                     left_index=True, right_index=True)
        
        # Then merge volume
        df = pd.merge(df, df_volume.drop('timestamp', axis=1),
                     left_index=True, right_index=True)
        
        # Add additional features that might be useful for ML models
        # Daily price change (%)
        df['price_change_pct'] = df['price'].pct_change() * 100
        
        # Daily volume change (%)
        df['volume_change_pct'] = df['volume'].pct_change() * 100
        
        # Market cap to volume ratio (indicates if price movement is backed by volume)
        df['mcap_to_volume_ratio'] = df['market_cap'] / df['volume']
        
        # 7-day moving averages
        df['price_ma7'] = df['price'].rolling(window=7).mean()
        df['volume_ma7'] = df['volume'].rolling(window=7).mean()
        
        # Volatility (standard deviation over 7 days)
        df['volatility'] = df['price'].rolling(window=7).std()
        
        # Reset index to prepare for future operations
        df = df.reset_index(drop=True)
        
        return df
        
    except (KeyError, ValueError) as e:
        print(f"Error processing data for {crypto_id}: {e}")
        return None

def get_additional_coin_data(crypto_id):
    """
    Get additional data about a cryptocurrency that might be useful for predictions
    
    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum')
        
    Returns:
        dict: Dictionary containing additional data
    """
    endpoint = f"coins/{crypto_id}"
    
    # Make API request with retry mechanism
    data = make_api_request(endpoint)
    
    if data is None:
        print(f"Failed to fetch additional data for {crypto_id}")
        return None
    
    try:
        # Extract relevant data with safe get operations to avoid KeyErrors
        additional_data = {
            'name': data.get('name', ''),
            'symbol': data.get('symbol', ''),
            'market_cap_rank': data.get('market_cap_rank', None)
        }
        
        # Safely get developer data
        developer_data = data.get('developer_data', {})
        additional_data.update({
            'github_forks': developer_data.get('forks', 0),
            'github_stars': developer_data.get('stars', 0),
            'github_subscribers': developer_data.get('subscribers', 0),
            'github_total_issues': developer_data.get('total_issues', 0),
            'github_closed_issues': developer_data.get('closed_issues', 0)
        })
        
        # Safely get community data
        community_data = data.get('community_data', {})
        additional_data.update({
            'community_score': community_data.get('community_score', 0)
        })
        
        # Other metrics
        additional_data.update({
            'sentiment_votes_up_percentage': data.get('sentiment_votes_up_percentage', 0),
            'coingecko_score': data.get('coingecko_score', 0),
            'developer_score': data.get('developer_score', 0),
            'liquidity_score': data.get('liquidity_score', 0)
        })
        
        return additional_data
        
    except Exception as e:
        print(f"Error processing additional data for {crypto_id}: {e}")
        return None

def get_market_trends():
    """
    Get global cryptocurrency market trends and data
    
    Returns:
        dict: Dictionary containing global market data
    """
    global_endpoint = "global"
    trending_endpoint = "search/trending"
    
    market_data = {}
    
    # Make API request for global data with retry mechanism
    global_data = make_api_request(global_endpoint)
    
    if global_data is None:
        print("Failed to fetch global market data")
        return None
    
    try:
        # Extract global data
        global_data = global_data.get('data', {})
        
        market_data = {
            'total_market_cap_usd': global_data.get('total_market_cap', {}).get('usd', 0),
            'total_volume_usd': global_data.get('total_volume', {}).get('usd', 0),
            'market_cap_percentage': {
                'btc': global_data.get('market_cap_percentage', {}).get('btc', 0),
                'eth': global_data.get('market_cap_percentage', {}).get('eth', 0)
            },
            'market_cap_change_percentage_24h_usd': global_data.get('market_cap_change_percentage_24h_usd', 0),
            'active_cryptocurrencies': global_data.get('active_cryptocurrencies', 0),
            'markets': global_data.get('markets', 0)
        }
        
        # Add a small delay before making another API request
        time.sleep(0.5)
        
        # Make API request for trending data with retry mechanism
        trending_data = make_api_request(trending_endpoint)
        
        if trending_data is not None:
            # Extract trending coins
            trending_coins = []
            for coin in trending_data.get('coins', []):
                item = coin.get('item', {})
                trending_coins.append({
                    'id': item.get('id', ''),
                    'name': item.get('name', ''),
                    'symbol': item.get('symbol', ''),
                    'market_cap_rank': item.get('market_cap_rank', 0)
                })
            
            market_data['trending_coins'] = trending_coins
        
        return market_data
        
    except Exception as e:
        print(f"Error processing market trends data: {e}")
        return None

def get_external_factors(from_date, to_date=None):
    """
    Get external factors that might affect cryptocurrency prices.
    This is a placeholder function - in a real application, you might fetch:
    - Stock market indices (S&P 500, NASDAQ)
    - Gold prices
    - Currency exchange rates
    - Interest rates
    - Economic indicators
    
    Args:
        from_date (str): Start date in format 'YYYY-MM-DD'
        to_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to current date.
        
    Returns:
        DataFrame: DataFrame containing external factors
    """
    # Note: This is a placeholder. In a real application, you would:
    # 1. Use APIs like Alpha Vantage, Yahoo Finance, or FRED
    # 2. Collect and process the data
    # 3. Return a DataFrame with the external factors
    
    if to_date is None:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    # Create a sample DataFrame with dates
    start = datetime.strptime(from_date, '%Y-%m-%d')
    end = datetime.strptime(to_date, '%Y-%m-%d')
    
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    
    df = pd.DataFrame({'date': dates})
    
    # Add placeholder columns for external factors
    # In a real application, these would contain actual data
    df['sp500'] = 0
    df['nasdaq'] = 0
    df['gold_price'] = 0
    df['usd_index'] = 0
    df['interest_rate'] = 0
    
    return df

def calculate_technical_indicators(df):
    """
    Calculate technical indicators based on price data
    
    Args:
        df (DataFrame): DataFrame with price data
        
    Returns:
        DataFrame: DataFrame with technical indicators added
    """
    # Make a copy to avoid modifying the original
    df_with_indicators = df.copy()
    
    # Create price series for calculations
    price = df_with_indicators['price']
    
    # --- Moving Averages ---
    # Add 20-day and 50-day moving averages
    df_with_indicators['ma20'] = price.rolling(window=20).mean()
    df_with_indicators['ma50'] = price.rolling(window=50).mean()
    
    # --- RSI (Relative Strength Index) ---
    # Calculate daily price changes
    delta = price.diff()
    
    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Calculate average gain and loss over 14 periods (typical for RSI)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    df_with_indicators['rsi'] = 100 - (100 / (1 + rs))
    
    # --- MACD (Moving Average Convergence Divergence) ---
    # Calculate the MACD line (12-day EMA - 26-day EMA)
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    
    # Calculate the signal line (9-day EMA of MACD line)
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    # Calculate the MACD histogram (MACD line - signal line)
    df_with_indicators['macd'] = macd_line
    df_with_indicators['macd_signal'] = signal_line
    df_with_indicators['macd_histogram'] = macd_line - signal_line
    
    # --- Bollinger Bands ---
    # Calculate 20-day moving average and standard deviation
    middle_band = price.rolling(window=20).mean()
    std_dev = price.rolling(window=20).std()
    
    # Calculate upper and lower bands (typically 2 standard deviations)
    df_with_indicators['bollinger_middle'] = middle_band
    df_with_indicators['bollinger_upper'] = middle_band + (std_dev * 2)
    df_with_indicators['bollinger_lower'] = middle_band - (std_dev * 2)
    
    # --- Stochastic Oscillator ---
    # Look back period (typically 14 days)
    period = 14
    
    # Find the lowest low and highest high in the period
    low_min = price.rolling(window=period).min()
    high_max = price.rolling(window=period).max()
    
    # Calculate %K (current close - lowest low) / (highest high - lowest low) * 100
    k_percent = 100 * ((price - low_min) / (high_max - low_min))
    
    # Calculate %D (3-day SMA of %K)
    d_percent = k_percent.rolling(window=3).mean()
    
    df_with_indicators['stoch_k'] = k_percent
    df_with_indicators['stoch_d'] = d_percent
    
    # --- Average True Range (ATR) ---
    high = price * 1.01  # Simulating high price (since we don't have actual high/low)
    low = price * 0.99   # Simulating low price
    
    # Calculate true range
    tr1 = high - low
    tr2 = abs(high - price.shift())
    tr3 = abs(low - price.shift())
    
    # True range is the maximum of these three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR (14-day average of true range)
    df_with_indicators['atr'] = tr.rolling(window=14).mean()
    
    # --- On-Balance Volume (OBV) ---
    volume = df_with_indicators['volume']
    obv = np.zeros(len(price))
    for i in range(1, len(price)):
        if price.iloc[i] > price.iloc[i-1]:
            obv[i] = obv[i-1] + volume.iloc[i]
        elif price.iloc[i] < price.iloc[i-1]:
            obv[i] = obv[i-1] - volume.iloc[i]
        else:
            obv[i] = obv[i-1]
    
    df_with_indicators['obv'] = obv
    
    # --- Price Rate of Change (ROC) ---
    # Calculate 14-day rate of change
    df_with_indicators['price_roc'] = price.pct_change(periods=14) * 100
    
    # --- Commodity Channel Index (CCI) ---
    # Typical price (close since we don't have high/low)
    typical_price = price
    
    # Calculate 20-day SMA of typical price
    sma_tp = typical_price.rolling(window=20).mean()
    
    # Calculate mean deviation
    mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    
    # Calculate CCI
    df_with_indicators['cci'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # --- Williams %R ---
    # Calculate Williams %R for 14-day period
    df_with_indicators['williams_r'] = -100 * ((high_max - price) / (high_max - low_min))
    
    return df_with_indicators

def get_historical_data_with_indicators(crypto_id, days):
    """
    Get historical data including calculated technical indicators
    
    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., 'bitcoin')
        days (int): Number of days of historical data
        
    Returns:
        DataFrame: DataFrame with price data and technical indicators
    """
    # Get base historical data
    df = get_historical_data(crypto_id, days)
    
    if df is None:
        return None
    
    # Add technical indicators
    df_with_indicators = calculate_technical_indicators(df)
    
    return df_with_indicators

def collect_comprehensive_data(crypto_id, days):
    """
    Collect comprehensive data for cryptocurrency price prediction
    
    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., 'bitcoin')
        days (int): Number of days of historical data
        
    Returns:
        dict: Dictionary containing all collected data
    """
    # Get historical price data with technical indicators
    price_data = get_historical_data_with_indicators(crypto_id, days)
    
    # Get additional coin data
    coin_data = get_additional_coin_data(crypto_id)
    
    # Get market trends
    market_trends = get_market_trends()
    
    # Calculate start date for external factors (based on days parameter)
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Get external factors
    external_factors = get_external_factors(start_date)
    
    # Combine all data
    comprehensive_data = {
        'price_data': price_data,
        'coin_data': coin_data,
        'market_trends': market_trends,
        'external_factors': external_factors
    }
    
    return comprehensive_data

# Example usage
if __name__ == "__main__":
    print("Using CoinGecko Premium API with key:", API_KEY)
    print(f"Base URL: {PRO_API_BASE_URL}")
    print("Collecting cryptocurrency data...\n")
    
    # Test API connectivity by pinging the server
    print("Testing API connection...")
    ping_response = make_api_request("ping")
    if ping_response is not None:
        print("✅ API connection successful!")
    else:
        print("❌ API connection failed!")
    
    # Get 30 days of Bitcoin price data
    print("\nFetching Bitcoin historical data...")
    bitcoin_data = get_historical_data('bitcoin', 30)
    
    if bitcoin_data is not None:
        print("Bitcoin price data sample:")
        print(bitcoin_data.head())
        
        # Add a small delay before making another API request
        time.sleep(1)
        
        # Show example with technical indicators
        print("\nFetching Bitcoin data with technical indicators...")
        bitcoin_indicators = get_historical_data_with_indicators('bitcoin', 60)  # Need more days for indicators
        
        if bitcoin_indicators is not None:
            print("\nBitcoin data with technical indicators (sample):")
            # Select a subset of columns to display
            indicator_cols = ['timestamp', 'price', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']
            print(bitcoin_indicators[indicator_cols].iloc[30:35])  # Show after indicators have been calculated
            
        # Add a small delay before making another API request
        time.sleep(1)
            
        # Get 30 days of Ethereum price data
        print("\nFetching Ethereum historical data...")
        ethereum_data = get_historical_data('ethereum', 30)
        
        if ethereum_data is not None:
            print("\nEthereum price data sample:")
            print(ethereum_data.head())
            
        # Add a small delay before making another API request
        time.sleep(1)
            
        # Get additional data about Bitcoin
        print("\nFetching additional Bitcoin data...")
        btc_additional = get_additional_coin_data('bitcoin')
        if btc_additional is not None:
            print("\nAdditional Bitcoin data:")
            for key, value in btc_additional.items():
                print(f"{key}: {value}")
                
        # Add a small delay before making another API request
        time.sleep(1)
                
        # Get market trends
        print("\nFetching global market trends...")
        trends = get_market_trends()
        if trends is not None:
            print("\nGlobal Market Trends:")
            for key, value in trends.items():
                if key != 'trending_coins':
                    print(f"{key}: {value}")
            
            print("\nTrending Coins:")
            for coin in trends.get('trending_coins', [])[:3]:  # Show top 3
                print(f"  {coin.get('name')} ({coin.get('symbol')})")
                
        # Add a small delay before making another API request
        time.sleep(1)
                
        # Test comprehensive data collection
        print("\nCollecting comprehensive data for Bitcoin (sample output)...")
        comprehensive_data = collect_comprehensive_data('bitcoin', 7)
        if comprehensive_data is not None:
            print("✅ Comprehensive data collected successfully!")
