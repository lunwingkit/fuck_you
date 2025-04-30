# python 01_data_collection/fetch_from_yfinance_ti_top25sp500.py
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import pytz
import yfinance as yf

# Try to import TA-Lib, fallback to custom indicators if not available
try:
    import talib
    USE_TALIB = True
except ImportError:
    USE_TALIB = False
    print("TA-Lib not installed, using custom indicator calculations.")

# Set up logging
today = datetime.now(pytz.UTC)
timestamp = today.strftime('%Y%m%d_%H%M%S')
LOG_DIR = 'log'
LOG_FILE = f'{LOG_DIR}/download_sp500_top25_techind_{timestamp}.log'

# Create log directory if it doesn't exist
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Output folder
OUTPUT_DIR = 'data/daily_stock_price'
OUTPUT_FILE = f'{OUTPUT_DIR}/sp500_top25_technical_indicators.csv'

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# List of top 25 company symbols
TOP_25_SYMBOLS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'META', 'BRK-B', 'GOOGL', 'AVGO',
    'TSLA', 'LLY', 'JPM', 'WMT', 'V', 'XOM', 'MA', 'UNH', 'ORCL', 'NFLX',
    'COST', 'JNJ', 'PG', 'ABBV', 'HD', 'BAC'
]

# Custom indicator functions (used if TA-Lib is not available)
def custom_sma(series, period=20):
    return series.rolling(window=period).mean()

def custom_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def custom_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def custom_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band

def custom_atr(high, low, close, period=14):
    # Calculate true range (TR)
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def custom_obv(close, volume):
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()
    return obv

def download_and_calculate_indicators():
    try:
        logger.info("Starting data download for top 25 S&P500 stocks")
        print("Starting data download for top 25 S&P500 stocks")

        all_data = []

        for symbol in TOP_25_SYMBOLS:
            logger.info(f"Downloading data for {symbol}")
            print(f"Downloading data for {symbol}")

            yf_symbol = symbol.replace('BRK-B', 'BRK-B')  # Yahoo uses BRK-B as-is

            # Download data
            df = yf.download(
                yf_symbol,
                start='2000-01-01',
                interval='1d',
                auto_adjust=True,
                progress=False
            )

            if df.empty:
                logger.warning(f"No data found for {symbol}, skipping.")
                print(f"No data found for {symbol}, skipping.")
                continue

            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df = df[[('Close', symbol), ('High', symbol), ('Low', symbol), ('Open', symbol), ('Volume', symbol)]]
                df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
            else:
                df = df[['Close', 'High', 'Low', 'Open', 'Volume']]

            # Ensure index is DatetimeIndex
            df.index.name = 'Date'
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Log DataFrame structure
            logger.info(f"DataFrame shape for {symbol}: {df.shape}")
            logger.info(f"Columns for {symbol}: {list(df.columns)}")
            logger.info(f"Index type for {symbol}: {type(df.index)}")

            # Add technical indicators
            try:
                if USE_TALIB:
                    # TA-Lib indicators
                    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
                    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
                    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(
                        df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
                    )
                    df['BB_Upper'], df['BB_Lower'] = talib.BBANDS(
                        df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2
                    )[1:]
                    df['ATR_14'] = talib.ATR(
                        df['High'], df['Low'], df['Close'], timeperiod=14
                    )
                    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
                    logger.info(f"TA-Lib indicators calculated for {symbol}")
                else:
                    # Custom indicators
                    df['SMA_20'] = custom_sma(df['Close'], period=20)
                    df['RSI_14'] = custom_rsi(df['Close'], period=14)
                    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = custom_macd(
                        df['Close'], fast=12, slow=26, signal=9
                    )
                    df['BB_Upper'], df['BB_Lower'] = custom_bollinger_bands(
                        df['Close'], period=20, std_dev=2
                    )
                    df['ATR_14'] = custom_atr(
                        df['High'], df['Low'], df['Close'], period=14
                    )
                    df['OBV'] = custom_obv(df['Close'], df['Volume'])
                    logger.info(f"Custom indicators calculated for {symbol}")

                # Additional features
                # Lagged values
                df['Close_Lag_1'] = df['Close'].shift(1)
                df['Close_Lag_2'] = df['Close'].shift(2)
                df['Close_Lag_3'] = df['Close'].shift(3)
                df['Close_Lag_5'] = df['Close'].shift(5)
                df['Volume_Lag_1'] = df['Volume'].shift(1)
                df['Volume_Lag_3'] = df['Volume'].shift(3)

                # Daily return
                df['Daily_Return'] = df['Close'].pct_change()

                # Volatility
                df['Volatility_20'] = df['Close'].rolling(window=20).std()

                # Price ranges
                df['High_Low_Range'] = df['High'] - df['Low']
                df['Open_Close_Range'] = df['Close'] - df['Open']

                # MACD histogram slope
                df['MACD_Hist_Slope'] = df['MACD_Hist'].diff()

            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
                print(f"Error calculating indicators for {symbol}: {e}")
                continue

            df['symbol'] = symbol
            df['date'] = df.index

            # Reset index and rearrange columns
            df = df.reset_index(drop=True)
            cols = [
                'symbol', 'date', 'Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
                'BB_Upper', 'BB_Lower', 'ATR_14', 'OBV',
                'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5',
                'Volume_Lag_1', 'Volume_Lag_3', 'Daily_Return', 'Volatility_20',
                'High_Low_Range', 'Open_Close_Range', 'MACD_Hist_Slope'
            ]
            df = df[cols]

            all_data.append(df)

            logger.info(f"Data and indicators processed for {symbol}")
            print(f"Data and indicators processed for {symbol}")

        # Combine all stocks
        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            result_df.to_csv(OUTPUT_FILE, index=False)
            logger.info(f"Data saved to {OUTPUT_FILE}")
            print(f"Data saved to {OUTPUT_FILE}")
        else:
            logger.error("No data collected for any symbol.")
            print("No data collected for any symbol.")

    except Exception as e:
        logger.error(f"Error during data download: {e}")
        print(f"Error during data download: {e}")
        exit(1)

if __name__ == "__main__":
    logger.info("Script started")
    print("Script started")
    download_and_calculate_indicators()
    logger.info("Script finished")
    print("Script finished")