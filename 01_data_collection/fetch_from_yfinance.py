# python 01_data_collection/fetch_from_yfinance.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import logging
import pytz

# Set up logging to a file with dynamic timestamp
today = datetime.now(pytz.UTC)
timestamp = today.strftime('%Y%m%d_%H%M%S')
LOG_DIR = 'log'
LOG_FILE = f'{LOG_DIR}/fetch_yfinance_daily_{timestamp}.log'

# Create log directory if it doesn't exist
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# File paths
CONSTITUENTS_FILE = 'data/index/sp500_constituents.csv'
OUTPUT_DIR = 'data/daily_stock_price'
OUTPUT_FILE = f'{OUTPUT_DIR}/sp500_daily_data_yfinance.csv'

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Check if output file exists and load existing dates per symbol
if os.path.exists(OUTPUT_FILE):
    logger.info(f"Loading existing data from {OUTPUT_FILE}")
    existing_data = pd.read_csv(OUTPUT_FILE)
    existing_data['date'] = pd.to_datetime(existing_data['date'], utc=True)
    existing_dates_by_symbol = {
        symbol: set(existing_data[existing_data['symbol'] == symbol]['date'])
        for symbol in existing_data['symbol'].unique()
    }
else:
    logger.info(f"Creating new output file {OUTPUT_FILE}")
    existing_dates_by_symbol = {}
    pd.DataFrame(columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']).to_csv(OUTPUT_FILE, index=False)

# Load symbols from constituents file
try:
    symbols = pd.read_csv(CONSTITUENTS_FILE)['Symbol'].tolist()
except FileNotFoundError:
    logger.error(f"Constituents file {CONSTITUENTS_FILE} not found")
    print(f"Error: Constituents file {CONSTITUENTS_FILE} not found")
    exit(1)

# Determine the latest possible data point (yesterday relative to today)
end_date = (today - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
logger.info(f"Running job on {today.strftime('%Y-%m-%d')}, fetching missing data up to {end_date.strftime('%Y-%m-%d')}")

# Fetch and append data for each symbol
for symbol in symbols:
    print(f"Fetching daily data for {symbol}...")
    logger.info(f"Fetching daily data for {symbol}")
    try:
        # Determine the earliest date to fetch
        if symbol in existing_dates_by_symbol:
            latest_existing_date = max(existing_dates_by_symbol[symbol])
            start_date = latest_existing_date + timedelta(days=1)
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # If no data exists, fetch from the earliest available date
            start_date = None  # Use period='max' to get all data

        # Set end date to yesterday
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Skip fetch if start_date is after the end_date (no new data needed)
        if start_date and start_date > end_date:
            print(f"No new data needed for {symbol} up to {end_date.strftime('%Y-%m-%d')}")
            logger.info(f"No new data needed for {symbol} up to {end_date.strftime('%Y-%m-%d')}")
            continue

        # Fetch daily data using yfinance
        stock = yf.Ticker(symbol)
        if start_date:
            start_date_str = start_date.strftime('%Y-%m-%d')
            data = stock.history(start=start_date_str, end=end_date_str, interval='1d')
        else:
            # Fetch all available data if no prior data exists
            data = stock.history(period='max', interval='1d')

        # Rename columns to match your format
        data = data.reset_index()
        data = data.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        data['symbol'] = symbol
        data['date'] = pd.to_datetime(data['date'], utc=True)

        # Filter out any dates that might still exist (edge case)
        if symbol in existing_dates_by_symbol:
            data = data[~data['date'].isin(existing_dates_by_symbol[symbol])]

        # Append new data to CSV if there’s anything to add
        if not data.empty:
            header = not os.path.exists(OUTPUT_FILE) or os.stat(OUTPUT_FILE).st_size == 0
            data[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']].to_csv(
                OUTPUT_FILE, mode='a', header=header, index=False
            )
            print(f"Added {len(data)} new rows for {symbol} up to {end_date.strftime('%Y-%m-%d')}")
            logger.info(f"Added {len(data)} new rows for {symbol} up to {end_date.strftime('%Y-%m-%d')}")
            
            # Update in-memory set of dates
            if symbol not in existing_dates_by_symbol:
                existing_dates_by_symbol[symbol] = set()
            existing_dates_by_symbol[symbol].update(data['date'])
        else:
            print(f"No new data for {symbol} up to {end_date.strftime('%Y-%m-%d')}")
            logger.info(f"No new data for {symbol} up to {end_date.strftime('%Y-%m-%d')}")

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        logger.error(f"Error fetching {symbol}: {e}")

    # Small delay to avoid potential rate limiting
    time.sleep(0.5)

print(f"Data collection complete. All data saved to {OUTPUT_FILE}")
logger.info(f"Data collection complete. All data saved to {OUTPUT_FILE}")