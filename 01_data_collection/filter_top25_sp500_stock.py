# python 01_data_collection/filter_top25_sp500_stock.py
import pandas as pd
import os
import logging
from datetime import datetime
import pytz

# Set up logging
today = datetime.now(pytz.UTC)
timestamp = today.strftime('%Y%m%d_%H%M%S')
LOG_DIR = 'log'
LOG_FILE = f'{LOG_DIR}/filter_sp500_data_{timestamp}.log'

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
INPUT_FILE = 'data/daily_stock_price/sp500_daily_data_yfinance.csv'
OUTPUT_DIR = 'data/daily_stock_price'
OUTPUT_FILE = f'{OUTPUT_DIR}/sp500_top25_daily_data.csv'

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# List of top 25 company symbols from the provided CSV
TOP_25_SYMBOLS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'META', 'BRK-B', 'GOOGL', 'AVGO', 
    'TSLA', 'LLY', 'JPM', 'WMT', 'V', 'XOM', 'MA', 'UNH', 'ORCL', 'NFLX', 
    'COST', 'JNJ', 'PG', 'ABBV', 'HD', 'BAC'
]

def filter_stock_data():
    try:
        logger.info(f"Loading data from {INPUT_FILE}")
        print(f"Loading data from {INPUT_FILE}")
        
        # Read the input CSV (considering its large size, use chunks if needed)
        df = pd.read_csv(INPUT_FILE)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        
        logger.info(f"Filtering data for {len(TOP_25_SYMBOLS)} symbols")
        print(f"Filtering data for {len(TOP_25_SYMBOLS)} symbols")
        
        # Filter for only the specified symbols
        filtered_df = df[df['symbol'].isin(TOP_25_SYMBOLS)]
        
        # Log the number of rows before and after filtering
        logger.info(f"Original data: {len(df)} rows")
        logger.info(f"Filtered data: {len(filtered_df)} rows")
        print(f"Original data: {len(df)} rows")
        print(f"Filtered data: {len(filtered_df)} rows")
        
        # Save the filtered data to a new CSV
        filtered_df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Filtered data saved to {OUTPUT_FILE}")
        print(f"Filtered data saved to {OUTPUT_FILE}")
        
    except FileNotFoundError:
        logger.error(f"Input file {INPUT_FILE} not found")
        print(f"Error: Input file {INPUT_FILE} not found")
        exit(1)
    except Exception as e:
        logger.error(f"Error during filtering: {e}")
        print(f"Error during filtering: {e}")
        exit(1)

if __name__ == "__main__":
    logger.info("Starting data filtering process")
    print("Starting data filtering process")
    filter_stock_data()
    logger.info("Data filtering process completed")
    print("Data filtering process completed")