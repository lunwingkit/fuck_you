import pandas as pd
from tabulate import tabulate
from datetime import timedelta

# File path
file_path = 'data/daily_stock_price/sp500_top25_technical_indicators.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Ensure 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Group by symbol to find start and end dates
periods = df.groupby('symbol').agg(
    Start_Date=('date', 'min'),
    End_Date=('date', 'max'),
    Record_Count=('date', 'count')
).reset_index()

# Function to check for missing trading days
def check_missing_dates(start_date, end_date, dates):
    # Generate all weekdays (Monday to Friday) in the period
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' for business days
    # Convert actual dates to set for comparison
    actual_dates = set(dates)
    # Find missing dates
    missing_dates = [d for d in all_dates if d not in actual_dates]
    return len(missing_dates), missing_dates

# Analyze missing data for each symbol
missing_info = []
for symbol in periods['symbol']:
    symbol_data = df[df['symbol'] == symbol]
    start_date = periods[periods['symbol'] == symbol]['Start_Date'].iloc[0]
    end_date = periods[periods['symbol'] == symbol]['End_Date'].iloc[0]
    num_missing, missing_dates = check_missing_dates(start_date, end_date, symbol_data['date'])
    missing_info.append({
        'Symbol': symbol,
        'Start_Date': start_date,
        'End_Date': end_date,
        'Record_Count': len(symbol_data),
        'Missing_Days': num_missing,
        'Missing_Dates': missing_dates if num_missing > 0 else 'None'
    })

# Create DataFrame for results
results = pd.DataFrame(missing_info)

# Format dates for display
results['Start_Date'] = results['Start_Date'].dt.strftime('%Y-%m-%d')
results['End_Date'] = results['End_Date'].dt.strftime('%Y-%m-%d')

# Display the table
print(tabulate(results[['Symbol', 'Start_Date', 'End_Date', 'Record_Count', 'Missing_Days']],
               headers=['Symbol', 'Start Date', 'End Date', 'Record Count', 'Missing Days'],
               tablefmt='psql', showindex=False))

# Print detailed missing dates if any
for index, row in results.iterrows():
    if row['Missing_Days'] > 0:
        print(f"\nMissing dates for {row['Symbol']} ({row['Missing_Days']} days):")
        print(row['Missing_Dates'])