#  python 01_data_collection/compare.py
import pandas as pd
from tabulate import tabulate

# File paths
file1 = 'data/daily_stock_price/sp500_top25_daily_data.csv'
file2 = 'data/daily_stock_price/sp500_top25_technical_indicators.csv'

# Read the CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Count daily records per symbol for each file
count1 = df1.groupby('symbol').size().reset_index(name='Daily_Count_File1')
count2 = df2.groupby('symbol').size().reset_index(name='Daily_Count_File2')

# Merge the counts on symbol
merged_counts = pd.merge(count1, count2, on='symbol', how='outer').fillna(0)

# Convert counts to integers
merged_counts['Daily_Count_File1'] = merged_counts['Daily_Count_File1'].astype(int)
merged_counts['Daily_Count_File2'] = merged_counts['Daily_Count_File2'].astype(int)

# Display the table
print(tabulate(merged_counts, headers=['Symbol', 'Daily Count (sp500_top25_daily_data)', 
                                      'Daily Count (sp500_top25_technical_indicators)'], 
               tablefmt='psql', showindex=False))