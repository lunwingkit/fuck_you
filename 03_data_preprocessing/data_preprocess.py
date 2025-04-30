# python 03_data_preprocessing/data_preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats.mstats import winsorize
import os

# Load data
df = pd.read_csv('data/daily_stock_price/sp500_top25_technical_indicators.csv')

# Create a copy to avoid modifying the original data
df_processed = df.copy()
df_processed['date'] = pd.to_datetime(df_processed['date'])

# 1. Handle missing values
# Forward-fill within each symbol
for col in df_processed.columns:
    if col != 'symbol':  # Exclude the grouping column
        df_processed[col] = df_processed.groupby('symbol')[col].ffill()
# Drop remaining missing values (e.g., initial rows where ffill can't be applied)
df_processed = df_processed.dropna()

# 2. Sort by symbol and date
df_processed = df_processed.sort_values(by=['symbol', 'date'])

# 3. Create target variable for regression (next day's Close price)
# Shift the Close price by -1 within each symbol to get the next day's price
df_processed['Next_Close'] = df_processed.groupby('symbol')['Close'].shift(-1)
# Drop rows where Next_Close is NaN (last row for each symbol)
df_processed = df_processed.dropna(subset=['Next_Close'])

# 4. Feature scaling
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'MACD', 
                'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower', 'ATR_14', 
                'OBV', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5', 
                'Volume_Lag_1', 'Volume_Lag_3', 'Volatility_20', 
                'High_Low_Range', 'Open_Close_Range', 'MACD_Hist_Slope']
scaler = StandardScaler()
df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

# Min-Max scale RSI_14 and Daily_Return
minmax_scaler = MinMaxScaler()
df_processed[['RSI_14', 'Daily_Return']] = minmax_scaler.fit_transform(df_processed[['RSI_14', 'Daily_Return']])

# Log transform Volume and OBV
df_processed['Volume'] = np.log1p(df_processed['Volume'])
df_processed['OBV'] = np.log1p(df_processed['OBV'].abs())

# 5. Encode categorical variables
df_processed = pd.get_dummies(df_processed, columns=['symbol'], prefix='symbol')

# 6. Apply PCA to price-related features
price_features = [col for col in df_processed.columns if col in ['Open', 'High', 'Low', 'Close', 'SMA_20', 'BB_Upper', 'BB_Lower', 
                                                                'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5']]
n_features = len(price_features)
n_components = min(3, n_features)  # Ensure n_components <= n_features
if n_features > 1:
    pca = PCA(n_components=n_components)
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(pca.fit_transform(df_processed[price_features]), 
                          columns=pca_columns)
    df_processed = df_processed.drop(price_features, axis=1)
    df_processed = pd.concat([df_processed, df_pca], axis=1)
else:
    print(f"Warning: Only {n_features} price feature(s) available for PCA. Skipping PCA step.")

# 7. Remove highly correlated features
corr_matrix = df_processed.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df_processed = df_processed.drop(to_drop, axis=1)

# 8. Handle outliers
df_processed['MACD'] = winsorize(df_processed['MACD'], limits=[0.01, 0.01])
df_processed['ATR_14'] = winsorize(df_processed['ATR_14'], limits=[0.01, 0.01])
df_processed['Daily_Return'] = winsorize(df_processed['Daily_Return'], limits=[0.01, 0.01])
df_processed['Daily_Return'] = df_processed['Daily_Return'].clip(lower=-0.3, upper=0.3)

# 9. Feature engineering
df_processed['Volatility_Ratio'] = df_processed['Volatility_20'] / df_processed['PC1']  # Use PC1 as proxy for price
df_processed['RSI_14_Change'] = df_processed['RSI_14'].diff()
df_processed['Trend_Direction'] = (df_processed['MACD_Hist_Slope'] > 0).astype(int)
df_processed['Day_of_Week'] = df_processed['date'].dt.dayofweek
df_processed['Month'] = df_processed['date'].dt.month
df_processed['Year'] = df_processed['date'].dt.year

# Handle missing values in new features
df_processed['RSI_14_Change'] = df_processed['RSI_14_Change'].fillna(0)  # Fill first row with 0
df_processed['Volatility_Ratio'] = df_processed['Volatility_Ratio'].fillna(df_processed['Volatility_Ratio'].mean())  # Fill with mean

# 10. Time-based train-test split
train_df = df_processed[df_processed['date'] < '2021-01-01']
test_df = df_processed[df_processed['date'] >= '2021-01-01']

# Combine train and test data for saving
final_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# 11. Save the preprocessed data to a new file
output_dir = 'data/daily_stock_price'
output_file = f'{output_dir}/sp500_top25_stock_data_postprocessed.csv'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the preprocessed data
final_df.to_csv(output_file, index=False)

# Final check
print("Preprocessing completed. Preprocessed data saved to:", output_file)
print("Final dataset shape:", final_df.shape)
print("Missing values in final dataset:", final_df.isnull().sum())