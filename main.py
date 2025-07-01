import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import shap

# Load the dataset
file_path = 'data/default/TCS_stock_history.csv'
data = pd.read_csv(file_path)

print('Initial Data Overview:')
print(data.info())
print('\nMissing values per column:')
print(data.isnull().sum())

# Clean the data: Remove or fill null/NaN/missing values
# Convert columns to appropriate types if needed
for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Remove rows with any NaN values
cleaned_data = data.dropna()

print('\nData after cleaning:')
print(cleaned_data.info())
print('\nMissing values per column after cleaning:')
print(cleaned_data.isnull().sum())

# Save cleaned data for further steps
cleaned_data.to_csv('TCS_stock_history_cleaned.csv', index=False)
print('\nCleaned data saved to TCS_stock_history_cleaned.csv')

# --- Step 2: Exploratory Data Analysis (EDA) and Visualization ---

# Load the cleaned data
df = pd.read_csv('TCS_stock_history_cleaned.csv', parse_dates=['Date'])

# Price Trends: Visualize Open, Close, High, Low prices over time
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Open'], label='Open', color='#008080', alpha=0.7)      # Teal
plt.plot(df['Date'], df['Close'], label='Close', color='#FFD700', alpha=0.7)    # Gold
plt.plot(df['Date'], df['High'], label='High', color='#FF00FF', alpha=0.7)      # Magenta
plt.plot(df['Date'], df['Low'], label='Low', color='#808080', alpha=0.7)        # Gray
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('TCS Stock Price Trends (Open, Close, High, Low)')
plt.legend()
plt.tight_layout()
plt.show()

# Volume Analysis (check for outliers)
q1 = df['Volume'].quantile(0.25)
q3 = df['Volume'].quantile(0.75)
iqr = q3 - q1
outlier_threshold = q3 + 1.5 * iqr
num_outliers = (df['Volume'] > outlier_threshold).sum()
if num_outliers > 0:
    print(f"Volume column has {num_outliers} outlier(s) above {outlier_threshold:.0f}.")

plt.figure(figsize=(14, 5))
plt.plot(df['Date'], df['Volume'], color='#00CED1')  # Cyan
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('TCS Stock Trading Volume Over Time')
plt.tight_layout()
plt.show()

# Moving Averages (50-day and 200-day)
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Close'], label='Close Price', color='#FFD700')   # Gold
plt.plot(df['Date'], df['MA50'], label='50-Day MA', color='#FF00FF')    # Magenta
plt.plot(df['Date'], df['MA200'], label='200-Day MA', color='#008080')  # Teal
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('TCS Stock Price with Moving Averages')
plt.legend()
plt.tight_layout()
plt.show()

# Correlation Heatmap (distinct color palette)
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='mako', fmt='.2f')
plt.title('Correlation Heatmap of Features')
plt.tight_layout()
plt.show()

# --- Step 3: Feature Engineering ---
print('\n--- Feature Engineering ---')

# Extract date-based features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Day_of_Week'] = df['Date'].dt.dayofweek

# Create lag feature: previous day's close price
df['Prev_Close'] = df['Close'].shift(1)

# Drop rows with NaN values from lag feature
df_features = df.dropna().copy()

print('Feature-engineered data (head):')
print(df_features.head())

# Save feature-engineered data for modeling
features_path = 'TCS_stock_history_features.csv'
df_features.to_csv(features_path, index=False)
print(f'Feature-engineered data saved to {features_path}')

# --- Step 3b: Advanced Feature Engineering ---
print('\n--- Advanced Feature Engineering ---')

# Add more lag features
df['Prev_Close_2'] = df['Close'].shift(2)
df['Prev_Close_3'] = df['Close'].shift(3)
df['Prev_Close_5'] = df['Close'].shift(5)

# Add rolling statistics
df['MA7'] = df['Close'].rolling(window=7).mean()
df['MA30'] = df['Close'].rolling(window=30).mean()
df['MA90'] = df['Close'].rolling(window=90).mean()
df['RollingStd7'] = df['Close'].rolling(window=7).std()
df['RollingStd30'] = df['Close'].rolling(window=30).std()

# Drop rows with new NaNs from lag/rolling features
df_advanced = df.dropna().copy()

print('Advanced feature-engineered data (head):')
print(df_advanced.head())

# Save advanced feature-engineered data for modeling
advanced_features_path = 'TCS_stock_history_features_advanced.csv'
df_advanced.to_csv(advanced_features_path, index=False)
print(f'Advanced feature-engineered data saved to {advanced_features_path}')

# --- Step 4: Model Building and Prediction with Advanced Features (Linear Regression) ---
print('\n--- Model Building and Prediction with Advanced Features (Linear Regression) ---')

# Load advanced feature-engineered data
df_advanced = pd.read_csv('TCS_stock_history_features_advanced.csv', parse_dates=['Date'])

# Select advanced features and target
advanced_feature_cols = [
    'Open', 'High', 'Low', 'Volume', 'Prev_Close', 'Prev_Close_2', 'Prev_Close_3', 'Prev_Close_5',
    'MA7', 'MA30', 'MA90', 'RollingStd7', 'RollingStd30', 'Day_of_Week', 'Month'
]
X_adv = df_advanced[advanced_feature_cols]
y_adv = df_advanced['Close']

# Chronological split (first 80% train, last 20% test)
n = len(df_advanced)
split_idx = int(n * 0.8)
X_adv = df_advanced[advanced_feature_cols]
y_adv = df_advanced['Close']

X_train_ts, X_test_ts = X_adv.iloc[:split_idx], X_adv.iloc[split_idx:]
y_train_ts, y_test_ts = y_adv.iloc[:split_idx], y_adv.iloc[split_idx:]

# Retrain model on time-series split
model_ts = LinearRegression()
model_ts.fit(X_train_ts, y_train_ts)
y_pred_ts = model_ts.predict(X_test_ts)

# Evaluation
mse_ts = mean_squared_error(y_test_ts, y_pred_ts)
r2_ts = r2_score(y_test_ts, y_pred_ts)
print(f'Time Series Split - Mean Squared Error: {mse_ts:.2f}')
print(f'Time Series Split - R^2 Score: {r2_ts:.4f}')

# Visualization: Actual vs Predicted
plt.figure(figsize=(14, 6))
plt.plot(df_advanced['Date'].iloc[split_idx:], y_test_ts, label='Actual Close', color='#008080')
plt.plot(df_advanced['Date'].iloc[split_idx:], y_pred_ts, label='Predicted Close', color='#FFD700', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Price (Time Series Split)')
plt.legend()
plt.tight_layout()
plt.show()

# Visualization: Residuals
residuals = y_test_ts - y_pred_ts
plt.figure(figsize=(14, 4))
plt.plot(df_advanced['Date'].iloc[split_idx:], residuals, color='#FF00FF')
plt.xlabel('Date')
plt.ylabel('Residual (Actual - Predicted)')
plt.title('Residuals Over Time (Time Series Split)')
plt.tight_layout()
plt.show()

# Feature Importance (Linear Regression Coefficients)
print('\nFeature Importances (Linear Regression Coefficients):')
for feat, coef in zip(advanced_feature_cols, model_ts.coef_):
    print(f'{feat}: {coef:.4f}')

# --- Step 7: Sliding Window LSTM (Keep Only This Model) ---
print('\n--- Sliding Window LSTM (Varying Window Size) ---')

window_sizes = [5, 10, 20]
best_r2 = -float('inf')
best_window = None
best_model = None
best_scaler = None
best_X_test = None
best_y_test = None
best_pred = None

for window_size in window_sizes:
    print(f'\nTraining Sliding Window LSTM with window size: {window_size}')
    close_prices = df_advanced['Close'].values.reshape(-1, 1)
    scaler_close = MinMaxScaler()
    close_scaled = scaler_close.fit_transform(close_prices)
    X_seq, y_seq = [], []
    for i in range(window_size, len(close_scaled)):
        X_seq.append(close_scaled[i-window_size:i, 0])
        y_seq.append(close_scaled[i, 0])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    split_idx = int(len(X_seq) * 0.8)
    X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
    y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]
    X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], 1))
    X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_seq, y_train_seq, epochs=30, batch_size=16, verbose=0)
    pred_scaled = model.predict(X_test_seq)
    pred = scaler_close.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    y_test_inv = scaler_close.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    mse = mean_squared_error(y_test_inv, pred)
    r2 = r2_score(y_test_inv, pred)
    print(f'Window Size {window_size} - MSE: {mse:.2f}, R^2: {r2:.4f}')
    if r2 > best_r2:
        best_r2 = r2
        best_window = window_size
        best_model = model
        best_scaler = scaler_close
        best_X_test = X_test_seq
        best_y_test = y_test_inv
        best_pred = pred

print(f'\nBest window size: {best_window} (R^2: {best_r2:.4f})')

# Plot actual vs predicted for best window size
plt.figure(figsize=(14, 6))
plt.plot(range(len(best_y_test)), best_y_test, label='Actual Close', color='#008080')
plt.plot(range(len(best_pred)), best_pred, label=f'Sliding Window LSTM (window={best_window}) Predicted Close', color='#FF1493', alpha=0.7)
plt.xlabel('Time Index')
plt.ylabel('Close Price')
plt.title(f'Sliding Window LSTM (window={best_window}): Actual vs Predicted Close Price (Time Series Split)')
plt.legend()
plt.tight_layout()
plt.show()

# --- Step 8: SHAP Explainability Using Surrogate Model ---
print('\n--- SHAP Explainability Using Surrogate Random Forest ---')
from sklearn.ensemble import RandomForestRegressor

# Flatten the LSTM input for SHAP (samples, window_size)
X_shap = best_X_test.reshape((best_X_test.shape[0], best_X_test.shape[1]))

# Train surrogate Random Forest on the same input/output
rf_surrogate = RandomForestRegressor(n_estimators=100, random_state=42)
rf_surrogate.fit(X_shap, best_y_test)

# Use SHAP TreeExplainer for feature importances
explainer = shap.TreeExplainer(rf_surrogate)
shap_values = explainer.shap_values(X_shap)

# SHAP summary plot for lag importances
shap.summary_plot(shap_values, X_shap, feature_names=[f'lag_{i+1}' for i in range(best_window)])

# --- Step 9: Business-Focused Visualizations and Reporting ---
print('\n--- Business-Focused Visualizations and Reporting ---')

# Calculate returns (percentage change)
actual_returns = pd.Series(best_y_test).pct_change().fillna(0)
predicted_returns = pd.Series(best_pred).pct_change().fillna(0)

# Plot predicted vs actual returns
plt.figure(figsize=(14, 5))
plt.plot(actual_returns, label='Actual Returns', color='#008080')
plt.plot(predicted_returns, label='Predicted Returns', color='#FF1493', alpha=0.7)
plt.xlabel('Time Index')
plt.ylabel('Returns')
plt.title('Actual vs Predicted Returns')
plt.legend()
plt.tight_layout()
plt.show()

# Plot cumulative returns
actual_cum_returns = (1 + actual_returns).cumprod() - 1
predicted_cum_returns = (1 + predicted_returns).cumprod() - 1
plt.figure(figsize=(14, 5))
plt.plot(actual_cum_returns, label='Actual Cumulative Returns', color='#008080')
plt.plot(predicted_cum_returns, label='Predicted Cumulative Returns', color='#FF1493', alpha=0.7)
plt.xlabel('Time Index')
plt.ylabel('Cumulative Returns')
plt.title('Actual vs Predicted Cumulative Returns')
plt.legend()
plt.tight_layout()
plt.show()

# Print performance report
print('\n--- Model Performance Report ---')
print(f'Best window size: {best_window}')
print(f'R^2 Score: {best_r2:.4f}')
print(f'Last Actual Close: {best_y_test[-1]:.2f}')
print(f'Last Predicted Close: {best_pred[-1]:.2f}')
print(f'Mean Squared Error: {mean_squared_error(best_y_test, best_pred):.2f}')

# --- Step 10: Schedule Retraining as New Data Arrives ---
def retrain_on_new_data(csv_path, window_size=10):
    print(f'\n--- Retraining Model on New Data from {csv_path} ---')
    df_new = pd.read_csv(csv_path, parse_dates=['Date'])
    close_prices = df_new['Close'].values.reshape(-1, 1)
    scaler_close = MinMaxScaler()
    close_scaled = scaler_close.fit_transform(close_prices)
    X_seq, y_seq = [], []
    for i in range(window_size, len(close_scaled)):
        X_seq.append(close_scaled[i-window_size:i, 0])
        y_seq.append(close_scaled[i, 0])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    split_idx = int(len(X_seq) * 0.8)
    X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
    y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]
    X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], 1))
    X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_seq, y_train_seq, epochs=30, batch_size=16, verbose=0)
    print('Retraining complete. Model is now updated with the latest data.')
    return model, scaler_close

# Example usage (uncomment to retrain when new data is available):
# new_model, new_scaler = retrain_on_new_data('TCS_stock_history_cleaned.csv', window_size=best_window) 