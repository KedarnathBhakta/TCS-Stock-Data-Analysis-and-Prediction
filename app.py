import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import shap
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import warnings

st.set_page_config(page_title='TCS Stock Price Prediction Dashboard', layout='wide')
st.title('📈 TCS Stock Price Prediction Dashboard')
st.markdown('''---''')

# --- Sidebar Controls ---
st.sidebar.header('Controls')
window_size = st.sidebar.selectbox('Select LSTM Window Size (days)', [5, 10, 20], index=1, help='Number of previous days used to predict the next close price.')
date_range = st.sidebar.slider('Select Date Range for Visualization', 0, 100, (0, 100), help='Zoom in on a percentage of the test period.')

# --- Upload Data ---
st.sidebar.header('Upload New Data')
uploaded_file = st.sidebar.file_uploader('Upload a CSV file with Date and Close columns', type=['csv'])

EPOCHS = 30
BATCH_SIZE = 16
MODEL_PATH = f'lstm_model_window{window_size}.joblib'

# --- Load or Prepare Data ---
def load_data(file):
    df = pd.read_csv(file)
    # Auto-detect 'Date' column (case-insensitive, strip spaces)
    date_col = None
    for col in df.columns:
        if col.strip().lower() == 'date':
            date_col = col
            break
    if date_col is None:
        st.error("CSV must contain a column named 'Date' (case-insensitive).")
        st.stop()
    df = df.rename(columns={date_col: 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

def prepare_sequences(df, window_size):
    # Auto-detect 'Close' column (case-insensitive, strip spaces)
    close_col = None
    for col in df.columns:
        if col.strip().lower() == 'close':
            close_col = col
            break
    if close_col is None:
        st.error("CSV must contain a column named 'Close' (case-insensitive).")
        st.stop()
    close_prices = df[close_col].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(close_prices)
    X_seq, y_seq = [], []
    for i in range(window_size, len(close_scaled)):
        X_seq.append(close_scaled[i-window_size:i, 0])
        y_seq.append(close_scaled[i, 0])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    return X_seq, y_seq, scaler

def train_lstm(X_train, y_train, window_size):
    model = Sequential()
    model.add(LSTM(50, input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    return model

# --- Main Logic ---
if uploaded_file:
    df = load_data(uploaded_file)
    st.success('Data uploaded successfully!')
else:
    df = pd.read_csv('data/default/TCS_stock_history_cleaned.csv', parse_dates=['Date'])
    st.info('Using default TCS_stock_history_cleaned.csv')

X_seq, y_seq, scaler = prepare_sequences(df, window_size)
split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# --- Model Save/Load ---
model_loaded = False
try:
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            model_loaded = True
            st.success('✅ Loaded existing trained model')
        except Exception as e:
            warnings.warn(f"Model load failed: {e}. Will retrain.")
            st.warning(f"Model load failed: {e}. Retraining...")
    else:
        st.info('Model file missing. Retraining...')
except Exception as e:
    warnings.warn(f"Error checking model: {e}. Will retrain.")
    st.warning(f"Error checking model: {e}. Retraining...")

if not model_loaded:
    with st.spinner('Training new model...'):
        model = train_lstm(X_train, y_train, window_size)
        joblib.dump(model, MODEL_PATH)
    st.success('✅ New model trained and saved (auto)')

pred_scaled = model.predict(X_test)
pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# --- Date Range Filter ---
start_idx = int(len(y_test_inv) * date_range[0] / 100)
end_idx = int(len(y_test_inv) * date_range[1] / 100)
if end_idx <= start_idx:
    end_idx = start_idx + 1

# --- Business Visualizations ---
st.header('Actual vs Predicted Close Price')
st.caption('Visualize how well the model tracks the actual stock price.')
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(range(end_idx - start_idx), y_test_inv[start_idx:end_idx], label='Actual Close', color='#008080')
ax1.plot(range(end_idx - start_idx), pred[start_idx:end_idx], label='Predicted Close', color='#FF1493', alpha=0.7)
ax1.set_xlabel('Time Index')
ax1.set_ylabel('Close Price')
ax1.set_title('Actual vs Predicted Close Price')
ax1.legend()
st.pyplot(fig1)

# Returns
actual_returns = pd.Series(y_test_inv).pct_change().fillna(0)
predicted_returns = pd.Series(pred).pct_change().fillna(0)

st.header('Actual vs Predicted Returns')
st.caption('Daily returns (percentage change) for actual and predicted prices.')
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(actual_returns[start_idx:end_idx], label='Actual Returns', color='#008080')
ax2.plot(predicted_returns[start_idx:end_idx], label='Predicted Returns', color='#FF1493', alpha=0.7)
ax2.set_xlabel('Time Index')
ax2.set_ylabel('Returns')
ax2.set_title('Actual vs Predicted Returns')
ax2.legend()
st.pyplot(fig2)

# Cumulative Returns
actual_cum_returns = (1 + actual_returns).cumprod() - 1
predicted_cum_returns = (1 + predicted_returns).cumprod() - 1
st.header('Actual vs Predicted Cumulative Returns')
st.caption('Cumulative returns show the growth of a hypothetical investment.')
fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(actual_cum_returns[start_idx:end_idx], label='Actual Cumulative Returns', color='#008080')
ax3.plot(predicted_cum_returns[start_idx:end_idx], label='Predicted Cumulative Returns', color='#FF1493', alpha=0.7)
ax3.set_xlabel('Time Index')
ax3.set_ylabel('Cumulative Returns')
ax3.set_title('Actual vs Predicted Cumulative Returns')
ax3.legend()
st.pyplot(fig3)

# --- Download Predictions ---
st.header('Download Predictions')
st.caption('Download the predicted and actual close prices for further analysis.')
pred_df = pd.DataFrame({
    'Actual_Close': y_test_inv,
    'Predicted_Close': pred
})
st.download_button('Download Predictions as CSV', pred_df.to_csv(index=False), file_name='predictions.csv', mime='text/csv')

# --- Business Insights ---
st.header('Business Insights')
st.caption('Key risk and performance metrics for actual and predicted returns.')
def sharpe_ratio(returns):
    return np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
def max_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()
st.write(f"**Actual Volatility:** {np.std(actual_returns) * np.sqrt(252):.4f}")
st.write(f"**Predicted Volatility:** {np.std(predicted_returns) * np.sqrt(252):.4f}")
st.write(f"**Actual Max Drawdown:** {max_drawdown(actual_returns):.2%}")
st.write(f"**Predicted Max Drawdown:** {max_drawdown(predicted_returns):.2%}")
st.write(f"**Actual Sharpe Ratio:** {sharpe_ratio(actual_returns):.2f}")
st.write(f"**Predicted Sharpe Ratio:** {sharpe_ratio(predicted_returns):.2f}")

# --- Model Performance Report ---
st.header('Model Performance Report')
st.caption('Summary of model accuracy and last predictions.')
st.write(f'**Window size:** {window_size}')
st.write(f'**R² Score:** {r2_score(y_test_inv, pred):.4f}')
st.write(f'**Mean Squared Error:** {mean_squared_error(y_test_inv, pred):.2f}')
st.write(f'**Last Actual Close:** {y_test_inv[-1]:.2f}')
st.write(f'**Last Predicted Close:** {pred[-1]:.2f}')

# --- SHAP Explainability (Surrogate Model) ---
st.header('SHAP Feature Importance (Surrogate Random Forest)')
st.caption('See which previous days (lags) are most important for the prediction.')
X_shap = X_test.reshape((X_test.shape[0], X_test.shape[1]))
rf_surrogate = RandomForestRegressor(n_estimators=100, random_state=42)
rf_surrogate.fit(X_shap, y_test_inv)
explainer = shap.TreeExplainer(rf_surrogate)
shap_values = explainer.shap_values(X_shap)
fig4 = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap, feature_names=[f'lag_{i+1}' for i in range(window_size)], show=False)
st.pyplot(fig4)

st.markdown('---')
st.info('To retrain the model with new data, upload a new CSV and refresh the app.')

if st.button("Retrain Model"):
    # Your retraining function here
    retrain_model()
    st.success("Model retrained and predictions updated!") 