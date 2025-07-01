#!/usr/bin/env python3
"""
TCS Stock Prediction Model Retraining Script
Automated script to retrain the LSTM model with latest data and update predictions.
Can be scheduled to run automatically using Task Scheduler (Windows) or cron (Linux).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import logging
from datetime import datetime
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrain.log'),
        logging.StreamHandler()
    ]
)

class TCSModelRetrainer:
    def __init__(self, window_size=10, epochs=30, batch_size=16):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = f'lstm_model_window{window_size}.joblib'
        self.scaler_path = f'scaler_window{window_size}.joblib'
        self.predictions_path = f'predictions_window{window_size}.csv'
        
    def fetch_latest_data(self, symbol='TCS.NS', days=365):
        """Fetch latest data from Yahoo Finance"""
        try:
            logging.info(f"Fetching latest data for {symbol}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f'{days}d')
            df = df.reset_index()
            df = df.rename(columns={'Date': 'Date', 'Close': 'Close'})
            df = df[['Date', 'Close']]
            df = df.sort_values('Date')
            logging.info(f"Fetched {len(df)} records from {df['Date'].min()} to {df['Date'].max()}")
            return df
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return None
    
    def load_local_data(self, file_path='data/default/TCS_stock_history_cleaned.csv'):
        """Load data from local CSV file"""
        try:
            logging.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path, parse_dates=['Date'])
            df = df.sort_values('Date')
            logging.info(f"Loaded {len(df)} records from {df['Date'].min()} to {df['Date'].max()}")
            return df
        except Exception as e:
            logging.error(f"Error loading local data: {e}")
            return None
    
    def prepare_sequences(self, df):
        """Prepare sequences for LSTM training"""
        close_prices = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        close_scaled = scaler.fit_transform(close_prices)
        
        X_seq, y_seq = [], []
        for i in range(self.window_size, len(close_scaled)):
            X_seq.append(close_scaled[i-self.window_size:i, 0])
            y_seq.append(close_scaled[i, 0])
        
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        return X_seq, y_seq, scaler
    
    def train_lstm(self, X_train, y_train):
        """Train LSTM model"""
        model = Sequential([
            LSTM(50, input_shape=(self.window_size, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        logging.info(f"Training LSTM model with {self.epochs} epochs, batch size {self.batch_size}")
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return model
    
    def evaluate_model(self, model, X_test, y_test, scaler):
        """Evaluate model performance"""
        pred_scaled = model.predict(X_test)
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        mse = mean_squared_error(y_test_inv, pred)
        r2 = r2_score(y_test_inv, pred)
        
        logging.info(f"Model Performance - MSE: {mse:.4f}, R²: {r2:.4f}")
        return pred, y_test_inv, mse, r2
    
    def save_model_and_predictions(self, model, scaler, pred, actual, df):
        """Save model, scaler, and predictions"""
        # Save model
        joblib.dump(model, self.model_path)
        logging.info(f"Model saved to {self.model_path}")
        
        # Save scaler
        joblib.dump(scaler, self.scaler_path)
        logging.info(f"Scaler saved to {self.scaler_path}")
        
        # Save predictions
        pred_df = pd.DataFrame({
            'Date': df['Date'].iloc[self.window_size:].iloc[-len(pred):],
            'Actual_Close': actual,
            'Predicted_Close': pred
        })
        pred_df.to_csv(self.predictions_path, index=False)
        logging.info(f"Predictions saved to {self.predictions_path}")
        
        return pred_df
    
    def retrain(self, use_yahoo_finance=True):
        """Main retraining function"""
        start_time = datetime.now()
        logging.info("=" * 50)
        logging.info("Starting TCS Model Retraining")
        logging.info("=" * 50)
        
        try:
            # Load data
            if use_yahoo_finance:
                df = self.fetch_latest_data()
                if df is None:
                    logging.warning("Failed to fetch from Yahoo Finance, trying local data")
                    df = self.load_local_data()
            else:
                df = self.load_local_data()
            
            if df is None:
                logging.error("Failed to load data from both sources")
                return False
            
            # Prepare sequences
            X_seq, y_seq, scaler = self.prepare_sequences(df)
            
            # Split data
            split_idx = int(len(X_seq) * 0.8)
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
            
            # Reshape for LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Train model
            model = self.train_lstm(X_train, y_train)
            
            # Evaluate model
            pred, actual, mse, r2 = self.evaluate_model(model, X_test, y_test, scaler)
            
            # Save everything
            pred_df = self.save_model_and_predictions(model, scaler, pred, actual, df)
            
            # Log final results
            end_time = datetime.now()
            duration = end_time - start_time
            logging.info("=" * 50)
            logging.info("Retraining Completed Successfully!")
            logging.info(f"Duration: {duration}")
            logging.info(f"Final R² Score: {r2:.4f}")
            logging.info(f"Final MSE: {mse:.4f}")
            logging.info(f"Last Actual Close: {actual[-1]:.2f}")
            logging.info(f"Last Predicted Close: {pred[-1]:.2f}")
            logging.info("=" * 50)
            
            return True
            
        except Exception as e:
            logging.error(f"Error during retraining: {e}")
            return False

def main():
    """Main function to run retraining"""
    # Create retrainer instance
    retrainer = TCSModelRetrainer(window_size=10, epochs=30, batch_size=16)
    
    # Try Yahoo Finance first, fallback to local data
    success = retrainer.retrain(use_yahoo_finance=True)
    
    if success:
        print("✅ Retraining completed successfully!")
        print("📊 Check retrain.log for detailed information")
    else:
        print("❌ Retraining failed! Check retrain.log for errors")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 