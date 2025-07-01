#!/usr/bin/env python3
"""
Test script to verify PyTorch LSTM model works correctly
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# PyTorch LSTM Model Class
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def test_pytorch_lstm():
    """Test the PyTorch LSTM model with sample data"""
    print("🧪 Testing PyTorch LSTM Model...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    df = pd.DataFrame({'Date': dates, 'Close': close_prices})
    
    print(f"✅ Created sample data with {len(df)} records")
    
    # Prepare sequences
    window_size = 10
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(close_prices)
    
    X_seq, y_seq = [], []
    for i in range(window_size, len(close_scaled)):
        X_seq.append(close_scaled[i-window_size:i, 0])
        y_seq.append(close_scaled[i, 0])
    
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    print(f"✅ Prepared sequences: X shape {X_seq.shape}, y shape {y_seq.shape}")
    
    # Split data
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1)
    
    print(f"✅ Converted to PyTorch tensors")
    
    # Create and train model
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    print("🔄 Training model...")
    model.train()
    for epoch in range(10):  # Quick test with 10 epochs
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()
    
    print(f"✅ Training completed. Final loss: {loss.item():.6f}")
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test_tensor).squeeze().numpy()
    
    # Inverse transform
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"✅ Made predictions. Predicted {len(pred)} values")
    print(f"📊 Sample predictions: {pred[:5]}")
    print(f"📊 Sample actual: {y_test_inv[:5]}")
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test_inv, pred)
    r2 = r2_score(y_test_inv, pred)
    
    print(f"📈 Model Performance:")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²: {r2:.4f}")
    
    print("🎉 PyTorch LSTM test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_pytorch_lstm()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 