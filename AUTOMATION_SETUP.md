# 🤖 TCS Stock Prediction - Automation Setup Guide

This guide will help you set up automated retraining for your TCS stock prediction model to keep predictions up-to-date.

## 📋 Overview

You have two automation options:

1. **Enhanced Streamlit Dashboard** (`app_enhanced.py`) - Interactive retraining within the dashboard
2. **Standalone Retrain Script** (`retrain.py`) - Automated script for scheduled execution

## 🚀 Option 1: Enhanced Streamlit Dashboard

### Features:
- ✅ Manual retraining with current data
- ✅ Automatic data fetching from Yahoo Finance
- ✅ File upload retraining
- ✅ Real-time progress tracking
- ✅ Model performance metrics

### Setup:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the enhanced dashboard
streamlit run app_enhanced.py
```

### Usage:
1. Open the dashboard in your browser
2. Go to the sidebar "🔄 Model Retraining" section
3. Choose your retraining method:
   - **Manual Retrain**: Retrain with current data
   - **Use Latest Yahoo Finance Data**: Fetch latest data and retrain
   - **Upload New Data**: Upload CSV and retrain
4. Click the retrain button and monitor progress

---

## ⏰ Option 2: Scheduled Automation with Standalone Script

### Features:
- ✅ Automated data fetching from Yahoo Finance
- ✅ Fallback to local data if online fetch fails
- ✅ Comprehensive logging
- ✅ Model and predictions saving
- ✅ Error handling and notifications

### Setup:

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Test the Script
```bash
python retrain.py
```

You should see output like:
```
✅ Retraining completed successfully!
📊 Check retrain.log for detailed information
```

#### Step 3: Schedule Automation

##### Windows Task Scheduler:
1. Open **Task Scheduler** (search in Start menu)
2. Click **"Create Basic Task"**
3. Name: `TCS Model Retraining`
4. Trigger: **Daily** (or your preferred frequency)
5. Action: **Start a program**
6. Program: `python`
7. Arguments: `retrain.py`
8. Start in: `C:\Users\YourUsername\Downloads\Unified Project\TCS Stock Data Analysis and Prediction`
9. Finish and test the task

##### Linux/Mac Cron:
```bash
# Edit crontab
crontab -e

# Add this line for daily retraining at 9 AM
0 9 * * * cd /path/to/your/project && python retrain.py

# Or for weekly retraining on Sundays at 9 AM
0 9 * * 0 cd /path/to/your/project && python retrain.py
```

##### Cloud Scheduling (AWS Lambda, Azure Functions, etc.):
1. Package the script and dependencies
2. Set up cloud function with appropriate trigger
3. Configure environment variables if needed

---

## 📊 Monitoring and Logs

### Log Files:
- `retrain.log` - Detailed retraining logs
- `predictions_window{size}.csv` - Latest predictions
- `lstm_model_window{size}.joblib` - Trained model
- `scaler_window{size}.joblib` - Data scaler

### Log Format:
```
2024-01-15 09:00:00 - INFO - Starting TCS Model Retraining
2024-01-15 09:00:05 - INFO - Fetching latest data for TCS.NS
2024-01-15 09:00:10 - INFO - Fetched 365 records from 2023-01-15 to 2024-01-15
2024-01-15 09:01:30 - INFO - Training LSTM model with 30 epochs, batch size 16
2024-01-15 09:02:15 - INFO - Model Performance - MSE: 123.4567, R²: 0.9876
2024-01-15 09:02:16 - INFO - Retraining Completed Successfully!
```

---

## 🔧 Configuration Options

### Modify `retrain.py` parameters:
```python
# In retrain.py, modify these parameters:
retrainer = TCSModelRetrainer(
    window_size=10,      # LSTM window size
    epochs=30,           # Training epochs
    batch_size=16        # Batch size
)

# Data source options:
retrainer.retrain(use_yahoo_finance=True)  # Use Yahoo Finance
retrainer.retrain(use_yahoo_finance=False) # Use local CSV
```

### Environment Variables (Optional):
```bash
# Set these in your environment for customization
export TCS_SYMBOL=TCS.NS
export DATA_DAYS=365
export WINDOW_SIZE=10
export EPOCHS=30
```

---

## 🚨 Troubleshooting

### Common Issues:

1. **Yahoo Finance API fails:**
   - Script automatically falls back to local data
   - Check internet connection
   - Verify stock symbol (TCS.NS for NSE)

2. **Memory issues:**
   - Reduce `batch_size` in retrain.py
   - Reduce `epochs` for faster training
   - Use smaller `window_size`

3. **Task Scheduler not working:**
   - Check if Python is in PATH
   - Verify working directory
   - Run task manually first

4. **Model performance degrades:**
   - Check retrain.log for errors
   - Verify data quality
   - Consider adjusting hyperparameters

### Error Recovery:
```bash
# Check logs
tail -f retrain.log

# Manual retraining
python retrain.py

# Verify model files exist
ls -la *.joblib *.csv
```

---

## 📈 Performance Optimization

### For Faster Retraining:
- Reduce `epochs` (e.g., 15-20 instead of 30)
- Use smaller `batch_size` (e.g., 8 instead of 16)
- Consider using GPU if available

### For Better Predictions:
- Increase `epochs` for more training
- Experiment with different `window_size` values
- Use more historical data (increase `days` parameter)

---

## 🔄 Integration with Existing Workflow

### With Streamlit Dashboard:
1. Run `retrain.py` on schedule
2. Dashboard automatically loads latest model
3. No manual intervention needed

### With External Systems:
- Model files are saved in standard formats
- Predictions CSV can be imported to other tools
- Logs provide monitoring integration points

---

## 📞 Support

If you encounter issues:
1. Check the `retrain.log` file for detailed error messages
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Test the script manually first: `python retrain.py`
4. Ensure you have write permissions in the project directory

---

## 🎯 Next Steps

1. **Start with manual testing**: Run `python retrain.py` manually
2. **Set up basic scheduling**: Use Task Scheduler or cron
3. **Monitor performance**: Check logs and model accuracy
4. **Optimize parameters**: Adjust based on your needs
5. **Scale up**: Consider cloud deployment for production use

Your TCS stock prediction system is now ready for automated, up-to-date predictions! 🚀 