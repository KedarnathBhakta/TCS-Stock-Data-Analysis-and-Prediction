📈 TCS Stock Data Analysis & Prediction

Welcome to the TCS Stock Data Analysis & Prediction project! This solution leverages advanced machine learning and deep learning (LSTM) to forecast TCS stock prices, with a focus on business usability, explainability, and automation.


🚀 Project Highlights

- Interactive Streamlit dashboard for business users
- Automated retraining with latest data (manual, scheduled, or via dashboard)
- Model explainability with SHAP
- Downloadable predictions and business metrics
- All data files organized in the data/ folder


📦 Project Structure

```
.
├── app_enhanced.py           # Streamlit dashboard app (with retraining)
├── app.py                    # (Optional) Original dashboard app
├── retrain.py                # Standalone retraining script
├── requirements.txt          # Python dependencies
├── AUTOMATION_SETUP.md       # Automation & deployment guide
├── README.md                 # Project documentation
├── data/                     # All CSV data and prediction files
│   ├── TCS_stock_history_cleaned.csv
│   ├── predictions_window10.csv
│   ├── ... (other CSVs)
├── lstm_model_windowXX.joblib   # Saved LSTM models (per window size)
├── scaler_windowXX.joblib       # Saved scalers (per window size)
├── retrain.log               # Retraining logs
├── venv/                     # Python virtual environment (not tracked in git)
```


📊 Features

- Upload new data or use live Yahoo Finance data
- Choose LSTM window size and date range interactively
- Visualize actual vs predicted prices, returns, and cumulative returns
- Download predictions as CSV
- Business metrics: volatility, max drawdown, Sharpe ratio
- SHAP feature importance for explainability
- Retrain model with a button or on a schedule


🔄 Automation & Retraining

- Use the dashboard for on-demand retraining (manual, Yahoo, or file upload)
- Use retrain.py for scheduled retraining (see AUTOMATION_SETUP.md)
- All new predictions and models are saved in the data/ folder


📁 Data Management

- All CSV files are now in the data/ folder for easy management
- Example files:
  - data/TCS_stock_history_cleaned.csv — Main historical data
  - data/predictions_window10.csv — Latest predictions
  - data/feature_enhanced_lstm_predictions.csv — Feature-engineered predictions
  - data/xgboost_predictions_advanced.csv — XGBoost model predictions


🛠️ Setup Instructions

1. Clone the repository and navigate to the project folder
2. Install dependencies:
   pip install -r requirements.txt
3. To launch the dashboard:
   streamlit run app_enhanced.py
4. To retrain automatically:
   python retrain.py
   (See AUTOMATION_SETUP.md for scheduling)


💡 Tips

- For best results, keep your data up-to-date in the data/ folder
- Use the dashboard for business-friendly analysis and explainability
- Check retrain.log for retraining history and troubleshooting


📞 Support

If you have questions or need help, check the logs, review the documentation, or reach out to your project lead.


🎉 Enjoy automated, explainable, and business-ready TCS stock predictions! 