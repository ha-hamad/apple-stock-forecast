# Apple Stock Price Forecasting with ARIMA

This project uses real historical stock data from Apple (AAPL) to build a time series forecasting model using ARIMA. The goal is to forecast stock prices for the next 30 business days using only Python.

---

## 📊 Key Features:
- Downloaded real stock price data (2020–2024) using `yfinance`
- Visualized the stock's historical closing prices
- Checked for stationarity using Augmented Dickey-Fuller test
- Transformed the data using differencing
- Built and trained an ARIMA(1,1,1) model with `statsmodels`
- Forecasted the next 30 days and plotted the predictions

---

## 🧠 Tools & Libraries Used:
- Python
- Pandas
- Matplotlib
- Seaborn
- Statsmodels
- YFinance

---

## 🧪 How to Run:
1. Clone the repository:
git clone https://github.com/yourusername/apple-stock-forecast.git
cd apple-stock-forecast

2. Install dependencies:
pip install -r requirements.txt

3. Run the script:
python forecast.py

--- 📌 Author
Made by (Hamad Albaker) ha-hamad as part of my data science project portfolio.
