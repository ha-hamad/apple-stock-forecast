# ------------------------------------------------------------
# 📊 Apple Stock Price Forecasting Using ARIMA (2020–2024)
# ------------------------------------------------------------

# Import essential libraries
import pandas as pd                      # For data manipulation
import matplotlib.pyplot as plt          # For visualization
import seaborn as sns                    # For styled plots
import yfinance as yf                    # To download stock data

# Set a clean plot style
sns.set(style="darkgrid")

# ------------------------------------------------------------
# Step 1: Download Apple (AAPL) stock data using Yahoo Finance
# ------------------------------------------------------------
# This fetches historical stock prices from Jan 2020 to Dec 2024
# We'll use the 'Close' price for modeling
df = yf.download('AAPL', start='2020-01-01', end='2024-12-31', auto_adjust=True)

# Make sure the index is in business-day frequency for modeling
df = df.asfreq('B')

# Preview the data
print("First few rows of the dataset:\n", df.head())

# ------------------------------------------------------------
# Step 2: Visualize the closing price to see the trend
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Closing Price')
plt.title('Apple (AAPL) Closing Price (2020–2024)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 3: Test for Stationarity using Augmented Dickey-Fuller
# ------------------------------------------------------------
# The ADF test checks if the time series is stationary (no trend or seasonality)
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import adfuller

# Drop missing or infinite values just in case
clean_close = df['Close'].replace([float('inf'), float('-inf')], pd.NA).dropna()

# Run ADF test
result = adfuller(clean_close)
print("\nADF Test p-value (original series):", result[1])
# A p-value > 0.05 means the series is non-stationary and needs transformation

# ------------------------------------------------------------
# Step 4: If not stationary, difference the series
# ------------------------------------------------------------
# First-order differencing removes the trend from the series
df['Diff_Close'] = df['Close'].diff()
df_diff = df['Diff_Close'].dropna()

# Re-run ADF test on differenced series
result_diff = adfuller(df_diff)
print("ADF Test after differencing (p-value):", result_diff[1])

# Visualize the differenced data
plt.figure(figsize=(12, 6))
plt.plot(df_diff, label='Differenced Closing Price')
plt.title('Differenced Apple Closing Price')
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 5: Fit the ARIMA Model
# ------------------------------------------------------------
from statsmodels.tsa.arima.model import ARIMA

# ARIMA(p=1, d=1, q=1) means:
# p = past values (AR), d = 1st differencing, q = past forecast errors (MA)
model = ARIMA(df['Close'], order=(1, 1, 1))
model_fit = model.fit()

# Display ARIMA model diagnostics
print("\nARIMA Model Summary:")
print(model_fit.summary())

# ------------------------------------------------------------
# Step 6: Forecast the next 30 business days
# ------------------------------------------------------------
forecast = model_fit.forecast(steps=30)

# Generate a range of future business dates
last_date = df.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

# ------------------------------------------------------------
# Step 7: Plot the forecast alongside historical data
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Historical Closing Price')
plt.plot(forecast_dates, forecast, label='Forecasted Price', color='red')
plt.title('Apple Stock Price Forecast (Next 30 Business Days)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()
