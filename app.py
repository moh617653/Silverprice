import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Silver Price Predictor", layout="wide")

st.title("Silver Price Analysis & Prediction (SI=F)")
st.write("Extracting the last 5 years of Silver prices from Yahoo Finance and forecasting the next 1 year.")

@st.cache_data
def load_data():
    ticker = "SI=F"
    df = yf.download(ticker, period="5y", interval="1d")
    df = df.dropna()
    return df

with st.spinner("Downloading data from Yahoo Finance..."):
    data = load_data()

if 'Close' not in data.columns:
    st.error("Could not find 'Close' prices in the downloaded data.")
    st.stop()

# We might get multi-indexed columns from yfinance sometimes
if isinstance(data.columns, pd.MultiIndex):
    close_series = data.xs('Close', level=0, axis=1).iloc[:, 0]
else:
    close_series = data['Close']

df_ts = pd.DataFrame({'Close': close_series})
df_ts.index = pd.to_datetime(df_ts.index)

# Ensure daily frequency by filling missing dates 
df_ts = df_ts.resample('B').ffill() # 'B' for business days

st.subheader("Historical Last 5 Years Silver Price")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_ts.index, y=df_ts['Close'], mode='lines', name='Historical Close Price'))
fig.update_layout(title="Silver Daily Close Price", xaxis_title="Date", yaxis_title="Price (USD)", hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)

# Train-Test Split to calculate RMSE (last 252 business days = ~1 year)
n_test = 252
train, test = df_ts.iloc[:-n_test], df_ts.iloc[-n_test:]

with st.spinner("Training model for evaluation (Holt-Winters)..."):
    # Fit Exponential Smoothing
    model = ExponentialSmoothing(train['Close'], trend='add', seasonal=None, initialization_method="estimated")
    fit_model = model.fit()
    predictions = fit_model.forecast(len(test))
    
    rmse = np.sqrt(mean_squared_error(test['Close'], predictions))

st.write(f"### Model Evaluation on Last 1 Year")
st.success(f"**Root Mean Squared Error (RMSE)**: {rmse:.4f}")

with st.spinner("Forecasting next 1 year..."):
    # Fit on all data for future forecast
    full_model = ExponentialSmoothing(df_ts['Close'], trend='add', seasonal=None, initialization_method="estimated")
    full_fit = full_model.fit()
    
    # Predict next 252 business days (~1 year)
    future_forecast = full_fit.forecast(252)
    
st.subheader("Future Forecast (Next 1 Year)")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_ts.index, y=df_ts['Close'], mode='lines', name='Historical'))
fig2.add_trace(go.Scatter(x=future_forecast.index, y=future_forecast, mode='lines', name='Forecast (1 Year Ahead)', line=dict(color='orange')))
fig2.update_layout(title="Silver Price Future Prediction", xaxis_title="Date", yaxis_title="Price (USD)", hovermode='x unified')
st.plotly_chart(fig2, use_container_width=True)

st.write("---")
st.write("**Disclaimer:** This is a simple statistical forecast and should not be used as financial advice. Commodity markets are highly volatile.")
