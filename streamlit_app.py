
# streamlit_app.py

import streamlit as st
import yfinance as yf
from datetime import date, timedelta
from prophet import Prophet
import pandas as pd
import requests

st.set_page_config(page_title="AI Stock & Crypto Predictor", layout="wide")
st.title("📈 AI-Powered Stock & Crypto Predictor")

# --- Sidebar Inputs ---
st.sidebar.header("Select Asset")
asset = st.sidebar.text_input("Enter Ticker (e.g., BTC-USD, AAPL)", value="BTC-USD")
days_to_forecast = st.sidebar.slider("Days to Predict Ahead", 1, 30, 7)

# --- Load Data ---
@st.cache_data
def load_data(ticker):
    end = date.today()
    start = end - timedelta(days=730)
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)

    # Flatten MultiIndex if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(asset)
data_load_state.text("✔️ Data loaded successfully!")

# --- Display Historical Chart ---
st.subheader(f"Historical Chart for {asset}")
st.line_chart(data[['Date', 'Close']].set_index('Date'))

# --- Forecasting ---
st.subheader(f"Forecasting {days_to_forecast} days into the future")
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=days_to_forecast)
forecast = model.predict(future)
st.line_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].tail(days_to_forecast))

# --- Display Forecast Table ---
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_to_forecast))

# --- Optional: News Sentiment (Basic) ---
st.subheader("📰 News Headlines (Demo)")
news_api_key = "demo"  # Replace with real NewsAPI key later
news_url = f"https://newsapi.org/v2/everything?q={asset}&sortBy=publishedAt&apiKey={news_api_key}"

if news_api_key != "demo":
    try:
        response = requests.get(news_url)
        articles = response.json().get("articles", [])[:5]
        for article in articles:
            st.markdown(f"**[{article['title']}]({article['url']})**")
            st.caption(article['description'])
    except:
        st.warning("Unable to fetch news.")
else:
    st.info("🛠️ Add your NewsAPI key to enable live headlines.")

# --- Footer ---
st.caption("AI predictions are for informational purposes only and not financial advice.")
