import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Set up Streamlit page configuration
st.set_page_config(page_title="SBIN Predictor", layout="centered")
st.title("📈 SBIN Next-Day Price Predictor")

# function to calculate 11 features
def calculate_features(df):
    # Ensure dataframe is sorted chronologically
    df = df.sort_index()
    
    df['return_1'] = df['Close'].pct_change(1)
    df['return_3'] = df['Close'].pct_change(3)
    
    df['sma_5'] = df['Close'].rolling(window=5).mean()
    df['sma_10'] = df['Close'].rolling(window=10).mean()
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    
    df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['Vol_MA_5'] = df['Volume'].rolling(window=5).mean()
    
    df['volatility_10'] = df['return_1'].rolling(window=10).std()
    
    df['volume_change'] = df['Volume'].pct_change(1)
    
    df['momentum_5'] = (df['Close'] / df['Close'].shift(5)) - 1
    
    df['return_lag1'] = df['return_1'].shift(1)
    
    return df

# Load the pre-trained pipeline
@st.cache_resource
def load_model():
    try:
        return joblib.load("sbin_model.joblib")
    except FileNotFoundError:
        st.error("Error: 'sbin_model.joblib' not found. Please place it in the same directory.")
        return None

pipeline = load_model()

if pipeline is not None:
    if st.button("Fetch Latest Data & Predict"):
        with st.spinner("Downloading market data from Yahoo Finance..."):
            # Fetch past 60 days of data to ensure indicators (like SMA 20) have enough data to calculate
            end_date = datetime.today()
            start_date = end_date - timedelta(days=60)
            
            # Fetch SBIN data from National Stock Exchange (NSE)
            sbin_data = yf.download("SBIN.NS", start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            
        if not sbin_data.empty and len(sbin_data) >= 25:
            # Flatten multi-level columns if returned by modern yfinance
            if isinstance(sbin_data.columns, pd.MultiIndex):
                sbin_data.columns = sbin_data.columns.get_level_values(0)
                
            # Process features
            processed_df = calculate_features(sbin_data)
            
            # Extract the very last row (today's calculated features)
            latest_row = processed_df.tail(1)
            
            feature_cols = [
                "return_1", "return_3", "sma_5", "sma_10", "sma_20", 
                "ema_10", "Vol_MA_5", "volatility_10", "volume_change", 
                "momentum_5", "return_lag1"
            ]
            
            # Check for NaN values in the latest row due to indicator windows
            if latest_row[feature_cols].isnull().values.any():
                st.warning("Warning: Generated features contain NaN values. Results might be inaccurate.")
            
            # Prepare inputs for the pipeline
            X_latest = latest_row[feature_cols]
            
            # Convert the single-row DataFrame into a raw 2D numpy array to ignore feature name constraints
            X_numpy = X_latest.values

            # Predict using the pipeline
            prediction = pipeline.predict(X_numpy)
                        
            # Display Metrics safely
            last_close_date = latest_row.index[0].strftime('%Y-%m-%d')
            last_close_val = float(latest_row['Close'].values[0])
            
            #Use .item() to safely convert 1D/2D numpy array to a python scalar float
            predicted_val = float(prediction.item())
            
            st.success("🎉 Prediction generated successfully!")
            
            col1, col2 = st.columns(2)
            col1.metric(label=f"Last Recorded Close ({last_close_date})", value=f"₹{last_close_val:.2f}")
            col2.metric(label="Predicted Next-Day Close", value=f"₹{predicted_val:.2f}")
            
            # Visual dataframe showcase
            st.subheader("Extracted Features for Prediction")
            st.dataframe(X_latest.style.format("{:.4f}"))
        else:
            st.error("Failed to retrieve adequate stock data. Please check market hours or your connection.")
