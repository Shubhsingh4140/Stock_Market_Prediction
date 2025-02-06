import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st

# Set the title for the Streamlit app
st.title('Stock Trend Prediction')

# Take user input for the stock ticker 
user_input = st.text_input("Enter Stock Ticker", 'TSLA')

# Define the start and end date for fetching data
start = '2017-05-14'
end = '2019-05-12'

# Fetch the stock data from Tiingo using requests
api_key = 'ac0e4d9ca7e8f64702fcd180543becbfed55ea8f'
url = f'https://api.tiingo.com/tiingo/daily/{user_input}/prices?startDate={start}&endDate={end}&token={api_key}'

try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    st.write(df.head())  # Display the first few rows of the dataframe
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Visualization of the closing price over time
st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['close'], label='Close')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
st.pyplot(fig)
