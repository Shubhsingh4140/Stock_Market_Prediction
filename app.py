import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
# from keras.models import load_model
from tensorflow.keras.models import load_model

import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Set the title for the Streamlit app
st.title('Stock Trend Prediction')

# Take user input for the stock ticek
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

# Adding 100-day moving average
st.subheader('Closing Price Vs Time Chart With 100MA')
ma100 = df['close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['close'], label='Close')
plt.plot(ma100, label='100MA')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
st.pyplot(fig)

# Adding 100-day and 200-day moving averages
st.subheader('Closing Price Vs Time Chart With 100MA & 200MA')
ma200 = df['close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['close'], label='Close')
plt.plot(ma100, label='100MA')
plt.plot(ma200, label='200MA')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
st.pyplot(fig)

# Prepare data for training and testing
data_training = df['close'][0:int(len(df)*0.70)]
data_testing = df['close'][int(len(df)*0.70):]

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

# Load the trained model
# model = load_model('keras_model.h5')
# model = load_model("keras_model.h5", compile=False)
# model = load_model("keras_model_new.keras", compile=False)

import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "keras_model_new.keras")

model = load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)







# Prepare the test data
past_100_days = data_training.tail(100)
# final_df = past_100_days.append(data_testing, ignore_index=True)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df.values.reshape(-1, 1))

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
y_predicted = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Visualization of the predicted vs original prices
st.subheader('Predicted Vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label="Original Price")
plt.plot(y_predicted, 'r', label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
