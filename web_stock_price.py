import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


st.title("Stock Price Predictor App")

# User input for stock ID
stock = st.text_input("Enter the Stock ID", "GOOG")

# Setting the date range for the past 20 years
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

# Download the stock data from Yahoo Finance
google_data = yf.download(stock, start, end)

# Print the structure of the data for debugging
st.write(google_data.head())  # Display the first few rows
st.write(google_data.columns)  # Display column names

# Check if 'Close' column exists
if 'Close' not in google_data.columns:
    st.error("The 'Close' column is missing from the data.")
else:
    # Reset index if needed
    google_data = google_data.reset_index()

    # Display the stock data
    st.subheader("Stock Data")
    st.write(google_data)

    # Splitting data into training and test sets
    splitting_len = int(len(google_data) * 0.7)
    x_test = pd.DataFrame(google_data['Close'][splitting_len:])

    # Plot function to show graph
    def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
        fig = plt.figure(figsize=figsize)
        plt.plot(values, 'Orange')
        plt.plot(full_data['Close'], 'b')
        if extra_data:
            plt.plot(extra_dataset)
        return fig

    # Visualizations for moving averages
    st.subheader('Original Close Price and MA for 250 days')
    google_data['MA_for_250_days'] = google_data['Close'].rolling(250).mean()
    st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data, 0))

    st.subheader('Original Close Price and MA for 200 days')
    google_data['MA_for_200_days'] = google_data['Close'].rolling(200).mean()
    st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data, 0))

    st.subheader('Original Close Price and MA for 100 days')
    google_data['MA_for_100_days'] = google_data['Close'].rolling(100).mean()
    st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 0))

    st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
    st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

    # Scale the data for prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test)

    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Load the trained model
    model = load_model("Latest_stock_price_model.keras")

    # Make predictions
    predictions = model.predict(x_data)

    # Inverse scaling to get actual stock prices
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Create a DataFrame to show predictions vs actual values
    plotting_data = pd.DataFrame(
        {'original_test_data': inv_y_test.reshape(-1),
         'predictions': inv_pre.reshape(-1)},
        index=google_data.index[splitting_len + 100:]
    )

    # Display original vs predicted values
    st.subheader("Original values vs Predicted values")
    st.write(plotting_data)

    # Plot original vs predicted values
    st.subheader('Original Close Price vs Predicted Close price')
    fig = plt.figure(figsize=(15, 6))
    plt.plot(pd.concat([google_data['Close'][:splitting_len + 100], plotting_data], axis=0))
    plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
    st.pyplot(fig)
