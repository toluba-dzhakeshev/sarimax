import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def load_data(file):
    data = pd.read_excel(file)  
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    return data

def train_and_forecast(data):
    model = sm.tsa.SARIMAX(data, order=(0,1,1), seasonal_order=(2,0,1,52)).fit(disp=False)
    forecast_steps = 52
    forecast = model.forecast(steps=forecast_steps)
    return model, forecast

st.title('SARIMAX Time Series Forecasting')

uploaded_file = st.file_uploader("Upload your Excel file", type=["xls", "xlsx"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Data Preview:", data.head())

    data.set_index('Order Date', inplace=True)
    weekly_sales_sum = data.groupby(pd.Grouper(freq='W'))['Sales'].sum()
    
    train_size = int(len(weekly_sales_sum) * 0.8)
    train, test = weekly_sales_sum[:train_size], weekly_sales_sum[train_size:]

    st.write("Training model and forecasting...")
    model, forecast = train_and_forecast(weekly_sales_sum)

    preds = model.forecast(steps=len(test))
    mae = mean_absolute_error(test, preds)
    st.write(f"Mean Absolute Error on test data: {mae}")
    
    forecast_index = pd.date_range(start=weekly_sales_sum.index[-1] + pd.Timedelta(weeks=1), periods=52, freq='W')
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])
    
    plt.figure(figsize=(10, 5))
    plt.plot(weekly_sales_sum, label='Actual Sales')
    plt.plot(forecast_df, label='Forecast', color='red')
    plt.legend()
    plt.title('Weekly Sales Forecast using SARIMAX')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    st.pyplot(plt)
    
    st.write("Model Summary:")
    st.text(model.summary())

else:
    st.stop()
    