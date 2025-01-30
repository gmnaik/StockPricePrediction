import pandas as pd
import numpy as np
from sklearn.svm import SVR
import yfinance as yf
from datetime import date,datetime,timedelta
from sklearn.model_selection import train_test_split,TimeSeriesSplit,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,root_mean_squared_error
from decimal import Decimal
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def check_stationarity(series):
    df_test_statistic, df_p_value, _, _, _, _ = adfuller(series)
    return df_p_value

def forecast_price(stockprice,vix_data):
    stockprice.columns = stockprice.columns.droplevel(1)   #To drop Ticker column from a dataframe
    print("Inside forecast_price() function:\n",stockprice)
    
    # Convert 'Date' column to datetime format
    stockprice['Date'] = pd.to_datetime(stockprice['Date'])

    # Filter data from January 1, 2022, onwards
    stockprice_df = stockprice[stockprice['Date'] >= '2022-01-01']
    
    train_size = int(len(stockprice_df) - 30)
    
    X_train = stockprice_df[:train_size]
    X_test = stockprice_df[train_size:]
    y_train = stockprice_df[:train_size]
    y_test = stockprice_df[train_size:]
    
    print("train_size:",len(X_train))
    print("train_size:",len(X_test))
    #print("stockprice_df:\n",stockprice_df)
    
    stockprice_df_close = X_train['Close']
    stockprice_df_close = pd.to_numeric(stockprice_df_close, errors='coerce').dropna() 
    stockprice_df_close_test = pd.to_numeric(X_test['Close'], errors='coerce').dropna() 
    
    original_series = stockprice_df_close.copy()
    
    p_value = check_stationarity(stockprice_df_close)

    # Step 2: Apply differencing until the series becomes stationary
    d = 0
    while p_value > 0.05:  # p-value > 0.05 means non-stationary
        stockprice_df_close = stockprice_df_close.diff().dropna()  # Apply differencing
        d += 1  # Increment differencing order
        p_value = check_stationarity(stockprice_df_close)  # Check if it's stationary after differencing
        print(f"Order of Differencing: {d}, p-value: {p_value}")    
    
    best_aic = np.inf
    best_order = None
    for p in range(5):  # Try p from 0 to 4
        for q in range(5):  # Try q from 0 to 4
            try:
                model = ARIMA(stockprice_df_close, order=(p, d, q))  # d=1 since data was differenced
                model_fit = model.fit()
                aic = model_fit.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
            except:
                print("Best parameters for ARIMA model not found")

    print(f'Best ARIMA order based on AIC: {best_order}')
    
    model = ARIMA(stockprice_df_close, order=best_order)
    model_fit = model.fit()

    # Step 3: Make predictions on the test set
    forecast = model_fit.forecast(steps=len(stockprice_df_close_test))  # Forecast for the test set length
    forecast_inverted = forecast.copy()
    for i in range(d, 0, -1):
        last_value = original_series.iloc[-i]  # Anchor from original data
        forecast_orginal = np.cumsum(forecast_inverted) + last_value
        
    # Step 4: Evaluate the ARIMA model
    mae_arima = mean_absolute_error(stockprice_df_close_test, forecast_orginal)
    mse_arima = mean_squared_error(stockprice_df_close_test, forecast_orginal)
    rmse_arima = np.sqrt(mse)

    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')

    # Step 5: Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(stockprice_df_close_test.index, stockprice_df_close_test, label='Actual', color='blue')
    plt.plot(stockprice_df_close_test.index, forecast_orginal, label='Predicted', color='red')
    plt.title('Actual vs Predicted Stock Price (AAPL)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
        
        
        
        
        
        
            
                    
                    
                            