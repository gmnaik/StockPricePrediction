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
from pmdarima import auto_arima
#from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.statespace.sarimax import SARIMAX

#import numpy
#import pmdarima
#print(numpy.__version__)
#print(pmdarima.__version__)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def check_stationarity(series):
    df_test_statistic, df_p_value, _, _, _, _ = adfuller(series)
    return df_p_value

def inverse_diff(forecast, original_series, d):
    inverted = forecast.copy()
    for _ in range(d):
        inverted = np.cumsum(np.insert(inverted, 0, original_series.iloc[-d:].values))
        inverted = inverted[d:]
    return inverted

def forecast_price(stockprice,vix_data):
    stockprice.columns = stockprice.columns.droplevel(1)   #To drop Ticker column from a dataframe
    vix_data.columns = vix_data.columns.droplevel(1) 
    stockprice_original = stockprice.copy()
    vixdata_original = vix_data.copy()
    print("Inside forecast_price() function:\n",stockprice)
    print("vix data:\n",vix_data)
    
    # Convert 'Date' column to datetime format
    stockprice['Date'] = pd.to_datetime(stockprice['Date'])

    # Filter data from January 1, 2022, onwards
    stockprice_df = stockprice[stockprice['Date'] >= '2020-01-01']
    
    testsize = -90
    
    X_train = stockprice_df.iloc[:testsize]
    X_test = stockprice_df.iloc[testsize:]
    #y_train = stockprice_df[:train_size]
    #y_test = stockprice_df[train_size:]
    
    print("train_size:",len(X_train))
    print("train_size:",len(X_test))
    #print("stockprice_df:\n",stockprice_df)
    
    stockprice_df_close = X_train['Close']
    stockprice_df_close = pd.to_numeric(stockprice_df_close, errors='coerce').dropna() 
    stockprice_df_close_test = pd.to_numeric(X_test['Close'], errors='coerce').dropna() 
    
    print("stockprice_df_close_test before:\n",stockprice_df_close_test)
    
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
    
    '''
    model = auto_arima(
        stockprice_df_close,
        trace=True,
        suppress_warnings=True,
        stepwise=True,
        max_p=5, max_q=5
    )
    best_order = model.order
    '''
    print(f'Best ARIMA order based on AIC: {best_order}')
    print(f'Best ARIMA order based on AIC d value: {best_order[1]}')
    
    model = ARIMA(stockprice_df_close, order=best_order)
    model_fit = model.fit()
    print("model_fit:\n",model_fit)
    # Step 3: Make predictions on the test set
    
    
    forecast = model_fit.forecast(steps=len(stockprice_df_close_test))  # Forecast for the test set length
    #forecast_inverted = forecast.copy()
    #for i in range(d, 0, -1):
        #last_value = original_series.iloc[-i]  # Anchor from original data
        #forecast_original = np.cumsum(forecast_inverted) + last_value
    
    forecast_original = inverse_diff(forecast, original_series, best_order[1])
        
    # Step 4: Evaluate the ARIMA model
    mae_arima = mean_absolute_error(stockprice_df_close_test, forecast_original)
    mse_arima = mean_squared_error(stockprice_df_close_test, forecast_original)
    rmse_arima = np.sqrt(mse_arima)

    print(f'Mean Absolute Error (MAE): {mae_arima}')
    print(f'Mean Squared Error (MSE): {mse_arima}')
    print(f'Root Mean Squared Error (RMSE): {rmse_arima}')
    
    print("stockprice_df_close_test:\n",stockprice_df_close_test)
    
    print("forecast_inverted:\n",forecast_original)

    # Step 5: Plot actual vs predicted values
    #plt.figure(figsize=(10, 6))
    #plt.plot(stockprice_df_close_test.index, stockprice_df_close_test, label='Actual', color='blue')
    #plt.plot(stockprice_df_close_test.index, forecast_original, label='Predicted', color='red')
    #plt.xlabel('Date')
    #plt.ylabel('Price')
    #plt.legend()
    #plt.show()
    
    print("ARIMA model completed")
    
    ##########################################################################################################################
    
    stock_data = stockprice_original[['Close']].rename(columns={'Close': 'Stock_Close'})
    vix_data = vixdata_original[['Close']].rename(columns={'Close': 'VIX_Close'})
    
    df = stock_data.merge(vix_data, left_index=True, right_index=True)
    
    print("df:\n",df.head(5))
    
    
        
        
        
        
        
        
            
                    
                    
                            