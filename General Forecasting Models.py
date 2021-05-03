# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:36:19 2021

@author: satyajit.mahapatra
"""
import pandas as pd
import numpy as np

df  = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\airline_passengers.csv', index_col = 'Month', parse_dates = True)

df.index.freq = 'MS' #to enable statsmodels to run

#Creating train test split
train_data = df.iloc[:108]
test_data = df.iloc[108:]

#Forecasting using statsmodels and Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

fitted_model = ExponentialSmoothing(train_data['Thousands of Passengers'], trend = 'mul', seasonal='mul', seasonal_periods=12).fit()
test_predictions = fitted_model.forecast(36)

train_data['Thousands of Passengers'].plot(legend = True, label = 'TRAIN', figsize = (12,8))
test_data['Thousands of Passengers'].plot(legend = True, label = 'TEST')
test_predictions.plot(legend = True, label = 'Predictions', xlim = ['1958-01-01','1961-01-01'])

#Evaluating metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

mean_squared_error(test_data, test_predictions)
mean_absolute_error(test_data, test_predictions)

np.sqrt(mean_squared_error(test_data, test_predictions)) #root mean squared error

final_model = ExponentialSmoothing(df['Thousands of Passengers'], trend = 'mul', seasonal = 'mul', seasonal_periods=12).fit()
forecast_predictions = final_model.forecast(36)

df['Thousands of Passengers'].plot(figsize = (12,5))
forecast_predictions.plot()

df2 = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\samples.csv')
df2['a'].plot() #stationary data
df2['b'].plot() #non stationary data


#Differncing to make it stationary
from statsmodels.tsa.statespace.tools import diff

diff(df2['b'], k_diff = 1).plot() #stationary time series

#PACF and ACF
import statsmodels.api as sm
from statsmodels.tsa.stattools import acovf, acf, pacf, pacf_yw, pacf_ols

df1 = df.copy() #non stationary data
df.index.freq = 'MS' 

df2 = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\DailyTotalFemaleBirths.csv', index_col = 'Date', parse_dates = True)
df2 = df2.copy() #stationary data
df2.index.freq = 'D'

df = pd.DataFrame({'a':[13, 5, 11, 12, 9]})
acf(df['a'])

pacf_yw(df['a'], nlags = 4, method = 'mle') #maximum liklihood estimation = mle, yw = yiol walker equation
pacf_yw(df['a'], nlags = 4, method = 'unbiased')

pacf_ols(df['a'], nlags = 4) #ols = ordinary least squares equation

#plotting using statsmodels
from pandas.plotting import lag_plot

lag_plot(df1['Thousands of Passengers'])

lag_plot(df2['Births'])

#plotting acf and pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df1, lags = 40)

plot_acf(df2, lags = 40)

plot_pacf(df2, lags = 40, title='Daily Female Births')













