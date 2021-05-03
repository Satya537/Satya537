# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:30:44 2021

@author: satyajit.mahapatra
"""
import pandas as pd
import numpy as np
import matplotlib as plt
from statsmodels.tsa.filters.hp_filter import hpfilter
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\macrodata.csv')

gdp_cycle, gdp_trend = hpfilter(df['realgdp'], lamb = 1600)

df['trend'] = gdp_trend

gdp_trend.plot()

df[['trend','realgdp']].plot()

#ETS Decomposition

df_2  = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\airline_passengers.csv', index_col = 'Month', parse_dates = True)
df_2.dropna(inplace = True)

df_2.plot()

from statsmodels.tsa.seasonal import seasonal_decompose
df_2['Month'] = pd.to_datetime(df_2['Month'])

result = seasonal_decompose(df_2['Thousands of Passengers'], model = 'multiplicative')

#prints the trend, seasonality, residual component
result.seasonal
result.residual
result.trend

#ploting the components
result.plot()

#Exponenetially weighted moving average
df_2['6-month-SMA'] = df_2['Thousands of Passengers'].rolling(window = 6).mean()
df_2.plot()

df_2['EWMA'] = df_2['Thousands of Passengers'].ewm(span = 12).mean()
df_2[['Thousands of Passengers', 'EWMA']].plot()


##Holt Winters Method
df = df_2.copy()
df.index.freq = 'MS' #alias for the type of period under consideration

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

span = 12
alpha = 2/(span+1)

df['EWMA'] = df['Thousands of Passengers'].ewm(alpha = alpha, adjust = False).mean()

#Using Simple Exponential Smoothing
model = SimpleExpSmoothing(df['Thousands of Passengers'])
fitted_model = model.fit(smoothing_level = alpha, optimized = False) #smoothing level is the same as alpha
df['SES12'] = fitted_model.fittedvalues.shift(-1)

#Double and Triple Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df['DES_add_12'] = ExponentialSmoothing(df['Thousands of Passengers'], trend = 'add').fit().fittedvalues.shift(-1)
df[['Thousands of Passengers', 'SES12', 'DES_add_12']].iloc[:24].plot(figsize = (12,5))

df['DES_mult_12'] = ExponentialSmoothing(df['Thousands of Passengers'], trend = 'mul').fit().fittedvalues.shift(-1)
df[['Thousands of Passengers', 'SES12', 'DES_add_12', 'DES_mult_12']].iloc[:24].plot(figsize = (12,5))

#Triple Exponential Smoothing
df['TES_mul_12'] = ExponentialSmoothing(df['Thousands of Passengers'], trend = 'mul', seasonal = 'mul', seasonal_periods = 12).fit().fittedvalues.shift(-1)
df[['Thousands of Passengers', 'DES_mult_12', 'TES_mul_12']].iloc[:24].plot(figsize = (12,5))















