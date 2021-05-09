# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 17:26:34 2021

@author: satyajit.mahapatra
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

df = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\co2_mm_mlo.csv')

df['date'] = pd.to_datetime({'year':df['year'], 'month':df['month'], 'day':1})

df = df.set_index('date')
df.index.freq = 'MS'

df['interpolated'].plot()

result = seasonal_decompose(df['interpolated'], model = 'add')
result.plot()

auto_arima(df['interpolated'], seasonal = True, m = 12).summary()

train = df.iloc[:717]
test = df.iloc[717:]

model = SARIMAX(train['interpolated'], order = (0,1,1), seasonal_order = (1,0,1,12))

results = model.fit()

start = len(train)
end = len(train) + len(test) - 1

predictions = results.predict(start = start, end = end, typ = 'levels').rename('SARIMA Predictions')

test['interpolated'] .plot(legend = True, figsize = (12,8))
predictions.plot(legend = True)

from statsmodels.tools.eval_measures import rmse

error = rmse(test['interpolated'], predictions)
error

#Forecast into the unknown future
model = SARIMAX(df['interpolated'], order = (0,1,1), seasonal_order = (1,0,1,12))
results = model.fit()

fcast = results.predict(len(df), len(df) + 11, typ = 'level').rename('SARIMA FORECAST')

df['interpolated'].plot(legend = True, figsize = (12,9))
fcast.plot(legend = True, figsize = (12,9))

###SARIMAX













