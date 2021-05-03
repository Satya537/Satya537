# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 20:35:15 2021

@author: satyajit.mahapatra
"""
import pandas as pd
import numpy as np
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

df1 = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\airline_passengers.csv', index_col = 'Month', parse_dates = True)
df1.index.freq = 'MS'

#FemaleBirth
df2 = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\DailyTotalFemaleBirths.csv', index_col = 'Date', parse_dates = True)
df2.index.freq = 'D'

stepwise_fit = auto_arima(df2['Births'], start_p = 0, start_q = 0, max_p = 6, max_q = 3, seasonal = False, trace = True)
stepwise_fit.summary()

stepwise_ft= auto_arima(df1['Thousands of Passengers'], start_p =0, start_q =0, max_p = 4, max_q = 4, seasonal = True, trace = True, m = 12)
stepwise_fit.summary()

#Importing arima and arma models
from statsmodels.tsa.arima_model import ARMA, ARIMA, ARMAResults, ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pmdarima import auto_arima

df2 = df2[:120]

df4 = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\TradeInventories.csv', index_col = 'Date', parse_dates = True)
df4.index.freq = 'MS'

#using the ARMA model
df2['Births'].plot(figsize = (12,5))

#running the augmented dicky fuller test
def adf_test(series,title=''):
    from statsmodels.tsa.stattools import adfuller
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")

adf_test(df2['Births'])

auto_arima(df2['Births'], seasonal = False).summary()

train = df2.iloc[:90]
test = df2.iloc[90:]

model = ARMA(train['Births'], order = (2,2))
results = model.fit()
results.summary()

#using this to forecast
start = len(train)
end = len(train) + len(test) - 1

predictions = results.predict(start, end).rename('ARMA(2,2) Predictions')

test['Births'].plot(figsize = (12,8), legend = True)
predictions.plot(legend = True)


df4.plot(figsize = (12,9))

#Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df4['Inventories'], model = 'add')

result.plot()

auto_arima(df4['Inventories'], seasonal = False).summary()

from statsmodels.tsa.statespace.tools import diff

df4['Diff_1'] = diff(df4['Inventories'], k_diff = 1)

adf_test(df4['Diff_1'])

#Running the ACF and PACF plot for the value
plot_acf(df4['Inventories'], lags = 40)
plot_pacf(df4['Inventories'], lags = 40)


len(df4)

train= df4.iloc[:252]
test = df4.iloc[252:]

model = ARIMA(train['Inventories'], order = (1,1,1))
results = model.fit()
results.summary()

start = len(train)
end = len(train) + len(test) - 1

predictions = results.predict(start = start, end = end, typ = 'levels').rename('ARIMA (1,1,) Predictions')

test['Inventories'].plot(legend = True, figsize = (12,9))
predictions.plot(legend = True)

from statsmodels.tools.eval_measures import rmse

error = rmse(test['Inventories'], predictions)

test['Inventories'].mean()



























