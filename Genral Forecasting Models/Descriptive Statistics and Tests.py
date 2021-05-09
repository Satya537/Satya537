# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 18:10:55 2021

@author: satyajit.mahapatra
"""
import pandas as pd
import numpy as np

#Loading seasonal dataset
df1 = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\airline_passengers.csv', index_col = 'Month', parse_dates = True)
df1.index.freq = 'MS' #to enable statsmodels to run

#Loading non seasonal dataset
df2 = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\DailyTotalFemaleBirths.csv', index_col = 'Date', parse_dates = True)
df2.index.freq = 'D'

#Testing Augmented Dicky Fuller Test
from statsmodels.tsa.stattools import adfuller

adfuller(df1['Thousands of Passengers'])

dftest = adfuller(df1['Thousands of Passengers'],autolag='AIC')
dfout = pd.Series(dftest[0:4],index=['ADF test statistic','p-value','# lags used','# observations'])

for key,val in dftest[4].items():
    dfout[f'critical value ({key})']=val
print(dfout)

#Custom function for Dicky Fuller Test
from statsmodels.tsa.stattools import adfuller

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
        
df2.plot()

adf_test(df2['Births'])


df3 = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\samples.csv', index_col = 0, parse_dates = True)
df3.index.freq = 'MS'

df3[['a','d']].plot(figsize = (12,8))

#import causality test
from statsmodels.tsa.stattools import grangercausalitytests

grangercausalitytests(df3[['a','b']], maxlag = 3);

grangercausalitytests(df3[['b','d']], maxlag = 3); # p > 0.05 indicates causality

#Exploring seasonality
from statsmodels.graphics.tsaplots import month_plot, quarter_plot

month_plot(df1['Thousands of Passengers'])

dfq = df1['Thousands of Passengers'].resample(rule = 'Q').mean()

quarter_plot(dfq)













