# -*- coding: utf-8 -*-
"""
Created on Sun May  2 20:02:48 2021

@author: satyajit.mahapatra
"""
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\4.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\RestaurantVisitors.csv', index_col = 'date', parse_dates = True)

df.index.freq = "D"

#Dropping na values
df1 = df.dropna()

df1.columns

column_list = ['rest1', 'rest2', 'rest3', 'rest4', 'total']

for col in column_list:
    df1[col] = df1[col].astype(int)
    
#plotting df1 total
df1['total'].plot(figsize = (16,5))

#Testing the effect of holidays
ax = df1['total'].plot(figsize = (16,5))

for day in df1.query('holiday == 1').index:
    ax.axvline(x = day, color = 'black', alpha = 0.8);
    
#Ets decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df1['total'])
result.plot();
result.seasonal.plot();

#train test split
train = df1.iloc[:436]
test = df1.iloc[436:]


#using auto arima
from pmdarima import auto_arima

auto_arima(df1['total'], seasonal = True, m = 7).summary()

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train['total'], order = (1,0,0), seasonal_order = (2,0,0,7), enforce_invertibility = False)

results = model.fit()
results.summary()

start = len(train)
end = start + len(test) - 1

predictions = results.predict(start, end).rename('SARIMA Model')

test['total'].plot(legend = True, figsize = (15, 8))
predictions.plot(legend = True)

#Checking impact of holidays
ax = test['total'].plot(legend = True, figsize = (16,5))
predictions.plot(legend = True)

for day in test.query('holiday == 1').index:
    ax.axvline(x = day, color = 'black', alpha = 0.8);

#evaluate the model
from statsmodels.tools.eval_measures import rmse
rmse(test['total'], predictions)












