# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:29:42 2021

@author: satyajit.mahapatra
"""
import pandas as pd
import numpy as np

from statsmodels.tsa.ar_model import AR, ARResults

df = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\uspopulation.csv', index_col = 'DATE', parse_dates = True)

df.index.freq = 'MS'

df.plot()

#Splitting into train test
train = df.iloc[:84]
test = df.iloc[84:]

import warnings
warnings.filterwarnings('ignore')

model = AR(train['PopEst'])
AR1fit = model.fit(maxlag = 1)

AR1fit.k_ar
AR1fit.params

start = len(train)
end = len(train) + len(test) - 1

AR1fit.predict(start = start, end = end, dynamic = False)

predictions1 = AR1fit.predict(start = start, end = end, dynamic = False)

predictions1 = predictions1.rename('AR(1) Predictions')

test.plot(figsize = (12,8), legend = True)
predictions1.plot(figsize = (12,8), legend = True)

#Order 2 model
model2 = AR(train['PopEst']) 
AR2fit = model2.fit(maxlag = 2)

AR2fit.params

predictions2 = AR2fit.predict(start = start, end = end, dynamic = False)
predictions2 = predictions2.rename('AR(2) Predictions')

test.plot(figsize = (12,8), legend = True)
predictions1.plot(figsize = (12,8), legend = True)
predictions2.plot(figsize = (12,8), legend = True)

model3 = AR(train['PopEst'])
ARfit = model3.fit(ic = 't-stat')

ARfit.params

predictions8 = ARfit.predict(start, end)
predictions8 = predictions8.rename('AR(8) Predictions')

from sklearn.metrics import mean_squared_error
labels = ['AR1', 'AR2','AR8']

preds = [predictions1, predictions2, predictions8]


for i in range(3):
    error = mean_squared_error(test['PopEst'], preds[i])
    print(f'{labels[i]} MSE was :{error}')

test.plot(figsize = (12,8), legend = True)
predictions1.plot(figsize = (12,8), legend = True)
predictions2.plot(figsize = (12,8), legend = True)
predictions8.plot(figsize = (12,8), legend = True)

model_future = AR(df['PopEst'])
ARfit_future = model_future.fit()

future_predictions = ARfit_future.predict(start = len(df), end = len(df) + 12).rename('Forecast')

ARfit_future.params

df.plot(figsize = (12,8), legend = True)
future_predictions.plot(figsize = (12,8), legend = True)





