# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:50:59 2021

@author: satyajit.mahapatra
"""
import pandas as pd
from fbprophet import Prophet
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from statsmodels.tools.eval_measures import rmse
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric


df = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\Miles_Traveled.csv')

#Formatting
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

train = df.iloc[:576]
test = df.iloc[576:]

m = Prophet()
m.fit(train)

future = m.make_future_dataframe(periods = 12, freq = 'MS')
forecast = m.predict(future)

ax = forecast.plot(x = 'ds', y = 'yhat', label = 'Predictions', legend = True, figsize = (12,8))
test.plot(x = 'ds', y = 'y', label = 'True Test Data', legend = True, ax = ax, xlim = ('2018-01-01','2019-01-01'))

predictions = forecast.iloc[-12:]['yhat']

rmse(predictions, test['y'])
test['y'].mean()

#using prophet internal diagnostics
initial = 5 * 365
initial = str(initial) + ' days'

period = 5 * 365
period = str(period) + ' days'

horizon = 365
horizon = str(horizon) + ' days'

#Cross validation
df_cv = cross_validation(m, initial = initial, period = period, horizon = horizon)

#Error Evaluation
performance_metrics(df_cv)

#plotting cross validation
plot_cross_validation_metric(df_cv, metric = 'rmse')

















