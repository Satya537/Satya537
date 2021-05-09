# -*- coding: utf-8 -*-
"""
Created on Sun May  9 17:09:53 2021

@author: satyajit.mahapatra
"""
import pandas as pd
from fbprophet import Prophet
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from statsmodels.tools.eval_measures import rmse
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric, add_changepoints_to_plot


df = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\HospitalityEmployees.csv')

#Formatting
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods = 12, freq = 'MS')
forecast = m.predict(future)

fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)