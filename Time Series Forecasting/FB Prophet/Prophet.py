# -*- coding: utf-8 -*-
"""
Created on Sun May  9 12:10:11 2021

@author: satyajit.mahapatra
"""
import pandas as pd
from fbprophet import Prophet
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

df = pd.read_csv(r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\BeerWineLiquor.csv')

#renaming column names as per facebook prophet naming convention
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

#Declaring, fitting and predicting based on the model
m = Prophet()
m.fit(df)

#Placeholder to hold forecast
future = m.make_future_dataframe(periods = 24, freq = 'MS')

#creating forecast
forecast = m.predict(future)

forecast.columns

#keeping only the necessary columns
forecast[['ds','yhat_lower', 'yhat_upper','yhat']]

m.plot(forecast)


