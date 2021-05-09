# -*- coding: utf-8 -*-
"""
Created on Sun May  2 21:47:02 2021

@author: satyajit.mahapatra
"""
import pandas as pd
import numpy as np

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse

import warnings
warnings.filterwarnings("ignore")

file_path = r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\\'

df = pd.read_csv(file_path + 'M2SLMoneyStock.csv', index_col = 0, parse_dates = True)
df.index.freq = 'MS'

sp = pd.read_csv(file_path + 'PCEPersonalSpending.csv', index_col = 0, parse_dates = True)
sp.index.freq = 'MS'


df = df.join(sp)

#dropping null values
df.dropna(inplace = True)

df.plot(figsize = (12,8))

#ad fuller test
def adf_test(series,title=''):
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
        
#Checking across columns
adf_test(df['Money'])
adf_test(df['Spending'])

#both are not stationary,so we will differentiate
df_transformed = df.diff()

adf_test(df_transformed['Money']) #not stationary yet
adf_test(df_transformed['Spending']) #stationary 

#Differencing again to make it stationary 
df_transformed = df_transformed.diff().dropna()
adf_test(df_transformed['Money']) #stationary now

#performing train test split
nobs = 12

train = df_transformed[:-nobs]
test = df_transformed[-nobs:]

#GRID Search for order determination p for the AR part of the vector auto regression model, p not generated in auto arima
model = VAR(train)

for p in [1,2,3,4,5,6,7]:
    results = model.fit(p) #when fitting specify what order is required
    print(f'ORDER {p}')
    print(f'AIC: {results.aic}')
    print('\n')

#based on aic values VAR with p = 5
results = model.fit(5)
results.summary()

#predicitng the next 12 values 
#we need the last p lagged values right before the test starts 
#creating the numpy input array 
train.values[-5:] #numpy array of order p*2


z = results.forecast(y = train.values[-5:], steps = 12) #the forecast parameter accepts a numpy array of dimension (p*k) where p is the order and k is the number of variables 

#transforming the output into a dataframe
idx = pd.date_range('2015-01-01', periods = 12, freq = 'MS')

df_forecast = pd.DataFrame(z, index = idx, columns = ['Money_2d', 'Spending_2d']) #this is the differenced forecast

#we need to reverse the differenced

#Invert the transformation
# Add the most recent first difference from the training side of the original dataset to the forecast cumulative sum
df_forecast['Money1d'] = (df['Money'].iloc[-nobs-1]-df['Money'].iloc[-nobs-2]) + df_forecast['Money_2d'].cumsum()

# Now build the forecast values from the first difference set
df_forecast['MoneyForecast'] = df['Money'].iloc[-nobs-1] + df_forecast['Money1d'].cumsum()

# Add the most recent first difference from the training side of the original dataset to the forecast cumulative sum
df_forecast['Spending1d'] = (df['Spending'].iloc[-nobs-1]-df['Spending'].iloc[-nobs-2]) + df_forecast['Spending_2d'].cumsum()

# Now build the forecast values from the first difference set
df_forecast['SpendingForecast'] = df['Spending'].iloc[-nobs-1] + df_forecast['Spending1d'].cumsum()

#Plotting
test_range = df[-nobs:]
test_range.plot(figsize = (12,5))

df_forecast[['MoneyForecast', 'SpendingForecast']].plot()

test_range['Money'].plot(legend = True, figsize = (12,8))
df_forecast['MoneyForecast'].plot(legend = True)

test_range['Spending'].plot(legend = True, figsize = (12,8))
df_forecast['SpendingForecast'].plot(legend = True)

#Evaluating the model
rmse(test_range['Money'], df_forecast['MoneyForecast'])
test_range['Money'].mean()

rmse(test_range['Spending'], df_forecast['SpendingForecast'])
test_range['Spending'].mean()
















        
        
