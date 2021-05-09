# -*- coding: utf-8 -*-
"""
Created on Sun May  9 11:42:20 2021

@author: satyajit.mahapatra
"""
import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from statsmodels.tsa.api import VAR
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse

import warnings
warnings.filterwarnings("ignore")

file_path = r'C:\One-Drive\OneDrive - Tredence\Udemy Courses\Python for Time Series Data Analysis\1. Introduction\1.1 UDEMY_TSA_FINAL.zip\UDEMY_TSA_FINAL\Data\\'

df = pd.read_csv(file_path + 'M2SLMoneyStock.csv', index_col = 0, parse_dates = True)
df.index.freq = 'MS'

sp = pd.read_csv(file_path + 'PCEPersonalSpending.csv', index_col = 0, parse_dates = True)
sp.index.freq = 'MS'

#COmbning spending and money
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

#checking with auto arima to see what values for p and q to choose
auto_arima(df['Spending'], maxiter = 1000) #ooutput order = (1,1,2)
auto_arima(df['Money'], maxiter = 1000) #ooutput order = (1,2,2)

#the middle term suggests the differencing term, it suggests 1 for spending and 2nd order differncing for Money. 
#We are differencing them both on 2nd order now so that they can start at the same date

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

#Fitting hte VARMA Model
model = VARMAX(train, order = (1,2), trend = 'c')
results = model.fit(maxiter = 1000, disp = False)
results.summary()

#predicitng the next 12 values 

df_forecast = results.forecast(12) #Varma returns a dataframe forecast which is differenced twice now, convert it back into original

#Invert the transformation
# Add the most recent first difference from the training side of the original dataset to the forecast cumulative sum
df_forecast['Money1d'] = (df['Money'].iloc[-nobs-1]-df['Money'].iloc[-nobs-2]) + df_forecast['Money'].cumsum()

# Now build the forecast values from the first difference set
df_forecast['MoneyForecast'] = df['Money'].iloc[-nobs-1] + df_forecast['Money1d'].cumsum()

# Add the most recent first difference from the training side of the original dataset to the forecast cumulative sum
df_forecast['Spending1d'] = (df['Spending'].iloc[-nobs-1]-df['Spending'].iloc[-nobs-2]) + df_forecast['Spending'].cumsum()

# Now build the forecast values from the first difference set
df_forecast['SpendingForecast'] = df['Spending'].iloc[-nobs-1] + df_forecast['Spending1d'].cumsum()

#Combining the original forecast
df_final = pd.concat([df.iloc[-12:], df_forecast[['MoneyForecast','SpendingForecast']]],axis = 1)

#Plotting
df_final[['Money', 'MoneyForecast']].plot()
df_final[['Spending', 'SpendingForecast']].plot()

#Evaluating the model
rmse(df['Money'][-nobs:], df_forecast['MoneyForecast'])
df['Money'].mean()

rmse(df['Spending'][-nobs:], df_forecast['SpendingForecast'])
df['Spending'].mean()
















        
        
