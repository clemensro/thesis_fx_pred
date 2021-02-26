# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:36:16 2020

@author: Clemens
"""

#%%

#Imports

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
from pandas import read_csv
from datetime import datetime 

from math import sqrt

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

import pmdarima as pm
from pmdarima.arima import ARIMA
from bokeh.plotting import figure, show, output_notebook


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
%matplotlib inline

#%%

#Import Data

def GetData(fileName):
    return pd.read_csv(fileName, header=0, parse_dates=[0], index_col=0, encoding='utf-8')

df = GetData(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\data.csv')

#Choose currency pair
df_EURUSD = pd.DataFrame(df, columns=['EURUSD'])

#%%

#Decomposition

#Define Figuresize
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size


fig_size1 = plt.rcParams["figure.figsize"]
fig_size1[0] = 10
fig_size1[1] = 5
plt.rcParams["figure.figsize"] = fig_size1


fig = plt.figure()
df_EURUSD.EURUSD.plot(label = 'EURUSD')
plt.ylabel('EURUSD')
plt.legend()
plt.show()
fig.savefig(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\plots\\EURUSD_plot.pdf', bbox_inches='tight')



from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df_EURUSD.EURUSD, model='additive')
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)
fig1 = result.plot()
plt.show()


fig1.savefig(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\plots\\EURUSD_decompose.pdf', bbox_inches='tight')


#%%



from statsmodels.tsa.stattools import adfuller

#Converting to log return and applying Dickey Fuller

df_EURUSD['log_return'] = np.log(1 + df_EURUSD.EURUSD.pct_change())


#Original Series

print(" > Is the data stationary ?")
df_EURUSDtest = adfuller(df_EURUSD.EURUSD, autolag='AIC')
print("Test statistic = {:.3f}".format(df_EURUSDtest[0]))
print("P-value = {:.3f}".format(df_EURUSDtest[1]))
print("Critical values :")
for k, v in df_EURUSDtest[4].items():
    print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v<df_EURUSDtest[0] else "", 100-int(k[:-1])))

#Log return Series
print("\n > Is the de-trended data stationary ?")
df_EURUSDtest = adfuller(df_EURUSD.log_return.dropna(), autolag='AIC')
print("Test statistic = {:.3f}".format(df_EURUSDtest[0]))
print("P-value = {:.3f}".format(df_EURUSDtest[1]))
print("Critical values :")
for k, v in df_EURUSDtest[4].items():
    print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v<df_EURUSDtest[0] else "", 100-int(k[:-1])))
    



#%%

#Autocorrelation
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 3
plt.rcParams["figure.figsize"] = fig_size

# Original Series
fig, axes = plt.subplots(1, 2, sharex=True,)
plot_acf(df_EURUSD.EURUSD, ax=axes[0], alpha=.05, lags = 30)
axes[0].set(ylabel = 'Autocorrelation', xlabel='Lags')
plot_pacf(df_EURUSD.EURUSD, ax=axes[1], alpha=.05, lags = 30)
axes[1].set(ylabel = 'Partial autocorrelation', xlabel = 'Lags')
plt.show()
fig.savefig(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\plots\\EURUSD_acf.pdf', bbox_inches='tight')

#log return series
fig, axes = plt.subplots(1, 2, sharex=True)
plot_acf(df_EURUSD.log_return.dropna(), ax=axes[0], alpha=.05, lags = 30)
axes[0].set(ylabel = 'Autocorrelation', xlabel='Lags')
plot_pacf(df_EURUSD.log_return.dropna(), ax=axes[1], alpha=.05, lags = 30)
axes[1].set(ylabel = 'Partial autocorrelation', xlabel = 'Lags')
plt.show()
fig.savefig(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\plots\\EURUSD_acf_return.pdf', bbox_inches='tight')



#%%

# Split Data in Training Data (80%) and Test Data (20%)
NumberOfElements_EURUSD = int(len(df_EURUSD.log_return.dropna()))
TrainingSize_EURUSD = int(NumberOfElements_EURUSD * 0.8)
TrainingData_EURUSD = df_EURUSD.log_return.dropna()[:TrainingSize_EURUSD]
TestData_EURUSD = df_EURUSD.log_return.dropna()[TrainingSize_EURUSD:]

#%%

#Finding the best ARIMA order with auto_arima

stepwise_fit = pm.auto_arima(TrainingData_EURUSD, start_p=1, start_q=1, max_p=3, max_q=3,
                             start_P=0, seasonal=False, d=1, trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True,  # don't want convergence warnings
                             stepwise=True)  # set to stepwise
stepwise_fit.summary()


#%%

#Define ARIMA Forecasting Function
def StartARIMAForecasting(Actual, P, D, Q):
	model = ARIMA(Actual, order=(P, D, Q))
	model_fit = model.fit(disp=0)
	prediction = model_fit.forecast()[0]
	return prediction


#%%

#Random Walk forecast
Actual = [x for x in TrainingData_EURUSD]
Predictions_rw_EURUSD = list()

#In a for loop, predict values using ARIMA model
for timepoint in range(len(TestData_EURUSD)):
	ActualValue =  TestData_EURUSD[timepoint]
	#forcast value
	Prediction = StartARIMAForecasting(Actual, 0, 1, 0)
	print('Actual=%f, Predicted=%f' % (ActualValue, Prediction))
	#add it in the list
	Predictions_rw_EURUSD.append(Prediction)
	Actual.append(ActualValue)

#Predictions in Series format
Predictions_rw_EURUSD = pd.Series(Predictions_rw_EURUSD)
Predictions_rw_EURUSD = pd.Series((v[0] for v in Predictions_rw_EURUSD))
Predictions_rw_EURUSD.index = TestData_EURUSD.index.copy()


#Forecasting accuracy scores
def forecast_accuracy(forecast, actual):
    mae = np.mean(np.abs(forecast - actual))    # MAE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    return({'mae': mae, 'rmse':rmse})

print(forecast_accuracy(Predictions_rw_EURUSD, TestData_EURUSD))

#MAE: 0.00377972
#RMSE: 0.00498926

#Convert from log returns to actual values
from sklearn.model_selection import train_test_split

train_actual_EURUSD, test_actual_EURUSD = train_test_split(df_EURUSD.EURUSD, test_size = 0.2, shuffle = False)

Predictions_rw_EURUSD1 = Predictions_rw_EURUSD.iloc[1:]

pred_rw_EURUSD_actual = np.exp(Predictions_rw_EURUSD1)*test_actual_EURUSD
pred_rw_EURUSD_actual = pred_rw_EURUSD_actual.dropna()

mape = np.mean(np.abs(pred_rw_EURUSD_actual - test_actual_EURUSD)/np.abs(test_actual_EURUSD))  # MAPE
print(mape)

#MAPE: 0.00286045


#%%
#ARIMA Forecasting

Actual = [x for x in TrainingData_EURUSD]
Predictions_arima_EURUSD = list()

#in a for loop, predict values using ARIMA model
for timepoint in range(len(TestData_EURUSD)):
	ActualValue =  TestData_EURUSD[timepoint]
	#forcast value
	Prediction = StartARIMAForecasting(Actual, 0, 1, 3)
	print('Actual=%f, Predicted=%f' % (ActualValue, Prediction))
	#add it in the list
	Predictions_arima_EURUSD.append(Prediction)
	Actual.append(ActualValue)

#Predictions in Series format
Predictions_arima_EURUSD = pd.Series(Predictions_arima_EURUSD)
Predictions_arima_EURUSD = pd.Series((v[0] for v in Predictions_arima_EURUSD))
Predictions_arima_EURUSD.index = TestData_EURUSD.index.copy()


#Forecasting accuracy scores
def forecast_accuracy(forecast, actual):
    mae = np.mean(np.abs(forecast - actual))    # MAE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    return({'mae': mae, 'rmse':rmse})

print(forecast_accuracy(Predictions_arima_EURUSD, TestData_EURUSD))

#MAE: 0.002859522
#RMSE: 0.003699132

#Convert from log returns to actual values
from sklearn.model_selection import train_test_split

Predictions_arima_EURUSD1 = Predictions_arima_EURUSD.iloc[1:]

pred_arima_EURUSD_actual = np.exp(Predictions_arima_EURUSD1)*test_actual_EURUSD
pred_arima_EURUSD_actual = pred_arima_EURUSD_actual.dropna()


mape = np.mean(np.abs(pred_arima_EURUSD_actual - test_actual_EURUSD)/np.abs(test_actual_EURUSD))  # MAPE
print(mape)

#MAPE:0.00014795213245645539

    
#%%

#Plots with the result of the ARIMA, Random Walk and k-NN Regression forecast.
fig_size1 = plt.rcParams["figure.figsize"]
fig_size1[0] = 12
fig_size1[1] = 5
plt.rcParams["figure.figsize"] = fig_size1


fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey = True)
l1 = ax1.plot(test_actual_EURUSD[-20:], label='Actual values')
l2 = ax1.plot(pred_rw_EURUSD_actual[-20:], color='red', label='Forecast values')


l3 = ax2.plot(test_actual_EURUSD[-20:], label='Actual values')
l4 = ax2.plot(pred_arima_EURUSD_actual[-20:], color='red', label='Forecast values')


l5 = ax3.plot(test_actual_EURUSD[-20:], label='Actual values')
l6 = ax3.plot(pred_knn_EURUSD_actual[-20:], color='red', label='Forecast values' )


ax1.set(title='Random walk', ylabel='EURUSD', xlabel='Dates')
ax2.set(title='ARIMA(0,1,3)', ylabel='EURUSD', xlabel = 'Dates')
ax3.set(title='k-NN Regression with k = 40', ylabel = 'EURUSD', xlabel='Dates')
fig.autofmt_xdate()



fig.legend(('Actual values', 'Forecast values'), loc=(0.375,0.), borderaxespad = 2.5,
        ncol=2,)
fig.savefig(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\plots\\EURUSD_results_plot.pdf', bbox_inches='tight')


