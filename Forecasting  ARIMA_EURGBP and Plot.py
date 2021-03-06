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
df_EURGBP = pd.DataFrame(df, columns=['EURGBP'])

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
df_EURGBP.EURGBP.plot(label = 'EURGBP')
plt.ylabel('EURGBP')
plt.legend()
plt.show()
fig.savefig(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\plots\\EURGBP_plot.pdf', bbox_inches='tight')


from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df_EURGBP.EURGBP, model='additive')
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)
fig1 = result.plot()
plt.show()
fig1.savefig(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\plots\\EURGBP_decompose.pdf', bbox_inches='tight')

#%%


from statsmodels.tsa.stattools import adfuller
#Converting to log return and applying Dickey Fuller

df_EURGBP['log_return'] = np.log(1 + df_EURGBP.EURGBP.pct_change())


#Original Series
print(" > Is the data stationary ?")
df_EURGBPtest = adfuller(df_EURGBP.EURGBP, autolag='AIC')
print("Test statistic = {:.3f}".format(df_EURGBPtest[0]))
print("P-value = {:.3f}".format(df_EURGBPtest[1]))
print("Critical values :")
for k, v in df_EURGBPtest[4].items():
    print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v<df_EURGBPtest[0] else "", 100-int(k[:-1])))
    

#Log Return Series
print("\n > Is the de-trended data stationary ?")
df_EURGBPtest = adfuller(df_EURGBP.log_return.dropna(), autolag='AIC')
print("Test statistic = {:.3f}".format(df_EURGBPtest[0]))
print("P-value = {:.3f}".format(df_EURGBPtest[1]))
print("Critical values :")
for k, v in df_EURGBPtest[4].items():
    print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v<df_EURGBPtest[0] else "", 100-int(k[:-1])))
    



#%%

#Autocorrelation
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 3
plt.rcParams["figure.figsize"] = fig_size

# Original Series
fig, axes = plt.subplots(1, 2, sharex=True)
plot_acf(df_EURGBP.EURGBP, ax=axes[0], alpha=.05, lags = 30)
axes[0].set(ylabel = 'Autocorrelation', xlabel='Lags')
plot_pacf(df_EURGBP.EURGBP, ax=axes[1], alpha=.05, lags = 30)
axes[1].set(ylabel = 'Partial autocorrelation', xlabel = 'Lags')
plt.show()
fig.savefig(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\plots\\EURGBP_acf.pdf', bbox_inches='tight')


#log return series
fig, axes = plt.subplots(1, 2, sharex=True)
plot_acf(df_EURGBP.log_return.dropna(), ax=axes[0], alpha=.05, lags = 30)
axes[0].set(ylabel = 'Autocorrelation', xlabel='Lags')
plot_pacf(df_EURGBP.log_return.dropna(), ax=axes[1], alpha=.05, lags = 30)
axes[1].set(ylabel = 'Partial autocorrelation', xlabel = 'Lags')
plt.show()
fig.savefig(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\plots\\EURGBP_acf_return.pdf', bbox_inches='tight')


#%%

# Split Data in Training Data (80%) and Test Data (20%)
NumberOfElements_EURGBP = int(len(df_EURGBP.log_return.dropna()))
TrainingSize_EURGBP = int(NumberOfElements_EURGBP * 0.8)
TrainingData_EURGBP = df_EURGBP.log_return.dropna()[:TrainingSize_EURGBP]
TestData_EURGBP = df_EURGBP.log_return.dropna()[TrainingSize_EURGBP:]
#%%

#Finding the best ARIMA order with auto_arima

stepwise_fit = pm.auto_arima(TrainingData_EURGBP, start_p=1, start_q=1, max_p=3, max_q=3,
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

#Random walk forecast
Actual = [x for x in TrainingData_EURGBP]
Predictions_rw_EURGBP = list()

#in a for loop, predict values using ARIMA model
for timepoint in range(len(TestData_EURGBP)):
	ActualValue =  TestData_EURGBP[timepoint]
	#forcast value
	Prediction = StartARIMAForecasting(Actual, 0, 1, 0)
	print('Actual=%f, Predicted=%f' % (ActualValue, Prediction))
	#add it in the list
	Predictions_rw_EURGBP.append(Prediction)
	Actual.append(ActualValue)

Predictions_rw_EURGBP = pd.Series(Predictions_rw_EURGBP)
Predictions_rw_EURGBP = pd.Series((v[0] for v in Predictions_rw_EURGBP))
Predictions_rw_EURGBP.index = TestData_EURGBP.index.copy()


#Forecasting accuracy scores
def forecast_accuracy(forecast, actual):
    mae = np.mean(np.abs(forecast - actual))    # MAE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    return({'mae': mae, 'rmse':rmse})

print(forecast_accuracy(Predictions_rw_EURGBP, TestData_EURGBP))

#MAE: 0.004926379
#RMSE: 0.006679597

#Convert from log returns to actual values
from sklearn.model_selection import train_test_split

train_actual_EURGBP, test_actual_EURGBP = train_test_split(df_EURGBP.EURGBP, test_size = 0.2, shuffle = False)

Predictions_rw_EURGBP1 = Predictions_rw_EURGBP.iloc[1:]

pred_rw_EURGBP_actual = np.exp(Predictions_rw_EURGBP1)*test_actual_EURGBP
pred_rw_EURGBP_actual = pred_rw_EURGBP_actual.dropna()

mape = np.mean(np.abs(pred_rw_EURGBP_actual - test_actual_EURGBP)/np.abs(test_actual_EURGBP))  # MAPE
print(mape)

#MAPE: 0.0033127047

#%%
#ARIMA Forecasting

Actual = [x for x in TrainingData_EURGBP]
Predictions_arima_EURGBP = list()

#in a for loop, predict values using ARIMA model
for timepoint in range(len(TestData_EURGBP)):
	ActualValue =  TestData_EURGBP[timepoint]
	#forcast value
	Prediction = StartARIMAForecasting(Actual, 1, 1, 2)
	print('Actual=%f, Predicted=%f' % (ActualValue, Prediction))
	#add it in the list
	Predictions_arima_EURGBP.append(Prediction)
	Actual.append(ActualValue)

Predictions_arima_EURGBP = pd.Series(Predictions_arima_EURGBP)
Predictions_arima_EURGBP = pd.Series((v[0] for v in Predictions_arima_EURGBP))
Predictions_arima_EURGBP.index = TestData_EURGBP.index.copy()

#Forecasting accuracy scores
def forecast_accuracy(forecast, actual):
    mae = np.mean(np.abs(forecast - actual))    # MAE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    return({'mae': mae, 'rmse':rmse})

print(forecast_accuracy(Predictions_arima_EURGBP, TestData_EURGBP))

#MAE: 0.003332027
#RMSE: 0.0045855219

#Convert from log returns to actual values
from sklearn.model_selection import train_test_split

train_actual_EURGBP, test_actual_EURGBP = train_test_split(df_EURGBP.EURGBP, test_size = 0.2, shuffle = False)


Predictions_arima_EURGBP1 = Predictions_arima_EURGBP.iloc[1:]

pred_arima_EURGBP_actual = np.exp(Predictions_arima_EURGBP1)*test_actual_EURGBP
pred_arima_EURGBP_actual = pred_arima_EURGBP_actual.dropna()

mape = np.mean(np.abs(pred_arima_EURGBP_actual - test_actual_EURGBP)/np.abs(test_actual_EURGBP))  # MAPE
print(mape)

#MAPE: 0.00026672395
    

#%%

#Plots with the result of the ARIMA, Random Walk and k-NN Regression forecast.
fig_size1 = plt.rcParams["figure.figsize"]
fig_size1[0] = 12
fig_size1[1] = 5
plt.rcParams["figure.figsize"] = fig_size1


fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey = True)
l1 = ax1.plot(test_actual_EURGBP[-20:], label='Actual values')
l2 = ax1.plot(pred_rw_EURGBP_actual[-20:], color='red', label='Forecast values')


l3 = ax2.plot(test_actual_EURGBP[-20:], label='Actual values')
l4 = ax2.plot(pred_arima_EURGBP_actual[-20:], color='red', label='Forecast values')


l5 = ax3.plot(test_actual_EURGBP[-20:], label='Actual values')
l6 = ax3.plot(pred_knn_EURGBP_actual[-20:], color='red', label='Forecast values' )


ax1.set(title='Random walk', ylabel='EURGBP', xlabel='Dates')
ax2.set(title='ARIMA(1,1,2)', ylabel='EURGBP', xlabel = 'Dates')
ax3.set(title='k-NN Regression with k = 40', ylabel = 'EURGBP', xlabel='Dates')
fig.autofmt_xdate()



fig.legend(('Actual values', 'Forecast values'), loc=(0.375,0.), borderaxespad = 2.5,
        ncol=2,)
fig.savefig(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\plots\\EURGBP_results_plot.pdf', bbox_inches='tight')


