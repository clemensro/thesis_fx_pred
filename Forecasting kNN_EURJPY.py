# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:48:44 2020

@author: Clemens
"""

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
%matplotlib inline


#%%

#Import Data

def GetData(fileName):
    return pd.read_csv(fileName, header=0, parse_dates=[0], index_col=0, encoding='utf-8')

df = GetData(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\data.csv')

#Choose currency pair
df_EURUSD = pd.DataFrame(df, columns=['EURUSD'])

#%%


# Prepare Data for kNN Regression by taking log return and adding Lags 1-5
dfknn_EURJPY = df.EURJPY
dfknn_EURJPY = dfknn_EURJPY.to_frame()
dfknn_EURJPY['EURJPY'] = np.log(1 + dfknn_EURJPY.EURJPY.pct_change())
dfknn_EURJPY['lag1'] = dfknn_EURJPY.EURJPY.shift(1)
dfknn_EURJPY['lag2'] = dfknn_EURJPY.EURJPY.shift(2)
dfknn_EURJPY['lag3'] = dfknn_EURJPY.EURJPY.shift(3)
dfknn_EURJPY['lag4'] = dfknn_EURJPY.EURJPY.shift(4)
dfknn_EURJPY['lag5'] = dfknn_EURJPY.EURJPY.shift(5)
dfknn_EURJPY = dfknn_EURJPY.iloc[6:]




#%%
#Split Data into Train and Test Split




train_knn_EURJPY, test_knn_EURJPY = train_test_split(dfknn_EURJPY, test_size = 0.2, shuffle = False)

x_train_knn_EURJPY = train_knn_EURJPY.drop('EURJPY', axis=1)
y_train_knn_EURJPY = train_knn_EURJPY['EURJPY']

x_test_knn_EURJPY = test_knn_EURJPY.drop('EURJPY', axis = 1)
y_test_knn_EURJPY = test_knn_EURJPY['EURJPY']


#%%

#Search for optimal k value


rmse_val_EURJPY = [] #to store rmse values for different k

for K in range(60):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train_knn_EURJPY, y_train_knn_EURJPY)  #fit the model
    pred_knn_EURJPY = model.predict(x_test_knn_EURJPY) #make prediction on test set
    error = sqrt(sqrt(mean_squared_error(y_test_knn_EURJPY,pred_knn_EURJPY))) #calculate rmse
    rmse_val_EURJPY.append(error) #store mse values
    print('RMSE value for k= ' , K , 'is:', error)


curve = pd.DataFrame(rmse_val_EURJPY) #elbow curve 
curve.plot()

fig2 = plt.figure()
curve.plot()
plt.ylabel('Root mean squared error')
plt.xlabel('Number of neighbours k')
plt.legend('_nolegend_')

plt.savefig(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\plots\\EURJPY_knn_k.pdf', bbox_inches='tight')


#%%

#kNN Regression Forecast

model = neighbors.KNeighborsRegressor(n_neighbors = 40, weights = 'distance')

model.fit(x_train_knn_EURJPY, y_train_knn_EURJPY)  #fit the model
pred_knn_EURJPY = model.predict(x_test_knn_EURJPY) #make prediction on test set


#%%

#Forecasting error of log return series

pred_knn_EURJPY = pd.Series(pred_knn_EURJPY)

pred_knn_EURJPY.index = y_test_knn_EURJPY.index.copy()

#Calculate forecasting accuracy MAE and RMSE

def forecast_accuracy(forecast, actual):
    mae = np.mean(np.abs(forecast - actual))    # MAE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    return({'mae': mae, 'rmse':rmse})

print(forecast_accuracy(pred_knn_EURJPY, y_test_knn_EURJPY))

#MAE: 0.0035056574222035943
#RMSE: 0.004590752321090918

#%%

#Convert from log returns to actual values and print forecasting error MAPE

train_actual_EURJPY, test_actual_EURJPY = train_test_split(df_EURJPY.EURJPY, test_size = 0.2, shuffle = False)


pred_knn_EURJPY_actual = np.exp(pred_knn_EURJPY)*test_actual_EURJPY
pred_knn_EURJPY_actual =  pred_knn_EURJPY_actual.dropna()


mape = np.mean(np.abs(pred_knn_EURJPY_actual - test_actual_EURJPY)/np.abs(test_actual_EURJPY))  # MAPE
print(mape)

#MAPE: 0.000640997941236101