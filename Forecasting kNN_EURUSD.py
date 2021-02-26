# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:48:44 2020

@author: Clemens
"""
#Imports for k-NN Regression
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
dfknn_EURUSD = df.EURUSD
dfknn_EURUSD = dfknn_EURUSD.to_frame()
dfknn_EURUSD['EURUSD'] = np.log(1 + dfknn_EURUSD.EURUSD.pct_change())
dfknn_EURUSD['lag1'] = dfknn_EURUSD.EURUSD.shift(1)
dfknn_EURUSD['lag2'] = dfknn_EURUSD.EURUSD.shift(2)
dfknn_EURUSD['lag3'] = dfknn_EURUSD.EURUSD.shift(3)
dfknn_EURUSD['lag4'] = dfknn_EURUSD.EURUSD.shift(4)
dfknn_EURUSD['lag5'] = dfknn_EURUSD.EURUSD.shift(5)
dfknn_EURUSD = dfknn_EURUSD.iloc[6:]



#%%

#Split Data into train and test split



train_knn_EURUSD, test_knn_EURUSD = train_test_split(dfknn_EURUSD, test_size = 0.2, shuffle = False)

x_train_knn_EURUSD = train_knn_EURUSD.drop('EURUSD', axis=1)
y_train_knn_EURUSD = train_knn_EURUSD['EURUSD']

x_test_knn_EURUSD = test_knn_EURUSD.drop('EURUSD', axis = 1)
y_test_knn_EURUSD = test_knn_EURUSD['EURUSD']



#%%

#Search for optimal k value

rmse_val_EURUSD = [] #to store rmse values for different k

for K in range(60):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train_knn_EURUSD, y_train_knn_EURUSD)  #fit the model
    pred_knn_EURUSD = model.predict(x_test_knn_EURUSD) #make prediction on test set
    error = sqrt(sqrt(mean_squared_error(y_test_knn_EURUSD,pred_knn_EURUSD))) #calculate rmse
    rmse_val_EURUSD.append(error) #store mse values
    print('RMSE value for k= ' , K , 'is:', error)


curve = pd.DataFrame(rmse_val_EURUSD) #elbow curve 
curve.plot()

fig2 = plt.figure()
curve.plot()
plt.ylabel('Root mean squared error')
plt.xlabel('Number of neighbours k')
plt.legend('_nolegend_')

plt.savefig(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\plots\\EURUSD_knn_k.pdf', bbox_inches='tight')


#%%

#kNN Regression Forecast

model = neighbors.KNeighborsRegressor(n_neighbors = 40, weights = 'distance')

model.fit(x_train_knn_EURUSD, y_train_knn_EURUSD)  #fit the model
pred_knn_EURUSD = model.predict(x_test_knn_EURUSD) #make prediction on test set


#%%

#Forecasting error of log return series 

pred_knn_EURUSD = pd.Series(pred_knn_EURUSD)

pred_knn_EURUSD.index = y_test_knn_EURUSD.index.copy()


def forecast_accuracy(forecast, actual):
    mae = np.mean(np.abs(forecast - actual))    # MAE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    return({'mae': mae, 'rmse':rmse})

print(forecast_accuracy(pred_knn_EURUSD, y_test_knn_EURUSD))

#MAE: 0.002944062432299604
#RMSE: 0.003752472596916725

#%%
#Convert from log returns to actual values and print forecasting error MAPE

train_actual_EURUSD, test_actual_EURUSD = train_test_split(df_EURUSD.EURUSD, test_size = 0.2, shuffle = False)

pred_knn_EURUSD_actual = np.exp(pred_knn_EURUSD)*test_actual_EURUSD
pred_knn_EURUSD_actual =  pred_knn_EURUSD_actual.dropna()


mape = np.mean(np.abs(pred_knn_EURUSD_actual - test_actual_EURUSD)/np.abs(test_actual_EURUSD))  # MAPE
print(mape)

#MAPE: 0.000556816


