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
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
%matplotlib inline
#%%


#Import Data

def GetData(fileName):
    return pd.read_csv(fileName, header=0, parse_dates=[0], index_col=0, encoding='utf-8')

df = GetData(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\data.csv')

#Choose currency pair
df_EURGBP = pd.DataFrame(df, columns=['EURGBP'])



#%%
# Prepare Data for kNN Regression by taking log return and adding Lags 1-5
dfknn_EURGBP = df.EURGBP
dfknn_EURGBP = dfknn_EURGBP.to_frame()
dfknn_EURGBP['EURGBP'] = np.log(1 + dfknn_EURGBP.EURGBP.pct_change())
dfknn_EURGBP['lag1'] = dfknn_EURGBP.EURGBP.shift(1)
dfknn_EURGBP['lag2'] = dfknn_EURGBP.EURGBP.shift(2)
dfknn_EURGBP['lag3'] = dfknn_EURGBP.EURGBP.shift(3)
dfknn_EURGBP['lag4'] = dfknn_EURGBP.EURGBP.shift(4)
dfknn_EURGBP['lag5'] = dfknn_EURGBP.EURGBP.shift(5)
dfknn_EURGBP = dfknn_EURGBP.iloc[6:]





#%%

#Split Data into Train and Test Split

train_knn_EURGBP, test_knn_EURGBP = train_test_split(dfknn_EURGBP, test_size = 0.2, shuffle = False)

x_train_knn_EURGBP = train_knn_EURGBP.drop('EURGBP', axis=1)
y_train_knn_EURGBP = train_knn_EURGBP['EURGBP']

x_test_knn_EURGBP = test_knn_EURGBP.drop('EURGBP', axis = 1)
y_test_knn_EURGBP = test_knn_EURGBP['EURGBP']




#%%

#Search for optimal k value

rmse_val_EURGBP = [] #to store rmse values for different k

for K in range(60):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train_knn_EURGBP, y_train_knn_EURGBP)  #fit the model
    pred_knn_EURGBP = model.predict(x_test_knn_EURGBP) #make prediction on test set
    error = sqrt(sqrt(mean_squared_error(y_test_knn_EURGBP,pred_knn_EURGBP))) #calculate rmse
    rmse_val_EURGBP.append(error) #store mse values
    print('RMSE value for k= ' , K , 'is:', error)


curve = pd.DataFrame(rmse_val_EURGBP) #elbow curve 

fig2 = plt.figure()
curve.plot()
plt.ylabel('Root mean squared error')
plt.xlabel('Number of neighbours k')
plt.legend('EURGBP')

plt.savefig(r'C:\\Users\\Clemens\\Documents\Uni Freiburg\\Bachelor Arbeit\\Python\\fx_pred\\plots\\EURGBP_knn_k.pdf', bbox_inches='tight')


#%%

#kNN Regression Forecast

model = neighbors.KNeighborsRegressor(n_neighbors = 40, weights = 'distance')

model.fit(x_train_knn_EURGBP, y_train_knn_EURGBP)  #fit the model
pred_knn_EURGBP = model.predict(x_test_knn_EURGBP) #make prediction on test set


#%%

#Forecasting error of log return series

pred_knn_EURGBP = pd.Series(pred_knn_EURGBP)

pred_knn_EURGBP.index = y_test_knn_EURGBP.index.copy()


plt.plot(y_test_knn_EURGBP[:50])
plt.plot(pred_knn_EURGBP[:50], color='red')
plt.show()

def forecast_accuracy(forecast, actual):
    mae = np.mean(np.abs(forecast - actual))    # MAE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    return({'mae': mae, 'rmse':rmse})

print(forecast_accuracy(pred_knn_EURGBP, y_test_knn_EURGBP))

#MAE: 0.0033485282
#RMSE: 0.004598941

#%%

#Convert from log returns to actual values and print forecasting error MAPE

train_actual_EURGBP, test_actual_EURGBP = train_test_split(df_EURGBP.EURGBP, test_size = 0.2, shuffle = False)

pred_knn_EURGBP_actual = np.exp(pred_knn_EURGBP)*test_actual_EURGBP
pred_knn_EURGBP_actual =  pred_knn_EURGBP_actual.dropna()

mape = np.mean(np.abs(pred_knn_EURGBP_actual - test_actual_EURGBP)/np.abs(test_actual_EURGBP))  # MAPE
print(mape)

#MAPE: 0.000527595