import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

trainDataset = pd.read_csv('Google_Stock_price_Train.csv')
trainDataset = trainDataset.iloc[:,4:5].values
trainDataset = np.array(trainDataset.tolist())

import re

sample = []
for data in trainDataset:
    i = re.sub(',','',str(data))
    i = i[2:-2]
    sample.append([float(i)])
    
sample = np.array(sample)

from sklearn.preprocessing import MinMaxScaler

mScaler = MinMaxScaler(feature_range=(0,1))

trainDataset = mScaler.fit_transform(sample)

x = trainDataset[:-1,:]

y = trainDataset[1:,:]

x = np.reshape(x, (x.shape[0], x.shape[1],1))
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(units=32,activation='sigmoid',input_shape=(None,1)))
model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(x,y,epochs=100,batch_size=32)

realStock = pd.read_csv('Google_Stock_Price_Test.csv')
realStock = realStock.iloc[:,4:5].values

inpStock = mScaler.transform(realStock)
inpStock = np.reshape(inpStock,(len(inpStock),1,1))

model.predict(inpStock)
actualPrediction = mScaler.inverse_transform(model.predict(inpStock))

plt.plot(realStock,'r')
plt.plot(actualPrediction,'b')

