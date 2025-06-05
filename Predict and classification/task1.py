import pandas as pd
import sklearn
import keras
import tensorflow as tf
import numpy as np

path = "hourly_wages_data.csv"
# read in data using pandas
full_dataset = pd.read_csv(path)
# check data has been read in properly
print(full_dataset.head())

from sklearn.model_selection import train_test_split

train_ds, test_ds = train_test_split(full_dataset, test_size=0.1)

#Видаляємо цільову змінну
train_X = train_ds.drop(columns=['wage_per_hour'])
test_X = test_ds.drop(columns=['wage_per_hour'])

# check that the target variable has been removed
train_X.head()

train_y = train_ds['wage_per_hour']
test_y = test_ds['wage_per_hour']

from keras.models import Sequential
from keras.layers import Dense

# create model
# get number of columns in training data
n_cols = train_X.shape[1]
print(n_cols)

#Дає змогу не перенавчитися
from keras.callbacks import EarlyStopping
#set early stopping monitor so the model stops training when
# it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)
# #train model

#training a new model on the same data to show the effect of #increasing model capacity
#create model
model_mc = Sequential()
#add model layers
model_mc.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model_mc.add(Dense(90, activation='relu'))
model_mc.add(Dense(50, activation='relu'))
model_mc.add(Dense(1))
#compile model using mse as a measure of model performance
model_mc.compile(optimizer='adam', loss='mean_squared_error')
#train model
model_mc.fit(train_X, train_y, validation_split=0.2, epochs=30)

test_y_predictions = model_mc.predict(test_X)

print(test_y_predictions)

print (np.array(test_y),'-----------------------------------------------------------------------\n', test_y_predictions)
t = np.subtract(np.array(test_y), test_y_predictions)
print(t)

mape = 100. * np.mean(np.abs(np.array(test_y) - test_y_predictions) / np.array(test_y))
#По результату точність доволі мала
print(mape)