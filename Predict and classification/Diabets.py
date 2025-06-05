import numpy as np
import pandas as pd
import sklearn
import keras
import tensorflow
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from Hourly_wages import early_stopping_monitor

#read in training data
df_2 = pd.read_csv('diabetes_data.csv')
#view data structure
print(df_2)
#random_state=1-Параметр якщо потрібно щоб дані моделі не дуже відрізнялися
train_df_2, test_df_2 = train_test_split(df_2, test_size=0.1)
print("train_df_2---------------------------------------------------------------------\n",train_df_2)
print("test_df_2---------------------------------------------------------------------\n",test_df_2)
train_X_2 = train_df_2.drop(columns=['diabetes'])
test_X_2 = test_df_2.drop(columns=['diabetes'])
#check that the target variable has been removed
print("train_X_2---------------------------------------------------------------------\n",train_X_2)
print("test_X_2----------------------------------------------------------------------\n",test_X_2)

from keras.utils import to_categorical
#one-hot encode target column
train_y_2 = to_categorical(train_df_2['diabetes'])
test_y_2 = to_categorical(test_df_2['diabetes'])
#vcheck that target column has been converted
print("train_y_2---------------------------------------------------------------------\n",train_y_2[0:5])
print("test_y_2----------------------------------------------------------------------\n",test_y_2[0:5])

#create model(мережа)
model_2 = Sequential()
#get number of columns in training data
n_cols_2 = train_X_2.shape[1]
#add layers to model
model_2.add(Dense(250, activation='relu', input_shape=(n_cols_2,)))
model_2.add(Dense(250, activation='relu'))
model_2.add(Dense(250, activation='relu'))
model_2.add(Dense(2, activation='softmax'))
#compile model using accuracy to measure model performance
#optimizer-алгоритм навчання; loss-функція втрат; metrics - точність
model_2.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])
print("model_2------------------------------------------------------------------------\n",model_2)
#validation_split-вибірка для перевірки
model_2.fit(train_X_2, train_y_2, epochs=30, validation_split=0.2,
callbacks=[early_stopping_monitor])

#test model
test_y_predictions = model_2.predict(test_X_2)
print("test_y_predictions--------------------------------\n",test_y_predictions)
#Бінарізація
test_y_predictions_bin = np.argmax(test_y_predictions, axis = 1)
print("binarization test_y_predictions_bin-------------------------\n",test_y_predictions_bin)
test_y_true = test_df_2['diabetes'].to_numpy()
# confusion_matrix
from sklearn.metrics import confusion_matrix
#Порівняння тренованої вибірки та оригінальної
confusion_matrix(test_y_true, test_y_predictions_bin, labels=[0, 1])
print("Confusion_matrix-------------------------------------------------\n",confusion_matrix(test_y_true, test_y_predictions_bin, labels=[0, 1]))

# classification report
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print("classification_report---------------------------------------------\n",classification_report(test_y_true, test_y_predictions_bin,
target_names=target_names))