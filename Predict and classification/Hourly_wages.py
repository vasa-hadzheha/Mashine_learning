import pandas as pd
import sklearn
import keras
import tensorflow

path = "hourly_wages_data.csv"
# завантажуємо наш датасет
train_df = pd.read_csv(path)
# виводимо вигляд датасету в консоль
print(train_df.shape)
print(train_df)

#завантажуємо функцію спліт для розбиття вибірки на навчальну та тестову
from sklearn.model_selection import train_test_split
#розбили train_df на дві вибірки випадковим образом береться 90% датасету для змінної-train_df та 10% для змінної-test_df
train_df, test_df = train_test_split(train_df, test_size=0.1)
print("Train_df_______________________________", train_df.shape)
print(train_df)
print("Test_df_______________________________", test_df.shape)
print(test_df)

#Робимо змінні train_X test_X це будуть наші входи ми видаляємо з наших вибірок для навчальної та тестової цільову ознаку (стовпець wage_per_hour для того щоб потім спрогнозувати його значення)
train_X = train_df.drop(columns=['wage_per_hour'])
print("Train_X_______________________________", train_X.shape)
print(train_X)
test_X = test_df.drop(columns=['wage_per_hour'])
print("Test_X_______________________________", test_X.shape)
print(test_X)

# check that the target variable has been removed
train_X.head()
# Вставляємо Цей стовпець що ми видалили із входів у нашу цільову змінну Y для тренувальної та тестової вибірки
train_y = train_df['wage_per_hour']
print("Train_Y_______________________________", train_y.shape)
print(train_y)
test_y = test_df['wage_per_hour']
print("Test_Y_______________________________", test_y.shape)
print(test_y)

#Тип моделі, яку ми будемо використовувати - Sequential. із бібліотеки keras
from keras.models import Sequential
# Dense - це шар, Dense - стандартний тип шару, який працює в більшості випадків.
from keras.layers import Dense

# create model
model = Sequential()
# get number of columns in training data
n_cols = train_X.shape[1]
print(n_cols)

from keras.callbacks import EarlyStopping
#set early stopping monitor so the model stops training when
# it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)
# #train model
# model.fit(train_X, train_y, validation_split=0.2, epochs=30,callbacks=[early_stopping_monitor])
# #example on how to use our newly trained model on how to make
# #predictions on unseen data (we will pretend our new data is #saved in a
# #dataframe called 'test_X').
# test_y_predictions = model.predict(test_X)
# print(test_X)

#training a new model on the same data to show the effect of #increasing model capacity
#create model
model_mc = Sequential()
#add model layers
model_mc.add(Dense(200, activation='relu', input_shape=(n_cols,)))
model_mc.add(Dense(200, activation='relu'))
model_mc.add(Dense(200, activation='relu'))
model_mc.add(Dense(1))
#compile model using mse as a measure of model performance
model_mc.compile(optimizer='adam', loss='mean_squared_error')
#train model
model_mc.fit(train_X, train_y, validation_split=0.2, epochs=30,callbacks=[early_stopping_monitor])