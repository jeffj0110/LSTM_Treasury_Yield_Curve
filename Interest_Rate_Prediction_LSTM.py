# Prototype Treasury Yield prediction using Keras and Tensorflow.
# J. Jones 
# April 2021
#
# Script creates a LSTM Time Series multivariant model, runs through training data and then does a validation with a separate set of data.
# Uses GDP, CPI, Fed Discount Rate, Chinese Discount Rate, Wages and Unemployment as independent variables and predicts the US Treasury Yield Curve. 
#
# A final prediction with forecasted dependent economic variables is generated for 3/31/2021.
import keras
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import LeakyReLU, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


# load the dataset
Training_Data_Set = 'LSTM_IR_Training_Data.csv'
frame = pd.read_csv(Training_Data_Set)

# need to convert date strings to number of days since Jan 1, 1970 for Keras
# insert column at beginning of list
frame.insert(0,'DaysSince01011970',0)

start_date = datetime(1970,1,1)
rowcount = 0
for EachDate in frame['Date_MMDDYYYY']:
    date_time_str = frame.iloc[rowcount,1]
    date_time_obj = datetime.strptime(date_time_str, '%m/%d/%Y')
    dayssince1970 = date_time_obj - start_date
    frame.iloc[rowcount,0] = dayssince1970.days
    rowcount += 1

#Remove the text column from the dataframe
frame = frame.drop('Date_MMDDYYYY',1)

# Convert dataframe to numpy array

IntRate_Dataset = np.array(frame)
X = IntRate_Dataset[:,0:14]
y = IntRate_Dataset[:,14:21]

scaler_X = MinMaxScaler()
scaler_X.fit(X)
xscale = scaler_X.transform(X)

# Create Timeseries Training Data
timesteps = 1
Input_Columns = 14
train_generator = TimeseriesGenerator(xscale, y, length=timesteps, sampling_rate=1, batch_size=timesteps)

from keras import backend as K

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )



#define Keras model
model = Sequential()
model.add(LSTM(units=75, return_sequences=True, input_shape=(timesteps, Input_Columns)))
model.add(Dropout(0.1))
model.add(LSTM(units=75, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(units=75, return_sequences=True))
model.add(LeakyReLU(alpha=.5))
model.add(Dropout(0.1))
model.add(Dense(75))
model.add(Dense(7))



# compile the keras model
# optimizer = keras.optimizers.Adam(lr=0.005)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', coeff_determination])

print(time.strftime("%H:%M:%S"))

# fit the keras model on the dataset
model.fit(train_generator, steps_per_epoch=1, epochs=9000, batch_size=4)
model.summary()
print(time.strftime("%H:%M:%S"))

# evaluate the keras model
# clear out previous values
dfs = []


Val_Data_Set = 'LSTM_IR_Validation_Data.csv'
frame = pd.read_csv(Val_Data_Set) 

# need to convert date strings to number of days since Jan 1, 1970 for Keras
# insert column at beginning of list
frame.insert(0,'DaysSince01011970',0)

start_date = datetime(1970,1,1)
rowcount = 0
for EachDate in frame['Date_MMDDYYYY']:
    date_time_str = frame.iloc[rowcount,1]
    date_time_obj = datetime.strptime(date_time_str, '%m/%d/%Y')
    dayssince1970 = date_time_obj - start_date
    frame.iloc[rowcount,0] = dayssince1970.days
    rowcount += 1

#Remove the text column from the dataframe
frame = frame.drop('Date_MMDDYYYY',1)


# Convert to numpy arrays for transformation
IntRate_Dataset = np.array(frame)
X = IntRate_Dataset[:,0:14]
y = IntRate_Dataset[:,14:21]

scaler_X = MinMaxScaler()
scaler_X.fit(X)
xscale = scaler_X.transform(X)

# Prepare results array for inserting predicted results
resulting_array = np.concatenate((X, y), axis=1)

# Create Timeseries validation Data
timesteps = 1
#nput_Columns = 14
Validation_Series = TimeseriesGenerator(xscale, y, length=timesteps, sampling_rate=1, batch_size=timesteps)

# Create Timeseries validation Data
timesteps = 1
#nput_Columns = 14
for rowcnter in range(len(X)-1) :
    Validation_Series = TimeseriesGenerator(xscale[rowcnter:rowcnter+2], y[rowcnter:rowcnter+2], length=timesteps, sampling_rate=1, batch_size=timesteps)
    results = model.predict(Validation_Series)
    resulting_array[rowcnter+1:rowcnter+2,14:21] = results

df = pd.DataFrame(data=resulting_array)
df.columns=["Days","Real GDP growth","Real disposable income growth","Unemployment rate","CPI inflation rate","Fed Disc Rate", "China Disc Rate",
            "Prev 3-month Treasury rate", "Prev 6-month Treasury rate", "Prev 1-year Treasury yield", "Prev 2-year Treasury yield", "Prev 3-year Treasury yield", "Prev 5-year Treasury yield", "Prev 10-year Treasury yield",
            "3-month Treasury rate", "6-month Treasury rate","1-year Treasury yield","2-year Treasury yield","3-year Treasury yield","5-year Treasury yield","10-year Treasury yield"]
df.insert(0,'Quarter_End'," ")

start_date = datetime(1970,1,1)
rowcount = 0
for DayCounter in df['Days']:
    qend = start_date + timedelta(days=DayCounter)
    df.iloc[rowcount,0] = qend.strftime("%Y/%m/%d")
    rowcount += 1

df.sort_values("Quarter_End", ascending=True, inplace=True )
df.drop("Days",1)
df.to_csv('LSTM_IR_Validation_Results.csv', index=False)