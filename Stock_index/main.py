import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
from sklearn.preprocessing import MinMaxScaler
import seaborn as sb

data = pd.read_csv("./dataset/VNI_010721_311221.csv")
print(data.info())
for i in range(0, len(data)):
    data['Price'][i]    = float(data['Price'][i].replace(',',''))
    data['Open'][i]     = float(data['Open'][i].replace(',',''))
    data['High'][i]     = float(data['High'][i].replace(',',''))
    data['Low'][i]      = float(data['Low'][i].replace(',',''))
    if(data['Vol.'][i][-1]      == "K"):
        data['Vol.'][i]      = float(data['Vol.'][i].replace('K','').replace(',',''))*1000
    elif(data['Vol.'][i][-1]    == "M"):
        data['Vol.'][i]      = float(data['Vol.'][i].replace('M','').replace(',',''))*1000000
    else:
        data['Vol.'][i]      = float(data['Vol.'][i].replace(',',''))

def reverse_array(myArray):
    for i in range(0, int(len(myArray)/2)):
        temp                            =   myArray[i]         
        myArray[i]                 =   myArray[len(myArray) - 1 - i]
        myArray[len(myArray) - 1 - i] =   temp
    return myArray

data['Date']        =   reverse_array(data['Date'])
data['High']        =   reverse_array(data['High'])
data['Low']         =   reverse_array(data['Low'])
data['Open']        =   reverse_array(data['Open'])
data['Price']       =   reverse_array(data['Price'])
data['Vol.']        =   reverse_array(data['Vol.'])
data['Change %']    =   reverse_array(data['Change %'])

print(data)

data.plot('Date','Price',color="red")
plt.show()
features = ['Open', 'High', 'Low', 'Price', 'Vol.']
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(data[col])
plt.show()

plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.boxplot(data[col])
plt.show()
# 1. Filter out the closing market price data
close_data = data.filter(['Price'])
 
# 2. Convert the data into array for easy evaluation
dataset = close_data.values
 
# 3. Scale/Normalize the data to make all values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
 
# 4. Creating training data size : 70% of the data
training_data_len = math.ceil(len(dataset) *0.7)
train_data = scaled_data[0:training_data_len  , : ]
 
# 5. Separating the data into x and y data
x_train_data=[]
y_train_data =[]
for i in range(60,len(train_data)):
    x_train_data=list(x_train_data)
    y_train_data=list(y_train_data)
    x_train_data.append(train_data[i-60:i,0])
    y_train_data.append(train_data[i,0])
 
    # 6. Converting the training x and y values to numpy arrays
    x_train_data1, y_train_data1 = np.array(x_train_data), np.array(y_train_data)
 
    # 7. Reshaping training s and y data to make the calculations easier
    x_train_data2 = np.reshape(x_train_data1, (x_train_data1.shape[0],x_train_data1.shape[1],1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train_data2.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train_data2, y_train_data1, batch_size=1, epochs=1)

# 1. Creating a dataset for testing
test_data = scaled_data[training_data_len - 60: , : ]
x_test = []
y_test =  dataset[training_data_len : , : ]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
 
# 2.  Convert the values into arrays for easier computation
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
 
# 3. Making predictions on the testing data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

train = data[:training_data_len]
valid = data[training_data_len:]
 
valid['Predictions'] = predictions
 
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Price')
 
plt.plot(train['Price'])
plt.plot(valid[['Price', 'Predictions']])
 
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
 
plt.show()
