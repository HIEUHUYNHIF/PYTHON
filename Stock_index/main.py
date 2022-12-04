import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
from sklearn.preprocessing import MinMaxScaler
import seaborn as sb

data = pd.read_csv("./dataset/VNI/VNI_010721_310722.csv")
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

data = data[::-1].reset_index(drop=True)

print(data)

data.plot('Date','Price',color="red")
plt.show()
features = ['Open', 'High', 'Low', 'Price', 'Vol.']
plt.subplots(figsize=(10,5))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(data[col])
plt.show()

# 1. Filter out the closing market price data
close_data = data.filter(['Price'])
 
# 2. Convert the data into array for easy evaluation
dataset = close_data.values
 
# 3. Scale/Normalize the data to make all values between 0 and 112
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
 
# 4. Creating training data size : 80% of the data
trainDataLen = math.ceil(len(dataset) *0.5)
trainData = scaled_data[0:trainDataLen  , : ]
 
# 5. Separating the data into x and y data
trainDataX=[]
trainDataY =[]
for i in range(3,len(trainData)):
    trainDataX=list(trainDataX)
    trainDataY=list(trainDataY)
    trainDataX.append(trainData[i-3:i,0])
    trainDataY.append(trainData[i,0])
 
    # 6. Converting the training x and y values to numpy arrays
    trainDataX1, trainDataY1 = np.array(trainDataX), np.array(trainDataY)
 
    # 7. Reshaping training s and y data to make the calculations easier
    trainDataX2 = np.reshape(trainDataX1, (trainDataX1.shape[0],trainDataX1.shape[1],1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(trainDataX2.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(trainDataX2, trainDataY1, batch_size=1, epochs=1)

# 1. Creating a dataset for testing
testData = scaled_data[trainDataLen - 3: , : ]
testX = []
testY =  dataset[trainDataLen : , : ]
for i in range(3,len(testData)):
    testX.append(testData[i-3:i,0])
 
# 2.  Convert the values into arrays for easier computation
testX = np.array(testX)
testX = np.reshape(testX, (testX.shape[0],testX.shape[1],1))
 
# 3. Making predictions on the testing data
predictions = model.predict(testX)
predictions = scaler.inverse_transform(predictions)

rmse=np.sqrt(np.mean(((predictions- testY)**2)))
print(rmse)

train = data[:trainDataLen]
valid = data[trainDataLen:]
 
valid['Predictions'] = predictions
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Price')
 
plt.plot(train['Price'])
plt.plot(valid[['Price', 'Predictions']])
 
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
 
plt.show()
