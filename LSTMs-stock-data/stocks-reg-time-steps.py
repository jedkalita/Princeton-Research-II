# LSTM for adj close price of stock problem with time step regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix

def my_mean_squared_error(inv_y, inv_yhat):
    mse = 0
    for i in range(len(inv_y)):
        mse += float (math.pow((inv_y[i] - inv_yhat[i]), 2) / math.pow(inv_y[i], 2))
    mse = mse / len(inv_y)
    return mse

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('EOD-KO.csv', usecols=[11], engine='python', skipfooter=3)
dataframe = dataframe.reindex(index=dataframe.index[::-1])
dataset = dataframe.values
dataset = dataset.astype('float32')
#print(dataset)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
#print(trainX.shape) #(9395, 3, 1)
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
#print(testX.shape) #(4626, 3, 1)
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(400, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=20, validation_data=(testX, testY), verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
print(trainY)
print(testY)
print(trainPredict)
print(testPredict)
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
trainScore_normalized = math.sqrt(my_mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score (unnormalized): %f RMSE' % (trainScore))
print('Train Score (normalized): %f RMSE' % (trainScore_normalized))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
testScore_normalized = math.sqrt(my_mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score (unnormalized): %f RMSE' % (testScore))
print('Test Score (normalized): %f RMSE' % (testScore_normalized))


history = [x for x in trainY[0]]
predictions = list()
for i in range(len(testY[0])):
    # make prediction
    predictions.append(history[-1])
    # observation
    history.append(testY[0][i])

print(predictions)

persistence_rmse_normalized = math.sqrt(my_mean_squared_error(testY[0], predictions))
print('Testing against persistence model (normalized): %f' % persistence_rmse_normalized)
predictions2 = list()
for i in range(len(trainY[0]) - 1):
    # make prediction
    predictions2.append(trainY[0][i + 1])
    # observation
    #history.append(trainY[0][i])
predictions2.append(testY[0][1])

persistence_rmse_normalized2 = math.sqrt(my_mean_squared_error(trainY[0], predictions2))
print('Testing against persistence model for training (normalized): %f' % persistence_rmse_normalized2)
persistence_rmse2 = math.sqrt(mean_squared_error(trainY[0], predictions2))
print('Testing against persistence model for training (unnormalized): %f' % persistence_rmse2)


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()