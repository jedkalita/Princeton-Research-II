from pandas import read_csv
from matplotlib import pyplot
import math
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def my_mean_squared_error(inv_y, inv_yhat):
    mse = 0
    for i in range(len(inv_y)):
        mse += float (math.pow((inv_y[i] - inv_yhat[i]), 2) / math.pow(inv_y[i], 2))
    mse = mse / len(inv_y)
    return mse

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    #print("No of variables = %d" %n_vars)
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# load dataset
dataset = read_csv('EOD-KO.csv', header=0, index_col=0)
dataset = dataset.reindex(index=dataset.index[::-1])
#print(dataset)
#print(type(dataset))
values = dataset.values
#print(values)
'''encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])'''
values = values[:, 7:12] #until values, it seems to be on the right track
#print(values)
#values.reverse()
#print(values.shape) #(14032, 5)
# ensure all data is float
values = values.astype('float32')
#print(values)
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#print(scaled.shape) #(14032, 12) - 12 features in the beginning
#print(scaled)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
#reframed = series_to_supervised(values, 1, 1)
#print(reframed)
# drop columns we don't want to predict
#reframed.drop(reframed.columns[[0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23]], axis=1, inplace=True)
reframed.drop(reframed.columns[[5, 6, 7, 9]], axis=1, inplace=True) #now it is scaled between (0, 1)
print(reframed.head())
print(reframed.tail())
#var8 - adj open, var9 - adj high, var10 - adj low, var11 - adj close, var12 - volume traded
#we are predicting adj close here (hence column 8 is saved)

# split into train and test sets
values = reframed.values

#print(values)
n_train = math.ceil(len(values) * 0.7)
#n_train = n_train - 22
print("No of training examples = %d" %n_train)
n_test = len(values) - n_train
print("No of test examples = %d" %n_test)
train = values[:n_train, :]
print(train)
test = values[n_train:, :]
print(test)
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
#print("Printing test_y")
history = [x for x in train_y]
predictions = list()
for i in range(len(test_y)):
    # make prediction
    predictions.append(history[-1])
    # observation
    history.append(test_y[i])

persistence_rmse_normalized = sqrt(my_mean_squared_error(test_y, predictions))

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
'''model.add(LSTM(2, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(LSTM(2, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(LSTM(2, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))'''
model.add(LSTM(12, input_shape=(train_X.shape[1], train_X.shape[2])))
'''batch = 1
model.add(LSTM(2, batch_input_shape=(batch, train_X.shape[1], train_X.shape[2]), stateful=True))'''
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
#batch = math.floor(n_train / 50)
repeats = 10
for i in range(repeats):
    print("Repeat #: %d" %i)
    history = model.fit(train_X, train_y, epochs=100, batch_size=140, validation_data=(test_X, test_y), verbose=2, shuffle=False)

'''nb_epochs = 5
for i in range(nb_epochs):
    history = model.fit(train_X, train_y, epochs=1, batch_size=batch, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    model.reset_states()'''


# make a prediction
yhat = model.predict(test_X) #it will return adj close in scaled format - the predicted version
print(yhat)
#print(yhat.shape) #(4209, 1)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#print(test_X.shape) #(4209, 5)
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
#print(inv_yhat.shape) #(4209, 5)
#print(inv_yhat)
inv_yhat = scaler.inverse_transform(inv_yhat) #initially when the scaler is called on values, the values
#has to be of the proper dimensions in terms of the number of features
inv_yhat = inv_yhat[:,0]
print(inv_yhat)
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
print(inv_y)
print(len(inv_y))
print(inv_yhat)
print(len(inv_yhat))
rmse_unnormalized = sqrt(mean_squared_error(inv_y, inv_yhat))



rmse = sqrt(my_mean_squared_error(inv_y, inv_yhat))
print('Test RMSE (normalized): %f' % rmse)
print('Test RMSE (unnormalized): %f' % rmse_unnormalized)


print('Testing against persistence model (normalized): %f' % persistence_rmse_normalized)

predictions2 = list()
for i in range(len(train_y) - 1):
    # make prediction
    predictions2.append(train_y[i + 1])
    # observation
    #history.append(trainY[0][i])
predictions2.append(test_y[1])

'''persistence_rmse_normalized2 = math.sqrt(my_mean_squared_error(train_y, predictions2))
print('Testing against persistence model for training (normalized): %f' % persistence_rmse_normalized2)'''
persistence_rmse2 = math.sqrt(mean_squared_error(train_y, predictions2))
print('Testing against persistence model for training (unnormalized): %f' % persistence_rmse2)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()