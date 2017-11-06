from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array
import pandas
import math

def my_mean_squared_error(inv_y, inv_yhat):
    mse = 0
    for i in range(len(inv_y)):
        mse += float (math.pow((inv_y[i] - inv_yhat[i]), 2) / math.pow(inv_y[i], 2))
    mse = mse / len(inv_y)
    return mse

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
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

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test

# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]

# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i - 1])
    return inverted

# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        rmse_normalized = sqrt(my_mean_squared_error(actual, predicted))
        print('t+%d RMSE (unnormalized): %f' % ((i + 1), rmse))
        print('t+%d RMSE (normalized): %f' % ((i + 1), rmse_normalized))


# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    pyplot.plot(series)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series[off_s]] + forecasts[i]
        pyplot.plot(xaxis, yaxis, color='red')
    # show the plot
    pyplot.show()

# load the dataset
dataframe = pandas.read_csv('EOD-KO.csv', usecols=[11], engine='python', skipfooter=3)
dataframe = dataframe.reindex(index=dataframe.index[::-1])
series = dataframe.values
series = series.astype('float32')

#print(len(series))

# configure
n_lag = 1
n_seq = 3
n_test = 4209
n_epochs = 3
n_batch = 1
n_neurons = 10

# prepare data
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
train_size = len(train)
test_size = len(test)
series_size = train_size + test_size
'''print(train_size)
print(len(series))
print(n_test)
print(len(test))'''
'''print(test)
print(train)'''

# fit model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)

# make forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)

# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+2)

'''print(actual)
print(forecasts)'''
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)

# walk-forward validations on the test data
walk_for_1 = list()
walk_for_2 = list()
walk_for_3 = list()
history = [x for x in series[0:train_size]]
for i in range(len(test)):
    # make prediction
    walk_for_1.append(history[-1])
    walk_for_2.append(history[-2])
    walk_for_3.append(history[-3])
    # observation
    history.append(series[train_size + i])

rmse_persistence_unnormalized_1 = sqrt(mean_squared_error(series[train_size:series_size,:], walk_for_1))
print('Persistence Model (forecast 1) RMSE against test set (unnormalized): %f' % rmse_persistence_unnormalized_1)
rmse_persistence_normalized_1 = sqrt(my_mean_squared_error(series[train_size:series_size,:], walk_for_1))
print('Persistence Model (forecast 1) RMSE against test set (normalized): %f' % rmse_persistence_normalized_1)

rmse_persistence_unnormalized_2 = sqrt(mean_squared_error(series[train_size:series_size,:], walk_for_2))
print('Persistence Model (forecast 2) RMSE against test set (unnormalized): %f' % rmse_persistence_unnormalized_2)
rmse_persistence_normalized_2 = sqrt(my_mean_squared_error(series[train_size:series_size,:], walk_for_2))
print('Persistence Model (forecast 2) RMSE against test set (normalized): %f' % rmse_persistence_normalized_2)

rmse_persistence_unnormalized_3 = sqrt(mean_squared_error(series[train_size:series_size,:], walk_for_3))
print('Persistence Model (forecast 3) RMSE against test set (unnormalized): %f' % rmse_persistence_unnormalized_3)
rmse_persistence_normalized_3 = sqrt(my_mean_squared_error(series[train_size:series_size,:], walk_for_3))
print('Persistence Model (forecast 3) RMSE against test set (normalized): %f' % rmse_persistence_normalized_3)


'''# make a persistence forecast
def persistence(last_ob, n_seq):
    return [last_ob for i in range(n_seq)]


# evaluate the persistence model
def make_forecasts_persistence(train, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = persistence(X[-1], n_seq)
        # store the forecast
        forecasts.append(forecast)
    return forecasts


# evaluate the RMSE for each forecast time step
def evaluate_forecasts_persistence(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = test[:, (n_lag + i)]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('Persistence t+%d RMSE: %f' % ((i + 1), rmse))

# make forecasts for persistence model
forecasts_persistence = make_forecasts_persistence(train, test, n_lag, n_seq)
# evaluate forecasts for persistence model
print("..........PERSISTENCE MODEL RESULTS............")
evaluate_forecasts_persistence(test, forecasts, n_lag, n_seq)'''

# plot forecasts
# plot forecasts for lstm model
plot_forecasts(series, forecasts, n_test+2)
# plot forecasts for persistence model
#plot_forecasts(series, forecasts_persistence, n_test+2)

