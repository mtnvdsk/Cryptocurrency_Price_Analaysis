from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
#import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error

def predictionstart(dataset):
    df = pd.read_csv(dataset, delimiter=',',
                     usecols=['Date', 'Open', 'High', 'Low', 'Close'])
    print('Loaded data from the Kaggle repository')

    # Sort DataFrame by date
    df = df.sort_values('Date')

    # Double check the result
    df.head()
    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), (df['Low'] + df['High']) / 2.0)
    plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.show()
    # First calculate the mid prices from the highest and lowest
    high_prices = df.loc[:, 'High'].as_matrix()
    low_prices = df.loc[:, 'Low'].as_matrix()
    mid_prices = (high_prices + low_prices) / 2.0
    train_data = mid_prices[:11000]
    test_data = mid_prices[11000:]
    # Scale the data to be between 0 and 1
    # When scaling remember! You normalize both test and train data with respect to training data
    # Because you are not supposed to have access to test data
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1, 1)
    test_data = test_data.reshape(-1, 1)
    # Train the Scaler with training data and smooth data
    smoothing_window_size = 2500
    for di in range(0, 10000, smoothing_window_size):
        scaler.fit(train_data[di:di + smoothing_window_size, :])
        train_data[di:di + smoothing_window_size, :] = scaler.transform(train_data[di:di + smoothing_window_size, :])

    # You normalize the last bit of remaining data
    scaler.fit(train_data[di + smoothing_window_size:, :])
    train_data[di + smoothing_window_size:, :] = scaler.transform(train_data[di + smoothing_window_size:, :])
    # Reshape both train and test data
    train_data = train_data.reshape(-1)

    # Normalize test data
    test_data = scaler.transform(test_data).reshape(-1)
    # Now perform exponential moving average smoothing
    # So the data will have a smoother curve than the original ragged data
    EMA = 0.0
    gamma = 0.1
    for ti in range(11000):
        EMA = gamma * train_data[ti] + (1 - gamma) * EMA
        train_data[ti] = EMA

    # Used for visualization and test purposes
    all_mid_data = np.concatenate([train_data, test_data], axis=0)
    window_size = 100
    N = train_data.size
    std_avg_predictions = []
    std_avg_x = []
    mse_errors = []

    for pred_idx in range(window_size, N):

        if pred_idx >= N:
            date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
        else:
            date = df.loc[pred_idx, 'Date']

        std_avg_predictions.append(np.mean(train_data[pred_idx - window_size:pred_idx]))
        mse_errors.append((std_avg_predictions[-1] - train_data[pred_idx]) ** 2)
        std_avg_x.append(date)

    print('MSE error for standard averaging: %.5f' % (0.5 * np.mean(mse_errors)))

    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True')
    plt.plot(range(window_size, N), std_avg_predictions, color='orange', label='Prediction')
    # plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.show()

    window_size = 100
    N = train_data.size

    run_avg_predictions = []
    run_avg_x = []

    mse_errors = []

    running_mean = 0.0
    run_avg_predictions.append(running_mean)

    decay = 0.5

    for pred_idx in range(1, N):
        running_mean = running_mean * decay + (1.0 - decay) * train_data[pred_idx - 1]
        run_avg_predictions.append(running_mean)
        mse_errors.append((run_avg_predictions[-1] - train_data[pred_idx]) ** 2)
        run_avg_x.append(date)

    print('MSE error for EMA averaging: %.5f' % (0.5 * np.mean(mse_errors)))
    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True')
    plt.plot(range(0, N), run_avg_predictions, color='orange', label='Prediction')
    # plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.show()

    class DataGeneratorSeq(object):

        def __init__(self, prices, batch_size, num_unroll):
            self._prices = prices
            self._prices_length = len(self._prices) - num_unroll
            self._batch_size = batch_size
            self._num_unroll = num_unroll
            self._segments = self._prices_length // self._batch_size
            self._cursor = [offset * self._segments for offset in range(self._batch_size)]

        def next_batch(self):

            batch_data = np.zeros((self._batch_size), dtype=np.float32)
            batch_labels = np.zeros((self._batch_size), dtype=np.float32)

            for b in range(self._batch_size):
                if self._cursor[b] + 1 >= self._prices_length:
                    # self._cursor[b] = b * self._segments
                    self._cursor[b] = np.random.randint(0, (b + 1) * self._segments)

                batch_data[b] = self._prices[self._cursor[b]]
                batch_labels[b] = self._prices[self._cursor[b] + np.random.randint(0, 5)]
                self._cursor[b] = (self._cursor[b] + 1) % self._prices_length

            return batch_data, batch_labels

        def unroll_batches(self):

            unroll_data, unroll_labels = [], []
            init_data, init_label = None, None
            for ui in range(self._num_unroll):
                data, labels = self.next_batch()

                unroll_data.append(data)
                unroll_labels.append(labels)

            return unroll_data, unroll_labels

        def reset_indices(self):
            for b in range(self._batch_size):
                self._cursor[b] = np.random.randint(0, min((b + 1) * self._segments, self._prices_length - 1))

    dg = DataGeneratorSeq(train_data, 5, 5)
    u_data, u_labels = dg.unroll_batches()

    for ui, (dat, lbl) in enumerate(zip(u_data, u_labels)):
        print('\n\nUnrolled index %d' % ui)
        dat_ind = dat
        lbl_ind = lbl
        print('\tInputs: ', dat)
        print('\n\tOutput:', lbl)
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            X.append(a)
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    def lstm_prediction(train_data, test_data):
        # Reshape train_data for LSTM
        train_data = train_data.reshape(-1, 1)
        test_data = test_data.reshape(-1, 1)

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_data = scaler.fit_transform(train_data)
        scaled_test_data = scaler.transform(test_data)

        # Create datasets
        time_step = 60  # Number of previous days to consider for prediction
        X_train, y_train = create_dataset(scaled_train_data, time_step)
        X_test, y_test = create_dataset(scaled_test_data, time_step)

        # Reshape input to be [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))  # Output layer

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32)

        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transform to get actual prices
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        # Plotting results
        plt.figure(figsize=(14, 7))
        plt.plot(range(len(train_data)), train_data, label='Actual Prices', color='blue')
        plt.plot(range(time_step, len(train_predict) + time_step), train_predict, label='Train Predictions', color='orange')
        plt.plot(range(len(train_data) + (time_step * 2), len(train_data) + (time_step * 2) + len(test_predict)), test_predict, label='Test Predictions', color='red')
        plt.title('Cryptocurrency Price Prediction using LSTM')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transform to get actual prices
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        # Calculate MSE and RMSE for LSTM
        lstm_mse = mean_squared_error(train_data[time_step:], train_predict)
        lstm_rmse = np.sqrt(lstm_mse)

        print(f'LSTM Mean Squared Error: {lstm_mse:.5f}')
        print(f'LSTM Root Mean Squared Error: {lstm_rmse:.5f}')

    # Call the LSTM prediction function
    lstm_prediction(train_data, test_data)
    

    def ann_prediction(train_data, test_data):
        # Reshape train_data for ANN
        train_data = train_data.reshape(-1, 1)
        test_data = test_data.reshape(-1, 1)

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_data = scaler.fit_transform(train_data)
        scaled_test_data = scaler.transform(test_data)

        # Create datasets for ANN
        time_step = 60  # Number of previous days to consider for prediction
        X_train, y_train = create_dataset(scaled_train_data, time_step)
        X_test, y_test = create_dataset(scaled_test_data, time_step)

        # Reshape input to be [samples, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

        # Build the ANN model
        model = Sequential()
        model.add(Dense(50, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))  # Output layer

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=32)

        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transform to get actual prices
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        # Plotting results
        plt.figure(figsize=(14, 7))
        plt.plot(range(len(train_data)), train_data, label='Actual Prices', color='blue')
        plt.plot(range(time_step, len(train_predict) + time_step), train_predict, label='Train Predictions', color='orange')
        plt.plot(range(len(train_data) + (time_step * 2), len(train_data) + (time_step * 2) + len(test_predict)), test_predict, label='Test Predictions', color='red')
        plt.title('Cryptocurrency Price Prediction using ANN')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        ann_mse = mean_squared_error(train_data[time_step:], train_predict)
        ann_rmse = np.sqrt(ann_mse)

        print(f'ANN Mean Squared Error: {ann_mse:.5f}')
        print(f'ANN Root Mean Squared Error: {ann_rmse:.5f}')
    # Call the ANN prediction function
    ann_prediction(train_data, test_data)
