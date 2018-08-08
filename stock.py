import quandl
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


class Stock:

    @staticmethod
    def create_data_set(data_set, look_back=1):
        data_x, data_y = [], []
        for i in range(len(data_set) - look_back - 1):
            a = data_set[i:(i + look_back), 0]
            data_x.append(a)
            data_y.append(data_set[i + look_back, 0])
        return numpy.array(data_x), numpy.array(data_y)

    def __init__(self, ticker, exchange):
        numpy.random.seed(7)

        self.ticker = ticker.upper()
        self.exchange = exchange.upper()

        try:
            quandl.ApiConfig.api_key = 't7g5V4bGamZ5nxYwxuns'
            stock = quandl.get('%s/%s' % (exchange, ticker))
            stock = stock.dropna()
            print(stock.head())
            print(stock.tail())
            print(stock.shape)

        except Exception as e:
            print('Error Retrieving Data.')
            print(e)
            return

        self.stock = stock.reset_index(level=0)

        self.stock['DS'] = self.stock['Date']
        self.stock['X'] = self.stock['Close']
        self.stock['Change'] = self.stock['Close'] - self.stock['Open']

        self.data_frame = []
        self.data_set = []
        self.train_size = -1
        self.test_size = -1
        self.train = []
        self.test = []
        self.look_back = 1
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.model = Sequential()

        self.min_date = min(self.stock['DS'])
        self.max_date = max(self.stock['DS'])

        print('{} Stock Initialized. Data covers {} to {}.'.format(self.ticker,
                                                                   self.min_date.date(),
                                                                   self.max_date.date()))

    def plot_stock_history(self):
        plt.plot(self.stock['Date'], self.stock['X'], linewidth=0.6, label='X')
        plt.xlabel('Date')
        plt.ylabel('INR Rs')
        plt.title('%s Stock History' % self.ticker)
        plt.show()

    def load_data_set(self):
        self.data_frame = pandas.DataFrame(data={'X': self.stock['X']})
        self.data_set = self.data_frame.values

    def normalize(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_set = scaler.fit_transform(self.data_set)

    def split_data_set(self, split_fraction=0.67):
        self.train_size = int(len(self.data_set) * split_fraction)
        self.test_size = len(self.data_set) - self.train_size
        self.train, self.test = self.data_set[0:self.train_size, :], self.data_set[self.train_size:len(self.data_set), :]

    def reshape_input(self, look_back=1):
        self.look_back = look_back
        self.train_x, self.train_y = self.create_data_set(self.train, self.look_back)
        self.test_x, self.test_y = self.create_data_set(self.test, self.look_back)
        self.train_x = numpy.reshape(self.train_x, (self.train_x.shape[0], 1, self.train_x.shape[1]))
        self.test_x = numpy.reshape(self.test_x, (self.test_x.shape[0], 1, self.test_x.shape[1]))

    def create_lstm_model(self, loss='mean_squared_error', optimizer='adam'):
        self.model.add(LSTM(4, input_shape=(1, self.look_back)))
        self.model.add(Dense(1))
        self.model.compile(loss=loss, optimizer=optimizer)

    def fit_model(self, epochs=100, batch_size=1, verbose=2):
        self.model.fit(self.train_x, self.train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
