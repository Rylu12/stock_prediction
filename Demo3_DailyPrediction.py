import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import yfinance as yf


class DailyPrediction:
    """
    This class file is designed to be a program for quick daily stock price trend predictions.
    """

    def __init__(self, ticker = 'SPY'):

        self.ticker = ticker
        data = yf.Ticker(ticker)
        self.stock_train = data.history(period='10y')
        self.stock_train.drop(columns=['Dividends', 'Volume', 'Stock Splits'], inplace=True)

    def _create_buy_sig(self):

        # Length of days to average
        avg_length = 25

        self.stock_train['before_avg'] = self.stock_train['Close'].rolling(window=avg_length).mean()
        self.stock_train['Buy_sig'] = np.ones(len(self.stock_train['Close']))

        for i in range(0, len(self.stock_train)):

            after_avg = 0
            if i == (len(self.stock_train) - avg_length):
                break

            # Get future rolling moving average
            for j in range(0, avg_length):
                after_avg = after_avg + self.stock_train['Close'].iloc[i + j]

            forward_avg = after_avg / avg_length

            if self.stock_train['before_avg'].iloc[i] < forward_avg:
                self.stock_train['Buy_sig'].iloc[i] = 1
            else:
                self.stock_train['Buy_sig'].iloc[i] = 0



    def _get_EMA(self):

        for n in range(3, 100, 2):
            ema = self.stock_train['Close'].iloc[0]
            ema_list = [ema]

            for i in range(1, len(self.stock_train['Close'])):
                ema = self.stock_train['Close'].iloc[i] * (2 / (1 + n)) + ema * (1 - (2 / (1 + n)))
                ema_list.append(ema)
            self.stock_train['EMA'] = ema_list

            # Compare if above or below ema
            slope_ema = [1]
            for j in range(1, len(ema_list)):

                if ema_list[j] < self.stock_train['Close'].iloc[j]:
                    slope_ema.append(1)
                else:
                    slope_ema.append(0)

            self.stock_train['ema_%d' % n] = slope_ema
        self.stock_train.drop(['EMA'], inplace=True, axis=1)


    def _get_RSI(self):
        # Get RSI indicators

        change_gains = [0]
        change_loss = [0]
        for i in range(1, len(self.stock_train)):
            change = self.stock_train['Close'].iloc[i] - self.stock_train['Close'].iloc[i - 1]
            change_gains.append(change if change > 0 else 0)
            change_loss.append(abs(change) if change < 0 else 0)

        self.stock_train['change_gains'] = change_gains
        self.stock_train['change_loss'] = change_loss

        # The RSI window range
        for i in range(3, 31, 1):
            x = self.stock_train['change_gains'].rolling(window=i).mean() / self.stock_train['change_loss'].rolling(
                window=i).mean()
            rsi = 100 - (100 / (1 + x)) + .001
            self.stock_train['RSI_%d' % i] = rsi


    def _get_WillR(self):

        for i in range(3, 100, 2):
            High_high = self.stock_train['High'].rolling(window=i).max()
            Low_low = self.stock_train['Low'].rolling(window=i).min()

            self.stock_train['Will_R_%d' % i] = -100 * (High_high - self.stock_train['Close']) / (High_high - Low_low)


    def _get_Stochastic(self):

        for i in range(3, 35, 1):
            High_price = self.stock_train['Close'].rolling(window=i).max()
            Low_price = self.stock_train['Close'].rolling(window=i).min()

            self.stock_train['Stochastic_%d' % i] = (self.stock_train['Close'] - Low_price) / (High_price - Low_price)


    def _get_feat(self, db_name):
        db = create_engine('sqlite:///' + db_name)

        conn = db.connect()

        fetch = conn.execute("""
        PRAGMA table_info('SPY_LogitReg')
        """).fetchall()

        feat_indicators = [b for a, b, c, d, e, f in fetch][1:8]

        conn.close()

        return feat_indicators


    def predict_trend(self, db_name):

        self._create_buy_sig()
        self._get_EMA()
        self._get_RSI()
        self._get_WillR()
        self._get_Stochastic()
        feat_indicators = self._get_feat(db_name)

        self.stock_train.dropna(axis=0, inplace=True)
        self.stock_train.reset_index(inplace=True)
        self.y = self.stock_train['Buy_sig'].values
        self.x = self.stock_train[feat_indicators].values

        x_train = self.x[0:-1]
        x_test = self.x[-1]
        y_train = self.y[0:-1]
       # y_test = self.y[-1]

        train_scaler = StandardScaler()
        scaled_x_train = train_scaler.fit_transform(x_train)
        scaled_x_test = train_scaler.transform([x_test])

        reg = LogisticRegression(max_iter=10000, tol=0.0001, C=1, random_state=42, n_jobs=-1)

        reg.fit(scaled_x_train, y_train)

        y_pred = reg.predict(scaled_x_test)

        return y_pred

obj = DailyPrediction(ticker = 'SPY')
pred = obj.predict_trend('All_stock.sql')

if pred == 1:
    print('Up Trend ', pred)
else:
    print('Down Trend ', pred)

















