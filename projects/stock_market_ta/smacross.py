from backtesting.test import SMA
from datetime import datetime
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting import Backtest

def dateparse(datestring):
    return datetime.strptime(datestring, '%Y-%m-%d')

class Sma4Cross(Strategy):
    n1 = 50
    n2 = 100
    n_enter = 20
    n_exit = 10

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
        self.sma_enter = self.I(SMA, self.data.Close, self.n_enter)
        self.sma_exit = self.I(SMA, self.data.Close, self.n_exit)

    def next(self):

        if not self.position:

            # On upwards trend, if price closes above
            # "entry" MA, go long

            # Here, even though the operands are arrays, this
            # works by implicitly comparing the two last values
            if self.sma1 > self.sma2:
                if crossover(self.data.Close, self.sma_enter):
                    self.buy()

            # On downwards trend, if price closes below
            # "entry" MA, go short

            else:
                if crossover(self.sma_enter, self.data.Close):
                    self.sell()

        # But if we already hold a position and the price
        # closes back below (above) "exit" MA, close the position

        else:
            if (self.position.is_long and
                    crossover(self.sma_exit, self.data.Close)
                    or
                    self.position.is_short and
                    crossover(self.data.Close, self.sma_exit)):
                self.position.close()


AAPL = pd.read_csv('data/daily/AAPL.csv', parse_dates=True, date_parser=dateparse, index_col='timestamp')
AAPL.drop(['close','dividend_amount','split_coefficient'], inplace=True, axis=1)
AAPL.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
backtest = Backtest(AAPL, Sma4Cross, commission=0)

stats, heatmap = backtest.optimize(
    n1=range(10, 110, 5),
    n2=range(20, 210, 5),
    n_enter=range(15, 35, 5),
    n_exit=range(10, 25, 5),
    constraint=lambda p: p.n_exit < p.n_enter < p.n1 < p.n2,
    maximize='Equity Final [$]',
    return_heatmap=True)
