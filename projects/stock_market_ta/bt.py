import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.test import SMA, GOOG
from sqlalchemy import create_engine
from tqdm import tqdm
import numpy as np
import tulipy as tl
from datetime import datetime
from backtesting.lib import crossover

core = 99
HOST = 'localhost'
USER = 'postgres'
PASS = 'password'
db = 'stock_market'
conn_string = f"host={HOST} port=5432 dbname={db} user={USER} password={PASS}"
engine = create_engine(f'postgresql://{USER}:{PASS}@{HOST}:5432/{db}')


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def dateparse(datestring):
    return datetime.strptime(datestring, '%Y-%m-%d')

def RSI(array, n, ob, os):
    """Relative strength index"""
    # Approximate; good enough
    rs = tl.rsi(np.array(array), n)
    rs = np.insert(rs, 0, np.repeat((ob+os)/2, n))
    return rs

class SmaCross(Strategy):
    n1 = 10
    n2 = 30
    n3 = 10
    n4 = 30
    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
        self.sma3 = self.I(SMA, self.data.Close, self.n3)
        self.sma4 = self.I(SMA, self.data.Close, self.n4)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma3, self.sma4):
            self.sell()

class RSI_Strat(Strategy):
    n1 = 30
    rsi_ob = 70 # overbought
    rsi_os = 30 # oversold

    def init(self):
        self.rsi = self.I(RSI, array=self.data.Close, n=self.n1, ob=self.rsi_ob, os=self.rsi_os)

    def next(self):
        if not self.position and self.rsi[-1] < self.rsi_os:
            self.buy()
        elif self.rsi[-1] > self.rsi_ob:
            self.sell()


# SMA TEST
tickers = pd.read_sql('SELECT DISTINCT(ticker) FROM market_data', engine).ticker.values.tolist()
tickers = list(divide_chunks(tickers, 1))[core]
failed_tests = []
n1_range = range(5, 205, 5)
n2_range = range(5, 205, 5)
n3_range = range(5, 205, 5)
n4_range = range(5, 205, 5)
i = 40*40*40*40*len(tickers)
pbar = tqdm(total=i)

for ticker in tickers:
    df = pd.read_sql(f'SELECT timestamp AS "index", open AS "Open", high as "High", low as "Low", adjusted_close as '
                     f'"Close", volume as "Volume" FROM market_data WHERE ticker = \'{ticker}\' AND timestamp <= \'2017-12-31\' ORDER BY "index" ASC',
                     engine, index_col='index')
    df.index = pd.to_datetime(df.index)

    bt = Backtest(df, SmaCross, cash=10000, commission=0)
    for n1 in n1_range:
        for n2 in n2_range:
            for n3 in n3_range:
                for n4 in n4_range:
                    if (n1 < n2) and (n3 < n4):
                        try:
                            output = bt.run(n1=n1, n2=n2, n3=n3, n4=n4)
                            output['ticker'] = ticker
                            output = output.to_frame().transpose()
                            output.columns = [c.replace('%','') for c in output.columns]
                            output._strategy = str(output._strategy).split('\n')[0]
                            output.to_sql('test_results', if_exists='append', con=engine)
                        except:
                            failed_tests.append({'ticker':ticker, 'n1':n1,'n2':n2, 'n3':n3,'n4':n4})
                    pbar.update(1)
'''
#RSI TEST

ob = range(5,95,5)
os = range(5,95,5)
n = range(2,90,1)
for ticker in tickers:
    df = pd.read_sql(f'SELECT timestamp AS "index", open AS "Open", high as "High", low as "Low", adjusted_close as '
                     f'"Close", volume as "Volume" FROM market_data WHERE ticker = \'{ticker}\' ORDER BY "index" ASC',
                     engine, index_col='index')
    df.index = pd.to_datetime(df.index)
    bt = Backtest(df, RSI_Strat, cash=10000, commission=0)
    for ob_val in ob:
        for os_val in os:
            if ob_val > os_val:
                for n_val in n:
                    try:
                        output = bt.run(n1 = n_val, rsi_ob = ob_val, rsi_os = os_val)
                        output['ticker'] = ticker
                        output = output.to_frame().transpose()
                        output.columns = [c.replace('%', '') for c in output.columns]
                        output._strategy = str(output._strategy).split('\n')[0]
                        output.to_sql('test_results', if_exists='append', con=engine)
                    except:
                        failed_tests.append({'ticker': ticker, 'n1': n_val, 'rsi_ob': ob_val, 'rsi_os': os_val})
                    pbar.update(1)
'''