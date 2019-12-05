import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.test import SMA
from sqlalchemy import create_engine
from tqdm import tqdm
from datetime import datetime
from backtesting.lib import crossover
from sklearn.


HOST = 'localhost'
USER = 'postgres'
PASS = 'password'
db = 'stock_market'
conn_string = f"host={HOST} port=5432 dbname={db} user={USER} password={PASS}"
engine = create_engine(f'postgresql://{USER}:{PASS}@{HOST}:5432/{db}')

def dateparse(datestring):
    return datetime.strptime(datestring, '%Y-%m-%d')

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

i = 608400

strategies = [
    {'n1': 60, 'n2': 200, 'n3': 65, 'n4': 165},
    {'n1': 70, 'n2': 200, 'n3': 70, 'n4': 190},
    {'n1': 120, 'n2': 185, 'n3': 50, 'n4': 175},
    {'n1': 145, 'n2': 195, 'n3': 120, 'n4': 165},
    {'n1': 65, 'n2': 195, 'n3': 70, 'n4': 175},
    {'n1': 65, 'n2': 200, 'n3': 70, 'n4': 175},
    {'n1': 110, 'n2': 200, 'n3': 55, 'n4': 160},
    {'n1': 60, 'n2': 195, 'n3': 65, 'n4': 165},
    {'n1': 170, 'n2': 200, 'n3': 50, 'n4': 175},
    {'n1': 85, 'n2': 200, 'n3': 55, 'n4': 175}
]

file = open('non_stationary.txt', 'r')
tickers = file.readlines()
tickers = [t[:-1] for t in tickers]
trades = pd.DataFrame(columns=['Open Date', 'Close Date', 'Strategy', 'Price at Buy', 'Price at Sell', 'Win', 'Gain', 'Ticker'])
for ticker in tqdm(tickers):
    df = pd.read_sql(f'SELECT timestamp AS "index", open AS "Open", high as "High", low as "Low", adjusted_close as '
                     f'"Close", volume as "Volume" FROM market_data WHERE ticker = \'{ticker}\' ORDER BY "index" ASC',
                     engine, index_col='index')
    df.index = pd.to_datetime(df.index)

    s = 1
    for strategy in strategies:
        df[f"{strategy['n1']}d ma"] = df.Close.rolling(window=strategy['n1']).mean()
        df[f"{strategy['n2']}d ma"] = df.Close.rolling(window=strategy['n2']).mean()
        df[f"{strategy['n3']}d ma"] = df.Close.rolling(window=strategy['n3']).mean()
        df[f"{strategy['n4']}d ma"] = df.Close.rolling(window=strategy['n4']).mean()
        df[f's{s} buy'] = df[f"{strategy['n1']}d ma"] > df[f"{strategy['n2']}d ma"]
        df[f's{s} sell'] = df[f"{strategy['n3']}d ma"] < df[f"{strategy['n4']}d ma"]
        s += 1

    for i in range(1, 11):
        in_trade = False
        for idx, row in df.iterrows():
            if not in_trade:
                if row[f's{i} buy'] == 1:
                    open_date = idx
                    open_price = row.Close
                    in_trade = True
            else:
                if row[f's{i} sell'] == 1:
                    close_date = idx
                    close_price = row.Close
                    in_trade = False
                    tdf = pd.DataFrame(columns=['Open Date', 'Close Date', 'Strategy', 'Price at Buy', 'Price at Sell', 'Win', 'Gain', 'Ticker'],
                                       data=[[open_date, close_date, i, open_price, close_price, close_price > open_price, close_price-open_price, ticker]])
                    trades = trades.append(tdf, sort=False)
trades.to_csv('trades.csv')
trades.to_sql('trades', if_exists='append', con=engine)

X = pd.concat([trades,pd.get_dummies(trades.Strategy)], axis=1)
X.reset_index(drop=True, inplace=True)

X['unique_opens'] = X.apply(lambda x: str(x['Open Date']) + x['Ticker'], axis=1)

cols_for_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'Win']
X['Win'] = X.apply(lambda x: int(x['Win']), axis=1)
melted = X.pivot_table(index='unique_opens', aggfunc=np.sum, values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
melted_wins = X.pivot_table(index='unique_opens', aggfunc=np.max, values='Win')
melted_gains = X.pivot_table(index='unique_opens', aggfunc=np.max, values='Gain')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(melted, melted_wins, test_size=0.33, random_state=0)

clf = RandomForestClassifier(n_estimators=10000, random_state=0, verbose=1, n_jobs=-1)
clf.fit(X_train, y_train)

acc_score = accuracy_score(y_test, clf.predict(X_test))
confusion_matrix(y_test, clf.predict(X_test))
tn, fp, fn, tp = confusion_matrix(y_test, clf.predict(X_test)).ravel()
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2*(precision*recall)/(precision+recall)
accuracy = (tp+tn)/(tp+fp+fn+tn)

y_preds = clf.predict(X_test)


test_gains = melted_gains.loc[y_test.index]

probs = clf.predict_proba(X_test)