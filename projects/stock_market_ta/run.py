import luigi
import time
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import SignalStrategy, TrailingStrategy
from backtesting.test import SMA, GOOG

class HelloWorld(luigi.Task):
    def requires(self):
        return None

    def output(self):
        return luigi.LocalTarget('helloworld.txt')

    def run(self):
        #time.sleep(15)
        with self.output().open('w') as outfile:
            outfile.write('Hello World!\n')
        #time.sleep(15)


class NameSubstituter(luigi.Task):
    name = luigi.Parameter()

    def requires(self):
        return HelloWorld()

    def output(self):
        return luigi.LocalTarget(self.input().path + '.name_' + self.name)

    def run(self):
        #time.sleep(15)
        with self.input().open() as infile, self.output().open('w') as outfile:
            text = infile.read()
            text = text.replace('World', self.name)
            outfile.write(text)
        #time.sleep(15)


class Backtester(luigi.Task):
    n1 = luigi.IntParameter()
    n2 = luigi.IntParameter()
    def requires(self):
        return None

    def output(self):
        return luigi.LocalTarget(f'{self.n1}-{self.n2}.csv')

    def run(self):
        #time.sleep(15)

        class SmaCross(SignalStrategy, TrailingStrategy):


            def init(self):
                n1 = luigi.IntParameter()
                n2 = luigi.IntParameter()
                super().init()
                sma1 = self.I(SMA, self.data.Close, n1)
                sma2 = self.I(SMA, self.data.Close, n2)

                signal = (pd.Series(sma1) > sma2).astype(int).diff().fillna(0)

                self.set_signal(signal)

                self.set_trailing_sl(4)

        #time.sleep(5)
        bt = Backtest(GOOG, SmaCross, cash=10000, commission=.002)
        output = bt.run()
        output.to_csv(f'{self.n1}-{self.n2}.csv')


if __name__ == '__main__':
    luigi.run()