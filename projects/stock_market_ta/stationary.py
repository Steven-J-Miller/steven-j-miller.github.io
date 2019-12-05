from statsmodels.tsa.stattools import adfuller
import pandas as pd
import os
from tqdm import tqdm

stationary = []
non_stationary = []
uncertain = []

files = os.listdir('data/daily')

for file in tqdm(files):
    data = pd.read_csv(f'data/daily/{file}')
    data.sort_values(axis=0, by='timestamp', inplace=True)
    adf_test = adfuller(data['adjusted_close'].values)

    if adf_test[0] < adf_test[4]['1%'] and adf_test[1] < .05:
        stationary.append(file[:-4])
    elif adf_test[0] > adf_test[4]['1%'] and adf_test[1] > .05:
        non_stationary.append(file[:-4])
    else:
        uncertain.append(file[:-4])

with open('stationary.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % ticker for ticker in stationary)
with open('non_stationary.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % ticker for ticker in non_stationary)
with open('uncertain.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % ticker for ticker in uncertain)
'''
intra_stationary = []
intra_non_stationary = []
intra_uncertain = []

files = os.listdir('data/intraday')

for file in tqdm(files):
data = pd.read_csv(f'data/intraday/{file}')
data.sort_values(axis=0, by='timestamp', inplace=True)
adf_test = adfuller(data['close'].values)

if adf_test[0] < adf_test[4]['1%'] and adf_test[1] < .05:
    intra_stationary.append(file[:-4])
elif adf_test[0] > adf_test[4]['5%'] and adf_test[1] > .05:
    intra_non_stationary.append(file[:-4])
else:
    intra_uncertain.append(file[:-4])
'''