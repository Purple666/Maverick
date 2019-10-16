import configparser
from itertools import product
import random

import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from keras.utils import to_categorical

from model import Maverick

QUANTILES_NUM = 6
TIME_SERIES = 48
LOOK_FORWARD = 2
DATA_SPLIT = 0.2

# Config and connect 
config = configparser.ConfigParser()
config.read('env/oanda.ini')
client = oandapyV20.API(access_token=config['oanda']['api_key'], environment='live')

# request candles
candle_data = instruments.InstrumentsCandles(
    instrument="EUR_USD", # AUD_CAD EUR_USD
    params={"count": "5000", "granularity": "M5"}
)
client.request(candle_data)
candles = candle_data.response['candles']
candles = [
    [float(i['mid']['o']) for i in candles if i['complete']],
    [float(i['mid']['h']) for i in candles if i['complete']],
    [float(i['mid']['l']) for i in candles if i['complete']],
    [float(i['mid']['c']) for i in candles if i['complete']]
]


x_train = []
y_train = []
for series in range(len(candles[0])-TIME_SERIES+1-LOOK_FORWARD):
    series_result = np.zeros((TIME_SERIES, 0))
    dif = candles[3][series+TIME_SERIES+LOOK_FORWARD-1] - candles[3][series+TIME_SERIES-1]
    if dif > 0: # Price has gone up
        y_train.append(2)
    elif dif < 0: # Price has gone down
        y_train.append(0)
    else: # Price has stayed the same
        y_train.append(1)
    for ohlc in candles:
        result = to_categorical(pd.qcut(ohlc[series:series+TIME_SERIES], QUANTILES_NUM, labels=False))
        series_result = np.append(series_result, result, axis=1)
    x_train.append(series_result)
x_train = np.array(x_train)
y_train = to_categorical(y_train)

# normalize data to even long to short
total_long = len(list(filter(lambda x: x[2] == 1, y_train)))
total_middle = len(list(filter(lambda x: x[1] == 1, y_train)))
total_short = len(list(filter(lambda x: x[0] == 1, y_train)))
print("Long:", total_long)
print("middle:", total_middle)
print("Short:", total_short)

to_delete = []
index = None
difference = 0
if total_long > total_short:
    index = 2
    difference = total_long - total_short
elif total_long < total_short:
    index = 0
    difference = total_short - total_long
if difference > 0:
    for n, row in enumerate(y_train):
        if row[index]:
            to_delete.append(n)
    random.shuffle(to_delete)
    x_train = np.delete(x_train, to_delete[:difference], axis=0)
    y_train = np.delete(y_train, to_delete[:difference], axis=0)

# split train and test data
rows = {'long': [], 'short': []}
for n, row in enumerate(y_train):
    if row[2]:
        rows['long'].append(n)
    elif row[0]:
        rows['short'].append(n)
random.shuffle(rows['long'])
random.shuffle(rows['short'])

split = int(min([total_long, total_short])*DATA_SPLIT)
val_rows = rows['long'][:split]
val_rows.extend(rows['short'][:split])

x_val = x_train[val_rows]
y_val = y_train[val_rows]
x_train = np.delete(x_train, val_rows, axis=0)
y_train = np.delete(y_train, val_rows, axis=0)


print("Long and short now even")

total_history = []
X = Maverick(x_train, y_train, validation_data=(x_val, y_val), callbacks=True)
# for i in range(10):
# history = X.run_model(300, 0.1, 0.01, 22, 34, 48)
# total_history.append((history.history['val_loss'][-1], history.history['val_acc'][-1]))
# print(total_history)
# X.bayesian_optimization(
#     batch_size=(300, 300), 
#     drop_out=(0.1, 0.8), 
#     learning_rate=(0.01, 0.1), 
#     neurons=((6, 36), (12, 48), (12, 48))
# )

X.run_model(300, 0.1, 0.01, 22, 34, 48)
