import configparser
from itertools import product

import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, SpatialDropout1D
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


OHLC = ['open', 'high', 'low', 'close']
MA = 50
TIMESERIES_LEN = 50
QUANTILES_NUM = 10

# Config and connect 
config = configparser.ConfigParser()
config.read('env/oanda.ini')
client = oandapyV20.API(access_token=config['oanda']['api_key'], environment='live')

# request candles
candle_data = instruments.InstrumentsCandles(
    instrument="AUD_CAD", 
    params={"count": "5000", "granularity": "H1"}
)
client.request(candle_data)
candles = candle_data.response['candles']
print("Candles 1/5")

# build dataframe
candles_dict = {i: [] for i in OHLC}
for candle in candles:
    if candle['complete']:
        for i in OHLC:
            candles_dict[i].append(float(candle['mid'][i[0]]))
df = pd.DataFrame(candles_dict)
print("Build dataframe 2/5")

# y_train
y_train = []
for index, row in df.iterrows():
    if row['close'] > row['open']:
        y_train.append(2)
    elif row['close'] < row['open']:
        y_train.append(0)
    else:
        y_train.append(1)
y_train = to_categorical(y_train)
print("Build y_train 3/5")

# Add features to dataframe
df['close_mean'] = df['close'].rolling(MA).mean()
pct_dict = {f'{i}_pct': [] for i in OHLC}
for index, row in df.iterrows():
    for i in OHLC:
        result = ((row[i] / row['close_mean']) - 1) * 100
        if not np.isnan(result): 
            pct_dict[f'{i}_pct'].append(result)

# Feature engineer 
q_dict = {f'{i}_q_{n}': [] for i in OHLC for n in range(QUANTILES_NUM)}
for i in OHLC:
    for ts in range(len(pct_dict[f'{i}_pct'])-TIMESERIES_LEN+1):
        for n in range(QUANTILES_NUM):
            temp_list = []
            for qv in pd.qcut(pct_dict[f'{i}_pct'][ts:ts+TIMESERIES_LEN], QUANTILES_NUM, labels=False):
                temp_list.append(1 if qv == n else 0)
            q_dict[f'{i}_q_{n}'].append(temp_list)
print("Feature engineer 4/5")


temp_df = pd.DataFrame(q_dict)
x_train = []
for _, row in temp_df.iterrows():
    new_series = []
    for num in range(len(row[0])):
        new_row = []
        for i in OHLC:
            for n in range(QUANTILES_NUM):
                new_row.append(row[f'{i}_q_{n}'][num])
        new_series.append(new_row)
    x_train.append(new_series)
print("Finale Feature organization 5/5")

# Run model
x_train = np.array(x_train)[:-1]
y_train = y_train[-len(x_train):]

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

worst = {
    'val_loss': None,
    'val_acc': None,
    'params': None
}
best = {
    'val_loss': None,
    'val_acc': None,
    'params': None
}
callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
for drop_out, lr in product(range(1, 9), range(0, 101, 10)):
    if not lr:
        continue
    drop_out = drop_out/10
    lr = lr/1000
    model = Sequential()
    model.add(LSTM(40, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(SpatialDropout1D(drop_out))
    model.add(LSTM(80, return_sequences=True))
    # model.add(SpatialDropout1D(drop_out))
    # model.add(LSTM(40, return_sequences=True))
    model.add(SpatialDropout1D(drop_out))
    model.add(LSTM(40, return_sequences=False))
    model.add(Dense(y_train.shape[1], activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=200, epochs=100, validation_split=0.4, verbose=1, callbacks=callbacks)
    result = {
        'val_loss': history.history['val_loss'][-1],
        'val_acc': history.history['val_acc'][-1],
        'params': {'drop_out': drop_out, 'learning_rate': lr}
    }
    if not best['val_loss'] or best['val_loss'] > history.history['val_loss'][-1]:
        best.update(result)
    if not worst['val_loss'] or worst['val_loss'] < history.history['val_loss'][-1]:
        worst.update(result)
    print("current:", drop_out, lr)
    print("worst:", worst)
    print("best:", best)