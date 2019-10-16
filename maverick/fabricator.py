import configparser
import random

import numpy as np
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from keras.utils import to_categorical


class Candles:
    def __get__(self, instance, owner):
        candle_data = instruments.InstrumentsCandles(
            instrument=instance._instrument, 
            params={
                "count": str(instance._candle_count), 
                "granularity": instance._granularity
            }
        )
        instance._client.request(candle_data)
        candles = candle_data.response['candles']
        return [
            [float(i['mid']['o']) for i in candles if i['complete']],
            [float(i['mid']['h']) for i in candles if i['complete']],
            [float(i['mid']['l']) for i in candles if i['complete']],
            [float(i['mid']['c']) for i in candles if i['complete']]
        ]


class Fabricator:
    candles = Candles()

    def __init__(self, client, instrument, candle_count, granularity):
        assert isinstance(client, dict), "`client` must be dictionary with the key `file`, or `api_key`"
        if 'file' in client.keys() and 'api_key' not in client.keys():
            config = configparser.ConfigParser()
            config.read(client['file'])
            api_key = config['oanda']['api_key']
        else:
            api_key = client['api_key']
        self._client = oandapyV20.API(access_token=api_key, environment='live')
        self._instrument = instrument
        self._candle_count = candle_count
        self._granularity = granularity

    def x_y_train(self, time_series, look_forward, quantile_count, data_split=None):
        x_train = []
        y_train = []
        candles = self.candles
        for series in range(len(candles[0])-time_series+1-look_forward):
            series_result = np.zeros((time_series, 0))
            dif = candles[3][series+time_series+look_forward-1] - candles[3][series+time_series-1]
            if dif > 0: # Price has gone up
                y_train.append(2)
            elif dif < 0: # Price has gone down
                y_train.append(0)
            else: # Price has stayed the same
                y_train.append(1)
            for ohlc in candles:
                result = to_categorical(pd.qcut(ohlc[series:series+time_series], quantile_count, labels=False))
                series_result = np.append(series_result, result, axis=1)
            x_train.append(series_result)
        x_train = np.array(x_train)
        y_train = to_categorical(y_train)

        # normalize data to even long to short
        total_long = len(list(filter(lambda x: x[2] == 1, y_train)))
        total_middle = len(list(filter(lambda x: x[1] == 1, y_train)))
        total_short = len(list(filter(lambda x: x[0] == 1, y_train)))
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
        if not data_split:
            return x_train, y_train

        # split train and test data
        rows = {'long': [], 'short': []}
        for n, row in enumerate(y_train):
            if row[2]:
                rows['long'].append(n)
            elif row[0]:
                rows['short'].append(n)
        random.shuffle(rows['long'])
        random.shuffle(rows['short'])

        split = int(min([total_long, total_short])*data_split)
        val_rows = rows['long'][:split]
        val_rows.extend(rows['short'][:split])

        x_val = x_train[val_rows]
        y_val = y_train[val_rows]
        x_train = np.delete(x_train, val_rows, axis=0)
        y_train = np.delete(y_train, val_rows, axis=0)
        return x_train, y_train, x_val, y_val

    def x_train(self):
        pass

if __name__ == '__main__':
    x = Fabricator({'file': 'env/oanda.ini'}, 'EUR_USD', 5000, 'M5')
    x_train, y_train, x_val, y_val = x.x_y_train(48, 2, 6, data_split=0.2)