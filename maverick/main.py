import configparser

import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments


OHLC = ['open', 'high', 'low', 'close']
MA = 2
TIMESERIES_LEN = 4
QUANTILES_NUM = 2

# Config and connect 
config = configparser.ConfigParser()
config.read('env/oanda.ini')
client = oandapyV20.API(access_token=config['oanda']['api_key'], environment='live')

# request candles
candle_data = instruments.InstrumentsCandles(
    instrument="AUD_CAD", 
    params={"count": "10", "granularity": "H1"}
)
client.request(candle_data)
candles = candle_data.response['candles']

# build dataframe
candles_dict = {i: [] for i in OHLC}
for candle in candles:
    if candle['complete']:
        for i in OHLC:
            candles_dict[i].append(float(candle['mid'][i[0]]))
df = pd.DataFrame(candles_dict)

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

print(pd.DataFrame(q_dict))


# for i in OHLC:
#     print(pd.qcut(pct_dict[f'{i}_pct'], QUANTILES_NUM, labels=False), i)
#     for n in range()

# print(pd.qcut(pct_dict['open_pct'], 4, labels=False))


# print(df_pct.quantile([0.25, 0.5, 0.75]))
# print(pct_dict['open_pct'])

# test_df = pd.DataFrame([[1, 1], [2, 100], [3, 100], [4, 100], [5, 12]], columns=['a', 'b'])
# print(test_df.quantile([0.25, 0.5, 0.75]))

# df = pd.concat([df, df_pct], axis=1, sort=False)
# print(df)
# df = df.dropna().reset_index(drop=True)

# if __name__ == '__main__':
#     print(df)
    # for index, row in df.iterrows():
    #     print(row)