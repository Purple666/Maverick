import configparser

import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments


OHLC = ['open', 'high', 'low', 'close']

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
df['close_mean'] = df['close'].rolling(2).mean()
pct_dict = {f'{i}_pct': [] for i in OHLC}
for index, row in df.iterrows():
    for i in OHLC:
        pct_dict[f'{i}_pct'].append(((row[i] / row['close_mean']) - 1) * 100)
df_pct = pd.DataFrame(pct_dict)

print(df_pct.quantile([0.25, 0.5, 0.75]))
print(pct_dict['open_pct'])
print(pd.qcut(pct_dict['open_pct'], 4, labels=False))

# test_df = pd.DataFrame([[1, 1], [2, 100], [3, 100], [4, 100], [5, 12]], columns=['a', 'b'])
# print(test_df.quantile([0.25, 0.5, 0.75]))

# df = pd.concat([df, df_pct], axis=1, sort=False)
# print(df)
# df = df.dropna().reset_index(drop=True)

# if __name__ == '__main__':
#     print(df)
    # for index, row in df.iterrows():
    #     print(row)