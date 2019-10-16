import configparser

import oandapyV20
import oandapyV20.endpoints.instruments as instruments


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