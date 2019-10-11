import configparser
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

config = configparser.ConfigParser()
config.read(f'oanda.ini')
client = oandapyV20.API(access_token=config['oanda']['api_key'], environment='live')