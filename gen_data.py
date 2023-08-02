import random
from binance import Client
import pandas as pd
"""
RSI_BTC = 0.0
RSI_USDT = 1000.0

MA_BTC = 0.0
MA_USDT = 1000.0

MA_PERIOD = 3
EMA_PERIOD = 4
RSI_PERIOD = 16
OVERBOUGHT = 26
OVERSOLD = 68

"""

client = Client()


frame = pd.DataFrame(client.get_historical_klines("BTCUSDT",'15m', '15 days ago UTC'))
frame = frame.iloc[:,:6]
frame = pd.DataFrame(frame)
frame.columns = ['T', 'O', 'H','L','C', 'V']
frame = frame.astype(float)
frame = frame.pct_change()
frame = frame.iloc[1:,:]
print("saving")
frame.to_csv('data.txt', header = None, sep=',', mode='w')
print("done")


