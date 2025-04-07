import MetaTrader5 as mt5
import pandas as pd

def get_symbol_data(symbol, n=500):
    if not mt5.initialize():
        print("MT5 init error")
        return pd.DataFrame()
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume"]]
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df