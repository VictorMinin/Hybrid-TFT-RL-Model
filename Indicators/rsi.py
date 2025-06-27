import pandas_ta as ta
import pandas as pd
from wrappers.time_it import timeit


@timeit
def calculate_rsi(df: pd.DataFrame, timeframe: int, leng = 14) -> pd.DataFrame:
    """
    Calculates the RSI value for the timeframe and length given.

    Parameters:
    df (pd.DataFrame): DataFrame with columns ['High', 'Low', 'Close']
    timeframe (int): The timeframe of df used for naming of new column
    length (int): The length used in rsi calculation

    Returns:
    pd.DataFrame: DataFrame with added columns ['RSI_tf']
    """

    rsi = ta.rsi(df["Close"], length = leng)
    df[f'RSI_{leng}_{timeframe}'] = rsi

    return df