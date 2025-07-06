import pandas as pd
from wrappers.time_it import timeit

class AverageTrueRange:
    """
     This class takes in a pandas DataFrame with OHLC data and calculates the
     Average True Range values for the given period.

     It returns a dataframe with the calculated Average True Range values.
    """
    def __init__(self, period=14):
        self.period = period

    @timeit
    def calculate_atr(self, df: pd.DataFrame, timeframe: int):
        """
        Calculates the Average True Range (ATR) for a given DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with 'High', 'Low', and 'Close' columns.
            timeframe (int): Timeframe for which the indicator is being calculated for.
        Returns:
            pd.DataFrame: The original DataFrame with an added 'ATR' column.

        """
        # Make a copy to avoid modifying the original DataFrame
        df_atr = df.copy()

        # Shift the 'Close' column to get the previous candle's close
        df_atr['previous_close'] = df_atr['Close'].shift(1)

        # Calculate the three components of True Range
        df_atr['tr1'] = df_atr['High'] - df_atr['Low']
        df_atr['tr2'] = abs(df_atr['High'] - df_atr['previous_close'])
        df_atr['tr3'] = abs(df_atr['Low'] - df_atr['previous_close'])

        # Calculate the True Range (TR) which is the max of the three components
        df_atr['true_range'] = df_atr[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Calculate the Average True Range (ATR) using a simple moving average (SMA)
        df_atr[f'atr_{timeframe}'] = df_atr['true_range'].rolling(window=self.period).mean()

        return df_atr[[f'atr_{timeframe}']]