import pandas as pd
import numpy as np
from numba import njit
from wrappers.time_it import  timeit

@njit
def optimized_computation(close: pd.Series, channel_period: int):
    n = len(close)
    upper = np.empty(n)
    lower = np.empty(n)
    upper[0] = close[0]
    lower[0] = close[0]

    for i in range(1,n):
        src = close[i]
        prev_upper = upper[i-1]
        prev_lower = lower[i-1]

        if src > prev_upper:
            upper[i] = src
        else:
            upper[i] = prev_upper - (prev_upper - prev_lower) / channel_period

        if src < prev_lower:
            lower[i] = src
        else:
            lower[i] = prev_lower + (prev_upper - prev_lower) / channel_period

    return upper, lower

class GChannelIndicator:
    """
    This class takes a pandas DataFrame with OHLC data and calculates the
    G-Channel values (Upper, Lower, Middle bands). These values will then be
    converted into G%, G_Width, Upper_Slope, Middle_Slope, and Lower_Slope values.

    The G-Channel indicator is a recently invented indicator, this implementation
    is based on the original research paper:
    https://mpra.ub.uni-muenchen.de/95806/1/MPRA_paper_95806.pdf.

    It returns a DataFrame with these additional G%, G_Width, Upper_Slope, Middle_Slope, and
    Lower_Slope columns.
    """
    def __init__(self, channel_period: int = 100):
        """
        Initializes the GChannelIndicator with a default period of 100.

        Args:
            channel_period (int): The period for the channel calculation.
        """
        if channel_period < 1:
            raise ValueError("Channel Period must be a positive integer.")
        self.channel_period = channel_period

    @timeit
    def calculate_g_channel(self, ohlc_df: pd.DataFrame, timeframe: int) -> pd.DataFrame:
        """
        Calculates the G-Channel indicator values.

        Args:
            timeframe (int): The timeframe which the indicator is being calculated for.
            ohlc_df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns.


        Returns:
            pd.DataFrame: The DataFrame with newly calculated indicator columns.
        """
        # Ensure the required 'Close' column exists
        if 'Close' not in ohlc_df.columns:
            raise ValueError("DataFrame must contain a 'Close' column.")

        df = ohlc_df.copy()
        close = df['Close'].values.astype(np.float64)

        upper, lower = optimized_computation(close, self.channel_period)

        # Assign results
        df['UpperBuffer']  = upper
        df['LowerBuffer']  = lower
        # Calculate the Middle Buffer
        df['MiddleBuffer'] = (df['UpperBuffer'] + df['LowerBuffer']) / 2

        # Calculate G_Width
        df[f'G_Width_{timeframe}'] = df['UpperBuffer'] - df['LowerBuffer']

        df[f'G_Width_{timeframe}'] = upper - lower
        df[f'Upper_Slope_{timeframe}'] = np.diff(np.insert(upper, 0, np.nan))
        df[f'Middle_Slope_{timeframe}'] = np.diff(np.insert((upper + lower) / 2, 0, np.nan))
        df[f'Lower_Slope_{timeframe}'] = np.diff(np.insert(lower, 0, np.nan))

        # G% calculation
        g_percent = np.full(len(close), np.nan)

        middle = df['MiddleBuffer'].values
        # Prices above Middle Band
        # When price is at UpperBand, (UpperBand - MiddleBand) / (UpperBand - MiddleBand) = 1
        # When price is at MiddleBand, (MiddleBand - MiddleBand) / (UpperBand - MiddleBand) = 0
        upper_mask = close >= middle
        # Prices below Middle Band
        # When price is at MiddleBand, (MiddleBand - MiddleBand) / (MiddleBand - LowerBand) = 0
        # When price is at LowerBand, (LowerBand - MiddleBand) / (MiddleBand - LowerBand) = -1
        lower_mask = close < middle

        g_percent[upper_mask] = (close[upper_mask] - middle[upper_mask]) / (upper[upper_mask] - middle[upper_mask])
        g_percent[lower_mask] = (close[lower_mask] - middle[lower_mask]) / (middle[lower_mask] - lower[lower_mask])
        df[f'G%_{timeframe}'] = g_percent

        # Drop the unnecessary columns (due to differencing of timeseries, we don't want undifferenced vales)
        df = df.drop(columns=['UpperBuffer', 'LowerBuffer', 'MiddleBuffer'])

        return df