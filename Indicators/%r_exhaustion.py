import pandas as pd
import pandas_ta as pta
from wrappers.time_it import timeit

class RTrendExhaustion:
    """
    A Python implementation of the "%R Trend Exhaustion" Pine Script indicator.

    This class takes in a pandas DataFrame with OHLC data and calculates the
    calculates two Williams %R oscillator values (a fast and a slow one).

    It returns a dataframe with the calculated Williams %R oscillator values slow_r,
    fast_r, and r_diff.
    """
    def __init__(self,
                 short_length: int = 21,
                 short_smoothing_length: int = 7,
                 long_length: int = 112,
                 long_smoothing_length: int = 3,
                 smooth_type: str = 'ema',
                 src: str = 'Close'):
        """
        Initializes the R_Trend_Exhaustion indicator with specified settings.

        Args:
            short_length (int): The lookback period for the fast %R.
            short_smoothing_length (int): The smoothing period for the fast %R.
            long_length (int): The lookback period for the slow %R.
            long_smoothing_length (int): The smoothing period for the slow %R.
            smooth_type (str): The type of moving average for smoothing.
                              Supports 'sma', 'ema', 'hma', 'rma', and 'wma'.
            src (str): The source price series to use ('close', 'high', etc.).
        """
        self.short_length = short_length
        self.short_smoothing_length = short_smoothing_length
        self.long_length = long_length
        self.long_smoothing_length = long_smoothing_length
        self.smooth_type = smooth_type.lower()
        self.src_col = src.lower()

    def get_ma(self, series: pd.Series, length: int, ma_type: str, volume: pd.Series = None) -> pd.Series:
        """
        Applies a selected moving average to a series using pandas-ta.
        """
        series = series.copy()
        # Fill NA to prevent issues with some TA calculations
        series.fillna(0, inplace=True)

        ma_type = ma_type.lower()

        if ma_type == 'sma':
            return pta.sma(series, length)
        elif ma_type == 'ema':
            return pta.ema(series, length)
        elif ma_type == 'hma':
            return pta.hma(series, length)
        elif ma_type == 'rma':
            return pta.rma(series, length)
        elif ma_type == 'wma':
            return pta.wma(series, length)
        else:
            raise ValueError(f"Unsupported smoothing type: {ma_type}")

    @timeit
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the %R values based on the input DataFrame.

        Args:
            df (pd.DataFrame): A DataFrame with 'high', 'low', 'close' columns
                               and a datetime index.

        Returns:
            pd.DataFrame: A DataFrame with the datetime index, 'fast_r', 'slow_r', and 'r_diff' columns.
        """
        if not all(col in df.columns for col in ['High', 'Low', self.src_col]):
            raise ValueError(f"Input DataFrame must contain 'High', 'Low', and '{self.src_col}' columns.")

        df_copy = df.copy()

        # Calculate Williams %R using pandas-ta built-in function 'willr'
        s_percent_r = pta.willr(df_copy['High'], df_copy['Low'], df_copy[self.src_col], length=self.short_length)
        l_percent_r = pta.willr(df_copy['High'], df_copy['Low'], df_copy[self.src_col], length=self.long_length)

        # Apply smoothing
        if self.short_smoothing_length > 1:
            s_percent_r = self.get_ma(s_percent_r, self.short_smoothing_length, self.smooth_type)

        if self.long_smoothing_length > 1:
            l_percent_r = self.get_ma(l_percent_r, self.long_smoothing_length, self.smooth_type)

        # Create the output DataFrame
        df_output = pd.DataFrame({
            'fast_r': s_percent_r,
            'slow_r': l_percent_r
        }, index=df_copy.index) # Use the index of the copied dataframe

        df_output['r_diff'] = df_output['fast_r'] - df_output['slow_r']

        return df_output