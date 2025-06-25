import pandas as pd
import numpy as np
from wrappers.time_it import  timeit

class GChannelIndicator:
    """
    This class takes a pandas DataFrame with OHLC data and calculates the
    G-Channel values (Upper, Lower, Middle bands). These values will then be
    converted into G_Width, Upper_Slope, Middle_Slope, and Lower_Slope values.

    The G-Channel indicator is a recently invented indicator, this implementation
    is based on the original research paper:
    https://mpra.ub.uni-muenchen.de/95806/1/MPRA_paper_95806.pdf.

    It returns a DataFrame with these additional G_Width, Upper_Slope, Middle_Slope, and
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
    def calculate(self, ohlc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the G-Channel indicator values.

        Args:
            ohlc_df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns.


        Returns:
            pd.DataFrame: The DataFrame with newly calculated indicator columns.
        """
        # Ensure the required 'Close' column exists
        if 'Close' not in ohlc_df.columns:
            raise ValueError("DataFrame must contain a 'Close' column.")

        df = ohlc_df.copy()

        # Initialize calculation buffer columns with NaN
        df['UpperBuffer']  = np.nan
        df['LowerBuffer']  = np.nan

        # Get integer locations for columns
        close_loc = df.columns.get_loc('Close')
        upper_loc = df.columns.get_loc('UpperBuffer')
        lower_loc = df.columns.get_loc('LowerBuffer')

        # Iteratively calculate the G-Channel using .iloc
        for i in range(len(df)):
            src = df.iloc[i, close_loc]

            if i == 0:
                # Initialize first bar values
                df.iloc[i, upper_loc] = src
                df.iloc[i, lower_loc] = src
            else:
                # Get previous buffer values
                prev_a_buffer = df.iloc[i-1, upper_loc]
                prev_b_buffer = df.iloc[i-1, lower_loc]

                # Calculate current aBuffer (Upper)
                if src > prev_a_buffer:
                    a_buffer = src
                else:
                    if self.channel_period > 0:
                        a_buffer = prev_a_buffer - (prev_a_buffer - prev_b_buffer) / self.channel_period
                    else:
                        a_buffer = prev_a_buffer

                # Calculate current bBuffer (Lower)
                if src < prev_b_buffer:
                    b_buffer = src
                else:
                    if self.channel_period > 0:
                        b_buffer = prev_b_buffer + (prev_a_buffer - prev_b_buffer) / self.channel_period
                    else:
                        b_buffer = prev_b_buffer

                df.iloc[i, upper_loc] = a_buffer
                df.iloc[i, lower_loc] = b_buffer

        # Calculate the Middle Buffer
        df['MiddleBuffer'] = (df['UpperBuffer'] + df['LowerBuffer']) / 2

        # Calculate G_Width
        df['G_Width'] = df['UpperBuffer'] - df['LowerBuffer']

        # Calculate the slopes (1 bar diff)
        df['Upper_Slope'] = df['UpperBuffer'].diff()
        df['Middle_Slope'] = df['MiddleBuffer'].diff()
        df['Lower_Slope'] = df['LowerBuffer'].diff()

        # Drop the unnecessary columns (due to differencing of timeseries, we don't want undifferenced vales)
        df = df.drop(columns=['UpperBuffer', 'LowerBuffer', 'MiddleBuffer'])

        return df