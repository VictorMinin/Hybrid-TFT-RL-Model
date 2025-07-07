import pandas as pd
import numpy as np
from wrappers.time_it import timeit

class LinearRegressionSlopeCalculator:
    """
    A class to calculate the Linear Regression Slope
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the class with the dataframe.

        Args:
            df (pd.DataFrame): A DataFrame containing at least a 'Close' column.
                               The index should be a datetime index.
        """
        if 'Close' not in df.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column.")
        self.df = df.copy()

    @timeit
    def calculate(self, timeframe: int, period: int = 20):
        """
        Calculates the linear regression slope over a rolling window.

        Args:
            period (int): The lookback period for the linear regression calculation.
            timeframe (int): The timeframe for which the indicator is being calculated.

        Returns:
            pd.DataFrame: The original DataFrame with an added 'lr_slope' column.

        """
        # Initialize the new column with NaN
        self.df[f'lr_slope_{timeframe}'] = np.nan

        # Create the independent variable (time sequence)
        x = np.arange(period)

        # Loop through the dataframe to calculate the rolling regression slope
        for i in range(period - 1, len(self.df)):
            # Get the data for the current window
            window_slice = self.df.iloc[i - period + 1 : i + 1]

            # Use 'Close' price for the regression calculation
            y = window_slice['Close'].values

            # --- Perform Linear Regression to get the slope ---
            # np.polyfit returns [slope, intercept]
            slope, _ = np.polyfit(x, y, 1)

            # --- Store the calculated slope ---
            current_index = self.df.index[i]
            self.df.loc[current_index, f'lr_slope_{timeframe}'] = slope

        return self.df