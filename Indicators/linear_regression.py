import pandas as pd
import numpy as np
from wrappers.time_it import timeit
import numba

@numba.jit(nopython=True)
def optimized_linreg_calculation(prices, output_array, period):
    sum_x = 0.0
    sum_x_squared = 0.0
    for n in range(period):
        sum_x += n
        sum_x_squared += n * n
    denominator = period * sum_x_squared - sum_x * sum_x

    if denominator == 0:
        return # Exit early if we divide by 0

    for i in range(period - 1, len(prices)):
        y_window = prices[i - period + 1 : i + 1]

        sum_y = 0.0
        sum_xy = 0.0

        for j in range(period):
            sum_y += y_window[j]
            sum_xy += j * y_window[j]

        #    Formula: N * Σ(xy) - Σx * Σy
        numerator = period * sum_xy - sum_x * sum_y
        slope = numerator / denominator
        output_array[i] = slope
    return output_array

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
        close_prices = self.df['Close'].to_numpy()
        output_slopes = np.full(len(self.df), np.nan)

        # Create the independent variable (time sequence)
        x = np.arange(period)
        output_slopes = optimized_linreg_calculation(close_prices, output_slopes, period)

        column_name = f'lr_slope_{timeframe}_length_{period}'
        self.df[column_name] = output_slopes

        return self.df