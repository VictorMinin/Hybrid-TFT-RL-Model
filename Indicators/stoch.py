import pandas as pd
from wrappers.time_it import timeit

@timeit
def calculate_stochastic_oscillator(df: pd.DataFrame, timeframe: int, k_period=14, d_period=3) -> pd.DataFrame:
    """
    Calculate the stochastic oscillator values %K and %D.

    Parameters:
    df (pd.DataFrame): DataFrame with columns ['High', 'Low', 'Close']
    timeframe (int): The timeframe of df used for naming of new column
    k_period (int): The period for %K calculation (default is 14)
    d_period (int): The period for %D calculation (default is 3)

    Returns:
    pd.DataFrame: DataFrame with added columns ['%K_tf', '%D_tf']
    """
    df = df.copy()

    df['Lowest Low'] = df['Low'].rolling(window=k_period, min_periods=1).min()
    df['Highest High'] = df['High'].rolling(window=k_period, min_periods=1).max()

    df[f'%K{timeframe}'] = 100 * ((df['Close'] - df['Lowest Low']) / (df['Highest High'] - df['Lowest Low']))
    df[f'%D{timeframe}'] = df[f'%K{timeframe}'].rolling(window=d_period, min_periods=1).mean()
    # ADD IN SLOPE OF %D POSSIBLY and need to add in k below or above d column
    df.drop(columns=['Lowest Low', 'Highest High', f'%K{timeframe}'], inplace=True)

    return df