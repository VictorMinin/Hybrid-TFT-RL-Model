import pandas as pd
from wrappers.time_it import timeit

@timeit
def calculate_stochastic_oscillator(df: pd.DataFrame, timeframe: int, k_period=12, d_period=3, smooth_k=3) -> pd.DataFrame:
    """
    Calculate the stochastic oscillator values %K and %D.

    Parameters:
    df (pd.DataFrame): DataFrame with columns ['High', 'Low', 'Close']
    timeframe (int): The timeframe of df used for naming of new column
    k_period (int): The period for %K calculation (default is 12)
    d_period (int): The period for %D calculation (default is 3)
    smooth_k (int): The period for smoothing k (default is 3)

    Returns:
    pd.DataFrame: DataFrame with added columns ['%K_tf', '%D_tf', 'K_above_D_tf']
    """
    df = df.copy()

    df['Lowest Low'] = df['Low'].rolling(window=k_period, min_periods=1).min()
    df['Highest High'] = df['High'].rolling(window=k_period, min_periods=1).max()

    denominator = (df['Highest High'] - df['Lowest Low'])
    raw_k_calc = 100 * ((df['Close'] - df['Lowest Low']) / denominator)
    df['Raw %K'] = raw_k_calc.fillna(0) # Handle cases where denominator is 0/ NaN

    # Smooth the Raw %K to get the Slow %K
    k_col_name = f'%K_{timeframe}'
    df[k_col_name] = df['Raw %K'].rolling(window=smooth_k, min_periods=1).mean()

    # Smooth the Slow %K to get the Slow %D
    d_col_name = f'%D_{timeframe}'
    df[d_col_name] = df[k_col_name].rolling(window=d_period, min_periods=1).mean()

    # Add the column indicating whether %K is above or below %D
    # 1 for K above D, 0 for K below or equal to D
    k_above_d_col_name = f'K_above_D_{timeframe}'
    df[k_above_d_col_name] = (df[k_col_name] > df[d_col_name]).astype(int)

    # Drop intermediate columns
    df.drop(columns=['Lowest Low', 'Highest High', 'Raw %K'], inplace=True)

    return df