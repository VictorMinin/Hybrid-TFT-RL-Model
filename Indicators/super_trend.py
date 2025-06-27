import pandas as pd

def calculate_supertrend(df_ohlc: pd.DataFrame, atr_period: int = 10, atr_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Calculates the Supertrend indicator, providing separate columns for uptrends and downtrends.

    Args:
        df_ohlc (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', and 'Close' columns.
        atr_period (int): The period for the ATR calculation. Default is 10.
        atr_multiplier (float): The multiplier for the ATR value. Default is 3.0.

    Returns:
        pd.DataFrame: The original DataFrame with added 'Supertrend_Up' and 'Supertrend_Down' columns.
    """
    if not all(col in df_ohlc.columns for col in ['High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns.")

    df = df_ohlc.copy()

    # Calculate True Range (TR)
    high_low = df['High'] - df['Low']
    high_close_prev = abs(df['High'] - df['Close'].shift(1))
    low_close_prev = abs(df['Low'] - df['Close'].shift(1))

    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

    # Calculate Average True Range (ATR)
    atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()

    # Calculate the basic upper and lower bands
    basic_up_band = (df['High'] + df['Low']) / 2 + (atr_multiplier * atr)
    basic_low_band = (df['High'] + df['Low']) / 2 - (atr_multiplier * atr)

    # Initialize final bands and trend arrays
    final_up_band = [0.0] * len(df)
    final_low_band = [0.0] * len(df)
    trend = [1] * len(df)  # Start with an uptrend assumption

    # --- Core Supertrend Logic ---
    for i in range(1, len(df)):
        # Determine the final bands based on the previous close
        if df['Close'][i-1] <= final_up_band[i-1]:
            final_up_band[i] = min(basic_up_band[i], final_up_band[i-1])
        else:
            final_up_band[i] = basic_up_band[i]

        if df['Close'][i-1] >= final_low_band[i-1]:
            final_low_band[i] = max(basic_low_band[i], final_low_band[i-1])
        else:
            final_low_band[i] = basic_low_band[i]

        # Determine the trend
        if trend[i-1] == 1 and df['Close'][i] < final_low_band[i]:
            trend[i] = -1
        elif trend[i-1] == -1 and df['Close'][i] > final_up_band[i]:
            trend[i] = 1
        else:
            trend[i] = trend[i-1]

    # Create the final Supertrend columns for plotting with -1 for flag as to which trend is invalid
    supertrend_up = [final_low_band[i] if trend[i] == 1 else -1 for i in range(len(df))]
    supertrend_down = [final_up_band[i] if trend[i] == -1 else -1 for i in range(len(df))]

    df['Supertrend_Up'] = supertrend_up
    df['Supertrend_Down'] = supertrend_down

    # The first value is always NaN
    df.at[df.index[0], 'Supertrend_Up'] = float('nan')
    df.at[df.index[0], 'Supertrend_Down'] = float('nan')

    return df