import pandas as pd

class Parabolic_SAR:
    """
     This class takes in a pandas DataFrame with OHLC data and calculates the
     Parabolic SAR values for the given indicator settings.

    It returns a dataframe with the calculated Parabolic SAR values.
    """
    def __init__(self, initial_af=0.02, max_af=0.2, step_af=0.02):
        self.initial_af = initial_af
        self.max_af = max_af
        self.step_af = step_af

    def calculate_psar(self, df: pd.DataFrame):
        """
    Calculates the Parabolic SAR for a given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'High', 'Low', and 'Close' columns.

    Returns:
        pd.DataFrame: The original DataFrame with an added 'PSAR' column.
    """
        # Make a copy to avoid modifying the original DataFrame
        df_psar = df.copy()

        # Initialize columns for PSAR calculation
        df_psar['psar'] = 0.0
        df_psar['trend'] = 0  # 1 for uptrend, -1 for downtrend
        df_psar['ep'] = 0.0   # Extreme Point
        df_psar['af'] = 0.0   # Acceleration Factor

        # Set initial values for the first row
        df_psar.at[0, 'trend'] = 1  # Assume initial uptrend
        df_psar.at[0, 'psar'] = df_psar.at[0, 'Low']
        df_psar.at[0, 'ep'] = df_psar.at[0, 'High']
        df_psar.at[0, 'af'] = self.initial_af

        # Iterate through the DataFrame starting from the second row
        for i in range(1, len(df_psar)):
            prev_psar = df_psar.at[i-1, 'psar']
            prev_trend = df_psar.at[i-1, 'trend']
            prev_ep = df_psar.at[i-1, 'ep']
            prev_af = df_psar.at[i-1, 'af']

            # Calculate the current PSAR
            current_psar = prev_psar + prev_af * (prev_ep - prev_psar)

            # --- Trend Reversal Check ---
            # If in an uptrend, but the current low is below the PSAR, switch to downtrend
            if prev_trend == 1 and df_psar.at[i, 'Low'] < current_psar:
                trend = -1
                psar = prev_ep  # New PSAR is the previous Extreme Point
                ep = df_psar.at[i, 'Low']
                af = self.initial_af
            # If in a downtrend, but the current high is above the PSAR, switch to uptrend
            elif prev_trend == -1 and df_psar.at[i, 'High'] > current_psar:
                trend = 1
                psar = prev_ep  # New PSAR is the previous Extreme Point
                ep = df_psar.at[i, 'High']
                af = self.initial_af
            # --- No Trend Reversal ---
            else:
                trend = prev_trend
                psar = current_psar
                af = prev_af
                # Update Extreme Point and Acceleration Factor
                if trend == 1: # Uptrend
                    if df_psar.at[i, 'High'] > prev_ep:
                        ep = df_psar.at[i, 'High']
                        af = min(self.max_af, af + self.step_af)
                    else:
                        ep = prev_ep
                else: # Downtrend
                    if df_psar.at[i, 'Low'] < prev_ep:
                        ep = df_psar.at[i, 'Low']
                        af = min(self.max_af, af + self.step_af)
                    else:
                        ep = prev_ep

            # Ensure PSAR is not beyond the current period's high/low
            if trend == 1:
                psar = min(psar, df_psar.at[i-1, 'Low'], df_psar.at[i, 'Low'])
            else:
                psar = max(psar, df_psar.at[i-1, 'High'], df_psar.at[i, 'High'])


            # Store calculated values
            df_psar.at[i, 'trend'] = trend
            df_psar.at[i, 'psar'] = psar
            df_psar.at[i, 'ep'] = ep
            df_psar.at[i, 'af'] = af

        return df_psar[['psar']]