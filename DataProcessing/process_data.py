import os
from typing import Dict
import pandas as pd
from Indicators.jurik_ma import calculate_jurik_filter
from Indicators.g_channel import GChannelIndicator
from Indicators.average_true_range import AverageTrueRange
from Indicators.stoch import calculate_stochastic_oscillator
from Indicators.rsi import calculate_rsi
from Indicators.r_exhaustion import RTrendExhaustion
from Indicators.jurik_dmx import calculate_dmx_numba
from DataProcessing.normalize_features import normalize_features
from wrappers.time_it import timeit

@timeit
def merge_timeframe_dfs(eurusd_dfs: Dict[str, pd.DataFrame], prediction_time_frame: int) -> pd.DataFrame:
    """
    Arguments:
        eurusd_dfs (Dict[str, pd.DataFrame]): A dictionary of dataframes of the timeframes 1Min, 15Min, 30Min, 60Min, 240Min
        prediction_time_frame (int): The target timeframe to align other DataFrames to

    Returns:
        df (pd.DataFrame): A single unified DataFrame aligned to the prediction_time_frame's index
    """
    # Reindex all DataFrames to match the prediction timeframe's index
    target_index = eurusd_dfs[f'EURUSD_{prediction_time_frame}'].index
    for key, df in eurusd_dfs.items():
        if key != f'EURUSD_{prediction_time_frame}':
            df = df.reindex(target_index, method='ffill')
            eurusd_dfs[key] = df
    eurusd_merged = pd.concat([df for df in eurusd_dfs.values()], axis=1)
    return eurusd_merged

@timeit
def truncate(eurusd_df: pd.DataFrame) -> pd.DataFrame:
    # Start at 2004-01-30 00:00:00+00:00
    # End at 2024-06-19 17:00:00+00:00
    start_date = "2004-01-30 00:00:00+00:00"
    end_date = "2024-06-19 17:00:00+00:00"

    eurusd_df_truncated = eurusd_df.truncate(before=start_date, after=end_date)
    eurusd_df_truncated = eurusd_df_truncated.fillna(eurusd_df_truncated.mean())

    return eurusd_df_truncated

@timeit
def process_data(prediction_time_frame: int) -> pd.DataFrame:
    """


    """

    timeframes = (1, 5, 15, 30, 60, 240)
    eurusd_dfs = {}
    raw_data_path = "C:/Users/victo/IdeaProjects/Hybrid-TFT-RL-Model-/data/raw"
    processed_data_path = "C:/Users/victo/IdeaProjects/Hybrid-TFT-RL-Model-/data/processed"
    os.chdir(raw_data_path)
    g_channel = GChannelIndicator()
    atr = AverageTrueRange()
    r_exhaustion = RTrendExhaustion()
    for tf in timeframes:
        df = pd.read_csv(f"EURUSD_{tf}MIN.csv")

        df['Time'] = pd.to_datetime(df['Time'], utc=True)  # Assume it's UTC
        # Convert from UTC to US/Eastern time (EST/EDT will be handled automatically)
        df['Time'] = df['Time'].dt.tz_convert('US/Eastern')
        df.set_index('Time', inplace=True)  # Set index for all DataFrames

        df = calculate_rsi(df, tf)
        df = g_channel.calculate_g_channel(df, tf)
        df = calculate_stochastic_oscillator(df, tf)
        df = atr.calculate_atr(df, tf)
        df = r_exhaustion.calculate(df,tf)
        df = calculate_dmx_numba(df, tf)
        df = calculate_jurik_filter(df, tf)
        df.drop(columns=['Open', 'High', 'Low'], inplace=True)
        if tf != prediction_time_frame:
            df.drop(columns=['Close'], inplace=True)
        eurusd_dfs[f"EURUSD_{tf}"] = df

    eurusd_df = merge_timeframe_dfs(eurusd_dfs, prediction_time_frame)
    os.chdir(processed_data_path)
    eurusd_df = truncate(eurusd_df)
    eurusd_df.to_csv("X.csv")
    normalized_eurusd_df = normalize_features(eurusd_df)
    normalized_eurusd_df.to_csv("X_normalized.csv")

    return eurusd_df

process_data(60)