import pandas as pd
import numpy as np
from numba import njit
from wrappers.time_it import timeit

@njit
def numba_jma_calculation(src: np.array, length: int, phase: int, volty_10: np.array, temp_avg: np.array) -> np.array:
    """
    Calculates the Jurik Moving Average using Numba compiled Numpy for speed.

    Arguments:
    src (np.array): A numpy array of the dataframe's 'Close' values.
    length (int): The length used for internal JMA calculations.
    phase (int): The phase for the JMA, controlling overshoot and smoothness.
    volty_10 (np.array): Used for internal JMA calculations.
    temp_avg (np.array): Used for internal JMA calculations.

    Returns: A numpy array of the JMA values.
    """
    length = max(length, 1)
    len1 = max(np.log(np.sqrt(0.5 * (length - 1))) / np.log(2.0) + 2.0, 0)
    pow1 = max(len1 - 2.0, 0.5)
    div = 1.0 / (10.0 + 10.0 * (min(max(length-10, 0), 100)) / 100)
    avg_len = 65
    len2 = np.sqrt(0.5 * (length - 1)) * len1
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    phase_ratio = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100 + 1.5

    e0 = 0.0
    e1 = 0.0
    e2 = 0.0
    jma1 = 0.0

    bsmax_prev = src[0]
    bsmin_prev = src[0]
    avolty_prev = 0
    vsum_prev = 0
    jma_result = np.zeros(len(src))

    for i in range(len(src)):
        current_src = src[i]
        current_volty_10 = volty_10[i]
        current_temp_avg = temp_avg[i]

        del1 = current_src - bsmax_prev
        del2 = current_src - bsmin_prev

        volty = abs(del1) if abs(del1) > abs(del2) else abs(del2)
        vsum = vsum_prev + div * (volty - current_volty_10)

        if i <= avg_len:
            avolty = avolty_prev + 2.0 * (vsum - avolty_prev) / (avg_len + 1)
        else:
            avolty = current_temp_avg

        dVolty = volty / avolty if avolty > 0 else 0
        dVolty = min(max(dVolty, 1), pow(len1, 1.0/pow1))
        pow2 = pow(dVolty, pow1)

        Kv = pow(len2 / (len2 + 1), np.sqrt(pow2))

        bsmax = current_src if del1 > 0 else current_src - Kv * del1
        bsmin = current_src if del2 < 0 else current_src - Kv * del2

        alpha = beta**pow2

        e0 = (1 - alpha) * current_src + alpha * e0
        e1 = (current_src - e0) * (1 - beta) + beta * e1
        e2 = (e0 + phase_ratio * e1 - jma1) * pow(1 - alpha, 2) + pow(alpha, 2) * e2
        jma1 = e2 + jma1

        bsmin_prev = bsmin
        bsmax_prev = bsmax
        avolty_prev = avolty
        vsum_prev = vsum

        jma_result[i] = jma1

    return jma_result

@timeit
def calculate_jurik_filter(df: pd.DataFrame, timeframe: int, src_column='Close', length=15, phase=0, double_smooth=False) -> pd.DataFrame:
    """
    Calculates the slope of the Jurik Moving Average (JMA).

    Arguments:
    df (pd.DataFrame): Input DataFrame containing the source data.
    timeframe (int): The timeframe for the JMA calculation (used for column naming).
    src_column (str): The name of the column to use as the source price.
    len_param (int): The length parameter for the JMA calculation.
    phase (int): The phase parameter for the JMA, controlling overshoot and smoothness.
    double_smooth (bool): If True, applies the JMA calculation a second time.

    Returns:
    pd.Dataframe: The original Dataframe with the JMA slope values added in.
    """
    avg_len = 65
    src = df[src_column]
    volty_10 = src.diff().abs().rolling(10).sum().fillna(0.0)
    temp_avg = volty_10.rolling(avg_len).mean().fillna(0.0)

    src_np = src.to_numpy()
    volty_10_np = volty_10.to_numpy()
    temp_avg_np = temp_avg.to_numpy()

    jma_result = numba_jma_calculation(src_np, length, phase, volty_10_np, temp_avg_np)

    if double_smooth:
        jma_result = numba_jma_calculation(jma_result, length, phase, volty_10_np, temp_avg_np)

    jma_series = pd.Series(jma_result, index=src.index)

    # We shall calculate the slope of the MA to ensure all of our indicators are differenced
    jma_slope = jma_series.diff()
    jma_slope_col_name = f"jma_slope_{timeframe}"
    df[jma_slope_col_name] = jma_slope

    return df

