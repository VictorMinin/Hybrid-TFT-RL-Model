import pandas as pd
import numpy as np
from numba import njit
from wrappers.time_it import timeit # Assuming you have this decorator

@timeit
def calculate_dmx_numba(ohlc_df: pd.DataFrame, timeframe: int, length=32, phase=0, sig_len=5) -> pd.DataFrame:
    """
    Calculates the Jurik DMX signal line using a Numba-optimized kernel.

    Args:
        ohlc_df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns.
        timeframe (int): The timeframe for the column name in the result.
        length (int): The length for the main DMX calculation.
        phase (int): The phase for the JMA filters.
        sig_len (int): The length for the final signal line smoothing.

    Returns:
        pd.DataFrame: The original DataFrame with the DMX signal line column added.
    """
    required_cols = {'Open', 'High', 'Low', 'Close'}
    if not required_cols.issubset(ohlc_df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_cols}")

    high = ohlc_df['High'].to_numpy(dtype=np.float64)
    low = ohlc_df['Low'].to_numpy(dtype=np.float64)
    close = ohlc_df['Close'].to_numpy(dtype=np.float64)

    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]

    high_prev = np.roll(high, 1)
    high_prev[0] = high[0]

    low_prev = np.roll(low, 1)
    low_prev[0] = low[0]

    up = high - high_prev
    down = low_prev - low

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))

    signal_series = _jurik_dmx_kernel(
        tr,
        plus_dm,
        minus_dm,
        length,
        phase,
        sig_len
    )

    output_df = ohlc_df.copy()

    # Add the new signal line column to the copied DataFrame
    output_df[f'dmx_signal_{timeframe}'] = signal_series

    return output_df

@njit
def _jurik_dmx_kernel(tr: np.ndarray, plus_dm: np.ndarray, minus_dm: np.ndarray, length: int, phase: int, sig_len: int) -> np.ndarray:
    JMA_TRUR   = 0
    JMA_PLUS   = 1
    JMA_MINUS  = 2
    JMA_SIGNAL = 3
    NUM_CHANNELS = 4

    # State variables
    jma_e0 = np.zeros(NUM_CHANNELS, dtype=np.float64)
    jma_e1 = np.zeros(NUM_CHANNELS, dtype=np.float64)
    jma_e2 = np.zeros(NUM_CHANNELS, dtype=np.float64)
    jma_jma1 = np.zeros(NUM_CHANNELS, dtype=np.float64)
    jma_bsmax = np.zeros(NUM_CHANNELS, dtype=np.float64)
    jma_bsmin = np.zeros(NUM_CHANNELS, dtype=np.float64)
    jma_vsum = np.zeros(NUM_CHANNELS, dtype=np.float64)
    jma_avolty = np.zeros(NUM_CHANNELS, dtype=np.float64)
    jma_bar_index = np.zeros(NUM_CHANNELS, dtype=np.int64)
    jma_volty_hist = np.zeros((NUM_CHANNELS, 10), dtype=np.float64)
    jma_vsum_hist = np.zeros((NUM_CHANNELS, 65), dtype=np.float64)

    output_signal = np.empty(len(tr), dtype=np.float64)

    for i in range(len(tr)):
        current_tr = tr[i]
        current_plus_dm = plus_dm[i]
        current_minus_dm = minus_dm[i]

        trur = _jurik_filter_numba(current_tr, length, phase, JMA_TRUR, jma_e0, jma_e1, jma_e2, jma_jma1, jma_bsmax, jma_bsmin, jma_vsum, jma_avolty, jma_bar_index, jma_volty_hist, jma_vsum_hist)
        plus_filt = _jurik_filter_numba(current_plus_dm, length, phase, JMA_PLUS, jma_e0, jma_e1, jma_e2, jma_jma1, jma_bsmax, jma_bsmin, jma_vsum, jma_avolty, jma_bar_index, jma_volty_hist, jma_vsum_hist)
        minus_filt = _jurik_filter_numba(current_minus_dm, length, phase, JMA_MINUS, jma_e0, jma_e1, jma_e2, jma_jma1, jma_bsmax, jma_bsmin, jma_vsum, jma_avolty, jma_bar_index, jma_volty_hist, jma_vsum_hist)

        plus = 100.0 * plus_filt / trur if trur != 0 else 0.0
        minus = 100.0 * minus_filt / trur if trur != 0 else 0.0
        trigger = plus - minus

        signal = _jurik_filter_numba(trigger, sig_len, phase, JMA_SIGNAL, jma_e0, jma_e1, jma_e2, jma_jma1, jma_bsmax, jma_bsmin, jma_vsum, jma_avolty, jma_bar_index, jma_volty_hist, jma_vsum_hist)

        output_signal[i] = signal

    return output_signal


@njit
def _jurik_filter_numba(src, length, phase, channel_idx, jma_e0, jma_e1, jma_e2, jma_jma1, jma_bsmax, jma_bsmin, jma_vsum, jma_avolty, jma_bar_index, jma_volty_hist, jma_vsum_hist):
    # --- Step 1: Unpack state for the current channel ---
    bar_index = jma_bar_index[channel_idx]

    if bar_index == 0:
        bsmax = src
        bsmin = src
        e0 = src
        jma1 = src
    else:
        bsmax = jma_bsmax[channel_idx]
        bsmin = jma_bsmin[channel_idx]
        e0 = jma_e0[channel_idx]
        jma1 = jma_jma1[channel_idx]

    e1 = jma_e1[channel_idx]
    e2 = jma_e2[channel_idx]
    vsum = jma_vsum[channel_idx]
    avolty = jma_avolty[channel_idx]

    # --- Step 2: Core JMA calculation logic---
    len1 = max(np.log(np.sqrt(0.5 * (length - 1))) / np.log(2.0) + 2.0, 0) if length > 1 else 0
    pow1 = max(len1 - 2.0, 0.5)
    del1 = src - bsmax
    del2 = src - bsmin
    volty = abs(del1) if abs(del1) > abs(del2) else abs(del2)

    volty_hist_channel = jma_volty_hist[channel_idx]
    volty_hist_channel = np.roll(volty_hist_channel, 1)
    last_volty = volty_hist_channel[-1]
    volty_hist_channel[0] = volty
    jma_volty_hist[channel_idx] = volty_hist_channel

    div = 1.0 / (10.0 + 10.0 * (min(max(length - 10, 0), 100)) / 100)
    vsum = vsum + div * (volty - last_volty)

    avg_len = 65
    vsum_hist_channel = jma_vsum_hist[channel_idx]
    vsum_hist_channel = np.roll(vsum_hist_channel, 1)
    vsum_hist_channel[0] = vsum
    jma_vsum_hist[channel_idx] = vsum_hist_channel

    if bar_index <= avg_len:
        avolty = avolty + 2.0 * (vsum - avolty) / (avg_len + 1)
    else:
        avolty = np.mean(vsum_hist_channel)

    d_volty = volty / avolty if avolty > 0 else 0
    d_volty = min(d_volty, pow(len1, 1.0 / pow1)) if pow1 != 0 else min(d_volty, 1e9)
    d_volty = max(d_volty, 1)

    pow2 = d_volty ** pow1
    len2 = np.sqrt(0.5 * (length - 1)) * len1 if length > 1 else 0
    kv = pow(len2 / (len2 + 1), np.sqrt(pow2)) if len2 > 0 else 0

    bsmax = src if del1 > 0 else src - kv * del1
    bsmin = src if del2 < 0 else src - kv * del2

    phase_ratio = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100 + 1.5)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2) if length > 1 else 0
    alpha = beta ** pow2

    e0 = (1 - alpha) * src + alpha * e0
    e1 = (src - e0) * (1 - beta) + beta * e1
    e2 = (e0 + phase_ratio * e1 - jma1) * ((1 - alpha) ** 2) + (alpha ** 2) * e2
    final_jma1 = jma1 + e2

    # --- Step 3: Updated state put into the main arrays ---
    jma_e0[channel_idx] = e0
    jma_e1[channel_idx] = e1
    jma_e2[channel_idx] = e2
    jma_jma1[channel_idx] = final_jma1
    jma_bsmax[channel_idx] = bsmax
    jma_bsmin[channel_idx] = bsmin
    jma_vsum[channel_idx] = vsum
    jma_avolty[channel_idx] = avolty
    jma_bar_index[channel_idx] = bar_index + 1

    return final_jma1