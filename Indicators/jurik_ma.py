import pandas as pd
import numpy as np
from wrappers.time_it import timeit

@timeit
def calculate_jurik_filter(df: pd.DataFrame, timeframe: int, src_column='Close', len_param=15, phase=0, filter_param=0, double_smooth=False) -> pd.Series:
    """
    Calculates the slope of the Jurik Moving Average (JMA).

    Arguments:
    df (pd.DataFrame): Input DataFrame containing the source data.
    timeframe (int): The timeframe for the JMA calculation (used for column naming).
    src_column (str): The name of the column to use as the source price.
    len_param (int): The length parameter for the JMA calculation.
    phase (int): The phase parameter for the JMA, controlling overshoot and smoothness.
    filter_param (float): If > 0, applies a price filter based on standard deviation.
    double_smooth (bool): If True, applies the JMA calculation a second time.

    Returns:
    pd.Series: A series of the JMA slope values.
    """
    def jurik_ma(src, length, phase):
        src = src.copy()
        length = max(length, 1)
        len1 = max(np.log(np.sqrt(0.5 * (length - 1))) / np.log(2.0) + 2.0, 0)
        pow1 = max(len1 - 2.0, 0.5)

        def calc_jma(row):
            nonlocal e0, e1, e2, jma1

            del1 = row['src'] - row['bsmax']
            del2 = row['src'] - row['bsmin']

            volty = abs(del1) if abs(del1) > abs(del2) else abs(del2)
            vsum = row['vsum'] + div * (volty - row['volty_10'])

            if row.name <= avgLen:
                avolty = row['avolty'] + 2.0 * (vsum - row['avolty']) / (avgLen + 1)
            else:
                avolty = row['temp_avg']

            dVolty = volty / avolty if avolty > 0 else 0
            dVolty = min(max(dVolty, 1), pow(len1, 1.0/pow1))
            pow2 = pow(dVolty, pow1)

            Kv = pow(len2 / (len2 + 1), np.sqrt(pow2))

            bsmax = row['src'] if del1 > 0 else row['src'] - Kv * del1
            bsmin = row['src'] if del2 < 0 else row['src'] - Kv * del2

            alpha = pow(beta, pow2)

            e0 = (1 - alpha) * row['src'] + alpha * e0
            e1 = (row['src'] - e0) * (1 - beta) + beta * e1
            e2 = (e0 + phaseRatio * e1 - jma1) * pow(1 - alpha, 2) + pow(alpha, 2) * e2
            jma1 = e2 + jma1

            return pd.Series({
                'jma': jma1,
                'bsmax': bsmax,
                'bsmin': bsmin,
                'vsum': vsum,
                'avolty': avolty
            })

        div = 1.0 / (10.0 + 10.0 * (min(max(length-10, 0), 100)) / 100)
        avgLen = 65
        len2 = np.sqrt(0.5 * (length - 1)) * len1
        beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
        phaseRatio = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100 + 1.5

        e0 = e1 = e2 = jma1 = 0.0

        src = pd.DataFrame({'src': src})
        src['volty_10'] = src['src'].diff().abs().rolling(10).sum()
        src['temp_avg'] = src['volty_10'].rolling(avgLen).mean()
        src['vsum'] = src['volty_10'].cumsum()
        src['avolty'] = 0
        src['bsmax'] = src['bsmin'] = src['src']

        result = src.apply(calc_jma, axis=1)
        return result['jma']

    def filter_price(price, length, filter_val):
        filtdev = filter_val * price.rolling(length).std()
        return price.where(abs(price - price.shift()) >= filtdev, price.shift())

    src = df[src_column]

    if filter_param > 0:
        src = filter_price(src, len_param, filter_param)

    jma = jurik_ma(src, len_param, phase)

    if double_smooth:
        jma = jurik_ma(jma, len_param, phase)

    if filter_param > 0:
        jma = filter_price(jma, len_param, filter_param)

    # We shall calculate the slope of the MA to ensure all of our indicators are differenced
    jma_slope = jma.diff()
    jma_slope.name = f"jma_slope_{timeframe}"

    return jma_slope