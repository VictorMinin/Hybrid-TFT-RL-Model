import pandas as pd
import numpy as np
from wrappers.time_it import timeit

class JurikDMX:
    """
    A Python implementation of the Jurik DMX Histogram [Loxx] PineScript indicator.

    This class takes IN a pandas DataFrame with OHLC data and calculates the
    Jurik DMX signal line.

    It returns a dataframe with the calculated Jurik DMX signal line.
    """
    def __init__(self, ohlc_df: pd.DataFrame):
        # Ensure the input is a DataFrame
        if not isinstance(ohlc_df, pd.DataFrame):
            raise TypeError("ohlc_df must be a pandas DataFrame.")

        self.df = ohlc_df.copy()

        required_cols = {'Open', 'High', 'Low', 'Close'}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_cols}")

        # Initialize internal state variables used for iterative calculations
        self._initialize_state()

    def _initialize_state(self):
        """Initializes or resets state variables for calculations."""
        self._hp_prev = 0.0
        self._src_prev_hp = 0.0
        self._out_prev1_ess = 0.0
        self._out_prev2_ess = 0.0
        self._src_prev1_ess = 0.0
        self._src_prev2_ess = 0.0
        self._jma_state = {}

    def high_pass_filter(self, src, max_len):
        """High-pass filter component."""
        c = 360 * np.pi / 180
        alpha = (1 - np.sin(c / max_len)) / np.cos(c / max_len) if max_len > 0 else 0
        hp = 0.5 * (1 + alpha) * (src - self._src_prev_hp) + alpha * self._hp_prev
        self._src_prev_hp = src
        self._hp_prev = hp
        return hp

    def ehlers_ss_filter(self, src, length):
        """Ehlers Super Smoother Filter component."""
        s = 1.414
        a = np.exp(-s * np.pi / length) if length > 0 else 0
        b = 2 * a * np.cos(s * np.pi / length) if length > 0 else 0
        c2 = b
        c3 = -a * a
        c1 = 1 - c2 - c3
        out = (c1 * (src + self._src_prev1_ess) / 2 +
               c2 * self._out_prev1_ess +
               c3 * self._out_prev2_ess)
        self._src_prev2_ess = self._src_prev1_ess
        self._src_prev1_ess = src
        self._out_prev2_ess = self._out_prev1_ess
        self._out_prev1_ess = out if not np.isnan(out) else (self._src_prev1_ess if not np.isnan(self._src_prev1_ess) else src)
        return out

    def jurik_filter(self, src, length, phase, key):
        """Jurik Moving Average (JMA) implementation."""
        if key not in self._jma_state:
            self._jma_state[key] = {
                'bsmax': src, 'bsmin': src, 'vsum': 0.0,
                'avolty': 0.0, 'volty_hist': np.zeros(10), 'e0': src,
                'e1': 0.0, 'e2': 0.0, 'jma1': src, 'bar_index': 0,
                'vsum_hist': np.zeros(65)
            }
        st = self._jma_state[key]
        len1 = max(np.log(np.sqrt(0.5 * (length - 1))) / np.log(2.0) + 2.0, 0) if length > 1 else 0
        pow1 = max(len1 - 2.0, 0.5)
        del1 = src - st['bsmax']
        del2 = src - st['bsmin']
        volty = abs(del1) if abs(del1) > abs(del2) else abs(del2)
        st['volty_hist'] = np.roll(st['volty_hist'], 1)
        st['volty_hist'][0] = volty
        div = 1.0 / (10.0 + 10.0 * (min(max(length - 10, 0), 100)) / 100)
        st['vsum'] += div * (volty - st['volty_hist'][-1])
        avg_len = 65
        st['vsum_hist'] = np.roll(st['vsum_hist'], 1)
        st['vsum_hist'][0] = st['vsum']
        if st['bar_index'] <= avg_len:
            st['avolty'] += 2.0 * (st['vsum'] - st['avolty']) / (avg_len + 1)
        else:
            st['avolty'] = np.mean(st['vsum_hist'])
        st['bar_index'] += 1
        d_volty = volty / st['avolty'] if st['avolty'] > 0 else 0
        d_volty = min(d_volty, pow(len1, 1.0 / pow1)) if pow1 != 0 else min(d_volty, 1e9)
        d_volty = max(d_volty, 1)
        pow2 = d_volty ** pow1
        len2 = np.sqrt(0.5 * (length - 1)) * len1 if length > 1 else 0
        kv = pow(len2 / (len2 + 1), np.sqrt(pow2)) if len2 > 0 else 0
        st['bsmax'] = src if del1 > 0 else src - kv * del1
        st['bsmin'] = src if del2 < 0 else src - kv * del2
        phase_ratio = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100 + 1.5)
        beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2) if length > 1 else 0
        alpha = beta ** pow2
        st['e0'] = (1 - alpha) * src + alpha * st['e0']
        st['e1'] = (src - st['e0']) * (1 - beta) + beta * st['e1']
        st['e2'] = (st['e0'] + phase_ratio * st['e1'] - st['jma1']) * ((1 - alpha) ** 2) + (alpha ** 2) * st['e2']
        st['jma1'] += st['e2']
        return st['jma1']

    @timeit
    def calculate(self, length=32, phase=0, sig_len=5):
        self._initialize_state()
        high_prev = self.df['High'].shift(1).fillna(self.df['High'])
        low_prev = self.df['Low'].shift(1).fillna(self.df['Low'])
        close_prev = self.df['Close'].shift(1).fillna(self.df['Close'])
        up = self.df['High'] - high_prev
        down = low_prev - self.df['Low']
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        tr = np.maximum(self.df['High'] - self.df['Low'],
                        np.maximum(abs(self.df['High'] - close_prev),
                                   abs(self.df['Low'] - close_prev)))
        signal_series = []
        for i in range(len(self.df)):
            current_tr = tr.iloc[i]
            current_plus_dm = plus_dm[i]
            current_minus_dm = minus_dm[i]
            out_len = length
            trur = self.jurik_filter(current_tr, out_len, phase, key='trur_jma')
            plus_filt = self.jurik_filter(current_plus_dm, out_len, phase, key='plus_jma')
            minus_filt = self.jurik_filter(current_minus_dm, out_len, phase, key='minus_jma')
            plus = 100 * plus_filt / trur if trur != 0 else 0
            minus = 100 * minus_filt / trur if trur != 0 else 0
            trigger = plus - minus
            signal = self.jurik_filter(trigger, sig_len, phase, key='signal_jma')
            signal_series.append(signal)
        result_df = pd.DataFrame(index=self.df.index)
        result_df['signal'] = signal_series
        self.result_df = result_df.copy()
        return result_df
