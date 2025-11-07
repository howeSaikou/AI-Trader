import pandas as pd
import talib
from typing import Dict, Optional

class MACDSignalDetector:
    """
    MACD信号识别器，用于检测MACD在0轴上方、金叉、峰值递增等条件。
    支持多周期联动分析。
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def detect_signal(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        检测MACD是否满足信号条件：
        - MACD在0轴上方
        - 至少有一个金叉
        - 形成两个以上峰值且后一个高于前一个
        - 处于第二个峰值开始下降过程但未触0轴
        """
        # 计算MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], 
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )
        
        # 判断MACD是否在0轴上方
        macd_above_zero = (df['macd'] > 0).all()
        
        # 判断是否有金叉（MACD从下向上穿过信号线）
        cross_up = (df['macd'].diff() > 0) & (df['macd'] < df['macd_signal'].shift(1))
        has_golden_cross = cross_up.any()
        
        # 找到MACD峰值
        peaks = []
        for i in range(1, len(df)-1):
            if df['macd'].iloc[i] > df['macd'].iloc[i-1] and df['macd'].iloc[i] > df['macd'].iloc[i+1]:
                peaks.append(i)
        
        # 判断是否有至少两个峰值且递增
        if len(peaks) >= 2:
            peak_values = [df['macd'].iloc[p] for p in peaks]
            increasing_peaks = all(peak_values[i] < peak_values[i+1] for i in range(len(peak_values)-1))
        else:
            increasing_peaks = False
        
        # 判断是否处于第二个峰值开始下降过程但未触0轴
        if len(peaks) >= 2:
            second_peak_idx = peaks[1]
            after_second_peak = df.iloc[second_peak_idx:].copy()
            descending_after_peak = (after_second_peak['macd'].diff() < 0).all()
            not_touch_zero = (after_second_peak['macd'] > 0).all()
            in_descending_phase = descending_after_peak and not_touch_zero
        else:
            in_descending_phase = False
        
        return {
            'macd_above_zero': macd_above_zero,
            'has_golden_cross': has_golden_cross,
            'increasing_peaks': increasing_peaks,
            'in_descending_phase': in_descending_phase
        }