import pandas as pd
from typing import Dict, List, Optional
import talib

class TradeSignalDetector:
    """
    开仓信号检测器，在小周期中寻找符合大周期信号的开仓机会。
    支持多周期联动分析。
    """
    
    def __init__(self, periods: List[str] = ['4h', '1h', '30m']):
        self.periods = periods
    
    def find_entry_signal(self, large_df: pd.DataFrame, small_dfs: Dict[str, pd.DataFrame]) -> Optional[str]:
        """
        在小周期中寻找开仓信号：
        - MACD从0轴下上升到0轴上
        - 回调过程中MACD向0轴下走
        - MACD接近或形成金叉
        - 0轴下峰值 < 0轴上峰值
        """
        for period in self.periods:
            df = small_dfs.get(period)
            if df is None or len(df) < 10:
                continue
            
            # 计算MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # 检查MACD是否从0轴下上升到0轴上
            below_zero = df['macd'] < 0
            above_zero = df['macd'] > 0
            crossed_up = (df['macd'].diff() > 0) & (df['macd'] < df['macd'].shift(1))
            
            # 检查回调过程中MACD向0轴下走
            recent_downward = (df['macd'].diff() < 0).tail(5).any()
            
            # 检查是否接近或形成金叉
            golden_cross = (df['macd'].diff() > 0) & (df['macd'] < df['macd_signal'].shift(1))
            near_golden_cross = golden_cross.tail(5).any()
            
            # 检查0轴下峰值 < 0轴上峰值
            below_peak = df[below_zero]['macd'].max()
            above_peak = df[above_zero]['macd'].max()
            peak_condition = below_peak < above_peak
            
            if (crossed_up.any() and recent_downward and near_golden_cross and peak_condition):
                return period
        
        return None