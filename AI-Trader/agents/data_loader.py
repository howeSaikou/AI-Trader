import pandas as pd
from typing import Dict, Optional

class DataLoader:
    """
    数据加载器，支持分钟级数据的加载与多周期重采样。
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_data(self, symbol: str, interval: str = '1min') -> pd.DataFrame:
        """
        加载指定股票的分钟级数据。
        """
        # 这里假设数据已经以CSV格式存储
        df = pd.read_csv(f"{self.data_path}/{symbol}_{interval}.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df
    
    def resample_to_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """
        将分钟级数据重采样到指定周期。
        """
        return df.resample(period).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })