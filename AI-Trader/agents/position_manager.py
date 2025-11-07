import pandas as pd
from typing import Dict, Optional

class PositionManager:
    """
    仓位管理器，负责设置止盈止损和动态提损。
    """
    
    def __init__(self):
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.price_map = {}
    
    def set_take_profit_and_stop_loss(self, df: pd.DataFrame, high: float, low: float) -> Dict[str, float]:
        """
        设置止盈止损：
        - 止盈：-0.2对应的价格
        - 区域止损：根据回调最低点到达的区间设置止损
        """
        # 创建价格映射表
        self.price_map = {}
        for i in range(11):
            key = round(i / 10, 1)
            value = low + (high - low) * (1 - key)
            self.price_map[key] = value
        
        # 设置止盈价
        self.take_profit_price = self.price_map[0.2]
        
        # 设置区域止损
        current_low = df['low'].min()
        if current_low <= self.price_map[0.4]:
            self.stop_loss_price = self.price_map[0.6]
        elif current_low <= self.price_map[0.5]:
            self.stop_loss_price = self.price_map[0.7]
        elif current_low <= self.price_map[0.6]:
            self.stop_loss_price = self.price_map[0.8]
        else:
            self.stop_loss_price = self.price_map[0.9]
        
        return {
            'take_profit_price': self.take_profit_price,
            'stop_loss_price': self.stop_loss_price
        }
    
    def update_stop_loss(self, current_price: float) -> None:
        """
        动态提损：
        - 价格上涨2份：止损设为回调最低点
        - 价格达0.1：止损设为成本价
        - 价格达0：止损设为0.1对应价格
        """
        if self.entry_price is None:
            return
        
        price_diff = current_price - self.entry_price
        price_units = price_diff / (self.price_map[1.0] - self.price_map[0.0])
        
        if price_units >= 2:
            # 价格上涨了2份，止损设为回调最低点
            self.stop_loss_price = self.price_map[1.0]
        elif price_units >= 0.1:
            # 价格达0.1，止损设为成本价
            self.stop_loss_price = self.entry_price
        elif price_units >= 0:
            # 价格达0，止损设为0.1对应价格
            self.stop_loss_price = self.price_map[0.1]