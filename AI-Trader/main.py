from agents.macd_signal_detector import MACDSignalDetector
from agents.trade_signal_detector import TradeSignalDetector
from agents.position_manager import PositionManager
from agents.data_loader import DataLoader

def run_trading_strategy(symbol: str, start_date: str, end_date: str):
    """
    主运行流程，整合各模块实现完整交易策略。
    """
    # 初始化数据加载器
    data_loader = DataLoader("data")
    
    # 加载日级数据
    daily_df = data_loader.load_data(symbol, '1d')
    daily_df = data_loader.resample_to_period(daily_df, '1d')
    
    # 加载小周期数据
    small_dfs = {}
    for period in ['4h', '1h', '30m']:
        df = data_loader.load_data(symbol, period)
        df = data_loader.resample_to_period(df, period)
        small_dfs[period] = df
    
    # 初始化信号检测器
    macd_detector = MACDSignalDetector()
    trade_detector = TradeSignalDetector()
    position_manager = PositionManager()
    
    # 逐日遍历
    for date in daily_df.index:
        # 获取当日数据
        daily_data = daily_df.loc[date]
        
        # 识别信号
        signal_result = macd_detector.detect_signal(daily_data)
        
        if not signal_result['macd_above_zero'] or not signal_result['has_golden_cross'] or \
           not signal_result['increasing_peaks'] or not signal_result['in_descending_phase']:
            continue
        
        # 持续监测，每5分钟检测一次变化
        # 这里简化为直接使用小周期数据进行检测
        entry_period = trade_detector.find_entry_signal(daily_data, small_dfs)
        
        if entry_period is not None:
            # 开仓
            entry_price = small_dfs[entry_period]['close'].iloc[-1]
            position_manager.entry_price = entry_price
            
            # 设置止盈止损
            high = daily_data['high']
            low = daily_data['low']
            stop_loss_info = position_manager.set_take_profit_and_stop_loss(daily_data, high, low)
            
            # 持续监测，动态提损
            while True:
                # 模拟持续监测
                current_price = small_dfs[entry_period]['close'].iloc[-1]
                position_manager.update_stop_loss(current_price)
                
                # 检查是否触发止盈或止损
                if current_price >= stop_loss_info['take_profit_price']:
                    print(f"止盈触发，价格: {current_price}")
                    break
                elif current_price <= stop_loss_info['stop_loss_price']:
                    print(f"止损触发，价格: {current_price}")
                    break
                
                # 模拟等待下一个5分钟
                import time
                time.sleep(300)