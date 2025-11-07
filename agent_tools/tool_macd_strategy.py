import os
import sys
from typing import Dict, List, Any
import pandas as pd
import talib
import numpy as np
from fastmcp import FastMCP

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tools.general_tools import get_config_value
from tools.price_tools import get_open_prices

mcp = FastMCP("MACDStrategyTools")

# 支持的所有时间周期
ALL_TIMEFRAMES = ["1d", "4h", "1h", "30m", "15m", "5m", "1m"]

def identify_uptrends(price_data: pd.DataFrame) -> List[Dict]:
    """
    识别上涨趋势波段
    
    Args:
        price_data: 包含时间、开盘价、最高价、最低价、收盘价的DataFrame
    
    Returns:
        List[Dict]: 上涨趋势波段列表，每个元素包含起止时间、最高价、最低价等信息
    """
    if len(price_data) < 10:  # 数据不足，无法识别趋势
        return []
    
    # 计算MACD指标
    macd, macd_signal, macd_hist = talib.MACD(
        price_data['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    # 计算价格变化率
    price_changes = price_data['close'].pct_change()
    
    # 简化处理：通过价格变化和MACD识别上涨趋势
    uptrends = []
    in_uptrend = False
    start_idx = 0
    
    for i in range(1, len(price_data)):
        # 判断是否进入上涨趋势
        if not in_uptrend and (
            (price_changes.iloc[i] > 0.005 and macd.iloc[i] > 0) or  # 价格上涨且MACD为正
            (macd.iloc[i] > macd_signal.iloc[i] and macd.iloc[i-1] <= macd_signal.iloc[i-1])  # MACD金叉
        ):
            in_uptrend = True
            start_idx = i
            
        # 判断是否结束上涨趋势
        elif in_uptrend and (
            (price_changes.iloc[i] < -0.005 and macd.iloc[i] < 0) or  # 价格下跌且MACD为负
            (macd.iloc[i] < macd_signal.iloc[i] and macd.iloc[i-1] >= macd_signal.iloc[i-1])  # MACD死叉
        ):
            in_uptrend = False
            # 记录一个上涨趋势波段
            if i - start_idx > 2:  # 至少持续3个周期
                uptrend_data = price_data.iloc[start_idx:i+1]
                uptrends.append({
                    'start_time': uptrend_data.index[0],
                    'end_time': uptrend_data.index[-1],
                    'low': uptrend_data['low'].min(),
                    'high': uptrend_data['high'].max(),
                    'start_idx': start_idx,
                    'end_idx': i
                })
    
    # 如果仍在上涨趋势中，也记录下来
    if in_uptrend and len(price_data) - start_idx > 2:
        uptrend_data = price_data.iloc[start_idx:]
        uptrends.append({
            'start_time': uptrend_data.index[0],
            'end_time': uptrend_data.index[-1],
            'low': uptrend_data['low'].min(),
            'high': uptrend_data['high'].max(),
            'start_idx': start_idx,
            'end_idx': len(price_data) - 1
        })
    
    return uptrends

def get_price_data(symbol: str, timeframe: str, periods: int = 100) -> pd.DataFrame:
    """
    获取指定股票和时间周期的价格数据（模拟实现）
    
    Args:
        symbol: 股票代码
        timeframe: 时间周期 ('1d', '4h', '1h', '30m', '15m', '5m', '1m')
        periods: 获取的数据周期数
        
    Returns:
        pd.DataFrame: 包含时间、开盘价、最高价、最低价、收盘价的DataFrame
    """
    # 这里应该从实际数据源获取数据
    # 为简化起见，我们生成模拟数据
    # 实际实现中需要从本地数据文件或API获取真实数据
    
    # 生成模拟数据
    np.random.seed(42)  # 固定随机种子以保证结果一致
    dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq='1min')
    prices = 100 + np.cumsum(np.random.randn(periods) * 0.1)  # 随机游走生成价格
    
    # 确保高低开收价格合理
    opens = prices[:-1]  # 开盘价
    closes = prices[1:]  # 收盘价
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(len(opens)))  # 最高价
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(len(opens)))  # 最低价
    
    # 构造DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes
    }, index=dates[:-1])
    
    # 根据时间周期进行重采样
    if timeframe != '1m':
        # 简化处理：实际应该根据timeframe进行正确的重采样
        if timeframe == '5m':
            df = df.resample('5T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
        elif timeframe == '15m':
            df = df.resample('15T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
        elif timeframe == '30m':
            df = df.resample('30T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
        elif timeframe == '1h':
            df = df.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
        elif timeframe == '4h':
            df = df.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
        elif timeframe == '1d':
            df = df.resample('1D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
    
    return df

@mcp.tool()
def detect_macd_signal(symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
    """
    检测MACD信号
    
    Args:
        symbol: 股票代码
        timeframe: 时间周期 ('1d', '4h', '1h', '30m', '15m', '5m', '1m')
        
    Returns:
        Dict包含信号检测结果
    """
    # 获取当前交易日期
    today_date = get_config_value("TODAY_DATE")
    
    # 获取价格数据
    price_data = get_price_data(symbol, timeframe, periods=100)
    
    # 识别上涨趋势
    uptrends = identify_uptrends(price_data)
    
    # 至少需要两个上涨趋势
    if len(uptrends) < 2:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "date": today_date,
            "signal_detected": False,
            "reason": "未检测到足够的上涨趋势"
        }
    
    # 获取第二段上涨
    second_uptrend = uptrends[1]
    
    # 检查是否满足MACD信号条件
    uptrend_data = price_data.iloc[second_uptrend['start_idx']:second_uptrend['end_idx']+1]
    macd, macd_signal, macd_hist = talib.MACD(
        uptrend_data['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    # 检查条件：
    # 1. MACD都在0轴上方
    # 2. 至少有一个金叉
    # 3. MACD至少形成有两个峰值，且MACD的峰值一个比一个高
    # 4. 目前是在MACD第2个峰值刚开始往下走的过程中
    
    macd_above_zero = (macd > 0).all()
    
    # 检查金叉
    golden_cross = ((macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))).any()
    
    # 检查峰值条件
    peaks = []
    peak_indices = []
    for i in range(1, len(macd) - 1):
        if macd.iloc[i] > macd.iloc[i-1] and macd.iloc[i] > macd.iloc[i+1] and macd.iloc[i] > 0:
            peaks.append(macd.iloc[i])
            peak_indices.append(i)
    
    # 至少有两个峰值且后一个比前一个高
    increasing_peaks = len(peaks) >= 2 and peaks[-1] > peaks[-2]
    
    # 检查是否在第二峰值下降过程中
    in_descending_phase = False
    if len(peak_indices) >= 2:
        # 获取第二个峰值的索引
        second_peak_idx = peak_indices[-1]
        # 检查峰值之后是否开始下降
        if second_peak_idx < len(macd) - 1:
            after_peak = macd.iloc[second_peak_idx:]
            if len(after_peak) > 1 and after_peak.iloc[0] > after_peak.iloc[-1]:
                in_descending_phase = True
    
    signal_detected = macd_above_zero and golden_cross and increasing_peaks and in_descending_phase
    
    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "date": today_date,
        "signal_detected": signal_detected,
        "macd_above_zero": macd_above_zero,
        "has_golden_cross": golden_cross,
        "increasing_peaks": increasing_peaks,
        "in_descending_phase": in_descending_phase,
        "second_uptrend": second_uptrend
    }
    
    return result

@mcp.tool()
def find_entry_signal(symbol: str, detected_timeframe: str = "1d") -> Dict[str, Any]:
    """
    在更小周期中寻找开仓信号
    
    Args:
        symbol: 股票代码
        detected_timeframe: 检测到信号的周期
        
    Returns:
        Dict包含开仓信号信息
    """
    # 获取当前交易日期
    today_date = get_config_value("TODAY_DATE")
    
    # 根据检测到信号的周期，确定向下三个更小周期
    try:
        idx = ALL_TIMEFRAMES.index(detected_timeframe)
        # 获取更小的三个周期
        sub_timeframes = ALL_TIMEFRAMES[idx+1:idx+4] if idx+4 <= len(ALL_TIMEFRAMES) else ALL_TIMEFRAMES[idx+1:]
    except ValueError:
        # 如果未找到周期，默认使用较小的周期
        sub_timeframes = ["4h", "1h", "30m"]
    
    # 在小周期中寻找开仓信号
    entry_period = None
    entry_details = None
    
    for tf in sub_timeframes:
        # 获取小周期价格数据
        price_data = get_price_data(symbol, tf, periods=50)
        
        if len(price_data) < 10:
            continue
            
        # 计算MACD
        macd, macd_signal, macd_hist = talib.MACD(
            price_data['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # 检查开仓条件：
        # 1. MACD从0轴下上升到0轴上
        # 2. 回调过程中MACD向0轴下走
        # 3. MACD接近或形成金叉
        # 4. 0轴下峰值 < 0轴上峰值
        
        if len(macd) > 5:
            # 检查是否有从负到正的穿越
            # 找到最后一次MACD从负到正的穿越点
            cross_up_indices = []
            for i in range(1, len(macd)):
                if macd.iloc[i] > 0 and macd.iloc[i-1] <= 0:
                    cross_up_indices.append(i)
            
            if cross_up_indices:
                last_cross_up_idx = cross_up_indices[-1]
                # 检查穿越后是否回调
                if last_cross_up_idx < len(macd) - 1:
                    after_cross = macd.iloc[last_cross_up_idx:]
                    if len(after_cross) > 1 and after_cross.iloc[0] > after_cross.iloc[-1]:
                        # 在回调过程中检查是否接近金叉
                        near_golden_cross = False
                        # 检查最近几个周期是否接近金叉
                        recent_data = macd.iloc[-5:]
                        recent_signal = macd_signal.iloc[-5:]
                        for i in range(1, len(recent_data)):
                            if (recent_data.iloc[i] > recent_signal.iloc[i] and 
                                recent_data.iloc[i-1] <= recent_signal.iloc[i-1]):
                                near_golden_cross = True
                                break
                        
                        # 检查0轴下峰值和0轴上峰值
                        below_zero_peak = macd[macd < 0].max() if len(macd[macd < 0]) > 0 else None
                        above_zero_peak = macd[macd > 0].max() if len(macd[macd > 0]) > 0 else None
                        
                        if (below_zero_peak is not None and above_zero_peak is not None and 
                            below_zero_peak < above_zero_peak):
                            entry_period = tf
                            entry_details = {
                                "macd_values": macd.tail(5).tolist(),
                                "macd_signal_values": macd_signal.tail(5).tolist(),
                                "below_zero_peak": below_zero_peak,
                                "above_zero_peak": above_zero_peak
                            }
                            break
    
    result = {
        "symbol": symbol,
        "detected_timeframe": detected_timeframe,
        "entry_period": entry_period,
        "sub_timeframes": sub_timeframes,
        "entry_details": entry_details,
        "date": today_date
    }
    
    if entry_period:
        # 获取入场价格
        try:
            price_data = get_open_prices(today_date, [symbol])
            entry_price = price_data.get(f"{symbol}_price", 0)
            result["entry_price"] = entry_price
        except Exception as e:
            result["error"] = f"无法获取价格数据: {str(e)}"
    
    return result

@mcp.tool()
def calculate_stop_loss_take_profit(symbol: str, high: float, low: float, current_lowest_price: float = None) -> Dict[str, Any]:
    """
    计算止盈止损价格
    
    Args:
        symbol: 股票代码
        high: 最高价
        low: 最低价
        current_lowest_price: 当前回调中的最低价格（用于区域止损计算）
        
    Returns:
        Dict包含止盈止损价格
    """
    # 创建价格映射表 (将第二段上涨分成10份)
    price_map = {}
    for i in range(11):
        key = round(i / 10, 1)
        value = low + (high - low) * (1 - key)
        price_map[key] = value
    
    # 设置止盈价 (-0.2对应价格)
    take_profit_price = price_map[0.2]
    
    # 设置区域止损 (根据回调最低点到达的区间设置止损)
    stop_loss_price = low  # 默认设置为最低价
    
    # 如果提供了当前最低价，则根据区域止损规则计算
    if current_lowest_price is not None:
        # 确定当前最低价在哪个区间
        for i in range(10):
            upper_bound = price_map[i/10]
            lower_bound = price_map[(i+1)/10]
            if lower_bound <= current_lowest_price <= upper_bound:
                # 根据规则：回调最低点达X区间，则止损设于X+0.2对应价格
                # 例如：到达0.4区间，则止损设为0.6；到达0.5区间，则止损设为0.7
                stop_loss_price = price_map[(i+2)/10]
                break
    
    result = {
        "symbol": symbol,
        "high": high,
        "low": low,
        "take_profit_price": take_profit_price,
        "stop_loss_price": stop_loss_price,
        "price_map": price_map
    }
    
    return result

@mcp.tool()
def update_dynamic_stop_loss(symbol: str, current_price: float, entry_price: float, low: float) -> Dict[str, Any]:
    """
    动态更新止损价格
    
    Args:
        symbol: 股票代码
        current_price: 当前价格
        entry_price: 入场价格
        low: 最低价格
        
    Returns:
        Dict包含更新后的止损价格
    """
    # 计算价格变化
    price_diff = current_price - entry_price
    price_range = 0.1  # 假设价格区间为0.1，实际应该根据历史数据计算
    
    stop_loss_price = entry_price  # 默认止损价为成本价
    
    # 动态提损逻辑
    if price_range > 0:
        price_units = price_diff / price_range
        
        if price_units >= 2:
            # 价格上涨了2份，止损设为回调最低点
            stop_loss_price = low
        elif price_units >= 0.1:
            # 价格达0.1，止损设为成本价
            stop_loss_price = entry_price
        elif price_units >= 0:
            # 价格达0，止损设为0.1对应价格
            # 这里需要根据price_map计算，简化处理
            stop_loss_price = entry_price - 0.1 * (entry_price - low)
    
    result = {
        "symbol": symbol,
        "current_price": current_price,
        "entry_price": entry_price,
        "stop_loss_price": stop_loss_price,
        "updated": stop_loss_price != entry_price
    }
    
    return result

if __name__ == "__main__":
    port = int(os.getenv("MACD_STRATEGY_HTTP_PORT", "8006"))
    mcp.run(transport="streamable-http", port=port)