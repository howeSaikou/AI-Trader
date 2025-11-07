#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
启动所有MCP服务的脚本
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_tools.tool_math import mcp as math_mcp
from agent_tools.tool_get_price_local import mcp as price_mcp
from agent_tools.tool_trade import mcp as trade_mcp
from agent_tools.tool_jina_search import mcp as search_mcp
from agent_tools.tool_alphavantage_news import mcp as news_mcp
from agent_tools.tool_macd_strategy import mcp as macd_strategy_mcp  # 添加MACD策略工具

async def start_service(mcp_instance, port, service_name):
    """启动单个MCP服务"""
    try:
        print(f"🚀 启动 {service_name} 服务 (端口: {port})...")
        await mcp_instance.run(transport="streamable-http", port=port)
    except Exception as e:
        print(f"❌ {service_name} 服务启动失败: {e}")

async def main():
    """主函数 - 启动所有MCP服务"""
    print("🚀 启动所有MCP服务...")
    
    # 定义服务配置
    services = [
        (math_mcp, int(os.getenv("MATH_HTTP_PORT", "8000")), "Math"),
        (search_mcp, int(os.getenv("SEARCH_HTTP_PORT", "8001")), "Search"),
        (trade_mcp, int(os.getenv("TRADE_HTTP_PORT", "8002")), "Trade"),
        (price_mcp, int(os.getenv("GETPRICE_HTTP_PORT", "8003")), "Price"),
        (news_mcp, int(os.getenv("NEWS_HTTP_PORT", "8005")), "News"),
        (macd_strategy_mcp, int(os.getenv("MACD_STRATEGY_HTTP_PORT", "8006")), "MACD Strategy")  # 添加MACD策略服务
    ]
    
    # 创建任务列表
    tasks = [
        asyncio.create_task(start_service(mcp_instance, port, name))
        for mcp_instance, port, name in services
    ]
    
    try:
        # 等待所有任务完成
        await asyncio.gather(*tasks, return_exceptions=True)
    except KeyboardInterrupt:
        print("\n⚠️  收到中断信号，正在停止所有服务...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        print("✅ 所有服务已停止")
    except Exception as e:
        print(f"❌ 服务运行出错: {e}")

if __name__ == "__main__":
    asyncio.run(main())
