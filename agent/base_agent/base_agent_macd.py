"""
MACD策略专用代理类
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from prompts.agent_prompt_macd import get_macd_strategy_system_prompt
from tools.general_tools import (extract_conversation, extract_tool_messages,
                                 get_config_value, write_config_value)
from tools.price_tools import add_no_trade_record

# Load environment variables
load_dotenv()

# 支持的所有时间周期
ALL_TIMEFRAMES = ["1d", "4h", "1h", "30m", "15m", "5m", "1m"]


class BaseAgentMACD:
    """
    MACD策略专用代理类

    主要功能：
    1. MCP工具管理和连接
    2. AI模型创建和配置
    3. 交易执行和决策循环
    4. 日志记录和管理
    5. 持仓和配置管理
    """

    # 默认纳斯达克100股票代码
    DEFAULT_STOCK_SYMBOLS = [
        "NVDA",
        "MSFT",
        "AAPL",
        "GOOG",
        "GOOGL",
        "AMZN",
        "META",
        "AVGO",
        "TSLA",
        "NFLX",
        "PLTR",
        "COST",
        "ASML",
        "AMD",
        "CSCO",
        "AZN",
        "TMUS",
        "MU",
        "LIN",
        "PEP",
        "SHOP",
        "APP",
        "INTU",
        "AMAT",
        "LRCX",
        "PDD",
        "QCOM",
        "ARM",
        "INTC",
        "BKNG",
        "AMGN",
        "TXN",
        "ISRG",
        "GILD",
        "KLAC",
        "PANW",
        "ADBE",
        "HON",
        "CRWD",
        "CEG",
        "ADI",
        "ADP",
        "DASH",
        "CMCSA",
        "VRTX",
        "MELI",
        "SBUX",
        "CDNS",
        "ORLY",
        "SNPS",
        "MSTR",
        "MDLZ",
        "ABNB",
        "MRVL",
        "CTAS",
        "TRI",
        "MAR",
        "MNST",
        "CSX",
        "ADSK",
        "PYPL",
        "FTNT",
        "AEP",
        "WDAY",
        "REGN",
        "ROP",
        "NXPI",
        "DDOG",
        "AXON",
        "ROST",
        "IDXX",
        "EA",
        "PCAR",
        "FAST",
        "EXC",
        "TTWO",
        "XEL",
        "ZS",
        "PAYX",
        "WBD",
        "BKR",
        "CPRT",
        "CCEP",
        "FANG",
        "TEAM",
        "CHTR",
        "KDP",
        "MCHP",
        "GEHC",
        "VRSK",
        "CTSH",
        "CSGP",
        "KHC",
        "ODFL",
        "DXCM",
        "BIIB",
        "ON",
        "CTVA",
        "MDB",
        "TTD",
        "SIRI",
        "WBA",
        "GFS",
        "DLTR",
        "NDAQ",
        "FSLR",
        "EXPE",
        "VRSN",
        "ENPH",
        "LCID",
        "INCY",
        "7799.T",
    ]

    def __init__(
        self,
        signature: str,
        basemodel: str = "gpt-4-turbo",
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        initial_cash: float = 100000.0,
        max_steps: int = 15,
        base_delay: float = 1.0,
        max_retries: int = 3,
        stock_symbols: Optional[List[str]] = None,
        market: str = "us",
        data_path: str = "./data",
        log_path: str = "./data/agent_data",
        init_date: str = "2023-01-01",
        end_date: str = "2024-01-01",
    ):
        """
        初始化BaseAgentMACD

        Args:
            signature: 代理签名（模型名称）
            basemodel: 基础模型名称
            openai_api_key: OpenAI API密钥
            openai_base_url: OpenAI基础URL
            initial_cash: 初始资金
            max_steps: 每个交易会话的最大步骤数
            base_delay: 重试基础延迟
            max_retries: 最大重试次数
            stock_symbols: 交易的股票代码列表
            market: 市场类型（"us"或"cn"）
            data_path: 数据路径
            log_path: 日志路径
            init_date: 初始化日期
            end_date: 结束日期
        """
        # 基础配置
        self.signature = signature
        self.basemodel = basemodel
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        self.initial_cash = initial_cash
        self.max_steps = max_steps
        self.base_delay = base_delay
        self.max_retries = max_retries
        self.stock_symbols = stock_symbols or self.DEFAULT_STOCK_SYMBOLS.copy()
        self.market = market
        self.data_path = data_path
        self.base_log_path = log_path
        self.init_date = init_date
        self.end_date = end_date

        # 持仓文件路径
        self.position_file = os.path.join(self.data_path, "agent_data", self.signature, "position", "position.jsonl")

        # MCP客户端和工具
        self.client = None
        self.tools = None
        self.model = None
        self.agent = None

        # MCP配置
        self.mcp_config = {
            "math": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('MATH_HTTP_PORT', '8000')}/mcp",
            },
            "search": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('SEARCH_HTTP_PORT', '8001')}/mcp",
            },
            "trade": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('TRADE_HTTP_PORT', '8002')}/mcp",
            },
            "price": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('GETPRICE_HTTP_PORT', '8003')}/mcp",
            },
            "macd_strategy": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('MACD_STRATEGY_HTTP_PORT', '8006')}/mcp",
            },
        }

    async def initialize(self) -> None:
        """初始化MCP客户端和AI模型"""
        print(f"🚀 初始化代理: {self.signature}")

        # 验证OpenAI配置
        if not self.openai_api_key:
            raise ValueError(
                "❌ OpenAI API密钥未设置。请在环境变量或配置文件中配置OPENAI_API_KEY。"
            )
        if not self.openai_base_url:
            print("⚠️  OpenAI基础URL未设置，使用默认值")

        try:
            # 创建MCP客户端
            self.client = MultiServerMCPClient(self.mcp_config)

            # 获取工具
            self.tools = await self.client.get_tools()
            if not self.tools:
                print("⚠️  警告: 未加载到MCP工具。MCP服务可能未运行。")
                print(f"   MCP配置: {self.mcp_config}")
            else:
                print(f"✅ 已加载 {len(self.tools)} 个MCP工具")
        except Exception as e:
            raise RuntimeError(
                f"❌ 初始化MCP客户端失败: {e}\n"
                f"   请确保MCP服务在配置的端口上运行。\n"
                f"   运行: python agent_tools/start_mcp_services.py"
            )

        try:
            # 创建AI模型 - 为DeepSeek模型使用自定义的DeepSeekChatOpenAI
            # 处理tool_calls.args格式差异（JSON字符串 vs 字典）
            if "deepseek" in self.basemodel.lower():
                self.model = ChatOpenAI(
                    model=self.basemodel,
                    base_url=self.openai_base_url,
                    api_key=self.openai_api_key,
                    max_retries=3,
                    timeout=30,
                )
            else:
                self.model = ChatOpenAI(
                    model=self.basemodel,
                    base_url=self.openai_base_url,
                    api_key=self.openai_api_key,
                    max_retries=3,
                    timeout=30,
                )
        except Exception as e:
            raise RuntimeError(f"❌ 初始化AI模型失败: {e}")

        # 注意: agent将在run_trading_session()中基于特定日期创建
        # 因为system_prompt需要当前日期和价格信息

        print(f"✅ 代理 {self.signature} 初始化完成")

    def _setup_logging(self, today_date: str) -> str:
        """设置日志文件路径"""
        log_path = os.path.join(self.base_log_path, self.signature, "log", today_date)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        return os.path.join(log_path, "log.jsonl")

    def _log_message(self, log_file: str, new_messages: List[Dict[str, str]]) -> None:
        """记录消息到日志文件"""
        log_entry = {
            "signature": self.signature,
            "new_messages": new_messages
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    async def _ainvoke_with_retry(self, message: List[Dict[str, str]]) -> Any:
        """带重试的代理调用"""
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self.agent.ainvoke({"messages": message}, {"recursion_limit": 100})
            except Exception as e:
                if attempt == self.max_retries:
                    raise e
                print(f"⚠️ 尝试 {attempt} 失败，{self.base_delay * attempt} 秒后重试...")
                print(f"错误详情: {e}")
                await asyncio.sleep(self.base_delay * attempt)

    async def run_trading_session(self, today_date: str) -> None:
        """
        运行单日交易会话

        Args:
            today_date: 交易日期
        """
        print(f"📈 开始交易会话: {today_date}")

        # 设置日志
        log_file = self._setup_logging(today_date)
        write_config_value("LOG_FILE", log_file)
        # 更新系统提示词
        self.agent = create_agent(
            self.model,
            tools=self.tools,
            system_prompt=get_macd_strategy_system_prompt(today_date, self.signature, self.market, self.stock_symbols),
        )

        # 初始用户查询
        user_query = [{"role": "user", "content": f"请分析并更新今日({today_date})的持仓。在多个周期（日、4小时、1小时、30分钟、15分钟、5分钟、1分钟）上识别MACD信号。"}]
        message = user_query.copy()

        # 记录初始消息
        self._log_message(log_file, user_query)

        # 交易循环
        current_step = 0
        while current_step < self.max_steps:
            current_step += 1
            print(f"🔄 第 {current_step}/{self.max_steps} 步")

            try:
                # 调用代理
                response = await self._ainvoke_with_retry(message)

                # 提取代理响应
                agent_response = extract_conversation(response, "final")

                # 检查停止信号
                if "STOP_SIGNAL" in agent_response:
                    print("✅ 收到停止信号，交易会话结束")
                    print(agent_response)
                    self._log_message(log_file, [{"role": "assistant", "content": agent_response}])
                    break

                # 提取工具消息
                tool_msgs = extract_tool_messages(response)
                tool_response = "\n".join([msg.content for msg in tool_msgs])

                # 准备新消息
                new_messages = [
                    {"role": "assistant", "content": agent_response},
                    {"role": "user", "content": f"工具结果: {tool_response}"},
                ]

                # 添加新消息
                message.extend(new_messages)

                # 记录消息
                self._log_message(log_file, new_messages[0])
                self._log_message(log_file, new_messages[1])

            except Exception as e:
                print(f"❌ 交易会话错误: {str(e)}")
                print(f"错误详情: {e}")
                raise

        # 处理交易结果
        await self._handle_trading_result(today_date)

    async def _handle_trading_result(self, today_date: str) -> None:
        """处理交易结果"""
        if_trade = get_config_value("IF_TRADE")
        if if_trade:
            write_config_value("IF_TRADE", False)
            print("✅ 交易完成")
        else:
            print("📊 无交易，保持持仓")
            try:
                add_no_trade_record(today_date, self.signature)
            except NameError as e:
                print(f"❌ NameError: {e}")
                raise
            write_config_value("IF_TRADE", False)

    def register_agent(self) -> None:
        """注册新代理，创建初始持仓"""
        # 检查position.jsonl文件是否已存在
        if os.path.exists(self.position_file):
            print(f"⚠️ 持仓文件 {self.position_file} 已存在，跳过注册")
            return

        # 确保目录结构存在
        position_dir = os.path.join(self.data_path, "position")
        if not os.path.exists(position_dir):
            os.makedirs(position_dir)
            print(f"📁 创建持仓目录: {position_dir}")

        # 创建初始持仓
        init_position = {symbol: 0 for symbol in self.stock_symbols}
        init_position["CASH"] = self.initial_cash

        with open(self.position_file, "w") as f:  # 使用"w"模式确保创建新文件
            f.write(json.dumps({"date": self.init_date, "id": 0, "positions": init_position}) + "\n")

        print(f"✅ 代理 {self.signature} 注册完成")
        print(f"📁 持仓文件: {self.position_file}")
        currency_symbol = "¥" if self.market == "cn" else "$"
        print(f"💰 初始资金: {currency_symbol}{self.initial_cash:,.2f}")
        print(f"📊 股票数量: {len(self.stock_symbols)}")

    def get_trading_dates(self, init_date: str, end_date: str) -> List[str]:
        """
        获取交易日期列表，按merged.jsonl中的实际交易日过滤

        Args:
            init_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日期列表（排除周末和节假日）
        """
        from tools.price_tools import is_trading_day

        dates = []
        max_date = None

        if not os.path.exists(self.position_file):
            self.register_agent()
            max_date = init_date
        else:
            # 读取现有持仓文件，找到最新日期
            with open(self.position_file, "r") as f:
                for line in f:
                    doc = json.loads(line)
                    current_date = doc["date"]
                    if max_date is None:
                        max_date = current_date
                    else:
                        current_date_obj = datetime.strptime(current_date, "%Y-%m-%d")
                        max_date_obj = datetime.strptime(max_date, "%Y-%m-%d")
                        if current_date_obj > max_date_obj:
                            max_date = current_date

        # 检查是否需要处理新日期
        max_date_obj = datetime.strptime(max_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

        if end_date_obj <= max_date_obj:
            return []

        # 生成交易日期列表，按实际交易日过滤
        trading_dates = []
        current_date = max_date_obj + timedelta(days=1)

        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")
            # 检查这天是否是merged.jsonl中的实际交易日
            if is_trading_day(date_str, market=self.market):
                trading_dates.append(date_str)
            current_date += timedelta(days=1)

        return trading_dates

    async def run_with_retry(self, today_date: str) -> None:
        """带重试的运行方法"""
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"🔄 尝试运行 {self.signature} - {today_date} (第 {attempt} 次尝试)")
                await self.run_trading_session(today_date)
                print(f"✅ {self.signature} - {today_date} 运行成功")
                return
            except Exception as e:
                print(f"❌ 第 {attempt} 次尝试失败: {str(e)}")
                if attempt == self.max_retries:
                    print(f"💥 {self.signature} - {today_date} 所有重试均失败")
                    raise
                else:
                    wait_time = self.base_delay * attempt
                    print(f"⏳ 等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)

    async def run_date_range(self, init_date: str, end_date: str) -> None:
        """
        运行日期范围内的所有交易日

        Args:
            init_date: 开始日期
            end_date: 结束日期
        """
        print(f"📅 运行日期范围: {init_date} 到 {end_date}")

        # 获取交易日期列表
        trading_dates = self.get_trading_dates(init_date, end_date)

        if not trading_dates:
            print(f"ℹ️ 没有需要处理的交易日")
            return

        print(f"📊 需要处理的交易日: {trading_dates}")

        # 运行每个交易日
        for today_date in trading_dates:
            write_config_value("TODAY_DATE", today_date)
            await self.run_with_retry(today_date)
            
            # 如果是实时模式，可能需要等待一段时间再处理下一个交易日
            # 这里可以添加适当的延迟逻辑