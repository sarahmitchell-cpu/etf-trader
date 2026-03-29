#!/usr/bin/env python3
"""
股票数据公共模块 — 策略D/E共用

提供:
  - STOCK_POOL: 28只龙头股配置
  - DATA_DIR: 数据缓存目录
  - fetch_yahoo(): Yahoo Finance周线数据获取
  - fetch_stock(): 带缓存的单只股票数据获取
  - load_all_data(): 加载全部股票周线价格矩阵
"""

from __future__ import annotations

import pandas as pd
import requests
import os
import sys
import time
from typing import Optional

# ============================================================
# 配置
# ============================================================

DATA_DIR = os.environ.get(
    'ETF_DATA_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
)

CACHE_MAX_AGE_DAYS = 3

# 龙头股池 (28只, 覆盖15个主要行业)
# ⚠️ 严重警告: 手工选股存在幸存者偏差(survivorship bias)
# 这28只股票是2024年回看时选出的行业领军者，用它们回测2021-2026会高估收益。
# 真实可交易的收益应打折30-50%。策略D/G通过baostock历史成分股解决了此问题。
STOCK_POOL = {
    '600519.SS': {'name': '贵州茅台', 'code': '600519', 'sector': '白酒'},
    '000858.SZ': {'name': '五粮液', 'code': '000858', 'sector': '白酒'},
    '600887.SS': {'name': '伊利股份', 'code': '600887', 'sector': '食品'},
    '002714.SZ': {'name': '牧原股份', 'code': '002714', 'sector': '养殖'},
    '601318.SS': {'name': '中国平安', 'code': '601318', 'sector': '保险'},
    '600036.SS': {'name': '招商银行', 'code': '600036', 'sector': '银行'},
    '002415.SZ': {'name': '海康威视', 'code': '002415', 'sector': '安防'},
    '300750.SZ': {'name': '宁德时代', 'code': '300750', 'sector': '电池'},
    '600276.SS': {'name': '恒瑞医药', 'code': '600276', 'sector': '创新药'},
    '300760.SZ': {'name': '迈瑞医疗', 'code': '300760', 'sector': '医疗器械'},
    '601012.SS': {'name': '隆基绿能', 'code': '601012', 'sector': '光伏'},
    '300274.SZ': {'name': '阳光电源', 'code': '300274', 'sector': '逆变器'},
    '000333.SZ': {'name': '美的集团', 'code': '000333', 'sector': '家电'},
    '600690.SS': {'name': '海尔智家', 'code': '600690', 'sector': '家电'},
    '002594.SZ': {'name': '比亚迪', 'code': '002594', 'sector': '新能源车'},
    '600893.SS': {'name': '航发动力', 'code': '600893', 'sector': '航发'},
    '601668.SS': {'name': '中国建筑', 'code': '601668', 'sector': '建筑'},
    '600585.SS': {'name': '海螺水泥', 'code': '600585', 'sector': '水泥'},
    '601899.SS': {'name': '紫金矿业', 'code': '601899', 'sector': '有色'},
    '600028.SS': {'name': '中国石化', 'code': '600028', 'sector': '石油'},
    '002371.SZ': {'name': '北方华创', 'code': '002371', 'sector': '半导体设备'},
    '000063.SZ': {'name': '中兴通讯', 'code': '000063', 'sector': '通信设备'},
    '600941.SS': {'name': '中国移动', 'code': '600941', 'sector': '运营商'},
    '002230.SZ': {'name': '科大讯飞', 'code': '002230', 'sector': 'AI'},
    '600900.SS': {'name': '长江电力', 'code': '600900', 'sector': '水电'},
    '601088.SS': {'name': '中国神华', 'code': '601088', 'sector': '煤炭'},
    '601006.SS': {'name': '大秦铁路', 'code': '601006', 'sector': '铁路'},
    '001979.SZ': {'name': '招商蛇口', 'code': '001979', 'sector': '地产'},
}


# ============================================================
# 数据获取
# ============================================================

def fetch_yahoo(ticker: str, days: int = 2000) -> Optional[pd.DataFrame]:
    """
    从Yahoo Finance获取周线数据。

    Args:
        ticker: Yahoo Finance格式的股票代码 (如 '600519.SS')
        days: 获取最近多少天的数据

    Returns:
        DataFrame with DatetimeIndex, columns=['close', 'adjclose'], or None on failure
    """
    end_ts = int(time.time())
    start_ts = end_ts - days * 86400
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
    params = {'period1': str(start_ts), 'period2': str(end_ts), 'interval': '1wk'}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/120.0.0.0 Safari/537.36',
    }

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  [429] {ticker} 限流, 等待{wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            if r.status_code != 200:
                return None

            data = r.json()
            chart = data.get('chart', {}).get('result', [])
            if not chart:
                return None

            timestamps = chart[0].get('timestamp', [])
            quote = chart[0].get('indicators', {}).get('quote', [{}])[0]
            closes = quote.get('close', [])
            adjclose_list = chart[0].get('indicators', {}).get('adjclose', [{}])
            adjcloses = adjclose_list[0].get('adjclose', closes) if adjclose_list else closes

            rows = []
            for ts, c, ac in zip(timestamps, closes, adjcloses):
                if c is not None and ac is not None:
                    rows.append({
                        'date': pd.Timestamp(ts, unit='s').normalize(),
                        'close': float(c),
                        'adjclose': float(ac),
                    })

            df = pd.DataFrame(rows)
            df = df.drop_duplicates(subset='date', keep='last').set_index('date').sort_index()
            return df

        except Exception as ex:
            print(f"  [ERR] {ticker}: {ex}", file=sys.stderr)
            if attempt < 2:
                time.sleep(5)
    return None


def fetch_stock(ticker: str, info: dict, days: int = 2000,
                cache_max_age: float = CACHE_MAX_AGE_DAYS) -> Optional[pd.Series]:
    """
    获取单只股票周线复权收盘价，带CSV缓存。

    优先读缓存(未过期时)，否则从Yahoo获取并写入缓存，
    最后兜底使用过期缓存。

    Args:
        ticker: Yahoo Finance格式的股票代码
        info: 股票信息字典 (需含 'name')
        days: 获取天数
        cache_max_age: 缓存最大有效天数

    Returns:
        adjclose Series (DatetimeIndex), or None on failure
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    safe = ticker.replace('.', '_')
    csv_path = os.path.join(DATA_DIR, f'sd_{safe}_weekly.csv')

    # 1. 未过期缓存
    if os.path.exists(csv_path):
        age = (time.time() - os.path.getmtime(csv_path)) / 86400
        if age <= cache_max_age:
            df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
            if len(df) > 0:
                print(f"  {info['name']}: 缓存 ({len(df)}周)")
                return df['adjclose'] if 'adjclose' in df.columns else df['close']

    # 2. Yahoo获取
    df = fetch_yahoo(ticker, days)
    if df is not None and len(df) > 0:
        df.to_csv(csv_path)
        print(f"  {info['name']}: Yahoo ({len(df)}周)")
        return df['adjclose'] if 'adjclose' in df.columns else df['close']

    # 3. 过期缓存兜底
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        if len(df) > 0:
            print(f"  {info['name']}: 过期缓存")
            return df['adjclose'] if 'adjclose' in df.columns else df['close']

    print(f"  {info['name']}: FAILED", file=sys.stderr)
    return None


def load_all_data(stock_pool: dict = None, for_backtest: bool = False,
                  min_stocks: int = 15, min_weeks: int = 20,
                  sleep_between: float = 3.0) -> Optional[pd.DataFrame]:
    """
    加载股票池所有股票的周线价格矩阵。

    Args:
        stock_pool: 股票池字典 (默认使用 STOCK_POOL)
        for_backtest: True=获取更长历史 (2000天), False=仅获取近期 (200天)
        min_stocks: 最少成功加载股票数 (低于此数返回None)
        min_weeks: 每只股票最少周数据量 (低于此数跳过该股)
        sleep_between: 每只股票请求间隔秒数

    Returns:
        DataFrame (DatetimeIndex, columns=tickers), or None on failure
    """
    if stock_pool is None:
        stock_pool = STOCK_POOL

    days = 2000 if for_backtest else 200
    print(f"加载数据 ({'回测' if for_backtest else '信号'}模式)...")

    prices = {}
    for ticker, info in stock_pool.items():
        s = fetch_stock(ticker, info, days)
        if s is not None and len(s) > min_weeks:
            prices[ticker] = s
        if sleep_between > 0:
            time.sleep(sleep_between)

    if len(prices) < min_stocks:
        print(f"[ERROR] 仅加载{len(prices)}只，不足{min_stocks}只", file=sys.stderr)
        return None

    price_df = pd.DataFrame(prices).dropna(how='all')
    print(f"  数据矩阵: {price_df.shape[0]}周 x {price_df.shape[1]}股")
    if len(price_df) > 0:
        print(f"  {price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")
    return price_df
