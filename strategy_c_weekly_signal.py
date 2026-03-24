#!/usr/bin/env python3
"""
Strategy C: A股跨境ETF轮动周度信号 (Top1纯动量版)

每周六运行，输出下周调仓建议。

核心逻辑:
  1. 标的池: 恒生科技ETF(513180) + 恒生医疗ETF(513060) + 恒生高股息ETF(159726)
  2. Top1动量轮动: 4周动量排名，100%持有最强的1只ETF
  3. 无趋势过滤、无Regime择时、无波动率缩放 — 简洁有效

修正后回测表现 (2022.05~2026.03, ~3.8年, 无前瞻偏差):
  - CAGR=18.9%, MDD=-20.4%, Sharpe=0.76
  - 2023:-4%, 2024:+22%, 2025:+55%

数据来源: Yahoo Finance API
交易成本: 8bps单边 (A股跨境ETF, 无印花税)

用法:
  python3 strategy_c_weekly_signal.py              # 正常运行
  python3 strategy_c_weekly_signal.py --json       # 仅输出JSON
  python3 strategy_c_weekly_signal.py --backtest   # 运行完整回测
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List

# ============================================================
# 配置
# ============================================================

DATA_DIR = os.environ.get(
    'ETF_DATA_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
)

# 标的池: A股跨境ETF (追踪恒生系指数)
ETF_POOL = {
    '513180.SS': {'name': '恒生科技ETF', 'code': '513180', 'factor': '科技'},
    '513060.SS': {'name': '恒生医疗ETF', 'code': '513060', 'factor': '医疗'},
    '159726.SZ': {'name': '恒生高股息ETF', 'code': '159726', 'factor': '高股息'},
}

# 策略参数
PARAMS = {
    'momentum_lookback': 4,     # 动量回看周数
    'txn_cost_bps': 8,          # 单边交易成本 (基点)
}

# 缓存过期天数
CACHE_MAX_AGE_DAYS = 3


# ============================================================
# 数据获取 (Yahoo Finance)
# ============================================================

def _fetch_yahoo(ticker: str, days: int = 2000) -> Optional[pd.DataFrame]:
    """从Yahoo Finance获取日线数据。"""
    end_ts = int(time.time())
    start_ts = end_ts - days * 86400
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
    params = {
        'period1': str(start_ts),
        'period2': str(end_ts),
        'interval': '1d',
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/120.0.0.0 Safari/537.36',
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  [WARN] Yahoo 429限流, 等待{wait}秒重试...", file=sys.stderr)
                time.sleep(wait)
                continue
            if r.status_code != 200:
                print(f"  [WARN] Yahoo HTTP {r.status_code}: {ticker}", file=sys.stderr)
                return None

            data = r.json()
            chart = data.get('chart', {}).get('result', [])
            if not chart:
                print(f"  [WARN] Yahoo无数据: {ticker}", file=sys.stderr)
                return None

            timestamps = chart[0].get('timestamp', [])
            closes = chart[0].get('indicators', {}).get('quote', [{}])[0].get('close', [])

            if not timestamps or not closes:
                return None

            rows = []
            for ts, c in zip(timestamps, closes):
                if c is not None:
                    rows.append({
                        'date': pd.Timestamp(ts, unit='s').normalize(),
                        'close': float(c),
                    })

            df = pd.DataFrame(rows)
            df = df.drop_duplicates(subset='date', keep='last')
            df = df.set_index('date').sort_index()
            return df

        except Exception as ex:
            print(f"  [WARN] Yahoo请求失败 {ticker} (尝试{attempt+1}): {ex}", file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(3)

    return None


def fetch_data(ticker: str, name: str, days: int = 2000) -> Optional[pd.DataFrame]:
    """获取数据，支持缓存。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    safe_name = ticker.replace('^', '').replace('.', '_')
    csv_path = os.path.join(DATA_DIR, f'sc_{safe_name}_daily.csv')

    # 检查缓存
    if os.path.exists(csv_path):
        file_age_days = (time.time() - os.path.getmtime(csv_path)) / 86400
        if file_age_days <= CACHE_MAX_AGE_DAYS:
            df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
            if len(df) > 0:
                print(f"  {name}: 使用缓存 ({len(df)}行, {file_age_days:.1f}天前)")
                return df[['close']]
        else:
            print(f"  {name}: 缓存过期 ({file_age_days:.1f}天), 刷新...")

    # 从Yahoo获取
    df = _fetch_yahoo(ticker, days)
    if df is not None and len(df) > 0:
        df.to_csv(csv_path)
        print(f"  {name}: Yahoo获取成功 ({len(df)}行)")
        return df[['close']]

    # 兜底: 过期缓存
    if os.path.exists(csv_path):
        print(f"  [WARN] API失败, 使用过期缓存: {name}", file=sys.stderr)
        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        if len(df) > 0:
            return df[['close']]

    print(f"  [ERROR] 无法获取数据: {name} ({ticker})", file=sys.stderr)
    return None


def load_all_data(for_backtest: bool = False) -> Optional[Dict[str, pd.Series]]:
    """
    加载所有ETF数据并转换为周频。

    Returns:
        etf_weekly dict 或 None
    """
    days = 2000 if for_backtest else 600
    print(f"加载数据 ({'回测模式' if for_backtest else '信号模式'})...")

    etf_daily = {}
    for ticker, info in ETF_POOL.items():
        df = fetch_data(ticker, info['name'], days=days)
        if df is None:
            print(f"  [ERROR] 无法加载 {info['name']}", file=sys.stderr)
            return None
        etf_daily[ticker] = df['close']
        time.sleep(3)  # Yahoo限流保护

    # 转换为周频 (每周五收盘)
    etf_weekly = {}
    for ticker, series in etf_daily.items():
        etf_weekly[ticker] = series.resample('W-FRI').last().dropna()

    # 对齐日期
    common_idx = etf_weekly[list(etf_weekly.keys())[0]].index
    for ticker in etf_weekly:
        common_idx = common_idx.intersection(etf_weekly[ticker].index)
    for ticker in etf_weekly:
        etf_weekly[ticker] = etf_weekly[ticker].loc[common_idx]

    print(f"  数据对齐完成: {len(common_idx)} 周 "
          f"({common_idx[0].strftime('%Y-%m-%d')} ~ {common_idx[-1].strftime('%Y-%m-%d')})")

    return etf_weekly


# ============================================================
# 策略核心逻辑 (纯动量)
# ============================================================

def select_top1(etf_weekly: Dict[str, pd.Series], idx: int) -> Tuple[Optional[str], List[dict]]:
    """
    Top1纯动量选择，无过滤。

    Args:
        etf_weekly: ETF周价格字典
        idx: 当前决策点 (使用 [:idx] 数据, 即 iloc[idx-1] 为最新价)

    Returns:
        (selected_ticker, momentum_details)
    """
    lookback = PARAMS['momentum_lookback']

    if idx < lookback + 1:
        return None, []

    # 计算4周动量
    momenta = []
    for ticker, prices in etf_weekly.items():
        if idx <= len(prices):
            mom = prices.iloc[idx-1] / prices.iloc[idx-1-lookback] - 1
            momenta.append((ticker, float(mom)))

    # 按动量排名
    ranked = sorted(momenta, key=lambda x: -x[1])

    details = []
    for rank_i, (ticker, mom) in enumerate(ranked):
        info = ETF_POOL[ticker]
        detail = {
            'ticker': ticker,
            'name': info['name'],
            'code': info['code'],
            'factor': info['factor'],
            'momentum_4w': round(mom * 100, 2),
            'rank': rank_i + 1,
            'selected': (rank_i == 0),
            'position_pct': 100.0 if rank_i == 0 else 0.0,
        }
        details.append(detail)

    selected = ranked[0][0] if ranked else None
    return selected, details


def generate_signal(etf_weekly: Dict[str, pd.Series]) -> dict:
    """生成本周信号。"""
    first_ticker = list(etf_weekly.keys())[0]
    latest_date = etf_weekly[first_ticker].index[-1]
    idx = len(etf_weekly[first_ticker])

    # Top1动量选择
    selected, momentum_details = select_top1(etf_weekly, idx)

    # 构建信号
    signal = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': 'Strategy C (跨境ETF纯动量轮动)',
        'variant': 'Top1 Pure Momentum (4w)',
        'total_position_pct': 100.0 if selected else 0.0,
        'cash_pct': 0.0 if selected else 100.0,
        'selected_etf': ETF_POOL[selected]['name'] if selected else '现金',
        'selected_code': ETF_POOL[selected]['code'] if selected else '-',
        'holdings': momentum_details,
        'action_summary': [],
    }

    # 生成中文调仓建议
    signal['action_summary'] = [
        f"本周持仓: {signal['selected_etf']} ({signal['selected_code']})",
        f"仓位: 100%",
        f"",
        f"动量排名 (4周):",
    ]
    for h in momentum_details:
        sel_icon = '>> ' if h.get('selected') else '   '
        signal['action_summary'].append(
            f"{sel_icon}#{h['rank']} {h['code']} {h['name']}: {h['momentum_4w']:+.1f}%"
        )

    return signal


# ============================================================
# 回测
# ============================================================

def run_backtest(etf_weekly: Dict[str, pd.Series]) -> dict:
    """
    运行完整回测。

    设计:
      - 在第 i 周末, 使用 [:i] 的数据做决策 (无前瞻偏差)
      - 决策后的仓位承受第 i→i+1 周的实际收益
      - 包含交易成本 8bps
    """
    print("\n运行回测...")

    txn_cost = PARAMS['txn_cost_bps'] / 10000
    tickers = list(etf_weekly.keys())
    lookback = PARAMS['momentum_lookback']
    min_weeks = lookback + 2

    etf_returns = {t: etf_weekly[t].pct_change() for t in tickers}

    nav = [1.0]
    dates = []
    weekly_holdings = []
    prev_holding = None
    total_txn_cost = 0.0

    for i in range(min_weeks, len(etf_weekly[tickers[0]]) - 1):
        # ---- 决策阶段 (使用 [:i+1] 即 iloc[0..i] 的数据) ----
        decision_idx = i + 1  # select_top1 uses [:idx], so idx=i+1 means data up to iloc[i]
        selected, _ = select_top1(etf_weekly, decision_idx)

        if selected is None:
            continue

        # ---- 收益阶段: 第 i+1 周的收益 ----
        ret_date = etf_weekly[tickers[0]].index[i + 1]
        week_ret = etf_returns[selected].iloc[i + 1]
        if pd.isna(week_ret):
            week_ret = 0.0

        # 交易成本
        if selected != prev_holding:
            if prev_holding is not None:
                period_txn = 2 * txn_cost  # 卖旧买新
            else:
                period_txn = txn_cost  # 首次建仓
        else:
            period_txn = 0.0
        total_txn_cost += period_txn

        portfolio_ret = week_ret - period_txn
        nav.append(nav[-1] * (1 + portfolio_ret))
        dates.append(ret_date)
        weekly_holdings.append(selected)
        prev_holding = selected

    nav_series = pd.Series(nav[1:], index=dates)

    # 计算指标
    total_years = (dates[-1] - dates[0]).days / 365.25
    cagr = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (1 / total_years) - 1

    drawdown = nav_series / nav_series.cummax() - 1
    mdd = drawdown.min()

    weekly_rets = nav_series.pct_change().dropna()
    sharpe = (weekly_rets.mean() * 52 - 0.025) / (weekly_rets.std() * np.sqrt(52)) if weekly_rets.std() > 0 else 0  # rf=2.5%
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    # 持仓分布
    holding_counts = pd.Series(weekly_holdings).value_counts()
    holding_dist = {}
    for h, cnt in holding_counts.items():
        holding_dist[ETF_POOL[h]['name']] = round(cnt / len(weekly_holdings) * 100, 1)

    # 胜率
    win_rate = (weekly_rets > 0).sum() / len(weekly_rets) * 100

    # 年度收益
    annual = nav_series.resample('YE').last().pct_change().dropna()
    annual_returns = {str(d.year): round(v * 100, 1) for d, v in annual.items()}

    # 换手次数
    trades = sum(1 for j in range(1, len(weekly_holdings))
                 if weekly_holdings[j] != weekly_holdings[j-1])

    result = {
        'strategy': 'Strategy C: Top1纯动量 (4周)',
        'period': f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}",
        'total_weeks': len(nav_series),
        'total_years': round(total_years, 1),
        'cagr_pct': round(cagr * 100, 1),
        'total_return_pct': round((nav_series.iloc[-1] - 1) * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round(win_rate, 1),
        'holding_distribution': holding_dist,
        'trade_count': trades,
        'annual_returns': annual_returns,
        'total_txn_cost_pct': round(total_txn_cost * 100, 2),
        'txn_cost_bps': PARAMS['txn_cost_bps'],
        'note': '无前瞻偏差: T周决策→T+1周收益, 交易成本8bp',
    }

    print(f"\n回测结果:")
    print(f"  期间: {result['period']} ({result['total_years']}年)")
    print(f"  CAGR: {result['cagr_pct']}%")
    print(f"  总收益: {result['total_return_pct']}%")
    print(f"  最大回撤: {result['mdd_pct']}%")
    print(f"  Sharpe: {result['sharpe']}")
    print(f"  Calmar: {result['calmar']}")
    print(f"  胜率: {result['win_rate_pct']}%")
    print(f"  换手次数: {result['trade_count']}")
    print(f"  累计交易成本: {result['total_txn_cost_pct']}%")
    print(f"\n持仓分布:")
    for name, pct in holding_dist.items():
        print(f"  {name}: {pct}%")
    print(f"\n年度收益:")
    for year, ret in sorted(annual_returns.items()):
        print(f"  {year}: {ret:+.1f}%")

    return result


# ============================================================
# 输出格式化
# ============================================================

def format_signal_text(signal: dict) -> str:
    """格式化信号为可读文本 (适合Telegram/终端)。"""
    lines = [
        f"{'='*40}",
        f"策略C 跨境ETF纯动量轮动 周度信号",
        f"日期: {signal['date']}",
        f"{'='*40}",
        f"",
    ]

    for line in signal['action_summary']:
        lines.append(line)

    lines.extend([
        f"",
        f"策略: Top1纯动量, 4周回看, 100%持仓",
        f"生成时间: {signal['generated_at']}",
    ])

    return '\n'.join(lines)


# ============================================================
# 主入口
# ============================================================

def main():
    args = sys.argv[1:]
    json_only = '--json' in args
    do_backtest = '--backtest' in args

    if not json_only:
        print("=" * 50)
        print(f"策略C: 跨境ETF纯动量轮动 周度信号")
        print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)

    # 加载数据
    result = load_all_data(for_backtest=do_backtest)
    if result is None:
        print("[ERROR] 数据加载失败!", file=sys.stderr)
        sys.exit(1)

    etf_weekly = result

    if do_backtest:
        bt_result = run_backtest(etf_weekly)
        output_path = os.path.join(DATA_DIR, 'strategy_c_backtest_result.json')
        with open(output_path, 'w') as f:
            json.dump(bt_result, f, indent=2, ensure_ascii=False)
        print(f"\n回测结果已保存: {output_path}")
        return

    # 生成本周信号
    signal = generate_signal(etf_weekly)

    # 保存JSON
    output_path = os.path.join(DATA_DIR, 'strategy_c_latest_signal.json')
    with open(output_path, 'w') as f:
        json.dump(signal, f, indent=2, ensure_ascii=False)

    if json_only:
        print(json.dumps(signal, indent=2, ensure_ascii=False))
    else:
        print(format_signal_text(signal))
        print(f"\n信号已保存: {output_path}")

    return signal


if __name__ == '__main__':
    main()
