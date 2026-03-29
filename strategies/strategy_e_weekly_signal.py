#!/usr/bin/env python3
"""
Strategy E: A股价值投资策略 (逆向价值+质量) — V3 无幸存者偏差版

每月运行(每4周)，输出调仓建议。

核心逻辑:
  1. 股票池: CSI300 真实历史成分股 (baostock, 无幸存者偏差)
  2. 价值因子: 10周动量反转 (近期跌多=便宜, 逆向排名)
  3. 质量因子: 12周波动率 (低波=高质量), 权重20%
  4. 安全过滤: 4周动量>-20% (排除暴跌/基本面恶化)
  5. 行业约束: 每行业最多2只 (sector_max=2)
  6. Top10等权, 月度调仓

V3 改进:
  - 从28只手工选股迁移到CSI300历史成分股，彻底消除幸存者偏差
  - 使用baostock历史成分股列表 (与策略D共享数据基础设施)
  - 股票池从28只扩大到~300只，因子选股更有统计意义
  - Top N从4提升到10，分散化更充分

数据来源: baostock (历史成分股 + 周线价格 + 行业分类)
交易成本: 8bps单边

用法:
  python3 strategy_e_weekly_signal.py              # 正常运行
  python3 strategy_e_weekly_signal.py --json       # 仅输出JSON
  python3 strategy_e_weekly_signal.py --backtest   # 运行完整回测
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import json
import os
import sys
import time
import baostock as bs
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'baostock_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# Strategy Parameters
# ============================================================

PARAMS = {
    'value_lookback': 10,       # 价值因子回看周数
    'vol_lookback': 12,         # 波动率回看周数
    'vol_weight': 0.2,          # 质量因子权重 (0=纯价值, 1=纯质量)
    'top_n': 10,                # 选前N名
    'rebal_freq': 4,            # 调仓频率 (周)
    'max_drawdown_filter': -0.20,  # 4周动量安全过滤
    'txn_cost_bps': 8,
    'sector_max': 2,            # 每行业最多选N只
    'warmup': 20,               # 预热期(周)
}

# CSI 300 rebalance dates (semi-annual, shared with strategy D)
REBALANCE_DATES = [
    '2020-12-14', '2021-06-15', '2021-12-13', '2022-06-13',
    '2022-12-12', '2023-06-12', '2023-12-11', '2024-06-17',
    '2024-12-16', '2025-06-16', '2025-12-15',
]


# ============================================================
# Data Loading (reuse Strategy D infrastructure)
# ============================================================

def fetch_constituents():
    """Fetch CSI 300 historical constituents from baostock"""
    cache_path = os.path.join(CACHE_DIR, 'csi300_constituents_history.json')
    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 7:
            with open(cache_path) as f:
                return json.load(f)

    lg = bs.login()
    constituent_map = {}
    all_stocks = set()

    for d in REBALANCE_DATES:
        rs = bs.query_hs300_stocks(date=d)
        stocks = []
        while rs.next():
            row = rs.get_row_data()
            stocks.append(row[1])
        constituent_map[d] = stocks
        all_stocks.update(stocks)
        print(f"  {d}: {len(stocks)} stocks")

    bs.logout()

    data = {
        'dates': REBALANCE_DATES,
        'constituents': constituent_map,
        'all_unique_stocks': sorted(all_stocks),
        'total_unique': len(all_stocks),
    }

    with open(cache_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)

    return data


def fetch_industries(stock_list):
    """Get industry classification for all stocks"""
    cache_path = os.path.join(CACHE_DIR, 'csi300_industries.json')
    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 30:
            with open(cache_path) as f:
                data = json.load(f)
            missing = [s for s in stock_list if s not in data]
            if len(missing) < 10:
                return data

    lg = bs.login()
    industries = {}
    for i, stock in enumerate(stock_list):
        if i % 100 == 0:
            print(f"  Fetching industries: {i}/{len(stock_list)}...")
        rs = bs.query_stock_industry(code=stock)
        while rs.next():
            row = rs.get_row_data()
            if len(row) >= 4:
                industries[stock] = {
                    'name': row[2],
                    'industry': row[3],
                    'classification': row[4] if len(row) > 4 else '',
                }
                break
        if stock not in industries:
            industries[stock] = {'name': '?', 'industry': '其他', 'classification': ''}

    bs.logout()

    with open(cache_path, 'w') as f:
        json.dump(industries, f, ensure_ascii=False, indent=1)

    return industries


def fetch_weekly_prices(stock_list):
    """Fetch weekly close prices for all stocks"""
    cache_path = os.path.join(CACHE_DIR, 'csi300_all_weekly_prices.pkl')
    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 3:
            df = pd.read_pickle(cache_path)
            missing = [s for s in stock_list if s not in df.columns]
            if len(missing) < 20:
                return df

    lg = bs.login()
    all_prices = {}
    total = len(stock_list)

    for i, stock in enumerate(stock_list):
        if i % 50 == 0:
            print(f"  Fetching prices: {i}/{total}...")

        rs = bs.query_history_k_data_plus(
            stock, 'date,close',
            start_date='2020-09-01', end_date='2026-12-31',
            frequency='w', adjustflag='1'
        )

        data = []
        while rs.next():
            row = rs.get_row_data()
            try:
                data.append({'date': row[0], 'close': float(row[1])})
            except (ValueError, IndexError):
                continue

        if len(data) > 20:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            all_prices[stock] = df['close']

        if i % 100 == 99:
            time.sleep(2)

    bs.logout()

    price_df = pd.DataFrame(all_prices).sort_index()
    price_df = price_df[price_df.index >= '2021-01-01']
    price_df = price_df.ffill(limit=2)
    price_df.to_pickle(cache_path)
    print(f"Price matrix: {price_df.shape[0]} weeks x {price_df.shape[1]} stocks")
    return price_df


def build_constituent_mask(price_df, const):
    """Build boolean mask: True if stock is in CSI300 at that week"""
    mask = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
    rebal_dates = const['dates']
    constituents = const['constituents']

    for idx, date in enumerate(price_df.index):
        date_str = date.strftime('%Y-%m-%d')
        active_date = None
        for d in rebal_dates:
            if d <= date_str:
                active_date = d
            else:
                break
        if active_date is None:
            active_date = rebal_dates[0]
        active_stocks = set(constituents[active_date]) & set(price_df.columns)
        mask.loc[date, list(active_stocks)] = True

    return mask


# ============================================================
# Strategy Core: Value Reversal + Quality (on CSI300 constituents)
# ============================================================

def select_value_quality(price_df: pd.DataFrame, mask: pd.DataFrame,
                         industries: dict, idx: int) -> Tuple[List[str], List[dict]]:
    """
    逆向价值+质量选股 (严格无前瞻, 无幸存者偏差)

    仅在当期CSI300成分股中选股 (通过mask过滤)
    """
    vlb = PARAMS['value_lookback']
    vol_lb = PARAMS['vol_lookback']
    vw = PARAMS['vol_weight']
    top_n = PARAMS['top_n']
    ddf = PARAMS['max_drawdown_filter']

    if idx < max(vlb, vol_lb) + 2:
        return [], []

    # Only consider stocks in CSI300 at this point in time
    active = mask.iloc[idx]
    active_stocks = [col for col in price_df.columns if active.get(col, False)]

    if len(active_stocks) < top_n:
        return [], []

    returns = price_df.pct_change(fill_method=None)

    # Calculate factors for active constituent stocks
    scores = {}
    for col in active_stocks:
        if (idx - vlb < 0 or
            pd.isna(price_df[col].iloc[idx]) or
            pd.isna(price_df[col].iloc[idx - vlb]) or
            price_df[col].iloc[idx - vlb] <= 0):
            continue

        # Value factor: negative momentum = high value
        mom_10w = float(price_df[col].iloc[idx] / price_df[col].iloc[idx - vlb] - 1)

        # Quality factor: low volatility
        ret_slice = returns[col].iloc[max(0, idx-vol_lb):idx+1].dropna()
        if len(ret_slice) < 4:
            continue
        vol = float(ret_slice.std())

        # Safety filter: exclude stocks crashing >20% in 4 weeks
        if idx >= 4:
            mom_4w = float(price_df[col].iloc[idx] / price_df[col].iloc[idx-4] - 1)
            if mom_4w < ddf:
                continue
        else:
            mom_4w = 0.0

        scores[col] = {
            'value_score': -mom_10w,
            'quality_score': -vol,
            'mom_10w': mom_10w,
            'mom_4w': mom_4w,
            'vol_12w': vol,
        }

    if len(scores) < top_n:
        return [], []

    # Rank and composite
    tickers = list(scores.keys())
    value_ranks = pd.Series({t: scores[t]['value_score'] for t in tickers}).rank(ascending=False)
    quality_ranks = pd.Series({t: scores[t]['quality_score'] for t in tickers}).rank(ascending=False)
    composite = (1 - vw) * value_ranks + vw * quality_ranks

    # Sector constraint
    sector_max = PARAMS.get('sector_max')
    if sector_max:
        sorted_tickers = list(composite.sort_values().index)
        selected = []
        sector_count = defaultdict(int)
        for t in sorted_tickers:
            if len(selected) >= top_n:
                break
            sec = industries.get(t, {}).get('industry', '?')
            if sector_count[sec] < sector_max:
                selected.append(t)
                sector_count[sec] += 1
    else:
        selected = list(composite.nsmallest(top_n).index)

    # Build details
    details = []
    for rank_i, t in enumerate(composite.sort_values().index[:30]):
        info = industries.get(t, {'name': t, 'industry': '?'})
        s = scores[t]
        details.append({
            'ticker': t,
            'name': info.get('name', '?'),
            'code': t.split('.')[1] if '.' in t else t,
            'sector': info.get('industry', '?'),
            'mom_10w': round(s['mom_10w'] * 100, 1),
            'mom_4w': round(s['mom_4w'] * 100, 1),
            'vol_12w': round(s['vol_12w'] * 100, 2),
            'value_rank': int(value_ranks[t]),
            'quality_rank': int(quality_ranks[t]),
            'composite_rank': rank_i + 1,
            'selected': t in selected,
            'position_pct': round(100 / top_n, 1) if t in selected else 0.0,
        })

    return selected, details


# ============================================================
# Signal Generation
# ============================================================

def generate_signal(price_df: pd.DataFrame, mask: pd.DataFrame,
                    industries: dict) -> dict:
    """Generate current signal"""
    idx = len(price_df) - 1
    latest_date = price_df.index[idx]

    selected, details = select_value_quality(price_df, mask, industries, idx)

    signal = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': 'Strategy E V3 (逆向价值+质量, CSI300成分股, 无幸存者偏差)',
        'variant': f"Value{PARAMS['value_lookback']}w VW{PARAMS['vol_weight']} Top{PARAMS['top_n']}",
        'rebalance_freq': f"每{PARAMS['rebal_freq']}周(月度)",
        'universe': 'CSI300 real constituents (baostock)',
        'total_stocks': len(selected),
        'position_per_stock': round(100 / PARAMS['top_n'], 1) if selected else 0,
        'selected_stocks': [],
        'all_rankings': details,
        'action_summary': [],
    }

    for d in details:
        if d['selected']:
            signal['selected_stocks'].append({
                'code': d['code'],
                'name': d['name'],
                'sector': d['sector'],
                'mom_10w': d['mom_10w'],
                'vol_12w': d['vol_12w'],
            })

    # Summary text
    lines = [
        f"逻辑: 在CSI300成分股中寻找近期被低估的优质股 (无幸存者偏差)",
        f"",
        f"本期持仓 ({len(selected)}只, 等权{round(100/PARAMS['top_n'],1)}%每只):",
        f"",
    ]
    for d in details:
        if d['selected']:
            lines.append(f"  >> #{d['composite_rank']} {d['code']} {d['name']} [{d['sector']}] "
                        f"10w:{d['mom_10w']:+.1f}% 4w:{d['mom_4w']:+.1f}% Vol:{d['vol_12w']:.1f}%")

    lines.append("")
    lines.append(f"注: 已过滤4周跌幅>{abs(PARAMS['max_drawdown_filter'])*100:.0f}%的股票, "
                 f"每行业最多{PARAMS['sector_max']}只")

    signal['action_summary'] = lines
    return signal


# ============================================================
# Backtest
# ============================================================

def run_backtest(price_df: pd.DataFrame, mask: pd.DataFrame,
                 industries: dict) -> dict:
    print("\n运行回测 (CSI300成分股, 无幸存者偏差)...")
    txn_cost = PARAMS['txn_cost_bps'] / 10000
    returns = price_df.pct_change(fill_method=None)
    warmup = max(PARAMS['value_lookback'], PARAMS['vol_lookback'], PARAMS['warmup']) + 5

    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        selected, _ = select_value_quality(price_df, mask, industries, i)
        if not selected:
            i += 1
            continue

        selected_set = set(selected)
        new_buys = selected_set - prev_holdings
        sold = prev_holdings - selected_set
        turnover = (len(new_buys) + len(sold)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        hold_end = min(i + PARAMS['rebal_freq'], len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            rets = []
            for s in selected:
                r = returns[s].iloc[j]
                if not pd.isna(r):
                    rets.append(float(r))
            port_ret = np.mean(rets) if rets else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    if not dates:
        return {'error': 'No trades'}

    nav_s = pd.Series(nav[1:], index=dates)
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / years) - 1
    dd = nav_s / nav_s.cummax() - 1
    mdd = dd.min()
    wr = pd.Series(weekly_rets)
    sharpe = (wr.mean() * 52 - 0.025) / (wr.std() * np.sqrt(52)) if wr.std() > 0 else 0  # rf=2.5%
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    win_rate = (wr > 0).sum() / len(wr) * 100

    annual = nav_s.resample('YE').last().pct_change().dropna()
    annual_returns = {str(d.year): round(v * 100, 1) for d, v in annual.items()}

    result = {
        'strategy': f"Strategy E V3: 逆向价值+质量 CSI300 Top{PARAMS['top_n']}",
        'universe': 'CSI300 real constituents (baostock, no survivorship bias)',
        'period': f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}",
        'years': round(years, 1),
        'cagr_pct': round(cagr * 100, 1),
        'total_return_pct': round((nav_s.iloc[-1] - 1) * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round(win_rate, 1),
        'annual_returns': annual_returns,
        'total_txn_pct': round(total_txn * 100, 2),
        'note': '无前瞻偏差, 无幸存者偏差: T周决策->T+1周收益, 交易成本8bp, CSI300历史成分股',
    }

    print(f"\n回测结果:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


# ============================================================
# Output
# ============================================================

def format_signal_text(signal: dict) -> str:
    lines = [
        f"{'='*55}",
        f"策略E V3 逆向价值+质量 (CSI300, 无幸存者偏差)",
        f"日期: {signal['date']}",
        f"调仓频率: {signal['rebalance_freq']}",
        f"{'='*55}",
        "",
    ]
    for line in signal['action_summary']:
        lines.append(line)
    lines.extend([
        "",
        f"参数: {signal['variant']}",
        f"生成时间: {signal['generated_at']}",
    ])
    return '\n'.join(lines)


def load_data():
    """Load all CSI300 data"""
    print("Loading CSI300 data for Strategy E V3...")
    const = fetch_constituents()
    all_stocks = const['all_unique_stocks']
    print(f"  {len(all_stocks)} unique stocks")

    industries = fetch_industries(all_stocks)
    price_df = fetch_weekly_prices(all_stocks)
    mask = build_constituent_mask(price_df, const)
    return price_df, mask, industries, const


def main():
    args = sys.argv[1:]
    json_only = '--json' in args
    do_backtest = '--backtest' in args

    if not json_only:
        print("=" * 55)
        print(f"策略E V3: 逆向价值+质量 (CSI300成分股, 无幸存者偏差)")
        print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 55)

    price_df, mask, industries, const = load_data()

    if do_backtest:
        result = run_backtest(price_df, mask, industries)
        out = os.path.join(DATA_DIR, 'strategy_e_backtest_result.json')
        with open(out, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n已保存: {out}")
        return

    signal = generate_signal(price_df, mask, industries)
    out = os.path.join(DATA_DIR, 'strategy_e_latest_signal.json')
    with open(out, 'w') as f:
        json.dump(signal, f, indent=2, ensure_ascii=False)

    if json_only:
        print(json.dumps(signal, indent=2, ensure_ascii=False))
    else:
        print(format_signal_text(signal))
        print(f"\n已保存: {out}")

    return signal


if __name__ == '__main__':
    main()
