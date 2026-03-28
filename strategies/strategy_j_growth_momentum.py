#!/usr/bin/env python3
"""
Strategy J: 成长+动量双因子选股 (Growth-Momentum Dual Factor)

灵感来源: 创成长指数, 扩展到沪深300+中证500中大盘股票池
核心逻辑: 选择半年期EPS+营收增长最快的中大盘股票

因子构建:
  - EPS增长率(HoH): 从PE_TTM和股价推导, EPS = Price/PE_TTM, 半年变化率
  - 营收增长率(HoH): 从PS_TTM和股价推导, Rev = Price/PS_TTM, 半年变化率
  - 综合排名: EPS排名 + 营收排名 等权平均

回测结果 (2021~2026, CSI300+CSI500):
  Top10 月调仓: 年化33.9% 回撤-28.8% Sharpe=1.13 Calmar=1.18
  Top15 月调仓: 年化26.2% 回撤-28.0% Sharpe=0.97 Calmar=0.94
  Top20 月调仓: 年化24.2% 回撤-23.6% Sharpe=0.95 Calmar=1.02

执行参数:
  - 股票池: 沪深300 + 中证500 (~700只)
  - 选股: Top15 综合成长排名
  - 权重: 等权
  - 调仓: 每月(4周)
  - 数据: baostock周线 + 日度估值(PE_TTM, PS_TTM)

数据源: SQLite (已缓存)
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import os
import sys
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DB_PATH = os.path.join(DATA_DIR, 'stock_data.db')

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'top_n': 15,           # Number of stocks to hold
    'rebal_weeks': 4,      # Rebalance every N weeks
    'growth_lookback': 26, # HoH = 26 weeks (~6 months)
    'min_pe': 0,           # Only positive PE (profitable companies)
    'capital': 1_000_000,  # Default capital for position sizing
}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_constituents():
    """Get CSI300 + CSI500 constituents with names."""
    conn = get_db()
    stocks_300 = pd.read_sql(
        'SELECT stock_code, stock_name FROM index_constituents WHERE index_code="000300"', conn
    )
    stocks_500 = pd.read_sql(
        'SELECT stock_code, stock_name FROM index_constituents WHERE index_code="000905"', conn
    )
    conn.close()
    all_stocks = pd.concat([stocks_300, stocks_500]).drop_duplicates('stock_code')
    return all_stocks


def load_data(lookback_months=12):
    """Load price and valuation data from SQLite."""
    conn = get_db()
    start_date = (datetime.now() - timedelta(days=lookback_months * 31)).strftime('%Y-%m-%d')

    all_stocks = get_constituents()
    all_codes = list(all_stocks['stock_code'])

    # Weekly prices
    weekly = pd.read_sql(
        f'SELECT code, date, close FROM stock_weekly WHERE date >= "{start_date}"', conn
    )
    weekly['date'] = pd.to_datetime(weekly['date'])
    weekly = weekly[weekly['code'].isin(all_codes)]
    price_pivot = weekly.pivot(index='date', columns='code', values='close').sort_index()

    # Daily valuations -> resample to weekly
    val = pd.read_sql(
        f'SELECT code, date, pe_ttm, ps_ttm FROM stock_daily_valuation WHERE date >= "{start_date}"', conn
    )
    val['date'] = pd.to_datetime(val['date'])
    val = val[val['code'].isin(all_codes)]
    val['week'] = val['date'].dt.to_period('W').dt.end_time.dt.normalize()
    val_weekly = val.groupby(['code', 'week']).last().reset_index()

    pe_pivot = val_weekly.pivot(index='week', columns='code', values='pe_ttm').sort_index()
    ps_pivot = val_weekly.pivot(index='week', columns='code', values='ps_ttm').sort_index()

    # Align
    common = list(set(price_pivot.columns) & set(pe_pivot.columns))
    pe_al = pe_pivot.reindex(price_pivot.index, method='ffill')[common]
    ps_al = ps_pivot.reindex(price_pivot.index, method='ffill')[common]
    price_df = price_pivot[common]

    conn.close()
    return price_df, pe_al, ps_al, all_stocks


def compute_growth_signal(price_df, pe_df, ps_df, lookback=26):
    """
    Compute HoH growth signal.
    EPS = Price / PE_TTM (only positive PE)
    Rev = Price / PS_TTM (only positive PS)
    Growth = current / lookback_weeks_ago - 1
    Signal = average percentile rank of EPS growth + Revenue growth
    """
    pe_clean = pe_df.replace(0, np.nan).where(pe_df > 0)
    ps_clean = ps_df.replace(0, np.nan).where(ps_df > 0)

    eps = price_df / pe_clean
    rev = price_df / ps_clean

    eps_growth = (eps / eps.shift(lookback) - 1).clip(-5, 50)
    rev_growth = (rev / rev.shift(lookback) - 1).clip(-5, 50)

    eps_rank = eps_growth.rank(axis=1, pct=True)
    rev_rank = rev_growth.rank(axis=1, pct=True)
    composite = (eps_rank + rev_rank) / 2

    return composite, eps_growth, rev_growth


def get_current_picks(top_n=None):
    """Get current top stock picks based on latest growth signal."""
    if top_n is None:
        top_n = CONFIG['top_n']

    price_df, pe_df, ps_df, all_stocks = load_data()
    name_map = dict(zip(all_stocks['stock_code'], all_stocks['stock_name']))

    signal, eps_g, rev_g = compute_growth_signal(
        price_df, pe_df, ps_df, CONFIG['growth_lookback']
    )

    latest = signal.iloc[-1].dropna()
    top = latest.nlargest(top_n)

    picks = []
    for code, score in top.items():
        price = price_df.iloc[-1].get(code, 0)
        eg = eps_g.iloc[-1].get(code, 0)
        rg = rev_g.iloc[-1].get(code, 0)
        picks.append({
            'code': code,
            'name': name_map.get(code, '?'),
            'price': round(float(price), 2) if not np.isnan(price) else 0,
            'eps_growth_pct': round(float(eg) * 100, 1) if not np.isnan(eg) else 0,
            'rev_growth_pct': round(float(rg) * 100, 1) if not np.isnan(rg) else 0,
            'composite_rank': round(float(score), 4),
        })

    return {
        'date': price_df.index[-1].strftime('%Y-%m-%d'),
        'strategy': 'J',
        'description': 'Growth HoH (EPS+Revenue) Top' + str(top_n),
        'top_n': top_n,
        'rebal_weeks': CONFIG['rebal_weeks'],
        'picks': picks,
    }


def position_sizing(picks, capital=None):
    """Calculate position sizes for equal-weight portfolio."""
    if capital is None:
        capital = CONFIG['capital']

    per_stock = capital / len(picks)
    positions = []
    for p in picks:
        if p['price'] > 0:
            shares = int(per_stock / p['price'] / 100) * 100  # Round to 100
            actual = shares * p['price']
        else:
            shares = 0
            actual = 0
        positions.append({
            **p,
            'target_shares': shares,
            'target_amount': round(actual, 0),
        })
    return positions


def main():
    """Generate current signal and print portfolio."""
    print("=" * 70)
    print("Strategy J: 成长因子选股 (Growth HoH)")
    print("股票池: 沪深300 + 中证500")
    print("=" * 70)

    result = get_current_picks()
    positions = position_sizing(result['picks'])

    print(f"\n信号日期: {result['date']}")
    print(f"选股数: {result['top_n']}, 调仓周期: {result['rebal_weeks']}周")
    print(f"本金: {CONFIG['capital']/10000:.0f}万")
    print()

    print(f"{'#':>2} {'代码':>8} {'名称':<10} {'现价':>8} {'股数':>6} {'金额':>8} "
          f"{'EPS增长':>8} {'营收增长':>8} {'排名':>6}")
    print("-" * 80)

    total_invested = 0
    for i, p in enumerate(positions, 1):
        total_invested += p['target_amount']
        print(f"{i:2d} {p['code']:>8} {p['name']:<10} {p['price']:8.2f} "
              f"{p['target_shares']:6d} {p['target_amount']/10000:7.2f}万 "
              f"{p['eps_growth_pct']:+7.1f}% {p['rev_growth_pct']:+7.1f}% "
              f"{p['composite_rank']:6.4f}")

    print(f"\n总投入: {total_invested/10000:.2f}万 / {CONFIG['capital']/10000:.0f}万")

    # Save to JSON
    output = {
        **result,
        'positions': positions,
        'generated': datetime.now().isoformat(),
    }
    output_path = os.path.join(DATA_DIR, 'strategy_j_latest_signal.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
