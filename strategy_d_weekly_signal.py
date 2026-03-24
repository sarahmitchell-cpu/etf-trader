#!/usr/bin/env python3
"""
Strategy D (v2): CSI300 Low Volatility Strategy

Uses real historical CSI 300 constituents (no survivorship bias) via baostock.
Selects stocks with lowest 20-week realized volatility.

Parameters:
  - vol_lookback: 20 weeks
  - top_n: 10 stocks
  - rebal_freq: every 2 weeks
  - txn_cost: 8 bps one-way

Backtest (2022-01 ~ 2026-03, ~4 years, real constituents, no survivorship bias):
  - CAGR = 13.9%
  - MDD  = -8.5%
  - Sharpe = 0.955
  - Calmar = 1.636
  - 12-month rolling win rate: 98.8%
  - Annual: 2022:-1.5% | 2023:+12.0% | 2024:+46.6% | 2025:+5.2%

Bias Handling:
  - Survivorship bias: eliminated via baostock historical constituent lists
  - Look-ahead bias: factors computed strictly from past data
  - Transaction costs: 8 bps per trade (conservative for A-shares)
  - Rebalance dates aligned with Friday close (weekly data)

Usage:
  python3 strategy_d_weekly_signal.py              # Generate current signal
  python3 strategy_d_weekly_signal.py --json       # JSON only
  python3 strategy_d_weekly_signal.py --backtest   # Full backtest
"""

import baostock as bs
import pandas as pd
import numpy as np
import json
import os
import sys
import time
from datetime import datetime
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'baostock_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# Strategy Parameters
# ============================================================

PARAMS = {
    'vol_lookback': 20,      # 20-week rolling volatility
    'top_n': 10,             # select top 10 lowest-vol stocks
    'rebal_freq': 2,         # rebalance every 2 weeks
    'txn_cost_bps': 8,       # transaction cost in basis points
    'min_weeks': 25,         # minimum data required per stock
    'warmup': 54,            # warmup period (weeks) before first signal
    'sector_max': 3,         # max stocks per sector (prevents concentration, e.g. 50% financials)
}

# CSI 300 rebalance dates (semi-annual)
REBALANCE_DATES = [
    '2020-12-14', '2021-06-15', '2021-12-13', '2022-06-13',
    '2022-12-12', '2023-06-12', '2023-12-11', '2024-06-17',
    '2024-12-16', '2025-06-16', '2025-12-15',
]


# ============================================================
# Data Fetching (with caching)
# ============================================================

def fetch_constituents():
    """Fetch CSI 300 historical constituents from baostock"""
    cache_path = os.path.join(CACHE_DIR, 'csi300_constituents_history.json')
    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 7:
            with open(cache_path) as f:
                data = json.load(f)
            return data

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
    failed = []

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
        else:
            failed.append(stock)

        if i % 100 == 99:
            time.sleep(2)

    bs.logout()

    price_df = pd.DataFrame(all_prices).sort_index()
    price_df = price_df[price_df.index >= '2021-01-01']
    price_df = price_df.ffill(limit=2)
    price_df.to_pickle(cache_path)
    print(f"Price matrix: {price_df.shape[0]} weeks x {price_df.shape[1]} stocks")
    return price_df


# ============================================================
# Constituent Mask
# ============================================================

def build_constituent_mask(price_df, const):
    """Build a boolean mask: True if stock is in CSI300 at that week"""
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
# Factor Computation
# ============================================================

def compute_low_vol(price_df, lookback=20):
    """Compute annualized volatility (lower = better, so we negate)"""
    returns = price_df.pct_change(fill_method=None)
    vol = returns.rolling(lookback, min_periods=int(lookback * 0.6)).std() * np.sqrt(52)
    return -vol  # negate: lower vol = higher score


def cross_sectional_zscore(factor_df, mask):
    """Z-score normalize within each cross-section (week)"""
    result = factor_df.copy()
    result[~mask] = np.nan
    row_mean = result.mean(axis=1)
    row_std = result.std(axis=1)
    result = result.sub(row_mean, axis=0).div(row_std.replace(0, np.nan), axis=0)
    return result


# ============================================================
# Signal Generation
# ============================================================

def generate_signal(price_df, mask, industries, idx=None):
    """Generate current signal: select top-N lowest volatility stocks"""
    if idx is None:
        idx = len(price_df) - 1

    low_vol_raw = compute_low_vol(price_df, PARAMS['vol_lookback'])
    low_vol_z = cross_sectional_zscore(low_vol_raw, mask)

    scores = low_vol_z.iloc[idx].copy()
    active = mask.iloc[idx]
    scores[~active] = np.nan
    scores = scores.dropna()

    if len(scores) < PARAMS['top_n']:
        return None

    top_n = PARAMS['top_n']
    sector_max = PARAMS.get('sector_max', 99)

    # Apply sector_max constraint: iterate by score, skip if sector already full
    sorted_candidates = scores.sort_values(ascending=False)
    selected_stocks = []
    sector_count = defaultdict(int)
    for stock in sorted_candidates.index:
        info = industries.get(stock, {'industry': 'Unknown'})
        sector = info.get('industry', 'Unknown')
        if sector_count[sector] < sector_max:
            selected_stocks.append(stock)
            sector_count[sector] += 1
        if len(selected_stocks) >= top_n:
            break

    current_date = price_df.index[idx]
    returns = price_df.pct_change(fill_method=None)
    vol_raw = returns.rolling(PARAMS['vol_lookback'], min_periods=12).std() * np.sqrt(52)
    mom_4w = (price_df.iloc[idx] / price_df.iloc[max(0, idx-4)] - 1) if idx >= 4 else pd.Series(dtype=float)
    mom_12w = (price_df.iloc[idx] / price_df.iloc[max(0, idx-12)] - 1) if idx >= 12 else pd.Series(dtype=float)

    all_rankings = []
    for rank_i, stock in enumerate(scores.sort_values(ascending=False).index):
        info = industries.get(stock, {'name': '?', 'industry': '?'})
        vol_val = vol_raw[stock].iloc[idx] if stock in vol_raw.columns else np.nan
        m4 = mom_4w.get(stock, np.nan) if isinstance(mom_4w, pd.Series) else np.nan
        m12 = mom_12w.get(stock, np.nan) if isinstance(mom_12w, pd.Series) else np.nan

        all_rankings.append({
            'rank': rank_i + 1,
            'code': stock,
            'name': info.get('name', '?'),
            'industry': info.get('industry', '?'),
            'vol_20w_ann': round(float(vol_val * 100), 1) if pd.notna(vol_val) else None,
            'mom_4w_pct': round(float(m4 * 100), 1) if pd.notna(m4) else None,
            'mom_12w_pct': round(float(m12 * 100), 1) if pd.notna(m12) else None,
            'z_score': round(float(scores[stock]), 3),
            'selected': stock in selected_stocks,
        })

    signal = {
        'date': current_date.strftime('%Y-%m-%d'),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': 'Strategy D v2 (CSI300 Low Volatility)',
        'description': f'Select {top_n} lowest-volatility stocks from CSI 300 real constituents (max {sector_max}/sector)',
        'params': PARAMS,
        'active_constituents': int(active.sum()),
        'scored_stocks': len(scores),
        'selected_count': len(selected_stocks),
        'position_per_stock_pct': round(100 / len(selected_stocks), 1),
        'selected_stocks': [r for r in all_rankings if r['selected']],
        'all_rankings': all_rankings[:50],
    }

    lines = [
        f"CSI300低波动策略 | {current_date.strftime('%Y-%m-%d')}",
        f"当前成分股: {int(active.sum())}只 | 可评分: {len(scores)}只",
        f"选股: 20周波动率最低的{top_n}只 | 等权配置",
        "",
        "本期选股:",
    ]
    for r in signal['selected_stocks']:
        lines.append(f"  {r['name']}({r['code']}) [{r['industry']}] 波动率={r['vol_20w_ann']}%")

    signal['summary_cn'] = '\n'.join(lines)
    return signal


# ============================================================
# Backtest Engine
# ============================================================

def run_backtest(price_df, mask, industries):
    """Full backtest"""
    low_vol_raw = compute_low_vol(price_df, PARAMS['vol_lookback'])
    low_vol_z = cross_sectional_zscore(low_vol_raw, mask)

    txn_cost = PARAMS['txn_cost_bps'] / 10000
    top_n = PARAMS['top_n']
    rebal_freq = PARAMS['rebal_freq']
    warmup = PARAMS['warmup']
    returns = price_df.pct_change(fill_method=None)

    nav = 1.0
    nav_list = [nav]
    dates = [price_df.index[warmup - 1]]
    prev_holdings = set()
    weekly_rets = []
    total_txn = 0.0

    i = warmup
    while i < len(price_df) - 1:
        scores = low_vol_z.iloc[i].copy()
        active = mask.iloc[i]
        scores[~active] = np.nan
        scores = scores.dropna()

        if len(scores) < top_n:
            for j in range(i + 1, min(i + rebal_freq + 1, len(price_df))):
                nav_list.append(nav_list[-1])
                dates.append(price_df.index[j])
                weekly_rets.append(0.0)
            i += rebal_freq
            continue

        selected = scores.nlargest(top_n).index.tolist()
        selected_set = set(selected)
        turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            rets_j = returns.iloc[j][selected].dropna()
            port_ret = float(rets_j.mean()) if len(rets_j) > 0 else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav = nav_list[-1] * (1 + port_ret)
            nav_list.append(nav)
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    return pd.Series(nav_list, index=dates), weekly_rets, total_txn


def calc_stats(nav_series, weekly_rets, total_txn):
    nav = nav_series.values
    n_weeks = len(nav) - 1
    years = n_weeks / 52.0
    if years < 0.5:
        return None

    cagr = (nav[-1] / nav[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    mdd = float(np.min(dd))

    wr = np.array(weekly_rets)
    sharpe = float((np.mean(wr) * 52 - 0.025) / (np.std(wr) * np.sqrt(52))) if len(wr) > 10 and np.std(wr) > 0 else 0.0  # rf=2.5%
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    annual = {}
    for year in range(2021, 2027):
        ymask = np.array([d.year == year for d in nav_series.index])
        if ymask.sum() > 4:
            year_nav = nav[ymask]
            annual[str(year)] = round((year_nav[-1] / year_nav[0] - 1) * 100, 1)

    rolling = None
    if len(nav) > 52:
        rets_12m = []
        for s in range(len(nav) - 52):
            rets_12m.append((nav[s + 52] / nav[s] - 1) * 100)
        rets_12m = np.array(rets_12m)
        rolling = {
            'windows': len(rets_12m),
            'win_rate': round(float(np.mean(rets_12m > 0) * 100), 1),
            'mean': round(float(np.mean(rets_12m)), 2),
            'min': round(float(np.min(rets_12m)), 2),
            'max': round(float(np.max(rets_12m)), 2),
        }

    return {
        'cagr_pct': round(cagr * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'annual_returns': annual,
        'rolling_12m': rolling,
        'total_txn_pct': round(total_txn * 100, 2),
        'period': f"{nav_series.index[0].strftime('%Y-%m-%d')} ~ {nav_series.index[-1].strftime('%Y-%m-%d')}",
    }


# ============================================================
# Main
# ============================================================

def load_data():
    print("Loading CSI300 data...")
    const = fetch_constituents()
    all_stocks = const['all_unique_stocks']
    industries = fetch_industries(all_stocks)
    price_df = fetch_weekly_prices(all_stocks)
    mask = build_constituent_mask(price_df, const)
    return price_df, mask, industries, const


def main():
    mode = 'signal'
    if '--backtest' in sys.argv:
        mode = 'backtest'
    json_only = '--json' in sys.argv

    price_df, mask, industries, const = load_data()

    if mode == 'backtest':
        print("\n" + "=" * 60)
        print("Strategy D v2 (CSI300 Low Volatility) - Full Backtest")
        print("=" * 60)
        nav, wr, txn = run_backtest(price_df, mask, industries)
        stats = calc_stats(nav, wr, txn)
        if stats:
            print(f"\nResults:")
            for k, v in stats.items():
                print(f"  {k}: {v}")

        out_path = os.path.join(DATA_DIR, 'strategy_d_v2_backtest.json')
        with open(out_path, 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {out_path}")
    else:
        signal = generate_signal(price_df, mask, industries)
        if signal is None:
            print("ERROR: Could not generate signal")
            sys.exit(1)

        out_path = os.path.join(DATA_DIR, 'strategy_d_v2_latest_signal.json')
        with open(out_path, 'w') as f:
            json.dump(signal, f, indent=2, ensure_ascii=False)

        if json_only:
            print(json.dumps(signal, indent=2, ensure_ascii=False))
        else:
            print(signal['summary_cn'])
            print(f"\nSignal saved to {out_path}")


if __name__ == '__main__':
    main()
