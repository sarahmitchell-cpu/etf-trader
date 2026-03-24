#!/usr/bin/env python3
"""
Strategy G: CSI500 Low Volatility + Low PB Value Strategy (Mean Reversion)

Uses real historical CSI 500 constituents (no survivorship bias) via baostock.
Composite factor: 50% Low Volatility (12w) + 50% Low PB value.

Parameters:
  - vol_lookback: 12 weeks
  - top_n: 10 stocks
  - rebal_freq: every 2 weeks
  - factor weights: low_vol=0.5, low_pb=0.5
  - txn_cost: 8 bps one-way

Backtest (2021-01 ~ 2026-03, ~5 years, real constituents, no survivorship bias):
  - CAGR = 13.5%
  - MDD  = -11.9%
  - Sharpe = 0.794
  - Calmar = 1.132
  - 12-month rolling win rate: 97.5% (only 4 losing windows out of 160)
  - Annual: 2022:+8.8% | 2023:+4.1% | 2024:+38.7% | 2025:+12.0%

Bias Handling:
  - Survivorship bias: eliminated via baostock historical CSI500 constituent lists
    (928 unique stocks across 11 rebalance periods, 2020-12 to 2025-12)
  - Look-ahead bias: factors computed strictly from past data, PE/PB from daily
    data resampled to weekly (last available before rebalance date)
  - Transaction costs: 8 bps per trade (conservative for A-shares)
  - Data coverage: ~400/928 stocks have PE/PB data (baostock limitation)
    Only stocks with both price AND PE/PB data are eligible for selection
  - Rebalance dates aligned with Friday close (weekly data)

Strategy Rationale:
  - CSI 500 mid-cap stocks: pure momentum FAILS (tested, all negative CAGR)
  - Pure mean reversion (buy dips) also fails (dips = real deterioration)
  - Low PB captures "genuinely cheap" stocks (fundamental anchor)
  - Low volatility filters out distressed/speculative names
  - Combination: buy cheap, stable mid-caps = A-share value investing sweet spot

Usage:
  python3 strategy_g_weekly_signal.py              # Generate current signal
  python3 strategy_g_weekly_signal.py --json       # JSON only
  python3 strategy_g_weekly_signal.py --backtest   # Full backtest
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
    'vol_lookback': 12,      # 12-week rolling volatility
    'top_n': 10,             # select top 10 stocks
    'rebal_freq': 2,         # rebalance every 2 weeks
    'txn_cost_bps': 8,       # transaction cost in basis points
    'factor_weights': {
        'low_vol': 0.5,
        'low_pb': 0.5,
    },
    'warmup': 54,            # warmup period (weeks)
}

# CSI 500 rebalance dates (semi-annual)
REBALANCE_DATES = [
    '2020-12-14', '2021-06-15', '2021-12-13', '2022-06-13',
    '2022-12-12', '2023-06-12', '2023-12-11', '2024-06-17',
    '2024-12-16', '2025-06-16', '2025-12-15',
]


# ============================================================
# Data Fetching (with caching)
# ============================================================

def fetch_constituents():
    """Fetch CSI 500 historical constituents from baostock"""
    cache_path = os.path.join(CACHE_DIR, 'csi500_constituents_history.json')
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
        rs = bs.query_zz500_stocks(date=d)
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
    cache_path = os.path.join(CACHE_DIR, 'csi500_industries.json')
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
    cache_path = os.path.join(CACHE_DIR, 'csi500_all_weekly_prices.pkl')
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


def fetch_fundamentals(stock_list):
    """Fetch daily PE/PB and resample to weekly"""
    cache_path = os.path.join(CACHE_DIR, 'csi500_weekly_fundamentals.pkl')

    existing = {}
    if os.path.exists(cache_path):
        old = pd.read_pickle(cache_path)
        if isinstance(old, dict) and len(old.get('pe', {})) > 10:
            existing = old
            print(f"Loaded PE/PB cache: PE={len(existing.get('pe',{}))}, PB={len(existing.get('pb',{}))}")

    to_fetch = [s for s in stock_list if s not in existing.get('pe', {})]
    print(f"Need to fetch PE/PB: {len(to_fetch)} stocks")

    if not to_fetch:
        return existing

    lg = bs.login()
    pe_data = dict(existing.get('pe', {}))
    pb_data = dict(existing.get('pb', {}))
    failed = []
    t0 = time.time()

    for i, stock in enumerate(to_fetch):
        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Fetching PE/PB: {i}/{len(to_fetch)} ({elapsed:.0f}s)...")

        rs = bs.query_history_k_data_plus(
            stock, 'date,peTTM,pbMRQ',
            start_date='2020-09-01', end_date='2026-12-31',
            frequency='d', adjustflag='1'
        )

        dates_list, pe_list, pb_list = [], [], []
        while rs.next():
            row = rs.get_row_data()
            try:
                d = row[0]
                pe = float(row[1]) if row[1] and row[1] != '' else np.nan
                pb = float(row[2]) if row[2] and row[2] != '' else np.nan
                dates_list.append(d)
                pe_list.append(pe)
                pb_list.append(pb)
            except (ValueError, IndexError):
                continue

        if len(dates_list) > 20:
            idx = pd.to_datetime(dates_list)
            weekly_pe = pd.Series(pe_list, index=idx).resample('W-FRI').last()
            weekly_pb = pd.Series(pb_list, index=idx).resample('W-FRI').last()
            pe_data[stock] = weekly_pe.dropna()
            pb_data[stock] = weekly_pb.dropna()
        else:
            failed.append(stock)

        if (i + 1) % 200 == 0:
            pd.to_pickle({'pe': pe_data, 'pb': pb_data}, cache_path)

        if i % 100 == 99:
            time.sleep(1)

    bs.logout()

    result = {'pe': pe_data, 'pb': pb_data}
    pd.to_pickle(result, cache_path)
    print(f"PE/PB done! PE: {len(pe_data)}, PB: {len(pb_data)}, Failed: {len(failed)}")
    return result


# ============================================================
# Constituent Mask
# ============================================================

def build_constituent_mask(price_df, const):
    """Build boolean mask: True if stock is in CSI500 at that week"""
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

def compute_factors(price_df, pb_df):
    """Compute low volatility and low PB factors"""
    returns = price_df.pct_change(fill_method=None)

    # Low Volatility (12-week)
    vol12 = returns.rolling(12, min_periods=8).std() * np.sqrt(52)
    low_vol = -vol12  # negate: lower vol = higher score

    # Low PB (value)
    pb_clean = pb_df.copy()
    pb_clean[pb_clean <= 0] = np.nan
    pb_clean[pb_clean > 30] = np.nan
    low_pb = -pb_clean  # negate: lower PB = higher score

    return low_vol, low_pb


def cross_sectional_zscore(factor_df, mask):
    """Z-score normalize within each cross-section"""
    result = factor_df.copy()
    result[~mask] = np.nan
    row_mean = result.mean(axis=1)
    row_std = result.std(axis=1)
    result = result.sub(row_mean, axis=0).div(row_std.replace(0, np.nan), axis=0)
    return result


# ============================================================
# Signal Generation
# ============================================================

def generate_signal(price_df, mask, pb_df, industries, idx=None):
    """Generate current signal"""
    if idx is None:
        idx = len(price_df) - 1

    low_vol, low_pb = compute_factors(price_df, pb_df)
    low_vol_z = cross_sectional_zscore(low_vol, mask)
    low_pb_z = cross_sectional_zscore(low_pb, mask)

    # Composite score
    w = PARAMS['factor_weights']
    score_df = low_vol_z * w['low_vol'] + low_pb_z * w['low_pb']

    scores = score_df.iloc[idx].copy()
    active = mask.iloc[idx]
    scores[~active] = np.nan
    scores = scores.dropna()

    if len(scores) < PARAMS['top_n']:
        return None

    top_n = PARAMS['top_n']
    selected_stocks = scores.nlargest(top_n).index.tolist()

    current_date = price_df.index[idx]
    returns = price_df.pct_change(fill_method=None)
    vol_raw = returns.rolling(PARAMS['vol_lookback'], min_periods=8).std() * np.sqrt(52)
    mom_4w = (price_df.iloc[idx] / price_df.iloc[max(0, idx-4)] - 1) if idx >= 4 else pd.Series(dtype=float)
    mom_12w = (price_df.iloc[idx] / price_df.iloc[max(0, idx-12)] - 1) if idx >= 12 else pd.Series(dtype=float)

    all_rankings = []
    for rank_i, stock in enumerate(scores.sort_values(ascending=False).index[:50]):
        info = industries.get(stock, {'name': '?', 'industry': '?'})
        vol_val = vol_raw[stock].iloc[idx] if stock in vol_raw.columns else np.nan
        pb_val = pb_df[stock].iloc[idx] if stock in pb_df.columns else np.nan
        m4 = mom_4w.get(stock, np.nan) if isinstance(mom_4w, pd.Series) else np.nan
        m12 = mom_12w.get(stock, np.nan) if isinstance(mom_12w, pd.Series) else np.nan

        all_rankings.append({
            'rank': rank_i + 1,
            'code': stock,
            'name': info.get('name', '?'),
            'industry': info.get('industry', '?'),
            'vol_12w_ann': round(float(vol_val * 100), 1) if pd.notna(vol_val) else None,
            'pb': round(float(pb_val), 2) if pd.notna(pb_val) else None,
            'mom_4w_pct': round(float(m4 * 100), 1) if pd.notna(m4) else None,
            'mom_12w_pct': round(float(m12 * 100), 1) if pd.notna(m12) else None,
            'composite_score': round(float(scores[stock]), 3),
            'selected': stock in selected_stocks,
        })

    signal = {
        'date': current_date.strftime('%Y-%m-%d'),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': 'Strategy G (CSI500 LV+PB Mean Reversion)',
        'description': 'Select 10 lowest-vol + lowest-PB stocks from CSI 500 real constituents',
        'params': PARAMS,
        'active_constituents': int(active.sum()),
        'scored_stocks': len(scores),
        'selected_count': len(selected_stocks),
        'position_per_stock_pct': round(100 / len(selected_stocks), 1),
        'selected_stocks': [r for r in all_rankings if r['selected']],
        'all_rankings': all_rankings,
    }

    lines = [
        f"CSI500低波价值策略 | {current_date.strftime('%Y-%m-%d')}",
        f"当前成分股: {int(active.sum())}只 | 可评分: {len(scores)}只",
        f"选股: 50%低波动+50%低PB | Top {top_n} | 等权配置",
        "",
        "本期选股:",
    ]
    for r in signal['selected_stocks']:
        pb_str = f"PB={r['pb']}" if r['pb'] else "PB=N/A"
        vol_str = f"波动率={r['vol_12w_ann']}%" if r['vol_12w_ann'] else "波动率=N/A"
        lines.append(f"  {r['name']}({r['code']}) [{r['industry']}] {vol_str} {pb_str}")

    signal['summary_cn'] = '\n'.join(lines)
    return signal


# ============================================================
# Backtest Engine
# ============================================================

def run_backtest(price_df, mask, pb_df, industries):
    """Full backtest"""
    low_vol, low_pb = compute_factors(price_df, pb_df)
    low_vol_z = cross_sectional_zscore(low_vol, mask)
    low_pb_z = cross_sectional_zscore(low_pb, mask)

    w = PARAMS['factor_weights']
    score_df = low_vol_z * w['low_vol'] + low_pb_z * w['low_pb']

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
        scores = score_df.iloc[i].copy()
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
    print("Loading CSI500 data...")
    const = fetch_constituents()
    all_stocks = const['all_unique_stocks']
    print(f"  {len(all_stocks)} unique stocks")

    industries = fetch_industries(all_stocks)
    price_df = fetch_weekly_prices(all_stocks)

    print("Loading PE/PB fundamentals...")
    fund = fetch_fundamentals(all_stocks)
    pb_dict = fund.get('pb', {})
    pb_df = pd.DataFrame(pb_dict).sort_index()
    pb_df = pb_df.reindex(price_df.index, method='ffill')
    print(f"  PB matrix: {pb_df.shape}")

    mask = build_constituent_mask(price_df, const)
    return price_df, mask, pb_df, industries, const


def main():
    mode = 'signal'
    if '--backtest' in sys.argv:
        mode = 'backtest'
    json_only = '--json' in sys.argv

    price_df, mask, pb_df, industries, const = load_data()

    if mode == 'backtest':
        print("\n" + "=" * 60)
        print("Strategy G (CSI500 LV+PB) - Full Backtest")
        print("=" * 60)
        nav, wr, txn = run_backtest(price_df, mask, pb_df, industries)
        stats = calc_stats(nav, wr, txn)
        if stats:
            print(f"\nResults:")
            for k, v in stats.items():
                print(f"  {k}: {v}")

        out_path = os.path.join(DATA_DIR, 'strategy_g_backtest.json')
        with open(out_path, 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {out_path}")
    else:
        signal = generate_signal(price_df, mask, pb_df, industries)
        if signal is None:
            print("ERROR: Could not generate signal")
            sys.exit(1)

        out_path = os.path.join(DATA_DIR, 'strategy_g_latest_signal.json')
        with open(out_path, 'w') as f:
            json.dump(signal, f, indent=2, ensure_ascii=False)

        if json_only:
            print(json.dumps(signal, indent=2, ensure_ascii=False))
        else:
            print(signal['summary_cn'])
            print(f"\nSignal saved to {out_path}")


if __name__ == '__main__':
    main()
