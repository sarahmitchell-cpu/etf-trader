#!/usr/bin/env python3
"""
Momentum Strategy Fair Backtest (NO Survivorship Bias)

Fix: Use point-in-time CSI300+CSI500 constituents at each semi-annual rebalance.
Data: baostock (free), weekly prices, as far back as possible (2010+).

Strategies tested:
1. Pure momentum (price return) Top10/15/20, various lookbacks
2. Equal-weight benchmark (point-in-time constituents)
3. Biased versions (current constituents) for comparison

Output: /Users/claw/etf-trader/data/momentum_fair_backtest.json
"""

import baostock as bs
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import pickle
from datetime import datetime, timedelta
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'baostock_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Semi-annual rebalance dates from 2010 to 2025
# CSI300 & CSI500 rebalance in June and December each year
REBALANCE_DATES = []
for year in range(2010, 2026):
    REBALANCE_DATES.append(f'{year}-06-15')
    REBALANCE_DATES.append(f'{year}-12-15')
# Add 2026 if needed
# REBALANCE_DATES.append('2026-06-15')

print(f"Total rebalance dates: {len(REBALANCE_DATES)}")
print(f"Range: {REBALANCE_DATES[0]} to {REBALANCE_DATES[-1]}")


# ============================================================
# Data Fetching
# ============================================================

def fetch_all_historical_constituents(force=False):
    """Fetch CSI300+CSI500 constituents at every semi-annual rebalance date."""
    cache_path = os.path.join(CACHE_DIR, 'momentum_fair_constituents.json')
    if not force and os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 30:
            with open(cache_path) as f:
                data = json.load(f)
            print(f"Loaded cached constituents: {len(data['dates'])} dates, {data['total_unique']} unique stocks")
            return data

    lg = bs.login()
    if lg.error_code != '0':
        print(f"Login failed: {lg.error_msg}")
        sys.exit(1)

    constituent_map = {}
    all_stocks = set()

    for d in REBALANCE_DATES:
        stocks_300 = set()
        stocks_500 = set()

        # CSI300
        rs = bs.query_hs300_stocks(date=d)
        while rs.next():
            row = rs.get_row_data()
            stocks_300.add(row[1])

        # CSI500
        rs = bs.query_zz500_stocks(date=d)
        while rs.next():
            row = rs.get_row_data()
            stocks_500.add(row[1])

        combined = stocks_300 | stocks_500
        constituent_map[d] = {
            'csi300': sorted(stocks_300),
            'csi500': sorted(stocks_500),
            'combined': sorted(combined),
        }
        all_stocks.update(combined)
        print(f"  {d}: CSI300={len(stocks_300)}, CSI500={len(stocks_500)}, combined={len(combined)}, overlap={len(stocks_300 & stocks_500)}")

    bs.logout()

    data = {
        'dates': REBALANCE_DATES,
        'constituents': constituent_map,
        'all_unique_stocks': sorted(all_stocks),
        'total_unique': len(all_stocks),
    }
    with open(cache_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"\nTotal unique stocks across all periods: {len(all_stocks)}")
    return data


def get_constituent_at_date(constituent_data, target_date):
    """Get the active constituent list at a given date (point-in-time)."""
    target = pd.Timestamp(target_date)
    dates = sorted(constituent_data['dates'])

    # Find the most recent rebalance date <= target_date
    active_date = None
    for d in dates:
        if pd.Timestamp(d) <= target:
            active_date = d
        else:
            break

    if active_date is None:
        # Before first rebalance, use first available
        active_date = dates[0]

    return set(constituent_data['constituents'][active_date]['combined'])


def fetch_weekly_prices(all_stocks, start_date='2009-01-01', end_date='2026-03-28', force=False):
    """Fetch weekly close prices for all stocks from baostock."""
    cache_path = os.path.join(CACHE_DIR, 'momentum_fair_weekly_prices.pkl')
    if not force and os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 7:
            df = pd.read_pickle(cache_path)
            print(f"Loaded cached weekly prices: {df.shape[0]} weeks x {df.shape[1]} stocks")
            return df

    lg = bs.login()
    if lg.error_code != '0':
        print(f"Login failed: {lg.error_msg}")
        sys.exit(1)

    all_data = {}
    total = len(all_stocks)
    failed = []
    t0 = time.time()

    for i, code in enumerate(all_stocks):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"  Fetching {i+1}/{total} ({code}) ... ETA {eta:.0f}s")

        rs = bs.query_history_k_data_plus(
            code,
            "date,close",
            start_date=start_date,
            end_date=end_date,
            frequency="w",
            adjustflag="2",  # 前复权
        )

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            failed.append(code)
            continue

        df = pd.DataFrame(rows, columns=['date', 'close'])
        df['date'] = pd.to_datetime(df['date'])
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.set_index('date')['close']
        all_data[code] = df

    bs.logout()

    print(f"\nFetched {len(all_data)}/{total} stocks, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed[:20]}...")

    # Combine into DataFrame
    price_df = pd.DataFrame(all_data)
    price_df = price_df.sort_index()
    price_df.to_pickle(cache_path)
    print(f"Weekly price matrix: {price_df.shape[0]} weeks x {price_df.shape[1]} stocks")
    print(f"Date range: {price_df.index[0]} to {price_df.index[-1]}")
    return price_df


# ============================================================
# Backtest Engine
# ============================================================

def compute_momentum(price_df, idx, lookback=12, skip=1):
    """Compute momentum (price return) for all stocks at index position idx."""
    end_idx = idx - skip
    start_idx = end_idx - lookback
    if start_idx < 0 or end_idx < 0:
        return {}

    mom = {}
    for col in price_df.columns:
        p_end = price_df[col].iloc[end_idx]
        p_start = price_df[col].iloc[start_idx]
        if pd.notna(p_end) and pd.notna(p_start) and p_start > 0:
            mom[col] = p_end / p_start - 1
    return mom


def backtest_momentum(price_df, constituent_data, mom_lookback=12, mom_skip=1,
                      top_n=10, rebal_freq=4, txn_cost_bps=8, use_fair=True,
                      start_date=None):
    """
    Momentum backtest with optional point-in-time constituent filtering.

    If use_fair=True: at each rebalance, only select from stocks in the index at that time.
    If use_fair=False: use ALL stocks in price_df (biased - equivalent to using current constituents).
    """
    txn_cost = txn_cost_bps / 10000
    returns = price_df.pct_change()
    warmup = mom_lookback + mom_skip + 1

    nav = [1.0]
    dates = []
    prev_holdings = set()
    weekly_returns = []
    holding_log = []

    i = warmup
    if start_date:
        # Find index position for start_date
        start_ts = pd.Timestamp(start_date)
        for j in range(warmup, len(price_df)):
            if price_df.index[j] >= start_ts:
                i = j
                break

    while i < len(price_df) - 1:
        current_date = price_df.index[i]

        # Compute momentum for all stocks
        mom = compute_momentum(price_df, i, mom_lookback, mom_skip)
        if len(mom) < 20:
            i += 1
            continue

        # Filter by point-in-time constituents if fair mode
        if use_fair and constituent_data:
            active_stocks = get_constituent_at_date(constituent_data, current_date)
            mom = {k: v for k, v in mom.items() if k in active_stocks}

        if len(mom) < top_n:
            i += 1
            continue

        # Rank and select top N
        ranked = sorted(mom.items(), key=lambda x: -x[1])
        selected = set(t for t, _ in ranked[:top_n])

        # Transaction costs
        new_buys = selected - prev_holdings
        sold = prev_holdings - selected
        turnover = len(new_buys) + len(sold)
        period_txn = turnover / max(len(selected), 1) * txn_cost

        # Hold for rebal_freq weeks
        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            week_rets = []
            for s in selected:
                if s in returns.columns and pd.notna(returns[s].iloc[j]):
                    week_rets.append(returns[s].iloc[j])
            port_ret = np.mean(week_rets) if week_rets else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_returns.append(port_ret)

        holding_log.append({
            'date': str(current_date.date()),
            'top3': [(t, round(v*100,1)) for t, v in ranked[:3]],
            'pool_size': len(mom),
        })

        prev_holdings = selected
        i = hold_end

    if not dates or len(dates) < 10:
        return {'error': 'Insufficient data'}

    nav_s = pd.Series(nav[1:], index=dates)
    yrs = (dates[-1] - dates[0]).days / 365.25
    if yrs <= 0:
        return {'error': 'Zero duration'}

    cagr = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / yrs) - 1
    dd = nav_s / nav_s.cummax() - 1
    mdd = dd.min()
    wr = pd.Series(weekly_returns)
    sharpe = wr.mean() / wr.std() * np.sqrt(52) if wr.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    win_rate = (wr > 0).sum() / len(wr) * 100

    # Annual returns
    annual = nav_s.resample('YE').last().pct_change().dropna()
    annual_str = ' | '.join([f"{y.year}:{v:+.1%}" for y, v in annual.items()])

    return {
        'cagr_pct': round(cagr * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round(win_rate, 1),
        'num_weeks': len(weekly_returns),
        'years': round(yrs, 1),
        'annual_returns': annual_str,
        'date_range': f"{dates[0].date()} to {dates[-1].date()}",
        'num_rebalances': len(holding_log),
        'final_nav': round(nav_s.iloc[-1], 4),
    }


def backtest_equal_weight(price_df, constituent_data, rebal_freq=4, txn_cost_bps=8,
                          use_fair=True, start_date=None):
    """Equal-weight benchmark: hold ALL stocks in the current index equally."""
    txn_cost = txn_cost_bps / 10000
    returns = price_df.pct_change()

    nav = [1.0]
    dates = []
    weekly_returns = []
    prev_holdings = set()

    i = 1
    if start_date:
        start_ts = pd.Timestamp(start_date)
        for j in range(1, len(price_df)):
            if price_df.index[j] >= start_ts:
                i = j
                break

    rebal_counter = 0
    current_holdings = set()

    while i < len(price_df):
        current_date = price_df.index[i]

        # Rebalance every rebal_freq weeks
        if rebal_counter % rebal_freq == 0:
            if use_fair and constituent_data:
                eligible = get_constituent_at_date(constituent_data, current_date)
                # Only include stocks that have price data
                current_holdings = set(s for s in eligible if s in returns.columns and pd.notna(price_df[s].iloc[i]))
            else:
                current_holdings = set(s for s in price_df.columns if pd.notna(price_df[s].iloc[i]))

            # Transaction costs on turnover
            new_buys = current_holdings - prev_holdings
            sold = prev_holdings - current_holdings
            turnover = len(new_buys) + len(sold)
            period_txn = turnover / max(len(current_holdings), 1) * txn_cost * 0.1  # much less turnover for EW
            prev_holdings = current_holdings
        else:
            period_txn = 0

        # Compute equal-weight return
        week_rets = []
        for s in current_holdings:
            if s in returns.columns and pd.notna(returns[s].iloc[i]):
                week_rets.append(returns[s].iloc[i])

        port_ret = np.mean(week_rets) if week_rets else 0.0
        if rebal_counter % rebal_freq == 0:
            port_ret -= period_txn

        nav.append(nav[-1] * (1 + port_ret))
        dates.append(current_date)
        weekly_returns.append(port_ret)
        rebal_counter += 1
        i += 1

    if not dates or len(dates) < 10:
        return {'error': 'Insufficient data'}

    nav_s = pd.Series(nav[1:], index=dates)
    yrs = (dates[-1] - dates[0]).days / 365.25
    if yrs <= 0:
        return {'error': 'Zero duration'}

    cagr = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / yrs) - 1
    dd = nav_s / nav_s.cummax() - 1
    mdd = dd.min()
    wr = pd.Series(weekly_returns)
    sharpe = wr.mean() / wr.std() * np.sqrt(52) if wr.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    annual = nav_s.resample('YE').last().pct_change().dropna()
    annual_str = ' | '.join([f"{y.year}:{v:+.1%}" for y, v in annual.items()])

    return {
        'cagr_pct': round(cagr * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'num_weeks': len(weekly_returns),
        'years': round(yrs, 1),
        'annual_returns': annual_str,
        'date_range': f"{dates[0].date()} to {dates[-1].date()}",
        'final_nav': round(nav_s.iloc[-1], 4),
    }


# ============================================================
# Main
# ============================================================

def main():
    t_start = time.time()
    print("=" * 70)
    print("MOMENTUM FAIR BACKTEST (Survivorship Bias Free)")
    print("=" * 70)

    # 1. Fetch historical constituents
    print("\n[1/3] Fetching historical CSI300+CSI500 constituents...")
    constituent_data = fetch_all_historical_constituents()
    print(f"  Total unique stocks: {constituent_data['total_unique']}")

    # 2. Fetch weekly prices for ALL historical stocks
    print("\n[2/3] Fetching weekly prices for all historical stocks...")
    all_stocks = constituent_data['all_unique_stocks']
    price_df = fetch_weekly_prices(all_stocks, start_date='2009-01-01', end_date='2026-03-28')
    print(f"  Price matrix: {price_df.shape}")

    # 3. Run backtests
    print("\n[3/3] Running backtests...")

    results = []

    # Parameter grid
    lookbacks = [8, 12, 20, 26]
    top_ns = [10, 15, 20, 30]
    rebal_freqs = [2, 4]  # 2-week and 4-week

    # A) FAIR backtests (point-in-time constituents)
    print("\n--- FAIR Backtests (no survivorship bias) ---")
    for lb in lookbacks:
        for tn in top_ns:
            for rf in rebal_freqs:
                label = f"FAIR_Mom_LB{lb}_Top{tn}_R{rf}w"
                r = backtest_momentum(price_df, constituent_data,
                                      mom_lookback=lb, mom_skip=1,
                                      top_n=tn, rebal_freq=rf,
                                      use_fair=True)
                if 'error' not in r:
                    r['strategy'] = label
                    r['type'] = 'fair'
                    results.append(r)
                    print(f"  {label}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Sharpe={r['sharpe']}")
                else:
                    print(f"  {label}: ERROR - {r['error']}")

    # B) BIASED backtests (all stocks, simulating current constituents)
    print("\n--- BIASED Backtests (survivorship bias) ---")
    for lb in [12, 20]:
        for tn in [10, 15, 20]:
            for rf in [4]:
                label = f"BIASED_Mom_LB{lb}_Top{tn}_R{rf}w"
                r = backtest_momentum(price_df, constituent_data,
                                      mom_lookback=lb, mom_skip=1,
                                      top_n=tn, rebal_freq=rf,
                                      use_fair=False)
                if 'error' not in r:
                    r['strategy'] = label
                    r['type'] = 'biased'
                    results.append(r)
                    print(f"  {label}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Sharpe={r['sharpe']}")

    # C) Equal-weight benchmarks
    print("\n--- Equal-Weight Benchmarks ---")
    for use_fair, label in [(True, 'EW_Fair'), (False, 'EW_Biased')]:
        r = backtest_equal_weight(price_df, constituent_data, rebal_freq=4,
                                  use_fair=use_fair)
        if 'error' not in r:
            r['strategy'] = label
            r['type'] = 'benchmark'
            results.append(r)
            print(f"  {label}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Sharpe={r['sharpe']}")

    # 4. Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Sorted by Sharpe Ratio")
    print("=" * 70)
    print(f"{'Strategy':<35} {'Type':<8} {'CAGR%':>7} {'MDD%':>7} {'Sharpe':>7} {'Calmar':>7} {'Years':>5}")
    print("-" * 85)

    results.sort(key=lambda x: -x.get('sharpe', 0))
    for r in results:
        print(f"{r['strategy']:<35} {r.get('type',''):<8} {r['cagr_pct']:>7.1f} {r['mdd_pct']:>7.1f} {r['sharpe']:>7.3f} {r.get('calmar',0):>7.3f} {r.get('years',0):>5.1f}")

    # 5. Bias analysis
    print("\n" + "=" * 70)
    print("SURVIVORSHIP BIAS IMPACT")
    print("=" * 70)

    fair_dict = {r['strategy'].replace('FAIR_', '').replace('BIASED_', ''): r
                 for r in results if r.get('type') == 'fair'}
    biased_dict = {r['strategy'].replace('FAIR_', '').replace('BIASED_', ''): r
                   for r in results if r.get('type') == 'biased'}

    for key in sorted(set(fair_dict.keys()) & set(biased_dict.keys())):
        f = fair_dict[key]
        b = biased_dict[key]
        bias = b['cagr_pct'] - f['cagr_pct']
        print(f"  {key}: Fair CAGR={f['cagr_pct']}% vs Biased CAGR={b['cagr_pct']}% → Bias={bias:+.1f}pp")

    # Annual returns for best fair strategy
    best_fair = [r for r in results if r.get('type') == 'fair']
    if best_fair:
        best = max(best_fair, key=lambda x: x['sharpe'])
        print(f"\nBest fair strategy: {best['strategy']}")
        print(f"  CAGR={best['cagr_pct']}% MDD={best['mdd_pct']}% Sharpe={best['sharpe']}")
        print(f"  Annual: {best.get('annual_returns', 'N/A')}")
        print(f"  Date range: {best.get('date_range', 'N/A')}")

    # 6. Save results
    output_path = os.path.join(DATA_DIR, 'momentum_fair_backtest.json')
    output = {
        'results': results,
        'constituent_info': {
            'total_unique_stocks': constituent_data['total_unique'],
            'rebalance_dates': constituent_data['dates'],
        },
        'price_info': {
            'shape': list(price_df.shape),
            'date_range': f"{price_df.index[0].date()} to {price_df.index[-1].date()}",
        },
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': round(time.time() - t_start),
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Total elapsed: {time.time() - t_start:.0f}s")


if __name__ == '__main__':
    main()
