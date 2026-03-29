#!/usr/bin/env python3
"""
Strategy J Fair Backtest: Growth+Momentum Dual Factor (NO Survivorship Bias)

Original Strategy J used CURRENT CSI300+CSI500 constituents applied retroactively,
creating survivorship bias (stocks that did well are more likely in today's index).

This script fixes that by:
1. Using baostock historical constituent queries at each semi-annual rebalance date
2. At each monthly rebalance, only selecting from stocks IN the index at that time
3. Fetching PE_TTM and PS_TTM for ALL historically-included stocks
4. Comparing biased vs unbiased results

Data: baostock (free, no API key needed)
"""

import baostock as bs
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import pickle
from datetime import datetime
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'baostock_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Semi-annual index rebalance dates (CSI300 & CSI500 rebalance together)
REBALANCE_DATES = [
    '2020-06-15',
    '2020-12-14',
    '2021-06-15',
    '2021-12-13',
    '2022-06-13',
    '2022-12-12',
    '2023-06-12',
    '2023-12-11',
    '2024-06-17',
    '2024-12-16',
    '2025-06-16',
    '2025-12-15',
]


# ============================================================
# Data Fetching
# ============================================================

def fetch_historical_constituents(force=False):
    """Fetch CSI300+CSI500 constituents at each semi-annual rebalance date."""
    cache_path = os.path.join(CACHE_DIR, 'csi300_500_combined_constituents.json')
    if not force and os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 30:
            with open(cache_path) as f:
                data = json.load(f)
            print(f"Loaded cached combined constituents: {len(data['dates'])} dates, {data['total_unique']} unique stocks")
            return data

    lg = bs.login()
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
        print(f"  {d}: CSI300={len(stocks_300)}, CSI500={len(stocks_500)}, combined={len(combined)}")

    bs.logout()

    data = {
        'dates': REBALANCE_DATES,
        'constituents': constituent_map,
        'all_unique_stocks': sorted(all_stocks),
        'total_unique': len(all_stocks),
    }

    with open(cache_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Total unique stocks across all periods: {len(all_stocks)}")
    return data


def get_active_constituents(date_str, constituent_data):
    """Get which stocks were in CSI300+CSI500 at a given date."""
    rebal_dates = constituent_data['dates']
    constituents = constituent_data['constituents']

    active_date = None
    for d in rebal_dates:
        if d <= date_str:
            active_date = d
        else:
            break

    if active_date is None:
        active_date = rebal_dates[0]

    return set(constituents[active_date]['combined'])


def fetch_weekly_fundamentals(stock_list, force=False):
    """Fetch daily PE_TTM and PS_TTM, resample to weekly (Friday close)."""
    cache_path = os.path.join(CACHE_DIR, 'strategy_j_weekly_fundamentals.pkl')
    if not force and os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 7:
            data = pd.read_pickle(cache_path)
            print(f"Loaded cached fundamentals: PE {data['pe'].shape}, PS {data['ps'].shape}")
            return data

    lg = bs.login()
    all_pe = {}
    all_ps = {}
    all_close = {}
    total = len(stock_list)
    failed = []

    for i, stock in enumerate(stock_list):
        if i % 50 == 0:
            print(f"  Fetching fundamentals: {i}/{total}...")

        # Must use daily frequency for peTTM/psTTM (not available in weekly)
        rs = bs.query_history_k_data_plus(
            stock,
            'date,close,peTTM,psTTM',
            start_date='2020-01-01',
            end_date='2026-03-31',
            frequency='d',
            adjustflag='1'  # post-adjusted prices
        )

        rows = []
        while rs.next():
            row = rs.get_row_data()
            try:
                d = row[0]
                c = float(row[1]) if row[1] else np.nan
                pe = float(row[2]) if row[2] else np.nan
                ps = float(row[3]) if row[3] else np.nan
                rows.append({'date': d, 'close': c, 'pe': pe, 'ps': ps})
            except (ValueError, IndexError):
                continue

        if len(rows) > 50:
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            # Resample to weekly (Friday), take last value of each week
            weekly = df.resample('W-FRI').last().dropna(how='all')
            all_pe[stock] = weekly['pe']
            all_ps[stock] = weekly['ps']
            all_close[stock] = weekly['close']
        else:
            failed.append(stock)

        # Rate limiting
        if i % 200 == 199:
            time.sleep(1)

    bs.logout()

    print(f"Fetched {len(all_pe)} stocks, {len(failed)} failed")
    if failed:
        print(f"  Failed ({len(failed)}): {failed[:20]}...")

    pe_df = pd.DataFrame(all_pe).sort_index()
    ps_df = pd.DataFrame(all_ps).sort_index()
    close_df = pd.DataFrame(all_close).sort_index()

    # Keep from 2020 onward, forward fill small gaps
    start = pd.Timestamp('2020-06-01')
    pe_df = pe_df[pe_df.index >= start].ffill(limit=2)
    ps_df = ps_df[ps_df.index >= start].ffill(limit=2)
    close_df = close_df[close_df.index >= start].ffill(limit=2)

    data = {'pe': pe_df, 'ps': ps_df, 'close': close_df}
    pd.to_pickle(data, cache_path)
    print(f"Saved: PE {pe_df.shape}, PS {ps_df.shape}, Close {close_df.shape}")
    return data


# ============================================================
# Strategy Logic
# ============================================================

def compute_growth_signal(close_df, pe_df, ps_df, lookback=26):
    """
    Compute HoH growth signal.
    EPS = Price / PE_TTM (only positive PE)
    Rev = Price / PS_TTM (only positive PS)
    Growth = current / lookback_weeks_ago - 1
    Signal = average percentile rank of EPS growth + Revenue growth
    """
    pe_clean = pe_df.replace(0, np.nan).where(pe_df > 0)
    ps_clean = ps_df.replace(0, np.nan).where(ps_df > 0)

    eps = close_df / pe_clean
    rev = close_df / ps_clean

    eps_growth = (eps / eps.shift(lookback) - 1).clip(-5, 50)
    rev_growth = (rev / rev.shift(lookback) - 1).clip(-5, 50)

    eps_rank = eps_growth.rank(axis=1, pct=True)
    rev_rank = rev_growth.rank(axis=1, pct=True)
    composite = (eps_rank + rev_rank) / 2

    return composite, eps_growth, rev_growth


def backtest_growth_momentum(close_df, pe_df, ps_df, constituent_data,
                              top_n=15, rebal_weeks=4, growth_lookback=26,
                              txn_bps=10, use_historical_constituents=True,
                              label=""):
    """
    Backtest Strategy J with or without survivorship bias fix.

    Args:
        use_historical_constituents: True = fair (no bias), False = use all stocks (biased)
    """
    # Align all dataframes
    common_cols = list(set(close_df.columns) & set(pe_df.columns) & set(ps_df.columns))
    close = close_df[common_cols]
    pe = pe_df.reindex(close.index, method='ffill')[common_cols]
    ps = ps_df.reindex(close.index, method='ffill')[common_cols]

    # Compute growth signal for all stocks
    signal, eps_g, rev_g = compute_growth_signal(close, pe, ps, growth_lookback)

    # Weekly returns
    returns = close.pct_change()

    warmup = growth_lookback + 2
    txn_cost = txn_bps / 10000

    nav = [1.0]
    dates = [close.index[warmup - 1]]
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []
    selection_log = []

    i = warmup
    while i < len(close) - 1:
        current_date = close.index[i]
        date_str = current_date.strftime('%Y-%m-%d')

        # Determine eligible universe
        if use_historical_constituents:
            eligible = get_active_constituents(date_str, constituent_data)
            eligible = eligible & set(common_cols)
        else:
            eligible = set(common_cols)

        # Get signal for eligible stocks
        sig_row = signal.iloc[i]
        eligible_signals = sig_row[list(eligible)].dropna()

        if len(eligible_signals) < top_n:
            # Hold through, no rebalance
            if prev_holdings:
                rets = [float(returns[s].iloc[i]) for s in prev_holdings
                        if s in returns.columns and not pd.isna(returns[s].iloc[i])]
                port_ret = np.mean(rets) if rets else 0.0
            else:
                port_ret = 0.0
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(current_date)
            weekly_rets.append(port_ret)
            i += 1
            continue

        # Select top_n stocks
        top_picks = eligible_signals.nlargest(top_n).index.tolist()

        # Log selection
        selection_log.append({
            'date': date_str,
            'eligible': len(eligible_signals),
            'selected': top_picks[:5],  # first 5 for logging
        })

        # Transaction costs
        selected_set = set(top_picks)
        if prev_holdings:
            turnover = len(selected_set - prev_holdings) / max(len(selected_set), 1)
        else:
            turnover = 1.0
        period_txn = turnover * txn_cost

        # Hold for rebal_weeks
        hold_end = min(i + rebal_weeks, len(close) - 1)
        for j in range(i, hold_end + 1):
            rets = [float(returns[s].iloc[j]) for s in top_picks
                    if s in returns.columns and pd.notna(returns[s].iloc[j])]
            port_ret = np.mean(rets) if rets else 0.0

            # Charge transaction cost on first week of holding period
            if j == i:
                port_ret -= period_txn
                total_txn += period_txn

            nav.append(nav[-1] * (1 + port_ret))
            dates.append(close.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end + 1

    nav_series = pd.Series(nav, index=dates)
    return nav_series, weekly_rets, total_txn, selection_log


def calc_stats(nav_series, weekly_rets, total_txn, label=""):
    """Calculate strategy statistics."""
    nav = nav_series.values
    if len(nav) < 10:
        return {'label': label, 'error': 'too short'}

    n_weeks = len(nav) - 1
    years = n_weeks / 52.0
    cagr = (nav[-1] / nav[0]) ** (1 / years) - 1 if years > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    mdd = float(np.min(dd))

    # Sharpe
    wr = np.array(weekly_rets)
    if len(wr) > 10 and np.std(wr) > 0:
        sharpe = float(np.mean(wr) / np.std(wr) * np.sqrt(52))
    else:
        sharpe = 0.0

    calmar = cagr / abs(mdd) if mdd != 0 else 0

    # Win rate
    win_rate = float(np.mean(wr > 0) * 100) if len(wr) > 0 else 0

    # Annual returns
    annual = {}
    for year in range(2021, 2027):
        mask = [d.year == year for d in nav_series.index]
        if sum(mask) > 4:
            year_nav = nav[np.array(mask)]
            annual[str(year)] = round((year_nav[-1] / year_nav[0] - 1) * 100, 1)

    return {
        'label': label,
        'cagr_pct': round(cagr * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round(win_rate, 1),
        'total_return_pct': round((nav[-1] / nav[0] - 1) * 100, 1),
        'years': round(years, 1),
        'total_txn_pct': round(total_txn * 100, 2),
        'annual_returns': annual,
        'period': f"{nav_series.index[0].strftime('%Y-%m-%d')} ~ {nav_series.index[-1].strftime('%Y-%m-%d')}",
    }


def build_buy_hold_benchmark(close_df, constituent_data, use_historical=True, rebal_weeks=4):
    """Equal-weight buy-and-hold of all constituents (with or without historical tracking)."""
    returns = close_df.pct_change()
    warmup = 28  # ~6 months

    nav = [1.0]
    dates = [close_df.index[warmup - 1]]
    weekly_rets = []

    for i in range(warmup, len(close_df)):
        current_date = close_df.index[i]
        date_str = current_date.strftime('%Y-%m-%d')

        if use_historical:
            eligible = get_active_constituents(date_str, constituent_data)
            eligible = eligible & set(close_df.columns)
        else:
            eligible = set(close_df.columns)

        rets = [float(returns[s].iloc[i]) for s in eligible
                if s in returns.columns and pd.notna(returns[s].iloc[i])]
        port_ret = np.mean(rets) if rets else 0.0
        nav.append(nav[-1] * (1 + port_ret))
        dates.append(current_date)
        weekly_rets.append(port_ret)

    return pd.Series(nav, index=dates), weekly_rets


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()

    print("=" * 70)
    print("Strategy J Fair Backtest: Growth+Momentum (No Survivorship Bias)")
    print("=" * 70)

    # Step 1: Historical constituents
    print("\n[1/3] Fetching historical constituents (CSI300+CSI500)...")
    constituent_data = fetch_historical_constituents()
    all_stocks = constituent_data['all_unique_stocks']
    print(f"  Total unique stocks: {len(all_stocks)}")

    # Step 2: Fetch fundamentals (PE_TTM, PS_TTM, close)
    print("\n[2/3] Fetching weekly fundamentals (PE_TTM, PS_TTM, close)...")
    fund_data = fetch_weekly_fundamentals(all_stocks)
    close_df = fund_data['close']
    pe_df = fund_data['pe']
    ps_df = fund_data['ps']
    print(f"  Close: {close_df.shape}, PE: {pe_df.shape}, PS: {ps_df.shape}")

    # Step 3: Run backtests
    print("\n[3/3] Running backtests...")

    configs = [
        # Fair (no survivorship bias)
        {'top_n': 10, 'rebal_weeks': 4, 'growth_lookback': 26, 'use_historical_constituents': True,
         'label': 'FAIR_Top10_4w_g26'},
        {'top_n': 15, 'rebal_weeks': 4, 'growth_lookback': 26, 'use_historical_constituents': True,
         'label': 'FAIR_Top15_4w_g26'},
        {'top_n': 20, 'rebal_weeks': 4, 'growth_lookback': 26, 'use_historical_constituents': True,
         'label': 'FAIR_Top20_4w_g26'},
        {'top_n': 15, 'rebal_weeks': 2, 'growth_lookback': 26, 'use_historical_constituents': True,
         'label': 'FAIR_Top15_2w_g26'},
        {'top_n': 15, 'rebal_weeks': 4, 'growth_lookback': 13, 'use_historical_constituents': True,
         'label': 'FAIR_Top15_4w_g13'},
        {'top_n': 15, 'rebal_weeks': 4, 'growth_lookback': 52, 'use_historical_constituents': True,
         'label': 'FAIR_Top15_4w_g52'},

        # Biased (current constituents applied retroactively) - for comparison
        {'top_n': 10, 'rebal_weeks': 4, 'growth_lookback': 26, 'use_historical_constituents': False,
         'label': 'BIASED_Top10_4w_g26'},
        {'top_n': 15, 'rebal_weeks': 4, 'growth_lookback': 26, 'use_historical_constituents': False,
         'label': 'BIASED_Top15_4w_g26'},
        {'top_n': 20, 'rebal_weeks': 4, 'growth_lookback': 26, 'use_historical_constituents': False,
         'label': 'BIASED_Top20_4w_g26'},
    ]

    results = []
    for cfg in configs:
        label = cfg.pop('label')
        print(f"\n  Running {label}...")
        nav, wr, txn, sel_log = backtest_growth_momentum(
            close_df, pe_df, ps_df, constituent_data, txn_bps=10, label=label, **cfg
        )
        cfg['label'] = label

        stats = calc_stats(nav, wr, txn, label)
        results.append(stats)

        if 'error' not in stats:
            print(f"    CAGR={stats['cagr_pct']}% MDD={stats['mdd_pct']}% "
                  f"Sharpe={stats['sharpe']} Calmar={stats['calmar']}")
            print(f"    Annual: {stats['annual_returns']}")
            print(f"    Txn cost total: {stats['total_txn_pct']}%")
        else:
            print(f"    ERROR: {stats['error']}")

    # Benchmarks
    print("\n  Running benchmarks...")

    # Fair equal-weight benchmark
    bm_nav, bm_wr = build_buy_hold_benchmark(close_df, constituent_data, use_historical=True)
    bm_stats = calc_stats(bm_nav, bm_wr, 0.0, 'BM_EqualWeight_Historical')
    results.append(bm_stats)
    print(f"    BM Historical: CAGR={bm_stats['cagr_pct']}% MDD={bm_stats['mdd_pct']}% Sharpe={bm_stats['sharpe']}")

    # Biased equal-weight benchmark
    bm2_nav, bm2_wr = build_buy_hold_benchmark(close_df, constituent_data, use_historical=False)
    bm2_stats = calc_stats(bm2_nav, bm2_wr, 0.0, 'BM_EqualWeight_AllStocks')
    results.append(bm2_stats)
    print(f"    BM All Stocks: CAGR={bm2_stats['cagr_pct']}% MDD={bm2_stats['mdd_pct']}% Sharpe={bm2_stats['sharpe']}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON: Fair vs Biased")
    print("=" * 70)
    print(f"{'Strategy':<30} {'CAGR%':>7} {'MDD%':>7} {'Sharpe':>7} {'Calmar':>7}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: -x.get('cagr_pct', -999)):
        if 'error' in r:
            continue
        print(f"{r['label']:<30} {r['cagr_pct']:>7.1f} {r['mdd_pct']:>7.1f} "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f}")

    # Survivorship bias impact
    print("\n" + "=" * 70)
    print("SURVIVORSHIP BIAS IMPACT")
    print("=" * 70)
    for cfg_name in ['Top10_4w_g26', 'Top15_4w_g26', 'Top20_4w_g26']:
        fair = next((r for r in results if r['label'] == f'FAIR_{cfg_name}'), None)
        biased = next((r for r in results if r['label'] == f'BIASED_{cfg_name}'), None)
        if fair and biased and 'error' not in fair and 'error' not in biased:
            cagr_diff = biased['cagr_pct'] - fair['cagr_pct']
            print(f"  {cfg_name}:")
            print(f"    Fair:   CAGR={fair['cagr_pct']:+.1f}% MDD={fair['mdd_pct']:.1f}% Sharpe={fair['sharpe']:.3f}")
            print(f"    Biased: CAGR={biased['cagr_pct']:+.1f}% MDD={biased['mdd_pct']:.1f}% Sharpe={biased['sharpe']:.3f}")
            print(f"    Bias inflates CAGR by {cagr_diff:+.1f}pp")

    # Save results
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Strategy J fair backtest with survivorship bias fix',
        'total_unique_stocks': len(all_stocks),
        'constituent_dates': REBALANCE_DATES,
        'data_shape': {
            'close': list(close_df.shape),
            'pe': list(pe_df.shape),
            'ps': list(ps_df.shape),
        },
        'strategies': sorted(results, key=lambda x: -x.get('calmar', -999)),
    }

    out_path = os.path.join(DATA_DIR, 'strategy_j_fair_backtest.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\nResults saved to {out_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
