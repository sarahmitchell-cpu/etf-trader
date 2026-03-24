#!/usr/bin/env python3
"""
CSI500 Full Historical Constituent Backtest
- Uses baostock for real historical constituent lists at each semi-annual rebalance
- Downloads weekly price data for ALL ~928 unique stocks
- Gets industry classification from baostock
- Runs Sector Rotation strategy using correct constituents at each point in time
- Performs rolling 12-month stability analysis
- NO survivorship bias
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

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'baostock_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Semi-annual rebalance dates for CSI 500
REBALANCE_DATES = [
    '2020-12-14',  # Need one before our backtest period for initial constituents
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


def get_constituents_all_dates():
    """Fetch CSI 500 constituents at each rebalance date"""
    cache_path = os.path.join(CACHE_DIR, 'csi500_constituents_history.json')
    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 7:
            with open(cache_path) as f:
                data = json.load(f)
            print(f"Loaded cached constituents: {len(data)} dates")
            return data

    lg = bs.login()
    constituent_map = {}
    all_stocks = set()

    for d in REBALANCE_DATES:
        rs = bs.query_zz500_stocks(date=d)
        stocks = []
        while rs.next():
            row = rs.get_row_data()
            stocks.append(row[1])  # e.g. sh.600006
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

    print(f"Total unique stocks: {len(all_stocks)}")
    return data


def get_industry_classification(stock_list):
    """Get industry classification for all stocks from baostock"""
    cache_path = os.path.join(CACHE_DIR, 'csi500_industries.json')
    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 30:
            with open(cache_path) as f:
                data = json.load(f)
            missing = [s for s in stock_list if s not in data]
            if len(missing) < 10:
                print(f"Loaded cached industries: {len(data)} stocks ({len(missing)} missing)")
                return data

    lg = bs.login()
    industries = {}
    total = len(stock_list)

    for i, stock in enumerate(stock_list):
        if i % 100 == 0:
            print(f"  Fetching industries: {i}/{total}...")
        rs = bs.query_stock_industry(code=stock)
        while rs.next():
            row = rs.get_row_data()
            # Fields: updateDate, code, code_name, industry, industryClassification
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

    print(f"Got industries for {len(industries)} stocks")
    return industries


def fetch_weekly_prices(stock_list):
    """Fetch weekly close prices for all stocks via baostock"""
    cache_path = os.path.join(CACHE_DIR, 'csi500_all_weekly_prices.pkl')
    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 3:
            df = pd.read_pickle(cache_path)
            missing = [s for s in stock_list if s not in df.columns]
            if len(missing) < 20:
                print(f"Loaded cached prices: {df.shape[0]} weeks x {df.shape[1]} stocks ({len(missing)} missing)")
                return df

    lg = bs.login()
    all_prices = {}
    total = len(stock_list)
    failed = []

    for i, stock in enumerate(stock_list):
        if i % 50 == 0:
            print(f"  Fetching prices: {i}/{total}...")

        rs = bs.query_history_k_data_plus(
            stock,
            'date,close',
            start_date='2020-09-01',
            end_date='2026-03-31',
            frequency='w',
            adjustflag='1'  # 后复权
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

        # Rate limiting
        if i % 100 == 99:
            time.sleep(2)

    bs.logout()

    print(f"Fetched {len(all_prices)} stocks, {len(failed)} failed")
    if failed:
        print(f"  Failed: {failed[:20]}...")

    price_df = pd.DataFrame(all_prices)
    price_df = price_df.sort_index()
    price_df = price_df[price_df.index >= '2021-01-01']
    price_df = price_df.ffill(limit=2)

    price_df.to_pickle(cache_path)
    print(f"Price matrix: {price_df.shape[0]} weeks x {price_df.shape[1]} stocks")
    return price_df


def get_active_constituents(date, constituent_data):
    """Get which stocks were in the CSI 500 at a given date"""
    rebal_dates = constituent_data['dates']
    constituents = constituent_data['constituents']

    # Find the most recent rebalance date <= given date
    active_date = None
    for d in rebal_dates:
        if d <= date.strftime('%Y-%m-%d'):
            active_date = d
        else:
            break

    if active_date is None:
        active_date = rebal_dates[0]

    return set(constituents[active_date])


def strategy_sector_rotation(price_df, constituent_data, industries,
                              mom_lookback=12, skip=0, top_sectors=3,
                              stocks_per_sector=2, rebal_freq=2, txn_bps=8):
    """Sector Rotation using real historical constituents"""
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    warmup = mom_lookback + skip + 2

    nav = [1.0]
    dates = [price_df.index[warmup - 1]]
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    # Track selections for analysis
    selection_log = []

    i = warmup
    while i < len(price_df) - 1:
        current_date = price_df.index[i]

        # Get active CSI 500 constituents at this date
        active = get_active_constituents(current_date, constituent_data)
        # Intersect with available price data
        available = active & set(price_df.columns)

        if len(available) < 50:
            i += 1
            continue

        # Compute momentum for available stocks
        mom_idx = i - skip
        start_idx = mom_idx - mom_lookback
        if start_idx < 0:
            i += 1
            continue

        mom = {}
        for col in available:
            p0 = price_df[col].iloc[start_idx]
            p1 = price_df[col].iloc[mom_idx]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                mom[col] = p1 / p0 - 1

        if len(mom) < 30:
            i += 1
            continue

        # Build sector mapping using industries
        sector_stocks = defaultdict(list)
        for s in mom:
            ind = industries.get(s, {}).get('industry', '其他')
            sector_stocks[ind].append(s)

        # Sector average momentum
        sector_mom = {}
        for sec, stocks in sector_stocks.items():
            if len(stocks) >= 2:  # Need at least 2 stocks per sector
                moms = [mom[s] for s in stocks if s in mom]
                if moms:
                    sector_mom[sec] = np.mean(moms)

        if len(sector_mom) < top_sectors:
            i += 1
            continue

        # Pick top sectors by momentum
        top_secs = sorted(sector_mom.items(), key=lambda x: -x[1])[:top_sectors]
        top_sec_names = [s for s, _ in top_secs]

        # Pick best stocks within each top sector
        selected = []
        for sec in top_sec_names:
            candidates = [(s, mom.get(s, -999)) for s in sector_stocks[sec] if s in mom]
            candidates.sort(key=lambda x: -x[1])
            for s, _ in candidates[:stocks_per_sector]:
                selected.append(s)

        if not selected:
            i += 1
            continue

        # Log selection
        selection_log.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'active_constituents': len(available),
            'with_momentum': len(mom),
            'top_sectors': top_sec_names,
            'selected': [f"{industries.get(s,{}).get('name','?')}({industries.get(s,{}).get('industry','?')})" for s in selected],
        })

        selected_set = set(selected)
        turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            rets = [float(returns[s].iloc[j]) for s in selected if s in returns.columns and not pd.isna(returns[s].iloc[j])]
            port_ret = np.mean(rets) if rets else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    return pd.Series(nav, index=dates), weekly_rets, total_txn, selection_log


def calc_stats(nav_series, weekly_rets, total_txn, label=""):
    """Calculate strategy statistics"""
    nav = nav_series.values
    total_ret = nav[-1] / nav[0] - 1
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
    win_rate = float(np.mean(np.array(weekly_rets) > 0) * 100)

    # Annual returns
    annual = {}
    for year in range(2021, 2027):
        mask = [d.year == year for d in nav_series.index]
        if sum(mask) > 4:
            year_nav = nav[mask]
            annual[str(year)] = round((year_nav[-1] / year_nav[0] - 1) * 100, 1)

    return {
        'label': label,
        'cagr_pct': round(cagr * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round(win_rate, 1),
        'total_return_pct': round(total_ret * 100, 1),
        'years': round(years, 1),
        'total_txn_pct': round(total_txn * 100, 2),
        'annual_returns': annual,
        'period': f"{nav_series.index[0].strftime('%Y-%m-%d')} ~ {nav_series.index[-1].strftime('%Y-%m-%d')}",
    }


def rolling_analysis(nav_series, window_weeks=52):
    """Rolling window return analysis"""
    nav = nav_series.values
    dates = nav_series.index

    results = []
    for start in range(len(nav) - window_weeks):
        end = start + window_weeks
        ret = nav[end] / nav[start] - 1
        results.append({
            'start': dates[start].strftime('%Y-%m-%d'),
            'end': dates[end].strftime('%Y-%m-%d'),
            'return_pct': round(ret * 100, 2)
        })

    returns = np.array([r['return_pct'] for r in results])

    summary = {
        'window_weeks': window_weeks,
        'total_windows': len(results),
        'positive_windows': int(np.sum(returns > 0)),
        'win_rate_pct': round(float(np.mean(returns > 0) * 100), 1),
        'mean_return_pct': round(float(np.mean(returns)), 2),
        'median_return_pct': round(float(np.median(returns)), 2),
        'std_return_pct': round(float(np.std(returns)), 2),
        'min_return_pct': round(float(np.min(returns)), 2),
        'max_return_pct': round(float(np.max(returns)), 2),
        'p5': round(float(np.percentile(returns, 5)), 2),
        'p10': round(float(np.percentile(returns, 10)), 2),
        'p25': round(float(np.percentile(returns, 25)), 2),
        'p75': round(float(np.percentile(returns, 75)), 2),
        'p90': round(float(np.percentile(returns, 90)), 2),
        'p95': round(float(np.percentile(returns, 95)), 2),
    }

    worst_idx = int(np.argmin(returns))
    best_idx = int(np.argmax(returns))
    summary['worst_period'] = results[worst_idx]
    summary['best_period'] = results[best_idx]

    # Max drawdown duration
    peak = nav[0]
    in_dd = False
    dd_start = 0
    max_dd_dur = 0
    for i in range(len(nav)):
        if nav[i] >= peak:
            if in_dd:
                max_dd_dur = max(max_dd_dur, i - dd_start)
                in_dd = False
            peak = nav[i]
        else:
            if not in_dd:
                dd_start = i
                in_dd = True
    if in_dd:
        max_dd_dur = max(max_dd_dur, len(nav) - 1 - dd_start)
    summary['max_drawdown_duration_weeks'] = max_dd_dur

    return summary, sorted(results, key=lambda x: x['return_pct'])


def main():
    t0 = time.time()

    print("=" * 60)
    print("CSI500 Full Constituent Backtest (No Survivorship Bias)")
    print("=" * 60)

    # Step 1: Get historical constituents
    print("\n[1/4] Fetching historical constituents...")
    constituent_data = get_constituents_all_dates()

    all_stocks = constituent_data['all_unique_stocks']
    print(f"  Total unique stocks: {len(all_stocks)}")

    # Step 2: Get industry classification
    print("\n[2/4] Fetching industry classifications...")
    industries = get_industry_classification(all_stocks)

    # Count industries
    ind_count = defaultdict(int)
    for s in all_stocks:
        ind = industries.get(s, {}).get('industry', '其他')
        ind_count[ind] += 1
    print(f"  Industries: {len(ind_count)}")
    for ind, cnt in sorted(ind_count.items(), key=lambda x: -x[1])[:15]:
        print(f"    {ind}: {cnt}")

    # Step 3: Fetch all price data
    print("\n[3/4] Fetching weekly prices for all stocks...")
    price_df = fetch_weekly_prices(all_stocks)
    print(f"  Matrix: {price_df.shape[0]} weeks x {price_df.shape[1]} stocks")

    # Step 4: Run strategies
    print("\n[4/4] Running strategies...")

    configs = [
        {'mom_lookback': 12, 'skip': 0, 'top_sectors': 3, 'stocks_per_sector': 2, 'rebal_freq': 2,
         'label': 'SR_m12_top3_sps2_2w'},
        {'mom_lookback': 12, 'skip': 0, 'top_sectors': 3, 'stocks_per_sector': 2, 'rebal_freq': 4,
         'label': 'SR_m12_top3_sps2_4w'},
        {'mom_lookback': 12, 'skip': 0, 'top_sectors': 5, 'stocks_per_sector': 2, 'rebal_freq': 2,
         'label': 'SR_m12_top5_sps2_2w'},
        {'mom_lookback': 12, 'skip': 1, 'top_sectors': 3, 'stocks_per_sector': 2, 'rebal_freq': 2,
         'label': 'SR_m12_sk1_top3_sps2_2w'},
        {'mom_lookback': 8, 'skip': 0, 'top_sectors': 3, 'stocks_per_sector': 2, 'rebal_freq': 2,
         'label': 'SR_m8_top3_sps2_2w'},
        {'mom_lookback': 20, 'skip': 0, 'top_sectors': 3, 'stocks_per_sector': 2, 'rebal_freq': 2,
         'label': 'SR_m20_top3_sps2_2w'},
        {'mom_lookback': 12, 'skip': 0, 'top_sectors': 3, 'stocks_per_sector': 3, 'rebal_freq': 2,
         'label': 'SR_m12_top3_sps3_2w'},
        {'mom_lookback': 12, 'skip': 0, 'top_sectors': 4, 'stocks_per_sector': 2, 'rebal_freq': 2,
         'label': 'SR_m12_top4_sps2_2w'},
    ]

    results = []
    best_nav = None
    best_label = None
    best_calmar = -999

    for cfg in configs:
        label = cfg.pop('label')
        print(f"\n  Running {label}...")
        nav, wr, txn, sel_log = strategy_sector_rotation(
            price_df, constituent_data, industries, **cfg, txn_bps=8
        )
        cfg['label'] = label

        if len(nav) < 52:
            print(f"    SKIP: too short ({len(nav)} weeks)")
            continue

        stats = calc_stats(nav, wr, txn, label)
        results.append(stats)
        print(f"    CAGR={stats['cagr_pct']}% MDD={stats['mdd_pct']}% Sharpe={stats['sharpe']} Calmar={stats['calmar']}")
        print(f"    Annual: {stats['annual_returns']}")

        if stats['calmar'] > best_calmar:
            best_calmar = stats['calmar']
            best_nav = nav
            best_label = label

    # Rolling analysis on best strategy
    print(f"\n{'='*60}")
    print(f"Best strategy: {best_label} (Calmar={best_calmar:.3f})")
    print(f"{'='*60}")

    if best_nav is not None and len(best_nav) > 52:
        print("\n=== Rolling 52-week (12-month) Analysis ===")
        rolling_summary, rolling_windows = rolling_analysis(best_nav, 52)
        print(f"  Windows: {rolling_summary['total_windows']}")
        print(f"  Win rate: {rolling_summary['win_rate_pct']}%")
        print(f"  Mean: {rolling_summary['mean_return_pct']}%")
        print(f"  Median: {rolling_summary['median_return_pct']}%")
        print(f"  Min: {rolling_summary['min_return_pct']}% ({rolling_summary['worst_period']['start']} ~ {rolling_summary['worst_period']['end']})")
        print(f"  Max: {rolling_summary['max_return_pct']}%")
        print(f"  P5/P10/P25: {rolling_summary['p5']}% / {rolling_summary['p10']}% / {rolling_summary['p25']}%")
        print(f"  P75/P90/P95: {rolling_summary['p75']}% / {rolling_summary['p90']}% / {rolling_summary['p95']}%")
        print(f"  Max DD duration: {rolling_summary['max_drawdown_duration_weeks']} weeks")

        # Print all windows
        print(f"\n=== All 52-week Windows (sorted) ===")
        for w in rolling_windows:
            print(f"  {w['start']} ~ {w['end']}: {w['return_pct']:+.1f}%")
    else:
        rolling_summary = None
        rolling_windows = None

    # Save results
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Full CSI500 historical constituent backtest (no survivorship bias)',
        'total_unique_stocks': len(all_stocks),
        'constituent_dates': REBALANCE_DATES,
        'price_matrix': f"{price_df.shape[0]} weeks x {price_df.shape[1]} stocks",
        'strategies': sorted(results, key=lambda x: -x['calmar']),
        'best_strategy': best_label,
        'rolling_52w_summary': rolling_summary,
        'rolling_52w_windows': rolling_windows,
    }

    out_path = os.path.join(DATA_DIR, 'csi500_full_constituent_backtest.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\nResults saved to {out_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
