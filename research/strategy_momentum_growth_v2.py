#!/usr/bin/env python3
"""
动量+成长 双因子策略研究 v2 — 消除存活偏差
使用历史成分股数据: 每个调仓日使用当时的沪深300+中证500成分股

Changes vs v1:
  - Load historical constituents from baostock_cache JSON
  - At each rebalancing date, only select from stocks that were in the index at that time
  - Compare survivorship-bias-free results with v1 results
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import os
import sys
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DB_PATH = os.path.join(DATA_DIR, 'stock_data.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

# ============================================================
# 1. Load historical constituents
# ============================================================
def load_historical_constituents():
    """
    Load historical CSI300 + CSI500 constituents.
    Returns a list of (effective_date, end_date, set_of_codes) tuples.
    Codes are converted from baostock format (sh.600000) to 6-digit (600000).
    """
    result_periods = []

    for fn in ['csi300_constituents_history.json', 'csi500_constituents_history.json']:
        path = os.path.join(DATA_DIR, 'baostock_cache', fn)
        with open(path) as f:
            data = json.load(f)

        dates = sorted(data['dates'])
        constituents = data['constituents']

        for i, dt_str in enumerate(dates):
            # Convert codes: sh.600000 -> 600000, sz.000001 -> 000001
            codes = set()
            for c in constituents[dt_str]:
                codes.add(c.split('.')[1])

            start = pd.Timestamp(dt_str)
            end = pd.Timestamp(dates[i+1]) - pd.Timedelta(days=1) if i < len(dates)-1 else pd.Timestamp('2030-01-01')

            result_periods.append((fn.split('_')[0], start, end, codes))

    return result_periods

def get_pool_at_date(constituent_periods, dt):
    """Get the union of CSI300 + CSI500 stocks effective at date dt."""
    pool = set()
    for idx_name, start, end, codes in constituent_periods:
        if start <= dt <= end:
            pool |= codes
    return pool

# ============================================================
# 1b. Load all stock data (wider universe for price/valuation)
# ============================================================
def load_all_data():
    """Load ALL stocks with price and valuation data (not filtered by constituents)."""
    conn = get_db()

    # Get all unique codes that have ever been in CSI300 or CSI500
    constituent_periods = load_historical_constituents()
    all_ever_codes = set()
    for _, _, _, codes in constituent_periods:
        all_ever_codes |= codes
    print(f"All-time unique stocks in CSI300+CSI500: {len(all_ever_codes)}")

    # Weekly prices for these stocks
    placeholders = ','.join(['?'] * len(all_ever_codes))
    codes_list = sorted(all_ever_codes)

    weekly = pd.read_sql(
        f"SELECT code, date, close, volume FROM stock_weekly WHERE date >= '2020-01-01' AND code IN ({placeholders})",
        conn, params=codes_list
    )
    weekly['date'] = pd.to_datetime(weekly['date'])
    price_pivot = weekly.pivot(index='date', columns='code', values='close').sort_index()
    volume_pivot = weekly.pivot(index='date', columns='code', values='volume').sort_index()
    print(f"Weekly prices: {price_pivot.shape[0]} weeks x {price_pivot.shape[1]} stocks")

    # Daily valuations -> resample to weekly
    val = pd.read_sql(
        f"SELECT code, date, pe_ttm, pb_mrq, ps_ttm FROM stock_daily_valuation WHERE date >= '2020-01-01' AND code IN ({placeholders})",
        conn, params=codes_list
    )
    val['date'] = pd.to_datetime(val['date'])
    val['week'] = val['date'].dt.to_period('W').dt.end_time.dt.normalize()
    val_weekly = val.groupby(['code', 'week']).last().reset_index()

    pe_pivot = val_weekly.pivot(index='week', columns='code', values='pe_ttm').sort_index()
    pb_pivot = val_weekly.pivot(index='week', columns='code', values='pb_mrq').sort_index()
    ps_pivot = val_weekly.pivot(index='week', columns='code', values='ps_ttm').sort_index()

    # Align
    common_codes = sorted(set(price_pivot.columns) & set(pe_pivot.columns))
    print(f"Stocks with both price and valuation data: {len(common_codes)}")

    pe_aligned = pe_pivot.reindex(price_pivot.index, method='ffill')[common_codes]
    pb_aligned = pb_pivot.reindex(price_pivot.index, method='ffill')[common_codes]
    ps_aligned = ps_pivot.reindex(price_pivot.index, method='ffill')[common_codes]
    price_df = price_pivot[common_codes]

    conn.close()
    return price_df, pe_aligned, pb_aligned, ps_aligned, common_codes, constituent_periods

# ============================================================
# 2. Compute growth factors (same as v1)
# ============================================================
def compute_growth_factors(price_df, pe_df, pb_df, ps_df, lookback_quarters=4):
    weeks_back = lookback_quarters * 13
    pe_clean = pe_df.replace(0, np.nan).where(pe_df > 0)
    ps_clean = ps_df.replace(0, np.nan).where(ps_df > 0)

    eps = price_df / pe_clean
    rev = price_df / ps_clean
    roe = pb_df / pe_clean

    eps_growth = eps / eps.shift(weeks_back) - 1
    rev_growth = rev / rev.shift(weeks_back) - 1
    roe_change = roe - roe.shift(weeks_back)

    for df in [eps_growth, rev_growth, roe_change]:
        df.clip(-5, 50, inplace=True)

    return eps_growth, rev_growth, roe_change

# ============================================================
# 3. Momentum factors (same as v1)
# ============================================================
def compute_momentum(price_df, lookback=12, skip=1):
    return price_df.shift(skip) / price_df.shift(lookback + skip) - 1

def compute_high52w(price_df):
    high_52w = price_df.rolling(52, min_periods=26).max()
    return price_df / high_52w - 1

# ============================================================
# 4. Ranking (same as v1)
# ============================================================
def rank_pct(df):
    return df.rank(axis=1, pct=True)

def composite_signal(growth_rank, mom_rank, growth_weight=0.5):
    return growth_weight * growth_rank + (1.0 - growth_weight) * mom_rank

# ============================================================
# 5. Backtest with historical constituents (KEY CHANGE)
# ============================================================
def backtest(price_df, signal_df, constituent_periods, top_n=15, rebal_weeks=4, start_date='2021-01-01'):
    """
    Long-only top_n stocks by signal, equal weight, rebalance every rebal_weeks.
    KEY: At each rebalancing, only consider stocks in the historical index pool.
    """
    dates = price_df.index[price_df.index >= pd.Timestamp(start_date)]
    if len(dates) < 10:
        return None

    portfolio_value = [1.0]
    current_holdings = []
    rebal_counter = 0

    for i in range(len(dates) - 1):
        dt = dates[i]
        dt_next = dates[i + 1]

        # Rebalance?
        if rebal_counter == 0 or len(current_holdings) == 0:
            # Get the pool of stocks valid at this date
            pool = get_pool_at_date(constituent_periods, dt)
            pool_codes = [c for c in signal_df.columns if c in pool]

            sig = signal_df.loc[dt, pool_codes].dropna()
            if len(sig) < top_n:
                portfolio_value.append(portfolio_value[-1])
                rebal_counter = (rebal_counter + 1) % rebal_weeks
                continue
            top_stocks = sig.nlargest(top_n).index.tolist()
            current_holdings = top_stocks
            rebal_counter = 1
        else:
            rebal_counter = (rebal_counter + 1) % rebal_weeks

        # Calculate returns
        if len(current_holdings) == 0:
            portfolio_value.append(portfolio_value[-1])
            continue

        p_now = price_df.loc[dt, current_holdings]
        p_next = price_df.loc[dt_next, current_holdings]
        valid = p_now.notna() & p_next.notna() & (p_now > 0)

        if valid.sum() == 0:
            portfolio_value.append(portfolio_value[-1])
            continue

        rets = (p_next[valid] / p_now[valid] - 1).mean()
        portfolio_value.append(portfolio_value[-1] * (1 + rets))

    idx = dates[:len(portfolio_value)]
    return pd.Series(portfolio_value, index=idx)

def calc_metrics(pv):
    if pv is None or len(pv) < 20:
        return None
    years = (pv.index[-1] - pv.index[0]).days / 365.25
    if years < 0.5:
        return None

    total_ret = pv.iloc[-1] / pv.iloc[0]
    cagr = total_ret ** (1/years) - 1
    cummax = pv.cummax()
    dd = pv / cummax - 1
    mdd = dd.min()
    weekly_rets = pv.pct_change().dropna()
    sharpe = weekly_rets.mean() / weekly_rets.std() * np.sqrt(52) if weekly_rets.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    annual = {}
    for year in range(pv.index[0].year, pv.index[-1].year + 1):
        yr_data = pv[pv.index.year == year]
        if len(yr_data) >= 2:
            annual[year] = yr_data.iloc[-1] / yr_data.iloc[0] - 1

    return {
        'cagr': round(cagr * 100, 1),
        'mdd': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'annual': {str(k): round(v*100, 1) for k, v in annual.items()}
    }

# ============================================================
# 6. Main
# ============================================================
def main():
    print("=" * 60)
    print("动量+成长 v2 — 消除存活偏差")
    print("使用历史成分股回测")
    print("=" * 60)

    # Load data
    price_df, pe_df, pb_df, ps_df, codes, constituent_periods = load_all_data()

    # Check how many stocks we need data for but don't have
    all_ever = set()
    for _, _, _, c in constituent_periods:
        all_ever |= c
    missing = all_ever - set(codes)
    print(f"Missing data for {len(missing)} stocks out of {len(all_ever)} total")
    if missing:
        print(f"  Sample missing: {sorted(missing)[:10]}")

    # Compute factors
    print("\n--- Computing factors ---")
    eps_g_yoy, rev_g_yoy, roe_c_yoy = compute_growth_factors(price_df, pe_df, pb_df, ps_df, lookback_quarters=4)
    eps_g_hoh, rev_g_hoh, roe_c_hoh = compute_growth_factors(price_df, pe_df, pb_df, ps_df, lookback_quarters=2)
    mom_12w = compute_momentum(price_df, lookback=12, skip=1)
    mom_20w = compute_momentum(price_df, lookback=20, skip=1)
    high52 = compute_high52w(price_df)

    # Rank
    print("--- Ranking ---")
    r_eps_yoy = rank_pct(eps_g_yoy)
    r_rev_yoy = rank_pct(rev_g_yoy)
    r_roe_yoy = rank_pct(roe_c_yoy)
    r_eps_hoh = rank_pct(eps_g_hoh)
    r_rev_hoh = rank_pct(rev_g_hoh)

    r_growth_yoy = (r_eps_yoy + r_rev_yoy + r_roe_yoy) / 3
    r_growth_ep_rev = (r_eps_yoy + r_rev_yoy) / 2
    r_growth_hoh = (r_eps_hoh + r_rev_hoh) / 2

    r_mom12 = rank_pct(mom_12w)
    r_mom20 = rank_pct(mom_20w)
    r_high52 = rank_pct(high52)

    # Strategies (focus on key ones to save time)
    strategies = {}

    # Pure baselines
    strategies['纯动量_12w'] = r_mom12
    strategies['纯动量_20w'] = r_mom20
    strategies['52周新高'] = r_high52
    strategies['纯成长_YoY(EPS+Rev+ROE)'] = r_growth_yoy
    strategies['纯成长_YoY(EPS+Rev)'] = r_growth_ep_rev
    strategies['纯成长_HoH(EPS+Rev)'] = r_growth_hoh
    strategies['纯EPS成长_YoY'] = r_eps_yoy
    strategies['纯营收成长_YoY'] = r_rev_yoy

    # Key combos
    for gw in [0.3, 0.4, 0.5, 0.6, 0.7]:
        mw = 1 - gw
        lg, lm = int(gw*10), int(mw*10)
        strategies[f'动量12w+成长YoY_G{lg}M{lm}'] = composite_signal(r_growth_yoy, r_mom12, gw)
        strategies[f'动量12w+EP成长_G{lg}M{lm}'] = composite_signal(r_growth_ep_rev, r_mom12, gw)
        strategies[f'动量20w+成长YoY_G{lg}M{lm}'] = composite_signal(r_growth_yoy, r_mom20, gw)
        strategies[f'动量12w+成长HoH_G{lg}M{lm}'] = composite_signal(r_growth_hoh, r_mom12, gw)
        strategies[f'52wH+成长YoY_G{lg}M{lm}'] = composite_signal(r_growth_yoy, r_high52, gw)

    for gw in [0.4, 0.5, 0.6]:
        lg, lm = int(gw*10), int((1-gw)*10)
        strategies[f'动量12w+EPS成长_G{lg}M{lm}'] = composite_signal(r_eps_yoy, r_mom12, gw)
        strategies[f'动量12w+营收成长_G{lg}M{lm}'] = composite_signal(r_rev_yoy, r_mom12, gw)

    print(f"\nTotal strategies: {len(strategies)}")

    configs = [
        {'top_n': 10, 'rebal_weeks': 2},
        {'top_n': 10, 'rebal_weeks': 4},
        {'top_n': 15, 'rebal_weeks': 2},
        {'top_n': 15, 'rebal_weeks': 4},
        {'top_n': 20, 'rebal_weeks': 4},
        {'top_n': 30, 'rebal_weeks': 4},
    ]

    print("\n--- Running backtests (historical constituents) ---")
    all_results = []
    total_tests = len(strategies) * len(configs)
    test_num = 0

    for strat_name, signal in strategies.items():
        for cfg in configs:
            test_num += 1
            if test_num % 50 == 0:
                print(f"  Progress: {test_num}/{total_tests}...")
            pv = backtest(price_df, signal, constituent_periods,
                         top_n=cfg['top_n'], rebal_weeks=cfg['rebal_weeks'],
                         start_date='2021-01-01')
            metrics = calc_metrics(pv)
            if metrics:
                all_results.append({
                    'strategy': strat_name,
                    'top_n': cfg['top_n'],
                    'rebal_weeks': cfg['rebal_weeks'],
                    **metrics
                })

    # Equal weight benchmark (use historical pool)
    eq_signal = pd.DataFrame(1.0, index=price_df.index, columns=price_df.columns)
    pv = backtest(price_df, eq_signal, constituent_periods, top_n=700, rebal_weeks=4, start_date='2021-01-01')
    metrics = calc_metrics(pv)
    if metrics:
        all_results.append({'strategy': '等权持有全部', 'top_n': 700, 'rebal_weeks': 4, **metrics})

    # ============================================================
    # Results
    # ============================================================
    df = pd.DataFrame(all_results)
    if len(df) == 0:
        print("ERROR: No results!")
        return

    print(f"\n{'='*80}")
    print(f"Total results: {len(df)}")
    print(f"{'='*80}")

    print("\n=== TOP 20 by Sharpe ===")
    for _, r in df.nlargest(20, 'sharpe').iterrows():
        ann = ' | '.join([f"{y}:{v:+.1f}%" for y,v in sorted(r['annual'].items())]) if r.get('annual') else ''
        print(f"  {r['strategy']:40s} Top{r['top_n']:2d} R{r['rebal_weeks']}w | "
              f"CAGR={r['cagr']:6.1f}% MDD={r['mdd']:6.1f}% Sharpe={r['sharpe']:.3f} Calmar={r['calmar']:.3f}  {ann}")

    print("\n=== Category comparison ===")
    categories = {
        '纯动量': df[df['strategy'].str.startswith('纯动量') | df['strategy'].str.startswith('52周')],
        '纯成长': df[df['strategy'].str.startswith('纯成长') | df['strategy'].str.startswith('纯EPS') | df['strategy'].str.startswith('纯营收')],
        '动量+成长': df[df['strategy'].str.contains(r'\+')],
        '等权基准': df[df['strategy'] == '等权持有全部'],
    }
    for cat_name, cat_df in categories.items():
        if len(cat_df) == 0:
            continue
        best = cat_df.loc[cat_df['sharpe'].idxmax()]
        ann_str = ' | '.join([f"{y}:{v:+.1f}%" for y,v in sorted(best['annual'].items())]) if best.get('annual') else ''
        print(f"\n  [{cat_name}]  {best['strategy']} Top{best['top_n']} R{best['rebal_weeks']}w")
        print(f"    CAGR={best['cagr']:.1f}% MDD={best['mdd']:.1f}% Sharpe={best['sharpe']:.3f} Calmar={best['calmar']:.3f}")
        print(f"    {ann_str}")

    # Head-to-head Top15 R4w
    print("\n=== Head-to-head: Top15 R4w ===")
    h2h = df[(df['top_n'] == 15) & (df['rebal_weeks'] == 4)].sort_values('sharpe', ascending=False)
    for _, r in h2h.head(20).iterrows():
        ann = ' | '.join([f"{y}:{v:+.1f}%" for y,v in sorted(r['annual'].items())]) if r.get('annual') else ''
        print(f"  {r['strategy']:40s} CAGR={r['cagr']:6.1f}% MDD={r['mdd']:6.1f}% Sharpe={r['sharpe']:.3f}  {ann}")

    # Compare with v1
    v1_path = os.path.join(DATA_DIR, 'momentum_growth_research.json')
    if os.path.exists(v1_path):
        with open(v1_path) as f:
            v1_data = json.load(f)
        v1_results = v1_data.get('results', [])
        print("\n\n=== V1 vs V2 比较 (存活偏差影响) ===")
        print(f"{'Strategy':40s} {'V1 Sharpe':>10} {'V2 Sharpe':>10} {'Delta':>8} {'V1 CAGR':>8} {'V2 CAGR':>8}")
        print("-" * 100)

        # Match by strategy + config
        v1_lookup = {}
        for r in v1_results:
            key = (r['strategy'], r['top_n'], r['rebal_weeks'])
            v1_lookup[key] = r

        compared = []
        for _, r2 in df.iterrows():
            key = (r2['strategy'], r2['top_n'], r2['rebal_weeks'])
            if key in v1_lookup:
                r1 = v1_lookup[key]
                delta_sharpe = r2['sharpe'] - r1['sharpe']
                compared.append((r2['strategy'], r2['top_n'], r2['rebal_weeks'],
                               r1['sharpe'], r2['sharpe'], delta_sharpe,
                               r1['cagr'], r2['cagr']))

        # Sort by v1 sharpe descending
        compared.sort(key=lambda x: -x[3])
        for name, tn, rw, s1, s2, ds, c1, c2 in compared[:30]:
            label = f"{name} T{tn}R{rw}"
            print(f"  {label:45s} {s1:8.3f}   {s2:8.3f}   {ds:+7.3f}   {c1:6.1f}%  {c2:6.1f}%")

        # Summary stats
        deltas = [x[5] for x in compared]
        print(f"\n  Average Sharpe change: {np.mean(deltas):+.3f}")
        print(f"  Median Sharpe change:  {np.median(deltas):+.3f}")
        print(f"  Strategies degraded:   {sum(1 for d in deltas if d < 0)}/{len(deltas)}")

    # Save
    output_path = os.path.join(DATA_DIR, 'momentum_growth_research_v2.json')
    with open(output_path, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'pool': 'CSI300+CSI500 (historical constituents)',
            'backtest_period': '2021~2026',
            'survivorship_bias': 'eliminated',
            'total_strategies': len(strategies),
            'total_configs': len(configs),
            'total_results': len(all_results),
            'results': all_results
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")
    print("\nDONE")

if __name__ == '__main__':
    main()
