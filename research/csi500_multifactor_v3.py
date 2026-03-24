#!/usr/bin/env python3
"""
CSI500 Multi-Factor Research v3 - VECTORIZED (No Survivorship Bias)
Pre-computes all factor matrices as DataFrames for fast ranking.
"""

import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'baostock_cache')


def load_data():
    with open(os.path.join(CACHE_DIR, 'csi500_constituents_history.json')) as f:
        const = json.load(f)
    with open(os.path.join(CACHE_DIR, 'csi500_industries.json')) as f:
        industries = json.load(f)
    price_df = pd.read_pickle(os.path.join(CACHE_DIR, 'csi500_all_weekly_prices.pkl'))

    # Load PE/PB if available
    fund_path = os.path.join(CACHE_DIR, 'csi500_weekly_fundamentals.pkl')
    pe_df, pb_df = None, None
    if os.path.exists(fund_path):
        fund = pd.read_pickle(fund_path)
        pe_dict = fund.get('pe', {})
        pb_dict = fund.get('pb', {})
        if pe_dict:
            pe_df = pd.DataFrame(pe_dict)
            pe_df = pe_df.sort_index()
            # Align to price_df index
            pe_df = pe_df.reindex(price_df.index, method='ffill')
            print(f"PE matrix: {pe_df.shape}")
        if pb_dict:
            pb_df = pd.DataFrame(pb_dict)
            pb_df = pb_df.sort_index()
            pb_df = pb_df.reindex(price_df.index, method='ffill')
            print(f"PB matrix: {pb_df.shape}")

    print(f"Prices: {price_df.shape[0]} weeks x {price_df.shape[1]} stocks")
    return const, industries, price_df, pe_df, pb_df


def build_constituent_mask(price_df, const):
    """Build a boolean DataFrame: True if stock is in CSI500 at that date"""
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


def precompute_factors(price_df, pe_df, pb_df):
    """Pre-compute all factor DataFrames (higher = better for all)"""
    factors = {}
    returns = price_df.pct_change()

    # 1. Low Volatility (12-week rolling) - negate so lower vol = higher score
    vol12 = returns.rolling(12, min_periods=8).std() * np.sqrt(52)
    factors['low_vol'] = -vol12

    # 2. Low Volatility (20-week)
    vol20 = returns.rolling(20, min_periods=12).std() * np.sqrt(52)
    factors['low_vol_20w'] = -vol20

    # 3. Momentum 12 months (52 weeks)
    factors['mom_12m'] = price_df / price_df.shift(52) - 1

    # 4. Momentum 6 months (26 weeks)
    factors['mom_6m'] = price_df / price_df.shift(26) - 1

    # 5. Momentum 3 months (13 weeks)
    factors['mom_3m'] = price_df / price_df.shift(13) - 1

    # 6. Mean Reversion (4-week) - negate so lower recent return = higher score
    factors['mean_rev'] = -(price_df / price_df.shift(4) - 1)

    # 7. Quality Sharpe (26-week rolling)
    roll_mean = returns.rolling(26, min_periods=13).mean()
    roll_std = returns.rolling(26, min_periods=13).std()
    factors['quality_sharpe'] = roll_mean / roll_std.replace(0, np.nan)

    # 8. Value PE - negate so lower PE = higher score
    if pe_df is not None:
        # Filter extreme values
        pe_clean = pe_df.copy()
        pe_clean[pe_clean <= 0] = np.nan
        pe_clean[pe_clean > 300] = np.nan
        factors['value_pe'] = -pe_clean

    # 9. Value PB
    if pb_df is not None:
        pb_clean = pb_df.copy()
        pb_clean[pb_clean <= 0] = np.nan
        pb_clean[pb_clean > 30] = np.nan
        factors['value_pb'] = -pb_clean

    print(f"Pre-computed {len(factors)} factors")
    return factors


def cross_sectional_zscore(factor_df, mask):
    """Z-score normalize factor values cross-sectionally at each date, only for active stocks"""
    result = factor_df.copy()
    result[~mask] = np.nan
    # Cross-sectional z-score per row
    row_mean = result.mean(axis=1)
    row_std = result.std(axis=1)
    result = result.sub(row_mean, axis=0).div(row_std.replace(0, np.nan), axis=0)
    return result


def run_backtest(price_df, mask, score_df, top_n=10, rebal_freq=2,
                 sector_cap=0, industries=None, txn_bps=8):
    """
    Run backtest given a score DataFrame (higher = buy).
    Returns nav series and weekly returns.
    """
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    warmup = 54  # need enough history

    nav = 1.0
    nav_list = [nav]
    dates = [price_df.index[warmup - 1]]
    prev_holdings = set()
    weekly_rets = []
    total_txn = 0.0

    i = warmup
    while i < len(price_df) - 1:
        # Get scores for this date, only active constituents
        scores = score_df.iloc[i].copy()
        active = mask.iloc[i]
        scores[~active] = np.nan
        scores = scores.dropna()

        if len(scores) < top_n:
            # Hold previous or skip
            for j in range(i + 1, min(i + rebal_freq + 1, len(price_df))):
                nav_list.append(nav_list[-1])
                dates.append(price_df.index[j])
                weekly_rets.append(0.0)
            i += rebal_freq
            continue

        # Select top stocks
        if sector_cap > 0 and industries is not None:
            # Sector-diversified selection
            scores_sorted = scores.sort_values(ascending=False)
            selected = []
            sector_count = defaultdict(int)
            for stock in scores_sorted.index:
                sec = industries.get(stock, {}).get('industry', '其他')
                if sector_count[sec] < sector_cap:
                    selected.append(stock)
                    sector_count[sec] += 1
                    if len(selected) >= top_n:
                        break
        else:
            selected = scores.nlargest(top_n).index.tolist()

        if not selected:
            i += 1
            continue

        n_stocks = len(selected)
        selected_set = set(selected)

        # Transaction cost
        turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(n_stocks, 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        # Hold period
        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            # Equal-weight portfolio return
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


def calc_stats(nav_series, weekly_rets, total_txn, label=""):
    nav = nav_series.values
    if len(nav) < 20:
        return None
    n_weeks = len(nav) - 1
    years = n_weeks / 52.0
    if years < 0.5:
        return None

    cagr = (nav[-1] / nav[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    mdd = float(np.min(dd))

    wr = np.array(weekly_rets)
    sharpe = float(np.mean(wr) / np.std(wr) * np.sqrt(52)) if len(wr) > 10 and np.std(wr) > 0 else 0.0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    annual = {}
    for year in range(2021, 2027):
        mask = np.array([d.year == year for d in nav_series.index])
        if mask.sum() > 4:
            year_nav = nav[mask]
            annual[str(year)] = round((year_nav[-1] / year_nav[0] - 1) * 100, 1)

    return {
        'label': label,
        'cagr_pct': round(cagr * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'annual_returns': annual,
        'txn_pct': round(total_txn * 100, 2),
    }


def rolling_12m(nav_series):
    nav = nav_series.values
    if len(nav) <= 52:
        return None
    rets = []
    for s in range(len(nav) - 52):
        rets.append((nav[s + 52] / nav[s] - 1) * 100)
    rets = np.array(rets)
    return {
        'windows': len(rets),
        'win_rate': round(float(np.mean(rets > 0) * 100), 1),
        'mean': round(float(np.mean(rets)), 2),
        'median': round(float(np.median(rets)), 2),
        'min': round(float(np.min(rets)), 2),
        'max': round(float(np.max(rets)), 2),
        'p10': round(float(np.percentile(rets, 10)), 2),
        'p25': round(float(np.percentile(rets, 25)), 2),
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("CSI500 Multi-Factor Research v3 (Vectorized)")
    print("=" * 60)

    # Load
    const, industries, price_df, pe_df, pb_df = load_data()

    # Build constituent mask
    print("\nBuilding constituent mask...")
    mask = build_constituent_mask(price_df, const)
    active_counts = mask.sum(axis=1)
    print(f"  Active stocks per week: min={active_counts.min()}, max={active_counts.max()}, mean={active_counts.mean():.0f}")

    # Pre-compute factors
    print("\nPre-computing factors...")
    factors = precompute_factors(price_df, pe_df, pb_df)

    # Z-score normalize all factors
    print("Z-score normalizing...")
    z_factors = {}
    for name, df in factors.items():
        z_factors[name] = cross_sectional_zscore(df, mask)

    print(f"\nSetup done in {time.time()-t0:.1f}s")

    # Define configs
    configs = []

    # Single factors
    single_factors = list(factors.keys())
    for f in single_factors:
        for top_n in [10, 15, 20, 30]:
            for rebal in [2, 4]:
                configs.append({
                    'factors': {f: 1.0},
                    'top_n': top_n,
                    'rebal': rebal,
                    'sec_cap': 0,
                    'label': f"{f}_top{top_n}_{rebal}w",
                })
        # With sector cap
        for top_n in [15, 20]:
            configs.append({
                'factors': {f: 1.0},
                'top_n': top_n,
                'rebal': 2,
                'sec_cap': 3,
                'label': f"{f}_top{top_n}_2w_sec3",
            })

    # Composite strategies
    composites = [
        ({'low_vol': 0.5, 'quality_sharpe': 0.5}, 'LV_QS'),
        ({'low_vol': 0.5, 'mean_rev': 0.5}, 'LV_MR'),
        ({'low_vol': 0.5, 'mom_6m': 0.5}, 'LV_Mom6'),
        ({'low_vol': 0.5, 'mom_12m': 0.5}, 'LV_Mom12'),
        ({'low_vol': 0.3, 'mom_12m': 0.3, 'mean_rev': 0.4}, 'LV_Mom_MR'),
        ({'low_vol': 0.4, 'quality_sharpe': 0.3, 'mean_rev': 0.3}, 'LV_QS_MR'),
        ({'mom_6m': 0.5, 'mean_rev': 0.5}, 'Mom6_MR'),
        ({'quality_sharpe': 0.5, 'mean_rev': 0.5}, 'QS_MR'),
        ({'low_vol': 0.5, 'mom_3m': 0.5}, 'LV_Mom3'),
    ]

    has_pe = 'value_pe' in factors
    has_pb = 'value_pb' in factors
    if has_pe:
        composites += [
            ({'low_vol': 0.5, 'value_pe': 0.5}, 'LV_PE'),
            ({'value_pe': 0.5, 'quality_sharpe': 0.5}, 'PE_QS'),
            ({'value_pe': 0.4, 'low_vol': 0.3, 'mom_12m': 0.3}, 'PE_LV_Mom'),
            ({'value_pe': 0.3, 'low_vol': 0.3, 'quality_sharpe': 0.2, 'mom_6m': 0.2}, 'PE_LV_QS_Mom'),
        ]
    if has_pb:
        composites += [
            ({'low_vol': 0.5, 'value_pb': 0.5}, 'LV_PB'),
            ({'value_pb': 0.5, 'quality_sharpe': 0.5}, 'PB_QS'),
        ]
    if has_pe and has_pb:
        composites += [
            ({'value_pe': 0.3, 'value_pb': 0.3, 'low_vol': 0.4}, 'PE_PB_LV'),
            ({'value_pe': 0.25, 'value_pb': 0.25, 'low_vol': 0.25, 'quality_sharpe': 0.25}, 'PE_PB_LV_QS'),
        ]

    for weights, name in composites:
        # Check all required factors exist
        if not all(f in z_factors for f in weights):
            continue
        for top_n in [10, 15, 20]:
            for rebal in [2, 4]:
                configs.append({
                    'factors': weights,
                    'top_n': top_n,
                    'rebal': rebal,
                    'sec_cap': 0,
                    'label': f"C_{name}_t{top_n}_{rebal}w",
                })
        # With sector cap
        configs.append({
            'factors': weights,
            'top_n': 15,
            'rebal': 2,
            'sec_cap': 3,
            'label': f"C_{name}_t15_2w_s3",
        })

    print(f"\nTotal configs: {len(configs)}")
    print("Running backtests...\n")

    results = []
    best_nav = None
    best_label = None
    best_calmar = -999

    for idx, cfg in enumerate(configs):
        if idx % 25 == 0:
            print(f"  Progress: {idx}/{len(configs)} ({time.time()-t0:.0f}s)")

        # Build composite score
        weights = cfg['factors']
        score_df = None
        for f, w in weights.items():
            if f not in z_factors:
                score_df = None
                break
            if score_df is None:
                score_df = z_factors[f] * w
            else:
                # Add with alignment (use mean where one factor is missing)
                score_df = score_df.add(z_factors[f] * w, fill_value=0)

        if score_df is None:
            continue

        try:
            nav, wr, txn = run_backtest(
                price_df, mask, score_df,
                top_n=cfg['top_n'], rebal_freq=cfg['rebal'],
                sector_cap=cfg['sec_cap'], industries=industries,
                txn_bps=8)

            stats = calc_stats(nav, wr, txn, cfg['label'])
            if stats is None:
                continue
            results.append(stats)

            if stats['calmar'] > best_calmar:
                best_calmar = stats['calmar']
                best_nav = nav
                best_label = cfg['label']

        except Exception as e:
            print(f"  ERROR {cfg['label']}: {e}")

    results.sort(key=lambda x: -x['calmar'])

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(results)} strategies tested in {time.time()-t0:.0f}s")
    print(f"{'='*60}")

    print("\nTop 30 by Calmar:")
    for i, r in enumerate(results[:30]):
        print(f"  {i+1}. {r['label']}")
        print(f"     CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Sharpe={r['sharpe']} Calmar={r['calmar']}")
        print(f"     Annual: {r['annual_returns']}")

    print(f"\nBottom 5:")
    for r in results[-5:]:
        print(f"  {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']}")

    # Factor group summary
    print(f"\n{'='*60}")
    print("Factor Group Summary (avg Calmar):")
    groups = defaultdict(list)
    for r in results:
        label = r['label']
        if label.startswith('C_'):
            parts = label.split('_t')[0]
            groups[parts].append(r)
        else:
            factor = '_'.join(label.split('_')[:-2]) if '_top' in label else label.split('_top')[0]
            groups[factor].append(r)

    summary_rows = []
    for group, items in sorted(groups.items()):
        avg_cagr = np.mean([r['cagr_pct'] for r in items])
        avg_mdd = np.mean([r['mdd_pct'] for r in items])
        avg_calmar = np.mean([r['calmar'] for r in items])
        best_c = max(r['calmar'] for r in items)
        summary_rows.append((group, len(items), avg_cagr, avg_mdd, avg_calmar, best_c))
        print(f"  {group} ({len(items)}): CAGR={avg_cagr:.1f}% MDD={avg_mdd:.1f}% Calmar={avg_calmar:.3f} best={best_c:.3f}")

    # Rolling 12m for best
    rolling = None
    if best_nav is not None and len(best_nav) > 52:
        print(f"\n=== Rolling 12m: {best_label} ===")
        rolling = rolling_12m(best_nav)
        if rolling:
            for k, v in rolling.items():
                print(f"  {k}: {v}")

    # Save
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Multi-factor CSI500 vectorized, real constituents, no survivorship bias',
        'factors_available': list(factors.keys()),
        'has_pe': has_pe,
        'has_pb': has_pb,
        'total_configs': len(configs),
        'total_results': len(results),
        'strategies': results,
        'best_strategy': best_label,
        'best_calmar': round(best_calmar, 3),
        'best_rolling_12m': rolling,
        'elapsed_seconds': round(time.time() - t0, 1),
    }

    out_path = os.path.join(DATA_DIR, 'csi500_multifactor_research.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_path}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
