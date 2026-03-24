#!/usr/bin/env python3
"""
CSI500 Fast Multi-Factor Research - optimized for speed.
Pre-computes all factor scores in vectorized form, then runs strategies.
"""
import pandas as pd
import numpy as np
import json
import os
import time
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'baostock_cache')


def load_data():
    with open(os.path.join(CACHE_DIR, 'csi500_constituents_history.json')) as f:
        cd = json.load(f)
    with open(os.path.join(CACHE_DIR, 'csi500_industries.json')) as f:
        industries = json.load(f)
    price_df = pd.read_pickle(os.path.join(CACHE_DIR, 'csi500_all_weekly_prices.pkl'))
    print(f"Prices: {price_df.shape}, Stocks: {len(cd['all_unique_stocks'])}")
    return cd, industries, price_df


def precompute_factors(price_df):
    """Vectorized factor computation for ALL stocks at ALL times."""
    print("Pre-computing factors...")
    t0 = time.time()
    returns = price_df.pct_change()

    factors = {}

    # Low volatility (12-week rolling std, annualized)
    vol12 = returns.rolling(12, min_periods=8).std() * np.sqrt(52)
    factors['low_vol'] = -vol12  # lower vol = higher score

    # Low volatility (26-week)
    vol26 = returns.rolling(26, min_periods=16).std() * np.sqrt(52)
    factors['low_vol_26w'] = -vol26

    # Momentum 12m (52-week return)
    factors['momentum_12m'] = price_df / price_df.shift(52) - 1

    # Momentum 6m
    factors['momentum_6m'] = price_df / price_df.shift(26) - 1

    # Mean reversion (negative 4-week return)
    factors['mean_reversion'] = -(price_df / price_df.shift(4) - 1)

    # Quality Sharpe (26-week rolling Sharpe)
    roll_mean = returns.rolling(26, min_periods=16).mean()
    roll_std = returns.rolling(26, min_periods=16).std()
    factors['quality_sharpe'] = roll_mean / roll_std

    print(f"  Done in {time.time()-t0:.1f}s")
    return factors


def get_active_stocks(date_str, cd):
    active_date = None
    for d in cd['dates']:
        if d <= date_str:
            active_date = d
        else:
            break
    if active_date is None:
        active_date = cd['dates'][0]
    return set(cd['constituents'][active_date])


def run_single_factor(factor_scores, price_df, cd, industries,
                      top_n=15, rebal_freq=2, sector_cap=0, txn_bps=8):
    """Run strategy using pre-computed factor scores."""
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    warmup = 55  # enough for 52-week momentum

    nav = 1.0
    nav_list = [1.0]
    dates = [price_df.index[warmup - 1]]
    prev_holdings = set()
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        date = price_df.index[i]
        date_str = date.strftime('%Y-%m-%d')
        active = get_active_stocks(date_str, cd)
        available = list(active & set(price_df.columns))

        if len(available) < 50:
            i += 1
            continue

        # Get scores for available stocks
        scores = factor_scores.iloc[i][available].dropna()
        if len(scores) < top_n:
            i += 1
            continue

        # Select top stocks
        if sector_cap > 0 and industries:
            # Group by sector
            sector_stocks = defaultdict(list)
            for s in scores.index:
                ind = industries.get(s, {}).get('industry', 'Other')
                sector_stocks[ind].append((s, scores[s]))
            for sec in sector_stocks:
                sector_stocks[sec].sort(key=lambda x: -x[1])

            selected = []
            sector_order = sorted(sector_stocks.keys(),
                                  key=lambda sec: -sector_stocks[sec][0][1] if sector_stocks[sec] else -999)
            sector_idx = {s: 0 for s in sector_order}
            while len(selected) < top_n:
                added = False
                for sec in sector_order:
                    if sector_idx[sec] < min(sector_cap, len(sector_stocks[sec])):
                        selected.append(sector_stocks[sec][sector_idx[sec]][0])
                        sector_idx[sec] += 1
                        added = True
                        if len(selected) >= top_n:
                            break
                if not added:
                    break
        else:
            selected = list(scores.nlargest(top_n).index)

        if not selected:
            i += 1
            continue

        selected_set = set(selected)
        turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            rets_j = returns.iloc[j][selected].dropna()
            port_ret = float(rets_j.mean()) if len(rets_j) > 0 else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav *= (1 + port_ret)
            nav_list.append(nav)
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    return pd.Series(nav_list, index=dates), weekly_rets


def run_composite(factor_dict, weights, price_df, cd, industries,
                  top_n=15, rebal_freq=2, sector_cap=0, txn_bps=8):
    """Composite z-score strategy using pre-computed factors."""
    # Z-score normalize each factor cross-sectionally at each time step
    z_factors = {}
    for name, w in weights.items():
        if name not in factor_dict:
            continue
        f = factor_dict[name]
        # Cross-sectional z-score
        mean = f.mean(axis=1)
        std = f.std(axis=1)
        z_factors[name] = f.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)

    # Weighted composite
    composite = None
    total_w = 0
    for name, w in weights.items():
        if name not in z_factors:
            continue
        if composite is None:
            composite = z_factors[name] * w
        else:
            composite = composite.add(z_factors[name] * w, fill_value=0)
        total_w += w

    if composite is None or total_w == 0:
        return pd.Series(dtype=float), []

    composite = composite / total_w

    return run_single_factor(composite, price_df, cd, industries,
                             top_n=top_n, rebal_freq=rebal_freq,
                             sector_cap=sector_cap, txn_bps=txn_bps)


def calc_stats(nav_series, weekly_rets, label=""):
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
    }


def rolling_analysis(nav_series, window=52):
    nav = nav_series.values
    rets = np.array([nav[s+window]/nav[s]-1 for s in range(len(nav)-window)]) * 100
    if len(rets) == 0:
        return {}
    return {
        'windows': len(rets),
        'win_rate_pct': round(float(np.mean(rets > 0) * 100), 1),
        'mean_pct': round(float(np.mean(rets)), 2),
        'median_pct': round(float(np.median(rets)), 2),
        'min_pct': round(float(np.min(rets)), 2),
        'max_pct': round(float(np.max(rets)), 2),
        'p10_pct': round(float(np.percentile(rets, 10)), 2),
        'p25_pct': round(float(np.percentile(rets, 25)), 2),
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("CSI500 Fast Multi-Factor Research")
    print("=" * 60)

    cd, industries, price_df = load_data()
    factors = precompute_factors(price_df)

    # Define configs: (type, params, label)
    configs = []

    # Single factors
    for fname in ['low_vol', 'low_vol_26w', 'momentum_12m', 'momentum_6m', 'mean_reversion', 'quality_sharpe']:
        for top_n in [10, 15, 20, 30]:
            for rebal in [2, 4]:
                configs.append(('single', fname, top_n, rebal, 0,
                                f"{fname}_top{top_n}_{rebal}w"))
        # sector cap
        for top_n in [15, 20]:
            configs.append(('single', fname, top_n, 2, 3,
                            f"{fname}_top{top_n}_2w_sec3"))

    # Composites
    composites = [
        ({'low_vol': 0.5, 'quality_sharpe': 0.5}, 'LV_QS'),
        ({'low_vol': 0.5, 'mean_reversion': 0.5}, 'LV_MR'),
        ({'low_vol': 0.3, 'momentum_12m': 0.3, 'mean_reversion': 0.4}, 'LV_Mom_MR'),
        ({'low_vol': 0.5, 'momentum_6m': 0.5}, 'LV_Mom6m'),
        ({'low_vol': 0.4, 'quality_sharpe': 0.3, 'mean_reversion': 0.3}, 'LV_QS_MR'),
        ({'quality_sharpe': 0.5, 'mean_reversion': 0.5}, 'QS_MR'),
        ({'low_vol': 0.4, 'momentum_6m': 0.3, 'mean_reversion': 0.3}, 'LV_Mom6_MR'),
        ({'low_vol_26w': 0.5, 'quality_sharpe': 0.5}, 'LV26_QS'),
        ({'low_vol_26w': 0.4, 'mean_reversion': 0.3, 'quality_sharpe': 0.3}, 'LV26_MR_QS'),
        ({'low_vol': 0.25, 'low_vol_26w': 0.25, 'quality_sharpe': 0.25, 'mean_reversion': 0.25}, 'Equal4'),
    ]

    for weights, name in composites:
        for top_n in [10, 15, 20]:
            for rebal in [2, 4]:
                configs.append(('composite', weights, top_n, rebal, 0,
                                f"C_{name}_top{top_n}_{rebal}w"))
        configs.append(('composite', weights, 15, 2, 3,
                        f"C_{name}_top15_2w_sec3"))

    print(f"\nTotal configs: {len(configs)}")
    print("Running strategies...\n")

    results = []
    best_nav = None
    best_label = None
    best_calmar = -999

    for idx, cfg in enumerate(configs):
        if idx % 30 == 0:
            print(f"  Progress: {idx}/{len(configs)} ({time.time()-t0:.0f}s)")

        try:
            if cfg[0] == 'single':
                _, fname, top_n, rebal, sec_cap, label = cfg
                nav, wr = run_single_factor(factors[fname], price_df, cd, industries,
                                            top_n=top_n, rebal_freq=rebal, sector_cap=sec_cap)
            else:
                _, weights, top_n, rebal, sec_cap, label = cfg
                nav, wr = run_composite(factors, weights, price_df, cd, industries,
                                        top_n=top_n, rebal_freq=rebal, sector_cap=sec_cap)

            stats = calc_stats(nav, wr, label)
            if stats is None:
                continue
            results.append(stats)

            if stats['calmar'] > best_calmar:
                best_calmar = stats['calmar']
                best_nav = nav
                best_label = label

        except Exception as e:
            print(f"  ERROR {cfg[-1]}: {e}")
            import traceback; traceback.print_exc()
            continue

    results.sort(key=lambda x: -x['calmar'])

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(results)} strategies completed")
    print(f"{'='*60}")

    print("\n=== TOP 25 BY CALMAR ===")
    for i, r in enumerate(results[:25]):
        print(f"  {i+1:2d}. {r['label']}")
        print(f"      CAGR={r['cagr_pct']:+.1f}%  MDD={r['mdd_pct']:.1f}%  Sharpe={r['sharpe']:.3f}  Calmar={r['calmar']:.3f}")
        print(f"      Annual: {r['annual_returns']}")

    print("\n=== BOTTOM 5 ===")
    for r in results[-5:]:
        print(f"  {r['label']}: CAGR={r['cagr_pct']:+.1f}%  MDD={r['mdd_pct']:.1f}%  Calmar={r['calmar']:.3f}")

    # Factor group averages
    print(f"\n=== FACTOR GROUP AVERAGES ===")
    groups = defaultdict(list)
    for r in results:
        lbl = r['label']
        if lbl.startswith('C_'):
            groups['Composite'].append(r)
        else:
            factor = lbl.split('_top')[0]
            groups[factor].append(r)

    for group in sorted(groups.keys()):
        items = groups[group]
        avg_cagr = np.mean([r['cagr_pct'] for r in items])
        avg_mdd = np.mean([r['mdd_pct'] for r in items])
        avg_calmar = np.mean([r['calmar'] for r in items])
        best = max(r['calmar'] for r in items)
        print(f"  {group:20s} ({len(items):2d}): CAGR={avg_cagr:+.1f}%  MDD={avg_mdd:.1f}%  Calmar={avg_calmar:.3f}  best={best:.3f}")

    # Rolling analysis on best
    rolling = None
    if best_nav is not None and len(best_nav) > 52:
        print(f"\n=== ROLLING 12M ANALYSIS: {best_label} ===")
        rolling = rolling_analysis(best_nav)
        for k, v in rolling.items():
            print(f"  {k}: {v}")

    # Save results
    output = {
        'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'CSI500 multi-factor with real constituents (no survivorship bias), price-based factors only',
        'total_configs': len(configs),
        'total_results': len(results),
        'strategies': results,
        'best_strategy': best_label,
        'best_calmar': round(best_calmar, 3),
        'best_rolling_12m': rolling,
    }

    out_path = os.path.join(DATA_DIR, 'csi500_multifactor_research.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_path}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
