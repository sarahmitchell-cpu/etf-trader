#!/usr/bin/env python3
"""
CSI500 Multi-Factor Research v2 (No Survivorship Bias)
- Uses cached prices + PE/PB fundamentals
- Factors: Low Vol, Value (PE/PB), Mean Reversion, Momentum, Composites
- Skips ROE (too slow to fetch for 928 stocks)
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


def load_all_data():
    """Load all cached data"""
    # Constituents
    with open(os.path.join(CACHE_DIR, 'csi500_constituents_history.json')) as f:
        constituent_data = json.load(f)
    print(f"Constituents: {len(constituent_data['all_unique_stocks'])} stocks")

    # Industries
    with open(os.path.join(CACHE_DIR, 'csi500_industries.json')) as f:
        industries = json.load(f)
    print(f"Industries: {len(industries)} stocks")

    # Prices
    price_df = pd.read_pickle(os.path.join(CACHE_DIR, 'csi500_all_weekly_prices.pkl'))
    print(f"Prices: {price_df.shape[0]} weeks x {price_df.shape[1]} stocks")

    # Fundamentals (PE/PB)
    fund_path = os.path.join(CACHE_DIR, 'csi500_weekly_fundamentals.pkl')
    fund_data = None
    if os.path.exists(fund_path):
        fund_data = pd.read_pickle(fund_path)
        pe_count = len(fund_data.get('pe', {}))
        pb_count = len(fund_data.get('pb', {}))
        print(f"Fundamentals: PE={pe_count}, PB={pb_count}")
    else:
        print("WARNING: No fundamental data cached! Run csi500_fetch_fundamentals.py first.")

    return constituent_data, industries, price_df, fund_data


def get_active_constituents(date, constituent_data):
    rebal_dates = constituent_data['dates']
    constituents = constituent_data['constituents']
    active_date = None
    for d in rebal_dates:
        if d <= date.strftime('%Y-%m-%d'):
            active_date = d
        else:
            break
    if active_date is None:
        active_date = rebal_dates[0]
    return set(constituents[active_date])


def compute_factor_scores(stocks, price_df, fund_data, industries, i, factor_type, vol_lookback=12):
    """Compute factor scores for a set of stocks at time index i"""
    current_date = price_df.index[i]
    pe_dict = fund_data.get('pe', {}) if fund_data else {}
    pb_dict = fund_data.get('pb', {}) if fund_data else {}
    scores = {}

    for stock in stocks:
        prices_slice = price_df[stock].iloc[max(0, i - max(vol_lookback, 52)):i + 1]
        valid_prices = prices_slice.dropna()
        if len(valid_prices) < 10:
            continue

        if factor_type == 'low_vol':
            rets = valid_prices.pct_change().dropna()
            if len(rets) >= vol_lookback // 2:
                vol = float(rets.tail(vol_lookback).std()) * np.sqrt(52)
                if vol > 0:
                    scores[stock] = -vol  # lower vol = higher score

        elif factor_type == 'value_pe':
            if stock in pe_dict:
                pe_s = pe_dict[stock]
                mask = pe_s.index <= current_date
                if mask.any():
                    pe = float(pe_s[mask].iloc[-1])
                    if not np.isnan(pe) and 0 < pe < 300:
                        scores[stock] = -pe  # lower PE = higher score

        elif factor_type == 'value_pb':
            if stock in pb_dict:
                pb_s = pb_dict[stock]
                mask = pb_s.index <= current_date
                if mask.any():
                    pb = float(pb_s[mask].iloc[-1])
                    if not np.isnan(pb) and 0 < pb < 30:
                        scores[stock] = -pb

        elif factor_type == 'momentum_12m':
            if len(valid_prices) >= 52:
                p0 = valid_prices.iloc[-52]
                p1 = valid_prices.iloc[-1]
                if p0 > 0:
                    scores[stock] = p1 / p0 - 1

        elif factor_type == 'momentum_6m':
            if len(valid_prices) >= 26:
                p0 = valid_prices.iloc[-26]
                p1 = valid_prices.iloc[-1]
                if p0 > 0:
                    scores[stock] = p1 / p0 - 1

        elif factor_type == 'mean_reversion':
            # Short-term reversal: worst 4-week return = buy
            if len(valid_prices) >= 4:
                p0 = valid_prices.iloc[-4]
                p1 = valid_prices.iloc[-1]
                if p0 > 0:
                    scores[stock] = -(p1 / p0 - 1)  # lower recent return = higher score

        elif factor_type == 'quality_sharpe':
            # Use 26-week Sharpe as quality proxy
            rets = valid_prices.pct_change().dropna()
            if len(rets) >= 26:
                r26 = rets.tail(26)
                if r26.std() > 0:
                    scores[stock] = float(r26.mean() / r26.std())

    return scores


def run_strategy(price_df, constituent_data, industries, fund_data,
                 factor_type='low_vol', top_n=10, rebal_freq=2,
                 vol_lookback=12, sector_cap=0, txn_bps=8):
    """Run a single-factor strategy"""
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change(fill_method=None)
    warmup = max(vol_lookback + 4, 54)  # need 52 weeks for momentum

    nav = [1.0]
    dates = [price_df.index[warmup - 1]]
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        current_date = price_df.index[i]
        active = get_active_constituents(current_date, constituent_data)
        available = active & set(price_df.columns)

        if len(available) < 50:
            i += 1
            continue

        scores = compute_factor_scores(available, price_df, fund_data, industries, i, factor_type, vol_lookback)

        if len(scores) < top_n:
            i += 1
            continue

        # Select stocks
        if sector_cap > 0:
            sector_stocks = defaultdict(list)
            for s in scores:
                ind = industries.get(s, {}).get('industry', '其他')
                sector_stocks[ind].append((s, scores[s]))
            for sec in sector_stocks:
                sector_stocks[sec].sort(key=lambda x: -x[1])

            selected = []
            sector_order = sorted(sector_stocks.keys(),
                                  key=lambda s: -sector_stocks[s][0][1] if sector_stocks[s] else -999)
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
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            selected = [s for s, _ in ranked[:top_n]]

        if not selected:
            i += 1
            continue

        selected_set = set(selected)
        turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            rets = []
            for s in selected:
                if s in returns.columns and j < len(returns):
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

    return pd.Series(nav, index=dates), weekly_rets, total_txn


def run_composite(price_df, constituent_data, industries, fund_data,
                  weights=None, top_n=10, rebal_freq=2,
                  vol_lookback=12, sector_cap=0, txn_bps=8):
    """Run composite z-score strategy"""
    if weights is None:
        weights = {'low_vol': 0.5, 'value_pe': 0.5}

    txn_cost = txn_bps / 10000
    returns = price_df.pct_change(fill_method=None)
    warmup = max(vol_lookback + 4, 54)

    nav = [1.0]
    dates = [price_df.index[warmup - 1]]
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        current_date = price_df.index[i]
        active = get_active_constituents(current_date, constituent_data)
        available = active & set(price_df.columns)

        if len(available) < 50:
            i += 1
            continue

        # Get raw scores for each factor
        raw = {}
        for factor in weights:
            raw[factor] = compute_factor_scores(available, price_df, fund_data, industries, i, factor, vol_lookback)

        # Z-score normalize each factor
        z_scores = {}
        for factor, values in raw.items():
            if len(values) < 20:
                continue
            arr = np.array(list(values.values()))
            mean, std = np.mean(arr), np.std(arr)
            if std > 0:
                z_scores[factor] = {s: (v - mean) / std for s, v in values.items()}

        # Composite
        all_stocks = set()
        for f in z_scores:
            all_stocks |= set(z_scores[f].keys())

        composite = {}
        min_factors = max(1, len(weights) // 2)
        for stock in all_stocks:
            total_w, total_s = 0, 0
            for f, w in weights.items():
                if stock in z_scores.get(f, {}):
                    total_s += w * z_scores[f][stock]
                    total_w += w
            if total_w > 0 and sum(1 for f in weights if stock in z_scores.get(f, {})) >= min_factors:
                composite[stock] = total_s / total_w

        if len(composite) < top_n:
            i += 1
            continue

        # Select
        if sector_cap > 0:
            sector_stocks = defaultdict(list)
            for s in composite:
                ind = industries.get(s, {}).get('industry', '其他')
                sector_stocks[ind].append((s, composite[s]))
            for sec in sector_stocks:
                sector_stocks[sec].sort(key=lambda x: -x[1])
            selected = []
            sector_order = sorted(sector_stocks.keys(),
                                  key=lambda s: -sector_stocks[s][0][1] if sector_stocks[s] else -999)
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
            ranked = sorted(composite.items(), key=lambda x: -x[1])
            selected = [s for s, _ in ranked[:top_n]]

        if not selected:
            i += 1
            continue

        selected_set = set(selected)
        turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            rets = []
            for s in selected:
                if s in returns.columns and j < len(returns):
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

    return pd.Series(nav, index=dates), weekly_rets, total_txn


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
        'annual_returns': annual,
        'total_txn_pct': round(total_txn * 100, 2),
    }


def rolling_analysis(nav_series, window=52):
    nav = nav_series.values
    dates = nav_series.index
    rets = []
    for s in range(len(nav) - window):
        rets.append(nav[s + window] / nav[s] - 1)
    rets = np.array(rets) * 100
    return {
        'windows': len(rets),
        'win_rate': round(float(np.mean(rets > 0) * 100), 1),
        'mean': round(float(np.mean(rets)), 2),
        'median': round(float(np.median(rets)), 2),
        'min': round(float(np.min(rets)), 2),
        'max': round(float(np.max(rets)), 2),
        'p10': round(float(np.percentile(rets, 10)), 2),
        'p25': round(float(np.percentile(rets, 25)), 2),
        'p75': round(float(np.percentile(rets, 75)), 2),
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("CSI500 Multi-Factor Research v2")
    print("=" * 60)

    constituent_data, industries, price_df, fund_data = load_all_data()

    has_fundamentals = fund_data is not None and len(fund_data.get('pe', {})) > 100

    # Define configs
    configs = []

    # === SINGLE FACTORS (price-based, always available) ===
    for factor in ['low_vol', 'momentum_12m', 'momentum_6m', 'mean_reversion', 'quality_sharpe']:
        for top_n in [10, 15, 20, 30]:
            for rebal in [2, 4]:
                configs.append(('single', factor, top_n, rebal, 0,
                                f"{factor}_top{top_n}_{rebal}w"))
        # With sector cap
        for top_n in [15, 20]:
            configs.append(('single', factor, top_n, 2, 3,
                            f"{factor}_top{top_n}_2w_sec3"))

    # === VALUE FACTORS (need fundamentals) ===
    if has_fundamentals:
        for factor in ['value_pe', 'value_pb']:
            for top_n in [10, 15, 20, 30]:
                for rebal in [2, 4]:
                    configs.append(('single', factor, top_n, rebal, 0,
                                    f"{factor}_top{top_n}_{rebal}w"))
            for top_n in [15, 20]:
                configs.append(('single', factor, top_n, 2, 3,
                                f"{factor}_top{top_n}_2w_sec3"))

    # === COMPOSITE STRATEGIES ===
    composites = [
        ({'low_vol': 0.5, 'quality_sharpe': 0.5}, 'LV50_QS50'),
        ({'low_vol': 0.5, 'mean_reversion': 0.5}, 'LV50_MR50'),
        ({'low_vol': 0.3, 'momentum_12m': 0.3, 'mean_reversion': 0.4}, 'LV30_Mom30_MR40'),
        ({'low_vol': 0.5, 'momentum_6m': 0.5}, 'LV50_Mom6m50'),
    ]

    if has_fundamentals:
        composites += [
            ({'low_vol': 0.5, 'value_pe': 0.5}, 'LV50_PE50'),
            ({'low_vol': 0.5, 'value_pb': 0.5}, 'LV50_PB50'),
            ({'value_pe': 0.5, 'quality_sharpe': 0.5}, 'PE50_QS50'),
            ({'value_pe': 0.3, 'value_pb': 0.3, 'low_vol': 0.4}, 'PE30_PB30_LV40'),
            ({'low_vol': 0.25, 'value_pe': 0.25, 'value_pb': 0.25, 'quality_sharpe': 0.25}, 'LV_PE_PB_QS_25each'),
            ({'low_vol': 0.3, 'value_pe': 0.2, 'quality_sharpe': 0.3, 'momentum_6m': 0.2}, 'LV30_PE20_QS30_Mom20'),
            ({'value_pe': 0.4, 'low_vol': 0.3, 'momentum_12m': 0.3}, 'PE40_LV30_Mom30'),
        ]

    for weights, name in composites:
        for top_n in [10, 15, 20]:
            for rebal in [2, 4]:
                configs.append(('composite', weights, top_n, rebal, 0,
                                f"Comp_{name}_top{top_n}_{rebal}w"))
        # With sector cap
        configs.append(('composite', weights, 15, 2, 3,
                        f"Comp_{name}_top15_2w_sec3"))

    print(f"\nTotal configs: {len(configs)}")
    print("Running strategies...\n")

    results = []
    best_nav = None
    best_label = None
    best_calmar = -999

    for idx, cfg in enumerate(configs):
        if idx % 20 == 0:
            print(f"  Progress: {idx}/{len(configs)} ({time.time()-t0:.0f}s)")

        try:
            if cfg[0] == 'single':
                _, factor, top_n, rebal, sec_cap, label = cfg
                nav, wr, txn = run_strategy(
                    price_df, constituent_data, industries, fund_data,
                    factor_type=factor, top_n=top_n, rebal_freq=rebal, sector_cap=sec_cap)
            else:
                _, weights, top_n, rebal, sec_cap, label = cfg
                nav, wr, txn = run_composite(
                    price_df, constituent_data, industries, fund_data,
                    weights=weights, top_n=top_n, rebal_freq=rebal, sector_cap=sec_cap)

            stats = calc_stats(nav, wr, txn, label)
            if stats is None:
                continue
            results.append(stats)

            if stats['calmar'] > best_calmar:
                best_calmar = stats['calmar']
                best_nav = nav
                best_label = label

        except Exception as e:
            print(f"  ERROR {cfg[-1]}: {e}")
            continue

    results.sort(key=lambda x: -x['calmar'])

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(results)} strategies")
    print(f"{'='*60}")

    print("\nTop 25 by Calmar:")
    for i, r in enumerate(results[:25]):
        print(f"  {i+1}. {r['label']}")
        print(f"     CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Sharpe={r['sharpe']} Calmar={r['calmar']}")
        print(f"     Annual: {r['annual_returns']}")

    print(f"\nBottom 5:")
    for r in results[-5:]:
        print(f"  {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']}")

    # Factor summary
    print(f"\n{'='*60}")
    print("Factor Group Averages:")
    groups = defaultdict(list)
    for r in results:
        label = r['label']
        if label.startswith('Comp_'):
            groups['Composite'].append(r)
        else:
            factor = label.split('_top')[0]
            groups[factor].append(r)

    for group, items in sorted(groups.items()):
        avg_cagr = np.mean([r['cagr_pct'] for r in items])
        avg_mdd = np.mean([r['mdd_pct'] for r in items])
        avg_calmar = np.mean([r['calmar'] for r in items])
        best_c = max(r['calmar'] for r in items)
        print(f"  {group} ({len(items)}): avg CAGR={avg_cagr:.1f}% MDD={avg_mdd:.1f}% Calmar={avg_calmar:.3f} (best={best_c:.3f})")

    # Rolling analysis on best
    rolling = None
    if best_nav is not None and len(best_nav) > 52:
        print(f"\n=== Rolling 12m for Best: {best_label} ===")
        rolling = rolling_analysis(best_nav)
        for k, v in rolling.items():
            print(f"  {k}: {v}")

    # Save
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Multi-factor CSI500 with real constituents (no survivorship bias)',
        'has_fundamentals': has_fundamentals,
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
