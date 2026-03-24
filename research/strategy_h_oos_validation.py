#!/usr/bin/env python3
"""
Strategy H Out-of-Sample Validation
====================================
1. Split: In-Sample 2015-01-01 ~ 2022-12-31, Out-of-Sample 2023-01-01 ~ 2026-03-24
2. Full parameter search on IS data only
3. Parameter neighborhood robustness check (nearby params should also work)
4. Validate surviving variants on OOS data
5. Compare with original H1-H8 selections
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings, json
warnings.filterwarnings('ignore')

# ========== Config ==========
INDICES = {
    '沪深300': '000300.SS',
    '上证50ETF': '510050.SS',
    '科创50ETF': '588000.SS',
    '恒生指数': '^HSI',
    '国企指数': '^HSCE',
    'H股ETF': '510900.SS',
}

CUM_DAYS_LIST = list(range(1, 11))
THRESHOLD_PCT_LIST = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
HOLD_DAYS_LIST = list(range(1, 51))
STOP_LOSS_LIST = [None, -3, -5, -7]

IS_START = '2015-01-01'
IS_END = '2022-12-31'
OOS_START = '2023-01-01'
OOS_END = '2026-03-24'
FULL_START = '2015-01-01'
FULL_END = '2026-03-24'

RISK_FREE_RATE = 0.02
MIN_TRADES = 3

# Original H variants for comparison
ORIGINAL_H = {
    'H1': {'ticker': '588000.SS', 'index': '科创50ETF', 'direction': 'dip', 'cum_days': 3, 'threshold_pct': 7, 'hold_days': 19, 'stop_loss': None},
    'H2': {'ticker': '000300.SS', 'index': '沪深300', 'direction': 'dip', 'cum_days': 8, 'threshold_pct': 4, 'hold_days': 11, 'stop_loss': -3},
    'H3': {'ticker': '588000.SS', 'index': '科创50ETF', 'direction': 'rally', 'cum_days': 5, 'threshold_pct': 6, 'hold_days': 14, 'stop_loss': -5},
    'H4': {'ticker': '588000.SS', 'index': '科创50ETF', 'direction': 'rally', 'cum_days': 1, 'threshold_pct': 3, 'hold_days': 2, 'stop_loss': None},
    'H5': {'ticker': '^HSI', 'index': '恒生指数', 'direction': 'dip', 'cum_days': 5, 'threshold_pct': 5, 'hold_days': 6, 'stop_loss': -7},
    'H6': {'ticker': '510050.SS', 'index': '上证50ETF', 'direction': 'dip', 'cum_days': 6, 'threshold_pct': 4, 'hold_days': 4, 'stop_loss': None},
    'H7': {'ticker': None, 'index': '通用', 'direction': 'dip', 'cum_days': 6, 'threshold_pct': 6, 'hold_days': 4, 'stop_loss': None},
    'H8': {'ticker': '000300.SS', 'index': '沪深300', 'direction': 'rally', 'cum_days': 1, 'threshold_pct': 2, 'hold_days': 2, 'stop_loss': None},
}

# ========== Backtest Engine ==========
def backtest_strategy(closes, n, cum_returns, cum_days, threshold_pct, hold_days, direction='dip', stop_loss_pct=None):
    if n < cum_days + hold_days + 10:
        return None, []
    nav = np.ones(n)
    position = False
    entry_price = 0
    entry_idx = 0
    hold_count = 0
    trades = []
    for i in range(1, n):
        if position:
            daily_ret = closes[i] / closes[i-1] - 1
            nav[i] = nav[i-1] * (1 + daily_ret)
            hold_count += 1
            current_trade_ret = closes[i] / entry_price - 1
            hit_stop = stop_loss_pct is not None and current_trade_ret * 100 <= stop_loss_pct
            if hold_count >= hold_days or hit_stop:
                trades.append({'return': current_trade_ret, 'hold_days': hold_count, 'stop_loss': hit_stop})
                position = False
        else:
            nav[i] = nav[i-1]
            cr = cum_returns[i]
            if not np.isnan(cr):
                trigger = (direction == 'dip' and cr <= -threshold_pct) or (direction == 'rally' and cr >= threshold_pct)
                if trigger:
                    position = True
                    entry_price = closes[i]
                    entry_idx = i
                    hold_count = 0
    if position:
        trades.append({'return': closes[-1] / entry_price - 1, 'hold_days': n - 1 - entry_idx, 'stop_loss': False})
    return nav, trades

def calc_metrics(nav, trades, n_days):
    if nav is None or n_days < 2:
        return None
    n_years = n_days / 252
    if n_years <= 0:
        return None
    annualized_return = (nav[-1] / nav[0]) ** (1 / n_years) - 1
    cummax = np.maximum.accumulate(nav)
    drawdown = (nav - cummax) / cummax
    max_drawdown = np.min(drawdown)
    daily_ret = np.diff(nav) / nav[:-1]
    std = np.std(daily_ret)
    sharpe = (np.mean(daily_ret) - RISK_FREE_RATE / 252) / std * np.sqrt(252) if std > 0 else 0
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    n_trades = len(trades)
    if n_trades > 0:
        rets = [t['return'] for t in trades]
        win_rate = sum(1 for r in rets if r > 0) / n_trades
    else:
        win_rate = 0
    return {
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'calmar': calmar,
        'n_trades': n_trades,
        'win_rate': win_rate,
    }

# ========== Data Download ==========
def download_all():
    """Download full period data and split into IS/OOS."""
    is_data = {}
    oos_data = {}
    full_data = {}

    for name, ticker in INDICES.items():
        print(f"Downloading {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=FULL_START, end=FULL_END, progress=False)
            if df is not None and len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[['Close']].dropna()
                df.columns = ['close']

                is_df = df[df.index <= IS_END]
                oos_df = df[df.index >= OOS_START]

                if len(is_df) > 100 and len(oos_df) > 50:
                    is_data[name] = is_df
                    oos_data[name] = oos_df
                    full_data[name] = df
                    print(f"  IS: {len(is_df)} days, OOS: {len(oos_df)} days, Full: {len(df)} days")
                else:
                    print(f"  Insufficient data for split")
            else:
                print(f"  Download insufficient")
        except Exception as e:
            print(f"  Failed: {e}")

    return is_data, oos_data, full_data

def precompute(data_dict):
    """Pre-compute cumulative returns for all indices."""
    result = {}
    for name, df in data_dict.items():
        closes = df['close'].values
        n = len(closes)
        cum_rets = {}
        for cd in CUM_DAYS_LIST:
            cr = np.full(n, np.nan)
            for i in range(cd, n):
                cr[i] = (closes[i] / closes[i - cd] - 1) * 100
            cum_rets[cd] = cr
        result[name] = {'closes': closes, 'n': n, 'cum_returns': cum_rets}
    return result

# ========== Run single variant on data ==========
def run_single(pc, cum_days, threshold_pct, hold_days, direction, stop_loss):
    nav, trades = backtest_strategy(
        pc['closes'], pc['n'], pc['cum_returns'][cum_days],
        cum_days, threshold_pct, hold_days, direction, stop_loss
    )
    return calc_metrics(nav, trades, pc['n'])

# ========== Parameter Neighborhood Check ==========
def check_neighborhood(pc_dict, idx_name, cum_days, threshold_pct, hold_days, direction, stop_loss, min_sharpe=0.0):
    """Check if nearby parameter combinations also perform well."""
    # Neighborhood: cum_days +/-1, threshold +/-1, hold_days +/-2
    cd_range = [cd for cd in [cum_days-1, cum_days, cum_days+1] if 1 <= cd <= 10]
    th_candidates = sorted(set(THRESHOLD_PCT_LIST) & set(range(max(2, threshold_pct-1), threshold_pct+2)))
    if not th_candidates:
        th_candidates = [threshold_pct]
    hd_range = [hd for hd in range(max(1, hold_days-2), min(51, hold_days+3))]

    total = 0
    good = 0
    sharpes = []

    pc = pc_dict[idx_name]
    for cd in cd_range:
        for th in th_candidates:
            for hd in hd_range:
                m = run_single(pc, cd, th, hd, direction, stop_loss)
                if m and m['n_trades'] >= MIN_TRADES:
                    total += 1
                    sharpes.append(m['sharpe'])
                    if m['sharpe'] > min_sharpe:
                        good += 1

    if total == 0:
        return 0, 0, 0

    return good / total, np.mean(sharpes), total

# ========== Main ==========
def main():
    print("=" * 90)
    print("Strategy H Out-of-Sample Validation")
    print(f"In-Sample: {IS_START} ~ {IS_END}")
    print(f"Out-of-Sample: {OOS_START} ~ {OOS_END}")
    print("=" * 90)

    is_data, oos_data, full_data = download_all()
    print(f"\nLoaded {len(is_data)} indices\n")

    if not is_data:
        print("No data available!")
        return

    is_pc = precompute(is_data)
    oos_pc = precompute(oos_data)
    full_pc = precompute(full_data)

    # ================================================================
    # PART 1: Validate original H1-H8 on IS vs OOS
    # ================================================================
    print("\n" + "=" * 90)
    print("PART 1: Original H1-H8 In-Sample vs Out-of-Sample Performance")
    print("=" * 90)

    report_lines = []
    header = f"{'Variant':<6} {'Index':<12} {'Dir':<6} {'Params':<25} | {'IS Sharpe':>10} {'IS CAGR':>10} {'IS MDD':>10} {'IS Trades':>10} | {'OOS Sharpe':>10} {'OOS CAGR':>10} {'OOS MDD':>10} {'OOS Trades':>10} | {'Decay%':>8}"
    print(header)
    print("-" * len(header))

    h_results = {}
    for hid, hv in ORIGINAL_H.items():
        idx = hv['index']
        direction = hv['direction']
        cd = hv['cum_days']
        th = hv['threshold_pct']
        hd = hv['hold_days']
        sl = hv['stop_loss']

        if hv['ticker'] is None:
            # H7 universal - test on all indices
            is_sharpes = []
            oos_sharpes = []
            for idx_name in is_pc:
                is_m = run_single(is_pc[idx_name], cd, th, hd, direction, sl)
                oos_m = run_single(oos_pc[idx_name], cd, th, hd, direction, sl)
                if is_m and is_m['n_trades'] >= 2:
                    is_sharpes.append(is_m['sharpe'])
                if oos_m and oos_m['n_trades'] >= 1:
                    oos_sharpes.append(oos_m['sharpe'])

            is_avg = np.mean(is_sharpes) if is_sharpes else 0
            oos_avg = np.mean(oos_sharpes) if oos_sharpes else 0
            decay = ((oos_avg - is_avg) / abs(is_avg) * 100) if is_avg != 0 else 0

            params = f"{cd}d {'drop' if direction=='dip' else 'rise'}>{th}% hold{hd}d"
            if sl: params += f" SL{sl}%"
            print(f"{hid:<6} {'Universal':<12} {direction:<6} {params:<25} | {is_avg:>10.2f} {'':>10} {'':>10} {len(is_sharpes):>10} | {oos_avg:>10.2f} {'':>10} {'':>10} {len(oos_sharpes):>10} | {decay:>7.0f}%")
            h_results[hid] = {'is_sharpe': is_avg, 'oos_sharpe': oos_avg, 'decay': decay}
        else:
            if idx not in is_pc:
                print(f"{hid:<6} {idx:<12} -> NO DATA")
                continue

            is_m = run_single(is_pc[idx], cd, th, hd, direction, sl)
            oos_m = run_single(oos_pc[idx], cd, th, hd, direction, sl)

            is_sharpe = is_m['sharpe'] if is_m else 0
            is_cagr = is_m['annualized_return'] * 100 if is_m else 0
            is_mdd = is_m['max_drawdown'] * 100 if is_m else 0
            is_trades = is_m['n_trades'] if is_m else 0

            oos_sharpe = oos_m['sharpe'] if oos_m else 0
            oos_cagr = oos_m['annualized_return'] * 100 if oos_m else 0
            oos_mdd = oos_m['max_drawdown'] * 100 if oos_m else 0
            oos_trades = oos_m['n_trades'] if oos_m else 0

            decay = ((oos_sharpe - is_sharpe) / abs(is_sharpe) * 100) if is_sharpe != 0 else 0

            params = f"{cd}d {'drop' if direction=='dip' else 'rise'}>{th}% hold{hd}d"
            if sl: params += f" SL{sl}%"
            print(f"{hid:<6} {idx:<12} {direction:<6} {params:<25} | {is_sharpe:>10.2f} {is_cagr:>9.1f}% {is_mdd:>9.1f}% {is_trades:>10} | {oos_sharpe:>10.2f} {oos_cagr:>9.1f}% {oos_mdd:>9.1f}% {oos_trades:>10} | {decay:>7.0f}%")
            h_results[hid] = {'is_sharpe': is_sharpe, 'oos_sharpe': oos_sharpe, 'decay': decay,
                              'is_cagr': is_cagr, 'oos_cagr': oos_cagr, 'is_mdd': is_mdd, 'oos_mdd': oos_mdd,
                              'is_trades': is_trades, 'oos_trades': oos_trades}

    # Summary
    survived = sum(1 for v in h_results.values() if v['oos_sharpe'] > 0)
    print(f"\nSummary: {survived}/{len(h_results)} variants have positive OOS Sharpe")
    for hid, v in h_results.items():
        status = "PASS" if v['oos_sharpe'] > 0 else "FAIL"
        print(f"  {hid}: IS={v['is_sharpe']:.2f} -> OOS={v['oos_sharpe']:.2f} (decay {v['decay']:.0f}%) [{status}]")

    # ================================================================
    # PART 2: Neighborhood Robustness Check on IS data
    # ================================================================
    print("\n" + "=" * 90)
    print("PART 2: Parameter Neighborhood Robustness (IS data)")
    print("=" * 90)

    for hid, hv in ORIGINAL_H.items():
        if hv['ticker'] is None:
            # H7 universal - check neighborhood across all indices
            all_robust = []
            for idx_name in is_pc:
                ratio, avg_s, total = check_neighborhood(
                    is_pc, idx_name, hv['cum_days'], hv['threshold_pct'],
                    hv['hold_days'], hv['direction'], hv['stop_loss'], min_sharpe=0
                )
                all_robust.append(ratio)
            avg_robust = np.mean(all_robust) if all_robust else 0
            print(f"  {hid} (Universal): Avg neighborhood positive rate = {avg_robust*100:.0f}%")
        else:
            idx = hv['index']
            if idx not in is_pc:
                print(f"  {hid}: NO DATA for {idx}")
                continue
            ratio, avg_sharpe, total = check_neighborhood(
                is_pc, idx, hv['cum_days'], hv['threshold_pct'],
                hv['hold_days'], hv['direction'], hv['stop_loss'], min_sharpe=0
            )
            print(f"  {hid} ({idx}): {ratio*100:.0f}% of {total} neighbors have Sharpe>0, avg neighbor Sharpe={avg_sharpe:.2f}")

    # ================================================================
    # PART 3: Full IS parameter search -> top candidates -> OOS validation
    # ================================================================
    print("\n" + "=" * 90)
    print("PART 3: Full IS Search -> OOS Validation (find robust strategies)")
    print("=" * 90)

    # Run full search on IS data
    print("Running full parameter search on IS data...")
    is_results = []
    count = 0
    total_combos = len(CUM_DAYS_LIST) * len(THRESHOLD_PCT_LIST) * len(HOLD_DAYS_LIST) * len(STOP_LOSS_LIST) * 2 * len(is_pc)

    for direction_tag, direction in [('dip', 'dip'), ('rally', 'rally')]:
        for cum_days in CUM_DAYS_LIST:
            for threshold in THRESHOLD_PCT_LIST:
                for hold_days in HOLD_DAYS_LIST:
                    for stop_loss in STOP_LOSS_LIST:
                        for idx_name in is_pc:
                            count += 1
                            if count % 20000 == 0:
                                print(f"  Progress: {count}/{total_combos} ({count*100//total_combos}%)...")

                            m = run_single(is_pc[idx_name], cum_days, threshold, hold_days, direction, stop_loss)
                            if m and m['n_trades'] >= MIN_TRADES:
                                is_results.append({
                                    'index': idx_name,
                                    'direction': direction,
                                    'cum_days': cum_days,
                                    'threshold_pct': threshold,
                                    'hold_days': hold_days,
                                    'stop_loss': stop_loss if stop_loss is not None else 0,
                                    'has_stop_loss': stop_loss is not None,
                                    **m
                                })

    is_df = pd.DataFrame(is_results)
    print(f"\nIS search: {len(is_df)} valid strategies")

    if len(is_df) == 0:
        return

    # Filter IS top candidates (Sharpe > 0.5, >= 5 trades)
    top_is = is_df[(is_df['sharpe'] > 0.5) & (is_df['n_trades'] >= 5)].copy()
    print(f"IS candidates with Sharpe > 0.5 and >= 5 trades: {len(top_is)}")

    # Validate top IS candidates on OOS
    print("\nValidating top IS candidates on OOS data...")
    oos_validated = []

    for _, row in top_is.iterrows():
        idx = row['index']
        if idx not in oos_pc:
            continue

        sl = row['stop_loss'] if row['has_stop_loss'] else None
        oos_m = run_single(oos_pc[idx], int(row['cum_days']), int(row['threshold_pct']),
                          int(row['hold_days']), row['direction'], sl)

        if oos_m and oos_m['n_trades'] >= 1:
            oos_validated.append({
                'index': idx,
                'direction': row['direction'],
                'cum_days': int(row['cum_days']),
                'threshold_pct': int(row['threshold_pct']),
                'hold_days': int(row['hold_days']),
                'stop_loss': sl,
                'is_sharpe': row['sharpe'],
                'is_cagr': row['annualized_return'] * 100,
                'is_mdd': row['max_drawdown'] * 100,
                'is_trades': row['n_trades'],
                'is_win_rate': row['win_rate'],
                'oos_sharpe': oos_m['sharpe'],
                'oos_cagr': oos_m['annualized_return'] * 100,
                'oos_mdd': oos_m['max_drawdown'] * 100,
                'oos_trades': oos_m['n_trades'],
                'oos_win_rate': oos_m['win_rate'],
            })

    val_df = pd.DataFrame(oos_validated)
    print(f"OOS validated: {len(val_df)} strategies")

    # Filter: OOS Sharpe > 0 (strategy still works out of sample)
    passed = val_df[val_df['oos_sharpe'] > 0].copy()
    passed['decay'] = (passed['oos_sharpe'] - passed['is_sharpe']) / passed['is_sharpe'].abs() * 100
    print(f"OOS Sharpe > 0 (passed): {len(passed)} strategies ({len(passed)*100//max(1,len(val_df))}%)")

    # Show best OOS-validated strategies per index
    print(f"\n{'='*90}")
    print("Best OOS-Validated Strategies per Index (sorted by OOS Sharpe)")
    print(f"{'='*90}")

    for idx_name in sorted(is_pc.keys()):
        idx_passed = passed[passed['index'] == idx_name]
        if len(idx_passed) == 0:
            print(f"\n  {idx_name}: No strategies survived OOS validation")
            continue

        print(f"\n  === {idx_name} ({len(idx_passed)} survived) ===")
        top5 = idx_passed.nlargest(5, 'oos_sharpe')
        for _, r in top5.iterrows():
            dir_str = 'drop' if r['direction'] == 'dip' else 'rise'
            sl_str = f" SL{r['stop_loss']}%" if r['stop_loss'] else ""
            print(f"    {r['cum_days']}d {dir_str}>{r['threshold_pct']}% hold{r['hold_days']}d{sl_str}")
            print(f"      IS: Sharpe={r['is_sharpe']:.2f} CAGR={r['is_cagr']:.1f}% MDD={r['is_mdd']:.1f}% Trades={r['is_trades']:.0f} WR={r['is_win_rate']*100:.0f}%")
            print(f"      OOS: Sharpe={r['oos_sharpe']:.2f} CAGR={r['oos_cagr']:.1f}% MDD={r['oos_mdd']:.1f}% Trades={r['oos_trades']:.0f} WR={r['oos_win_rate']*100:.0f}%")
            print(f"      Decay: {r['decay']:.0f}%")

    # ================================================================
    # PART 4: Recommend new robust variants
    # ================================================================
    print(f"\n{'='*90}")
    print("PART 4: Recommended Robust Variants (IS Sharpe>0.5, OOS Sharpe>0, Decay<70%)")
    print(f"{'='*90}")

    robust = passed[(passed['decay'] > -70) & (passed['is_trades'] >= 5)].copy()
    # Score: weighted combo of IS and OOS sharpe
    robust['score'] = robust['oos_sharpe'] * 0.6 + robust['is_sharpe'] * 0.4

    # Pick diverse set: different indices, different directions
    selected = []
    used_combos = set()

    for _, r in robust.nlargest(200, 'score').iterrows():
        key = (r['index'], r['direction'])
        if key in used_combos:
            continue
        used_combos.add(key)
        selected.append(r)
        if len(selected) >= 8:
            break

    # If not enough diversity, fill from top scores
    if len(selected) < 8:
        for _, r in robust.nlargest(50, 'score').iterrows():
            already = any(s['cum_days'] == r['cum_days'] and s['threshold_pct'] == r['threshold_pct']
                        and s['hold_days'] == r['hold_days'] and s['index'] == r['index'] for s in selected)
            if not already:
                selected.append(r)
                if len(selected) >= 8:
                    break

    print(f"\nRecommended {len(selected)} robust variants:")
    for i, r in enumerate(selected):
        hid = f"H{i+1}'"
        dir_str = '超跌' if r['direction'] == 'dip' else '追涨'
        sl_str = f" 止损{r['stop_loss']}%" if r['stop_loss'] else ""
        print(f"\n  {hid}: {r['index']} {dir_str} {r['cum_days']}日{'跌' if r['direction']=='dip' else '涨'}>{r['threshold_pct']}% 持{r['hold_days']}日{sl_str}")
        print(f"    IS:  Sharpe={r['is_sharpe']:.2f}  CAGR={r['is_cagr']:.1f}%  MDD={r['is_mdd']:.1f}%  Trades={r['is_trades']:.0f}  WR={r['is_win_rate']*100:.0f}%")
        print(f"    OOS: Sharpe={r['oos_sharpe']:.2f}  CAGR={r['oos_cagr']:.1f}%  MDD={r['oos_mdd']:.1f}%  Trades={r['oos_trades']:.0f}  WR={r['oos_win_rate']*100:.0f}%")
        print(f"    Decay: {r['decay']:.0f}%  Score: {r['score']:.2f}")

    # ================================================================
    # PART 5: Cross-index universal strategies with OOS
    # ================================================================
    print(f"\n{'='*90}")
    print("PART 5: Cross-Index Universal Strategies (IS+OOS)")
    print(f"{'='*90}")

    # Group by params, count indices where both IS and OOS Sharpe > 0
    if len(passed) > 0:
        grp = passed.groupby(['direction', 'cum_days', 'threshold_pct', 'hold_days', 'stop_loss']).agg({
            'is_sharpe': 'mean',
            'oos_sharpe': 'mean',
            'is_cagr': 'mean',
            'oos_cagr': 'mean',
            'index': 'count',
            'decay': 'mean',
        }).rename(columns={'index': 'n_indices'}).reset_index()

        universal = grp[grp['n_indices'] >= 3].nlargest(15, 'oos_sharpe')

        print(f"\nUniversal strategies (>=3 indices, OOS validated):")
        for _, r in universal.iterrows():
            dir_str = '超跌' if r['direction'] == 'dip' else '追涨'
            sl_str = f" SL{r['stop_loss']}%" if r['stop_loss'] else ""
            print(f"  {dir_str} {int(r['cum_days'])}d>{'drop' if r['direction']=='dip' else 'rise'}{int(r['threshold_pct'])}% hold{int(r['hold_days'])}d{sl_str}")
            print(f"    {int(r['n_indices'])} indices | IS avg Sharpe={r['is_sharpe']:.2f} | OOS avg Sharpe={r['oos_sharpe']:.2f} | Decay={r['decay']:.0f}%")

    # Save full validated results
    if len(val_df) > 0:
        val_df.to_csv('/tmp/strategy_h_oos_results.csv', index=False, encoding='utf-8-sig')
        print(f"\nFull OOS results saved to /tmp/strategy_h_oos_results.csv ({len(val_df)} records)")

    # Save summary
    summary = {
        'original_h_results': {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in h_results.items()},
        'is_total': len(is_df),
        'is_passed_sharpe_05': len(top_is),
        'oos_validated': len(val_df),
        'oos_passed': len(passed),
        'pass_rate': f"{len(passed)*100//max(1,len(val_df))}%",
        'recommended_variants': [{
            'index': r['index'], 'direction': r['direction'],
            'cum_days': int(r['cum_days']), 'threshold_pct': int(r['threshold_pct']),
            'hold_days': int(r['hold_days']), 'stop_loss': r['stop_loss'],
            'is_sharpe': round(r['is_sharpe'], 2), 'oos_sharpe': round(r['oos_sharpe'], 2),
            'decay': round(r['decay'], 0), 'score': round(r['score'], 2),
        } for r in selected],
    }
    with open('/tmp/strategy_h_oos_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"Summary saved to /tmp/strategy_h_oos_summary.json")

    print(f"\n{'='*90}")
    print("DONE")
    print(f"{'='*90}")

if __name__ == '__main__':
    main()
