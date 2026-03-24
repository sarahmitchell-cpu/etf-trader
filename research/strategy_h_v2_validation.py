#!/usr/bin/env python3
"""
Strategy H V2 Validation — Full Audit Fix
==========================================
Addresses ALL issues from V2 audit report:
  1. Data from local CSV (reproducible, no yfinance drift)
  2. Next-day open price for entry (no look-ahead bias)
  3. 10bps round-trip transaction costs
  4. IS/OOS split: IS 2015-2022 (param selection), OOS 2023-2026 (blind validation)
  5. OOS data NEVER used for param selection
  6. Minimum 5 OOS trades to count as "pass"
  7. Neighborhood robustness check on IS data only

Usage:
  python3 research/strategy_h_v2_validation.py
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'strategy_h')

INDICES = {
    '沪深300':   'csi300.csv',
    '上证50':    'sse50.csv',
    '科创50':    'star50.csv',
    '恒生指数':  'hsi.csv',
    '国企指数':  'hscei.csv',
}

IS_END = '2022-12-31'
OOS_START = '2023-01-01'

RISK_FREE_RATE = 0.02
TRANSACTION_COST_BPS = 10  # 10bps round-trip (5bps each way)
COST_RATE = TRANSACTION_COST_BPS / 10000  # 0.001

# Parameter search space
CUM_DAYS_LIST = list(range(1, 11))
THRESHOLD_PCT_LIST = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
HOLD_DAYS_LIST = list(range(1, 51))
STOP_LOSS_LIST = [None, -3, -5, -7]

MIN_IS_TRADES = 5   # minimum trades in IS to consider
MIN_OOS_TRADES = 5  # minimum trades in OOS to count as validated (audit requirement)

# ============================================================
# Data Loading
# ============================================================
def load_data():
    """Load all index data from local CSV files."""
    data = {}
    for name, filename in INDICES.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {name}")
            continue
        df = pd.read_csv(path, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        # Ensure numeric
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['close', 'open'])
        data[name] = df
        print(f"  {name}: {len(df)} days, {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    return data


def split_data(data):
    """Split into IS and OOS periods."""
    is_data = {}
    oos_data = {}
    for name, df in data.items():
        is_df = df[df['date'] <= IS_END].copy().reset_index(drop=True)
        oos_df = df[df['date'] >= OOS_START].copy().reset_index(drop=True)
        if len(is_df) > 50:
            is_data[name] = is_df
        if len(oos_df) > 50:
            oos_data[name] = oos_df
        print(f"  {name}: IS={len(is_df)} days, OOS={len(oos_df)} days")
    return is_data, oos_data


# ============================================================
# Backtest Engine V2 (Fixed: next-day open + costs)
# ============================================================
def backtest_v2(closes, opens, cum_days, threshold_pct, hold_days, direction='dip',
                stop_loss_pct=None):
    """
    Fixed backtest engine:
    - Signal detected at day i close
    - Entry at day i+1 open (next-day open, no look-ahead)
    - Exit at day i+hold_days close (or stop-loss trigger)
    - Transaction costs: COST_RATE on entry and exit
    """
    n = len(closes)
    if n < cum_days + hold_days + 10:
        return None, []

    # Pre-compute cumulative returns for signal detection (uses close prices)
    cum_ret = np.full(n, np.nan)
    for d in range(cum_days, n):
        cum_ret[d] = (closes[d] / closes[d - cum_days] - 1) * 100

    # Generate signals (at close of day i)
    if direction == 'dip':
        signals = np.array([not np.isnan(cr) and cr <= -threshold_pct for cr in cum_ret])
    else:
        signals = np.array([not np.isnan(cr) and cr >= threshold_pct for cr in cum_ret])

    # NAV tracking
    nav = np.ones(n)
    position = False
    entry_price = 0.0
    entry_idx = 0
    hold_count = 0
    trades = []

    for i in range(1, n):
        if position:
            # Track daily P&L using close-to-close
            daily_ret = closes[i] / closes[i - 1] - 1
            nav[i] = nav[i - 1] * (1 + daily_ret)
            hold_count += 1

            # Check stop-loss and exit conditions
            current_trade_ret = closes[i] / entry_price - 1
            hit_stop = (stop_loss_pct is not None and current_trade_ret * 100 <= stop_loss_pct)

            if hold_count >= hold_days or hit_stop:
                exit_price = closes[i]
                gross_ret = exit_price / entry_price - 1
                net_ret = gross_ret - COST_RATE  # deduct exit cost
                # Adjust NAV for transaction cost at exit
                nav[i] = nav[i] * (1 - COST_RATE)

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'gross_return': gross_ret,
                    'net_return': net_ret,
                    'hold_days': hold_count,
                    'stop_loss': hit_stop,
                })
                position = False
        else:
            nav[i] = nav[i - 1]
            # Signal at day i-1 close -> enter at day i open
            if i > 0 and signals[i - 1] and not np.isnan(opens[i]) and opens[i] > 0:
                position = True
                entry_price = opens[i]  # FIXED: next-day open
                entry_idx = i
                hold_count = 0
                # Apply entry transaction cost
                nav[i] = nav[i] * (1 - COST_RATE)
                # Adjust for the gap: entry at open, but nav tracks close-to-close
                # So on entry day, the return is from open to close
                if closes[i] > 0:
                    entry_day_ret = closes[i] / opens[i] - 1
                    nav[i] = nav[i] * (1 + entry_day_ret)

    # Force close open position
    if position:
        exit_price = closes[-1]
        gross_ret = exit_price / entry_price - 1
        net_ret = gross_ret - COST_RATE
        nav[-1] = nav[-1] * (1 - COST_RATE)
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': n - 1,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'gross_return': gross_ret,
            'net_return': net_ret,
            'hold_days': hold_count,
            'stop_loss': False,
        })

    return nav, trades


def calc_metrics(nav, trades, n_days):
    """Calculate performance metrics from NAV and trades."""
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
        rets = [t['net_return'] for t in trades]
        win_rate = sum(1 for r in rets if r > 0) / n_trades
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        plr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    else:
        win_rate = plr = 0

    return {
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'calmar': calmar,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_loss_ratio': plr,
    }


# ============================================================
# Precompute & Run
# ============================================================
def precompute(data_dict):
    """Pre-compute cumulative returns for all indices."""
    result = {}
    for name, df in data_dict.items():
        closes = df['close'].values
        opens = df['open'].values
        n = len(closes)
        cum_rets = {}
        for cd in CUM_DAYS_LIST:
            cr = np.full(n, np.nan)
            for i in range(cd, n):
                if closes[i - cd] > 0:
                    cr[i] = (closes[i] / closes[i - cd] - 1) * 100
            cum_rets[cd] = cr
        result[name] = {'closes': closes, 'opens': opens, 'n': n, 'cum_returns': cum_rets}
    return result


def run_single(pc, cum_days, threshold_pct, hold_days, direction, stop_loss):
    """Run a single backtest on precomputed data."""
    nav, trades = backtest_v2(
        pc['closes'], pc['opens'],
        cum_days, threshold_pct, hold_days, direction, stop_loss
    )
    return calc_metrics(nav, trades, pc['n']), trades


def check_neighborhood(pc, cum_days, threshold_pct, hold_days, direction, stop_loss, min_sharpe=0.0):
    """Check if nearby parameter combinations also perform well (IS data only)."""
    cd_range = [cd for cd in [cum_days-1, cum_days, cum_days+1] if 1 <= cd <= 10]
    # Find nearest thresholds in search space
    th_candidates = [t for t in THRESHOLD_PCT_LIST if abs(t - threshold_pct) <= 2]
    if not th_candidates:
        th_candidates = [threshold_pct]
    hd_range = [hd for hd in range(max(1, hold_days-3), min(51, hold_days+4))]

    total = 0
    good = 0
    sharpes = []

    for cd in cd_range:
        for th in th_candidates:
            for hd in hd_range:
                m, _ = run_single(pc, cd, th, hd, direction, stop_loss)
                if m and m['n_trades'] >= MIN_IS_TRADES:
                    total += 1
                    sharpes.append(m['sharpe'])
                    if m['sharpe'] > min_sharpe:
                        good += 1

    if total == 0:
        return 0, 0, 0
    return good / total, np.mean(sharpes), total


# ============================================================
# Main Pipeline
# ============================================================
def main():
    print("=" * 90)
    print("Strategy H V2 Validation — Full Audit Fix")
    print(f"  Data source: Local CSV ({DATA_DIR})")
    print(f"  Engine fixes: Next-day open entry + {TRANSACTION_COST_BPS}bps round-trip costs")
    print(f"  IS period: 2015-01-01 ~ {IS_END}")
    print(f"  OOS period: {OOS_START} ~ 2026-03-24")
    print(f"  Min IS trades: {MIN_IS_TRADES}, Min OOS trades: {MIN_OOS_TRADES}")
    print("=" * 90)

    # Step 1: Load data
    print("\n[Step 1] Loading data from local CSV...")
    all_data = load_data()
    print(f"  Loaded {len(all_data)} indices")

    is_data, oos_data = split_data(all_data)
    is_pc = precompute(is_data)
    oos_pc = precompute(oos_data)
    full_pc = precompute(all_data)

    # Step 2: Full IS parameter search
    print(f"\n[Step 2] Full IS parameter search...")
    is_results = []
    count = 0
    total_combos = len(CUM_DAYS_LIST) * len(THRESHOLD_PCT_LIST) * len(HOLD_DAYS_LIST) * len(STOP_LOSS_LIST) * 2 * len(is_pc)
    print(f"  Total parameter combinations: {total_combos:,}")

    for direction in ['dip', 'rally']:
        for cum_days in CUM_DAYS_LIST:
            for threshold in THRESHOLD_PCT_LIST:
                for hold_days in HOLD_DAYS_LIST:
                    for stop_loss in STOP_LOSS_LIST:
                        for idx_name in is_pc:
                            count += 1
                            if count % 50000 == 0:
                                print(f"  Progress: {count:,}/{total_combos:,} ({count*100//total_combos}%)...")

                            m, trades = run_single(is_pc[idx_name], cum_days, threshold, hold_days, direction, stop_loss)
                            if m and m['n_trades'] >= MIN_IS_TRADES:
                                is_results.append({
                                    'index': idx_name,
                                    'direction': direction,
                                    'cum_days': cum_days,
                                    'threshold_pct': threshold,
                                    'hold_days': hold_days,
                                    'stop_loss': stop_loss,
                                    'sharpe': m['sharpe'],
                                    'cagr': m['annualized_return'] * 100,
                                    'mdd': m['max_drawdown'] * 100,
                                    'n_trades': m['n_trades'],
                                    'win_rate': m['win_rate'],
                                    'calmar': m['calmar'],
                                })

    is_df = pd.DataFrame(is_results)
    print(f"\n  IS search complete: {len(is_df):,} valid strategies (>= {MIN_IS_TRADES} trades)")

    if len(is_df) == 0:
        print("  No valid strategies found!")
        return

    # Step 3: Filter IS candidates
    print(f"\n[Step 3] Filtering IS candidates...")
    top_is = is_df[(is_df['sharpe'] > 0.5)].copy()
    print(f"  IS Sharpe > 0.5: {len(top_is):,} strategies")

    # Distribution by index
    print(f"\n  IS candidates by index:")
    for idx in sorted(is_pc.keys()):
        idx_count = len(top_is[top_is['index'] == idx])
        print(f"    {idx}: {idx_count}")

    # Step 4: Neighborhood robustness check (IS data only)
    print(f"\n[Step 4] Neighborhood robustness check on IS candidates...")
    # For efficiency, only check unique (index, direction, cum_days, threshold, hold_days, stop_loss) combos
    # Group by params, take top per-index
    robust_candidates = []
    seen = set()

    for _, row in top_is.nlargest(2000, 'sharpe').iterrows():
        key = (row['index'], row['direction'], row['cum_days'], row['threshold_pct'],
               row['hold_days'], row['stop_loss'] if row['stop_loss'] is not None else 0)
        if key in seen:
            continue
        seen.add(key)

        ratio, avg_sharpe, total = check_neighborhood(
            is_pc[row['index']], int(row['cum_days']), int(row['threshold_pct']),
            int(row['hold_days']), row['direction'],
            row['stop_loss'] if pd.notna(row['stop_loss']) else None,
            min_sharpe=0.0
        )

        if ratio >= 0.6 and total >= 5:  # At least 60% of neighbors also positive
            robust_candidates.append({
                **row.to_dict(),
                'neighbor_pos_rate': ratio,
                'neighbor_avg_sharpe': avg_sharpe,
                'neighbor_count': total,
            })

    robust_df = pd.DataFrame(robust_candidates)
    print(f"  Robust candidates (>=60% neighbors positive, >=5 neighbors): {len(robust_df)}")

    if len(robust_df) == 0:
        print("  No robust candidates found! Lowering threshold...")
        # Fallback: lower neighborhood requirement
        for _, row in top_is.nlargest(500, 'sharpe').iterrows():
            key = (row['index'], row['direction'], row['cum_days'], row['threshold_pct'],
                   row['hold_days'], row['stop_loss'] if row['stop_loss'] is not None else 0)
            if key in seen:
                continue
            seen.add(key)

            ratio, avg_sharpe, total = check_neighborhood(
                is_pc[row['index']], int(row['cum_days']), int(row['threshold_pct']),
                int(row['hold_days']), row['direction'],
                row['stop_loss'] if pd.notna(row['stop_loss']) else None,
                min_sharpe=0.0
            )
            if ratio >= 0.4:
                robust_candidates.append({
                    **row.to_dict(),
                    'neighbor_pos_rate': ratio,
                    'neighbor_avg_sharpe': avg_sharpe,
                    'neighbor_count': total,
                })
        robust_df = pd.DataFrame(robust_candidates)
        print(f"  Robust candidates (relaxed >=40%): {len(robust_df)}")

    # Step 5: Select 8 variants from IS + neighborhood ONLY (no OOS peeking!)
    print(f"\n[Step 5] Selecting 8 variants from IS + neighborhood (NO OOS DATA USED)...")
    # Score = IS Sharpe * 0.5 + neighbor_avg_sharpe * 0.3 + neighbor_pos_rate * IS_sharpe * 0.2
    if len(robust_df) > 0:
        robust_df['score'] = (
            robust_df['sharpe'] * 0.5 +
            robust_df['neighbor_avg_sharpe'] * 0.3 +
            robust_df['neighbor_pos_rate'] * robust_df['sharpe'] * 0.2
        )
    else:
        # Fallback to raw IS results
        robust_df = top_is.nlargest(100, 'sharpe').copy()
        robust_df['score'] = robust_df['sharpe']
        robust_df['neighbor_pos_rate'] = 0
        robust_df['neighbor_avg_sharpe'] = 0
        robust_df['neighbor_count'] = 0

    # Pick diverse set: different indices and directions
    selected = []
    used_combos = set()

    for _, r in robust_df.nlargest(500, 'score').iterrows():
        key = (r['index'], r['direction'])
        if key in used_combos:
            continue
        used_combos.add(key)
        selected.append(r.to_dict())
        if len(selected) >= 8:
            break

    # Fill remaining slots with best scores (different params)
    if len(selected) < 8:
        for _, r in robust_df.nlargest(100, 'score').iterrows():
            already = any(
                s['cum_days'] == r['cum_days'] and s['threshold_pct'] == r['threshold_pct']
                and s['hold_days'] == r['hold_days'] and s['index'] == r['index']
                and s['direction'] == r['direction']
                for s in selected
            )
            if not already:
                selected.append(r.to_dict())
                if len(selected) >= 8:
                    break

    print(f"\n  Selected {len(selected)} variants (IS + neighborhood only):")
    for i, s in enumerate(selected):
        dir_zh = '超跌' if s['direction'] == 'dip' else '追涨'
        sl_str = f" SL{s['stop_loss']}%" if s['stop_loss'] else ""
        print(f"    H{i+1}: {s['index']} {dir_zh} {s['cum_days']}日>"
              f"{s['threshold_pct']}% 持{s['hold_days']}日{sl_str}")
        print(f"         IS: Sharpe={s['sharpe']:.2f} CAGR={s['cagr']:.1f}% MDD={s['mdd']:.1f}% "
              f"Trades={s['n_trades']} WR={s['win_rate']*100:.0f}%")
        print(f"         Neighbors: {s['neighbor_pos_rate']*100:.0f}% positive "
              f"(avg Sharpe={s['neighbor_avg_sharpe']:.2f}, n={s['neighbor_count']:.0f})")

    # Step 6: BLIND OOS validation (one-time test, no iteration)
    print(f"\n[Step 6] BLIND OOS Validation (one-time test, NO iteration)...")
    print(f"  Minimum OOS trades to pass: {MIN_OOS_TRADES}")

    final_results = []
    for i, s in enumerate(selected):
        hid = f"H{i+1}"
        idx = s['index']

        # Run on OOS data
        if idx in oos_pc:
            sl = s['stop_loss'] if pd.notna(s.get('stop_loss')) and s.get('stop_loss') is not None else None
            oos_m, oos_trades = run_single(
                oos_pc[idx], int(s['cum_days']), int(s['threshold_pct']),
                int(s['hold_days']), s['direction'], sl
            )
        else:
            oos_m = None
            oos_trades = []

        # Run on full period
        if idx in full_pc:
            sl = s['stop_loss'] if pd.notna(s.get('stop_loss')) and s.get('stop_loss') is not None else None
            full_m, full_trades = run_single(
                full_pc[idx], int(s['cum_days']), int(s['threshold_pct']),
                int(s['hold_days']), s['direction'], sl
            )
        else:
            full_m = None
            full_trades = []

        oos_sharpe = oos_m['sharpe'] if oos_m else None
        oos_trades_n = oos_m['n_trades'] if oos_m else 0
        oos_cagr = oos_m['annualized_return'] * 100 if oos_m else None
        oos_mdd = oos_m['max_drawdown'] * 100 if oos_m else None
        oos_wr = oos_m['win_rate'] if oos_m else None

        # Determine pass/fail
        if oos_m is None:
            status = "NO DATA"
        elif oos_trades_n < MIN_OOS_TRADES:
            status = f"INSUFFICIENT ({oos_trades_n} trades < {MIN_OOS_TRADES})"
        elif oos_sharpe > 0:
            decay = (oos_sharpe - s['sharpe']) / abs(s['sharpe']) * 100 if s['sharpe'] != 0 else 0
            status = f"PASS (decay {decay:.0f}%)"
        else:
            status = "FAIL (negative OOS Sharpe)"

        result = {
            'variant': hid,
            'index': idx,
            'direction': s['direction'],
            'cum_days': int(s['cum_days']),
            'threshold_pct': int(s['threshold_pct']),
            'hold_days': int(s['hold_days']),
            'stop_loss': s['stop_loss'],
            'is_sharpe': round(s['sharpe'], 2),
            'is_cagr': round(s['cagr'], 1),
            'is_mdd': round(s['mdd'], 1),
            'is_trades': int(s['n_trades']),
            'is_win_rate': round(s['win_rate'] * 100),
            'neighbor_pos_rate': round(s['neighbor_pos_rate'] * 100),
            'oos_sharpe': round(oos_sharpe, 2) if oos_sharpe is not None else None,
            'oos_cagr': round(oos_cagr, 1) if oos_cagr is not None else None,
            'oos_mdd': round(oos_mdd, 1) if oos_mdd is not None else None,
            'oos_trades': oos_trades_n,
            'oos_win_rate': round(oos_wr * 100) if oos_wr is not None else None,
            'full_sharpe': round(full_m['sharpe'], 2) if full_m else None,
            'full_cagr': round(full_m['annualized_return'] * 100, 1) if full_m else None,
            'full_mdd': round(full_m['max_drawdown'] * 100, 1) if full_m else None,
            'full_trades': full_m['n_trades'] if full_m else None,
            'full_win_rate': round(full_m['win_rate'] * 100) if full_m else None,
            'status': status,
        }
        final_results.append(result)

        dir_zh = '超跌' if s['direction'] == 'dip' else '追涨'
        sl_str = f" SL{s['stop_loss']}%" if s['stop_loss'] else ""
        print(f"\n  {hid}: {idx} {dir_zh} {s['cum_days']}日>{s['threshold_pct']}% "
              f"持{s['hold_days']}日{sl_str}")
        print(f"    IS:   Sharpe={s['sharpe']:.2f}  CAGR={s['cagr']:.1f}%  MDD={s['mdd']:.1f}%  "
              f"Trades={s['n_trades']}  WR={s['win_rate']*100:.0f}%")
        if oos_m:
            print(f"    OOS:  Sharpe={oos_sharpe:.2f}  CAGR={oos_cagr:.1f}%  MDD={oos_mdd:.1f}%  "
                  f"Trades={oos_trades_n}  WR={oos_wr*100:.0f}%")
        else:
            print(f"    OOS:  NO DATA")
        if full_m:
            print(f"    Full: Sharpe={full_m['sharpe']:.2f}  CAGR={full_m['annualized_return']*100:.1f}%  "
                  f"MDD={full_m['max_drawdown']*100:.1f}%  Trades={full_m['n_trades']}  "
                  f"WR={full_m['win_rate']*100:.0f}%")
        print(f"    -> {status}")

    # Step 7: Summary
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")

    passed = [r for r in final_results if 'PASS' in r['status']]
    failed = [r for r in final_results if 'FAIL' in r['status']]
    insufficient = [r for r in final_results if 'INSUFFICIENT' in r['status']]
    no_data = [r for r in final_results if 'NO DATA' in r['status']]

    print(f"  Total variants: {len(final_results)}")
    print(f"  PASS: {len(passed)}")
    print(f"  FAIL: {len(failed)}")
    print(f"  INSUFFICIENT OOS trades: {len(insufficient)}")
    print(f"  NO DATA: {len(no_data)}")

    testable = len(passed) + len(failed)
    if testable > 0:
        print(f"  Pass rate (testable only): {len(passed)}/{testable} = {len(passed)*100//testable}%")

    # Step 8: Also do a broad OOS validation of all IS candidates
    print(f"\n[Step 8] Broad OOS validation statistics (all IS Sharpe>0.5 candidates)...")
    oos_stats = []
    for _, row in top_is.iterrows():
        idx = row['index']
        if idx not in oos_pc:
            continue
        sl = row['stop_loss'] if pd.notna(row['stop_loss']) and row['stop_loss'] is not None else None
        oos_m, oos_trades = run_single(
            oos_pc[idx], int(row['cum_days']), int(row['threshold_pct']),
            int(row['hold_days']), row['direction'], sl
        )
        if oos_m:
            oos_stats.append({
                'index': idx,
                'direction': row['direction'],
                'is_sharpe': row['sharpe'],
                'oos_sharpe': oos_m['sharpe'],
                'oos_trades': oos_m['n_trades'],
                'oos_pass_5trades': oos_m['sharpe'] > 0 and oos_m['n_trades'] >= MIN_OOS_TRADES,
                'oos_pass_any': oos_m['sharpe'] > 0,
            })

    oos_stats_df = pd.DataFrame(oos_stats)
    if len(oos_stats_df) > 0:
        total_tested = len(oos_stats_df)
        pass_any = oos_stats_df['oos_pass_any'].sum()
        pass_5 = oos_stats_df['oos_pass_5trades'].sum()
        print(f"  Total IS candidates with OOS data: {total_tested}")
        print(f"  OOS Sharpe > 0 (any trades): {pass_any} ({pass_any*100//total_tested}%)")
        print(f"  OOS Sharpe > 0 AND >= {MIN_OOS_TRADES} trades: {pass_5} ({pass_5*100//total_tested}%)")

        print(f"\n  By index:")
        for idx in sorted(oos_stats_df['index'].unique()):
            idx_df = oos_stats_df[oos_stats_df['index'] == idx]
            n = len(idx_df)
            p_any = idx_df['oos_pass_any'].sum()
            p_5 = idx_df['oos_pass_5trades'].sum()
            med_trades = idx_df['oos_trades'].median()
            print(f"    {idx}: {n} tested, pass(any)={p_any}({p_any*100//max(1,n)}%), "
                  f"pass(5+trades)={p_5}({p_5*100//max(1,n)}%), median OOS trades={med_trades:.0f}")

    # Save results
    output_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'strategy_h')
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, 'v2_validation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'engine': 'v2 (next-day open + 10bps costs)',
            'is_period': f'2015-01-01 ~ {IS_END}',
            'oos_period': f'{OOS_START} ~ 2026-03-24',
            'min_is_trades': MIN_IS_TRADES,
            'min_oos_trades': MIN_OOS_TRADES,
            'is_total_strategies': len(is_df),
            'is_sharpe_gt_05': len(top_is),
            'robust_candidates': len(robust_df),
            'variants': final_results,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # Save IS results for reference
    is_csv_path = os.path.join(output_dir, 'v2_is_results.csv')
    is_df.to_csv(is_csv_path, index=False, encoding='utf-8-sig')
    print(f"  IS results saved to {is_csv_path}")

    if len(oos_stats_df) > 0:
        oos_csv_path = os.path.join(output_dir, 'v2_oos_stats.csv')
        oos_stats_df.to_csv(oos_csv_path, index=False, encoding='utf-8-sig')
        print(f"  OOS stats saved to {oos_csv_path}")

    print(f"\n{'='*90}")
    print("DONE — Strategy H V2 Validation Complete")
    print(f"{'='*90}")

    return final_results


if __name__ == '__main__':
    main()
