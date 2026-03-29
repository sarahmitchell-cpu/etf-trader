#!/usr/bin/env python3
"""
Strategy P: Holding Period Research
====================================
Test different exit strategies:
1. Original: exit when price > MA (variable holding period)
2. Fixed holding periods: 5, 10, 15, 20, 30, 40, 60, 90, 120 days
3. Combined: min(fixed_days, price > MA)
4. Trailing stop exits

Entry signal: 2of3 factors (ERP>0.7, Vol>P80+below MA120, VolRatio>1.5)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import baostock as bs
from datetime import datetime

def fetch_index_data(bs_code, start_date='2010-01-01'):
    rs = bs.query_history_k_data_plus(
        bs_code, "date,open,high,low,close,volume,amount",
        start_date=start_date,
        end_date=datetime.now().strftime('%Y-%m-%d'),
        frequency="d", adjustflag="3"
    )
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())
    if not data:
        return None
    df = pd.DataFrame(data, columns=rs.fields)
    df['date'] = pd.to_datetime(df['date'])
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.set_index('date').sort_index()
    return df[df['close'] > 0]


def rolling_percentile_fast(series, window, min_periods=252):
    arr = series.values
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        chunk = arr[start:i+1]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) >= min_periods:
            result[i] = np.mean(valid[-1] >= valid)
    return pd.Series(result, index=series.index)


def compute_entry_signals(df, erp_t=0.7, vol_t=80, ma_p=120, vol_r=1.5):
    """Compute 2of3 entry signals"""
    close = df['close']
    volume = df['volume']
    ret = close.pct_change()

    # Factor 1: ERP/Value
    long_ma = close.rolling(756, min_periods=504).mean()
    value_score = (long_ma - close) / long_ma
    value_pct = rolling_percentile_fast(value_score, 504, 252)
    f1 = value_pct > erp_t

    # Factor 2: Vol panic + below MA
    vol_20 = ret.rolling(20).std() * np.sqrt(252) * 100
    vol_pct = rolling_percentile_fast(vol_20, 504, 252)
    ma = close.rolling(ma_p, min_periods=ma_p).mean()
    f2 = (vol_pct > (vol_t / 100.0)) & (close < ma)

    # Factor 3: Volume anomaly
    vol_5d = volume.rolling(5).mean()
    vol_20d = volume.rolling(20).mean()
    vol_ratio = vol_5d / vol_20d
    f3 = vol_ratio > vol_r

    # 2of3 mode
    score = f1.astype(int) + f2.astype(int) + f3.astype(int)
    entry_mask = score >= 2

    return entry_mask, ma, close


def backtest_exit_strategy(close, entry_mask, exit_func, label=""):
    """
    Generic backtester with custom exit function.
    exit_func(entry_idx, current_idx, close_series, ma_series, extra) -> bool
    """
    close_arr = close.values
    idx = close.index
    trades = []
    in_trade = False
    entry_i = 0
    entry_price = 0

    for i in range(1, len(close_arr)):
        if np.isnan(close_arr[i]):
            continue

        entry_signal = bool(entry_mask.values[i]) if not np.isnan(entry_mask.values[i]) else False

        if not in_trade and entry_signal:
            in_trade = True
            entry_i = i
            entry_price = close_arr[i]
        elif in_trade:
            should_exit = exit_func(entry_i, i, close_arr)
            if should_exit:
                in_trade = False
                exit_price = close_arr[i]
                pnl = (exit_price / entry_price - 1) * 100
                days = (idx[i] - idx[entry_i]).days
                trades.append({
                    'entry': str(idx[entry_i].date()),
                    'exit': str(idx[i].date()),
                    'days': days,
                    'pnl_pct': round(pnl, 2)
                })

    # Close open trade
    if in_trade:
        exit_price = close_arr[-1]
        pnl = (exit_price / entry_price - 1) * 100
        days = (idx[-1] - idx[entry_i]).days
        trades.append({
            'entry': str(idx[entry_i].date()),
            'exit': str(idx[-1].date()) + '*',
            'days': days,
            'pnl_pct': round(pnl, 2),
            'open': True
        })

    return trades


def analyze_trades(trades, label=""):
    if not trades:
        return None
    closed = [t for t in trades if not t.get('open')]
    if not closed:
        return None
    pnls = [t['pnl_pct'] for t in closed]
    days_list = [t['days'] for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls) * 100
    avg_pnl = np.mean(pnls)
    avg_days = np.mean(days_list)
    med_days = np.median(days_list)
    total_pnl = sum(pnls)
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999

    return {
        'label': label,
        'n_trades': len(closed),
        'win_rate': round(wr, 1),
        'avg_pnl': round(avg_pnl, 1),
        'total_pnl': round(total_pnl, 1),
        'avg_days': round(avg_days, 0),
        'med_days': round(med_days, 0),
        'pf': round(pf, 1),
        'max_win': round(max(pnls), 1),
        'max_loss': round(min(pnls), 1),
        'trades': closed
    }


def run_research():
    print("=" * 70)
    print("  Strategy P: Holding Period Research")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    bs.login()

    indices = {
        '沪深300': 'sh.000300',
        '中证500': 'sh.000905',
        '中证1000': 'sh.000852',
        '创业板指': 'sz.399006',
    }

    # Fixed holding periods to test
    fixed_periods = [5, 10, 15, 20, 30, 40, 60, 90, 120]

    all_results = {}

    for name, code in indices.items():
        print(f"\n{'=' * 70}")
        print(f"  {name} ({code})")
        print(f"{'=' * 70}")

        df = fetch_index_data(code, '2010-01-01')
        if df is None or len(df) < 1000:
            print(f"  Data error")
            continue

        entry_mask, ma, close = compute_entry_signals(df)
        close_arr = close.values
        ma_arr = ma.values

        results = []

        # === Strategy 1: Original MA exit ===
        def exit_above_ma(entry_i, i, c):
            if np.isnan(ma_arr[i]):
                return False
            return c[i] > ma_arr[i]

        trades = backtest_exit_strategy(close, entry_mask, exit_above_ma)
        r = analyze_trades(trades, "MA Exit (price>MA120)")
        if r:
            results.append(r)
            print(f"\n  [MA Exit] Trades={r['n_trades']} WR={r['win_rate']}% AvgPnL={r['avg_pnl']:+.1f}% AvgDays={r['avg_days']:.0f} MedDays={r['med_days']:.0f}")

        # === Strategy 2: Fixed holding periods ===
        for hold_days in fixed_periods:
            def make_exit(hd):
                def exit_fixed(entry_i, i, c):
                    # Count trading days, not calendar days
                    return (i - entry_i) >= hd
                return exit_fixed

            trades = backtest_exit_strategy(close, entry_mask, make_exit(hold_days))
            r = analyze_trades(trades, f"Fixed {hold_days}d")
            if r:
                results.append(r)

        # === Strategy 3: MA exit with max holding cap ===
        for cap in [30, 60, 90, 120]:
            def make_capped(cap_d):
                def exit_capped(entry_i, i, c):
                    if (i - entry_i) >= cap_d:
                        return True
                    if np.isnan(ma_arr[i]):
                        return False
                    return c[i] > ma_arr[i]
                return exit_capped

            trades = backtest_exit_strategy(close, entry_mask, make_capped(cap))
            r = analyze_trades(trades, f"MA+Cap{cap}d")
            if r:
                results.append(r)

        # === Strategy 4: Trailing stop ===
        for stop_pct in [5, 8, 10, 15]:
            def make_trail(sp):
                def exit_trail(entry_i, i, c):
                    peak = np.max(c[entry_i:i+1])
                    drawdown = (peak - c[i]) / peak * 100
                    return drawdown >= sp
                return exit_trail

            trades = backtest_exit_strategy(close, entry_mask, make_trail(stop_pct))
            r = analyze_trades(trades, f"Trail-{stop_pct}%")
            if r:
                results.append(r)

        # === Strategy 5: Profit target ===
        for target in [5, 8, 10, 15, 20]:
            def make_target(tp):
                def exit_target(entry_i, i, c):
                    gain = (c[i] / c[entry_i] - 1) * 100
                    return gain >= tp
                return exit_target

            trades = backtest_exit_strategy(close, entry_mask, make_target(target))
            r = analyze_trades(trades, f"Target+{target}%")
            if r:
                results.append(r)

        # === Strategy 6: Combined profit target + stop loss ===
        for tp, sl in [(10, -5), (15, -5), (10, -8), (15, -8), (20, -10)]:
            def make_combo(tp_v, sl_v):
                def exit_combo(entry_i, i, c):
                    gain = (c[i] / c[entry_i] - 1) * 100
                    return gain >= tp_v or gain <= sl_v
                return exit_combo

            trades = backtest_exit_strategy(close, entry_mask, make_combo(tp, sl))
            r = analyze_trades(trades, f"TP{tp}/SL{sl}%")
            if r:
                results.append(r)

        # === Display all results ===
        print(f"\n  {'Exit Strategy':<20s} | {'#Tr':>4s} {'WR%':>5s} {'AvgP':>6s} {'TotP':>7s} {'AvgD':>5s} {'MedD':>5s} {'PF':>5s} {'MaxW':>6s} {'MaxL':>6s}")
        print(f"  {'-' * 85}")
        for r in results:
            print(f"  {r['label']:<20s} | {r['n_trades']:4d} {r['win_rate']:5.1f} {r['avg_pnl']:+5.1f}% {r['total_pnl']:+6.1f}% {r['avg_days']:5.0f} {r['med_days']:5.0f} {r['pf']:5.1f} {r['max_win']:+5.1f}% {r['max_loss']:+5.1f}%")

        # Find best by different criteria
        if results:
            best_wr = max(results, key=lambda x: (x['win_rate'], x['avg_pnl']))
            best_pnl = max(results, key=lambda x: x['total_pnl'])
            best_sharpe_like = max(results, key=lambda x: x['avg_pnl'] / max(abs(x['max_loss']), 1) * x['win_rate'] / 100)

            print(f"\n  Best WR:     {best_wr['label']} (WR={best_wr['win_rate']}% AvgPnL={best_wr['avg_pnl']:+.1f}%)")
            print(f"  Best Total:  {best_pnl['label']} (Total={best_pnl['total_pnl']:+.1f}% WR={best_pnl['win_rate']}%)")
            print(f"  Best Risk-Adj: {best_sharpe_like['label']} (WR={best_sharpe_like['win_rate']}% PF={best_sharpe_like['pf']})")

        all_results[name] = results

        # === Show trade-level detail for top strategies ===
        print(f"\n  --- Trade Details for MA Exit ---")
        ma_result = results[0] if results else None
        if ma_result:
            for t in ma_result['trades']:
                marker = '+' if t['pnl_pct'] > 0 else ''
                print(f"    {t['entry']} -> {t['exit']}  {t['days']:3d}d  {marker}{t['pnl_pct']:.1f}%")

    bs.logout()

    # === CROSS-INDEX SUMMARY ===
    print(f"\n\n{'=' * 70}")
    print("  CROSS-INDEX HOLDING PERIOD SUMMARY")
    print(f"{'=' * 70}")

    # Aggregate fixed period results across indices
    for hold_days in fixed_periods:
        label = f"Fixed {hold_days}d"
        wrs = []
        pnls = []
        for name, results in all_results.items():
            for r in results:
                if r['label'] == label:
                    wrs.append(r['win_rate'])
                    pnls.append(r['avg_pnl'])
        if wrs:
            print(f"  {label:<15s}: AvgWR={np.mean(wrs):5.1f}%  AvgPnL={np.mean(pnls):+5.1f}%  (across {len(wrs)} indices)")

    print()
    # MA exit comparison
    label = "MA Exit (price>MA120)"
    wrs = []
    pnls = []
    for name, results in all_results.items():
        for r in results:
            if r['label'] == label:
                wrs.append(r['win_rate'])
                pnls.append(r['avg_pnl'])
    if wrs:
        print(f"  {'MA Exit':<15s}: AvgWR={np.mean(wrs):5.1f}%  AvgPnL={np.mean(pnls):+5.1f}%  (across {len(wrs)} indices)")

    print()
    # Best strategies per criterion across all indices
    print("  Recommended: See per-index tables above for best exit strategy.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    run_research()
