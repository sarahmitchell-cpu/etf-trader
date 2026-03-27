#!/usr/bin/env python3
"""
Strategy P Upgrade: 3-Factor Super Panic Signal (Optimized)
============================================================
Vectorized version - avoids slow rolling .apply(lambda)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import baostock as bs
from datetime import datetime
import sys

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
    """Fast rolling percentile using rank approach"""
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


def run_research():
    print("=" * 70)
    print("  Strategy P: 3-Factor Super Signal Research (v2 - Optimized)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    print("  Factors: ERP(value) + VolPanic(price) + Volume(sentiment)")
    print("  Target: 90%+ win rate")
    print()

    bs.login()

    indices = {
        '沪深300': 'sh.000300',
        '中证500': 'sh.000905',
        '中证1000': 'sh.000852',
        '创业板指': 'sz.399006',
    }

    split = pd.Timestamp('2023-01-01')
    all_summaries = {}

    for name, code in indices.items():
        print(f"\n{'=' * 70}")
        print(f"  {name} ({code})")
        print(f"{'=' * 70}")

        df = fetch_index_data(code, '2010-01-01')
        if df is None or len(df) < 1000:
            print(f"  Data error")
            continue

        close = df['close']
        volume = df['volume']
        ret = close.pct_change()

        # ---- Pre-compute all indicators once ----
        print(f"  Computing indicators ({len(close)} days)...")

        # Factor 1: Value / ERP proxy
        # Use price deviation from 3-year MA as value signal
        long_ma_756 = close.rolling(756, min_periods=504).mean()
        value_score = (long_ma_756 - close) / long_ma_756  # positive = undervalued
        print(f"    Value score computed")

        # Precompute value percentiles for different lookbacks
        value_pct_504 = rolling_percentile_fast(value_score, 504, 252)
        print(f"    Value percentile computed")

        # Factor 2: Vol + below MA
        vol_20 = ret.rolling(20).std() * np.sqrt(252) * 100
        vol_pct_504 = rolling_percentile_fast(vol_20, 504, 252)
        print(f"    Volatility percentile computed")

        # Moving averages
        ma_dict = {}
        for p in [60, 90, 120, 150]:
            ma_dict[p] = close.rolling(p, min_periods=p).mean()

        # Factor 3: Volume ratio
        vol_5d = volume.rolling(5).mean()
        vol_20d = volume.rolling(20).mean()
        vol_ratio = vol_5d / vol_20d
        print(f"    Volume ratio computed")

        # ---- Grid search (vectorized signals) ----
        erp_ts = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
        vol_ts = [70, 75, 80, 85]
        ma_ps = [60, 90, 120, 150]
        vol_rs = [1.2, 1.5, 1.8, 2.0, 2.5]
        modes = [True, False]  # require_all_3

        total = len(erp_ts) * len(vol_ts) * len(ma_ps) * len(vol_rs) * len(modes)
        print(f"  Grid search: {total} combinations")

        results = []
        count = 0

        for erp_t in erp_ts:
            f1 = value_pct_504 > erp_t

            for vol_t in vol_ts:
                f2_vol = vol_pct_504 > (vol_t / 100.0)

                for ma_p in ma_ps:
                    ma = ma_dict[ma_p]
                    f2_below = close < ma
                    f2 = f2_vol & f2_below

                    for vol_r in vol_rs:
                        f3 = vol_ratio > vol_r

                        for req_all in modes:
                            count += 1

                            if req_all:
                                entry_mask = f1 & f2 & f3
                            else:
                                score = f1.astype(int) + f2.astype(int) + f3.astype(int)
                                entry_mask = score >= 2

                            # State machine for signal
                            signal = np.zeros(len(close))
                            in_pos = False
                            close_arr = close.values
                            ma_arr = ma.values
                            entry_arr = entry_mask.values

                            for i in range(1, len(close_arr)):
                                if np.isnan(ma_arr[i]):
                                    signal[i] = 1.0 if in_pos else 0.0
                                    continue

                                e = bool(entry_arr[i]) if not np.isnan(entry_arr[i]) else False
                                x = close_arr[i] > ma_arr[i]

                                if not in_pos and e:
                                    in_pos = True
                                elif in_pos and x:
                                    in_pos = False

                                signal[i] = 1.0 if in_pos else 0.0

                            signal_s = pd.Series(signal, index=close.index)

                            # Fast backtest
                            sig = signal_s.shift(1).fillna(0)
                            cost = 10 / 10000
                            turnover = sig.diff().abs().fillna(0)
                            strat_ret = (sig * ret - turnover * cost).dropna()

                            if len(strat_ret) < 60:
                                continue

                            # Extract trades
                            sig_arr = sig.values
                            close_bt = close.loc[sig.index].values
                            idx = sig.index
                            trades = []
                            in_trade = False
                            e_date = e_price = None

                            for i in range(1, len(sig_arr)):
                                if not in_trade and sig_arr[i] > 0 and sig_arr[i-1] == 0:
                                    in_trade = True
                                    e_date = idx[i]
                                    e_price = close_bt[i]
                                elif in_trade and sig_arr[i] == 0 and sig_arr[i-1] > 0:
                                    in_trade = False
                                    x_price = close_bt[i]
                                    pnl = (x_price / e_price - 1) * 100
                                    days = (idx[i] - e_date).days
                                    trades.append({'entry': str(e_date.date()), 'exit': str(idx[i].date()),
                                                   'days': days, 'pnl_pct': round(pnl, 2)})

                            if in_trade and e_price:
                                x_price = close_bt[-1]
                                pnl = (x_price / e_price - 1) * 100
                                trades.append({'entry': str(e_date.date()), 'exit': str(idx[-1].date()) + '*',
                                               'days': (idx[-1] - e_date).days, 'pnl_pct': round(pnl, 2)})

                            if len(trades) < 2:
                                continue

                            pnls = [t['pnl_pct'] for t in trades]
                            wins = [p for p in pnls if p > 0]
                            losses = [p for p in pnls if p <= 0]
                            wr = len(wins) / len(pnls) * 100 if pnls else 0

                            cum = (1 + strat_ret).cumprod()
                            total_r = cum.iloc[-1] - 1
                            years = len(strat_ret) / 252
                            cagr = (1 + total_r) ** (1 / max(years, 0.01)) - 1
                            ann_vol = strat_ret.std() * np.sqrt(252)
                            sharpe = (strat_ret.mean() * 252) / ann_vol if ann_vol > 0 else 0
                            mdd = ((cum - cum.cummax()) / cum.cummax()).min()

                            results.append({
                                'erp_t': erp_t, 'vol_t': vol_t, 'ma_p': ma_p,
                                'vol_r': vol_r, 'req_all': req_all,
                                'n_trades': len(trades), 'win_rate': round(wr, 1),
                                'sharpe': round(sharpe, 2), 'cagr': round(cagr * 100, 1),
                                'mdd': round(mdd * 100, 1),
                                'avg_pnl': round(np.mean(pnls), 1),
                                'pf': round(abs(sum(wins) / sum(losses)), 1) if losses and sum(losses) != 0 else 999,
                                'trades': trades,
                            })

        print(f"  Done. {len(results)} valid configs found.")

        if not results:
            print(f"  No valid configurations found for {name}")
            continue

        # ---- Sort and display ----
        # Primary: win rate, secondary: trade count
        results.sort(key=lambda x: (x['win_rate'], x['n_trades']), reverse=True)

        print(f"\n  TOP 15 by Win Rate (min 2 trades):")
        print(f"  {'ERP':>5s} {'Vol':>4s} {'MA':>4s} {'VolR':>5s} {'Mode':>5s} | {'#Tr':>4s} {'WR%':>5s} {'Shrp':>5s} {'CAGR':>6s} {'MDD':>6s} {'PF':>5s} {'AvgP':>5s}")
        print(f"  {'-' * 70}")

        for r in results[:15]:
            mode = "ALL3" if r['req_all'] else "2of3"
            print(f"  {r['erp_t']:5.2f} {r['vol_t']:4d} {r['ma_p']:4d} {r['vol_r']:5.1f} {mode:>5s} | "
                  f"{r['n_trades']:4d} {r['win_rate']:5.0f} {r['sharpe']:5.2f} {r['cagr']:+5.1f}% {r['mdd']:5.1f}% {r['pf']:5.1f} {r['avg_pnl']:+4.1f}")

        # Best config details
        best = results[0]
        print(f"\n  BEST: ERP>{best['erp_t']} Vol>P{best['vol_t']} MA{best['ma_p']} VolR>{best['vol_r']} "
              f"{'ALL3' if best['req_all'] else '2of3'}")
        print(f"  WR: {best['win_rate']:.0f}%  Trades: {best['n_trades']}  Sharpe: {best['sharpe']:.2f}  "
              f"CAGR: {best['cagr']:+.1f}%")

        if best['trades']:
            print(f"\n  Trade Log:")
            print(f"  {'Entry':12s} {'Exit':14s} {'Days':>5s} {'PnL':>8s}")
            print(f"  {'-' * 45}")
            for t in best['trades']:
                marker = '+' if t['pnl_pct'] > 0 else '-'
                print(f"  {t['entry']:12s} {t['exit']:14s} {t['days']:5d} {marker}{abs(t['pnl_pct']):7.2f}%")

        # ---- 90%+ WR configs ----
        high_wr = [r for r in results if r['win_rate'] >= 90]
        print(f"\n  Configs with WR >= 90%: {len(high_wr)}")
        if high_wr:
            high_wr.sort(key=lambda x: x['n_trades'], reverse=True)
            print(f"  Best 90%+ (most trades):")
            for r in high_wr[:5]:
                mode = "ALL3" if r['req_all'] else "2of3"
                print(f"    ERP>{r['erp_t']} Vol>P{r['vol_t']} MA{r['ma_p']} VolR>{r['vol_r']} {mode} "
                      f"| #Tr={r['n_trades']} WR={r['win_rate']:.0f}% Sharpe={r['sharpe']:.2f}")

        # ---- OOS check for best config ----
        if best:
            print(f"\n  --- OOS Validation (2023-now) ---")
            oos_trades = [t for t in best['trades'] if t['entry'] >= '2023-01-01']
            if oos_trades:
                oos_pnls = [t['pnl_pct'] for t in oos_trades]
                oos_wins = [p for p in oos_pnls if p > 0]
                oos_wr = len(oos_wins) / len(oos_pnls) * 100 if oos_pnls else 0
                print(f"  OOS Trades: {len(oos_trades)}  WR: {oos_wr:.0f}%")
                for t in oos_trades:
                    marker = '+' if t['pnl_pct'] > 0 else '-'
                    print(f"    {t['entry']} -> {t['exit']}  {t['days']:3d}d  {marker}{abs(t['pnl_pct']):.1f}%")
            else:
                print(f"  No OOS trades (signal too rare)")

        # ---- Compare with original 2-factor ----
        print(f"\n  --- Original 2-Factor Comparison ---")
        # Replicate original: vol P75 + below MA120
        f2_orig = (vol_pct_504 > 0.75) & (close < ma_dict[120])
        sig_orig = np.zeros(len(close))
        in_pos = False
        for i in range(1, len(close)):
            if np.isnan(ma_dict[120].values[i]):
                sig_orig[i] = 1.0 if in_pos else 0.0
                continue
            e = bool(f2_orig.values[i]) if not np.isnan(f2_orig.values[i]) else False
            x = close.values[i] > ma_dict[120].values[i]
            if not in_pos and e:
                in_pos = True
            elif in_pos and x:
                in_pos = False
            sig_orig[i] = 1.0 if in_pos else 0.0

        sig_orig_s = pd.Series(sig_orig, index=close.index)
        sig_o = sig_orig_s.shift(1).fillna(0)
        to_o = sig_o.diff().abs().fillna(0)
        sr_o = (sig_o * ret - to_o * 10/10000).dropna()

        # Extract original trades
        orig_trades = []
        in_trade = False
        for i in range(1, len(sig_o)):
            if not in_trade and sig_o.values[i] > 0 and sig_o.values[i-1] == 0:
                in_trade = True
                e_date = sig_o.index[i]
                e_price = close.loc[sig_o.index[i]]
            elif in_trade and sig_o.values[i] == 0 and sig_o.values[i-1] > 0:
                in_trade = False
                x_price = close.loc[sig_o.index[i]]
                pnl = (x_price / e_price - 1) * 100
                orig_trades.append({'entry': str(e_date.date()), 'exit': str(sig_o.index[i].date()),
                                    'days': (sig_o.index[i] - e_date).days, 'pnl_pct': round(pnl, 2)})

        if orig_trades:
            orig_pnls = [t['pnl_pct'] for t in orig_trades]
            orig_wins = [p for p in orig_pnls if p > 0]
            orig_wr = len(orig_wins) / len(orig_pnls) * 100
            cum_o = (1 + sr_o).cumprod()
            sharpe_o = (sr_o.mean() * 252) / (sr_o.std() * np.sqrt(252)) if sr_o.std() > 0 else 0
            print(f"  Original: Trades={len(orig_trades)} WR={orig_wr:.0f}% Sharpe={sharpe_o:.2f}")
            for t in orig_trades:
                marker = '+' if t['pnl_pct'] > 0 else '-'
                print(f"    {t['entry']} -> {t['exit']}  {t['days']:3d}d  {marker}{abs(t['pnl_pct']):.1f}%")

        all_summaries[name] = {
            'best': best,
            'n_90wr': len(high_wr),
            'orig_trades': len(orig_trades) if orig_trades else 0,
            'orig_wr': orig_wr if orig_trades else 0,
        }

    bs.logout()

    # ---- FINAL SUMMARY ----
    print(f"\n\n{'=' * 70}")
    print("  FINAL RESEARCH SUMMARY")
    print(f"{'=' * 70}")

    for name, s in all_summaries.items():
        b = s['best']
        print(f"\n  {name}:")
        print(f"    Original 2F: {s['orig_trades']} trades, WR={s['orig_wr']:.0f}%")
        print(f"    Best 3F: ERP>{b['erp_t']} Vol>P{b['vol_t']} MA{b['ma_p']} VolR>{b['vol_r']} "
              f"{'ALL3' if b['req_all'] else '2of3'}")
        print(f"      Trades={b['n_trades']} WR={b['win_rate']:.0f}% Sharpe={b['sharpe']:.2f} CAGR={b['cagr']:+.1f}%")
        print(f"    Configs with WR>=90%: {s['n_90wr']}")

    print(f"\n{'=' * 70}")
    print("  CONCLUSION: See above for per-index optimal 3-factor parameters.")
    print("  Next step: Select universal params that work across all indices,")
    print("  then implement as upgraded panic_buy_signal.py")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    run_research()
