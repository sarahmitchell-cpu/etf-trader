#!/usr/bin/env python3
"""
Deep Dive: Volatility + MA Timing Strategy (波动率+均线择时)
Based on HighVol80&BelowMA250 which showed OOS Sharpe 1.47-1.69

Strategy Logic:
- High volatility (fear) + price below long-term MA (oversold) = BUY
- This is a contrarian "buy the panic dip" strategy
- Exit when conditions normalize

This script does:
1. Parameter grid search (vol window, vol percentile, MA period, vol lookback)
2. Multi-index testing (6 indices)
3. Trade-level analysis (win rate, avg gain/loss, holding days)
4. Enhanced versions: with exit rules, partial positions, combined signals
5. Complete trading system design

Author: Sarah Mitchell / VisionClaw
Date: 2026-03-25
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import baostock as bs
import json
from datetime import datetime
from itertools import product

# ============================================================
# DATA
# ============================================================

def get_index_prices():
    """Fetch all index daily prices"""
    bs.login()

    indices = {
        '沪深300': 'sh.000300',
        '中证500': 'sh.000905',
        '中证1000': 'sh.000852',
        '上证50': 'sh.000016',
        '创业板指': 'sz.399006',
        '上证指数': 'sh.000001',
    }

    prices = {}
    for name, code in indices.items():
        try:
            rs = bs.query_history_k_data_plus(
                code, "date,open,high,low,close,volume",
                start_date='2010-01-01',
                end_date=datetime.now().strftime('%Y-%m-%d'),
                frequency="d", adjustflag="3"
            )
            data = []
            while rs.error_code == '0' and rs.next():
                data.append(rs.get_row_data())
            if data:
                df = pd.DataFrame(data, columns=rs.fields)
                df['date'] = pd.to_datetime(df['date'])
                for c in ['open','high','low','close','volume']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                df = df.set_index('date').sort_index()
                df = df[df['close'] > 0]
                prices[name] = df
                print(f"  {name}: {len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}")
        except Exception as e:
            print(f"  {name} error: {e}")

    bs.logout()
    return prices

# ============================================================
# STRATEGY CORE
# ============================================================

def generate_vol_ma_signal(close, vol_window=20, vol_lookback=504, vol_pct_thresh=80,
                            ma_period=250, exit_mode='condition_clear'):
    """
    Generate HighVol & BelowMA signal

    Args:
        close: price series
        vol_window: window for calculating realized volatility (days)
        vol_lookback: lookback for vol percentile ranking (days)
        vol_pct_thresh: percentile threshold (e.g. 80 = top 20%)
        ma_period: MA period for trend filter
        exit_mode: 'condition_clear' (exit when signal turns off) or
                   'above_ma' (exit only when price goes above MA)

    Returns:
        signal series (1=long, 0=cash)
    """
    # Realized volatility
    ret = close.pct_change()
    vol = ret.rolling(vol_window).std() * np.sqrt(252) * 100  # annualized %

    # Vol percentile (rolling)
    vol_pct = vol.rolling(vol_lookback, min_periods=min(252, vol_lookback)).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5, raw=True
    )

    # MA
    ma = close.rolling(ma_period, min_periods=ma_period).mean()

    # Entry: high vol + below MA
    entry_cond = (vol_pct > vol_pct_thresh / 100) & (close < ma)

    if exit_mode == 'condition_clear':
        # Simple: signal = entry condition
        signal = entry_cond.astype(float)
    elif exit_mode == 'above_ma':
        # Hold until price goes above MA (more persistent)
        signal = pd.Series(0.0, index=close.index)
        in_pos = False
        for i in range(1, len(close)):
            if pd.isna(entry_cond.iloc[i]) or pd.isna(ma.iloc[i]):
                signal.iloc[i] = 0
                continue
            if not in_pos and entry_cond.iloc[i]:
                in_pos = True
            elif in_pos and close.iloc[i] > ma.iloc[i]:
                in_pos = False
            signal.iloc[i] = 1.0 if in_pos else 0.0
    elif exit_mode == 'vol_normalize':
        # Hold until vol drops below median
        signal = pd.Series(0.0, index=close.index)
        in_pos = False
        for i in range(1, len(close)):
            if pd.isna(entry_cond.iloc[i]) or pd.isna(vol_pct.iloc[i]):
                signal.iloc[i] = 0
                continue
            if not in_pos and entry_cond.iloc[i]:
                in_pos = True
            elif in_pos and vol_pct.iloc[i] < 0.5:
                in_pos = False
            signal.iloc[i] = 1.0 if in_pos else 0.0
    elif exit_mode == 'trailing_stop':
        # Enter on signal, exit on trailing stop -8%
        signal = pd.Series(0.0, index=close.index)
        in_pos = False
        peak = 0
        for i in range(1, len(close)):
            if pd.isna(entry_cond.iloc[i]):
                signal.iloc[i] = 0
                continue
            if not in_pos and entry_cond.iloc[i]:
                in_pos = True
                peak = close.iloc[i]
            elif in_pos:
                peak = max(peak, close.iloc[i])
                if close.iloc[i] < peak * 0.92:  # -8% trailing stop
                    in_pos = False
            signal.iloc[i] = 1.0 if in_pos else 0.0

    return signal


def generate_enhanced_signal(close, vol_window=20, vol_lookback=504, vol_pct_thresh=80,
                              ma_period=250, enhancement='none'):
    """Enhanced versions of the base strategy"""
    ret = close.pct_change()
    vol = ret.rolling(vol_window).std() * np.sqrt(252) * 100
    vol_pct = vol.rolling(vol_lookback, min_periods=min(252, vol_lookback)).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5, raw=True
    )
    ma_long = close.rolling(ma_period, min_periods=ma_period).mean()

    if enhancement == 'gradual':
        # Gradual position sizing based on vol percentile
        # Higher vol = larger position (more fear = more buying)
        entry_cond = (close < ma_long)
        weight = pd.Series(0.0, index=close.index)
        mask = entry_cond & (vol_pct > 0.6)
        weight[mask] = 0.25
        mask = entry_cond & (vol_pct > 0.7)
        weight[mask] = 0.5
        mask = entry_cond & (vol_pct > 0.8)
        weight[mask] = 0.75
        mask = entry_cond & (vol_pct > 0.9)
        weight[mask] = 1.0
        return weight

    elif enhancement == 'dual_ma':
        # Add short MA cross: enter when vol high + below long MA + above short MA (reversal starting)
        ma_short = close.rolling(20, min_periods=20).mean()
        entry = (vol_pct > vol_pct_thresh / 100) & (close < ma_long) & (close > ma_short)
        return entry.astype(float)

    elif enhancement == 'vol_expanding':
        # Enter when vol is expanding (vol > vol's own MA) + below MA
        vol_ma = vol.rolling(60).mean()
        entry = (vol > vol_ma) & (vol_pct > vol_pct_thresh / 100) & (close < ma_long)
        return entry.astype(float)

    elif enhancement == 'rsi_confirm':
        # Add RSI oversold confirmation
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        entry = (vol_pct > vol_pct_thresh / 100) & (close < ma_long) & (rsi < 40)
        return entry.astype(float)

    elif enhancement == 'distance_filter':
        # Only buy when price is significantly below MA (not just slightly)
        dist = (close - ma_long) / ma_long
        entry = (vol_pct > vol_pct_thresh / 100) & (dist < -0.05)  # at least 5% below MA
        return entry.astype(float)

    elif enhancement == 'combined_best':
        # Combine: high vol + below long MA + above short MA (reversal) + significant dip
        ma_short = close.rolling(20).mean()
        dist = (close - ma_long) / ma_long
        # Enter: high vol + far below long MA + starting to recover (above short MA)
        signal = pd.Series(0.0, index=close.index)
        in_pos = False
        for i in range(max(ma_period, vol_lookback), len(close)):
            if pd.isna(vol_pct.iloc[i]) or pd.isna(ma_long.iloc[i]) or pd.isna(ma_short.iloc[i]):
                continue
            entry = (vol_pct.iloc[i] > vol_pct_thresh/100) and (dist.iloc[i] < -0.03)
            exit_cond = close.iloc[i] > ma_long.iloc[i]

            if not in_pos and entry:
                in_pos = True
            elif in_pos and exit_cond:
                in_pos = False
            signal.iloc[i] = 1.0 if in_pos else 0.0
        return signal

    return pd.Series(0.0, index=close.index)


# ============================================================
# BACKTEST & ANALYSIS
# ============================================================

def backtest_strategy(close, signal, cost_bps=10):
    """
    Backtest with next-day execution and detailed trade analysis
    Returns metrics dict and trade list
    """
    signal = signal.shift(1).fillna(0)  # next-day execution
    ret = close.pct_change()
    cost = cost_bps / 10000
    turnover = signal.diff().abs().fillna(0)
    strat_ret = (signal * ret - turnover * cost).dropna()

    if len(strat_ret) < 60:
        return None, []

    # Performance metrics
    cum = (1 + strat_ret).cumprod()
    total = cum.iloc[-1] - 1
    years = len(strat_ret) / 252
    cagr = (1 + total) ** (1/max(years, 0.01)) - 1
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe = (strat_ret.mean() * 252) / ann_vol if ann_vol > 0 else 0
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    # Win rate (daily)
    active_days = strat_ret[signal.loc[strat_ret.index] > 0]
    daily_win_rate = (active_days > 0).mean() * 100 if len(active_days) > 0 else 0

    # Trade-level analysis
    trades = []
    in_trade = False
    entry_date = None
    entry_price = None

    sig_aligned = signal.loc[close.index].fillna(0)
    for i in range(1, len(sig_aligned)):
        if not in_trade and sig_aligned.iloc[i] > 0 and sig_aligned.iloc[i-1] == 0:
            in_trade = True
            entry_date = sig_aligned.index[i]
            entry_price = close.iloc[i]
        elif in_trade and (sig_aligned.iloc[i] == 0 and sig_aligned.iloc[i-1] > 0):
            in_trade = False
            exit_date = sig_aligned.index[i]
            exit_price = close.iloc[i]
            pnl = (exit_price / entry_price - 1) * 100
            days = (exit_date - entry_date).days
            trades.append({
                'entry': str(entry_date.date()),
                'exit': str(exit_date.date()),
                'days': days,
                'pnl_pct': round(pnl, 2),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
            })

    # Close open trade
    if in_trade and entry_price is not None:
        exit_price = close.iloc[-1]
        pnl = (exit_price / entry_price - 1) * 100
        days = (close.index[-1] - entry_date).days
        trades.append({
            'entry': str(entry_date.date()),
            'exit': str(close.index[-1].date()) + ' (open)',
            'days': days,
            'pnl_pct': round(pnl, 2),
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
        })

    # Trade stats
    if trades:
        pnls = [t['pnl_pct'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        trade_win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        avg_days = np.mean([t['days'] for t in trades])
        max_win = max(pnls) if pnls else 0
        max_loss = min(pnls) if pnls else 0
    else:
        trade_win_rate = avg_win = avg_loss = profit_factor = avg_days = max_win = max_loss = 0

    # B&H comparison
    bh_ret = ret.loc[strat_ret.index].dropna()
    bh_cum = (1 + bh_ret).cumprod()
    bh_total = bh_cum.iloc[-1] - 1
    bh_cagr = (1 + bh_total) ** (1/max(years, 0.01)) - 1
    bh_vol = bh_ret.std() * np.sqrt(252)
    bh_sharpe = (bh_ret.mean() * 252) / bh_vol if bh_vol > 0 else 0
    bh_mdd = ((bh_cum - bh_cum.cummax()) / bh_cum.cummax()).min()

    metrics = {
        'cagr': round(cagr * 100, 2),
        'ann_vol': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'mdd': round(mdd * 100, 2),
        'calmar': round(calmar, 3),
        'years': round(years, 1),
        'pct_in_market': round(signal.loc[strat_ret.index].mean() * 100, 1),
        'n_trades': len(trades),
        'trade_win_rate': round(trade_win_rate, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999,
        'avg_holding_days': round(avg_days, 1),
        'max_win': round(max_win, 2),
        'max_loss': round(max_loss, 2),
        'daily_win_rate': round(daily_win_rate, 1),
        'bh_cagr': round(bh_cagr * 100, 2),
        'bh_sharpe': round(bh_sharpe, 3),
        'bh_mdd': round(bh_mdd * 100, 2),
        'start': str(strat_ret.index[0].date()),
        'end': str(strat_ret.index[-1].date()),
    }

    return metrics, trades


# ============================================================
# MAIN RESEARCH
# ============================================================

def run_research():
    print("=" * 70)
    print("DEEP DIVE: Volatility + MA Timing Strategy (波动率+均线择时)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. Get data
    print("\n[1/5] Fetching index prices...")
    prices = get_index_prices()

    if not prices:
        print("ERROR: No price data!")
        return

    # 2. Parameter grid search
    print("\n[2/5] Parameter grid search...")

    split_date = pd.Timestamp('2023-01-01')

    param_grid = {
        'vol_window': [10, 20, 30, 60],
        'vol_lookback': [252, 504, 756],
        'vol_pct_thresh': [60, 70, 80, 90],
        'ma_period': [120, 200, 250, 500],
    }

    all_results = []

    for idx_name, price_df in prices.items():
        close = price_df['close']
        print(f"\n  --- {idx_name} ---")
        count = 0

        for vw, vl, vp, mp in product(
            param_grid['vol_window'],
            param_grid['vol_lookback'],
            param_grid['vol_pct_thresh'],
            param_grid['ma_period']
        ):
            signal = generate_vol_ma_signal(close, vol_window=vw, vol_lookback=vl,
                                             vol_pct_thresh=vp, ma_period=mp,
                                             exit_mode='condition_clear')

            # Full period
            metrics, _ = backtest_strategy(close, signal)
            if metrics:
                metrics['index'] = idx_name
                metrics['params'] = f"Vol{vw}_LB{vl}_P{vp}_MA{mp}"
                metrics['vol_window'] = vw
                metrics['vol_lookback'] = vl
                metrics['vol_pct_thresh'] = vp
                metrics['ma_period'] = mp
                metrics['exit_mode'] = 'condition_clear'
                metrics['period'] = 'FULL'
                all_results.append(metrics)

            # IS period
            close_is = close[close.index < split_date]
            signal_is = signal[signal.index < split_date]
            metrics_is, _ = backtest_strategy(close_is, signal_is)
            if metrics_is:
                metrics_is['index'] = idx_name
                metrics_is['params'] = f"Vol{vw}_LB{vl}_P{vp}_MA{mp}"
                metrics_is['vol_window'] = vw
                metrics_is['vol_lookback'] = vl
                metrics_is['vol_pct_thresh'] = vp
                metrics_is['ma_period'] = mp
                metrics_is['exit_mode'] = 'condition_clear'
                metrics_is['period'] = 'IS'
                all_results.append(metrics_is)

            # OOS period
            close_oos = close[close.index >= split_date]
            signal_oos = signal[signal.index >= split_date]
            metrics_oos, _ = backtest_strategy(close_oos, signal_oos)
            if metrics_oos:
                metrics_oos['index'] = idx_name
                metrics_oos['params'] = f"Vol{vw}_LB{vl}_P{vp}_MA{mp}"
                metrics_oos['vol_window'] = vw
                metrics_oos['vol_lookback'] = vl
                metrics_oos['vol_pct_thresh'] = vp
                metrics_oos['ma_period'] = mp
                metrics_oos['exit_mode'] = 'condition_clear'
                metrics_oos['period'] = 'OOS'
                all_results.append(metrics_oos)

            count += 1

        print(f"    Tested {count} parameter combos")

    # 3. Exit mode comparison (for best params)
    print("\n[3/5] Testing exit modes...")

    exit_modes = ['condition_clear', 'above_ma', 'vol_normalize', 'trailing_stop']

    # Use the original params as baseline
    base_params = {'vol_window': 20, 'vol_lookback': 504, 'vol_pct_thresh': 80, 'ma_period': 250}

    for idx_name, price_df in prices.items():
        close = price_df['close']
        print(f"  {idx_name}:")

        for exit_mode in exit_modes:
            signal = generate_vol_ma_signal(close, **base_params, exit_mode=exit_mode)

            for period_name, c, s in [
                ('FULL', close, signal),
                ('IS', close[close.index < split_date], signal[signal.index < split_date]),
                ('OOS', close[close.index >= split_date], signal[signal.index >= split_date]),
            ]:
                metrics, _ = backtest_strategy(c, s)
                if metrics:
                    metrics['index'] = idx_name
                    metrics['params'] = f"Base_{exit_mode}"
                    metrics['vol_window'] = base_params['vol_window']
                    metrics['vol_lookback'] = base_params['vol_lookback']
                    metrics['vol_pct_thresh'] = base_params['vol_pct_thresh']
                    metrics['ma_period'] = base_params['ma_period']
                    metrics['exit_mode'] = exit_mode
                    metrics['period'] = period_name
                    all_results.append(metrics)
                    if period_name == 'OOS':
                        print(f"    {exit_mode:20s} Sharpe={metrics['sharpe']:.2f} CAGR={metrics['cagr']:.1f}% MDD={metrics['mdd']:.1f}% InMkt={metrics['pct_in_market']:.0f}%")

    # 4. Enhanced strategies
    print("\n[4/5] Testing enhanced strategies...")

    enhancements = ['gradual', 'dual_ma', 'vol_expanding', 'rsi_confirm', 'distance_filter', 'combined_best']

    for idx_name, price_df in prices.items():
        close = price_df['close']
        print(f"  {idx_name}:")

        for enh in enhancements:
            signal = generate_enhanced_signal(close, **base_params, enhancement=enh)

            for period_name, c, s in [
                ('FULL', close, signal),
                ('IS', close[close.index < split_date], signal[signal.index < split_date]),
                ('OOS', close[close.index >= split_date], signal[signal.index >= split_date]),
            ]:
                metrics, _ = backtest_strategy(c, s)
                if metrics:
                    metrics['index'] = idx_name
                    metrics['params'] = f"Enhanced_{enh}"
                    metrics['vol_window'] = base_params['vol_window']
                    metrics['vol_lookback'] = base_params['vol_lookback']
                    metrics['vol_pct_thresh'] = base_params['vol_pct_thresh']
                    metrics['ma_period'] = base_params['ma_period']
                    metrics['exit_mode'] = enh
                    metrics['period'] = period_name
                    all_results.append(metrics)
                    if period_name == 'OOS':
                        print(f"    {enh:20s} Sharpe={metrics['sharpe']:.2f} CAGR={metrics['cagr']:.1f}% MDD={metrics['mdd']:.1f}% InMkt={metrics['pct_in_market']:.0f}%")

    # 5. Analysis
    print("\n[5/5] Analyzing results...")
    df = pd.DataFrame(all_results)

    # Save full results
    csv_path = '/Users/claw/etf-trader/data/vol_ma_deepdive_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} results to {csv_path}")

    # OOS Analysis
    oos = df[df['period'] == 'OOS'].copy()
    is_df = df[df['period'] == 'IS'].copy()

    print(f"\n{'='*70}")
    print("RESULTS ANALYSIS")
    print(f"{'='*70}")

    # Top OOS by Sharpe (per index)
    print(f"\n--- Top 5 OOS Sharpe per Index ---")
    for idx_name in prices.keys():
        idx_oos = oos[oos['index'] == idx_name].nlargest(5, 'sharpe')
        print(f"\n  {idx_name}:")
        for _, r in idx_oos.iterrows():
            print(f"    {r['params']:30s} Sharpe={r['sharpe']:.3f} CAGR={r['cagr']:.1f}% MDD={r['mdd']:.1f}% WinRate={r['trade_win_rate']:.0f}% InMkt={r['pct_in_market']:.0f}%")

    # Global top 20 OOS
    print(f"\n--- Global Top 20 OOS Sharpe ---")
    top20 = oos.nlargest(20, 'sharpe')
    for _, r in top20.iterrows():
        print(f"  {r['index']:8s} {r['params']:30s} Sharpe={r['sharpe']:.3f} CAGR={r['cagr']:.1f}% MDD={r['mdd']:.1f}% Calmar={r['calmar']:.2f}")

    # IS→OOS consistency
    print(f"\n--- IS→OOS Consistency (both Sharpe > 0) ---")
    consistent = []
    for idx_name in prices.keys():
        idx_is = is_df[is_df['index'] == idx_name]
        idx_oos = oos[oos['index'] == idx_name]

        for _, oos_row in idx_oos.iterrows():
            is_match = idx_is[(idx_is['params'] == oos_row['params']) & (idx_is['exit_mode'] == oos_row['exit_mode'])]
            if len(is_match) > 0:
                is_sharpe = is_match.iloc[0]['sharpe']
                if is_sharpe > 0 and oos_row['sharpe'] > 0:
                    consistent.append({
                        'index': idx_name,
                        'params': oos_row['params'],
                        'is_sharpe': is_sharpe,
                        'oos_sharpe': oos_row['sharpe'],
                        'oos_cagr': oos_row['cagr'],
                        'oos_mdd': oos_row['mdd'],
                        'oos_calmar': oos_row['calmar'],
                        'oos_win_rate': oos_row['trade_win_rate'],
                        'oos_pct_in_market': oos_row['pct_in_market'],
                    })

    if consistent:
        cdf = pd.DataFrame(consistent).sort_values('oos_sharpe', ascending=False)
        print(f"  {len(cdf)} consistent combos found")
        for _, r in cdf.head(20).iterrows():
            print(f"  {r['index']:8s} {r['params']:30s} IS={r['is_sharpe']:.2f}->OOS={r['oos_sharpe']:.2f} CAGR={r['oos_cagr']:.1f}% MDD={r['oos_mdd']:.1f}%")

    # Detailed trade analysis for top strategies
    print(f"\n{'='*70}")
    print("DETAILED TRADE ANALYSIS (Top Strategies)")
    print(f"{'='*70}")

    # Pick top 3 consistent strategies across indices
    if consistent:
        top_consistent = cdf.head(6)
        for _, row in top_consistent.iterrows():
            idx_name = row['index']
            params_str = row['params']
            close = prices[idx_name]['close']

            # Parse params
            if params_str.startswith('Vol'):
                parts = params_str.split('_')
                vw = int(parts[0].replace('Vol',''))
                vl = int(parts[1].replace('LB',''))
                vp = int(parts[2].replace('P',''))
                mp = int(parts[3].replace('MA',''))
                signal = generate_vol_ma_signal(close, vol_window=vw, vol_lookback=vl,
                                                vol_pct_thresh=vp, ma_period=mp)
            elif params_str.startswith('Base_'):
                exit_mode = params_str.replace('Base_', '')
                signal = generate_vol_ma_signal(close, **base_params, exit_mode=exit_mode)
            elif params_str.startswith('Enhanced_'):
                enh = params_str.replace('Enhanced_', '')
                signal = generate_enhanced_signal(close, **base_params, enhancement=enh)
            else:
                continue

            # OOS only
            close_oos = close[close.index >= split_date]
            signal_oos = signal[signal.index >= split_date]
            metrics, trades = backtest_strategy(close_oos, signal_oos)

            if metrics and trades:
                print(f"\n  {idx_name} | {params_str}")
                print(f"  Sharpe={metrics['sharpe']:.3f} CAGR={metrics['cagr']:.1f}% MDD={metrics['mdd']:.1f}%")
                print(f"  Trades={metrics['n_trades']} WinRate={metrics['trade_win_rate']:.0f}% AvgWin={metrics['avg_win']:.1f}% AvgLoss={metrics['avg_loss']:.1f}%")
                print(f"  ProfitFactor={metrics['profit_factor']:.2f} AvgHold={metrics['avg_holding_days']:.0f}d")
                print(f"  {'Entry':12s} {'Exit':16s} {'Days':>5s} {'PnL%':>8s}")
                print(f"  {'-'*50}")
                for t in trades:
                    emoji = '+' if t['pnl_pct'] > 0 else '-'
                    print(f"  {t['entry']:12s} {t['exit']:16s} {t['days']:5d} {emoji}{abs(t['pnl_pct']):7.2f}%")

    # Parameter sensitivity analysis
    print(f"\n{'='*70}")
    print("PARAMETER SENSITIVITY (OOS Sharpe, 沪深300)")
    print(f"{'='*70}")

    oos_300 = oos[(oos['index'] == '沪深300') & (oos['exit_mode'] == 'condition_clear')].copy()
    if len(oos_300) > 0:
        for param_name in ['vol_window', 'vol_lookback', 'vol_pct_thresh', 'ma_period']:
            print(f"\n  {param_name}:")
            grouped = oos_300.groupby(param_name)['sharpe'].agg(['mean', 'std', 'max', 'count'])
            for val, row in grouped.iterrows():
                print(f"    {val:6.0f} -> mean={row['mean']:.3f} std={row['std']:.3f} max={row['max']:.3f} (n={row['count']:.0f})")

    # Current signal status
    print(f"\n{'='*70}")
    print("CURRENT SIGNAL STATUS")
    print(f"{'='*70}")

    for idx_name, price_df in prices.items():
        close = price_df['close']
        ret = close.pct_change()
        vol20 = ret.rolling(20).std() * np.sqrt(252) * 100
        vol_pct = vol20.rolling(504, min_periods=252).apply(
            lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5, raw=True
        )
        ma250 = close.rolling(250).mean()

        latest = close.iloc[-1]
        latest_vol = vol20.iloc[-1] if not pd.isna(vol20.iloc[-1]) else 0
        latest_vol_pct = vol_pct.iloc[-1] if not pd.isna(vol_pct.iloc[-1]) else 0
        latest_ma = ma250.iloc[-1] if not pd.isna(ma250.iloc[-1]) else 0
        dist_ma = (latest - latest_ma) / latest_ma * 100 if latest_ma > 0 else 0
        signal_on = latest_vol_pct > 0.8 and latest < latest_ma

        status = "ON (BUY)" if signal_on else "OFF (CASH)"
        print(f"  {idx_name:8s} Price={latest:8.1f} MA250={latest_ma:8.1f} Dist={dist_ma:+.1f}% Vol={latest_vol:.1f}% VolPct={latest_vol_pct*100:.0f}% -> {status}")

    # Save summary
    summary = {
        'total_results': len(df),
        'oos_results': len(oos),
        'global_top10_oos': oos.nlargest(10, 'sharpe')[['index','params','sharpe','cagr','mdd','calmar','trade_win_rate','pct_in_market','n_trades']].to_dict('records'),
        'consistent_top10': consistent[:10] if consistent else [],
        'parameter_sensitivity': {},
    }

    if len(oos_300) > 0:
        for param_name in ['vol_window', 'vol_lookback', 'vol_pct_thresh', 'ma_period']:
            grouped = oos_300.groupby(param_name)['sharpe'].agg(['mean', 'max']).reset_index()
            summary['parameter_sensitivity'][param_name] = grouped.to_dict('records')

    json_path = '/Users/claw/etf-trader/data/vol_ma_deepdive_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved to {json_path}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return df, summary


if __name__ == '__main__':
    df, summary = run_research()
