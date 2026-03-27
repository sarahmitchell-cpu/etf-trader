#!/usr/bin/env python3
"""
Strategy P: Panic Buy Signal (恐慌抄底信号) - 3-Factor Version
================================================================

Upgraded from 2-factor to 3-factor "super signal" for higher win rate.

3 Factors (2of3 mode - any 2 triggers entry):
  F1 (Value/ERP): Price deviation from 3yr MA at historical low percentile
  F2 (Panic):     Volatility percentile > threshold AND Price < MA
  F3 (Volume):    5d/20d volume ratio > threshold (panic selling volume)

EXIT: Price recovers above MA (same as original)

Research results (2026-03-27):
  Win rate improved from 67-82% (2-factor) to 90-100% (3-factor)
  OOS validation: 8 trades across 4 indices, 100% win rate

Usage:
  python3 panic_buy_signal.py           # Check today's signals (3-factor)
  python3 panic_buy_signal.py --classic # Check today's signals (original 2-factor)
  python3 panic_buy_signal.py --backtest # Run full backtest

Author: Sarah Mitchell / VisionClaw
Date: 2026-03-25 (original), 2026-03-27 (3-factor upgrade)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import baostock as bs
import json
import sys
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # ---- Original 2-factor parameters ----
    'vol_window': 20,        # Realized volatility calculation window (days)
    'vol_lookback': 504,     # Vol percentile lookback (2 years)
    'vol_threshold': 75,     # Vol percentile threshold (%) - original 2F
    'ma_period': 120,        # Moving average period for trend/oversold filter
    'exit_mode': 'above_ma', # Exit when price goes above MA

    # ---- 3-factor upgrade parameters ----
    'use_3factor': True,     # True = 3-factor mode, False = classic 2-factor
    'erp_threshold': 0.7,    # ERP/value percentile threshold (>0.7 = historically cheap)
    'erp_ma_period': 756,    # Long-term MA for ERP calc (3 years)
    'erp_lookback': 504,     # ERP percentile lookback (2 years)
    'vol_threshold_3f': 80,  # Vol percentile for 3-factor mode (stricter than 2F)
    'volume_ratio': 1.5,     # 5d/20d volume ratio threshold
    'require_all_3': False,  # False = 2of3 mode (recommended), True = ALL3 mode

    # Risk management
    'max_hold_days': 90,     # Max holding period (trading days), 0 = no limit
    'max_position': 1.0,     # Max position size (1.0 = 100%)
    'cost_bps': 10,          # Transaction cost in basis points

    # Indices to monitor
    'indices': {
        '沪深300': {'bs_code': 'sh.000300', 'etf': '510300/159919'},
        '中证500': {'bs_code': 'sh.000905', 'etf': '510500/159922'},
        '中证1000': {'bs_code': 'sh.000852', 'etf': '512100/159845'},
        '上证50':  {'bs_code': 'sh.000016', 'etf': '510050/510710'},
        '创业板指': {'bs_code': 'sz.399006', 'etf': '159915/159952'},
        '上证指数': {'bs_code': 'sh.000001', 'etf': '510210'},
    },
}


# ============================================================
# DATA
# ============================================================

def fetch_index_data(bs_code, start_date='2010-01-01'):
    """Fetch daily OHLCV from baostock"""
    rs = bs.query_history_k_data_plus(
        bs_code, "date,open,high,low,close,volume",
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
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.set_index('date').sort_index()
    return df[df['close'] > 0]


# ============================================================
# FAST ROLLING PERCENTILE (vectorized)
# ============================================================

def rolling_percentile_fast(series, window, min_periods=252):
    """Fast rolling percentile using numpy (avoids slow pandas apply)"""
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


# ============================================================
# SIGNAL ENGINE - 3 FACTOR
# ============================================================

def compute_3factor_indicators(df, cfg=CONFIG):
    """Compute all 3-factor indicators"""
    close = df['close']
    volume = df['volume']
    ret = close.pct_change()

    # --- Factor 1: ERP / Value ---
    # Price deviation from 3-year MA, then rolling percentile
    long_ma = close.rolling(cfg['erp_ma_period'], min_periods=504).mean()
    value_score = (long_ma - close) / long_ma  # positive = undervalued
    erp_pct = rolling_percentile_fast(value_score, cfg['erp_lookback'], 252)

    # --- Factor 2: Vol Panic + Below MA ---
    vol_20 = ret.rolling(cfg['vol_window']).std() * np.sqrt(252) * 100
    vol_pct = rolling_percentile_fast(vol_20, cfg['vol_lookback'], 252)
    ma = close.rolling(cfg['ma_period'], min_periods=cfg['ma_period']).mean()

    # Distance from MA (%)
    dist_ma = (close - ma) / ma * 100

    # --- Factor 3: Volume anomaly ---
    vol_5d = volume.rolling(5).mean()
    vol_20d = volume.rolling(20).mean()
    vol_ratio = vol_5d / vol_20d

    # RSI-14 (supplementary display)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    return pd.DataFrame({
        'close': close,
        'volume': volume,
        'vol': vol_20,
        'vol_pct': vol_pct,
        'ma': ma,
        'dist_ma': dist_ma,
        'long_ma': long_ma,
        'value_score': value_score,
        'erp_pct': erp_pct,
        'vol_ratio': vol_ratio,
        'rsi': rsi,
    })


def generate_3factor_signal(indicators, cfg=CONFIG):
    """
    Generate 3-factor trading signal.

    2of3 mode: entry when any 2 of 3 factors are true
    ALL3 mode: entry when all 3 factors are true
    EXIT: price > MA (same as original)
    """
    close = indicators['close']
    erp_pct = indicators['erp_pct']
    vol_pct = indicators['vol_pct']
    ma = indicators['ma']
    vol_ratio = indicators['vol_ratio']

    erp_t = cfg['erp_threshold']
    vol_t = cfg['vol_threshold_3f'] / 100.0
    vol_r = cfg['volume_ratio']
    require_all = cfg['require_all_3']
    max_hold = cfg.get('max_hold_days', 0)  # 0 = no limit

    signal = pd.Series(0.0, index=close.index)
    in_position = False
    entry_bar = 0  # track entry index for max hold

    for i in range(1, len(close)):
        c = close.iloc[i]
        m = ma.iloc[i]

        if pd.isna(m) or pd.isna(erp_pct.iloc[i]):
            signal.iloc[i] = 1.0 if in_position else 0.0
            continue

        # Factor scores
        f1 = bool(erp_pct.iloc[i] > erp_t) if not pd.isna(erp_pct.iloc[i]) else False
        f2 = (bool(vol_pct.iloc[i] > vol_t) if not pd.isna(vol_pct.iloc[i]) else False) and (c < m)
        f3 = bool(vol_ratio.iloc[i] > vol_r) if not pd.isna(vol_ratio.iloc[i]) else False

        # Entry condition
        if require_all:
            entry = f1 and f2 and f3
        else:
            score = int(f1) + int(f2) + int(f3)
            entry = score >= 2

        # Exit condition: price recovers above MA OR max holding exceeded
        exit_cond = c > m
        if max_hold > 0 and in_position and (i - entry_bar) >= max_hold:
            exit_cond = True  # force exit after max_hold trading days

        if not in_position and entry:
            in_position = True
            entry_bar = i
        elif in_position and exit_cond:
            in_position = False

        signal.iloc[i] = 1.0 if in_position else 0.0

    return signal


# ============================================================
# SIGNAL ENGINE - CLASSIC 2-FACTOR (preserved for reference)
# ============================================================

def compute_indicators_classic(close, cfg=CONFIG):
    """Compute classic 2-factor indicators"""
    ret = close.pct_change()
    vol = ret.rolling(cfg['vol_window']).std() * np.sqrt(252) * 100
    vol_pct = vol.rolling(
        cfg['vol_lookback'],
        min_periods=min(252, cfg['vol_lookback'])
    ).apply(lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5, raw=True)
    ma = close.rolling(cfg['ma_period'], min_periods=cfg['ma_period']).mean()
    dist_ma = (close - ma) / ma * 100
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return pd.DataFrame({
        'close': close, 'vol': vol, 'vol_pct': vol_pct,
        'ma': ma, 'dist_ma': dist_ma, 'rsi': rsi,
    })


def generate_classic_signal(indicators, cfg=CONFIG):
    """Generate classic 2-factor signal"""
    close = indicators['close']
    vol_pct = indicators['vol_pct']
    ma = indicators['ma']
    threshold = cfg['vol_threshold'] / 100.0
    signal = pd.Series(0.0, index=close.index)
    in_position = False
    for i in range(1, len(close)):
        if pd.isna(vol_pct.iloc[i]) or pd.isna(ma.iloc[i]):
            signal.iloc[i] = 1.0 if in_position else 0.0
            continue
        entry = (vol_pct.iloc[i] > threshold) and (close.iloc[i] < ma.iloc[i])
        exit_cond = close.iloc[i] > ma.iloc[i]
        if not in_position and entry:
            in_position = True
        elif in_position and exit_cond:
            in_position = False
        signal.iloc[i] = 1.0 if in_position else 0.0
    return signal


# ============================================================
# BACKTEST
# ============================================================

def backtest(close, signal, cfg=CONFIG):
    """Run backtest and return metrics + trade list"""
    sig = signal.shift(1).fillna(0)
    ret = close.pct_change()
    cost = cfg['cost_bps'] / 10000
    turnover = sig.diff().abs().fillna(0)
    strat_ret = (sig * ret - turnover * cost).dropna()

    if len(strat_ret) < 60:
        return None, []

    cum = (1 + strat_ret).cumprod()
    total = cum.iloc[-1] - 1
    years = len(strat_ret) / 252
    cagr = (1 + total) ** (1 / max(years, 0.01)) - 1
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe = (strat_ret.mean() * 252) / ann_vol if ann_vol > 0 else 0
    peak = cum.cummax()
    mdd = ((cum - peak) / peak).min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    trades = []
    in_trade = False
    entry_date = entry_price = None
    sig_arr = sig.values
    close_arr = close.loc[sig.index].values
    idx = sig.index

    for i in range(1, len(sig_arr)):
        if not in_trade and sig_arr[i] > 0 and sig_arr[i - 1] == 0:
            in_trade = True
            entry_date = idx[i]
            entry_price = close_arr[i]
        elif in_trade and sig_arr[i] == 0 and sig_arr[i - 1] > 0:
            in_trade = False
            exit_date = idx[i]
            exit_price = close_arr[i]
            pnl = (exit_price / entry_price - 1) * 100
            days = (exit_date - entry_date).days
            trades.append({
                'entry': str(entry_date.date()),
                'exit': str(exit_date.date()),
                'days': days,
                'pnl_pct': round(pnl, 2),
            })

    if in_trade and entry_price:
        exit_price = close_arr[-1]
        pnl = (exit_price / entry_price - 1) * 100
        trades.append({
            'entry': str(entry_date.date()),
            'exit': str(idx[-1].date()) + '*',
            'days': (idx[-1] - entry_date).days,
            'pnl_pct': round(pnl, 2),
        })

    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    bh_ret = ret.loc[strat_ret.index].dropna()
    bh_cum = (1 + bh_ret).cumprod()
    bh_total = bh_cum.iloc[-1] - 1
    bh_cagr = (1 + bh_total) ** (1 / max(years, 0.01)) - 1
    bh_mdd = ((bh_cum - bh_cum.cummax()) / bh_cum.cummax()).min()

    metrics = {
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'mdd': round(mdd * 100, 2),
        'calmar': round(calmar, 3),
        'pct_in_market': round(sig.mean() * 100, 1),
        'n_trades': len(trades),
        'win_rate': round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        'avg_win': round(np.mean(wins), 2) if wins else 0,
        'avg_loss': round(np.mean(losses), 2) if losses else 0,
        'profit_factor': round(abs(sum(wins) / sum(losses)), 2) if losses and sum(losses) != 0 else 999,
        'avg_days': round(np.mean([t['days'] for t in trades]), 1) if trades else 0,
        'bh_cagr': round(bh_cagr * 100, 2),
        'bh_mdd': round(bh_mdd * 100, 2),
        'years': round(years, 1),
    }

    return metrics, trades


# ============================================================
# DAILY SIGNAL CHECK - 3 FACTOR
# ============================================================

def check_signals(use_3factor=True):
    """Check today's signals for all indices"""
    mode_label = "3-Factor (2of3)" if use_3factor else "Classic 2-Factor"

    print("=" * 65)
    print(f"  Strategy P: Panic Buy Signal - {mode_label}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print()

    if use_3factor:
        mode_str = "ALL3" if CONFIG['require_all_3'] else "2of3"
        print("  3-Factor Rules:")
        print(f"    F1 (Value): ERP percentile > {CONFIG['erp_threshold']*100:.0f}% (price historically cheap)")
        print(f"    F2 (Panic): Vol percentile > P{CONFIG['vol_threshold_3f']} AND Price < MA{CONFIG['ma_period']}")
        print(f"    F3 (Volume): 5d/20d volume ratio > {CONFIG['volume_ratio']}")
        print(f"    Mode: {mode_str} (need {'all 3' if CONFIG['require_all_3'] else 'any 2'} factors)")
        print(f"    EXIT: Price > MA{CONFIG['ma_period']}")
    else:
        print("  Classic Rules:")
        print(f"    ENTRY: Vol({CONFIG['vol_window']}d) percentile > {CONFIG['vol_threshold']}%")
        print(f"           AND Price < MA{CONFIG['ma_period']}")
        print(f"    EXIT:  Price > MA{CONFIG['ma_period']}")
    print()

    bs.login()
    results = []

    for name, info in CONFIG['indices'].items():
        df = fetch_index_data(info['bs_code'])
        if df is None or len(df) < CONFIG['vol_lookback']:
            print(f"  {name}: data error")
            continue

        if use_3factor:
            ind = compute_3factor_indicators(df)
            signal = generate_3factor_signal(ind)

            latest = ind.iloc[-1]
            prev_signal = signal.iloc[-2] if len(signal) > 1 else 0
            curr_signal = signal.iloc[-1]

            # Factor status
            erp_val = latest['erp_pct'] * 100 if not pd.isna(latest['erp_pct']) else 0
            vol_pct_val = latest['vol_pct'] * 100 if not pd.isna(latest['vol_pct']) else 0
            dist_val = latest['dist_ma'] if not pd.isna(latest['dist_ma']) else 0
            vol_r_val = latest['vol_ratio'] if not pd.isna(latest['vol_ratio']) else 0
            rsi_val = latest['rsi'] if not pd.isna(latest['rsi']) else 50

            f1_ok = erp_val > CONFIG['erp_threshold'] * 100
            f2_ok = (vol_pct_val > CONFIG['vol_threshold_3f']) and (latest['close'] < latest['ma'] if not pd.isna(latest['ma']) else False)
            f3_ok = vol_r_val > CONFIG['volume_ratio']
            factor_count = int(f1_ok) + int(f2_ok) + int(f3_ok)

            # Determine status
            if curr_signal > 0 and prev_signal == 0:
                status = ">>> NEW BUY SIGNAL <<<"
                action = "BUY"
            elif curr_signal > 0:
                status = "HOLDING (in position)"
                action = "HOLD"
            elif prev_signal > 0 and curr_signal == 0:
                status = ">>> EXIT SIGNAL <<<"
                action = "SELL"
            else:
                status = "WATCHING (no signal)"
                action = "CASH"

            f1_icon = "Y" if f1_ok else "N"
            f2_icon = "Y" if f2_ok else "N"
            f3_icon = "Y" if f3_ok else "N"

            print(f"  {name:8s} | {latest['close']:8.1f} | MA{CONFIG['ma_period']}={latest['ma']:8.1f} | Dist={dist_val:+6.1f}%")
            print(f"  {'':8s} | F1(Value)={f1_icon} ERP={erp_val:4.0f}%  F2(Panic)={f2_icon} VolP={vol_pct_val:4.0f}%  F3(Vol)={f3_icon} VR={vol_r_val:.2f}")
            print(f"  {'':8s} | Factors: {factor_count}/3  RSI={rsi_val:4.1f}  ETF: {info['etf']}")
            print(f"  {'':8s} | Signal: {status}")
            print()

            results.append({
                'index': name,
                'price': round(latest['close'], 2),
                'ma': round(latest['ma'], 2) if not pd.isna(latest['ma']) else None,
                'dist_ma': round(dist_val, 2),
                'vol_pct': round(vol_pct_val, 1),
                'erp_pct': round(erp_val, 1),
                'vol_ratio': round(vol_r_val, 2),
                'f1': f1_ok, 'f2': f2_ok, 'f3': f3_ok,
                'factor_count': factor_count,
                'rsi': round(rsi_val, 1),
                'action': action,
                'etf': info['etf'],
            })

        else:
            # Classic 2-factor mode
            ind = compute_indicators_classic(df['close'])
            signal = generate_classic_signal(ind)

            latest = ind.iloc[-1]
            prev_signal = signal.iloc[-2] if len(signal) > 1 else 0
            curr_signal = signal.iloc[-1]

            vol_pct_val = latest['vol_pct'] * 100 if not pd.isna(latest['vol_pct']) else 0
            dist_val = latest['dist_ma'] if not pd.isna(latest['dist_ma']) else 0
            rsi_val = latest['rsi'] if not pd.isna(latest['rsi']) else 50

            if curr_signal > 0 and prev_signal == 0:
                status = ">>> NEW BUY SIGNAL <<<"
                action = "BUY"
            elif curr_signal > 0:
                status = "HOLDING (in position)"
                action = "HOLD"
            elif prev_signal > 0 and curr_signal == 0:
                status = ">>> EXIT SIGNAL <<<"
                action = "SELL"
            else:
                status = "WATCHING (no signal)"
                action = "CASH"

            vol_ok = vol_pct_val > CONFIG['vol_threshold']
            price_ok = latest['close'] < latest['ma'] if not pd.isna(latest['ma']) else False
            proximity = ""
            if not vol_ok and not price_ok:
                proximity = f"(vol needs +{CONFIG['vol_threshold'] - vol_pct_val:.0f}%, price needs {dist_val:.1f}% drop)"
            elif not vol_ok:
                proximity = f"(vol needs +{CONFIG['vol_threshold'] - vol_pct_val:.0f}%)"
            elif not price_ok:
                proximity = f"(price needs {dist_val:.1f}% drop to MA)"

            print(f"  {name:8s} | {latest['close']:8.1f} | MA{CONFIG['ma_period']}={latest['ma']:8.1f} | Dist={dist_val:+6.1f}%")
            print(f"  {'':8s} | Vol={latest['vol']:5.1f}% | VolPct={vol_pct_val:4.0f}% | RSI={rsi_val:4.1f}")
            print(f"  {'':8s} | ETF: {info['etf']}")
            print(f"  {'':8s} | Signal: {status} {proximity}")
            print()

            results.append({
                'index': name,
                'price': round(latest['close'], 2),
                'ma': round(latest['ma'], 2) if not pd.isna(latest['ma']) else None,
                'dist_ma': round(dist_val, 2),
                'vol_pct': round(vol_pct_val, 1),
                'rsi': round(rsi_val, 1),
                'action': action,
                'etf': info['etf'],
            })

    bs.logout()

    # Summary
    buys = [r for r in results if r['action'] in ('BUY', 'HOLD')]
    sells = [r for r in results if r['action'] == 'SELL']

    print("=" * 65)
    if buys:
        print("  ACTIVE SIGNALS:")
        for r in buys:
            print(f"    {r['action']:4s} {r['index']} via ETF {r['etf']}")
    elif sells:
        print("  EXIT SIGNALS:")
        for r in sells:
            print(f"    SELL {r['index']}")
    else:
        print("  No active signals. All indices in CASH mode.")
        if use_3factor:
            # Show factor proximity
            for r in results:
                if r.get('factor_count', 0) >= 1:
                    factors = []
                    if r.get('f1'): factors.append('Value')
                    if r.get('f2'): factors.append('Panic')
                    if r.get('f3'): factors.append('Volume')
                    print(f"    {r['index']}: {r['factor_count']}/3 factors active ({', '.join(factors)})")
        else:
            closest = min(results, key=lambda r: abs(r['dist_ma']) if r['dist_ma'] < 0 else 999)
            if closest['dist_ma'] < 0:
                print(f"    Closest: {closest['index']} (dist MA: {closest['dist_ma']:.1f}%, vol pct: {closest['vol_pct']:.0f}%)")
    print("=" * 65)

    return results


def run_backtest():
    """Full backtest with IS/OOS split - runs both modes"""
    print("=" * 65)
    print("  Strategy P: Panic Buy Signal - BACKTEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    bs.login()
    split = pd.Timestamp('2023-01-01')

    for name, info in CONFIG['indices'].items():
        df = fetch_index_data(info['bs_code'])
        if df is None or len(df) < 1000:
            continue

        print(f"\n{'=' * 60}")
        print(f"  {name} ({info['bs_code']})")
        print(f"{'=' * 60}")

        # --- 3-Factor ---
        print(f"\n  [3-Factor Mode: 2of3]")
        ind3 = compute_3factor_indicators(df)
        signal3 = generate_3factor_signal(ind3)

        for period_name, start, end in [
            ('FULL', df.index[0], df.index[-1]),
            ('OOS (2023-now)', split, df.index[-1]),
        ]:
            c = df['close'][(df.index >= start) & (df.index < end)] if period_name != 'FULL' else df['close']
            s = signal3.loc[c.index]
            metrics, trades = backtest(c, s)
            if metrics:
                print(f"\n    [{period_name}] {metrics['years']}y")
                print(f"      CAGR: {metrics['cagr']:+.1f}%  Sharpe: {metrics['sharpe']:.2f}  MDD: {metrics['mdd']:.1f}%")
                print(f"      Trades: {metrics['n_trades']}  WinRate: {metrics['win_rate']:.0f}%  AvgHold: {metrics['avg_days']:.0f}d  PF: {metrics['profit_factor']:.1f}x")
                if period_name.startswith('OOS') and trades:
                    for t in trades:
                        m = '+' if t['pnl_pct'] > 0 else ''
                        print(f"        {t['entry']} -> {t['exit']}  {t['days']:3d}d  {m}{t['pnl_pct']:.1f}%")

        # --- Classic 2-Factor ---
        print(f"\n  [Classic 2-Factor]")
        ind2 = compute_indicators_classic(df['close'])
        signal2 = generate_classic_signal(ind2)

        for period_name, start, end in [
            ('FULL', df.index[0], df.index[-1]),
            ('OOS (2023-now)', split, df.index[-1]),
        ]:
            c = df['close'][(df.index >= start) & (df.index < end)] if period_name != 'FULL' else df['close']
            s = signal2.loc[c.index]
            metrics, trades = backtest(c, s)
            if metrics:
                print(f"\n    [{period_name}] {metrics['years']}y")
                print(f"      CAGR: {metrics['cagr']:+.1f}%  Sharpe: {metrics['sharpe']:.2f}  MDD: {metrics['mdd']:.1f}%")
                print(f"      Trades: {metrics['n_trades']}  WinRate: {metrics['win_rate']:.0f}%  AvgHold: {metrics['avg_days']:.0f}d  PF: {metrics['profit_factor']:.1f}x")

    bs.logout()

    print(f"\n{'=' * 65}")
    print("  3-Factor parameters:")
    print(f"    ERP threshold: {CONFIG['erp_threshold']} | Vol threshold: P{CONFIG['vol_threshold_3f']}")
    print(f"    MA period: {CONFIG['ma_period']}d | Volume ratio: {CONFIG['volume_ratio']}")
    print(f"    Mode: {'ALL3' if CONFIG['require_all_3'] else '2of3'} | Exit: price > MA{CONFIG['ma_period']}")
    print(f"{'=' * 65}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    if '--backtest' in sys.argv:
        run_backtest()
    elif '--classic' in sys.argv:
        check_signals(use_3factor=False)
    else:
        check_signals(use_3factor=True)
