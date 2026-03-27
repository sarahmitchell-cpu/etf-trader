#!/usr/bin/env python3
"""
Strategy P Upgrade Research: 3-Factor Super Panic Signal
=========================================================

Research script to test a 3-factor "super signal" for panic bottom-fishing
with target 90%+ win rate.

Three Factors:
  1. ERP (Equity Risk Premium / 股债性价比) - Valuation factor
     = 1/PE(沪深300) - 10Y Treasury Yield
     Signal: ERP percentile > threshold (stocks cheap vs bonds)

  2. Short-term Price Decline - Price factor (existing Strategy P logic)
     = Volatility percentile > threshold AND price < MA

  3. Volume Anomaly - Sentiment factor
     = Abnormally high volume (capitulation selling)
     Volume ratio vs 20d avg > threshold

Super Signal = Factor1 AND Factor2 AND Factor3

Author: Sarah Mitchell / VisionClaw
Date: 2026-03-27
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import baostock as bs
from datetime import datetime, timedelta
import sys

# ============================================================
# DATA FETCHING
# ============================================================

def fetch_index_data(bs_code, start_date='2010-01-01'):
    """Fetch daily OHLCV from baostock"""
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


def fetch_bond_yield():
    """
    Fetch 10Y China Government Bond yield proxy.
    We use the CSI Bond Index or approximate with a fixed schedule.
    Since baostock doesn't have bond yields directly, we'll use
    a synthetic approach based on available data.
    """
    # Try to get 10Y bond yield from a local CSV or generate synthetic
    # For research, we'll use the inverse relationship with bond ETF prices
    # 511010 (国债ETF) as proxy - its price moves inversely with yields

    # Alternative: Use PE-based ERP directly with historical PE data
    return None


def fetch_index_pe(bs_code, start_date='2010-01-01'):
    """Fetch PE ratio history for an index using baostock"""
    # baostock provides PE for some indices
    rs = bs.query_history_k_data_plus(
        bs_code, "date,close,peTTM",
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
    for c in ['close', 'peTTM']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.set_index('date').sort_index()
    return df


# ============================================================
# FACTOR CALCULATIONS
# ============================================================

def calc_factor1_erp(close, pe=None, bond_yield_pct=3.0, lookback=504):
    """
    Factor 1: Equity Risk Premium (股债性价比)

    ERP = Earnings Yield - Bond Yield = 1/PE - BondYield
    When ERP is high -> stocks are cheap relative to bonds -> bullish

    Since we may not have real-time bond yield data, we use two approaches:
    A) If PE available: ERP percentile based on 1/PE ranking
    B) Fallback: Use close price distance from long-term trend as value proxy

    Signal: ERP percentile > threshold (e.g., top 20% = stocks very cheap)
    """
    if pe is not None and len(pe.dropna()) > 252:
        # Use actual PE data
        earnings_yield = 1.0 / pe  # Earnings yield
        # Approximate bond yield as constant or slowly changing
        # For better accuracy, this should use actual bond yield data
        erp = earnings_yield - bond_yield_pct / 100.0

        # Rolling percentile of ERP
        erp_pct = erp.rolling(lookback, min_periods=252).apply(
            lambda x: (x.iloc[-1] >= x).mean() if len(x) > 1 else 0.5
        )
        return erp, erp_pct
    else:
        # Fallback: Use price deviation from 3-year mean as value proxy
        # Lower price vs history = higher value = higher "ERP"
        long_ma = close.rolling(756, min_periods=504).mean()  # 3-year MA
        value_score = (long_ma - close) / long_ma  # Higher = more undervalued

        value_pct = value_score.rolling(lookback, min_periods=252).apply(
            lambda x: (x.iloc[-1] >= x).mean() if len(x) > 1 else 0.5
        )
        return value_score, value_pct


def calc_factor2_price_decline(close, vol_window=20, vol_lookback=504,
                                vol_threshold=75, ma_period=120):
    """
    Factor 2: Short-term Price Decline (existing Strategy P logic)

    Conditions:
    - Realized volatility percentile > threshold (panic selling)
    - Price below MA (downtrend/oversold)
    """
    ret = close.pct_change()
    vol = ret.rolling(vol_window).std() * np.sqrt(252) * 100

    vol_pct = vol.rolling(vol_lookback, min_periods=min(252, vol_lookback)).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    )

    ma = close.rolling(ma_period, min_periods=ma_period).mean()
    below_ma = close < ma
    high_vol = vol_pct > (vol_threshold / 100.0)

    factor2_signal = below_ma & high_vol

    return vol, vol_pct, ma, factor2_signal


def calc_factor3_volume(volume, short_window=5, long_window=20,
                         threshold=2.0, lookback=504):
    """
    Factor 3: Volume Anomaly (成交量异常 - 恐慌放量)

    Several volume signals:
    A) Volume ratio: recent volume vs longer-term average
    B) Volume percentile: current volume rank in history

    High volume during decline = capitulation = bullish contrarian

    Signal: Volume ratio > threshold (e.g., 2x = double normal volume)
    """
    vol_short = volume.rolling(short_window).mean()
    vol_long = volume.rolling(long_window).mean()
    vol_ratio = vol_short / vol_long

    # Volume percentile (rolling)
    vol_pct = volume.rolling(lookback, min_periods=252).apply(
        lambda x: (x.iloc[-1] >= x).mean() if len(x) > 1 else 0.5
    )

    # Binary signal
    high_volume = vol_ratio > threshold

    return vol_ratio, vol_pct, high_volume


# ============================================================
# COMBINED SIGNAL & BACKTEST
# ============================================================

def generate_super_signal(close, volume, pe=None,
                          erp_threshold=0.7,
                          vol_window=20, vol_lookback=504,
                          vol_threshold=75, ma_period=120,
                          volume_ratio_threshold=1.5,
                          exit_mode='above_ma',
                          require_all_3=True):
    """
    Generate the 3-factor super signal.

    Parameters:
    - erp_threshold: ERP percentile threshold (0.7 = top 30%)
    - vol_threshold: Volatility percentile threshold (75 = top 25%)
    - ma_period: MA period for price factor
    - volume_ratio_threshold: Volume ratio threshold (1.5 = 50% above avg)
    - require_all_3: If True, all 3 factors must fire. If False, 2 of 3.
    """
    # Factor 1: ERP / Value
    erp, erp_pct = calc_factor1_erp(close, pe)

    # Factor 2: Price decline (vol + below MA)
    vol, vol_pct_series, ma, f2_signal = calc_factor2_price_decline(
        close, vol_window, vol_lookback, vol_threshold, ma_period
    )

    # Factor 3: Volume anomaly
    vol_ratio, vol_volume_pct, f3_signal = calc_factor3_volume(
        volume, threshold=volume_ratio_threshold
    )

    # Factor 1 signal
    f1_signal = erp_pct > erp_threshold

    # Combine
    if require_all_3:
        entry_signal = f1_signal & f2_signal & f3_signal
    else:
        # 2 of 3
        score = f1_signal.astype(int) + f2_signal.astype(int) + f3_signal.astype(int)
        entry_signal = score >= 2

    # State machine (same exit logic as Strategy P)
    signal = pd.Series(0.0, index=close.index)
    in_position = False

    for i in range(1, len(close)):
        if pd.isna(ma.iloc[i]):
            signal.iloc[i] = 1.0 if in_position else 0.0
            continue

        entry = bool(entry_signal.iloc[i]) if not pd.isna(entry_signal.iloc[i]) else False

        if exit_mode == 'above_ma':
            exit_cond = close.iloc[i] > ma.iloc[i]
        else:
            exit_cond = close.iloc[i] > ma.iloc[i]

        if not in_position and entry:
            in_position = True
        elif in_position and exit_cond:
            in_position = False

        signal.iloc[i] = 1.0 if in_position else 0.0

    # Return detailed info
    factors_df = pd.DataFrame({
        'close': close,
        'ma': ma,
        'vol': vol,
        'vol_pct': vol_pct_series,
        'erp_pct': erp_pct,
        'vol_ratio': vol_ratio,
        'f1_erp': f1_signal.astype(int),
        'f2_price': f2_signal.astype(int),
        'f3_volume': f3_signal.astype(int),
        'entry': entry_signal.astype(int),
        'signal': signal,
    })

    return signal, factors_df


def backtest(close, signal, cost_bps=10):
    """Run backtest and return metrics + trade list"""
    sig = signal.shift(1).fillna(0)
    ret = close.pct_change()
    cost = cost_bps / 10000
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

    # Trade extraction
    trades = []
    in_trade = False
    entry_date = entry_price = None
    sig_arr = sig.values
    close_arr = close.loc[sig.index].values
    idx = sig.index

    for i in range(1, len(sig_arr)):
        if not in_trade and sig_arr[i] > 0 and sig_arr[i-1] == 0:
            in_trade = True
            entry_date = idx[i]
            entry_price = close_arr[i]
        elif in_trade and sig_arr[i] == 0 and sig_arr[i-1] > 0:
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
    }

    return metrics, trades


# ============================================================
# PARAMETER GRID SEARCH
# ============================================================

def grid_search(close, volume, pe=None):
    """
    Test multiple parameter combinations to find optimal 3-factor setup.
    Focus on maximizing win rate while maintaining reasonable trade count.
    """
    results = []

    # Parameter grid
    erp_thresholds = [0.6, 0.7, 0.8, 0.85, 0.9]
    vol_thresholds = [70, 75, 80, 85]
    ma_periods = [60, 90, 120, 150]
    volume_ratios = [1.2, 1.5, 1.8, 2.0, 2.5]
    require_modes = [True, False]  # all 3 vs 2 of 3

    total = len(erp_thresholds) * len(vol_thresholds) * len(ma_periods) * len(volume_ratios) * len(require_modes)
    print(f"  Testing {total} parameter combinations...")

    count = 0
    for erp_t in erp_thresholds:
        for vol_t in vol_thresholds:
            for ma_p in ma_periods:
                for vol_r in volume_ratios:
                    for req_all in require_modes:
                        count += 1
                        if count % 100 == 0:
                            print(f"    Progress: {count}/{total}")

                        try:
                            signal, _ = generate_super_signal(
                                close, volume, pe,
                                erp_threshold=erp_t,
                                vol_threshold=vol_t,
                                ma_period=ma_p,
                                volume_ratio_threshold=vol_r,
                                require_all_3=req_all,
                            )

                            metrics, trades = backtest(close, signal)

                            if metrics and metrics['n_trades'] >= 3:
                                results.append({
                                    'erp_t': erp_t,
                                    'vol_t': vol_t,
                                    'ma_p': ma_p,
                                    'vol_r': vol_r,
                                    'req_all': req_all,
                                    **metrics,
                                    'trades': trades,
                                })
                        except Exception as e:
                            pass

    return results


# ============================================================
# COMPARISON: 2-FACTOR vs 3-FACTOR
# ============================================================

def compare_strategies(close, volume, pe=None):
    """Compare original 2-factor vs various 3-factor configurations"""
    from panic_buy_signal import compute_indicators, generate_signal as gen_orig, backtest as bt_orig

    print("\n" + "=" * 70)
    print("  COMPARISON: Original 2-Factor vs 3-Factor Super Signal")
    print("=" * 70)

    # Original Strategy P
    ind = compute_indicators(close)
    orig_signal = gen_orig(ind)
    orig_metrics, orig_trades = bt_orig(close, orig_signal)

    if orig_metrics:
        print(f"\n  [Original 2-Factor] Vol P75 + Below MA120")
        print(f"    Trades: {orig_metrics['n_trades']}  WinRate: {orig_metrics['win_rate']:.0f}%  "
              f"Sharpe: {orig_metrics['sharpe']:.2f}  CAGR: {orig_metrics['cagr']:+.1f}%")

    # Test key 3-factor configs
    configs = [
        {'erp_t': 0.7, 'vol_t': 75, 'ma_p': 120, 'vol_r': 1.5, 'req': True, 'name': '3F Conservative'},
        {'erp_t': 0.8, 'vol_t': 75, 'ma_p': 120, 'vol_r': 1.5, 'req': True, 'name': '3F Moderate'},
        {'erp_t': 0.8, 'vol_t': 80, 'ma_p': 120, 'vol_r': 2.0, 'req': True, 'name': '3F Aggressive'},
        {'erp_t': 0.85, 'vol_t': 80, 'ma_p': 120, 'vol_r': 1.5, 'req': True, 'name': '3F Ultra'},
        {'erp_t': 0.7, 'vol_t': 75, 'ma_p': 120, 'vol_r': 1.5, 'req': False, 'name': '2of3 Conservative'},
        {'erp_t': 0.8, 'vol_t': 80, 'ma_p': 120, 'vol_r': 2.0, 'req': False, 'name': '2of3 Aggressive'},
    ]

    for cfg in configs:
        signal, factors = generate_super_signal(
            close, volume, pe,
            erp_threshold=cfg['erp_t'],
            vol_threshold=cfg['vol_t'],
            ma_period=cfg['ma_p'],
            volume_ratio_threshold=cfg['vol_r'],
            require_all_3=cfg['req'],
        )
        metrics, trades = backtest(close, signal)

        if metrics:
            mode = "ALL3" if cfg['req'] else "2of3"
            print(f"\n  [{cfg['name']}] ERP>{cfg['erp_t']} Vol>P{cfg['vol_t']} MA{cfg['ma_p']} VolR>{cfg['vol_r']} ({mode})")
            print(f"    Trades: {metrics['n_trades']}  WinRate: {metrics['win_rate']:.0f}%  "
                  f"Sharpe: {metrics['sharpe']:.2f}  CAGR: {metrics['cagr']:+.1f}%  MDD: {metrics['mdd']:.1f}%")
            if trades:
                print(f"    Trade Log:")
                for t in trades:
                    marker = '+' if t['pnl_pct'] > 0 else '-'
                    print(f"      {t['entry']} -> {t['exit']}  {t['days']:3d}d  {marker}{abs(t['pnl_pct']):.1f}%")
        else:
            print(f"\n  [{cfg['name']}] No trades or insufficient data")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  Strategy P Upgrade: 3-Factor Super Panic Signal Research")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    print("  Three Factors:")
    print("    1. ERP (股债性价比) - Valuation: stocks cheap vs bonds")
    print("    2. Price Decline (短期跌幅) - Vol P75 + Below MA")
    print("    3. Volume Anomaly (放量) - Capitulation selling")
    print("    Target: 90%+ win rate super signal")
    print()

    bs.login()

    # Test on key indices
    indices = {
        '沪深300': 'sh.000300',
        '中证500': 'sh.000905',
        '中证1000': 'sh.000852',
        '创业板指': 'sz.399006',
    }

    split = pd.Timestamp('2023-01-01')

    all_results = {}

    for name, code in indices.items():
        print(f"\n{'=' * 70}")
        print(f"  {name} ({code})")
        print(f"{'=' * 70}")

        df = fetch_index_data(code, start_date='2010-01-01')
        if df is None or len(df) < 1000:
            print(f"  Data error for {name}")
            continue

        close = df['close']
        volume = df['volume']

        # Try to get PE data
        pe = None
        try:
            pe_df = fetch_index_pe(code)
            if pe_df is not None and 'peTTM' in pe_df.columns:
                pe = pe_df['peTTM'].reindex(close.index).interpolate()
                valid_pe = pe.dropna()
                if len(valid_pe) > 252:
                    print(f"  PE data available: {len(valid_pe)} days, range {valid_pe.min():.1f}-{valid_pe.max():.1f}")
                else:
                    pe = None
                    print(f"  PE data insufficient, using price-based value proxy")
        except:
            pe = None
            print(f"  PE data not available, using price-based value proxy")

        # ---- FULL PERIOD Analysis ----
        print(f"\n  --- FULL PERIOD ({close.index[0].date()} to {close.index[-1].date()}) ---")

        # Grid search on full period first to find best params
        print(f"\n  Grid Search (full period):")
        results = grid_search(close, volume, pe)

        if results:
            # Sort by win rate (primary) then by number of trades (secondary)
            results.sort(key=lambda x: (x['win_rate'], x['n_trades']), reverse=True)

            print(f"\n  Top 10 by Win Rate (min 3 trades):")
            print(f"  {'ERP':>5s} {'Vol':>4s} {'MA':>4s} {'VolR':>5s} {'Mode':>5s} | {'#Tr':>4s} {'WR%':>5s} {'Sharpe':>7s} {'CAGR':>7s} {'MDD':>6s} {'PF':>5s}")
            print(f"  {'-' * 65}")

            for r in results[:10]:
                mode = "ALL3" if r['req_all'] else "2of3"
                print(f"  {r['erp_t']:5.2f} {r['vol_t']:4d} {r['ma_p']:4d} {r['vol_r']:5.1f} {mode:>5s} | "
                      f"{r['n_trades']:4d} {r['win_rate']:5.0f} {r['sharpe']:7.2f} {r['cagr']:+6.1f}% {r['mdd']:5.1f}% {r['profit_factor']:5.1f}")

            # Show trades for the best configuration
            best = results[0]
            print(f"\n  BEST CONFIG: ERP>{best['erp_t']} Vol>P{best['vol_t']} MA{best['ma_p']} VolR>{best['vol_r']} "
                  f"{'ALL3' if best['req_all'] else '2of3'}")
            print(f"  Win Rate: {best['win_rate']:.0f}%  Trades: {best['n_trades']}  "
                  f"Sharpe: {best['sharpe']:.2f}  CAGR: {best['cagr']:+.1f}%")

            if best['trades']:
                print(f"\n  Trade Log:")
                print(f"  {'Entry':12s} {'Exit':14s} {'Days':>5s} {'PnL':>8s}")
                print(f"  {'-' * 45}")
                for t in best['trades']:
                    marker = '+' if t['pnl_pct'] > 0 else '-'
                    print(f"  {t['entry']:12s} {t['exit']:14s} {t['days']:5d} {marker}{abs(t['pnl_pct']):7.2f}%")

            all_results[name] = results

        # ---- OOS Analysis with best params ----
        if results:
            best = results[0]
            print(f"\n  --- OOS VALIDATION (2023-now) ---")

            oos_close = close[close.index >= split]
            oos_volume = volume[volume.index >= split]
            oos_pe = pe[pe.index >= split] if pe is not None else None

            if len(oos_close) > 60:
                # Need full history for indicators but only measure OOS trades
                signal, factors = generate_super_signal(
                    close, volume, pe,
                    erp_threshold=best['erp_t'],
                    vol_threshold=best['vol_t'],
                    ma_period=best['ma_p'],
                    volume_ratio_threshold=best['vol_r'],
                    require_all_3=best['req_all'],
                )

                # OOS only
                oos_signal = signal[signal.index >= split]
                oos_close_bt = close[close.index >= split]

                oos_metrics, oos_trades = backtest(oos_close_bt, oos_signal)

                if oos_metrics:
                    print(f"  OOS Trades: {oos_metrics['n_trades']}  WinRate: {oos_metrics['win_rate']:.0f}%  "
                          f"Sharpe: {oos_metrics['sharpe']:.2f}  CAGR: {oos_metrics['cagr']:+.1f}%")
                    if oos_trades:
                        for t in oos_trades:
                            marker = '+' if t['pnl_pct'] > 0 else '-'
                            print(f"    {t['entry']} -> {t['exit']}  {t['days']:3d}d  {marker}{abs(t['pnl_pct']):.1f}%")
                else:
                    print(f"  No OOS trades (signal too rare in this period)")

    bs.logout()

    # ---- SUMMARY ----
    print(f"\n\n{'=' * 70}")
    print("  RESEARCH SUMMARY")
    print(f"{'=' * 70}")

    for name, results in all_results.items():
        if results:
            best = results[0]
            high_wr = [r for r in results if r['win_rate'] >= 90]
            print(f"\n  {name}:")
            print(f"    Best WR: {best['win_rate']:.0f}% ({best['n_trades']} trades)")
            print(f"    Configs with WR>=90%: {len(high_wr)}")
            if high_wr:
                # Among 90%+ WR configs, find best by trade count
                high_wr.sort(key=lambda x: x['n_trades'], reverse=True)
                top = high_wr[0]
                print(f"    Best 90%+ config (most trades): ERP>{top['erp_t']} Vol>P{top['vol_t']} "
                      f"MA{top['ma_p']} VolR>{top['vol_r']} {'ALL3' if top['req_all'] else '2of3'}")
                print(f"      Trades: {top['n_trades']}  WR: {top['win_rate']:.0f}%  Sharpe: {top['sharpe']:.2f}")

    print(f"\n{'=' * 70}")


if __name__ == '__main__':
    main()
