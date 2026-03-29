"""
300价值指数(000919)择时策略全面研究
测试多种择时机制: MA趋势、双均线、波动率缩放、动量、RSI、均值回归、布林带等
"""
from __future__ import annotations

import json
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ── data ──────────────────────────────────────────────────────────────
def fetch_index(code='000919', name='300价值'):
    import akshare as ak
    df = ak.index_zh_a_hist(symbol=code, period='daily',
                            start_date='20050101', end_date='20260401')
    df = df.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open',
                            '最高': 'high', '最低': 'low', '成交量': 'volume'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  {name}({code}): {len(df)} rows, {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]


def fetch_csi_tr(code='H00919', name='300价值TR'):
    """Try CSI total return index"""
    try:
        import akshare as ak
        df = ak.index_zh_a_hist(symbol=code.replace('H', ''), period='daily',
                                start_date='20050101', end_date='20260401')
        if df is not None and len(df) > 100:
            df = df.rename(columns={'日期': 'date', '收盘': 'close'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            return df[['date', 'close']]
    except Exception:
        pass
    return None


# ── helpers ───────────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def backtest(df, signal_col='signal', txn_cost_bps=8):
    """Run backtest given a signal column (0~1 position size)"""
    d = df.copy()
    d['ret'] = d['close'].pct_change()
    d['signal'] = d[signal_col].fillna(0)
    d['trades'] = d['signal'].diff().abs()
    cost = txn_cost_bps / 10000
    d['strat_ret'] = d['signal'] * d['ret'] - d['trades'] * cost
    d['cum_ret'] = (1 + d['strat_ret']).cumprod()
    d['cum_bh'] = (1 + d['ret']).cumprod()

    valid = d.dropna(subset=['strat_ret'])
    if len(valid) < 252:
        return None

    years = len(valid) / 252
    total_ret = valid['cum_ret'].iloc[-1]
    cagr = total_ret ** (1 / years) - 1
    ann_vol = valid['strat_ret'].std() * np.sqrt(252)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0
    dd = valid['cum_ret'] / valid['cum_ret'].cummax() - 1
    max_dd = dd.min()
    n_trades = int(valid['trades'].gt(0.01).sum())

    # Buy & hold
    bh_total = valid['cum_bh'].iloc[-1]
    bh_cagr = bh_total ** (1 / years) - 1

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate (monthly)
    d_m = valid.set_index('date')['strat_ret'].resample('M').sum()
    win_rate = (d_m > 0).mean()

    return {
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'max_dd': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'ann_vol': round(ann_vol * 100, 2),
        'win_rate_m': round(win_rate * 100, 1),
        'trades': n_trades,
        'bh_cagr': round(bh_cagr * 100, 2),
        'years': round(years, 1),
    }


# ── strategies ────────────────────────────────────────────────────────

def strategy_pure_ma(df, ma_period):
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['signal'] = (d['close'] > d['ma']).astype(float).shift(1)
    return backtest(d)


def strategy_dual_ma(df, fast, slow):
    d = df.copy()
    d['ma_fast'] = d['close'].rolling(fast).mean()
    d['ma_slow'] = d['close'].rolling(slow).mean()
    d['signal'] = (d['ma_fast'] > d['ma_slow']).astype(float).shift(1)
    return backtest(d)


def strategy_ma_vol_scaling(df, ma_period, vol_window=20, vol_target=0.15):
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['ret'] = d['close'].pct_change()
    d['realized_vol'] = d['ret'].rolling(vol_window).std() * np.sqrt(252)
    d['trend'] = (d['close'] > d['ma']).astype(float)
    d['vol_scale'] = (vol_target / d['realized_vol']).clip(0, 1)
    d['signal'] = (d['trend'] * d['vol_scale']).shift(1)
    return backtest(d)


def strategy_ma_trailing_stop(df, ma_period, stop_pct=0.08):
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['ret'] = d['close'].pct_change()

    pos = 0.0
    peak = 0.0
    signals = []
    for i in range(len(d)):
        if pd.isna(d['ma'].iloc[i]):
            signals.append(0.0)
            continue
        close = d['close'].iloc[i]
        ma = d['ma'].iloc[i]
        if pos == 0:
            if close > ma:
                pos = 1.0
                peak = close
        else:
            peak = max(peak, close)
            if close < peak * (1 - stop_pct) or close < ma:
                pos = 0.0
                peak = 0.0
        signals.append(pos)
    d['signal'] = pd.Series(signals, index=d.index).shift(1)
    return backtest(d)


def strategy_ma_rsi(df, ma_period, rsi_period=14, rsi_low=30, rsi_high=70):
    """MA trend + RSI filter: reduce position at RSI extremes"""
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['rsi'] = calc_rsi(d['close'], rsi_period)
    d['trend'] = (d['close'] > d['ma']).astype(float)
    # When RSI > high → reduce to 50%; when RSI < low & trend down → stay out
    d['rsi_scale'] = 1.0
    d.loc[d['rsi'] > rsi_high, 'rsi_scale'] = 0.5
    d['signal'] = (d['trend'] * d['rsi_scale']).shift(1)
    return backtest(d)


def strategy_bollinger_mean_reversion(df, bb_period=20, bb_std=2.0, ma_filter=None):
    """Bollinger band mean reversion: buy at lower band, sell at upper band"""
    d = df.copy()
    d['bb_mid'] = d['close'].rolling(bb_period).mean()
    d['bb_std'] = d['close'].rolling(bb_period).std()
    d['bb_upper'] = d['bb_mid'] + bb_std * d['bb_std']
    d['bb_lower'] = d['bb_mid'] - bb_std * d['bb_std']

    pos = 0.0
    signals = []
    for i in range(len(d)):
        if pd.isna(d['bb_mid'].iloc[i]):
            signals.append(0.0)
            continue
        close = d['close'].iloc[i]
        if close <= d['bb_lower'].iloc[i]:
            pos = 1.0
        elif close >= d['bb_upper'].iloc[i]:
            pos = 0.0
        signals.append(pos)
    d['signal'] = pd.Series(signals, index=d.index).shift(1)

    if ma_filter:
        d['ma_filt'] = d['close'].rolling(ma_filter).mean()
        d.loc[d['close'] < d['ma_filt'], 'signal'] = 0.0

    return backtest(d)


def strategy_momentum(df, lookback=20, hold=20):
    """Momentum: go long when past N-day return > 0"""
    d = df.copy()
    d['mom'] = d['close'].pct_change(lookback)
    d['signal'] = (d['mom'] > 0).astype(float).shift(1)
    return backtest(d)


def strategy_ma_slope(df, ma_period=60, slope_window=5):
    """Only long when MA is rising (positive slope)"""
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['ma_slope'] = d['ma'].pct_change(slope_window)
    d['trend'] = (d['close'] > d['ma']).astype(float)
    d['slope_up'] = (d['ma_slope'] > 0).astype(float)
    d['signal'] = (d['trend'] * d['slope_up']).shift(1)
    return backtest(d)


def strategy_volume_breakout(df, ma_period=60, vol_mult=1.5):
    """MA trend + volume confirmation: need above-average volume for entry"""
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['vol_ma'] = d['volume'].rolling(20).mean()
    d['vol_high'] = d['volume'] > d['vol_ma'] * vol_mult

    pos = 0.0
    signals = []
    for i in range(len(d)):
        if pd.isna(d['ma'].iloc[i]) or pd.isna(d['vol_ma'].iloc[i]):
            signals.append(0.0)
            continue
        close = d['close'].iloc[i]
        ma = d['ma'].iloc[i]
        if pos == 0:
            if close > ma and d['vol_high'].iloc[i]:
                pos = 1.0
        else:
            if close < ma:
                pos = 0.0
        signals.append(pos)
    d['signal'] = pd.Series(signals, index=d.index).shift(1)
    return backtest(d)


def strategy_drawdown_reentry(df, ma_period=60, dd_threshold=-0.10):
    """MA trend + re-enter only after a drawdown from peak"""
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['rolling_max'] = d['close'].rolling(252).max()
    d['dd_from_peak'] = d['close'] / d['rolling_max'] - 1

    d['trend'] = (d['close'] > d['ma']).astype(float)
    # More aggressive: enter only when recovering from DD
    d['was_in_dd'] = (d['dd_from_peak'] < dd_threshold).rolling(60).max()
    d['signal'] = d['trend'].shift(1)
    return backtest(d)


def strategy_regime_filter(df, ma_trend=60, vol_window=60, vol_threshold=0.25):
    """Regime-based: only enter in low-vol regime with trend"""
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_trend).mean()
    d['ret'] = d['close'].pct_change()
    d['realized_vol'] = d['ret'].rolling(vol_window).std() * np.sqrt(252)
    d['trend'] = (d['close'] > d['ma']).astype(float)
    d['low_vol'] = (d['realized_vol'] < vol_threshold).astype(float)
    d['signal'] = (d['trend'] * d['low_vol']).shift(1)
    return backtest(d)


def strategy_ema_crossover(df, fast=12, slow=26):
    """EMA crossover (MACD-style without signal line)"""
    d = df.copy()
    d['ema_fast'] = d['close'].ewm(span=fast).mean()
    d['ema_slow'] = d['close'].ewm(span=slow).mean()
    d['signal'] = (d['ema_fast'] > d['ema_slow']).astype(float).shift(1)
    return backtest(d)


def strategy_macd_signal(df, fast=12, slow=26, sig=9):
    """MACD: long when MACD > signal line"""
    d = df.copy()
    d['ema_fast'] = d['close'].ewm(span=fast).mean()
    d['ema_slow'] = d['close'].ewm(span=slow).mean()
    d['macd'] = d['ema_fast'] - d['ema_slow']
    d['macd_signal'] = d['macd'].ewm(span=sig).mean()
    d['signal'] = (d['macd'] > d['macd_signal']).astype(float).shift(1)
    return backtest(d)


def strategy_adaptive_ma(df, short_ma=20, long_ma=120):
    """Adaptive: use short MA in uptrend, long MA in downtrend"""
    d = df.copy()
    d['ma_short'] = d['close'].rolling(short_ma).mean()
    d['ma_long'] = d['close'].rolling(long_ma).mean()
    d['uptrend'] = d['close'] > d['ma_long']
    # In uptrend, use shorter MA (faster exit); in downtrend, use longer MA (slower entry)
    d['adaptive_ma'] = np.where(d['uptrend'], d['ma_short'], d['ma_long'])
    d['signal'] = (d['close'] > d['adaptive_ma']).astype(float).shift(1)
    return backtest(d)


def strategy_combined_score(df, ma_period=60):
    """Combined scoring: MA + RSI + momentum + vol regime"""
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['rsi'] = calc_rsi(d['close'], 14)
    d['mom_20'] = d['close'].pct_change(20)
    d['ret'] = d['close'].pct_change()
    d['vol'] = d['ret'].rolling(20).std() * np.sqrt(252)

    # Score 0-4
    d['s_trend'] = (d['close'] > d['ma']).astype(float)
    d['s_rsi'] = ((d['rsi'] > 30) & (d['rsi'] < 70)).astype(float)
    d['s_mom'] = (d['mom_20'] > 0).astype(float)
    d['s_vol'] = (d['vol'] < 0.25).astype(float)

    d['score'] = d['s_trend'] + d['s_rsi'] + d['s_mom'] + d['s_vol']
    # Position: 0 if score<=1, 0.5 if score==2, 1.0 if score>=3
    d['pos'] = 0.0
    d.loc[d['score'] == 2, 'pos'] = 0.5
    d.loc[d['score'] >= 3, 'pos'] = 1.0
    d['signal'] = d['pos'].shift(1)
    return backtest(d)


# value-specific strategies
def strategy_pe_timing(df, pe_df=None, ma_period=60):
    """PE-based timing for value index: buy when PE is low (value is cheap)"""
    # We don't have PE data easily, skip or use price-based proxy
    # Use price relative to 3-year range as PE proxy
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['pct_rank'] = d['close'].rolling(756).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    d['trend'] = (d['close'] > d['ma']).astype(float)
    # Buy when trend up AND price not at extreme high (< 80th percentile of 3yr)
    d['not_expensive'] = (d['pct_rank'] < 0.8).astype(float)
    d['signal'] = (d['trend'] * d['not_expensive']).shift(1)
    return backtest(d)


def strategy_mean_reversion_value(df, lookback=60, z_entry=-1.0, z_exit=0.5):
    """Mean reversion: value index tends to revert. Buy when z-score is low."""
    d = df.copy()
    d['ma'] = d['close'].rolling(lookback).mean()
    d['std'] = d['close'].rolling(lookback).std()
    d['zscore'] = (d['close'] - d['ma']) / d['std']

    pos = 0.0
    signals = []
    for i in range(len(d)):
        if pd.isna(d['zscore'].iloc[i]):
            signals.append(0.0)
            continue
        z = d['zscore'].iloc[i]
        if pos == 0 and z <= z_entry:
            pos = 1.0
        elif pos == 1 and z >= z_exit:
            pos = 0.0
        signals.append(pos)
    d['signal'] = pd.Series(signals, index=d.index).shift(1)
    return backtest(d)


# ── main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("300价值指数(000919) 择时策略全面研究")
    print("=" * 70)

    print("\n[1] Fetching data...")
    df = fetch_index('000919', '300价值')

    results = []

    # ── 1. Pure MA ────────────────────────────────────────────────────
    print("\n[2] Testing Pure MA strategies...")
    for ma in [5, 10, 20, 40, 60, 80, 120, 200, 250]:
        r = strategy_pure_ma(df, ma)
        if r:
            r['strategy'] = f'MA{ma}'
            r['type'] = 'pure_ma'
            results.append(r)
            print(f"  MA{ma}: CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── 2. Dual MA ────────────────────────────────────────────────────
    print("\n[3] Testing Dual MA strategies...")
    for fast, slow in [(5, 20), (10, 40), (10, 60), (20, 60), (20, 120),
                        (40, 120), (40, 200), (60, 200), (60, 250)]:
        r = strategy_dual_ma(df, fast, slow)
        if r:
            r['strategy'] = f'DualMA_{fast}_{slow}'
            r['type'] = 'dual_ma'
            results.append(r)
            print(f"  DualMA {fast}/{slow}: CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── 3. EMA crossover ──────────────────────────────────────────────
    print("\n[4] Testing EMA crossover...")
    for fast, slow in [(5, 20), (12, 26), (20, 60), (20, 120)]:
        r = strategy_ema_crossover(df, fast, slow)
        if r:
            r['strategy'] = f'EMA_{fast}_{slow}'
            r['type'] = 'ema'
            results.append(r)
            print(f"  EMA {fast}/{slow}: CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── 4. MACD ───────────────────────────────────────────────────────
    print("\n[5] Testing MACD...")
    for fast, slow, sig in [(12, 26, 9), (8, 21, 5), (5, 35, 5)]:
        r = strategy_macd_signal(df, fast, slow, sig)
        if r:
            r['strategy'] = f'MACD_{fast}_{slow}_{sig}'
            r['type'] = 'macd'
            results.append(r)
            print(f"  MACD {fast}/{slow}/{sig}: CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── 5. MA + Vol Scaling ───────────────────────────────────────────
    print("\n[6] Testing MA + Vol Scaling...")
    for ma in [40, 60, 120]:
        for vt in [0.10, 0.12, 0.15, 0.20]:
            r = strategy_ma_vol_scaling(df, ma, vol_target=vt)
            if r:
                r['strategy'] = f'MA{ma}_Vol{int(vt*100)}%'
                r['type'] = 'ma_vol'
                results.append(r)
                print(f"  MA{ma}+Vol{int(vt*100)}%: CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── 6. MA + Trailing Stop ─────────────────────────────────────────
    print("\n[7] Testing MA + Trailing Stop...")
    for ma in [40, 60, 120]:
        for stop in [0.05, 0.08, 0.10, 0.15]:
            r = strategy_ma_trailing_stop(df, ma, stop)
            if r:
                r['strategy'] = f'MA{ma}_Stop{int(stop*100)}%'
                r['type'] = 'trailing_stop'
                results.append(r)
                print(f"  MA{ma}+Stop{int(stop*100)}%: CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── 7. MA + RSI ───────────────────────────────────────────────────
    print("\n[8] Testing MA + RSI filter...")
    for ma in [40, 60, 120]:
        r = strategy_ma_rsi(df, ma)
        if r:
            r['strategy'] = f'MA{ma}_RSI'
            r['type'] = 'ma_rsi'
            results.append(r)
            print(f"  MA{ma}+RSI: CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── 8. MA Slope ───────────────────────────────────────────────────
    print("\n[9] Testing MA Slope strategies...")
    for ma in [40, 60, 120]:
        for sw in [5, 10, 20]:
            r = strategy_ma_slope(df, ma, sw)
            if r:
                r['strategy'] = f'MA{ma}_Slope{sw}'
                r['type'] = 'ma_slope'
                results.append(r)
                print(f"  MA{ma}+Slope{sw}: CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── 9. Bollinger Mean Reversion ───────────────────────────────────
    print("\n[10] Testing Bollinger Mean Reversion...")
    for period in [20, 40, 60]:
        for std in [1.5, 2.0, 2.5]:
            r = strategy_bollinger_mean_reversion(df, period, std)
            if r:
                r['strategy'] = f'BB_{period}_{std}'
                r['type'] = 'bollinger'
                results.append(r)
                print(f"  BB({period},{std}): CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── 10. Bollinger + MA filter ─────────────────────────────────────
    print("\n[11] Testing Bollinger + MA filter...")
    for period, std, ma in [(20, 2.0, 60), (20, 2.0, 120), (40, 2.0, 120)]:
        r = strategy_bollinger_mean_reversion(df, period, std, ma_filter=ma)
        if r:
            r['strategy'] = f'BB_{period}_{std}_MA{ma}'
            r['type'] = 'bollinger_ma'
            results.append(r)
            print(f"  BB({period},{std})+MA{ma}: CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── 11. Momentum ──────────────────────────────────────────────────
    print("\n[12] Testing Momentum strategies...")
    for lb in [5, 10, 20, 40, 60, 120]:
        r = strategy_momentum(df, lb)
        if r:
            r['strategy'] = f'MOM_{lb}'
            r['type'] = 'momentum'
            results.append(r)
            print(f"  MOM({lb}): CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── 12. Regime Filter ─────────────────────────────────────────────
    print("\n[13] Testing Regime Filter...")
    for vt in [0.20, 0.25, 0.30]:
        r = strategy_regime_filter(df, 60, vol_threshold=vt)
        if r:
            r['strategy'] = f'Regime_MA60_Vol{int(vt*100)}'
            r['type'] = 'regime'
            results.append(r)
            print(f"  Regime MA60+Vol<{int(vt*100)}%: CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── 13. Volume Breakout ───────────────────────────────────────────
    print("\n[14] Testing Volume Breakout...")
    for ma in [40, 60, 120]:
        for vm in [1.3, 1.5, 2.0]:
            r = strategy_volume_breakout(df, ma, vm)
            if r:
                r['strategy'] = f'VolBreak_MA{ma}_{vm}x'
                r['type'] = 'vol_breakout'
                results.append(r)
                print(f"  VolBreak MA{ma}+{vm}x: CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── 14. Adaptive MA ───────────────────────────────────────────────
    print("\n[15] Testing Adaptive MA...")
    for short, long in [(10, 60), (20, 120), (20, 200)]:
        r = strategy_adaptive_ma(df, short, long)
        if r:
            r['strategy'] = f'AdaptMA_{short}_{long}'
            r['type'] = 'adaptive_ma'
            results.append(r)
            print(f"  AdaptMA {short}/{long}: CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── 15. Combined Score ────────────────────────────────────────────
    print("\n[16] Testing Combined Score...")
    for ma in [40, 60, 120]:
        r = strategy_combined_score(df, ma)
        if r:
            r['strategy'] = f'Combined_MA{ma}'
            r['type'] = 'combined'
            results.append(r)
            print(f"  Combined MA{ma}: CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── 16. Mean Reversion (value-specific) ───────────────────────────
    print("\n[17] Testing Mean Reversion (value-specific)...")
    for lb in [40, 60, 120]:
        for z_entry in [-1.0, -1.5, -2.0]:
            for z_exit in [0.0, 0.5, 1.0]:
                r = strategy_mean_reversion_value(df, lb, z_entry, z_exit)
                if r:
                    r['strategy'] = f'MeanRev_{lb}_z{z_entry}_{z_exit}'
                    r['type'] = 'mean_reversion'
                    results.append(r)

    print(f"\n  Mean reversion: {sum(1 for r in results if r['type']=='mean_reversion')} combos tested")
    mr_results = [r for r in results if r['type'] == 'mean_reversion']
    if mr_results:
        best_mr = max(mr_results, key=lambda x: x['sharpe'])
        print(f"  Best: {best_mr['strategy']}: CAGR={best_mr['cagr']}% Sharpe={best_mr['sharpe']} MaxDD={best_mr['max_dd']}%")

    # ── 17. PE percentile timing (price-based proxy) ──────────────────
    print("\n[18] Testing PE-proxy timing...")
    for ma in [40, 60, 120]:
        r = strategy_pe_timing(df, ma_period=ma)
        if r:
            r['strategy'] = f'PEProxy_MA{ma}'
            r['type'] = 'pe_proxy'
            results.append(r)
            print(f"  PEProxy MA{ma}: CAGR={r['cagr']}% Sharpe={r['sharpe']} MaxDD={r['max_dd']}%")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"TOTAL: {len(results)} strategies tested")
    print("=" * 70)

    # Sort by Sharpe
    results.sort(key=lambda x: x['sharpe'], reverse=True)

    print("\n📊 TOP 15 by Sharpe:")
    print(f"{'Strategy':<30} {'CAGR%':>7} {'Sharpe':>7} {'MaxDD%':>8} {'Calmar':>7} {'WinM%':>6} {'Trades':>6}")
    print("-" * 75)
    for r in results[:15]:
        print(f"{r['strategy']:<30} {r['cagr']:>7} {r['sharpe']:>7} {r['max_dd']:>8} {r['calmar']:>7} {r['win_rate_m']:>6} {r['trades']:>6}")

    print("\n📊 TOP 10 by Calmar (CAGR/MaxDD):")
    results_calmar = sorted(results, key=lambda x: x['calmar'], reverse=True)
    print(f"{'Strategy':<30} {'CAGR%':>7} {'Sharpe':>7} {'MaxDD%':>8} {'Calmar':>7} {'WinM%':>6}")
    print("-" * 70)
    for r in results_calmar[:10]:
        print(f"{r['strategy']:<30} {r['cagr']:>7} {r['sharpe']:>7} {r['max_dd']:>8} {r['calmar']:>7} {r['win_rate_m']:>6}")

    print("\n📊 TOP 10 by lowest MaxDD (CAGR > 5%):")
    results_dd = sorted([r for r in results if r['cagr'] > 5],
                        key=lambda x: x['max_dd'], reverse=True)
    print(f"{'Strategy':<30} {'CAGR%':>7} {'Sharpe':>7} {'MaxDD%':>8} {'Calmar':>7}")
    print("-" * 60)
    for r in results_dd[:10]:
        print(f"{r['strategy']:<30} {r['cagr']:>7} {r['sharpe']:>7} {r['max_dd']:>8} {r['calmar']:>7}")

    # Buy & hold reference
    bh = results[0] if results else {}
    if bh:
        print(f"\n📌 Buy & Hold: CAGR={bh['bh_cagr']}% (for reference)")

    # Save results
    out_path = '/Users/claw/etf-trader/data/csi300_value_timing_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to {out_path}")

    # Value-specific insights
    print("\n" + "=" * 70)
    print("💡 300价值指数择时特征分析")
    print("=" * 70)

    # Compare trend-following vs mean-reversion
    tf_results = [r for r in results if r['type'] in ('pure_ma', 'dual_ma', 'ema', 'macd')]
    mr_results = [r for r in results if r['type'] == 'mean_reversion']

    if tf_results:
        avg_tf_sharpe = np.mean([r['sharpe'] for r in tf_results])
        best_tf = max(tf_results, key=lambda x: x['sharpe'])
        print(f"\n趋势跟踪类 (avg Sharpe={avg_tf_sharpe:.3f}):")
        print(f"  Best: {best_tf['strategy']} Sharpe={best_tf['sharpe']} CAGR={best_tf['cagr']}% MaxDD={best_tf['max_dd']}%")

    if mr_results:
        avg_mr_sharpe = np.mean([r['sharpe'] for r in mr_results])
        best_mr = max(mr_results, key=lambda x: x['sharpe'])
        print(f"\n均值回归类 (avg Sharpe={avg_mr_sharpe:.3f}):")
        print(f"  Best: {best_mr['strategy']} Sharpe={best_mr['sharpe']} CAGR={best_mr['cagr']}% MaxDD={best_mr['max_dd']}%")

    vol_results = [r for r in results if r['type'] == 'ma_vol']
    if vol_results:
        best_vol = max(vol_results, key=lambda x: x['sharpe'])
        print(f"\n波动率缩放类:")
        print(f"  Best: {best_vol['strategy']} Sharpe={best_vol['sharpe']} CAGR={best_vol['cagr']}% MaxDD={best_vol['max_dd']}%")


if __name__ == '__main__':
    main()
