#!/usr/bin/env python3
"""
大趋势动量 + 小趋势逆向 组合策略研究
Trend Following (big picture) + Mean Reversion (short-term) Combined Strategy

核心思路: 顺大势、逆小势
- 大趋势用长均线判断方向(MA60/MA120/MA250)
- 小趋势用短期超跌信号做逆向入场(RSI/回调幅度/MA20偏离)

标的: 多个因子指数(300价值/300红利/红利低波/自由现金流等)
数据源: CSI Official Total Return Index
回测期: 2005~2026

Author: Sarah Mitchell / VisionClaw
Date: 2026-03-29
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import requests
import json, os, sys
from datetime import datetime

DATA_DIR = '/Users/claw/etf-trader/data'

# ============================================================
# 1. DATA LOADING (reuse from V2)
# ============================================================

def fetch_csi_index(code, name, start='20050101', end='20260329'):
    """Fetch index data from CSI official API"""
    url = 'https://www.csindex.com.cn/csindex-home/perf/index-perf'
    params = {'indexCode': code, 'startDate': start, 'endDate': end}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
        'Referer': 'https://www.csindex.com.cn/'
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        data = r.json()
        if str(data.get('code')) != '200' or not data.get('data'):
            print(f"  {name} ({code}): API error")
            return None
        items = data['data']
        df = pd.DataFrame(items)
        df['date'] = pd.to_datetime(df['tradeDate'], format='%Y%m%d')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df[['date', 'close']].dropna().set_index('date').sort_index()
        df = df[df['close'] > 0]
        df = df[~df.index.duplicated(keep='first')]
        total_ret = df['close'].iloc[-1] / df['close'].iloc[0] - 1
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        print(f"  {name} ({code}): {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} rows, CAGR={cagr*100:.1f}%")
        return df
    except Exception as e:
        print(f"  {name} ({code}): ERROR {e}")
        return None

def load_indices():
    """Load key factor indices"""
    print("[1] Loading CSI official total return indices...")
    indices = {
        '300价值': 'H00919',         # 沪深300价值全收益 (API verified)
        '300成长': 'H00918',         # 沪深300成长全收益 (API verified)
        '300相对成长': 'H00920',     # 沪深300相对成长全收益 (was mislabeled 300红利)
        '300相对价值': 'H00921',     # 沪深300相对价值全收益 (was mislabeled 300低波)
        '红利低波': 'H20269',
        '自由现金流': '932365',
        '基本面50': 'H00925',
        '沪深300': 'H00300',
        '中证红利': 'H00922',
    }
    factors = {}
    for name, code in indices.items():
        df = fetch_csi_index(code, name)
        if df is not None and len(df) > 250:
            factors[name] = df
    print(f"  Loaded {len(factors)} indices\n")
    return factors

# ============================================================
# 2. INDICATORS
# ============================================================

def calc_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_atr(prices, period=20):
    """Calculate ATR using close-to-close (simplified for index)"""
    returns = prices.pct_change().abs()
    return returns.rolling(period).mean() * prices

# ============================================================
# 3. METRICS
# ============================================================

def calc_metrics(returns, name='Strategy'):
    """Calculate strategy metrics"""
    returns = returns.dropna()
    if len(returns) < 252:
        return None

    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1
    years = len(returns) / 252
    cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0

    # Max drawdown
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min() * 100

    # Monthly win rate
    monthly = returns.resample('ME').sum()
    win_rate = (monthly > 0).mean() * 100

    # Trade count (signal changes)
    calmar = cagr / abs(max_dd/100) if max_dd != 0 else 0

    return {
        'name': name,
        'cagr': round(cagr * 100, 2),
        'vol': round(vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'max_dd': round(max_dd, 2),
        'calmar': round(calmar, 3),
        'win_rate_monthly': round(win_rate, 1),
        'years': round(years, 1),
        'total_ret': round(total_ret * 100, 1),
    }

# ============================================================
# 4. STRATEGY IMPLEMENTATIONS
# ============================================================

def strat_buy_hold(df):
    """Baseline: buy and hold"""
    return df['close'].pct_change().dropna()

def strat_ma_only(df, ma_period=60):
    """Pure MA timing (baseline from V2)"""
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['ret'] = d['close'].pct_change()
    d['signal'] = (d['close'] > d['ma']).shift(1).astype(float)
    d['strat_ret'] = d['ret'] * d['signal']
    return d['strat_ret'].dropna()

def strat_trend_reversal_rsi(df, trend_ma=60, rsi_period=14, rsi_buy=30, rsi_sell=70):
    """
    Strategy 1: MA trend + RSI reversal
    - Trend up (close > MA_trend): use RSI for entry/exit
      - RSI < rsi_buy: full position (oversold in uptrend = buy)
      - RSI > rsi_sell: half position (overbought = reduce)
      - Otherwise: maintain current position
    - Trend down: 0 position
    """
    d = df.copy()
    d['ma'] = d['close'].rolling(trend_ma).mean()
    d['rsi'] = calc_rsi(d['close'], rsi_period)
    d['ret'] = d['close'].pct_change()

    # Build position signal
    d['trend_up'] = d['close'] > d['ma']

    # Position: 0 = cash, 0.5 = half, 1.0 = full
    position = pd.Series(0.0, index=d.index)
    current_pos = 0.0

    for i in range(1, len(d)):
        if not d['trend_up'].iloc[i]:
            current_pos = 0.0
        else:
            rsi_val = d['rsi'].iloc[i]
            if pd.isna(rsi_val):
                current_pos = 1.0 if d['trend_up'].iloc[i] else 0.0
            elif rsi_val < rsi_buy:
                current_pos = 1.0  # Oversold in uptrend -> full
            elif rsi_val > rsi_sell:
                current_pos = 0.5  # Overbought -> reduce
            # else: maintain current_pos
        position.iloc[i] = current_pos

    d['signal'] = position.shift(1)
    d['strat_ret'] = d['ret'] * d['signal']
    return d['strat_ret'].dropna()

def strat_trend_reversal_rsi_v2(df, trend_ma=60, rsi_period=14, rsi_buy=35, rsi_sell=65):
    """
    Strategy 1b: Same but with tighter RSI thresholds
    """
    return strat_trend_reversal_rsi(df, trend_ma, rsi_period, rsi_buy, rsi_sell)

def strat_trend_pullback_ma20(df, trend_ma=60, short_ma=20):
    """
    Strategy 2: MA trend + MA20 pullback
    - Trend up (close > MA_trend):
      - Price pulls back to MA20 -> buy (full position)
      - Price above MA20 by >5% -> maintain
    - Trend down: 0 position
    """
    d = df.copy()
    d['ma_trend'] = d['close'].rolling(trend_ma).mean()
    d['ma_short'] = d['close'].rolling(short_ma).mean()
    d['ret'] = d['close'].pct_change()
    d['trend_up'] = d['close'] > d['ma_trend']
    d['near_ma20'] = d['close'] <= d['ma_short'] * 1.02  # Within 2% of MA20
    d['above_ma20'] = d['close'] > d['ma_short']

    position = pd.Series(0.0, index=d.index)
    current_pos = 0.0

    for i in range(1, len(d)):
        if not d['trend_up'].iloc[i]:
            current_pos = 0.0
        else:
            if d['near_ma20'].iloc[i]:
                current_pos = 1.0  # Pullback to MA20 in uptrend -> buy
            elif not d['above_ma20'].iloc[i]:
                current_pos = 0.5  # Below MA20 but above trend MA -> reduce
            # If above MA20, maintain position
            if current_pos == 0.0 and d['trend_up'].iloc[i]:
                current_pos = 0.5  # At least half position in uptrend
        position.iloc[i] = current_pos

    d['signal'] = position.shift(1)
    d['strat_ret'] = d['ret'] * d['signal']
    return d['strat_ret'].dropna()

def strat_trend_dip_buy(df, trend_ma=60, dip_pct=-0.05, dip_window=10):
    """
    Strategy 3: MA trend + dip buying
    - Trend up (close > MA_trend):
      - If price dropped >dip_pct% in last dip_window days -> full position
      - Otherwise -> 70% position
    - Trend down: 0 position
    """
    d = df.copy()
    d['ma_trend'] = d['close'].rolling(trend_ma).mean()
    d['ret'] = d['close'].pct_change()
    d['trend_up'] = d['close'] > d['ma_trend']
    d['recent_change'] = d['close'].pct_change(dip_window)

    position = pd.Series(0.0, index=d.index)

    for i in range(1, len(d)):
        if not d['trend_up'].iloc[i]:
            position.iloc[i] = 0.0
        else:
            rc = d['recent_change'].iloc[i]
            if pd.notna(rc) and rc < dip_pct:
                position.iloc[i] = 1.0  # Dip in uptrend -> full
            else:
                position.iloc[i] = 0.7  # Normal uptrend -> 70%

    d['signal'] = position.shift(1)
    d['strat_ret'] = d['ret'] * d['signal']
    return d['strat_ret'].dropna()

def strat_trend_atr_pullback(df, trend_ma=60, atr_period=20, atr_mult=1.5):
    """
    Strategy 4: MA trend + ATR-based pullback
    - Trend up:
      - Price drops > atr_mult * ATR from recent high -> full position (buy dip)
      - Otherwise -> 60% position
    - Trend down: 0 position
    """
    d = df.copy()
    d['ma_trend'] = d['close'].rolling(trend_ma).mean()
    d['ret'] = d['close'].pct_change()
    d['trend_up'] = d['close'] > d['ma_trend']
    d['atr'] = d['close'].pct_change().abs().rolling(atr_period).mean() * d['close']
    d['recent_high'] = d['close'].rolling(atr_period).max()
    d['pullback'] = d['recent_high'] - d['close']

    position = pd.Series(0.0, index=d.index)

    for i in range(1, len(d)):
        if not d['trend_up'].iloc[i]:
            position.iloc[i] = 0.0
        else:
            atr_val = d['atr'].iloc[i]
            pb_val = d['pullback'].iloc[i]
            if pd.notna(atr_val) and pd.notna(pb_val) and atr_val > 0:
                if pb_val > atr_mult * atr_val:
                    position.iloc[i] = 1.0  # Big pullback -> full
                else:
                    position.iloc[i] = 0.6
            else:
                position.iloc[i] = 0.6

    d['signal'] = position.shift(1)
    d['strat_ret'] = d['ret'] * d['signal']
    return d['strat_ret'].dropna()

def strat_dual_trend_rsi(df, long_ma=250, short_ma=60, rsi_period=14):
    """
    Strategy 5: Dual MA trend + RSI
    - Strong uptrend (close > MA250 AND close > MA60): full position
    - Weak uptrend (close > MA250 but < MA60):
      - RSI < 30 -> full position (oversold bounce)
      - Otherwise -> 30% position
    - Downtrend (close < MA250): 0 position
    """
    d = df.copy()
    d['ma_long'] = d['close'].rolling(long_ma).mean()
    d['ma_short'] = d['close'].rolling(short_ma).mean()
    d['rsi'] = calc_rsi(d['close'], rsi_period)
    d['ret'] = d['close'].pct_change()

    position = pd.Series(0.0, index=d.index)

    for i in range(1, len(d)):
        close = d['close'].iloc[i]
        ma_l = d['ma_long'].iloc[i]
        ma_s = d['ma_short'].iloc[i]
        rsi_val = d['rsi'].iloc[i]

        if pd.isna(ma_l) or pd.isna(ma_s):
            continue

        if close < ma_l:
            position.iloc[i] = 0.0  # Downtrend
        elif close > ma_s:
            position.iloc[i] = 1.0  # Strong uptrend
        else:
            # Weak uptrend (above MA250 but below MA60)
            if pd.notna(rsi_val) and rsi_val < 30:
                position.iloc[i] = 1.0  # Oversold bounce
            else:
                position.iloc[i] = 0.3  # Cautious

    d['signal'] = position.shift(1)
    d['strat_ret'] = d['ret'] * d['signal']
    return d['strat_ret'].dropna()

def strat_trend_reversal_bollinger(df, trend_ma=60, bb_period=20, bb_std=2.0):
    """
    Strategy 6: MA trend + Bollinger Band reversal
    - Trend up:
      - Price touches lower BB -> full position (mean reversion)
      - Price touches upper BB -> 50% position (take some profit)
      - Otherwise -> 70% position
    - Trend down: 0 position
    """
    d = df.copy()
    d['ma_trend'] = d['close'].rolling(trend_ma).mean()
    d['bb_mid'] = d['close'].rolling(bb_period).mean()
    d['bb_std'] = d['close'].rolling(bb_period).std()
    d['bb_upper'] = d['bb_mid'] + bb_std * d['bb_std']
    d['bb_lower'] = d['bb_mid'] - bb_std * d['bb_std']
    d['ret'] = d['close'].pct_change()
    d['trend_up'] = d['close'] > d['ma_trend']

    position = pd.Series(0.0, index=d.index)
    current_pos = 0.0

    for i in range(1, len(d)):
        if not d['trend_up'].iloc[i]:
            current_pos = 0.0
        else:
            close = d['close'].iloc[i]
            bb_l = d['bb_lower'].iloc[i]
            bb_u = d['bb_upper'].iloc[i]

            if pd.notna(bb_l) and pd.notna(bb_u):
                if close <= bb_l:
                    current_pos = 1.0  # Touch lower BB -> full
                elif close >= bb_u:
                    current_pos = 0.5  # Touch upper BB -> reduce
                # else maintain

            if current_pos == 0.0 and d['trend_up'].iloc[i]:
                current_pos = 0.7

        position.iloc[i] = current_pos

    d['signal'] = position.shift(1)
    d['strat_ret'] = d['ret'] * d['signal']
    return d['strat_ret'].dropna()

def strat_ma_with_rsi_filter(df, ma_period=60, rsi_period=14, rsi_entry=40):
    """
    Strategy 7: Simple MA + RSI entry filter
    - Only enter when close > MA AND RSI < rsi_entry (not overbought)
    - Exit when close < MA
    - This avoids chasing tops
    """
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['rsi'] = calc_rsi(d['close'], rsi_period)
    d['ret'] = d['close'].pct_change()
    d['above_ma'] = d['close'] > d['ma']

    position = pd.Series(0.0, index=d.index)
    current_pos = 0.0

    for i in range(1, len(d)):
        if not d['above_ma'].iloc[i]:
            current_pos = 0.0  # Below MA -> exit
        else:
            rsi_val = d['rsi'].iloc[i]
            if current_pos == 0.0:
                # Only enter if RSI not overbought
                if pd.notna(rsi_val) and rsi_val < rsi_entry:
                    current_pos = 1.0
                # else wait
            else:
                current_pos = 1.0  # Already in, stay in
        position.iloc[i] = current_pos

    d['signal'] = position.shift(1)
    d['strat_ret'] = d['ret'] * d['signal']
    return d['strat_ret'].dropna()

def strat_trend_momentum_reversal_switch(df, trend_ma=120, mom_lookback=20, reversal_lookback=5):
    """
    Strategy 8: Trend-dependent momentum vs reversal switch
    - Strong trend (close > MA * 1.05): use momentum (stay in if recent returns positive)
    - Near MA (within 5%): use reversal (buy dips, sell rallies)
    - Below MA: cash
    """
    d = df.copy()
    d['ma'] = d['close'].rolling(trend_ma).mean()
    d['ret'] = d['close'].pct_change()
    d['mom'] = d['close'].pct_change(mom_lookback)
    d['rev'] = d['close'].pct_change(reversal_lookback)
    d['dist_to_ma'] = (d['close'] - d['ma']) / d['ma']

    position = pd.Series(0.0, index=d.index)

    for i in range(1, len(d)):
        dist = d['dist_to_ma'].iloc[i]
        mom = d['mom'].iloc[i]
        rev = d['rev'].iloc[i]

        if pd.isna(dist) or pd.isna(mom):
            continue

        if dist < 0:
            position.iloc[i] = 0.0  # Below MA -> cash
        elif dist > 0.05:
            # Strong trend -> momentum
            if mom > 0:
                position.iloc[i] = 1.0
            else:
                position.iloc[i] = 0.5
        else:
            # Near MA -> reversal
            if pd.notna(rev) and rev < -0.03:
                position.iloc[i] = 1.0  # Short-term dip near MA -> buy
            elif pd.notna(rev) and rev > 0.03:
                position.iloc[i] = 0.3  # Short-term rally near MA -> reduce
            else:
                position.iloc[i] = 0.6

    d['signal'] = position.shift(1)
    d['strat_ret'] = d['ret'] * d['signal']
    return d['strat_ret'].dropna()

def strat_adaptive_trend_reversal(df, fast_ma=20, slow_ma=60, trend_ma=250):
    """
    Strategy 9: Adaptive - uses regime detection
    - Bull regime (close > MA250):
      - Fast MA > Slow MA: full position (momentum)
      - Fast MA < Slow MA but close > MA250: buy on dips (逆向)
    - Bear regime: 0 position
    """
    d = df.copy()
    d['ma_fast'] = d['close'].rolling(fast_ma).mean()
    d['ma_slow'] = d['close'].rolling(slow_ma).mean()
    d['ma_trend'] = d['close'].rolling(trend_ma).mean()
    d['ret'] = d['close'].pct_change()
    d['rsi'] = calc_rsi(d['close'], 14)

    position = pd.Series(0.0, index=d.index)

    for i in range(1, len(d)):
        close = d['close'].iloc[i]
        ma_t = d['ma_trend'].iloc[i]
        ma_f = d['ma_fast'].iloc[i]
        ma_s = d['ma_slow'].iloc[i]
        rsi = d['rsi'].iloc[i]

        if pd.isna(ma_t) or pd.isna(ma_f) or pd.isna(ma_s):
            continue

        if close < ma_t:
            position.iloc[i] = 0.0  # Bear -> cash
        elif ma_f > ma_s:
            position.iloc[i] = 1.0  # Bull + momentum -> full
        else:
            # Bull but pullback (fast < slow)
            if pd.notna(rsi) and rsi < 35:
                position.iloc[i] = 1.0  # Oversold in bull -> buy
            elif pd.notna(rsi) and rsi < 50:
                position.iloc[i] = 0.7
            else:
                position.iloc[i] = 0.4

    d['signal'] = position.shift(1)
    d['strat_ret'] = d['ret'] * d['signal']
    return d['strat_ret'].dropna()

# ============================================================
# 5. MAIN: RUN ALL STRATEGIES ON ALL FACTORS
# ============================================================

def count_trades(returns_series, threshold=0.001):
    """Estimate number of position changes"""
    # Approximate from strategy returns
    active = (returns_series.abs() > threshold).astype(int)
    changes = active.diff().abs().sum() / 2
    return int(changes)

def run_all():
    factors = load_indices()
    if not factors:
        print("ERROR: No data loaded!")
        return

    # Define all strategies to test
    strategies = [
        # Baselines
        ('BuyHold', lambda df: strat_buy_hold(df)),
        ('MA60_Only', lambda df: strat_ma_only(df, 60)),
        ('MA120_Only', lambda df: strat_ma_only(df, 120)),
        ('MA250_Only', lambda df: strat_ma_only(df, 250)),

        # Trend + RSI reversal
        ('MA60+RSI(30/70)', lambda df: strat_trend_reversal_rsi(df, 60, 14, 30, 70)),
        ('MA60+RSI(35/65)', lambda df: strat_trend_reversal_rsi(df, 60, 14, 35, 65)),
        ('MA60+RSI(40/60)', lambda df: strat_trend_reversal_rsi(df, 60, 14, 40, 60)),
        ('MA120+RSI(30/70)', lambda df: strat_trend_reversal_rsi(df, 120, 14, 30, 70)),
        ('MA250+RSI(30/70)', lambda df: strat_trend_reversal_rsi(df, 250, 14, 30, 70)),

        # Trend + MA20 pullback
        ('MA60+MA20回调', lambda df: strat_trend_pullback_ma20(df, 60, 20)),
        ('MA120+MA20回调', lambda df: strat_trend_pullback_ma20(df, 120, 20)),

        # Trend + dip buying
        ('MA60+逢跌5%加仓', lambda df: strat_trend_dip_buy(df, 60, -0.05, 10)),
        ('MA60+逢跌3%加仓', lambda df: strat_trend_dip_buy(df, 60, -0.03, 10)),
        ('MA60+逢跌8%加仓', lambda df: strat_trend_dip_buy(df, 60, -0.08, 20)),

        # Trend + ATR pullback
        ('MA60+ATR回调(1.5x)', lambda df: strat_trend_atr_pullback(df, 60, 20, 1.5)),
        ('MA60+ATR回调(2.0x)', lambda df: strat_trend_atr_pullback(df, 60, 20, 2.0)),

        # Dual trend + RSI
        ('MA250+MA60+RSI', lambda df: strat_dual_trend_rsi(df, 250, 60, 14)),
        ('MA250+MA60+RSI(松)', lambda df: strat_dual_trend_rsi(df, 250, 60, 14)),

        # Bollinger Band reversal
        ('MA60+布林带(2σ)', lambda df: strat_trend_reversal_bollinger(df, 60, 20, 2.0)),
        ('MA60+布林带(1.5σ)', lambda df: strat_trend_reversal_bollinger(df, 60, 20, 1.5)),

        # MA + RSI entry filter
        ('MA60+RSI入场<40', lambda df: strat_ma_with_rsi_filter(df, 60, 14, 40)),
        ('MA60+RSI入场<50', lambda df: strat_ma_with_rsi_filter(df, 60, 14, 50)),
        ('MA60+RSI入场<60', lambda df: strat_ma_with_rsi_filter(df, 60, 14, 60)),

        # Trend-dependent switch
        ('MA120动量逆向切换', lambda df: strat_trend_momentum_reversal_switch(df, 120, 20, 5)),
        ('MA60动量逆向切换', lambda df: strat_trend_momentum_reversal_switch(df, 60, 20, 5)),

        # Adaptive
        ('自适应(20/60/250)', lambda df: strat_adaptive_trend_reversal(df, 20, 60, 250)),
        ('自适应(10/60/120)', lambda df: strat_adaptive_trend_reversal(df, 10, 60, 120)),
    ]

    all_results = []

    print(f"\n[2] Running {len(strategies)} strategies on {len(factors)} indices...")
    print(f"    Total combinations: {len(strategies) * len(factors)}\n")

    for factor_name, factor_df in factors.items():
        print(f"\n{'='*60}")
        print(f"  Factor: {factor_name}")
        print(f"{'='*60}")
        print(f"  {'Strategy':<30} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Vol':>7} {'WinR%':>6}")
        print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*6}")

        for strat_name, strat_fn in strategies:
            try:
                returns = strat_fn(factor_df)
                metrics = calc_metrics(returns, f"{strat_name}_{factor_name}")
                if metrics:
                    metrics['strategy'] = strat_name
                    metrics['factor'] = factor_name
                    all_results.append(metrics)
                    print(f"  {strat_name:<30} {metrics['cagr']:>6.1f}% {metrics['sharpe']:>7.3f} "
                          f"{metrics['max_dd']:>7.1f}% {metrics['vol']:>6.1f}% {metrics['win_rate_monthly']:>5.1f}")
            except Exception as e:
                print(f"  {strat_name:<30} ERROR: {e}")

    # ============================================================
    # 6. SUMMARY & RANKING
    # ============================================================

    print(f"\n\n{'='*80}")
    print(f"  OVERALL RANKING (Top 30 by Sharpe)")
    print(f"{'='*80}")

    sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)

    print(f"  {'#':>3} {'Strategy+Factor':<45} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Vol':>7} {'WinR%':>6}")
    print(f"  {'-'*3} {'-'*45} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*6}")

    for i, r in enumerate(sorted_results[:30]):
        label = f"{r['strategy']}+{r['factor']}"
        marker = " ***" if r['sharpe'] > 1.0 else ""
        print(f"  {i+1:>3} {label:<45} {r['cagr']:>6.1f}% {r['sharpe']:>7.3f} "
              f"{r['max_dd']:>7.1f}% {r['vol']:>6.1f}% {r['win_rate_monthly']:>5.1f}{marker}")

    # Compare combo strategies vs pure MA60
    print(f"\n\n{'='*80}")
    print(f"  KEY COMPARISON: Combo vs Pure MA60 (per factor)")
    print(f"{'='*80}")

    for factor_name in factors.keys():
        ma60_result = next((r for r in all_results if r['strategy'] == 'MA60_Only' and r['factor'] == factor_name), None)
        if not ma60_result:
            continue

        print(f"\n  [{factor_name}] MA60 baseline: CAGR={ma60_result['cagr']}% Sharpe={ma60_result['sharpe']} "
              f"MaxDD={ma60_result['max_dd']}% WinRate={ma60_result['win_rate_monthly']}%")

        factor_results = [r for r in all_results if r['factor'] == factor_name and r['strategy'] != 'BuyHold' and r['strategy'] != 'MA60_Only']
        factor_sorted = sorted(factor_results, key=lambda x: x['sharpe'], reverse=True)

        for r in factor_sorted[:5]:
            sharpe_diff = r['sharpe'] - ma60_result['sharpe']
            arrow = "↑" if sharpe_diff > 0 else "↓"
            print(f"    {r['strategy']:<35} CAGR={r['cagr']:>6.1f}% Sharpe={r['sharpe']:>6.3f}({arrow}{abs(sharpe_diff):.3f}) "
                  f"MaxDD={r['max_dd']:>7.1f}% WinR={r['win_rate_monthly']:>5.1f}%")

    # Save results
    output = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'description': '大趋势动量+小趋势逆向 组合策略研究',
        'total_combos': len(all_results),
        'all_results': sorted_results,
    }

    out_path = os.path.join(DATA_DIR, 'trend_reversal_combo_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n\nResults saved to {out_path}")
    print(f"Total combinations tested: {len(all_results)}")

if __name__ == '__main__':
    run_all()
