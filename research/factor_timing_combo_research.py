#!/usr/bin/env python3
"""
A股因子择时+轮动组合策略研究
Factor Timing & Rotation Strategy Research for A-shares

因子覆盖: 价值(300价值), 红利(红利指数/中证红利), 低波(300低波), 成长(300成长/中证成长), 现金流(自由现金流)
择时方法: MA均线/双均线/波动率/ERP/动量轮动
基准: 沪深300

数据源: baostock(价格), akshare(国债收益率/宏观), 已有csv

Author: Sarah Mitchell / VisionClaw
Date: 2026-03-28
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import baostock as bs
import akshare as ak
import json, os, traceback
from datetime import datetime

DATA_DIR = '/Users/claw/etf-trader/data'
OUT_DIR = DATA_DIR

# ============================================================
# 1. DATA LOADING
# ============================================================

def load_index_from_csv(filename, date_col='date', close_col='close'):
    """Load index from existing CSV"""
    fp = os.path.join(DATA_DIR, filename)
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp)
    if '日期' in df.columns and date_col == 'date' and 'date' not in df.columns:
        date_col = '日期'
    if '收盘' in df.columns and close_col == 'close' and 'close' not in df.columns:
        close_col = '收盘'
    df['date'] = pd.to_datetime(df[date_col])
    df['close'] = pd.to_numeric(df[close_col], errors='coerce')
    df = df[['date', 'close']].dropna().set_index('date').sort_index()
    return df

def load_index_from_baostock(code, name, start='2005-01-01', end='2026-03-28'):
    """Load index from baostock"""
    print(f"  Fetching {name} ({code}) from baostock...")
    rs = bs.query_history_k_data_plus(
        code, "date,close", start_date=start, end_date=end, frequency="d"
    )
    data = []
    while (rs.error_code == '0') and rs.next():
        data.append(rs.get_row_data())
    if not data:
        print(f"    -> No data for {name}")
        return None
    df = pd.DataFrame(data, columns=['date', 'close'])
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna().set_index('date').sort_index()
    df = df[df['close'] > 0]
    print(f"    -> {name}: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} rows")
    return df

def load_index_from_akshare(code, name):
    """Load index from akshare (for indices not in baostock)"""
    print(f"  Fetching {name} ({code}) from akshare...")
    try:
        df = ak.index_zh_a_hist(symbol=code, period="daily", start_date="20050101", end_date="20260328")
        df['date'] = pd.to_datetime(df['日期'])
        df['close'] = pd.to_numeric(df['收盘'], errors='coerce')
        df = df[['date', 'close']].dropna().set_index('date').sort_index()
        df = df[df['close'] > 0]
        print(f"    -> {name}: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} rows")
        return df
    except Exception as e:
        print(f"    -> Failed: {e}")
        return None

def get_bond_yield():
    """Get 10-year government bond yield"""
    print("  Fetching 10Y bond yield...")
    try:
        df = ak.bond_zh_us_rate(start_date="20050101")
        df = df[['日期', '中国国债收益率10年']].copy()
        df.columns = ['date', 'yield_10y']
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['yield_10y']).set_index('date').sort_index()
        if df['yield_10y'].mean() > 1:
            df['yield_10y'] = df['yield_10y'] / 100.0
        print(f"    -> Bond yield: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} rows")
        return df
    except Exception as e:
        print(f"    -> Failed: {e}")
        return None


def load_all_factors():
    """Load all factor index data"""
    print("\n[1] Loading factor index data...")
    bs.login()

    factors = {}

    # Factor indices available from baostock (2010+)
    bs_indices = {
        '300价值': 'sh.000918',
        '300成长': 'sh.000919',
        '中证红利': 'sh.000922',
        '基本面50': 'sh.000925',
        '红利指数': 'sh.000015',
        '沪深300': 'sh.000300',
        '中证500': 'sh.000905',
        '中证1000': 'sh.000852',
    }

    for name, code in bs_indices.items():
        df = load_index_from_baostock(code, name)
        if df is not None and len(df) > 500:
            factors[name] = df

    bs.logout()

    # Try akshare for additional indices
    ak_indices = {
        '300红利': '000920',
        '300低波': '000921',
        '中证成长': '000923',
    }

    for name, code in ak_indices.items():
        # First try CSV
        csv_name = f"{code}_daily.csv"
        df = load_index_from_csv(csv_name, date_col='date', close_col='收盘')
        if df is None:
            df = load_index_from_csv(csv_name)
        if df is not None and len(df) > 500:
            factors[name] = df
            print(f"    -> {name} from CSV: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} rows")
        else:
            df = load_index_from_akshare(code, name)
            if df is not None and len(df) > 500:
                factors[name] = df

    # Try akshare for newer factor indices
    newer_indices = {
        '中证低波红利': '931157',
        '自由现金流': '932365',
    }
    for name, code in newer_indices.items():
        csv_name = f"{code}CNY010_daily.csv" if code == '932365' else f"{code}_daily.csv"
        df = load_index_from_csv(csv_name)
        if df is not None and len(df) > 200:
            factors[name] = df
            print(f"    -> {name} from CSV: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} rows")
        else:
            df = load_index_from_akshare(code, name)
            if df is not None and len(df) > 200:
                factors[name] = df

    print(f"\n  Loaded {len(factors)} factor indices")
    for name, df in factors.items():
        total_ret = df['close'].iloc[-1] / df['close'].iloc[0] - 1
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        print(f"    {name}: {df.index[0].date()} ~ {df.index[-1].date()}, CAGR={cagr*100:.1f}%")

    return factors


# ============================================================
# 2. BACKTEST ENGINE
# ============================================================

def calc_metrics(returns, name=''):
    """Calculate performance metrics from daily returns series"""
    if len(returns) < 20:
        return None

    cum_ret = (1 + returns).cumprod()
    total_ret = cum_ret.iloc[-1] - 1
    years = len(returns) / 252
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0

    # Max drawdown
    running_max = cum_ret.cummax()
    drawdown = cum_ret / running_max - 1
    max_dd = drawdown.min()

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate (monthly)
    monthly = returns.resample('ME').sum()
    win_rate = (monthly > 0).mean()

    return {
        'name': name,
        'cagr': round(cagr * 100, 2),
        'vol': round(vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'max_dd': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'total_ret': round(total_ret * 100, 1),
        'years': round(years, 1),
        'win_rate_monthly': round(win_rate * 100, 1),
    }


def backtest_buy_hold(prices, name='BuyHold'):
    """Simple buy and hold"""
    returns = prices['close'].pct_change().dropna()
    return calc_metrics(returns, name)


def backtest_ma_timing(prices, ma_period=120, name='MA'):
    """MA timing: above MA = full position, below MA = cash (0 return)"""
    df = prices.copy()
    df['ma'] = df['close'].rolling(ma_period).mean()
    df['ret'] = df['close'].pct_change()
    df['signal'] = (df['close'] > df['ma']).shift(1).astype(float)  # T-1 signal
    df['strat_ret'] = df['ret'] * df['signal']
    return calc_metrics(df['strat_ret'].dropna(), name)


def backtest_dual_ma(prices, fast=20, slow=120, name='DualMA'):
    """Dual MA: fast > slow = full position, else cash"""
    df = prices.copy()
    df['ma_fast'] = df['close'].rolling(fast).mean()
    df['ma_slow'] = df['close'].rolling(slow).mean()
    df['ret'] = df['close'].pct_change()
    df['signal'] = (df['ma_fast'] > df['ma_slow']).shift(1).astype(float)
    df['strat_ret'] = df['ret'] * df['signal']
    return calc_metrics(df['strat_ret'].dropna(), name)


def backtest_vol_timing(prices, vol_window=20, vol_target=0.15, name='VolTiming'):
    """Volatility targeting: scale position to target vol"""
    df = prices.copy()
    df['ret'] = df['close'].pct_change()
    df['vol'] = df['ret'].rolling(vol_window).std() * np.sqrt(252)
    df['weight'] = (vol_target / df['vol']).clip(0, 1.5).shift(1)
    df['strat_ret'] = df['ret'] * df['weight']
    return calc_metrics(df['strat_ret'].dropna(), name)


def backtest_momentum_rotation(factor_prices, lookback=60, top_n=2, rebal_freq=20, name='MomRot'):
    """Momentum rotation: pick top_n factors by recent return, rebalance every rebal_freq days"""
    # Align all factors to common dates
    all_close = pd.DataFrame()
    for fname, fdf in factor_prices.items():
        all_close[fname] = fdf['close']
    all_close = all_close.dropna(how='all').ffill()

    # Daily returns
    all_ret = all_close.pct_change()

    # Generate signals
    strat_ret = pd.Series(0.0, index=all_ret.index)
    rebal_dates = all_ret.index[lookback::rebal_freq]

    current_weights = {}
    for i, date in enumerate(all_ret.index):
        if i < lookback:
            continue

        if date in rebal_dates or not current_weights:
            # Calculate momentum (lookback return)
            mom = {}
            for col in all_close.columns:
                if pd.notna(all_close[col].iloc[i]) and pd.notna(all_close[col].iloc[max(0, i-lookback)]):
                    v0 = all_close[col].iloc[max(0, i-lookback)]
                    v1 = all_close[col].iloc[i]
                    if v0 > 0:
                        mom[col] = v1 / v0 - 1

            if len(mom) >= top_n:
                sorted_factors = sorted(mom.items(), key=lambda x: x[1], reverse=True)
                top_factors = [f[0] for f in sorted_factors[:top_n]]
                w = 1.0 / top_n
                current_weights = {f: w for f in top_factors}
            elif mom:
                w = 1.0 / len(mom)
                current_weights = {f: w for f in mom}

        day_ret = 0.0
        for f, w in current_weights.items():
            if f in all_ret.columns and pd.notna(all_ret[f].iloc[i]):
                day_ret += w * all_ret[f].iloc[i]
        strat_ret.iloc[i] = day_ret

    return calc_metrics(strat_ret.dropna(), name)


def backtest_inverse_momentum_rotation(factor_prices, lookback=60, top_n=2, rebal_freq=20, name='RevRot'):
    """Inverse momentum (reversal) rotation: pick WORST performing factors"""
    all_close = pd.DataFrame()
    for fname, fdf in factor_prices.items():
        all_close[fname] = fdf['close']
    all_close = all_close.dropna(how='all').ffill()
    all_ret = all_close.pct_change()

    strat_ret = pd.Series(0.0, index=all_ret.index)
    rebal_dates = all_ret.index[lookback::rebal_freq]

    current_weights = {}
    for i, date in enumerate(all_ret.index):
        if i < lookback:
            continue

        if date in rebal_dates or not current_weights:
            mom = {}
            for col in all_close.columns:
                if pd.notna(all_close[col].iloc[i]) and pd.notna(all_close[col].iloc[max(0, i-lookback)]):
                    v0 = all_close[col].iloc[max(0, i-lookback)]
                    v1 = all_close[col].iloc[i]
                    if v0 > 0:
                        mom[col] = v1 / v0 - 1

            if len(mom) >= top_n:
                sorted_factors = sorted(mom.items(), key=lambda x: x[1])  # ascending = worst first
                top_factors = [f[0] for f in sorted_factors[:top_n]]
                w = 1.0 / top_n
                current_weights = {f: w for f in top_factors}

        day_ret = 0.0
        for f, w in current_weights.items():
            if f in all_ret.columns and pd.notna(all_ret[f].iloc[i]):
                day_ret += w * all_ret[f].iloc[i]
        strat_ret.iloc[i] = day_ret

    return calc_metrics(strat_ret.dropna(), name)


def backtest_equal_weight(factor_prices, name='EqualWeight'):
    """Equal weight all factors, rebalance daily"""
    all_close = pd.DataFrame()
    for fname, fdf in factor_prices.items():
        all_close[fname] = fdf['close']
    all_close = all_close.dropna(how='all').ffill()
    all_ret = all_close.pct_change()
    strat_ret = all_ret.mean(axis=1)
    return calc_metrics(strat_ret.dropna(), name)


def backtest_equal_weight_ma_timing(factor_prices, ma_period=120, name='EW+MA'):
    """Equal weight all factors with MA timing on each"""
    all_close = pd.DataFrame()
    for fname, fdf in factor_prices.items():
        all_close[fname] = fdf['close']
    all_close = all_close.dropna(how='all').ffill()
    all_ret = all_close.pct_change()

    # MA signal for each factor
    signals = pd.DataFrame()
    for col in all_close.columns:
        ma = all_close[col].rolling(ma_period).mean()
        signals[col] = (all_close[col] > ma).shift(1).astype(float)

    timed_ret = all_ret * signals
    strat_ret = timed_ret.mean(axis=1)
    return calc_metrics(strat_ret.dropna(), name)


def backtest_momentum_rotation_with_ma(factor_prices, lookback=60, top_n=2, rebal_freq=20, ma_period=120, name='MomRot+MA'):
    """Momentum rotation + MA filter: only hold factors that are above their MA"""
    all_close = pd.DataFrame()
    for fname, fdf in factor_prices.items():
        all_close[fname] = fdf['close']
    all_close = all_close.dropna(how='all').ffill()
    all_ret = all_close.pct_change()

    # MA for each factor
    all_ma = all_close.rolling(ma_period).mean()

    strat_ret = pd.Series(0.0, index=all_ret.index)
    rebal_dates = all_ret.index[lookback::rebal_freq]

    current_weights = {}
    for i, date in enumerate(all_ret.index):
        if i < max(lookback, ma_period):
            continue

        if date in rebal_dates or not current_weights:
            mom = {}
            for col in all_close.columns:
                if pd.notna(all_close[col].iloc[i]) and pd.notna(all_close[col].iloc[max(0, i-lookback)]):
                    # MA filter
                    if all_close[col].iloc[i] < all_ma[col].iloc[i]:
                        continue  # Skip factors below MA
                    v0 = all_close[col].iloc[max(0, i-lookback)]
                    v1 = all_close[col].iloc[i]
                    if v0 > 0:
                        mom[col] = v1 / v0 - 1

            if len(mom) >= 1:
                n = min(top_n, len(mom))
                sorted_factors = sorted(mom.items(), key=lambda x: x[1], reverse=True)
                top_factors = [f[0] for f in sorted_factors[:n]]
                w = 1.0 / n
                current_weights = {f: w for f in top_factors}
            else:
                current_weights = {}  # All cash

        day_ret = 0.0
        for f, w in current_weights.items():
            if f in all_ret.columns and pd.notna(all_ret[f].iloc[i]):
                day_ret += w * all_ret[f].iloc[i]
        strat_ret.iloc[i] = day_ret

    return calc_metrics(strat_ret.dropna(), name)


def backtest_risk_parity(factor_prices, vol_window=60, rebal_freq=20, name='RiskParity'):
    """Risk parity: weight inversely proportional to volatility"""
    all_close = pd.DataFrame()
    for fname, fdf in factor_prices.items():
        all_close[fname] = fdf['close']
    all_close = all_close.dropna(how='all').ffill()
    all_ret = all_close.pct_change()

    strat_ret = pd.Series(0.0, index=all_ret.index)
    rebal_dates = all_ret.index[vol_window::rebal_freq]

    current_weights = {}
    for i, date in enumerate(all_ret.index):
        if i < vol_window:
            continue

        if date in rebal_dates or not current_weights:
            vols = {}
            for col in all_ret.columns:
                v = all_ret[col].iloc[max(0,i-vol_window):i].std()
                if v > 0 and not np.isnan(v):
                    vols[col] = v

            if vols:
                inv_vols = {k: 1.0/v for k, v in vols.items()}
                total = sum(inv_vols.values())
                current_weights = {k: v/total for k, v in inv_vols.items()}

        day_ret = 0.0
        for f, w in current_weights.items():
            if f in all_ret.columns and pd.notna(all_ret[f].iloc[i]):
                day_ret += w * all_ret[f].iloc[i]
        strat_ret.iloc[i] = day_ret

    return calc_metrics(strat_ret.dropna(), name)


def backtest_value_tilt_timing(factor_prices, bond_yield_df, name='ValueTilt'):
    """
    When market is cheap (high ERP), tilt towards value/dividend factors
    When market is expensive (low ERP), tilt towards growth/quality
    Use CSI300 PE as proxy (from price level)
    """
    # We approximate ERP using 沪深300 inverse PE vs bond yield
    if '沪深300' not in factor_prices or bond_yield_df is None:
        return None

    hs300 = factor_prices['沪深300'].copy()
    # Use rolling PE approximation: normalize price by 5-year trend as pseudo earnings yield
    # Actually, let's use a simpler approach: use price level relative to MA as a value signal

    all_close = pd.DataFrame()
    value_factors = []
    growth_factors = []
    for fname, fdf in factor_prices.items():
        all_close[fname] = fdf['close']
        if any(k in fname for k in ['价值', '红利', '低波', '基本面', '现金流']):
            value_factors.append(fname)
        elif any(k in fname for k in ['成长', '500', '1000']):
            growth_factors.append(fname)

    if not value_factors or not growth_factors:
        return None

    all_close = all_close.dropna(how='all').ffill()
    all_ret = all_close.pct_change()

    # Market cheapness signal: 沪深300 price relative to 3-year MA
    hs300_in = all_close['沪深300'].copy() if '沪深300' in all_close.columns else None
    if hs300_in is None:
        return None

    ma_long = hs300_in.rolling(750).mean()  # ~3 year MA
    cheapness = (ma_long / hs300_in - 1).clip(-0.5, 0.5)  # positive = cheap
    # Normalize to 0-1 for value weight
    value_weight = ((cheapness + 0.5) / 1.0).clip(0.3, 0.7)  # value always 30-70%

    strat_ret = pd.Series(0.0, index=all_ret.index)
    for i in range(1, len(all_ret)):
        if pd.isna(value_weight.iloc[i-1]):
            continue
        vw = value_weight.iloc[i-1]
        gw = 1.0 - vw

        val_ret = 0
        val_cnt = 0
        for f in value_factors:
            if f in all_ret.columns and pd.notna(all_ret[f].iloc[i]):
                val_ret += all_ret[f].iloc[i]
                val_cnt += 1

        grow_ret = 0
        grow_cnt = 0
        for f in growth_factors:
            if f in all_ret.columns and pd.notna(all_ret[f].iloc[i]):
                grow_ret += all_ret[f].iloc[i]
                grow_cnt += 1

        day_ret = 0
        if val_cnt > 0:
            day_ret += vw * (val_ret / val_cnt)
        if grow_cnt > 0:
            day_ret += gw * (grow_ret / grow_cnt)
        strat_ret.iloc[i] = day_ret

    return calc_metrics(strat_ret.dropna(), name)


# ============================================================
# 3. MAIN RESEARCH
# ============================================================

def run_research():
    print("=" * 70)
    print("A股因子择时+轮动组合策略研究")
    print("=" * 70)

    # Load data
    factors = load_all_factors()
    bond_yield = get_bond_yield()

    if len(factors) < 3:
        print("ERROR: Not enough factor data loaded!")
        return

    # Determine common start date (need enough history)
    # Use factors that start from at least 2010
    factor_starts = {k: v.index[0] for k, v in factors.items()}
    print(f"\nFactor start dates:")
    for k, v in sorted(factor_starts.items(), key=lambda x: x[1]):
        print(f"  {k}: {v.date()}")

    # Use 2010-01-01 as common start for most analyses
    common_start = pd.Timestamp('2010-06-01')  # Ensure enough MA warmup

    # Filter factors with enough data
    usable_factors = {}
    for name, df in factors.items():
        df_filtered = df[df.index >= common_start]
        if len(df_filtered) > 500:
            usable_factors[name] = df_filtered

    print(f"\nUsable factors (from {common_start.date()}): {list(usable_factors.keys())}")

    # Separate into "pure factor" indices (exclude benchmarks for rotation)
    benchmark_names = ['沪深300', '中证500', '中证1000']
    pure_factors = {k: v for k, v in usable_factors.items() if k not in benchmark_names}

    results = []

    # ==========================================
    # Part A: Buy-and-Hold per factor
    # ==========================================
    print("\n" + "=" * 60)
    print("Part A: Buy-and-Hold (各因子持有不动)")
    print("=" * 60)

    for name, df in sorted(usable_factors.items()):
        r = backtest_buy_hold(df, f"BuyHold_{name}")
        if r:
            results.append({**r, 'category': 'BuyHold'})
            print(f"  {name}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # ==========================================
    # Part B: Single Factor + MA Timing
    # ==========================================
    print("\n" + "=" * 60)
    print("Part B: 单因子 + 均线择时")
    print("=" * 60)

    ma_periods = [60, 120, 250]
    for name, df in sorted(pure_factors.items()):
        for ma in ma_periods:
            r = backtest_ma_timing(df, ma_period=ma, name=f"MA{ma}_{name}")
            if r:
                results.append({**r, 'category': f'MA{ma}_Timing'})
        # Also dual MA
        for fast, slow in [(20, 120), (60, 250)]:
            r = backtest_dual_ma(df, fast=fast, slow=slow, name=f"DualMA{fast}_{slow}_{name}")
            if r:
                results.append({**r, 'category': f'DualMA{fast}_{slow}'})

    # Print best per factor
    for name in sorted(pure_factors.keys()):
        factor_results = [r for r in results if name in r['name'] and 'MA' in r['name']]
        if factor_results:
            best = max(factor_results, key=lambda x: x['sharpe'])
            bh = next((r for r in results if r['name'] == f'BuyHold_{name}'), None)
            bh_sharpe = bh['sharpe'] if bh else 0
            print(f"  {name}: 最优={best['name']}, CAGR={best['cagr']:.1f}%, Sharpe={best['sharpe']:.2f} (vs BH Sharpe={bh_sharpe:.2f}), MaxDD={best['max_dd']:.1f}%")

    # ==========================================
    # Part C: Volatility Timing per factor
    # ==========================================
    print("\n" + "=" * 60)
    print("Part C: 单因子 + 波动率择时")
    print("=" * 60)

    for name, df in sorted(pure_factors.items()):
        for vt in [0.12, 0.15, 0.20]:
            r = backtest_vol_timing(df, vol_target=vt, name=f"VolTgt{int(vt*100)}%_{name}")
            if r:
                results.append({**r, 'category': f'VolTiming_{int(vt*100)}'})

    for name in sorted(pure_factors.keys()):
        factor_results = [r for r in results if name in r['name'] and 'VolTgt' in r['name']]
        if factor_results:
            best = max(factor_results, key=lambda x: x['sharpe'])
            print(f"  {name}: 最优={best['name']}, CAGR={best['cagr']:.1f}%, Sharpe={best['sharpe']:.2f}, MaxDD={best['max_dd']:.1f}%")

    # ==========================================
    # Part D: Factor Rotation Strategies
    # ==========================================
    print("\n" + "=" * 60)
    print("Part D: 因子轮动策略")
    print("=" * 60)

    # D1: Equal weight
    r = backtest_equal_weight(pure_factors, name='等权因子组合')
    if r:
        results.append({**r, 'category': 'Rotation'})
        print(f"  等权因子组合: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # D2: Equal weight + MA timing
    for ma in [60, 120, 250]:
        r = backtest_equal_weight_ma_timing(pure_factors, ma_period=ma, name=f'等权+MA{ma}')
        if r:
            results.append({**r, 'category': 'EW_MA_Timing'})
            print(f"  等权+MA{ma}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # D3: Momentum rotation
    for lb in [20, 60, 120, 250]:
        for tn in [1, 2, 3]:
            for rf in [20, 60]:
                r = backtest_momentum_rotation(pure_factors, lookback=lb, top_n=tn, rebal_freq=rf,
                                              name=f'动量轮动_LB{lb}_Top{tn}_R{rf}')
                if r:
                    results.append({**r, 'category': 'MomRotation'})

    # Best momentum rotation
    mom_results = [r for r in results if r['category'] == 'MomRotation']
    if mom_results:
        best = max(mom_results, key=lambda x: x['sharpe'])
        print(f"  最优动量轮动: {best['name']}, CAGR={best['cagr']:.1f}%, Sharpe={best['sharpe']:.2f}, MaxDD={best['max_dd']:.1f}%")
        # Top 5
        top5 = sorted(mom_results, key=lambda x: x['sharpe'], reverse=True)[:5]
        for r in top5:
            print(f"    {r['name']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # D4: Inverse momentum (reversal) rotation
    for lb in [20, 60, 120]:
        for tn in [1, 2]:
            r = backtest_inverse_momentum_rotation(pure_factors, lookback=lb, top_n=tn, rebal_freq=20,
                                                   name=f'反转轮动_LB{lb}_Top{tn}')
            if r:
                results.append({**r, 'category': 'ReversalRotation'})

    rev_results = [r for r in results if r['category'] == 'ReversalRotation']
    if rev_results:
        best = max(rev_results, key=lambda x: x['sharpe'])
        print(f"  最优反转轮动: {best['name']}, CAGR={best['cagr']:.1f}%, Sharpe={best['sharpe']:.2f}, MaxDD={best['max_dd']:.1f}%")

    # D5: Risk parity
    r = backtest_risk_parity(pure_factors, name='风险平价')
    if r:
        results.append({**r, 'category': 'RiskParity'})
        print(f"  风险平价: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # D6: Momentum rotation + MA filter
    for lb in [60, 120]:
        for tn in [2, 3]:
            for ma in [120, 250]:
                r = backtest_momentum_rotation_with_ma(pure_factors, lookback=lb, top_n=tn, rebal_freq=20,
                                                       ma_period=ma, name=f'动量+MA{ma}_LB{lb}_Top{tn}')
                if r:
                    results.append({**r, 'category': 'MomRot_MA'})

    momma_results = [r for r in results if r['category'] == 'MomRot_MA']
    if momma_results:
        best = max(momma_results, key=lambda x: x['sharpe'])
        print(f"  最优动量+MA: {best['name']}, CAGR={best['cagr']:.1f}%, Sharpe={best['sharpe']:.2f}, MaxDD={best['max_dd']:.1f}%")

    # D7: Value tilt timing
    r = backtest_value_tilt_timing(usable_factors, bond_yield, name='价值成长择时轮动')
    if r:
        results.append({**r, 'category': 'ValueTilt'})
        print(f"  价值成长择时: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # ==========================================
    # Part E: Summary & Rankings
    # ==========================================
    print("\n" + "=" * 60)
    print("SUMMARY: TOP 20 STRATEGIES BY SHARPE")
    print("=" * 60)

    top20 = sorted(results, key=lambda x: x['sharpe'], reverse=True)[:20]
    print(f"{'Rank':<5} {'Strategy':<40} {'CAGR':>6} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7}")
    print("-" * 75)
    for i, r in enumerate(top20, 1):
        print(f"{i:<5} {r['name']:<40} {r['cagr']:>5.1f}% {r['sharpe']:>7.3f} {r['max_dd']:>6.1f}% {r['calmar']:>7.3f}")

    # Also show best in each category
    print("\n" + "=" * 60)
    print("BEST PER CATEGORY")
    print("=" * 60)

    categories = sorted(set(r['category'] for r in results))
    for cat in categories:
        cat_results = [r for r in results if r['category'] == cat]
        best = max(cat_results, key=lambda x: x['sharpe'])
        print(f"  {cat:<25} {best['name']:<40} CAGR={best['cagr']:.1f}%, Sharpe={best['sharpe']:.3f}, MaxDD={best['max_dd']:.1f}%")

    # ==========================================
    # Part F: In-Sample vs Out-of-Sample
    # ==========================================
    print("\n" + "=" * 60)
    print("IN-SAMPLE (2010-2021) vs OUT-OF-SAMPLE (2022-2026)")
    print("=" * 60)

    # Re-run top strategies with IS/OOS split
    is_end = pd.Timestamp('2021-12-31')
    oos_start = pd.Timestamp('2022-01-01')

    # Test top strategies from each category
    best_per_cat = {}
    for cat in categories:
        cat_results = [r for r in results if r['category'] == cat]
        if cat_results:
            best_per_cat[cat] = max(cat_results, key=lambda x: x['sharpe'])

    # For IS/OOS, re-run key strategies
    print(f"\n{'Strategy':<35} {'IS_CAGR':>8} {'IS_Sharpe':>10} {'OOS_CAGR':>9} {'OOS_Sharpe':>11} {'IS_DD':>7} {'OOS_DD':>7}")
    print("-" * 90)

    # Buy and hold benchmark
    for name, df in [('沪深300', usable_factors.get('沪深300')), ('中证红利', pure_factors.get('中证红利')), ('红利指数', pure_factors.get('红利指数'))]:
        if df is None:
            continue
        is_r = backtest_buy_hold(df[df.index <= is_end], f'BH_{name}_IS')
        oos_r = backtest_buy_hold(df[df.index >= oos_start], f'BH_{name}_OOS')
        if is_r and oos_r:
            print(f"  BuyHold_{name:<25} {is_r['cagr']:>7.1f}% {is_r['sharpe']:>10.3f} {oos_r['cagr']:>8.1f}% {oos_r['sharpe']:>11.3f} {is_r['max_dd']:>6.1f}% {oos_r['max_dd']:>6.1f}%")

    # MA timing on top factors
    for name in ['中证红利', '红利指数', '300价值']:
        if name not in pure_factors:
            continue
        df = pure_factors[name]
        for ma in [120, 250]:
            is_r = backtest_ma_timing(df[df.index <= is_end], ma_period=ma, name=f'MA{ma}_{name}_IS')
            oos_r = backtest_ma_timing(df[df.index >= oos_start], ma_period=ma, name=f'MA{ma}_{name}_OOS')
            if is_r and oos_r:
                print(f"  MA{ma}_{name:<25} {is_r['cagr']:>7.1f}% {is_r['sharpe']:>10.3f} {oos_r['cagr']:>8.1f}% {oos_r['sharpe']:>11.3f} {is_r['max_dd']:>6.1f}% {oos_r['max_dd']:>6.1f}%")

    # Equal weight + MA
    is_factors = {k: v[v.index <= is_end] for k, v in pure_factors.items()}
    oos_factors = {k: v[v.index >= oos_start] for k, v in pure_factors.items()}

    r_is = backtest_equal_weight(is_factors, '等权_IS')
    r_oos = backtest_equal_weight(oos_factors, '等权_OOS')
    if r_is and r_oos:
        print(f"  {'等权因子组合':<35} {r_is['cagr']:>7.1f}% {r_is['sharpe']:>10.3f} {r_oos['cagr']:>8.1f}% {r_oos['sharpe']:>11.3f} {r_is['max_dd']:>6.1f}% {r_oos['max_dd']:>6.1f}%")

    for ma in [120, 250]:
        r_is = backtest_equal_weight_ma_timing(is_factors, ma_period=ma, name=f'等权+MA{ma}_IS')
        r_oos = backtest_equal_weight_ma_timing(oos_factors, ma_period=ma, name=f'等权+MA{ma}_OOS')
        if r_is and r_oos:
            print(f"  {'等权+MA' + str(ma):<35} {r_is['cagr']:>7.1f}% {r_is['sharpe']:>10.3f} {r_oos['cagr']:>8.1f}% {r_oos['sharpe']:>11.3f} {r_is['max_dd']:>6.1f}% {r_oos['max_dd']:>6.1f}%")

    # Risk parity
    r_is = backtest_risk_parity(is_factors, name='风险平价_IS')
    r_oos = backtest_risk_parity(oos_factors, name='风险平价_OOS')
    if r_is and r_oos:
        print(f"  {'风险平价':<35} {r_is['cagr']:>7.1f}% {r_is['sharpe']:>10.3f} {r_oos['cagr']:>8.1f}% {r_oos['sharpe']:>11.3f} {r_is['max_dd']:>6.1f}% {r_oos['max_dd']:>6.1f}%")

    # Save results
    output = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'common_start': str(common_start.date()),
        'factors_used': list(usable_factors.keys()),
        'pure_factors': list(pure_factors.keys()),
        'total_strategies': len(results),
        'all_results': results,
        'top20_by_sharpe': top20,
    }

    out_path = os.path.join(OUT_DIR, 'factor_timing_combo_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == '__main__':
    run_research()
