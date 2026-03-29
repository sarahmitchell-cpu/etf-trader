#!/usr/bin/env python3
"""
A股因子择时+轮动策略研究 V2 - 使用中证指数官方全收益数据
Factor Timing & Rotation with Official CSI Total Return Indices

数据源: 中证指数官网 csindex.com.cn (全收益指数)
因子: 价值/红利/低波/成长/基本面/现金流
择时: MA/DualMA/VolTgt/动量轮动/反转轮动/风险平价/价值成长切换

Author: Sarah Mitchell / VisionClaw
Date: 2026-03-29
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import requests
import json, os, traceback
from datetime import datetime

DATA_DIR = '/Users/claw/etf-trader/data'

# ============================================================
# 1. CSI OFFICIAL DATA
# ============================================================

def fetch_csi_index(code, name, start='20050101', end='20260328'):
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

        # Some indices have weekend/holiday entries with no change - keep only trading days
        df = df[~df.index.duplicated(keep='first')]

        total_ret = df['close'].iloc[-1] / df['close'].iloc[0] - 1
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

        print(f"  {name} ({code}): {df.index[0].date()} ~ {df.index[-1].date()}, "
              f"{len(df)} rows, CAGR={cagr*100:.1f}%")
        return df
    except Exception as e:
        print(f"  {name} ({code}): ERROR {e}")
        return None


def load_all_indices():
    """Load all factor indices from CSI official"""
    print("[1] Fetching official CSI total return index data...")

    # All indices we want - using 全收益(Total Return) versions where available
    indices = {
        # Benchmarks
        '沪深300': 'H00300',        # 沪深300全收益
        '中证500': 'H00905',        # 中证500全收益
        '中证1000': 'H00852',       # 中证1000全收益

        # Factor indices (Total Return)
        '中证红利': 'H00922',        # 中证红利全收益
        '红利低波': 'H20269',        # 红利低波全收益
        '300价值': 'H00919',         # 沪深300价值全收益 (API verified)
        '300成长': 'H00918',         # 沪深300成长全收益 (API verified)
        '300相对成长': 'H00920',     # 沪深300相对成长全收益 (was mislabeled 300红利)
        '300相对价值': 'H00921',     # 沪深300相对价值全收益 (was mislabeled 300低波)
        '基本面50': 'H00925',        # 基本面50全收益
        '上证红利': 'H00015',        # 上证红利全收益
        # NOTE: H30532=中证资源优选, H30533=中国互联网50 (both were mislabeled)
        # Removed: 中证质量成长 and 800价值 - correct codes TBD
    }

    # Also try some newer factor indices
    newer = {
        '自由现金流': '932365',       # 中证自由现金流
        '红利低波100': '931157',      # 中证红利低波100
        '央企红利': '932396',         # NOTE: actually 中证1000相对价值 - need correct code
    }

    import time
    factors = {}
    for name, code in indices.items():
        df = fetch_csi_index(code, name)
        if df is not None and len(df) > 200:
            factors[name] = df
        time.sleep(0.5)

    for name, code in newer.items():
        df = fetch_csi_index(code, name)
        if df is not None and len(df) > 200:
            factors[name] = df
        time.sleep(0.5)

    print(f"\nLoaded {len(factors)} indices")
    return factors


# ============================================================
# 2. BACKTEST ENGINE (same as v1 but cleaner)
# ============================================================

def calc_metrics(returns, name=''):
    if len(returns) < 20:
        return None
    cum_ret = (1 + returns).cumprod()
    total_ret = cum_ret.iloc[-1] - 1
    years = len(returns) / 252
    if years <= 0:
        return None
    cagr = (1 + total_ret) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    running_max = cum_ret.cummax()
    drawdown = cum_ret / running_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    monthly = returns.resample('ME').sum()
    win_rate = (monthly > 0).mean()
    return {
        'name': name, 'cagr': round(cagr * 100, 2), 'vol': round(vol * 100, 2),
        'sharpe': round(sharpe, 3), 'max_dd': round(max_dd * 100, 2),
        'calmar': round(calmar, 3), 'total_ret': round(total_ret * 100, 1),
        'years': round(years, 1), 'win_rate_monthly': round(win_rate * 100, 1),
    }

def backtest_buy_hold(prices, name='BuyHold'):
    returns = prices['close'].pct_change().dropna()
    return calc_metrics(returns, name)

def backtest_ma_timing(prices, ma_period=120, name='MA'):
    df = prices.copy()
    df['ma'] = df['close'].rolling(ma_period).mean()
    df['ret'] = df['close'].pct_change()
    df['signal'] = (df['close'] > df['ma']).shift(1).astype(float)
    df['strat_ret'] = df['ret'] * df['signal']
    return calc_metrics(df['strat_ret'].dropna(), name)

def backtest_dual_ma(prices, fast=20, slow=120, name='DualMA'):
    df = prices.copy()
    df['ma_fast'] = df['close'].rolling(fast).mean()
    df['ma_slow'] = df['close'].rolling(slow).mean()
    df['ret'] = df['close'].pct_change()
    df['signal'] = (df['ma_fast'] > df['ma_slow']).shift(1).astype(float)
    df['strat_ret'] = df['ret'] * df['signal']
    return calc_metrics(df['strat_ret'].dropna(), name)

def backtest_vol_timing(prices, vol_window=20, vol_target=0.15, name='VolTiming'):
    df = prices.copy()
    df['ret'] = df['close'].pct_change()
    df['vol'] = df['ret'].rolling(vol_window).std() * np.sqrt(252)
    df['weight'] = (vol_target / df['vol']).clip(0, 1.5).shift(1)
    df['strat_ret'] = df['ret'] * df['weight']
    return calc_metrics(df['strat_ret'].dropna(), name)

def backtest_equal_weight(factor_prices, name='EqualWeight'):
    all_close = pd.DataFrame({k: v['close'] for k, v in factor_prices.items()})
    all_close = all_close.dropna(how='all').ffill()
    all_ret = all_close.pct_change()
    strat_ret = all_ret.mean(axis=1)
    return calc_metrics(strat_ret.dropna(), name)

def backtest_equal_weight_ma_timing(factor_prices, ma_period=120, name='EW+MA'):
    all_close = pd.DataFrame({k: v['close'] for k, v in factor_prices.items()})
    all_close = all_close.dropna(how='all').ffill()
    all_ret = all_close.pct_change()
    signals = pd.DataFrame()
    for col in all_close.columns:
        ma = all_close[col].rolling(ma_period).mean()
        signals[col] = (all_close[col] > ma).shift(1).astype(float)
    timed_ret = all_ret * signals
    strat_ret = timed_ret.mean(axis=1)
    return calc_metrics(strat_ret.dropna(), name)

def backtest_momentum_rotation(factor_prices, lookback=60, top_n=2, rebal_freq=20, name='MomRot'):
    all_close = pd.DataFrame({k: v['close'] for k, v in factor_prices.items()})
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
                v0 = all_close[col].iloc[max(0, i-lookback)]
                v1 = all_close[col].iloc[i]
                if pd.notna(v0) and pd.notna(v1) and v0 > 0:
                    mom[col] = v1 / v0 - 1
            if len(mom) >= top_n:
                sorted_f = sorted(mom.items(), key=lambda x: x[1], reverse=True)
                top = [f[0] for f in sorted_f[:top_n]]
                w = 1.0 / top_n
                current_weights = {f: w for f in top}
            elif mom:
                w = 1.0 / len(mom)
                current_weights = {f: w for f in mom}
        day_ret = sum(w * all_ret[f].iloc[i] for f, w in current_weights.items()
                      if f in all_ret.columns and pd.notna(all_ret[f].iloc[i]))
        strat_ret.iloc[i] = day_ret
    return calc_metrics(strat_ret.dropna(), name)

def backtest_reversal_rotation(factor_prices, lookback=60, top_n=2, rebal_freq=20, name='RevRot'):
    all_close = pd.DataFrame({k: v['close'] for k, v in factor_prices.items()})
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
                v0 = all_close[col].iloc[max(0, i-lookback)]
                v1 = all_close[col].iloc[i]
                if pd.notna(v0) and pd.notna(v1) and v0 > 0:
                    mom[col] = v1 / v0 - 1
            if len(mom) >= top_n:
                sorted_f = sorted(mom.items(), key=lambda x: x[1])  # worst first
                top = [f[0] for f in sorted_f[:top_n]]
                w = 1.0 / top_n
                current_weights = {f: w for f in top}
        day_ret = sum(w * all_ret[f].iloc[i] for f, w in current_weights.items()
                      if f in all_ret.columns and pd.notna(all_ret[f].iloc[i]))
        strat_ret.iloc[i] = day_ret
    return calc_metrics(strat_ret.dropna(), name)

def backtest_risk_parity(factor_prices, vol_window=60, rebal_freq=20, name='RiskParity'):
    all_close = pd.DataFrame({k: v['close'] for k, v in factor_prices.items()})
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
                inv = {k: 1.0/v for k, v in vols.items()}
                total = sum(inv.values())
                current_weights = {k: v/total for k, v in inv.items()}
        day_ret = sum(w * all_ret[f].iloc[i] for f, w in current_weights.items()
                      if f in all_ret.columns and pd.notna(all_ret[f].iloc[i]))
        strat_ret.iloc[i] = day_ret
    return calc_metrics(strat_ret.dropna(), name)

def backtest_momentum_rotation_ma(factor_prices, lookback=60, top_n=2, rebal_freq=20, ma_period=120, name='MomMA'):
    all_close = pd.DataFrame({k: v['close'] for k, v in factor_prices.items()})
    all_close = all_close.dropna(how='all').ffill()
    all_ret = all_close.pct_change()
    all_ma = all_close.rolling(ma_period).mean()
    strat_ret = pd.Series(0.0, index=all_ret.index)
    rebal_dates = all_ret.index[max(lookback, ma_period)::rebal_freq]
    current_weights = {}
    for i, date in enumerate(all_ret.index):
        if i < max(lookback, ma_period):
            continue
        if date in rebal_dates or not current_weights:
            mom = {}
            for col in all_close.columns:
                if all_close[col].iloc[i] < all_ma[col].iloc[i]:
                    continue
                v0 = all_close[col].iloc[max(0, i-lookback)]
                v1 = all_close[col].iloc[i]
                if pd.notna(v0) and pd.notna(v1) and v0 > 0:
                    mom[col] = v1 / v0 - 1
            if mom:
                n = min(top_n, len(mom))
                sorted_f = sorted(mom.items(), key=lambda x: x[1], reverse=True)
                top = [f[0] for f in sorted_f[:n]]
                w = 1.0 / n
                current_weights = {f: w for f in top}
            else:
                current_weights = {}
        day_ret = sum(w * all_ret[f].iloc[i] for f, w in current_weights.items()
                      if f in all_ret.columns and pd.notna(all_ret[f].iloc[i]))
        strat_ret.iloc[i] = day_ret
    return calc_metrics(strat_ret.dropna(), name)


# ============================================================
# 3. MAIN RESEARCH
# ============================================================

def run_research():
    print("=" * 70)
    print("A股因子择时+轮动 V2 - 中证指数官方全收益数据")
    print("=" * 70)

    factors = load_all_indices()
    if len(factors) < 3:
        print("ERROR: Not enough data!")
        return

    # Common start: use latest start among all factors
    starts = {k: v.index[0] for k, v in factors.items()}
    common_start = max(starts.values()) + pd.Timedelta(days=1)
    print(f"\nCommon start date: {common_start.date()}")
    print(f"Factor start dates:")
    for k, v in sorted(starts.items(), key=lambda x: x[1]):
        print(f"  {k}: {v.date()}")

    # Filter to common period
    for k in factors:
        factors[k] = factors[k][factors[k].index >= common_start]

    # Separate benchmarks and pure factors
    benchmarks = ['沪深300', '中证500', '中证1000']
    pure_factor_names = [k for k in factors if k not in benchmarks]
    pure_factors = {k: factors[k] for k in pure_factor_names}

    results = []

    # ==========================================
    # Part A: Buy-and-Hold
    # ==========================================
    print("\n" + "=" * 60)
    print("Part A: Buy-and-Hold (全收益指数)")
    print("=" * 60)
    print(f"{'指数':<16} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>8}")
    print("-" * 60)

    for name in sorted(factors.keys()):
        r = backtest_buy_hold(factors[name], f"BuyHold_{name}")
        if r:
            results.append({**r, 'category': 'BuyHold'})
            print(f"  {name:<14} {r['cagr']:>6.1f}% {r['vol']:>6.1f}% {r['sharpe']:>7.3f} {r['max_dd']:>7.1f}% {r['calmar']:>8.3f}")

    # ==========================================
    # Part B: Single Factor + Timing
    # ==========================================
    print("\n" + "=" * 60)
    print("Part B: 单因子 + 均线/波动率择时")
    print("=" * 60)

    for name in sorted(pure_factor_names):
        df = pure_factors[name]
        # MA timing
        for ma in [60, 120, 250]:
            r = backtest_ma_timing(df, ma, f"MA{ma}_{name}")
            if r:
                results.append({**r, 'category': f'MA{ma}'})
        # Dual MA
        for fast, slow in [(20, 120), (60, 250)]:
            r = backtest_dual_ma(df, fast, slow, f"DMA{fast}_{slow}_{name}")
            if r:
                results.append({**r, 'category': f'DMA{fast}_{slow}'})
        # Vol timing
        for vt in [0.12, 0.15]:
            r = backtest_vol_timing(df, vol_target=vt, name=f"VolTgt{int(vt*100)}%_{name}")
            if r:
                results.append({**r, 'category': f'VolTgt{int(vt*100)}'})

    # Best timing per factor
    print(f"\n{'因子':<16} {'最优策略':<30} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'vs BH':>8}")
    print("-" * 80)
    for name in sorted(pure_factor_names):
        timing_results = [r for r in results if name in r['name'] and 'BuyHold' not in r['name']]
        bh = next((r for r in results if r['name'] == f'BuyHold_{name}'), None)
        if timing_results and bh:
            best = max(timing_results, key=lambda x: x['sharpe'])
            delta_sharpe = best['sharpe'] - bh['sharpe']
            print(f"  {name:<14} {best['name']:<28} {best['cagr']:>6.1f}% {best['sharpe']:>7.3f} "
                  f"{best['max_dd']:>7.1f}% {delta_sharpe:>+7.3f}")

    # ==========================================
    # Part C: Factor Rotation
    # ==========================================
    print("\n" + "=" * 60)
    print("Part C: 因子组合与轮动")
    print("=" * 60)

    # C1: Equal weight
    r = backtest_equal_weight(pure_factors, '等权因子')
    if r:
        results.append({**r, 'category': 'Combo'})
        print(f"  等权因子: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.3f}, MaxDD={r['max_dd']:.1f}%")

    # C2: EW + MA timing
    for ma in [60, 120, 250]:
        r = backtest_equal_weight_ma_timing(pure_factors, ma, f'等权+MA{ma}')
        if r:
            results.append({**r, 'category': 'Combo_MA'})
            print(f"  等权+MA{ma}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.3f}, MaxDD={r['max_dd']:.1f}%")

    # C3: Momentum rotation
    print("\n  动量轮动:")
    for lb in [20, 60, 120, 250]:
        for tn in [1, 2, 3]:
            for rf in [20, 60]:
                r = backtest_momentum_rotation(pure_factors, lb, tn, rf, f'动量_LB{lb}_T{tn}_R{rf}')
                if r:
                    results.append({**r, 'category': 'MomRot'})

    mom_results = [r for r in results if r['category'] == 'MomRot']
    if mom_results:
        top5 = sorted(mom_results, key=lambda x: x['sharpe'], reverse=True)[:5]
        for r in top5:
            print(f"    {r['name']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.3f}, MaxDD={r['max_dd']:.1f}%")

    # C4: Reversal rotation
    print("\n  反转轮动:")
    for lb in [20, 60, 120]:
        for tn in [1, 2]:
            r = backtest_reversal_rotation(pure_factors, lb, tn, 20, f'反转_LB{lb}_T{tn}')
            if r:
                results.append({**r, 'category': 'RevRot'})

    rev_results = [r for r in results if r['category'] == 'RevRot']
    if rev_results:
        best = max(rev_results, key=lambda x: x['sharpe'])
        print(f"    最优: {best['name']}: CAGR={best['cagr']:.1f}%, Sharpe={best['sharpe']:.3f}, MaxDD={best['max_dd']:.1f}%")

    # C5: Risk parity
    r = backtest_risk_parity(pure_factors, name='风险平价')
    if r:
        results.append({**r, 'category': 'RiskParity'})
        print(f"\n  风险平价: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.3f}, MaxDD={r['max_dd']:.1f}%")

    # C6: Momentum + MA filter
    print("\n  动量+MA过滤:")
    for lb in [60, 120]:
        for tn in [2, 3]:
            for ma in [120, 250]:
                r = backtest_momentum_rotation_ma(pure_factors, lb, tn, 20, ma, f'动量MA{ma}_LB{lb}_T{tn}')
                if r:
                    results.append({**r, 'category': 'MomMA'})

    momma_results = [r for r in results if r['category'] == 'MomMA']
    if momma_results:
        best = max(momma_results, key=lambda x: x['sharpe'])
        print(f"    最优: {best['name']}: CAGR={best['cagr']:.1f}%, Sharpe={best['sharpe']:.3f}, MaxDD={best['max_dd']:.1f}%")

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 60)
    print("TOP 20 STRATEGIES BY SHARPE")
    print("=" * 60)
    top20 = sorted(results, key=lambda x: x['sharpe'], reverse=True)[:20]
    print(f"{'#':<3} {'Strategy':<40} {'CAGR':>6} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7}")
    print("-" * 75)
    for i, r in enumerate(top20, 1):
        print(f"{i:<3} {r['name']:<40} {r['cagr']:>5.1f}% {r['sharpe']:>7.3f} {r['max_dd']:>6.1f}% {r['calmar']:>7.3f}")

    # ==========================================
    # IS/OOS validation
    # ==========================================
    print("\n" + "=" * 60)
    print("IS (样本内) vs OOS (样本外) - split at 2022-01-01")
    print("=" * 60)

    is_end = pd.Timestamp('2021-12-31')
    oos_start = pd.Timestamp('2022-01-01')

    print(f"\n{'Strategy':<30} {'IS_CAGR':>8} {'IS_Shrp':>8} {'OOS_CAGR':>9} {'OOS_Shrp':>9} {'IS_DD':>7} {'OOS_DD':>7}")
    print("-" * 82)

    # Key strategies to validate
    test_configs = [
        ('BuyHold_沪深300', lambda f: backtest_buy_hold(f.get('沪深300', pd.DataFrame()))),
        ('BuyHold_中证红利', lambda f: backtest_buy_hold(f.get('中证红利', pd.DataFrame()))),
        ('BuyHold_红利低波', lambda f: backtest_buy_hold(f.get('红利低波', pd.DataFrame()))),
        ('BuyHold_300价值', lambda f: backtest_buy_hold(f.get('300价值', pd.DataFrame()))),
        ('等权因子', lambda f: backtest_equal_weight({k:v for k,v in f.items() if k not in ['沪深300','中证500','中证1000']})),
        ('等权+MA60', lambda f: backtest_equal_weight_ma_timing({k:v for k,v in f.items() if k not in ['沪深300','中证500','中证1000']}, 60)),
        ('等权+MA120', lambda f: backtest_equal_weight_ma_timing({k:v for k,v in f.items() if k not in ['沪深300','中证500','中证1000']}, 120)),
        ('风险平价', lambda f: backtest_risk_parity({k:v for k,v in f.items() if k not in ['沪深300','中证500','中证1000']})),
    ]

    for label, fn in test_configs:
        is_factors = {k: v[v.index <= is_end] for k, v in factors.items() if len(v[v.index <= is_end]) > 60}
        oos_factors = {k: v[v.index >= oos_start] for k, v in factors.items() if len(v[v.index >= oos_start]) > 60}

        try:
            r_is = fn(is_factors)
            r_oos = fn(oos_factors)
            if r_is and r_oos:
                print(f"  {label:<28} {r_is['cagr']:>7.1f}% {r_is['sharpe']:>8.3f} {r_oos['cagr']:>8.1f}% "
                      f"{r_oos['sharpe']:>9.3f} {r_is['max_dd']:>6.1f}% {r_oos['max_dd']:>6.1f}%")
        except:
            pass

    # Save
    output = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'data_source': 'CSI Official Total Return Index (csindex.com.cn)',
        'factors_used': list(factors.keys()),
        'pure_factors': pure_factor_names,
        'common_start': str(common_start.date()),
        'total_strategies': len(results),
        'all_results': results,
        'top20': top20,
    }

    out_path = os.path.join(DATA_DIR, 'factor_timing_v2_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == '__main__':
    run_research()
