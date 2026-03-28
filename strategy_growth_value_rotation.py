#!/usr/bin/env python3
"""
沪深300成长 vs 价值 轮动策略研究
使用 000918(300成长) 和 000919(300价值) 的全收益模拟

测试:
1. 固定周期轮动 (1/2/3/4/6/8/12/24/52周)
2. 动量轮动 (过去N周谁涨得好就持谁)
3. 均线轮动 (成长/价值相对强弱的MA交叉)
4. RSI轮动 (相对强弱RSI)
5. 均值回归 (谁跌得多就买谁)
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def load_data():
    """Load daily price data for growth and value indices."""
    growth = pd.read_csv(os.path.join(DATA_DIR, '000918_daily.csv'))
    value = pd.read_csv(os.path.join(DATA_DIR, '000919_daily.csv'))

    growth['日期'] = pd.to_datetime(growth['日期'])
    value['日期'] = pd.to_datetime(value['日期'])

    growth = growth.set_index('日期').sort_index()
    value = value.set_index('日期').sort_index()

    # Align dates
    common_dates = growth.index.intersection(value.index)
    g = growth.loc[common_dates, '收盘'].copy()
    v = value.loc[common_dates, '收盘'].copy()

    print(f"Data range: {common_dates[0].date()} ~ {common_dates[-1].date()}, {len(common_dates)} trading days")
    return g, v

def calc_metrics(pv, benchmark=None):
    """Calculate strategy metrics."""
    if pv is None or len(pv) < 50:
        return None
    years = (pv.index[-1] - pv.index[0]).days / 365.25
    if years < 1:
        return None

    total_ret = pv.iloc[-1] / pv.iloc[0]
    cagr = total_ret ** (1/years) - 1
    cummax = pv.cummax()
    dd = pv / cummax - 1
    mdd = dd.min()
    daily_rets = pv.pct_change().dropna()
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    # Annual returns
    annual = {}
    for year in range(pv.index[0].year, pv.index[-1].year + 1):
        yr = pv[pv.index.year == year]
        if len(yr) >= 10:
            annual[str(year)] = round((yr.iloc[-1] / yr.iloc[0] - 1) * 100, 1)

    result = {
        'cagr': round(cagr * 100, 1),
        'mdd': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'annual': annual,
        'total_return': round((total_ret - 1) * 100, 1),
    }

    # Excess return vs benchmark
    if benchmark is not None:
        bm_ret = benchmark.iloc[-1] / benchmark.iloc[0]
        bm_cagr = bm_ret ** (1/years) - 1
        result['excess_cagr'] = round((cagr - bm_cagr) * 100, 1)

    return result

# ============================================================
# Strategy 1: Buy and Hold benchmarks
# ============================================================
def buy_and_hold(g, v, start):
    """Pure buy-and-hold benchmarks."""
    g = g[g.index >= start]
    v = v[v.index >= start]
    results = {}

    # Growth only
    pv_g = g / g.iloc[0]
    results['纯持有成长'] = calc_metrics(pv_g)

    # Value only
    pv_v = v / v.iloc[0]
    results['纯持有价值'] = calc_metrics(pv_v)

    # 50/50 rebalanced monthly
    dates = g.index
    pv = pd.Series(1.0, index=dates)
    for i in range(1, len(dates)):
        g_ret = g.iloc[i] / g.iloc[i-1] - 1
        v_ret = v.iloc[i] / v.iloc[i-1] - 1
        pv.iloc[i] = pv.iloc[i-1] * (1 + 0.5 * g_ret + 0.5 * v_ret)
    results['50/50均配'] = calc_metrics(pv)

    return results

# ============================================================
# Strategy 2: Fixed period rotation (momentum-based)
# ============================================================
def fixed_period_rotation(g, v, period_days, lookback_days, start):
    """
    Every period_days, compare past lookback_days return of growth vs value.
    Hold the winner for the next period.
    """
    g = g[g.index >= start].copy()
    v = v[v.index >= start].copy()
    dates = g.index

    pv = pd.Series(1.0, index=dates)
    holding = None  # 'growth' or 'value'
    last_rebal = 0

    for i in range(1, len(dates)):
        # Rebalance?
        if i - last_rebal >= period_days or holding is None:
            # Look back
            lb_start = max(0, i - lookback_days)
            g_ret = g.iloc[i] / g.iloc[lb_start] - 1
            v_ret = v.iloc[i] / v.iloc[lb_start] - 1
            holding = 'growth' if g_ret > v_ret else 'value'
            last_rebal = i

        # Daily return
        if holding == 'growth':
            ret = g.iloc[i] / g.iloc[i-1] - 1
        else:
            ret = v.iloc[i] / v.iloc[i-1] - 1
        pv.iloc[i] = pv.iloc[i-1] * (1 + ret)

    return pv

# ============================================================
# Strategy 3: MA crossover on relative strength
# ============================================================
def ma_rotation(g, v, fast_ma, slow_ma, start):
    """
    Compute growth/value ratio. When fast MA > slow MA, hold growth; else hold value.
    """
    ratio = g / v
    fast = ratio.rolling(fast_ma).mean()
    slow = ratio.rolling(slow_ma).mean()

    g = g[g.index >= start].copy()
    v = v[v.index >= start].copy()
    fast = fast[fast.index >= start]
    slow = slow[slow.index >= start]
    dates = g.index

    pv = pd.Series(1.0, index=dates)
    for i in range(1, len(dates)):
        dt = dates[i]
        if pd.isna(fast.loc[dt]) or pd.isna(slow.loc[dt]):
            # Not enough data, hold 50/50
            g_ret = g.iloc[i] / g.iloc[i-1] - 1
            v_ret = v.iloc[i] / v.iloc[i-1] - 1
            ret = 0.5 * g_ret + 0.5 * v_ret
        elif fast.loc[dt] > slow.loc[dt]:
            ret = g.iloc[i] / g.iloc[i-1] - 1
        else:
            ret = v.iloc[i] / v.iloc[i-1] - 1
        pv.iloc[i] = pv.iloc[i-1] * (1 + ret)

    return pv

# ============================================================
# Strategy 4: RSI on relative strength
# ============================================================
def rsi_rotation(g, v, rsi_period, threshold_high, threshold_low, start):
    """
    Compute RSI of growth/value ratio.
    RSI > threshold_high: growth is overbought -> switch to value
    RSI < threshold_low: growth is oversold -> switch to growth
    In between: keep current position.
    """
    ratio = g / v
    delta = ratio.diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    g = g[g.index >= start].copy()
    v = v[v.index >= start].copy()
    rsi = rsi[rsi.index >= start]
    dates = g.index

    pv = pd.Series(1.0, index=dates)
    holding = 'growth'  # default

    for i in range(1, len(dates)):
        dt = dates[i]
        rsi_val = rsi.loc[dt] if dt in rsi.index and not pd.isna(rsi.loc[dt]) else 50

        # Mean reversion: if growth RSI high, switch to value; if low, switch to growth
        if rsi_val > threshold_high:
            holding = 'value'
        elif rsi_val < threshold_low:
            holding = 'growth'

        if holding == 'growth':
            ret = g.iloc[i] / g.iloc[i-1] - 1
        else:
            ret = v.iloc[i] / v.iloc[i-1] - 1
        pv.iloc[i] = pv.iloc[i-1] * (1 + ret)

    return pv

# ============================================================
# Strategy 5: Momentum RSI (trend-following version)
# ============================================================
def momentum_rsi_rotation(g, v, rsi_period, threshold_high, threshold_low, start):
    """
    Trend-following RSI: RSI high = growth strong -> hold growth.
    RSI low = growth weak -> hold value.
    """
    ratio = g / v
    delta = ratio.diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    g = g[g.index >= start].copy()
    v = v[v.index >= start].copy()
    rsi = rsi[rsi.index >= start]
    dates = g.index

    pv = pd.Series(1.0, index=dates)
    holding = 'growth'

    for i in range(1, len(dates)):
        dt = dates[i]
        rsi_val = rsi.loc[dt] if dt in rsi.index and not pd.isna(rsi.loc[dt]) else 50

        # Momentum: RSI high = growth strong -> stay growth
        if rsi_val > threshold_high:
            holding = 'growth'
        elif rsi_val < threshold_low:
            holding = 'value'

        if holding == 'growth':
            ret = g.iloc[i] / g.iloc[i-1] - 1
        else:
            ret = v.iloc[i] / v.iloc[i-1] - 1
        pv.iloc[i] = pv.iloc[i-1] * (1 + ret)

    return pv

# ============================================================
# Strategy 6: Breakout rotation
# ============================================================
def breakout_rotation(g, v, lookback, start):
    """
    If growth/value ratio breaks above N-day high -> hold growth.
    If breaks below N-day low -> hold value.
    """
    ratio = g / v
    high_n = ratio.rolling(lookback).max()
    low_n = ratio.rolling(lookback).min()

    g = g[g.index >= start].copy()
    v = v[v.index >= start].copy()
    dates = g.index

    pv = pd.Series(1.0, index=dates)
    holding = 'growth'

    for i in range(1, len(dates)):
        dt = dates[i]
        r = ratio.loc[dt] if dt in ratio.index else None
        h = high_n.loc[dt] if dt in high_n.index else None
        l = low_n.loc[dt] if dt in low_n.index else None

        if r is not None and h is not None and l is not None and not pd.isna(h):
            if r >= h:
                holding = 'growth'
            elif r <= l:
                holding = 'value'

        if holding == 'growth':
            ret = g.iloc[i] / g.iloc[i-1] - 1
        else:
            ret = v.iloc[i] / v.iloc[i-1] - 1
        pv.iloc[i] = pv.iloc[i-1] * (1 + ret)

    return pv

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("沪深300 成长 vs 价值 轮动策略研究")
    print("=" * 70)

    g, v = load_data()

    # Test from 2012 (need some lookback) and from 2015, 2018, 2021
    start_dates = ['2012-01-01', '2015-01-01', '2018-01-01', '2021-01-01']

    all_results = []

    for start in start_dates:
        print(f"\n{'='*70}")
        print(f"Start: {start}")
        print(f"{'='*70}")

        gs = g[g.index >= start]
        vs = v[v.index >= start]
        bm_5050 = pd.Series(index=gs.index, dtype=float)
        bm_5050.iloc[0] = 1.0
        for i in range(1, len(bm_5050)):
            gr = gs.iloc[i]/gs.iloc[i-1]-1
            vr = vs.iloc[i]/vs.iloc[i-1]-1
            bm_5050.iloc[i] = bm_5050.iloc[i-1]*(1+0.5*gr+0.5*vr)

        results = {}

        # Benchmarks
        pv_g = gs / gs.iloc[0]
        pv_v = vs / vs.iloc[0]
        results['纯持有成长'] = calc_metrics(pv_g)
        results['纯持有价值'] = calc_metrics(pv_v)
        results['50/50均配'] = calc_metrics(bm_5050)

        # Fixed period momentum rotation
        for period in [5, 10, 20, 40, 60, 120, 240]:
            for lookback in [20, 40, 60, 120, 240]:
                if lookback < period:
                    continue
                pv = fixed_period_rotation(g, v, period, lookback, start)
                m = calc_metrics(pv, bm_5050)
                if m:
                    name = f'动量轮动_P{period}d_L{lookback}d'
                    results[name] = m

        # MA crossover
        for fast in [5, 10, 20]:
            for slow in [20, 40, 60, 120, 240]:
                if fast >= slow:
                    continue
                pv = ma_rotation(g, v, fast, slow, start)
                m = calc_metrics(pv, bm_5050)
                if m:
                    results[f'MA交叉_F{fast}_S{slow}'] = m

        # RSI mean-reversion
        for period in [14, 20, 40, 60]:
            for hi, lo in [(70, 30), (65, 35), (60, 40), (75, 25), (80, 20)]:
                pv = rsi_rotation(g, v, period, hi, lo, start)
                m = calc_metrics(pv, bm_5050)
                if m:
                    results[f'RSI均值回归_P{period}_H{hi}L{lo}'] = m

        # RSI momentum (trend-following)
        for period in [14, 20, 40, 60]:
            for hi, lo in [(55, 45), (60, 40), (65, 35), (70, 30)]:
                pv = momentum_rsi_rotation(g, v, period, hi, lo, start)
                m = calc_metrics(pv, bm_5050)
                if m:
                    results[f'RSI趋势_P{period}_H{hi}L{lo}'] = m

        # Breakout
        for lb in [20, 40, 60, 120, 240]:
            pv = breakout_rotation(g, v, lb, start)
            m = calc_metrics(pv, bm_5050)
            if m:
                results[f'突破轮动_{lb}d'] = m

        # Sort by Sharpe
        sorted_results = sorted(results.items(), key=lambda x: -x[1]['sharpe'])

        print(f"\nTotal strategies tested: {len(results)}")
        print(f"\n--- TOP 20 by Sharpe ---")
        for name, m in sorted_results[:20]:
            excess = f" excess={m['excess_cagr']:+.1f}%" if 'excess_cagr' in m else ""
            print(f"  {name:35s} CAGR={m['cagr']:6.1f}% MDD={m['mdd']:6.1f}% Sharpe={m['sharpe']:.3f} Calmar={m['calmar']:.3f}{excess}")

        print(f"\n--- TOP 10 by Calmar ---")
        sorted_calmar = sorted(results.items(), key=lambda x: -x[1]['calmar'])
        for name, m in sorted_calmar[:10]:
            print(f"  {name:35s} CAGR={m['cagr']:6.1f}% MDD={m['mdd']:6.1f}% Sharpe={m['sharpe']:.3f} Calmar={m['calmar']:.3f}")

        # Show annual returns for top 5
        print(f"\n--- Annual returns (Top 5 + benchmarks) ---")
        show_list = sorted_results[:5]
        # Add benchmarks
        for bm_name in ['纯持有成长', '纯持有价值', '50/50均配']:
            if bm_name in results:
                show_list.append((bm_name, results[bm_name]))

        for name, m in show_list:
            ann_str = ' | '.join([f"{y}:{v:+.1f}%" for y, v in sorted(m['annual'].items())])
            print(f"  {name:35s} {ann_str}")

        # Store for JSON
        for name, m in results.items():
            all_results.append({
                'start': start,
                'strategy': name,
                **m
            })

    # ============================================================
    # Robustness: check which strategies are consistently good
    # ============================================================
    print(f"\n\n{'='*70}")
    print("Robustness: strategies that rank top-20 in ALL start periods")
    print(f"{'='*70}")

    # Group by strategy, collect sharpe across periods
    strat_sharpes = {}
    for r in all_results:
        name = r['strategy']
        if name not in strat_sharpes:
            strat_sharpes[name] = []
        strat_sharpes[name].append((r['start'], r['sharpe']))

    # Rank by average sharpe across all periods
    avg_sharpes = []
    for name, sharpes in strat_sharpes.items():
        if len(sharpes) == len(start_dates):
            avg = np.mean([s for _, s in sharpes])
            mn = min(s for _, s in sharpes)
            avg_sharpes.append((name, avg, mn, sharpes))

    avg_sharpes.sort(key=lambda x: -x[1])
    print(f"\n--- Top 20 by Average Sharpe (across all periods) ---")
    for name, avg, mn, sharpes in avg_sharpes[:20]:
        detail = ' | '.join([f"{s:.3f}" for _, s in sorted(sharpes)])
        print(f"  {name:35s} Avg={avg:.3f} Min={mn:.3f}  [{detail}]")

    # Save
    output_path = os.path.join(DATA_DIR, 'growth_value_rotation_research.json')
    with open(output_path, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'indices': {'growth': '000918', 'value': '000919'},
            'start_dates': start_dates,
            'total_results': len(all_results),
            'results': all_results
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")
    print("\nDONE")

if __name__ == '__main__':
    main()
