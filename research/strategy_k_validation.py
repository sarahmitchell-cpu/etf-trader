#!/usr/bin/env python3
"""
Strategy K Comprehensive Validation
====================================
1. Out-of-sample testing (CSI300 Growth/Value)
2. Cross-market validation:
   - CSI500 Growth/Value (000920/000921)
   - CSI800 Growth/Value (000922/000923)
   - US Russell 1000 Growth/Value (IWF/IWD)
   - US S&P500 Growth/Value (SPYG/SPYV)
3. Walk-forward analysis
4. Transaction cost sensitivity
"""

import pandas as pd
import numpy as np
import json
import sys
import os

# ============================================================
# Core Strategy Logic
# ============================================================

def compute_rsi(series, period=7):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def momentum_rsi_rotation(growth, value, rsi_period=7, high=55, low=45,
                           start_date='2010-01-01', end_date='2026-12-31',
                           tx_cost=0.001):
    """
    Trend-following RSI rotation.
    RSI > high -> hold growth; RSI < low -> hold value
    """
    ratio = growth / value
    rsi = compute_rsi(ratio, rsi_period)

    # Align dates
    mask = (growth.index >= start_date) & (growth.index <= end_date)
    growth = growth[mask]
    value = value[mask]
    rsi = rsi[mask]

    if len(growth) < 50:
        return None

    position = 'cash'  # start neutral
    equity = 1.0
    positions = []
    trades = 0

    for i in range(1, len(growth)):
        dt = growth.index[i]
        r = rsi.iloc[i-1]  # use previous day's RSI

        new_pos = position
        if pd.notna(r):
            if r > high:
                new_pos = 'growth'
            elif r < low:
                new_pos = 'value'

        # Calculate return
        if position == 'growth':
            ret = growth.iloc[i] / growth.iloc[i-1] - 1
        elif position == 'value':
            ret = value.iloc[i] / value.iloc[i-1] - 1
        else:
            ret = 0

        # Apply transaction cost on switch
        if new_pos != position and position != 'cash':
            equity *= (1 - tx_cost)
            trades += 1
        if new_pos != position and new_pos != 'cash':
            equity *= (1 - tx_cost)
            if position == 'cash':
                trades += 1

        equity *= (1 + ret)
        position = new_pos
        positions.append({'date': dt, 'equity': equity, 'position': position})

    if not positions:
        return None

    df = pd.DataFrame(positions).set_index('date')

    # Calculate metrics
    total_days = (df.index[-1] - df.index[0]).days
    if total_days <= 0:
        return None
    years = total_days / 365.25
    total_return = df['equity'].iloc[-1] / df['equity'].iloc[0] - 1
    cagr = (df['equity'].iloc[-1] / df['equity'].iloc[0]) ** (1/years) - 1 if years > 0 else 0

    # Max drawdown
    peak = df['equity'].cummax()
    dd = (df['equity'] - peak) / peak
    mdd = dd.min()

    # Sharpe (annualized, daily returns)
    daily_ret = df['equity'].pct_change().dropna()
    if len(daily_ret) > 10 and daily_ret.std() > 0:
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
    else:
        sharpe = 0

    # Calmar
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    # Annual returns
    df['year'] = df.index.year
    annual = {}
    for yr, grp in df.groupby('year'):
        if len(grp) > 10:
            yr_ret = grp['equity'].iloc[-1] / grp['equity'].iloc[0] - 1
            annual[yr] = yr_ret

    win_years = sum(1 for v in annual.values() if v > 0)
    total_years = len(annual)
    win_rate = win_years / total_years if total_years > 0 else 0

    return {
        'cagr': round(cagr * 100, 2),
        'mdd': round(mdd * 100, 2),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'total_return': round(total_return * 100, 2),
        'trades': trades,
        'years': round(years, 1),
        'win_rate': round(win_rate * 100, 1),
        'annual': {str(k): round(v*100, 2) for k, v in annual.items()},
    }

def buy_and_hold(series, start_date='2010-01-01', end_date='2026-12-31'):
    """Buy and hold benchmark"""
    mask = (series.index >= start_date) & (series.index <= end_date)
    s = series[mask]
    if len(s) < 2:
        return None
    total_days = (s.index[-1] - s.index[0]).days
    years = total_days / 365.25
    total_return = s.iloc[-1] / s.iloc[0] - 1
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1/years) - 1 if years > 0 else 0
    peak = s.cummax()
    mdd = ((s - peak) / peak).min()
    daily_ret = s.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    return {
        'cagr': round(cagr * 100, 2),
        'mdd': round(mdd * 100, 2),
        'sharpe': round(sharpe, 3),
        'total_return': round(total_return * 100, 2),
    }

# ============================================================
# Data Loading
# ============================================================

def load_cn_index(code):
    """Load Chinese index from local CSV or akshare"""
    local_path = f'/Users/claw/etf-trader/data/{code}_daily.csv'
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        if '日期' in df.columns:
            df['date'] = pd.to_datetime(df['日期'])
            df['close'] = df['收盘']
        else:
            df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df['close']

    import akshare as ak
    df = ak.index_zh_a_hist(symbol=code, period='daily', start_date='20050101', end_date='20260328')
    df['date'] = pd.to_datetime(df['日期'])
    df = df.set_index('date').sort_index()
    # Save for reuse
    df.to_csv(local_path)
    return df['收盘'].astype(float)

def load_us_etf(ticker):
    """Load US ETF data via akshare"""
    cache_path = f'/Users/claw/etf-trader/data/{ticker}_daily.csv'
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df['close']

    import akshare as ak
    df = ak.stock_us_daily(symbol=ticker, adjust='qfq')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df.to_csv(cache_path)
    return df['close']

# ============================================================
# Test Suite
# ============================================================

def test_out_of_sample(growth, value, label):
    """
    Split data: in-sample (first 60%) vs out-of-sample (last 40%)
    Also do 3-fold walk-forward
    """
    results = {}

    # Align
    common = growth.index.intersection(value.index)
    growth = growth[common]
    value = value[common]

    total_len = len(common)
    split_60 = int(total_len * 0.6)
    split_date = common[split_60]

    # Grid search on in-sample
    best_sharpe = -999
    best_params = (7, 55, 45)

    is_start = str(common[0].date())
    is_end = str(split_date.date())
    oos_start = str(split_date.date())
    oos_end = str(common[-1].date())

    print(f"\n{'='*60}")
    print(f"[{label}] Out-of-Sample Test")
    print(f"  In-sample:  {is_start} to {is_end} ({split_60} days)")
    print(f"  Out-of-sample: {oos_start} to {oos_end} ({total_len - split_60} days)")
    print(f"{'='*60}")

    for period in [5, 7, 10, 14, 20]:
        for high, low in [(55, 45), (60, 40), (65, 35), (70, 30)]:
            r = momentum_rsi_rotation(growth, value, period, high, low,
                                       start_date=is_start, end_date=is_end)
            if r and r['sharpe'] > best_sharpe:
                best_sharpe = r['sharpe']
                best_params = (period, high, low)

    p, h, l = best_params
    print(f"\n  Best in-sample params: RSI({p}), H{h}/L{l} (Sharpe={best_sharpe:.3f})")

    # Test best params on out-of-sample
    is_result = momentum_rsi_rotation(growth, value, p, h, l,
                                       start_date=is_start, end_date=is_end)
    oos_result = momentum_rsi_rotation(growth, value, p, h, l,
                                        start_date=oos_start, end_date=oos_end)

    # Also test default params (P7 H55/L45) on both periods
    is_default = momentum_rsi_rotation(growth, value, 7, 55, 45,
                                        start_date=is_start, end_date=is_end)
    oos_default = momentum_rsi_rotation(growth, value, 7, 55, 45,
                                         start_date=oos_start, end_date=oos_end)

    # Full period
    full_result = momentum_rsi_rotation(growth, value, p, h, l)
    full_default = momentum_rsi_rotation(growth, value, 7, 55, 45)

    # Benchmarks
    bh_growth = buy_and_hold(growth, oos_start, oos_end)
    bh_value = buy_and_hold(value, oos_start, oos_end)

    print(f"\n  --- Optimized Params: RSI({p}) H{h}/L{l} ---")
    if is_result:
        print(f"  In-sample:      CAGR {is_result['cagr']:6.1f}%  MDD {is_result['mdd']:6.1f}%  Sharpe {is_result['sharpe']:.3f}")
    if oos_result:
        print(f"  Out-of-sample:  CAGR {oos_result['cagr']:6.1f}%  MDD {oos_result['mdd']:6.1f}%  Sharpe {oos_result['sharpe']:.3f}")

    print(f"\n  --- Default Params: RSI(7) H55/L45 ---")
    if is_default:
        print(f"  In-sample:      CAGR {is_default['cagr']:6.1f}%  MDD {is_default['mdd']:6.1f}%  Sharpe {is_default['sharpe']:.3f}")
    if oos_default:
        print(f"  Out-of-sample:  CAGR {oos_default['cagr']:6.1f}%  MDD {oos_default['mdd']:6.1f}%  Sharpe {oos_default['sharpe']:.3f}")

    print(f"\n  --- Benchmarks (Out-of-sample period) ---")
    if bh_growth:
        print(f"  Buy&Hold Growth: CAGR {bh_growth['cagr']:6.1f}%  MDD {bh_growth['mdd']:6.1f}%  Sharpe {bh_growth['sharpe']:.3f}")
    if bh_value:
        print(f"  Buy&Hold Value:  CAGR {bh_value['cagr']:6.1f}%  MDD {bh_value['mdd']:6.1f}%  Sharpe {bh_value['sharpe']:.3f}")

    results['label'] = label
    results['in_sample_period'] = f"{is_start} to {is_end}"
    results['oos_period'] = f"{oos_start} to {oos_end}"
    results['best_params'] = f"RSI({p}) H{h}/L{l}"
    results['optimized'] = {'in_sample': is_result, 'out_of_sample': oos_result}
    results['default_p7'] = {'in_sample': is_default, 'out_of_sample': oos_default}
    results['full_period'] = {'optimized': full_result, 'default': full_default}
    results['benchmarks_oos'] = {'growth_bh': bh_growth, 'value_bh': bh_value}

    return results


def test_walk_forward(growth, value, label, n_folds=4):
    """Walk-forward analysis: train on fold N, test on fold N+1"""
    common = growth.index.intersection(value.index)
    growth = growth[common]
    value = value[common]

    fold_size = len(common) // n_folds

    print(f"\n{'='*60}")
    print(f"[{label}] Walk-Forward Analysis ({n_folds} folds)")
    print(f"{'='*60}")

    wf_results = []

    for i in range(n_folds - 1):
        train_start = str(common[i * fold_size].date())
        train_end = str(common[(i + 1) * fold_size - 1].date())
        test_start = str(common[(i + 1) * fold_size].date())
        test_end = str(common[min((i + 2) * fold_size - 1, len(common) - 1)].date())

        # Find best params on training fold
        best_sharpe = -999
        best_params = (7, 55, 45)

        for period in [5, 7, 10, 14, 20]:
            for high, low in [(55, 45), (60, 40), (65, 35)]:
                r = momentum_rsi_rotation(growth, value, period, high, low,
                                           start_date=train_start, end_date=train_end)
                if r and r['sharpe'] > best_sharpe:
                    best_sharpe = r['sharpe']
                    best_params = (period, high, low)

        p, h, l = best_params

        # Test on next fold
        test_result = momentum_rsi_rotation(growth, value, p, h, l,
                                             start_date=test_start, end_date=test_end)
        default_result = momentum_rsi_rotation(growth, value, 7, 55, 45,
                                                start_date=test_start, end_date=test_end)

        print(f"\n  Fold {i+1}: Train {train_start}~{train_end} -> Test {test_start}~{test_end}")
        print(f"    Best params: RSI({p}) H{h}/L{l}")
        if test_result:
            print(f"    Optimized: CAGR {test_result['cagr']:6.1f}%  MDD {test_result['mdd']:6.1f}%  Sharpe {test_result['sharpe']:.3f}")
        if default_result:
            print(f"    Default:   CAGR {default_result['cagr']:6.1f}%  MDD {default_result['mdd']:6.1f}%  Sharpe {default_result['sharpe']:.3f}")

        wf_results.append({
            'fold': i + 1,
            'train': f"{train_start}~{train_end}",
            'test': f"{test_start}~{test_end}",
            'best_params': f"RSI({p}) H{h}/L{l}",
            'optimized_test': test_result,
            'default_test': default_result,
        })

    return wf_results


def test_transaction_costs(growth, value, label):
    """Sensitivity analysis for transaction costs"""
    common = growth.index.intersection(value.index)
    growth = growth[common]
    value = value[common]

    print(f"\n{'='*60}")
    print(f"[{label}] Transaction Cost Sensitivity (P7 H55/L45)")
    print(f"{'='*60}")

    costs = [0, 0.0005, 0.001, 0.002, 0.003, 0.005]
    tc_results = []

    for cost in costs:
        r = momentum_rsi_rotation(growth, value, 7, 55, 45, tx_cost=cost)
        if r:
            print(f"  Cost {cost*100:.2f}%: CAGR {r['cagr']:6.1f}%  MDD {r['mdd']:6.1f}%  Sharpe {r['sharpe']:.3f}  Trades {r['trades']}")
            tc_results.append({'cost_bps': cost * 10000, 'result': r})

    return tc_results


def test_param_stability(growth, value, label):
    """Test all parameter combinations to show stability"""
    common = growth.index.intersection(value.index)
    growth = growth[common]
    value = value[common]

    print(f"\n{'='*60}")
    print(f"[{label}] Parameter Stability Grid")
    print(f"{'='*60}")

    header = f"  {'RSI':>5} {'H/L':>7} {'CAGR':>7} {'MDD':>7} {'Sharpe':>7} {'Trades':>7}"
    print(header)
    print(f"  {'-'*43}")

    grid_results = []

    for period in [5, 7, 10, 14, 20]:
        for high, low in [(55, 45), (60, 40), (65, 35), (70, 30)]:
            r = momentum_rsi_rotation(growth, value, period, high, low)
            if r:
                print(f"  {period:>5} {high}/{low:>5} {r['cagr']:6.1f}% {r['mdd']:6.1f}% {r['sharpe']:7.3f} {r['trades']:>7}")
                grid_results.append({
                    'period': period, 'high': high, 'low': low,
                    'cagr': r['cagr'], 'mdd': r['mdd'], 'sharpe': r['sharpe'],
                    'trades': r['trades']
                })

    return grid_results


# ============================================================
# Main
# ============================================================

def main():
    all_results = {}

    # ---- 1. CSI300 Growth/Value (our primary market) ----
    print("\n" + "=" * 70)
    print("MARKET 1: CSI300 Growth (000918) / Value (000919)")
    print("=" * 70)

    g300 = load_cn_index('000918')
    v300 = load_cn_index('000919')

    r1_oos = test_out_of_sample(g300, v300, 'CSI300')
    r1_wf = test_walk_forward(g300, v300, 'CSI300')
    r1_tc = test_transaction_costs(g300, v300, 'CSI300')
    r1_grid = test_param_stability(g300, v300, 'CSI300')

    all_results['CSI300'] = {
        'oos': r1_oos, 'walk_forward': r1_wf,
        'tx_cost': r1_tc, 'param_grid': r1_grid
    }

    # ---- 2. CSI500 Growth/Value ----
    print("\n" + "=" * 70)
    print("MARKET 2: CSI500 Growth (000920) / Value (000921)")
    print("=" * 70)

    g500 = load_cn_index('000920')
    v500 = load_cn_index('000921')

    r2_oos = test_out_of_sample(g500, v500, 'CSI500')
    r2_wf = test_walk_forward(g500, v500, 'CSI500')
    r2_tc = test_transaction_costs(g500, v500, 'CSI500')
    r2_grid = test_param_stability(g500, v500, 'CSI500')

    all_results['CSI500'] = {
        'oos': r2_oos, 'walk_forward': r2_wf,
        'tx_cost': r2_tc, 'param_grid': r2_grid
    }

    # ---- 3. CSI800 Growth/Value ----
    print("\n" + "=" * 70)
    print("MARKET 3: CSI800 Growth (000922) / Value (000923)")
    print("=" * 70)

    g800 = load_cn_index('000922')
    v800 = load_cn_index('000923')

    r3_oos = test_out_of_sample(g800, v800, 'CSI800')
    r3_wf = test_walk_forward(g800, v800, 'CSI800')
    r3_tc = test_transaction_costs(g800, v800, 'CSI800')
    r3_grid = test_param_stability(g800, v800, 'CSI800')

    all_results['CSI800'] = {
        'oos': r3_oos, 'walk_forward': r3_wf,
        'tx_cost': r3_tc, 'param_grid': r3_grid
    }

    # ---- 4. US Russell 1000 Growth/Value ----
    print("\n" + "=" * 70)
    print("MARKET 4: US Russell 1000 Growth (IWF) / Value (IWD)")
    print("=" * 70)

    g_russ = load_us_etf('IWF')
    v_russ = load_us_etf('IWD')

    r4_oos = test_out_of_sample(g_russ, v_russ, 'Russell1000')
    r4_wf = test_walk_forward(g_russ, v_russ, 'Russell1000')
    r4_tc = test_transaction_costs(g_russ, v_russ, 'Russell1000')
    r4_grid = test_param_stability(g_russ, v_russ, 'Russell1000')

    all_results['Russell1000'] = {
        'oos': r4_oos, 'walk_forward': r4_wf,
        'tx_cost': r4_tc, 'param_grid': r4_grid
    }

    # ---- 5. US S&P500 Growth/Value ----
    print("\n" + "=" * 70)
    print("MARKET 5: US S&P500 Growth (SPYG) / Value (SPYV)")
    print("=" * 70)

    g_sp = load_us_etf('SPYG')
    v_sp = load_us_etf('SPYV')

    r5_oos = test_out_of_sample(g_sp, v_sp, 'SP500')
    r5_wf = test_walk_forward(g_sp, v_sp, 'SP500')
    r5_tc = test_transaction_costs(g_sp, v_sp, 'SP500')
    r5_grid = test_param_stability(g_sp, v_sp, 'SP500')

    all_results['SP500'] = {
        'oos': r5_oos, 'walk_forward': r5_wf,
        'tx_cost': r5_tc, 'param_grid': r5_grid
    }

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("CROSS-MARKET SUMMARY (Default P7 H55/L45, full period)")
    print("=" * 70)

    summary = []
    for market, data in all_results.items():
        oos = data['oos']
        full = oos.get('full_period', {}).get('default')
        oos_r = oos.get('default_p7', {}).get('out_of_sample')

        if full:
            print(f"\n  {market:>12}:")
            print(f"    Full:  CAGR {full['cagr']:6.1f}%  MDD {full['mdd']:6.1f}%  Sharpe {full['sharpe']:.3f}  WinRate {full.get('win_rate', 'N/A')}")
        if oos_r:
            print(f"    OOS:   CAGR {oos_r['cagr']:6.1f}%  MDD {oos_r['mdd']:6.1f}%  Sharpe {oos_r['sharpe']:.3f}")

        summary.append({
            'market': market,
            'full': full,
            'oos': oos_r,
        })

    all_results['summary'] = summary

    # Save results
    output_path = '/Users/claw/etf-trader/data/strategy_k_validation.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == '__main__':
    main()
