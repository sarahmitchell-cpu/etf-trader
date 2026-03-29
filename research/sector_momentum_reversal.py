#!/usr/bin/env python3
"""
行业Beta动量反转研究 (Sector Momentum & Reversal Research)

研究内容:
  1. 行业横截面动量 (Cross-sectional momentum): 买入过去表现最好的行业
  2. 行业横截面反转 (Cross-sectional reversal): 买入过去表现最差的行业
  3. 不同回望期 (lookback): 5d, 10d, 20d, 60d, 120d, 250d
  4. 不同持仓期 (holding): 5d, 10d, 20d, 60d
  5. 动量+反转组合: 长期动量 + 短期反转

数据: 中证行业指数 via akshare
回测期: 2012~2026

Author: Sarah Mitchell / VisionClaw
Date: 2026-03-29
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import akshare as ak
import json, os, sys, time
from datetime import datetime
from itertools import product

DATA_DIR = '/Users/claw/etf-trader/data'

# ============================================================
# 1. SECTOR INDEX DEFINITIONS
# ============================================================

SECTOR_INDICES = {
    # 中证全指行业 (GICS sectors)
    '全指消费': '000990',
    '全指金融': '000991',
    '全指信息': '000992',
    '全指医药': '000993',
    '全指能源': '000994',
    '全指材料': '000995',
    '全指公用': '000998',
    # 行业主题指数
    '中证军工': '930633',
    '证券公司': '399975',
    '中证钢铁': '930608',
    '有色金属': '930651',
    '中证养殖': '399812',
    '中证白酒': '930606',
    '中证银行': '399986',
    '芯片产业': '930707',
    '中证红利': '000922',
}

# ============================================================
# 2. DATA LOADING (via akshare)
# ============================================================

def fetch_index_akshare(code, name, start='20100101', end='20260329'):
    """Fetch index data via akshare"""
    try:
        time.sleep(0.5)
        df = ak.index_zh_a_hist(symbol=code, period='daily',
                                start_date=start, end_date=end)
        if df is None or len(df) == 0:
            print(f"    {name} ({code}): No data")
            return None

        df = df.rename(columns={'日期': 'date', '收盘': 'close'})
        df['date'] = pd.to_datetime(df['date'])
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df[['date', 'close']].dropna().set_index('date').sort_index()
        df = df[df['close'] > 0]
        df = df[~df.index.duplicated(keep='first')]
        print(f"    {name} ({code}): {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} rows")
        return df
    except Exception as e:
        print(f"    {name} ({code}): ERROR {e}")
        return None


def load_all_sectors():
    """Load all sector indices"""
    print("[1] Loading sector indices via akshare...")
    sectors = {}
    for name, code in SECTOR_INDICES.items():
        df = fetch_index_akshare(code, name)
        if df is not None and len(df) > 500:
            sectors[name] = df
    print(f"    Loaded {len(sectors)} sectors\n")
    return sectors


def build_price_panel(sectors):
    """Build aligned price panel (DataFrame with sectors as columns)"""
    all_dfs = {}
    for name, df in sectors.items():
        all_dfs[name] = df['close']

    panel = pd.DataFrame(all_dfs)
    panel = panel.dropna()
    panel = panel[panel.index >= '2012-01-01']

    print(f"[2] Price panel: {panel.index[0].date()} ~ {panel.index[-1].date()}, "
          f"{len(panel)} trading days, {panel.shape[1]} sectors\n")
    return panel


# ============================================================
# 3. STRATEGY IMPLEMENTATIONS
# ============================================================

def strategy_momentum(panel, lookback, holding, top_n=3, long_only=True):
    """
    Cross-sectional momentum: buy top N sectors by past returns
    """
    returns_daily = panel.pct_change()
    momentum = panel.pct_change(lookback)

    n_sectors = panel.shape[1]
    top_n = min(top_n, n_sectors)

    portfolio_returns = pd.Series(0.0, index=panel.index)
    rebalance_dates = set(panel.index[lookback::holding])
    current_weights = pd.Series(0.0, index=panel.columns)

    for i in range(lookback, len(panel)):
        date = panel.index[i]

        if date in rebalance_dates:
            prev_date_idx = i - 1
            if prev_date_idx < lookback:
                continue

            mom = momentum.iloc[prev_date_idx]
            if mom.isna().all():
                continue

            ranked = mom.dropna().sort_values(ascending=False)

            if long_only:
                current_weights = pd.Series(0.0, index=panel.columns)
                for sec in ranked.index[:top_n]:
                    current_weights[sec] = 1.0 / top_n
            else:
                current_weights = pd.Series(0.0, index=panel.columns)
                for sec in ranked.index[:top_n]:
                    current_weights[sec] = 0.5 / top_n
                for sec in ranked.index[-top_n:]:
                    current_weights[sec] = -0.5 / top_n

        day_ret = returns_daily.iloc[i]
        portfolio_returns.iloc[i] = (current_weights * day_ret).sum()

    return portfolio_returns.iloc[lookback:]


def strategy_reversal(panel, lookback, holding, bottom_n=3, long_only=True):
    """
    Cross-sectional reversal: buy bottom N sectors by past returns (losers)
    """
    returns_daily = panel.pct_change()
    momentum = panel.pct_change(lookback)

    n_sectors = panel.shape[1]
    bottom_n = min(bottom_n, n_sectors)

    portfolio_returns = pd.Series(0.0, index=panel.index)
    rebalance_dates = set(panel.index[lookback::holding])
    current_weights = pd.Series(0.0, index=panel.columns)

    for i in range(lookback, len(panel)):
        date = panel.index[i]

        if date in rebalance_dates:
            prev_date_idx = i - 1
            if prev_date_idx < lookback:
                continue

            mom = momentum.iloc[prev_date_idx]
            if mom.isna().all():
                continue

            ranked = mom.dropna().sort_values(ascending=True)

            if long_only:
                current_weights = pd.Series(0.0, index=panel.columns)
                for sec in ranked.index[:bottom_n]:
                    current_weights[sec] = 1.0 / bottom_n
            else:
                current_weights = pd.Series(0.0, index=panel.columns)
                for sec in ranked.index[:bottom_n]:
                    current_weights[sec] = 0.5 / bottom_n
                for sec in ranked.index[-bottom_n:]:
                    current_weights[sec] = -0.5 / bottom_n

        day_ret = returns_daily.iloc[i]
        portfolio_returns.iloc[i] = (current_weights * day_ret).sum()

    return portfolio_returns.iloc[lookback:]


def strategy_combo_mom_rev(panel, long_lookback, short_lookback, holding, top_n=3):
    """
    Combo: Long-term momentum + Short-term reversal
    - Use long_lookback momentum to select sector universe (top half)
    - Within that, use short_lookback reversal to pick entries (buy short-term losers)
    """
    returns_daily = panel.pct_change()
    long_mom = panel.pct_change(long_lookback)
    short_mom = panel.pct_change(short_lookback)

    n_sectors = panel.shape[1]
    top_n = min(top_n, n_sectors)

    portfolio_returns = pd.Series(0.0, index=panel.index)
    start = max(long_lookback, short_lookback)
    rebalance_dates = set(panel.index[start::holding])
    current_weights = pd.Series(0.0, index=panel.columns)

    for i in range(start, len(panel)):
        date = panel.index[i]

        if date in rebalance_dates:
            prev_idx = i - 1
            if prev_idx < start:
                continue

            lm = long_mom.iloc[prev_idx].dropna()
            sm = short_mom.iloc[prev_idx].dropna()

            if lm.empty or sm.empty:
                continue

            # Step 1: Long-term momentum filter (top half)
            lm_ranked = lm.sort_values(ascending=False)
            top_half = lm_ranked.index[:max(n_sectors // 2, top_n)]

            # Step 2: Within top half, short-term reversal (buy losers)
            sm_filtered = sm[sm.index.isin(top_half)]
            sm_ranked = sm_filtered.sort_values(ascending=True)
            picks = sm_ranked.index[:top_n]

            current_weights = pd.Series(0.0, index=panel.columns)
            for sec in picks:
                current_weights[sec] = 1.0 / len(picks)

        day_ret = returns_daily.iloc[i]
        portfolio_returns.iloc[i] = (current_weights * day_ret).sum()

    return portfolio_returns.iloc[start:]


def strategy_equal_weight(panel):
    """Baseline: equal-weight all sectors"""
    returns_daily = panel.pct_change()
    return returns_daily.mean(axis=1).dropna()


# ============================================================
# 4. METRICS
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

    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min() * 100

    monthly = returns.resample('ME').sum()
    win_rate = (monthly > 0).mean() * 100

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
# 5. MAIN RESEARCH
# ============================================================

def run_all():
    sectors = load_all_sectors()
    if len(sectors) < 8:
        print(f"ERROR: Need at least 8 sectors, got {len(sectors)}!")
        return

    panel = build_price_panel(sectors)
    if len(panel) < 500:
        print("ERROR: Insufficient data!")
        return

    all_results = []

    # ========================================
    # A. Baseline
    # ========================================
    print("[3] Running baselines...")
    ew_ret = strategy_equal_weight(panel)
    m = calc_metrics(ew_ret, 'EqualWeight_AllSectors')
    if m:
        m['strategy_type'] = 'baseline'
        m['lookback'] = 0
        m['holding'] = 0
        m['top_n'] = panel.shape[1]
        all_results.append(m)
        print(f"    EqualWeight: CAGR={m['cagr']:.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:.1f}%")

    # ========================================
    # B. Pure Momentum (long-only)
    # ========================================
    print("\n[4] Testing pure momentum strategies (buy winners)...")
    lookbacks = [5, 10, 20, 60, 120, 250]
    holdings = [5, 10, 20, 60]
    top_ns = [2, 3, 4, 5]

    count = 0
    for lb, hd, tn in product(lookbacks, holdings, top_ns):
        count += 1
        name = f'MOM_LB{lb}_HD{hd}_TOP{tn}'
        try:
            ret = strategy_momentum(panel, lb, hd, tn, long_only=True)
            m = calc_metrics(ret, name)
            if m:
                m['strategy_type'] = 'momentum'
                m['lookback'] = lb
                m['holding'] = hd
                m['top_n'] = tn
                all_results.append(m)
        except Exception as e:
            pass
    print(f"    Tested {count} momentum combos")

    # ========================================
    # C. Pure Reversal (long-only)
    # ========================================
    print("\n[5] Testing pure reversal strategies (buy losers)...")
    count = 0
    for lb, hd, tn in product(lookbacks, holdings, top_ns):
        count += 1
        name = f'REV_LB{lb}_HD{hd}_BOT{tn}'
        try:
            ret = strategy_reversal(panel, lb, hd, tn, long_only=True)
            m = calc_metrics(ret, name)
            if m:
                m['strategy_type'] = 'reversal'
                m['lookback'] = lb
                m['holding'] = hd
                m['top_n'] = tn
                all_results.append(m)
        except Exception as e:
            pass
    print(f"    Tested {count} reversal combos")

    # ========================================
    # D. Long-Short Momentum
    # ========================================
    print("\n[6] Testing long-short momentum...")
    count = 0
    for lb, hd in product([20, 60, 120, 250], [20, 60]):
        for tn in [2, 3]:
            count += 1
            name = f'MOM_LS_LB{lb}_HD{hd}_N{tn}'
            try:
                ret = strategy_momentum(panel, lb, hd, tn, long_only=False)
                m = calc_metrics(ret, name)
                if m:
                    m['strategy_type'] = 'momentum_ls'
                    m['lookback'] = lb
                    m['holding'] = hd
                    m['top_n'] = tn
                    all_results.append(m)
            except:
                pass
    print(f"    Tested {count} long-short combos")

    # ========================================
    # E. Combo: Long-term MOM + Short-term REV
    # ========================================
    print("\n[7] Testing combo: long-term MOM + short-term REV...")
    long_lookbacks = [60, 120, 250]
    short_lookbacks = [5, 10, 20]
    combo_holdings = [5, 10, 20]
    combo_top_ns = [2, 3, 4]
    count = 0
    for llb, slb, hd, tn in product(long_lookbacks, short_lookbacks, combo_holdings, combo_top_ns):
        if slb >= llb:
            continue
        count += 1
        name = f'COMBO_LMOM{llb}_SREV{slb}_HD{hd}_N{tn}'
        try:
            ret = strategy_combo_mom_rev(panel, llb, slb, hd, tn)
            m = calc_metrics(ret, name)
            if m:
                m['strategy_type'] = 'combo_mom_rev'
                m['lookback'] = llb
                m['short_lookback'] = slb
                m['holding'] = hd
                m['top_n'] = tn
                all_results.append(m)
        except:
            pass
    print(f"    Tested {count} combo strategies")

    # ========================================
    # F. Long-Short Reversal
    # ========================================
    print("\n[8] Testing long-short reversal...")
    count = 0
    for lb, hd in product([5, 10, 20, 60], [5, 10, 20]):
        for tn in [2, 3]:
            count += 1
            name = f'REV_LS_LB{lb}_HD{hd}_N{tn}'
            try:
                ret = strategy_reversal(panel, lb, hd, tn, long_only=False)
                m = calc_metrics(ret, name)
                if m:
                    m['strategy_type'] = 'reversal_ls'
                    m['lookback'] = lb
                    m['holding'] = hd
                    m['top_n'] = tn
                    all_results.append(m)
            except:
                pass
    print(f"    Tested {count} long-short reversal combos")

    # ========================================
    # RESULTS
    # ========================================
    print(f"\n\n{'='*90}")
    print(f"  TOTAL STRATEGIES TESTED: {len(all_results)}")
    print(f"{'='*90}")

    sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)

    # Top 25 overall
    print(f"\n  TOP 25 BY SHARPE RATIO:")
    print(f"  {'#':>3} {'Name':<48} {'Type':<14} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'WinR%':>6}")
    print(f"  {'-'*3} {'-'*48} {'-'*14} {'-'*7} {'-'*7} {'-'*8} {'-'*6}")
    for i, r in enumerate(sorted_results[:25]):
        print(f"  {i+1:>3} {r['name']:<48} {r['strategy_type']:<14} "
              f"{r['cagr']:>6.1f}% {r['sharpe']:>7.3f} {r['max_dd']:>7.1f}% {r['win_rate_monthly']:>5.1f}")

    # Best by strategy type
    for stype in ['momentum', 'reversal', 'combo_mom_rev', 'momentum_ls', 'reversal_ls']:
        subset = [r for r in sorted_results if r['strategy_type'] == stype]
        if not subset:
            continue
        print(f"\n\n  BEST {stype.upper()} (Top 10):")
        print(f"  {'#':>3} {'Name':<48} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'WinR%':>6}")
        print(f"  {'-'*3} {'-'*48} {'-'*7} {'-'*7} {'-'*8} {'-'*6}")
        for i, r in enumerate(subset[:10]):
            print(f"  {i+1:>3} {r['name']:<48} "
                  f"{r['cagr']:>6.1f}% {r['sharpe']:>7.3f} {r['max_dd']:>7.1f}% {r['win_rate_monthly']:>5.1f}")

    # Heatmap: Momentum effect by lookback x holding
    print(f"\n\n{'='*90}")
    print(f"  MOMENTUM SHARPE HEATMAP (top_n=3)")
    hdr = "LB \\ HD"
    print(f"  {hdr:<10} {'5d':>8} {'10d':>8} {'20d':>8} {'60d':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for lb in lookbacks:
        row = f"  {lb}d{'':<7}"
        for hd in holdings:
            r = next((x for x in all_results if x['name'] == f'MOM_LB{lb}_HD{hd}_TOP3'), None)
            row += f" {r['sharpe']:>7.3f}" if r else f" {'N/A':>7}"
        print(row)

    print(f"\n  REVERSAL SHARPE HEATMAP (bottom_n=3)")
    print(f"  {hdr:<10} {'5d':>8} {'10d':>8} {'20d':>8} {'60d':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for lb in lookbacks:
        row = f"  {lb}d{'':<7}"
        for hd in holdings:
            r = next((x for x in all_results if x['name'] == f'REV_LB{lb}_HD{hd}_BOT3'), None)
            row += f" {r['sharpe']:>7.3f}" if r else f" {'N/A':>7}"
        print(row)

    # MOM vs REV comparison
    print(f"\n\n{'='*90}")
    print(f"  MOMENTUM vs REVERSAL (top_n=3)")
    print(f"  {'Config':<20} {'MOM_S':>8} {'REV_S':>8} {'Winner':>8} {'MOM_CAGR':>9} {'REV_CAGR':>9}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")
    for lb in lookbacks:
        for hd in holdings:
            mr = next((x for x in all_results if x['name'] == f'MOM_LB{lb}_HD{hd}_TOP3'), None)
            rr = next((x for x in all_results if x['name'] == f'REV_LB{lb}_HD{hd}_BOT3'), None)
            if mr and rr:
                w = 'MOM' if mr['sharpe'] > rr['sharpe'] else 'REV'
                print(f"  LB{lb:>3} HD{hd:>3}{'':<8} "
                      f"{mr['sharpe']:>8.3f} {rr['sharpe']:>8.3f} {w:>8} "
                      f"{mr['cagr']:>8.1f}% {rr['cagr']:>8.1f}%")

    # Save results
    output = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'description': '行业Beta动量反转研究',
        'sectors': list(SECTOR_INDICES.keys()),
        'period': f"{panel.index[0].date()} ~ {panel.index[-1].date()}",
        'total_strategies': len(all_results),
        'all_results': sorted_results,
    }

    out_path = os.path.join(DATA_DIR, 'sector_momentum_reversal_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n\nResults saved to {out_path}")
    print(f"Total strategies tested: {len(all_results)}")


if __name__ == '__main__':
    run_all()
