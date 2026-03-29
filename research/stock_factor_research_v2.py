#!/usr/bin/env python3
"""
A股个股因子组合回测研究 V2 (Practical Portfolio Construction)

基于V1的发现，深入研究:
  1. 多因子组合 (低换手+动量, 低换手+价值, 三因子等)
  2. 实际组合构建 (Top N stocks, 不是quintile sort)
  3. 交易成本 (单边30bps)
  4. 不同持仓数量 (30/50/100)
  5. 不同调仓周期 (月度/季度)
  6. 分大中小盘 (CSI300 vs CSI500成分)
  7. 因子衰减分析

Author: Sarah Mitchell / VisionClaw
Date: 2026-03-29
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import akshare as ak
import baostock as bs
import json, os, sys, time, pickle
from datetime import datetime
from collections import defaultdict

DATA_DIR = '/Users/claw/etf-trader/data'
CACHE_DIR = os.path.join(DATA_DIR, 'stock_cache_v2')
os.makedirs(CACHE_DIR, exist_ok=True)

RISK_FREE_RATE = 0.025
TC_ONE_SIDE = 0.003  # 30bps per side (more realistic for A-shares)
START_DATE = '20150101'
END_DATE = '20260328'
BACKTEST_START = '2016-01-31'


# ============================================================
# 1. DATA LOADING (reuse V1 cache)
# ============================================================

def get_csi800_codes():
    """Get CSI300+CSI500 constituent codes."""
    cache_file = os.path.join(CACHE_DIR, 'csi800_codes.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    lg = bs.login()
    codes = set()
    for query_fn in [bs.query_hs300_stocks, bs.query_zz500_stocks]:
        rs = query_fn(date='2024-12-31')
        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            code = row[1].split('.')[1]
            codes.add(code)
    bs.logout()
    codes = sorted(codes)
    with open(cache_file, 'wb') as f:
        pickle.dump(codes, f)
    return codes


def get_csi300_codes():
    """Get CSI300 constituent codes only."""
    cache_file = os.path.join(CACHE_DIR, 'csi300_codes.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    lg = bs.login()
    codes = set()
    rs = bs.query_hs300_stocks(date='2024-12-31')
    while rs.error_code == '0' and rs.next():
        row = rs.get_row_data()
        code = row[1].split('.')[1]
        codes.add(code)
    bs.logout()
    codes = sorted(codes)
    with open(cache_file, 'wb') as f:
        pickle.dump(codes, f)
    return codes


def fetch_stock_akshare(code):
    """Fetch daily data for a single stock via akshare."""
    cache_file = os.path.join(CACHE_DIR, f'{code}_ak.pkl')
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 86400 * 3:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    try:
        df = ak.stock_zh_a_hist(symbol=code, period='daily',
                                start_date=START_DATE, end_date=END_DATE, adjust='qfq')
        if df is None or len(df) < 100:
            return None
        df = df.rename(columns={
            '日期': 'date', '收盘': 'close', '开盘': 'open',
            '最高': 'high', '最低': 'low',
            '成交量': 'volume', '成交额': 'amount',
            '涨跌幅': 'pctChg', '换手率': 'turnover',
        })
        df['date'] = pd.to_datetime(df['date'])
        for c in ['close', 'open', 'high', 'low', 'volume', 'amount', 'pctChg', 'turnover']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.set_index('date').sort_index()
        df = df[df['volume'] > 0]
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        return df
    except:
        return None


def fetch_stock_valuation_bs(code):
    """Fetch PE/PB via baostock."""
    cache_file = os.path.join(CACHE_DIR, f'{code}_val.pkl')
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 86400 * 3:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    prefix = 'sh' if code.startswith('6') else 'sz'
    bs_code = f'{prefix}.{code}'
    try:
        rs = bs.query_history_k_data_plus(
            bs_code, fields='date,peTTM,pbMRQ',
            frequency='d', start_date='2015-01-01', end_date='2026-03-28',
            adjustflag='2')
        data = []
        while rs.error_code == '0' and rs.next():
            data.append(rs.get_row_data())
        if not data:
            return None
        df = pd.DataFrame(data, columns=['date', 'peTTM', 'pbMRQ'])
        df['date'] = pd.to_datetime(df['date'])
        df['peTTM'] = pd.to_numeric(df['peTTM'], errors='coerce')
        df['pbMRQ'] = pd.to_numeric(df['pbMRQ'], errors='coerce')
        df = df.set_index('date').sort_index()
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        return df
    except:
        return None


def load_all_data(codes):
    """Load all stock data from cache."""
    print("  Loading price/volume data...")
    all_data = {}
    failed = 0
    for i, code in enumerate(codes):
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(codes)} ...")
        df = fetch_stock_akshare(code)
        if df is not None and len(df) > 120:
            all_data[code] = df
        else:
            failed += 1
        time.sleep(0.02)
    print(f"  Loaded {len(all_data)} stocks, failed: {failed}")
    return all_data


def load_valuation_data(codes):
    """Load PE/PB data from cache."""
    print("  Loading PE/PB data...")
    val_data = {}
    # Only try loading from cache (don't login to baostock if no cache)
    for code in codes:
        cache_file = os.path.join(CACHE_DIR, f'{code}_val.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                val_data[code] = pickle.load(f)
    print(f"  Loaded valuation for {len(val_data)} stocks")
    return val_data


# ============================================================
# 2. FACTOR COMPUTATION (enhanced)
# ============================================================

def get_month_ends(all_data):
    all_dates = set()
    for code, df in all_data.items():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)
    me_df = pd.DataFrame({'date': all_dates})
    me_df = me_df[me_df['date'] >= pd.Timestamp(BACKTEST_START)]
    me_df['ym'] = me_df['date'].dt.to_period('M')
    month_ends = me_df.groupby('ym')['date'].max().tolist()
    return sorted(month_ends)


def compute_factors(all_data, val_data, month_ends):
    """Compute factor values at each month-end."""
    records = []
    for mi, dt in enumerate(month_ends):
        if (mi + 1) % 12 == 0:
            print(f"    Month {mi+1}/{len(month_ends)} ({dt.date()})...")
        for code, df in all_data.items():
            hist = df[df.index <= dt]
            if len(hist) < 120:
                continue
            if (dt - hist.index[-1]).days > 5:
                continue

            last = hist.iloc[-1]
            n = len(hist)
            rec = {'date': dt, 'code': code, 'close': last['close']}

            # SIZE: log daily amount
            rec['size'] = np.log(hist['amount'].iloc[-20:].mean() + 1)

            # RETURNS
            if n >= 22:
                rec['ret_1m'] = hist['close'].iloc[-1] / hist['close'].iloc[-22] - 1
            if n >= 66:
                rec['ret_3m'] = hist['close'].iloc[-1] / hist['close'].iloc[-66] - 1
            if n >= 132:
                rec['mom_6m'] = hist['close'].iloc[-22] / hist['close'].iloc[-132] - 1
            if n >= 264:
                rec['mom_12m'] = hist['close'].iloc[-22] / hist['close'].iloc[-264] - 1

            # LOW VOLATILITY
            if n >= 20:
                daily_ret = hist['pctChg'].iloc[-20:] / 100.0
                rec['vol_20d'] = daily_ret.std() * np.sqrt(252)

            # TURNOVER
            if n >= 20:
                rec['turn_20d'] = hist['turnover'].iloc[-20:].mean()

            # ILLIQUIDITY (Amihud)
            if n >= 20:
                recent = hist.iloc[-20:]
                amihud = (recent['pctChg'].abs() / 100.0 / (recent['amount'] + 1)).mean()
                rec['illiquidity'] = amihud

            # VALUATION
            if code in val_data:
                vdf = val_data[code]
                vhist = vdf[vdf.index <= dt]
                if len(vhist) > 0:
                    vlast = vhist.iloc[-1]
                    pe = vlast.get('peTTM', np.nan)
                    pb = vlast.get('pbMRQ', np.nan)
                    if pd.notna(pe) and pe > 0:
                        rec['ep'] = 1.0 / pe
                    if pd.notna(pb) and pb > 0:
                        rec['bp'] = 1.0 / pb

            records.append(rec)

    return pd.DataFrame(records)


# ============================================================
# 3. TOP-N PORTFOLIO BACKTEST (new!)
# ============================================================

def build_composite_signal(cur, factor_weights):
    """
    Build composite signal from multiple factors.
    factor_weights: dict of {factor_name: (weight, ascending)}
      ascending=True means lower value = better (e.g., low turnover)
      ascending=False means higher value = better (e.g., high momentum)
    """
    composite = pd.Series(0.0, index=cur.index)
    valid_mask = pd.Series(True, index=cur.index)

    for fname, (weight, ascending) in factor_weights.items():
        if fname not in cur.columns:
            return None
        vals = cur[fname]
        valid_mask &= vals.notna()
        if ascending:
            # Lower is better -> rank ascending (1=best)
            rank_pct = vals.rank(ascending=True, pct=True)
        else:
            # Higher is better -> rank descending (1=best, but pct=True makes it 0~1)
            rank_pct = vals.rank(ascending=False, pct=True)
        composite += weight * rank_pct

    composite[~valid_mask] = np.nan
    return composite


def topn_portfolio_backtest(factor_df, strategy_name, factor_weights,
                            top_n=50, rebal_months=1, tc=TC_ONE_SIDE,
                            universe_filter=None):
    """
    Top-N portfolio backtest with transaction costs.

    Args:
        factor_weights: {factor_name: (weight, ascending)}
        top_n: number of stocks to hold
        rebal_months: rebalance every N months (1=monthly, 3=quarterly)
        tc: one-side transaction cost
        universe_filter: set of codes to restrict to (e.g., CSI300 only)
    """
    dates = sorted(factor_df['date'].unique())

    nav = 1.0
    nav_history = []
    prev_holdings = set()
    rebal_count = 0

    for i in range(len(dates) - 1):
        dt = dates[i]
        dt_next = dates[i + 1]

        cur = factor_df[factor_df['date'] == dt].copy()
        if universe_filter:
            cur = cur[cur['code'].isin(universe_filter)]

        # Check if rebalance month
        do_rebal = (rebal_count % rebal_months == 0)
        rebal_count += 1

        if do_rebal:
            # Build composite signal
            composite = build_composite_signal(cur, factor_weights)
            if composite is None:
                nav_history.append({'date': dt, 'nav': nav})
                continue

            cur = cur.copy()
            cur['signal'] = composite.values if len(composite) == len(cur) else composite
            cur = cur.dropna(subset=['signal'])

            if len(cur) < top_n:
                nav_history.append({'date': dt, 'nav': nav})
                continue

            # Select top N (lowest signal = best, since rank_pct ascending for "good")
            top = cur.nsmallest(top_n, 'signal')
            new_holdings = set(top['code'].tolist())
        else:
            new_holdings = prev_holdings.copy()

        # Get forward returns for held stocks
        nxt = factor_df[factor_df['date'] == dt_next][['code', 'close']].rename(
            columns={'close': 'close_next'})
        held = factor_df[(factor_df['date'] == dt) & (factor_df['code'].isin(new_holdings))]
        held = held.merge(nxt, on='code', how='inner')

        if len(held) == 0:
            nav_history.append({'date': dt, 'nav': nav})
            continue

        held['fwd_ret'] = held['close_next'] / held['close'] - 1
        # Winsorize
        p01, p99 = held['fwd_ret'].quantile([0.01, 0.99])
        held['fwd_ret'] = held['fwd_ret'].clip(p01, p99)

        port_ret = held['fwd_ret'].mean()

        # Transaction costs
        if do_rebal and prev_holdings:
            turnover = len(new_holdings - prev_holdings) / max(len(new_holdings), 1)
            tc_cost = turnover * tc * 2  # buy + sell
        elif not prev_holdings:
            tc_cost = tc  # initial buy only
        else:
            tc_cost = 0

        nav *= (1 + port_ret - tc_cost)
        nav_history.append({'date': dt, 'nav': nav})
        prev_holdings = new_holdings

    if not nav_history or len(nav_history) < 12:
        return None

    nav_df = pd.DataFrame(nav_history).set_index('date')
    total_days = (nav_df.index[-1] - nav_df.index[0]).days
    years = total_days / 365.25
    if years < 1:
        return None

    cagr = (nav_df['nav'].iloc[-1]) ** (1 / years) - 1
    monthly_ret = nav_df['nav'].pct_change().dropna()
    ann_vol = monthly_ret.std() * np.sqrt(12)
    sharpe = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    dd = nav_df['nav'] / nav_df['nav'].cummax() - 1
    max_dd = dd.min()

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        'strategy': strategy_name,
        'CAGR': round(cagr * 100, 2),
        'AnnVol': round(ann_vol * 100, 2),
        'Sharpe': round(sharpe, 3),
        'MaxDD': round(max_dd * 100, 2),
        'Calmar': round(calmar, 3),
        'FinalNAV': round(nav_df['nav'].iloc[-1], 4),
        'years': round(years, 1),
    }


# ============================================================
# 4. BENCHMARK: Equal Weight & CSI800
# ============================================================

def equal_weight_benchmark(factor_df, universe_filter=None):
    """Equal weight all stocks in universe each month."""
    dates = sorted(factor_df['date'].unique())
    nav = 1.0
    nav_history = []

    for i in range(len(dates) - 1):
        dt = dates[i]
        dt_next = dates[i + 1]

        cur = factor_df[factor_df['date'] == dt].copy()
        if universe_filter:
            cur = cur[cur['code'].isin(universe_filter)]

        nxt = factor_df[factor_df['date'] == dt_next][['code', 'close']].rename(
            columns={'close': 'close_next'})
        cur = cur.merge(nxt, on='code', how='inner')
        cur['fwd_ret'] = cur['close_next'] / cur['close'] - 1

        if len(cur) < 30:
            nav_history.append({'date': dt, 'nav': nav})
            continue

        p01, p99 = cur['fwd_ret'].quantile([0.01, 0.99])
        cur['fwd_ret'] = cur['fwd_ret'].clip(p01, p99)
        port_ret = cur['fwd_ret'].mean()
        nav *= (1 + port_ret)
        nav_history.append({'date': dt, 'nav': nav})

    if not nav_history or len(nav_history) < 12:
        return None

    nav_df = pd.DataFrame(nav_history).set_index('date')
    total_days = (nav_df.index[-1] - nav_df.index[0]).days
    years = total_days / 365.25
    cagr = (nav_df['nav'].iloc[-1]) ** (1 / years) - 1
    monthly_ret = nav_df['nav'].pct_change().dropna()
    ann_vol = monthly_ret.std() * np.sqrt(12)
    sharpe = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    dd = nav_df['nav'] / nav_df['nav'].cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        'strategy': 'EqualWeight',
        'CAGR': round(cagr * 100, 2),
        'AnnVol': round(ann_vol * 100, 2),
        'Sharpe': round(sharpe, 3),
        'MaxDD': round(max_dd * 100, 2),
        'Calmar': round(calmar, 3),
        'FinalNAV': round(nav_df['nav'].iloc[-1], 4),
        'years': round(years, 1),
    }


# ============================================================
# 5. FACTOR DECAY ANALYSIS
# ============================================================

def factor_decay_analysis(factor_df, factor_weights, top_n=50, max_hold_months=6):
    """Test how signal decays over different holding periods."""
    results = []
    for hold in range(1, max_hold_months + 1):
        r = topn_portfolio_backtest(
            factor_df,
            strategy_name=f'hold_{hold}m',
            factor_weights=factor_weights,
            top_n=top_n,
            rebal_months=hold,
            tc=TC_ONE_SIDE
        )
        if r:
            r['hold_months'] = hold
            results.append(r)
    return results


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("A股个股因子组合回测研究 V2 (Practical Portfolio)")
    print("=" * 70)
    t_start = time.time()

    # Step 1: Load data
    print("\n=== Step 1: Load Data ===")
    codes = get_csi800_codes()
    csi300_codes = get_csi300_codes()
    csi500_codes = [c for c in codes if c not in set(csi300_codes)]
    print(f"  CSI800: {len(codes)}, CSI300: {len(csi300_codes)}, CSI500: {len(csi500_codes)}")

    all_data = load_all_data(codes)
    val_data = load_valuation_data(list(all_data.keys()))

    # Step 2: Compute factors
    print(f"\n=== Step 2: Compute Factors ===")
    month_ends = get_month_ends(all_data)
    print(f"  Month-ends: {len(month_ends)} ({month_ends[0].date()} ~ {month_ends[-1].date()})")

    # Check for cached factor_df
    factor_cache = os.path.join(CACHE_DIR, 'factor_df_v2.pkl')
    if os.path.exists(factor_cache):
        mtime = os.path.getmtime(factor_cache)
        if time.time() - mtime < 86400 * 1:
            print("  Loading cached factor_df...")
            with open(factor_cache, 'rb') as f:
                factor_df = pickle.load(f)
            print(f"  Loaded {len(factor_df)} records")
        else:
            factor_df = None
    else:
        factor_df = None

    if factor_df is None:
        t0 = time.time()
        factor_df = compute_factors(all_data, val_data, month_ends)
        print(f"  Factor computation: {time.time()-t0:.0f}s, {len(factor_df)} records")
        with open(factor_cache, 'wb') as f:
            pickle.dump(factor_df, f)
        print(f"  Cached factor_df")

    print(f"  Avg stocks/month: {len(factor_df)/len(month_ends):.0f}")

    # Step 3: Benchmarks
    print(f"\n=== Step 3: Benchmarks ===")
    csi300_set = set(csi300_codes)
    csi500_set = set(csi500_codes)

    bm_all = equal_weight_benchmark(factor_df)
    bm_300 = equal_weight_benchmark(factor_df, universe_filter=csi300_set)
    bm_500 = equal_weight_benchmark(factor_df, universe_filter=csi500_set)

    print(f"  EW CSI800: CAGR={bm_all['CAGR']:.1f}%, Sharpe={bm_all['Sharpe']:.3f}, MaxDD={bm_all['MaxDD']:.1f}%")
    print(f"  EW CSI300: CAGR={bm_300['CAGR']:.1f}%, Sharpe={bm_300['Sharpe']:.3f}, MaxDD={bm_300['MaxDD']:.1f}%")
    print(f"  EW CSI500: CAGR={bm_500['CAGR']:.1f}%, Sharpe={bm_500['Sharpe']:.3f}, MaxDD={bm_500['MaxDD']:.1f}%")

    # Step 4: Strategy definitions
    print(f"\n=== Step 4: Portfolio Strategies ===")

    strategies = []

    # --- SINGLE FACTORS ---
    single_factors = {
        'LowTurn': {'turn_20d': (1.0, True)},
        'Mom6M': {'mom_6m': (1.0, False)},
        'Mom12M': {'mom_12m': (1.0, False)},
        'Value_EP': {'ep': (1.0, False)},
        'Value_BP': {'bp': (1.0, False)},
        'LowVol': {'vol_20d': (1.0, True)},
        'Reversal_1M': {'ret_1m': (1.0, True)},
        'Illiquidity': {'illiquidity': (1.0, False)},
    }

    # --- DOUBLE FACTORS ---
    double_factors = {
        'LowTurn+Mom6M': {'turn_20d': (1.0, True), 'mom_6m': (1.0, False)},
        'LowTurn+Mom12M': {'turn_20d': (1.0, True), 'mom_12m': (1.0, False)},
        'LowTurn+EP': {'turn_20d': (1.0, True), 'ep': (1.0, False)},
        'LowTurn+BP': {'turn_20d': (1.0, True), 'bp': (1.0, False)},
        'Mom6M+EP': {'mom_6m': (1.0, False), 'ep': (1.0, False)},
        'Mom12M+EP': {'mom_12m': (1.0, False), 'ep': (1.0, False)},
        'Mom6M+BP': {'mom_6m': (1.0, False), 'bp': (1.0, False)},
        'LowTurn+LowVol': {'turn_20d': (1.0, True), 'vol_20d': (1.0, True)},
        'LowVol+EP': {'vol_20d': (1.0, True), 'ep': (1.0, False)},
        'LowVol+Mom6M': {'vol_20d': (1.0, True), 'mom_6m': (1.0, False)},
        'Rev1M+LowTurn': {'ret_1m': (1.0, True), 'turn_20d': (1.0, True)},
    }

    # --- TRIPLE FACTORS ---
    triple_factors = {
        'LowTurn+Mom6M+EP': {'turn_20d': (1.0, True), 'mom_6m': (1.0, False), 'ep': (1.0, False)},
        'LowTurn+Mom12M+EP': {'turn_20d': (1.0, True), 'mom_12m': (1.0, False), 'ep': (1.0, False)},
        'LowTurn+Mom6M+BP': {'turn_20d': (1.0, True), 'mom_6m': (1.0, False), 'bp': (1.0, False)},
        'LowTurn+LowVol+EP': {'turn_20d': (1.0, True), 'vol_20d': (1.0, True), 'ep': (1.0, False)},
        'LowTurn+LowVol+Mom6M': {'turn_20d': (1.0, True), 'vol_20d': (1.0, True), 'mom_6m': (1.0, False)},
        'Mom6M+EP+LowVol': {'mom_6m': (1.0, False), 'ep': (1.0, False), 'vol_20d': (1.0, True)},
        'Rev1M+LowTurn+EP': {'ret_1m': (1.0, True), 'turn_20d': (1.0, True), 'ep': (1.0, False)},
    }

    # --- QUAD FACTORS ---
    quad_factors = {
        'LowTurn+Mom6M+EP+LowVol': {
            'turn_20d': (1.0, True), 'mom_6m': (1.0, False),
            'ep': (1.0, False), 'vol_20d': (1.0, True)
        },
        'LowTurn+Mom12M+EP+LowVol': {
            'turn_20d': (1.0, True), 'mom_12m': (1.0, False),
            'ep': (1.0, False), 'vol_20d': (1.0, True)
        },
    }

    all_factor_defs = {}
    all_factor_defs.update(single_factors)
    all_factor_defs.update(double_factors)
    all_factor_defs.update(triple_factors)
    all_factor_defs.update(quad_factors)

    all_results = []

    # --- Part A: CSI800 universe, different top_n ---
    print("\n--- Part A: CSI800, varying top_n, monthly rebalance ---")
    for top_n in [30, 50, 100]:
        for name, weights in all_factor_defs.items():
            strat_name = f'{name}_top{top_n}_M1'
            r = topn_portfolio_backtest(factor_df, strat_name, weights,
                                        top_n=top_n, rebal_months=1)
            if r:
                r['universe'] = 'CSI800'
                r['top_n'] = top_n
                r['rebal'] = '1M'
                r['factor_combo'] = name
                all_results.append(r)
                if top_n == 50:  # only print top50
                    print(f"  {strat_name:<35s} CAGR={r['CAGR']:>6.1f}% Sharpe={r['Sharpe']:>6.3f} MaxDD={r['MaxDD']:>6.1f}% Calmar={r['Calmar']:>6.3f}")

    # --- Part B: CSI300 only (large cap) ---
    print("\n--- Part B: CSI300 only, top 30, monthly ---")
    for name, weights in all_factor_defs.items():
        strat_name = f'{name}_CSI300_top30'
        r = topn_portfolio_backtest(factor_df, strat_name, weights,
                                    top_n=30, rebal_months=1, universe_filter=csi300_set)
        if r:
            r['universe'] = 'CSI300'
            r['top_n'] = 30
            r['rebal'] = '1M'
            r['factor_combo'] = name
            all_results.append(r)
            print(f"  {strat_name:<35s} CAGR={r['CAGR']:>6.1f}% Sharpe={r['Sharpe']:>6.3f} MaxDD={r['MaxDD']:>6.1f}%")

    # --- Part C: CSI500 only (mid cap) ---
    print("\n--- Part C: CSI500 only, top 50, monthly ---")
    for name, weights in all_factor_defs.items():
        strat_name = f'{name}_CSI500_top50'
        r = topn_portfolio_backtest(factor_df, strat_name, weights,
                                    top_n=50, rebal_months=1, universe_filter=csi500_set)
        if r:
            r['universe'] = 'CSI500'
            r['top_n'] = 50
            r['rebal'] = '1M'
            r['factor_combo'] = name
            all_results.append(r)
            print(f"  {strat_name:<35s} CAGR={r['CAGR']:>6.1f}% Sharpe={r['Sharpe']:>6.3f} MaxDD={r['MaxDD']:>6.1f}%")

    # --- Part D: Quarterly rebalance (top strategies) ---
    print("\n--- Part D: Quarterly rebalance, CSI800, top 50 ---")
    for name, weights in all_factor_defs.items():
        strat_name = f'{name}_top50_Q'
        r = topn_portfolio_backtest(factor_df, strat_name, weights,
                                    top_n=50, rebal_months=3)
        if r:
            r['universe'] = 'CSI800'
            r['top_n'] = 50
            r['rebal'] = '3M'
            r['factor_combo'] = name
            all_results.append(r)
            print(f"  {strat_name:<35s} CAGR={r['CAGR']:>6.1f}% Sharpe={r['Sharpe']:>6.3f} MaxDD={r['MaxDD']:>6.1f}%")

    # --- Part E: No TC comparison (top 3 combos) ---
    print("\n--- Part E: No TC comparison ---")
    best_by_sharpe = sorted([r for r in all_results if r['universe'] == 'CSI800' and r['top_n'] == 50 and r['rebal'] == '1M'],
                            key=lambda x: x['Sharpe'], reverse=True)[:5]
    for orig in best_by_sharpe:
        name = orig['factor_combo']
        weights = all_factor_defs.get(name)
        if not weights:
            continue
        strat_name = f'{name}_top50_noTC'
        r = topn_portfolio_backtest(factor_df, strat_name, weights,
                                    top_n=50, rebal_months=1, tc=0)
        if r:
            print(f"  {strat_name:<35s} CAGR={r['CAGR']:>6.1f}% (TC: {orig['CAGR']:>6.1f}%) "
                  f"Sharpe={r['Sharpe']:>6.3f} (TC: {orig['Sharpe']:>6.3f})")

    # --- Part F: Factor decay analysis (best combo) ---
    print("\n--- Part F: Factor Decay Analysis ---")
    if best_by_sharpe:
        best_name = best_by_sharpe[0]['factor_combo']
        best_weights = all_factor_defs[best_name]
        print(f"  Analyzing decay for: {best_name}")
        decay_results = factor_decay_analysis(factor_df, best_weights, top_n=50, max_hold_months=6)
        for dr in decay_results:
            print(f"  Hold {dr['hold_months']}M: CAGR={dr['CAGR']:>6.1f}%, Sharpe={dr['Sharpe']:>6.3f}, MaxDD={dr['MaxDD']:>6.1f}%")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: TOP 30 STRATEGIES (by Sharpe, CSI800, monthly rebal)")
    print("=" * 70)

    csi800_monthly = [r for r in all_results if r['universe'] == 'CSI800' and r['rebal'] == '1M']
    csi800_monthly.sort(key=lambda x: x['Sharpe'], reverse=True)

    hdr = f"{'Strategy':<40s} {'CAGR':>7s} {'Vol':>6s} {'Sharpe':>7s} {'MaxDD':>7s} {'Calmar':>7s} {'N':>4s}"
    print(hdr)
    print("-" * len(hdr))
    for r in csi800_monthly[:30]:
        print(f"{r['strategy']:<40s} {r['CAGR']:>6.1f}% {r['AnnVol']:>5.1f}% {r['Sharpe']:>7.3f} {r['MaxDD']:>6.1f}% {r['Calmar']:>7.3f} {r['top_n']:>4d}")

    # Benchmarks line
    print(f"\n{'--- BENCHMARKS ---':<40s}")
    for bm, label in [(bm_all, 'EW_CSI800'), (bm_300, 'EW_CSI300'), (bm_500, 'EW_CSI500')]:
        print(f"{label:<40s} {bm['CAGR']:>6.1f}% {bm['AnnVol']:>5.1f}% {bm['Sharpe']:>7.3f} {bm['MaxDD']:>6.1f}% {bm['Calmar']:>7.3f}")

    # Best by universe
    print(f"\n{'--- BEST BY UNIVERSE ---':<40s}")
    for univ in ['CSI800', 'CSI300', 'CSI500']:
        subset = [r for r in all_results if r['universe'] == univ]
        if subset:
            best = max(subset, key=lambda x: x['Sharpe'])
            print(f"  {univ}: {best['strategy']} → Sharpe={best['Sharpe']:.3f}, CAGR={best['CAGR']:.1f}%, MaxDD={best['MaxDD']:.1f}%")

    # Save all results
    out_path = os.path.join(DATA_DIR, 'stock_factor_v2_results.json')
    # Make results JSON-serializable
    save_results = {
        'benchmarks': {
            'EW_CSI800': bm_all,
            'EW_CSI300': bm_300,
            'EW_CSI500': bm_500,
        },
        'strategies': all_results,
        'best_by_sharpe_csi800': csi800_monthly[:10] if csi800_monthly else [],
    }
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Total: {len(all_results)} strategies tested in {total_time:.0f}s")
    print(f"Results saved to {out_path}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
