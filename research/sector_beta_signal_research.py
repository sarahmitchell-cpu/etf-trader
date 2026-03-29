#!/usr/bin/env python3
"""
行业Beta信号研究 (Sector Beta Signal Research)

研究内容:
  1. 各行业对大盘(沪深300)的滚动Beta
  2. Beta轮动策略: 牛市高beta, 熊市低beta
  3. Beta + 动量组合信号
  4. Beta regime切换
  5. 低beta异象 (low beta anomaly)
  6. Beta变化率信号 (beta acceleration)

数据: 中证行业指数 via akshare
基准: 沪深300
回测期: 尽可能长 (2012~2026)

Author: Sarah Mitchell / VisionClaw
Date: 2026-03-29
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import akshare as ak
import json, os, sys, time
from datetime import datetime

DATA_DIR = '/Users/claw/etf-trader/data'

# ============================================================
# 1. SECTOR INDEX DEFINITIONS
# ============================================================

SECTOR_INDICES = {
    # 中证全指行业 (GICS-like, all started ~2011)
    '全指消费': '000990',
    '全指金融': '000991',
    '全指信息': '000992',
    '全指医药': '000993',
    '全指能源': '000994',
    '全指材料': '000995',
    '全指工业': '000996',
    '全指可选': '000997',
    '全指公用': '000998',
    '全指电信': '000999',
    # Long-history theme indices only
    '中证军工': '930633',
    '证券公司': '399975',
    '中证银行': '399986',
    # Removed: 中证新能(399808), 中证半导(930665), 中证白酒(930606) - shorter history
}

MARKET_INDEX = '000300'  # 沪深300 as market proxy
RISK_FREE_RATE = 0.025   # ~2.5% annual

# ============================================================
# 2. DATA FETCHING
# ============================================================

def fetch_index_data(code, name, start='20120101', end='20260401'):
    """Fetch daily index data via akshare."""
    try:
        df = ak.index_zh_a_hist(symbol=code, period='daily',
                                start_date=start, end_date=end)
        if df is None or len(df) < 100:
            return None
        df = df.rename(columns={'日期': 'date', '收盘': 'close'})
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'close']].sort_values('date').reset_index(drop=True)
        df['close'] = df['close'].astype(float)
        df = df.set_index('date')
        df.columns = [name]
        return df
    except Exception as e:
        print(f"  ERROR fetching {code} ({name}): {e}")
        return None


def load_all_data():
    """Load market + sector data, align dates."""
    print("Loading market index (沪深300)...")
    mkt = fetch_index_data(MARKET_INDEX, 'market')
    if mkt is None:
        raise ValueError("Cannot fetch market data")

    sectors = {}
    for name, code in SECTOR_INDICES.items():
        print(f"  Loading {name} ({code})...")
        df = fetch_index_data(code, name)
        if df is not None:
            sectors[name] = df
        time.sleep(0.3)

    # Merge all - use outer join so we keep full market history
    all_data = mkt.copy()
    for name, df in sectors.items():
        all_data = all_data.join(df, how='outer')

    # Forward fill small gaps, keep NaN for sectors that don't exist yet
    all_data = all_data.sort_index()
    # Only keep rows where market exists
    all_data = all_data.dropna(subset=['market'])

    print(f"\nLoaded {len(sectors)} sectors, {len(all_data)} trading days")
    print(f"Period: {all_data.index[0].date()} ~ {all_data.index[-1].date()}")
    # Show per-sector data availability
    for c in all_data.columns:
        valid = all_data[c].dropna()
        if len(valid) > 0:
            print(f"  {c:12s}: {valid.index[0].date()} ~ {valid.index[-1].date()} ({len(valid)} rows)")
    return all_data


# ============================================================
# 3. BETA CALCULATION
# ============================================================

def calc_rolling_beta(sector_ret, market_ret, window=60):
    """Calculate rolling beta using OLS."""
    cov = sector_ret.rolling(window).cov(market_ret)
    var = market_ret.rolling(window).var()
    beta = cov / var
    return beta


def calc_all_betas(data, windows=[20, 60, 120, 250]):
    """Calculate rolling betas for all sectors."""
    rets = data.pct_change()
    mkt_ret = rets['market']
    sector_names = [c for c in data.columns if c != 'market']

    betas = {}
    for w in windows:
        beta_df = pd.DataFrame(index=data.index)
        for name in sector_names:
            beta_df[name] = calc_rolling_beta(rets[name], mkt_ret, w)
        betas[w] = beta_df

    return rets, betas


# ============================================================
# 4. MARKET REGIME DETECTION
# ============================================================

def detect_regime(mkt_data, ma_period=60):
    """Detect bull/bear regime using MA."""
    mkt_close = mkt_data['market']
    ma = mkt_close.rolling(ma_period).mean()
    regime = (mkt_close > ma).astype(int)  # 1=bull, 0=bear
    return regime


def detect_regime_dual_ma(mkt_data, fast=20, slow=120):
    """Dual MA regime detection."""
    mkt_close = mkt_data['market']
    ma_fast = mkt_close.rolling(fast).mean()
    ma_slow = mkt_close.rolling(slow).mean()
    regime = (ma_fast > ma_slow).astype(int)
    return regime


# ============================================================
# 5. STRATEGY IMPLEMENTATIONS
# ============================================================

def strategy_equal_weight(rets, sector_names):
    """Baseline: equal weight all sectors."""
    strat_ret = rets[sector_names].mean(axis=1)
    return strat_ret


def strategy_beta_rotation(rets, betas_df, regime, sector_names,
                           n_high=3, n_low=3, txn_cost_bps=15):
    """
    Beta rotation: high-beta sectors in bull, low-beta in bear.
    Rebalance weekly.
    """
    # Weekly rebalance dates
    dates = rets.index
    rebal_dates = rets.resample('W-FRI').last().index

    positions = pd.DataFrame(0.0, index=dates, columns=sector_names)

    for i, d in enumerate(rebal_dates):
        if d not in betas_df.index:
            continue
        beta_row = betas_df.loc[d, sector_names].dropna()
        if len(beta_row) < n_high + n_low:
            continue

        reg = regime.loc[:d].iloc[-1] if d in regime.index else regime.asof(d)

        if reg == 1:  # Bull - buy high beta
            top = beta_row.nlargest(n_high).index.tolist()
            weight = 1.0 / n_high
            for s in top:
                # Hold until next rebalance
                next_d = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
                mask = (dates > d) & (dates <= next_d)
                positions.loc[mask, s] = weight
        else:  # Bear - buy low beta
            bot = beta_row.nsmallest(n_low).index.tolist()
            weight = 1.0 / n_low
            for s in bot:
                next_d = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
                mask = (dates > d) & (dates <= next_d)
                positions.loc[mask, s] = weight

    strat_ret = (positions * rets[sector_names]).sum(axis=1)

    # Transaction costs
    turnover = positions.diff().abs().sum(axis=1)
    strat_ret -= turnover * txn_cost_bps / 10000

    return strat_ret


def strategy_low_beta(rets, betas_df, sector_names, n=3, txn_cost_bps=15):
    """Low beta anomaly: always buy lowest beta sectors."""
    dates = rets.index
    rebal_dates = rets.resample('W-FRI').last().index
    positions = pd.DataFrame(0.0, index=dates, columns=sector_names)

    for i, d in enumerate(rebal_dates):
        if d not in betas_df.index:
            continue
        beta_row = betas_df.loc[d, sector_names].dropna()
        if len(beta_row) < n:
            continue
        bot = beta_row.nsmallest(n).index.tolist()
        weight = 1.0 / n
        next_d = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        mask = (dates > d) & (dates <= next_d)
        for s in bot:
            positions.loc[mask, s] = weight

    strat_ret = (positions * rets[sector_names]).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    strat_ret -= turnover * txn_cost_bps / 10000
    return strat_ret


def strategy_high_beta(rets, betas_df, sector_names, n=3, txn_cost_bps=15):
    """Always buy highest beta sectors."""
    dates = rets.index
    rebal_dates = rets.resample('W-FRI').last().index
    positions = pd.DataFrame(0.0, index=dates, columns=sector_names)

    for i, d in enumerate(rebal_dates):
        if d not in betas_df.index:
            continue
        beta_row = betas_df.loc[d, sector_names].dropna()
        if len(beta_row) < n:
            continue
        top = beta_row.nlargest(n).index.tolist()
        weight = 1.0 / n
        next_d = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        mask = (dates > d) & (dates <= next_d)
        for s in top:
            positions.loc[mask, s] = weight

    strat_ret = (positions * rets[sector_names]).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    strat_ret -= turnover * txn_cost_bps / 10000
    return strat_ret


def strategy_beta_momentum_combo(rets, betas_df, sector_names,
                                  mom_lookback=20, n=3, txn_cost_bps=15):
    """
    Combo: rank by beta-adjusted momentum (alpha = ret - beta * mkt_ret).
    Buy top alpha sectors.
    """
    dates = rets.index
    rebal_dates = rets.resample('W-FRI').last().index
    positions = pd.DataFrame(0.0, index=dates, columns=sector_names)

    for i, d in enumerate(rebal_dates):
        if d not in betas_df.index:
            continue
        idx = dates.get_loc(d)
        if idx < mom_lookback:
            continue

        beta_row = betas_df.loc[d, sector_names].dropna()
        valid_sectors = beta_row.index.tolist()
        if len(valid_sectors) < n:
            continue

        # Past mom_lookback returns
        past_rets = rets.iloc[idx - mom_lookback:idx]
        sector_mom = past_rets[valid_sectors].sum()
        mkt_mom = past_rets['market'].sum()

        # Alpha = sector_return - beta * market_return
        alpha = sector_mom - beta_row[valid_sectors] * mkt_mom

        top = alpha.nlargest(n).index.tolist()
        weight = 1.0 / n
        next_d = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        mask = (dates > d) & (dates <= next_d)
        for s in top:
            positions.loc[mask, s] = weight

    strat_ret = (positions * rets[sector_names]).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    strat_ret -= turnover * txn_cost_bps / 10000
    return strat_ret


def strategy_beta_change(rets, betas_df, sector_names,
                          change_lookback=20, n=3, txn_cost_bps=15):
    """
    Beta acceleration: buy sectors with rising beta (beta increasing).
    """
    dates = rets.index
    rebal_dates = rets.resample('W-FRI').last().index
    positions = pd.DataFrame(0.0, index=dates, columns=sector_names)

    beta_change = betas_df[sector_names].diff(change_lookback)

    for i, d in enumerate(rebal_dates):
        if d not in beta_change.index:
            continue
        chg = beta_change.loc[d, sector_names].dropna()
        if len(chg) < n:
            continue
        top = chg.nlargest(n).index.tolist()
        weight = 1.0 / n
        next_d = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        mask = (dates > d) & (dates <= next_d)
        for s in top:
            positions.loc[mask, s] = weight

    strat_ret = (positions * rets[sector_names]).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    strat_ret -= turnover * txn_cost_bps / 10000
    return strat_ret


def strategy_beta_regime_timed(rets, betas_df, regime, sector_names,
                                n_high=3, n_low=3, txn_cost_bps=15):
    """
    Beta rotation with market timing:
    Bull: top-N high beta sectors
    Bear: cash (0 exposure)
    """
    dates = rets.index
    rebal_dates = rets.resample('W-FRI').last().index
    positions = pd.DataFrame(0.0, index=dates, columns=sector_names)

    for i, d in enumerate(rebal_dates):
        if d not in betas_df.index:
            continue
        beta_row = betas_df.loc[d, sector_names].dropna()
        if len(beta_row) < n_high:
            continue

        reg = regime.loc[:d].iloc[-1] if d in regime.index else regime.asof(d)

        if reg == 1:  # Bull
            top = beta_row.nlargest(n_high).index.tolist()
            weight = 1.0 / n_high
            next_d = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
            mask = (dates > d) & (dates <= next_d)
            for s in top:
                positions.loc[mask, s] = weight
        # Bear: stay cash (positions already 0)

    strat_ret = (positions * rets[sector_names]).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    strat_ret -= turnover * txn_cost_bps / 10000
    return strat_ret


def strategy_beta_weighted(rets, betas_df, regime, sector_names, txn_cost_bps=15):
    """
    Beta-weighted: weight sectors proportional to beta in bull,
    inverse-beta weighted in bear.
    """
    dates = rets.index
    rebal_dates = rets.resample('W-FRI').last().index
    positions = pd.DataFrame(0.0, index=dates, columns=sector_names)

    for i, d in enumerate(rebal_dates):
        if d not in betas_df.index:
            continue
        beta_row = betas_df.loc[d, sector_names].dropna()
        if len(beta_row) < 3:
            continue
        # Clip betas to positive
        beta_pos = beta_row.clip(lower=0.1)

        reg = regime.loc[:d].iloc[-1] if d in regime.index else regime.asof(d)

        if reg == 1:  # Bull - weight by beta
            weights = beta_pos / beta_pos.sum()
        else:  # Bear - weight by inverse beta
            inv_beta = 1.0 / beta_pos
            weights = inv_beta / inv_beta.sum()

        next_d = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        mask = (dates > d) & (dates <= next_d)
        for s in weights.index:
            positions.loc[mask, s] = weights[s]

    strat_ret = (positions * rets[sector_names]).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    strat_ret -= turnover * txn_cost_bps / 10000
    return strat_ret


# ============================================================
# 6. BACKTEST ENGINE
# ============================================================

def calc_metrics(strat_ret, name, start_date=None):
    """Calculate strategy performance metrics."""
    if start_date:
        strat_ret = strat_ret.loc[start_date:]

    strat_ret = strat_ret.dropna()
    if len(strat_ret) < 252:
        return None

    cum = (1 + strat_ret).cumprod()
    years = len(strat_ret) / 252
    total_ret = cum.iloc[-1] - 1
    cagr = (1 + total_ret) ** (1 / years) - 1

    vol = strat_ret.std() * np.sqrt(252)
    sharpe = (cagr - RISK_FREE_RATE) / vol if vol > 0 else 0

    peak = cum.cummax()
    dd = cum / peak - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Monthly win rate
    monthly = strat_ret.resample('ME').sum()
    win_rate = (monthly > 0).mean() * 100

    return {
        'name': name,
        'cagr': round(cagr * 100, 2),
        'vol': round(vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'max_dd': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'win_rate_m': round(win_rate, 1),
        'years': round(years, 1),
        'total_ret': round(total_ret * 100, 1),
    }


# ============================================================
# 7. MAIN RESEARCH
# ============================================================

def main():
    print("=" * 70)
    print("行业Beta信号研究 (Sector Beta Signal Research)")
    print("=" * 70)

    # Load data
    data = load_all_data()
    sector_names = [c for c in data.columns if c != 'market']
    print(f"Sectors: {sector_names}")

    # Calculate returns and betas
    print("\nCalculating rolling betas...")
    rets, betas = calc_all_betas(data, windows=[20, 60, 120, 250])

    # Detect market regimes
    print("Detecting market regimes...")
    regime_ma60 = detect_regime(data, 60)
    regime_ma120 = detect_regime(data, 120)
    regime_dual = detect_regime_dual_ma(data, 20, 120)

    # Start date for backtest (need enough data for beta calc)
    start = '2013-01-01'

    results = []

    # ---- BASELINES ----
    print("\n--- Baselines ---")
    # Equal weight
    ew_ret = strategy_equal_weight(rets, sector_names)
    m = calc_metrics(ew_ret, 'EqualWeight', start)
    if m:
        m['strategy_type'] = 'baseline'
        results.append(m)
        print(f"  {m['name']:45s} CAGR={m['cagr']:5.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:6.1f}%")

    # Market (沪深300)
    mkt_m = calc_metrics(rets['market'], 'Market_CSI300', start)
    if mkt_m:
        mkt_m['strategy_type'] = 'baseline'
        results.append(mkt_m)
        print(f"  {mkt_m['name']:45s} CAGR={mkt_m['cagr']:5.1f}% Sharpe={mkt_m['sharpe']:.3f} MaxDD={mkt_m['max_dd']:6.1f}%")

    # ---- LOW BETA / HIGH BETA ----
    print("\n--- Low/High Beta Strategies ---")
    for beta_window in [60, 120, 250]:
        for n in [2, 3, 4, 5]:
            # Low beta
            sr = strategy_low_beta(rets, betas[beta_window], sector_names, n=n)
            name = f"LowBeta_B{beta_window}_N{n}"
            m = calc_metrics(sr, name, start)
            if m:
                m['strategy_type'] = 'low_beta'
                results.append(m)
                print(f"  {m['name']:45s} CAGR={m['cagr']:5.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:6.1f}%")

            # High beta
            sr = strategy_high_beta(rets, betas[beta_window], sector_names, n=n)
            name = f"HighBeta_B{beta_window}_N{n}"
            m = calc_metrics(sr, name, start)
            if m:
                m['strategy_type'] = 'high_beta'
                results.append(m)
                print(f"  {m['name']:45s} CAGR={m['cagr']:5.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:6.1f}%")

    # ---- BETA ROTATION (regime-dependent) ----
    print("\n--- Beta Rotation Strategies ---")
    regimes = {
        'MA60': regime_ma60,
        'MA120': regime_ma120,
        'DualMA': regime_dual,
    }
    for reg_name, regime in regimes.items():
        for beta_window in [60, 120]:
            for n in [2, 3, 4]:
                sr = strategy_beta_rotation(
                    rets, betas[beta_window], regime, sector_names,
                    n_high=n, n_low=n)
                name = f"BetaRot_{reg_name}_B{beta_window}_N{n}"
                m = calc_metrics(sr, name, start)
                if m:
                    m['strategy_type'] = 'beta_rotation'
                    results.append(m)
                    print(f"  {m['name']:45s} CAGR={m['cagr']:5.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:6.1f}%")

    # ---- BETA ROTATION + TIMING (bear=cash) ----
    print("\n--- Beta Rotation + Market Timing (bear=cash) ---")
    for reg_name, regime in regimes.items():
        for beta_window in [60, 120]:
            for n in [2, 3, 4]:
                sr = strategy_beta_regime_timed(
                    rets, betas[beta_window], regime, sector_names,
                    n_high=n)
                name = f"BetaTimedCash_{reg_name}_B{beta_window}_N{n}"
                m = calc_metrics(sr, name, start)
                if m:
                    m['strategy_type'] = 'beta_timed_cash'
                    results.append(m)
                    print(f"  {m['name']:45s} CAGR={m['cagr']:5.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:6.1f}%")

    # ---- BETA-WEIGHTED ----
    print("\n--- Beta-Weighted Strategies ---")
    for reg_name, regime in regimes.items():
        for beta_window in [60, 120]:
            sr = strategy_beta_weighted(rets, betas[beta_window], regime, sector_names)
            name = f"BetaWeighted_{reg_name}_B{beta_window}"
            m = calc_metrics(sr, name, start)
            if m:
                m['strategy_type'] = 'beta_weighted'
                results.append(m)
                print(f"  {m['name']:45s} CAGR={m['cagr']:5.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:6.1f}%")

    # ---- BETA-ADJUSTED MOMENTUM (ALPHA) ----
    print("\n--- Beta-Adjusted Momentum (Alpha) ---")
    for beta_window in [60, 120]:
        for mom_lb in [10, 20, 60]:
            for n in [2, 3, 4]:
                sr = strategy_beta_momentum_combo(
                    rets, betas[beta_window], sector_names,
                    mom_lookback=mom_lb, n=n)
                name = f"BetaAlpha_B{beta_window}_M{mom_lb}_N{n}"
                m = calc_metrics(sr, name, start)
                if m:
                    m['strategy_type'] = 'beta_alpha'
                    results.append(m)
                    print(f"  {m['name']:45s} CAGR={m['cagr']:5.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:6.1f}%")

    # ---- BETA CHANGE (ACCELERATION) ----
    print("\n--- Beta Change / Acceleration ---")
    for beta_window in [60, 120]:
        for chg_lb in [10, 20, 40]:
            for n in [2, 3, 4]:
                sr = strategy_beta_change(
                    rets, betas[beta_window], sector_names,
                    change_lookback=chg_lb, n=n)
                name = f"BetaChg_B{beta_window}_C{chg_lb}_N{n}"
                m = calc_metrics(sr, name, start)
                if m:
                    m['strategy_type'] = 'beta_change'
                    results.append(m)
                    print(f"  {m['name']:45s} CAGR={m['cagr']:5.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:6.1f}%")

    # ---- SUMMARY ----
    print("\n" + "=" * 70)
    print(f"Total strategies: {len(results)}")
    results.sort(key=lambda x: x['sharpe'], reverse=True)

    print("\n=== TOP 20 BY SHARPE ===")
    for r in results[:20]:
        print(f"  {r['name']:45s} type={r['strategy_type']:16s} "
              f"CAGR={r['cagr']:5.1f}% Sharpe={r['sharpe']:.3f} "
              f"MaxDD={r['max_dd']:6.1f}% Calmar={r['calmar']:.3f}")

    print("\n=== TOP 10 BY CALMAR ===")
    results_c = sorted(results, key=lambda x: x['calmar'], reverse=True)
    for r in results_c[:10]:
        print(f"  {r['name']:45s} type={r['strategy_type']:16s} "
              f"CAGR={r['cagr']:5.1f}% Sharpe={r['sharpe']:.3f} "
              f"MaxDD={r['max_dd']:6.1f}% Calmar={r['calmar']:.3f}")

    print("\n=== STRATEGY TYPE AVERAGES ===")
    type_groups = {}
    for r in results:
        t = r['strategy_type']
        if t not in type_groups:
            type_groups[t] = []
        type_groups[t].append(r)

    for t, group in sorted(type_groups.items()):
        avg_cagr = np.mean([r['cagr'] for r in group])
        avg_sharpe = np.mean([r['sharpe'] for r in group])
        avg_dd = np.mean([r['max_dd'] for r in group])
        best_sharpe = max(r['sharpe'] for r in group)
        print(f"  {t:20s}: n={len(group):3d}, avgCAGR={avg_cagr:5.1f}%, "
              f"avgSharpe={avg_sharpe:.3f}, avgMaxDD={avg_dd:6.1f}%, bestSharpe={best_sharpe:.3f}")

    # Save results
    output = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'description': '行业Beta信号研究',
        'sectors': list(SECTOR_INDICES.keys()),
        'market': 'CSI300',
        'period': f"{data.index[0].date()} ~ {data.index[-1].date()}",
        'backtest_start': start,
        'total_strategies': len(results),
        'all_results': results,
    }

    out_path = os.path.join(DATA_DIR, 'sector_beta_signal_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")


if __name__ == '__main__':
    main()
