#!/usr/bin/env python3
"""
CSI500 Dynamic Allocation Research - Phase 3
Goal: Preserve most of the SR's 22% CAGR while reducing MDD from -27%.

Strategies tested:
  1. Regime-Switch: 100% SR normally, shift to blend only when risk is high
  2. Adaptive Allocation: Dynamically adjust SR/QM ratio based on market regime
  3. Portfolio-Level Trailing Stop: Cut total exposure when portfolio draws down
  4. SR + Sector Trend Confirmation: Only enter sectors that are BOTH top-momentum AND above their own MA
  5. SR with Wider Diversification Variants: top4/top5 sectors with more stocks
  6. SR + Drawdown-Triggered Cash Buffer: Go partial cash after portfolio peak drawdown exceeds threshold
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import json
import os
import sys
import time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi500_quality_momentum_research import (
    CSI500_POOL, load_all_data, get_momentum, get_volatility,
    rank_percentile, inverse_rank, DATA_DIR
)
from csi500_drawdown_reduction import (
    calc_stats, build_pool_index, get_sr_selections, get_qm_selections
)


# ============================================================
# Strategy 1: Regime Switch
# Normal: 100% SR. When market risk is high: shift to blend.
# Risk signals: pool index below MA, or recent vol spike
# ============================================================

def run_regime_switch(price_df, ma_weeks=15, blend_sr_pct=0.3, rebal_freq=4, txn_bps=8):
    """
    Normal regime: 100% SR (full returns)
    Risk regime (index < MA): blend SR*blend_sr_pct + QM*(1-blend_sr_pct)
    """
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    sr_sel = get_sr_selections(price_df, 12, 0, 3, 2)
    qm_sel = get_qm_selections(price_df, 20, 24, 0, 8, 0.3, 0.7)
    pool_idx = build_pool_index(price_df)

    warmup = max(26, ma_weeks + 2)
    nav = [1.0]
    dates = []
    weekly_rets = []
    prev_sr = set()
    prev_qm = set()
    total_txn = 0.0
    risk_weeks = 0
    total_weeks = 0

    i = warmup
    while i < len(price_df) - 1:
        if i not in sr_sel:
            i += 1
            continue

        # Determine regime
        idx_slice = pool_idx.iloc[max(0, i-ma_weeks):i+1]
        if len(idx_slice) < ma_weeks:
            i += 1
            continue
        ma_val = idx_slice.rolling(ma_weeks).mean().iloc[-1]
        current_val = idx_slice.iloc[-1]
        is_risky = current_val < ma_val

        sr_stocks = sr_sel[i]
        qm_stocks = qm_sel.get(i, sr_stocks)  # fallback to SR if QM not available

        hold_end = min(i + rebal_freq, len(price_df) - 1)

        if is_risky:
            # Blend mode
            sr_w = blend_sr_pct
            qm_w = 1.0 - blend_sr_pct
            risk_weeks += (hold_end - i)

            sr_set = set(sr_stocks)
            qm_set = set(qm_stocks)
            turnover_sr = len(sr_set - prev_sr) / max(len(sr_set), 1)
            turnover_qm = len(qm_set - prev_qm) / max(len(qm_set), 1)
            period_txn = (turnover_sr * sr_w + turnover_qm * qm_w) * txn_cost
            total_txn += period_txn

            for j in range(i + 1, hold_end + 1):
                sr_rets = [float(returns[s].iloc[j]) for s in sr_stocks if not pd.isna(returns[s].iloc[j])]
                qm_rets = [float(returns[s].iloc[j]) for s in qm_stocks if not pd.isna(returns[s].iloc[j])]
                sr_ret = np.mean(sr_rets) if sr_rets else 0.0
                qm_ret = np.mean(qm_rets) if qm_rets else 0.0
                port_ret = sr_w * sr_ret + qm_w * qm_ret
                if j == i + 1:
                    port_ret -= period_txn
                nav.append(nav[-1] * (1 + port_ret))
                dates.append(price_df.index[j])
                weekly_rets.append(port_ret)
                total_weeks += 1

            prev_sr = sr_set
            prev_qm = qm_set
        else:
            # Normal mode: 100% SR
            selected_set = set(sr_stocks)
            turnover = (len(selected_set - prev_sr) + len(prev_sr - selected_set)) / max(len(selected_set), 1)
            period_txn = turnover * txn_cost
            total_txn += period_txn

            for j in range(i + 1, hold_end + 1):
                rets = [float(returns[s].iloc[j]) for s in sr_stocks if not pd.isna(returns[s].iloc[j])]
                port_ret = np.mean(rets) if rets else 0.0
                if j == i + 1:
                    port_ret -= period_txn
                nav.append(nav[-1] * (1 + port_ret))
                dates.append(price_df.index[j])
                weekly_rets.append(port_ret)
                total_weeks += 1

            prev_sr = selected_set
            prev_qm = set()

        i = hold_end

    label = f"RegimeSwitch_MA{ma_weeks}_blend{int(blend_sr_pct*100)}"
    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'method': 'regime_switch', 'ma_weeks': ma_weeks,
                       'blend_sr_pct': blend_sr_pct,
                       'risk_pct': round(risk_weeks / max(total_weeks, 1) * 100, 1)})


# ============================================================
# Strategy 2: Adaptive Allocation (continuous)
# SR weight = f(regime_score). Stronger trend = more SR.
# ============================================================

def run_adaptive_allocation(price_df, ma_weeks=15, min_sr=0.2, max_sr=1.0, rebal_freq=4, txn_bps=8):
    """
    SR weight varies continuously based on how far above/below MA the index is.
    Above MA by 5%+ => max_sr (e.g., 100% SR)
    Below MA by 5%+ => min_sr (e.g., 20% SR, 80% QM)
    Linear interpolation in between.
    """
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    sr_sel = get_sr_selections(price_df, 12, 0, 3, 2)
    qm_sel = get_qm_selections(price_df, 20, 24, 0, 8, 0.3, 0.7)
    pool_idx = build_pool_index(price_df)

    warmup = max(26, ma_weeks + 2)
    nav = [1.0]
    dates = []
    weekly_rets = []
    prev_sr = set()
    prev_qm = set()
    total_txn = 0.0

    i = warmup
    while i < len(price_df) - 1:
        if i not in sr_sel:
            i += 1
            continue

        idx_slice = pool_idx.iloc[max(0, i-ma_weeks):i+1]
        if len(idx_slice) < ma_weeks:
            i += 1
            continue
        ma_val = idx_slice.rolling(ma_weeks).mean().iloc[-1]
        current_val = idx_slice.iloc[-1]

        # Deviation from MA as percentage
        dev = (current_val - ma_val) / ma_val
        # Map [-0.05, +0.05] -> [min_sr, max_sr]
        sr_weight = min_sr + (max_sr - min_sr) * np.clip((dev + 0.05) / 0.10, 0, 1)
        qm_weight = 1.0 - sr_weight

        sr_stocks = sr_sel[i]
        qm_stocks = qm_sel.get(i, sr_stocks)

        sr_set = set(sr_stocks)
        qm_set = set(qm_stocks)
        turnover_sr = len(sr_set - prev_sr) / max(len(sr_set), 1)
        turnover_qm = len(qm_set - prev_qm) / max(len(qm_set), 1)
        period_txn = (turnover_sr * sr_weight + turnover_qm * qm_weight) * txn_cost
        total_txn += period_txn

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            sr_rets = [float(returns[s].iloc[j]) for s in sr_stocks if not pd.isna(returns[s].iloc[j])]
            qm_rets = [float(returns[s].iloc[j]) for s in qm_stocks if not pd.isna(returns[s].iloc[j])]
            sr_ret = np.mean(sr_rets) if sr_rets else 0.0
            qm_ret = np.mean(qm_rets) if qm_rets else 0.0
            port_ret = sr_weight * sr_ret + qm_weight * qm_ret
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_sr = sr_set
        prev_qm = qm_set
        i = hold_end

    label = f"Adaptive_MA{ma_weeks}_sr{int(min_sr*100)}-{int(max_sr*100)}"
    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'method': 'adaptive', 'ma_weeks': ma_weeks,
                       'min_sr': min_sr, 'max_sr': max_sr})


# ============================================================
# Strategy 3: Portfolio-Level Trailing Stop
# When portfolio NAV drops X% from peak, reduce to Y% exposure
# ============================================================

def run_portfolio_trailing_stop(price_df, stop_pct=-0.10, reduced_exposure=0.3,
                                 recovery_pct=0.0, rebal_freq=4, txn_bps=8):
    """
    Track portfolio NAV peak. When drawdown exceeds stop_pct, reduce exposure.
    Resume full exposure when drawdown recovers to recovery_pct.
    """
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    sr_sel = get_sr_selections(price_df, 12, 0, 3, 2)

    warmup = 14
    nav = [1.0]
    dates = []
    weekly_rets = []
    prev_holdings = set()
    total_txn = 0.0
    peak_nav = 1.0
    in_protection = False

    i = warmup
    while i < len(price_df) - 1:
        if i not in sr_sel:
            i += 1
            continue

        # Check if we should be in protection mode
        current_nav = nav[-1]
        if current_nav > peak_nav:
            peak_nav = current_nav
        dd = (current_nav - peak_nav) / peak_nav

        if dd <= stop_pct and not in_protection:
            in_protection = True
        elif dd >= recovery_pct and in_protection:
            in_protection = False
            peak_nav = current_nav  # Reset peak

        exposure = reduced_exposure if in_protection else 1.0

        selected = sr_sel[i]
        selected_set = set(selected)
        turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            rets = [float(returns[s].iloc[j]) for s in selected if not pd.isna(returns[s].iloc[j])]
            port_ret = np.mean(rets) if rets else 0.0
            port_ret *= exposure
            if j == i + 1:
                port_ret -= period_txn * exposure
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

            # Update peak/dd tracking intra-period
            if nav[-1] > peak_nav:
                peak_nav = nav[-1]

        prev_holdings = selected_set
        i = hold_end

    label = f"PortStop_{int(abs(stop_pct)*100)}pct_exp{int(reduced_exposure*100)}_rec{int(recovery_pct*100)}"
    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'method': 'portfolio_stop', 'stop_pct': stop_pct,
                       'reduced_exposure': reduced_exposure, 'recovery_pct': recovery_pct})


# ============================================================
# Strategy 4: Sector Trend Confirmation
# Only enter a sector if it's BOTH top-momentum AND above its own MA
# ============================================================

def run_sector_trend_confirm(price_df, mom_lookback=12, sector_ma=12,
                              top_sectors=3, stocks_per_sector=2,
                              fallback_to_qm=True, rebal_freq=4, txn_bps=8):
    """
    Like SR but with an additional filter:
    Only enter sectors whose own sector index is above its MA.
    If a top-3 sector fails the filter, skip it (hold fewer stocks or use QM).
    """
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()

    sector_stocks = defaultdict(list)
    for s in price_df.columns:
        sec = CSI500_POOL.get(s, {}).get('sector', '?')
        sector_stocks[sec].append(s)

    # Build sector indices
    sector_indices = {}
    for sec, stocks in sector_stocks.items():
        sec_rets = returns[stocks].mean(axis=1)
        sector_indices[sec] = (1 + sec_rets).cumprod()

    qm_sel = get_qm_selections(price_df, 20, 24, 0, 8, 0.3, 0.7) if fallback_to_qm else {}

    warmup = max(mom_lookback + 2, sector_ma + 2)
    nav = [1.0]
    dates = []
    weekly_rets = []
    prev_holdings = set()
    total_txn = 0.0

    i = warmup
    while i < len(price_df) - 1:
        mom = get_momentum(price_df, i, mom_lookback)

        sector_mom = {}
        for sec, stocks in sector_stocks.items():
            sec_moms = [mom[s] for s in stocks if s in mom]
            if sec_moms:
                sector_mom[sec] = np.mean(sec_moms)

        if len(sector_mom) < 3:
            i += 1
            continue

        # Rank sectors by momentum
        ranked_sectors = sorted(sector_mom.items(), key=lambda x: -x[1])

        # Filter: sector must be above its own MA
        selected = []
        sectors_tried = 0
        for sec, _ in ranked_sectors:
            if sectors_tried >= top_sectors + 3:  # look at a few more sectors as backup
                break
            sec_idx = sector_indices.get(sec)
            if sec_idx is None or len(sec_idx) < sector_ma:
                continue

            sec_slice = sec_idx.iloc[max(0, i-sector_ma):i+1]
            if len(sec_slice) < sector_ma:
                continue
            sec_ma = sec_slice.rolling(sector_ma).mean().iloc[-1]
            sec_current = sec_slice.iloc[-1]

            if sec_current > sec_ma:
                # Sector confirmed
                stocks = sector_stocks[sec]
                stock_moms = [(s, mom[s]) for s in stocks if s in mom]
                stock_moms.sort(key=lambda x: -x[1])
                for s, _ in stock_moms[:stocks_per_sector]:
                    selected.append(s)
                sectors_tried += 1
                if sectors_tried >= top_sectors:
                    break
            else:
                sectors_tried += 1

        # If fewer sectors confirmed, fill with QM stocks
        if fallback_to_qm and len(selected) < top_sectors * stocks_per_sector:
            qm_stocks = qm_sel.get(i, [])
            needed = top_sectors * stocks_per_sector - len(selected)
            for s in qm_stocks:
                if s not in selected and needed > 0:
                    selected.append(s)
                    needed -= 1

        if not selected:
            i += rebal_freq
            continue

        selected_set = set(selected)
        turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            rets = [float(returns[s].iloc[j]) for s in selected if not pd.isna(returns[s].iloc[j])]
            port_ret = np.mean(rets) if rets else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    label = f"SectorConfirm_m{mom_lookback}_sma{sector_ma}_t{top_sectors}_s{stocks_per_sector}"
    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'method': 'sector_confirm', 'mom_lookback': mom_lookback,
                       'sector_ma': sector_ma, 'top_sectors': top_sectors,
                       'stocks_per_sector': stocks_per_sector,
                       'fallback_to_qm': fallback_to_qm})


# ============================================================
# Strategy 5: Regime Switch with Vol-based trigger
# ============================================================

def run_vol_regime_switch(price_df, vol_lookback=12, vol_threshold_pct=75,
                           blend_sr_pct=0.3, rebal_freq=4, txn_bps=8):
    """
    Normal: 100% SR
    When recent volatility exceeds its Xth percentile of history: switch to blend
    """
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    sr_sel = get_sr_selections(price_df, 12, 0, 3, 2)
    qm_sel = get_qm_selections(price_df, 20, 24, 0, 8, 0.3, 0.7)

    # Calculate rolling vol of pool index
    pool_idx = build_pool_index(price_df)
    pool_ret = pool_idx.pct_change()

    warmup = max(26, vol_lookback + 2)
    nav = [1.0]
    dates = []
    weekly_rets = []
    prev_sr = set()
    prev_qm = set()
    total_txn = 0.0
    risk_weeks = 0
    total_weeks = 0

    i = warmup
    while i < len(price_df) - 1:
        if i not in sr_sel:
            i += 1
            continue

        # Calculate current vol and its historical percentile
        current_vol = pool_ret.iloc[max(0, i-vol_lookback):i+1].std()
        hist_vol = pool_ret.iloc[:i+1].rolling(vol_lookback).std().dropna()
        if len(hist_vol) < 20:
            i += 1
            continue
        vol_pctile = (hist_vol < current_vol).mean() * 100
        is_high_vol = vol_pctile >= vol_threshold_pct

        sr_stocks = sr_sel[i]
        qm_stocks = qm_sel.get(i, sr_stocks)

        hold_end = min(i + rebal_freq, len(price_df) - 1)

        if is_high_vol:
            sr_w = blend_sr_pct
            qm_w = 1.0 - blend_sr_pct
            risk_weeks += (hold_end - i)
        else:
            sr_w = 1.0
            qm_w = 0.0

        sr_set = set(sr_stocks)
        qm_set = set(qm_stocks)
        turnover_sr = len(sr_set - prev_sr) / max(len(sr_set), 1)
        turnover_qm = len(qm_set - prev_qm) / max(len(qm_set), 1) if qm_w > 0 else 0
        period_txn = (turnover_sr * sr_w + turnover_qm * qm_w) * txn_cost
        total_txn += period_txn

        for j in range(i + 1, hold_end + 1):
            sr_rets = [float(returns[s].iloc[j]) for s in sr_stocks if not pd.isna(returns[s].iloc[j])]
            sr_ret = np.mean(sr_rets) if sr_rets else 0.0
            if qm_w > 0:
                qm_rets = [float(returns[s].iloc[j]) for s in qm_stocks if not pd.isna(returns[s].iloc[j])]
                qm_ret = np.mean(qm_rets) if qm_rets else 0.0
                port_ret = sr_w * sr_ret + qm_w * qm_ret
            else:
                port_ret = sr_ret
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)
            total_weeks += 1

        prev_sr = sr_set
        prev_qm = qm_set
        i = hold_end

    label = f"VolRegime_lb{vol_lookback}_th{vol_threshold_pct}_blend{int(blend_sr_pct*100)}"
    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'method': 'vol_regime_switch', 'vol_lookback': vol_lookback,
                       'vol_threshold_pct': vol_threshold_pct,
                       'blend_sr_pct': blend_sr_pct,
                       'risk_pct': round(risk_weeks / max(total_weeks, 1) * 100, 1)})


# ============================================================
# Strategy 6: Dual Momentum (absolute + relative)
# Only go into SR when pool index itself has positive momentum
# Otherwise hold QM as defensive
# ============================================================

def run_dual_momentum(price_df, abs_mom_weeks=12, rebal_freq=4, txn_bps=8):
    """
    Absolute momentum filter on pool index:
    - Pool index momentum > 0: 100% SR
    - Pool index momentum <= 0: 100% QM (defensive)
    """
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    sr_sel = get_sr_selections(price_df, 12, 0, 3, 2)
    qm_sel = get_qm_selections(price_df, 20, 24, 0, 8, 0.3, 0.7)
    pool_idx = build_pool_index(price_df)

    warmup = max(26, abs_mom_weeks + 2)
    nav = [1.0]
    dates = []
    weekly_rets = []
    prev_holdings = set()
    total_txn = 0.0
    sr_weeks = 0
    total_weeks = 0

    i = warmup
    while i < len(price_df) - 1:
        if i not in sr_sel:
            i += 1
            continue

        # Absolute momentum of pool index
        if i >= abs_mom_weeks:
            pool_mom = pool_idx.iloc[i] / pool_idx.iloc[i - abs_mom_weeks] - 1
        else:
            pool_mom = 0

        use_sr = pool_mom > 0
        if use_sr:
            selected = sr_sel[i]
            sr_weeks += 1
        else:
            selected = qm_sel.get(i, sr_sel[i])

        selected_set = set(selected)
        turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            rets = [float(returns[s].iloc[j]) for s in selected if not pd.isna(returns[s].iloc[j])]
            port_ret = np.mean(rets) if rets else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)
            total_weeks += 1

        prev_holdings = selected_set
        i = hold_end

    label = f"DualMom_abs{abs_mom_weeks}w"
    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'method': 'dual_momentum', 'abs_mom_weeks': abs_mom_weeks,
                       'sr_pct': round(sr_weeks / max(total_weeks / rebal_freq, 1) * 100, 1)})


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("Loading data...")
    price_df = load_all_data()
    print(f"Data loaded: {price_df.shape[0]} weeks x {price_df.shape[1]} stocks")

    results = []

    # --- Strategy 1: Regime Switch ---
    print("\n=== Regime Switch (MA trigger) ===")
    for ma in [10, 12, 15, 20]:
        for blend_sr in [0.2, 0.3, 0.5]:
            r = run_regime_switch(price_df, ma_weeks=ma, blend_sr_pct=blend_sr)
            if r:
                results.append(r)
                print(f"  {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']} risk_time={r.get('risk_pct','?')}%")

    # --- Strategy 2: Adaptive Allocation ---
    print("\n=== Adaptive Allocation ===")
    for ma in [10, 12, 15, 20]:
        for min_sr in [0.2, 0.3]:
            r = run_adaptive_allocation(price_df, ma_weeks=ma, min_sr=min_sr, max_sr=1.0)
            if r:
                results.append(r)
                print(f"  {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']}")

    # --- Strategy 3: Portfolio Trailing Stop ---
    print("\n=== Portfolio Trailing Stop ===")
    for stop in [-0.08, -0.10, -0.12, -0.15]:
        for exp in [0.3, 0.5]:
            for rec in [-0.03, 0.0]:
                r = run_portfolio_trailing_stop(price_df, stop_pct=stop,
                                                 reduced_exposure=exp, recovery_pct=rec)
                if r:
                    results.append(r)
                    print(f"  {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']}")

    # --- Strategy 4: Sector Trend Confirmation ---
    print("\n=== Sector Trend Confirmation ===")
    for sector_ma in [8, 10, 12, 15]:
        for fb in [True, False]:
            r = run_sector_trend_confirm(price_df, mom_lookback=12, sector_ma=sector_ma,
                                          top_sectors=3, stocks_per_sector=2, fallback_to_qm=fb)
            if r:
                results.append(r)
                print(f"  {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']}")

    # --- Strategy 5: Vol Regime Switch ---
    print("\n=== Vol Regime Switch ===")
    for vlb in [8, 12, 20]:
        for vth in [70, 80]:
            for bsr in [0.2, 0.3]:
                r = run_vol_regime_switch(price_df, vol_lookback=vlb,
                                           vol_threshold_pct=vth, blend_sr_pct=bsr)
                if r:
                    results.append(r)
                    print(f"  {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']} risk={r.get('risk_pct','?')}%")

    # --- Strategy 6: Dual Momentum ---
    print("\n=== Dual Momentum ===")
    for am in [8, 10, 12, 15, 20]:
        r = run_dual_momentum(price_df, abs_mom_weeks=am)
        if r:
            results.append(r)
            print(f"  {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']} SR%={r.get('sr_pct','?')}")

    # Sort by calmar
    results.sort(key=lambda x: -x['calmar'])

    # Summary
    print(f"\n{'='*70}")
    print(f"Total variants: {len(results)}")
    print(f"\nTop 15 by Calmar:")
    for r in results[:15]:
        print(f"  {r['label']:50s} CAGR={r['cagr_pct']:6.1f}% MDD={r['mdd_pct']:6.1f}% Calmar={r['calmar']:.3f} Sharpe={r['sharpe']:.3f}")

    # Save results
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_variants': len(results),
        'ranked_by_calmar': results,
        'best_per_method': {}
    }

    methods = set(r['method'] for r in results)
    for m in methods:
        m_results = [r for r in results if r['method'] == m]
        if m_results:
            output['best_per_method'][m] = max(m_results, key=lambda x: x['calmar'])

    out_path = os.path.join(DATA_DIR, 'csi500_dynamic_allocation.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
