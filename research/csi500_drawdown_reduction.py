#!/usr/bin/env python3
"""
CSI500 Drawdown Reduction Research
Tests 5 methods to reduce drawdown on the best sector rotation strategy:
  1. Trend Filter (MA on pool index -> cash when below)
  2. Blend SR + Quality-Momentum
  3. Volatility Timing (reduce position when vol spikes)
  4. Inverse-Vol Position Sizing
  5. Stop Loss per stock
  + combinations of the best methods
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
from itertools import product

# Reuse everything from the original research
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi500_quality_momentum_research import (
    CSI500_POOL, load_all_data, get_momentum, get_volatility,
    rank_percentile, inverse_rank, DATA_DIR
)

# ============================================================
# Shared stats
# ============================================================

def calc_stats(nav, dates, weekly_rets, total_txn, label, params):
    if not dates or len(dates) < 20:
        return None
    nav_s = pd.Series(nav[1:], index=dates)
    years = (dates[-1] - dates[0]).days / 365.25
    if years < 1:
        return None

    cagr = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / years) - 1
    dd = nav_s / nav_s.cummax() - 1
    mdd = dd.min()
    wr = pd.Series(weekly_rets)
    sharpe = wr.mean() / wr.std() * np.sqrt(52) if wr.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    win_rate = (wr > 0).sum() / len(wr) * 100

    annual = nav_s.resample('YE').last().pct_change().dropna()
    annual_returns = {str(d.year): round(v * 100, 1) for d, v in annual.items()}

    result = {
        'label': label,
        'cagr_pct': round(cagr * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round(win_rate, 1),
        'total_return_pct': round((nav_s.iloc[-1] - 1) * 100, 1),
        'years': round(years, 1),
        'total_txn_pct': round(total_txn * 100, 2),
        'annual_returns': annual_returns,
        'period': f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}",
    }
    result.update(params)
    return result


def build_pool_index(price_df):
    """Build equal-weight index of the pool as trend proxy"""
    returns = price_df.pct_change()
    idx_ret = returns.mean(axis=1)
    idx_nav = (1 + idx_ret).cumprod()
    return idx_nav


# ============================================================
# Baseline: Original Sector Rotation (SR_m12_sk0_top3_sps2)
# ============================================================

def get_sr_selections(price_df, mom_lookback=12, skip=0, top_sectors=3,
                       stocks_per_sector=2):
    """Return dict: {week_idx: [selected_stocks]} for all rebalance points"""
    sector_stocks = defaultdict(list)
    for s in price_df.columns:
        sec = CSI500_POOL.get(s, {}).get('sector', '?')
        sector_stocks[sec].append(s)

    warmup = mom_lookback + skip + 2
    selections = {}

    for i in range(warmup, len(price_df)):
        mom_idx = i - skip
        mom = get_momentum(price_df, mom_idx, mom_lookback)

        sector_mom = {}
        for sec, stocks in sector_stocks.items():
            sec_moms = [mom[s] for s in stocks if s in mom]
            if sec_moms:
                sector_mom[sec] = np.mean(sec_moms)

        if len(sector_mom) < 3:
            continue

        ranked_sectors = sorted(sector_mom.items(), key=lambda x: -x[1])[:top_sectors]
        selected = []
        for sec, _ in ranked_sectors:
            stocks = sector_stocks[sec]
            stock_moms = [(s, mom[s]) for s in stocks if s in mom]
            stock_moms.sort(key=lambda x: -x[1])
            for s, _ in stock_moms[:stocks_per_sector]:
                selected.append(s)

        if selected:
            selections[i] = selected

    return selections


def run_baseline(price_df, rebal_freq=4, txn_bps=8):
    """Run the original SR_m12_sk0_top3_sps2 strategy"""
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    selections = get_sr_selections(price_df, 12, 0, 3, 2)

    warmup = 14  # 12 + 0 + 2
    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        if i not in selections:
            i += 1
            continue

        selected = selections[i]
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

    return calc_stats(nav, dates, weekly_rets, total_txn, "Baseline_SR_m12_top3_sps2",
                      {'method': 'baseline'})


# ============================================================
# Method 1: Trend Filter (MA on pool index)
# ============================================================

def run_trend_filter(price_df, ma_weeks=20, rebal_freq=4, txn_bps=8):
    """Go to cash when pool index is below its MA"""
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    selections = get_sr_selections(price_df, 12, 0, 3, 2)
    pool_idx = build_pool_index(price_df)

    warmup = max(14, ma_weeks + 2)
    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []
    cash_weeks = 0
    total_weeks = 0

    i = warmup
    while i < len(price_df) - 1:
        if i not in selections:
            i += 1
            continue

        # Check trend: is pool index above its MA?
        idx_slice = pool_idx.iloc[max(0, i-ma_weeks):i+1]
        if len(idx_slice) < ma_weeks:
            i += 1
            continue

        ma_val = idx_slice.rolling(ma_weeks).mean().iloc[-1]
        current_val = idx_slice.iloc[-1]
        in_uptrend = current_val > ma_val

        hold_end = min(i + rebal_freq, len(price_df) - 1)

        if in_uptrend:
            selected = selections[i]
            selected_set = set(selected)
            turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(len(selected_set), 1)
            period_txn = turnover * txn_cost
            total_txn += period_txn

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
        else:
            # Cash: 0% return
            if prev_holdings:
                total_txn += txn_cost  # sell cost
            for j in range(i + 1, hold_end + 1):
                nav.append(nav[-1])
                dates.append(price_df.index[j])
                weekly_rets.append(0.0)
                cash_weeks += 1
                total_weeks += 1
            prev_holdings = set()

        i = hold_end

    label = f"TrendFilter_MA{ma_weeks}"
    result = calc_stats(nav, dates, weekly_rets, total_txn, label,
                        {'method': 'trend_filter', 'ma_weeks': ma_weeks,
                         'cash_pct': round(cash_weeks / max(total_weeks, 1) * 100, 1)})
    return result


# ============================================================
# Method 2: Blend SR + Quality-Momentum
# ============================================================

def get_qm_selections(price_df, mom_lookback=20, vol_lookback=24, skip=0,
                       top_n=8, mom_weight=0.3, vol_weight=0.7):
    """Get QM strategy selections for each week"""
    warmup = max(mom_lookback, vol_lookback) + skip + 2
    selections = {}

    for i in range(warmup, len(price_df)):
        mom_idx = i - skip
        mom = get_momentum(price_df, mom_idx, mom_lookback)
        vol = get_volatility(price_df, mom_idx, vol_lookback)

        common = set(mom.keys()) & set(vol.keys())
        if len(common) < 10:
            continue

        mom_rank = rank_percentile({k: mom[k] for k in common})
        vol_rank = inverse_rank({k: vol[k] for k in common})

        composite = {s: mom_weight * mom_rank[s] + vol_weight * vol_rank[s] for s in common}
        ranked = sorted(composite.items(), key=lambda x: -x[1])
        selections[i] = [s for s, _ in ranked[:top_n]]

    return selections


def run_blend(price_df, sr_weight=0.5, rebal_freq=4, txn_bps=8):
    """Blend SR and QM strategies by running both and averaging returns"""
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    sr_sel = get_sr_selections(price_df, 12, 0, 3, 2)
    qm_sel = get_qm_selections(price_df, 20, 24, 0, 8, 0.3, 0.7)

    warmup = 26  # max warmup of both
    qm_weight = 1.0 - sr_weight
    nav = [1.0]
    dates = []
    prev_sr = set()
    prev_qm = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        sr_stocks = sr_sel.get(i, None)
        qm_stocks = qm_sel.get(i, None)
        if sr_stocks is None and qm_stocks is None:
            i += 1
            continue

        hold_end = min(i + rebal_freq, len(price_df) - 1)

        # Compute turnover for both
        sr_set = set(sr_stocks) if sr_stocks else prev_sr
        qm_set = set(qm_stocks) if qm_stocks else prev_qm

        sr_to = (len(sr_set - prev_sr) + len(prev_sr - sr_set)) / max(len(sr_set), 1) if sr_stocks else 0
        qm_to = (len(qm_set - prev_qm) + len(prev_qm - qm_set)) / max(len(qm_set), 1) if qm_stocks else 0
        period_txn = (sr_weight * sr_to + qm_weight * qm_to) * txn_cost
        total_txn += period_txn

        for j in range(i + 1, hold_end + 1):
            sr_rets = [float(returns[s].iloc[j]) for s in sr_set if not pd.isna(returns[s].iloc[j])]
            qm_rets = [float(returns[s].iloc[j]) for s in qm_set if not pd.isna(returns[s].iloc[j])]
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

    label = f"Blend_SR{int(sr_weight*100)}_QM{int(qm_weight*100)}"
    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'method': 'blend', 'sr_weight': sr_weight})


# ============================================================
# Method 3: Volatility Timing
# ============================================================

def run_vol_timing(price_df, vol_lookback=12, vol_threshold_pct=75,
                    min_exposure=0.3, rebal_freq=4, txn_bps=8):
    """Reduce exposure when recent vol is in top quartile historically"""
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    selections = get_sr_selections(price_df, 12, 0, 3, 2)

    # Compute rolling pool vol
    pool_ret = returns.mean(axis=1)
    rolling_vol = pool_ret.rolling(vol_lookback).std()
    # Historical percentile of vol
    expanding_vol_rank = rolling_vol.expanding().rank(pct=True)

    warmup = max(14, vol_lookback + 2)
    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        if i not in selections:
            i += 1
            continue

        selected = selections[i]
        selected_set = set(selected)

        # Determine exposure
        vol_pct = expanding_vol_rank.iloc[i]
        if pd.isna(vol_pct):
            exposure = 1.0
        elif vol_pct > (vol_threshold_pct / 100):
            # High vol: reduce exposure linearly
            excess = (vol_pct - vol_threshold_pct / 100) / (1 - vol_threshold_pct / 100)
            exposure = max(min_exposure, 1.0 - excess * (1.0 - min_exposure))
        else:
            exposure = 1.0

        turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            rets = [float(returns[s].iloc[j]) for s in selected if not pd.isna(returns[s].iloc[j])]
            port_ret = np.mean(rets) if rets else 0.0
            port_ret = port_ret * exposure  # scale by exposure
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    label = f"VolTiming_lb{vol_lookback}_th{vol_threshold_pct}_min{int(min_exposure*100)}"
    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'method': 'vol_timing', 'vol_lookback': vol_lookback,
                       'vol_threshold_pct': vol_threshold_pct, 'min_exposure': min_exposure})


# ============================================================
# Method 4: Inverse-Vol Position Sizing
# ============================================================

def run_invvol_sizing(price_df, vol_lookback=12, rebal_freq=4, txn_bps=8):
    """Weight each stock inversely proportional to its volatility"""
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    selections = get_sr_selections(price_df, 12, 0, 3, 2)

    warmup = max(14, vol_lookback + 2)
    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        if i not in selections:
            i += 1
            continue

        selected = selections[i]
        selected_set = set(selected)

        # Compute inverse-vol weights
        vol = get_volatility(price_df, i, vol_lookback)
        weights = {}
        for s in selected:
            v = vol.get(s, None)
            if v and v > 0:
                weights[s] = 1.0 / v
            else:
                weights[s] = 1.0  # fallback equal

        total_w = sum(weights.values())
        weights = {s: w / total_w for s, w in weights.items()}

        turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            port_ret = 0.0
            for s in selected:
                r = returns[s].iloc[j]
                if not pd.isna(r):
                    port_ret += weights.get(s, 0) * float(r)
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    label = f"InvVol_lb{vol_lookback}"
    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'method': 'invvol_sizing', 'vol_lookback': vol_lookback})


# ============================================================
# Method 5: Stop Loss
# ============================================================

def run_stop_loss(price_df, stop_pct=-0.10, rebal_freq=4, txn_bps=8):
    """Drop individual stocks that fall >stop_pct from entry during hold period"""
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    selections = get_sr_selections(price_df, 12, 0, 3, 2)

    warmup = 14
    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        if i not in selections:
            i += 1
            continue

        selected = list(selections[i])
        selected_set = set(selected)
        turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        # Track entry prices
        entry_prices = {}
        for s in selected:
            p = price_df[s].iloc[i]
            if pd.notna(p):
                entry_prices[s] = float(p)

        active = list(selected)
        hold_end = min(i + rebal_freq, len(price_df) - 1)

        for j in range(i + 1, hold_end + 1):
            # Check stop loss
            stopped = []
            for s in active:
                p = price_df[s].iloc[j]
                if pd.notna(p) and s in entry_prices and entry_prices[s] > 0:
                    ret_from_entry = float(p) / entry_prices[s] - 1
                    if ret_from_entry < stop_pct:
                        stopped.append(s)
                        total_txn += txn_cost / len(selected)

            for s in stopped:
                active.remove(s)

            if active:
                rets = [float(returns[s].iloc[j]) for s in active if not pd.isna(returns[s].iloc[j])]
                port_ret = np.mean(rets) if rets else 0.0
            else:
                port_ret = 0.0  # all stopped, hold cash

            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    label = f"StopLoss_{int(abs(stop_pct)*100)}pct"
    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'method': 'stop_loss', 'stop_pct': stop_pct})


# ============================================================
# Combo: Trend Filter + InvVol + Blend
# ============================================================

def run_trend_invvol(price_df, ma_weeks=20, vol_lookback=12,
                      rebal_freq=4, txn_bps=8):
    """Trend filter + inverse-vol sizing combined"""
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    selections = get_sr_selections(price_df, 12, 0, 3, 2)
    pool_idx = build_pool_index(price_df)

    warmup = max(14, ma_weeks + 2, vol_lookback + 2)
    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []
    cash_weeks = 0
    total_weeks = 0

    i = warmup
    while i < len(price_df) - 1:
        if i not in selections:
            i += 1
            continue

        idx_slice = pool_idx.iloc[max(0, i-ma_weeks):i+1]
        if len(idx_slice) < ma_weeks:
            i += 1
            continue

        ma_val = idx_slice.rolling(ma_weeks).mean().iloc[-1]
        current_val = idx_slice.iloc[-1]
        in_uptrend = current_val > ma_val

        hold_end = min(i + rebal_freq, len(price_df) - 1)

        if in_uptrend:
            selected = selections[i]
            selected_set = set(selected)

            # Inverse-vol weights
            vol = get_volatility(price_df, i, vol_lookback)
            weights = {}
            for s in selected:
                v = vol.get(s, None)
                if v and v > 0:
                    weights[s] = 1.0 / v
                else:
                    weights[s] = 1.0
            total_w = sum(weights.values())
            weights = {s: w / total_w for s, w in weights.items()}

            turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(len(selected_set), 1)
            period_txn = turnover * txn_cost
            total_txn += period_txn

            for j in range(i + 1, hold_end + 1):
                port_ret = 0.0
                for s in selected:
                    r = returns[s].iloc[j]
                    if not pd.isna(r):
                        port_ret += weights.get(s, 0) * float(r)
                if j == i + 1:
                    port_ret -= period_txn
                nav.append(nav[-1] * (1 + port_ret))
                dates.append(price_df.index[j])
                weekly_rets.append(port_ret)
                total_weeks += 1

            prev_holdings = selected_set
        else:
            if prev_holdings:
                total_txn += txn_cost
            for j in range(i + 1, hold_end + 1):
                nav.append(nav[-1])
                dates.append(price_df.index[j])
                weekly_rets.append(0.0)
                cash_weeks += 1
                total_weeks += 1
            prev_holdings = set()

        i = hold_end

    label = f"Trend{ma_weeks}_InvVol{vol_lookback}"
    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'method': 'trend_invvol', 'ma_weeks': ma_weeks,
                       'vol_lookback': vol_lookback,
                       'cash_pct': round(cash_weeks / max(total_weeks, 1) * 100, 1)})


def run_trend_blend(price_df, ma_weeks=20, sr_weight=0.5,
                     rebal_freq=4, txn_bps=8):
    """Trend filter on blended SR+QM"""
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    sr_sel = get_sr_selections(price_df, 12, 0, 3, 2)
    qm_sel = get_qm_selections(price_df, 20, 24, 0, 8, 0.3, 0.7)
    pool_idx = build_pool_index(price_df)

    qm_weight = 1.0 - sr_weight
    warmup = max(26, ma_weeks + 2)
    nav = [1.0]
    dates = []
    prev_sr = set()
    prev_qm = set()
    total_txn = 0.0
    weekly_rets = []
    cash_weeks = 0
    total_weeks = 0

    i = warmup
    while i < len(price_df) - 1:
        sr_stocks = sr_sel.get(i, None)
        qm_stocks = qm_sel.get(i, None)
        if sr_stocks is None and qm_stocks is None:
            i += 1
            continue

        idx_slice = pool_idx.iloc[max(0, i-ma_weeks):i+1]
        if len(idx_slice) < ma_weeks:
            i += 1
            continue

        ma_val = idx_slice.rolling(ma_weeks).mean().iloc[-1]
        current_val = idx_slice.iloc[-1]
        in_uptrend = current_val > ma_val

        hold_end = min(i + rebal_freq, len(price_df) - 1)

        if in_uptrend:
            sr_set = set(sr_stocks) if sr_stocks else prev_sr
            qm_set = set(qm_stocks) if qm_stocks else prev_qm

            sr_to = (len(sr_set - prev_sr) + len(prev_sr - sr_set)) / max(len(sr_set), 1) if sr_stocks else 0
            qm_to = (len(qm_set - prev_qm) + len(prev_qm - qm_set)) / max(len(qm_set), 1) if qm_stocks else 0
            period_txn = (sr_weight * sr_to + qm_weight * qm_to) * txn_cost
            total_txn += period_txn

            for j in range(i + 1, hold_end + 1):
                sr_rets = [float(returns[s].iloc[j]) for s in sr_set if not pd.isna(returns[s].iloc[j])]
                qm_rets = [float(returns[s].iloc[j]) for s in qm_set if not pd.isna(returns[s].iloc[j])]
                sr_ret = np.mean(sr_rets) if sr_rets else 0.0
                qm_ret = np.mean(qm_rets) if qm_rets else 0.0
                port_ret = sr_weight * sr_ret + qm_weight * qm_ret
                if j == i + 1:
                    port_ret -= period_txn
                nav.append(nav[-1] * (1 + port_ret))
                dates.append(price_df.index[j])
                weekly_rets.append(port_ret)
                total_weeks += 1

            prev_sr = sr_set
            prev_qm = qm_set
        else:
            if prev_sr or prev_qm:
                total_txn += txn_cost
            for j in range(i + 1, hold_end + 1):
                nav.append(nav[-1])
                dates.append(price_df.index[j])
                weekly_rets.append(0.0)
                cash_weeks += 1
                total_weeks += 1
            prev_sr = set()
            prev_qm = set()

        i = hold_end

    label = f"TrendBlend_MA{ma_weeks}_SR{int(sr_weight*100)}"
    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'method': 'trend_blend', 'ma_weeks': ma_weeks,
                       'sr_weight': sr_weight,
                       'cash_pct': round(cash_weeks / max(total_weeks, 1) * 100, 1)})


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("CSI500 Drawdown Reduction Research")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    price_df = load_all_data()
    if price_df is None:
        sys.exit(1)

    all_results = []

    # Baseline
    print("\n[BASELINE] SR_m12_sk0_top3_sps2")
    r = run_baseline(price_df)
    if r:
        all_results.append(r)
        print(f"  CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Sharpe={r['sharpe']} Calmar={r['calmar']}")
        print(f"  Annual: {r['annual_returns']}")

    # Method 1: Trend Filter
    print("\n[METHOD 1] Trend Filter (various MA periods)")
    for ma in [10, 15, 20, 26, 30]:
        r = run_trend_filter(price_df, ma_weeks=ma)
        if r:
            all_results.append(r)
            print(f"  MA{ma}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Cal={r['calmar']} Cash={r.get('cash_pct', '?')}%")
            print(f"    Annual: {r['annual_returns']}")

    # Method 2: Blend SR + QM
    print("\n[METHOD 2] Blend SR + Quality-Momentum")
    for sr_w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        r = run_blend(price_df, sr_weight=sr_w)
        if r:
            all_results.append(r)
            print(f"  SR{int(sr_w*100)}/QM{int((1-sr_w)*100)}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Cal={r['calmar']}")
            print(f"    Annual: {r['annual_returns']}")

    # Method 3: Volatility Timing
    print("\n[METHOD 3] Volatility Timing")
    for vlb, vth, me in [(8, 70, 0.3), (8, 75, 0.3), (12, 70, 0.3),
                          (12, 75, 0.3), (12, 75, 0.5), (12, 80, 0.3),
                          (20, 75, 0.3)]:
        r = run_vol_timing(price_df, vol_lookback=vlb, vol_threshold_pct=vth, min_exposure=me)
        if r:
            all_results.append(r)
            print(f"  lb{vlb}_th{vth}_min{int(me*100)}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Cal={r['calmar']}")

    # Method 4: Inverse-Vol Sizing
    print("\n[METHOD 4] Inverse-Volatility Position Sizing")
    for vlb in [8, 12, 20, 24]:
        r = run_invvol_sizing(price_df, vol_lookback=vlb)
        if r:
            all_results.append(r)
            print(f"  lb{vlb}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Cal={r['calmar']}")
            print(f"    Annual: {r['annual_returns']}")

    # Method 5: Stop Loss
    print("\n[METHOD 5] Stop Loss")
    for sp in [-0.05, -0.08, -0.10, -0.15, -0.20]:
        r = run_stop_loss(price_df, stop_pct=sp)
        if r:
            all_results.append(r)
            print(f"  {int(abs(sp)*100)}%: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Cal={r['calmar']}")
            print(f"    Annual: {r['annual_returns']}")

    # Combos
    print("\n[COMBOS] Best combinations")
    for ma in [15, 20, 26]:
        for vlb in [12, 20]:
            r = run_trend_invvol(price_df, ma_weeks=ma, vol_lookback=vlb)
            if r:
                all_results.append(r)
                print(f"  Trend{ma}+InvVol{vlb}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Cal={r['calmar']} Cash={r.get('cash_pct', '?')}%")

    for ma in [15, 20, 26]:
        for sr_w in [0.5, 0.6]:
            r = run_trend_blend(price_df, ma_weeks=ma, sr_weight=sr_w)
            if r:
                all_results.append(r)
                print(f"  TrendBlend_MA{ma}_SR{int(sr_w*100)}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Cal={r['calmar']} Cash={r.get('cash_pct', '?')}%")

    # ──────────── Summary ────────────
    print("\n" + "=" * 70)
    print("SUMMARY: All methods ranked by Calmar")
    print("=" * 70)
    ranked = sorted(all_results, key=lambda x: -x['calmar'])
    for i, r in enumerate(ranked):
        marker = " ⭐" if r['method'] == 'baseline' else ""
        print(f"  #{i+1} {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% "
              f"Sh={r['sharpe']} Cal={r['calmar']}{marker}")
        print(f"       Annual: {r['annual_returns']}")

    # Group by method
    print("\n" + "=" * 70)
    print("BEST per Method:")
    print("=" * 70)
    methods = {}
    for r in all_results:
        m = r.get('method', 'unknown')
        if m not in methods or r['calmar'] > methods[m]['calmar']:
            methods[m] = r
    for m, r in sorted(methods.items(), key=lambda x: -x[1]['calmar']):
        print(f"  {m}: {r['label']}")
        print(f"    CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Sharpe={r['sharpe']} Calmar={r['calmar']}")
        print(f"    Annual: {r['annual_returns']}")

    # Save
    out_path = os.path.join(DATA_DIR, 'csi500_drawdown_reduction.json')
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_variants': len(all_results),
        'ranked_by_calmar': ranked,
        'best_per_method': methods,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
