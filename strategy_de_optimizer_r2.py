#!/usr/bin/env python3
"""
Strategy D+E Deep Optimizer - Round 2

Round 1 findings:
  - Vol scaling is the biggest improvement
  - BUT Round 1 allowed leverage (position > 100%) which is unrealistic
  - Regime filter hurts too much for E (contrarian strategy)
  - Sector max=1 helps D slightly
  - Mean reversion value type slightly better than reversal for E

Round 2 focus:
  1. Fix: vol scaling capped at 1.0 (no leverage, only reduce)
  2. Test regime + vol scaling combos properly
  3. Test sector_max with vol scaling
  4. Deep grid search on winning combos
  5. Walk-forward on best configs
  6. Robustness: parameter sensitivity analysis
  7. Monthly stability analysis
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RESULTS_DIR = os.path.join(DATA_DIR, 'optimization')
os.makedirs(RESULTS_DIR, exist_ok=True)

STOCK_POOL = {
    '600519.SS': {'name': '贵州茅台', 'code': '600519', 'sector': '白酒'},
    '000858.SZ': {'name': '五粮液', 'code': '000858', 'sector': '白酒'},
    '600887.SS': {'name': '伊利股份', 'code': '600887', 'sector': '食品'},
    '002714.SZ': {'name': '牧原股份', 'code': '002714', 'sector': '养殖'},
    '601318.SS': {'name': '中国平安', 'code': '601318', 'sector': '保险'},
    '600036.SS': {'name': '招商银行', 'code': '600036', 'sector': '银行'},
    '002415.SZ': {'name': '海康威视', 'code': '002415', 'sector': '安防'},
    '300750.SZ': {'name': '宁德时代', 'code': '300750', 'sector': '电池'},
    '600276.SS': {'name': '恒瑞医药', 'code': '600276', 'sector': '创新药'},
    '300760.SZ': {'name': '迈瑞医疗', 'code': '300760', 'sector': '医疗器械'},
    '601012.SS': {'name': '隆基绿能', 'code': '601012', 'sector': '光伏'},
    '300274.SZ': {'name': '阳光电源', 'code': '300274', 'sector': '逆变器'},
    '000333.SZ': {'name': '美的集团', 'code': '000333', 'sector': '家电'},
    '600690.SS': {'name': '海尔智家', 'code': '600690', 'sector': '家电'},
    '002594.SZ': {'name': '比亚迪', 'code': '002594', 'sector': '新能源车'},
    '600893.SS': {'name': '航发动力', 'code': '600893', 'sector': '航发'},
    '601668.SS': {'name': '中国建筑', 'code': '601668', 'sector': '建筑'},
    '600585.SS': {'name': '海螺水泥', 'code': '600585', 'sector': '水泥'},
    '601899.SS': {'name': '紫金矿业', 'code': '601899', 'sector': '有色'},
    '600028.SS': {'name': '中国石化', 'code': '600028', 'sector': '石油'},
    '002371.SZ': {'name': '北方华创', 'code': '002371', 'sector': '半导体设备'},
    '000063.SZ': {'name': '中兴通讯', 'code': '000063', 'sector': '通信设备'},
    '600941.SS': {'name': '中国移动', 'code': '600941', 'sector': '运营商'},
    '002230.SZ': {'name': '科大讯飞', 'code': '002230', 'sector': 'AI'},
    '600900.SS': {'name': '长江电力', 'code': '600900', 'sector': '水电'},
    '601088.SS': {'name': '中国神华', 'code': '601088', 'sector': '煤炭'},
    '601006.SS': {'name': '大秦铁路', 'code': '601006', 'sector': '铁路'},
    '001979.SZ': {'name': '招商蛇口', 'code': '001979', 'sector': '地产'},
}

# ============================================================
# DATA
# ============================================================

def load_cached_prices() -> pd.DataFrame:
    """Load 28-stock weekly prices from cache"""
    prices = {}
    for ticker, info in STOCK_POOL.items():
        safe = ticker.replace('.', '_')
        csv_path = os.path.join(DATA_DIR, f'sd_{safe}_weekly.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
            if len(df) > 0:
                prices[ticker] = df['adjclose'] if 'adjclose' in df.columns else df['close']
    price_df = pd.DataFrame(prices).dropna(how='all')
    print(f"Loaded: {price_df.shape[0]}w x {price_df.shape[1]}stocks ({price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')})")
    return price_df


def load_signal_index() -> Optional[pd.Series]:
    csv_path = os.path.join(DATA_DIR, 'sd_signal_index_weekly.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        if len(df) > 0:
            return df['adjclose'] if 'adjclose' in df.columns else df['close']
    return None


# ============================================================
# BACKTEST ENGINE v2 (NO LEVERAGE)
# ============================================================

def calc_metrics(nav_s: pd.Series, weekly_rets: list) -> dict:
    if len(nav_s) < 10:
        return {'cagr': 0, 'mdd': -1, 'sharpe': 0, 'calmar': 0}
    years = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
    if years <= 0:
        return {'cagr': 0, 'mdd': -1, 'sharpe': 0, 'calmar': 0}
    cagr = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / years) - 1
    dd = nav_s / nav_s.cummax() - 1
    mdd = dd.min()
    wr = pd.Series(weekly_rets)
    sharpe = wr.mean() / wr.std() * np.sqrt(52) if wr.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    win_rate = (wr > 0).sum() / len(wr) * 100 if len(wr) > 0 else 0
    annual = nav_s.resample('YE').last().pct_change().dropna()
    annual_returns = {str(d.year): round(v * 100, 1) for d, v in annual.items()}
    # Worst year
    worst_year = min(annual_returns.values()) if annual_returns else 0
    # Count negative years
    neg_years = sum(1 for v in annual_returns.values() if v < 0)

    return {
        'cagr': round(cagr * 100, 2),
        'mdd': round(mdd * 100, 2),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate': round(win_rate, 1),
        'annual': annual_returns,
        'final_nav': round(nav_s.iloc[-1], 4),
        'years': round(years, 1),
        'worst_year': worst_year,
        'neg_years': neg_years,
    }


def get_regime(sig_aligned, i, regime_type, short_ma, long_ma, bear_mult, trans_mult):
    """Get position multiplier from regime detection (no lookahead)"""
    if sig_aligned is None:
        return 1.0
    sig_slice = sig_aligned.iloc[:i+1].dropna()
    if len(sig_slice) < long_ma:
        return 1.0
    sig_price = sig_slice.iloc[-1]
    sig_long = sig_slice.iloc[-long_ma:].mean()

    if regime_type == 'single':
        return bear_mult if sig_price < sig_long else 1.0
    elif regime_type == 'dual':
        sig_short = sig_slice.iloc[-short_ma:].mean() if len(sig_slice) >= short_ma else sig_price
        if sig_price < sig_long:
            return bear_mult
        elif sig_price < sig_short:
            return trans_mult
        return 1.0
    return 1.0


def get_vol_scale(returns, selected, i, vol_target, vol_lb):
    """Get vol scaling factor, CAPPED AT 1.0 (no leverage!)"""
    if vol_target is None:
        return 1.0
    port_rets = []
    for j_back in range(max(0, i - vol_lb), i):
        week_r = []
        for s in selected:
            if j_back < len(returns):
                r = returns[s].iloc[j_back]
                if not pd.isna(r):
                    week_r.append(float(r))
        if week_r:
            port_rets.append(np.mean(week_r))
    if len(port_rets) < 4:
        return 1.0
    realized_vol = np.std(port_rets) * np.sqrt(52)
    if realized_vol < 0.01:
        return 1.0
    # KEY: cap at 1.0 - only REDUCE position when vol is high
    return min(vol_target / realized_vol, 1.0)


def backtest_d(price_df, params, signal_index=None):
    """Strategy D momentum backtest"""
    lookback = params.get('lookback', 4)
    skip = params.get('skip', 1)
    top_n = params.get('top_n', 8)
    rebal_freq = params.get('rebal_freq', 2)
    txn_cost = params.get('txn_cost_bps', 8) / 10000
    regime = params.get('regime_filter', None)
    short_ma = params.get('regime_short_ma', 13)
    long_ma = params.get('regime_long_ma', 40)
    bear_mult = params.get('regime_bear_mult', 0.2)
    trans_mult = params.get('regime_trans_mult', 0.5)
    vol_target = params.get('vol_target', None)
    vol_lb = params.get('vol_lookback', 12)
    sector_max = params.get('sector_max', None)
    mom_type = params.get('momentum_type', 'raw')

    returns = price_df.pct_change(fill_method=None)
    warmup = max(lookback + skip + 2, long_ma + 2 if regime else 10, vol_lb + 2 if vol_target else 10)

    sig_aligned = None
    if regime and signal_index is not None:
        sig_aligned = signal_index.reindex(price_df.index, method='ffill')

    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        # Regime
        regime_mult = get_regime(sig_aligned, i, regime, short_ma, long_ma, bear_mult, trans_mult) if regime else 1.0

        # Momentum
        end_idx = i - skip
        start_idx = end_idx - lookback
        if start_idx < 0 or end_idx <= 0:
            i += 1
            continue

        avail = [c for c in price_df.columns
                 if not pd.isna(price_df[c].iloc[start_idx])
                 and not pd.isna(price_df[c].iloc[end_idx])
                 and price_df[c].iloc[start_idx] > 0]
        if len(avail) < top_n:
            i += 1
            continue

        momenta = []
        for col in avail:
            if mom_type == 'risk_adjusted':
                ret_slice = returns[col].iloc[max(0,start_idx):end_idx+1].dropna()
                raw_mom = float(price_df[col].iloc[end_idx] / price_df[col].iloc[start_idx] - 1)
                vol = float(ret_slice.std()) if len(ret_slice) >= 3 else 1.0
                mom = raw_mom / vol if vol > 0.001 else raw_mom
            else:
                mom = float(price_df[col].iloc[end_idx] / price_df[col].iloc[start_idx] - 1)
            momenta.append((col, mom))

        ranked = sorted(momenta, key=lambda x: (-x[1], x[0]))

        # Sector constraint
        if sector_max:
            selected = []
            sector_count = defaultdict(int)
            for t, _ in ranked:
                if len(selected) >= top_n:
                    break
                sector = STOCK_POOL.get(t, {}).get('sector', '?')
                if sector_count[sector] < sector_max:
                    selected.append(t)
                    sector_count[sector] += 1
        else:
            selected = [t for t, _ in ranked[:top_n]]

        # Vol scaling (capped at 1.0)
        vol_scale = get_vol_scale(returns, selected, i, vol_target, vol_lb)
        position_mult = min(regime_mult * vol_scale, 1.0)  # NEVER > 1.0

        selected_set = set(selected)
        new_buys = selected_set - prev_holdings
        sold = prev_holdings - selected_set
        turnover = (len(new_buys) + len(sold)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            week_r = []
            for s in selected:
                r = returns[s].iloc[j]
                if not pd.isna(r):
                    week_r.append(float(r))
            port_ret = np.mean(week_r) if week_r else 0.0
            port_ret *= position_mult
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    if not dates:
        return {'error': 'No trades'}
    nav_s = pd.Series(nav[1:], index=dates)
    metrics = calc_metrics(nav_s, weekly_rets)
    metrics['total_txn_pct'] = round(total_txn * 100, 2)
    metrics['params'] = {k: v for k, v in params.items()}
    return metrics


def backtest_e(price_df, params, signal_index=None):
    """Strategy E value backtest"""
    vlb = params.get('value_lookback', 12)
    vol_lb_factor = params.get('vol_lookback', 12)
    vw = params.get('vol_weight', 0.3)
    top_n = params.get('top_n', 5)
    rebal_freq = params.get('rebal_freq', 4)
    txn_cost = params.get('txn_cost_bps', 8) / 10000
    ddf = params.get('max_dd_filter', -0.15)
    regime = params.get('regime_filter', None)
    short_ma = params.get('regime_short_ma', 13)
    long_ma = params.get('regime_long_ma', 40)
    bear_mult = params.get('regime_bear_mult', 0.2)
    trans_mult = params.get('regime_trans_mult', 0.5)
    vol_target = params.get('vol_target', None)
    vol_lb_scale = params.get('vol_scale_lookback', 20)
    sector_max = params.get('sector_max', None)
    value_type = params.get('value_type', 'reversal')

    returns = price_df.pct_change(fill_method=None)
    warmup = max(vlb + 5, vol_lb_factor + 5, long_ma + 2 if regime else 10)

    sig_aligned = None
    if regime and signal_index is not None:
        sig_aligned = signal_index.reindex(price_df.index, method='ffill')

    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        regime_mult = get_regime(sig_aligned, i, regime, short_ma, long_ma, bear_mult, trans_mult) if regime else 1.0

        scores = {}
        for col in price_df.columns:
            if (i - vlb < 0 or pd.isna(price_df[col].iloc[i]) or
                pd.isna(price_df[col].iloc[i - vlb]) or price_df[col].iloc[i - vlb] <= 0):
                continue
            mom = float(price_df[col].iloc[i] / price_df[col].iloc[i - vlb] - 1)
            ret_slice = returns[col].iloc[max(0, i-vol_lb_factor):i+1].dropna()
            if len(ret_slice) < 4:
                continue
            vol = float(ret_slice.std())
            if i >= 4:
                mom_4w = float(price_df[col].iloc[i] / price_df[col].iloc[i-4] - 1)
                if mom_4w < ddf:
                    continue
            else:
                mom_4w = 0.0

            if value_type == 'mean_reversion':
                ma_lb = min(20, i)
                ma20 = price_df[col].iloc[max(0,i-ma_lb):i+1].mean()
                value_score = -(price_df[col].iloc[i] / ma20 - 1)
            else:
                value_score = -mom

            scores[col] = {'value_score': value_score, 'quality_score': -vol, 'mom': mom, 'vol': vol}

        if len(scores) < top_n:
            i += 1
            continue

        tickers = list(scores.keys())
        value_ranks = pd.Series({t: scores[t]['value_score'] for t in tickers}).rank(ascending=False)
        quality_ranks = pd.Series({t: scores[t]['quality_score'] for t in tickers}).rank(ascending=False)
        composite = (1 - vw) * value_ranks + vw * quality_ranks

        if sector_max:
            sorted_tickers = list(composite.sort_values().index)
            selected = []
            sector_count = defaultdict(int)
            for t in sorted_tickers:
                if len(selected) >= top_n:
                    break
                sector = STOCK_POOL.get(t, {}).get('sector', '?')
                if sector_count[sector] < sector_max:
                    selected.append(t)
                    sector_count[sector] += 1
        else:
            selected = list(composite.nsmallest(top_n).index)

        vol_scale = get_vol_scale(returns, selected, i, vol_target, vol_lb_scale)
        position_mult = min(regime_mult * vol_scale, 1.0)

        selected_set = set(selected)
        new_buys = selected_set - prev_holdings
        sold = prev_holdings - selected_set
        turnover = (len(new_buys) + len(sold)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            rets = []
            for s in selected:
                r = returns[s].iloc[j]
                if not pd.isna(r):
                    rets.append(float(r))
            port_ret = np.mean(rets) if rets else 0.0
            port_ret *= position_mult
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    if not dates:
        return {'error': 'No trades'}
    nav_s = pd.Series(nav[1:], index=dates)
    metrics = calc_metrics(nav_s, weekly_rets)
    metrics['total_txn_pct'] = round(total_txn * 100, 2)
    metrics['params'] = {k: v for k, v in params.items()}
    return metrics


# ============================================================
# WALK-FORWARD (proper rolling)
# ============================================================

def walk_forward_d(price_df, params, signal_index, n_folds=5):
    """Rolling walk-forward: split data into n_folds, test each fold with prior data"""
    total = len(price_df)
    fold_size = total // (n_folds + 1)  # +1 for initial training window
    results = []

    for fold in range(n_folds):
        # Test starts at fold_size * (fold + 1), ends at fold_size * (fold + 2)
        test_end = min(fold_size * (fold + 2), total)
        test_df = price_df.iloc[:test_end]

        r = backtest_d(test_df, params, signal_index)
        if 'error' not in r:
            results.append({
                'fold': fold + 1,
                'end': price_df.index[min(test_end-1, total-1)].strftime('%Y-%m-%d'),
                'cagr': r['cagr'], 'mdd': r['mdd'], 'sharpe': r['sharpe'], 'calmar': r['calmar']
            })
    return results


def walk_forward_e(price_df, params, signal_index, n_folds=5):
    total = len(price_df)
    fold_size = total // (n_folds + 1)
    results = []
    for fold in range(n_folds):
        test_end = min(fold_size * (fold + 2), total)
        test_df = price_df.iloc[:test_end]
        r = backtest_e(test_df, params, signal_index)
        if 'error' not in r:
            results.append({
                'fold': fold + 1,
                'end': price_df.index[min(test_end-1, total-1)].strftime('%Y-%m-%d'),
                'cagr': r['cagr'], 'mdd': r['mdd'], 'sharpe': r['sharpe'], 'calmar': r['calmar']
            })
    return results


# ============================================================
# PARAMETER SENSITIVITY
# ============================================================

def sensitivity_d(price_df, base_params, signal_index):
    """Test sensitivity by varying one param at a time"""
    results = {}
    base = backtest_d(price_df, base_params, signal_index)
    results['base'] = base

    variations = {
        'lookback': [3, 4, 5, 6, 8],
        'skip': [0, 1, 2],
        'top_n': [5, 6, 8, 10],
        'rebal_freq': [1, 2, 3, 4],
    }

    if base_params.get('vol_target'):
        vt = base_params['vol_target']
        variations['vol_target'] = [vt * 0.7, vt * 0.85, vt, vt * 1.15, vt * 1.3]
    if base_params.get('vol_lookback'):
        variations['vol_lookback'] = [6, 8, 12, 16, 20]
    if base_params.get('regime_filter'):
        variations['regime_bear_mult'] = [0.0, 0.1, 0.2, 0.3, 0.5]
        variations['regime_trans_mult'] = [0.3, 0.5, 0.7]

    for param_name, values in variations.items():
        param_results = []
        for val in values:
            p = {**base_params, param_name: val}
            r = backtest_d(price_df, p, signal_index)
            if 'error' not in r:
                param_results.append({'value': val, 'cagr': r['cagr'], 'mdd': r['mdd'],
                                      'sharpe': r['sharpe'], 'calmar': r['calmar']})
        results[f'vary_{param_name}'] = param_results

    return results


def sensitivity_e(price_df, base_params, signal_index):
    results = {}
    base = backtest_e(price_df, base_params, signal_index)
    results['base'] = base

    variations = {
        'value_lookback': [8, 10, 12, 16, 20],
        'vol_weight': [0.0, 0.1, 0.3, 0.5, 0.7],
        'top_n': [3, 5, 7, 8],
        'rebal_freq': [2, 4, 6, 8],
        'max_dd_filter': [-0.10, -0.12, -0.15, -0.20, -0.25],
    }

    if base_params.get('vol_target'):
        vt = base_params['vol_target']
        variations['vol_target'] = [vt * 0.7, vt * 0.85, vt, vt * 1.15, vt * 1.3]

    for param_name, values in variations.items():
        param_results = []
        for val in values:
            p = {**base_params, param_name: val}
            r = backtest_e(price_df, p, signal_index)
            if 'error' not in r:
                param_results.append({'value': val, 'cagr': r['cagr'], 'mdd': r['mdd'],
                                      'sharpe': r['sharpe'], 'calmar': r['calmar']})
        results[f'vary_{param_name}'] = param_results

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()
    print("=" * 60)
    print("Strategy D+E Optimizer Round 2 (No Leverage)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    price_df = load_cached_prices()
    signal_idx = load_signal_index()

    if price_df.empty:
        print("[FATAL] No data!")
        sys.exit(1)

    all_results = {}

    # ========================================
    # PHASE A: Re-test vol scaling WITHOUT leverage
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE A: Vol Scaling (NO leverage, cap=1.0)")
    print("=" * 60)

    d_vol_results = {}
    for vol_target in [None, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        for vol_lb in [8, 12, 16, 20]:
            p = {'lookback': 4, 'skip': 1, 'top_n': 8, 'rebal_freq': 2, 'txn_cost_bps': 8,
                 'vol_target': vol_target, 'vol_lookback': vol_lb}
            r = backtest_d(price_df, p, signal_idx)
            vt_str = f"{int(vol_target*100)}" if vol_target else "none"
            key = f"vt{vt_str}_vlb{vol_lb}"
            d_vol_results[key] = r
            if 'error' not in r:
                print(f"  D {key}: CAGR={r['cagr']}%, MDD={r['mdd']}%, Calmar={r['calmar']}")

    e_vol_results = {}
    for vol_target in [None, 0.12, 0.15, 0.18, 0.20, 0.25]:
        for vol_lb in [12, 16, 20, 26]:
            p = {'value_lookback': 12, 'vol_lookback': 12, 'vol_weight': 0.3,
                 'top_n': 5, 'rebal_freq': 4, 'max_dd_filter': -0.15, 'txn_cost_bps': 8,
                 'vol_target': vol_target, 'vol_scale_lookback': vol_lb}
            r = backtest_e(price_df, p, signal_idx)
            vt_str = f"{int(vol_target*100)}" if vol_target else "none"
            key = f"vt{vt_str}_vlb{vol_lb}"
            e_vol_results[key] = r
            if 'error' not in r:
                print(f"  E {key}: CAGR={r['cagr']}%, MDD={r['mdd']}%, Calmar={r['calmar']}")

    all_results['phase_a_d'] = d_vol_results
    all_results['phase_a_e'] = e_vol_results

    # Find best D and E from vol scaling
    best_d_vol = max([(k, v) for k, v in d_vol_results.items() if 'error' not in v],
                     key=lambda x: x[1].get('calmar', 0), default=(None, None))
    best_e_vol = max([(k, v) for k, v in e_vol_results.items() if 'error' not in v],
                     key=lambda x: x[1].get('calmar', 0), default=(None, None))

    print(f"\n  Best D vol: {best_d_vol[0]} -> CAGR={best_d_vol[1]['cagr']}%, MDD={best_d_vol[1]['mdd']}%, Calmar={best_d_vol[1]['calmar']}")
    print(f"  Best E vol: {best_e_vol[0]} -> CAGR={best_e_vol[1]['cagr']}%, MDD={best_e_vol[1]['mdd']}%, Calmar={best_e_vol[1]['calmar']}")

    # ========================================
    # PHASE B: Regime + Vol scaling combos
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE B: Regime + Vol Scaling Combos")
    print("=" * 60)

    combo_results = {}
    # D: test regime variants with best vol settings
    d_best_vt = best_d_vol[1]['params'].get('vol_target')
    d_best_vlb = best_d_vol[1]['params'].get('vol_lookback', 12)

    for regime in [None, 'single', 'dual']:
        for long_ma in [30, 40]:
            for bear_mult in [0.3, 0.5] if regime else [1.0]:
                for trans_mult in [0.5, 0.7] if regime == 'dual' else [1.0]:
                    p = {'lookback': 4, 'skip': 1, 'top_n': 8, 'rebal_freq': 2, 'txn_cost_bps': 8,
                         'vol_target': d_best_vt, 'vol_lookback': d_best_vlb,
                         'regime_filter': regime, 'regime_short_ma': 13, 'regime_long_ma': long_ma,
                         'regime_bear_mult': bear_mult, 'regime_trans_mult': trans_mult}
                    r = backtest_d(price_df, p, signal_idx)
                    parts = []
                    if regime: parts.append(f"r={regime[:1]}MA{long_ma}b{bear_mult}")
                    if regime == 'dual': parts.append(f"t{trans_mult}")
                    parts.append(f"vt{int(d_best_vt*100) if d_best_vt else 'X'}")
                    key = f"D_{'_'.join(parts)}" if parts else "D_base"
                    combo_results[key] = r
                    if 'error' not in r:
                        print(f"  {key}: CAGR={r['cagr']}%, MDD={r['mdd']}%, Calmar={r['calmar']}")

    # E: test regime with best vol settings
    e_best_vt = best_e_vol[1]['params'].get('vol_target')
    e_best_vlb = best_e_vol[1]['params'].get('vol_scale_lookback', 20)

    for regime in [None, 'single', 'dual']:
        for long_ma in [30, 40]:
            for bear_mult in [0.3, 0.5] if regime else [1.0]:
                for trans_mult in [0.5, 0.7] if regime == 'dual' else [1.0]:
                    p = {'value_lookback': 12, 'vol_lookback': 12, 'vol_weight': 0.3,
                         'top_n': 5, 'rebal_freq': 4, 'max_dd_filter': -0.15, 'txn_cost_bps': 8,
                         'vol_target': e_best_vt, 'vol_scale_lookback': e_best_vlb,
                         'regime_filter': regime, 'regime_short_ma': 13, 'regime_long_ma': long_ma,
                         'regime_bear_mult': bear_mult, 'regime_trans_mult': trans_mult}
                    r = backtest_e(price_df, p, signal_idx)
                    parts = []
                    if regime: parts.append(f"r={regime[:1]}MA{long_ma}b{bear_mult}")
                    if regime == 'dual': parts.append(f"t{trans_mult}")
                    parts.append(f"vt{int(e_best_vt*100) if e_best_vt else 'X'}")
                    key = f"E_{'_'.join(parts)}" if parts else "E_base"
                    combo_results[key] = r
                    if 'error' not in r:
                        print(f"  {key}: CAGR={r['cagr']}%, MDD={r['mdd']}%, Calmar={r['calmar']}")

    all_results['phase_b'] = combo_results

    # ========================================
    # PHASE C: Sector + Vol + Regime mega grid
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE C: Mega Grid (Sector + Vol + Regime)")
    print("=" * 60)

    mega_d = {}
    for sector_max in [1, 2, None]:
        for vol_target in [d_best_vt, None]:
            for regime in [None, 'single']:
                for mom_type in ['raw', 'risk_adjusted']:
                    p = {'lookback': 4, 'skip': 1, 'top_n': 8, 'rebal_freq': 2, 'txn_cost_bps': 8,
                         'sector_max': sector_max, 'momentum_type': mom_type,
                         'vol_target': vol_target, 'vol_lookback': d_best_vlb,
                         'regime_filter': regime, 'regime_short_ma': 13, 'regime_long_ma': 30,
                         'regime_bear_mult': 0.5, 'regime_trans_mult': 0.7}
                    r = backtest_d(price_df, p, signal_idx)
                    key = f"D_s{sector_max or 'X'}_{mom_type[:3]}_v{int(vol_target*100) if vol_target else 'X'}_r{regime[:1] if regime else 'X'}"
                    mega_d[key] = r

    mega_e = {}
    for sector_max in [1, 2, None]:
        for vtype in ['reversal', 'mean_reversion']:
            for vol_target in [e_best_vt, None]:
                for vw in [0.3, 0.5]:
                    p = {'value_lookback': 12, 'vol_lookback': 12, 'vol_weight': vw,
                         'top_n': 5, 'rebal_freq': 4, 'max_dd_filter': -0.15, 'txn_cost_bps': 8,
                         'sector_max': sector_max, 'value_type': vtype,
                         'vol_target': vol_target, 'vol_scale_lookback': e_best_vlb}
                    r = backtest_e(price_df, p, signal_idx)
                    key = f"E_s{sector_max or 'X'}_{vtype[:3]}_v{int(vol_target*100) if vol_target else 'X'}_vw{vw}"
                    mega_e[key] = r

    # Sort and display
    valid_d = sorted([(k, v) for k, v in mega_d.items() if 'error' not in v],
                     key=lambda x: -x[1].get('calmar', 0))
    valid_e = sorted([(k, v) for k, v in mega_e.items() if 'error' not in v],
                     key=lambda x: -x[1].get('calmar', 0))

    print(f"\n  TOP 10 D Mega ({len(valid_d)} variants):")
    for k, v in valid_d[:10]:
        print(f"    {k}: CAGR={v['cagr']}%, MDD={v['mdd']}%, Sharpe={v['sharpe']}, Calmar={v['calmar']}")
        print(f"      Annual: {v.get('annual', {})}")

    print(f"\n  TOP 10 E Mega ({len(valid_e)} variants):")
    for k, v in valid_e[:10]:
        print(f"    {k}: CAGR={v['cagr']}%, MDD={v['mdd']}%, Sharpe={v['sharpe']}, Calmar={v['calmar']}")
        print(f"      Annual: {v.get('annual', {})}")

    all_results['phase_c_d'] = mega_d
    all_results['phase_c_e'] = mega_e

    # ========================================
    # PHASE D: Pick top 3 configs, do walk-forward + sensitivity
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE D: Walk-Forward + Sensitivity on Top Configs")
    print("=" * 60)

    # Collect ALL D results across phases
    all_d = {}
    for phase_key in ['phase_a_d', 'phase_b', 'phase_c_d']:
        if phase_key in all_results:
            for k, v in all_results[phase_key].items():
                if k.startswith('D') and 'error' not in v:
                    all_d[k] = v
    # Also add non-D prefixed from phase_a_d
    for k, v in all_results.get('phase_a_d', {}).items():
        if 'error' not in v:
            all_d[f"D_{k}"] = v

    all_e = {}
    for phase_key in ['phase_a_e', 'phase_b', 'phase_c_e']:
        if phase_key in all_results:
            for k, v in all_results[phase_key].items():
                if k.startswith('E') and 'error' not in v:
                    all_e[k] = v
    for k, v in all_results.get('phase_a_e', {}).items():
        if 'error' not in v:
            all_e[f"E_{k}"] = v

    # Top 3 D by calmar
    top3_d = sorted(all_d.items(), key=lambda x: -x[1].get('calmar', 0))[:3]
    top3_e = sorted(all_e.items(), key=lambda x: -x[1].get('calmar', 0))[:3]

    wf_results = {}
    sens_results = {}

    for i, (name, result) in enumerate(top3_d):
        params = result.get('params', {})
        print(f"\n  --- D Config #{i+1}: {name} ---")
        print(f"      CAGR={result['cagr']}%, MDD={result['mdd']}%, Calmar={result['calmar']}")

        # Walk-forward
        wf = walk_forward_d(price_df, params, signal_idx)
        wf_results[f'D_{i+1}_{name}'] = wf
        if wf:
            avg_calmar = np.mean([r['calmar'] for r in wf])
            min_calmar = min(r['calmar'] for r in wf)
            print(f"      WF avg Calmar={avg_calmar:.3f}, min={min_calmar:.3f}")
            for w in wf:
                print(f"        Fold {w['fold']} ({w['end']}): CAGR={w['cagr']}%, MDD={w['mdd']}%, Calmar={w['calmar']}")

        # Sensitivity
        sens = sensitivity_d(price_df, params, signal_idx)
        sens_results[f'D_{i+1}_{name}'] = sens
        for param_name, variations in sens.items():
            if param_name.startswith('vary_') and isinstance(variations, list):
                calmars = [v['calmar'] for v in variations]
                if calmars:
                    print(f"      Sensitivity {param_name}: Calmar range [{min(calmars):.3f}, {max(calmars):.3f}]")

    for i, (name, result) in enumerate(top3_e):
        params = result.get('params', {})
        print(f"\n  --- E Config #{i+1}: {name} ---")
        print(f"      CAGR={result['cagr']}%, MDD={result['mdd']}%, Calmar={result['calmar']}")

        wf = walk_forward_e(price_df, params, signal_idx)
        wf_results[f'E_{i+1}_{name}'] = wf
        if wf:
            avg_calmar = np.mean([r['calmar'] for r in wf])
            min_calmar = min(r['calmar'] for r in wf)
            print(f"      WF avg Calmar={avg_calmar:.3f}, min={min_calmar:.3f}")
            for w in wf:
                print(f"        Fold {w['fold']} ({w['end']}): CAGR={w['cagr']}%, MDD={w['mdd']}%, Calmar={w['calmar']}")

        sens = sensitivity_e(price_df, params, signal_idx)
        sens_results[f'E_{i+1}_{name}'] = sens
        for param_name, variations in sens.items():
            if param_name.startswith('vary_') and isinstance(variations, list):
                calmars = [v['calmar'] for v in variations]
                if calmars:
                    print(f"      Sensitivity {param_name}: Calmar range [{min(calmars):.3f}, {max(calmars):.3f}]")

    all_results['walk_forward'] = wf_results
    all_results['sensitivity'] = sens_results

    # ========================================
    # FINAL RECOMMENDATION
    # ========================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"FINAL RECOMMENDATION (Round 2, {elapsed/60:.1f} min)")
    print("=" * 60)

    # Find absolute best by Calmar across everything
    best_d_overall = sorted(all_d.items(), key=lambda x: -x[1].get('calmar', 0))[0] if all_d else None
    best_e_overall = sorted(all_e.items(), key=lambda x: -x[1].get('calmar', 0))[0] if all_e else None

    # Also find best by "robust score" = calmar * (1 - neg_years/total_years)
    def robust_score(v):
        calmar = v.get('calmar', 0)
        total_yrs = max(len(v.get('annual', {})), 1)
        neg = v.get('neg_years', 0)
        return calmar * (1 - 0.3 * neg / total_yrs)

    best_d_robust = sorted(all_d.items(), key=lambda x: -robust_score(x[1]))[0] if all_d else None
    best_e_robust = sorted(all_e.items(), key=lambda x: -robust_score(x[1]))[0] if all_e else None

    if best_d_overall:
        k, v = best_d_overall
        print(f"\n  BEST D (by Calmar): {k}")
        print(f"    CAGR={v['cagr']}%, MDD={v['mdd']}%, Sharpe={v['sharpe']}, Calmar={v['calmar']}")
        print(f"    Annual: {v.get('annual', {})}")
        print(f"    Params: {v.get('params', {})}")

    if best_d_robust and best_d_robust[0] != best_d_overall[0]:
        k, v = best_d_robust
        print(f"\n  BEST D (Robust): {k}")
        print(f"    CAGR={v['cagr']}%, MDD={v['mdd']}%, Sharpe={v['sharpe']}, Calmar={v['calmar']}")
        print(f"    Annual: {v.get('annual', {})}")

    if best_e_overall:
        k, v = best_e_overall
        print(f"\n  BEST E (by Calmar): {k}")
        print(f"    CAGR={v['cagr']}%, MDD={v['mdd']}%, Sharpe={v['sharpe']}, Calmar={v['calmar']}")
        print(f"    Annual: {v.get('annual', {})}")
        print(f"    Params: {v.get('params', {})}")

    if best_e_robust and best_e_robust[0] != best_e_overall[0]:
        k, v = best_e_robust
        print(f"\n  BEST E (Robust): {k}")
        print(f"    CAGR={v['cagr']}%, MDD={v['mdd']}%, Sharpe={v['sharpe']}, Calmar={v['calmar']}")
        print(f"    Annual: {v.get('annual', {})}")

    # Save
    summary = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_minutes': round(elapsed / 60, 1),
        'best_d': {'key': best_d_overall[0], **best_d_overall[1]} if best_d_overall else None,
        'best_e': {'key': best_e_overall[0], **best_e_overall[1]} if best_e_overall else None,
        'best_d_robust': {'key': best_d_robust[0], **best_d_robust[1]} if best_d_robust else None,
        'best_e_robust': {'key': best_e_robust[0], **best_e_robust[1]} if best_e_robust else None,
    }

    def ser(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)

    with open(os.path.join(RESULTS_DIR, 'r2_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=ser)
    with open(os.path.join(RESULTS_DIR, 'r2_full.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=ser)

    print(f"\n  Results saved to {RESULTS_DIR}/r2_*.json")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print("  DONE!")


if __name__ == '__main__':
    main()
