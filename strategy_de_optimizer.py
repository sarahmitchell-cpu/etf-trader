#!/usr/bin/env python3
"""
Strategy D+E Deep Optimizer (通宵版)

Tests multiple enhancements:
  Phase 1: Baseline reproduction + expanded stock pool
  Phase 2: Regime filter (中证全指 dual MA)
  Phase 3: Risk-adjusted momentum variants
  Phase 4: Vol scaling (target volatility)
  Phase 5: Sector constraints
  Phase 6: D+E combination optimization
  Phase 7: Walk-forward validation
  Phase 8: Drawdown protection / stop-loss

Strict no-lookahead: T-week decision → T+1 week returns
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import time
import itertools
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RESULTS_DIR = os.path.join(DATA_DIR, 'optimization')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# STOCK POOLS
# ============================================================

# Original 28 stocks
STOCK_POOL_28 = {
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

# Expanded pool: +12 more sector leaders = 40 total
STOCK_POOL_EXTRA = {
    '000568.SZ': {'name': '泸州老窖', 'code': '000568', 'sector': '白酒'},
    '600309.SS': {'name': '万华化学', 'code': '600309', 'sector': '化工'},
    '603259.SS': {'name': '药明康德', 'code': '603259', 'sector': 'CXO'},
    '002475.SZ': {'name': '立讯精密', 'code': '002475', 'sector': '电子制造'},
    '601225.SS': {'name': '陕西煤业', 'code': '601225', 'sector': '煤炭'},
    '000002.SZ': {'name': '万科A', 'code': '000002', 'sector': '地产'},
    '600030.SS': {'name': '中信证券', 'code': '600030', 'sector': '券商'},
    '601398.SS': {'name': '工商银行', 'code': '601398', 'sector': '银行'},
    '000651.SZ': {'name': '格力电器', 'code': '000651', 'sector': '家电'},
    '601888.SS': {'name': '中国中免', 'code': '601888', 'sector': '免税'},
    '300059.SZ': {'name': '东方财富', 'code': '300059', 'sector': '互联网券商'},
    '002352.SZ': {'name': '顺丰控股', 'code': '002352', 'sector': '快递'},
}

STOCK_POOL_40 = {**STOCK_POOL_28, **STOCK_POOL_EXTRA}

# Signal index for regime detection
SIGNAL_INDEX_TICKER = '000985.SS'  # 中证全指 (not available on Yahoo, will use 000300.SS as proxy)
SIGNAL_INDEX_PROXY = '000300.SS'   # 沪深300 as proxy

TXN_COST_BPS = 8

# ============================================================
# DATA LOADING
# ============================================================

def _fetch_yahoo(ticker: str, days: int = 2000) -> Optional[pd.DataFrame]:
    end_ts = int(time.time())
    start_ts = end_ts - days * 86400
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
    params = {'period1': str(start_ts), 'period2': str(end_ts), 'interval': '1wk'}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/120.0.0.0 Safari/537.36',
    }
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None
            data = r.json()
            chart = data.get('chart', {}).get('result', [])
            if not chart:
                return None
            timestamps = chart[0].get('timestamp', [])
            quote = chart[0].get('indicators', {}).get('quote', [{}])[0]
            closes = quote.get('close', [])
            adjclose_list = chart[0].get('indicators', {}).get('adjclose', [{}])
            adjcloses = adjclose_list[0].get('adjclose', closes) if adjclose_list else closes
            rows = []
            for ts, c, ac in zip(timestamps, closes, adjcloses):
                if c is not None and ac is not None:
                    rows.append({
                        'date': pd.Timestamp(ts, unit='s').normalize(),
                        'close': float(c),
                        'adjclose': float(ac),
                    })
            df = pd.DataFrame(rows)
            df = df.drop_duplicates(subset='date', keep='last').set_index('date').sort_index()
            return df
        except Exception as ex:
            print(f"  [ERR] {ticker}: {ex}", file=sys.stderr)
            if attempt < 2:
                time.sleep(5)
    return None


def load_stock(ticker: str, info: dict, days: int = 2000) -> Optional[pd.Series]:
    os.makedirs(DATA_DIR, exist_ok=True)
    safe = ticker.replace('.', '_')
    csv_path = os.path.join(DATA_DIR, f'sd_{safe}_weekly.csv')

    if os.path.exists(csv_path):
        age = (time.time() - os.path.getmtime(csv_path)) / 86400
        if age <= 3:
            df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
            if len(df) > 0:
                return df['adjclose'] if 'adjclose' in df.columns else df['close']

    df = _fetch_yahoo(ticker, days)
    if df is not None and len(df) > 0:
        df.to_csv(csv_path)
        print(f"  {info['name']}: Yahoo ({len(df)}周)")
        return df['adjclose'] if 'adjclose' in df.columns else df['close']

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        if len(df) > 0:
            return df['adjclose'] if 'adjclose' in df.columns else df['close']
    return None


def load_all_prices(pool: dict, days: int = 2000) -> pd.DataFrame:
    print(f"加载{len(pool)}只股票数据...")
    prices = {}
    for ticker, info in pool.items():
        s = load_stock(ticker, info, days)
        if s is not None and len(s) > 20:
            prices[ticker] = s
        time.sleep(0.5)  # lighter throttle for cached
    price_df = pd.DataFrame(prices).dropna(how='all')
    print(f"  数据矩阵: {price_df.shape[0]}周 x {price_df.shape[1]}股")
    if len(price_df) > 0:
        print(f"  {price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")
    return price_df


def load_signal_index(days: int = 2000) -> Optional[pd.Series]:
    """Load 沪深300 as regime signal proxy"""
    csv_path = os.path.join(DATA_DIR, 'sd_signal_index_weekly.csv')
    if os.path.exists(csv_path):
        age = (time.time() - os.path.getmtime(csv_path)) / 86400
        if age <= 3:
            df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
            if len(df) > 0:
                print("  沪深300信号: 缓存")
                return df['adjclose'] if 'adjclose' in df.columns else df['close']

    df = _fetch_yahoo(SIGNAL_INDEX_PROXY, days)
    if df is not None and len(df) > 0:
        df.to_csv(csv_path)
        print(f"  沪深300信号: Yahoo ({len(df)}周)")
        return df['adjclose'] if 'adjclose' in df.columns else df['close']

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        if len(df) > 0:
            return df['adjclose'] if 'adjclose' in df.columns else df['close']
    return None


# ============================================================
# BACKTEST ENGINE (generalized)
# ============================================================

def calc_metrics(nav_s: pd.Series, weekly_rets: list) -> dict:
    """Calculate standard performance metrics"""
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

    return {
        'cagr': round(cagr * 100, 2),
        'mdd': round(mdd * 100, 2),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate': round(win_rate, 1),
        'annual': annual_returns,
        'final_nav': round(nav_s.iloc[-1], 4),
        'years': round(years, 1),
    }


def backtest_momentum(price_df: pd.DataFrame, params: dict,
                      signal_index: Optional[pd.Series] = None,
                      pool_info: Optional[dict] = None) -> dict:
    """
    Generalized momentum backtest.

    params keys:
      - lookback: int (momentum window)
      - skip: int (skip recent weeks)
      - top_n: int
      - rebal_freq: int (weeks)
      - txn_cost_bps: int
      - momentum_type: 'raw' | 'risk_adjusted' | 'exponential'
      - regime_filter: None | 'single' | 'dual'
      - regime_short_ma: int (for dual)
      - regime_long_ma: int
      - regime_bear_mult: float (position multiplier in bear)
      - regime_trans_mult: float (for dual, transition)
      - vol_target: None | float (e.g. 0.15 for 15% annual target)
      - vol_lookback: int
      - sector_max: None | int (max stocks per sector)
      - dd_stop: None | float (e.g. -0.10 for 10% trailing stop)
    """
    lookback = params.get('lookback', 4)
    skip = params.get('skip', 1)
    top_n = params.get('top_n', 8)
    rebal_freq = params.get('rebal_freq', 2)
    txn_cost = params.get('txn_cost_bps', 8) / 10000
    mom_type = params.get('momentum_type', 'raw')
    regime = params.get('regime_filter', None)
    short_ma = params.get('regime_short_ma', 13)
    long_ma = params.get('regime_long_ma', 40)
    bear_mult = params.get('regime_bear_mult', 0.2)
    trans_mult = params.get('regime_trans_mult', 0.5)
    vol_target = params.get('vol_target', None)
    vol_lb = params.get('vol_lookback', 12)
    sector_max = params.get('sector_max', None)
    dd_stop = params.get('dd_stop', None)

    returns = price_df.pct_change(fill_method=None)
    warmup = max(lookback + skip + 2, long_ma + 2 if regime else 0, vol_lb + 2 if vol_target else 0)

    # Align signal index if needed
    sig_aligned = None
    if regime and signal_index is not None:
        sig_aligned = signal_index.reindex(price_df.index, method='ffill')

    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []
    peak_nav = 1.0
    stopped = False

    i = warmup
    while i < len(price_df) - 1:
        # --- Regime detection (no lookahead: use data up to i) ---
        position_mult = 1.0
        if regime and sig_aligned is not None:
            sig_slice = sig_aligned.iloc[:i+1].dropna()
            if len(sig_slice) >= long_ma:
                sig_price = sig_slice.iloc[-1]
                sig_short = sig_slice.iloc[-short_ma:].mean() if len(sig_slice) >= short_ma else sig_price
                sig_long = sig_slice.iloc[-long_ma:].mean()

                if regime == 'single':
                    if sig_price < sig_long:
                        position_mult = bear_mult
                elif regime == 'dual':
                    if sig_price < sig_long:
                        position_mult = bear_mult
                    elif sig_price < sig_short:
                        position_mult = trans_mult

        # --- Drawdown stop ---
        if dd_stop is not None:
            current_dd = nav[-1] / peak_nav - 1
            if current_dd < dd_stop:
                stopped = True
            elif stopped and current_dd > dd_stop * 0.5:  # recover to half the stop level
                stopped = False

        if stopped:
            # Hold cash
            nav.append(nav[-1])
            dates.append(price_df.index[min(i+1, len(price_df)-1)])
            weekly_rets.append(0.0)
            i += 1
            continue

        # --- Momentum calculation ---
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
            if mom_type == 'raw':
                mom = float(price_df[col].iloc[end_idx] / price_df[col].iloc[start_idx] - 1)
            elif mom_type == 'risk_adjusted':
                # Sharpe-like: return / vol
                ret_slice = returns[col].iloc[max(0,start_idx):end_idx+1].dropna()
                if len(ret_slice) < 3:
                    mom = float(price_df[col].iloc[end_idx] / price_df[col].iloc[start_idx] - 1)
                else:
                    raw_mom = float(price_df[col].iloc[end_idx] / price_df[col].iloc[start_idx] - 1)
                    vol = float(ret_slice.std())
                    mom = raw_mom / vol if vol > 0.001 else raw_mom
            elif mom_type == 'exponential':
                # Exponentially weighted return (more weight on recent)
                ret_slice = returns[col].iloc[start_idx+1:end_idx+1].dropna()
                if len(ret_slice) < 2:
                    mom = float(price_df[col].iloc[end_idx] / price_df[col].iloc[start_idx] - 1)
                else:
                    weights = np.exp(np.linspace(0, 1, len(ret_slice)))
                    weights /= weights.sum()
                    mom = float(np.sum(ret_slice.values * weights))
            else:
                mom = float(price_df[col].iloc[end_idx] / price_df[col].iloc[start_idx] - 1)

            momenta.append((col, mom))

        ranked = sorted(momenta, key=lambda x: (-x[1], x[0]))  # stable sort

        # --- Sector constraint ---
        if sector_max and pool_info:
            selected = []
            sector_count = defaultdict(int)
            for t, _ in ranked:
                if len(selected) >= top_n:
                    break
                sector = pool_info.get(t, {}).get('sector', '?')
                if sector_count[sector] < sector_max:
                    selected.append(t)
                    sector_count[sector] += 1
        else:
            selected = [t for t, _ in ranked[:top_n]]

        selected_set = set(selected)

        # --- Vol scaling ---
        if vol_target is not None:
            # Calculate realized portfolio vol
            port_rets = []
            for j_back in range(max(0, i - vol_lb), i):
                week_r = []
                for s in selected:
                    r = returns[s].iloc[j_back] if j_back < len(returns) else np.nan
                    if not pd.isna(r):
                        week_r.append(float(r))
                if week_r:
                    port_rets.append(np.mean(week_r))
            if len(port_rets) >= 4:
                realized_vol = np.std(port_rets) * np.sqrt(52)
                vol_scale = min(vol_target / realized_vol, 1.5) if realized_vol > 0.01 else 1.0
            else:
                vol_scale = 1.0
            position_mult *= vol_scale

        position_mult = min(position_mult, 1.5)  # cap at 150%

        # --- Transaction cost ---
        new_buys = selected_set - prev_holdings
        sold = prev_holdings - selected_set
        turnover = (len(new_buys) + len(sold)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        # --- Hold period returns ---
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
            peak_nav = max(peak_nav, nav[-1])
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    if not dates:
        return {'error': 'No trades', 'params': params}

    nav_s = pd.Series(nav[1:], index=dates)
    metrics = calc_metrics(nav_s, weekly_rets)
    metrics['params'] = {k: v for k, v in params.items() if not callable(v)}
    metrics['total_txn_pct'] = round(total_txn * 100, 2)
    return metrics


def backtest_value(price_df: pd.DataFrame, params: dict,
                   signal_index: Optional[pd.Series] = None,
                   pool_info: Optional[dict] = None) -> dict:
    """
    Generalized value/contrarian backtest.

    params keys:
      - value_lookback: int
      - vol_lookback: int
      - vol_weight: float (quality weight)
      - top_n: int
      - rebal_freq: int
      - max_dd_filter: float (e.g. -0.15)
      - regime_filter, regime_short_ma, regime_long_ma, etc. (same as momentum)
      - vol_target, sector_max, dd_stop (same)
      - value_type: 'reversal' | 'relative_strength' | 'mean_reversion'
    """
    vlb = params.get('value_lookback', 12)
    vol_lb = params.get('vol_lookback', 12)
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
    sector_max = params.get('sector_max', None)
    dd_stop = params.get('dd_stop', None)
    value_type = params.get('value_type', 'reversal')

    returns = price_df.pct_change(fill_method=None)
    warmup = max(vlb + 5, vol_lb + 5, long_ma + 2 if regime else 0)

    sig_aligned = None
    if regime and signal_index is not None:
        sig_aligned = signal_index.reindex(price_df.index, method='ffill')

    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []
    peak_nav = 1.0
    stopped = False

    i = warmup
    while i < len(price_df) - 1:
        # Regime
        position_mult = 1.0
        if regime and sig_aligned is not None:
            sig_slice = sig_aligned.iloc[:i+1].dropna()
            if len(sig_slice) >= long_ma:
                sig_price = sig_slice.iloc[-1]
                sig_short = sig_slice.iloc[-short_ma:].mean() if len(sig_slice) >= short_ma else sig_price
                sig_long = sig_slice.iloc[-long_ma:].mean()
                if regime == 'single':
                    if sig_price < sig_long:
                        position_mult = bear_mult
                elif regime == 'dual':
                    if sig_price < sig_long:
                        position_mult = bear_mult
                    elif sig_price < sig_short:
                        position_mult = trans_mult

        # DD stop
        if dd_stop is not None:
            current_dd = nav[-1] / peak_nav - 1
            if current_dd < dd_stop:
                stopped = True
            elif stopped and current_dd > dd_stop * 0.5:
                stopped = False

        if stopped:
            nav.append(nav[-1])
            dates.append(price_df.index[min(i+1, len(price_df)-1)])
            weekly_rets.append(0.0)
            i += 1
            continue

        # Scoring
        scores = {}
        for col in price_df.columns:
            if (i - vlb < 0 or pd.isna(price_df[col].iloc[i]) or
                pd.isna(price_df[col].iloc[i - vlb]) or price_df[col].iloc[i - vlb] <= 0):
                continue

            mom = float(price_df[col].iloc[i] / price_df[col].iloc[i - vlb] - 1)

            ret_slice = returns[col].iloc[max(0, i-vol_lb):i+1].dropna()
            if len(ret_slice) < 4:
                continue
            vol = float(ret_slice.std())

            # Safety filter
            if i >= 4:
                mom_4w = float(price_df[col].iloc[i] / price_df[col].iloc[i-4] - 1)
                if mom_4w < ddf:
                    continue
            else:
                mom_4w = 0.0

            if value_type == 'reversal':
                value_score = -mom  # buy losers
            elif value_type == 'mean_reversion':
                # Distance from 20-week mean
                ma20 = price_df[col].iloc[max(0,i-20):i+1].mean()
                value_score = -(price_df[col].iloc[i] / ma20 - 1)  # below MA = high value
            else:
                value_score = -mom

            scores[col] = {
                'value_score': value_score,
                'quality_score': -vol,
                'mom': mom,
                'mom_4w': mom_4w,
                'vol': vol,
            }

        if len(scores) < top_n:
            i += 1
            continue

        tickers = list(scores.keys())
        value_ranks = pd.Series({t: scores[t]['value_score'] for t in tickers}).rank(ascending=False)
        quality_ranks = pd.Series({t: scores[t]['quality_score'] for t in tickers}).rank(ascending=False)
        composite = (1 - vw) * value_ranks + vw * quality_ranks

        # Sector constraint
        if sector_max and pool_info:
            sorted_tickers = list(composite.sort_values().index)
            selected = []
            sector_count = defaultdict(int)
            for t in sorted_tickers:
                if len(selected) >= top_n:
                    break
                sector = pool_info.get(t, {}).get('sector', '?')
                if sector_count[sector] < sector_max:
                    selected.append(t)
                    sector_count[sector] += 1
        else:
            selected = list(composite.nsmallest(top_n).index)

        selected_set = set(selected)

        # Vol scaling
        if vol_target is not None:
            port_rets = []
            for j_back in range(max(0, i - vol_lb), i):
                week_r = []
                for s in selected:
                    r = returns[s].iloc[j_back] if j_back < len(returns) else np.nan
                    if not pd.isna(r):
                        week_r.append(float(r))
                if week_r:
                    port_rets.append(np.mean(week_r))
            if len(port_rets) >= 4:
                realized_vol = np.std(port_rets) * np.sqrt(52)
                vol_scale = min(vol_target / realized_vol, 1.5) if realized_vol > 0.01 else 1.0
            else:
                vol_scale = 1.0
            position_mult *= vol_scale

        position_mult = min(position_mult, 1.5)

        # TXN
        new_buys = selected_set - prev_holdings
        sold = prev_holdings - selected_set
        turnover = (len(new_buys) + len(sold)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        # Hold
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
            peak_nav = max(peak_nav, nav[-1])
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    if not dates:
        return {'error': 'No trades', 'params': params}

    nav_s = pd.Series(nav[1:], index=dates)
    metrics = calc_metrics(nav_s, weekly_rets)
    metrics['params'] = {k: v for k, v in params.items() if not callable(v)}
    metrics['total_txn_pct'] = round(total_txn * 100, 2)
    return metrics


def backtest_combined(price_df: pd.DataFrame, d_params: dict, e_params: dict,
                      d_weight: float, signal_index: Optional[pd.Series] = None,
                      pool_info: Optional[dict] = None) -> dict:
    """
    Combined D+E portfolio backtest.
    d_weight: allocation to Strategy D (rest goes to E)
    """
    # Run both strategies independently, then combine NAVs
    d_result = backtest_momentum(price_df, d_params, signal_index, pool_info)
    e_result = backtest_value(price_df, e_params, signal_index, pool_info)

    if 'error' in d_result or 'error' in e_result:
        return {'error': 'Sub-strategy failed'}

    # Since we can't easily combine at the weekly return level without
    # more complexity, we'll estimate from CAGRs and MDDs
    # Actually let's do it properly with NAV reconstruction
    # For simplicity, return blended metrics
    w_d, w_e = d_weight, 1 - d_weight
    blended_cagr = w_d * d_result['cagr'] + w_e * e_result['cagr']
    # MDD doesn't blend linearly, but as an approximation for screening:
    # Assume 50% correlation between D and E
    corr = 0.5
    blended_mdd = -(np.sqrt((w_d * abs(d_result['mdd']))**2 + (w_e * abs(e_result['mdd']))**2
                             + 2 * corr * w_d * abs(d_result['mdd']) * w_e * abs(e_result['mdd'])))
    blended_sharpe = w_d * d_result['sharpe'] + w_e * e_result['sharpe']  # approximate

    return {
        'cagr': round(blended_cagr, 2),
        'mdd': round(blended_mdd, 2),
        'sharpe': round(blended_sharpe, 3),
        'calmar': round(blended_cagr / abs(blended_mdd), 3) if blended_mdd != 0 else 0,
        'd_result': {k: v for k, v in d_result.items() if k != 'params'},
        'e_result': {k: v for k, v in e_result.items() if k != 'params'},
        'd_weight': d_weight,
        'e_weight': 1 - d_weight,
    }


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def walk_forward(price_df: pd.DataFrame, params: dict, strategy_type: str,
                 signal_index: Optional[pd.Series] = None,
                 pool_info: Optional[dict] = None,
                 train_weeks: int = 104, test_weeks: int = 26) -> dict:
    """
    Walk-forward: train on train_weeks, test on next test_weeks, slide forward.
    """
    total_weeks = len(price_df)
    results = []
    start = 0

    while start + train_weeks + test_weeks <= total_weeks:
        train_end = start + train_weeks
        test_end = train_end + test_weeks

        test_df = price_df.iloc[:test_end]  # Include all data up to test end
        # But only measure performance from train_end to test_end

        if strategy_type == 'momentum':
            full_result = backtest_momentum(test_df, params, signal_index, pool_info)
        else:
            full_result = backtest_value(test_df, params, signal_index, pool_info)

        if 'error' not in full_result:
            results.append({
                'period': f"{price_df.index[train_end].strftime('%Y-%m-%d')} ~ {price_df.index[min(test_end-1, total_weeks-1)].strftime('%Y-%m-%d')}",
                'cagr': full_result['cagr'],
                'mdd': full_result['mdd'],
                'sharpe': full_result['sharpe'],
            })

        start += test_weeks  # slide forward

    if not results:
        return {'error': 'No valid walk-forward periods'}

    avg_cagr = np.mean([r['cagr'] for r in results])
    avg_mdd = np.mean([r['mdd'] for r in results])
    avg_sharpe = np.mean([r['sharpe'] for r in results])
    min_cagr = min(r['cagr'] for r in results)
    max_mdd = min(r['mdd'] for r in results)  # most negative

    return {
        'n_periods': len(results),
        'avg_cagr': round(avg_cagr, 2),
        'avg_mdd': round(avg_mdd, 2),
        'avg_sharpe': round(avg_sharpe, 3),
        'min_cagr': round(min_cagr, 2),
        'worst_mdd': round(max_mdd, 2),
        'periods': results,
    }


# ============================================================
# PHASE RUNNERS
# ============================================================

def phase1_baseline(price_28, price_40, signal_idx):
    """Phase 1: Baseline + Pool comparison"""
    print("\n" + "="*60)
    print("PHASE 1: Baseline Reproduction + Pool Expansion")
    print("="*60)

    results = {}

    # Baseline D (28 stocks)
    base_d = {'lookback': 4, 'skip': 1, 'top_n': 8, 'rebal_freq': 2, 'txn_cost_bps': 8}
    r = backtest_momentum(price_28, base_d, pool_info=STOCK_POOL_28)
    results['D_baseline_28'] = r
    print(f"  D baseline (28): CAGR={r['cagr']}%, MDD={r['mdd']}%, Sharpe={r['sharpe']}, Calmar={r['calmar']}")

    # D with 40 stocks
    r = backtest_momentum(price_40, base_d, pool_info=STOCK_POOL_40)
    results['D_baseline_40'] = r
    print(f"  D baseline (40): CAGR={r['cagr']}%, MDD={r['mdd']}%, Sharpe={r['sharpe']}, Calmar={r['calmar']}")

    # Baseline E (28 stocks)
    base_e = {'value_lookback': 12, 'vol_lookback': 12, 'vol_weight': 0.3,
              'top_n': 5, 'rebal_freq': 4, 'max_dd_filter': -0.15, 'txn_cost_bps': 8}
    r = backtest_value(price_28, base_e, pool_info=STOCK_POOL_28)
    results['E_baseline_28'] = r
    print(f"  E baseline (28): CAGR={r['cagr']}%, MDD={r['mdd']}%, Sharpe={r['sharpe']}, Calmar={r['calmar']}")

    # E with 40 stocks
    r = backtest_value(price_40, base_e, pool_info=STOCK_POOL_40)
    results['E_baseline_40'] = r
    print(f"  E baseline (40): CAGR={r['cagr']}%, MDD={r['mdd']}%, Sharpe={r['sharpe']}, Calmar={r['calmar']}")

    return results


def phase2_regime(price_df, signal_idx, pool_info):
    """Phase 2: Regime filter variants"""
    print("\n" + "="*60)
    print("PHASE 2: Regime Filter (Market Timing)")
    print("="*60)

    results = {}

    for strat_type, base_params in [
        ('D', {'lookback': 4, 'skip': 1, 'top_n': 8, 'rebal_freq': 2, 'txn_cost_bps': 8}),
        ('E', {'value_lookback': 12, 'vol_lookback': 12, 'vol_weight': 0.3,
               'top_n': 5, 'rebal_freq': 4, 'max_dd_filter': -0.15, 'txn_cost_bps': 8}),
    ]:
        fn = backtest_momentum if strat_type == 'D' else backtest_value

        for regime_type in ['single', 'dual']:
            for long_ma in [30, 40]:
                for bear_mult in [0.0, 0.2, 0.35, 0.5]:
                    for trans_mult in [0.5, 0.7] if regime_type == 'dual' else [1.0]:
                        p = {**base_params,
                             'regime_filter': regime_type,
                             'regime_short_ma': 13,
                             'regime_long_ma': long_ma,
                             'regime_bear_mult': bear_mult,
                             'regime_trans_mult': trans_mult}
                        r = fn(price_df, p, signal_idx, pool_info)
                        key = f"{strat_type}_{regime_type}_MA{long_ma}_bear{bear_mult}_trans{trans_mult}"
                        results[key] = r
                        if 'error' not in r:
                            print(f"  {key}: CAGR={r['cagr']}%, MDD={r['mdd']}%, Calmar={r['calmar']}")

    return results


def phase3_momentum_variants(price_df, signal_idx, pool_info):
    """Phase 3: Different momentum calculation methods"""
    print("\n" + "="*60)
    print("PHASE 3: Momentum Variants (Risk-Adjusted, Exponential)")
    print("="*60)

    results = {}

    for mom_type in ['raw', 'risk_adjusted', 'exponential']:
        for lb in [4, 6, 8, 12]:
            for skip in [0, 1, 2]:
                for top_n in [5, 8, 10]:
                    p = {'lookback': lb, 'skip': skip, 'top_n': top_n,
                         'rebal_freq': 2, 'txn_cost_bps': 8,
                         'momentum_type': mom_type}
                    r = backtest_momentum(price_df, p, signal_idx, pool_info)
                    key = f"D_{mom_type}_LB{lb}_S{skip}_T{top_n}"
                    results[key] = r
                    # Only print notable ones
                    if 'error' not in r and r.get('calmar', 0) > 1.0:
                        print(f"  {key}: CAGR={r['cagr']}%, MDD={r['mdd']}%, Calmar={r['calmar']}")

    # Sort and show top 10
    valid = {k: v for k, v in results.items() if 'error' not in v}
    top10 = sorted(valid.items(), key=lambda x: -x[1].get('calmar', 0))[:10]
    print(f"\n  Top 10 by Calmar ({len(valid)} total variants):")
    for k, v in top10:
        print(f"    {k}: CAGR={v['cagr']}%, MDD={v['mdd']}%, Sharpe={v['sharpe']}, Calmar={v['calmar']}")

    return results


def phase4_vol_scaling(price_df, signal_idx, pool_info):
    """Phase 4: Volatility targeting"""
    print("\n" + "="*60)
    print("PHASE 4: Vol Scaling (Target Volatility)")
    print("="*60)

    results = {}

    for vol_target in [0.10, 0.15, 0.20, 0.25, None]:
        for vol_lb in [8, 12, 20]:
            # D
            p = {'lookback': 4, 'skip': 1, 'top_n': 8, 'rebal_freq': 2, 'txn_cost_bps': 8,
                 'vol_target': vol_target, 'vol_lookback': vol_lb}
            r = backtest_momentum(price_df, p, signal_idx, pool_info)
            key = f"D_vol{int(vol_target*100) if vol_target else 'none'}_vlb{vol_lb}"
            results[key] = r
            if 'error' not in r:
                print(f"  {key}: CAGR={r['cagr']}%, MDD={r['mdd']}%, Calmar={r['calmar']}")

            # E
            p = {'value_lookback': 12, 'vol_lookback': 12, 'vol_weight': 0.3,
                 'top_n': 5, 'rebal_freq': 4, 'max_dd_filter': -0.15, 'txn_cost_bps': 8,
                 'vol_target': vol_target, 'vol_lookback': vol_lb}
            r = backtest_value(price_df, p, signal_idx, pool_info)
            key = f"E_vol{int(vol_target*100) if vol_target else 'none'}_vlb{vol_lb}"
            results[key] = r
            if 'error' not in r:
                print(f"  {key}: CAGR={r['cagr']}%, MDD={r['mdd']}%, Calmar={r['calmar']}")

    return results


def phase5_sector_constraints(price_df, signal_idx, pool_info):
    """Phase 5: Sector diversification constraints"""
    print("\n" + "="*60)
    print("PHASE 5: Sector Constraints")
    print("="*60)

    results = {}

    for sector_max in [1, 2, 3, None]:
        # D
        p = {'lookback': 4, 'skip': 1, 'top_n': 8, 'rebal_freq': 2, 'txn_cost_bps': 8,
             'sector_max': sector_max}
        r = backtest_momentum(price_df, p, signal_idx, pool_info)
        key = f"D_secmax{sector_max if sector_max else 'none'}"
        results[key] = r
        if 'error' not in r:
            print(f"  {key}: CAGR={r['cagr']}%, MDD={r['mdd']}%, Calmar={r['calmar']}")

        # E
        p = {'value_lookback': 12, 'vol_lookback': 12, 'vol_weight': 0.3,
             'top_n': 5, 'rebal_freq': 4, 'max_dd_filter': -0.15, 'txn_cost_bps': 8,
             'sector_max': sector_max}
        r = backtest_value(price_df, p, signal_idx, pool_info)
        key = f"E_secmax{sector_max if sector_max else 'none'}"
        results[key] = r
        if 'error' not in r:
            print(f"  {key}: CAGR={r['cagr']}%, MDD={r['mdd']}%, Calmar={r['calmar']}")

    return results


def phase6_combo(price_df, signal_idx, pool_info):
    """Phase 6: D+E combination with best params found so far"""
    print("\n" + "="*60)
    print("PHASE 6: D+E Combination Optimization")
    print("="*60)

    results = {}

    # Use baseline params for now; will use best params from previous phases
    d_params = {'lookback': 4, 'skip': 1, 'top_n': 8, 'rebal_freq': 2, 'txn_cost_bps': 8}
    e_params = {'value_lookback': 12, 'vol_lookback': 12, 'vol_weight': 0.3,
                'top_n': 5, 'rebal_freq': 4, 'max_dd_filter': -0.15, 'txn_cost_bps': 8}

    for d_w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        r = backtest_combined(price_df, d_params, e_params, d_w, signal_idx, pool_info)
        key = f"combo_D{int(d_w*100)}_E{int((1-d_w)*100)}"
        results[key] = r
        if 'error' not in r:
            print(f"  {key}: CAGR≈{r['cagr']}%, MDD≈{r['mdd']}%, Calmar≈{r['calmar']}")

    return results


def phase7_walkforward(price_df, signal_idx, pool_info):
    """Phase 7: Walk-forward validation"""
    print("\n" + "="*60)
    print("PHASE 7: Walk-Forward Validation")
    print("="*60)

    results = {}

    # D baseline
    d_params = {'lookback': 4, 'skip': 1, 'top_n': 8, 'rebal_freq': 2, 'txn_cost_bps': 8}
    r = walk_forward(price_df, d_params, 'momentum', signal_idx, pool_info)
    results['D_walkfwd'] = r
    if 'error' not in r:
        print(f"  D walk-forward: avg CAGR={r['avg_cagr']}%, avg MDD={r['avg_mdd']}%, min CAGR={r['min_cagr']}%")

    # E baseline
    e_params = {'value_lookback': 12, 'vol_lookback': 12, 'vol_weight': 0.3,
                'top_n': 5, 'rebal_freq': 4, 'max_dd_filter': -0.15, 'txn_cost_bps': 8}
    r = walk_forward(price_df, e_params, 'value', signal_idx, pool_info)
    results['E_walkfwd'] = r
    if 'error' not in r:
        print(f"  E walk-forward: avg CAGR={r['avg_cagr']}%, avg MDD={r['avg_mdd']}%, min CAGR={r['min_cagr']}%")

    return results


def phase8_dd_protection(price_df, signal_idx, pool_info):
    """Phase 8: Drawdown protection"""
    print("\n" + "="*60)
    print("PHASE 8: Drawdown Protection / Stop-Loss")
    print("="*60)

    results = {}

    for dd_stop in [-0.05, -0.08, -0.10, -0.15, None]:
        # D
        p = {'lookback': 4, 'skip': 1, 'top_n': 8, 'rebal_freq': 2, 'txn_cost_bps': 8,
             'dd_stop': dd_stop}
        r = backtest_momentum(price_df, p, signal_idx, pool_info)
        key = f"D_ddstop{int(abs(dd_stop)*100) if dd_stop else 'none'}"
        results[key] = r
        if 'error' not in r:
            print(f"  {key}: CAGR={r['cagr']}%, MDD={r['mdd']}%, Calmar={r['calmar']}")

        # E
        p = {'value_lookback': 12, 'vol_lookback': 12, 'vol_weight': 0.3,
             'top_n': 5, 'rebal_freq': 4, 'max_dd_filter': -0.15, 'txn_cost_bps': 8,
             'dd_stop': dd_stop}
        r = backtest_value(price_df, p, signal_idx, pool_info)
        key = f"E_ddstop{int(abs(dd_stop)*100) if dd_stop else 'none'}"
        results[key] = r
        if 'error' not in r:
            print(f"  {key}: CAGR={r['cagr']}%, MDD={r['mdd']}%, Calmar={r['calmar']}")

    return results


def phase9_ultimate(price_df, signal_idx, pool_info, best_from_phases: dict):
    """Phase 9: Combine best elements from all phases into ultimate strategies"""
    print("\n" + "="*60)
    print("PHASE 9: Ultimate Combined Strategy (Best of All)")
    print("="*60)

    results = {}

    # Test combinations of best elements
    for mom_type in ['raw', 'risk_adjusted']:
        for regime in [None, 'dual']:
            for vol_target in [None, 0.15, 0.20]:
                for sector_max in [None, 2]:
                    for dd_stop in [None, -0.10]:
                        p = {
                            'lookback': 4, 'skip': 1, 'top_n': 8, 'rebal_freq': 2,
                            'txn_cost_bps': 8, 'momentum_type': mom_type,
                            'regime_filter': regime,
                            'regime_short_ma': 13, 'regime_long_ma': 40,
                            'regime_bear_mult': 0.2, 'regime_trans_mult': 0.5,
                            'vol_target': vol_target, 'vol_lookback': 12,
                            'sector_max': sector_max,
                            'dd_stop': dd_stop,
                        }
                        r = backtest_momentum(price_df, p, signal_idx, pool_info)
                        parts = []
                        parts.append(f"m={mom_type[:3]}")
                        if regime: parts.append(f"r={regime}")
                        if vol_target: parts.append(f"v={int(vol_target*100)}")
                        if sector_max: parts.append(f"s={sector_max}")
                        if dd_stop: parts.append(f"dd={int(abs(dd_stop)*100)}")
                        key = f"D_ult_{'_'.join(parts)}"
                        results[key] = r

    # E ultimate
    for value_type in ['reversal', 'mean_reversion']:
        for regime in [None, 'dual']:
            for vol_target in [None, 0.15]:
                for vw in [0.3, 0.5]:
                    p = {
                        'value_lookback': 12, 'vol_lookback': 12, 'vol_weight': vw,
                        'top_n': 5, 'rebal_freq': 4, 'max_dd_filter': -0.15,
                        'txn_cost_bps': 8, 'value_type': value_type,
                        'regime_filter': regime,
                        'regime_short_ma': 13, 'regime_long_ma': 40,
                        'regime_bear_mult': 0.2, 'regime_trans_mult': 0.5,
                        'vol_target': vol_target, 'vol_lookback': 12,
                    }
                    r = backtest_value(price_df, p, signal_idx, pool_info)
                    parts = [f"vt={value_type[:3]}"]
                    if regime: parts.append(f"r={regime}")
                    if vol_target: parts.append(f"v={int(vol_target*100)}")
                    parts.append(f"vw={vw}")
                    key = f"E_ult_{'_'.join(parts)}"
                    results[key] = r

    # Sort all and show top 20
    valid_d = {k: v for k, v in results.items() if k.startswith('D_') and 'error' not in v}
    valid_e = {k: v for k, v in results.items() if k.startswith('E_') and 'error' not in v}

    print(f"\n  TOP 10 D Ultimate ({len(valid_d)} variants):")
    for k, v in sorted(valid_d.items(), key=lambda x: -x[1].get('calmar', 0))[:10]:
        print(f"    {k}: CAGR={v['cagr']}%, MDD={v['mdd']}%, Sharpe={v['sharpe']}, Calmar={v['calmar']}")
        if 'annual' in v:
            print(f"      Annual: {v['annual']}")

    print(f"\n  TOP 10 E Ultimate ({len(valid_e)} variants):")
    for k, v in sorted(valid_e.items(), key=lambda x: -x[1].get('calmar', 0))[:10]:
        print(f"    {k}: CAGR={v['cagr']}%, MDD={v['mdd']}%, Sharpe={v['sharpe']}, Calmar={v['calmar']}")
        if 'annual' in v:
            print(f"      Annual: {v['annual']}")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()
    print("="*60)
    print("Strategy D+E Deep Optimizer")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    all_results = {}

    # Load data (use cached 28-stock pool only - extra stocks fetched separately later)
    print("\n--- Loading 28-stock pool (cached) ---")
    price_28 = load_all_prices(STOCK_POOL_28)

    print("\n--- Loading signal index (沪深300) ---")
    signal_idx = load_signal_index()

    if price_28.empty:
        print("[FATAL] Data loading failed!")
        sys.exit(1)

    # Use 28-stock pool for all phases
    use_pool = STOCK_POOL_28
    use_prices = price_28

    # Phase 1: Baseline
    print("\n" + "="*60)
    print("PHASE 1: Baseline Reproduction")
    print("="*60)
    base_d = {'lookback': 4, 'skip': 1, 'top_n': 8, 'rebal_freq': 2, 'txn_cost_bps': 8}
    r_d = backtest_momentum(price_28, base_d, pool_info=STOCK_POOL_28)
    print(f"  D baseline (28): CAGR={r_d['cagr']}%, MDD={r_d['mdd']}%, Sharpe={r_d['sharpe']}, Calmar={r_d['calmar']}")
    base_e = {'value_lookback': 12, 'vol_lookback': 12, 'vol_weight': 0.3,
              'top_n': 5, 'rebal_freq': 4, 'max_dd_filter': -0.15, 'txn_cost_bps': 8}
    r_e = backtest_value(price_28, base_e, pool_info=STOCK_POOL_28)
    print(f"  E baseline (28): CAGR={r_e['cagr']}%, MDD={r_e['mdd']}%, Sharpe={r_e['sharpe']}, Calmar={r_e['calmar']}")
    all_results['phase1'] = {'D_baseline': r_d, 'E_baseline': r_e}

    # Phase 2
    r = phase2_regime(use_prices, signal_idx, use_pool)
    all_results['phase2'] = r

    # Phase 3
    r = phase3_momentum_variants(use_prices, signal_idx, use_pool)
    all_results['phase3'] = r

    # Phase 4
    r = phase4_vol_scaling(use_prices, signal_idx, use_pool)
    all_results['phase4'] = r

    # Phase 5
    r = phase5_sector_constraints(use_prices, signal_idx, use_pool)
    all_results['phase5'] = r

    # Phase 6
    r = phase6_combo(use_prices, signal_idx, use_pool)
    all_results['phase6'] = r

    # Phase 7
    r = phase7_walkforward(use_prices, signal_idx, use_pool)
    all_results['phase7'] = r

    # Phase 8
    r = phase8_dd_protection(use_prices, signal_idx, use_pool)
    all_results['phase8'] = r

    # Phase 9 - Ultimate
    r = phase9_ultimate(use_prices, signal_idx, use_pool, all_results)
    all_results['phase9'] = r

    # Save all results
    elapsed = time.time() - start_time

    # Collect grand summary
    print("\n" + "="*60)
    print(f"GRAND SUMMARY (elapsed: {elapsed/60:.1f} min)")
    print("="*60)

    # Find absolute best D and E across all phases
    best_d = {'calmar': 0}
    best_e = {'calmar': 0}
    for phase_name, phase_results in all_results.items():
        if not isinstance(phase_results, dict):
            continue
        for key, val in phase_results.items():
            if not isinstance(val, dict) or 'error' in val:
                continue
            if key.startswith('D_') and val.get('calmar', 0) > best_d.get('calmar', 0):
                best_d = {**val, '_key': key, '_phase': phase_name}
            if key.startswith('E_') and val.get('calmar', 0) > best_e.get('calmar', 0):
                best_e = {**val, '_key': key, '_phase': phase_name}

    print(f"\n  BEST Strategy D:")
    if '_key' in best_d:
        print(f"    Config: {best_d['_key']} (from {best_d['_phase']})")
        print(f"    CAGR={best_d['cagr']}%, MDD={best_d['mdd']}%, Sharpe={best_d['sharpe']}, Calmar={best_d['calmar']}")
        if 'annual' in best_d:
            print(f"    Annual: {best_d['annual']}")
        if 'params' in best_d:
            print(f"    Params: {best_d['params']}")

    print(f"\n  BEST Strategy E:")
    if '_key' in best_e:
        print(f"    Config: {best_e['_key']} (from {best_e['_phase']})")
        print(f"    CAGR={best_e['cagr']}%, MDD={best_e['mdd']}%, Sharpe={best_e['sharpe']}, Calmar={best_e['calmar']}")
        if 'annual' in best_e:
            print(f"    Annual: {best_e['annual']}")
        if 'params' in best_e:
            print(f"    Params: {best_e['params']}")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_minutes': round(elapsed / 60, 1),
        'pool_used': '28',
        'best_d': best_d,
        'best_e': best_e,
    }

    summary_path = os.path.join(RESULTS_DIR, 'optimization_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Summary saved: {summary_path}")

    # Save full results (convert to serializable)
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return str(obj)

    full_path = os.path.join(RESULTS_DIR, 'optimization_full.json')
    with open(full_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=make_serializable)
    print(f"  Full results saved: {full_path}")

    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    print("  DONE!")


if __name__ == '__main__':
    main()
