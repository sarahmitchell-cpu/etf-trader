#!/usr/bin/env python3
"""
CSI500 Mid-Cap Research: Quality+Momentum & Sector Rotation
Tests multiple strategy approaches for mid-cap stocks:
  A. Quality+Momentum composite (monthly rebalance)
  B. Sector Rotation (pick top sectors, best stock per sector)
  C. Low Volatility selection
  D. Mean Reversion (contrarian)
  E. Pure Momentum with monthly rebalance (compare vs biweekly)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import akshare as ak
from datetime import datetime
from collections import defaultdict
from itertools import product

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Same 50-stock pool from CSI500 research
CSI500_POOL = {
    'sz002064': {'name': '华峰化学', 'code': '002064', 'sector': '化工'},
    'sh600426': {'name': '华鲁恒升', 'code': '600426', 'sector': '化工'},
    'sz002601': {'name': '龙蟒佰利', 'code': '002601', 'sector': '化工'},
    'sh601100': {'name': '恒立液压', 'code': '601100', 'sector': '机械'},
    'sz300124': {'name': '汇川技术', 'code': '300124', 'sector': '机械'},
    'sz002008': {'name': '大族激光', 'code': '002008', 'sector': '机械'},
    'sz002049': {'name': '紫光国微', 'code': '002049', 'sector': '电子'},
    'sz300661': {'name': '圣邦股份', 'code': '300661', 'sector': '电子'},
    'sh603501': {'name': '韦尔股份', 'code': '603501', 'sector': '电子'},
    'sz002241': {'name': '歌尔股份', 'code': '002241', 'sector': '电子'},
    'sz300122': {'name': '智飞生物', 'code': '300122', 'sector': '医药'},
    'sh603259': {'name': '药明康德', 'code': '603259', 'sector': '医药'},
    'sz000661': {'name': '长春高新', 'code': '000661', 'sector': '医药'},
    'sh603288': {'name': '海天味业', 'code': '603288', 'sector': '食品'},
    'sz002568': {'name': '百润股份', 'code': '002568', 'sector': '食品'},
    'sz300498': {'name': '温氏股份', 'code': '300498', 'sector': '养殖'},
    'sz002459': {'name': '晶澳科技', 'code': '002459', 'sector': '光伏'},
    'sz300014': {'name': '亿纬锂能', 'code': '300014', 'sector': '锂电'},
    'sz002812': {'name': '恩捷股份', 'code': '002812', 'sector': '锂电'},
    'sz002179': {'name': '中航光电', 'code': '002179', 'sector': '军工'},
    'sh600760': {'name': '中航沈飞', 'code': '600760', 'sector': '军工'},
    'sz300033': {'name': '同花顺', 'code': '300033', 'sector': '软件'},
    'sz002236': {'name': '大华股份', 'code': '002236', 'sector': '安防'},
    'sz002460': {'name': '赣锋锂业', 'code': '002460', 'sector': '锂矿'},
    'sh600362': {'name': '江西铜业', 'code': '600362', 'sector': '有色'},
    'sz002466': {'name': '天齐锂业', 'code': '002466', 'sector': '锂矿'},
    'sz002920': {'name': '德赛西威', 'code': '002920', 'sector': '汽车电子'},
    'sh603596': {'name': '伯特利', 'code': '603596', 'sector': '汽车零部件'},
    'sh601689': {'name': '拓普集团', 'code': '601689', 'sector': '汽车零部件'},
    'sz002271': {'name': '东方雨虹', 'code': '002271', 'sector': '建材'},
    'sh601155': {'name': '新城控股', 'code': '601155', 'sector': '地产'},
    'sh600999': {'name': '招商证券', 'code': '600999', 'sector': '券商'},
    'sh601688': {'name': '华泰证券', 'code': '601688', 'sector': '券商'},
    'sz002602': {'name': '世纪华通', 'code': '002602', 'sector': '游戏'},
    'sz300413': {'name': '芒果超媒', 'code': '300413', 'sector': '传媒'},
    'sh600886': {'name': '国投电力', 'code': '600886', 'sector': '电力'},
    'sh601006': {'name': '大秦铁路', 'code': '601006', 'sector': '铁路'},
    'sh600115': {'name': '中国东航', 'code': '600115', 'sector': '航空'},
    'sh603868': {'name': '飞科电器', 'code': '603868', 'sector': '小家电'},
    'sz002572': {'name': '索菲亚', 'code': '002572', 'sector': '家居'},
    'sz300782': {'name': '卓胜微', 'code': '300782', 'sector': '半导体'},
    'sz300628': {'name': '亿联网络', 'code': '300628', 'sector': '通信'},
    'sh600348': {'name': '阳泉煤业', 'code': '600348', 'sector': '煤炭'},
    'sz000932': {'name': '华菱钢铁', 'code': '000932', 'sector': '钢铁'},
    'sz002311': {'name': '海大集团', 'code': '002311', 'sector': '饲料'},
    'sz000876': {'name': '新希望', 'code': '000876', 'sector': '养殖'},
    'sz002475': {'name': '立讯精密', 'code': '002475', 'sector': '消费电子'},
    'sh601127': {'name': '赛力斯', 'code': '601127', 'sector': '新能源车'},
    'sz002385': {'name': '大北农', 'code': '002385', 'sector': '饲料'},
    'sh600309': {'name': '万华化学', 'code': '600309', 'sector': '化工'},
}


def fetch_stock_daily(symbol, name, retries=3):
    """Fetch daily hfq data via akshare, resample to weekly close"""
    cache_path = os.path.join(DATA_DIR, f'csi500_{symbol}_weekly.csv')

    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 3:
            df = pd.read_csv(cache_path, parse_dates=['date'], index_col='date')
            if len(df) > 50:
                print(f"  {name}: cache ({len(df)}w)")
                return df['close']

    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_daily(symbol=symbol, adjust="hfq")
            if df is None or len(df) < 200:
                print(f"  {name}: too few data ({len(df) if df is not None else 0})")
                return None
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            df = df[df.index >= '2021-01-01']
            weekly = df['close'].resample('W-FRI').last().dropna()
            if len(weekly) < 50:
                print(f"  {name}: too few weeks ({len(weekly)})")
                return None
            out = pd.DataFrame({'close': weekly})
            out.index.name = 'date'
            out.to_csv(cache_path)
            print(f"  {name}: OK ({len(weekly)}w)")
            return weekly
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                print(f"  {name}: ERROR {e}")
                return None
    return None


def load_all_data():
    print(f"Loading {len(CSI500_POOL)} stocks via akshare (daily->weekly)...")
    prices = {}
    for symbol, info in CSI500_POOL.items():
        s = fetch_stock_daily(symbol, info['name'])
        if s is not None:
            prices[symbol] = s
        time.sleep(0.5)
    print(f"\nLoaded {len(prices)}/{len(CSI500_POOL)} stocks")
    if len(prices) < 25:
        print("[ERROR] Need 25+")
        return None
    price_df = pd.DataFrame(prices).dropna(how='all')
    price_df = price_df.ffill(limit=2)
    print(f"  Matrix: {price_df.shape[0]}w x {price_df.shape[1]} stocks")
    print(f"  Range: {price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")
    return price_df


# ============================================================
# Helper: compute factor scores at a given time index
# ============================================================

def get_momentum(price_df, idx, lookback):
    """Return momentum (pct return) for each stock over lookback weeks ending at idx"""
    start = idx - lookback
    if start < 0:
        return {}
    mom = {}
    for col in price_df.columns:
        p0 = price_df[col].iloc[start]
        p1 = price_df[col].iloc[idx]
        if pd.notna(p0) and pd.notna(p1) and p0 > 0:
            mom[col] = p1 / p0 - 1
    return mom


def get_volatility(price_df, idx, lookback):
    """Return realized weekly volatility for each stock over lookback weeks"""
    start = max(0, idx - lookback)
    if idx - start < 8:
        return {}
    vol = {}
    for col in price_df.columns:
        chunk = price_df[col].iloc[start:idx+1].dropna()
        if len(chunk) < 8:
            continue
        rets = chunk.pct_change().dropna()
        if len(rets) > 4:
            vol[col] = float(rets.std())
    return vol


def rank_percentile(scores):
    """Convert raw scores to percentile ranks [0, 1]. Higher = better rank."""
    if not scores:
        return {}
    items = sorted(scores.items(), key=lambda x: x[1])
    n = len(items)
    return {k: i / (n - 1) if n > 1 else 0.5 for i, (k, v) in enumerate(items)}


def inverse_rank(scores):
    """Inverse rank: lower raw score = higher rank (for volatility)"""
    if not scores:
        return {}
    items = sorted(scores.items(), key=lambda x: -x[1])  # higher vol first
    n = len(items)
    return {k: i / (n - 1) if n > 1 else 0.5 for i, (k, v) in enumerate(items)}


# ============================================================
# Strategy A: Quality (low vol) + Momentum composite
# ============================================================

def strategy_quality_momentum(price_df, mom_lookback=12, vol_lookback=24,
                               skip=1, top_n=8, rebal_freq=4,
                               mom_weight=0.5, vol_weight=0.5,
                               sector_max=None, txn_bps=8, label=""):
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    warmup = max(mom_lookback, vol_lookback) + skip + 2

    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        mom_idx = i - skip
        mom = get_momentum(price_df, mom_idx, mom_lookback)
        vol = get_volatility(price_df, mom_idx, vol_lookback)

        # Intersection of stocks with both factors
        common = set(mom.keys()) & set(vol.keys())
        if len(common) < 10:
            i += 1
            continue

        mom_rank = rank_percentile({k: mom[k] for k in common})
        vol_rank = inverse_rank({k: vol[k] for k in common})  # low vol = high rank

        composite = {}
        for s in common:
            composite[s] = mom_weight * mom_rank[s] + vol_weight * vol_rank[s]

        # Sort by composite score, apply sector constraint
        ranked = sorted(composite.items(), key=lambda x: -x[1])
        if sector_max:
            selected = []
            sector_count = defaultdict(int)
            for s, _ in ranked:
                if len(selected) >= top_n:
                    break
                sec = CSI500_POOL.get(s, {}).get('sector', '?')
                if sector_count[sec] < sector_max:
                    selected.append(s)
                    sector_count[sec] += 1
        else:
            selected = [s for s, _ in ranked[:top_n]]

        if not selected:
            i += 1
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

    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'mom_lookback': mom_lookback, 'vol_lookback': vol_lookback,
                       'skip': skip, 'top_n': top_n, 'rebal_freq': rebal_freq,
                       'mom_weight': mom_weight, 'vol_weight': vol_weight,
                       'sector_max': sector_max})


# ============================================================
# Strategy B: Sector Rotation
# ============================================================

def strategy_sector_rotation(price_df, mom_lookback=12, skip=1,
                              top_sectors=5, stocks_per_sector=1,
                              rebal_freq=4, txn_bps=8, label=""):
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    warmup = mom_lookback + skip + 2

    # Build sector mapping
    sector_stocks = defaultdict(list)
    for s in price_df.columns:
        sec = CSI500_POOL.get(s, {}).get('sector', '?')
        sector_stocks[sec].append(s)

    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        mom_idx = i - skip
        mom = get_momentum(price_df, mom_idx, mom_lookback)

        # Calculate average sector momentum
        sector_mom = {}
        for sec, stocks in sector_stocks.items():
            sec_moms = [mom[s] for s in stocks if s in mom]
            if sec_moms:
                sector_mom[sec] = np.mean(sec_moms)

        if len(sector_mom) < 3:
            i += 1
            continue

        # Pick top sectors
        ranked_sectors = sorted(sector_mom.items(), key=lambda x: -x[1])[:top_sectors]

        # Within each top sector, pick best stock(s)
        selected = []
        for sec, _ in ranked_sectors:
            stocks = sector_stocks[sec]
            stock_moms = [(s, mom[s]) for s in stocks if s in mom]
            stock_moms.sort(key=lambda x: -x[1])
            for s, _ in stock_moms[:stocks_per_sector]:
                selected.append(s)

        if not selected:
            i += 1
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

    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'mom_lookback': mom_lookback, 'skip': skip,
                       'top_sectors': top_sectors, 'stocks_per_sector': stocks_per_sector,
                       'rebal_freq': rebal_freq})


# ============================================================
# Strategy C: Pure Low Volatility
# ============================================================

def strategy_low_vol(price_df, vol_lookback=24, top_n=10,
                      rebal_freq=4, sector_max=None, txn_bps=8, label=""):
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    warmup = vol_lookback + 2

    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        vol = get_volatility(price_df, i, vol_lookback)
        if len(vol) < 10:
            i += 1
            continue

        # Sort by lowest volatility
        ranked = sorted(vol.items(), key=lambda x: x[1])

        if sector_max:
            selected = []
            sector_count = defaultdict(int)
            for s, _ in ranked:
                if len(selected) >= top_n:
                    break
                sec = CSI500_POOL.get(s, {}).get('sector', '?')
                if sector_count[sec] < sector_max:
                    selected.append(s)
                    sector_count[sec] += 1
        else:
            selected = [s for s, _ in ranked[:top_n]]

        if not selected:
            i += 1
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

    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'vol_lookback': vol_lookback, 'top_n': top_n,
                       'rebal_freq': rebal_freq, 'sector_max': sector_max})


# ============================================================
# Strategy D: Mean Reversion (buy losers)
# ============================================================

def strategy_mean_reversion(price_df, lookback=4, skip=0, top_n=8,
                              rebal_freq=2, sector_max=None, txn_bps=8, label=""):
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    warmup = lookback + skip + 2

    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        mom_idx = i - skip
        mom = get_momentum(price_df, mom_idx, lookback)
        if len(mom) < 10:
            i += 1
            continue

        # Sort by LOWEST momentum (buy losers)
        ranked = sorted(mom.items(), key=lambda x: x[1])

        if sector_max:
            selected = []
            sector_count = defaultdict(int)
            for s, _ in ranked:
                if len(selected) >= top_n:
                    break
                sec = CSI500_POOL.get(s, {}).get('sector', '?')
                if sector_count[sec] < sector_max:
                    selected.append(s)
                    sector_count[sec] += 1
        else:
            selected = [s for s, _ in ranked[:top_n]]

        if not selected:
            i += 1
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

    return calc_stats(nav, dates, weekly_rets, total_txn, label,
                      {'lookback': lookback, 'skip': skip, 'top_n': top_n,
                       'rebal_freq': rebal_freq, 'sector_max': sector_max})


# ============================================================
# Strategy E: Equal Weight Benchmark
# ============================================================

def strategy_equal_weight(price_df, label="EqualWeight"):
    """Buy and hold all stocks equally"""
    returns = price_df.pct_change()
    warmup = 2
    nav = [1.0]
    dates = []
    weekly_rets = []

    for i in range(warmup, len(price_df)):
        rets = [float(returns[s].iloc[i]) for s in price_df.columns if not pd.isna(returns[s].iloc[i])]
        port_ret = np.mean(rets) if rets else 0.0
        nav.append(nav[-1] * (1 + port_ret))
        dates.append(price_df.index[i])
        weekly_rets.append(port_ret)

    return calc_stats(nav, dates, weekly_rets, 0.0, label, {'strategy': 'equal_weight'})


# ============================================================
# Stats calculation
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


def main():
    print("=" * 70)
    print("CSI500 Mid-Cap Research: Quality+Momentum & Sector Rotation")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    price_df = load_all_data()
    if price_df is None:
        sys.exit(1)

    all_results = {}

    # ──────────── Benchmark: Equal Weight ────────────
    print("\n" + "=" * 70)
    print("BENCHMARK: Equal Weight Hold All 50 Stocks")
    print("=" * 70)
    ew = strategy_equal_weight(price_df)
    if ew:
        print(f"  CAGR={ew['cagr_pct']}% MDD={ew['mdd_pct']}% Sharpe={ew['sharpe']} Calmar={ew['calmar']}")
        print(f"  Annual: {ew['annual_returns']}")
        all_results['equal_weight'] = ew

    # ──────────── Strategy A: Quality + Momentum ────────────
    print("\n" + "=" * 70)
    print("STRATEGY A: Quality (Low Vol) + Momentum Composite")
    print("=" * 70)
    a_results = []
    configs_a = list(product(
        [8, 12, 20],       # mom_lookback
        [12, 24],          # vol_lookback
        [0, 1],            # skip
        [8, 10, 15],       # top_n
        [4],               # rebal_freq (monthly)
        [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)],  # (mom_w, vol_w)
        [2, None],         # sector_max
    ))
    print(f"  {len(configs_a)} parameter combinations")
    for count, (ml, vl, sk, tn, rf, (mw, vw), sm) in enumerate(configs_a, 1):
        sm_str = f"sec{sm}" if sm else "nosec"
        label = f"QM_m{ml}_v{vl}_sk{sk}_T{tn}_mw{int(mw*10)}vw{int(vw*10)}_{sm_str}"
        print(f"  [{count}/{len(configs_a)}] {label}...", end=" ", flush=True)
        r = strategy_quality_momentum(price_df, mom_lookback=ml, vol_lookback=vl,
                                       skip=sk, top_n=tn, rebal_freq=rf,
                                       mom_weight=mw, vol_weight=vw,
                                       sector_max=sm, label=label)
        if r:
            a_results.append(r)
            print(f"CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Sh={r['sharpe']} Cal={r['calmar']}")
        else:
            print("SKIP")
    all_results['quality_momentum'] = a_results

    # ──────────── Strategy B: Sector Rotation ────────────
    print("\n" + "=" * 70)
    print("STRATEGY B: Sector Rotation")
    print("=" * 70)
    b_results = []
    configs_b = list(product(
        [8, 12, 20],       # mom_lookback
        [0, 1],            # skip
        [3, 5, 7],         # top_sectors
        [1, 2],            # stocks_per_sector
        [4],               # rebal_freq (monthly)
    ))
    print(f"  {len(configs_b)} parameter combinations")
    for count, (ml, sk, ts, sps, rf) in enumerate(configs_b, 1):
        label = f"SR_m{ml}_sk{sk}_top{ts}_sps{sps}"
        print(f"  [{count}/{len(configs_b)}] {label}...", end=" ", flush=True)
        r = strategy_sector_rotation(price_df, mom_lookback=ml, skip=sk,
                                      top_sectors=ts, stocks_per_sector=sps,
                                      rebal_freq=rf, label=label)
        if r:
            b_results.append(r)
            print(f"CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Sh={r['sharpe']} Cal={r['calmar']}")
        else:
            print("SKIP")
    all_results['sector_rotation'] = b_results

    # ──────────── Strategy C: Low Volatility ────────────
    print("\n" + "=" * 70)
    print("STRATEGY C: Pure Low Volatility")
    print("=" * 70)
    c_results = []
    configs_c = list(product(
        [12, 24, 36],      # vol_lookback
        [8, 10, 15],       # top_n
        [4],               # rebal_freq
        [2, None],         # sector_max
    ))
    print(f"  {len(configs_c)} parameter combinations")
    for count, (vl, tn, rf, sm) in enumerate(configs_c, 1):
        sm_str = f"sec{sm}" if sm else "nosec"
        label = f"LV_v{vl}_T{tn}_{sm_str}"
        print(f"  [{count}/{len(configs_c)}] {label}...", end=" ", flush=True)
        r = strategy_low_vol(price_df, vol_lookback=vl, top_n=tn,
                              rebal_freq=rf, sector_max=sm, label=label)
        if r:
            c_results.append(r)
            print(f"CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Sh={r['sharpe']} Cal={r['calmar']}")
        else:
            print("SKIP")
    all_results['low_volatility'] = c_results

    # ──────────── Strategy D: Mean Reversion ────────────
    print("\n" + "=" * 70)
    print("STRATEGY D: Mean Reversion (Buy Losers)")
    print("=" * 70)
    d_results = []
    configs_d = list(product(
        [4, 8, 12],        # lookback
        [0, 1],            # skip
        [8, 10],           # top_n
        [2, 4],            # rebal_freq
        [2, None],         # sector_max
    ))
    print(f"  {len(configs_d)} parameter combinations")
    for count, (lb, sk, tn, rf, sm) in enumerate(configs_d, 1):
        sm_str = f"sec{sm}" if sm else "nosec"
        label = f"MR_lb{lb}_sk{sk}_T{tn}_rf{rf}_{sm_str}"
        print(f"  [{count}/{len(configs_d)}] {label}...", end=" ", flush=True)
        r = strategy_mean_reversion(price_df, lookback=lb, skip=sk, top_n=tn,
                                     rebal_freq=rf, sector_max=sm, label=label)
        if r:
            d_results.append(r)
            print(f"CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Sh={r['sharpe']} Cal={r['calmar']}")
        else:
            print("SKIP")
    all_results['mean_reversion'] = d_results

    # ──────────── Summary ────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Best of Each Strategy Category")
    print("=" * 70)

    # Collect all results
    all_flat = []
    for cat, results in all_results.items():
        if isinstance(results, list):
            all_flat.extend(results)
        elif results:
            all_flat.append(results)

    if not all_flat:
        print("No results!")
        sys.exit(1)

    # Top 10 overall by Calmar
    print("\nTOP 10 Overall by Calmar:")
    sc = sorted(all_flat, key=lambda x: -x['calmar'])
    for i, r in enumerate(sc[:10]):
        print(f"  #{i+1} {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% "
              f"Sh={r['sharpe']} Cal={r['calmar']} Win={r['win_rate_pct']}%")
        print(f"       {r['annual_returns']}")

    print("\nTOP 10 Overall by CAGR:")
    sg = sorted(all_flat, key=lambda x: -x['cagr_pct'])
    for i, r in enumerate(sg[:10]):
        print(f"  #{i+1} {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% "
              f"Sh={r['sharpe']} Cal={r['calmar']} Win={r['win_rate_pct']}%")
        print(f"       {r['annual_returns']}")

    print("\nTOP 10 Overall by Sharpe:")
    ss = sorted(all_flat, key=lambda x: -x['sharpe'])
    for i, r in enumerate(ss[:10]):
        print(f"  #{i+1} {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% "
              f"Sh={r['sharpe']} Cal={r['calmar']} Win={r['win_rate_pct']}%")
        print(f"       {r['annual_returns']}")

    # Best per category
    print("\nBEST per Category (by Calmar):")
    for cat in ['quality_momentum', 'sector_rotation', 'low_volatility', 'mean_reversion']:
        results = all_results.get(cat, [])
        if isinstance(results, list) and results:
            best = max(results, key=lambda x: x['calmar'])
            print(f"  {cat}: {best['label']}")
            print(f"    CAGR={best['cagr_pct']}% MDD={best['mdd_pct']}% "
                  f"Sh={best['sharpe']} Cal={best['calmar']}")
            print(f"    {best['annual_returns']}")

    # Save results
    out_path = os.path.join(DATA_DIR, 'csi500_quality_momentum_research.json')
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pool_size': len(CSI500_POOL),
        'data_stocks': price_df.shape[1],
        'data_weeks': price_df.shape[0],
        'equal_weight': ew,
        'top10_calmar': sc[:10],
        'top10_cagr': sg[:10],
        'top10_sharpe': ss[:10],
        'best_per_category': {},
    }
    for cat in ['quality_momentum', 'sector_rotation', 'low_volatility', 'mean_reversion']:
        results = all_results.get(cat, [])
        if isinstance(results, list) and results:
            best_cal = max(results, key=lambda x: x['calmar'])
            best_cagr = max(results, key=lambda x: x['cagr_pct'])
            best_sharpe = max(results, key=lambda x: x['sharpe'])
            output['best_per_category'][cat] = {
                'best_calmar': best_cal,
                'best_cagr': best_cagr,
                'best_sharpe': best_sharpe,
                'total_combos': len(results),
            }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
