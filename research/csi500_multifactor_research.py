#!/usr/bin/env python3
"""
CSI500 Multi-Factor Research v2 (No Survivorship Bias)
- Reuses cached historical constituents + industries + prices from Phase 4
- Tests: Low Volatility, Value (PE/PB), Momentum, Multi-factor combos
- ROE removed (too many API calls; replaced by earnings yield 1/PE and PB)
- Incremental fundamental data caching
"""

import baostock as bs
import pandas as pd
import numpy as np
import json
import os
import sys
import time
from datetime import datetime
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'baostock_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

REBALANCE_DATES = [
    '2020-12-14', '2021-06-15', '2021-12-13', '2022-06-13',
    '2022-12-12', '2023-06-12', '2023-12-11', '2024-06-17',
    '2024-12-16', '2025-06-16', '2025-12-15',
]


def load_cached_data():
    """Load cached constituents, industries, and prices from Phase 4"""
    const_path = os.path.join(CACHE_DIR, 'csi500_constituents_history.json')
    with open(const_path) as f:
        constituent_data = json.load(f)
    all_stocks = constituent_data['all_unique_stocks']
    print(f"Loaded constituents: {len(all_stocks)} unique stocks")

    ind_path = os.path.join(CACHE_DIR, 'csi500_industries.json')
    with open(ind_path) as f:
        industries = json.load(f)
    print(f"Loaded industries: {len(industries)} stocks")

    price_path = os.path.join(CACHE_DIR, 'csi500_all_weekly_prices.pkl')
    price_df = pd.read_pickle(price_path)
    print(f"Loaded prices: {price_df.shape[0]} weeks x {price_df.shape[1]} stocks")

    return constituent_data, industries, price_df, all_stocks


def fetch_fundamental_data(stock_list):
    """Fetch PE, PB from weekly k-line data with incremental caching"""
    cache_path = os.path.join(CACHE_DIR, 'csi500_weekly_fundamentals.pkl')

    # Try loading existing cache
    pe_data = {}
    pb_data = {}
    if os.path.exists(cache_path):
        try:
            data = pd.read_pickle(cache_path)
            pe_data = data.get('pe', {})
            pb_data = data.get('pb', {})
            if len(pe_data) > 100:
                print(f"Loaded cached fundamentals: PE for {len(pe_data)} stocks, PB for {len(pb_data)} stocks")
                # If we have >80% coverage, skip fetching
                if len(pe_data) >= len(stock_list) * 0.8:
                    return data
        except Exception as e:
            print(f"Cache load error: {e}, refetching...")
            pe_data = {}
            pb_data = {}

    # Figure out which stocks still need fetching
    need_fetch = [s for s in stock_list if s not in pe_data]
    if not need_fetch:
        return {'pe': pe_data, 'pb': pb_data}

    print(f"Need to fetch fundamentals for {len(need_fetch)} stocks (have {len(pe_data)} cached)")

    lg = bs.login()
    total = len(need_fetch)
    failed = []
    fetched = 0

    for i, stock in enumerate(need_fetch):
        if i % 100 == 0:
            print(f"  Fetching fundamentals: {i}/{total}... (total PE: {len(pe_data)})")
            sys.stdout.flush()

        try:
            # Must use daily frequency for PE/PB fields (weekly doesn't support them)
            rs = bs.query_history_k_data_plus(
                stock,
                'date,close,peTTM,pbMRQ',
                start_date='2020-09-01',
                end_date='2026-03-31',
                frequency='d',
                adjustflag='1'
            )

            dates_list = []
            pe_list = []
            pb_list = []
            while rs.next():
                row = rs.get_row_data()
                try:
                    d = row[0]
                    pe = float(row[2]) if row[2] and row[2] != '' else np.nan
                    pb = float(row[3]) if row[3] and row[3] != '' else np.nan
                    dates_list.append(d)
                    pe_list.append(pe)
                    pb_list.append(pb)
                except (ValueError, IndexError):
                    continue

            if len(dates_list) > 50:
                idx = pd.to_datetime(dates_list)
                # Resample daily to weekly (Friday close) to match price data
                pe_series = pd.Series(pe_list, index=idx).resample('W-FRI').last().dropna()
                pb_series = pd.Series(pb_list, index=idx).resample('W-FRI').last().dropna()
                if len(pe_series) > 20:
                    pe_data[stock] = pe_series
                    pb_data[stock] = pb_series
                    fetched += 1
                else:
                    failed.append(stock)
            else:
                failed.append(stock)
        except Exception as e:
            failed.append(stock)
            if i < 5:
                print(f"    Error fetching {stock}: {e}")

        # Save incrementally every 200 stocks
        if (i + 1) % 200 == 0:
            print(f"  Saving incremental cache ({len(pe_data)} stocks)...")
            pd.to_pickle({'pe': pe_data, 'pb': pb_data}, cache_path)

        # Brief pause every 300 to avoid rate limits
        if (i + 1) % 300 == 0:
            time.sleep(1)

    bs.logout()

    print(f"Fetched fundamentals for {fetched} new stocks, {len(failed)} failed")
    print(f"Total PE: {len(pe_data)}, PB: {len(pb_data)}")

    data = {'pe': pe_data, 'pb': pb_data}
    pd.to_pickle(data, cache_path)
    return data


def get_active_constituents(date, constituent_data):
    """Get which stocks were in the CSI 500 at a given date"""
    rebal_dates = constituent_data['dates']
    constituents = constituent_data['constituents']
    active_date = None
    for d in rebal_dates:
        if d <= date.strftime('%Y-%m-%d'):
            active_date = d
        else:
            break
    if active_date is None:
        active_date = rebal_dates[0]
    return set(constituents[active_date])


def strategy_factor(price_df, constituent_data, industries, fund_data,
                    factor_type='low_vol', top_n=10, rebal_freq=2,
                    vol_lookback=12, sector_cap=0, txn_bps=8):
    """
    Factor-based stock selection strategy
    factor_type: 'low_vol', 'value_pe', 'value_pb', 'earnings_yield',
                 'momentum_12m', 'momentum_6m', 'reversal_4w'
    """
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change(fill_method=None)
    warmup = max(vol_lookback + 4, 14)

    pe_dict = fund_data.get('pe', {}) if fund_data else {}
    pb_dict = fund_data.get('pb', {}) if fund_data else {}

    nav = [1.0]
    dates = [price_df.index[warmup - 1]]
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        current_date = price_df.index[i]
        active = get_active_constituents(current_date, constituent_data)
        available = active & set(price_df.columns)

        if len(available) < 50:
            i += 1
            continue

        scores = {}

        for stock in available:
            prices_slice = price_df[stock].iloc[max(0, i - vol_lookback):i + 1]
            valid_prices = prices_slice.dropna()
            if len(valid_prices) < vol_lookback // 2:
                continue

            if factor_type == 'low_vol':
                rets = valid_prices.pct_change().dropna()
                if len(rets) >= vol_lookback // 2:
                    vol = float(rets.std()) * np.sqrt(52)
                    if vol > 0:
                        scores[stock] = -vol

            elif factor_type == 'value_pe':
                if stock in pe_dict:
                    pe_series = pe_dict[stock]
                    mask = pe_series.index <= current_date
                    if mask.any():
                        pe = float(pe_series[mask].iloc[-1])
                        if not np.isnan(pe) and 0 < pe < 500:
                            scores[stock] = -pe  # lower PE = better

            elif factor_type == 'value_pb':
                if stock in pb_dict:
                    pb_series = pb_dict[stock]
                    mask = pb_series.index <= current_date
                    if mask.any():
                        pb = float(pb_series[mask].iloc[-1])
                        if not np.isnan(pb) and 0 < pb < 50:
                            scores[stock] = -pb

            elif factor_type == 'earnings_yield':
                # 1/PE as quality proxy (higher = cheaper + more profitable)
                if stock in pe_dict:
                    pe_series = pe_dict[stock]
                    mask = pe_series.index <= current_date
                    if mask.any():
                        pe = float(pe_series[mask].iloc[-1])
                        if not np.isnan(pe) and pe > 1:
                            scores[stock] = 1.0 / pe

            elif factor_type == 'momentum_12m':
                start_idx = max(0, i - 52)
                p0 = price_df[stock].iloc[start_idx]
                p1 = price_df[stock].iloc[i]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    scores[stock] = p1 / p0 - 1

            elif factor_type == 'momentum_6m':
                start_idx = max(0, i - 26)
                p0 = price_df[stock].iloc[start_idx]
                p1 = price_df[stock].iloc[i]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    scores[stock] = p1 / p0 - 1

            elif factor_type == 'reversal_4w':
                # Short-term reversal: buy recent losers
                start_idx = max(0, i - 4)
                p0 = price_df[stock].iloc[start_idx]
                p1 = price_df[stock].iloc[i]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    scores[stock] = -(p1 / p0 - 1)  # negative = buy losers

            elif factor_type == 'low_vol_value':
                # Combined: low vol + low PE
                rets = valid_prices.pct_change().dropna()
                vol_ok = False
                pe_ok = False
                vol_z = 0
                pe_z = 0
                if len(rets) >= vol_lookback // 2:
                    vol = float(rets.std()) * np.sqrt(52)
                    if vol > 0:
                        vol_z = -vol
                        vol_ok = True
                if stock in pe_dict:
                    pe_series = pe_dict[stock]
                    mask = pe_series.index <= current_date
                    if mask.any():
                        pe = float(pe_series[mask].iloc[-1])
                        if not np.isnan(pe) and 0 < pe < 500:
                            pe_z = -pe / 100
                            pe_ok = True
                if vol_ok and pe_ok:
                    scores[stock] = vol_z + pe_z

        if len(scores) < top_n:
            i += 1
            continue

        # Rank and select with optional sector cap
        if sector_cap > 0:
            sector_stocks = defaultdict(list)
            for s in scores:
                ind = industries.get(s, {}).get('industry', 'Other')
                sector_stocks[ind].append((s, scores[s]))
            for sec in sector_stocks:
                sector_stocks[sec].sort(key=lambda x: -x[1])

            selected = []
            sector_order = sorted(sector_stocks.keys(),
                                  key=lambda s: -sector_stocks[s][0][1] if sector_stocks[s] else -999)
            sector_idx = {s: 0 for s in sector_order}
            while len(selected) < top_n:
                added = False
                for sec in sector_order:
                    if sector_idx[sec] < min(sector_cap, len(sector_stocks[sec])):
                        selected.append(sector_stocks[sec][sector_idx[sec]][0])
                        sector_idx[sec] += 1
                        added = True
                        if len(selected) >= top_n:
                            break
                if not added:
                    break
        else:
            ranked = sorted(scores.items(), key=lambda x: -x[1])
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
            rets = []
            for s in selected:
                if s in returns.columns and j < len(returns):
                    r = returns[s].iloc[j]
                    if not pd.isna(r):
                        rets.append(float(r))
            port_ret = np.mean(rets) if rets else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    return pd.Series(nav, index=dates), weekly_rets, total_txn


def strategy_composite(price_df, constituent_data, industries, fund_data,
                        weights=None, top_n=10, rebal_freq=2, vol_lookback=12,
                        sector_cap=0, txn_bps=8):
    """
    Composite z-score factor strategy
    weights: dict of factor_name -> weight
    Supported factors: low_vol, value_pe, value_pb, earnings_yield, momentum_12m, momentum_6m, reversal_4w
    """
    if weights is None:
        weights = {'low_vol': 0.5, 'value_pe': 0.5}

    txn_cost = txn_bps / 10000
    returns = price_df.pct_change(fill_method=None)
    warmup = max(vol_lookback + 4, 14)

    pe_dict = fund_data.get('pe', {}) if fund_data else {}
    pb_dict = fund_data.get('pb', {}) if fund_data else {}

    nav = [1.0]
    dates = [price_df.index[warmup - 1]]
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        current_date = price_df.index[i]
        active = get_active_constituents(current_date, constituent_data)
        available = active & set(price_df.columns)

        if len(available) < 50:
            i += 1
            continue

        raw_factors = defaultdict(dict)

        for stock in available:
            prices_slice = price_df[stock].iloc[max(0, i - max(vol_lookback, 52)):i + 1]
            valid_prices = prices_slice.dropna()
            if len(valid_prices) < 8:
                continue

            if 'low_vol' in weights:
                rets_slice = price_df[stock].iloc[max(0, i - vol_lookback):i + 1].dropna().pct_change().dropna()
                if len(rets_slice) >= vol_lookback // 2:
                    vol = float(rets_slice.std()) * np.sqrt(52)
                    if vol > 0:
                        raw_factors['low_vol'][stock] = -vol

            if 'value_pe' in weights and stock in pe_dict:
                pe_series = pe_dict[stock]
                mask = pe_series.index <= current_date
                if mask.any():
                    pe = float(pe_series[mask].iloc[-1])
                    if not np.isnan(pe) and 0 < pe < 500:
                        raw_factors['value_pe'][stock] = -pe

            if 'value_pb' in weights and stock in pb_dict:
                pb_series = pb_dict[stock]
                mask = pb_series.index <= current_date
                if mask.any():
                    pb = float(pb_series[mask].iloc[-1])
                    if not np.isnan(pb) and 0 < pb < 50:
                        raw_factors['value_pb'][stock] = -pb

            if 'earnings_yield' in weights and stock in pe_dict:
                pe_series = pe_dict[stock]
                mask = pe_series.index <= current_date
                if mask.any():
                    pe = float(pe_series[mask].iloc[-1])
                    if not np.isnan(pe) and pe > 1:
                        raw_factors['earnings_yield'][stock] = 1.0 / pe

            if 'momentum_12m' in weights:
                start_idx = max(0, i - 52)
                p0 = price_df[stock].iloc[start_idx]
                p1 = price_df[stock].iloc[i]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    raw_factors['momentum_12m'][stock] = p1 / p0 - 1

            if 'momentum_6m' in weights:
                start_idx = max(0, i - 26)
                p0 = price_df[stock].iloc[start_idx]
                p1 = price_df[stock].iloc[i]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    raw_factors['momentum_6m'][stock] = p1 / p0 - 1

            if 'reversal_4w' in weights:
                start_idx = max(0, i - 4)
                p0 = price_df[stock].iloc[start_idx]
                p1 = price_df[stock].iloc[i]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    raw_factors['reversal_4w'][stock] = -(p1 / p0 - 1)

        # Z-score normalization per factor
        z_scores = defaultdict(dict)
        for factor_name, values in raw_factors.items():
            if len(values) < 20:
                continue
            arr = np.array(list(values.values()))
            mean = np.mean(arr)
            std = np.std(arr)
            if std > 0:
                for stock, val in values.items():
                    z_scores[factor_name][stock] = (val - mean) / std

        # Composite score
        all_stocks_scored = set()
        for factor_name in z_scores:
            all_stocks_scored |= set(z_scores[factor_name].keys())

        composite = {}
        min_weight_coverage = sum(weights.values()) * 0.4  # Need at least 40% of weight covered
        for stock in all_stocks_scored:
            total_weight = 0
            total_score = 0
            for factor_name, w in weights.items():
                if stock in z_scores.get(factor_name, {}):
                    total_score += w * z_scores[factor_name][stock]
                    total_weight += w
            if total_weight >= min_weight_coverage:
                composite[stock] = total_score / total_weight

        if len(composite) < top_n:
            i += 1
            continue

        # Sector diversification
        if sector_cap > 0:
            sector_stocks = defaultdict(list)
            for s in composite:
                ind = industries.get(s, {}).get('industry', 'Other')
                sector_stocks[ind].append((s, composite[s]))
            for sec in sector_stocks:
                sector_stocks[sec].sort(key=lambda x: -x[1])

            selected = []
            sector_order = sorted(sector_stocks.keys(),
                                  key=lambda s: -sector_stocks[s][0][1] if sector_stocks[s] else -999)
            sector_idx = {s: 0 for s in sector_order}
            while len(selected) < top_n:
                added = False
                for sec in sector_order:
                    if sector_idx[sec] < min(sector_cap, len(sector_stocks[sec])):
                        selected.append(sector_stocks[sec][sector_idx[sec]][0])
                        sector_idx[sec] += 1
                        added = True
                        if len(selected) >= top_n:
                            break
                if not added:
                    break
        else:
            ranked = sorted(composite.items(), key=lambda x: -x[1])
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
            rets = []
            for s in selected:
                if s in returns.columns and j < len(returns):
                    r = returns[s].iloc[j]
                    if not pd.isna(r):
                        rets.append(float(r))
            port_ret = np.mean(rets) if rets else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    return pd.Series(nav, index=dates), weekly_rets, total_txn


def calc_stats(nav_series, weekly_rets, total_txn, label=""):
    """Calculate strategy statistics"""
    nav = nav_series.values
    if len(nav) < 10:
        return None
    total_ret = nav[-1] / nav[0] - 1
    n_weeks = len(nav) - 1
    years = n_weeks / 52.0

    cagr = (nav[-1] / nav[0]) ** (1 / years) - 1 if years > 0.5 else 0

    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    mdd = float(np.min(dd))

    wr = np.array(weekly_rets)
    if len(wr) > 10 and np.std(wr) > 0:
        sharpe = float(np.mean(wr) / np.std(wr) * np.sqrt(52))
    else:
        sharpe = 0.0

    calmar = cagr / abs(mdd) if mdd != 0 else 0
    win_rate = float(np.mean(np.array(weekly_rets) > 0) * 100)

    annual = {}
    for year in range(2021, 2027):
        mask = [d.year == year for d in nav_series.index]
        if sum(mask) > 4:
            year_nav = nav[mask]
            annual[str(year)] = round((year_nav[-1] / year_nav[0] - 1) * 100, 1)

    return {
        'label': label,
        'cagr_pct': round(cagr * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round(win_rate, 1),
        'total_return_pct': round(total_ret * 100, 1),
        'years': round(years, 1),
        'total_txn_pct': round(total_txn * 100, 2),
        'annual_returns': annual,
        'period': f"{nav_series.index[0].strftime('%Y-%m-%d')} ~ {nav_series.index[-1].strftime('%Y-%m-%d')}",
    }


def rolling_analysis(nav_series, window_weeks=52):
    """Rolling window return analysis"""
    nav = nav_series.values
    dates = nav_series.index
    results = []
    for start in range(len(nav) - window_weeks):
        end = start + window_weeks
        ret = nav[end] / nav[start] - 1
        results.append({
            'start': dates[start].strftime('%Y-%m-%d'),
            'end': dates[end].strftime('%Y-%m-%d'),
            'return_pct': round(ret * 100, 2)
        })
    rets_arr = np.array([r['return_pct'] for r in results])
    return {
        'windows': len(results),
        'win_rate': round(float(np.mean(rets_arr > 0) * 100), 1),
        'mean': round(float(np.mean(rets_arr)), 2),
        'median': round(float(np.median(rets_arr)), 2),
        'min': round(float(np.min(rets_arr)), 2),
        'max': round(float(np.max(rets_arr)), 2),
        'p10': round(float(np.percentile(rets_arr, 10)), 2),
        'p25': round(float(np.percentile(rets_arr, 25)), 2),
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("CSI500 Multi-Factor Research v2 (No Survivorship Bias)")
    print("=" * 60)

    # Load cached data
    print("\n[1/2] Loading cached data...")
    constituent_data, industries, price_df, all_stocks = load_cached_data()

    # Fetch fundamental data (PE/PB only, no ROE)
    print("\n[2/2] Fetching fundamental data (PE/PB)...")
    fund_data = fetch_fundamental_data(all_stocks)
    pe_count = len(fund_data.get('pe', {}))
    pb_count = len(fund_data.get('pb', {}))
    print(f"  PE data: {pe_count} stocks, PB data: {pb_count} stocks")

    # Run strategies
    print("\nRunning factor strategies...")

    configs = []

    # === Single factors ===
    # Low Volatility
    for top_n in [8, 10, 15, 20, 30]:
        for rebal in [2, 4]:
            configs.append({
                'type': 'single', 'factor_type': 'low_vol',
                'top_n': top_n, 'rebal_freq': rebal, 'sector_cap': 0,
                'label': f"LowVol_top{top_n}_{rebal}w",
            })

    # Value PE
    for top_n in [8, 10, 15, 20, 30]:
        for rebal in [2, 4]:
            configs.append({
                'type': 'single', 'factor_type': 'value_pe',
                'top_n': top_n, 'rebal_freq': rebal, 'sector_cap': 0,
                'label': f"ValuePE_top{top_n}_{rebal}w",
            })

    # Value PB
    for top_n in [10, 15, 20]:
        for rebal in [2, 4]:
            configs.append({
                'type': 'single', 'factor_type': 'value_pb',
                'top_n': top_n, 'rebal_freq': rebal, 'sector_cap': 0,
                'label': f"ValuePB_top{top_n}_{rebal}w",
            })

    # Earnings Yield (1/PE)
    for top_n in [10, 15, 20]:
        for rebal in [2, 4]:
            configs.append({
                'type': 'single', 'factor_type': 'earnings_yield',
                'top_n': top_n, 'rebal_freq': rebal, 'sector_cap': 0,
                'label': f"EarnYield_top{top_n}_{rebal}w",
            })

    # Momentum 12m / 6m
    for mom_type in ['momentum_12m', 'momentum_6m']:
        for top_n in [10, 15, 20]:
            for rebal in [2, 4]:
                label = f"{'Mom12m' if '12' in mom_type else 'Mom6m'}_top{top_n}_{rebal}w"
                configs.append({
                    'type': 'single', 'factor_type': mom_type,
                    'top_n': top_n, 'rebal_freq': rebal, 'sector_cap': 0,
                    'label': label,
                })

    # Short-term reversal
    for top_n in [10, 15, 20]:
        for rebal in [2, 4]:
            configs.append({
                'type': 'single', 'factor_type': 'reversal_4w',
                'top_n': top_n, 'rebal_freq': rebal, 'sector_cap': 0,
                'label': f"Reversal4w_top{top_n}_{rebal}w",
            })

    # Single factors with sector cap
    for factor in ['low_vol', 'value_pe', 'earnings_yield']:
        for top_n in [15, 20]:
            label_map = {'low_vol': 'LowVol', 'value_pe': 'ValuePE', 'earnings_yield': 'EarnYield'}
            configs.append({
                'type': 'single', 'factor_type': factor,
                'top_n': top_n, 'rebal_freq': 2, 'sector_cap': 3,
                'label': f"{label_map[factor]}_top{top_n}_2w_sec3",
            })

    # === Composite z-score strategies ===
    composite_weights = [
        # Two-factor
        ({'low_vol': 0.5, 'value_pe': 0.5}, 'LV50_PE50'),
        ({'low_vol': 0.5, 'value_pb': 0.5}, 'LV50_PB50'),
        ({'low_vol': 0.5, 'earnings_yield': 0.5}, 'LV50_EY50'),
        ({'value_pe': 0.5, 'reversal_4w': 0.5}, 'PE50_Rev50'),
        ({'low_vol': 0.5, 'reversal_4w': 0.5}, 'LV50_Rev50'),
        ({'low_vol': 0.7, 'value_pe': 0.3}, 'LV70_PE30'),
        ({'low_vol': 0.3, 'value_pe': 0.7}, 'LV30_PE70'),
        # Three-factor
        ({'low_vol': 0.4, 'value_pe': 0.3, 'value_pb': 0.3}, 'LV40_PE30_PB30'),
        ({'low_vol': 0.33, 'value_pe': 0.33, 'reversal_4w': 0.34}, 'LV33_PE33_Rev34'),
        ({'low_vol': 0.4, 'value_pe': 0.3, 'reversal_4w': 0.3}, 'LV40_PE30_Rev30'),
        ({'low_vol': 0.4, 'earnings_yield': 0.3, 'reversal_4w': 0.3}, 'LV40_EY30_Rev30'),
        # Four-factor
        ({'low_vol': 0.3, 'value_pe': 0.25, 'value_pb': 0.25, 'reversal_4w': 0.2}, 'LV30_PE25_PB25_Rev20'),
        ({'low_vol': 0.25, 'value_pe': 0.25, 'earnings_yield': 0.25, 'reversal_4w': 0.25}, 'LV25_PE25_EY25_Rev25'),
        # With momentum (to compare)
        ({'low_vol': 0.4, 'value_pe': 0.3, 'momentum_6m': 0.3}, 'LV40_PE30_Mom6m30'),
        ({'low_vol': 0.4, 'value_pe': 0.3, 'momentum_12m': 0.3}, 'LV40_PE30_Mom12m30'),
    ]

    for weights, name in composite_weights:
        for top_n in [10, 15, 20]:
            for rebal in [2, 4]:
                configs.append({
                    'type': 'composite', 'weights': weights,
                    'top_n': top_n, 'rebal_freq': rebal, 'sector_cap': 0,
                    'label': f"Comp_{name}_top{top_n}_{rebal}w",
                })
        # With sector cap
        for top_n in [15, 20]:
            configs.append({
                'type': 'composite', 'weights': weights,
                'top_n': top_n, 'rebal_freq': 2, 'sector_cap': 3,
                'label': f"Comp_{name}_top{top_n}_2w_sec3",
            })

    print(f"Total configs to test: {len(configs)}")

    results = []
    best_nav = None
    best_label = None
    best_calmar = -999

    for idx, cfg in enumerate(configs):
        label = cfg['label']
        if idx % 20 == 0:
            elapsed = time.time() - t0
            print(f"\n  Progress: {idx}/{len(configs)} ({elapsed:.0f}s elapsed)...")
            sys.stdout.flush()

        try:
            if cfg['type'] == 'single':
                nav, wr, txn = strategy_factor(
                    price_df, constituent_data, industries, fund_data,
                    factor_type=cfg['factor_type'], top_n=cfg['top_n'],
                    rebal_freq=cfg['rebal_freq'], sector_cap=cfg['sector_cap'],
                )
            else:
                nav, wr, txn = strategy_composite(
                    price_df, constituent_data, industries, fund_data,
                    weights=cfg['weights'], top_n=cfg['top_n'],
                    rebal_freq=cfg['rebal_freq'], sector_cap=cfg['sector_cap'],
                )

            if len(nav) < 52:
                continue

            stats = calc_stats(nav, wr, txn, label)
            if stats is None:
                continue

            results.append(stats)

            if stats['calmar'] > best_calmar:
                best_calmar = stats['calmar']
                best_nav = nav
                best_label = label

            if stats['calmar'] > 0.1 or idx % 30 == 0:
                print(f"  {label}: CAGR={stats['cagr_pct']}% MDD={stats['mdd_pct']}% Calmar={stats['calmar']}")

        except Exception as e:
            print(f"  ERROR {label}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Sort by Calmar
    results.sort(key=lambda x: -x['calmar'])

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(results)} strategies tested")
    print(f"{'='*60}")

    print("\nTop 20 by Calmar:")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1}. {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Sharpe={r['sharpe']} Calmar={r['calmar']}")
        print(f"     Annual: {r['annual_returns']}")

    print(f"\nBottom 5:")
    for r in results[-5:]:
        print(f"  {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']}")

    # By factor type summary
    print(f"\n{'='*60}")
    print("FACTOR TYPE SUMMARY (best per type):")
    factor_best = {}
    for r in results:
        ftype = r['label'].split('_')[0]
        if ftype not in factor_best or r['calmar'] > factor_best[ftype]['calmar']:
            factor_best[ftype] = r
    for ftype, r in sorted(factor_best.items(), key=lambda x: -x[1]['calmar']):
        print(f"  {ftype}: {r['label']} CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']}")

    # Rolling analysis on top 3
    rolling_results = {}
    top3 = results[:3] if len(results) >= 3 else results
    for r in top3:
        label = r['label']
        # Re-run to get NAV series
        cfg = None
        for c in configs:
            if c['label'] == label:
                cfg = c
                break
        if cfg:
            try:
                if cfg['type'] == 'single':
                    nav, wr, txn = strategy_factor(
                        price_df, constituent_data, industries, fund_data,
                        factor_type=cfg['factor_type'], top_n=cfg['top_n'],
                        rebal_freq=cfg['rebal_freq'], sector_cap=cfg['sector_cap'],
                    )
                else:
                    nav, wr, txn = strategy_composite(
                        price_df, constituent_data, industries, fund_data,
                        weights=cfg['weights'], top_n=cfg['top_n'],
                        rebal_freq=cfg['rebal_freq'], sector_cap=cfg['sector_cap'],
                    )
                if len(nav) > 52:
                    rolling = rolling_analysis(nav, 52)
                    rolling_results[label] = rolling
                    print(f"\n=== Rolling 12m: {label} ===")
                    print(f"  Win rate: {rolling['win_rate']}%, Mean: {rolling['mean']}%, Min: {rolling['min']}%, Max: {rolling['max']}%")
            except:
                pass

    # Equal-weight benchmark
    print("\n=== Equal-Weight CSI500 Benchmark ===")
    try:
        nav_ew, wr_ew, txn_ew = strategy_factor(
            price_df, constituent_data, industries, fund_data,
            factor_type='momentum_12m', top_n=50, rebal_freq=4,
        )
        stats_ew = calc_stats(nav_ew, wr_ew, txn_ew, 'EqualWeight_top50')
        if stats_ew:
            print(f"  CAGR={stats_ew['cagr_pct']}% MDD={stats_ew['mdd_pct']}% Calmar={stats_ew['calmar']}")
            results.append(stats_ew)
    except Exception as e:
        print(f"  Error: {e}")

    # Save
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Multi-factor CSI500 research v2 with real historical constituents (no survivorship bias, no ROE)',
        'data_stats': {
            'total_stocks': len(all_stocks),
            'pe_coverage': pe_count,
            'pb_coverage': pb_count,
            'price_matrix': f"{price_df.shape[0]} weeks x {price_df.shape[1]} stocks",
        },
        'total_configs': len(configs),
        'total_results': len(results),
        'strategies': results,
        'best_strategy': best_label,
        'best_calmar': best_calmar,
        'rolling_12m_top3': rolling_results,
        'factor_summary': {k: {'label': v['label'], 'cagr': v['cagr_pct'], 'mdd': v['mdd_pct'], 'calmar': v['calmar']}
                          for k, v in factor_best.items()},
    }

    out_path = os.path.join(DATA_DIR, 'csi500_multifactor_research.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\nResults saved to {out_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
