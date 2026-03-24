#!/usr/bin/env python3
"""
CSI500 Growth Factor Research (No Survivorship Bias)
- Uses real historical constituents from baostock
- Growth factors: YOYNI (净利润增长), YOYPNI (归母净利润增长), ROE, Revenue Growth
- Combined with low volatility and momentum for comparison
- Tests pure growth and growth-combo strategies
"""

import baostock as bs
import pandas as pd
import numpy as np
import json
import os
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


# ============================================================
# STEP 1: Data Loading (reuse CSI500 caches where possible)
# ============================================================

def get_constituents():
    """Load CSI 500 constituents (reuse existing cache)"""
    cache_path = os.path.join(CACHE_DIR, 'csi500_constituents_history.json')
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        print(f"Loaded cached CSI500 constituents: {len(data['dates'])} dates, {data['total_unique']} unique stocks")
        return data

    # Fetch if not cached
    lg = bs.login()
    constituent_map = {}
    all_stocks = set()
    for d in REBALANCE_DATES:
        rs = bs.query_zz500_stocks(date=d)
        stocks = []
        while rs.next():
            row = rs.get_row_data()
            stocks.append(row[1])
        constituent_map[d] = stocks
        all_stocks.update(stocks)
        print(f"  {d}: {len(stocks)} stocks")
    bs.logout()

    data = {
        'dates': REBALANCE_DATES,
        'constituents': constituent_map,
        'all_unique_stocks': sorted(all_stocks),
        'total_unique': len(all_stocks),
    }
    with open(cache_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)
    return data


def load_weekly_prices(stock_list):
    """Load cached weekly prices or fetch"""
    cache_path = os.path.join(CACHE_DIR, 'csi500_all_weekly_prices.pkl')
    if os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)
        missing = [s for s in stock_list if s not in df.columns]
        print(f"Loaded cached CSI500 prices: {df.shape[0]} weeks x {df.shape[1]} stocks ({len(missing)} missing)")
        return df

    # Fetch
    lg = bs.login()
    all_prices = {}
    failed = []
    total = len(stock_list)

    for i, stock in enumerate(stock_list):
        if i % 50 == 0:
            print(f"  Fetching prices: {i}/{total}...")
        rs = bs.query_history_k_data_plus(
            stock, 'date,close',
            start_date='2020-09-01', end_date='2026-03-31',
            frequency='w', adjustflag='1'
        )
        data = []
        while rs.next():
            row = rs.get_row_data()
            try:
                data.append({'date': row[0], 'close': float(row[1])})
            except (ValueError, IndexError):
                continue
        if len(data) > 20:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            all_prices[stock] = df['close']
        else:
            failed.append(stock)
        if i % 100 == 99:
            time.sleep(2)

    bs.logout()
    print(f"Fetched {len(all_prices)} stocks, {len(failed)} failed")

    price_df = pd.DataFrame(all_prices).sort_index()
    price_df = price_df[price_df.index >= '2021-01-01']
    price_df = price_df.ffill(limit=2)
    price_df.to_pickle(cache_path)
    return price_df


def load_fundamentals():
    """Load cached PE/PB"""
    cache_path = os.path.join(CACHE_DIR, 'csi500_weekly_fundamentals.pkl')
    if os.path.exists(cache_path):
        data = pd.read_pickle(cache_path)
        print(f"Loaded cached PE/PB: PE={len(data.get('pe', {}))}, PB={len(data.get('pb', {}))}")
        return data
    return {'pe': {}, 'pb': {}}


def fetch_growth_data(stock_list):
    """Fetch quarterly growth + profit data for all stocks"""
    cache_path = os.path.join(CACHE_DIR, 'csi500_quarterly_growth.pkl')

    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 7:
            data = pd.read_pickle(cache_path)
            cached_stocks = set(data.get('fetched_stocks', []))
            missing = [s for s in stock_list if s not in cached_stocks]
            if len(missing) < 50:
                print(f"Loaded cached growth data: {len(cached_stocks)} stocks, {len(missing)} missing")
                return data
            else:
                print(f"Cache has {len(cached_stocks)} stocks, need {len(missing)} more")

    # Load partial cache
    existing = {}
    if os.path.exists(cache_path):
        existing = pd.read_pickle(cache_path)

    fetched_stocks = set(existing.get('fetched_stocks', []))
    growth_records = list(existing.get('growth_records', []))
    profit_records = list(existing.get('profit_records', []))

    to_fetch = [s for s in stock_list if s not in fetched_stocks]
    print(f"Fetching growth data for {len(to_fetch)} stocks...")

    if not to_fetch:
        return existing

    lg = bs.login()
    t0 = time.time()

    # Quarters to fetch: 2020Q4 to 2025Q4
    quarters = []
    for y in range(2020, 2026):
        for q in range(1, 5):
            if y == 2020 and q < 4:
                continue
            quarters.append((y, q))

    for i, stock in enumerate(to_fetch):
        if i % 100 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (len(to_fetch) - i)
            print(f"  Growth data: {i}/{len(to_fetch)} ({elapsed:.0f}s, ETA {eta:.0f}s)")

        for year, quarter in quarters:
            # Growth data
            rs = bs.query_growth_data(code=stock, year=year, quarter=quarter)
            while rs.next():
                row = rs.get_row_data()
                try:
                    growth_records.append({
                        'code': stock,
                        'year': year,
                        'quarter': quarter,
                        'pub_date': row[1],
                        'stat_date': row[2],
                        'yoy_equity': float(row[3]) if row[3] else np.nan,  # 净资产同比
                        'yoy_asset': float(row[4]) if row[4] else np.nan,   # 总资产同比
                        'yoy_ni': float(row[5]) if row[5] else np.nan,      # 净利润同比
                        'yoy_eps': float(row[6]) if row[6] else np.nan,     # EPS同比
                        'yoy_pni': float(row[7]) if row[7] else np.nan,     # 归母净利润同比
                    })
                except (ValueError, IndexError):
                    pass

            # Profit data
            rs = bs.query_profit_data(code=stock, year=year, quarter=quarter)
            while rs.next():
                row = rs.get_row_data()
                try:
                    profit_records.append({
                        'code': stock,
                        'year': year,
                        'quarter': quarter,
                        'pub_date': row[1],
                        'stat_date': row[2],
                        'roe': float(row[3]) if row[3] else np.nan,
                        'np_margin': float(row[4]) if row[4] else np.nan,
                        'gp_margin': float(row[5]) if row[5] else np.nan,
                        'net_profit': float(row[6]) if row[6] else np.nan,
                        'eps_ttm': float(row[7]) if row[7] else np.nan,
                        'revenue': float(row[8]) if row[8] else np.nan,
                    })
                except (ValueError, IndexError):
                    pass

        fetched_stocks.add(stock)

        # Checkpoint every 200 stocks
        if (i + 1) % 200 == 0:
            result = {
                'fetched_stocks': sorted(fetched_stocks),
                'growth_records': growth_records,
                'profit_records': profit_records,
            }
            pd.to_pickle(result, cache_path)
            print(f"  Checkpoint: {len(fetched_stocks)} stocks, {len(growth_records)} growth records")

        if i % 50 == 49:
            time.sleep(1)

    bs.logout()

    result = {
        'fetched_stocks': sorted(fetched_stocks),
        'growth_records': growth_records,
        'profit_records': profit_records,
    }
    pd.to_pickle(result, cache_path)
    print(f"Growth data done! {len(fetched_stocks)} stocks, {len(growth_records)} growth records, {len(profit_records)} profit records")
    return result


def build_quarterly_factors(growth_data, price_df):
    """Convert quarterly growth/profit data into weekly factor DataFrames aligned with price_df"""

    growth_df = pd.DataFrame(growth_data['growth_records'])
    profit_df = pd.DataFrame(growth_data['profit_records'])

    # Map quarter end dates
    def quarter_end(year, quarter):
        month = quarter * 3
        if month == 3:
            return f"{year}-03-31"
        elif month == 6:
            return f"{year}-06-30"
        elif month == 9:
            return f"{year}-09-30"
        else:
            return f"{year}-12-31"

    # Publication lag: quarterly data becomes available ~1-2 months after quarter end
    # Use pub_date if available, otherwise assume: Q1->Apr30, Q2->Aug31, Q3->Oct31, Q4->Apr30 next year
    def assumed_pub_date(year, quarter):
        if quarter == 1:
            return f"{year}-04-30"
        elif quarter == 2:
            return f"{year}-08-31"
        elif quarter == 3:
            return f"{year}-10-31"
        else:
            return f"{year+1}-04-30"

    factors = {}
    weekly_index = price_df.index

    # Process growth factors
    if len(growth_df) > 0:
        print(f"  Processing {len(growth_df)} growth records...")
        growth_df['avail_date'] = growth_df.apply(
            lambda r: r['pub_date'] if pd.notna(r.get('pub_date')) and r['pub_date'] != ''
            else assumed_pub_date(r['year'], r['quarter']),
            axis=1
        )
        growth_df['avail_date'] = pd.to_datetime(growth_df['avail_date'])

        for factor_name, col_name in [
            ('growth_ni', 'yoy_ni'),      # 净利润增长
            ('growth_pni', 'yoy_pni'),     # 归母净利润增长
            ('growth_eps', 'yoy_eps'),     # EPS增长
            ('growth_equity', 'yoy_equity'), # 净资产增长
        ]:
            factor_weekly = pd.DataFrame(np.nan, index=weekly_index, columns=price_df.columns)

            # Group by stock
            for stock, grp in growth_df.groupby('code'):
                if stock not in price_df.columns:
                    continue
                grp = grp.sort_values('avail_date')
                for _, row in grp.iterrows():
                    val = row.get(col_name, np.nan)
                    if pd.isna(val):
                        continue
                    # Clip extreme growth values
                    val = np.clip(val, -2.0, 5.0)
                    avail = row['avail_date']
                    # Fill forward from avail_date until next update
                    mask = weekly_index >= avail
                    factor_weekly.loc[mask, stock] = val

                # Actually we need to handle overlapping updates properly
                # Re-do: for each week, use the most recent available data
                valid_rows = grp[grp[col_name].notna()].sort_values('avail_date')
                if len(valid_rows) == 0:
                    continue

                for week_date in weekly_index:
                    available = valid_rows[valid_rows['avail_date'] <= week_date]
                    if len(available) > 0:
                        latest = available.iloc[-1]
                        val = np.clip(latest[col_name], -2.0, 5.0)
                        factor_weekly.at[week_date, stock] = val

            factors[factor_name] = factor_weekly
            non_null = factor_weekly.notna().sum().sum()
            print(f"    {factor_name}: {non_null} non-null values, {(factor_weekly.notna().any(axis=0)).sum()} stocks")

    # Process profit factors (ROE, margins)
    if len(profit_df) > 0:
        print(f"  Processing {len(profit_df)} profit records...")
        profit_df['avail_date'] = profit_df.apply(
            lambda r: r['pub_date'] if pd.notna(r.get('pub_date')) and r['pub_date'] != ''
            else assumed_pub_date(r['year'], r['quarter']),
            axis=1
        )
        profit_df['avail_date'] = pd.to_datetime(profit_df['avail_date'])

        for factor_name, col_name in [
            ('roe', 'roe'),
            ('np_margin', 'np_margin'),
            ('gp_margin', 'gp_margin'),
        ]:
            factor_weekly = pd.DataFrame(np.nan, index=weekly_index, columns=price_df.columns)

            for stock, grp in profit_df.groupby('code'):
                if stock not in price_df.columns:
                    continue
                valid_rows = grp[grp[col_name].notna()].sort_values('avail_date')
                if len(valid_rows) == 0:
                    continue

                for week_date in weekly_index:
                    available = valid_rows[valid_rows['avail_date'] <= week_date]
                    if len(available) > 0:
                        latest = available.iloc[-1]
                        val = latest[col_name]
                        if factor_name == 'roe':
                            val = np.clip(val, -1.0, 1.0)
                        elif factor_name in ('np_margin', 'gp_margin'):
                            val = np.clip(val, -2.0, 2.0)
                        factor_weekly.at[week_date, stock] = val

            factors[factor_name] = factor_weekly
            non_null = factor_weekly.notna().sum().sum()
            print(f"    {factor_name}: {non_null} non-null values, {(factor_weekly.notna().any(axis=0)).sum()} stocks")

    # Composite growth score: average of available growth metrics
    growth_cols = ['growth_ni', 'growth_pni', 'growth_eps']
    available_growth = [c for c in growth_cols if c in factors]
    if available_growth:
        composite = None
        count = None
        for c in available_growth:
            f = factors[c]
            if composite is None:
                composite = f.fillna(0)
                count = f.notna().astype(float)
            else:
                composite = composite + f.fillna(0)
                count = count + f.notna().astype(float)
        factors['growth_composite'] = composite / count.replace(0, np.nan)
        print(f"    growth_composite: combined from {available_growth}")

    # Growth + quality composite: growth_ni * ROE (growth firms with high profitability)
    if 'growth_ni' in factors and 'roe' in factors:
        # Simple product: high growth + high ROE
        g = factors['growth_ni'].copy()
        r = factors['roe'].copy()
        # Normalize each before combining
        factors['growth_quality'] = g  # will be z-scored later, combined in configs

    return factors


# ============================================================
# STEP 2: Backtest Engine (same as existing framework)
# ============================================================

def build_constituent_mask(price_df, const):
    mask = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
    rebal_dates = const['dates']
    constituents = const['constituents']

    for idx, date in enumerate(price_df.index):
        date_str = date.strftime('%Y-%m-%d')
        active_date = None
        for d in rebal_dates:
            if d <= date_str:
                active_date = d
            else:
                break
        if active_date is None:
            active_date = rebal_dates[0]
        active_stocks = set(constituents[active_date]) & set(price_df.columns)
        mask.loc[date, list(active_stocks)] = True
    return mask


def cross_sectional_zscore(factor_df, mask):
    result = factor_df.copy()
    result[~mask] = np.nan
    row_mean = result.mean(axis=1)
    row_std = result.std(axis=1)
    result = result.sub(row_mean, axis=0).div(row_std.replace(0, np.nan), axis=0)
    return result


def precompute_price_factors(price_df, pe_df, pb_df):
    """Compute price-based factors (same as v3)"""
    factors = {}
    returns = price_df.pct_change(fill_method=None)

    # Low Volatility
    vol12 = returns.rolling(12, min_periods=8).std() * np.sqrt(52)
    factors['low_vol'] = -vol12

    vol20 = returns.rolling(20, min_periods=12).std() * np.sqrt(52)
    factors['low_vol_20w'] = -vol20

    # Momentum
    factors['mom_12m'] = price_df / price_df.shift(52) - 1
    factors['mom_6m'] = price_df / price_df.shift(26) - 1
    factors['mom_3m'] = price_df / price_df.shift(13) - 1

    # Mean Reversion
    factors['mean_rev'] = -(price_df / price_df.shift(4) - 1)

    # Quality Sharpe
    roll_mean = returns.rolling(26, min_periods=13).mean()
    roll_std = returns.rolling(26, min_periods=13).std()
    factors['quality_sharpe'] = roll_mean / roll_std.replace(0, np.nan)

    # Value
    if pe_df is not None:
        pe_clean = pe_df.copy()
        pe_clean[pe_clean <= 0] = np.nan
        pe_clean[pe_clean > 300] = np.nan
        factors['value_pe'] = -pe_clean

    if pb_df is not None:
        pb_clean = pb_df.copy()
        pb_clean[pb_clean <= 0] = np.nan
        pb_clean[pb_clean > 30] = np.nan
        factors['value_pb'] = -pb_clean

    return factors


def run_backtest(price_df, mask, score_df, top_n=10, rebal_freq=2,
                 sector_cap=0, industries=None, txn_bps=8):
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change(fill_method=None)
    warmup = 54

    nav = 1.0
    nav_list = [nav]
    dates = [price_df.index[warmup - 1]]
    prev_holdings = set()
    weekly_rets = []
    total_txn = 0.0

    i = warmup
    while i < len(price_df) - 1:
        scores = score_df.iloc[i].copy()
        active = mask.iloc[i]
        scores[~active] = np.nan
        scores = scores.dropna()

        if len(scores) < top_n:
            for j in range(i + 1, min(i + rebal_freq + 1, len(price_df))):
                nav_list.append(nav_list[-1])
                dates.append(price_df.index[j])
                weekly_rets.append(0.0)
            i += rebal_freq
            continue

        if sector_cap > 0 and industries is not None:
            scores_sorted = scores.sort_values(ascending=False)
            selected = []
            sector_count = defaultdict(int)
            for stock in scores_sorted.index:
                sec = industries.get(stock, {}).get('industry', '其他')
                if sector_count[sec] < sector_cap:
                    selected.append(stock)
                    sector_count[sec] += 1
                    if len(selected) >= top_n:
                        break
        else:
            selected = scores.nlargest(top_n).index.tolist()

        if not selected:
            i += 1
            continue

        n_stocks = len(selected)
        selected_set = set(selected)
        turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(n_stocks, 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            rets_j = returns.iloc[j][selected].dropna()
            port_ret = float(rets_j.mean()) if len(rets_j) > 0 else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav = nav_list[-1] * (1 + port_ret)
            nav_list.append(nav)
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    return pd.Series(nav_list, index=dates), weekly_rets, total_txn


def calc_stats(nav_series, weekly_rets, total_txn, label=""):
    nav = nav_series.values
    if len(nav) < 20:
        return None
    n_weeks = len(nav) - 1
    years = n_weeks / 52.0
    if years < 0.5:
        return None

    cagr = (nav[-1] / nav[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    mdd = float(np.min(dd))

    wr = np.array(weekly_rets)
    sharpe = float(np.mean(wr) / np.std(wr) * np.sqrt(52)) if len(wr) > 10 and np.std(wr) > 0 else 0.0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    annual = {}
    for year in range(2021, 2027):
        ymask = np.array([d.year == year for d in nav_series.index])
        if ymask.sum() > 4:
            year_nav = nav[ymask]
            annual[str(year)] = round((year_nav[-1] / year_nav[0] - 1) * 100, 1)

    return {
        'label': label,
        'cagr_pct': round(cagr * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'annual_returns': annual,
        'txn_pct': round(total_txn * 100, 2),
    }


def rolling_12m(nav_series):
    nav = nav_series.values
    if len(nav) <= 52:
        return None
    rets = []
    for s in range(len(nav) - 52):
        rets.append((nav[s + 52] / nav[s] - 1) * 100)
    rets = np.array(rets)
    return {
        'windows': len(rets),
        'win_rate': round(float(np.mean(rets > 0) * 100), 1),
        'mean': round(float(np.mean(rets)), 2),
        'median': round(float(np.median(rets)), 2),
        'min': round(float(np.min(rets)), 2),
        'max': round(float(np.max(rets)), 2),
        'p10': round(float(np.percentile(rets, 10)), 2),
        'p25': round(float(np.percentile(rets, 25)), 2),
    }


# ============================================================
# STEP 3: Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("CSI500 Growth Factor Research (No Survivorship Bias)")
    print("=" * 60)

    # 1. Load constituents
    print("\n[1/5] Loading CSI500 historical constituents...")
    const = get_constituents()
    all_stocks = const['all_unique_stocks']
    print(f"  {len(all_stocks)} unique stocks")

    # 2. Load industries
    print("\n[2/5] Loading industries...")
    ind_path = os.path.join(CACHE_DIR, 'csi500_industries.json')
    industries = {}
    if os.path.exists(ind_path):
        with open(ind_path) as f:
            industries = json.load(f)
        print(f"  Loaded {len(industries)} industry records")

    # 3. Load prices
    print("\n[3/5] Loading weekly prices...")
    price_df = load_weekly_prices(all_stocks)

    # 4. Load PE/PB
    print("\n[4/5] Loading PE/PB...")
    fund = load_fundamentals()
    pe_df, pb_df = None, None
    pe_dict = fund.get('pe', {})
    pb_dict = fund.get('pb', {})
    if pe_dict:
        pe_df = pd.DataFrame(pe_dict).sort_index()
        pe_df = pe_df.reindex(price_df.index, method='ffill')
        print(f"  PE matrix: {pe_df.shape}")
    if pb_dict:
        pb_df = pd.DataFrame(pb_dict).sort_index()
        pb_df = pb_df.reindex(price_df.index, method='ffill')
        print(f"  PB matrix: {pb_df.shape}")

    # 5. Fetch growth data
    print("\n[5/5] Fetching quarterly growth data...")
    growth_data = fetch_growth_data(all_stocks)

    print(f"\nData loading done in {time.time()-t0:.0f}s")

    # Build constituent mask
    print("\nBuilding constituent mask...")
    mask = build_constituent_mask(price_df, const)
    active_counts = mask.sum(axis=1)
    print(f"  Active stocks per week: min={active_counts.min()}, max={active_counts.max()}")

    # Pre-compute price-based factors
    print("\nPre-computing price factors...")
    price_factors = precompute_price_factors(price_df, pe_df, pb_df)

    # Build quarterly growth factors
    print("\nBuilding quarterly growth factors...")
    growth_factors = build_quarterly_factors(growth_data, price_df)

    # Merge all factors
    all_factors = {**price_factors, **growth_factors}
    print(f"\nTotal factors: {len(all_factors)}")
    for name in all_factors:
        f = all_factors[name]
        coverage = f.notna().sum().sum() / (f.shape[0] * f.shape[1]) * 100
        print(f"  {name}: coverage={coverage:.1f}%")

    # Z-score normalize
    print("\nZ-score normalizing...")
    z_factors = {}
    for name, df in all_factors.items():
        z_factors[name] = cross_sectional_zscore(df, mask)

    # ============================================================
    # Define strategy configurations - GROWTH FOCUSED
    # ============================================================
    configs = []

    # --- Pure growth factors ---
    growth_single = ['growth_ni', 'growth_pni', 'growth_eps', 'growth_equity',
                     'growth_composite', 'roe', 'np_margin', 'gp_margin']
    for f in growth_single:
        if f not in z_factors:
            continue
        for top_n in [10, 15, 20, 30]:
            for rebal in [4, 8]:  # Growth factors update quarterly, less frequent rebalance
                configs.append({
                    'factors': {f: 1.0},
                    'top_n': top_n,
                    'rebal': rebal,
                    'sec_cap': 0,
                    'label': f"G_{f}_t{top_n}_{rebal}w",
                })
        # With sector cap
        for top_n in [15, 20]:
            configs.append({
                'factors': {f: 1.0},
                'top_n': top_n,
                'rebal': 4,
                'sec_cap': 3,
                'label': f"G_{f}_t{top_n}_4w_s3",
            })

    # --- Growth + Low Volatility combos ---
    growth_lv_combos = [
        ({'growth_ni': 0.6, 'low_vol': 0.4}, 'GNI_LV64'),
        ({'growth_ni': 0.5, 'low_vol': 0.5}, 'GNI_LV55'),
        ({'growth_ni': 0.4, 'low_vol': 0.6}, 'GNI_LV46'),
        ({'growth_ni': 0.7, 'low_vol': 0.3}, 'GNI_LV73'),
        ({'growth_pni': 0.6, 'low_vol': 0.4}, 'GPNI_LV64'),
        ({'growth_pni': 0.5, 'low_vol': 0.5}, 'GPNI_LV55'),
        ({'growth_composite': 0.6, 'low_vol': 0.4}, 'GC_LV64'),
        ({'growth_composite': 0.5, 'low_vol': 0.5}, 'GC_LV55'),
        ({'growth_composite': 0.5, 'low_vol_20w': 0.5}, 'GC_LV20_55'),
        ({'growth_ni': 0.5, 'low_vol_20w': 0.5}, 'GNI_LV20_55'),
        ({'roe': 0.5, 'low_vol': 0.5}, 'ROE_LV55'),
        ({'roe': 0.6, 'low_vol': 0.4}, 'ROE_LV64'),
        ({'roe': 0.4, 'low_vol': 0.6}, 'ROE_LV46'),
    ]

    # --- Growth + Value combos ---
    growth_val_combos = [
        ({'growth_ni': 0.5, 'value_pb': 0.5}, 'GNI_PB55'),
        ({'growth_ni': 0.5, 'value_pe': 0.5}, 'GNI_PE55'),
        ({'growth_ni': 0.4, 'value_pe': 0.3, 'value_pb': 0.3}, 'GNI_PEPB'),
        ({'growth_composite': 0.5, 'value_pb': 0.5}, 'GC_PB55'),
        ({'roe': 0.5, 'value_pb': 0.5}, 'ROE_PB55'),
        ({'roe': 0.4, 'value_pe': 0.3, 'value_pb': 0.3}, 'ROE_PEPB'),
    ]

    # --- Growth + Momentum combos ---
    growth_mom_combos = [
        ({'growth_ni': 0.5, 'mom_6m': 0.5}, 'GNI_M6_55'),
        ({'growth_ni': 0.5, 'mom_3m': 0.5}, 'GNI_M3_55'),
        ({'growth_ni': 0.6, 'mom_6m': 0.4}, 'GNI_M6_64'),
        ({'growth_composite': 0.5, 'mom_6m': 0.5}, 'GC_M6_55'),
        ({'roe': 0.4, 'mom_6m': 0.3, 'growth_ni': 0.3}, 'ROE_M6_GNI'),
    ]

    # --- Triple combos: Growth + LV + Value ---
    triple_combos = [
        ({'growth_ni': 0.4, 'low_vol': 0.3, 'value_pb': 0.3}, 'GNI_LV_PB'),
        ({'growth_ni': 0.4, 'low_vol': 0.3, 'value_pe': 0.3}, 'GNI_LV_PE'),
        ({'growth_ni': 0.3, 'low_vol': 0.4, 'value_pb': 0.3}, 'GNI3_LV4_PB3'),
        ({'growth_composite': 0.4, 'low_vol': 0.3, 'value_pb': 0.3}, 'GC_LV_PB'),
        ({'roe': 0.3, 'growth_ni': 0.3, 'low_vol': 0.4}, 'ROE_GNI_LV'),
        ({'roe': 0.3, 'growth_ni': 0.3, 'value_pb': 0.4}, 'ROE_GNI_PB'),
        ({'growth_ni': 0.3, 'low_vol': 0.3, 'value_pb': 0.2, 'roe': 0.2}, 'GNI_LV_PB_ROE'),
        ({'growth_ni': 0.3, 'low_vol_20w': 0.3, 'value_pb': 0.2, 'value_pe': 0.2}, 'GNI_LV20_PEPB'),
    ]

    # --- Quad combos: Growth + LV + Value + Quality ---
    quad_combos = [
        ({'growth_ni': 0.3, 'low_vol': 0.2, 'value_pb': 0.2, 'quality_sharpe': 0.3}, 'GNI_LV_PB_QS'),
        ({'growth_ni': 0.25, 'roe': 0.25, 'low_vol': 0.25, 'value_pb': 0.25}, 'GNI_ROE_LV_PB'),
        ({'growth_composite': 0.3, 'low_vol': 0.3, 'value_pb': 0.2, 'quality_sharpe': 0.2}, 'GC_LV_PB_QS'),
    ]

    all_composites = growth_lv_combos + growth_val_combos + growth_mom_combos + triple_combos + quad_combos

    for weights, name in all_composites:
        if not all(f in z_factors for f in weights):
            continue
        for top_n in [10, 15, 20]:
            for rebal in [2, 4]:
                configs.append({
                    'factors': weights,
                    'top_n': top_n,
                    'rebal': rebal,
                    'sec_cap': 0,
                    'label': f"C_{name}_t{top_n}_{rebal}w",
                })
        # Best params with sector cap
        configs.append({
            'factors': weights,
            'top_n': 15,
            'rebal': 4,
            'sec_cap': 3,
            'label': f"C_{name}_t15_4w_s3",
        })

    # --- Reference: existing best strategies (LV+PB from v3) ---
    ref_combos = [
        ({'low_vol': 0.5, 'value_pb': 0.5}, 'REF_LV_PB'),
        ({'value_pe': 0.3, 'value_pb': 0.3, 'low_vol': 0.4}, 'REF_PE_PB_LV'),
    ]
    for weights, name in ref_combos:
        if not all(f in z_factors for f in weights):
            continue
        for top_n in [10, 15]:
            configs.append({
                'factors': weights,
                'top_n': top_n,
                'rebal': 2,
                'sec_cap': 0,
                'label': f"{name}_t{top_n}_2w",
            })

    print(f"\nTotal configs: {len(configs)}")
    print("Running backtests...\n")

    results = []
    best_nav = None
    best_label = None
    best_calmar = -999

    for idx, cfg in enumerate(configs):
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(configs)} ({time.time()-t0:.0f}s)")

        weights = cfg['factors']
        score_df = None
        for f, w in weights.items():
            if f not in z_factors:
                score_df = None
                break
            if score_df is None:
                score_df = z_factors[f] * w
            else:
                score_df = score_df.add(z_factors[f] * w, fill_value=0)

        if score_df is None:
            continue

        try:
            nav, wr, txn = run_backtest(
                price_df, mask, score_df,
                top_n=cfg['top_n'], rebal_freq=cfg['rebal'],
                sector_cap=cfg['sec_cap'], industries=industries,
                txn_bps=8)

            stats = calc_stats(nav, wr, txn, cfg['label'])
            if stats is None:
                continue
            results.append(stats)

            if stats['calmar'] > best_calmar:
                best_calmar = stats['calmar']
                best_nav = nav
                best_label = cfg['label']

        except Exception as e:
            print(f"  ERROR {cfg['label']}: {e}")

    results.sort(key=lambda x: -x['calmar'])

    # ============================================================
    # Print Results
    # ============================================================
    print(f"\n{'='*60}")
    print(f"CSI500 GROWTH FACTOR RESULTS: {len(results)} strategies in {time.time()-t0:.0f}s")
    print(f"{'='*60}")

    print("\nTop 30 by Calmar:")
    for i, r in enumerate(results[:30]):
        print(f"  {i+1}. {r['label']}")
        print(f"     CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Sharpe={r['sharpe']} Calmar={r['calmar']}")
        print(f"     Annual: {r['annual_returns']}")

    # Growth-only strategies
    print(f"\n{'='*60}")
    print("PURE GROWTH STRATEGIES (G_ prefix):")
    growth_only = [r for r in results if r['label'].startswith('G_')]
    growth_only.sort(key=lambda x: -x['calmar'])
    for r in growth_only[:15]:
        print(f"  {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']}")

    # Growth combo strategies
    print(f"\nGROWTH COMBO STRATEGIES (C_ prefix with growth):")
    growth_combo = [r for r in results if r['label'].startswith('C_') and
                    any(g in r['label'] for g in ['GNI', 'GPNI', 'GC_', 'ROE'])]
    growth_combo.sort(key=lambda x: -x['calmar'])
    for r in growth_combo[:20]:
        print(f"  {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']}")

    # Reference strategies
    print(f"\nREFERENCE (existing LV+PB best):")
    refs = [r for r in results if r['label'].startswith('REF_')]
    for r in refs:
        print(f"  {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']}")

    # Factor group summary
    print(f"\n{'='*60}")
    print("Factor Group Summary (avg Calmar):")
    groups = defaultdict(list)
    for r in results:
        label = r['label']
        if label.startswith('G_'):
            parts = '_'.join(label.split('_')[1:-2])
            groups[f'G_{parts}'].append(r)
        elif label.startswith('C_'):
            parts = label.split('_t')[0]
            groups[parts].append(r)
        else:
            groups[label.split('_t')[0]].append(r)

    for group, items in sorted(groups.items(), key=lambda x: -max(r['calmar'] for r in x[1])):
        avg_cagr = np.mean([r['cagr_pct'] for r in items])
        avg_mdd = np.mean([r['mdd_pct'] for r in items])
        best_c = max(r['calmar'] for r in items)
        print(f"  {group} ({len(items)}): avgCAGR={avg_cagr:.1f}% avgMDD={avg_mdd:.1f}% bestCalmar={best_c:.3f}")

    pos = sum(1 for r in results if r['cagr_pct'] > 0)
    print(f"\nPositive CAGR: {pos}/{len(results)} ({pos/len(results)*100:.0f}%)")

    # Rolling 12m for best
    rolling = None
    if best_nav is not None and len(best_nav) > 52:
        print(f"\n=== Rolling 12m: {best_label} ===")
        rolling = rolling_12m(best_nav)
        if rolling:
            for k, v in rolling.items():
                print(f"  {k}: {v}")

    # Compare with existing CSI500 best (LV+PB)
    csi500_v3_path = os.path.join(DATA_DIR, 'csi500_multifactor_research.json')
    if os.path.exists(csi500_v3_path):
        with open(csi500_v3_path) as f:
            v3 = json.load(f)
        print(f"\n=== vs Existing CSI500 Best (v3 LV+PB) ===")
        print(f"  v3 best: {v3.get('best_strategy')} Calmar={v3.get('best_calmar')}")
        if results:
            print(f"  Growth best: {results[0]['label']} Calmar={results[0]['calmar']}")

    # Save
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'CSI500 Growth Factor Research, real constituents, no survivorship bias',
        'factors_available': list(all_factors.keys()),
        'growth_factors': [f for f in all_factors if f.startswith('growth') or f in ('roe', 'np_margin', 'gp_margin')],
        'total_unique_stocks': len(all_stocks),
        'total_configs': len(configs),
        'total_results': len(results),
        'positive_count': pos,
        'strategies': results,
        'best_strategy': best_label,
        'best_calmar': round(best_calmar, 3),
        'best_rolling_12m': rolling,
        'elapsed_seconds': round(time.time() - t0, 1),
    }

    out_path = os.path.join(DATA_DIR, 'csi500_growth_factor_research.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_path}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
