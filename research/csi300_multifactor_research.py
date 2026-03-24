#!/usr/bin/env python3
"""
CSI300 Multi-Factor Research (No Survivorship Bias)
- Fetches real historical constituents at each semi-annual rebalance via baostock
- Downloads weekly prices + daily PE/PB for all unique stocks
- Runs vectorized multi-factor backtest with same methodology as CSI500 v3
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

# Semi-annual rebalance dates for CSI 300
REBALANCE_DATES = [
    '2020-12-14',
    '2021-06-15',
    '2021-12-13',
    '2022-06-13',
    '2022-12-12',
    '2023-06-12',
    '2023-12-11',
    '2024-06-17',
    '2024-12-16',
    '2025-06-16',
    '2025-12-15',
]


# ============================================================
# STEP 1: Data Fetching
# ============================================================

def get_constituents():
    """Fetch CSI 300 constituents at each rebalance date"""
    cache_path = os.path.join(CACHE_DIR, 'csi300_constituents_history.json')
    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 7:
            with open(cache_path) as f:
                data = json.load(f)
            print(f"Loaded cached CSI300 constituents: {len(data['dates'])} dates, {data['total_unique']} unique stocks")
            return data

    lg = bs.login()
    constituent_map = {}
    all_stocks = set()

    for d in REBALANCE_DATES:
        rs = bs.query_hs300_stocks(date=d)
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

    print(f"Total unique CSI300 stocks: {len(all_stocks)}")
    return data


def get_industries(stock_list):
    """Get industry classification for all stocks"""
    cache_path = os.path.join(CACHE_DIR, 'csi300_industries.json')
    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 30:
            with open(cache_path) as f:
                data = json.load(f)
            missing = [s for s in stock_list if s not in data]
            if len(missing) < 10:
                print(f"Loaded cached CSI300 industries: {len(data)} stocks ({len(missing)} missing)")
                return data

    lg = bs.login()
    industries = {}
    total = len(stock_list)

    for i, stock in enumerate(stock_list):
        if i % 100 == 0:
            print(f"  Fetching industries: {i}/{total}...")
        rs = bs.query_stock_industry(code=stock)
        while rs.next():
            row = rs.get_row_data()
            if len(row) >= 4:
                industries[stock] = {
                    'name': row[2],
                    'industry': row[3],
                    'classification': row[4] if len(row) > 4 else '',
                }
                break
        if stock not in industries:
            industries[stock] = {'name': '?', 'industry': '其他', 'classification': ''}

    bs.logout()

    with open(cache_path, 'w') as f:
        json.dump(industries, f, ensure_ascii=False, indent=1)

    print(f"Got industries for {len(industries)} stocks")
    return industries


def fetch_weekly_prices(stock_list):
    """Fetch weekly close prices for all stocks"""
    cache_path = os.path.join(CACHE_DIR, 'csi300_all_weekly_prices.pkl')
    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 3:
            df = pd.read_pickle(cache_path)
            missing = [s for s in stock_list if s not in df.columns]
            if len(missing) < 20:
                print(f"Loaded cached CSI300 prices: {df.shape[0]} weeks x {df.shape[1]} stocks ({len(missing)} missing)")
                return df

    lg = bs.login()
    all_prices = {}
    total = len(stock_list)
    failed = []

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
    if failed:
        print(f"  Failed: {failed[:20]}...")

    price_df = pd.DataFrame(all_prices).sort_index()
    price_df = price_df[price_df.index >= '2021-01-01']
    price_df = price_df.ffill(limit=2)
    price_df.to_pickle(cache_path)
    print(f"Price matrix: {price_df.shape[0]} weeks x {price_df.shape[1]} stocks")
    return price_df


def fetch_fundamentals(stock_list):
    """Fetch daily PE/PB and resample to weekly"""
    cache_path = os.path.join(CACHE_DIR, 'csi300_weekly_fundamentals.pkl')

    existing = {}
    if os.path.exists(cache_path):
        old = pd.read_pickle(cache_path)
        if isinstance(old, dict) and len(old.get('pe', {})) > 10:
            existing = old
            print(f"Loaded partial PE/PB cache: PE={len(existing.get('pe',{}))}, PB={len(existing.get('pb',{}))}")

    to_fetch = [s for s in stock_list if s not in existing.get('pe', {})]
    print(f"Need to fetch PE/PB: {len(to_fetch)} stocks")

    if not to_fetch:
        print("All PE/PB data cached!")
        return existing

    lg = bs.login()
    pe_data = dict(existing.get('pe', {}))
    pb_data = dict(existing.get('pb', {}))
    failed = []
    t0 = time.time()

    for i, stock in enumerate(to_fetch):
        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Fetching PE/PB: {i}/{len(to_fetch)} ({elapsed:.0f}s)...")

        rs = bs.query_history_k_data_plus(
            stock, 'date,peTTM,pbMRQ',
            start_date='2020-09-01', end_date='2026-03-31',
            frequency='d', adjustflag='1'
        )

        dates_list, pe_list, pb_list = [], [], []
        while rs.next():
            row = rs.get_row_data()
            try:
                d = row[0]
                pe = float(row[1]) if row[1] and row[1] != '' else np.nan
                pb = float(row[2]) if row[2] and row[2] != '' else np.nan
                dates_list.append(d)
                pe_list.append(pe)
                pb_list.append(pb)
            except (ValueError, IndexError):
                continue

        if len(dates_list) > 20:
            idx = pd.to_datetime(dates_list)
            weekly_pe = pd.Series(pe_list, index=idx).resample('W-FRI').last()
            weekly_pb = pd.Series(pb_list, index=idx).resample('W-FRI').last()
            pe_data[stock] = weekly_pe.dropna()
            pb_data[stock] = weekly_pb.dropna()
        else:
            failed.append(stock)

        if (i + 1) % 200 == 0:
            pd.to_pickle({'pe': pe_data, 'pb': pb_data}, cache_path)
            print(f"  Checkpoint: PE={len(pe_data)}, PB={len(pb_data)}")

        if i % 100 == 99:
            time.sleep(1)

    bs.logout()

    result = {'pe': pe_data, 'pb': pb_data}
    pd.to_pickle(result, cache_path)
    print(f"PE/PB done! PE: {len(pe_data)}, PB: {len(pb_data)}, Failed: {len(failed)}")
    return result


# ============================================================
# STEP 2: Multi-Factor Backtest Engine (same as CSI500 v3)
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


def precompute_factors(price_df, pe_df, pb_df):
    factors = {}
    returns = price_df.pct_change(fill_method=None)

    # Low Volatility (12-week)
    vol12 = returns.rolling(12, min_periods=8).std() * np.sqrt(52)
    factors['low_vol'] = -vol12

    # Low Volatility (20-week)
    vol20 = returns.rolling(20, min_periods=12).std() * np.sqrt(52)
    factors['low_vol_20w'] = -vol20

    # Momentum 12m, 6m, 3m
    factors['mom_12m'] = price_df / price_df.shift(52) - 1
    factors['mom_6m'] = price_df / price_df.shift(26) - 1
    factors['mom_3m'] = price_df / price_df.shift(13) - 1

    # Mean Reversion
    factors['mean_rev'] = -(price_df / price_df.shift(4) - 1)

    # Quality Sharpe
    roll_mean = returns.rolling(26, min_periods=13).mean()
    roll_std = returns.rolling(26, min_periods=13).std()
    factors['quality_sharpe'] = roll_mean / roll_std.replace(0, np.nan)

    # Value PE
    if pe_df is not None:
        pe_clean = pe_df.copy()
        pe_clean[pe_clean <= 0] = np.nan
        pe_clean[pe_clean > 300] = np.nan
        factors['value_pe'] = -pe_clean

    # Value PB
    if pb_df is not None:
        pb_clean = pb_df.copy()
        pb_clean[pb_clean <= 0] = np.nan
        pb_clean[pb_clean > 30] = np.nan
        factors['value_pb'] = -pb_clean

    # Dividend yield proxy (1/PB * earnings proxy) - for CSI300 large-caps
    # Size factor (market cap proxy via price level - negate for small within 300)
    # Not adding these for now, same factor set as CSI500

    print(f"Pre-computed {len(factors)} factors")
    return factors


def cross_sectional_zscore(factor_df, mask):
    result = factor_df.copy()
    result[~mask] = np.nan
    row_mean = result.mean(axis=1)
    row_std = result.std(axis=1)
    result = result.sub(row_mean, axis=0).div(row_std.replace(0, np.nan), axis=0)
    return result


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
    print("CSI300 Multi-Factor Research (No Survivorship Bias)")
    print("=" * 60)

    # 1. Fetch constituents
    print("\n[1/4] Fetching CSI300 historical constituents...")
    const = get_constituents()
    all_stocks = const['all_unique_stocks']
    print(f"  {len(all_stocks)} unique stocks across {len(const['dates'])} rebalance dates")

    # 2. Fetch industries
    print("\n[2/4] Fetching industry classifications...")
    industries = get_industries(all_stocks)

    # 3. Fetch prices
    print("\n[3/4] Fetching weekly prices...")
    price_df = fetch_weekly_prices(all_stocks)

    # 4. Fetch PE/PB
    print("\n[4/4] Fetching PE/PB fundamentals...")
    fund = fetch_fundamentals(all_stocks)

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

    print(f"\nData fetching done in {time.time()-t0:.0f}s")
    print(f"Prices: {price_df.shape[0]} weeks x {price_df.shape[1]} stocks")

    # Build constituent mask
    print("\nBuilding constituent mask...")
    mask = build_constituent_mask(price_df, const)
    active_counts = mask.sum(axis=1)
    print(f"  Active stocks per week: min={active_counts.min()}, max={active_counts.max()}, mean={active_counts.mean():.0f}")

    # Pre-compute factors
    print("\nPre-computing factors...")
    factors = precompute_factors(price_df, pe_df, pb_df)

    # Z-score normalize
    print("Z-score normalizing...")
    z_factors = {}
    for name, df in factors.items():
        z_factors[name] = cross_sectional_zscore(df, mask)

    print(f"\nSetup done in {time.time()-t0:.1f}s")

    # Define configs
    configs = []

    # Single factors
    single_factors = list(factors.keys())
    for f in single_factors:
        for top_n in [10, 15, 20, 30]:
            for rebal in [2, 4]:
                configs.append({
                    'factors': {f: 1.0},
                    'top_n': top_n,
                    'rebal': rebal,
                    'sec_cap': 0,
                    'label': f"{f}_top{top_n}_{rebal}w",
                })
        for top_n in [15, 20]:
            configs.append({
                'factors': {f: 1.0},
                'top_n': top_n,
                'rebal': 2,
                'sec_cap': 3,
                'label': f"{f}_top{top_n}_2w_sec3",
            })

    # Composite strategies
    composites = [
        ({'low_vol': 0.5, 'quality_sharpe': 0.5}, 'LV_QS'),
        ({'low_vol': 0.5, 'mean_rev': 0.5}, 'LV_MR'),
        ({'low_vol': 0.5, 'mom_6m': 0.5}, 'LV_Mom6'),
        ({'low_vol': 0.5, 'mom_12m': 0.5}, 'LV_Mom12'),
        ({'low_vol': 0.3, 'mom_12m': 0.3, 'mean_rev': 0.4}, 'LV_Mom_MR'),
        ({'low_vol': 0.4, 'quality_sharpe': 0.3, 'mean_rev': 0.3}, 'LV_QS_MR'),
        ({'mom_6m': 0.5, 'mean_rev': 0.5}, 'Mom6_MR'),
        ({'quality_sharpe': 0.5, 'mean_rev': 0.5}, 'QS_MR'),
        ({'low_vol': 0.5, 'mom_3m': 0.5}, 'LV_Mom3'),
    ]

    has_pe = 'value_pe' in factors
    has_pb = 'value_pb' in factors
    if has_pe:
        composites += [
            ({'low_vol': 0.5, 'value_pe': 0.5}, 'LV_PE'),
            ({'value_pe': 0.5, 'quality_sharpe': 0.5}, 'PE_QS'),
            ({'value_pe': 0.4, 'low_vol': 0.3, 'mom_12m': 0.3}, 'PE_LV_Mom'),
            ({'value_pe': 0.3, 'low_vol': 0.3, 'quality_sharpe': 0.2, 'mom_6m': 0.2}, 'PE_LV_QS_Mom'),
        ]
    if has_pb:
        composites += [
            ({'low_vol': 0.5, 'value_pb': 0.5}, 'LV_PB'),
            ({'value_pb': 0.5, 'quality_sharpe': 0.5}, 'PB_QS'),
        ]
    if has_pe and has_pb:
        composites += [
            ({'value_pe': 0.3, 'value_pb': 0.3, 'low_vol': 0.4}, 'PE_PB_LV'),
            ({'value_pe': 0.25, 'value_pb': 0.25, 'low_vol': 0.25, 'quality_sharpe': 0.25}, 'PE_PB_LV_QS'),
            ({'value_pe': 0.4, 'value_pb': 0.4, 'low_vol': 0.2}, 'PE_PB_lv'),
            ({'value_pe': 0.5, 'value_pb': 0.5}, 'PE_PB'),
        ]

    for weights, name in composites:
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
        configs.append({
            'factors': weights,
            'top_n': 15,
            'rebal': 2,
            'sec_cap': 3,
            'label': f"C_{name}_t15_2w_s3",
        })

    print(f"\nTotal configs: {len(configs)}")
    print("Running backtests...\n")

    results = []
    best_nav = None
    best_label = None
    best_calmar = -999

    for idx, cfg in enumerate(configs):
        if idx % 25 == 0:
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

    print(f"\n{'='*60}")
    print(f"CSI300 RESULTS: {len(results)} strategies tested in {time.time()-t0:.0f}s")
    print(f"{'='*60}")

    print("\nTop 30 by Calmar:")
    for i, r in enumerate(results[:30]):
        print(f"  {i+1}. {r['label']}")
        print(f"     CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Sharpe={r['sharpe']} Calmar={r['calmar']}")
        print(f"     Annual: {r['annual_returns']}")

    print(f"\nBottom 5:")
    for r in results[-5:]:
        print(f"  {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']}")

    # Positive/negative count
    pos = sum(1 for r in results if r['cagr_pct'] > 0)
    print(f"\nPositive CAGR: {pos}/{len(results)} ({pos/len(results)*100:.0f}%)")

    # Factor group summary
    print(f"\n{'='*60}")
    print("Factor Group Summary (avg Calmar):")
    groups = defaultdict(list)
    for r in results:
        label = r['label']
        if label.startswith('C_'):
            parts = label.split('_t')[0]
            groups[parts].append(r)
        else:
            factor = '_'.join(label.split('_')[:-2]) if '_top' in label else label.split('_top')[0]
            groups[factor].append(r)

    for group, items in sorted(groups.items(), key=lambda x: -max(r['calmar'] for r in x[1])):
        avg_cagr = np.mean([r['cagr_pct'] for r in items])
        avg_mdd = np.mean([r['mdd_pct'] for r in items])
        avg_calmar = np.mean([r['calmar'] for r in items])
        best_c = max(r['calmar'] for r in items)
        print(f"  {group} ({len(items)}): CAGR={avg_cagr:.1f}% MDD={avg_mdd:.1f}% Calmar={avg_calmar:.3f} best={best_c:.3f}")

    # Rolling 12m for best
    rolling = None
    if best_nav is not None and len(best_nav) > 52:
        print(f"\n=== Rolling 12m: {best_label} ===")
        rolling = rolling_12m(best_nav)
        if rolling:
            for k, v in rolling.items():
                print(f"  {k}: {v}")

    # Compare with CSI500 best
    csi500_path = os.path.join(DATA_DIR, 'csi500_multifactor_research.json')
    if os.path.exists(csi500_path):
        with open(csi500_path) as f:
            csi500 = json.load(f)
        print(f"\n=== CSI300 vs CSI500 Comparison ===")
        if results:
            print(f"  CSI300 best: {results[0]['label']} CAGR={results[0]['cagr_pct']}% MDD={results[0]['mdd_pct']}% Calmar={results[0]['calmar']}")
        print(f"  CSI500 best: {csi500.get('best_strategy','')} Calmar={csi500.get('best_calmar','')}")
        if csi500.get('strategies'):
            best500 = csi500['strategies'][0]
            print(f"  CSI500 top1: {best500['label']} CAGR={best500['cagr_pct']}% MDD={best500['mdd_pct']}% Calmar={best500['calmar']}")

    # Save
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Multi-factor CSI300, real historical constituents, no survivorship bias',
        'factors_available': list(factors.keys()),
        'has_pe': has_pe,
        'has_pb': has_pb,
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

    out_path = os.path.join(DATA_DIR, 'csi300_multifactor_research.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_path}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
