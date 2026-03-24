#!/usr/bin/env python3
"""
CSI500 Growth Factor Research - FAST version
Uses existing cached price/constituent data + akshare for bulk financial data
"""

import akshare as ak
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'baostock_cache')


def load_cached_data():
    """Load existing cached data from previous CSI500 research"""
    # Constituents
    with open(os.path.join(CACHE_DIR, 'csi500_constituents_history.json')) as f:
        const = json.load(f)
    print(f"Constituents: {const['total_unique']} unique stocks, {len(const['dates'])} rebalance dates")

    # Weekly prices
    price_df = pd.read_pickle(os.path.join(CACHE_DIR, 'csi500_all_weekly_prices.pkl'))
    print(f"Prices: {price_df.shape[0]} weeks x {price_df.shape[1]} stocks")

    # PE/PB
    fund = pd.read_pickle(os.path.join(CACHE_DIR, 'csi500_weekly_fundamentals.pkl'))
    pe_dict = fund.get('pe', {})
    pb_dict = fund.get('pb', {})
    pe_df = pd.DataFrame(pe_dict).sort_index().reindex(price_df.index, method='ffill') if pe_dict else None
    pb_df = pd.DataFrame(pb_dict).sort_index().reindex(price_df.index, method='ffill') if pb_dict else None
    print(f"PE: {pe_df.shape if pe_df is not None else 'None'}, PB: {pb_df.shape if pb_df is not None else 'None'}")

    # Industries
    ind_path = os.path.join(CACHE_DIR, 'csi500_industries.json')
    industries = {}
    if os.path.exists(ind_path):
        with open(ind_path) as f:
            industries = json.load(f)

    return const, price_df, pe_df, pb_df, industries


def fetch_financial_data_akshare(stock_list, price_df):
    """Fetch financial data using akshare for bulk retrieval"""
    cache_path = os.path.join(CACHE_DIR, 'csi500_akshare_financials.pkl')

    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 7:
            data = pd.read_pickle(cache_path)
            print(f"Loaded cached akshare financials: {len(data)} items")
            return data

    print("Fetching financial data via akshare...")

    # Convert baostock codes to akshare codes (sh.600000 -> 600000)
    def bs_to_ak(code):
        return code.split('.')[1]

    def ak_to_bs(code):
        if code.startswith('6'):
            return f'sh.{code}'
        else:
            return f'sz.{code}'

    financial_data = {}
    failed = []
    total = len(stock_list)

    for i, bs_code in enumerate(stock_list):
        if i % 100 == 0:
            print(f"  Progress: {i}/{total}...")

        ak_code = bs_to_ak(bs_code)
        if bs_code not in price_df.columns:
            continue

        try:
            # Get financial indicators
            df = ak.stock_financial_abstract_ths(symbol=ak_code, indicator="按报告期")
            if df is not None and len(df) > 0:
                financial_data[bs_code] = df
        except Exception as e:
            failed.append((bs_code, str(e)))
            continue

        if i % 50 == 49:
            time.sleep(1)

    print(f"Got financial data for {len(financial_data)} stocks, {len(failed)} failed")

    pd.to_pickle(financial_data, cache_path)
    return financial_data


def fetch_growth_from_efinance(stock_list, price_df):
    """Alternative: fetch key metrics using simple approach with baostock
    Only fetch 2 years of annual data (4 quarters) per stock"""
    import baostock as bs

    cache_path = os.path.join(CACHE_DIR, 'csi500_growth_simple.pkl')
    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 7:
            data = pd.read_pickle(cache_path)
            print(f"Loaded cached simple growth: {len(data)} stocks")
            return data

    print("Fetching growth data (simplified - annual only)...")
    lg = bs.login()

    result = {}
    total = len(stock_list)
    t0 = time.time()

    for i, stock in enumerate(stock_list):
        if stock not in price_df.columns:
            continue
        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Progress: {i}/{total} ({elapsed:.0f}s)")

        stock_data = {}
        # Only annual Q4 data for 2022, 2023, 2024
        for year in [2022, 2023, 2024]:
            # Growth
            rs = bs.query_growth_data(code=stock, year=year, quarter=4)
            while rs.next():
                row = rs.get_row_data()
                try:
                    stock_data[f'yoy_ni_{year}'] = float(row[5]) if row[5] else None
                    stock_data[f'yoy_pni_{year}'] = float(row[7]) if row[7] else None
                    stock_data[f'yoy_eps_{year}'] = float(row[6]) if row[6] else None
                except Exception:
                    pass

            # Profit
            rs = bs.query_profit_data(code=stock, year=year, quarter=4)
            while rs.next():
                row = rs.get_row_data()
                try:
                    stock_data[f'roe_{year}'] = float(row[3]) if row[3] else None
                    stock_data[f'np_margin_{year}'] = float(row[4]) if row[4] else None
                    stock_data[f'revenue_{year}'] = float(row[8]) if row[8] else None
                except Exception:
                    pass

        if stock_data:
            result[stock] = stock_data

        # Checkpoint
        if (i + 1) % 200 == 0:
            pd.to_pickle(result, cache_path)
            elapsed = time.time() - t0
            print(f"  Checkpoint: {len(result)} stocks ({elapsed:.0f}s)")

    bs.logout()

    pd.to_pickle(result, cache_path)
    print(f"Growth data done! {len(result)} stocks in {time.time()-t0:.0f}s")
    return result


def build_growth_factors(growth_data, price_df):
    """Build weekly growth factor DataFrames from annual data"""
    weekly_index = price_df.index

    # For each stock, we have 2022/2023/2024 annual data
    # Growth data becomes available approximately:
    # 2022 annual -> available after April 2023
    # 2023 annual -> available after April 2024
    # 2024 annual -> available after April 2025

    availability = {
        2022: pd.Timestamp('2023-04-30'),
        2023: pd.Timestamp('2024-04-30'),
        2024: pd.Timestamp('2025-04-30'),
    }

    factors = {}

    for factor_name, key_prefix in [
        ('growth_ni', 'yoy_ni'),
        ('growth_pni', 'yoy_pni'),
        ('growth_eps', 'yoy_eps'),
        ('roe', 'roe'),
        ('np_margin', 'np_margin'),
    ]:
        factor_df = pd.DataFrame(np.nan, index=weekly_index, columns=price_df.columns)
        count = 0

        for stock, data in growth_data.items():
            if stock not in price_df.columns:
                continue

            for year in [2022, 2023, 2024]:
                val = data.get(f'{key_prefix}_{year}')
                if val is None:
                    continue

                # Clip
                if 'growth' in factor_name:
                    val = np.clip(val, -2.0, 5.0)
                elif factor_name == 'roe':
                    val = np.clip(val, -1.0, 1.0)

                avail = availability[year]
                next_avail = availability.get(year + 1, pd.Timestamp('2099-12-31'))

                # Fill from avail_date to next_avail_date
                mask = (weekly_index >= avail) & (weekly_index < next_avail)
                factor_df.loc[mask, stock] = val
                count += 1

        non_null = factor_df.notna().sum().sum()
        stock_coverage = (factor_df.notna().any(axis=0)).sum()
        print(f"  {factor_name}: {non_null} values, {stock_coverage} stocks")
        factors[factor_name] = factor_df

    # Revenue growth (CAGR from revenue data)
    rev_growth_df = pd.DataFrame(np.nan, index=weekly_index, columns=price_df.columns)
    for stock, data in growth_data.items():
        if stock not in price_df.columns:
            continue

        rev_2022 = data.get('revenue_2022')
        rev_2023 = data.get('revenue_2023')
        rev_2024 = data.get('revenue_2024')

        # 2023 YOY revenue growth
        if rev_2022 and rev_2023 and rev_2022 > 0:
            yoy = (rev_2023 / rev_2022 - 1)
            yoy = np.clip(yoy, -2.0, 5.0)
            mask = (weekly_index >= availability[2023]) & (weekly_index < availability.get(2024, pd.Timestamp('2099-12-31')))
            rev_growth_df.loc[mask, stock] = yoy

        # 2024 YOY revenue growth
        if rev_2023 and rev_2024 and rev_2023 > 0:
            yoy = (rev_2024 / rev_2023 - 1)
            yoy = np.clip(yoy, -2.0, 5.0)
            mask = weekly_index >= availability[2024]
            rev_growth_df.loc[mask, stock] = yoy

    factors['rev_growth'] = rev_growth_df

    # Composite growth
    g_cols = ['growth_ni', 'growth_pni', 'growth_eps']
    available = [c for c in g_cols if c in factors]
    if available:
        composite = None
        cnt = None
        for c in available:
            f = factors[c]
            if composite is None:
                composite = f.fillna(0)
                cnt = f.notna().astype(float)
            else:
                composite = composite + f.fillna(0)
                cnt = cnt + f.notna().astype(float)
        factors['growth_composite'] = composite / cnt.replace(0, np.nan)
        print(f"  growth_composite: from {available}")

    return factors


# ============================================================
# Backtest Engine (same as v3)
# ============================================================

def build_constituent_mask(price_df, const):
    mask = pd.DataFrame(False, index=price_df.index, columns=price_df.columns)
    for idx, date in enumerate(price_df.index):
        date_str = date.strftime('%Y-%m-%d')
        active_date = None
        for d in const['dates']:
            if d <= date_str:
                active_date = d
            else:
                break
        if active_date is None:
            active_date = const['dates'][0]
        active_stocks = set(const['constituents'][active_date]) & set(price_df.columns)
        mask.loc[date, list(active_stocks)] = True
    return mask


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
        'label': label, 'cagr_pct': round(cagr * 100, 1), 'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3), 'calmar': round(calmar, 3),
        'annual_returns': annual, 'txn_pct': round(total_txn * 100, 2),
    }


def rolling_12m(nav_series):
    nav = nav_series.values
    if len(nav) <= 52:
        return None
    rets = np.array([(nav[s + 52] / nav[s] - 1) * 100 for s in range(len(nav) - 52)])
    return {
        'windows': len(rets), 'win_rate': round(float(np.mean(rets > 0) * 100), 1),
        'mean': round(float(np.mean(rets)), 2), 'median': round(float(np.median(rets)), 2),
        'min': round(float(np.min(rets)), 2), 'max': round(float(np.max(rets)), 2),
        'p10': round(float(np.percentile(rets, 10)), 2), 'p25': round(float(np.percentile(rets, 25)), 2),
    }


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("CSI500 Growth Factor Research (Fast Version)")
    print("=" * 60)

    # Load cached data
    print("\n[1/3] Loading cached data...")
    const, price_df, pe_df, pb_df, industries = load_cached_data()

    # Fetch growth data (simplified - annual only)
    print("\n[2/3] Fetching growth data (annual, 3 years)...")
    all_stocks = const['all_unique_stocks']
    growth_data = fetch_growth_from_efinance(all_stocks, price_df)

    # Build factors
    print("\n[3/3] Building factors...")

    # Price-based factors
    returns = price_df.pct_change(fill_method=None)
    price_factors = {}

    vol12 = returns.rolling(12, min_periods=8).std() * np.sqrt(52)
    price_factors['low_vol'] = -vol12

    vol20 = returns.rolling(20, min_periods=12).std() * np.sqrt(52)
    price_factors['low_vol_20w'] = -vol20

    price_factors['mom_12m'] = price_df / price_df.shift(52) - 1
    price_factors['mom_6m'] = price_df / price_df.shift(26) - 1
    price_factors['mom_3m'] = price_df / price_df.shift(13) - 1
    price_factors['mean_rev'] = -(price_df / price_df.shift(4) - 1)

    roll_mean = returns.rolling(26, min_periods=13).mean()
    roll_std = returns.rolling(26, min_periods=13).std()
    price_factors['quality_sharpe'] = roll_mean / roll_std.replace(0, np.nan)

    if pe_df is not None:
        pe_clean = pe_df.copy()
        pe_clean[pe_clean <= 0] = np.nan
        pe_clean[pe_clean > 300] = np.nan
        price_factors['value_pe'] = -pe_clean
    if pb_df is not None:
        pb_clean = pb_df.copy()
        pb_clean[pb_clean <= 0] = np.nan
        pb_clean[pb_clean > 30] = np.nan
        price_factors['value_pb'] = -pb_clean

    # Growth factors
    growth_factors = build_growth_factors(growth_data, price_df)

    all_factors = {**price_factors, **growth_factors}
    print(f"\nTotal factors: {len(all_factors)}")

    # Build mask
    mask = build_constituent_mask(price_df, const)
    print(f"Active stocks: min={mask.sum(axis=1).min()}, max={mask.sum(axis=1).max()}")

    # Z-score
    print("Z-scoring...")
    z_factors = {name: cross_sectional_zscore(df, mask) for name, df in all_factors.items()}

    # Check growth factor coverage within mask
    for gf in ['growth_ni', 'growth_pni', 'roe', 'rev_growth', 'growth_composite']:
        if gf in z_factors:
            valid = (z_factors[gf].notna() & mask).sum().sum()
            total = mask.sum().sum()
            print(f"  {gf}: {valid}/{total} coverage = {valid/total*100:.1f}%")

    # ============================================================
    # Define configs
    # ============================================================
    configs = []

    # Pure growth
    for f in ['growth_ni', 'growth_pni', 'growth_eps', 'roe', 'np_margin', 'rev_growth', 'growth_composite']:
        if f not in z_factors:
            continue
        for top_n in [10, 15, 20, 30]:
            for rebal in [4, 8]:
                configs.append({'factors': {f: 1.0}, 'top_n': top_n, 'rebal': rebal,
                               'sec_cap': 0, 'label': f"G_{f}_t{top_n}_{rebal}w"})

    # Growth + Low Volatility
    for gf in ['growth_ni', 'growth_pni', 'growth_composite', 'roe']:
        if gf not in z_factors:
            continue
        for lv in ['low_vol', 'low_vol_20w']:
            for gw, lw in [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7)]:
                for top_n in [10, 15, 20]:
                    for rebal in [2, 4]:
                        label = f"C_{gf}_{lv}_g{int(gw*10)}l{int(lw*10)}_t{top_n}_{rebal}w"
                        configs.append({'factors': {gf: gw, lv: lw}, 'top_n': top_n,
                                       'rebal': rebal, 'sec_cap': 0, 'label': label})

    # Growth + Value
    for gf in ['growth_ni', 'growth_composite', 'roe']:
        if gf not in z_factors:
            continue
        for vf in ['value_pe', 'value_pb']:
            if vf not in z_factors:
                continue
            for gw, vw in [(0.6, 0.4), (0.5, 0.5), (0.4, 0.6)]:
                for top_n in [10, 15, 20]:
                    configs.append({'factors': {gf: gw, vf: vw}, 'top_n': top_n, 'rebal': 4,
                                   'sec_cap': 0, 'label': f"C_{gf}_{vf}_g{int(gw*10)}v{int(vw*10)}_t{top_n}_4w"})

    # Growth + LV + Value (triple)
    for gf in ['growth_ni', 'growth_composite', 'roe']:
        if gf not in z_factors:
            continue
        for vf in ['value_pb', 'value_pe']:
            if vf not in z_factors:
                continue
            combos = [
                (0.4, 0.3, 0.3), (0.3, 0.4, 0.3), (0.3, 0.3, 0.4),
                (0.5, 0.3, 0.2), (0.5, 0.2, 0.3),
            ]
            for gw, lw, vw in combos:
                for top_n in [10, 15, 20]:
                    for rebal in [2, 4]:
                        label = f"C_{gf}_lv_{vf}_g{int(gw*10)}l{int(lw*10)}v{int(vw*10)}_t{top_n}_{rebal}w"
                        configs.append({'factors': {gf: gw, 'low_vol': lw, vf: vw}, 'top_n': top_n,
                                       'rebal': rebal, 'sec_cap': 0, 'label': label})

    # Growth + Momentum
    for gf in ['growth_ni', 'growth_composite']:
        if gf not in z_factors:
            continue
        for mf in ['mom_6m', 'mom_3m']:
            for gw, mw in [(0.6, 0.4), (0.5, 0.5), (0.7, 0.3)]:
                for top_n in [15, 20]:
                    configs.append({'factors': {gf: gw, mf: mw}, 'top_n': top_n, 'rebal': 4,
                                   'sec_cap': 0, 'label': f"C_{gf}_{mf}_g{int(gw*10)}m{int(mw*10)}_t{top_n}_4w"})

    # Growth + Quality Sharpe
    for gf in ['growth_ni', 'roe']:
        if gf not in z_factors:
            continue
        for gw, qw in [(0.5, 0.5), (0.6, 0.4)]:
            for top_n in [10, 15, 20]:
                configs.append({'factors': {gf: gw, 'quality_sharpe': qw}, 'top_n': top_n, 'rebal': 4,
                               'sec_cap': 0, 'label': f"C_{gf}_qs_g{int(gw*10)}q{int(qw*10)}_t{top_n}_4w"})

    # Reference: existing best (LV+PB)
    for vf in ['value_pb']:
        if vf in z_factors:
            configs.append({'factors': {'low_vol': 0.5, vf: 0.5}, 'top_n': 10, 'rebal': 2,
                           'sec_cap': 0, 'label': 'REF_LV_PB_t10_2w'})

    print(f"\nTotal configs: {len(configs)}")
    print("Running backtests...\n")

    results = []
    best_nav = None
    best_label = None
    best_calmar = -999

    for idx, cfg in enumerate(configs):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(configs)} ({time.time()-t0:.0f}s)")

        weights = cfg['factors']
        if not all(f in z_factors for f in weights):
            continue

        score_df = None
        for f, w in weights.items():
            if score_df is None:
                score_df = z_factors[f] * w
            else:
                score_df = score_df.add(z_factors[f] * w, fill_value=0)

        try:
            nav, wr, txn = run_backtest(price_df, mask, score_df,
                                        top_n=cfg['top_n'], rebal_freq=cfg['rebal'],
                                        sector_cap=cfg['sec_cap'], industries=industries, txn_bps=8)
            stats = calc_stats(nav, wr, txn, cfg['label'])
            if stats is None:
                continue
            results.append(stats)
            if stats['calmar'] > best_calmar:
                best_calmar = stats['calmar']
                best_nav = nav
                best_label = cfg['label']
        except Exception as e:
            pass

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

    # Growth-only
    print(f"\nPURE GROWTH (G_ prefix):")
    for r in [r for r in results if r['label'].startswith('G_')][:10]:
        print(f"  {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']}")

    # Growth combos
    print(f"\nGROWTH COMBOS (top 20):")
    growth_combos = [r for r in results if 'growth' in r['label'].lower() or 'roe' in r['label'].lower()]
    for r in growth_combos[:20]:
        print(f"  {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Calmar={r['calmar']}")

    # Factor group summary
    print(f"\nFactor Group Summary:")
    groups = defaultdict(list)
    for r in results:
        if r['label'].startswith('G_'):
            base = '_'.join(r['label'].split('_')[1:3])
            groups[f'G_{base}'].append(r)
        elif r['label'].startswith('C_'):
            # Extract factor combo name
            parts = r['label'].split('_t')[0]
            groups[parts].append(r)
        else:
            groups[r['label'].split('_t')[0]].append(r)

    for group, items in sorted(groups.items(), key=lambda x: -max(r['calmar'] for r in x[1]))[:30]:
        best_c = max(r['calmar'] for r in items)
        avg_cagr = np.mean([r['cagr_pct'] for r in items])
        avg_mdd = np.mean([r['mdd_pct'] for r in items])
        print(f"  {group} ({len(items)}): avgCAGR={avg_cagr:.1f}% avgMDD={avg_mdd:.1f}% bestCalmar={best_c:.3f}")

    pos = sum(1 for r in results if r['cagr_pct'] > 0)
    print(f"\nPositive: {pos}/{len(results)} ({pos/len(results)*100:.0f}%)")

    # Rolling 12m
    rolling = None
    if best_nav is not None and len(best_nav) > 52:
        print(f"\n=== Rolling 12m: {best_label} ===")
        rolling = rolling_12m(best_nav)
        if rolling:
            for k, v in rolling.items():
                print(f"  {k}: {v}")

    # Save
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'CSI500 Growth Factor Research (fast, annual data)',
        'growth_data_stocks': len(growth_data),
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
