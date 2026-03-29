#!/usr/bin/env python3
"""
A股个股因子研究 V3 - 历史成分股修正 (Survivorship-Bias Free)

核心改进:
  - 使用baostock获取每个调仓日的真实CSI300/CSI500成分股
  - 每月只用当时的真实成分股做选股，消除survivorship bias
  - 对比V2(当前成分股)结果，量化bias影响
  - 重点测试: 低换手+6M动量 (V2最佳策略)

Author: Sarah Mitchell / VisionClaw
Date: 2026-03-29
"""
from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import akshare as ak
import baostock as bs
import json, os, sys, time, pickle
from datetime import datetime
from collections import defaultdict

DATA_DIR = '/Users/claw/etf-trader/data'
CACHE_DIR = os.path.join(DATA_DIR, 'stock_cache_v3')
os.makedirs(CACHE_DIR, exist_ok=True)

# Reuse V2 price cache
PRICE_CACHE_DIR = os.path.join(DATA_DIR, 'stock_cache_v2')

RISK_FREE_RATE = 0.025
TC_ONE_SIDE = 0.003
START_DATE = '20150101'
END_DATE = '20260328'
BACKTEST_START = '2016-01-31'


# ============================================================
# 1. HISTORICAL CONSTITUENT DATA
# ============================================================

def get_historical_constituents(index_type='hs300', start_year=2016, end_year=2026):
    """
    Get historical constituents for CSI300 or CSI500 at each quarter-end.
    Returns dict: {date_str: set(codes)}
    """
    cache_file = os.path.join(CACHE_DIR, f'{index_type}_hist_constituents.pkl')
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 86400 * 7:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

    lg = bs.login()

    query_fn = bs.query_hs300_stocks if index_type == 'hs300' else bs.query_zz500_stocks

    # Query at each quarter-end and mid-quarter to capture changes
    dates = []
    for y in range(start_year, end_year + 1):
        for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            # Use last day of each month
            if m == 12:
                dates.append(f'{y}-{m:02d}-31')
            elif m in [1, 3, 5, 7, 8, 10]:
                dates.append(f'{y}-{m:02d}-31')
            elif m in [4, 6, 9, 11]:
                dates.append(f'{y}-{m:02d}-30')
            else:  # Feb
                dates.append(f'{y}-{m:02d}-28')

    constituents = {}
    prev_codes = None
    changes = 0

    for dt_str in dates:
        rs = query_fn(date=dt_str)
        codes = set()
        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            code = row[1].split('.')[1]  # sh.600000 -> 600000
            codes.add(code)

        if codes:
            constituents[dt_str] = codes
            if prev_codes and codes != prev_codes:
                changes += 1
            prev_codes = codes

    bs.logout()

    print(f"  {index_type}: {len(constituents)} dates, {changes} changes detected")

    with open(cache_file, 'wb') as f:
        pickle.dump(constituents, f)

    return constituents


def get_constituents_at_date(const_dict, target_date):
    """Get the most recent constituent list for a given date."""
    target_str = target_date.strftime('%Y-%m-%d')
    best_date = None
    for dt_str in sorted(const_dict.keys()):
        if dt_str <= target_str:
            best_date = dt_str
        else:
            break
    if best_date:
        return const_dict[best_date]
    return set()


# ============================================================
# 2. DATA LOADING (reuse V2 cache + expand universe)
# ============================================================

def fetch_stock_akshare(code):
    """Fetch daily data for a single stock via akshare (check V2 cache first)."""
    # Check V2 cache first
    cache_file = os.path.join(PRICE_CACHE_DIR, f'{code}_ak.pkl')
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 86400 * 3:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

    # Fall back to new cache
    new_cache = os.path.join(CACHE_DIR, f'{code}_ak.pkl')
    if os.path.exists(new_cache):
        mtime = os.path.getmtime(new_cache)
        if time.time() - mtime < 86400 * 3:
            with open(new_cache, 'rb') as f:
                return pickle.load(f)

    try:
        df = ak.stock_zh_a_hist(symbol=code, period='daily',
                                start_date=START_DATE, end_date=END_DATE, adjust='qfq')
        if df is None or len(df) < 100:
            return None
        df = df.rename(columns={
            '日期': 'date', '收盘': 'close', '开盘': 'open',
            '最高': 'high', '最低': 'low',
            '成交量': 'volume', '成交额': 'amount',
            '涨跌幅': 'pctChg', '换手率': 'turnover',
        })
        df['date'] = pd.to_datetime(df['date'])
        for c in ['close', 'open', 'high', 'low', 'volume', 'amount', 'pctChg', 'turnover']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.set_index('date').sort_index()
        df = df[df['volume'] > 0]
        with open(new_cache, 'wb') as f:
            pickle.dump(df, f)
        return df
    except:
        return None


def fetch_stock_valuation_bs(code):
    """Fetch PE/PB via baostock (check V2 cache first)."""
    for cache_dir in [PRICE_CACHE_DIR, CACHE_DIR]:
        cache_file = os.path.join(cache_dir, f'{code}_val.pkl')
        if os.path.exists(cache_file):
            mtime = os.path.getmtime(cache_file)
            if time.time() - mtime < 86400 * 3:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

    prefix = 'sh' if code.startswith('6') else 'sz'
    bs_code = f'{prefix}.{code}'
    try:
        rs = bs.query_history_k_data_plus(
            bs_code, fields='date,peTTM,pbMRQ',
            frequency='d', start_date='2015-01-01', end_date='2026-03-28',
            adjustflag='2')
        data = []
        while rs.error_code == '0' and rs.next():
            data.append(rs.get_row_data())
        if not data:
            return None
        df = pd.DataFrame(data, columns=['date', 'peTTM', 'pbMRQ'])
        df['date'] = pd.to_datetime(df['date'])
        df['peTTM'] = pd.to_numeric(df['peTTM'], errors='coerce')
        df['pbMRQ'] = pd.to_numeric(df['pbMRQ'], errors='coerce')
        df = df.set_index('date').sort_index()
        new_cache = os.path.join(CACHE_DIR, f'{code}_val.pkl')
        with open(new_cache, 'wb') as f:
            pickle.dump(df, f)
        return df
    except:
        return None


# ============================================================
# 3. FACTOR COMPUTATION
# ============================================================

def get_month_ends(all_data):
    all_dates = set()
    for code, df in all_data.items():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)
    me_df = pd.DataFrame({'date': all_dates})
    me_df = me_df[me_df['date'] >= pd.Timestamp(BACKTEST_START)]
    me_df['ym'] = me_df['date'].dt.to_period('M')
    month_ends = me_df.groupby('ym')['date'].max().tolist()
    return sorted(month_ends)


def compute_factors_for_stocks(all_data, val_data, month_ends, stock_codes):
    """Compute factors only for given stocks at each month end."""
    records = []
    for mi, dt in enumerate(month_ends):
        if (mi + 1) % 24 == 0:
            print(f"    Month {mi+1}/{len(month_ends)} ({dt.date()})...")
        for code in stock_codes:
            if code not in all_data:
                continue
            df = all_data[code]
            hist = df[df.index <= dt]
            if len(hist) < 120:
                continue
            if (dt - hist.index[-1]).days > 5:
                continue

            last = hist.iloc[-1]
            n = len(hist)
            rec = {'date': dt, 'code': code, 'close': last['close']}

            # SIZE
            rec['size'] = np.log(hist['amount'].iloc[-20:].mean() + 1)

            # RETURNS
            if n >= 22:
                rec['ret_1m'] = hist['close'].iloc[-1] / hist['close'].iloc[-22] - 1
            if n >= 66:
                rec['ret_3m'] = hist['close'].iloc[-1] / hist['close'].iloc[-66] - 1
            if n >= 132:
                rec['mom_6m'] = hist['close'].iloc[-22] / hist['close'].iloc[-132] - 1
            if n >= 264:
                rec['mom_12m'] = hist['close'].iloc[-22] / hist['close'].iloc[-264] - 1

            # VOLATILITY
            if n >= 20:
                daily_ret = hist['pctChg'].iloc[-20:] / 100.0
                rec['vol_20d'] = daily_ret.std() * np.sqrt(252)

            # TURNOVER
            if n >= 20:
                rec['turn_20d'] = hist['turnover'].iloc[-20:].mean()

            # ILLIQUIDITY
            if n >= 20:
                recent = hist.iloc[-20:]
                amihud = (recent['pctChg'].abs() / 100.0 / (recent['amount'] + 1)).mean()
                rec['illiquidity'] = amihud

            # VALUATION
            if code in val_data:
                vdf = val_data[code]
                vhist = vdf[vdf.index <= dt]
                if len(vhist) > 0:
                    vlast = vhist.iloc[-1]
                    pe = vlast.get('peTTM', np.nan)
                    pb = vlast.get('pbMRQ', np.nan)
                    if pd.notna(pe) and pe > 0:
                        rec['ep'] = 1.0 / pe
                    if pd.notna(pb) and pb > 0:
                        rec['bp'] = 1.0 / pb

            records.append(rec)

    return pd.DataFrame(records)


# ============================================================
# 4. PORTFOLIO BACKTEST WITH HISTORICAL CONSTITUENTS
# ============================================================

def build_composite_signal(cur, factor_weights):
    """Build composite signal from multiple factors."""
    composite = pd.Series(0.0, index=cur.index)
    valid_mask = pd.Series(True, index=cur.index)
    for fname, (weight, ascending) in factor_weights.items():
        if fname not in cur.columns:
            return None
        vals = cur[fname]
        valid_mask &= vals.notna()
        if ascending:
            rank_pct = vals.rank(ascending=True, pct=True)
        else:
            rank_pct = vals.rank(ascending=False, pct=True)
        composite += weight * rank_pct
    composite[~valid_mask] = np.nan
    return composite


def backtest_with_hist_constituents(all_data, val_data, month_ends,
                                     hs300_const, zz500_const,
                                     factor_weights, strategy_name,
                                     universe='CSI300', top_n=30,
                                     rebal_months=1, tc=TC_ONE_SIDE):
    """
    Backtest using historical constituents at each rebalance date.
    """
    nav = 1.0
    nav_history = []
    prev_holdings = set()
    rebal_count = 0

    for i in range(len(month_ends) - 1):
        dt = month_ends[i]
        dt_next = month_ends[i + 1]

        # Get constituents at this date
        if universe == 'CSI300':
            eligible = get_constituents_at_date(hs300_const, dt)
        elif universe == 'CSI500':
            eligible = get_constituents_at_date(zz500_const, dt)
        else:  # CSI800
            eligible = get_constituents_at_date(hs300_const, dt) | get_constituents_at_date(zz500_const, dt)

        if not eligible:
            nav_history.append({'date': dt, 'nav': nav})
            rebal_count += 1
            continue

        do_rebal = (rebal_count % rebal_months == 0)
        rebal_count += 1

        if do_rebal:
            # Compute factors for eligible stocks at this date
            eligible_with_data = [c for c in eligible if c in all_data]
            if len(eligible_with_data) < top_n:
                nav_history.append({'date': dt, 'nav': nav})
                continue

            # Build factor data for this month
            records = []
            for code in eligible_with_data:
                df = all_data[code]
                hist = df[df.index <= dt]
                if len(hist) < 120:
                    continue
                if (dt - hist.index[-1]).days > 5:
                    continue

                last = hist.iloc[-1]
                n = len(hist)
                rec = {'code': code, 'close': last['close']}

                if n >= 22:
                    rec['ret_1m'] = hist['close'].iloc[-1] / hist['close'].iloc[-22] - 1
                if n >= 66:
                    rec['ret_3m'] = hist['close'].iloc[-1] / hist['close'].iloc[-66] - 1
                if n >= 132:
                    rec['mom_6m'] = hist['close'].iloc[-22] / hist['close'].iloc[-132] - 1
                if n >= 264:
                    rec['mom_12m'] = hist['close'].iloc[-22] / hist['close'].iloc[-264] - 1
                if n >= 20:
                    daily_ret = hist['pctChg'].iloc[-20:] / 100.0
                    rec['vol_20d'] = daily_ret.std() * np.sqrt(252)
                if n >= 20:
                    rec['turn_20d'] = hist['turnover'].iloc[-20:].mean()
                if n >= 20:
                    recent = hist.iloc[-20:]
                    rec['illiquidity'] = (recent['pctChg'].abs() / 100.0 / (recent['amount'] + 1)).mean()

                # Valuation
                if code in val_data:
                    vdf = val_data[code]
                    vhist = vdf[vdf.index <= dt]
                    if len(vhist) > 0:
                        vlast = vhist.iloc[-1]
                        pe = vlast.get('peTTM', np.nan)
                        pb = vlast.get('pbMRQ', np.nan)
                        if pd.notna(pe) and pe > 0:
                            rec['ep'] = 1.0 / pe
                        if pd.notna(pb) and pb > 0:
                            rec['bp'] = 1.0 / pb

                records.append(rec)

            if len(records) < top_n:
                nav_history.append({'date': dt, 'nav': nav})
                continue

            cur = pd.DataFrame(records)
            composite = build_composite_signal(cur, factor_weights)
            if composite is None or composite.notna().sum() < top_n:
                nav_history.append({'date': dt, 'nav': nav})
                continue

            cur['signal'] = composite.values
            cur = cur.dropna(subset=['signal'])
            top = cur.nsmallest(top_n, 'signal')
            new_holdings = set(top['code'].tolist())
            held_prices = dict(zip(top['code'], top['close']))
        else:
            new_holdings = prev_holdings.copy()
            held_prices = {}
            for code in new_holdings:
                if code in all_data:
                    df = all_data[code]
                    hist = df[df.index <= dt]
                    if len(hist) > 0:
                        held_prices[code] = hist.iloc[-1]['close']

        # Get forward returns
        fwd_rets = []
        for code in new_holdings:
            if code not in all_data:
                continue
            df = all_data[code]
            # Get price at dt and dt_next
            hist_cur = df[df.index <= dt]
            hist_nxt = df[df.index <= dt_next]
            if len(hist_cur) == 0 or len(hist_nxt) == 0:
                continue
            p_cur = hist_cur.iloc[-1]['close']
            p_nxt = hist_nxt.iloc[-1]['close']
            if p_cur > 0:
                fwd_rets.append(p_nxt / p_cur - 1)

        if not fwd_rets:
            nav_history.append({'date': dt, 'nav': nav})
            continue

        # Winsorize
        fwd_arr = np.array(fwd_rets)
        p01, p99 = np.percentile(fwd_arr, [1, 99])
        fwd_arr = np.clip(fwd_arr, p01, p99)
        port_ret = fwd_arr.mean()

        # TC
        if do_rebal and prev_holdings:
            turnover = len(new_holdings - prev_holdings) / max(len(new_holdings), 1)
            tc_cost = turnover * tc * 2
        elif not prev_holdings:
            tc_cost = tc
        else:
            tc_cost = 0

        nav *= (1 + port_ret - tc_cost)
        nav_history.append({'date': dt, 'nav': nav})
        prev_holdings = new_holdings

    if not nav_history or len(nav_history) < 12:
        return None

    nav_df = pd.DataFrame(nav_history).set_index('date')
    total_days = (nav_df.index[-1] - nav_df.index[0]).days
    years = total_days / 365.25
    if years < 1:
        return None

    cagr = (nav_df['nav'].iloc[-1]) ** (1 / years) - 1
    monthly_ret = nav_df['nav'].pct_change().dropna()
    ann_vol = monthly_ret.std() * np.sqrt(12)
    sharpe = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    dd = nav_df['nav'] / nav_df['nav'].cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Yearly returns
    nav_df['year'] = nav_df.index.year
    yearly = {}
    for year, grp in nav_df.groupby('year'):
        if len(grp) >= 2:
            yr_ret = grp['nav'].iloc[-1] / grp['nav'].iloc[0] - 1
            yearly[int(year)] = round(yr_ret * 100, 2)

    return {
        'strategy': strategy_name,
        'CAGR': round(cagr * 100, 2),
        'AnnVol': round(ann_vol * 100, 2),
        'Sharpe': round(sharpe, 3),
        'MaxDD': round(max_dd * 100, 2),
        'Calmar': round(calmar, 3),
        'FinalNAV': round(nav_df['nav'].iloc[-1], 4),
        'years': round(years, 1),
        'yearly_returns': yearly,
    }


def equal_weight_hist_benchmark(all_data, month_ends, const_dict, universe_name):
    """Equal weight benchmark using historical constituents."""
    nav = 1.0
    nav_history = []

    for i in range(len(month_ends) - 1):
        dt = month_ends[i]
        dt_next = month_ends[i + 1]
        eligible = get_constituents_at_date(const_dict, dt)

        if not eligible:
            nav_history.append({'date': dt, 'nav': nav})
            continue

        fwd_rets = []
        for code in eligible:
            if code not in all_data:
                continue
            df = all_data[code]
            hist_cur = df[df.index <= dt]
            hist_nxt = df[df.index <= dt_next]
            if len(hist_cur) == 0 or len(hist_nxt) == 0:
                continue
            p_cur = hist_cur.iloc[-1]['close']
            p_nxt = hist_nxt.iloc[-1]['close']
            if p_cur > 0:
                fwd_rets.append(p_nxt / p_cur - 1)

        if not fwd_rets:
            nav_history.append({'date': dt, 'nav': nav})
            continue

        fwd_arr = np.array(fwd_rets)
        p01, p99 = np.percentile(fwd_arr, [1, 99])
        fwd_arr = np.clip(fwd_arr, p01, p99)
        nav *= (1 + fwd_arr.mean())
        nav_history.append({'date': dt, 'nav': nav})

    if not nav_history or len(nav_history) < 12:
        return None

    nav_df = pd.DataFrame(nav_history).set_index('date')
    total_days = (nav_df.index[-1] - nav_df.index[0]).days
    years = total_days / 365.25
    cagr = (nav_df['nav'].iloc[-1]) ** (1 / years) - 1
    monthly_ret = nav_df['nav'].pct_change().dropna()
    ann_vol = monthly_ret.std() * np.sqrt(12)
    sharpe = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    dd = nav_df['nav'] / nav_df['nav'].cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    nav_df['year'] = nav_df.index.year
    yearly = {}
    for year, grp in nav_df.groupby('year'):
        if len(grp) >= 2:
            yr_ret = grp['nav'].iloc[-1] / grp['nav'].iloc[0] - 1
            yearly[int(year)] = round(yr_ret * 100, 2)

    return {
        'strategy': f'EW_{universe_name}_hist',
        'CAGR': round(cagr * 100, 2),
        'AnnVol': round(ann_vol * 100, 2),
        'Sharpe': round(sharpe, 3),
        'MaxDD': round(max_dd * 100, 2),
        'Calmar': round(calmar, 3),
        'FinalNAV': round(nav_df['nav'].iloc[-1], 4),
        'years': round(years, 1),
        'yearly_returns': yearly,
    }


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("A股个股因子研究 V3 - 历史成分股修正")
    print("=" * 70)
    t_start = time.time()

    # Step 1: Get historical constituents
    print("\n=== Step 1: Historical Constituents ===")
    hs300_const = get_historical_constituents('hs300', 2016, 2026)
    zz500_const = get_historical_constituents('zz500', 2016, 2026)

    # Collect all codes that were ever in CSI300 or CSI500
    all_codes_ever = set()
    for codes in hs300_const.values():
        all_codes_ever |= codes
    for codes in zz500_const.values():
        all_codes_ever |= codes
    print(f"  Total unique codes ever in CSI800: {len(all_codes_ever)}")

    # Step 2: Load price data for all historical constituents
    print(f"\n=== Step 2: Loading Price Data ({len(all_codes_ever)} stocks) ===")
    all_data = {}
    failed = 0
    codes_list = sorted(all_codes_ever)
    for i, code in enumerate(codes_list):
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(codes_list)} ...")
        df = fetch_stock_akshare(code)
        if df is not None and len(df) > 60:
            all_data[code] = df
        else:
            failed += 1
        time.sleep(0.02)
    print(f"  Loaded {len(all_data)} stocks, failed: {failed}")

    # Step 3: Load valuation data
    print(f"\n=== Step 3: Loading Valuation Data ===")
    val_data = {}
    need_bs = []
    for code in all_data.keys():
        found = False
        for cache_dir in [PRICE_CACHE_DIR, CACHE_DIR]:
            cache_file = os.path.join(cache_dir, f'{code}_val.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    val_data[code] = pickle.load(f)
                found = True
                break
        if not found:
            need_bs.append(code)

    if need_bs:
        print(f"  Need to fetch {len(need_bs)} valuations via baostock...")
        lg = bs.login()
        for i, code in enumerate(need_bs):
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(need_bs)} ...")
            df = fetch_stock_valuation_bs(code)
            if df is not None:
                val_data[code] = df
        bs.logout()

    print(f"  Loaded valuation for {len(val_data)} stocks")

    # Step 4: Month ends
    month_ends = get_month_ends(all_data)
    print(f"\n  Month-ends: {len(month_ends)} ({month_ends[0].date()} ~ {month_ends[-1].date()})")

    # Step 5: Benchmarks with historical constituents
    print(f"\n=== Step 4: Historical Constituent Benchmarks ===")
    bm_300 = equal_weight_hist_benchmark(all_data, month_ends, hs300_const, 'CSI300')
    bm_500 = equal_weight_hist_benchmark(all_data, month_ends, zz500_const, 'CSI500')

    # CSI800 = union
    hs800_const = {}
    for dt_str in sorted(set(list(hs300_const.keys()) + list(zz500_const.keys()))):
        c300 = hs300_const.get(dt_str, set())
        c500 = zz500_const.get(dt_str, set())
        if c300 or c500:
            hs800_const[dt_str] = c300 | c500
    bm_800 = equal_weight_hist_benchmark(all_data, month_ends, hs800_const, 'CSI800')

    print(f"  EW CSI300 (hist): CAGR={bm_300['CAGR']:.1f}%, Sharpe={bm_300['Sharpe']:.3f}, MaxDD={bm_300['MaxDD']:.1f}%")
    print(f"  EW CSI500 (hist): CAGR={bm_500['CAGR']:.1f}%, Sharpe={bm_500['Sharpe']:.3f}, MaxDD={bm_500['MaxDD']:.1f}%")
    print(f"  EW CSI800 (hist): CAGR={bm_800['CAGR']:.1f}%, Sharpe={bm_800['Sharpe']:.3f}, MaxDD={bm_800['MaxDD']:.1f}%")
    print(f"  Yearly: CSI300={bm_300['yearly_returns']}")
    print(f"  Yearly: CSI500={bm_500['yearly_returns']}")

    # Step 6: Strategy tests
    print(f"\n=== Step 5: Strategy Tests (Historical Constituents) ===")

    # Define strategies to test
    strategies = {
        # Single factors
        'LowTurn': {'turn_20d': (1.0, True)},
        'Mom6M': {'mom_6m': (1.0, False)},
        'Mom12M': {'mom_12m': (1.0, False)},
        'LowVol': {'vol_20d': (1.0, True)},
        'Value_EP': {'ep': (1.0, False)},
        'Value_BP': {'bp': (1.0, False)},
        # Best combos from V2
        'LowTurn+Mom6M': {'turn_20d': (1.0, True), 'mom_6m': (1.0, False)},
        'LowTurn+Mom12M': {'turn_20d': (1.0, True), 'mom_12m': (1.0, False)},
        'LowTurn+BP': {'turn_20d': (1.0, True), 'bp': (1.0, False)},
        'LowTurn+EP': {'turn_20d': (1.0, True), 'ep': (1.0, False)},
        'LowVol+Mom6M': {'vol_20d': (1.0, True), 'mom_6m': (1.0, False)},
        'Mom6M+EP': {'mom_6m': (1.0, False), 'ep': (1.0, False)},
        'Mom6M+BP': {'mom_6m': (1.0, False), 'bp': (1.0, False)},
        # Triple
        'LowTurn+Mom6M+EP': {'turn_20d': (1.0, True), 'mom_6m': (1.0, False), 'ep': (1.0, False)},
        'LowTurn+Mom6M+BP': {'turn_20d': (1.0, True), 'mom_6m': (1.0, False), 'bp': (1.0, False)},
        'LowTurn+LowVol+Mom6M': {'turn_20d': (1.0, True), 'vol_20d': (1.0, True), 'mom_6m': (1.0, False)},
        'LowTurn+Mom12M+EP+LowVol': {
            'turn_20d': (1.0, True), 'mom_12m': (1.0, False),
            'ep': (1.0, False), 'vol_20d': (1.0, True)
        },
    }

    all_results = []

    # Test across universes and top_n
    configs = [
        ('CSI300', 30, '1M'),
        ('CSI300', 50, '1M'),
        ('CSI800', 30, '1M'),
        ('CSI800', 50, '1M'),
        ('CSI800', 100, '1M'),
        ('CSI500', 50, '1M'),
        # Quarterly
        ('CSI300', 30, '3M'),
        ('CSI800', 50, '3M'),
    ]

    for universe, top_n, rebal in configs:
        rebal_m = 1 if rebal == '1M' else 3
        print(f"\n--- {universe}, top {top_n}, {rebal} rebal ---")
        for name, weights in strategies.items():
            strat_name = f'{name}_{universe}_top{top_n}_{rebal}'
            r = backtest_with_hist_constituents(
                all_data, val_data, month_ends,
                hs300_const, zz500_const,
                weights, strat_name,
                universe=universe, top_n=top_n,
                rebal_months=rebal_m
            )
            if r:
                r['universe'] = universe
                r['top_n'] = top_n
                r['rebal'] = rebal
                r['factor_combo'] = name
                all_results.append(r)
                print(f"  {strat_name:<45s} CAGR={r['CAGR']:>6.1f}% Sharpe={r['Sharpe']:>6.3f} MaxDD={r['MaxDD']:>6.1f}%")
            else:
                print(f"  {strat_name:<45s} FAILED (not enough data)")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: ALL STRATEGIES (Historical Constituents)")
    print("=" * 70)

    # Sort by Sharpe
    all_results.sort(key=lambda x: x['Sharpe'], reverse=True)

    hdr = f"{'Strategy':<50s} {'CAGR':>7s} {'Vol':>6s} {'Sharpe':>7s} {'MaxDD':>7s} {'Calmar':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_results[:30]:
        print(f"{r['strategy']:<50s} {r['CAGR']:>6.1f}% {r['AnnVol']:>5.1f}% {r['Sharpe']:>7.3f} {r['MaxDD']:>6.1f}% {r['Calmar']:>7.3f}")

    print(f"\n{'--- BENCHMARKS (Historical Constituents) ---'}")
    for bm, label in [(bm_800, 'EW_CSI800'), (bm_300, 'EW_CSI300'), (bm_500, 'EW_CSI500')]:
        print(f"  {label:<30s} CAGR={bm['CAGR']:>6.1f}% Sharpe={bm['Sharpe']:>6.3f} MaxDD={bm['MaxDD']:>6.1f}%")

    # V2 vs V3 comparison
    print(f"\n{'--- V2 (Current Constituents) vs V3 (Historical) ---'}")
    print(f"  V2 EW CSI300: CAGR=22.6%, Sharpe=0.983  (from V2 results)")
    print(f"  V3 EW CSI300: CAGR={bm_300['CAGR']:.1f}%, Sharpe={bm_300['Sharpe']:.3f}")
    print(f"  V2 Best CSI300: LowTurn+Mom6M top30 Sharpe=1.086, CAGR=22.7%")

    # Find best CSI300 in V3
    csi300_v3 = [r for r in all_results if r['universe'] == 'CSI300' and r['rebal'] == '1M']
    if csi300_v3:
        best = csi300_v3[0]
        print(f"  V3 Best CSI300: {best['factor_combo']} top{best['top_n']} Sharpe={best['Sharpe']:.3f}, CAGR={best['CAGR']:.1f}%")

    # Yearly returns for best strategy
    if all_results:
        best = all_results[0]
        print(f"\n  Best Strategy Yearly Returns ({best['strategy']}):")
        for yr, ret in sorted(best.get('yearly_returns', {}).items()):
            print(f"    {yr}: {ret:>6.1f}%")

    # Save results
    out_path = os.path.join(DATA_DIR, 'stock_factor_v3_results.json')
    save_data = {
        'benchmarks_hist': {
            'EW_CSI800': bm_800,
            'EW_CSI300': bm_300,
            'EW_CSI500': bm_500,
        },
        'strategies': all_results,
        'total_unique_stocks': len(all_codes_ever),
        'loaded_stocks': len(all_data),
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Total: {len(all_results)} strategies tested in {total_time:.0f}s")
    print(f"Results saved to {out_path}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
