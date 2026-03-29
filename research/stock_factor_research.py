#!/usr/bin/env python3
"""
A股个股因子截面回测研究 (Individual Stock Factor Cross-Sectional Research)

研究内容:
  1. 市值因子 (Size) - 日均成交额代理
  2. 短期反转 (1M return)
  3. 中期反转 (3M return)
  4. 动量因子 (6M skip 1M, 12M skip 1M)
  5. 低波因子 (20d realized vol)
  6. 换手率因子 (20d avg turnover)
  7. 非流动性 (Amihud)
  8. 量比 (5d/60d volume ratio)
  9. 估值因子 (PE/PB) - 用baostock补充

宇宙: CSI800 (沪深300+中证500)
周期: 2016-01 ~ 2026-03
频率: 月度调仓, 五分位组合

数据源: akshare (价量) + baostock (成分股/估值)

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

DATA_DIR = '/Users/claw/etf-trader/data'
CACHE_DIR = os.path.join(DATA_DIR, 'stock_cache_v2')
os.makedirs(CACHE_DIR, exist_ok=True)

RISK_FREE_RATE = 0.025
TC_ONE_SIDE = 0.0015

START_DATE = '20150101'
END_DATE = '20260328'
BACKTEST_START = '2016-01-31'


# ============================================================
# 1. GET CSI800 UNIVERSE
# ============================================================

def get_csi800_codes():
    """Get CSI300+CSI500 constituent codes."""
    cache_file = os.path.join(CACHE_DIR, 'csi800_codes.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    lg = bs.login()

    codes = set()
    for query_fn in [bs.query_hs300_stocks, bs.query_zz500_stocks]:
        rs = query_fn(date='2024-12-31')
        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            # Convert sh.600000 -> 600000
            code = row[1].split('.')[1]
            codes.add(code)

    bs.logout()

    codes = sorted(codes)
    with open(cache_file, 'wb') as f:
        pickle.dump(codes, f)

    print(f"  CSI800 universe: {len(codes)} stocks")
    return codes


# ============================================================
# 2. FETCH DATA VIA AKSHARE (FAST)
# ============================================================

def fetch_stock_akshare(code):
    """Fetch daily data for a single stock via akshare."""
    cache_file = os.path.join(CACHE_DIR, f'{code}_ak.pkl')
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 86400 * 3:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

    try:
        df = ak.stock_zh_a_hist(symbol=code, period='daily',
                                start_date=START_DATE, end_date=END_DATE,
                                adjust='qfq')
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
        df = df[df['volume'] > 0]  # remove suspended days

        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)

        return df
    except Exception as e:
        return None


def fetch_stock_valuation_bs(code):
    """Fetch PE/PB via baostock for a single stock."""
    cache_file = os.path.join(CACHE_DIR, f'{code}_val.pkl')
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
            adjustflag='2'
        )
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

        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        return df
    except:
        return None


def load_all_data(codes):
    """Load all stock data."""
    print("  Fetching price/volume data via akshare...")
    all_data = {}
    failed = 0

    for i, code in enumerate(codes):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(codes)} ...")
        df = fetch_stock_akshare(code)
        if df is not None and len(df) > 120:
            all_data[code] = df
        else:
            failed += 1
        time.sleep(0.05)  # small delay to avoid rate limit

    print(f"  Loaded {len(all_data)} stocks, failed: {failed}")
    return all_data


def load_valuation_data(codes):
    """Load PE/PB data via baostock for all stocks."""
    print("  Fetching PE/PB data via baostock...")
    lg = bs.login()

    val_data = {}
    for i, code in enumerate(codes):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(codes)} ...")
        df = fetch_stock_valuation_bs(code)
        if df is not None:
            val_data[code] = df

    bs.logout()
    print(f"  Loaded valuation for {len(val_data)} stocks")
    return val_data


# ============================================================
# 3. FACTOR COMPUTATION
# ============================================================

def get_month_ends(all_data):
    """Get all month-end trading dates."""
    all_dates = set()
    for code, df in all_data.items():
        all_dates.update(df.index.tolist())

    all_dates = sorted(all_dates)
    me_df = pd.DataFrame({'date': all_dates})
    me_df = me_df[me_df['date'] >= pd.Timestamp(BACKTEST_START)]
    me_df['ym'] = me_df['date'].dt.to_period('M')
    month_ends = me_df.groupby('ym')['date'].max().tolist()
    return sorted(month_ends)


def compute_factors(all_data, val_data, month_ends):
    """Compute factor values at each month-end for all stocks."""
    records = []

    for mi, dt in enumerate(month_ends):
        if (mi + 1) % 12 == 0:
            print(f"    Month {mi+1}/{len(month_ends)} ({dt.date()})...")

        for code, df in all_data.items():
            hist = df[df.index <= dt]
            if len(hist) < 120:
                continue

            # Check stock is actively traded near month-end
            if (dt - hist.index[-1]).days > 5:
                continue

            last = hist.iloc[-1]
            n = len(hist)
            rec = {'date': dt, 'code': code, 'close': last['close']}

            # SIZE: log daily amount (proxy for market cap)
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

            # LOW VOLATILITY
            if n >= 20:
                daily_ret = hist['pctChg'].iloc[-20:] / 100.0
                rec['vol_20d'] = daily_ret.std() * np.sqrt(252)

            # TURNOVER
            if n >= 20:
                rec['turn_20d'] = hist['turnover'].iloc[-20:].mean()

            # ILLIQUIDITY (Amihud)
            if n >= 20:
                recent = hist.iloc[-20:]
                amihud = (recent['pctChg'].abs() / 100.0 / (recent['amount'] + 1)).mean()
                rec['illiquidity'] = amihud

            # VOLUME TREND (5d/60d)
            if n >= 60:
                v5 = hist['volume'].iloc[-5:].mean()
                v60 = hist['volume'].iloc[-60:].mean()
                rec['vol_trend'] = v5 / v60 if v60 > 0 else np.nan

            # VALUATION (PE/PB from baostock)
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
# 4. QUINTILE BACKTEST
# ============================================================

def quintile_backtest(factor_df, factor_name, ascending=True, n_q=5):
    """Run quintile sort backtest for a factor."""
    dates = sorted(factor_df['date'].unique())

    results_list = []

    for i in range(len(dates) - 1):
        dt = dates[i]
        dt_next = dates[i + 1]

        cur = factor_df[factor_df['date'] == dt].copy()
        cur = cur.dropna(subset=[factor_name])

        if len(cur) < n_q * 10:
            continue

        # Get forward return
        nxt = factor_df[factor_df['date'] == dt_next][['code', 'close']].rename(
            columns={'close': 'close_next'})
        cur = cur.merge(nxt, on='code', how='inner')
        cur['fwd_ret'] = cur['close_next'] / cur['close'] - 1

        if len(cur) < n_q * 10:
            continue

        # Winsorize fwd_ret at 1st/99th percentile to reduce outlier impact
        p01, p99 = cur['fwd_ret'].quantile([0.01, 0.99])
        cur['fwd_ret'] = cur['fwd_ret'].clip(p01, p99)

        # Rank and quintile
        cur['rank'] = cur[factor_name].rank(ascending=ascending, method='first')
        cur['q'] = pd.qcut(cur['rank'], n_q, labels=False) + 1

        # IC
        ic = cur[factor_name].corr(cur['fwd_ret'], method='spearman')

        row = {'date': dt, 'IC': ic, 'n': len(cur)}
        for q in range(1, n_q + 1):
            row[f'Q{q}'] = cur[cur['q'] == q]['fwd_ret'].mean()
        row['LS'] = row[f'Q{n_q}'] - row['Q1']

        results_list.append(row)

    if not results_list:
        return None

    rdf = pd.DataFrame(results_list).set_index('date')

    # Compute NAVs and metrics
    metrics = {}
    for col in [f'Q{q}' for q in range(1, n_q + 1)] + ['LS']:
        nav = (1 + rdf[col]).cumprod()
        total_days = (nav.index[-1] - nav.index[0]).days
        years = total_days / 365.25
        if years < 1 or nav.iloc[-1] <= 0:
            continue

        cagr = (nav.iloc[-1]) ** (1 / years) - 1
        monthly_ret = rdf[col]
        ann_vol = monthly_ret.std() * np.sqrt(12)
        sharpe = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
        dd = nav / nav.cummax() - 1
        max_dd = dd.min()

        metrics[col] = {
            'CAGR': round(cagr * 100, 2),
            'AnnVol': round(ann_vol * 100, 2),
            'Sharpe': round(sharpe, 3),
            'MaxDD': round(max_dd * 100, 2),
        }

    ic_s = rdf['IC']
    return {
        'quintile_metrics': metrics,
        'avg_IC': round(ic_s.mean(), 4),
        'ICIR': round(ic_s.mean() / ic_s.std(), 4) if ic_s.std() > 0 else 0,
        'IC_positive_pct': round((ic_s > 0).mean() * 100, 1),
        'avg_stocks': int(rdf['n'].mean()),
        'n_months': len(rdf),
    }


# ============================================================
# 5. MULTI-FACTOR COMBO
# ============================================================

def compute_combo_factors(factor_df):
    """Compute composite factor signals."""
    dates = sorted(factor_df['date'].unique())

    combo_records = []
    for dt in dates:
        cur = factor_df[factor_df['date'] == dt].copy()
        if len(cur) < 50:
            continue

        # Rank each factor (percentile rank within cross-section)
        # For "good direction": higher rank = better
        rank_cols = {}

        # Small size is good → ascending rank (small = high rank)
        if 'size' in cur.columns:
            rank_cols['size_rank'] = cur['size'].rank(ascending=False, pct=True)

        # High EP/BP is good
        if 'ep' in cur.columns:
            rank_cols['ep_rank'] = cur['ep'].rank(ascending=True, pct=True)
        if 'bp' in cur.columns:
            rank_cols['bp_rank'] = cur['bp'].rank(ascending=True, pct=True)

        # Low ret_1m is good (reversal)
        if 'ret_1m' in cur.columns:
            rank_cols['rev_1m_rank'] = cur['ret_1m'].rank(ascending=False, pct=True)

        # Low volatility is good
        if 'vol_20d' in cur.columns:
            rank_cols['lowvol_rank'] = cur['vol_20d'].rank(ascending=False, pct=True)

        # Low turnover is good
        if 'turn_20d' in cur.columns:
            rank_cols['lowturn_rank'] = cur['turn_20d'].rank(ascending=False, pct=True)

        for col, vals in rank_cols.items():
            cur[col] = vals

        # COMBO 1: Reversal + Low Vol
        if 'rev_1m_rank' in rank_cols and 'lowvol_rank' in rank_cols:
            cur['combo_rev_lowvol'] = cur['rev_1m_rank'] + cur['lowvol_rank']

        # COMBO 2: Reversal + Low Turnover
        if 'rev_1m_rank' in rank_cols and 'lowturn_rank' in rank_cols:
            cur['combo_rev_lowturn'] = cur['rev_1m_rank'] + cur['lowturn_rank']

        # COMBO 3: Value + Low Vol
        if 'bp_rank' in rank_cols and 'lowvol_rank' in rank_cols:
            cur['combo_value_lowvol'] = cur['bp_rank'] + cur['lowvol_rank']

        # COMBO 4: Value + Reversal + Low Vol (triple factor)
        if all(k in rank_cols for k in ['bp_rank', 'rev_1m_rank', 'lowvol_rank']):
            cur['combo_val_rev_lowvol'] = cur['bp_rank'] + cur['rev_1m_rank'] + cur['lowvol_rank']

        # COMBO 5: Reversal + Low Vol + Low Turnover
        if all(k in rank_cols for k in ['rev_1m_rank', 'lowvol_rank', 'lowturn_rank']):
            cur['combo_rev_lowvol_lowturn'] = cur['rev_1m_rank'] + cur['lowvol_rank'] + cur['lowturn_rank']

        # COMBO 6: Small + Value + Reversal
        if all(k in rank_cols for k in ['size_rank', 'bp_rank', 'rev_1m_rank']):
            cur['combo_small_val_rev'] = cur['size_rank'] + cur['bp_rank'] + cur['rev_1m_rank']

        combo_records.append(cur)

    return pd.concat(combo_records, ignore_index=True)


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("A股个股因子截面回测研究 (CSI800)")
    print("=" * 70)

    # Step 1: Universe
    print("\n=== Step 1: Universe ===")
    codes = get_csi800_codes()

    # Step 2: Price/Volume data
    print(f"\n=== Step 2: Loading price data ({len(codes)} stocks) ===")
    t0 = time.time()
    all_data = load_all_data(codes)
    print(f"  Price data loaded in {time.time()-t0:.0f}s")

    # Step 3: Valuation data
    print(f"\n=== Step 3: Loading valuation data ===")
    t0 = time.time()
    val_data = load_valuation_data(list(all_data.keys()))
    print(f"  Valuation data loaded in {time.time()-t0:.0f}s")

    # Step 4: Compute factors
    print(f"\n=== Step 4: Computing monthly factors ===")
    month_ends = get_month_ends(all_data)
    print(f"  Month-ends: {len(month_ends)} ({month_ends[0].date()} ~ {month_ends[-1].date()})")

    t0 = time.time()
    factor_df = compute_factors(all_data, val_data, month_ends)
    print(f"  Factor computation: {time.time()-t0:.0f}s, {len(factor_df)} records")
    print(f"  Avg stocks/month: {len(factor_df)/len(month_ends):.0f}")

    # Step 5: Single factor backtests
    print(f"\n=== Step 5: Single Factor Backtests ===")

    factors = [
        ('size', True, '市值(成交额代理)', 'Q1=small'),
        ('ret_1m', True, '1月收益(反转)', 'Q1=losers'),
        ('ret_3m', True, '3月收益(反转)', 'Q1=losers'),
        ('mom_6m', True, '6月动量(skip1M)', 'Q1=low_mom'),
        ('mom_12m', True, '12月动量(skip1M)', 'Q1=low_mom'),
        ('vol_20d', True, '20日波动率', 'Q1=low_vol'),
        ('turn_20d', True, '20日换手率', 'Q1=low_turn'),
        ('illiquidity', True, '非流动性Amihud', 'Q1=liquid'),
        ('vol_trend', True, '量比5d/60d', 'Q1=low_vol_trend'),
        ('ep', True, '盈利收益率1/PE', 'Q1=low_ep'),
        ('bp', True, '账面市值比1/PB', 'Q1=low_bp'),
    ]

    all_results = {}

    for fname, asc, desc, note in factors:
        if fname not in factor_df.columns:
            print(f"\n  SKIP {desc}: no data")
            continue

        valid_pct = factor_df[fname].notna().mean() * 100
        if valid_pct < 30:
            print(f"\n  SKIP {desc}: only {valid_pct:.0f}% valid")
            continue

        print(f"\n--- {desc} ({fname}) [{note}] ---")
        result = quintile_backtest(factor_df, fname, ascending=asc)

        if result is None:
            print("  Not enough data")
            continue

        result['factor'] = fname
        result['description'] = desc
        all_results[fname] = result

        print(f"  IC: {result['avg_IC']:.4f}, ICIR: {result['ICIR']:.4f}, IC>0: {result['IC_positive_pct']:.1f}%")
        print(f"  Stocks/month: {result['avg_stocks']}, Months: {result['n_months']}")

        qm = result['quintile_metrics']
        for q in ['Q1', 'Q3', 'Q5', 'LS']:
            if q in qm:
                print(f"  {q}: CAGR={qm[q]['CAGR']:>6.1f}%, Sharpe={qm[q]['Sharpe']:>6.3f}, MaxDD={qm[q]['MaxDD']:>6.1f}%")

        # Monotonicity
        cagrs = [qm[f'Q{q}']['CAGR'] for q in range(1, 6) if f'Q{q}' in qm]
        if len(cagrs) == 5:
            mono = all(cagrs[i+1] >= cagrs[i] for i in range(4)) or \
                   all(cagrs[i+1] <= cagrs[i] for i in range(4))
            print(f"  Quintiles: {['%.1f' % c for c in cagrs]} {'MONOTONIC' if mono else ''}")

    # Step 6: Combo factors
    print(f"\n=== Step 6: Multi-Factor Combos ===")
    combo_df = compute_combo_factors(factor_df)

    combo_factors = [
        ('combo_rev_lowvol', True, '反转+低波'),
        ('combo_rev_lowturn', True, '反转+低换手'),
        ('combo_value_lowvol', True, '价值+低波'),
        ('combo_val_rev_lowvol', True, '价值+反转+低波'),
        ('combo_rev_lowvol_lowturn', True, '反转+低波+低换手'),
        ('combo_small_val_rev', True, '小市值+价值+反转'),
    ]

    for fname, asc, desc in combo_factors:
        if fname not in combo_df.columns:
            continue

        print(f"\n--- {desc} ({fname}) ---")
        result = quintile_backtest(combo_df, fname, ascending=asc)

        if result is None:
            print("  Not enough data")
            continue

        result['factor'] = fname
        result['description'] = desc
        all_results[fname] = result

        print(f"  IC: {result['avg_IC']:.4f}, ICIR: {result['ICIR']:.4f}, IC>0: {result['IC_positive_pct']:.1f}%")

        qm = result['quintile_metrics']
        for q in ['Q1', 'Q3', 'Q5', 'LS']:
            if q in qm:
                print(f"  {q}: CAGR={qm[q]['CAGR']:>6.1f}%, Sharpe={qm[q]['Sharpe']:>6.3f}, MaxDD={qm[q]['MaxDD']:>6.1f}%")

        cagrs = [qm[f'Q{q}']['CAGR'] for q in range(1, 6) if f'Q{q}' in qm]
        if len(cagrs) == 5:
            mono = all(cagrs[i+1] >= cagrs[i] for i in range(4)) or \
                   all(cagrs[i+1] <= cagrs[i] for i in range(4))
            print(f"  Quintiles: {['%.1f' % c for c in cagrs]} {'MONOTONIC' if mono else ''}")

    # Step 7: Summary
    print("\n" + "=" * 70)
    print("FACTOR SUMMARY (ranked by |ICIR|)")
    print("=" * 70)
    hdr = f"{'Factor':<28s} {'AvgIC':>7s} {'ICIR':>7s} {'IC>0%':>6s} {'Q1':>7s} {'Q5':>7s} {'LS_CAGR':>8s} {'LS_Shp':>7s}"
    print(hdr)
    print("-" * len(hdr))

    sorted_res = sorted(all_results.values(), key=lambda x: abs(x['ICIR']), reverse=True)
    for r in sorted_res:
        qm = r['quintile_metrics']
        q1 = qm.get('Q1', {}).get('CAGR', 0)
        q5 = qm.get('Q5', {}).get('CAGR', 0)
        ls = qm.get('LS', {}).get('CAGR', 0)
        ls_s = qm.get('LS', {}).get('Sharpe', 0)
        print(f"{r['description']:<28s} {r['avg_IC']:>7.4f} {r['ICIR']:>7.4f} {r['IC_positive_pct']:>5.1f}% {q1:>6.1f}% {q5:>6.1f}% {ls:>7.1f}% {ls_s:>7.3f}")

    # Save
    out_path = os.path.join(DATA_DIR, 'stock_factor_research_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved {len(all_results)} factor results to {out_path}")


if __name__ == '__main__':
    main()
