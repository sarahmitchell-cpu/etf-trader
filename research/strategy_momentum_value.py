#!/usr/bin/env python3
"""
动量价值策略 (Momentum + Value)
股票池: 沪深300 + 中证500 成分股 (~800只)
因子:
  - 动量: N周价格动量 (skip最近1周避免短期反转)
  - 价值: EP (1/PE_TTM) + BP (1/PB_MRQ) 复合
数据源: baostock (weekly price + daily PE/PB)
回测: 2021~2026, 周度/月度调仓
"""

import pandas as pd
import numpy as np
import baostock as bs
import akshare as ak
import os
import json
import time
import sys
import sqlite3
from datetime import datetime, timedelta
from itertools import product

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DB_PATH = os.path.join(DATA_DIR, 'stock_data.db')
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# Database helper
# ============================================================
def get_db():
    """Get SQLite connection with WAL mode"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

def init_db():
    """Create tables if not exist"""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stock_weekly (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL,
            volume REAL,
            PRIMARY KEY (code, date)
        );
        CREATE TABLE IF NOT EXISTS stock_daily_valuation (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL,
            pe_ttm REAL,
            pb_mrq REAL,
            ps_ttm REAL,
            PRIMARY KEY (code, date)
        );
        CREATE TABLE IF NOT EXISTS index_constituents (
            index_code TEXT NOT NULL,
            stock_code TEXT NOT NULL,
            stock_name TEXT,
            fetch_date TEXT NOT NULL,
            PRIMARY KEY (index_code, stock_code, fetch_date)
        );
        CREATE INDEX IF NOT EXISTS idx_weekly_date ON stock_weekly(date);
        CREATE INDEX IF NOT EXISTS idx_valuation_date ON stock_daily_valuation(date);
    """)
    conn.close()

# ============================================================
# Data fetching with DB cache
# ============================================================

def get_constituents(index_code='000300', force_refresh=False):
    """Get index constituents, cache in DB for 7 days"""
    conn = get_db()
    # Check cache
    if not force_refresh:
        rows = conn.execute(
            "SELECT stock_code, stock_name FROM index_constituents WHERE index_code=? AND fetch_date >= ?",
            (index_code, (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
        ).fetchall()
        if rows:
            conn.close()
            return [(r[0], r[1]) for r in rows]

    # Fetch from akshare
    df = ak.index_stock_cons(symbol=index_code)
    stocks = [(row['品种代码'], row['品种名称']) for _, row in df.iterrows()]

    today = datetime.now().strftime('%Y-%m-%d')
    conn.execute("DELETE FROM index_constituents WHERE index_code=?", (index_code,))
    conn.executemany(
        "INSERT OR REPLACE INTO index_constituents (index_code, stock_code, stock_name, fetch_date) VALUES (?,?,?,?)",
        [(index_code, code, name, today) for code, name in stocks]
    )
    conn.commit()
    conn.close()
    return stocks

def code_to_baostock(code):
    """Convert 6-digit code to baostock format"""
    if code.startswith('6') or code.startswith('9'):
        return f'sh.{code}'
    else:
        return f'sz.{code}'

def fetch_weekly_prices(codes, start='2020-06-01', end='2026-03-28'):
    """Fetch weekly prices for all codes, cache in DB"""
    conn = get_db()

    # Check which codes need fetching
    need_fetch = []
    for code in codes:
        row = conn.execute(
            "SELECT MAX(date) FROM stock_weekly WHERE code=? AND date >= ?",
            (code, start)
        ).fetchone()
        if row[0] is None or row[0] < '2026-03-20':
            need_fetch.append(code)

    if need_fetch:
        print(f"需要获取 {len(need_fetch)}/{len(codes)} 只股票的周线数据...")
        lg = bs.login()

        for i, code in enumerate(need_fetch):
            bs_code = code_to_baostock(code)
            rs = bs.query_history_k_data_plus(
                bs_code, 'date,close,volume',
                start_date=start, end_date=end,
                frequency='w', adjustflag='2'  # 前复权
            )
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())

            if rows:
                for r in rows:
                    try:
                        conn.execute(
                            "INSERT OR REPLACE INTO stock_weekly (code, date, close, volume) VALUES (?,?,?,?)",
                            (code, r[0], float(r[1]) if r[1] else None, float(r[2]) if r[2] else None)
                        )
                    except:
                        pass
                conn.commit()

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(need_fetch)}] 已获取")
            time.sleep(0.05)

        bs.logout()
        print(f"  周线数据获取完成")

    # Load all from DB
    df = pd.read_sql_query(
        f"SELECT code, date, close FROM stock_weekly WHERE date >= ? AND date <= ? AND code IN ({','.join('?' * len(codes))})",
        conn, params=[start, end] + list(codes)
    )
    conn.close()

    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot(index='date', columns='code', values='close')
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()
    return pivot

def fetch_daily_valuations(codes, start='2020-06-01', end='2026-03-28'):
    """Fetch daily PE/PB for all codes, cache in DB"""
    conn = get_db()

    # Check which codes need fetching
    need_fetch = []
    for code in codes:
        row = conn.execute(
            "SELECT MAX(date) FROM stock_daily_valuation WHERE code=? AND date >= ?",
            (code, start)
        ).fetchone()
        if row[0] is None or row[0] < '2026-03-20':
            need_fetch.append(code)

    if need_fetch:
        print(f"需要获取 {len(need_fetch)}/{len(codes)} 只股票的估值数据...")
        lg = bs.login()

        for i, code in enumerate(need_fetch):
            bs_code = code_to_baostock(code)
            rs = bs.query_history_k_data_plus(
                bs_code, 'date,close,peTTM,pbMRQ,psTTM',
                start_date=start, end_date=end,
                frequency='d', adjustflag='2'
            )
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())

            if rows:
                for r in rows:
                    try:
                        conn.execute(
                            "INSERT OR REPLACE INTO stock_daily_valuation (code, date, close, pe_ttm, pb_mrq, ps_ttm) VALUES (?,?,?,?,?,?)",
                            (code, r[0],
                             float(r[1]) if r[1] else None,
                             float(r[2]) if r[2] else None,
                             float(r[3]) if r[3] else None,
                             float(r[4]) if r[4] else None)
                        )
                    except:
                        pass
                conn.commit()

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(need_fetch)}] 已获取估值")
            time.sleep(0.05)

        bs.logout()
        print(f"  估值数据获取完成")
    conn.close()

def get_valuation_at_date(codes, date_str, conn=None):
    """Get PE/PB for stocks on or before a specific date"""
    close_conn = False
    if conn is None:
        conn = get_db()
        close_conn = True

    # Get the most recent valuation on or before the date
    results = {}
    for code in codes:
        row = conn.execute(
            "SELECT pe_ttm, pb_mrq FROM stock_daily_valuation WHERE code=? AND date<=? ORDER BY date DESC LIMIT 1",
            (code, date_str)
        ).fetchone()
        if row and row[0] and row[1]:
            pe = float(row[0])
            pb = float(row[1])
            if pe > 0 and pb > 0:  # Valid positive PE/PB
                results[code] = {'pe_ttm': pe, 'pb_mrq': pb}

    if close_conn:
        conn.close()
    return results


# ============================================================
# Backtest Engine
# ============================================================

def compute_momentum(price_df, idx, lookback=12, skip=1):
    """Compute momentum for all stocks at index position idx"""
    end_idx = idx - skip
    start_idx = end_idx - lookback
    if start_idx < 0 or end_idx < 0:
        return {}

    mom = {}
    for col in price_df.columns:
        p_end = price_df[col].iloc[end_idx]
        p_start = price_df[col].iloc[start_idx]
        if pd.notna(p_end) and pd.notna(p_start) and p_start > 0:
            mom[col] = p_end / p_start - 1
    return mom

def compute_value_score(valuations):
    """
    Compute value score from PE/PB
    Higher EP (1/PE) + higher BP (1/PB) = more value
    Returns percentile rank (0~1, 1=cheapest)
    """
    if not valuations:
        return {}

    ep = {k: 1.0 / v['pe_ttm'] for k, v in valuations.items() if v['pe_ttm'] > 0}
    bp = {k: 1.0 / v['pb_mrq'] for k, v in valuations.items() if v['pb_mrq'] > 0}

    common = set(ep.keys()) & set(bp.keys())
    if len(common) < 5:
        return {}

    # Rank each factor (higher = cheaper = more value)
    ep_series = pd.Series({k: ep[k] for k in common})
    bp_series = pd.Series({k: bp[k] for k in common})

    ep_rank = ep_series.rank(pct=True)
    bp_rank = bp_series.rank(pct=True)

    # Composite: 50% EP + 50% BP
    value_score = (ep_rank + bp_rank) / 2
    return value_score.to_dict()

def backtest_momentum_value(price_df, valuation_dates, all_codes,
                            mom_lookback=12, mom_skip=1,
                            mom_weight=0.5, val_weight=0.5,
                            top_n=10, rebal_freq=4,
                            txn_cost_bps=8,
                            min_stocks=20):
    """
    Momentum + Value composite strategy backtest

    Args:
        price_df: weekly price DataFrame
        valuation_dates: dict of date -> valuations
        mom_lookback: momentum lookback (weeks)
        mom_skip: skip recent N weeks
        mom_weight: weight on momentum factor
        val_weight: weight on value factor
        top_n: number of stocks to hold
        rebal_freq: rebalance every N weeks
        txn_cost_bps: transaction cost basis points (one-way)
    """
    txn_cost = txn_cost_bps / 10000
    returns = price_df.pct_change()

    warmup = mom_lookback + mom_skip + 1
    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_returns = []
    holding_log = []
    conn = get_db()

    i = warmup
    while i < len(price_df) - 1:
        decision_date = price_df.index[i]
        date_str = decision_date.strftime('%Y-%m-%d')

        # 1. Compute momentum scores
        mom_scores = compute_momentum(price_df, i, mom_lookback, mom_skip)

        # 2. Get valuations at this date
        avail_codes = list(mom_scores.keys())
        valuations = get_valuation_at_date(avail_codes, date_str, conn)

        # 3. Compute value scores
        value_scores = compute_value_score(valuations)

        # 4. Compute composite score for stocks with both factors
        common = set(mom_scores.keys()) & set(value_scores.keys())
        if len(common) < min_stocks:
            i += 1
            continue

        # Rank momentum (higher = stronger momentum = better)
        mom_series = pd.Series({k: mom_scores[k] for k in common})
        mom_rank = mom_series.rank(pct=True)

        # Value rank is already 0~1
        val_rank = pd.Series({k: value_scores[k] for k in common})

        # Composite
        composite = mom_weight * mom_rank + val_weight * val_rank

        # Select top N
        ranked = composite.sort_values(ascending=False)
        selected = set(ranked.index[:top_n])

        # Transaction cost
        new_buys = selected - prev_holdings
        sold = prev_holdings - selected
        turnover = len(new_buys) + len(sold)
        period_txn = turnover / max(len(selected), 1) * txn_cost
        total_txn += period_txn

        # Hold for rebal_freq weeks
        hold_end = min(i + rebal_freq, len(price_df) - 1)

        for j in range(i + 1, hold_end + 1):
            week_rets = []
            for s in selected:
                r = returns[s].iloc[j] if s in returns.columns else np.nan
                if pd.notna(r):
                    week_rets.append(r)

            port_ret = np.mean(week_rets) if week_rets else 0.0

            if j == i + 1:
                port_ret -= period_txn

            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_returns.append(port_ret)

        holding_log.append({
            'date': date_str,
            'holdings': list(selected)[:5],
            'top_composite': round(float(ranked.iloc[0]), 3) if len(ranked) > 0 else 0,
        })

        prev_holdings = selected
        i = hold_end

    conn.close()

    if not dates:
        return {'error': 'No trades generated'}

    nav_series = pd.Series(nav[1:], index=dates)
    total_years = (dates[-1] - dates[0]).days / 365.25

    if total_years <= 0:
        return {'error': 'Period too short'}

    cagr = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (1 / total_years) - 1
    dd = nav_series / nav_series.cummax() - 1
    mdd = dd.min()
    wr = pd.Series(weekly_returns)
    sharpe = wr.mean() / wr.std() * np.sqrt(52) if wr.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    win_rate = (wr > 0).sum() / len(wr) * 100

    annual = nav_series.resample('YE').last().pct_change().dropna()
    annual_returns = {str(d.year): round(v * 100, 1) for d, v in annual.items()}

    return {
        'strategy': f'MomVal_LB{mom_lookback}_Skip{mom_skip}_MW{mom_weight:.1f}_VW{val_weight:.1f}_Top{top_n}_R{rebal_freq}w',
        'mom_lookback': mom_lookback,
        'mom_skip': mom_skip,
        'mom_weight': mom_weight,
        'val_weight': val_weight,
        'top_n': top_n,
        'rebal_freq': rebal_freq,
        'period': f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}",
        'years': round(total_years, 1),
        'cagr_pct': round(cagr * 100, 1),
        'total_return_pct': round((nav_series.iloc[-1] - 1) * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round(win_rate, 1),
        'annual_returns': annual_returns,
        'total_txn_pct': round(total_txn * 100, 2),
        'num_rebalances': len(holding_log),
        'pool_size': price_df.shape[1],
    }


def backtest_pure_momentum(price_df, mom_lookback=12, mom_skip=1,
                           top_n=10, rebal_freq=4, txn_cost_bps=8):
    """Pure momentum strategy (for comparison)"""
    txn_cost = txn_cost_bps / 10000
    returns = price_df.pct_change()
    warmup = mom_lookback + mom_skip + 1
    nav = [1.0]
    dates = []
    prev_holdings = set()
    weekly_returns = []

    i = warmup
    while i < len(price_df) - 1:
        mom = compute_momentum(price_df, i, mom_lookback, mom_skip)
        if len(mom) < 20:
            i += 1
            continue

        ranked = sorted(mom.items(), key=lambda x: -x[1])
        selected = set(t for t, _ in ranked[:top_n])

        new_buys = selected - prev_holdings
        sold = prev_holdings - selected
        turnover = len(new_buys) + len(sold)
        period_txn = turnover / max(len(selected), 1) * txn_cost

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            week_rets = [returns[s].iloc[j] for s in selected if s in returns.columns and pd.notna(returns[s].iloc[j])]
            port_ret = np.mean(week_rets) if week_rets else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_returns.append(port_ret)

        prev_holdings = selected
        i = hold_end

    if not dates:
        return {'error': 'No trades'}

    nav_s = pd.Series(nav[1:], index=dates)
    yrs = (dates[-1] - dates[0]).days / 365.25
    cagr = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / yrs) - 1
    dd = nav_s / nav_s.cummax() - 1
    mdd = dd.min()
    wr = pd.Series(weekly_returns)
    sharpe = wr.mean() / wr.std() * np.sqrt(52) if wr.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    annual = nav_s.resample('YE').last().pct_change().dropna()

    return {
        'strategy': f'PureMom_LB{mom_lookback}_Top{top_n}_R{rebal_freq}w',
        'cagr_pct': round(cagr * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round((wr > 0).sum() / len(wr) * 100, 1),
        'annual_returns': {str(d.year): round(v * 100, 1) for d, v in annual.items()},
    }


def backtest_pure_value(price_df, all_codes, top_n=10, rebal_freq=4, txn_cost_bps=8):
    """Pure value strategy (for comparison)"""
    txn_cost = txn_cost_bps / 10000
    returns = price_df.pct_change()
    nav = [1.0]
    dates = []
    prev_holdings = set()
    weekly_returns = []
    conn = get_db()

    # Start after enough data
    i = 13  # ~3 months warmup
    while i < len(price_df) - 1:
        date_str = price_df.index[i].strftime('%Y-%m-%d')
        avail = [c for c in price_df.columns if pd.notna(price_df[c].iloc[i])]
        valuations = get_valuation_at_date(avail, date_str, conn)
        value_scores = compute_value_score(valuations)

        if len(value_scores) < 20:
            i += 1
            continue

        ranked = pd.Series(value_scores).sort_values(ascending=False)
        selected = set(ranked.index[:top_n])

        new_buys = selected - prev_holdings
        sold = prev_holdings - selected
        turnover = len(new_buys) + len(sold)
        period_txn = turnover / max(len(selected), 1) * txn_cost

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            week_rets = [returns[s].iloc[j] for s in selected if s in returns.columns and pd.notna(returns[s].iloc[j])]
            port_ret = np.mean(week_rets) if week_rets else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_returns.append(port_ret)

        prev_holdings = selected
        i = hold_end

    conn.close()

    if not dates:
        return {'error': 'No trades'}

    nav_s = pd.Series(nav[1:], index=dates)
    yrs = (dates[-1] - dates[0]).days / 365.25
    cagr = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / yrs) - 1
    dd = nav_s / nav_s.cummax() - 1
    mdd = dd.min()
    wr = pd.Series(weekly_returns)
    sharpe = wr.mean() / wr.std() * np.sqrt(52) if wr.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    annual = nav_s.resample('YE').last().pct_change().dropna()

    return {
        'strategy': f'PureValue_Top{top_n}_R{rebal_freq}w',
        'cagr_pct': round(cagr * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round((wr > 0).sum() / len(wr) * 100, 1),
        'annual_returns': {str(d.year): round(v * 100, 1) for d, v in annual.items()},
    }


def backtest_benchmark(price_df):
    """Equal-weight buy-and-hold benchmark"""
    returns = price_df.pct_change()
    avg_ret = returns.mean(axis=1).dropna()
    nav = (1 + avg_ret).cumprod()
    dates = nav.index
    yrs = (dates[-1] - dates[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    dd = nav / nav.cummax() - 1
    mdd = dd.min()
    sharpe = avg_ret.mean() / avg_ret.std() * np.sqrt(52) if avg_ret.std() > 0 else 0

    annual = nav.resample('YE').last().pct_change().dropna()

    return {
        'strategy': '等权持有全部',
        'cagr_pct': round(cagr * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(cagr / abs(mdd), 3) if mdd != 0 else 0,
        'annual_returns': {str(d.year): round(v * 100, 1) for d, v in annual.items()},
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("动量价值策略研究 (CSI300 + CSI500)")
    print("=" * 70)

    # 1. Initialize DB
    init_db()

    # 2. Get constituents
    print("\n获取指数成分股...")
    csi300 = get_constituents('000300')
    csi500 = get_constituents('000905')

    # Combine and deduplicate
    all_stocks = {}
    for code, name in csi300:
        all_stocks[code] = name
    for code, name in csi500:
        all_stocks[code] = name

    all_codes = sorted(all_stocks.keys())
    print(f"  沪深300: {len(csi300)}只, 中证500: {len(csi500)}只")
    print(f"  合并去重: {len(all_codes)}只")

    # 3. Fetch data
    print("\n获取价格和估值数据...")
    price_df = fetch_weekly_prices(all_codes, start='2020-06-01', end='2026-03-28')
    fetch_daily_valuations(all_codes, start='2020-06-01', end='2026-03-28')

    print(f"  价格矩阵: {price_df.shape[0]}周 x {price_df.shape[1]}股")
    print(f"  时间范围: {price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")

    # 4. Benchmark
    print("\n" + "=" * 70)
    print("基准")
    print("=" * 70)
    bench = backtest_benchmark(price_df)
    print(f"  等权持有: CAGR={bench['cagr_pct']}%, MDD={bench['mdd_pct']}%, Sharpe={bench['sharpe']}")
    print(f"  年度: {bench['annual_returns']}")

    # 5. Parameter sweep - Momentum + Value
    print("\n" + "=" * 70)
    print("动量价值策略参数搜索")
    print("=" * 70)

    results = []

    # Core sweep
    param_grid = [
        # (mom_lookback, mom_skip, mom_weight, val_weight, top_n, rebal_freq)
        # Vary momentum lookback
        (8, 1, 0.5, 0.5, 10, 4),
        (12, 1, 0.5, 0.5, 10, 4),
        (20, 1, 0.5, 0.5, 10, 4),
        (26, 1, 0.5, 0.5, 10, 4),
        # Vary factor weights
        (12, 1, 0.7, 0.3, 10, 4),  # momentum heavy
        (12, 1, 0.3, 0.7, 10, 4),  # value heavy
        (12, 1, 0.6, 0.4, 10, 4),
        (12, 1, 0.4, 0.6, 10, 4),
        # Vary top_n
        (12, 1, 0.5, 0.5, 5, 4),
        (12, 1, 0.5, 0.5, 15, 4),
        (12, 1, 0.5, 0.5, 20, 4),
        (12, 1, 0.5, 0.5, 30, 4),
        # Vary rebal frequency
        (12, 1, 0.5, 0.5, 10, 2),
        (12, 1, 0.5, 0.5, 10, 8),
        # Skip recent
        (12, 0, 0.5, 0.5, 10, 4),
        (12, 2, 0.5, 0.5, 10, 4),
        # Best combos guess
        (12, 1, 0.6, 0.4, 15, 4),
        (20, 1, 0.5, 0.5, 15, 4),
        (12, 1, 0.5, 0.5, 15, 2),
        (20, 1, 0.6, 0.4, 10, 4),
    ]

    for lb, skip, mw, vw, tn, rf in param_grid:
        r = backtest_momentum_value(price_df, {}, all_codes,
                                    mom_lookback=lb, mom_skip=skip,
                                    mom_weight=mw, val_weight=vw,
                                    top_n=tn, rebal_freq=rf)
        if 'error' not in r:
            results.append(r)
            print(f"  {r['strategy']}: CAGR={r['cagr_pct']}%, MDD={r['mdd_pct']}%, Sharpe={r['sharpe']}, Calmar={r['calmar']}")

    # 6. Comparison strategies
    print("\n" + "=" * 70)
    print("对比策略")
    print("=" * 70)

    # Pure momentum
    pm = backtest_pure_momentum(price_df, mom_lookback=12, mom_skip=1, top_n=10, rebal_freq=4)
    if 'error' not in pm:
        print(f"  纯动量 Top10: CAGR={pm['cagr_pct']}%, MDD={pm['mdd_pct']}%, Sharpe={pm['sharpe']}")

    # Pure value
    pv = backtest_pure_value(price_df, all_codes, top_n=10, rebal_freq=4)
    if 'error' not in pv:
        print(f"  纯价值 Top10: CAGR={pv['cagr_pct']}%, MDD={pv['mdd_pct']}%, Sharpe={pv['sharpe']}")

    # Pure momentum with different top_n
    pm20 = backtest_pure_momentum(price_df, mom_lookback=12, mom_skip=1, top_n=20, rebal_freq=4)
    if 'error' not in pm20:
        print(f"  纯动量 Top20: CAGR={pm20['cagr_pct']}%, MDD={pm20['mdd_pct']}%, Sharpe={pm20['sharpe']}")

    # 7. Sort and display results
    results.sort(key=lambda x: -x['sharpe'])

    print("\n" + "=" * 70)
    print("排名 (按Sharpe)")
    print("=" * 70)
    print(f"{'Strategy':<55} {'CAGR':>6} {'MDD':>7} {'Sharpe':>7} {'Calmar':>7} {'Win%':>6}")
    print("-" * 90)

    # Add benchmarks to display
    all_display = results + [
        {**bench, 'strategy': '📊 等权持有'},
        {**pm, 'strategy': '📈 纯动量 Top10'} if 'error' not in pm else None,
        {**pv, 'strategy': '💰 纯价值 Top10'} if 'error' not in pv else None,
        {**pm20, 'strategy': '📈 纯动量 Top20'} if 'error' not in pm20 else None,
    ]
    all_display = [x for x in all_display if x is not None]
    all_display.sort(key=lambda x: -x.get('sharpe', 0))

    for r in all_display:
        s = r.get('strategy', r.get('name', ''))[:54]
        print(f"  {s:<54} {r['cagr_pct']:>5.1f}% {r['mdd_pct']:>6.1f}% {r['sharpe']:>7.3f} {r.get('calmar', 0):>7.3f} {r.get('win_rate_pct', 0):>5.1f}%")

    # 8. Annual returns for top 3
    print("\n年度收益 (Top 3 + 基准):")
    for r in results[:3]:
        ann = ' '.join(f"{y}:{v:+.0f}%" for y, v in sorted(r['annual_returns'].items()))
        print(f"  {r['strategy'][:40]}: {ann}")
    ann = ' '.join(f"{y}:{v:+.0f}%" for y, v in sorted(bench['annual_returns'].items()))
    print(f"  等权持有: {ann}")

    # 9. Save results
    output = {
        'benchmark': bench,
        'pure_momentum': pm if 'error' not in pm else None,
        'pure_value': pv if 'error' not in pv else None,
        'momentum_value_results': results,
        'best': results[0] if results else None,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pool': f'CSI300({len(csi300)}) + CSI500({len(csi500)}) = {len(all_codes)} stocks',
    }

    out_file = os.path.join(DATA_DIR, 'momentum_value_research.json')
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {out_file}")
    print(f"数据库: {DB_PATH}")


if __name__ == '__main__':
    main()
