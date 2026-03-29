#!/usr/bin/env python3
"""
扩展纯动量策略研究
股票池: 沪深300 + 中证500 (~800只)
回测: 2015~2026 (10年+)
动量信号:
  1. 价格动量 (经典N周涨幅)
  2. 风险调整动量 (收益/波动率 = 信息比率)
  3. 成交量加权动量
  4. 双动量 (绝对动量 + 相对动量)
  5. 加速动量 (短期/长期)
  6. 52周新高距离
  7. 混合动量 (多因子加权)
数据源: baostock (cached in SQLite)
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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DB_PATH = os.path.join(DATA_DIR, 'stock_data.db')
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# DB helpers (reuse from strategy_momentum_value.py)
# ============================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stock_weekly (
            code TEXT NOT NULL, date TEXT NOT NULL,
            close REAL, volume REAL,
            PRIMARY KEY (code, date)
        );
        CREATE TABLE IF NOT EXISTS index_constituents (
            index_code TEXT NOT NULL, stock_code TEXT NOT NULL,
            stock_name TEXT, fetch_date TEXT NOT NULL,
            PRIMARY KEY (index_code, stock_code, fetch_date)
        );
        CREATE INDEX IF NOT EXISTS idx_weekly_date ON stock_weekly(date);
    """)
    conn.close()

def code_to_baostock(code):
    if code.startswith('6') or code.startswith('9'):
        return f'sh.{code}'
    return f'sz.{code}'

def get_constituents(index_code='000300'):
    conn = get_db()
    rows = conn.execute(
        "SELECT stock_code, stock_name FROM index_constituents WHERE index_code=? AND fetch_date >= ?",
        (index_code, (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
    ).fetchall()
    if rows:
        conn.close()
        return [(r[0], r[1]) for r in rows]
    df = ak.index_stock_cons(symbol=index_code)
    stocks = [(row['品种代码'], row['品种名称']) for _, row in df.iterrows()]
    today = datetime.now().strftime('%Y-%m-%d')
    conn.execute("DELETE FROM index_constituents WHERE index_code=?", (index_code,))
    conn.executemany(
        "INSERT OR REPLACE INTO index_constituents VALUES (?,?,?,?)",
        [(index_code, c, n, today) for c, n in stocks]
    )
    conn.commit()
    conn.close()
    return stocks

def fetch_weekly_prices(codes, start='2014-01-01', end='2026-03-28'):
    """Fetch weekly prices, extending back to 2014 for longer backtest"""
    conn = get_db()
    need_fetch = []
    for code in codes:
        row = conn.execute(
            "SELECT MIN(date), MAX(date) FROM stock_weekly WHERE code=?", (code,)
        ).fetchone()
        if row[0] is None or row[0] > '2014-06-01' or row[1] < '2026-03-20':
            need_fetch.append(code)

    if need_fetch:
        print(f"需要获取 {len(need_fetch)}/{len(codes)} 只股票的周线数据(2014~2026)...")
        lg = bs.login()
        for i, code in enumerate(need_fetch):
            bs_code = code_to_baostock(code)
            rs = bs.query_history_k_data_plus(
                bs_code, 'date,close,volume',
                start_date=start, end_date=end,
                frequency='w', adjustflag='2'
            )
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rows:
                for r in rows:
                    try:
                        conn.execute(
                            "INSERT OR REPLACE INTO stock_weekly VALUES (?,?,?,?)",
                            (code, r[0], float(r[1]) if r[1] else None, float(r[2]) if r[2] else None)
                        )
                    except: pass
                conn.commit()
            if (i+1) % 100 == 0:
                print(f"  [{i+1}/{len(need_fetch)}] 已获取")
            time.sleep(0.03)
        bs.logout()
        print(f"  周线数据获取完成")

    df = pd.read_sql_query(
        f"SELECT code, date, close FROM stock_weekly WHERE date >= ? AND date <= ? AND code IN ({','.join('?'*len(codes))})",
        conn, params=[start, end] + list(codes)
    )
    conn.close()
    if df.empty: return pd.DataFrame(), pd.DataFrame()

    pivot_close = df.pivot(index='date', columns='code', values='close')
    pivot_close.index = pd.to_datetime(pivot_close.index)
    pivot_close = pivot_close.sort_index()

    # Also get volume
    conn = get_db()
    df_vol = pd.read_sql_query(
        f"SELECT code, date, volume FROM stock_weekly WHERE date >= ? AND date <= ? AND code IN ({','.join('?'*len(codes))})",
        conn, params=[start, end] + list(codes)
    )
    conn.close()
    pivot_vol = df_vol.pivot(index='date', columns='code', values='volume')
    pivot_vol.index = pd.to_datetime(pivot_vol.index)
    pivot_vol = pivot_vol.sort_index()

    return pivot_close, pivot_vol


# ============================================================
# Momentum Signal Functions
# ============================================================

def signal_price_momentum(price_df, idx, lookback=12, skip=1):
    """Classic price momentum: return over [idx-lookback-skip, idx-skip]"""
    end = idx - skip
    start = end - lookback
    if start < 0 or end < 0: return {}
    scores = {}
    for col in price_df.columns:
        p_end = price_df[col].iloc[end]
        p_start = price_df[col].iloc[start]
        if pd.notna(p_end) and pd.notna(p_start) and p_start > 0:
            scores[col] = p_end / p_start - 1
    return scores

def signal_risk_adjusted_momentum(price_df, idx, lookback=12, skip=1):
    """Risk-adjusted momentum: return / volatility (like Sharpe)"""
    end = idx - skip
    start = end - lookback
    if start < 0 or end < 0: return {}
    scores = {}
    for col in price_df.columns:
        window = price_df[col].iloc[start:end+1]
        if window.notna().sum() < lookback * 0.7: continue
        rets = window.pct_change().dropna()
        if len(rets) < 3 or rets.std() == 0: continue
        scores[col] = rets.mean() / rets.std()
    return scores

def signal_volume_weighted_momentum(price_df, vol_df, idx, lookback=12, skip=1):
    """Volume-weighted momentum: weight recent returns by volume"""
    end = idx - skip
    start = end - lookback
    if start < 0 or end < 0: return {}
    scores = {}
    for col in price_df.columns:
        if col not in vol_df.columns: continue
        prices = price_df[col].iloc[start:end+1]
        volumes = vol_df[col].iloc[start:end+1]
        if prices.notna().sum() < lookback * 0.7: continue
        rets = prices.pct_change().dropna()
        vols = volumes.reindex(rets.index)
        if len(rets) < 3: continue
        # Normalize volumes
        v = np.array(vols.values, dtype=float)
        r = np.array(rets.values, dtype=float)
        # Align lengths
        min_len = min(len(v), len(r))
        v = v[:min_len]
        r = r[:min_len]
        mask = ~np.isnan(v) & ~np.isnan(r)
        if mask.sum() < 3: continue
        v_sum = np.nansum(v[mask])
        if v_sum == 0: continue
        weights = v[mask] / v_sum
        scores[col] = float(np.sum(r[mask] * weights))
    return scores

def signal_dual_momentum(price_df, idx, lookback=12, skip=1):
    """Dual momentum: only select stocks with positive absolute momentum + rank by relative"""
    end = idx - skip
    start = end - lookback
    if start < 0 or end < 0: return {}
    scores = {}
    for col in price_df.columns:
        p_end = price_df[col].iloc[end]
        p_start = price_df[col].iloc[start]
        if pd.notna(p_end) and pd.notna(p_start) and p_start > 0:
            ret = p_end / p_start - 1
            if ret > 0:  # absolute momentum filter
                scores[col] = ret
    return scores

def signal_acceleration(price_df, idx, short_lb=4, long_lb=20, skip=1):
    """Acceleration: short-term momentum minus long-term (trend acceleration)"""
    end = idx - skip
    if end - long_lb < 0: return {}
    scores = {}
    for col in price_df.columns:
        p_end = price_df[col].iloc[end]
        p_short = price_df[col].iloc[end - short_lb]
        p_long = price_df[col].iloc[end - long_lb]
        if all(pd.notna(x) and x > 0 for x in [p_end, p_short, p_long]):
            short_ret = p_end / p_short - 1
            long_ret = p_end / p_long - 1
            scores[col] = short_ret - long_ret * (short_lb / long_lb)  # acceleration
    return scores

def signal_52w_high(price_df, idx, skip=1):
    """Distance from 52-week high: closer to high = stronger"""
    end = idx - skip
    if end < 52: return {}
    scores = {}
    for col in price_df.columns:
        window = price_df[col].iloc[max(0, end-52):end+1]
        if window.notna().sum() < 40: continue
        high = window.max()
        current = price_df[col].iloc[end]
        if pd.notna(high) and pd.notna(current) and high > 0:
            scores[col] = current / high  # 0~1, closer to 1 = near high
    return scores

def signal_composite_momentum(price_df, vol_df, idx, lookback=12, skip=1):
    """Composite: 40% price_mom + 30% risk_adj + 30% 52w_high"""
    pm = signal_price_momentum(price_df, idx, lookback, skip)
    ra = signal_risk_adjusted_momentum(price_df, idx, lookback, skip)
    wh = signal_52w_high(price_df, idx, skip)

    common = set(pm.keys()) & set(ra.keys()) & set(wh.keys())
    if len(common) < 20: return {}

    # Rank each
    pm_s = pd.Series({k: pm[k] for k in common}).rank(pct=True)
    ra_s = pd.Series({k: ra[k] for k in common}).rank(pct=True)
    wh_s = pd.Series({k: wh[k] for k in common}).rank(pct=True)

    composite = 0.4 * pm_s + 0.3 * ra_s + 0.3 * wh_s
    return composite.to_dict()


# ============================================================
# Backtest Engine
# ============================================================

def backtest(price_df, signal_func, signal_kwargs={},
             top_n=10, rebal_freq=4, txn_cost_bps=8,
             min_stocks=20, label=''):
    """Generic backtest engine"""
    txn_cost = txn_cost_bps / 10000
    returns = price_df.pct_change()

    warmup = signal_kwargs.get('lookback', 20) + signal_kwargs.get('skip', 1) + 5
    if 'long_lb' in signal_kwargs:
        warmup = signal_kwargs['long_lb'] + 5

    nav = [1.0]
    dates = []
    prev_holdings = set()
    weekly_returns = []

    i = max(warmup, 52)  # at least 52 weeks for 52w high
    while i < len(price_df) - 1:
        scores = signal_func(idx=i)

        if len(scores) < min_stocks:
            i += 1
            continue

        ranked = pd.Series(scores).sort_values(ascending=False)
        selected = set(ranked.index[:top_n])

        new_buys = selected - prev_holdings
        sold = prev_holdings - selected
        turnover = len(new_buys) + len(sold)
        period_txn = turnover / max(len(selected), 1) * txn_cost

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            week_rets = [returns[s].iloc[j] for s in selected
                         if s in returns.columns and pd.notna(returns[s].iloc[j])]
            port_ret = np.mean(week_rets) if week_rets else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_returns.append(port_ret)

        prev_holdings = selected
        i = hold_end

    if not dates:
        return {'label': label, 'error': 'No trades'}

    nav_s = pd.Series(nav[1:], index=dates)
    yrs = (dates[-1] - dates[0]).days / 365.25
    if yrs <= 0:
        return {'label': label, 'error': 'Too short'}

    cagr = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / yrs) - 1
    dd = nav_s / nav_s.cummax() - 1
    mdd = dd.min()
    wr = pd.Series(weekly_returns)
    sharpe = wr.mean() / wr.std() * np.sqrt(52) if wr.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    win_rate = (wr > 0).sum() / len(wr) * 100

    annual = nav_s.resample('YE').last().pct_change().dropna()
    annual_returns = {str(d.year): round(v * 100, 1) for d, v in annual.items()}

    return {
        'label': label,
        'cagr_pct': round(cagr * 100, 1),
        'total_return_pct': round((nav_s.iloc[-1] - 1) * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round(win_rate, 1),
        'annual_returns': annual_returns,
        'years': round(yrs, 1),
        'period': f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}",
    }


def backtest_benchmark(price_df):
    returns = price_df.pct_change()
    avg_ret = returns.mean(axis=1).dropna()
    # Skip first 52 weeks to match
    avg_ret = avg_ret.iloc[52:]
    nav = (1 + avg_ret).cumprod()
    dates = nav.index
    yrs = (dates[-1] - dates[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    dd = nav / nav.cummax() - 1
    mdd = dd.min()
    sharpe = avg_ret.mean() / avg_ret.std() * np.sqrt(52) if avg_ret.std() > 0 else 0
    annual = nav.resample('YE').last().pct_change().dropna()
    return {
        'label': '等权持有全部',
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
    print("扩展纯动量策略研究 (2015~2026, 10年+)")
    print("CSI300 + CSI500, 多种动量信号")
    print("=" * 70)

    init_db()

    # Get constituents
    print("\n获取指数成分股...")
    csi300 = get_constituents('000300')
    csi500 = get_constituents('000905')
    all_stocks = {}
    for c, n in csi300: all_stocks[c] = n
    for c, n in csi500: all_stocks[c] = n
    all_codes = sorted(all_stocks.keys())
    print(f"  合并: {len(all_codes)}只")

    # Fetch data (extended to 2014)
    print("\n获取价格数据 (2014~2026)...")
    price_df, vol_df = fetch_weekly_prices(all_codes, start='2014-01-01', end='2026-03-28')
    print(f"  价格矩阵: {price_df.shape[0]}周 x {price_df.shape[1]}股")
    print(f"  时间: {price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")

    # Benchmark
    bench = backtest_benchmark(price_df)
    print(f"\n基准 等权持有: CAGR={bench['cagr_pct']}%, MDD={bench['mdd_pct']}%, Sharpe={bench['sharpe']}")

    results = []

    # ============================================================
    # 1. Price Momentum - vary lookback, skip, top_n, rebal
    # ============================================================
    print("\n" + "=" * 70)
    print("1. 价格动量 (经典)")
    print("=" * 70)

    pm_configs = [
        # (lookback, skip, top_n, rebal)
        (4, 1, 10, 2), (4, 1, 10, 4),
        (8, 1, 10, 4), (8, 1, 20, 4),
        (12, 1, 5, 4), (12, 1, 10, 2), (12, 1, 10, 4), (12, 1, 15, 4), (12, 1, 20, 4), (12, 1, 30, 4),
        (12, 0, 10, 4), (12, 2, 10, 4),
        (20, 1, 10, 4), (20, 1, 15, 4), (20, 1, 20, 4),
        (26, 1, 10, 4), (26, 1, 15, 4), (26, 1, 20, 4),
        (52, 1, 10, 4), (52, 1, 15, 4), (52, 1, 20, 4),
    ]

    for lb, skip, tn, rf in pm_configs:
        label = f"PriceMom_LB{lb}_S{skip}_Top{tn}_R{rf}w"
        r = backtest(price_df,
                     lambda idx, lookback=lb, skip=skip: signal_price_momentum(price_df, idx, lookback, skip),
                     signal_kwargs={'lookback': lb, 'skip': skip},
                     top_n=tn, rebal_freq=rf, label=label)
        if 'error' not in r:
            results.append(r)
            print(f"  {label}: CAGR={r['cagr_pct']}%, MDD={r['mdd_pct']}%, Sharpe={r['sharpe']}")

    # ============================================================
    # 2. Risk-Adjusted Momentum
    # ============================================================
    print("\n" + "=" * 70)
    print("2. 风险调整动量 (收益/波动率)")
    print("=" * 70)

    for lb, tn, rf in [(12, 10, 4), (12, 15, 4), (12, 20, 4), (20, 10, 4), (20, 15, 4), (26, 15, 4)]:
        label = f"RiskAdjMom_LB{lb}_Top{tn}_R{rf}w"
        r = backtest(price_df,
                     lambda idx, lookback=lb: signal_risk_adjusted_momentum(price_df, idx, lookback, 1),
                     signal_kwargs={'lookback': lb},
                     top_n=tn, rebal_freq=rf, label=label)
        if 'error' not in r:
            results.append(r)
            print(f"  {label}: CAGR={r['cagr_pct']}%, MDD={r['mdd_pct']}%, Sharpe={r['sharpe']}")

    # ============================================================
    # 3. Volume-Weighted Momentum
    # ============================================================
    print("\n" + "=" * 70)
    print("3. 成交量加权动量")
    print("=" * 70)

    for lb, tn, rf in [(12, 10, 4), (12, 15, 4), (20, 10, 4), (20, 15, 4)]:
        label = f"VolWtMom_LB{lb}_Top{tn}_R{rf}w"
        r = backtest(price_df,
                     lambda idx, lookback=lb: signal_volume_weighted_momentum(price_df, vol_df, idx, lookback, 1),
                     signal_kwargs={'lookback': lb},
                     top_n=tn, rebal_freq=rf, label=label)
        if 'error' not in r:
            results.append(r)
            print(f"  {label}: CAGR={r['cagr_pct']}%, MDD={r['mdd_pct']}%, Sharpe={r['sharpe']}")

    # ============================================================
    # 4. Dual Momentum (absolute + relative)
    # ============================================================
    print("\n" + "=" * 70)
    print("4. 双动量 (仅选正收益股票)")
    print("=" * 70)

    for lb, tn, rf in [(12, 10, 4), (12, 15, 4), (20, 10, 4), (20, 15, 4), (26, 15, 4)]:
        label = f"DualMom_LB{lb}_Top{tn}_R{rf}w"
        r = backtest(price_df,
                     lambda idx, lookback=lb: signal_dual_momentum(price_df, idx, lookback, 1),
                     signal_kwargs={'lookback': lb},
                     top_n=tn, rebal_freq=rf, label=label)
        if 'error' not in r:
            results.append(r)
            print(f"  {label}: CAGR={r['cagr_pct']}%, MDD={r['mdd_pct']}%, Sharpe={r['sharpe']}")

    # ============================================================
    # 5. Acceleration Momentum
    # ============================================================
    print("\n" + "=" * 70)
    print("5. 加速动量 (短期-长期)")
    print("=" * 70)

    for short, long, tn, rf in [(4, 20, 10, 4), (4, 20, 15, 4), (4, 12, 10, 4), (8, 26, 15, 4)]:
        label = f"AccelMom_S{short}L{long}_Top{tn}_R{rf}w"
        r = backtest(price_df,
                     lambda idx, s=short, l=long: signal_acceleration(price_df, idx, s, l, 1),
                     signal_kwargs={'long_lb': long},
                     top_n=tn, rebal_freq=rf, label=label)
        if 'error' not in r:
            results.append(r)
            print(f"  {label}: CAGR={r['cagr_pct']}%, MDD={r['mdd_pct']}%, Sharpe={r['sharpe']}")

    # ============================================================
    # 6. 52-Week High Momentum
    # ============================================================
    print("\n" + "=" * 70)
    print("6. 52周新高距离")
    print("=" * 70)

    for tn, rf in [(10, 4), (15, 4), (20, 4), (10, 2)]:
        label = f"52wHigh_Top{tn}_R{rf}w"
        r = backtest(price_df,
                     lambda idx: signal_52w_high(price_df, idx, 1),
                     signal_kwargs={'lookback': 52},
                     top_n=tn, rebal_freq=rf, label=label)
        if 'error' not in r:
            results.append(r)
            print(f"  {label}: CAGR={r['cagr_pct']}%, MDD={r['mdd_pct']}%, Sharpe={r['sharpe']}")

    # ============================================================
    # 7. Composite Momentum
    # ============================================================
    print("\n" + "=" * 70)
    print("7. 复合动量 (40%价格 + 30%风险调整 + 30%52周高)")
    print("=" * 70)

    for lb, tn, rf in [(12, 10, 4), (12, 15, 4), (20, 10, 4), (20, 15, 4), (20, 20, 4)]:
        label = f"CompositeMom_LB{lb}_Top{tn}_R{rf}w"
        r = backtest(price_df,
                     lambda idx, lookback=lb: signal_composite_momentum(price_df, vol_df, idx, lookback, 1),
                     signal_kwargs={'lookback': lb},
                     top_n=tn, rebal_freq=rf, label=label)
        if 'error' not in r:
            results.append(r)
            print(f"  {label}: CAGR={r['cagr_pct']}%, MDD={r['mdd_pct']}%, Sharpe={r['sharpe']}")

    # ============================================================
    # Summary
    # ============================================================
    results.sort(key=lambda x: -x.get('sharpe', 0))

    print("\n" + "=" * 70)
    print(f"总排名 (共{len(results)}组, 按Sharpe排序)")
    print("=" * 70)
    print(f"{'#':<4} {'Strategy':<45} {'CAGR':>6} {'MDD':>7} {'Sharpe':>7} {'Calmar':>7} {'Win%':>6}")
    print("-" * 85)

    # Add benchmark
    all_display = results + [{**bench, 'label': '📊 等权持有'}]
    all_display.sort(key=lambda x: -x.get('sharpe', 0))

    for i, r in enumerate(all_display, 1):
        lab = r.get('label', '')[:44]
        print(f"{i:<4} {lab:<45} {r['cagr_pct']:>5.1f}% {r['mdd_pct']:>6.1f}% {r['sharpe']:>7.3f} {r.get('calmar',0):>7.3f} {r.get('win_rate_pct',0):>5.1f}%")

    # Top 5 annual returns
    print(f"\nTop 5 年度收益:")
    for r in results[:5]:
        ann = ' '.join(f"{y}:{v:+.0f}%" for y, v in sorted(r.get('annual_returns', {}).items()))
        print(f"  {r['label'][:40]}: {ann}")
    ann = ' '.join(f"{y}:{v:+.0f}%" for y, v in sorted(bench.get('annual_returns', {}).items()))
    print(f"  等权持有: {ann}")

    # By signal type
    print(f"\n各类动量信号最优:")
    signal_types = {
        'PriceMom': '价格动量',
        'RiskAdjMom': '风险调整',
        'VolWtMom': '量价动量',
        'DualMom': '双动量',
        'AccelMom': '加速动量',
        '52wHigh': '52周新高',
        'CompositeMom': '复合动量',
    }
    for prefix, name in signal_types.items():
        group = [r for r in results if r['label'].startswith(prefix)]
        if group:
            best = max(group, key=lambda x: x['sharpe'])
            print(f"  {name:<12} {best['label']:<40} CAGR={best['cagr_pct']}% Sharpe={best['sharpe']} MDD={best['mdd_pct']}%")

    # Save
    output = {
        'benchmark': bench,
        'all_results': results,
        'best_overall': results[0] if results else None,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pool': f'CSI300+CSI500 = {len(all_codes)} stocks',
        'backtest_period': f'{price_df.index[0].strftime("%Y-%m-%d")} ~ {price_df.index[-1].strftime("%Y-%m-%d")}',
    }
    out_file = os.path.join(DATA_DIR, 'momentum_extended_research.json')
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_file}")


if __name__ == '__main__':
    main()
