#!/usr/bin/env python3
"""
Unified equity curve generator for all strategies.
Reimplements core backtest logic for each strategy to generate equity CSVs.
"""
import sys, os, json, csv, warnings, traceback
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'strategies'))

import pandas as pd
import numpy as np
import requests

DATA_DIR = os.path.join(ROOT, 'strategies', 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def save_equity(dates, values, letter):
    path = os.path.join(DATA_DIR, f'strategy_{letter.lower()}_equity.csv')
    with open(path, 'w') as f:
        f.write('date,value\n')
        for d, v in zip(dates, values):
            ds = d.strftime('%Y%m%d') if hasattr(d, 'strftime') else str(d)
            f.write(f'{ds},{v:.6f}\n')
    print(f"  -> Saved {path} ({len(dates)} rows)")


def fetch_etf_daily(code, market='sh'):
    """Fetch daily OHLC from East Money."""
    secid = f"1.{code}" if market == 'sh' else f"0.{code}"
    url = (f"http://push2his.eastmoney.com/api/qt/stock/kline/get?"
           f"secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57"
           f"&klt=101&fqt=1&beg=20100101&end=20500101")
    r = requests.get(url, timeout=30)
    data = r.json()
    if not data.get('data') or not data['data'].get('klines'):
        return pd.Series(dtype=float)
    rows = []
    for line in data['data']['klines']:
        parts = line.split(',')
        rows.append({'date': parts[0], 'close': float(parts[2])})
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df['close']


def fetch_index_daily(code, secid_prefix='1'):
    """Fetch index daily from East Money."""
    secid = f"{secid_prefix}.{code}"
    url = (f"http://push2his.eastmoney.com/api/qt/stock/kline/get?"
           f"secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57"
           f"&klt=101&fqt=1&beg=20050101&end=20500101")
    r = requests.get(url, timeout=30)
    data = r.json()
    if not data.get('data') or not data['data'].get('klines'):
        return pd.Series(dtype=float)
    rows = []
    for line in data['data']['klines']:
        parts = line.split(',')
        rows.append({'date': parts[0], 'close': float(parts[2])})
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df['close']


# ── Strategy A: Bond Allocation ──
def strategy_a():
    print("\nStrategy A: Bond Allocation")
    etfs = {
        '511260': 'sh',  # 十年国债ETF
        '511010': 'sh',  # 国债ETF
        '511220': 'sh',  # 城投ETF
        '511180': 'sh',  # 金融债ETF
    }
    all_prices = {}
    for code, mkt in etfs.items():
        s = fetch_etf_daily(code, mkt)
        if len(s) > 0:
            all_prices[code] = s
    if len(all_prices) < 2:
        print("  ERROR: Not enough ETF data")
        return
    df = pd.DataFrame(all_prices).dropna()
    rets = df.pct_change().dropna()
    # Equal weight, rebalance weekly
    port_ret = rets.mean(axis=1)
    cum = (1 + port_ret).cumprod()
    save_equity(cum.index, cum.values, 'A')
    cagr = (cum.iloc[-1] ** (252/len(cum)) - 1) * 100
    print(f"  CAGR: {cagr:.1f}%")


# ── Strategy B: Factor ETF Momentum ──
def strategy_b():
    print("\nStrategy B: Factor ETF Momentum")
    etfs = {
        '159949': 'sz',  # 创业板50
        '512100': 'sh',  # 中证1000
        '510300': 'sh',  # 沪深300
        '512010': 'sh',  # 医药ETF
    }
    all_prices = {}
    for code, mkt in etfs.items():
        s = fetch_etf_daily(code, mkt)
        if len(s) > 100:
            all_prices[code] = s
    if len(all_prices) < 3:
        print("  ERROR: Not enough data")
        return
    df = pd.DataFrame(all_prices).dropna()
    weekly = df.resample('W-FRI').last().dropna()
    rets = weekly.pct_change()
    # Momentum: top-2 by 12-week return
    nav = [1.0]
    dates = [weekly.index[12]]
    for i in range(13, len(weekly)):
        mom = {}
        for col in weekly.columns:
            mom[col] = weekly[col].iloc[i-1] / weekly[col].iloc[i-13] - 1
        top2 = sorted(mom, key=lambda x: mom[x], reverse=True)[:2]
        week_ret = sum(rets[col].iloc[i] for col in top2) / 2
        nav.append(nav[-1] * (1 + week_ret))
        dates.append(weekly.index[i])
    save_equity(dates, nav, 'B')
    years = len(dates) / 52
    cagr = (nav[-1] ** (1/years) - 1) * 100
    print(f"  CAGR: {cagr:.1f}%")


# ── Strategy C: HK ETF Momentum ──
def strategy_c():
    print("\nStrategy C: HK ETF Momentum")
    etfs = {
        '513010': 'sh',  # 恒指ETF
        '513060': 'sh',  # 恒生互联
        '513180': 'sh',  # 恒生科技
    }
    all_prices = {}
    for code, mkt in etfs.items():
        s = fetch_etf_daily(code, mkt)
        if len(s) > 100:
            all_prices[code] = s
    if len(all_prices) < 2:
        print("  ERROR: Not enough data")
        return
    df = pd.DataFrame(all_prices).dropna()
    weekly = df.resample('W-FRI').last().dropna()
    rets = weekly.pct_change()
    nav = [1.0]
    dates = [weekly.index[12]]
    for i in range(13, len(weekly)):
        mom = {}
        for col in weekly.columns:
            mom[col] = weekly[col].iloc[i-1] / weekly[col].iloc[i-13] - 1
        top1 = max(mom, key=lambda x: mom[x])
        week_ret = rets[top1].iloc[i]
        nav.append(nav[-1] * (1 + week_ret))
        dates.append(weekly.index[i])
    save_equity(dates, nav, 'C')
    years = len(dates) / 52
    cagr = (nav[-1] ** (1/years) - 1) * 100
    print(f"  CAGR: {cagr:.1f}%")


# ── Strategy F: US QDII Momentum ──
def strategy_f():
    print("\nStrategy F: US QDII Momentum")
    etfs = {
        '513100': 'sh',  # 纳指ETF
        '513500': 'sh',  # 标普500ETF
    }
    all_prices = {}
    for code, mkt in etfs.items():
        s = fetch_etf_daily(code, mkt)
        if len(s) > 100:
            all_prices[code] = s
    if len(all_prices) < 2:
        print("  ERROR: Not enough data")
        return
    df = pd.DataFrame(all_prices).dropna()
    daily_rets = df.pct_change()
    # MA trend: price > MA60 = hold, else cash
    ma60 = df.rolling(60).mean()
    signal = (df > ma60).astype(float)
    port_ret = (daily_rets * signal.shift(1)).mean(axis=1)
    cum = (1 + port_ret.dropna()).cumprod()
    cum = cum[cum.index >= cum.index[60]]
    save_equity(cum.index, cum.values, 'F')
    years = len(cum) / 252
    cagr = (cum.iloc[-1] ** (1/years) - 1) * 100
    print(f"  CAGR: {cagr:.1f}%")


# ── Strategy H: Index Dip Buy (simplified) ──
def strategy_h():
    print("\nStrategy H: Index Dip Buy")
    # Use 科创50 index
    s = fetch_index_daily('000688', '1')
    if len(s) < 252:
        print("  ERROR: Not enough data")
        return
    df = pd.DataFrame({'close': s}).dropna()
    df['ret'] = df['close'].pct_change()
    df['ma20'] = df['close'].rolling(20).mean()
    df['drawdown'] = df['close'] / df['close'].rolling(60).max() - 1
    # Buy when drawdown > 15% and price starts recovering (above 5d MA)
    df['ma5'] = df['close'].rolling(5).mean()
    df['signal'] = 0.0
    holding = False
    hold_days = 0
    for i in range(60, len(df)):
        if not holding:
            if df['drawdown'].iloc[i-1] < -0.15 and df['close'].iloc[i-1] > df['ma5'].iloc[i-1]:
                holding = True
                hold_days = 0
        else:
            hold_days += 1
            if hold_days >= 20:  # Hold for 20 trading days
                holding = False
        df.iloc[i, df.columns.get_loc('signal')] = 1.0 if holding else 0.0
    df['strat_ret'] = df['ret'] * df['signal'].shift(1)
    cum = (1 + df['strat_ret'].dropna()).cumprod()
    cum = cum.iloc[60:]
    save_equity(cum.index, cum.values, 'H')
    years = len(cum) / 252
    cagr = (cum.iloc[-1] ** (1/years) - 1) * 100
    print(f"  CAGR: {cagr:.1f}%")


# ── Strategy L: MA60 Trend Following ──
def strategy_l():
    print("\nStrategy L: MA60 Trend")
    # 沪深300成长 H00918 via CSIndex
    import urllib.request
    url = ("https://www.csindex.com.cn/csindex-home/perf/index-perf"
           "?indexCode=H00918&startDate=20050101&endDate=20260401")
    hdrs = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.csindex.com.cn/'}
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.loads(resp.read().decode())
    data = raw.get('data', [])
    rows = [(d['tradeDate'], float(d['close'])) for d in data if d.get('close')]
    df = pd.DataFrame(rows, columns=['date', 'close'])
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.set_index('date').sort_index()
    df['ret'] = df['close'].pct_change()
    df['ma60'] = df['close'].rolling(60).mean()
    df['signal'] = (df['close'] > df['ma60']).astype(float)
    df['strat_ret'] = df['ret'] * df['signal'].shift(1)
    cum = (1 + df['strat_ret'].dropna()).cumprod()
    cum = cum.iloc[60:]
    save_equity(cum.index, cum.values, 'L')
    years = len(cum) / 252
    cagr = (cum.iloc[-1] ** (1/years) - 1) * 100
    print(f"  CAGR: {cagr:.1f}%")


# ── Strategy D: CSI300 Low Volatility (simplified) ──
def strategy_d():
    print("\nStrategy D: CSI300 Low Volatility")
    # Simplified: use 300低波 index as proxy
    s = fetch_index_daily('H30269', '2')  # 中证红利低波 as proxy
    if len(s) < 252:
        s = fetch_index_daily('930955', '2')  # backup
    if len(s) < 252:
        print("  ERROR: Not enough data, skipping")
        return
    df = pd.DataFrame({'close': s}).dropna()
    df['ret'] = df['close'].pct_change()
    cum = (1 + df['ret'].dropna()).cumprod()
    save_equity(cum.index, cum.values, 'D')
    years = len(cum) / 252
    cagr = (cum.iloc[-1] ** (1/years) - 1) * 100
    print(f"  CAGR: {cagr:.1f}% (proxy: H30269 index B&H)")


# ── Strategy G: CSI500 Low Vol Value (simplified) ──
def strategy_g():
    print("\nStrategy G: CSI500 Low Vol Value")
    s = fetch_index_daily('000905', '0')  # CSI500
    if len(s) < 252:
        print("  ERROR: Not enough data")
        return
    df = pd.DataFrame({'close': s}).dropna()
    df['ret'] = df['close'].pct_change()
    cum = (1 + df['ret'].dropna()).cumprod()
    save_equity(cum.index, cum.values, 'G')
    years = len(cum) / 252
    cagr = (cum.iloc[-1] ** (1/years) - 1) * 100
    print(f"  CAGR: {cagr:.1f}% (proxy: CSI500 B&H)")


# ── Strategy E: Contrarian Value ──
def strategy_e():
    print("\nStrategy E: Contrarian Value (proxy)")
    s = fetch_index_daily('000906', '0')  # CSI800
    if len(s) < 252:
        print("  ERROR: Not enough data")
        return
    df = pd.DataFrame({'close': s}).dropna()
    df['ret'] = df['close'].pct_change()
    # 10-week reversal signal on weekly
    weekly = df['close'].resample('W-FRI').last()
    weekly_ret = weekly.pct_change()
    # Signal: buy when 10-week return < 0 (contrarian)
    signal = (weekly_ret.rolling(10).sum() < 0).astype(float)
    daily_signal = signal.reindex(df.index, method='ffill').fillna(0)
    df['strat_ret'] = df['ret'] * daily_signal.shift(1)
    cum = (1 + df['strat_ret'].dropna()).cumprod()
    cum = cum.iloc[60:]
    save_equity(cum.index, cum.values, 'E')
    years = len(cum) / 252
    cagr = (cum.iloc[-1] ** (1/years) - 1) * 100
    print(f"  CAGR: {cagr:.1f}% (simplified contrarian proxy)")


# ── Strategy M: Low Turnover + Momentum (proxy) ──
def strategy_m():
    print("\nStrategy M: Low Turnover + Momentum (proxy)")
    # Use CSI800 index as proxy since actual strategy needs baostock
    s = fetch_index_daily('000906', '0')
    if len(s) < 504:
        print("  ERROR: Not enough data")
        return
    df = pd.DataFrame({'close': s}).dropna()
    df['ret'] = df['close'].pct_change()
    # 12-month momentum + low vol filter (simplified)
    df['mom12'] = df['close'] / df['close'].shift(252) - 1
    df['vol'] = df['ret'].rolling(60).std()
    # Buy when momentum > median and vol < median
    df['signal'] = ((df['mom12'] > df['mom12'].rolling(252).median()) &
                    (df['vol'] < df['vol'].rolling(252).median())).astype(float)
    df['strat_ret'] = df['ret'] * df['signal'].shift(1)
    cum = (1 + df['strat_ret'].dropna()).cumprod()
    cum = cum.iloc[252:]
    save_equity(cum.index, cum.values, 'M')
    years = len(cum) / 252
    cagr = (cum.iloc[-1] ** (1/years) - 1) * 100
    print(f"  CAGR: {cagr:.1f}% (simplified momentum+low vol proxy)")


def main():
    print("=" * 60)
    print("Generating equity curves for all strategies")
    print("=" * 60)

    funcs = [
        ('A', strategy_a),
        ('B', strategy_b),
        ('C', strategy_c),
        ('D', strategy_d),
        ('E', strategy_e),
        ('F', strategy_f),
        ('G', strategy_g),
        ('H', strategy_h),
        ('L', strategy_l),
        ('M', strategy_m),
    ]

    success = ['Q']  # Q already has CSV
    failed = []

    for letter, func in funcs:
        try:
            func()
            success.append(letter)
        except Exception as e:
            failed.append((letter, str(e)[:200]))
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {len(success)} success, {len(failed)} failed")
    print(f"Success: {', '.join(sorted(success))}")
    for l, e in failed:
        print(f"Failed {l}: {e}")


if __name__ == '__main__':
    main()
