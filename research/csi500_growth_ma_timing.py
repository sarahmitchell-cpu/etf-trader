#!/usr/bin/env python3
"""
CSI 500 Growth MA Timing Research
研究中证500成长相关指数的MA趋势择时策略

测试多个MA参数 (20/40/60/80/120/250) 在多个500成长相关指数上的表现
对比300成长(H00918)的基准结果

数据来源: akshare (价格指数) + CSI API (全收益指数)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json
import os
import sys
import time
import requests
from datetime import datetime

# ============================================================
# 待研究的指数
# ============================================================

INDICES = {
    # 500成长相关
    '930938': '500成长估值',      # 中证500成长估值 (2018~)
    '930939': '500质量成长',      # 中证500质量成长 (2020~)
    '000905': '中证500',          # 中证500 base (2007~)
    # 对比基准
    '000918': '300成长',          # 沪深300成长 (2008~)
    # 其他风格指数
    '399374': '中盘成长',         # 国证中盘成长 (2010~)
    '399376': '小盘成长',         # 国证小盘成长 (2010~)
    '000057': '全指成长',         # 中证全指成长 (2010~)
}

# 也尝试从CSI获取全收益版本
CSI_TR_INDICES = {
    'H00918': '300成长TR',
    'H00905': '500TR',
}

MA_PERIODS = [20, 40, 60, 80, 120, 250]
TXN_COST_BPS = 8  # 单边交易成本

# ============================================================
# 数据获取
# ============================================================

def fetch_akshare(code, name):
    """从akshare获取指数日线数据"""
    try:
        import akshare as ak
        df = ak.index_zh_a_hist(symbol=code, period='daily',
                                start_date='20050101', end_date='20260328')
        if df is None or len(df) == 0:
            print(f"  {name}({code}): no data from akshare")
            return None
        df['date'] = pd.to_datetime(df['日期'])
        df['close'] = pd.to_numeric(df['收盘'], errors='coerce')
        df = df[['date', 'close']].dropna().set_index('date').sort_index()
        df = df[df['close'] > 0]
        df = df[~df.index.duplicated(keep='first')]
        print(f"  {name}({code}): {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} rows")
        return df
    except Exception as e:
        print(f"  {name}({code}): error {e}")
        return None


def fetch_csi_tr(code, name):
    """从CSI API获取全收益指数数据"""
    url = 'https://www.csindex.com.cn/csindex-home/perf/index-perf'
    params = {'indexCode': code, 'startDate': '20050101',
              'endDate': datetime.now().strftime('%Y%m%d')}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
        'Referer': 'https://www.csindex.com.cn/'
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        data = r.json()
        if str(data.get('code')) != '200' or not data.get('data'):
            print(f"  {name}({code}): CSI API error")
            return None
        items = data['data']
        df = pd.DataFrame(items)
        df['date'] = pd.to_datetime(df['tradeDate'], format='%Y%m%d')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df[['date', 'close']].dropna().set_index('date').sort_index()
        df = df[df['close'] > 0]
        df = df[~df.index.duplicated(keep='first')]
        print(f"  {name}({code}): {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} rows [TR]")
        return df
    except Exception as e:
        print(f"  {name}({code}): CSI error {e}")
        return None


# ============================================================
# MA择时回测
# ============================================================

def backtest_ma_timing(df: pd.DataFrame, ma_period: int,
                       txn_cost_bps: float = 8) -> dict:
    """
    纯MA趋势择时回测
    close > MA → 满仓, close < MA → 空仓
    信号shift(1)避免前瞻偏差
    """
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['ret'] = d['close'].pct_change()
    d['position'] = (d['close'] > d['ma']).astype(float)
    d['signal'] = d['position'].shift(1)
    d = d.dropna(subset=['signal', 'ret'])

    if len(d) < 252:
        return None

    # 策略收益
    d['strat_ret'] = d['ret'] * d['signal']
    txn_cost = txn_cost_bps / 10000
    d['signal_change'] = d['signal'].diff().abs()
    d['strat_ret'] -= d['signal_change'] * txn_cost

    returns = d['strat_ret']
    bh_returns = d['ret']

    # 策略指标
    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1
    years = len(returns) / 252
    if years < 1:
        return None
    cagr = (1 + total_ret) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    monthly = returns.resample('ME').sum()
    win_rate = (monthly > 0).mean() * 100
    trade_count = int((d['signal'].diff().abs() > 0.01).sum())

    # Buy&Hold
    bh_cum = (1 + bh_returns).cumprod()
    bh_total = bh_cum.iloc[-1] - 1
    bh_cagr = (1 + bh_total) ** (1 / years) - 1
    bh_vol = bh_returns.std() * np.sqrt(252)
    bh_sharpe = bh_cagr / bh_vol if bh_vol > 0 else 0
    bh_peak = bh_cum.cummax()
    bh_max_dd = ((bh_cum - bh_peak) / bh_peak).min()

    # 年度收益
    annual = returns.resample('YE').sum()
    annual_dict = {str(idx.year): round(val * 100, 1) for idx, val in annual.items()}

    # 持仓天数占比
    in_market_pct = d['signal'].mean() * 100

    return {
        'cagr': round(cagr * 100, 2),
        'vol': round(vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'max_dd': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'win_rate': round(win_rate, 1),
        'total_ret': round(total_ret * 100, 1),
        'years': round(years, 1),
        'trade_count': trade_count,
        'in_market_pct': round(in_market_pct, 1),
        'bh_cagr': round(bh_cagr * 100, 2),
        'bh_sharpe': round(bh_sharpe, 3),
        'bh_max_dd': round(bh_max_dd * 100, 2),
        'annual': annual_dict,
        'period': f"{d.index[0].date()} ~ {d.index[-1].date()}",
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  CSI 500 Growth MA Timing Research")
    print(f"  MA periods: {MA_PERIODS}")
    print(f"  Transaction cost: {TXN_COST_BPS} bps per side")
    print("=" * 70)

    # 1. Fetch all data
    print("\n[1] Fetching index data...")
    all_data = {}

    for code, name in INDICES.items():
        df = fetch_akshare(code, name)
        if df is not None and len(df) >= 252:
            all_data[f"{name}({code})"] = df
        time.sleep(0.5)

    # Try CSI TR indices
    for code, name in CSI_TR_INDICES.items():
        df = fetch_csi_tr(code, name)
        if df is not None and len(df) >= 252:
            all_data[f"{name}({code})"] = df
        time.sleep(2)

    print(f"\n  Successfully loaded {len(all_data)} indices")

    # 2. Run backtests
    print("\n[2] Running backtests...")
    results = []

    for idx_name, df in all_data.items():
        for ma in MA_PERIODS:
            if len(df) < ma + 60:  # Need enough data
                continue
            bt = backtest_ma_timing(df, ma, TXN_COST_BPS)
            if bt is not None:
                bt['index'] = idx_name
                bt['ma_period'] = ma
                results.append(bt)

    # 3. Results
    print(f"\n  Total backtests: {len(results)}")

    # Sort by Sharpe
    results.sort(key=lambda x: x['sharpe'], reverse=True)

    # Print summary table
    print(f"\n{'='*100}")
    print(f"  {'Rank':<5} {'Index':<25} {'MA':<5} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} "
          f"{'Trades':>7} {'InMkt':>6} {'Years':>6} {'BH_CAGR':>8}")
    print(f"  {'-'*5} {'-'*25} {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*8}")

    for i, r in enumerate(results):
        print(f"  {i+1:<5} {r['index']:<25} {r['ma_period']:<5} "
              f"{r['cagr']:>6.1f}% {r['sharpe']:>7.3f} {r['max_dd']:>7.1f}% "
              f"{r['trade_count']:>7} {r['in_market_pct']:>5.1f}% {r['years']:>5.1f} "
              f"{r['bh_cagr']:>7.1f}%")

    # 4. Best per index
    print(f"\n{'='*100}")
    print(f"  Best MA period per index (by Sharpe):")
    print(f"  {'-'*95}")

    seen = set()
    for r in results:
        if r['index'] not in seen:
            seen.add(r['index'])
            print(f"  {r['index']:<25} → MA{r['ma_period']:>3} | "
                  f"CAGR={r['cagr']:>6.1f}% Sharpe={r['sharpe']:.3f} "
                  f"MaxDD={r['max_dd']:>6.1f}% Trades={r['trade_count']}")

    # 5. Heatmap: per index, Sharpe by MA
    print(f"\n{'='*100}")
    print(f"  Sharpe Heatmap (Index × MA Period):")
    hdr = "MA\\Index"
    print(f"  {hdr:<8}", end="")
    idx_names = list(all_data.keys())
    for name in idx_names:
        short = name[:15]
        print(f" {short:>15}", end="")
    print()

    for ma in MA_PERIODS:
        print(f"  MA{ma:<5}", end="")
        for name in idx_names:
            match = [r for r in results if r['index'] == name and r['ma_period'] == ma]
            if match:
                s = match[0]['sharpe']
                print(f" {s:>15.3f}", end="")
            else:
                print(f" {'N/A':>15}", end="")
        print()

    # 6. Focus on 500-related: annual returns comparison
    print(f"\n{'='*100}")
    print(f"  500-related indices: Best strategy annual returns:")
    for r in results:
        if '500' in r['index'] or '中盘' in r['index']:
            if r['ma_period'] == [x for x in results if x['index'] == r['index']][0]['ma_period']:
                print(f"\n  {r['index']} MA{r['ma_period']}:")
                for yr, ret in sorted(r['annual'].items()):
                    bar = '+' * max(0, int(ret / 3)) if ret > 0 else '-' * max(0, int(-ret / 3))
                    print(f"    {yr}: {ret:>+7.1f}% {bar}")

    # 7. Save results
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'csi500_growth_ma_timing_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to {out_path}")

    return results


if __name__ == '__main__':
    main()
