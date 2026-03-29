#!/usr/bin/env python3
"""
Strategy L: 300成长 纯MA60趋势择时
(CSI 300 Growth Index Pure MA60 Trend Timing)

核心逻辑:
  极简趋势跟踪 — 收盘价在MA60上方满仓，下方空仓

  1. 标的: 沪深300成长全收益指数 (CSI 300 Growth TR, code H00918)
     - 对应ETF: 310398 (沪深300成长ETF) 或类似产品
  2. 趋势判断: 收盘价 > MA60 → 满仓; 收盘价 < MA60 → 空仓
  3. 信号延迟: 所有信号T日生成，T+1执行 (无前瞻性偏差)

回测表现 (2005~2026, 约20.5年, 全收益指数, 含8bps交易成本):
  - CAGR ≈ 18.9%, Sharpe ≈ 1.050, MaxDD ≈ -31.7%
  - 相比MA60+MA20回调: 更高CAGR, 更低交易频率, 更低交易成本拖累

对比基准:
  - 300成长 Buy&Hold: CAGR ≈ 14.5%, MaxDD > -70%

研究来源: research/trend_reversal_combo.py (243组合全因子回测)
数据来源: CSI Official Total Return Index API (csindex.com.cn)

用法:
  python3 strategy_l_weekly_signal.py              # 输出当前信号
  python3 strategy_l_weekly_signal.py --json       # JSON格式输出
  python3 strategy_l_weekly_signal.py --backtest   # 运行完整回测
  python3 strategy_l_weekly_signal.py --status     # 当前状态概览

Author: Sarah Mitchell / VisionClaw
Date: 2026-03-29
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import requests
import json
import sys
import os
from datetime import datetime, timedelta

# ============================================================
# 策略参数
# ============================================================

PARAMS = {
    'index_code': 'H00918',             # 沪深300成长全收益 (API verified)
    'index_name': '300成长',
    'trend_ma': 60,                      # 趋势判断均线
    'txn_cost_bps': 8,                   # 单边交易成本(bps)
}

# 可交易的ETF列表 (跟踪300成长或近似标的)
TRADABLE_ETFS = {
    '310398': '沪深300成长ETF (申万菱信)',
    '159918': '300ETF (嘉实, 近似)',
}

# ============================================================
# 数据获取
# ============================================================

def fetch_csi_index(code: str, name: str, start: str = '20050101',
                    end: str = None) -> pd.DataFrame | None:
    """从中证指数官方API获取全收益指数数据"""
    if end is None:
        end = datetime.now().strftime('%Y%m%d')

    url = 'https://www.csindex.com.cn/csindex-home/perf/index-perf'
    params = {'indexCode': code, 'startDate': start, 'endDate': end}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
        'Referer': 'https://www.csindex.com.cn/'
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        data = r.json()
        if str(data.get('code')) != '200' or not data.get('data'):
            print(f"  API error for {name} ({code})")
            return None

        items = data['data']
        df = pd.DataFrame(items)
        df['date'] = pd.to_datetime(df['tradeDate'], format='%Y%m%d')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df[['date', 'close']].dropna().set_index('date').sort_index()
        df = df[df['close'] > 0]
        df = df[~df.index.duplicated(keep='first')]
        return df
    except Exception as e:
        print(f"  ERROR fetching {name} ({code}): {e}")
        return None


# ============================================================
# 技术指标
# ============================================================

def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """计算RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ============================================================
# 策略核心: 纯MA60趋势择时
# ============================================================

def compute_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每日信号 — 纯MA60

    返回DataFrame包含:
      - close: 收盘价
      - ma_trend: MA60
      - trend_up: 趋势是否向上 (close > MA60)
      - position: 目标仓位 (0.0 空仓 / 1.0 满仓)
      - signal: 延迟一天的信号 (用于实际交易, 无前瞻偏差)
    """
    d = df.copy()
    trend_ma = PARAMS['trend_ma']

    d['ma_trend'] = d['close'].rolling(trend_ma).mean()
    d['rsi'] = calc_rsi(d['close'], 14)
    d['ret'] = d['close'].pct_change()

    # 趋势判断: close > MA60 → 满仓, 否则空仓
    d['trend_up'] = d['close'] > d['ma_trend']
    d['position'] = d['trend_up'].astype(float)

    # T日信号, T+1执行 (无前瞻偏差)
    d['signal'] = d['position'].shift(1)

    return d


def generate_signal(df: pd.DataFrame) -> dict:
    """生成当前交易信号"""
    d = compute_signal(df)

    latest = d.iloc[-1]

    # 当前状态
    close = latest['close']
    ma_trend = latest['ma_trend']
    rsi = latest['rsi']
    trend_up = latest['trend_up']
    position = latest['position']
    signal = latest['signal']  # 今天应执行的信号(昨天生成)

    # 距离MA的百分比
    dist_ma60 = (close - ma_trend) / ma_trend * 100 if pd.notna(ma_trend) else None

    # 判断信号变化
    prev_signal = d['signal'].iloc[-2] if len(d) > 2 else 0.0
    signal_changed = abs(signal - prev_signal) > 0.01 if pd.notna(prev_signal) else False

    # 生成建议
    if signal >= 1.0:
        action = '满仓持有'
        action_en = 'FULL_POSITION'
    else:
        action = '空仓观望'
        action_en = 'CASH'

    if signal_changed:
        if signal > prev_signal:
            action += ' (开仓↑)'
        else:
            action += ' (清仓↓)'

    result = {
        'strategy': 'L',
        'strategy_name': '纯MA60趋势择时',
        'index': PARAMS['index_name'],
        'index_code': PARAMS['index_code'],
        'date': str(d.index[-1].date()),
        'close': round(close, 2),
        'ma60': round(ma_trend, 2) if pd.notna(ma_trend) else None,
        'rsi14': round(rsi, 1) if pd.notna(rsi) else None,
        'dist_ma60_pct': round(dist_ma60, 2) if dist_ma60 is not None else None,
        'trend_up': bool(trend_up),
        'target_position': round(float(position), 2),
        'executable_signal': round(float(signal), 2),
        'action': action,
        'action_en': action_en,
        'signal_changed': bool(signal_changed),
        'tradable_etfs': TRADABLE_ETFS,
    }

    return result


# ============================================================
# 回测引擎
# ============================================================

def run_backtest(df: pd.DataFrame, verbose: bool = True) -> dict:
    """运行完整回测"""
    d = compute_signal(df)
    d = d.dropna(subset=['signal', 'ret'])

    # 策略收益 = 信号(T-1) * 实际收益(T)
    d['strat_ret'] = d['ret'] * d['signal']

    # 交易成本
    txn_cost = PARAMS['txn_cost_bps'] / 10000
    d['signal_change'] = d['signal'].diff().abs()
    d['strat_ret'] -= d['signal_change'] * txn_cost

    returns = d['strat_ret'].dropna()
    bh_returns = d['ret'].dropna()

    if len(returns) < 252:
        print("ERROR: Not enough data for backtest")
        return {}

    # 策略指标
    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1
    years = len(returns) / 252
    cagr = (1 + total_ret) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    monthly = returns.resample('ME').sum()
    win_rate = (monthly > 0).mean() * 100

    # Buy&Hold指标
    bh_cum = (1 + bh_returns).cumprod()
    bh_total = bh_cum.iloc[-1] - 1
    bh_cagr = (1 + bh_total) ** (1 / years) - 1
    bh_vol = bh_returns.std() * np.sqrt(252)
    bh_sharpe = bh_cagr / bh_vol if bh_vol > 0 else 0
    bh_peak = bh_cum.cummax()
    bh_dd = (bh_cum - bh_peak) / bh_peak
    bh_max_dd = bh_dd.min()

    # 年度收益
    annual = returns.resample('YE').sum()
    annual_dict = {str(idx.year): round(val * 100, 1) for idx, val in annual.items()}

    # 交易次数
    trade_changes = (d['signal'].diff().abs() > 0.01).sum()

    result = {
        'strategy': {
            'cagr': round(cagr * 100, 2),
            'vol': round(vol * 100, 2),
            'sharpe': round(sharpe, 3),
            'max_dd': round(max_dd * 100, 2),
            'calmar': round(calmar, 3),
            'win_rate_monthly': round(win_rate, 1),
            'total_ret': round(total_ret * 100, 1),
            'years': round(years, 1),
            'trade_count': int(trade_changes),
        },
        'buy_hold': {
            'cagr': round(bh_cagr * 100, 2),
            'vol': round(bh_vol * 100, 2),
            'sharpe': round(bh_sharpe, 3),
            'max_dd': round(bh_max_dd * 100, 2),
        },
        'annual_returns': annual_dict,
        'period': f"{d.index[0].date()} ~ {d.index[-1].date()}",
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Strategy L Backtest Results")
        print(f"  {PARAMS['index_name']} ({PARAMS['index_code']})")
        print(f"  纯MA{PARAMS['trend_ma']}趋势择时")
        print(f"  Period: {result['period']}")
        print(f"{'='*60}")

        s = result['strategy']
        b = result['buy_hold']
        print(f"\n  {'Metric':<25} {'Strategy':>12} {'Buy&Hold':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12}")
        print(f"  {'CAGR':<25} {s['cagr']:>11.2f}% {b['cagr']:>11.2f}%")
        print(f"  {'Volatility':<25} {s['vol']:>11.2f}% {b['vol']:>11.2f}%")
        print(f"  {'Sharpe':<25} {s['sharpe']:>12.3f} {b['sharpe']:>12.3f}")
        print(f"  {'Max Drawdown':<25} {s['max_dd']:>11.2f}% {b['max_dd']:>11.2f}%")
        print(f"  {'Calmar':<25} {s['calmar']:>12.3f} {'':>12}")
        print(f"  {'Monthly Win Rate':<25} {s['win_rate_monthly']:>11.1f}% {'':>12}")
        print(f"  {'Total Return':<25} {s['total_ret']:>10.1f}% {'':>12}")
        print(f"  {'Trade Count':<25} {s['trade_count']:>12} {'':>12}")

        print(f"\n  Annual Returns:")
        for yr, ret in sorted(result['annual_returns'].items()):
            bar = '+' * max(0, int(ret / 2)) if ret > 0 else '-' * max(0, int(-ret / 2))
            print(f"    {yr}: {ret:>+7.1f}% {bar}")

    return result


# ============================================================
# 状态概览
# ============================================================

def print_status(df: pd.DataFrame):
    """打印当前策略状态"""
    signal = generate_signal(df)

    print(f"\n{'='*50}")
    print(f"  Strategy L — {signal['strategy_name']}")
    print(f"  {signal['index']} ({signal['index_code']})")
    print(f"  Date: {signal['date']}")
    print(f"{'='*50}")
    print(f"  Close:     {signal['close']}")
    print(f"  MA60:      {signal['ma60']}  ({signal['dist_ma60_pct']:+.2f}%)")
    print(f"  RSI(14):   {signal['rsi14']}")
    print(f"  Trend:     {'↑ UP' if signal['trend_up'] else '↓ DOWN'}")
    print(f"  Position:  {signal['target_position']}")
    print(f"  Signal:    {signal['executable_signal']}")
    print(f"  Action:    {signal['action']}")
    if signal['signal_changed']:
        print(f"  ⚠️  SIGNAL CHANGED — 需要调仓!")
    print(f"{'='*50}")


# ============================================================
# MAIN
# ============================================================

def main():
    mode = 'signal'
    if '--backtest' in sys.argv:
        mode = 'backtest'
    elif '--json' in sys.argv:
        mode = 'json'
    elif '--status' in sys.argv:
        mode = 'status'

    # 获取数据
    print(f"Fetching {PARAMS['index_name']} ({PARAMS['index_code']})...")
    df = fetch_csi_index(PARAMS['index_code'], PARAMS['index_name'])
    if df is None or len(df) < PARAMS['trend_ma'] + 10:
        print("ERROR: Failed to fetch sufficient data")
        sys.exit(1)
    print(f"  Data: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} rows")

    if mode == 'backtest':
        result = run_backtest(df, verbose=True)
        # Save results
        out_path = os.path.join(os.path.dirname(__file__), 'data',
                                'strategy_l_backtest.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {out_path}")

    elif mode == 'json':
        signal = generate_signal(df)
        print(json.dumps(signal, ensure_ascii=False, indent=2))

    elif mode == 'status':
        print_status(df)

    else:
        # Default: print signal
        signal = generate_signal(df)
        print_status(df)

        # Also output brief JSON for automation
        print(f"\nJSON Signal:")
        brief = {
            'strategy': signal['strategy'],
            'date': signal['date'],
            'action': signal['action_en'],
            'position': signal['executable_signal'],
            'trend_up': signal['trend_up'],
            'close': signal['close'],
            'ma60': signal['ma60'],
        }
        print(json.dumps(brief, ensure_ascii=False))


if __name__ == '__main__':
    main()
