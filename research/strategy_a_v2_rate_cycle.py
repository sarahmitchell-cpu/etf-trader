#!/usr/bin/env python3
"""
Strategy A v2: 债券利率周期择时策略
(Bond Interest Rate Cycle Timing)

核心逻辑:
  1. 标的:
     - 降息周期(利率下行) → 持有长久期: 十年国债ETF(511260)
     - 加息周期(利率上行) → 持有短久期: 短债ETF(511360)
  2. 信号: 十年国债指数(H11009) 相对于N周移动平均线
     - 价格 > MA → 利率下行期 → 持有十年国债(赚久期收益)
     - 价格 < MA → 利率上行期 → 持有短债(避险)
  3. 调仓频率: 极低，仅在信号翻转时调仓
  4. 交易成本: 单边5bp
  5. 可选: 信用利差信号，决定是否用信用债替代国债

优势:
  - 交易频率极低(年均2-4次)
  - 逻辑清晰: 利率下行赚久期，上行保短债
  - 不依赖动量(债券动量效果差)

用法:
  python3 strategy_a_v2_rate_cycle.py --backtest     # 回测+网格搜索
  python3 strategy_a_v2_rate_cycle.py                # 实盘信号
  python3 strategy_a_v2_rate_cycle.py --json          # JSON输出

数据来源: 中证指数公司 (CSIndex)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List

# ============================================================
# 配置
# ============================================================

DATA_DIR = os.environ.get(
    'ETF_DATA_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
)

# 标的
LONG_DURATION = {'code': 'H11009', 'name': '十年国债', 'etf': '511260', 'etf_name': '十年国债ETF'}
SHORT_DURATION = {'code': 'H11006', 'name': '短债', 'etf': '511360', 'etf_name': '短融ETF'}
CREDIT_BOND = {'code': 'H11073', 'name': '信用债', 'etf': '511030', 'etf_name': '信用债ETF'}
BENCHMARK = {'code': 'H11001', 'name': '中证全债'}

# 默认参数 (网格搜索后选最优)
PARAMS = {
    'ma_period': 20,            # 移动平均周数
    'use_credit': False,        # 是否在利好时用信用债替代国债
    'txn_cost_bps': 5,
}

CACHE_MAX_AGE_DAYS = 3

# ============================================================
# 数据获取 (复用strategy_a的逻辑)
# ============================================================

def _fetch_csindex(code: str, days: int, csv_path: str) -> Optional[pd.DataFrame]:
    try:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        url = (
            f"https://www.csindex.com.cn/csindex-home/perf/index-perf"
            f"?indexCode={code}&startDate={start_date}&endDate={end_date}"
        )
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
            'Referer': 'https://www.csindex.com.cn/',
            'Accept': 'application/json',
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if 'data' not in data or not data['data']:
            return None
        records = data['data']
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['tradeDate'])
        df['close'] = df['close'].astype(float)
        df = df.set_index('date').sort_index()
        df[['close']].to_csv(csv_path)
        return df
    except Exception as e:
        print(f"  [WARN] CSIndex失败 ({code}): {e}")
        return None


def fetch_index(code: str, days: int = 600) -> Optional[pd.DataFrame]:
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f'{code}_daily.csv')

    if os.path.exists(csv_path):
        file_age_days = (time.time() - os.path.getmtime(csv_path)) / 86400
        if file_age_days <= CACHE_MAX_AGE_DAYS:
            df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
            if len(df) > 0:
                return df[['close']]

    df = _fetch_csindex(code, days, csv_path)
    if df is not None and len(df) > 0:
        return df[['close']]

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        if len(df) > 0:
            return df[['close']]
    return None


# ============================================================
# 回测引擎
# ============================================================

def run_backtest(long_weekly: pd.Series, short_weekly: pd.Series,
                 credit_weekly: Optional[pd.Series],
                 benchmark_weekly: pd.Series,
                 ma_period: int, use_credit: bool = False,
                 txn_cost_bps: int = 5) -> Dict:
    """
    利率周期择时回测
    - long_weekly: 十年国债指数周线
    - short_weekly: 短债指数周线
    - credit_weekly: 信用债指数周线 (可选)
    - benchmark_weekly: 中证全债周线
    - ma_period: MA周期
    """
    # 对齐
    all_series = [long_weekly, short_weekly, benchmark_weekly]
    if use_credit and credit_weekly is not None:
        all_series.append(credit_weekly)

    common_start = max(s.index[0] for s in all_series)
    common_end = min(s.index[-1] for s in all_series)

    long_w = long_weekly.loc[common_start:common_end]
    short_w = short_weekly.loc[common_start:common_end]
    bench_w = benchmark_weekly.loc[common_start:common_end]
    credit_w = credit_weekly.loc[common_start:common_end] if use_credit and credit_weekly is not None else None

    # 周收益率
    long_ret = long_w.pct_change()
    short_ret = short_w.pct_change()
    credit_ret = credit_w.pct_change() if credit_w is not None else None

    # 信号: 十年国债 > MA → 利率下行 → 持有长久期
    ma = long_w.rolling(ma_period).mean()
    signal = long_w > ma  # True = 持有长久期, False = 持有短债

    txn_cost = txn_cost_bps / 10000

    portfolio_value = [1.0]
    prev_holding = None  # 'long', 'short', 'credit'
    weekly_returns = []
    trade_count = 0

    start_idx = ma_period
    for i in range(1, len(long_w)):
        idx = long_w.index[i]

        if i < start_idx:
            weekly_returns.append(0.0)
            portfolio_value.append(portfolio_value[-1])
            continue

        # 决定持仓
        if signal.iloc[i - 1]:  # 用前一周信号
            if use_credit and credit_ret is not None:
                curr_holding = 'credit'
            else:
                curr_holding = 'long'
        else:
            curr_holding = 'short'

        # 交易成本 (仅在切换时)
        cost = 0.0
        if prev_holding is not None and curr_holding != prev_holding:
            cost = 2 * txn_cost  # 卖出+买入
            trade_count += 1

        # 收益
        if curr_holding == 'long':
            ret = long_ret.iloc[i] if not pd.isna(long_ret.iloc[i]) else 0
        elif curr_holding == 'credit':
            ret = credit_ret.iloc[i] if credit_ret is not None and not pd.isna(credit_ret.iloc[i]) else 0
        else:
            ret = short_ret.iloc[i] if not pd.isna(short_ret.iloc[i]) else 0

        ret -= cost
        weekly_returns.append(ret)
        portfolio_value.append(portfolio_value[-1] * (1 + ret))
        prev_holding = curr_holding

    common_idx = long_w.index
    portfolio_value = pd.Series(portfolio_value, index=common_idx)
    weekly_returns_s = pd.Series(weekly_returns, index=common_idx[1:])

    # 基准
    benchmark = bench_w / bench_w.iloc[0]

    # 统计
    years = (common_idx[-1] - common_idx[0]).days / 365.25
    total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    max_dd = drawdown.min()

    sharpe = (weekly_returns_s.mean() * 52 - 0.025) / (weekly_returns_s.std() * np.sqrt(52)) if weekly_returns_s.std() > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    bm_total = benchmark.iloc[-1] / benchmark.iloc[0] - 1
    bm_cagr = (1 + bm_total) ** (1 / years) - 1 if years > 0 else 0
    bm_running_max = benchmark.cummax()
    bm_dd = ((benchmark - bm_running_max) / bm_running_max).min()

    # 持有长久期的时间比例
    long_pct = signal.iloc[start_idx:].mean() * 100

    # 年度收益
    annual_returns = {}
    for year in range(common_idx[0].year, common_idx[-1].year + 1):
        year_data = portfolio_value[portfolio_value.index.year == year]
        if len(year_data) >= 2:
            annual_returns[year] = year_data.iloc[-1] / year_data.iloc[0] - 1

    # 年均交易次数
    trades_per_year = trade_count / years if years > 0 else 0

    return {
        'ma_period': ma_period,
        'use_credit': use_credit,
        'years': round(years, 1),
        'total_return': round(total_return * 100, 1),
        'cagr': round(cagr * 100, 1),
        'max_drawdown': round(max_dd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'benchmark_cagr': round(bm_cagr * 100, 1),
        'benchmark_max_dd': round(bm_dd * 100, 1),
        'long_duration_pct': round(long_pct, 1),
        'trade_count': trade_count,
        'trades_per_year': round(trades_per_year, 1),
        'annual_returns': {str(k): round(v * 100, 1) for k, v in annual_returns.items()},
        'portfolio_value': portfolio_value,
        'benchmark': benchmark,
    }


# ============================================================
# 网格搜索
# ============================================================

def grid_search(long_weekly, short_weekly, credit_weekly, benchmark_weekly):
    """网格搜索最优参数"""
    ma_periods = [8, 13, 16, 20, 26, 30, 40, 52]
    use_credit_options = [False, True]

    results = []
    print(f"\n{'='*80}")
    print(f"网格搜索: MA周期 x 是否使用信用债")
    print(f"{'='*80}")
    print(f"{'MA周期':>8} {'信用债':>6} {'CAGR%':>7} {'MDD%':>7} {'Sharpe':>8} {'Calmar':>8} {'交易/年':>8} {'长久期%':>8}")
    print(f"{'-'*80}")

    for ma in ma_periods:
        for use_credit in use_credit_options:
            r = run_backtest(long_weekly, short_weekly, credit_weekly,
                             benchmark_weekly, ma, use_credit)
            results.append(r)
            credit_str = '是' if use_credit else '否'
            print(f"{ma:>8} {credit_str:>6} {r['cagr']:>7.1f} {r['max_drawdown']:>7.1f} "
                  f"{r['sharpe']:>8.3f} {r['calmar']:>8.3f} {r['trades_per_year']:>8.1f} "
                  f"{r['long_duration_pct']:>8.1f}")

    # 按不同指标排序
    print(f"\n{'='*60}")
    print("Top 5 by Calmar:")
    by_calmar = sorted(results, key=lambda x: x['calmar'], reverse=True)[:5]
    for i, r in enumerate(by_calmar):
        print(f"  {i+1}. MA={r['ma_period']}, credit={r['use_credit']}: "
              f"CAGR={r['cagr']}%, MDD={r['max_drawdown']}%, "
              f"Calmar={r['calmar']}, trades/yr={r['trades_per_year']}")

    print("\nTop 5 by Sharpe:")
    by_sharpe = sorted(results, key=lambda x: x['sharpe'], reverse=True)[:5]
    for i, r in enumerate(by_sharpe):
        print(f"  {i+1}. MA={r['ma_period']}, credit={r['use_credit']}: "
              f"CAGR={r['cagr']}%, MDD={r['max_drawdown']}%, "
              f"Sharpe={r['sharpe']}, trades/yr={r['trades_per_year']}")

    print("\nTop 5 by CAGR:")
    by_cagr = sorted(results, key=lambda x: x['cagr'], reverse=True)[:5]
    for i, r in enumerate(by_cagr):
        print(f"  {i+1}. MA={r['ma_period']}, credit={r['use_credit']}: "
              f"CAGR={r['cagr']}%, MDD={r['max_drawdown']}%, "
              f"trades/yr={r['trades_per_year']}")

    return results


# ============================================================
# 信号生成 (实盘)
# ============================================================

def generate_signal() -> Dict:
    """生成本周调仓信号"""
    print("=" * 60)
    print("策略A v2: 债券利率周期择时 - 周度信号")
    print("=" * 60)

    print(f"\n加载数据...")
    long_df = fetch_index(LONG_DURATION['code'], days=600)
    short_df = fetch_index(SHORT_DURATION['code'], days=600)

    if long_df is None or short_df is None:
        return {'error': '无法加载数据'}

    long_weekly = long_df['close'].resample('W-FRI').last().dropna()
    short_weekly = short_df['close'].resample('W-FRI').last().dropna()

    ma = long_weekly.rolling(PARAMS['ma_period']).mean()
    latest_price = long_weekly.iloc[-1]
    latest_ma = ma.iloc[-1]
    is_bull = latest_price > latest_ma  # 利率下行=债券牛市

    signal = {
        'strategy': '策略A v2 - 债券利率周期择时',
        'date': str(long_weekly.index[-1].date()),
        'regime': '利率下行(债牛)' if is_bull else '利率上行(债熊)',
        'holding': LONG_DURATION['name'] if is_bull else SHORT_DURATION['name'],
        'etf': LONG_DURATION['etf'] if is_bull else SHORT_DURATION['etf'],
        'etf_name': LONG_DURATION['etf_name'] if is_bull else SHORT_DURATION['etf_name'],
        'signal_price': round(latest_price, 2),
        'signal_ma': round(latest_ma, 2),
        'ma_period': PARAMS['ma_period'],
    }

    if '--json' not in sys.argv:
        print(f"\n📊 信号日期: {signal['date']}")
        print(f"十年国债指数: {latest_price:.2f}")
        print(f"{PARAMS['ma_period']}周MA: {latest_ma:.2f}")
        print(f"\n📈 利率周期判断: {signal['regime']}")
        print(f"{'↑ 价格在MA上方 → 利率下行趋势 → 持有长久期' if is_bull else '↓ 价格在MA下方 → 利率上行趋势 → 持有短债避险'}")
        print(f"\n🎯 本周持仓: {signal['holding']} ({signal['etf']} {signal['etf_name']}) → 100%")

    return signal


# ============================================================
# 主入口
# ============================================================

def main():
    if '--backtest' in sys.argv:
        print("=" * 60)
        print("策略A v2: 债券利率周期择时 - 完整回测")
        print("=" * 60)

        days = 5500
        print(f"\n加载数据 (~{days//365}年)...")

        codes = [
            (LONG_DURATION, '十年国债'),
            (SHORT_DURATION, '短债'),
            (CREDIT_BOND, '信用债'),
            (BENCHMARK, '中证全债'),
        ]
        data = {}
        for info, label in codes:
            print(f"  {label} ({info['code']})...")
            df = fetch_index(info['code'], days=days)
            if df is None:
                print(f"  [ERROR] 无法加载 {label}")
                sys.exit(1)
            data[label] = df['close'].resample('W-FRI').last().dropna()
            print(f"    {data[label].index[0].date()} ~ {data[label].index[-1].date()} ({len(data[label])}周)")
            time.sleep(0.3)

        # 网格搜索
        results = grid_search(data['十年国债'], data['短债'], data['信用债'], data['中证全债'])

        # 用最佳参数运行详细回测
        best = sorted(results, key=lambda x: x['calmar'], reverse=True)[0]
        print(f"\n{'='*60}")
        print(f"最佳参数详细结果 (MA={best['ma_period']}, credit={best['use_credit']})")
        print(f"{'='*60}")
        print(f"  年化收益 (CAGR):    {best['cagr']}%")
        print(f"  最大回撤 (MDD):     {best['max_drawdown']}%")
        print(f"  夏普比率:           {best['sharpe']}")
        print(f"  卡玛比率:           {best['calmar']}")
        print(f"  总交易次数:         {best['trade_count']}")
        print(f"  年均交易次数:       {best['trades_per_year']}")
        print(f"  长久期持有比例:     {best['long_duration_pct']}%")
        print(f"\n  基准 (中证全债):")
        print(f"    年化收益:         {best['benchmark_cagr']}%")
        print(f"    最大回撤:         {best['benchmark_max_dd']}%")
        print(f"\n  年度收益:")
        for year, ret in sorted(best['annual_returns'].items()):
            print(f"    {year}: {ret:+.1f}%")

        # 保存
        result_data = {k: v for k, v in best.items()
                       if k not in ('portfolio_value', 'benchmark')}
        result_path = os.path.join(DATA_DIR, 'strategy_a_v2_backtest.json')
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {result_path}")

    else:
        signal = generate_signal()
        if '--json' in sys.argv:
            print(json.dumps(signal, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
