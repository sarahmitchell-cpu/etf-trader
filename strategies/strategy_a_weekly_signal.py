#!/usr/bin/env python3
"""
Strategy A: 债券分散持有策略
(Diversified Bond Buy-and-Hold)

核心逻辑:
  1. 标的池 (4个债券指数, 覆盖不同久期和信用):
     - 十年国债 (H11009) -> ETF 511260  (长久期利率债)
     - 城投债 (H11015) -> ETF 511220    (中久期信用债)
     - 信用债 (H11073) -> ETF 511030    (中短久期信用债)
     - 短债 (H11006) -> ETF 511360      (短久期, 现金替代)
  2. 配置: 等权分散持有 (各25%)
  3. 再平衡: 季度再平衡 (每季度末)
  4. 交易成本: 单边5bp

重要说明:
  - 回测使用中证债券指数 (非ETF价格), 含票息收入
  - 这是一个被动策略, 长期持有分散化的债券组合
  - 目标: 获取债券市场平均收益, 降低单一品种风险

用法:
  python3 strategy_a_weekly_signal.py              # 正常运行 (输出当前配置)
  python3 strategy_a_weekly_signal.py --json       # 仅输出JSON
  python3 strategy_a_weekly_signal.py --backtest   # 运行完整回测

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

# 债券指数池
BOND_POOL = {
    'H11009': {'name': '十年国债', 'etf': '511260', 'etf_name': '十年国债ETF'},
    'H11015': {'name': '城投债', 'etf': '511220', 'etf_name': '城投ETF'},
    'H11073': {'name': '信用债', 'etf': '511030', 'etf_name': '信用债ETF'},
    'H11006': {'name': '短债', 'etf': '511360', 'etf_name': '短融ETF'},
}

# 基准: 中证全债
BENCHMARK = {'code': 'H11001', 'name': '中证全债'}

# 策略参数
PARAMS = {
    'weights': {  # 等权分散
        '十年国债': 0.25,
        '城投债': 0.25,
        '信用债': 0.25,
        '短债': 0.25,
    },
    'rebalance_freq': 'quarter',  # 再平衡频率: quarter / month / none
    'rebalance_threshold': 0.05,  # 偏离阈值: 超过5%才再平衡
    'txn_cost_bps': 5,            # 单边交易成本
}

CACHE_MAX_AGE_DAYS = 3

# ============================================================
# 数据获取
# ============================================================

def _fetch_csindex(code: str, days: int, csv_path: str) -> Optional[pd.DataFrame]:
    """从中证指数公司获取数据"""
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
            print(f"  [WARN] CSIndex无数据: {code}")
            return None
        records = data['data']
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['tradeDate'])
        df['close'] = df['close'].astype(float)
        df = df.set_index('date').sort_index()
        df[['close']].to_csv(csv_path)
        print(f"  [OK] CSIndex: {code} ({len(df)}条)")
        return df
    except Exception as e:
        print(f"  [WARN] CSIndex失败 ({code}): {e}")
        return None


def fetch_index(code: str, days: int = 600) -> Optional[pd.DataFrame]:
    """获取指数日线数据"""
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
        print(f"  [WARN] API失败, 用过期缓存: {code}", file=sys.stderr)
        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        if len(df) > 0:
            return df[['close']]

    return None


# ============================================================
# 回测引擎
# ============================================================

def _get_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set:
    """获取再平衡日期 (季末/月末的最后一个交易日)"""
    dates = set()
    if freq == 'none':
        return dates

    s = pd.Series(range(len(index)), index=index)
    if freq == 'quarter':
        # 每季度最后一个交易日
        grouped = s.groupby([s.index.year, s.index.quarter])
    elif freq == 'month':
        grouped = s.groupby([s.index.year, s.index.month])
    else:
        return dates

    for _, group in grouped:
        dates.add(group.index[-1])
    return dates


def run_backtest(bond_weekly: Dict[str, pd.Series], bench_weekly: pd.Series) -> Dict:
    """运行分散持有回测"""

    target_weights = PARAMS['weights']
    bond_names = list(target_weights.keys())
    threshold = PARAMS['rebalance_threshold']
    txn_cost = PARAMS['txn_cost_bps'] / 10000

    # 对齐日期
    all_series = list(bond_weekly.values()) + [bench_weekly]
    common_start = max(s.index[0] for s in all_series)
    common_end = min(s.index[-1] for s in all_series)

    bond_aligned = {k: v.loc[common_start:common_end] for k, v in bond_weekly.items()}
    bench_aligned = bench_weekly.loc[common_start:common_end]
    common_idx = bench_aligned.index

    # 周收益率
    bond_ret = {k: v.pct_change() for k, v in bond_aligned.items()}

    # 再平衡日期
    rebal_dates = _get_rebalance_dates(common_idx, PARAMS['rebalance_freq'])

    # 模拟
    portfolio_value = [1.0]
    current_holdings = {n: target_weights[n] for n in bond_names}  # 初始按目标配置
    weekly_returns = []
    n_rebalances = 0

    for i in range(1, len(common_idx)):
        idx = common_idx[i]

        # 计算本周组合收益
        ret = 0.0
        new_holdings = {}
        total_value = 0.0
        for name in bond_names:
            w = current_holdings.get(name, 0)
            if w > 0 and idx in bond_ret[name].index:
                r = bond_ret[name].loc[idx]
                if not pd.isna(r):
                    ret += w * r
                    new_holdings[name] = w * (1 + r)
                else:
                    new_holdings[name] = w
            else:
                new_holdings[name] = w
            total_value += new_holdings[name]

        # 归一化持仓权重
        if total_value > 0:
            for name in bond_names:
                new_holdings[name] /= total_value

        # 检查是否需要再平衡
        if idx in rebal_dates and total_value > 0:
            max_drift = max(abs(new_holdings.get(n, 0) - target_weights[n]) for n in bond_names)
            if max_drift > threshold:
                # 计算换手成本
                turnover = sum(abs(target_weights[n] - new_holdings.get(n, 0)) for n in bond_names)
                cost = turnover * txn_cost
                ret -= cost
                new_holdings = {n: target_weights[n] for n in bond_names}
                n_rebalances += 1

        current_holdings = new_holdings
        weekly_returns.append(ret)
        portfolio_value.append(portfolio_value[-1] * (1 + ret))

    portfolio_value = pd.Series(portfolio_value, index=common_idx)
    weekly_returns = pd.Series(weekly_returns, index=common_idx[1:])

    # 基准: 中证全债买入持有
    benchmark = bench_aligned / bench_aligned.iloc[0]

    # 统计
    years = (common_idx[-1] - common_idx[0]).days / 365.25
    total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    max_dd = drawdown.min()

    sharpe = (weekly_returns.mean() * 52 - 0.025) / (weekly_returns.std() * np.sqrt(52)) if weekly_returns.std() > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    bm_total = benchmark.iloc[-1] / benchmark.iloc[0] - 1
    bm_cagr = (1 + bm_total) ** (1 / years) - 1 if years > 0 else 0
    bm_running_max = benchmark.cummax()
    bm_dd = ((benchmark - bm_running_max) / bm_running_max).min()

    # 年度收益
    annual_returns = {}
    for year in range(common_idx[0].year, common_idx[-1].year + 1):
        year_data = portfolio_value[portfolio_value.index.year == year]
        if len(year_data) >= 2:
            annual_returns[year] = year_data.iloc[-1] / year_data.iloc[0] - 1

    return {
        'years': round(years, 1),
        'total_return': round(total_return * 100, 1),
        'cagr': round(cagr * 100, 1),
        'max_drawdown': round(max_dd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'benchmark_cagr': round(bm_cagr * 100, 1),
        'benchmark_max_dd': round(bm_dd * 100, 1),
        'annual_returns': {str(k): round(v * 100, 1) for k, v in annual_returns.items()},
        'n_rebalances': n_rebalances,
        'rebalances_per_year': round(n_rebalances / years, 1) if years > 0 else 0,
        'target_weights': {k: round(v * 100) for k, v in target_weights.items()},
        'portfolio_value': portfolio_value,
        'benchmark': benchmark,
    }


# ============================================================
# 信号生成 (实盘)
# ============================================================

def generate_signal() -> Dict:
    """生成当前配置信号 (被动策略, 固定配置)"""
    print("=" * 60)
    print("策略A: 债券分散持有 - 当前配置")
    print("=" * 60)

    target_weights = PARAMS['weights']

    signal = {
        'strategy': '策略A-债券分散持有',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'type': 'passive_diversified',
        'rebalance_freq': PARAMS['rebalance_freq'],
        'allocations': {},
    }

    if '--json' not in sys.argv:
        print(f"\n📊 策略类型: 被动分散持有")
        print(f"再平衡频率: 每季度 (偏离>{PARAMS['rebalance_threshold']*100:.0f}%时)")
        print(f"\n🎯 目标配置:")

    for code, info in BOND_POOL.items():
        name = info['name']
        w = target_weights.get(name, 0)
        if w > 0:
            etf_str = f"{info['etf']} {info['etf_name']}"
            signal['allocations'][name] = round(w * 100)
            if '--json' not in sys.argv:
                print(f"  {name} ({etf_str}): {w*100:.0f}%")

    if '--json' not in sys.argv:
        print(f"\n💡 操作建议: 长期持有, 每季度检查偏离度, 超过5%时再平衡")

    return signal


# ============================================================
# 主入口
# ============================================================

def main():
    if '--backtest' in sys.argv:
        print("=" * 60)
        print("策略A: 债券分散持有 - 完整回测")
        print("=" * 60)

        days = 5500
        print(f"\n加载数据 (回测模式, ~{days//365}年)...")

        bond_weekly = {}
        for code, info in BOND_POOL.items():
            print(f"  {info['name']} ({code})...")
            df = fetch_index(code, days=days)
            if df is None:
                print(f"  [ERROR] 无法加载 {info['name']}")
                sys.exit(1)
            weekly = df['close'].resample('W-FRI').last().dropna()
            bond_weekly[info['name']] = weekly
            time.sleep(0.3)

        print(f"  {BENCHMARK['name']} ({BENCHMARK['code']})...")
        bench_df = fetch_index(BENCHMARK['code'], days=days)
        if bench_df is None:
            print("ERROR: 无法加载全债指数")
            sys.exit(1)
        bench_weekly = bench_df['close'].resample('W-FRI').last().dropna()

        for name, series in bond_weekly.items():
            print(f"  {name}: {series.index[0].date()} ~ {series.index[-1].date()} ({len(series)}周)")

        result = run_backtest(bond_weekly, bench_weekly)

        print(f"\n{'='*60}")
        print(f"回测结果 ({result['years']}年)")
        print(f"{'='*60}")
        print(f"  目标配置: {result['target_weights']}")
        print(f"  再平衡频率: {PARAMS['rebalance_freq']} (偏离>{PARAMS['rebalance_threshold']*100:.0f}%)")
        print(f"  再平衡次数: {result['n_rebalances']} ({result['rebalances_per_year']}次/年)")
        print(f"\n  年化收益 (CAGR):  {result['cagr']}%")
        print(f"  最大回撤 (MDD):   {result['max_drawdown']}%")
        print(f"  夏普比率:         {result['sharpe']}")
        print(f"  卡玛比率:         {result['calmar']}")
        print(f"\n  基准 (中证全债买入持有):")
        print(f"    年化收益:       {result['benchmark_cagr']}%")
        print(f"    最大回撤:       {result['benchmark_max_dd']}%")
        print(f"\n  年度收益:")
        for year, ret in sorted(result['annual_returns'].items()):
            print(f"    {year}: {ret:+.1f}%")

        result_data = {k: v for k, v in result.items()
                       if k not in ('portfolio_value', 'benchmark')}
        result_path = os.path.join(DATA_DIR, 'strategy_a_backtest.json')
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {result_path}")

    else:
        signal = generate_signal()
        if '--json' in sys.argv:
            print(json.dumps(signal, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
