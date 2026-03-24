#!/usr/bin/env python3
"""
Strategy A: 纯债ETF动量轮动策略
(Bond ETF Momentum Rotation)

核心逻辑:
  1. 标的池 (4个债券指数, 覆盖不同久期和信用):
     - 十年国债 (H11009) → ETF 511260
     - 城投债 (H11015) → ETF 511220
     - 信用债 (H11073) → ETF 511210 或类似
     - 短债 (H11006) → ETF 511360 (避险/现金替代)
  2. 动量排名: 4周收益率排名
     - Top1: 50%, Top2: 30%, Top3: 20%, Top4: 0%
  3. 防守机制: 当全债指数(H11001)跌破13周MA → 全切短债(避险)
  4. 调仓频率: 周频 (每周五)
  5. 交易成本: 单边5bp (债券ETF成本低)

重要说明:
  - 回测使用中证债券指数 (非ETF价格), 含票息收入
  - T-1 数据决策、T 期收益, 无前瞻偏差
  - 债券指数表现好于ETF实际表现 (ETF有跟踪误差)

用法:
  python3 strategy_a_weekly_signal.py              # 正常运行
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
from typing import Optional, Dict, Tuple, List

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

# 信号指数: 中证全债 (用于判断债市整体趋势)
SIGNAL_INDEX = {'code': 'H11001', 'name': '中证全债'}

# 策略参数
PARAMS = {
    'ma_defensive': 0,  # 0 = no defensive, pure momentum         # 防守均线周数 (全债指数)
    'momentum_lookback': 8,     # 动量回看周数
    'weights': [0.50, 0.30, 0.20, 0.00],  # Top1/2/3/4权重
    'txn_cost_bps': 5,          # 单边交易成本 (债券ETF低摩擦)
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
# 信号计算
# ============================================================

def compute_signal(bond_weekly: Dict[str, pd.Series], signal_weekly: pd.Series) -> pd.DataFrame:
    """
    计算每周配置信号。
    """
    bond_names = list(bond_weekly.keys())

    # 全债均线防守 (ma_defensive=0 表示不使用防守机制)
    ma_def = PARAMS['ma_defensive']
    if ma_def > 0:
        ma = signal_weekly.rolling(ma_def).mean()
        is_defensive = signal_weekly < ma
    else:
        is_defensive = pd.Series(False, index=signal_weekly.index)

    # 动量
    mom = pd.DataFrame()
    for name, series in bond_weekly.items():
        mom[name] = series.pct_change(PARAMS['momentum_lookback'])

    # 构建配置
    alloc = pd.DataFrame(0.0, index=signal_weekly.index, columns=bond_names + ['defensive'])
    safe_haven = '短债'  # 避险资产

    start_idx = max(PARAMS['momentum_lookback'], ma_def if ma_def > 0 else 0)
    for i in range(start_idx, len(signal_weekly)):
        idx = signal_weekly.index[i]

        if is_defensive.iloc[i]:
            # 防守: 全切短债
            if safe_haven in bond_names:
                alloc.loc[idx, safe_haven] = 1.0
            alloc.loc[idx, 'defensive'] = 1
        else:
            # 进攻: 动量排名分配
            if idx not in mom.index:
                if safe_haven in bond_names:
                    alloc.loc[idx, safe_haven] = 1.0
                continue

            mom_row = mom.loc[idx].dropna()
            if len(mom_row) == 0:
                if safe_haven in bond_names:
                    alloc.loc[idx, safe_haven] = 1.0
                continue

            ranked = mom_row.sort_values(ascending=False)
            weights = PARAMS['weights']
            for j in range(min(len(ranked), len(weights))):
                alloc.loc[idx, ranked.index[j]] = weights[j]

    return alloc


# ============================================================
# 回测引擎
# ============================================================

def run_backtest(bond_weekly: Dict[str, pd.Series], signal_weekly: pd.Series) -> Dict:
    """运行纯债动量轮动回测"""

    # 对齐日期
    all_series = list(bond_weekly.values()) + [signal_weekly]
    common_start = max(s.index[0] for s in all_series)
    common_end = min(s.index[-1] for s in all_series)

    bond_aligned = {k: v.loc[common_start:common_end] for k, v in bond_weekly.items()}
    signal_aligned = signal_weekly.loc[common_start:common_end]

    # 周收益率
    bond_ret = {k: v.pct_change() for k, v in bond_aligned.items()}

    # 信号
    alloc = compute_signal(bond_aligned, signal_aligned)

    # 交易成本
    txn_cost = PARAMS['txn_cost_bps'] / 10000

    bond_names = list(bond_weekly.keys())
    common_idx = signal_aligned.index

    portfolio_value = [1.0]
    prev_weights = {n: 0.0 for n in bond_names}
    weekly_returns = []

    for i in range(1, len(common_idx)):
        idx = common_idx[i]
        prev_idx = common_idx[i - 1]

        if prev_idx not in alloc.index:
            weekly_returns.append(0.0)
            portfolio_value.append(portfolio_value[-1])
            continue

        curr_weights = {n: alloc.loc[prev_idx, n] for n in bond_names}

        # 交易成本
        turnover = sum(abs(curr_weights.get(n, 0) - prev_weights.get(n, 0)) for n in bond_names)
        cost = turnover * txn_cost

        # 收益
        ret = 0.0
        for name in bond_names:
            w = curr_weights.get(name, 0)
            if w > 0 and idx in bond_ret[name].index:
                r = bond_ret[name].loc[idx]
                if not pd.isna(r):
                    ret += w * r
        ret -= cost

        weekly_returns.append(ret)
        portfolio_value.append(portfolio_value[-1] * (1 + ret))
        prev_weights = curr_weights

    portfolio_value = pd.Series(portfolio_value, index=common_idx)
    weekly_returns = pd.Series(weekly_returns, index=common_idx[1:])

    # 基准: 中证全债买入持有
    benchmark = signal_aligned / signal_aligned.iloc[0]

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

    # 防守比例
    defensive_pct = alloc['defensive'].mean() * 100 if 'defensive' in alloc.columns else 0

    # 各资产持仓时间占比
    holding_pct = {}
    for name in bond_names:
        holding_pct[name] = round((alloc[name] > 0).mean() * 100, 1)

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
        'defensive_pct': round(defensive_pct, 1),
        'holding_pct': holding_pct,
        'portfolio_value': portfolio_value,
        'benchmark': benchmark,
    }


# ============================================================
# 信号生成 (实盘)
# ============================================================

def generate_signal() -> Dict:
    """生成本周调仓信号"""
    print("=" * 60)
    print("策略A: 纯债ETF轮动 - 周度信号")
    print("=" * 60)

    # 加载数据
    bond_weekly = {}
    for code, info in BOND_POOL.items():
        print(f"  加载 {info['name']} ({code})...")
        df = fetch_index(code, days=600)
        if df is None:
            print(f"  [WARN] 跳过 {info['name']}")
            continue
        weekly = df['close'].resample('W-FRI').last().dropna()
        bond_weekly[info['name']] = weekly
        time.sleep(0.3)

    print(f"  加载 {SIGNAL_INDEX['name']} ({SIGNAL_INDEX['code']})...")
    sig_df = fetch_index(SIGNAL_INDEX['code'], days=600)
    if sig_df is None:
        return {'error': '无法加载全债指数'}
    signal_weekly = sig_df['close'].resample('W-FRI').last().dropna()

    # 计算信号
    alloc = compute_signal(bond_weekly, signal_weekly)
    latest = alloc.iloc[-1]

    ma_val = signal_weekly.rolling(PARAMS['ma_defensive']).mean().iloc[-1]
    price = signal_weekly.iloc[-1]
    is_def = latest.get('defensive', 0) > 0

    signal = {
        'strategy': '策略A-纯债ETF轮动',
        'date': str(signal_weekly.index[-1].date()),
        'defensive': bool(is_def),
        'allocations': {},
        'signal_price': round(price, 2),
        'signal_ma': round(ma_val, 2),
    }

    if '--json' not in sys.argv:
        print(f"\n📊 信号日期: {signal['date']}")
        print(f"中证全债: {price:.2f}, 13周MA: {ma_val:.2f}")
        print(f"状态: {'⚠️ 防守（全切短债）' if is_def else '✅ 正常轮动'}")
        print(f"\n🎯 配置建议:")

    bond_names = list(bond_weekly.keys())
    for name in bond_names:
        w = latest.get(name, 0)
        if w > 0:
            info = [v for v in BOND_POOL.values() if v['name'] == name]
            etf_str = f"{info[0]['etf']} {info[0]['etf_name']}" if info else name
            signal['allocations'][name] = round(w * 100)
            if '--json' not in sys.argv:
                print(f"  {name} ({etf_str}): {w*100:.0f}%")

    return signal


# ============================================================
# 主入口
# ============================================================

def main():
    if '--backtest' in sys.argv:
        print("=" * 60)
        print("策略A: 纯债ETF动量轮动 - 完整回测")
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

        print(f"  {SIGNAL_INDEX['name']} ({SIGNAL_INDEX['code']})...")
        sig_df = fetch_index(SIGNAL_INDEX['code'], days=days)
        if sig_df is None:
            print("ERROR: 无法加载全债指数")
            sys.exit(1)
        signal_weekly = sig_df['close'].resample('W-FRI').last().dropna()

        for name, series in bond_weekly.items():
            print(f"  {name}: {series.index[0].date()} ~ {series.index[-1].date()} ({len(series)}周)")

        result = run_backtest(bond_weekly, signal_weekly)

        print(f"\n{'='*60}")
        print(f"回测结果 ({result['years']}年)")
        print(f"{'='*60}")
        print(f"  年化收益 (CAGR):  {result['cagr']}%")
        print(f"  最大回撤 (MDD):   {result['max_drawdown']}%")
        print(f"  夏普比率:         {result['sharpe']}")
        print(f"  卡玛比率:         {result['calmar']}")
        print(f"\n  基准 (中证全债买入持有):")
        print(f"    年化收益:       {result['benchmark_cagr']}%")
        print(f"    最大回撤:       {result['benchmark_max_dd']}%")
        print(f"\n  防守比例:         {result['defensive_pct']}%")
        print(f"\n  各资产持仓时间占比:")
        for name, pct in result['holding_pct'].items():
            print(f"    {name}: {pct}%")
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
