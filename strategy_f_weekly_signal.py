#!/usr/bin/env python3
"""
Strategy F: 美股QDII动量轮动策略

核心逻辑:
  1. 标的池 (人民币计价的美股指数, 回测用):
     - 中证纳斯达克100 (H30533) → ETF 159941
     - 中证标普500 (H30140) → ETF 513500
     + 避险: 中证短债 (H11006) → ETF 511360
  2. 动量排名: 4周收益率排名, Top1配100%
  3. 防守机制: 纳指100低于20周MA → 全切短债避险
  4. 调仓频率: 周频 (每周五)
  5. 交易成本: 单边15bp (QDII溢价)

重要说明:
  - 回测使用中证编制的人民币计价指数, 已含汇率波动
  - T-1 数据决策、T 期收益, 无前瞻偏差
  - 实盘需注意QDII限额和折溢价风险
  - 513850(美国50ETF)成立太晚, 暂不纳入

用法:
  python3 strategy_f_weekly_signal.py              # 正常运行
  python3 strategy_f_weekly_signal.py --json       # 仅输出JSON
  python3 strategy_f_weekly_signal.py --backtest   # 运行完整回测

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

# 美股指数池 (人民币计价, 中证编制)
US_POOL = {
    'H30533': {'name': '纳指100', 'etf': '159941', 'etf_name': '纳指ETF'},
    'H30140': {'name': '标普500', 'etf': '513500', 'etf_name': '标普500ETF'},
}

# 避险资产: 短债
SAFE_HAVEN = {'code': 'H11006', 'name': '短债', 'etf': '511360', 'etf_name': '短融ETF'}

# 信号: 纳指100 (主力品种)
SIGNAL_CODE = 'H30533'

# 策略参数
PARAMS = {
    'momentum_lookback': 4,     # 动量回看周数
    'ma_fast': 8,               # 快速均线周数
    'ma_slow': 20,              # 慢速均线周数
    'dd_limit': -0.08,          # 回撤止损阈值 (-8%)
    'dd_lockout': 8,            # 止损后锁定周数
    'txn_cost_bps': 15,         # 单边交易成本 (QDII更高)
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
        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        if len(df) > 0:
            return df[['close']]

    return None


# ============================================================
# 信号计算
# ============================================================

def compute_regime(signal_series: pd.Series) -> pd.Series:
    """
    双均线判断市场状态:
    - bull: 价格 > 快MA 且 > 慢MA
    - transition: 价格在两条MA之间
    - bear: 价格 < 快MA 且 < 慢MA
    """
    ma_fast = signal_series.rolling(PARAMS['ma_fast']).mean()
    ma_slow = signal_series.rolling(PARAMS['ma_slow']).mean()

    regime = pd.Series('bear', index=signal_series.index)
    regime[(signal_series > ma_fast) | (signal_series > ma_slow)] = 'transition'
    regime[(signal_series > ma_fast) & (signal_series > ma_slow)] = 'bull'
    return regime


def compute_signal(us_weekly: Dict[str, pd.Series], safe_weekly: pd.Series) -> pd.DataFrame:
    """
    计算每周配置 (双MA + 回撤止损)。
    注意: 回撤止损在回测引擎中实现, 这里只处理MA信号。
    """
    us_names = list(us_weekly.keys())
    signal_name = [v['name'] for k, v in US_POOL.items() if k == SIGNAL_CODE][0]
    signal_series = us_weekly[signal_name]

    regime = compute_regime(signal_series)

    # 动量
    mom = pd.DataFrame()
    for name, series in us_weekly.items():
        mom[name] = series.pct_change(PARAMS['momentum_lookback'])

    # 配置
    alloc = pd.DataFrame(0.0, index=signal_series.index,
                         columns=us_names + ['safe', 'regime_str'])

    start_idx = max(PARAMS['momentum_lookback'], PARAMS['ma_slow'])
    for i in range(start_idx, len(signal_series)):
        idx = signal_series.index[i]
        r = regime.iloc[i]
        alloc.loc[idx, 'regime_str'] = r

        if r == 'bear':
            alloc.loc[idx, 'safe'] = 1.0
        elif r == 'transition':
            # 50% top momentum US + 50% safe
            mom_row = mom.loc[idx].dropna() if idx in mom.index else pd.Series()
            if len(mom_row) > 0:
                best = mom_row.sort_values(ascending=False).index[0]
                alloc.loc[idx, best] = 0.5
            alloc.loc[idx, 'safe'] = 0.5
        else:  # bull
            # 100% top momentum US
            mom_row = mom.loc[idx].dropna() if idx in mom.index else pd.Series()
            if len(mom_row) > 0:
                best = mom_row.sort_values(ascending=False).index[0]
                alloc.loc[idx, best] = 1.0
            else:
                alloc.loc[idx, 'safe'] = 1.0

    return alloc


# ============================================================
# 回测引擎
# ============================================================

def run_backtest(us_weekly: Dict[str, pd.Series], safe_weekly: pd.Series) -> Dict:
    """
    运行美股QDII动量回测。
    包含回撤止损机制: 当组合回撤超过dd_limit时, 全切短债并锁定dd_lockout周。
    """
    all_series = list(us_weekly.values()) + [safe_weekly]
    common_start = max(s.index[0] for s in all_series)
    common_end = min(s.index[-1] for s in all_series)

    us_aligned = {k: v.loc[common_start:common_end] for k, v in us_weekly.items()}
    safe_aligned = safe_weekly.loc[common_start:common_end]

    us_ret = {k: v.pct_change() for k, v in us_aligned.items()}
    safe_ret = safe_aligned.pct_change()

    alloc = compute_signal(us_aligned, safe_aligned)

    txn_cost = PARAMS['txn_cost_bps'] / 10000
    dd_limit = PARAMS['dd_limit']
    dd_lockout_weeks = PARAMS['dd_lockout']
    us_names = list(us_weekly.keys())
    all_assets = us_names + ['safe']

    common_idx = safe_aligned.index
    portfolio_value = [1.0]
    prev_weights = {a: 0.0 for a in all_assets}
    weekly_returns = []
    peak = 1.0
    dd_lockout = 0  # remaining lockout weeks

    for i in range(1, len(common_idx)):
        idx = common_idx[i]
        prev_idx = common_idx[i - 1]

        curr_pv = portfolio_value[-1]

        # Check drawdown lockout
        if dd_lockout > 0:
            dd_lockout -= 1
            # Force safe haven
            curr_weights = {a: 0.0 for a in all_assets}
            curr_weights['safe'] = 1.0
        else:
            # Check if drawdown exceeds limit
            curr_dd = (curr_pv - peak) / peak if peak > 0 else 0
            if curr_dd < dd_limit:
                dd_lockout = dd_lockout_weeks
                curr_weights = {a: 0.0 for a in all_assets}
                curr_weights['safe'] = 1.0
            elif prev_idx not in alloc.index:
                weekly_returns.append(0.0)
                portfolio_value.append(curr_pv)
                continue
            else:
                curr_weights = {}
                for name in us_names:
                    curr_weights[name] = alloc.loc[prev_idx, name]
                curr_weights['safe'] = alloc.loc[prev_idx, 'safe']

        turnover = sum(abs(curr_weights.get(a, 0) - prev_weights.get(a, 0)) for a in all_assets)
        cost = turnover * txn_cost

        ret = 0.0
        for name in us_names:
            w = curr_weights.get(name, 0)
            if w > 0 and idx in us_ret[name].index:
                r = us_ret[name].loc[idx]
                if not pd.isna(r):
                    ret += w * r
        w_safe = curr_weights.get('safe', 0)
        if w_safe > 0 and idx in safe_ret.index:
            r = safe_ret.loc[idx]
            if not pd.isna(r):
                ret += w_safe * r

        ret -= cost
        weekly_returns.append(ret)
        new_pv = curr_pv * (1 + ret)
        portfolio_value.append(new_pv)
        peak = max(peak, new_pv)
        prev_weights = curr_weights

    portfolio_value = pd.Series(portfolio_value, index=common_idx)
    weekly_returns = pd.Series(weekly_returns, index=common_idx[1:])

    # 基准: 等权持有两个美股指数
    bm_parts = [v / v.iloc[0] for v in us_aligned.values()]
    benchmark = sum(bm_parts) / len(bm_parts)

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

    annual_returns = {}
    for year in range(common_idx[0].year, common_idx[-1].year + 1):
        year_data = portfolio_value[portfolio_value.index.year == year]
        if len(year_data) >= 2:
            annual_returns[year] = year_data.iloc[-1] / year_data.iloc[0] - 1

    defensive_pct = (alloc['regime_str'] == 'bear').mean() * 100 if 'regime_str' in alloc.columns else 0

    # 持仓分布
    holding_pct = {}
    for name in us_names:
        holding_pct[name] = round((alloc[name] > 0).mean() * 100, 1)
    holding_pct['避险(短债)'] = round((alloc['safe'] > 0).mean() * 100, 1)

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
    print("策略F: 美股QDII动量 - 周度信号")
    print("=" * 60)

    us_weekly = {}
    for code, info in US_POOL.items():
        print(f"  加载 {info['name']} ({code})...")
        df = fetch_index(code, days=600)
        if df is None:
            print(f"  [WARN] 跳过 {info['name']}")
            continue
        weekly = df['close'].resample('W-FRI').last().dropna()
        us_weekly[info['name']] = weekly
        time.sleep(0.3)

    print(f"  加载 {SAFE_HAVEN['name']} ({SAFE_HAVEN['code']})...")
    safe_df = fetch_index(SAFE_HAVEN['code'], days=600)
    if safe_df is None:
        return {'error': '无法加载避险资产数据'}
    safe_weekly = safe_df['close'].resample('W-FRI').last().dropna()

    alloc = compute_signal(us_weekly, safe_weekly)
    latest = alloc.iloc[-1]

    signal_name = US_POOL[SIGNAL_CODE]['name']
    signal_series = us_weekly[signal_name]
    ma_val = signal_series.rolling(PARAMS['ma_slow']).mean().iloc[-1]
    price = signal_series.iloc[-1]
    is_def = latest.get('defensive', 0) > 0

    signal = {
        'strategy': '策略F-美股QDII动量',
        'date': str(signal_series.index[-1].date()),
        'defensive': bool(is_def),
        'allocations': {},
        'nasdaq_price': round(price, 2),
        'nasdaq_ma20w': round(ma_val, 2),
    }

    if '--json' not in sys.argv:
        print(f"\n📊 信号日期: {signal['date']}")
        print(f"纳指100: {price:.2f}, 20周MA: {ma_val:.2f}")
        print(f"状态: {'⚠️ 防守（全切短债）' if is_def else '✅ 进攻'}")
        print(f"\n🎯 配置建议:")

    us_names = list(us_weekly.keys())
    for name in us_names:
        w = latest.get(name, 0)
        if w > 0:
            info = [v for v in US_POOL.values() if v['name'] == name]
            etf_str = f"{info[0]['etf']} {info[0]['etf_name']}" if info else name
            signal['allocations'][name] = round(w * 100)
            if '--json' not in sys.argv:
                print(f"  {name} ({etf_str}): {w*100:.0f}%")

    safe_w = latest.get('safe', 0)
    if safe_w > 0:
        signal['allocations']['短债避险'] = round(safe_w * 100)
        if '--json' not in sys.argv:
            print(f"  短债避险 ({SAFE_HAVEN['etf']} {SAFE_HAVEN['etf_name']}): {safe_w*100:.0f}%")

    return signal


# ============================================================
# 主入口
# ============================================================

def main():
    if '--backtest' in sys.argv:
        print("=" * 60)
        print("策略F: 美股QDII动量 - 完整回测")
        print("=" * 60)

        days = 5500
        print(f"\n加载数据 (回测模式, ~{days//365}年)...")

        us_weekly = {}
        for code, info in US_POOL.items():
            print(f"  {info['name']} ({code})...")
            df = fetch_index(code, days=days)
            if df is None:
                print(f"  [ERROR] 无法加载 {info['name']}")
                sys.exit(1)
            weekly = df['close'].resample('W-FRI').last().dropna()
            us_weekly[info['name']] = weekly
            time.sleep(0.3)

        print(f"  {SAFE_HAVEN['name']} ({SAFE_HAVEN['code']})...")
        safe_df = fetch_index(SAFE_HAVEN['code'], days=days)
        if safe_df is None:
            print("ERROR: 无法加载避险资产")
            sys.exit(1)
        safe_weekly = safe_df['close'].resample('W-FRI').last().dropna()

        for name, series in us_weekly.items():
            print(f"  {name}: {series.index[0].date()} ~ {series.index[-1].date()} ({len(series)}周)")

        result = run_backtest(us_weekly, safe_weekly)

        print(f"\n{'='*60}")
        print(f"回测结果 ({result['years']}年)")
        print(f"{'='*60}")
        print(f"  年化收益 (CAGR):  {result['cagr']}%")
        print(f"  最大回撤 (MDD):   {result['max_drawdown']}%")
        print(f"  夏普比率:         {result['sharpe']}")
        print(f"  卡玛比率:         {result['calmar']}")
        print(f"  防守比例:         {result['defensive_pct']}%")
        print(f"\n  基准 (等权美股买入持有):")
        print(f"    年化收益:       {result['benchmark_cagr']}%")
        print(f"    最大回撤:       {result['benchmark_max_dd']}%")
        print(f"\n  持仓分布:")
        for name, pct in result['holding_pct'].items():
            print(f"    {name}: {pct}%")
        print(f"\n  年度收益:")
        for year, ret in sorted(result['annual_returns'].items()):
            print(f"    {year}: {ret:+.1f}%")

        result_data = {k: v for k, v in result.items()
                       if k not in ('portfolio_value', 'benchmark')}
        result_path = os.path.join(DATA_DIR, 'strategy_f_backtest.json')
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {result_path}")

    else:
        signal = generate_signal()
        if '--json' in sys.argv:
            print(json.dumps(signal, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
