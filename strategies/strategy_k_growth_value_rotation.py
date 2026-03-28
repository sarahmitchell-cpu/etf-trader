#!/usr/bin/env python3
"""
Strategy K: 300成长/价值 RSI动量轮动 (Growth-Value RSI Momentum Rotation)

核心逻辑:
  计算300成长(000918)与300价值(000919)价格比值的RSI指标。
  RSI > 55 → 成长风格占优，持有成长ETF
  RSI < 45 → 价值风格占优，持有价值ETF
  45 <= RSI <= 55 → 维持当前持仓（中性区不操作）

  这是趋势跟踪策略（非均值回归）：
  成长/价值比值走强(RSI高) → 继续持有成长（动量延续）
  成长/价值比值走弱(RSI低) → 切换到价值（动量延续）

可选叠加: 沪深300均线止损
  当沪深300 < N日均线时，全部转为现金（规避系统性风险）

回测结果 (2010-06 ~ 2026-03, ~16年, 含0.1%交易成本):
  P7 H55/L45:  CAGR 23.1%, MDD -39.1%, Sharpe 1.082
  P14 H55/L45: CAGR 15.9%, MDD -40.4%, Sharpe 0.798
  P7+MA120止损: CAGR 27.4%, MDD -18.1%, Sharpe 1.703 (14.8年)

  同期基准:
  纯持有成长: CAGR 3.7%, MDD -62.8%
  纯持有价值: CAGR 3.9%, MDD -40.8%

执行参数:
  - RSI周期: 7日（默认，可配置）
  - 阈值: 55/45
  - 标的ETF: 562310/159523(成长), 159510(价值)
  - 止损: 沪深300 MA120（可选，默认关闭）
  - 调仓频率: 信号驱动（非固定周期），约每年25次

数据源: 本地CSV (000918_daily.csv, 000919_daily.csv, 000300_daily.csv)

用法:
  python3 strategy_k_growth_value_rotation.py              # 生成当前信号
  python3 strategy_k_growth_value_rotation.py --json       # 仅输出JSON
  python3 strategy_k_growth_value_rotation.py --backtest   # 运行完整回测
  python3 strategy_k_growth_value_rotation.py --stoploss   # 启用MA止损回测
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

# ============================================================
# Configuration
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

PARAMS = {
    'rsi_period': 7,             # RSI周期（日）
    'threshold_high': 55,        # RSI > 此值 → 持有成长
    'threshold_low': 45,         # RSI < 此值 → 持有价值
    'txn_cost_bps': 10,          # 单边交易成本（基点）
    'ma_stoploss_enabled': False,  # 是否启用MA止损
    'ma_stoploss_period': 120,   # MA止损均线周期（日）
}

# 指数代码
INDICES = {
    'growth': {'code': '000918', 'name': '300成长', 'file': '000918_daily.csv'},
    'value':  {'code': '000919', 'name': '300价值', 'file': '000919_daily.csv'},
    'csi300': {'code': '000300', 'name': '沪深300',  'file': '000300_daily.csv'},
}

# ETF映射
ETFS = {
    'growth': [
        {'code': '562310', 'name': '300成长ETF(富国)'},
        {'code': '159523', 'name': '300成长ETF(易方达)'},
    ],
    'value': [
        {'code': '159510', 'name': '300价值ETF'},
    ],
}


# ============================================================
# Data Loading
# ============================================================

def load_index_data(index_key):
    """加载指数日线数据"""
    info = INDICES[index_key]
    filepath = os.path.join(DATA_DIR, info['file'])

    if index_key == 'csi300':
        # CSI300文件使用英文列名
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df['close']
    else:
        # 成长/价值使用中文列名
        df = pd.read_csv(filepath)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.set_index('日期').sort_index()
        return df['收盘']


def load_all_data():
    """加载所有数据并对齐日期"""
    growth = load_index_data('growth')
    value = load_index_data('value')

    # 对齐成长和价值的日期
    common = growth.index.intersection(value.index)
    g = growth.loc[common]
    v = value.loc[common]

    # 加载沪深300（如果需要止损）
    csi300 = None
    if PARAMS['ma_stoploss_enabled']:
        try:
            csi300 = load_index_data('csi300')
        except Exception as e:
            print(f"  [WARN] 无法加载沪深300数据: {e}", file=sys.stderr)

    print(f"数据范围: {common[0].date()} ~ {common[-1].date()}, {len(common)}个交易日")
    return g, v, csi300


# ============================================================
# RSI Computation
# ============================================================

def compute_rsi(series, period):
    """计算RSI指标"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_ratio_rsi(g, v, period=None):
    """计算成长/价值比值的RSI"""
    if period is None:
        period = PARAMS['rsi_period']
    ratio = g / v
    return compute_rsi(ratio, period)


# ============================================================
# Signal Generation
# ============================================================

def generate_signal(g, v, csi300=None):
    """
    生成当前交易信号。

    Returns:
        信号字典
    """
    rsi = compute_ratio_rsi(g, v)
    latest_date = g.index[-1]
    latest_rsi = rsi.iloc[-1]

    # 回溯最近的持仓状态
    holding = determine_current_holding(rsi)

    # MA止损检查
    in_market = True
    ma_val = None
    if PARAMS['ma_stoploss_enabled'] and csi300 is not None:
        ma = csi300.rolling(PARAMS['ma_stoploss_period']).mean()
        if latest_date in ma.index and not pd.isna(ma.loc[latest_date]):
            ma_val = ma.loc[latest_date]
            csi300_price = csi300.loc[latest_date]
            in_market = csi300_price >= ma_val

    if not in_market:
        action = 'cash'
        action_cn = '空仓（MA止损触发）'
    elif holding == 'growth':
        action = 'growth'
        action_cn = '持有成长ETF'
    else:
        action = 'value'
        action_cn = '持有价值ETF'

    # 近期RSI走势
    rsi_history = []
    for i in range(-5, 0):
        if abs(i) < len(rsi):
            d = rsi.index[i]
            rsi_history.append({
                'date': d.strftime('%Y-%m-%d'),
                'rsi': round(float(rsi.iloc[i]), 1),
            })

    signal = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': 'Strategy K (300成长/价值RSI动量轮动)',
        'params': {
            'rsi_period': PARAMS['rsi_period'],
            'threshold_high': PARAMS['threshold_high'],
            'threshold_low': PARAMS['threshold_low'],
            'ma_stoploss': PARAMS['ma_stoploss_enabled'],
            'ma_period': PARAMS['ma_stoploss_period'] if PARAMS['ma_stoploss_enabled'] else None,
        },
        'current_rsi': round(float(latest_rsi), 1),
        'rsi_zone': _rsi_zone(latest_rsi),
        'action': action,
        'action_cn': action_cn,
        'in_market': in_market,
        'growth_price': round(float(g.iloc[-1]), 2),
        'value_price': round(float(v.iloc[-1]), 2),
        'ratio': round(float(g.iloc[-1] / v.iloc[-1]), 4),
        'rsi_history': rsi_history,
        'etfs': ETFS,
        'action_summary': [],
    }

    # 构建中文摘要
    rsi_emoji = '🟢' if action == 'growth' else ('🔴' if action == 'value' else '⚪')
    signal['action_summary'] = [
        f"{rsi_emoji} 当前信号: {action_cn}",
        f"RSI({PARAMS['rsi_period']}): {signal['current_rsi']} ({signal['rsi_zone']})",
        f"成长/价值比: {signal['ratio']}",
        f"300成长: {signal['growth_price']}",
        f"300价值: {signal['value_price']}",
    ]

    if action == 'growth':
        etf_str = ' / '.join([f"{e['code']} {e['name']}" for e in ETFS['growth']])
        signal['action_summary'].append(f"操作: 全仓持有 {etf_str}")
    elif action == 'value':
        etf_str = ' / '.join([f"{e['code']} {e['name']}" for e in ETFS['value']])
        signal['action_summary'].append(f"操作: 全仓持有 {etf_str}")
    else:
        signal['action_summary'].append(f"操作: 空仓 (沪深300 < MA{PARAMS['ma_stoploss_period']})")

    if PARAMS['ma_stoploss_enabled'] and ma_val is not None:
        csi300_price = csi300.loc[latest_date] if latest_date in csi300.index else None
        if csi300_price is not None:
            signal['action_summary'].append(
                f"沪深300: {csi300_price:.2f} / MA{PARAMS['ma_stoploss_period']}: {ma_val:.2f} "
                f"({'上方' if in_market else '下方'})"
            )

    return signal


def determine_current_holding(rsi):
    """根据RSI历史回溯确定当前持仓"""
    holding = 'growth'  # 默认
    hi = PARAMS['threshold_high']
    lo = PARAMS['threshold_low']

    for i in range(len(rsi)):
        val = rsi.iloc[i]
        if pd.isna(val):
            continue
        if val > hi:
            holding = 'growth'
        elif val < lo:
            holding = 'value'

    return holding


def _rsi_zone(rsi_val):
    """RSI区间描述"""
    if pd.isna(rsi_val):
        return '数据不足'
    if rsi_val > PARAMS['threshold_high']:
        return '成长区（看多成长）'
    elif rsi_val < PARAMS['threshold_low']:
        return '价值区（看多价值）'
    else:
        return '中性区（维持现有）'


# ============================================================
# Backtest
# ============================================================

def run_backtest(g, v, csi300=None, start='2010-06-01',
                 rsi_period=None, hi=None, lo=None,
                 ma_stoploss=False, ma_period=120):
    """
    运行完整回测。

    策略规则:
      - T日收盘后计算RSI → T+1日按信号持仓
      - 交易日切换持仓时扣除交易成本
      - (可选) 沪深300 < MA时空仓

    Returns:
        (nav_series, metrics_dict, trades_count)
    """
    if rsi_period is None:
        rsi_period = PARAMS['rsi_period']
    if hi is None:
        hi = PARAMS['threshold_high']
    if lo is None:
        lo = PARAMS['threshold_low']

    txn_cost = PARAMS['txn_cost_bps'] / 10000

    # 计算RSI（全量数据，避免截断导致warmup不足）
    ratio = g / v
    rsi = compute_rsi(ratio, rsi_period)

    # MA止损
    csi_ma = None
    if ma_stoploss and csi300 is not None:
        csi_ma = csi300.rolling(ma_period).mean()

    # 截取回测区间
    g2 = g[g.index >= start].copy()
    v2 = v[v.index >= start].copy()
    rsi2 = rsi[rsi.index >= start]
    dates = g2.index

    pv = pd.Series(1.0, index=dates)
    holding = 'growth'
    trades = 0

    for i in range(1, len(dates)):
        dt = dates[i]
        rsi_val = rsi2.loc[dt] if dt in rsi2.index and not pd.isna(rsi2.loc[dt]) else 50

        old_holding = holding

        # MA止损检查
        if ma_stoploss and csi_ma is not None and dt in csi_ma.index:
            ma_val = csi_ma.loc[dt]
            if not pd.isna(ma_val) and dt in csi300.index and csi300.loc[dt] < ma_val:
                holding = 'cash'
            else:
                # 在市场中，按RSI决策
                if rsi_val > hi:
                    holding = 'growth'
                elif rsi_val < lo:
                    holding = 'value'
                elif holding == 'cash':
                    holding = 'growth'  # 从空仓重新入场默认成长
        else:
            # 无止损，纯RSI
            if rsi_val > hi:
                holding = 'growth'
            elif rsi_val < lo:
                holding = 'value'

        # 日收益
        if holding == 'growth':
            ret = g2.iloc[i] / g2.iloc[i-1] - 1
        elif holding == 'value':
            ret = v2.iloc[i] / v2.iloc[i-1] - 1
        else:
            ret = 0  # cash

        # 交易成本
        if old_holding != holding:
            trades += 1
            pv.iloc[i] = pv.iloc[i-1] * (1 + ret) * (1 - txn_cost)
        else:
            pv.iloc[i] = pv.iloc[i-1] * (1 + ret)

    # 计算指标
    metrics = calc_metrics(pv)
    if metrics:
        metrics['trades'] = trades
        years = (pv.index[-1] - pv.index[0]).days / 365.25
        metrics['trades_per_year'] = round(trades / years, 1)

    return pv, metrics, trades


def calc_metrics(pv):
    """计算策略指标"""
    if pv is None or len(pv) < 50:
        return None
    years = (pv.index[-1] - pv.index[0]).days / 365.25
    if years < 0.5:
        return None

    total_ret = pv.iloc[-1] / pv.iloc[0]
    cagr = total_ret ** (1/years) - 1
    mdd = (pv / pv.cummax() - 1).min()
    daily_rets = pv.pct_change().dropna()
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    # 年度收益
    annual = {}
    for year in range(pv.index[0].year, pv.index[-1].year + 1):
        yr = pv[pv.index.year == year]
        if len(yr) >= 10:
            annual[str(year)] = round((yr.iloc[-1] / yr.iloc[0] - 1) * 100, 1)

    # 年度胜率
    wins = sum(1 for v in annual.values() if v > 0)
    total_years = len(annual)

    return {
        'cagr_pct': round(cagr * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'total_return_pct': round((total_ret - 1) * 100, 1),
        'years': round(years, 1),
        'annual_returns': annual,
        'annual_win_rate': f"{wins}/{total_years}",
    }


def run_full_backtest(g, v, csi300):
    """运行完整对比回测"""
    start = '2010-06-01'

    print(f"\n{'='*70}")
    print(f"Strategy K 回测: {start} ~ {g.index[-1].date()}")
    print(f"{'='*70}")

    results = {}

    # 主策略: P7 H55/L45
    pv, m, trades = run_backtest(g, v, csi300, start, rsi_period=7)
    results['RSI_P7_H55L45'] = m
    print(f"\n主策略 RSI(7) H55/L45:")
    _print_metrics(m)

    # 备选: P14 H55/L45
    pv14, m14, _ = run_backtest(g, v, csi300, start, rsi_period=14)
    results['RSI_P14_H55L45'] = m14
    print(f"\n备选 RSI(14) H55/L45:")
    _print_metrics(m14)

    # MA止损版本
    if csi300 is not None:
        # 需要MA warmup，从2011-06开始
        ma_start = '2011-06-01'

        pv_ma120, m_ma120, _ = run_backtest(
            g, v, csi300, ma_start, rsi_period=7,
            ma_stoploss=True, ma_period=120
        )
        results['RSI_P7+MA120止损'] = m_ma120
        print(f"\nMA止损版 RSI(7)+MA120 (从{ma_start}):")
        _print_metrics(m_ma120)

        pv_ma60, m_ma60, _ = run_backtest(
            g, v, csi300, ma_start, rsi_period=7,
            ma_stoploss=True, ma_period=60
        )
        results['RSI_P7+MA60止损'] = m_ma60
        print(f"\nMA止损版 RSI(7)+MA60 (从{ma_start}):")
        _print_metrics(m_ma60)

    # 基准
    gs = g[g.index >= start]
    vs = v[v.index >= start]

    pv_g = gs / gs.iloc[0]
    m_g = calc_metrics(pv_g)
    results['纯持有成长'] = m_g
    print(f"\n基准-纯持有成长:")
    _print_metrics(m_g)

    pv_v = vs / vs.iloc[0]
    m_v = calc_metrics(pv_v)
    results['纯持有价值'] = m_v
    print(f"\n基准-纯持有价值:")
    _print_metrics(m_v)

    # 保存
    output_path = os.path.join(DATA_DIR, 'strategy_k_backtest.json')
    serializable = {}
    for k, v_dict in results.items():
        if v_dict:
            serializable[k] = v_dict
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n回测结果已保存: {output_path}")

    return results


def _print_metrics(m):
    """打印指标"""
    if m is None:
        print("  无有效数据")
        return
    print(f"  CAGR:    {m['cagr_pct']:>7.1f}%")
    print(f"  MDD:     {m['mdd_pct']:>7.1f}%")
    print(f"  Sharpe:  {m['sharpe']:>7.3f}")
    print(f"  Calmar:  {m['calmar']:>7.3f}")
    print(f"  总收益:  {m['total_return_pct']:>7.1f}%")
    if 'trades' in m:
        print(f"  交易次数: {m['trades']} (年均{m['trades_per_year']})")
    print(f"  年胜率:  {m.get('annual_win_rate', 'N/A')}")
    if 'annual_returns' in m:
        ann_str = ' | '.join([f"{y}:{v:+.1f}%" for y, v in sorted(m['annual_returns'].items())])
        print(f"  逐年: {ann_str}")


# ============================================================
# Output Formatting
# ============================================================

def format_signal_text(signal):
    """格式化信号为可读文本"""
    lines = [
        f"{'='*45}",
        f"策略K 成长/价值轮动信号",
        f"日期: {signal['date']}",
        f"{'='*45}",
        "",
    ]
    for line in signal['action_summary']:
        lines.append(line)
    lines.extend([
        "",
        f"策略参数: RSI({signal['params']['rsi_period']}) 阈值 {signal['params']['threshold_high']}/{signal['params']['threshold_low']}",
        f"生成时间: {signal['generated_at']}",
    ])
    return '\n'.join(lines)


# ============================================================
# Main
# ============================================================

def main():
    args = sys.argv[1:]
    json_only = '--json' in args
    do_backtest = '--backtest' in args
    use_stoploss = '--stoploss' in args

    if use_stoploss:
        PARAMS['ma_stoploss_enabled'] = True

    if not json_only:
        print("=" * 55)
        print(f"策略K: 300成长/价值 RSI动量轮动")
        print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 55)

    g, v, csi300 = load_all_data()

    if do_backtest:
        # 回测时需要沪深300数据
        if csi300 is None:
            try:
                csi300 = load_index_data('csi300')
            except Exception:
                pass
        run_full_backtest(g, v, csi300)
        return

    # 生成当前信号
    signal = generate_signal(g, v, csi300)

    # 保存
    output_path = os.path.join(DATA_DIR, 'strategy_k_latest_signal.json')
    with open(output_path, 'w') as f:
        json.dump(signal, f, indent=2, ensure_ascii=False)

    if json_only:
        print(json.dumps(signal, indent=2, ensure_ascii=False))
    else:
        print(format_signal_text(signal))
        print(f"\n信号已保存: {output_path}")

    return signal


if __name__ == '__main__':
    main()
