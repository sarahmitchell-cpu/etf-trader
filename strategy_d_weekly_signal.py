#!/usr/bin/env python3
"""
Strategy D: A股龙头股动量周度信号

每两周运行，输出调仓建议。

核心逻辑:
  1. 股票池: 28只各行业龙头A股
  2. 动量排名: 4周动量, 跳过最近1周(避免短期反转)
  3. Top8等权: 选动量最强的8只, 等权持有
  4. 行业约束: 每行业最多1只 (sector_max=1, 分散风险)
  5. 双周调仓

回测表现 (2021.03~2026.03, 5年, 无前瞻偏差):
  - CAGR=14.2%, MDD=-20.2%, Sharpe=0.81, Calmar=0.70
  - 年度: 2022:+6.4%, 2023:-0.9%, 2024:+24.6%, 2025:+28.3%
  - 基准(等权全持): CAGR=8.1%

数据来源: Yahoo Finance API (via stock_data_common)
交易成本: 8bps单边 (万3佣金+千1印花税折算)

用法:
  python3 strategy_d_weekly_signal.py              # 正常运行
  python3 strategy_d_weekly_signal.py --json       # 仅输出JSON
  python3 strategy_d_weekly_signal.py --backtest   # 运行完整回测
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
from collections import defaultdict
from typing import Tuple, List

from stock_data_common import STOCK_POOL, DATA_DIR, load_all_data

# ============================================================
# 策略参数
# ============================================================

PARAMS = {
    'momentum_lookback': 4,     # 动量回看周数
    'skip_recent': 1,           # 跳过最近N周 (避免短期反转)
    'top_n': 8,                 # 选前N名
    'rebal_freq': 2,            # 调仓频率 (周)
    'txn_cost_bps': 8,          # 单边交易成本
    'sector_max': 1,            # 每行业最多选N只 (None=不限)
}


# ============================================================
# 策略核心
# ============================================================

def select_top_momentum(price_df: pd.DataFrame, idx: int) -> Tuple[List[str], List[dict]]:
    """
    动量选股 (严格无前瞻)

    使用 price_df.iloc[:idx+1] 数据, 即最新价为 iloc[idx]
    计算: price[idx-skip] / price[idx-skip-lookback] - 1
    """
    lookback = PARAMS['momentum_lookback']
    skip = PARAMS['skip_recent']
    top_n = PARAMS['top_n']

    end_idx = idx - skip
    start_idx = end_idx - lookback

    if start_idx < 0 or end_idx <= 0:
        return [], []

    avail = [c for c in price_df.columns
             if not pd.isna(price_df[c].iloc[start_idx])
             and not pd.isna(price_df[c].iloc[end_idx])
             and price_df[c].iloc[start_idx] > 0]

    if len(avail) < 5:
        return [], []

    momenta = []
    for col in avail:
        mom = float(price_df[col].iloc[end_idx] / price_df[col].iloc[start_idx] - 1)
        momenta.append((col, mom))

    ranked = sorted(momenta, key=lambda x: (-x[1], x[0]))  # stable sort
    sector_max = PARAMS.get('sector_max')
    if sector_max:
        selected = []
        sector_count = defaultdict(int)
        for t, _ in ranked:
            if len(selected) >= top_n:
                break
            sec = STOCK_POOL.get(t, {}).get('sector', '?')
            if sector_count[sec] < sector_max:
                selected.append(t)
                sector_count[sec] += 1
    else:
        selected = [t for t, _ in ranked[:top_n]]

    details = []
    for rank_i, (ticker, mom) in enumerate(ranked):
        info = STOCK_POOL.get(ticker, {'name': ticker, 'code': ticker, 'sector': '?'})
        details.append({
            'ticker': ticker,
            'name': info['name'],
            'code': info['code'],
            'sector': info['sector'],
            'momentum_4w': round(mom * 100, 2),
            'rank': rank_i + 1,
            'selected': ticker in selected,
            'position_pct': round(100 / top_n, 1) if ticker in selected else 0.0,
        })

    return selected, details


def generate_signal(price_df: pd.DataFrame) -> dict:
    """生成本期信号"""
    idx = len(price_df) - 1
    latest_date = price_df.index[idx]

    selected, details = select_top_momentum(price_df, idx)

    signal = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': 'Strategy D (龙头股动量)',
        'variant': f"Top{PARAMS['top_n']} Mom{PARAMS['momentum_lookback']}w Skip{PARAMS['skip_recent']}",
        'rebalance_freq': f"每{PARAMS['rebal_freq']}周",
        'total_stocks': len(selected),
        'position_per_stock': round(100 / PARAMS['top_n'], 1) if selected else 0,
        'selected_stocks': [],
        'all_rankings': details,
        'action_summary': [],
    }

    for d in details:
        if d['selected']:
            signal['selected_stocks'].append({
                'code': d['code'],
                'name': d['name'],
                'sector': d['sector'],
                'momentum': d['momentum_4w'],
            })

    # 中文摘要
    lines = [
        f"本期持仓 ({len(selected)}只, 等权{round(100/PARAMS['top_n'],1)}%每只):",
        "",
    ]
    for d in details:
        if d['selected']:
            lines.append(f"  >> #{d['rank']} {d['code']} {d['name']} [{d['sector']}] {d['momentum_4w']:+.1f}%")

    lines.append("")
    lines.append("动量排名 (全部):")
    for d in details:
        icon = '>>' if d['selected'] else '  '
        lines.append(f"  {icon} #{d['rank']} {d['code']} {d['name']} [{d['sector']}] {d['momentum_4w']:+.1f}%")

    signal['action_summary'] = lines
    return signal


# ============================================================
# 回测
# ============================================================

def run_backtest(price_df: pd.DataFrame) -> dict:
    """完整回测 (严格无前瞻偏差)"""
    print("\n运行回测...")

    txn_cost = PARAMS['txn_cost_bps'] / 10000
    lookback = PARAMS['momentum_lookback']
    skip = PARAMS['skip_recent']
    top_n = PARAMS['top_n']
    rebal_freq = PARAMS['rebal_freq']

    returns = price_df.pct_change(fill_method=None)
    warmup = lookback + skip + 2

    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_returns_list = []
    holding_log = []

    i = warmup
    while i < len(price_df) - 1:
        selected, _ = select_top_momentum(price_df, i)
        if not selected:
            i += 1
            continue

        selected_set = set(selected)

        # Transaction cost
        new_buys = selected_set - prev_holdings
        sold = prev_holdings - selected_set
        turnover_pct = (len(new_buys) + len(sold)) / max(len(selected_set), 1)
        period_txn = turnover_pct * txn_cost
        total_txn += period_txn

        # Hold period returns
        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            week_rets = []
            for s in selected:
                r = returns[s].iloc[j]
                if not pd.isna(r):
                    week_rets.append(float(r))

            port_ret = np.mean(week_rets) if week_rets else 0.0
            if j == i + 1:
                port_ret -= period_txn

            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_returns_list.append(port_ret)

        holding_log.append({
            'date': price_df.index[i].strftime('%Y-%m-%d'),
            'stocks': [STOCK_POOL[s]['name'] for s in selected if s in STOCK_POOL],
        })

        prev_holdings = selected_set
        i = hold_end

    if not dates:
        return {'error': 'No trades'}

    nav_s = pd.Series(nav[1:], index=dates)
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / years) - 1
    dd = nav_s / nav_s.cummax() - 1
    mdd = dd.min()
    wr = pd.Series(weekly_returns_list)
    sharpe = wr.mean() / wr.std() * np.sqrt(52) if wr.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    win_rate = (wr > 0).sum() / len(wr) * 100

    annual = nav_s.resample('YE').last().pct_change().dropna()
    annual_returns = {str(d.year): round(v * 100, 1) for d, v in annual.items()}

    result = {
        'strategy': f'Strategy D: Top{top_n}龙头动量 (LB{lookback}w Skip{skip})',
        'period': f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}",
        'years': round(years, 1),
        'cagr_pct': round(cagr * 100, 1),
        'total_return_pct': round((nav_s.iloc[-1] - 1) * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round(win_rate, 1),
        'annual_returns': annual_returns,
        'total_txn_pct': round(total_txn * 100, 2),
        'num_rebalances': len(holding_log),
        'note': '无前瞻偏差: T周决策->T+1周收益, 交易成本8bp',
    }

    print(f"\n回测结果:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


# ============================================================
# 输出
# ============================================================

def format_signal_text(signal: dict) -> str:
    lines = [
        f"{'='*45}",
        f"策略D 龙头股动量 信号",
        f"日期: {signal['date']}",
        f"调仓频率: {signal['rebalance_freq']}",
        f"{'='*45}",
        "",
    ]
    for line in signal['action_summary']:
        lines.append(line)
    lines.extend([
        "",
        f"参数: {signal['variant']}",
        f"生成时间: {signal['generated_at']}",
    ])
    return '\n'.join(lines)


# ============================================================
# 主入口
# ============================================================

def main():
    args = sys.argv[1:]
    json_only = '--json' in args
    do_backtest = '--backtest' in args

    if not json_only:
        print("=" * 50)
        print(f"策略D: 龙头股动量 周度信号")
        print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)

    price_df = load_all_data(for_backtest=do_backtest)
    if price_df is None:
        print("[ERROR] 数据加载失败!", file=sys.stderr)
        sys.exit(1)

    if do_backtest:
        result = run_backtest(price_df)
        out = os.path.join(DATA_DIR, 'strategy_d_backtest_result.json')
        with open(out, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n已保存: {out}")
        return

    signal = generate_signal(price_df)
    out = os.path.join(DATA_DIR, 'strategy_d_latest_signal.json')
    with open(out, 'w') as f:
        json.dump(signal, f, indent=2, ensure_ascii=False)

    if json_only:
        print(json.dumps(signal, indent=2, ensure_ascii=False))
    else:
        print(format_signal_text(signal))
        print(f"\n已保存: {out}")

    return signal


if __name__ == '__main__':
    main()
