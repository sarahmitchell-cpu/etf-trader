#!/usr/bin/env python3
"""
Strategy E Research: A股价值投资策略 (逆向+质量多因子)

核心思路: 龙头股中寻找"被低估的优质股"
- 价值因子: 20周动量反转 (近期跌多的=便宜, 反向排名)
- 质量因子: 低波动率 (12周波动率, 低=稳健)
- 安全网: 4周动量>-10% (过滤暴跌股/基本面恶化)
- 选Top5-8, 等权, 月度调仓

学术基础: DeBondt & Thaler (1985) 长期反转效应
         + Ang et al. (2006) 低波动异象
"""

import pandas as pd
import numpy as np
import os
import json
import sys
from datetime import datetime
from itertools import product

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Use the same stock pool as Strategy D (already cached)
STOCK_POOL = {
    '600519.SS': {'name': '贵州茅台', 'sector': '白酒'},
    '000858.SZ': {'name': '五粮液', 'sector': '白酒'},
    '600887.SS': {'name': '伊利股份', 'sector': '食品'},
    '002714.SZ': {'name': '牧原股份', 'sector': '养殖'},
    '601318.SS': {'name': '中国平安', 'sector': '保险'},
    '600036.SS': {'name': '招商银行', 'sector': '银行'},
    '002415.SZ': {'name': '海康威视', 'sector': '安防'},
    '300750.SZ': {'name': '宁德时代', 'sector': '电池'},
    '600276.SS': {'name': '恒瑞医药', 'sector': '创新药'},
    '300760.SZ': {'name': '迈瑞医疗', 'sector': '医疗器械'},
    '601012.SS': {'name': '隆基绿能', 'sector': '光伏'},
    '300274.SZ': {'name': '阳光电源', 'sector': '逆变器'},
    '000333.SZ': {'name': '美的集团', 'sector': '家电'},
    '600690.SS': {'name': '海尔智家', 'sector': '家电'},
    '002594.SZ': {'name': '比亚迪', 'sector': '新能源车'},
    '600893.SS': {'name': '航发动力', 'sector': '航发'},
    '601668.SS': {'name': '中国建筑', 'sector': '建筑'},
    '600585.SS': {'name': '海螺水泥', 'sector': '水泥'},
    '601899.SS': {'name': '紫金矿业', 'sector': '有色'},
    '600028.SS': {'name': '中国石化', 'sector': '石油'},
    '002371.SZ': {'name': '北方华创', 'sector': '半导体设备'},
    '000063.SZ': {'name': '中兴通讯', 'sector': '通信设备'},
    '600941.SS': {'name': '中国移动', 'sector': '运营商'},
    '002230.SZ': {'name': '科大讯飞', 'sector': 'AI'},
    '600900.SS': {'name': '长江电力', 'sector': '水电'},
    '601088.SS': {'name': '中国神华', 'sector': '煤炭'},
    '601006.SS': {'name': '大秦铁路', 'sector': '铁路'},
    '001979.SZ': {'name': '招商蛇口', 'sector': '地产'},
}


def load_cached_prices():
    """Load cached weekly prices from Strategy D research"""
    csv_path = os.path.join(DATA_DIR, 'sd_all_weekly_prices.csv')
    if not os.path.exists(csv_path):
        print("ERROR: Run strategy_d_research.py first to cache price data!")
        sys.exit(1)
    df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
    print(f"Loaded: {df.shape[0]}周 x {df.shape[1]}股")
    return df


def backtest_value_quality(price_df, value_lookback=20, vol_lookback=12,
                           top_n=5, rebal_freq=4, max_drawdown_filter=-0.10,
                           vol_weight=0.5, txn_cost_bps=8):
    """
    逆向价值+质量 回测

    严格无前瞻偏差:
    - 在第 i 周末用 [:i+1] 数据做决策
    - 持仓承受 i+1 到 i+rebal_freq 的收益

    因子:
    - value_score = -momentum_Nw (跌得多=高value分)
    - quality_score = -volatility_12w (波动低=高quality分)
    - composite = (1-vol_weight) * value_rank + vol_weight * quality_rank
    - 过滤: 4周回撤 > max_drawdown_filter
    """
    txn_cost = txn_cost_bps / 10000
    returns = price_df.pct_change(fill_method=None)

    warmup = max(value_lookback, vol_lookback) + 5
    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        # Available stocks
        avail = []
        for col in price_df.columns:
            if (i - value_lookback >= 0 and
                not pd.isna(price_df[col].iloc[i]) and
                not pd.isna(price_df[col].iloc[i - value_lookback]) and
                price_df[col].iloc[i - value_lookback] > 0):
                avail.append(col)

        if len(avail) < top_n + 3:
            i += 1
            continue

        # Calculate factors
        scores = {}
        for col in avail:
            # Value factor: negative momentum = high value
            mom = price_df[col].iloc[i] / price_df[col].iloc[i - value_lookback] - 1
            value_score = -mom  # Losers get high score

            # Quality factor: low volatility = high quality
            ret_slice = returns[col].iloc[max(0, i-vol_lookback):i+1].dropna()
            if len(ret_slice) < 4:
                continue
            vol = ret_slice.std()
            quality_score = -vol  # Low vol = high score

            # Safety filter: 4-week momentum not too negative
            if i >= 4:
                mom_4w = price_df[col].iloc[i] / price_df[col].iloc[i-4] - 1
                if mom_4w < max_drawdown_filter:
                    continue  # Skip falling knives

            scores[col] = {
                'value': value_score,
                'quality': quality_score,
                'mom_20w': mom,
                'vol_12w': vol,
            }

        if len(scores) < top_n:
            i += 1
            continue

        # Rank and composite score
        tickers = list(scores.keys())
        value_ranks = pd.Series({t: scores[t]['value'] for t in tickers}).rank(ascending=False)
        quality_ranks = pd.Series({t: scores[t]['quality'] for t in tickers}).rank(ascending=False)
        composite = (1 - vol_weight) * value_ranks + vol_weight * quality_ranks

        # Select top N (lowest composite rank = best)
        selected = list(composite.nsmallest(top_n).index)
        selected_set = set(selected)

        # Transaction cost
        new_buys = selected_set - prev_holdings
        sold = prev_holdings - selected_set
        turnover = (len(new_buys) + len(sold)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        # Hold period
        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            rets = []
            for s in selected:
                r = returns[s].iloc[j]
                if not pd.isna(r):
                    rets.append(float(r))
            port_ret = np.mean(rets) if rets else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    if not dates:
        return {'error': 'No trades'}

    nav_s = pd.Series(nav[1:], index=dates)
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / years) - 1
    dd = nav_s / nav_s.cummax() - 1
    mdd = dd.min()
    wr = pd.Series(weekly_rets)
    sharpe = wr.mean() / wr.std() * np.sqrt(52) if wr.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    win_rate = (wr > 0).sum() / len(wr) * 100

    annual = nav_s.resample('YE').last().pct_change().dropna()
    annual_returns = {str(d.year): round(v * 100, 1) for d, v in annual.items()}

    return {
        'value_lb': value_lookback,
        'vol_lb': vol_lookback,
        'vol_weight': vol_weight,
        'top_n': top_n,
        'rebal_freq': rebal_freq,
        'dd_filter': max_drawdown_filter,
        'years': round(years, 1),
        'cagr_pct': round(cagr * 100, 1),
        'total_return_pct': round((nav_s.iloc[-1] - 1) * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round(win_rate, 1),
        'annual_returns': annual_returns,
        'total_txn_pct': round(total_txn * 100, 2),
    }


def main():
    price_df = load_cached_prices()

    print("\n" + "="*60)
    print("Strategy E: 逆向价值+质量 参数搜索")
    print("="*60)

    results = []
    for vlb, vw, tn, rf, ddf in product(
        [12, 20, 26],          # value lookback
        [0.0, 0.3, 0.5, 0.7], # vol weight
        [5, 8],                # top n
        [4],                   # rebal freq (monthly)
        [-0.15, -0.10, -0.05], # drawdown filter
    ):
        r = backtest_value_quality(price_df, value_lookback=vlb, vol_weight=vw,
                                   top_n=tn, rebal_freq=rf, max_drawdown_filter=ddf)
        if 'error' not in r:
            results.append(r)

    results.sort(key=lambda x: -x['sharpe'])

    print(f"\n共 {len(results)} 组参数")
    print(f"\nTop 15 by Sharpe:")
    print(f"{'VLB':>3} {'VW':>4} {'TopN':>4} {'DDF':>5} | {'CAGR':>6} {'MDD':>6} {'Sharpe':>6} {'Calmar':>6} {'Win%':>5} | Annual Returns")
    print("-" * 110)
    for r in results[:15]:
        ann = ' '.join(f"{y}:{v:+.0f}%" for y, v in sorted(r['annual_returns'].items()))
        print(f"{r['value_lb']:>3} {r['vol_weight']:>4.1f} {r['top_n']:>4} {r['dd_filter']:>5.2f} | "
              f"{r['cagr_pct']:>5.1f}% {r['mdd_pct']:>5.1f}% {r['sharpe']:>6.3f} {r['calmar']:>6.3f} {r['win_rate_pct']:>4.1f}% | {ann}")

    # Also test pure contrarian (no vol filter)
    print(f"\n\n--- Pure Contrarian (vol_weight=0) ---")
    pure = [r for r in results if r['vol_weight'] == 0.0]
    for r in pure[:5]:
        ann = ' '.join(f"{y}:{v:+.0f}%" for y, v in sorted(r['annual_returns'].items()))
        print(f"VLB={r['value_lb']:>2} Top{r['top_n']} DDF={r['dd_filter']:>5.2f} | "
              f"CAGR={r['cagr_pct']:>5.1f}% MDD={r['mdd_pct']:>5.1f}% Sharpe={r['sharpe']:.3f} | {ann}")

    # Also test pure quality (vol_weight=1 would be interesting but not in grid, use 0.7)
    print(f"\n--- Quality-Heavy (vol_weight=0.7) ---")
    qual = [r for r in results if r['vol_weight'] == 0.7]
    for r in qual[:5]:
        ann = ' '.join(f"{y}:{v:+.0f}%" for y, v in sorted(r['annual_returns'].items()))
        print(f"VLB={r['value_lb']:>2} Top{r['top_n']} DDF={r['dd_filter']:>5.2f} | "
              f"CAGR={r['cagr_pct']:>5.1f}% MDD={r['mdd_pct']:>5.1f}% Sharpe={r['sharpe']:.3f} | {ann}")

    # Save
    output = {'all_results': results, 'best': results[0] if results else None}
    with open(os.path.join(DATA_DIR, 'se_research_results.json'), 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存")


if __name__ == '__main__':
    main()
