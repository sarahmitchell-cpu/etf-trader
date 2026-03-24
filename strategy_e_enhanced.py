#!/usr/bin/env python3
"""
Strategy E Enhanced: A股多因子价值投资策略

在原策略E基础上，将单一"动量反转"价值因子升级为多维度价值评估体系。

核心改进:
  原版: 价值=10周涨跌幅反转 (跌多=便宜)
  增强: 价值=多因子综合评分
    - 中期反转 (10w return reversal, 权重35%)
    - 相对位置 (距52周低点的位置, 权重25%)
    - 均值回归 (20w return z-score, 权重20%)
    - 长短期背离 (4w vs 20w momentum divergence, 权重20%)

质量因子:
  - 低波动 (12w volatility, 权重60%)
  - 收益一致性 (正收益周占比, 权重40%)

安全过滤:
  - 4周跌幅 > -20% (同原版)
  - 额外: 排除波动率极端值 (top 10%)

参数:
  value_weight=0.75, quality_weight=0.25 (原版: 0.80/0.20)
  top_n=4, rebal_freq=4 (月度), sector_max=1

数据来源: Yahoo Finance API (via stock_data_common)
交易成本: 8bps单边

用法:
  python3 strategy_e_enhanced.py              # 正常运行
  python3 strategy_e_enhanced.py --json       # 仅输出JSON
  python3 strategy_e_enhanced.py --backtest   # 运行完整回测
  python3 strategy_e_enhanced.py --compare    # 对比原版 vs 增强版
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple, Dict

from stock_data_common import STOCK_POOL, DATA_DIR, load_all_data

# ============================================================
# 策略参数
# ============================================================

PARAMS = {
    # 价值因子参数
    'reversal_lookback': 10,       # 中期反转回看周数
    'long_lookback': 20,           # 长期回看周数 (z-score, 长短背离)
    'high_low_lookback': 52,       # 52周高低点回看

    # 价值子因子权重
    'w_reversal': 0.35,            # 中期反转权重
    'w_relative_pos': 0.25,        # 相对位置权重 (距52w低点)
    'w_zscore': 0.20,              # 均值回归z-score权重
    'w_divergence': 0.20,          # 长短期背离权重

    # 质量因子参数
    'vol_lookback': 12,            # 波动率回看周数
    'consistency_lookback': 12,    # 收益一致性回看周数

    # 质量子因子权重
    'w_low_vol': 0.60,             # 低波动权重
    'w_consistency': 0.40,         # 收益一致性权重

    # 综合权重
    'value_weight': 0.75,          # 价值因子总权重
    'quality_weight': 0.25,        # 质量因子总权重

    # 选股参数
    'top_n': 4,
    'rebal_freq': 4,               # 调仓频率 (周)
    'max_drawdown_filter': -0.20,  # 4周动量安全过滤
    'vol_percentile_filter': 0.90, # 排除波动率top 10%
    'txn_cost_bps': 8,
    'sector_max': 1,
}


# ============================================================
# 价值因子计算
# ============================================================

def compute_value_factors(price_df: pd.DataFrame, idx: int, col: str) -> Dict[str, float]:
    """
    计算多维度价值因子 (严格无前瞻)

    返回 dict: {factor_name: raw_score} (越高=越便宜/越有价值)
    返回 None 如果数据不足
    """
    rl = PARAMS['reversal_lookback']
    ll = PARAMS['long_lookback']
    hl = PARAMS['high_low_lookback']

    prices = price_df[col].iloc[:idx+1]
    if len(prices) < max(rl, ll, hl) + 2:
        return None
    if pd.isna(prices.iloc[idx]) or prices.iloc[idx] <= 0:
        return None

    current = float(prices.iloc[idx])
    returns = prices.pct_change(fill_method=None)

    # 1. 中期反转: -10w收益率 (跌多=高分)
    if idx - rl < 0 or pd.isna(prices.iloc[idx - rl]) or prices.iloc[idx - rl] <= 0:
        return None
    mom_10w = current / float(prices.iloc[idx - rl]) - 1
    reversal_score = -mom_10w

    # 2. 相对位置: 当前价格在52周高低点中的位置 (越接近低点=越便宜=高分)
    hl_start = max(0, idx - hl)
    window = prices.iloc[hl_start:idx+1].dropna()
    if len(window) < 10:
        return None
    high_52w = float(window.max())
    low_52w = float(window.min())
    if high_52w <= low_52w:
        relative_pos_score = 0.0
    else:
        # 0=在最高点, 1=在最低点
        relative_pos_score = (high_52w - current) / (high_52w - low_52w)

    # 3. 均值回归 z-score: 近20周收益的z-score (越负=跌越多=便宜)
    ll_start = max(0, idx - ll)
    ret_window = returns.iloc[ll_start:idx+1].dropna()
    if len(ret_window) < 8:
        return None
    ret_mean = float(ret_window.mean())
    ret_std = float(ret_window.std())
    if ret_std > 0:
        # 用最近一周收益的z-score (负z-score=低于均值=便宜)
        recent_ret = float(returns.iloc[idx]) if not pd.isna(returns.iloc[idx]) else 0.0
        zscore = (recent_ret - ret_mean) / ret_std
        zscore_score = -zscore  # 越负=越高分
    else:
        zscore_score = 0.0

    # 4. 长短期背离: 短期(4w)弱 但 长期(20w)不差 = 短期超卖
    if idx >= 4 and not pd.isna(prices.iloc[idx-4]) and prices.iloc[idx-4] > 0:
        mom_4w = current / float(prices.iloc[idx-4]) - 1
    else:
        mom_4w = 0.0
    if idx >= ll and not pd.isna(prices.iloc[idx-ll]) and prices.iloc[idx-ll] > 0:
        mom_20w = current / float(prices.iloc[idx-ll]) - 1
    else:
        mom_20w = 0.0
    # 背离 = 长期还行但短期跌了 (20w正 - 4w负 = 大正数 = 好)
    divergence_score = mom_20w - mom_4w

    return {
        'reversal': reversal_score,
        'relative_pos': relative_pos_score,
        'zscore': zscore_score,
        'divergence': divergence_score,
        'mom_10w': mom_10w,
        'mom_4w': mom_4w,
        'mom_20w': mom_20w,
    }


def compute_quality_factors(price_df: pd.DataFrame, idx: int, col: str) -> Dict[str, float]:
    """
    计算质量因子 (严格无前瞻)

    返回 dict: {factor_name: raw_score} (越高=质量越好)
    返回 None 如果数据不足
    """
    vl = PARAMS['vol_lookback']
    cl = PARAMS['consistency_lookback']

    returns = price_df[col].pct_change(fill_method=None)
    ret_slice = returns.iloc[max(0, idx-vl):idx+1].dropna()
    if len(ret_slice) < 4:
        return None

    # 1. 低波动: -volatility (越低=越好=越高分)
    vol = float(ret_slice.std())
    low_vol_score = -vol

    # 2. 收益一致性: 正收益周占比 (越高=越稳定=越好)
    cons_slice = returns.iloc[max(0, idx-cl):idx+1].dropna()
    if len(cons_slice) < 4:
        consistency_score = 0.5
    else:
        consistency_score = float((cons_slice > 0).sum() / len(cons_slice))

    return {
        'low_vol': low_vol_score,
        'consistency': consistency_score,
        'vol_12w': vol,
    }


# ============================================================
# 选股核心
# ============================================================

def select_value_quality(price_df: pd.DataFrame, idx: int) -> Tuple[List[str], List[dict]]:
    """
    多因子价值+质量选股 (严格无前瞻)
    """
    top_n = PARAMS['top_n']
    ddf = PARAMS['max_drawdown_filter']

    warmup = max(PARAMS['reversal_lookback'], PARAMS['long_lookback'],
                 PARAMS['high_low_lookback'], PARAMS['vol_lookback']) + 5
    if idx < warmup:
        return [], []

    # Step 1: 计算所有股票的原始因子
    stock_factors = {}
    for col in price_df.columns:
        vf = compute_value_factors(price_df, idx, col)
        if vf is None:
            continue
        qf = compute_quality_factors(price_df, idx, col)
        if qf is None:
            continue

        # Safety filter: 4周跌幅
        if vf['mom_4w'] < ddf:
            continue

        stock_factors[col] = {**vf, **qf}

    if len(stock_factors) < top_n:
        return [], []

    tickers = list(stock_factors.keys())

    # Step 2: 排除波动率极端值 (top 10%)
    vol_pct = PARAMS['vol_percentile_filter']
    vols = pd.Series({t: stock_factors[t]['vol_12w'] for t in tickers})
    vol_threshold = vols.quantile(vol_pct)
    tickers = [t for t in tickers if stock_factors[t]['vol_12w'] <= vol_threshold]
    if len(tickers) < top_n:
        return [], []

    # Step 3: 排名 - 价值子因子
    reversal_ranks = pd.Series({t: stock_factors[t]['reversal'] for t in tickers}).rank(ascending=False)
    relpos_ranks = pd.Series({t: stock_factors[t]['relative_pos'] for t in tickers}).rank(ascending=False)
    zscore_ranks = pd.Series({t: stock_factors[t]['zscore'] for t in tickers}).rank(ascending=False)
    divergence_ranks = pd.Series({t: stock_factors[t]['divergence'] for t in tickers}).rank(ascending=False)

    value_composite = (
        PARAMS['w_reversal'] * reversal_ranks +
        PARAMS['w_relative_pos'] * relpos_ranks +
        PARAMS['w_zscore'] * zscore_ranks +
        PARAMS['w_divergence'] * divergence_ranks
    )

    # Step 4: 排名 - 质量子因子
    lowvol_ranks = pd.Series({t: stock_factors[t]['low_vol'] for t in tickers}).rank(ascending=False)
    consistency_ranks = pd.Series({t: stock_factors[t]['consistency'] for t in tickers}).rank(ascending=False)

    quality_composite = (
        PARAMS['w_low_vol'] * lowvol_ranks +
        PARAMS['w_consistency'] * consistency_ranks
    )

    # Step 5: 综合排名
    final_composite = (
        PARAMS['value_weight'] * value_composite +
        PARAMS['quality_weight'] * quality_composite
    )

    # Step 6: 行业约束选股
    sector_max = PARAMS.get('sector_max')
    if sector_max:
        sorted_tickers = list(final_composite.sort_values().index)
        selected = []
        sector_count = defaultdict(int)
        for t in sorted_tickers:
            if len(selected) >= top_n:
                break
            sec = STOCK_POOL.get(t, {}).get('sector', '?')
            if sector_count[sec] < sector_max:
                selected.append(t)
                sector_count[sec] += 1
    else:
        selected = list(final_composite.nsmallest(top_n).index)

    # Build details
    details = []
    for rank_i, t in enumerate(final_composite.sort_values().index):
        info = STOCK_POOL.get(t, {'name': t, 'code': t, 'sector': '?'})
        sf = stock_factors[t]
        details.append({
            'ticker': t,
            'name': info['name'],
            'code': info.get('code', t),
            'sector': info.get('sector', '?'),
            'mom_10w': round(sf['mom_10w'] * 100, 1),
            'mom_4w': round(sf['mom_4w'] * 100, 1),
            'mom_20w': round(sf.get('mom_20w', 0) * 100, 1),
            'relative_pos': round(sf['relative_pos'] * 100, 1),
            'vol_12w': round(sf['vol_12w'] * 100, 2),
            'consistency': round(sf['consistency'] * 100, 1),
            'value_rank': int(value_composite.get(t, 0)),
            'quality_rank': int(quality_composite.get(t, 0)),
            'composite_rank': rank_i + 1,
            'selected': t in selected,
            'position_pct': round(100 / top_n, 1) if t in selected else 0.0,
        })

    return selected, details


# ============================================================
# 信号生成
# ============================================================

def generate_signal(price_df: pd.DataFrame) -> dict:
    idx = len(price_df) - 1
    latest_date = price_df.index[idx]

    selected, details = select_value_quality(price_df, idx)

    signal = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': 'Strategy E Enhanced (多因子价值+质量)',
        'variant': (f"Rev{PARAMS['reversal_lookback']}w "
                    f"VW{PARAMS['value_weight']}/{PARAMS['quality_weight']} "
                    f"Top{PARAMS['top_n']}"),
        'rebalance_freq': f"每{PARAMS['rebal_freq']}周(月度)",
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
                'mom_10w': d['mom_10w'],
                'mom_4w': d['mom_4w'],
                'relative_pos': d['relative_pos'],
                'vol_12w': d['vol_12w'],
            })

    lines = [
        f"逻辑: 多因子价值选股 (反转+相对位置+均值回归+长短背离)",
        f"",
        f"本期持仓 ({len(selected)}只, 等权{round(100/PARAMS['top_n'],1)}%每只):",
        f"",
    ]
    for d in details:
        if d['selected']:
            lines.append(f"  >> #{d['composite_rank']} {d['code']} {d['name']} [{d['sector']}] "
                        f"10w:{d['mom_10w']:+.1f}% 4w:{d['mom_4w']:+.1f}% "
                        f"位置:{d['relative_pos']:.0f}% Vol:{d['vol_12w']:.1f}%")

    lines.append("")
    lines.append("完整排名 (多因子综合):")
    for d in details:
        icon = '>>' if d['selected'] else '  '
        lines.append(f"  {icon} #{d['composite_rank']} {d['code']} {d['name']} [{d['sector']}] "
                    f"V-R:{d['value_rank']} Q-R:{d['quality_rank']} "
                    f"10w:{d['mom_10w']:+.1f}% Pos:{d['relative_pos']:.0f}%")

    lines.append("")
    lines.append(f"注: 已过滤4周跌>{abs(PARAMS['max_drawdown_filter'])*100:.0f}% + 波动率Top10%")

    signal['action_summary'] = lines
    return signal


# ============================================================
# 回测
# ============================================================

def run_backtest(price_df: pd.DataFrame, label: str = "Enhanced") -> dict:
    print(f"\n运行回测 ({label})...")
    txn_cost = PARAMS['txn_cost_bps'] / 10000
    returns = price_df.pct_change(fill_method=None)
    warmup = max(PARAMS['reversal_lookback'], PARAMS['long_lookback'],
                 PARAMS['high_low_lookback'], PARAMS['vol_lookback']) + 10

    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        selected, _ = select_value_quality(price_df, i)
        if not selected:
            i += 1
            continue

        selected_set = set(selected)
        new_buys = selected_set - prev_holdings
        sold = prev_holdings - selected_set
        turnover = (len(new_buys) + len(sold)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost
        total_txn += period_txn

        hold_end = min(i + PARAMS['rebal_freq'], len(price_df) - 1)
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

    result = {
        'strategy': f"Strategy E {label}: Top{PARAMS['top_n']}",
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
        'note': '无前瞻偏差: T周决策->T+1周收益, 交易成本8bp',
    }

    print(f"\n回测结果 ({label}):")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


def run_comparison(price_df: pd.DataFrame):
    """对比原版 vs 增强版"""
    print("=" * 60)
    print("策略E 原版 vs 增强版 对比")
    print("=" * 60)

    # 增强版
    enhanced_result = run_backtest(price_df, "Enhanced (多因子)")

    # 原版参数 (临时切换)
    saved_params = dict(PARAMS)
    PARAMS['w_reversal'] = 1.0
    PARAMS['w_relative_pos'] = 0.0
    PARAMS['w_zscore'] = 0.0
    PARAMS['w_divergence'] = 0.0
    PARAMS['w_low_vol'] = 1.0
    PARAMS['w_consistency'] = 0.0
    PARAMS['value_weight'] = 0.80
    PARAMS['quality_weight'] = 0.20
    PARAMS['vol_percentile_filter'] = 1.0  # 不过滤

    original_result = run_backtest(price_df, "Original (单因子反转)")

    # 恢复参数
    PARAMS.update(saved_params)

    print("\n" + "=" * 60)
    print("对比总结:")
    print("=" * 60)
    metrics = ['cagr_pct', 'mdd_pct', 'sharpe', 'calmar', 'win_rate_pct']
    labels = ['CAGR%', 'MDD%', 'Sharpe', 'Calmar', 'Win Rate%']
    print(f"{'指标':<12} {'原版':<12} {'增强版':<12} {'差异':<12}")
    print("-" * 48)
    for m, l in zip(metrics, labels):
        ov = original_result.get(m, 0)
        ev = enhanced_result.get(m, 0)
        diff = ev - ov
        print(f"{l:<12} {ov:<12} {ev:<12} {diff:+.1f}")

    print(f"\n年度收益对比:")
    orig_ann = original_result.get('annual_returns', {})
    enh_ann = enhanced_result.get('annual_returns', {})
    all_years = sorted(set(list(orig_ann.keys()) + list(enh_ann.keys())))
    for y in all_years:
        ov = orig_ann.get(y, 'N/A')
        ev = enh_ann.get(y, 'N/A')
        print(f"  {y}: 原版 {ov}% | 增强 {ev}%")

    return original_result, enhanced_result


# ============================================================
# 输出
# ============================================================

def format_signal_text(signal: dict) -> str:
    lines = [
        f"{'='*50}",
        f"策略E增强版 多因子价值投资 信号",
        f"日期: {signal['date']}",
        f"调仓频率: {signal['rebalance_freq']}",
        f"{'='*50}",
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


def main():
    args = sys.argv[1:]
    json_only = '--json' in args
    do_backtest = '--backtest' in args
    do_compare = '--compare' in args

    if not json_only:
        print("=" * 50)
        print(f"策略E增强版: 多因子价值+质量 信号")
        print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)

    price_df = load_all_data(for_backtest=(do_backtest or do_compare))
    if price_df is None:
        print("[ERROR] 数据加载失败!", file=sys.stderr)
        sys.exit(1)

    if do_compare:
        orig, enh = run_comparison(price_df)
        out = os.path.join(DATA_DIR, 'strategy_e_comparison.json')
        with open(out, 'w') as f:
            json.dump({'original': orig, 'enhanced': enh}, f, indent=2, ensure_ascii=False)
        print(f"\n已保存: {out}")
        return

    if do_backtest:
        result = run_backtest(price_df)
        out = os.path.join(DATA_DIR, 'strategy_e_enhanced_backtest.json')
        with open(out, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n已保存: {out}")
        return

    signal = generate_signal(price_df)
    out = os.path.join(DATA_DIR, 'strategy_e_enhanced_signal.json')
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
