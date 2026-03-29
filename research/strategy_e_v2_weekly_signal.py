import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
#!/usr/bin/env python3
"""
Strategy E v2: A股增强价值投资策略 (基本面价值+逆向+质量)

在原版Strategy E基础上，加入PE/PB基本面价值因子(akshare数据)。

核心逻辑:
  1. 股票池: 28只各行业龙头A股 (同策略D/E)
  2. 基本面价值因子 (30%权重):
     - PE(TTM)百分位: 当前PE在近3年历史中的百分位 (越低=越便宜)
     - PB百分位: 当前PB在近3年历史中的百分位 (越低=越便宜)
     - PE权重60% + PB权重40%
  3. 逆向动量因子 (20%权重): 10周负动量 (近期跌多=便宜)
  4. 质量因子 (30%权重): 12周低波动率 (低波=高质量)
  5. 盈利质量因子 (20%权重): ROE-PB组合 (高ROE+低PB=被低估的优质股)
     - 用 PB/PE 作为ROE代理 (ROE = PB/PE)
  6. 安全过滤: 4周动量>-20% (排除暴跌/基本面恶化)
  7. 行业约束: 每行业最多1只 (sector_max=1)
  8. Top4等权, 月度调仓

⚠️ 重要偏差警告:
  - 股票池(28只龙头股)是手工选择的当前行业领军者，存在严重幸存者偏差
  - 这些股票能成为当前龙头本身就是后验结果，回测收益可能被高估
  - 与策略D/G不同(使用baostock历史成分股消除幸存者偏差)，本策略无法消除此偏差
  - 建议: 将回测结果打折30-50%来估计真实未来表现

数据来源:
  - 价格: Yahoo Finance (via stock_data_common)
  - PE/PB: akshare stock_zh_valuation_baidu (百度股市通)

用法:
  python3 strategy_e_v2_weekly_signal.py              # 正常运行
  python3 strategy_e_v2_weekly_signal.py --json       # 仅输出JSON
  python3 strategy_e_v2_weekly_signal.py --backtest   # 运行完整回测
  python3 strategy_e_v2_weekly_signal.py --fetch-only # 仅获取/缓存基本面数据
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import json
import os
import sys
import time
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from stock_data_common import STOCK_POOL, DATA_DIR, load_all_data

# ============================================================
# 策略参数
# ============================================================

PARAMS = {
    # Factor weights (must sum to 1.0)
    'fundamental_weight': 0.30,   # PE/PB基本面价值
    'reversal_weight': 0.20,      # 逆向动量(负momentum)
    'quality_weight': 0.30,       # 低波动率
    'earnings_weight': 0.20,      # 盈利质量(ROE代理)

    # Lookback periods
    'value_lookback': 10,         # 逆向动量回看周数
    'vol_lookback': 12,           # 波动率回看周数
    'pe_percentile_window': 750,  # PE百分位回看天数 (~3年)

    # Portfolio construction
    'top_n': 4,
    'rebal_freq': 4,              # 调仓频率(周)
    'max_drawdown_filter': -0.20, # 4周动量安全过滤
    'txn_cost_bps': 8,
    'sector_max': 1,

    # PE/PB composite
    'pe_weight_in_fundamental': 0.6,  # PE在基本面因子中的权重
    'pb_weight_in_fundamental': 0.4,  # PB在基本面因子中的权重
}

# ============================================================
# 基本面数据获取与缓存
# ============================================================

FUND_CACHE_DIR = os.path.join(DATA_DIR, 'fundamental_cache')
FUND_CACHE_FILE = os.path.join(FUND_CACHE_DIR, 'pe_pb_daily.pkl')
FUND_CACHE_MAX_AGE_DAYS = 3


def _stock_code(ticker: str) -> str:
    """Convert Yahoo ticker to A-share code: '600519.SS' -> '600519'"""
    return ticker.split('.')[0]


def fetch_fundamental_data(stock_pool: dict = None,
                           force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Fetch PE(TTM) and PB daily history for all stocks via akshare.

    Returns:
        dict[ticker] -> DataFrame with columns ['pe', 'pb'], DatetimeIndex
    """
    if stock_pool is None:
        stock_pool = STOCK_POOL

    os.makedirs(FUND_CACHE_DIR, exist_ok=True)

    # Check cache
    if not force_refresh and os.path.exists(FUND_CACHE_FILE):
        age = (time.time() - os.path.getmtime(FUND_CACHE_FILE)) / 86400
        if age <= FUND_CACHE_MAX_AGE_DAYS:
            with open(FUND_CACHE_FILE, 'rb') as f:
                cached = pickle.load(f)
            print(f"  基本面数据: 从缓存加载 ({len(cached)}只, {age:.1f}天前)")
            return cached

    try:
        import akshare as ak
    except ImportError:
        print("[WARN] akshare未安装, 无法获取基本面数据", file=sys.stderr)
        return {}

    print("  获取PE/PB基本面数据 (akshare)...")
    result = {}
    for ticker, info in stock_pool.items():
        code = _stock_code(ticker)
        name = info['name']
        try:
            df_pe = ak.stock_zh_valuation_baidu(
                symbol=code, indicator="市盈率(TTM)", period="近五年")
            df_pb = ak.stock_zh_valuation_baidu(
                symbol=code, indicator="市净率", period="近五年")

            # Merge PE and PB
            df_pe = df_pe.rename(columns={'value': 'pe'}).set_index('date')
            df_pb = df_pb.rename(columns={'value': 'pb'}).set_index('date')
            df_pe.index = pd.to_datetime(df_pe.index)
            df_pb.index = pd.to_datetime(df_pb.index)

            merged = df_pe.join(df_pb, how='outer').sort_index()
            # Remove invalid values
            merged = merged[(merged['pe'] > 0) & (merged['pb'] > 0)]

            if len(merged) > 50:
                result[ticker] = merged
                print(f"    {name}: {len(merged)}天 PE={merged['pe'].iloc[-1]:.1f} PB={merged['pb'].iloc[-1]:.2f}")
            else:
                print(f"    {name}: 数据不足({len(merged)}天), 跳过")

            time.sleep(1.5)  # Rate limit

        except Exception as e:
            print(f"    {name}: 获取失败 - {e}")
            time.sleep(1)

    # Save cache
    if result:
        with open(FUND_CACHE_FILE, 'wb') as f:
            pickle.dump(result, f)
        print(f"  基本面数据缓存已保存: {len(result)}只股票")

    return result


def get_pe_pb_percentile(fund_data: Dict[str, pd.DataFrame],
                         ticker: str,
                         as_of_date: pd.Timestamp,
                         window_days: int = 750) -> Optional[Dict[str, float]]:
    """
    Calculate PE and PB percentile for a stock as of a given date.

    Returns dict with 'pe_pctl', 'pb_pctl', 'pe', 'pb', 'roe_proxy'
    or None if data unavailable.
    """
    if ticker not in fund_data:
        return None

    df = fund_data[ticker]

    # Get data up to as_of_date (no look-ahead)
    mask = df.index <= as_of_date
    available = df[mask]
    if len(available) < 60:  # Need at least ~2 months of data
        return None

    # Current values (latest available)
    current_pe = available['pe'].iloc[-1]
    current_pb = available['pb'].iloc[-1]

    if pd.isna(current_pe) or pd.isna(current_pb) or current_pe <= 0 or current_pb <= 0:
        return None

    # Historical window for percentile
    start_date = as_of_date - timedelta(days=window_days)
    window_data = available[available.index >= start_date]

    if len(window_data) < 60:
        window_data = available  # Use all available if window too short

    # Calculate percentiles (lower = cheaper = better for value)
    pe_pctl = (window_data['pe'] < current_pe).sum() / len(window_data)
    pb_pctl = (window_data['pb'] < current_pb).sum() / len(window_data)

    # ROE proxy: ROE = Net Income / Equity = (Price/PE) / (Price/PB) = PB/PE
    roe_proxy = current_pb / current_pe if current_pe > 0 else 0

    return {
        'pe': current_pe,
        'pb': current_pb,
        'pe_pctl': pe_pctl,
        'pb_pctl': pb_pctl,
        'roe_proxy': roe_proxy,
    }


# ============================================================
# 策略核心
# ============================================================

def select_enhanced_value(price_df: pd.DataFrame,
                          fund_data: Dict[str, pd.DataFrame],
                          idx: int) -> Tuple[List[str], List[dict]]:
    """
    增强版价值+质量选股 (基本面PE/PB + 逆向动量 + 低波 + 盈利质量)

    使用 price_df.iloc[:idx+1] 数据, 严格无前瞻。
    """
    vlb = PARAMS['value_lookback']
    vol_lb = PARAMS['vol_lookback']
    top_n = PARAMS['top_n']
    ddf = PARAMS['max_drawdown_filter']

    w_fund = PARAMS['fundamental_weight']
    w_rev = PARAMS['reversal_weight']
    w_qual = PARAMS['quality_weight']
    w_earn = PARAMS['earnings_weight']

    if idx < max(vlb, vol_lb) + 2:
        return [], []

    returns = price_df.pct_change(fill_method=None)
    as_of_date = price_df.index[idx]

    # Calculate factors for available stocks
    scores = {}
    for col in price_df.columns:
        if (idx - vlb < 0 or
            pd.isna(price_df[col].iloc[idx]) or
            pd.isna(price_df[col].iloc[idx - vlb]) or
            price_df[col].iloc[idx - vlb] <= 0):
            continue

        # --- Reversal factor: negative momentum = high value ---
        mom_10w = float(price_df[col].iloc[idx] / price_df[col].iloc[idx - vlb] - 1)

        # --- Quality factor: low volatility ---
        ret_slice = returns[col].iloc[max(0, idx-vol_lb):idx+1].dropna()
        if len(ret_slice) < 4:
            continue
        vol = float(ret_slice.std())

        # --- Safety filter ---
        if idx >= 4:
            mom_4w = float(price_df[col].iloc[idx] / price_df[col].iloc[idx-4] - 1)
            if mom_4w < ddf:
                continue
        else:
            mom_4w = 0.0

        # --- Fundamental factor: PE/PB percentile ---
        fund_info = get_pe_pb_percentile(
            fund_data, col, as_of_date,
            window_days=PARAMS['pe_percentile_window']
        )

        pe_pctl = None
        pb_pctl = None
        roe_proxy = None
        pe_val = None
        pb_val = None

        if fund_info is not None:
            pe_pctl = fund_info['pe_pctl']
            pb_pctl = fund_info['pb_pctl']
            roe_proxy = fund_info['roe_proxy']
            pe_val = fund_info['pe']
            pb_val = fund_info['pb']

        scores[col] = {
            'reversal_score': -mom_10w,      # Higher = more reversal (cheaper)
            'quality_score': -vol,            # Higher = lower vol (better quality)
            'pe_pctl': pe_pctl,              # Lower = cheaper
            'pb_pctl': pb_pctl,              # Lower = cheaper
            'roe_proxy': roe_proxy,          # Higher = better earnings
            'pe': pe_val,
            'pb': pb_val,
            'mom_10w': mom_10w,
            'mom_4w': mom_4w,
            'vol_12w': vol,
            'has_fundamental': fund_info is not None,
        }

    if len(scores) < top_n:
        return [], []

    tickers = list(scores.keys())

    # --- Rank each factor ---
    # 1. Reversal rank (higher reversal_score = better = lower rank number)
    reversal_ranks = pd.Series(
        {t: scores[t]['reversal_score'] for t in tickers}
    ).rank(ascending=False)

    # 2. Quality rank (higher quality_score = better)
    quality_ranks = pd.Series(
        {t: scores[t]['quality_score'] for t in tickers}
    ).rank(ascending=False)

    # 3. Fundamental rank (lower PE/PB percentile = cheaper = better)
    has_fund = [t for t in tickers if scores[t]['has_fundamental']]
    no_fund = [t for t in tickers if not scores[t]['has_fundamental']]

    if has_fund:
        pe_w = PARAMS['pe_weight_in_fundamental']
        pb_w = PARAMS['pb_weight_in_fundamental']

        pe_pctl_series = pd.Series({t: scores[t]['pe_pctl'] for t in has_fund})
        pb_pctl_series = pd.Series({t: scores[t]['pb_pctl'] for t in has_fund})

        # Lower percentile = cheaper = better (ascending=True gives rank 1 to lowest)
        pe_ranks_sub = pe_pctl_series.rank(ascending=True)
        pb_ranks_sub = pb_pctl_series.rank(ascending=True)

        # Composite fundamental rank within has_fund
        fund_composite_sub = pe_w * pe_ranks_sub + pb_w * pb_ranks_sub
        fund_ranks_sub = fund_composite_sub.rank(ascending=True)

        # Assign median rank to stocks without data
        median_rank = (len(tickers) + 1) / 2
        fundamental_ranks = pd.Series(dtype=float)
        for t in tickers:
            if t in fund_ranks_sub.index:
                fundamental_ranks[t] = fund_ranks_sub[t] * len(tickers) / len(has_fund)
            else:
                fundamental_ranks[t] = median_rank
    else:
        fundamental_ranks = pd.Series({t: (len(tickers) + 1) / 2 for t in tickers})

    # 4. Earnings quality rank: ROE proxy (PB/PE), higher = better
    if has_fund:
        roe_series = pd.Series({t: scores[t]['roe_proxy'] for t in has_fund})
        roe_ranks_sub = roe_series.rank(ascending=False)

        median_rank = (len(tickers) + 1) / 2
        earnings_ranks = pd.Series(dtype=float)
        for t in tickers:
            if t in roe_ranks_sub.index:
                earnings_ranks[t] = roe_ranks_sub[t] * len(tickers) / len(has_fund)
            else:
                earnings_ranks[t] = median_rank
    else:
        earnings_ranks = pd.Series({t: (len(tickers) + 1) / 2 for t in tickers})

    # --- Composite score (weighted rank) ---
    composite = (w_fund * fundamental_ranks +
                 w_rev * reversal_ranks +
                 w_qual * quality_ranks +
                 w_earn * earnings_ranks)

    # --- Sector constraint ---
    sector_max = PARAMS.get('sector_max')
    if sector_max:
        sorted_tickers = list(composite.sort_values().index)
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
        selected = list(composite.nsmallest(top_n).index)

    # --- Build details ---
    details = []
    for rank_i, t in enumerate(composite.sort_values().index):
        info = STOCK_POOL.get(t, {'name': t, 'code': t, 'sector': '?'})
        s = scores[t]
        d = {
            'ticker': t,
            'name': info['name'],
            'code': info.get('code', t),
            'sector': info.get('sector', '?'),
            'mom_10w': round(s['mom_10w'] * 100, 1),
            'mom_4w': round(s['mom_4w'] * 100, 1),
            'vol_12w': round(s['vol_12w'] * 100, 2),
            'reversal_rank': int(reversal_ranks[t]),
            'quality_rank': int(quality_ranks[t]),
            'fundamental_rank': round(fundamental_ranks[t], 1),
            'earnings_rank': round(earnings_ranks[t], 1),
            'composite_rank': rank_i + 1,
            'selected': t in selected,
            'position_pct': round(100 / top_n, 1) if t in selected else 0.0,
        }
        if s['has_fundamental']:
            d['pe'] = round(s['pe'], 1)
            d['pb'] = round(s['pb'], 2)
            d['pe_pctl'] = round(s['pe_pctl'] * 100, 1)
            d['pb_pctl'] = round(s['pb_pctl'] * 100, 1)
            d['roe_proxy'] = round(s['roe_proxy'] * 100, 2)
        details.append(d)

    return selected, details


# ============================================================
# 信号生成
# ============================================================

def generate_signal(price_df: pd.DataFrame,
                    fund_data: Dict[str, pd.DataFrame]) -> dict:
    idx = len(price_df) - 1
    latest_date = price_df.index[idx]

    selected, details = select_enhanced_value(price_df, fund_data, idx)

    signal = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': 'Strategy E v2 (增强价值投资)',
        'variant': (f"Fund{PARAMS['fundamental_weight']:.0%} "
                    f"Rev{PARAMS['reversal_weight']:.0%} "
                    f"Qual{PARAMS['quality_weight']:.0%} "
                    f"Earn{PARAMS['earnings_weight']:.0%} "
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
            stock_info = {
                'code': d['code'],
                'name': d['name'],
                'sector': d['sector'],
                'mom_10w': d['mom_10w'],
                'vol_12w': d['vol_12w'],
            }
            if 'pe' in d:
                stock_info['pe'] = d['pe']
                stock_info['pb'] = d['pb']
                stock_info['pe_pctl'] = d['pe_pctl']
            signal['selected_stocks'].append(stock_info)

    # Summary text
    lines = [
        f"逻辑: 基本面价值(PE/PB百分位) + 逆向动量 + 低波质量 + 盈利质量",
        f"权重: 基本面{PARAMS['fundamental_weight']:.0%} + 逆向{PARAMS['reversal_weight']:.0%}"
        f" + 质量{PARAMS['quality_weight']:.0%} + 盈利{PARAMS['earnings_weight']:.0%}",
        f"",
        f"本期持仓 ({len(selected)}只, 等权{round(100/PARAMS['top_n'],1)}%每只):",
        f"",
    ]
    for d in details:
        if d['selected']:
            pe_str = f" PE:{d['pe']:.1f}({d['pe_pctl']:.0f}%ile)" if 'pe' in d else ""
            pb_str = f" PB:{d['pb']:.2f}({d['pb_pctl']:.0f}%ile)" if 'pb' in d else ""
            lines.append(
                f"  >> #{d['composite_rank']} {d['code']} {d['name']} [{d['sector']}]"
                f" 10w:{d['mom_10w']:+.1f}% Vol:{d['vol_12w']:.1f}%"
                f"{pe_str}{pb_str}"
            )

    lines.append("")
    lines.append("完整排名 (4因子综合):")
    for d in details:
        icon = '>>' if d['selected'] else '  '
        pe_str = f" PE:{d.get('pe','?')}" if 'pe' in d else ""
        lines.append(
            f"  {icon} #{d['composite_rank']} {d['code']} {d['name']} [{d['sector']}]"
            f" F-R:{d['fundamental_rank']:.0f} R-R:{d['reversal_rank']}"
            f" Q-R:{d['quality_rank']} E-R:{d['earnings_rank']:.0f}{pe_str}"
        )

    lines.append("")
    lines.append(f"注: 已过滤4周跌幅>{abs(PARAMS['max_drawdown_filter'])*100:.0f}%的股票")
    signal['action_summary'] = lines
    return signal


# ============================================================
# 回测
# ============================================================

def run_backtest(price_df: pd.DataFrame,
                 fund_data: Dict[str, pd.DataFrame]) -> dict:
    print("\n运行增强回测...")
    txn_cost = PARAMS['txn_cost_bps'] / 10000
    returns = price_df.pct_change(fill_method=None)
    warmup = max(PARAMS['value_lookback'], PARAMS['vol_lookback']) + 5

    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        selected, _ = select_enhanced_value(price_df, fund_data, i)
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
    sharpe = (wr.mean() * 52 - 0.025) / (wr.std() * np.sqrt(52)) if wr.std() > 0 else 0  # rf=2.5%
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    win_rate = (wr > 0).sum() / len(wr) * 100

    annual = nav_s.resample('YE').last().pct_change().dropna()
    annual_returns = {str(d.year): round(v * 100, 1) for d, v in annual.items()}

    result = {
        'strategy': f"Strategy E v2: 增强价值投资 Top{PARAMS['top_n']}",
        'factors': (f"基本面{PARAMS['fundamental_weight']:.0%} + "
                    f"逆向{PARAMS['reversal_weight']:.0%} + "
                    f"质量{PARAMS['quality_weight']:.0%} + "
                    f"盈利{PARAMS['earnings_weight']:.0%}"),
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
        'note': '无前瞻偏差: T周决策->T+1周收益, 交易成本8bp, PE/PB百分位使用截止日前数据',
    }

    print(f"\n回测结果:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


def run_comparison_backtest(price_df: pd.DataFrame,
                            fund_data: Dict[str, pd.DataFrame]) -> dict:
    """Run both v1 and v2 backtests for comparison."""
    print("\n===== V2 增强版回测 =====")
    v2_result = run_backtest(price_df, fund_data)

    print("\n===== V1 原版回测 (对照) =====")
    # Temporarily switch to v1 weights
    saved = {
        'fundamental_weight': PARAMS['fundamental_weight'],
        'reversal_weight': PARAMS['reversal_weight'],
        'quality_weight': PARAMS['quality_weight'],
        'earnings_weight': PARAMS['earnings_weight'],
    }
    # V1 equivalent: 80% reversal + 20% quality, 0% fundamental, 0% earnings
    PARAMS['fundamental_weight'] = 0.0
    PARAMS['reversal_weight'] = 0.80
    PARAMS['quality_weight'] = 0.20
    PARAMS['earnings_weight'] = 0.0

    v1_result = run_backtest(price_df, fund_data)

    # Restore
    PARAMS.update(saved)

    comparison = {
        'v1_original': v1_result,
        'v2_enhanced': v2_result,
        'improvement': {
            'cagr_delta': round(v2_result.get('cagr_pct', 0) - v1_result.get('cagr_pct', 0), 1),
            'sharpe_delta': round(v2_result.get('sharpe', 0) - v1_result.get('sharpe', 0), 3),
            'mdd_delta': round(v2_result.get('mdd_pct', 0) - v1_result.get('mdd_pct', 0), 1),
            'calmar_delta': round(v2_result.get('calmar', 0) - v1_result.get('calmar', 0), 3),
        }
    }

    print("\n" + "=" * 60)
    print("对比总结:")
    print(f"  V1 原版:  CAGR={v1_result.get('cagr_pct')}% MDD={v1_result.get('mdd_pct')}% "
          f"Sharpe={v1_result.get('sharpe')} Calmar={v1_result.get('calmar')}")
    print(f"  V2 增强:  CAGR={v2_result.get('cagr_pct')}% MDD={v2_result.get('mdd_pct')}% "
          f"Sharpe={v2_result.get('sharpe')} Calmar={v2_result.get('calmar')}")
    imp = comparison['improvement']
    print(f"  改进:     CAGR{imp['cagr_delta']:+.1f}% MDD{imp['mdd_delta']:+.1f}% "
          f"Sharpe{imp['sharpe_delta']:+.3f} Calmar{imp['calmar_delta']:+.3f}")
    print("=" * 60)

    return comparison


# ============================================================
# 输出
# ============================================================

def format_signal_text(signal: dict) -> str:
    lines = [
        f"{'='*55}",
        f"策略E v2 增强价值投资 信号",
        f"日期: {signal['date']}",
        f"调仓频率: {signal['rebalance_freq']}",
        f"{'='*55}",
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
    fetch_only = '--fetch-only' in args

    if not json_only:
        print("=" * 55)
        print(f"策略E v2: 增强价值投资(基本面+逆向+质量+盈利)")
        print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 55)

    # Fetch fundamental data
    fund_data = fetch_fundamental_data()
    if fetch_only:
        print(f"\n基本面数据获取完成: {len(fund_data)}只股票")
        return

    # Load price data
    price_df = load_all_data(for_backtest=do_backtest)
    if price_df is None:
        print("[ERROR] 数据加载失败!", file=sys.stderr)
        sys.exit(1)

    if do_backtest:
        comparison = run_comparison_backtest(price_df, fund_data)
        out = os.path.join(DATA_DIR, 'strategy_e_v2_backtest_result.json')
        with open(out, 'w') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\n已保存: {out}")
        return

    signal = generate_signal(price_df, fund_data)
    out = os.path.join(DATA_DIR, 'strategy_e_v2_latest_signal.json')
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
