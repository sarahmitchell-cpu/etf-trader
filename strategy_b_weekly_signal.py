#!/usr/bin/env python3
"""
策略B: 增强版A股因子ETF周度信号 (Strategy B: Enhanced Factor ETF Weekly Signal)

基于深度研究的A股因子ETF择时策略，每周五收盘后运行，输出下周调仓建议。

核心逻辑:
  1. 因子池: 现金流(60%基准) + 国信价值(25%) + 红利低波(15%)
  2. 动态权重: 8周因子动量排名 → 最强50% / 中间30% / 最弱20%
  3. 双时间框架Regime (中证全指):
     - 强牛: 价格 > 13周MA 且 > 40周MA → 满仓
     - 过渡: 价格在两条MA之间 → 50%仓位
     - 熊市: 价格 < 13周MA 且 < 40周MA → 20%仓位
  4. 波动率缩放: 目标15%年化波动率, 12周回看

回测表现 (2014-2026, 信号: 中证全指):
  - CAGR: 23.2%, MDD: -7.1%, Calmar: 3.251

用法:
  python3 strategy_b_weekly_signal.py              # 正常运行
  python3 strategy_b_weekly_signal.py --json       # 仅输出JSON
  python3 strategy_b_weekly_signal.py --backtest   # 运行完整回测

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

# 数据目录 (用于缓存CSV)
DATA_DIR = os.environ.get('ETF_DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))

# 因子指数 (交易标的)
FACTOR_INDICES = {
    '932365': {'name': '现金流', 'etf': '159201', 'etf_name': '万家自由现金流ETF'},
    '931052': {'name': '国信价值', 'etf': '512040', 'etf_name': '国信价值ETF'},
    'H30269': {'name': '红利低波', 'etf': '515080', 'etf_name': '红利低波ETF'},
}

# 信号指数 (用于判断市场状态)
SIGNAL_INDEX = {'code': '000985', 'name': '中证全指'}

# 策略参数
PARAMS = {
    'ma_short': 13,         # 短期均线周数
    'ma_long': 40,          # 长期均线周数
    'vol_target': 0.15,     # 目标年化波动率
    'vol_lookback': 12,     # 波动率回看周数
    'momentum_lookback': 8, # 因子动量回看周数
    'momentum_weights': [0.50, 0.30, 0.20],  # 排名权重 (第1/2/3名)
    'regime_full': 1.0,     # 强牛仓位系数
    'regime_transition': 0.5,  # 过渡期仓位系数
    'regime_bear': 0.2,     # 熊市仓位系数
}


# ============================================================
# 数据获取
# ============================================================

def fetch_index(code: str, days: int = 600) -> pd.DataFrame | None:
    """
    获取指数日线数据。优先使用本地CSV缓存，缓存不存在则调用CSIndex API。

    Args:
        code: 指数代码 (如 '000985')
        days: 获取最近多少天的数据

    Returns:
        DataFrame with DatetimeIndex and 'close' column, or None on failure
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f'{code}_daily.csv')

    # 优先读本地缓存
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        if len(df) > 0:
            return df[['close']]

    # API获取
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.csindex.com.cn/',
        'Accept': 'application/json, text/plain, */*',
    }
    all_rows = []
    now = datetime.now()
    start_date = now - timedelta(days=days)

    for year in range(start_date.year, now.year + 1):
        s = f'{year}0101' if year > start_date.year else start_date.strftime('%Y%m%d')
        e = f'{year}1231' if year < now.year else now.strftime('%Y%m%d')
        url = (f'https://www.csindex.com.cn/csindex-home/perf/index-perf'
               f'?indexCode={code}&startDate={s}&endDate={e}')
        try:
            r = requests.get(url, headers=headers, timeout=30)
            resp = r.json()
            if resp.get('success') and resp.get('data'):
                data = resp['data']
                if isinstance(data, list):
                    all_rows.extend(data)
                elif isinstance(data, dict):
                    all_rows.append(data)
        except Exception as ex:
            print(f"  [WARN] API请求失败 {code}/{year}: {ex}", file=sys.stderr)
        time.sleep(0.5)

    if not all_rows:
        print(f"  [ERROR] 无法获取数据: {code}", file=sys.stderr)
        return None

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['tradeDate'])
    df['close'] = df['close'].astype(float)
    df = df[['date', 'close']].sort_values('date').drop_duplicates('date').set_index('date')
    df.to_csv(csv_path)
    return df


def load_all_data() -> tuple[dict[str, pd.Series], pd.Series] | None:
    """
    加载所有需要的指数数据并转换为周频。

    Returns:
        (factor_weekly, signal_weekly) 或 None
    """
    print("加载数据...")

    # 加载因子指数
    factor_daily = {}
    for code, info in FACTOR_INDICES.items():
        print(f"  {info['name']} ({code})...")
        df = fetch_index(code)
        if df is None:
            print(f"  [ERROR] 无法加载 {info['name']}", file=sys.stderr)
            return None
        factor_daily[info['name']] = df['close']
        time.sleep(0.3)

    # 加载信号指数
    print(f"  {SIGNAL_INDEX['name']} ({SIGNAL_INDEX['code']})...")
    sig_df = fetch_index(SIGNAL_INDEX['code'])
    if sig_df is None:
        print(f"  [ERROR] 无法加载 {SIGNAL_INDEX['name']}", file=sys.stderr)
        return None

    # 转换为周频 (每周五收盘)
    factor_weekly = {}
    for name, series in factor_daily.items():
        factor_weekly[name] = series.resample('W-FRI').last().dropna()

    signal_weekly = sig_df['close'].resample('W-FRI').last().dropna()

    # 对齐日期
    common_idx = signal_weekly.index
    for name in factor_weekly:
        common_idx = common_idx.intersection(factor_weekly[name].index)
    for name in factor_weekly:
        factor_weekly[name] = factor_weekly[name].loc[common_idx]
    signal_weekly = signal_weekly.loc[common_idx]

    print(f"  数据对齐完成: {len(common_idx)} 周 "
          f"({common_idx[0].strftime('%Y-%m-%d')} ~ {common_idx[-1].strftime('%Y-%m-%d')})")

    return factor_weekly, signal_weekly


# ============================================================
# 策略核心逻辑
# ============================================================

def compute_regime(signal_prices: pd.Series) -> tuple[str, float]:
    """
    双时间框架Regime判断。

    Args:
        signal_prices: 信号指数周价格序列

    Returns:
        (regime_label, regime_multiplier)
    """
    ma_short = signal_prices.rolling(PARAMS['ma_short']).mean()
    ma_long = signal_prices.rolling(PARAMS['ma_long']).mean()

    price = signal_prices.iloc[-1]
    above_short = price > ma_short.iloc[-1]
    above_long = price > ma_long.iloc[-1]

    if above_short and above_long:
        return 'strong_bull', PARAMS['regime_full']
    elif not above_short and not above_long:
        return 'bear', PARAMS['regime_bear']
    else:
        return 'transition', PARAMS['regime_transition']


def compute_momentum_weights(factor_weekly: dict[str, pd.Series]) -> dict[str, float]:
    """
    计算因子动量权重 (8周涨幅排名)。

    Args:
        factor_weekly: 因子周价格字典

    Returns:
        {因子名: 权重}
    """
    lookback = PARAMS['momentum_lookback']
    momenta = {}
    for name, prices in factor_weekly.items():
        if len(prices) >= lookback:
            mom = prices.iloc[-1] / prices.iloc[-lookback] - 1
            momenta[name] = mom

    ranked = sorted(momenta.items(), key=lambda x: x[1], reverse=True)
    weights = {}
    for i, (name, _) in enumerate(ranked):
        weights[name] = PARAMS['momentum_weights'][i]

    return weights


def compute_vol_scale(blend_returns: pd.Series) -> float:
    """
    计算波动率缩放系数。

    Args:
        blend_returns: 组合周收益率序列

    Returns:
        缩放系数 (0~1)
    """
    lookback = PARAMS['vol_lookback']
    if len(blend_returns) < lookback:
        return 1.0

    vol_annual = blend_returns.iloc[-lookback:].std() * np.sqrt(52)
    if vol_annual <= 0:
        return 1.0

    return min(PARAMS['vol_target'] / vol_annual, 1.0)


def generate_signal(factor_weekly: dict[str, pd.Series],
                    signal_weekly: pd.Series) -> dict:
    """
    生成本周信号。

    Args:
        factor_weekly: 因子周价格字典
        signal_weekly: 信号指数周价格序列

    Returns:
        信号字典
    """
    latest_date = signal_weekly.index[-1]

    # 1. Regime判断
    regime_label, regime_mult = compute_regime(signal_weekly)
    ma_short = signal_weekly.rolling(PARAMS['ma_short']).mean()
    ma_long = signal_weekly.rolling(PARAMS['ma_long']).mean()

    # 2. 因子动量权重
    mom_weights = compute_momentum_weights(factor_weekly)

    # 3. 计算组合收益率 (用动量权重)
    factor_returns = {name: prices.pct_change().dropna()
                      for name, prices in factor_weekly.items()}
    # 使用等权计算vol (因为动量权重每周变化, 用等权更稳定)
    blend_ret = sum(ret for ret in factor_returns.values()) / len(factor_returns)

    # 4. Vol缩放
    vol_scale = compute_vol_scale(blend_ret)
    vol_annual = blend_ret.iloc[-PARAMS['vol_lookback']:].std() * np.sqrt(52)

    # 5. 最终仓位
    total_position = vol_scale * regime_mult

    # 6. 因子动量排名详情
    lookback = PARAMS['momentum_lookback']
    momentum_detail = []
    for name in sorted(mom_weights, key=lambda x: mom_weights[x], reverse=True):
        mom = factor_weekly[name].iloc[-1] / factor_weekly[name].iloc[-lookback] - 1
        momentum_detail.append({
            'name': name,
            'etf': FACTOR_INDICES[[k for k, v in FACTOR_INDICES.items() if v['name'] == name][0]]['etf'],
            'etf_name': FACTOR_INDICES[[k for k, v in FACTOR_INDICES.items() if v['name'] == name][0]]['etf_name'],
            'momentum_8w': round(mom * 100, 2),
            'weight': mom_weights[name],
            'position_pct': round(total_position * mom_weights[name] * 100, 1),
        })

    # 7. 构建信号
    regime_labels_cn = {
        'strong_bull': '强牛 (价格 > 13周MA 且 > 40周MA)',
        'transition': '过渡 (价格在两条MA之间)',
        'bear': '熊市 (价格 < 13周MA 且 < 40周MA)',
    }

    signal = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': 'Strategy B (增强版)',
        'signal_index': {
            'name': f"{SIGNAL_INDEX['name']}({SIGNAL_INDEX['code']})",
            'price': round(signal_weekly.iloc[-1], 2),
            'ma_13w': round(ma_short.iloc[-1], 2),
            'ma_40w': round(ma_long.iloc[-1], 2),
        },
        'regime': {
            'label': regime_label,
            'label_cn': regime_labels_cn[regime_label],
            'multiplier': regime_mult,
        },
        'volatility': {
            'annual_12w': round(vol_annual * 100, 1),
            'target': round(PARAMS['vol_target'] * 100, 1),
            'scale': round(vol_scale, 3),
        },
        'total_position_pct': round(total_position * 100, 1),
        'cash_pct': round((1 - total_position) * 100, 1),
        'holdings': momentum_detail,
        'action_summary': [],
    }

    # 8. 生成中文调仓建议
    emoji = {'strong_bull': '🟢', 'transition': '🟡', 'bear': '🔴'}[regime_label]
    signal['action_summary'] = [
        f"{emoji} 市场状态: {regime_labels_cn[regime_label]}",
        f"总仓位: {signal['total_position_pct']}% (Regime {regime_mult*100:.0f}% × Vol缩放 {vol_scale*100:.0f}%)",
    ]
    for h in momentum_detail:
        signal['action_summary'].append(
            f"  {h['etf']} {h['etf_name']}: {h['position_pct']}% "
            f"(动量{h['momentum_8w']:+.1f}%, 权重{h['weight']*100:.0f}%)"
        )
    signal['action_summary'].append(f"  货币基金/现金: {signal['cash_pct']}%")

    return signal


# ============================================================
# 回测
# ============================================================

def run_backtest(factor_weekly: dict[str, pd.Series],
                 signal_weekly: pd.Series) -> dict:
    """
    运行完整回测。

    Returns:
        回测结果字典
    """
    print("\n运行回测...")

    # 需要至少40周数据来计算长期MA
    min_weeks = max(PARAMS['ma_long'], PARAMS['vol_lookback'], PARAMS['momentum_lookback'])

    factor_returns = {name: prices.pct_change() for name, prices in factor_weekly.items()}
    names = list(factor_weekly.keys())

    nav = [1.0]
    dates = []
    weekly_positions = []

    for i in range(min_weeks, len(signal_weekly)):
        date = signal_weekly.index[i]

        # Regime
        sig_slice = signal_weekly.iloc[:i+1]
        ma_s = sig_slice.rolling(PARAMS['ma_short']).mean()
        ma_l = sig_slice.rolling(PARAMS['ma_long']).mean()
        price = sig_slice.iloc[-1]
        above_short = price > ma_s.iloc[-1]
        above_long = price > ma_l.iloc[-1]

        if above_short and above_long:
            regime_mult = PARAMS['regime_full']
        elif not above_short and not above_long:
            regime_mult = PARAMS['regime_bear']
        else:
            regime_mult = PARAMS['regime_transition']

        # Momentum weights
        mom = {}
        lb = PARAMS['momentum_lookback']
        for name in names:
            prices_slice = factor_weekly[name].iloc[:i+1]
            if len(prices_slice) >= lb:
                mom[name] = prices_slice.iloc[-1] / prices_slice.iloc[-lb] - 1
        ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)
        weights = {}
        for j, (name, _) in enumerate(ranked):
            weights[name] = PARAMS['momentum_weights'][j]

        # Vol scaling
        blend_ret = sum(factor_returns[name].iloc[:i+1] * weights.get(name, 1/3)
                        for name in names)
        vol_lb = PARAMS['vol_lookback']
        recent = blend_ret.iloc[-vol_lb:]
        vol_ann = recent.std() * np.sqrt(52)
        vol_scale = min(PARAMS['vol_target'] / vol_ann, 1.0) if vol_ann > 0 else 1.0

        # Position
        position = vol_scale * regime_mult

        # Weekly return
        week_ret = sum(factor_returns[name].iloc[i] * weights.get(name, 1/3)
                       for name in names)
        portfolio_ret = position * week_ret
        nav.append(nav[-1] * (1 + portfolio_ret))
        dates.append(date)
        weekly_positions.append(position)

    nav_series = pd.Series(nav[1:], index=dates)

    # 计算指标
    total_years = (dates[-1] - dates[0]).days / 365.25
    cagr = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (1 / total_years) - 1

    drawdown = nav_series / nav_series.cummax() - 1
    mdd = drawdown.min()

    weekly_rets = nav_series.pct_change().dropna()
    sharpe = weekly_rets.mean() / weekly_rets.std() * np.sqrt(52) if weekly_rets.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    # 年度收益
    annual = nav_series.resample('YE').last().pct_change().dropna()
    annual_returns = {str(d.year): round(v * 100, 1) for d, v in annual.items()}

    result = {
        'period': f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}",
        'total_weeks': len(nav_series),
        'cagr_pct': round(cagr * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'annual_returns': annual_returns,
        'avg_position_pct': round(np.mean(weekly_positions) * 100, 1),
    }

    print(f"\n回测结果:")
    print(f"  期间: {result['period']}")
    print(f"  CAGR: {result['cagr_pct']}%")
    print(f"  最大回撤: {result['mdd_pct']}%")
    print(f"  Sharpe: {result['sharpe']}")
    print(f"  Calmar: {result['calmar']}")
    print(f"  平均仓位: {result['avg_position_pct']}%")
    print(f"\n年度收益:")
    for year, ret in sorted(annual_returns.items()):
        print(f"  {year}: {ret:+.1f}%")

    return result


# ============================================================
# 输出格式化
# ============================================================

def format_signal_text(signal: dict) -> str:
    """格式化信号为可读文本 (适合Telegram/终端)。"""
    lines = [
        f"{'='*40}",
        f"策略B 周度调仓信号",
        f"日期: {signal['date']}",
        f"{'='*40}",
        f"",
        f"信号指数: {signal['signal_index']['name']}",
        f"  价格: {signal['signal_index']['price']}",
        f"  13周MA: {signal['signal_index']['ma_13w']}",
        f"  40周MA: {signal['signal_index']['ma_40w']}",
        f"",
    ]

    for line in signal['action_summary']:
        lines.append(line)

    lines.extend([
        f"",
        f"波动率: {signal['volatility']['annual_12w']}% (目标{signal['volatility']['target']}%)",
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
        print(f"策略B: 增强版因子ETF周度信号")
        print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)

    # 加载数据
    result = load_all_data()
    if result is None:
        print("[ERROR] 数据加载失败!", file=sys.stderr)
        sys.exit(1)

    factor_weekly, signal_weekly = result

    if do_backtest:
        bt_result = run_backtest(factor_weekly, signal_weekly)
        output_path = os.path.join(DATA_DIR, 'backtest_result.json')
        with open(output_path, 'w') as f:
            json.dump(bt_result, f, indent=2, ensure_ascii=False)
        print(f"\n回测结果已保存: {output_path}")
        return

    # 生成本周信号
    signal = generate_signal(factor_weekly, signal_weekly)

    # 保存JSON
    output_path = os.path.join(DATA_DIR, 'latest_signal.json')
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
