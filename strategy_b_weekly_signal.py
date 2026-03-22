#!/usr/bin/env python3
"""
Strategy B: Enhanced Factor ETF Weekly Signal
(策略B: 增强版A股因子ETF周度信号)

每周五收盘后运行，输出下周调仓建议。

核心逻辑:
  1. 因子池: 现金流 + 国信价值 + 红利低波 + 创业成长 (全收益指数, 含分红再投资)
  2. 动态权重: 4周因子动量排名 → 第1名50% / 第2名30% / 第3名20% / 第4名0%
  3. 双时间框架Regime (中证全指):
     - 强牛: 价格 > 13周MA 且 > 40周MA → 满仓
     - 过渡: 价格在两条MA之间 → 70%仓位
     - 熊市: 价格 < 13周MA 且 < 40周MA → 35%仓位
  4. 波动率缩放: 目标15%年化波动率, 12周回看, 上限1.3x杠杆

重要说明:
  - 回测使用 T-1 数据决策、T 期收益, 无前瞻偏差
  - 回测包含交易成本 (默认单边 10bp)
  - 因子池为历史表现筛选, 存在选择偏差, 未来表现可能不同

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
DATA_DIR = os.environ.get(
    'ETF_DATA_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
)

# 因子指数 (交易标的) — 使用全收益指数 (Total Return Index, 含分红再投资)
# NOTE: 这4个因子是基于历史表现筛选的, 存在选择偏差 (selection bias).
# 回测结果可能高估未来表现. 建议定期审视因子有效性.
# v3: 加入创业成长, 与防守因子低相关, 牛市提供成长弹性.
# v4: 切换为全收益指数, 回测结果更贴近实盘 (ETF本身包含分红再投资).
#
# 全收益代码对照 (价格指数 → 全收益指数):
#   932365 → 932365CNY010 (中证现金流全收益)
#   931052 → H21052       (国信价值全收益)
#   H30269 → H20269       (红利低波全收益)
#   000958 → H00958       (创业成长全收益)
FACTOR_INDICES = {
    '932365CNY010': {'name': '现金流', 'etf': '159201', 'etf_name': '万家自由现金流ETF'},
    'H21052':       {'name': '国信价值', 'etf': '512040', 'etf_name': '国信价值ETF'},
    'H20269':       {'name': '红利低波', 'etf': '515080', 'etf_name': '红利低波ETF'},
    'H00958':       {'name': '创业成长', 'etf': '159967', 'etf_name': '创业板成长ETF'},
}

# 信号指数 (用于判断市场状态)
SIGNAL_INDEX = {'code': '000985', 'name': '中证全指'}

# 策略参数
PARAMS = {
    'ma_short': 13,             # 短期均线周数
    'ma_long': 40,              # 长期均线周数
    'vol_target': 0.15,         # 目标年化波动率
    'vol_lookback': 12,         # 波动率回看周数
    'momentum_lookback': 4,     # 因子动量回看周数 (v5: 8→4, 更快响应动量变化)
    'momentum_weights': [0.50, 0.30, 0.20, 0.00],  # 排名权重 (v5: 集中前3名, 第4名0%)
    'regime_full': 1.0,         # 强牛仓位系数
    'regime_transition': 0.7,   # 过渡期仓位系数 (v2: 0.5->0.7, 防守因子配得起更高底仓)
    'regime_bear': 0.35,        # 熊市仓位系数 (v2: 0.2->0.35, 因子跌幅远小于大盘)
    'vol_scale_cap': 1.3,       # Vol缩放上限 (v6: 1.0→1.3, 允许低波时适度加杠杆)
    'txn_cost_bps': 10,         # 单边交易成本 (基点, 10bp = 0.1%)
}

# 东方财富 (East Money) 备用数据源 secid 映射
# 当 CSIndex API 被WAF封锁时自动降级使用
# 注意: 东方财富仅提供价格指数, 不含全收益指数
EASTMONEY_SECID = {
    '000985': '1.000985',       # 中证全指 (上证)
    '932365CNY010': None,       # 全收益指数无东财数据
    'H21052': None,
    'H20269': None,
    'H00958': '1.000958',       # 降级为价格指数 000958 (创业成长股息率<0.3%, 差异可忽略)
}

# 缓存过期天数 (超过此天数重新从API获取)
CACHE_MAX_AGE_DAYS = 3


# ============================================================
# 数据获取
# ============================================================

def _fetch_csindex(code: str, days: int, csv_path: str) -> Optional[pd.DataFrame]:
    """从 CSIndex API 获取指数日线数据。"""
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
            if r.status_code == 403:
                print(f"  [WARN] CSIndex WAF封锁 (403), 跳过: {code}/{year}", file=sys.stderr)
                return None  # 被封锁, 立即放弃, 不再浪费请求
            resp = r.json()
            if resp.get('success') and resp.get('data'):
                data = resp['data']
                if isinstance(data, list):
                    all_rows.extend(data)
                elif isinstance(data, dict):
                    all_rows.append(data)
        except Exception as ex:
            print(f"  [WARN] CSIndex请求失败 {code}/{year}: {ex}", file=sys.stderr)
        time.sleep(1.0)  # 增加间隔, 降低被封风险

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['tradeDate'])
    df['close'] = df['close'].astype(float)
    df = df[['date', 'close']].sort_values('date').drop_duplicates('date').set_index('date')
    df.to_csv(csv_path)
    return df


def _fetch_eastmoney(code: str, days: int, csv_path: str) -> Optional[pd.DataFrame]:
    """从东方财富 API 获取指数日线数据 (备用数据源, 仅价格指数)。"""
    secid = EASTMONEY_SECID.get(code)
    if not secid:
        return None

    now = datetime.now()
    start_date = now - timedelta(days=days)
    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    params = {
        'secid': secid,
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101',   # daily
        'fqt': '1',
        'beg': start_date.strftime('%Y%m%d'),
        'end': now.strftime('%Y%m%d'),
        'lmt': '5000',
        'ut': 'fa5fd1943c7b386f172d6893dbbd4dc1',
    }
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        j = r.json()
        klines = j.get('data', {}).get('klines', []) if j.get('data') else []
        if not klines:
            return None
        rows = []
        for k in klines:
            parts = k.split(',')
            rows.append({'date': parts[0], 'close': float(parts[2])})
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df.to_csv(csv_path)
        print(f"  [INFO] 使用东方财富备用数据源: {code} ({len(df)}行)")
        return df
    except Exception as ex:
        print(f"  [WARN] 东方财富请求失败 {code}: {ex}", file=sys.stderr)
        return None


def fetch_index(code: str, days: int = 600) -> Optional[pd.DataFrame]:
    """
    获取指数日线数据。优先使用本地CSV缓存(若未过期)，否则依次尝试:
    1. CSIndex API (主数据源, 支持全收益指数)
    2. 东方财富 API (备用数据源, 仅价格指数)
    3. 过期的本地缓存 (最后兜底)

    Args:
        code: 指数代码 (如 '000985', '932365CNY010')
        days: 获取最近多少天的数据

    Returns:
        DataFrame with DatetimeIndex and 'close' column, or None on failure
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f'{code}_daily.csv')

    # 优先读本地缓存 (检查是否过期)
    if os.path.exists(csv_path):
        file_age_days = (time.time() - os.path.getmtime(csv_path)) / 86400
        if file_age_days <= CACHE_MAX_AGE_DAYS:
            df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
            if len(df) > 0:
                return df[['close']]
        else:
            print(f"  [INFO] 缓存已过期 ({file_age_days:.1f}天), 尝试刷新: {code}")

    # 数据源1: CSIndex API
    df = _fetch_csindex(code, days, csv_path)
    if df is not None and len(df) > 0:
        return df[['close']]

    # 数据源2: 东方财富 (仅部分指数可用)
    df = _fetch_eastmoney(code, days, csv_path)
    if df is not None and len(df) > 0:
        return df[['close']]

    # 兜底: 过期缓存
    if os.path.exists(csv_path):
        print(f"  [WARN] 所有API失败, 使用过期缓存: {code}", file=sys.stderr)
        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        if len(df) > 0:
            return df[['close']]

    print(f"  [ERROR] 无法获取数据: {code}", file=sys.stderr)
    return None


def load_all_data(for_backtest: bool = False) -> Optional[Tuple[Dict[str, pd.Series], pd.Series]]:
    """
    加载所有需要的指数数据并转换为周频。

    Args:
        for_backtest: 如果True, 获取更长时间的数据 (4500天 ~12年)

    Returns:
        (factor_weekly, signal_weekly) 或 None
    """
    days = 4500 if for_backtest else 600
    print(f"加载数据 ({'回测模式, ~12年' if for_backtest else '信号模式, ~2年'})...")

    # 加载因子指数
    factor_daily = {}
    for code, info in FACTOR_INDICES.items():
        print(f"  {info['name']} ({code})...")
        df = fetch_index(code, days=days)
        if df is None:
            print(f"  [ERROR] 无法加载 {info['name']}", file=sys.stderr)
            return None
        factor_daily[info['name']] = df['close']
        time.sleep(0.3)

    # 加载信号指数
    print(f"  {SIGNAL_INDEX['name']} ({SIGNAL_INDEX['code']})...")
    sig_df = fetch_index(SIGNAL_INDEX['code'], days=days)
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

def compute_regime(signal_prices: pd.Series) -> Tuple[str, float]:
    """
    双时间框架Regime判断。

    Args:
        signal_prices: 信号指数周价格序列 (应不包含当期决策周)

    Returns:
        (regime_label, regime_multiplier)
    """
    if len(signal_prices) < PARAMS['ma_long']:
        # 数据不足, 默认过渡期
        return 'transition', PARAMS['regime_transition']

    ma_short = signal_prices.rolling(PARAMS['ma_short']).mean()
    ma_long = signal_prices.rolling(PARAMS['ma_long']).mean()

    price = signal_prices.iloc[-1]
    ma_s_val = ma_short.iloc[-1]
    ma_l_val = ma_long.iloc[-1]

    # 检查NaN (前ma_long周可能为NaN)
    if pd.isna(ma_s_val) or pd.isna(ma_l_val):
        return 'transition', PARAMS['regime_transition']

    above_short = price > ma_s_val
    above_long = price > ma_l_val

    if above_short and above_long:
        return 'strong_bull', PARAMS['regime_full']
    elif not above_short and not above_long:
        return 'bear', PARAMS['regime_bear']
    else:
        return 'transition', PARAMS['regime_transition']


def compute_momentum_weights(factor_weekly: Dict[str, pd.Series]) -> Dict[str, float]:
    """
    计算因子动量权重 (8周涨幅排名)。

    Args:
        factor_weekly: 因子周价格字典 (应不包含当期决策周)

    Returns:
        {因子名: 权重}
    """
    lookback = PARAMS['momentum_lookback']
    momenta = {}
    for name, prices in factor_weekly.items():
        if len(prices) >= lookback:
            mom = prices.iloc[-1] / prices.iloc[-lookback] - 1
            momenta[name] = mom

    # 稳定排序: 动量相同时按名称排序, 避免依赖dict顺序
    ranked = sorted(momenta.items(), key=lambda x: (-x[1], x[0]))
    weights = {}
    for i, (name, _) in enumerate(ranked):
        if i < len(PARAMS['momentum_weights']):
            weights[name] = PARAMS['momentum_weights'][i]
        else:
            weights[name] = 0.0

    return weights


def compute_vol_scale(blend_returns: pd.Series) -> Tuple[float, float]:
    """
    计算波动率缩放系数。

    Args:
        blend_returns: 组合周收益率序列

    Returns:
        (缩放系数 0~1, 年化波动率)
    """
    lookback = PARAMS['vol_lookback']
    if len(blend_returns) < lookback:
        return 1.0, 0.0

    vol_annual = blend_returns.iloc[-lookback:].std() * np.sqrt(52)
    if vol_annual <= 0:
        return 1.0, 0.0

    scale = min(PARAMS['vol_target'] / vol_annual, PARAMS['vol_scale_cap'])
    return scale, vol_annual


def generate_signal(factor_weekly: Dict[str, pd.Series],
                    signal_weekly: pd.Series) -> dict:
    """
    生成本周信号。

    信号生成使用截至本周五的所有已知数据(因为是收盘后运行,
    数据已实现, 不存在前瞻问题)。

    Args:
        factor_weekly: 因子周价格字典
        signal_weekly: 信号指数周价格序列

    Returns:
        信号字典
    """
    latest_date = signal_weekly.index[-1]

    # 1. Regime判断 (使用全部历史, 当前价格已实现)
    regime_label, regime_mult = compute_regime(signal_weekly)
    ma_short = signal_weekly.rolling(PARAMS['ma_short']).mean()
    ma_long = signal_weekly.rolling(PARAMS['ma_long']).mean()

    # 2. 因子动量权重
    mom_weights = compute_momentum_weights(factor_weekly)

    # 3. 计算历史组合收益率 (每期用当期权重, 与回测一致)
    factor_returns = {name: prices.pct_change().dropna()
                      for name, prices in factor_weekly.items()}
    names = list(factor_weekly.keys())
    lookback_vol = PARAMS['vol_lookback']
    lookback_mom = PARAMS['momentum_lookback']
    n_weeks = len(signal_weekly)
    # 需要回溯足够多的周来积累 vol_lookback 个历史组合收益
    start_from = max(0, n_weeks - lookback_vol - 2)
    historical_port_rets = []
    for t in range(start_from, n_weeks):
        t_factor_slices = {name: factor_weekly[name].iloc[:t] for name in names}
        if len(t_factor_slices[names[0]]) < lookback_mom:
            continue
        t_weights = compute_momentum_weights(t_factor_slices)
        t_ret = sum(factor_returns[name].iloc[t] * t_weights.get(name, 1/len(names))
                    for name in names if t < len(factor_returns[name]))
        historical_port_rets.append(float(t_ret))
    blend_ret = pd.Series(historical_port_rets)

    # 4. Vol缩放
    vol_scale, vol_annual = compute_vol_scale(blend_ret)

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
        f"总仓位: {signal['total_position_pct']}% (Regime {regime_mult*100:.0f}% x Vol缩放 {vol_scale*100:.0f}%)",
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

def run_backtest(factor_weekly: Dict[str, pd.Series],
                 signal_weekly: pd.Series) -> dict:
    """
    运行完整回测。

    关键设计:
      - 在第 i 周, 使用 [:i] (即 T-1 及之前) 的数据做所有决策
      - 决策后的仓位承受第 i 周的实际收益 (factor_returns[i])
      - 这模拟了: 周五收盘后看数据 → 下周一开盘执行 → 承受下周收益
      - 包含交易成本: 仓位变化时收取单边成本

    Returns:
        回测结果字典
    """
    print("\n运行回测...")

    txn_cost = PARAMS['txn_cost_bps'] / 10000  # 转换为小数

    # 需要至少 ma_long+1 周数据来保证有足够的MA历史
    min_weeks = max(PARAMS['ma_long'], PARAMS['vol_lookback'], PARAMS['momentum_lookback']) + 1

    factor_returns = {name: prices.pct_change() for name, prices in factor_weekly.items()}
    names = list(factor_weekly.keys())

    nav = [1.0]
    dates = []
    weekly_positions = []
    total_txn_cost = 0.0
    prev_weights = {name: 0.0 for name in names}
    prev_position = 0.0

    # 记录每期实际组合收益 (用当期权重), 用于后续vol scaling
    # 修复: 之前用当前期权重回溯blend全部历史收益, 不是真实历史组合波动率
    historical_port_rets = []

    for i in range(min_weeks, len(signal_weekly)):
        date = signal_weekly.index[i]

        # ---- 决策阶段: 仅使用 [:i] 的数据 (T-1 及之前) ----

        # Regime: 用截至上周五 (index i-1) 的价格
        sig_slice = signal_weekly.iloc[:i]
        regime_label, regime_mult = compute_regime(sig_slice)

        # Momentum weights: 用截至上周五的因子价格
        factor_slices = {name: factor_weekly[name].iloc[:i] for name in names}
        mom_weights = compute_momentum_weights(factor_slices)

        # Vol scaling: 使用历史实际组合收益序列 (每期用当期权重计算的收益)
        if len(historical_port_rets) >= PARAMS['vol_lookback']:
            hist_ret_series = pd.Series(historical_port_rets)
            vol_scale, _ = compute_vol_scale(hist_ret_series)
        else:
            vol_scale = 1.0

        # Position
        position = vol_scale * regime_mult

        # ---- 交易成本: 计算仓位变化 ----
        turnover = 0.0
        for name in names:
            new_w = position * mom_weights.get(name, 0.0)
            old_w = prev_position * prev_weights.get(name, 0.0)
            turnover += abs(new_w - old_w)
        # 也考虑现金仓位变化 (卖出ETF=买入现金, 反之亦然, 只算ETF端)
        period_txn_cost = turnover * txn_cost
        total_txn_cost += period_txn_cost

        prev_weights = mom_weights.copy()
        prev_position = position

        # ---- 收益阶段: 使用第 i 周的实际收益 ----
        week_ret = sum(factor_returns[name].iloc[i] * mom_weights.get(name, 1/len(names))
                       for name in names)
        portfolio_ret = position * week_ret - period_txn_cost
        nav.append(nav[-1] * (1 + portfolio_ret))
        dates.append(date)
        weekly_positions.append(position)

        # 记录本期加权因子收益 (不含仓位缩放, 用于后续vol scaling)
        historical_port_rets.append(float(week_ret))

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
        'total_txn_cost_pct': round(total_txn_cost * 100, 2),
        'txn_cost_bps': PARAMS['txn_cost_bps'],
        'note': '回测使用T-1数据决策, 包含交易成本, 无前瞻偏差',
    }

    print(f"\n回测结果:")
    print(f"  期间: {result['period']}")
    print(f"  CAGR: {result['cagr_pct']}%")
    print(f"  最大回撤: {result['mdd_pct']}%")
    print(f"  Sharpe: {result['sharpe']}")
    print(f"  Calmar: {result['calmar']}")
    print(f"  平均仓位: {result['avg_position_pct']}%")
    print(f"  累计交易成本: {result['total_txn_cost_pct']}%")
    print(f"  (单边{PARAMS['txn_cost_bps']}bp, 无前瞻偏差)")
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
    result = load_all_data(for_backtest=do_backtest)
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
