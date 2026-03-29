#!/usr/bin/env python3
"""
A股个股因子策略 — 实盘选股脚本

策略 A: 低换手 + 12M动量  CSI800 Top20  月度调仓
策略 B: 低换手 + 12M动量 + 高EP  CSI800 Top20  月度调仓

回测绩效 (2016-01 ~ 2026-03, 历史成分股, 单边TC=30bps):
  策略A: CAGR=21.4%, Sharpe=0.966, MaxDD=-25.4%
  策略B: CAGR~19.7%, Sharpe~0.878 (Top30时)

选股逻辑:
  1. 每月底(最后一个交易日), 从当时CSI800成分股中选股
  2. 对每只股票计算: 20日平均换手率(rank, 升序), 12个月动量-skip1M(rank, 降序)
     策略B额外加入: 20日年化波动率(rank, 升序)
  3. 多因子等权合成: composite = sum(weight * rank_percentile)
     其中rank越小越好(即composite最低的30只入选)
  4. 等权买入Top30, 持有一个月, 下月底调仓
  5. 单边交易成本假设30bps (含佣金+滑点+冲击成本)

使用方法:
  python3 stock_strategy_live.py              # 显示最新选股结果
  python3 stock_strategy_live.py --backtest   # 运行回测

Author: Sarah Mitchell / VisionClaw
Date: 2026-03-29
"""
from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import akshare as ak
import baostock as bs
import json, os, sys, time, pickle, argparse
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = '/Users/claw/etf-trader/data'
CACHE_DIR = os.path.join(DATA_DIR, 'stock_cache_v3')
PRICE_CACHE_DIR = os.path.join(DATA_DIR, 'stock_cache_v2')
os.makedirs(CACHE_DIR, exist_ok=True)

RISK_FREE_RATE = 0.025
TC_ONE_SIDE = 0.003          # 单边30bps
START_DATE = '20150101'       # 数据起始(留足12M动量look-back)
END_DATE = datetime.now().strftime('%Y%m%d')
BACKTEST_START = '2016-01-31'

# 策略定义
STRATEGIES = {
    'A_LowTurn_Mom12M': {
        'name': '低换手+12M动量',
        'description': '经V3回测验证的最佳双因子策略 (Top20最优Sharpe)',
        'factors': {
            'turn_20d':  (1.0, True),    # 低换手: rank升序(换手率低的排前)
            'mom_12m':   (1.0, False),   # 12M动量: rank降序(动量高的排前)
        },
        'universe': 'CSI800',
        'top_n': 20,
        'rebal_months': 1,  # 月度
        'backtest_perf': {
            'CAGR': 21.4, 'Sharpe': 0.966, 'MaxDD': -25.4,
            'yearly': {2016: 38.1, 2017: 81.8, 2018: -17.2, 2019: 10.8,
                       2020: 72.8, 2021: -9.2, 2022: -0.8, 2023: 0.2,
                       2024: 17.3, 2025: 2.8}
        }
    },
    'B_LowTurn_Mom12M_EP': {
        'name': '低换手+12M动量+高EP(价值增强)',
        'description': '在策略A基础上加入EP(盈利收益率)因子, 偏向低估值动量股, 与A形成互补',
        'factors': {
            'turn_20d':  (1.0, True),    # 低换手: rank升序
            'mom_12m':   (1.0, False),   # 12M动量: rank降序
            'ep':        (1.0, False),   # 高EP(=1/PE): rank降序(EP高的排前)
        },
        'universe': 'CSI800',
        'top_n': 20,
        'rebal_months': 1,
        'backtest_perf': {
            'CAGR': 19.7, 'Sharpe': 0.878, 'MaxDD': -32.0,
            'yearly': {2016: 35.0, 2017: 57.5, 2018: -18.0, 2019: -4.5,
                       2020: 18.9, 2021: 13.8, 2022: 10.4, 2023: 16.5,
                       2024: 19.6, 2025: 12.0}
        }
    },
}


# ============================================================
# 1. 数据获取: 历史成分股
# ============================================================
def get_historical_constituents(index_type='hs300', start_year=2016, end_year=2026):
    """获取CSI300/CSI500每月末的历史成分股列表"""
    cache_file = os.path.join(CACHE_DIR, f'{index_type}_hist_constituents.pkl')
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 86400 * 7:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

    lg = bs.login()
    query_fn = bs.query_hs300_stocks if index_type == 'hs300' else bs.query_zz500_stocks

    dates = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            if m == 12:
                dates.append(f'{y}-{m:02d}-31')
            elif m in [1, 3, 5, 7, 8, 10]:
                dates.append(f'{y}-{m:02d}-31')
            elif m in [4, 6, 9, 11]:
                dates.append(f'{y}-{m:02d}-30')
            else:
                dates.append(f'{y}-{m:02d}-28')

    constituents = {}
    for dt_str in dates:
        rs = query_fn(date=dt_str)
        codes = set()
        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            code = row[1].split('.')[1]
            codes.add(code)
        if codes:
            constituents[dt_str] = codes
    bs.logout()
    print(f"  {index_type}: {len(constituents)} dates loaded")

    with open(cache_file, 'wb') as f:
        pickle.dump(constituents, f)
    return constituents


def get_constituents_at_date(const_dict, target_date):
    """获取某日期最近的成分股列表"""
    if isinstance(target_date, str):
        target_str = target_date
    else:
        target_str = target_date.strftime('%Y-%m-%d')
    best_date = None
    for dt_str in sorted(const_dict.keys()):
        if dt_str <= target_str:
            best_date = dt_str
        else:
            break
    return const_dict.get(best_date, set()) if best_date else set()


def get_csi800_at_date(hs300_const, zz500_const, target_date):
    """获取某日期的CSI800(=CSI300+CSI500)成分股"""
    c300 = get_constituents_at_date(hs300_const, target_date)
    c500 = get_constituents_at_date(zz500_const, target_date)
    return c300 | c500


# ============================================================
# 2. 数据获取: 行情 & 估值
# ============================================================
def fetch_stock_akshare(code):
    """获取单只股票日线数据(akshare, 前复权)"""
    for cache_dir in [PRICE_CACHE_DIR, CACHE_DIR]:
        cache_file = os.path.join(cache_dir, f'{code}_ak.pkl')
        if os.path.exists(cache_file):
            mtime = os.path.getmtime(cache_file)
            if time.time() - mtime < 86400 * 3:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

    try:
        df = ak.stock_zh_a_hist(symbol=code, period='daily',
                                start_date=START_DATE, end_date=END_DATE, adjust='qfq')
        if df is None or len(df) < 100:
            return None
        df = df.rename(columns={
            '日期': 'date', '收盘': 'close', '开盘': 'open',
            '最高': 'high', '最低': 'low',
            '成交量': 'volume', '成交额': 'amount',
            '涨跌幅': 'pctChg', '换手率': 'turnover',
        })
        df['date'] = pd.to_datetime(df['date'])
        for c in ['close', 'open', 'high', 'low', 'volume', 'amount', 'pctChg', 'turnover']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.set_index('date').sort_index()
        df = df[df['volume'] > 0]
        cache_file = os.path.join(CACHE_DIR, f'{code}_ak.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        return df
    except:
        return None


def fetch_stock_valuation_bs(code):
    """获取PE/PB估值(baostock)"""
    for cache_dir in [PRICE_CACHE_DIR, CACHE_DIR]:
        cache_file = os.path.join(cache_dir, f'{code}_val.pkl')
        if os.path.exists(cache_file):
            mtime = os.path.getmtime(cache_file)
            if time.time() - mtime < 86400 * 3:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

    prefix = 'sh' if code.startswith('6') else 'sz'
    bs_code = f'{prefix}.{code}'
    try:
        rs = bs.query_history_k_data_plus(
            bs_code, fields='date,peTTM,pbMRQ',
            frequency='d', start_date='2015-01-01', end_date='2026-03-31',
            adjustflag='2')
        data = []
        while rs.error_code == '0' and rs.next():
            data.append(rs.get_row_data())
        if not data:
            return None
        df = pd.DataFrame(data, columns=['date', 'peTTM', 'pbMRQ'])
        df['date'] = pd.to_datetime(df['date'])
        df['peTTM'] = pd.to_numeric(df['peTTM'], errors='coerce')
        df['pbMRQ'] = pd.to_numeric(df['pbMRQ'], errors='coerce')
        df = df.set_index('date').sort_index()
        cache_file = os.path.join(CACHE_DIR, f'{code}_val.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        return df
    except:
        return None


def get_stock_name(code):
    """获取股票名称(尽力而为)"""
    try:
        # Try from akshare spot data
        return None  # Will use code as fallback
    except:
        return None


# ============================================================
# 3. 因子计算
# ============================================================
def compute_factors_single(code, df, as_of_date, val_data=None):
    """
    计算单只股票在某一天的所有因子值。

    Args:
        code: 股票代码
        df: 该股票的日线DataFrame
        as_of_date: 计算截止日期
        val_data: 估值数据dict (可选)

    Returns:
        dict of factor values, or None if data insufficient
    """
    hist = df[df.index <= as_of_date]
    n = len(hist)

    # 至少需要264个交易日(约12个月)来计算12M动量
    if n < 264:
        return None

    # 最近交易日距离as_of_date不能超过5天(防止停牌股)
    if (as_of_date - hist.index[-1]).days > 5:
        return None

    last = hist.iloc[-1]
    rec = {
        'code': code,
        'close': last['close'],
        'last_date': hist.index[-1].strftime('%Y-%m-%d'),
    }

    # ---- 换手率因子: 20日平均换手率 ----
    # 换手率越低越好 → rank ascending
    rec['turn_20d'] = hist['turnover'].iloc[-20:].mean()

    # ---- 1M短期收益 (用于reversal, 非选股因子) ----
    rec['ret_1m'] = hist['close'].iloc[-1] / hist['close'].iloc[-22] - 1

    # ---- 6M动量 (skip最近1个月) ----
    if n >= 132:
        rec['mom_6m'] = hist['close'].iloc[-22] / hist['close'].iloc[-132] - 1

    # ---- 12M动量 (skip最近1个月) ----
    # 用t-22到t-264的收益, 避免短期反转
    rec['mom_12m'] = hist['close'].iloc[-22] / hist['close'].iloc[-264] - 1

    # ---- 20日年化波动率 ----
    daily_ret = hist['pctChg'].iloc[-20:] / 100.0
    rec['vol_20d'] = daily_ret.std() * np.sqrt(252)

    # ---- 估值因子 (可选) ----
    if val_data and code in val_data:
        vdf = val_data[code]
        vhist = vdf[vdf.index <= as_of_date]
        if len(vhist) > 0:
            vlast = vhist.iloc[-1]
            pe = vlast.get('peTTM', np.nan)
            pb = vlast.get('pbMRQ', np.nan)
            if pd.notna(pe) and pe > 0:
                rec['ep'] = 1.0 / pe
            if pd.notna(pb) and pb > 0:
                rec['bp'] = 1.0 / pb

    return rec


def build_composite_signal(df_factors, factor_weights):
    """
    多因子等权合成信号。

    对每个因子:
      - ascending=True: 值越小排名越靠前(如低换手、低波动)
      - ascending=False: 值越大排名越靠前(如高动量)

    rank_pct ∈ [0, 1], 值越小越好
    composite = Σ weight × rank_pct
    最终选composite最小的N只股票

    Args:
        df_factors: DataFrame, 每行一只股票, 列含因子值
        factor_weights: dict {factor_name: (weight, ascending)}

    Returns:
        Series of composite signals (lower = better)
    """
    composite = pd.Series(0.0, index=df_factors.index)
    valid_mask = pd.Series(True, index=df_factors.index)

    for fname, (weight, ascending) in factor_weights.items():
        if fname not in df_factors.columns:
            print(f"  WARNING: factor '{fname}' not found in data")
            return None

        vals = df_factors[fname]
        valid_mask &= vals.notna()

        # ascending=True: 值小的rank_pct小 (好)
        # ascending=False: 值大的rank_pct小 (好)
        if ascending:
            rank_pct = vals.rank(ascending=True, pct=True)
        else:
            rank_pct = vals.rank(ascending=False, pct=True)

        composite += weight * rank_pct

    composite[~valid_mask] = np.nan
    return composite


# ============================================================
# 4. 选股函数
# ============================================================
def select_stocks(all_data, val_data, hs300_const, zz500_const,
                  as_of_date, strategy_config):
    """
    在某一日期按策略选股。

    Returns:
        DataFrame with selected stocks and their factor values,
        sorted by composite signal (ascending)
    """
    factors = strategy_config['factors']
    universe = strategy_config['universe']
    top_n = strategy_config['top_n']

    # 1. 获取成分股
    if universe == 'CSI300':
        eligible = get_constituents_at_date(hs300_const, as_of_date)
    elif universe == 'CSI500':
        eligible = get_constituents_at_date(zz500_const, as_of_date)
    else:  # CSI800
        eligible = get_csi800_at_date(hs300_const, zz500_const, as_of_date)

    print(f"  成分股数量: {len(eligible)}")

    # 2. 计算因子
    records = []
    skipped = {'no_data': 0, 'insufficient': 0}
    for code in sorted(eligible):
        if code not in all_data:
            skipped['no_data'] += 1
            continue
        rec = compute_factors_single(code, all_data[code], as_of_date, val_data)
        if rec is None:
            skipped['insufficient'] += 1
            continue
        records.append(rec)

    print(f"  有效股票: {len(records)} (无数据: {skipped['no_data']}, 数据不足: {skipped['insufficient']})")

    if len(records) < top_n:
        print(f"  ERROR: 有效股票({len(records)})不足top_n({top_n})")
        return None

    df = pd.DataFrame(records)

    # 3. 合成信号
    composite = build_composite_signal(df, factors)
    if composite is None:
        print("  ERROR: 合成信号计算失败")
        return None

    df['composite'] = composite.values
    df = df.dropna(subset=['composite'])

    # 4. 选Top N
    df = df.sort_values('composite', ascending=True)
    selected = df.head(top_n).copy()
    selected['rank'] = range(1, len(selected) + 1)

    return selected


# ============================================================
# 5. 回测引擎
# ============================================================
def get_month_ends(all_data):
    """获取所有月末交易日"""
    all_dates = set()
    for code, df in all_data.items():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)
    me_df = pd.DataFrame({'date': all_dates})
    me_df = me_df[me_df['date'] >= pd.Timestamp(BACKTEST_START)]
    me_df['ym'] = me_df['date'].dt.to_period('M')
    month_ends = me_df.groupby('ym')['date'].max().tolist()
    return sorted(month_ends)


def run_backtest(all_data, val_data, month_ends, hs300_const, zz500_const,
                 strategy_key, strategy_config):
    """
    运行回测, 返回绩效指标和NAV序列。
    """
    factors = strategy_config['factors']
    universe = strategy_config['universe']
    top_n = strategy_config['top_n']
    rebal_months = strategy_config['rebal_months']

    print(f"\n{'='*60}")
    print(f"回测: {strategy_config['name']} ({universe} top{top_n} {rebal_months}M)")
    print(f"{'='*60}")

    nav = 1.0
    nav_history = []
    prev_holdings = set()
    rebal_count = 0
    total_turnover = 0
    rebal_events = 0

    for i in range(len(month_ends) - 1):
        dt = month_ends[i]
        dt_next = month_ends[i + 1]

        # 获取成分股
        if universe == 'CSI300':
            eligible = get_constituents_at_date(hs300_const, dt)
        elif universe == 'CSI500':
            eligible = get_constituents_at_date(zz500_const, dt)
        else:
            eligible = get_csi800_at_date(hs300_const, zz500_const, dt)

        if not eligible:
            nav_history.append({'date': dt, 'nav': nav})
            rebal_count += 1
            continue

        do_rebal = (rebal_count % rebal_months == 0)
        rebal_count += 1

        if do_rebal:
            # 计算因子并选股
            records = []
            for code in eligible:
                if code not in all_data:
                    continue
                rec = compute_factors_single(code, all_data[code], dt, val_data)
                if rec:
                    records.append(rec)

            if len(records) < top_n:
                nav_history.append({'date': dt, 'nav': nav})
                continue

            cur = pd.DataFrame(records)
            composite = build_composite_signal(cur, factors)
            if composite is None or composite.notna().sum() < top_n:
                nav_history.append({'date': dt, 'nav': nav})
                continue

            cur['signal'] = composite.values
            cur = cur.dropna(subset=['signal'])
            top = cur.nsmallest(top_n, 'signal')
            new_holdings = set(top['code'].tolist())
            rebal_events += 1
        else:
            new_holdings = prev_holdings.copy()

        # 计算持仓的前向收益
        fwd_rets = []
        for code in new_holdings:
            if code not in all_data:
                continue
            df = all_data[code]
            hist_cur = df[df.index <= dt]
            hist_nxt = df[df.index <= dt_next]
            if len(hist_cur) == 0 or len(hist_nxt) == 0:
                continue
            p_cur = hist_cur.iloc[-1]['close']
            p_nxt = hist_nxt.iloc[-1]['close']
            if p_cur > 0:
                fwd_rets.append(p_nxt / p_cur - 1)

        if not fwd_rets:
            nav_history.append({'date': dt, 'nav': nav})
            continue

        # Winsorize极端值
        fwd_arr = np.array(fwd_rets)
        p01, p99 = np.percentile(fwd_arr, [1, 99])
        fwd_arr = np.clip(fwd_arr, p01, p99)
        port_ret = fwd_arr.mean()

        # 交易成本
        if do_rebal and prev_holdings:
            turnover = len(new_holdings - prev_holdings) / max(len(new_holdings), 1)
            tc_cost = turnover * TC_ONE_SIDE * 2  # 双边
            total_turnover += turnover
        elif not prev_holdings:
            tc_cost = TC_ONE_SIDE  # 首次建仓只收单边
            total_turnover += 1.0
        else:
            tc_cost = 0

        nav *= (1 + port_ret - tc_cost)
        nav_history.append({'date': dt, 'nav': nav})
        prev_holdings = new_holdings

    if not nav_history or len(nav_history) < 12:
        return None

    # 计算绩效指标
    nav_df = pd.DataFrame(nav_history).set_index('date')
    total_days = (nav_df.index[-1] - nav_df.index[0]).days
    years = total_days / 365.25
    if years < 1:
        return None

    cagr = (nav_df['nav'].iloc[-1]) ** (1 / years) - 1
    monthly_ret = nav_df['nav'].pct_change().dropna()
    ann_vol = monthly_ret.std() * np.sqrt(12)
    sharpe = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    dd = nav_df['nav'] / nav_df['nav'].cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    avg_turnover = total_turnover / max(rebal_events, 1)

    # 年度收益
    nav_df['year'] = nav_df.index.year
    yearly = {}
    for year, grp in nav_df.groupby('year'):
        if len(grp) >= 2:
            yr_ret = grp['nav'].iloc[-1] / grp['nav'].iloc[0] - 1
            yearly[int(year)] = round(yr_ret * 100, 2)

    # 月度胜率
    win_rate = (monthly_ret > 0).sum() / len(monthly_ret) * 100

    result = {
        'strategy': strategy_key,
        'name': strategy_config['name'],
        'CAGR': round(cagr * 100, 2),
        'AnnVol': round(ann_vol * 100, 2),
        'Sharpe': round(sharpe, 3),
        'MaxDD': round(max_dd * 100, 2),
        'Calmar': round(calmar, 3),
        'WinRate': round(win_rate, 1),
        'AvgTurnover': round(avg_turnover * 100, 1),
        'FinalNAV': round(nav_df['nav'].iloc[-1], 4),
        'years': round(years, 1),
        'yearly_returns': yearly,
        'nav_series': nav_df['nav'].to_dict(),
    }

    return result


# ============================================================
# 6. 股票名称获取
# ============================================================
def load_stock_names():
    """批量获取A股股票名称映射"""
    cache_file = os.path.join(CACHE_DIR, 'stock_names.pkl')
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 86400 * 7:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    try:
        df = ak.stock_info_a_code_name()
        name_map = dict(zip(df['code'].astype(str).str.zfill(6), df['name']))
        with open(cache_file, 'wb') as f:
            pickle.dump(name_map, f)
        return name_map
    except:
        return {}


# ============================================================
# 7. 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='A股个股因子策略')
    parser.add_argument('--backtest', action='store_true', help='运行回测')
    parser.add_argument('--strategy', type=str, default='both',
                        choices=['A', 'B', 'both'], help='选择策略 (A/B/both)')
    parser.add_argument('--date', type=str, default=None,
                        help='选股日期 (YYYY-MM-DD), 默认最新')
    args = parser.parse_args()

    t_start = time.time()

    print("=" * 70)
    print("A股个股因子策略 — 实盘选股系统")
    print("=" * 70)

    # ---- 1. 加载历史成分股 ----
    print("\n[1/4] 加载CSI800历史成分股...")
    hs300_const = get_historical_constituents('hs300', 2016, 2026)
    zz500_const = get_historical_constituents('zz500', 2016, 2026)

    all_codes_ever = set()
    for codes in hs300_const.values():
        all_codes_ever |= codes
    for codes in zz500_const.values():
        all_codes_ever |= codes
    print(f"  历史CSI800累计股票数: {len(all_codes_ever)}")

    # ---- 2. 加载行情数据 ----
    print(f"\n[2/4] 加载行情数据 ({len(all_codes_ever)} stocks)...")
    all_data = {}
    failed = 0
    for i, code in enumerate(sorted(all_codes_ever)):
        if (i + 1) % 300 == 0:
            print(f"    {i+1}/{len(all_codes_ever)} ...")
        df = fetch_stock_akshare(code)
        if df is not None and len(df) > 60:
            all_data[code] = df
        else:
            failed += 1
        time.sleep(0.01)
    print(f"  成功加载: {len(all_data)} stocks, 失败: {failed}")

    # ---- 3. 加载估值数据 ----
    print(f"\n[3/4] 加载估值数据...")
    val_data = {}
    for code in all_data.keys():
        for cache_dir in [PRICE_CACHE_DIR, CACHE_DIR]:
            cache_file = os.path.join(cache_dir, f'{code}_val.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    val_data[code] = pickle.load(f)
                break
    print(f"  估值数据: {len(val_data)} stocks")

    # ---- 4. 获取股票名称 ----
    print(f"\n[4/4] 获取股票名称...")
    name_map = load_stock_names()
    print(f"  名称映射: {len(name_map)} stocks")

    # ============================================================
    # 回测模式
    # ============================================================
    if args.backtest:
        month_ends = get_month_ends(all_data)
        print(f"\n  月末日期数: {len(month_ends)} ({month_ends[0].date()} ~ {month_ends[-1].date()})")

        strategies_to_test = []
        if args.strategy in ['A', 'both']:
            strategies_to_test.append('A_LowTurn_Mom12M')
        if args.strategy in ['B', 'both']:
            strategies_to_test.append('B_LowTurn_Mom12M_EP')

        results = []
        for key in strategies_to_test:
            config = STRATEGIES[key]
            r = run_backtest(all_data, val_data, month_ends,
                            hs300_const, zz500_const, key, config)
            if r:
                results.append(r)
                print(f"\n  {r['name']}:")
                print(f"    CAGR={r['CAGR']:.1f}%, Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']:.1f}%")
                print(f"    年化波动={r['AnnVol']:.1f}%, Calmar={r['Calmar']:.3f}, 月胜率={r['WinRate']:.0f}%")
                print(f"    平均换手率={r['AvgTurnover']:.0f}%/次")
                print(f"    年度收益:")
                for yr, ret in sorted(r['yearly_returns'].items()):
                    marker = '🟢' if ret > 0 else '🔴'
                    print(f"      {yr}: {ret:>+7.1f}% {marker}")

        # 对比
        if len(results) == 2:
            a, b = results[0], results[1]
            print(f"\n{'='*60}")
            print(f"策略对比:")
            print(f"{'='*60}")
            print(f"  {'指标':<12s} {'策略A(双因子)':<18s} {'策略B(三因子)':<18s}")
            print(f"  {'CAGR':<12s} {a['CAGR']:>+6.1f}%{'':>10s} {b['CAGR']:>+6.1f}%")
            print(f"  {'Sharpe':<12s} {a['Sharpe']:>7.3f}{'':>9s} {b['Sharpe']:>7.3f}")
            print(f"  {'MaxDD':<12s} {a['MaxDD']:>6.1f}%{'':>10s} {b['MaxDD']:>6.1f}%")
            print(f"  {'Calmar':<12s} {a['Calmar']:>7.3f}{'':>9s} {b['Calmar']:>7.3f}")
            print(f"  {'月胜率':<10s} {a['WinRate']:>6.0f}%{'':>10s} {b['WinRate']:>6.0f}%")

        # 保存
        out_path = os.path.join(DATA_DIR, 'stock_strategy_live_backtest.json')
        save_results = []
        for r in results:
            r_save = {k: v for k, v in r.items() if k != 'nav_series'}
            r_save['nav_series'] = {str(k): v for k, v in r.get('nav_series', {}).items()}
            save_results.append(r_save)
        with open(out_path, 'w') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n回测结果已保存: {out_path}")

    # ============================================================
    # 选股模式 (默认)
    # ============================================================
    else:
        # 确定选股日期
        if args.date:
            as_of_date = pd.Timestamp(args.date)
        else:
            # 最新可用交易日
            all_dates = set()
            for code, df in list(all_data.items())[:100]:
                all_dates.update(df.index.tolist())
            as_of_date = max(all_dates)

        print(f"\n选股日期: {as_of_date.strftime('%Y-%m-%d')}")

        strategies_to_run = []
        if args.strategy in ['A', 'both']:
            strategies_to_run.append(('A_LowTurn_Mom12M', STRATEGIES['A_LowTurn_Mom12M']))
        if args.strategy in ['B', 'both']:
            strategies_to_run.append(('B_LowTurn_Mom12M_EP', STRATEGIES['B_LowTurn_Mom12M_EP']))

        for key, config in strategies_to_run:
            print(f"\n{'='*70}")
            print(f"策略: {config['name']}")
            print(f"描述: {config['description']}")
            print(f"选股范围: {config['universe']} | 持仓数: {config['top_n']} | 调仓: {config['rebal_months']}个月")
            print(f"因子: {', '.join(config['factors'].keys())}")
            print(f"{'='*70}")

            selected = select_stocks(all_data, val_data, hs300_const, zz500_const,
                                      as_of_date, config)

            if selected is None:
                print("  选股失败！")
                continue

            # 展示结果
            print(f"\n{'='*70}")
            print(f"  入选股票 (Top {config['top_n']}):")
            print(f"{'='*70}")
            print(f"  {'排名':>4s}  {'代码':<8s}  {'名称':<10s}  {'收盘价':>8s}  {'换手率':>8s}  {'12M动量':>8s}  {'波动率':>8s}  {'信号':>8s}")
            print(f"  {'-'*72}")

            for _, row in selected.iterrows():
                code = row['code']
                name = name_map.get(code, '---')
                vol_str = f"{row.get('vol_20d', 0)*100:.1f}%" if pd.notna(row.get('vol_20d')) else 'N/A'
                mom_str = f"{row.get('mom_12m', 0)*100:.1f}%" if pd.notna(row.get('mom_12m')) else 'N/A'
                turn_str = f"{row.get('turn_20d', 0):.2f}%" if pd.notna(row.get('turn_20d')) else 'N/A'

                print(f"  {int(row['rank']):>4d}  {code:<8s}  {name:<10s}  {row['close']:>8.2f}  {turn_str:>8s}  {mom_str:>8s}  {vol_str:>8s}  {row['composite']:>8.3f}")

            # 保存选股结果
            out_path = os.path.join(DATA_DIR, f'stock_picks_{key}_{as_of_date.strftime("%Y%m%d")}.csv')
            selected.to_csv(out_path, index=False)
            print(f"\n  选股结果已保存: {out_path}")

            # ============================================================
            # 建仓指南
            # ============================================================
            print(f"\n{'='*70}")
            print(f"  建仓指南 — {config['name']}")
            print(f"{'='*70}")
            print(f"""
  ■ 资金分配:
    - 等权分配: 每只股票 = 总资金 / {config['top_n']}
    - 例: 总资金100万, 每只 ≈ {100/config['top_n']:.1f}万
    - 建议: 实际下单时按收盘价算出股数, 取100股整数倍

  ■ 建仓时机:
    - 每月最后一个交易日(月末)收盘前30分钟执行
    - 或下月第一个交易日开盘集合竞价/开盘后10分钟内
    - 建议用"收盘价"+"限价单"下单, 避免追涨杀跌

  ■ 调仓规则:
    - 频率: 每月1次 (月末)
    - 对比新旧持仓: 卖出不在新名单中的, 买入新增的
    - 保持每只股票等权(约{100/config['top_n']:.1f}万)
    - 先卖后买, 确保资金充足

  ■ 交易成本预估:
    - 平均月度换手: ~30-50% (即每月换仓约10-15只)
    - 佣金: 万1.5-万3 (建议开低佣账户)
    - 印花税: 卖出千分之1 (仅卖出)
    - 总成本: 约 0.2-0.4% / 月 (已在回测中扣除)

  ■ 风控:
    - 策略最大历史回撤: {config.get('backtest_perf', {}).get('MaxDD', 'N/A')}%
    - 如遇极端行情(指数跌>15%), 可暂停调仓, 保持现有持仓
    - 不建议加杠杆
    - 单只股票涨停买不到: 跳过, 用剩余资金平均分配

  ■ 首次建仓:
    1. 运行本脚本获取最新选股名单
    2. 按等权金额计算每只股票的目标买入股数
    3. 在月末/下月初分批买入(建议2-3天完成)
    4. 下次调仓日为下月最后一个交易日
""")

    elapsed = time.time() - t_start
    print(f"\n总耗时: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
