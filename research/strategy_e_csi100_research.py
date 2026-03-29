#!/usr/bin/env python3
"""
Strategy E Research: CSI 100 (中证100) stock pool backtest
Compare with original 28 stocks and CSI A50 results.

Uses akshare to get CSI 100 constituents dynamically, then runs Strategy E
(reversal + low vol, sector_max=1) with various top_n configs.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import time
import warnings
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# Get CSI 100 constituents from akshare
# ============================================================

def get_csi100_constituents():
    """Get CSI 100 index constituents via akshare"""
    import akshare as ak

    cache_path = os.path.join(DATA_DIR, 'csi100_constituents.json')
    cache_age = 30  # days - constituents don't change often

    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= cache_age:
            with open(cache_path, 'r') as f:
                return json.load(f)

    try:
        # 中证100 index code: 000903
        df = ak.index_stock_cons(symbol="000903")
        if df is None or len(df) == 0:
            print("Failed to get CSI 100 constituents from akshare")
            # Try alternative
            df = ak.index_stock_cons_csindex(symbol="000903")

        constituents = {}
        for _, row in df.iterrows():
            code = str(row.get('品种代码', row.get('成分券代码', ''))).zfill(6)
            name = str(row.get('品种名称', row.get('成分券名称', '')))

            # Determine market
            if code.startswith(('60', '68')):
                market = 'sh'
            elif code.startswith(('00', '30')):
                market = 'sz'
            else:
                continue

            # Simple sector assignment based on code/name
            sector = assign_sector(code, name)
            constituents[code] = {'name': name, 'sector': sector, 'market': market}

        print(f"Got {len(constituents)} CSI 100 constituents")

        with open(cache_path, 'w') as f:
            json.dump(constituents, f, ensure_ascii=False, indent=2)

        return constituents

    except Exception as e:
        print(f"Error getting CSI 100 constituents: {e}")
        # Fallback: try loading cache even if expired
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None


def assign_sector(code, name):
    """Simple sector assignment based on stock name keywords"""
    sector_map = {
        '银行': '银行', '平安': '保险', '人寿': '保险', '太保': '保险',
        '证券': '券商', '中信': '券商', '茅台': '白酒', '五粮液': '白酒',
        '洋河': '白酒', '泸州': '白酒', '伊利': '食品', '海天': '食品',
        '蒙牛': '食品', '东鹏': '饮料', '片仔癀': '中药', '云南白药': '中药',
        '恒瑞': '创新药', '迈瑞': '医疗器械', '药明': 'CXO', '爱尔': '眼科',
        '美的': '家电', '海尔': '家电', '格力': '家电', '宁德': '电池',
        '比亚迪': '新能源车', '长城汽车': '汽车', '上汽': '汽车',
        '隆基': '光伏', '阳光电源': '逆变器', '海康': '安防', '大华': '安防',
        '中兴': '通信设备', '中国移动': '运营商', '中国电信': '运营商',
        '中国联通': '运营商', '科大讯飞': 'AI', '紫金': '有色', '北方稀土': '稀土',
        '中国神华': '煤炭', '中国石化': '石油', '中国石油': '石油',
        '长江电力': '水电', '三峡': '水电', '中国建筑': '建筑',
        '海螺': '水泥', '万华': '化工', '宝钢': '钢铁', '中国铝业': '铝',
        '京东方': '面板', '立讯': '消费电子', '工业富联': '消费电子',
        '中芯': '半导体', '北方华创': '半导体设备', '中际旭创': '光模块',
        '汇川': '工控', '国电南瑞': '电力设备', '三一': '工程机械',
        '中国中车': '轨交', '招商蛇口': '地产', '保利': '地产', '万科': '地产',
        '中国中免': '免税', '分众': '传媒', '牧原': '养殖',
        '中国人保': '保险', '新华保险': '保险', '中国太保': '保险',
        '工商银行': '银行', '建设银行': '银行', '农业银行': '银行',
        '中国银行': '银行', '交通银行': '银行', '招商银行': '银行',
        '兴业银行': '银行', '浦发银行': '银行', '民生银行': '银行',
        '光大银行': '银行', '中信银行': '银行',
        '航发': '航发', '中航': '军工', '光启': '军工',
        '福耀': '汽车零部件', '潍柴': '柴油机', '中国交建': '建筑',
        '中国电建': '建筑', '中国中铁': '建筑', '中国铁建': '建筑',
        '大秦铁路': '铁路', '京沪高铁': '铁路',
        '腾讯': '互联网', '阿里': '电商', '小商品城': '零售',
    }

    for keyword, sector in sector_map.items():
        if keyword in name:
            return sector

    # Fallback by code prefix
    if code.startswith('60'):
        return 'SH其他'
    elif code.startswith('00'):
        return 'SZ其他'
    elif code.startswith('30'):
        return '创业板'
    elif code.startswith('68'):
        return '科创板'
    return '其他'


# ============================================================
# Original 28 stocks (baseline)
# ============================================================

ORIGINAL_POOL = {
    '600519': {'name': '贵州茅台', 'sector': '白酒', 'market': 'sh'},
    '000858': {'name': '五粮液', 'sector': '白酒', 'market': 'sz'},
    '600887': {'name': '伊利股份', 'sector': '食品', 'market': 'sh'},
    '002714': {'name': '牧原股份', 'sector': '养殖', 'market': 'sh'},
    '601318': {'name': '中国平安', 'sector': '保险', 'market': 'sh'},
    '600036': {'name': '招商银行', 'sector': '银行', 'market': 'sh'},
    '002415': {'name': '海康威视', 'sector': '安防', 'market': 'sz'},
    '300750': {'name': '宁德时代', 'sector': '电池', 'market': 'sz'},
    '600276': {'name': '恒瑞医药', 'sector': '创新药', 'market': 'sh'},
    '300760': {'name': '迈瑞医疗', 'sector': '医疗器械', 'market': 'sz'},
    '601012': {'name': '隆基绿能', 'sector': '光伏', 'market': 'sh'},
    '300274': {'name': '阳光电源', 'sector': '逆变器', 'market': 'sz'},
    '000333': {'name': '美的集团', 'sector': '家电', 'market': 'sz'},
    '600690': {'name': '海尔智家', 'sector': '家电', 'market': 'sh'},
    '002594': {'name': '比亚迪', 'sector': '新能源车', 'market': 'sz'},
    '600893': {'name': '航发动力', 'sector': '航发', 'market': 'sh'},
    '601668': {'name': '中国建筑', 'sector': '建筑', 'market': 'sh'},
    '600585': {'name': '海螺水泥', 'sector': '水泥', 'market': 'sh'},
    '601899': {'name': '紫金矿业', 'sector': '有色', 'market': 'sh'},
    '600028': {'name': '中国石化', 'sector': '石油', 'market': 'sh'},
    '002371': {'name': '北方华创', 'sector': '半导体设备', 'market': 'sz'},
    '000063': {'name': '中兴通讯', 'sector': '通信设备', 'market': 'sz'},
    '600941': {'name': '中国移动', 'sector': '运营商', 'market': 'sh'},
    '002230': {'name': '科大讯飞', 'sector': 'AI', 'market': 'sz'},
    '600900': {'name': '长江电力', 'sector': '水电', 'market': 'sh'},
    '601088': {'name': '中国神华', 'sector': '煤炭', 'market': 'sh'},
    '601006': {'name': '大秦铁路', 'sector': '铁路', 'market': 'sh'},
    '001979': {'name': '招商蛇口', 'sector': '地产', 'market': 'sz'},
}


# ============================================================
# Data fetching via akshare
# ============================================================

def fetch_a_share_weekly(code, market, start='20210101', end='20260328'):
    """Fetch A-share weekly close via akshare TX source, with cache"""
    import akshare as ak

    cache_path = os.path.join(DATA_DIR, f'se_{code}_weekly.csv')
    cache_age = 3  # days

    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= cache_age:
            df = pd.read_csv(cache_path, parse_dates=['date'], index_col='date')
            if len(df) > 20:
                return df['close']

    try:
        symbol = f"{market}{code}"
        df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start, end_date=end, adjust='hfq')
        if df is None or len(df) == 0:
            return None

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        weekly = df['close'].resample('W-FRI').last().dropna()

        cache_df = pd.DataFrame({'close': weekly})
        cache_df.to_csv(cache_path)
        return weekly
    except Exception as e:
        print(f"    [ERR] {code}: {e}")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=['date'], index_col='date')
            if len(df) > 20:
                return df['close']
        return None


def load_pool(pool, label):
    """Load all stocks in a pool, return (price_df, pool_info_dict)"""
    print(f"\n{'='*60}")
    print(f"Loading: {label} ({len(pool)} stocks)")
    print(f"{'='*60}")

    prices = {}
    pool_info = {}
    failed = []

    for code, info in pool.items():
        market = info.get('market', 'sh')
        s = fetch_a_share_weekly(code, market)

        if s is not None and len(s) > 20:
            prices[code] = s
            pool_info[code] = info
            sys.stdout.write('.')
            sys.stdout.flush()
        else:
            failed.append(f"{info['name']}({code})")
        time.sleep(0.3)  # Rate limit

    print()
    if failed:
        print(f"  Failed ({len(failed)}): {', '.join(failed[:20])}")

    if len(prices) < 10:
        print(f"  ERROR: Only {len(prices)} loaded, need >= 10")
        return None, {}

    price_df = pd.DataFrame(prices).dropna(how='all')
    price_df = price_df.ffill(limit=2)

    print(f"  Matrix: {price_df.shape[0]}w x {price_df.shape[1]} stocks")
    if len(price_df) > 0:
        print(f"  Range: {price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")
    return price_df, pool_info


# ============================================================
# Strategy E logic
# ============================================================

DEFAULT_PARAMS = {
    'value_lookback': 10,
    'vol_lookback': 12,
    'vol_weight': 0.2,
    'top_n': 4,
    'rebal_freq': 4,
    'max_drawdown_filter': -0.20,
    'txn_cost_bps': 8,
    'sector_max': 1,
}


def select_stocks(price_df, idx, params, pool_info):
    vlb = params['value_lookback']
    vol_lb = params['vol_lookback']
    vw = params['vol_weight']
    top_n = params['top_n']
    ddf = params['max_drawdown_filter']

    if idx < max(vlb, vol_lb) + 2:
        return []

    returns = price_df.pct_change(fill_method=None)
    scores = {}

    for col in price_df.columns:
        if (idx - vlb < 0 or
            pd.isna(price_df[col].iloc[idx]) or
            pd.isna(price_df[col].iloc[idx - vlb]) or
            price_df[col].iloc[idx - vlb] <= 0):
            continue

        mom_10w = float(price_df[col].iloc[idx] / price_df[col].iloc[idx - vlb] - 1)
        ret_slice = returns[col].iloc[max(0, idx-vol_lb):idx+1].dropna()
        if len(ret_slice) < 4:
            continue
        vol = float(ret_slice.std())

        if idx >= 4:
            mom_4w = float(price_df[col].iloc[idx] / price_df[col].iloc[idx-4] - 1)
            if mom_4w < ddf:
                continue

        scores[col] = {'value_score': -mom_10w, 'quality_score': -vol}

    if len(scores) < top_n:
        return []

    tickers = list(scores.keys())
    value_ranks = pd.Series({t: scores[t]['value_score'] for t in tickers}).rank(ascending=False)
    quality_ranks = pd.Series({t: scores[t]['quality_score'] for t in tickers}).rank(ascending=False)
    composite = (1 - vw) * value_ranks + vw * quality_ranks

    sector_max = params.get('sector_max')
    if sector_max:
        sorted_tickers = list(composite.sort_values().index)
        selected = []
        sector_count = defaultdict(int)
        for t in sorted_tickers:
            if len(selected) >= top_n:
                break
            sec = pool_info.get(t, {}).get('sector', '?')
            if sector_count[sec] < sector_max:
                selected.append(t)
                sector_count[sec] += 1
    else:
        selected = list(composite.nsmallest(top_n).index)

    return selected


def run_backtest(price_df, pool_info, params=None, label=""):
    if params is None:
        params = DEFAULT_PARAMS.copy()

    txn_cost = params['txn_cost_bps'] / 10000
    returns = price_df.pct_change(fill_method=None)
    warmup = max(params['value_lookback'], params['vol_lookback']) + 5

    nav = [1.0]
    dates = []
    prev_holdings = set()
    weekly_rets = []
    trade_log = []

    i = warmup
    while i < len(price_df) - 1:
        selected = select_stocks(price_df, i, params, pool_info)
        if not selected:
            i += 1
            continue

        selected_set = set(selected)
        new_buys = selected_set - prev_holdings
        sold = prev_holdings - selected_set
        turnover = (len(new_buys) + len(sold)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost

        names = [pool_info.get(s, {}).get('name', s) for s in selected]
        trade_log.append({
            'date': price_df.index[i].strftime('%Y-%m-%d'),
            'stocks': names,
        })

        hold_end = min(i + params['rebal_freq'], len(price_df) - 1)
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

    if not dates or len(dates) < 10:
        return {'label': label, 'error': 'Insufficient trades'}

    nav_s = pd.Series(nav[1:], index=dates)
    years = (dates[-1] - dates[0]).days / 365.25
    if years <= 0.5:
        return {'label': label, 'error': 'Too short'}

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
        'label': label,
        'pool_size': price_df.shape[1],
        'period': f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}",
        'years': round(years, 1),
        'cagr_pct': round(cagr * 100, 1),
        'total_return_pct': round((nav_s.iloc[-1] - 1) * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round(win_rate, 1),
        'annual_returns': annual_returns,
        'last_trades': trade_log[-3:] if trade_log else [],
    }


def param_sweep(price_df, pool_info, label, top_ns=[4, 5, 6, 8, 10]):
    results = []
    for tn in top_ns:
        p = DEFAULT_PARAMS.copy()
        p['top_n'] = tn
        r = run_backtest(price_df, pool_info, p, f"{label}_Top{tn}")
        results.append(r)
        if 'error' not in r:
            print(f"  Top{tn}: CAGR={r['cagr_pct']:.1f}% MDD={r['mdd_pct']:.1f}% "
                  f"Sharpe={r['sharpe']:.3f} Calmar={r['calmar']:.3f}")
        else:
            print(f"  Top{tn}: {r['error']}")
    return results


# Also test different rebalance frequencies
def rebal_sweep(price_df, pool_info, label, top_n=8, freqs=[2, 4, 6, 8]):
    results = []
    for freq in freqs:
        p = DEFAULT_PARAMS.copy()
        p['top_n'] = top_n
        p['rebal_freq'] = freq
        r = run_backtest(price_df, pool_info, p, f"{label}_Top{top_n}_R{freq}w")
        results.append(r)
        if 'error' not in r:
            print(f"  Top{top_n} R{freq}w: CAGR={r['cagr_pct']:.1f}% MDD={r['mdd_pct']:.1f}% "
                  f"Sharpe={r['sharpe']:.3f} Calmar={r['calmar']:.3f}")
        else:
            print(f"  Top{top_n} R{freq}w: {r['error']}")
    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Strategy E CSI 100 Research")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_results = []

    # Pool 1: Original 28 (baseline)
    print("\n>>> POOL 1: Original 28 A-share leaders (baseline)")
    df1, info1 = load_pool(ORIGINAL_POOL, "Original-28")
    if df1 is not None:
        r1 = param_sweep(df1, info1, "Orig28", [4, 8])
        all_results.extend(r1)

    # Pool 2: CSI 100
    print("\n>>> POOL 2: CSI 100 (中证100, 100 large-cap A-shares)")
    csi100 = get_csi100_constituents()
    if csi100:
        print(f"  Constituents loaded: {len(csi100)} stocks")
        # Print some names for verification
        names = [v['name'] for v in list(csi100.values())[:10]]
        print(f"  Sample: {', '.join(names)} ...")

        df2, info2 = load_pool(csi100, "CSI-100")
        if df2 is not None:
            print("\n  --- Top N sweep ---")
            r2 = param_sweep(df2, info2, "CSI100", [4, 5, 6, 8, 10])
            all_results.extend(r2)

            print("\n  --- Rebalance freq sweep (Top8) ---")
            r2b = rebal_sweep(df2, info2, "CSI100", top_n=8, freqs=[2, 4, 6, 8])
            all_results.extend(r2b)

            print("\n  --- Rebalance freq sweep (Top10) ---")
            r2c = rebal_sweep(df2, info2, "CSI100", top_n=10, freqs=[2, 4, 6, 8])
            all_results.extend(r2c)
    else:
        print("  ERROR: Could not get CSI 100 constituents")

    # ---- Summary ----
    print("\n\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    header = f"{'Config':<25} {'Pool':>4} {'Years':>5} {'CAGR%':>7} {'MDD%':>7} {'Sharpe':>7} {'Calmar':>7} {'WinR%':>6} {'Annual Returns'}"
    print(header)
    print("-" * 120)
    for r in all_results:
        if 'error' in r:
            print(f"{r['label']:<25} {'?':>4} {'ERR':>5} {r.get('error','')}")
            continue
        annual_str = ' '.join(f"{k}:{v:+.0f}%" for k, v in sorted(r.get('annual_returns', {}).items()))
        print(f"{r['label']:<25} {r['pool_size']:>4} {r['years']:>5.1f} {r['cagr_pct']:>7.1f} {r['mdd_pct']:>7.1f} "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['win_rate_pct']:>6.1f} {annual_str}")

    # Save results
    out_path = os.path.join(DATA_DIR, 'strategy_e_csi100_research.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")

    return all_results


if __name__ == '__main__':
    main()
