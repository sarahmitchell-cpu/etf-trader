import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
#!/usr/bin/env python3
"""
Strategy E Expanded Research: CSI A50 + HK stocks pool comparison
Uses akshare (TX source for A-shares, EM source for HK) instead of Yahoo Finance.

Tests:
  Pool 1: Original 28 A-share leaders (baseline)
  Pool 2: CSI A50 (中证A50, 50 large-cap A-shares)
  Pool 3: CSI A50 + Major HK stocks (~75 stocks)
  Pool 4: All combined (~80+ stocks)

Uses same Strategy E logic: reversal + low vol, sector_max=1, Top4, 4-week rebal
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
# Stock Pools
# ============================================================

# Original 28 stocks (from stock_data_common.py)
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

# CSI A50 constituents (中证A50, index 930050) - additional stocks not in original
CSI_A50_EXTRA = {
    '000617': {'name': '中油资本', 'sector': '金融', 'market': 'sz'},
    '000725': {'name': '京东方A', 'sector': '面板', 'market': 'sz'},
    '000792': {'name': '盐湖股份', 'sector': '化工', 'market': 'sz'},
    '000938': {'name': '紫光股份', 'sector': 'IT服务', 'market': 'sz'},
    '000988': {'name': '华工科技', 'sector': '光通信', 'market': 'sz'},
    '002027': {'name': '分众传媒', 'sector': '传媒', 'market': 'sz'},
    '002475': {'name': '立讯精密', 'sector': '消费电子', 'market': 'sz'},
    '002625': {'name': '光启技术', 'sector': '军工', 'market': 'sz'},
    '300015': {'name': '爱尔眼科', 'sector': '眼科', 'market': 'sz'},
    '300124': {'name': '汇川技术', 'sector': '工控', 'market': 'sz'},
    '300308': {'name': '中际旭创', 'sector': '光模块', 'market': 'sz'},
    '300476': {'name': '胜宏科技', 'sector': 'PCB', 'market': 'sz'},
    '600019': {'name': '宝钢股份', 'sector': '钢铁', 'market': 'sh'},
    '600030': {'name': '中信证券', 'sector': '券商', 'market': 'sh'},
    '600031': {'name': '三一重工', 'sector': '工程机械', 'market': 'sh'},
    '600048': {'name': '保利发展', 'sector': '地产', 'market': 'sh'},
    '600111': {'name': '北方稀土', 'sector': '稀土', 'market': 'sh'},
    '600309': {'name': '万华化学', 'sector': '化工', 'market': 'sh'},
    '600406': {'name': '国电南瑞', 'sector': '电力设备', 'market': 'sh'},
    '600415': {'name': '小商品城', 'sector': '零售', 'market': 'sh'},
    '600436': {'name': '片仔癀', 'sector': '中药', 'market': 'sh'},
    '600660': {'name': '福耀玻璃', 'sector': '汽车零部件', 'market': 'sh'},
    '601600': {'name': '中国铝业', 'sector': '铝', 'market': 'sh'},
    '601766': {'name': '中国中车', 'sector': '轨交', 'market': 'sh'},
    '601816': {'name': '京沪高铁', 'sector': '铁路', 'market': 'sh'},
    '601888': {'name': '中国中免', 'sector': '免税', 'market': 'sh'},
    '603259': {'name': '药明康德', 'sector': 'CXO', 'market': 'sh'},
    '605499': {'name': '东鹏饮料', 'sector': '饮料', 'market': 'sh'},
    '688981': {'name': '中芯国际', 'sector': '半导体', 'market': 'sh'},
}

# Major Hong Kong stocks
HK_STOCKS = {
    '00700': {'name': '腾讯控股', 'sector': '互联网', 'market': 'hk'},
    '09988': {'name': '阿里巴巴', 'sector': '电商', 'market': 'hk'},
    '03690': {'name': '美团', 'sector': '本地生活', 'market': 'hk'},
    '09999': {'name': '网易', 'sector': '游戏', 'market': 'hk'},
    '09618': {'name': '京东集团', 'sector': '电商', 'market': 'hk'},
    '01810': {'name': '小米集团', 'sector': '消费电子', 'market': 'hk'},
    '02382': {'name': '舜宇光学', 'sector': '光学', 'market': 'hk'},
    '02269': {'name': '药明生物', 'sector': 'CXO', 'market': 'hk'},
    '01024': {'name': '快手', 'sector': '短视频', 'market': 'hk'},
    '00388': {'name': '港交所', 'sector': '交易所', 'market': 'hk'},
    '02020': {'name': '安踏体育', 'sector': '运动', 'market': 'hk'},
    '00175': {'name': '吉利汽车', 'sector': '汽车', 'market': 'hk'},
    '00005': {'name': '汇丰控股', 'sector': '银行', 'market': 'hk'},
    '01211': {'name': '比亚迪-HK', 'sector': '新能源车', 'market': 'hk'},
    '00968': {'name': '信义光能', 'sector': '光伏', 'market': 'hk'},
    '01347': {'name': '华虹半导体', 'sector': '半导体', 'market': 'hk'},
    '00285': {'name': '比亚迪电子', 'sector': 'EMS', 'market': 'hk'},
    '06098': {'name': '碧桂园服务', 'sector': '物管', 'market': 'hk'},
    '02318': {'name': '中国平安-HK', 'sector': '保险', 'market': 'hk'},
    '00941': {'name': '中国移动-HK', 'sector': '运营商', 'market': 'hk'},
}


# ============================================================
# Data fetching via akshare
# ============================================================

def fetch_a_share_weekly(code: str, market: str, start: str = '20210101', end: str = '20260328') -> Optional[pd.Series]:
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

        # Convert daily to weekly
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        weekly = df['close'].resample('W-FRI').last().dropna()

        cache_df = pd.DataFrame({'close': weekly})
        cache_df.to_csv(cache_path)
        return weekly
    except Exception as e:
        print(f"    [ERR] {code}: {e}")
        # Try expired cache
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=['date'], index_col='date')
            if len(df) > 20:
                return df['close']
        return None


def fetch_hk_weekly(code: str, start: str = '20210101', end: str = '20260328') -> Optional[pd.Series]:
    """Fetch HK stock weekly close via akshare"""
    import akshare as ak

    cache_path = os.path.join(DATA_DIR, f'se_hk_{code}_weekly.csv')
    cache_age = 3

    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= cache_age:
            df = pd.read_csv(cache_path, parse_dates=['date'], index_col='date')
            if len(df) > 20:
                return df['close']

    try:
        df = ak.stock_hk_hist(symbol=code, period='weekly', start_date=start, end_date=end, adjust='hfq')
        if df is None or len(df) == 0:
            return None

        df = df.rename(columns={'日期': 'date', '收盘': 'close'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        cache_df = pd.DataFrame({'close': df['close']})
        cache_df.to_csv(cache_path)
        return df['close']
    except Exception as e:
        print(f"    [ERR] HK-{code}: {e}")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=['date'], index_col='date')
            if len(df) > 20:
                return df['close']
        return None


def load_pool(pool: dict, label: str) -> Tuple[Optional[pd.DataFrame], dict]:
    """Load all stocks in a pool, return (price_df, pool_info_dict)"""
    print(f"\n{'='*60}")
    print(f"Loading: {label} ({len(pool)} stocks)")
    print(f"{'='*60}")

    prices = {}
    pool_info = {}
    failed = []

    for code, info in pool.items():
        market = info.get('market', 'sh')
        if market == 'hk':
            s = fetch_hk_weekly(code)
        else:
            s = fetch_a_share_weekly(code, market)

        if s is not None and len(s) > 20:
            key = f"{'HK' if market=='hk' else ''}{code}"
            prices[key] = s
            pool_info[key] = info
            # Print dot for progress
            sys.stdout.write('.')
            sys.stdout.flush()
        else:
            failed.append(f"{info['name']}({code})")
        time.sleep(0.5)

    print()
    if failed:
        print(f"  Failed ({len(failed)}): {', '.join(failed[:15])}")

    if len(prices) < 10:
        print(f"  ERROR: Only {len(prices)} loaded, need >= 10")
        return None, {}

    price_df = pd.DataFrame(prices).dropna(how='all')
    # Forward fill small gaps (max 2 weeks)
    price_df = price_df.ffill(limit=2)

    print(f"  Matrix: {price_df.shape[0]}w x {price_df.shape[1]} stocks")
    if len(price_df) > 0:
        print(f"  Range: {price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")
    return price_df, pool_info


# ============================================================
# Strategy E logic (identical to production)
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

        # Log trade
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


def param_sweep(price_df, pool_info, label, top_ns=[4, 5, 6, 8]):
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


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Strategy E Expanded Pool Research")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_results = []

    # Pool 1: Original 28
    print("\n>>> POOL 1: Original 28 A-share leaders (baseline)")
    df1, info1 = load_pool(ORIGINAL_POOL, "Original-28")
    if df1 is not None:
        r1 = param_sweep(df1, info1, "Orig28", [4])
        all_results.extend(r1)

    # Pool 2: CSI A50 (original overlaps + extras)
    print("\n>>> POOL 2: CSI A50 (50 large-cap A-shares)")
    csi_a50_full = {**ORIGINAL_POOL, **CSI_A50_EXTRA}
    # Remove duplicates by sector if any - keep unique codes
    df2, info2 = load_pool(csi_a50_full, "CSI-A50")
    if df2 is not None:
        r2 = param_sweep(df2, info2, "CSI-A50", [4, 5, 6, 8])
        all_results.extend(r2)

    # Pool 3: CSI A50 + HK
    print("\n>>> POOL 3: CSI A50 + Hong Kong stocks")
    combined = {**csi_a50_full, **HK_STOCKS}
    df3, info3 = load_pool(combined, "CSI-A50+HK")
    if df3 is not None:
        r3 = param_sweep(df3, info3, "A50+HK", [4, 5, 6, 8])
        all_results.extend(r3)

    # Pool 4: HK only (for comparison)
    print("\n>>> POOL 4: HK stocks only")
    df4, info4 = load_pool(HK_STOCKS, "HK-Only")
    if df4 is not None:
        r4 = param_sweep(df4, info4, "HK-Only", [4, 5])
        all_results.extend(r4)

    # ---- Summary ----
    print("\n\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    header = f"{'Config':<25} {'Pool':>4} {'Years':>5} {'CAGR%':>7} {'MDD%':>7} {'Sharpe':>7} {'Calmar':>7} {'WinR%':>6} {'Annual Returns'}"
    print(header)
    print("-" * 110)
    for r in all_results:
        if 'error' in r:
            print(f"{r['label']:<25} {'?':>4} {'ERR':>5} {r.get('error','')}")
            continue
        annual_str = ' '.join(f"{k}:{v:+.0f}%" for k, v in sorted(r.get('annual_returns', {}).items()))
        print(f"{r['label']:<25} {r['pool_size']:>4} {r['years']:>5.1f} {r['cagr_pct']:>7.1f} {r['mdd_pct']:>7.1f} "
              f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['win_rate_pct']:>6.1f} {annual_str}")

    # Save results
    out_path = os.path.join(DATA_DIR, 'strategy_e_expanded_research.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")

    return all_results


if __name__ == '__main__':
    main()
