#!/usr/bin/env python3
"""
Strategy E Fair Backtest: CSI A50 pool, starting from index creation date (2024-01-02)
to minimize survivorship bias. Uses current constituents but only backtests ~2.25 years.

Also attempts to reconstruct historical constituent changes from known adjustment dates.
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
from typing import List, Dict, Optional

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# CSI A50 Current Constituents (as of 2026-03-27)
# ============================================================
CSI_A50_POOL = {
    '000333': {'name': '美的集团', 'sector': '家电', 'market': 'sz'},
    '000617': {'name': '中油资本', 'sector': '金融', 'market': 'sz'},
    '000725': {'name': '京东方A', 'sector': '面板', 'market': 'sz'},
    '000792': {'name': '盐湖股份', 'sector': '化工', 'market': 'sz'},
    '000938': {'name': '紫光股份', 'sector': 'IT服务', 'market': 'sz'},
    '000988': {'name': '华工科技', 'sector': '光通信', 'market': 'sz'},
    '002027': {'name': '分众传媒', 'sector': '传媒', 'market': 'sz'},
    '002230': {'name': '科大讯飞', 'sector': 'AI', 'market': 'sz'},
    '002371': {'name': '北方华创', 'sector': '半导体', 'market': 'sz'},
    '002475': {'name': '立讯精密', 'sector': '消费电子', 'market': 'sz'},
    '002594': {'name': '比亚迪', 'sector': '新能源车', 'market': 'sz'},
    '002625': {'name': '光启技术', 'sector': '军工', 'market': 'sz'},
    '002714': {'name': '牧原股份', 'sector': '养殖', 'market': 'sz'},
    '300015': {'name': '爱尔眼科', 'sector': '眼科', 'market': 'sz'},
    '300124': {'name': '汇川技术', 'sector': '工控', 'market': 'sz'},
    '300308': {'name': '中际旭创', 'sector': '光模块', 'market': 'sz'},
    '300476': {'name': '胜宏科技', 'sector': 'PCB', 'market': 'sz'},
    '300750': {'name': '宁德时代', 'sector': '电池', 'market': 'sz'},
    '300760': {'name': '迈瑞医疗', 'sector': '医疗器械', 'market': 'sz'},
    '600019': {'name': '宝钢股份', 'sector': '钢铁', 'market': 'sh'},
    '600028': {'name': '中国石化', 'sector': '石油', 'market': 'sh'},
    '600030': {'name': '中信证券', 'sector': '券商', 'market': 'sh'},
    '600031': {'name': '三一重工', 'sector': '工程机械', 'market': 'sh'},
    '600036': {'name': '招商银行', 'sector': '银行', 'market': 'sh'},
    '600048': {'name': '保利发展', 'sector': '地产', 'market': 'sh'},
    '600111': {'name': '北方稀土', 'sector': '稀土', 'market': 'sh'},
    '600276': {'name': '恒瑞医药', 'sector': '创新药', 'market': 'sh'},
    '600309': {'name': '万华化学', 'sector': '化工', 'market': 'sh'},
    '600406': {'name': '国电南瑞', 'sector': '电力设备', 'market': 'sh'},
    '600415': {'name': '小商品城', 'sector': '零售', 'market': 'sh'},
    '600436': {'name': '片仔癀', 'sector': '中药', 'market': 'sh'},
    '600519': {'name': '贵州茅台', 'sector': '白酒', 'market': 'sh'},
    '600585': {'name': '海螺水泥', 'sector': '水泥', 'market': 'sh'},
    '600660': {'name': '福耀玻璃', 'sector': '汽车零部件', 'market': 'sh'},
    '600887': {'name': '伊利股份', 'sector': '食品', 'market': 'sh'},
    '600893': {'name': '航发动力', 'sector': '航发', 'market': 'sh'},
    '600900': {'name': '长江电力', 'sector': '水电', 'market': 'sh'},
    '600941': {'name': '中国移动', 'sector': '运营商', 'market': 'sh'},
    '601012': {'name': '隆基绿能', 'sector': '光伏', 'market': 'sh'},
    '601088': {'name': '中国神华', 'sector': '煤炭', 'market': 'sh'},
    '601318': {'name': '中国平安', 'sector': '保险', 'market': 'sh'},
    '601600': {'name': '中国铝业', 'sector': '铝', 'market': 'sh'},
    '601668': {'name': '中国建筑', 'sector': '建筑', 'market': 'sh'},
    '601766': {'name': '中国中车', 'sector': '轨交', 'market': 'sh'},
    '601816': {'name': '京沪高铁', 'sector': '铁路', 'market': 'sh'},
    '601888': {'name': '中国中免', 'sector': '免税', 'market': 'sh'},
    '601899': {'name': '紫金矿业', 'sector': '有色', 'market': 'sh'},
    '603259': {'name': '药明康德', 'sector': 'CXO', 'market': 'sh'},
    '605499': {'name': '东鹏饮料', 'sector': '饮料', 'market': 'sh'},
    '688981': {'name': '中芯国际', 'sector': '半导体', 'market': 'sh'},
}

# Known stocks that were REMOVED from A50 during adjustments (added later stocks that replaced them)
# Based on public information about A50 semi-annual adjustments
# 2024-06-16 adjustment: ~5 stocks changed
# 2024-12-16 adjustment: ~5 stocks changed  
# 2025-06-16 adjustment: ~5 stocks changed
# 2025-12-15 adjustment: ~5 stocks changed
# Stocks that were in original A50 but removed later (approximate, based on public sources):
REMOVED_STOCKS = {
    '600690': {'name': '海尔智家', 'sector': '家电', 'market': 'sh', 'removed': '2024-06-16'},
    '002415': {'name': '海康威视', 'sector': '安防', 'market': 'sz', 'removed': '2024-12-16'},
    '300274': {'name': '阳光电源', 'sector': '逆变器', 'market': 'sz', 'removed': '2025-06-16'},
    '000063': {'name': '中兴通讯', 'sector': '通信', 'market': 'sz', 'removed': '2024-06-16'},
    '601006': {'name': '大秦铁路', 'sector': '铁路', 'market': 'sh', 'removed': '2024-06-16'},
}

# Stocks added later (not in original 2024-01-02 list)
ADDED_LATER = {
    '300308': '2025-12-15',  # 中际旭创
    '002625': '2025-12-15',  # 光启技术
    '000988': '2025-12-15',  # 华工科技
    '300476': '2025-12-15',  # 胜宏科技
    '600111': '2025-06-16',  # 北方稀土
    '605499': '2025-06-16',  # 东鹏饮料
    '600048': '2025-06-16',  # 保利发展
    '000617': '2025-06-16',  # 中油资本
    '600415': '2024-12-16',  # 小商品城
    '002475': '2024-12-16',  # 立讯精密
    '600941': '2024-12-16',  # 中国移动
    '601600': '2024-12-16',  # 中国铝业
    '601816': '2024-12-16',  # 京沪高铁
}

def get_pool_at_date(date):
    """Get the approximate A50 constituent pool at a given date"""
    pool = dict(CSI_A50_POOL)
    # Remove stocks that were added after this date
    for code, add_date in ADDED_LATER.items():
        if date < pd.Timestamp(add_date):
            pool.pop(code, None)
    # Add back stocks that were removed after this date (they were in the pool before removal)
    for code, info in REMOVED_STOCKS.items():
        if date < pd.Timestamp(info['removed']):
            pool[code] = {k: v for k, v in info.items() if k != 'removed'}
    return pool

# ============================================================
# Data Fetching
# ============================================================
def fetch_weekly(code, market, start='20240101', end='20260328'):
    """Fetch weekly close prices using akshare TX source"""
    import akshare as ak
    try:
        if market == 'sh':
            symbol = f'sh{code}'
        else:
            symbol = f'sz{code}'
        df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start, end_date=end, adjust='hfq')
        if df is None or df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'] if 'date' in df.columns else df.iloc[:, 0])
        close_col = 'close' if 'close' in df.columns else df.columns[4]
        df = df.set_index('date')[close_col].astype(float)
        # Resample to weekly (Friday close)
        weekly = df.resample('W-FRI').last().dropna()
        weekly.name = code
        return weekly
    except Exception as e:
        print(f"  Error fetching {code}: {e}")
        return None

def load_all_data():
    """Load price data for all stocks (current + removed)"""
    all_stocks = dict(CSI_A50_POOL)
    all_stocks.update({k: {kk: vv for kk, vv in v.items() if kk != 'removed'} for k, v in REMOVED_STOCKS.items()})
    
    cache_file = os.path.join(DATA_DIR, 'a50_fair_backtest_prices.pkl')
    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < 3600:  # 1 hour cache
            print("Loading cached price data...")
            return pd.read_pickle(cache_file), all_stocks
    
    print(f"Fetching data for {len(all_stocks)} stocks...")
    series_list = []
    for i, (code, info) in enumerate(all_stocks.items()):
        print(f"  [{i+1}/{len(all_stocks)}] {code} {info['name']}...", end=' ')
        s = fetch_weekly(code, info['market'])
        if s is not None:
            series_list.append(s)
            print(f"OK ({len(s)} weeks)")
        else:
            print("FAILED")
        if i % 10 == 9:
            time.sleep(1)
    
    price_df = pd.concat(series_list, axis=1).sort_index()
    price_df.to_pickle(cache_file)
    print(f"Price data: {price_df.shape[0]} weeks x {price_df.shape[1]} stocks")
    return price_df, all_stocks

# ============================================================
# Strategy Logic (same as Strategy E: reversal + low vol)
# ============================================================
DEFAULT_PARAMS = {
    'value_lookback': 12,  # weeks for reversal
    'vol_lookback': 8,     # weeks for volatility
    'vol_weight': 0.2,     # 80% reversal + 20% low vol
    'top_n': 8,
    'rebal_weeks': 4,
    'sector_max': 1,
    'txn_cost_bps': 15,
    'max_drawdown_filter': -0.35,
}

def select_stocks(price_df, idx, params, pool_codes):
    """Select stocks using reversal + low vol composite"""
    vlb = params['value_lookback']
    vol_lb = params['vol_lookback']
    vw = params['vol_weight']
    top_n = params['top_n']
    
    if idx < max(vlb, vol_lb) + 2:
        return []
    
    available = [c for c in pool_codes if c in price_df.columns]
    window = price_df.iloc[max(0, idx-vlb-1):idx+1][available]
    
    # Reversal: negative return over lookback (buy losers)
    ret = window.iloc[-1] / window.iloc[0] - 1
    ret = ret.dropna()
    
    # Volatility: lower is better
    vol_window = price_df.iloc[max(0, idx-vol_lb):idx+1][available]
    vol = vol_window.pct_change().std()
    vol = vol.dropna()
    
    common = ret.index.intersection(vol.index)
    if len(common) < top_n:
        return list(common)
    
    ret = ret[common]
    vol = vol[common]
    
    # Composite: rank-based (lower rank = better)
    ret_rank = ret.rank()  # low return = low rank = good (reversal)
    vol_rank = vol.rank()  # low vol = low rank = good
    composite = (1 - vw) * ret_rank + vw * vol_rank
    
    # Sector constraint
    sector_max = params.get('sector_max')
    if sector_max:
        all_stocks_info = dict(CSI_A50_POOL)
        all_stocks_info.update({k: {kk: vv for kk, vv in v.items() if kk != 'removed'} for k, v in REMOVED_STOCKS.items()})
        sorted_stocks = composite.nsmallest(len(composite)).index.tolist()
        selected = []
        sector_count = defaultdict(int)
        for s in sorted_stocks:
            if len(selected) >= top_n:
                break
            sec = all_stocks_info.get(s, {}).get('sector', 'unknown')
            if sector_count[sec] < sector_max:
                selected.append(s)
                sector_count[sec] += 1
        return selected
    
    return list(composite.nsmallest(top_n).index)

def run_backtest(price_df, all_stocks_info, params, label="", use_dynamic_pool=False):
    """Run backtest with optional dynamic constituent pool"""
    txn_cost = params['txn_cost_bps'] / 10000
    warmup = max(params['value_lookback'], params['vol_lookback']) + 5
    rebal = params['rebal_weeks']
    dates = price_df.index
    
    # Start from 2024-01-05 (first Friday after index creation)
    start_date = pd.Timestamp('2024-01-05')
    start_idx = 0
    for i, d in enumerate(dates):
        if d >= start_date:
            start_idx = i
            break
    
    if start_idx < warmup:
        start_idx = warmup
    
    nav = [1.0]
    holdings = []
    trade_log = []
    
    for i in range(start_idx, len(dates)):
        week_num = i - start_idx
        
        if week_num % rebal == 0:
            # Determine pool at this date
            if use_dynamic_pool:
                pool = get_pool_at_date(dates[i])
                pool_codes = list(pool.keys())
            else:
                pool_codes = list(CSI_A50_POOL.keys())
            
            new_holdings = select_stocks(price_df, i, params, pool_codes)
            
            # Calculate turnover
            old_set = set(holdings)
            new_set = set(new_holdings)
            if holdings:
                turnover = len(old_set.symmetric_difference(new_set)) / max(len(old_set), len(new_set), 1)
            else:
                turnover = 0
            
            holdings = new_holdings
            trade_log.append({
                'date': dates[i].strftime('%Y-%m-%d'),
                'holdings': [f"{c}({all_stocks_info.get(c, {}).get('name', '?')})" for c in holdings],
                'turnover': round(turnover, 2),
                'pool_size': len(pool_codes),
            })
        
        if not holdings:
            nav.append(nav[-1])
            continue
        
        # Equal weight portfolio return
        valid = [c for c in holdings if c in price_df.columns and i > 0]
        if valid:
            rets = []
            for c in valid:
                if pd.notna(price_df.iloc[i][c]) and pd.notna(price_df.iloc[i-1][c]) and price_df.iloc[i-1][c] > 0:
                    rets.append(price_df.iloc[i][c] / price_df.iloc[i-1][c] - 1)
            if rets:
                port_ret = np.mean(rets)
                # Deduct transaction cost on rebalance weeks
                if (i - start_idx) % rebal == 0 and trade_log and trade_log[-1].get('turnover', 0) > 0:
                    port_ret -= trade_log[-1]['turnover'] * txn_cost
                nav.append(nav[-1] * (1 + port_ret))
            else:
                nav.append(nav[-1])
        else:
            nav.append(nav[-1])
    
    nav_s = pd.Series(nav[1:], index=dates[start_idx:])
    
    if len(nav_s) < 10:
        return {'label': label, 'error': 'Too few data points'}
    
    # Metrics
    total_ret = nav_s.iloc[-1] / nav_s.iloc[0] - 1
    years = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
    cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
    
    drawdown = nav_s / nav_s.cummax() - 1
    mdd = drawdown.min()
    
    weekly_rets = nav_s.pct_change().dropna()
    sharpe = weekly_rets.mean() / weekly_rets.std() * np.sqrt(52) if weekly_rets.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    
    win_rate = (weekly_rets > 0).sum() / len(weekly_rets) * 100
    
    # Annual returns
    annual_returns = {}
    for year in nav_s.index.year.unique():
        yr_data = nav_s[nav_s.index.year == year]
        if len(yr_data) > 1:
            yr_ret = yr_data.iloc[-1] / yr_data.iloc[0] - 1
            annual_returns[str(year)] = round(yr_ret * 100, 1)
    
    return {
        'label': label,
        'period': f"{nav_s.index[0].strftime('%Y-%m-%d')} ~ {nav_s.index[-1].strftime('%Y-%m-%d')}",
        'weeks': len(nav_s),
        'cagr_pct': round(cagr * 100, 1),
        'total_return_pct': round(total_ret * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round(win_rate, 1),
        'annual_returns': annual_returns,
        'num_rebalances': len(trade_log),
        'last_trades': trade_log[-3:] if trade_log else [],
    }

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Strategy E Fair Backtest: CSI A50 (from 2024-01-02)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    price_df, all_stocks_info = load_all_data()
    
    all_results = []
    
    # Test 1: Static pool (current constituents), various top_n
    print("\n>>> TEST 1: Static A50 pool (current constituents, 2024~now)")
    for tn in [4, 6, 8, 10]:
        p = DEFAULT_PARAMS.copy()
        p['top_n'] = tn
        r = run_backtest(price_df, all_stocks_info, p, f"Static_Top{tn}", use_dynamic_pool=False)
        all_results.append(r)
        if 'error' not in r:
            print(f"  Static Top{tn}: CAGR={r['cagr_pct']:.1f}% MDD={r['mdd_pct']:.1f}% Sharpe={r['sharpe']:.3f} Annual={r['annual_returns']}")
        else:
            print(f"  Static Top{tn}: {r['error']}")
    
    # Test 2: Dynamic pool (approximate historical constituents)
    print("\n>>> TEST 2: Dynamic A50 pool (time-varying constituents, 2024~now)")
    for tn in [4, 6, 8, 10]:
        p = DEFAULT_PARAMS.copy()
        p['top_n'] = tn
        r = run_backtest(price_df, all_stocks_info, p, f"Dynamic_Top{tn}", use_dynamic_pool=True)
        all_results.append(r)
        if 'error' not in r:
            print(f"  Dynamic Top{tn}: CAGR={r['cagr_pct']:.1f}% MDD={r['mdd_pct']:.1f}% Sharpe={r['sharpe']:.3f} Annual={r['annual_returns']}")
        else:
            print(f"  Dynamic Top{tn}: {r['error']}")
    
    # Test 3: Different rebalance frequencies
    print("\n>>> TEST 3: Rebalance frequency test (Dynamic Top8)")
    for rebal_w in [2, 4, 6, 8]:
        p = DEFAULT_PARAMS.copy()
        p['top_n'] = 8
        p['rebal_weeks'] = rebal_w
        r = run_backtest(price_df, all_stocks_info, p, f"Dynamic_Top8_R{rebal_w}w", use_dynamic_pool=True)
        all_results.append(r)
        if 'error' not in r:
            print(f"  R{rebal_w}w: CAGR={r['cagr_pct']:.1f}% MDD={r['mdd_pct']:.1f}% Sharpe={r['sharpe']:.3f}")
    
    # Summary
    print("\n\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Config':<25} {'Period':<30} {'CAGR%':>7} {'MDD%':>7} {'Sharpe':>7} {'Calmar':>7} {'WinRate':>8}")
    print("-" * 100)
    for r in all_results:
        if 'error' not in r:
            print(f"{r['label']:<25} {r['period']:<30} {r['cagr_pct']:>7.1f} {r['mdd_pct']:>7.1f} {r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['win_rate_pct']:>7.1f}%")
    
    # Save results
    save_path = os.path.join(DATA_DIR, 'strategy_e_a50_fair_backtest.json')
    with open(save_path, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {save_path}")
    
    # Best result
    valid = [r for r in all_results if 'error' not in r]
    if valid:
        best = max(valid, key=lambda x: x['sharpe'])
        print(f"\n🏆 Best by Sharpe: {best['label']} (CAGR={best['cagr_pct']}%, Sharpe={best['sharpe']}, MDD={best['mdd_pct']}%)")
        print(f"   Annual returns: {best['annual_returns']}")
        if best.get('last_trades'):
            print(f"   Last rebalance: {best['last_trades'][-1]}")

if __name__ == '__main__':
    main()
