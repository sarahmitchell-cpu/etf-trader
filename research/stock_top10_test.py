#!/usr/bin/env python3
"""Quick test: LowTurn+Mom12M CSI800 Top10 vs Top30 vs Top20"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'strategies'))
from strategy_m_stock_factor import *

def main():
    t_start = time.time()

    print("加载数据...")
    hs300_const = get_historical_constituents('hs300', 2016, 2026)
    zz500_const = get_historical_constituents('zz500', 2016, 2026)

    all_codes_ever = set()
    for codes in hs300_const.values():
        all_codes_ever |= codes
    for codes in zz500_const.values():
        all_codes_ever |= codes

    all_data = {}
    for code in sorted(all_codes_ever):
        df = fetch_stock_akshare(code)
        if df is not None and len(df) > 60:
            all_data[code] = df
    print(f"加载: {len(all_data)} stocks")

    val_data = {}
    for code in all_data.keys():
        for cache_dir in [PRICE_CACHE_DIR, CACHE_DIR]:
            cache_file = os.path.join(cache_dir, f'{code}_val.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    val_data[code] = pickle.load(f)
                break

    month_ends = get_month_ends(all_data)
    print(f"月末日期: {len(month_ends)}")

    # Test different top_n values
    for top_n in [10, 15, 20, 30]:
        config = {
            'name': f'低换手+12M动量 Top{top_n}',
            'factors': {
                'turn_20d': (1.0, True),
                'mom_12m':  (1.0, False),
            },
            'universe': 'CSI800',
            'top_n': top_n,
            'rebal_months': 1,
        }
        r = run_backtest(all_data, val_data, month_ends,
                         hs300_const, zz500_const, f'top{top_n}', config)
        if r:
            print(f"\n>>> Top{top_n}: CAGR={r['CAGR']:.1f}%, Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']:.1f}%, Calmar={r['Calmar']:.3f}, WinRate={r['WinRate']:.0f}%, Turnover={r['AvgTurnover']:.0f}%")
            print(f"    年度: {r['yearly_returns']}")

    print(f"\n总耗时: {time.time()-t_start:.0f}s")

if __name__ == '__main__':
    main()
