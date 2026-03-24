#!/usr/bin/env python3
"""
CSI500 Sector Rotation: Rolling 12-month stability analysis
- Replays the best SR strategy (m12, top3 sectors, 2 stocks/sector, 4-week rebal)
- Computes rolling 12-month (52-week) returns
- Reports: worst/best 12m, win rate, percentiles, all rolling windows
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import time
import akshare as ak
from datetime import datetime
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

CSI500_POOL = {
    'sz002064': {'name': '华峰化学', 'code': '002064', 'sector': '化工'},
    'sh600426': {'name': '华鲁恒升', 'code': '600426', 'sector': '化工'},
    'sz002601': {'name': '龙蟒佰利', 'code': '002601', 'sector': '化工'},
    'sh601100': {'name': '恒立液压', 'code': '601100', 'sector': '机械'},
    'sz300124': {'name': '汇川技术', 'code': '300124', 'sector': '机械'},
    'sz002008': {'name': '大族激光', 'code': '002008', 'sector': '机械'},
    'sz002049': {'name': '紫光国微', 'code': '002049', 'sector': '电子'},
    'sz300661': {'name': '圣邦股份', 'code': '300661', 'sector': '电子'},
    'sh603501': {'name': '韦尔股份', 'code': '603501', 'sector': '电子'},
    'sz002241': {'name': '歌尔股份', 'code': '002241', 'sector': '电子'},
    'sz300122': {'name': '智飞生物', 'code': '300122', 'sector': '医药'},
    'sh603259': {'name': '药明康德', 'code': '603259', 'sector': '医药'},
    'sz000661': {'name': '长春高新', 'code': '000661', 'sector': '医药'},
    'sh603288': {'name': '海天味业', 'code': '603288', 'sector': '食品'},
    'sz002568': {'name': '百润股份', 'code': '002568', 'sector': '食品'},
    'sz300498': {'name': '温氏股份', 'code': '300498', 'sector': '养殖'},
    'sz002459': {'name': '晶澳科技', 'code': '002459', 'sector': '光伏'},
    'sz300014': {'name': '亿纬锂能', 'code': '300014', 'sector': '锂电'},
    'sz002812': {'name': '恩捷股份', 'code': '002812', 'sector': '锂电'},
    'sz002179': {'name': '中航光电', 'code': '002179', 'sector': '军工'},
    'sh600760': {'name': '中航沈飞', 'code': '600760', 'sector': '军工'},
    'sz300033': {'name': '同花顺', 'code': '300033', 'sector': '软件'},
    'sz002236': {'name': '大华股份', 'code': '002236', 'sector': '安防'},
    'sz002460': {'name': '赣锋锂业', 'code': '002460', 'sector': '锂矿'},
    'sh600362': {'name': '江西铜业', 'code': '600362', 'sector': '有色'},
    'sz002466': {'name': '天齐锂业', 'code': '002466', 'sector': '锂矿'},
    'sz002920': {'name': '德赛西威', 'code': '002920', 'sector': '汽车电子'},
    'sh603596': {'name': '伯特利', 'code': '603596', 'sector': '汽车零部件'},
    'sh601689': {'name': '拓普集团', 'code': '601689', 'sector': '汽车零部件'},
    'sz002271': {'name': '东方雨虹', 'code': '002271', 'sector': '建材'},
    'sh601155': {'name': '新城控股', 'code': '601155', 'sector': '地产'},
    'sh600999': {'name': '招商证券', 'code': '600999', 'sector': '券商'},
    'sh601688': {'name': '华泰证券', 'code': '601688', 'sector': '券商'},
    'sz002602': {'name': '世纪华通', 'code': '002602', 'sector': '游戏'},
    'sz300413': {'name': '芒果超媒', 'code': '300413', 'sector': '传媒'},
    'sh600886': {'name': '国投电力', 'code': '600886', 'sector': '电力'},
    'sh601006': {'name': '大秦铁路', 'code': '601006', 'sector': '铁路'},
    'sh600115': {'name': '中国东航', 'code': '600115', 'sector': '航空'},
    'sh603868': {'name': '飞科电器', 'code': '603868', 'sector': '小家电'},
    'sz002572': {'name': '索菲亚', 'code': '002572', 'sector': '家居'},
    'sz300782': {'name': '卓胜微', 'code': '300782', 'sector': '半导体'},
    'sz300628': {'name': '亿联网络', 'code': '300628', 'sector': '通信'},
    'sh600348': {'name': '阳泉煤业', 'code': '600348', 'sector': '煤炭'},
    'sz000932': {'name': '华菱钢铁', 'code': '000932', 'sector': '钢铁'},
    'sz002311': {'name': '海大集团', 'code': '002311', 'sector': '饲料'},
    'sz000876': {'name': '新希望', 'code': '000876', 'sector': '养殖'},
    'sz002475': {'name': '立讯精密', 'code': '002475', 'sector': '消费电子'},
    'sh601127': {'name': '赛力斯', 'code': '601127', 'sector': '新能源车'},
    'sz002385': {'name': '大北农', 'code': '002385', 'sector': '饲料'},
    'sh600309': {'name': '万华化学', 'code': '600309', 'sector': '化工'},
}


def fetch_stock_daily(symbol, name, retries=3):
    cache_path = os.path.join(DATA_DIR, f'csi500_{symbol}_weekly.csv')
    if os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age <= 3:
            df = pd.read_csv(cache_path, parse_dates=['date'], index_col='date')
            if len(df) > 50:
                print(f"  {name}: cache ({len(df)}w)")
                return df['close']
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_daily(symbol=symbol, adjust="hfq")
            if df is None or len(df) < 200:
                return None
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            df = df[df.index >= '2021-01-01']
            weekly = df['close'].resample('W-FRI').last().dropna()
            if len(weekly) < 50:
                return None
            out = pd.DataFrame({'close': weekly})
            out.index.name = 'date'
            out.to_csv(cache_path)
            print(f"  {name}: OK ({len(weekly)}w)")
            return weekly
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                print(f"  {name}: ERROR {e}")
                return None
    return None


def load_all_data():
    print(f"Loading {len(CSI500_POOL)} stocks via akshare (daily->weekly)...")
    prices = {}
    for symbol, info in CSI500_POOL.items():
        s = fetch_stock_daily(symbol, info['name'])
        if s is not None:
            prices[symbol] = s
        time.sleep(0.3)
    print(f"\nLoaded {len(prices)}/{len(CSI500_POOL)} stocks")
    price_df = pd.DataFrame(prices).dropna(how='all').ffill(limit=2)
    print(f"  Matrix: {price_df.shape[0]}w x {price_df.shape[1]} stocks")
    print(f"  Range: {price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")
    return price_df


def get_momentum(price_df, idx, lookback):
    start = idx - lookback
    if start < 0:
        return {}
    mom = {}
    for col in price_df.columns:
        p0 = price_df[col].iloc[start]
        p1 = price_df[col].iloc[idx]
        if pd.notna(p0) and pd.notna(p1) and p0 > 0:
            mom[col] = p1 / p0 - 1
    return mom


def run_sector_rotation(price_df, mom_lookback=12, skip=0, top_sectors=3,
                        stocks_per_sector=2, rebal_freq=2, txn_bps=8):
    """Run SR strategy and return weekly NAV series"""
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    warmup = mom_lookback + skip + 2

    sector_stocks = defaultdict(list)
    for s in price_df.columns:
        sec = CSI500_POOL.get(s, {}).get('sector', '?')
        sector_stocks[sec].append(s)

    nav = [1.0]
    dates = [price_df.index[warmup - 1]]
    prev_holdings = set()

    i = warmup
    while i < len(price_df) - 1:
        mom_idx = i - skip
        mom = get_momentum(price_df, mom_idx, mom_lookback)

        # Sector average momentum
        sector_mom = {}
        for sec, stocks in sector_stocks.items():
            moms = [mom[s] for s in stocks if s in mom]
            if moms:
                sector_mom[sec] = np.mean(moms)

        top_secs = sorted(sector_mom.items(), key=lambda x: -x[1])[:top_sectors]
        top_sec_names = [s for s, _ in top_secs]

        # Pick best stocks per sector
        selected = []
        for sec in top_sec_names:
            candidates = [(s, mom.get(s, -999)) for s in sector_stocks[sec] if s in mom]
            candidates.sort(key=lambda x: -x[1])
            for s, _ in candidates[:stocks_per_sector]:
                selected.append(s)

        if not selected:
            i += 1
            continue

        selected_set = set(selected)
        turnover = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(len(selected_set), 1)
        period_txn = turnover * txn_cost

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            rets = [float(returns[s].iloc[j]) for s in selected if not pd.isna(returns[s].iloc[j])]
            port_ret = np.mean(rets) if rets else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])

        prev_holdings = selected_set
        i = hold_end

    return pd.Series(nav, index=dates)


def rolling_analysis(nav_series, window_weeks=52):
    """Compute rolling window returns for stability analysis"""
    results = []
    nav = nav_series.values
    dates = nav_series.index

    for start in range(len(nav) - window_weeks):
        end = start + window_weeks
        ret = nav[end] / nav[start] - 1
        start_date = dates[start].strftime('%Y-%m-%d')
        end_date = dates[end].strftime('%Y-%m-%d')
        results.append({
            'start': start_date,
            'end': end_date,
            'return_pct': round(ret * 100, 2)
        })

    returns = [r['return_pct'] for r in results]
    returns_arr = np.array(returns)

    summary = {
        'window_weeks': window_weeks,
        'total_windows': len(results),
        'positive_windows': int(np.sum(returns_arr > 0)),
        'win_rate_pct': round(float(np.mean(returns_arr > 0) * 100), 1),
        'mean_return_pct': round(float(np.mean(returns_arr)), 2),
        'median_return_pct': round(float(np.median(returns_arr)), 2),
        'std_return_pct': round(float(np.std(returns_arr)), 2),
        'min_return_pct': round(float(np.min(returns_arr)), 2),
        'max_return_pct': round(float(np.max(returns_arr)), 2),
        'p5_return_pct': round(float(np.percentile(returns_arr, 5)), 2),
        'p10_return_pct': round(float(np.percentile(returns_arr, 10)), 2),
        'p25_return_pct': round(float(np.percentile(returns_arr, 25)), 2),
        'p75_return_pct': round(float(np.percentile(returns_arr, 75)), 2),
        'p90_return_pct': round(float(np.percentile(returns_arr, 90)), 2),
        'p95_return_pct': round(float(np.percentile(returns_arr, 95)), 2),
    }

    # Find worst and best periods
    worst_idx = int(np.argmin(returns_arr))
    best_idx = int(np.argmax(returns_arr))
    summary['worst_period'] = results[worst_idx]
    summary['best_period'] = results[best_idx]

    # Monthly rolling (4-week) for finer granularity
    monthly_results = []
    for start in range(len(nav) - 4):
        end = start + 4
        ret = nav[end] / nav[start] - 1
        monthly_results.append(round(ret * 100, 2))

    monthly_arr = np.array(monthly_results)
    summary['monthly_4w'] = {
        'total_windows': len(monthly_results),
        'win_rate_pct': round(float(np.mean(monthly_arr > 0) * 100), 1),
        'mean_return_pct': round(float(np.mean(monthly_arr)), 2),
        'min_return_pct': round(float(np.min(monthly_arr)), 2),
        'max_return_pct': round(float(np.max(monthly_arr)), 2),
        'std_return_pct': round(float(np.std(monthly_arr)), 2),
    }

    # 6-month rolling (26 weeks)
    half_year_results = []
    for start in range(len(nav) - 26):
        end = start + 26
        ret = nav[end] / nav[start] - 1
        half_year_results.append({
            'start': dates[start].strftime('%Y-%m-%d'),
            'end': dates[end].strftime('%Y-%m-%d'),
            'return_pct': round(ret * 100, 2)
        })

    hy_returns = np.array([r['return_pct'] for r in half_year_results])
    summary['half_year_26w'] = {
        'total_windows': len(half_year_results),
        'win_rate_pct': round(float(np.mean(hy_returns > 0) * 100), 1),
        'mean_return_pct': round(float(np.mean(hy_returns)), 2),
        'min_return_pct': round(float(np.min(hy_returns)), 2),
        'max_return_pct': round(float(np.max(hy_returns)), 2),
        'worst_period': half_year_results[int(np.argmin(hy_returns))],
    }

    # Consecutive losing weeks
    weekly_rets = np.diff(nav) / nav[:-1]
    max_losing_streak = 0
    current_streak = 0
    for r in weekly_rets:
        if r < 0:
            current_streak += 1
            max_losing_streak = max(max_losing_streak, current_streak)
        else:
            current_streak = 0
    summary['max_consecutive_losing_weeks'] = max_losing_streak

    # Max drawdown duration
    peak = nav[0]
    in_dd = False
    dd_start = 0
    max_dd_duration = 0
    for i in range(len(nav)):
        if nav[i] >= peak:
            if in_dd:
                max_dd_duration = max(max_dd_duration, i - dd_start)
                in_dd = False
            peak = nav[i]
        else:
            if not in_dd:
                dd_start = i
                in_dd = True
    if in_dd:
        max_dd_duration = max(max_dd_duration, len(nav) - 1 - dd_start)
    summary['max_drawdown_duration_weeks'] = max_dd_duration

    return summary, results


def main():
    print("Loading data...")
    price_df = load_all_data()
    print(f"Data loaded: {len(price_df)} weeks x {price_df.shape[1]} stocks\n")

    print("Running Sector Rotation (best config: m12, top3, sps2, rebal2w)...")
    nav = run_sector_rotation(price_df, mom_lookback=12, skip=0,
                              top_sectors=3, stocks_per_sector=2, rebal_freq=2)
    print(f"NAV series: {len(nav)} weeks, {nav.index[0].strftime('%Y-%m-%d')} ~ {nav.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total return: {(nav.iloc[-1]/nav.iloc[0]-1)*100:.1f}%\n")

    # Rolling 52-week (12-month) analysis
    print("=== Rolling 52-week (12-month) Analysis ===")
    summary_52, windows_52 = rolling_analysis(nav, window_weeks=52)
    print(f"  Total windows: {summary_52['total_windows']}")
    print(f"  Win rate: {summary_52['win_rate_pct']}%")
    print(f"  Mean: {summary_52['mean_return_pct']}%")
    print(f"  Median: {summary_52['median_return_pct']}%")
    print(f"  Std: {summary_52['std_return_pct']}%")
    print(f"  Min: {summary_52['min_return_pct']}% ({summary_52['worst_period']['start']} ~ {summary_52['worst_period']['end']})")
    print(f"  Max: {summary_52['max_return_pct']}% ({summary_52['best_period']['start']} ~ {summary_52['best_period']['end']})")
    print(f"  P5/P10/P25: {summary_52['p5_return_pct']}% / {summary_52['p10_return_pct']}% / {summary_52['p25_return_pct']}%")
    print(f"  P75/P90/P95: {summary_52['p75_return_pct']}% / {summary_52['p90_return_pct']}% / {summary_52['p95_return_pct']}%")

    print(f"\n  Monthly (4w): win_rate={summary_52['monthly_4w']['win_rate_pct']}% mean={summary_52['monthly_4w']['mean_return_pct']}% min={summary_52['monthly_4w']['min_return_pct']}% max={summary_52['monthly_4w']['max_return_pct']}%")
    print(f"  Half-year (26w): win_rate={summary_52['half_year_26w']['win_rate_pct']}% mean={summary_52['half_year_26w']['mean_return_pct']}% min={summary_52['half_year_26w']['min_return_pct']}%")
    print(f"  Max consecutive losing weeks: {summary_52['max_consecutive_losing_weeks']}")
    print(f"  Max drawdown duration: {summary_52['max_drawdown_duration_weeks']} weeks")

    # All 52-week rolling windows sorted by return
    print("\n=== All 52-week Rolling Windows (sorted by return) ===")
    sorted_windows = sorted(windows_52, key=lambda x: x['return_pct'])
    for i, w in enumerate(sorted_windows):
        marker = " <<<WORST" if i == 0 else (" <<<BEST" if i == len(sorted_windows)-1 else "")
        print(f"  {w['start']} ~ {w['end']}: {w['return_pct']:+.1f}%{marker}")

    # Save results
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': 'SR_m12_top3_sps2_rebal2w',
        'nav_start': nav.index[0].strftime('%Y-%m-%d'),
        'nav_end': nav.index[-1].strftime('%Y-%m-%d'),
        'total_return_pct': round((nav.iloc[-1]/nav.iloc[0]-1)*100, 2),
        'rolling_52w_summary': summary_52,
        'rolling_52w_windows': sorted_windows,
    }

    out_path = os.path.join(DATA_DIR, 'csi500_rolling_stability.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
