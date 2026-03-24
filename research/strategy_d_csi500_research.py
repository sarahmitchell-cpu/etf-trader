#!/usr/bin/env python3
"""
Strategy D v2 Research: 中证500个股动量策略
使用 akshare stock_zh_a_daily 获取日线数据，再resample到周线
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import akshare as ak
from datetime import datetime
from collections import defaultdict
from itertools import product

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 中证500 代表性股票池 (50只)
# code前缀: sh/sz for akshare stock_zh_a_daily
# ============================================================

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
    """Fetch daily hfq data via akshare, resample to weekly close"""
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
                print(f"  {name}: too few data ({len(df) if df is not None else 0})")
                return None

            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            # Filter to 2021+
            df = df[df.index >= '2021-01-01']
            # Resample to weekly (Friday close)
            weekly = df['close'].resample('W-FRI').last().dropna()

            if len(weekly) < 50:
                print(f"  {name}: too few weeks ({len(weekly)})")
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
        time.sleep(1.0)

    print(f"\nLoaded {len(prices)}/{len(CSI500_POOL)} stocks")
    if len(prices) < 25:
        print(f"[ERROR] Need 25+, got {len(prices)}")
        return None

    price_df = pd.DataFrame(prices).dropna(how='all')
    price_df = price_df.ffill(limit=2)
    print(f"  Matrix: {price_df.shape[0]}w x {price_df.shape[1]} stocks")
    print(f"  Range: {price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")
    return price_df


# ============================================================
# Momentum selection & backtest (same as before)
# ============================================================

def select_top_momentum(price_df, idx, lookback, skip, top_n, sector_max=None):
    end_idx = idx - skip
    start_idx = end_idx - lookback
    if start_idx < 0 or end_idx <= 0:
        return []
    avail = [c for c in price_df.columns
             if not pd.isna(price_df[c].iloc[start_idx])
             and not pd.isna(price_df[c].iloc[end_idx])
             and price_df[c].iloc[start_idx] > 0]
    if len(avail) < 5:
        return []
    momenta = [(col, float(price_df[col].iloc[end_idx] / price_df[col].iloc[start_idx] - 1))
               for col in avail]
    ranked = sorted(momenta, key=lambda x: (-x[1], x[0]))

    if sector_max:
        selected = []
        sector_count = defaultdict(int)
        for t, _ in ranked:
            if len(selected) >= top_n:
                break
            sec = CSI500_POOL.get(t, {}).get('sector', '?')
            if sector_count[sec] < sector_max:
                selected.append(t)
                sector_count[sec] += 1
    else:
        selected = [t for t, _ in ranked[:top_n]]
    return selected


def run_backtest(price_df, lookback=8, skip=1, top_n=10, rebal_freq=2,
                 txn_bps=8, sector_max=1, label=""):
    txn_cost = txn_bps / 10000
    returns = price_df.pct_change()
    warmup = lookback + skip + 2

    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_rets = []

    i = warmup
    while i < len(price_df) - 1:
        selected = select_top_momentum(price_df, i, lookback, skip, top_n, sector_max)
        if not selected:
            i += 1
            continue
        selected_set = set(selected)
        turnover_pct = (len(selected_set - prev_holdings) + len(prev_holdings - selected_set)) / max(len(selected_set), 1)
        period_txn = turnover_pct * txn_cost
        total_txn += period_txn

        hold_end = min(i + rebal_freq, len(price_df) - 1)
        for j in range(i + 1, hold_end + 1):
            rets = [float(returns[s].iloc[j]) for s in selected if not pd.isna(returns[s].iloc[j])]
            port_ret = np.mean(rets) if rets else 0.0
            if j == i + 1:
                port_ret -= period_txn
            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_rets.append(port_ret)

        prev_holdings = selected_set
        i = hold_end

    if not dates or len(dates) < 20:
        return None
    nav_s = pd.Series(nav[1:], index=dates)
    years = (dates[-1] - dates[0]).days / 365.25
    if years < 1:
        return None

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
        'label': label, 'lookback': lookback, 'skip': skip, 'top_n': top_n,
        'rebal_freq': rebal_freq, 'sector_max': sector_max,
        'cagr_pct': round(cagr * 100, 1), 'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3), 'calmar': round(calmar, 3),
        'win_rate_pct': round(win_rate, 1),
        'total_return_pct': round((nav_s.iloc[-1] - 1) * 100, 1),
        'years': round(years, 1), 'total_txn_pct': round(total_txn * 100, 2),
        'annual_returns': annual_returns,
        'period': f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}",
    }


def grid_search(price_df):
    lookbacks = [4, 8, 12, 20]
    skips = [0, 1]
    top_ns = [8, 10, 15]
    sector_maxes = [1, 2, None]
    rebal_freqs = [2]

    combos = list(product(lookbacks, skips, top_ns, sector_maxes, rebal_freqs))
    results = []
    for count, (lb, sk, tn, sm, rf) in enumerate(combos, 1):
        sm_str = f"sec{sm}" if sm else "nosec"
        label = f"LB{lb}_SK{sk}_T{tn}_{sm_str}_RF{rf}"
        print(f"  [{count}/{len(combos)}] {label}...", end=" ", flush=True)
        r = run_backtest(price_df, lookback=lb, skip=sk, top_n=tn,
                         rebal_freq=rf, sector_max=sm, label=label)
        if r:
            results.append(r)
            print(f"CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% Sh={r['sharpe']} Cal={r['calmar']}")
        else:
            print("SKIP")
    return results


def main():
    print("=" * 60)
    print("Strategy D v2: CSI500 Mid-Cap Momentum Research")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pool: {len(CSI500_POOL)} stocks")
    print("=" * 60)

    price_df = load_all_data()
    if price_df is None:
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Baseline: Old D params (LB4 SK1 Top8 Sec1)")
    print("=" * 60)
    baseline = run_backtest(price_df, lookback=4, skip=1, top_n=8,
                            rebal_freq=2, sector_max=1, label="baseline")
    if baseline:
        print(f"  CAGR={baseline['cagr_pct']}% MDD={baseline['mdd_pct']}% "
              f"Sharpe={baseline['sharpe']} Calmar={baseline['calmar']}")
        print(f"  Annual: {baseline['annual_returns']}")

    print("\n" + "=" * 60)
    print("Grid Search")
    print("=" * 60)
    results = grid_search(price_df)

    if not results:
        print("No results!")
        sys.exit(1)

    for metric, key in [("Calmar", "calmar"), ("CAGR", "cagr_pct"), ("Sharpe", "sharpe")]:
        print(f"\n{'=' * 60}")
        print(f"TOP 10 by {metric}:")
        print("=" * 60)
        sr = sorted(results, key=lambda x: -x[key])
        for i, r in enumerate(sr[:10]):
            print(f"  #{i+1} {r['label']}: CAGR={r['cagr_pct']}% MDD={r['mdd_pct']}% "
                  f"Sh={r['sharpe']} Cal={r['calmar']} Win={r['win_rate_pct']}%")
            print(f"       {r['annual_returns']}")

    out_path = os.path.join(DATA_DIR, 'strategy_d_csi500_research.json')
    sc = sorted(results, key=lambda x: -x['calmar'])
    sg = sorted(results, key=lambda x: -x['cagr_pct'])
    ss = sorted(results, key=lambda x: -x['sharpe'])
    with open(out_path, 'w') as f:
        json.dump({
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pool_size': len(CSI500_POOL),
            'data_stocks': price_df.shape[1],
            'data_weeks': price_df.shape[0],
            'baseline': baseline,
            'total_combos': len(results),
            'top10_calmar': sc[:10],
            'top10_cagr': sg[:10],
            'top10_sharpe': ss[:10],
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
