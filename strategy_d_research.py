#!/usr/bin/env python3
"""
Strategy D Research: A股龙头股动量策略 研究+回测
股票池: ~28只各行业龙头
数据源: Yahoo Finance (weekly)
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import json
import sys
from datetime import datetime
from itertools import product

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 龙头股池 (28只, 覆盖主要行业)
# ============================================================
STOCK_POOL = {
    # 消费
    '600519.SS': {'name': '贵州茅台', 'sector': '白酒'},
    '000858.SZ': {'name': '五粮液', 'sector': '白酒'},
    '600887.SS': {'name': '伊利股份', 'sector': '食品'},
    '002714.SZ': {'name': '牧原股份', 'sector': '养殖'},
    # 金融
    '601318.SS': {'name': '中国平安', 'sector': '保险'},
    '600036.SS': {'name': '招商银行', 'sector': '银行'},
    # 科技
    '002415.SZ': {'name': '海康威视', 'sector': '安防'},
    '300750.SZ': {'name': '宁德时代', 'sector': '电池'},
    # 医药
    '600276.SS': {'name': '恒瑞医药', 'sector': '创新药'},
    '300760.SZ': {'name': '迈瑞医疗', 'sector': '医疗器械'},
    # 新能源
    '601012.SS': {'name': '隆基绿能', 'sector': '光伏'},
    '300274.SZ': {'name': '阳光电源', 'sector': '逆变器'},
    # 制造
    '000333.SZ': {'name': '美的集团', 'sector': '家电'},
    '600690.SS': {'name': '海尔智家', 'sector': '家电'},
    '002594.SZ': {'name': '比亚迪', 'sector': '新能源车'},
    # 军工
    '600893.SS': {'name': '航发动力', 'sector': '航发'},
    # 基建/地产
    '601668.SS': {'name': '中国建筑', 'sector': '建筑'},
    '600585.SS': {'name': '海螺水泥', 'sector': '水泥'},
    # 资源
    '601899.SS': {'name': '紫金矿业', 'sector': '有色'},
    '600028.SS': {'name': '中国石化', 'sector': '石油'},
    # 半导体
    '002371.SZ': {'name': '北方华创', 'sector': '半导体设备'},
    # 通信
    '000063.SZ': {'name': '中兴通讯', 'sector': '通信设备'},
    '600941.SS': {'name': '中国移动', 'sector': '运营商'},
    # 新消费/互联网
    '002230.SZ': {'name': '科大讯飞', 'sector': 'AI'},
    # 电力
    '600900.SS': {'name': '长江电力', 'sector': '水电'},
    # 煤炭
    '601088.SS': {'name': '中国神华', 'sector': '煤炭'},
    # 交运
    '601006.SS': {'name': '大秦铁路', 'sector': '铁路'},
    # 地产
    '001979.SZ': {'name': '招商蛇口', 'sector': '地产'},
}

print(f"股票池: {len(STOCK_POOL)} 只龙头股")


# ============================================================
# 数据获取
# ============================================================

def fetch_yahoo_weekly(ticker: str, years: int = 5) -> pd.DataFrame:
    """从Yahoo Finance获取周线数据"""
    cache_file = os.path.join(DATA_DIR, f'sd_{ticker.replace(".", "_").replace("^", "")}_weekly.csv')

    # 检查缓存 (3天内有效)
    if os.path.exists(cache_file):
        age = (time.time() - os.path.getmtime(cache_file)) / 86400
        if age <= 3:
            df = pd.read_csv(cache_file, parse_dates=['date'], index_col='date')
            return df

    end_ts = int(time.time())
    start_ts = end_ts - years * 365 * 86400
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
    params = {'period1': str(start_ts), 'period2': str(end_ts), 'interval': '1wk'}
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  [429] {ticker} 限流, 等待{wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            if r.status_code != 200:
                print(f"  [WARN] {ticker} HTTP {r.status_code}", file=sys.stderr)
                return pd.DataFrame()

            data = r.json()
            chart = data.get('chart', {}).get('result', [])
            if not chart:
                return pd.DataFrame()

            timestamps = chart[0].get('timestamp', [])
            quote = chart[0].get('indicators', {}).get('quote', [{}])[0]
            closes = quote.get('close', [])
            adjclose_list = chart[0].get('indicators', {}).get('adjclose', [{}])
            adjcloses = adjclose_list[0].get('adjclose', closes) if adjclose_list else closes

            rows = []
            for ts, c, ac in zip(timestamps, closes, adjcloses):
                if c is not None and ac is not None:
                    rows.append({
                        'date': pd.Timestamp(ts, unit='s').normalize(),
                        'close': float(c),
                        'adjclose': float(ac),
                    })

            df = pd.DataFrame(rows)
            df = df.drop_duplicates(subset='date', keep='last').set_index('date').sort_index()
            df.to_csv(cache_file)
            return df

        except Exception as ex:
            print(f"  [ERR] {ticker}: {ex}", file=sys.stderr)
            if attempt < 2:
                time.sleep(5)

    # Fallback to cache
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, parse_dates=['date'], index_col='date')
    return pd.DataFrame()


def load_all_stocks():
    """加载所有股票周线数据"""
    print("加载股票数据...")
    prices = {}
    failed = []

    for i, (ticker, info) in enumerate(STOCK_POOL.items()):
        df = fetch_yahoo_weekly(ticker)
        if len(df) > 0:
            prices[ticker] = df['adjclose']
            print(f"  [{i+1}/{len(STOCK_POOL)}] {info['name']} ({ticker}): {len(df)}周")
        else:
            failed.append(ticker)
            print(f"  [{i+1}/{len(STOCK_POOL)}] {info['name']} ({ticker}): FAILED")
        time.sleep(3)  # Rate limit protection

    if failed:
        print(f"\n失败: {len(failed)}只 - {[STOCK_POOL[t]['name'] for t in failed]}")

    # Build price DataFrame
    price_df = pd.DataFrame(prices)
    price_df = price_df.dropna(how='all')
    print(f"\n数据矩阵: {price_df.shape[0]}周 x {price_df.shape[1]}股")
    print(f"  时间范围: {price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")
    return price_df


# ============================================================
# 回测引擎 (严格无前瞻偏差)
# ============================================================

def backtest_momentum(price_df: pd.DataFrame,
                      lookback: int = 12,
                      skip_recent: int = 0,
                      top_n: int = 5,
                      rebal_freq: int = 4,
                      txn_cost_bps: int = 8,
                      min_stocks: int = 3) -> dict:
    """
    龙头股动量回测

    严格时序:
    - 在第 i 周末, 使用 price_df.iloc[:i+1] 数据计算动量
    - 选出 top_n 股票
    - 持有至下一个调仓日
    - 期间收益 = 实际第 i+1 周收益

    Args:
        lookback: 动量回看周数
        skip_recent: 跳过最近N周 (避免短期反转)
        top_n: 选前N名
        rebal_freq: 调仓频率 (周)
        txn_cost_bps: 单边交易成本
        min_stocks: 最少需要多少只有数据的股票才开始
    """
    txn_cost = txn_cost_bps / 10000
    returns = price_df.pct_change()

    warmup = lookback + skip_recent + 1
    nav = [1.0]
    dates = []
    prev_holdings = set()
    total_txn = 0.0
    weekly_returns = []
    holding_log = []

    i = warmup
    while i < len(price_df) - 1:
        # ---- Decision at week i, using data up to iloc[i] ----
        # Calculate momentum: return over [i-lookback-skip, i-skip]
        end_idx = i - skip_recent
        start_idx = end_idx - lookback

        if start_idx < 0 or end_idx <= 0:
            i += 1
            continue

        # Get available stocks at this point
        avail = price_df.columns[price_df.iloc[start_idx:end_idx+1].notna().all()]
        if len(avail) < min_stocks:
            i += 1
            continue

        # Momentum = price[end] / price[start] - 1
        mom = {}
        for col in avail:
            p_end = price_df[col].iloc[end_idx]
            p_start = price_df[col].iloc[start_idx]
            if p_start > 0 and not pd.isna(p_end) and not pd.isna(p_start):
                mom[col] = p_end / p_start - 1

        if len(mom) < min_stocks:
            i += 1
            continue

        # Select top N
        ranked = sorted(mom.items(), key=lambda x: -x[1])
        selected = set(t for t, _ in ranked[:top_n])

        # ---- Transaction cost ----
        new_buys = selected - prev_holdings
        sold = prev_holdings - selected
        turnover_positions = len(new_buys) + len(sold)
        period_txn = turnover_positions / max(len(selected), 1) * txn_cost
        total_txn += period_txn

        # ---- Hold for rebal_freq weeks, capturing actual returns ----
        hold_end = min(i + rebal_freq, len(price_df) - 1)

        for j in range(i + 1, hold_end + 1):
            # Week j return for selected stocks
            week_rets = []
            for s in selected:
                r = returns[s].iloc[j]
                if not pd.isna(r):
                    week_rets.append(r)

            if week_rets:
                # Equal weight
                port_ret = np.mean(week_rets)
            else:
                port_ret = 0.0

            # Apply txn cost only on first week of holding period
            if j == i + 1:
                port_ret -= period_txn

            nav.append(nav[-1] * (1 + port_ret))
            dates.append(price_df.index[j])
            weekly_returns.append(port_ret)

        holding_log.append({
            'decision_date': price_df.index[i].strftime('%Y-%m-%d'),
            'holdings': [STOCK_POOL[s]['name'] for s in selected if s in STOCK_POOL],
            'top_mom': round(ranked[0][1] * 100, 1) if ranked else 0,
        })

        prev_holdings = selected
        i = hold_end

    if not dates:
        return {'error': 'No trades generated'}

    nav_series = pd.Series(nav[1:], index=dates)
    total_years = (dates[-1] - dates[0]).days / 365.25

    if total_years <= 0:
        return {'error': 'Period too short'}

    cagr = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (1 / total_years) - 1
    dd = nav_series / nav_series.cummax() - 1
    mdd = dd.min()
    wr = pd.Series(weekly_returns)
    sharpe = wr.mean() / wr.std() * np.sqrt(52) if wr.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    win_rate = (wr > 0).sum() / len(wr) * 100

    # Annual returns
    annual = nav_series.resample('YE').last().pct_change().dropna()
    annual_returns = {str(d.year): round(v * 100, 1) for d, v in annual.items()}

    return {
        'lookback': lookback,
        'skip_recent': skip_recent,
        'top_n': top_n,
        'rebal_freq': rebal_freq,
        'period': f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}",
        'years': round(total_years, 1),
        'cagr_pct': round(cagr * 100, 1),
        'total_return_pct': round((nav_series.iloc[-1] - 1) * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'win_rate_pct': round(win_rate, 1),
        'annual_returns': annual_returns,
        'total_txn_pct': round(total_txn * 100, 2),
        'num_rebalances': len(holding_log),
    }


def backtest_benchmark(price_df: pd.DataFrame) -> dict:
    """等权持有基准"""
    returns = price_df.pct_change()
    avg_ret = returns.mean(axis=1).dropna()

    nav = (1 + avg_ret).cumprod()
    dates = nav.index
    total_years = (dates[-1] - dates[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / total_years) - 1
    dd = nav / nav.cummax() - 1
    mdd = dd.min()

    annual = nav.resample('YE').last().pct_change().dropna()
    annual_returns = {str(d.year): round(v * 100, 1) for d, v in annual.items()}

    return {
        'strategy': '等权持有全部龙头',
        'period': f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}",
        'years': round(total_years, 1),
        'cagr_pct': round(cagr * 100, 1),
        'mdd_pct': round(mdd * 100, 1),
        'sharpe': round(avg_ret.mean() / avg_ret.std() * np.sqrt(52), 3),
        'annual_returns': annual_returns,
    }


# ============================================================
# 主程序
# ============================================================

def main():
    price_df = load_all_stocks()

    if price_df.shape[1] < 10:
        print("ERROR: Too few stocks loaded!")
        sys.exit(1)

    # Save price data for later use
    price_df.to_csv(os.path.join(DATA_DIR, 'sd_all_weekly_prices.csv'))

    # Benchmark
    print("\n" + "="*60)
    print("基准: 等权持有")
    print("="*60)
    bench = backtest_benchmark(price_df)
    for k, v in bench.items():
        print(f"  {k}: {v}")

    # Parameter grid search
    print("\n" + "="*60)
    print("参数搜索")
    print("="*60)

    results = []
    param_grid = {
        'lookback': [4, 8, 12, 20],
        'skip_recent': [0, 1],
        'top_n': [3, 5, 8],
        'rebal_freq': [2, 4],
    }

    for lb, skip, tn, rf in product(
        param_grid['lookback'],
        param_grid['skip_recent'],
        param_grid['top_n'],
        param_grid['rebal_freq'],
    ):
        r = backtest_momentum(price_df, lookback=lb, skip_recent=skip, top_n=tn, rebal_freq=rf)
        if 'error' not in r:
            results.append(r)

    # Sort by Sharpe
    results.sort(key=lambda x: -x['sharpe'])

    print(f"\n共 {len(results)} 组参数")
    print(f"\nTop 10 by Sharpe:")
    print(f"{'LB':>3} {'Skip':>4} {'TopN':>4} {'Freq':>4} | {'CAGR':>6} {'MDD':>6} {'Sharpe':>6} {'Calmar':>6} {'Win%':>5} | Annual Returns")
    print("-" * 110)
    for r in results[:10]:
        ann = ' '.join(f"{y}:{v:+.0f}%" for y, v in sorted(r['annual_returns'].items()))
        print(f"{r['lookback']:>3} {r['skip_recent']:>4} {r['top_n']:>4} {r['rebal_freq']:>4} | "
              f"{r['cagr_pct']:>5.1f}% {r['mdd_pct']:>5.1f}% {r['sharpe']:>6.3f} {r['calmar']:>6.3f} {r['win_rate_pct']:>4.1f}% | {ann}")

    print(f"\nBottom 5 by Sharpe:")
    for r in results[-5:]:
        ann = ' '.join(f"{y}:{v:+.0f}%" for y, v in sorted(r['annual_returns'].items()))
        print(f"{r['lookback']:>3} {r['skip_recent']:>4} {r['top_n']:>4} {r['rebal_freq']:>4} | "
              f"{r['cagr_pct']:>5.1f}% {r['mdd_pct']:>5.1f}% {r['sharpe']:>6.3f} {r['calmar']:>6.3f} {r['win_rate_pct']:>4.1f}% | {ann}")

    # Save results
    output = {
        'benchmark': bench,
        'all_results': results,
        'best': results[0] if results else None,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(DATA_DIR, 'sd_research_results.json'), 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {os.path.join(DATA_DIR, 'sd_research_results.json')}")


if __name__ == '__main__':
    main()
