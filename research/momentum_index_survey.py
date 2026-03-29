"""
A股动量因子相关指数全面调研
查找以动量为主要因子的长期优秀指数
"""
from __future__ import annotations
import akshare as ak
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')


INDICES = {
    # 纯动量指数
    '930838': '动量指数',
    '930651': '300动量',
    '930652': '500动量',
    '931643': '动量30',
    '931068': 'A股动量',
    '399702': '深证动量',
    '931079': '动量60',
    '931657': '800动量',
    '932055': '中证动量',
    # 动量+其他因子
    '399405': '红利动量',
    '930955': '红利低波动量',
    # 成长类（含动量因子）
    '000918': '300成长',
    '399376': '小盘成长',
    '399374': '中盘成长',
    '000057': '全指成长',
    '399371': '国证成长',
    '930939': '500质量成长',
    '930938': '500成长估值',
    # 对比基准
    '000300': '沪深300',
    '000905': '中证500',
    '000852': '中证1000',
}


def analyze_index(code, name):
    try:
        df = ak.index_zh_a_hist(symbol=code, period='daily',
                                start_date='20050101', end_date='20260401')
        if df is None or len(df) < 200:
            return None

        first = str(df['日期'].iloc[0])
        last = str(df['日期'].iloc[-1])
        start_close = float(df['收盘'].iloc[0])
        end_close = float(df['收盘'].iloc[-1])
        years = (pd.to_datetime(last) - pd.to_datetime(first)).days / 365.25
        if years < 1:
            return None
        cagr = (end_close / start_close) ** (1 / years) - 1

        # Max drawdown
        closes = df['收盘'].astype(float).values
        peak = closes[0]
        max_dd = 0
        for c in closes:
            if c > peak:
                peak = c
            dd = (c - peak) / peak
            if dd < max_dd:
                max_dd = dd

        # Annual volatility
        rets = pd.Series(closes).pct_change().dropna()
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = cagr / ann_vol if ann_vol > 0 else 0

        return {
            'code': code,
            'name': name,
            'rows': len(df),
            'start': first,
            'end': last,
            'years': round(years, 1),
            'cagr': round(cagr * 100, 1),
            'max_dd': round(max_dd * 100, 1),
            'ann_vol': round(ann_vol * 100, 1),
            'sharpe': round(sharpe, 3),
            'calmar': round(cagr / abs(max_dd), 3) if max_dd != 0 else 0,
        }
    except Exception as e:
        print(f"  {code} {name}: ERROR {str(e)[:80]}")
        return None


def main():
    results = []
    for code, name in INDICES.items():
        print(f"  Fetching {code} {name}...")
        r = analyze_index(code, name)
        if r:
            results.append(r)

    results.sort(key=lambda x: x['cagr'], reverse=True)

    header = f"{'Code':>8s} {'Name':12s} {'Yrs':>4s} {'CAGR':>6s} {'MaxDD':>7s} {'Sharpe':>6s} {'Calmar':>7s} {'Vol':>5s}"
    print("\n" + header)
    print("-" * 65)
    for r in results:
        print(f"{r['code']:>8s} {r['name']:12s} {r['years']:4.1f} {r['cagr']:5.1f}% {r['max_dd']:6.1f}% {r['sharpe']:6.3f} {r['calmar']:7.3f} {r['ann_vol']:4.1f}%")

    with open('data/momentum_index_survey.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} results to data/momentum_index_survey.json")


if __name__ == '__main__':
    main()
