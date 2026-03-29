#!/usr/bin/env python3
"""
Strategy A (New): 可转债溢价择时策略
(Convertible Bond Premium Timing)

核心逻辑:
  使用溢价率代理信号在可转债ETF和纯债之间切换:
  - 当可转债"便宜"(溢价率低/压缩) → 持有可转债(享受股性上涨)
  - 当可转债"贵"(溢价率高/扩张) → 持有纯债(规避下跌)

溢价率代理信号 (因无长期历史溢价率数据):
  Signal 1 - CB/Equity Ratio:
    转债指数/沪深300 比值的趋势。比值低=溢价低=便宜
  Signal 2 - Equity Momentum:
    沪深300 > N周MA → 股市上行→转债有上涨空间 → 持有转债
  Signal 3 - CB Momentum:
    转债指数 > N周MA → 转债趋势向上 → 持有转债
  Signal 4 - Dual Signal:
    同时满足股市上行+转债比值合理 → 持有转债

标的:
  - 进攻: 中证转债指数(000832) → ETF 511380
  - 防守: 信用债(H11073) → ETF 511030
         或 短债(H11006) → ETF 511360

用法:
  python3 strategy_a_cb_premium.py --backtest
  python3 strategy_a_cb_premium.py
  python3 strategy_a_cb_premium.py --json
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
from typing import Optional, Dict

# ============================================================
DATA_DIR = os.environ.get(
    'ETF_DATA_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
)
CACHE_MAX_AGE_DAYS = 3

INDICES = {
    'CB':     {'code': '000832', 'name': '中证转债', 'etf': '511380', 'etf_name': '可转债ETF'},
    'EQ':     {'code': '000300', 'name': '沪深300'},
    'CREDIT': {'code': 'H11073', 'name': '信用债',   'etf': '511030', 'etf_name': '信用债ETF'},
    'SHORT':  {'code': 'H11006', 'name': '短债',     'etf': '511360', 'etf_name': '短融ETF'},
    'BENCH':  {'code': 'H11001', 'name': '中证全债'},
}

# ============================================================
# Data
# ============================================================

def _fetch_csindex(code: str, days: int, csv_path: str) -> Optional[pd.DataFrame]:
    try:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        url = (f"https://www.csindex.com.cn/csindex-home/perf/index-perf"
               f"?indexCode={code}&startDate={start_date}&endDate={end_date}")
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
                   'Referer': 'https://www.csindex.com.cn/', 'Accept': 'application/json'}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if 'data' not in data or not data['data']:
            return None
        df = pd.DataFrame(data['data'])
        df['date'] = pd.to_datetime(df['tradeDate'])
        df['close'] = df['close'].astype(float)
        df = df.set_index('date').sort_index()
        df[['close']].to_csv(csv_path)
        return df
    except Exception as e:
        print(f"  [WARN] CSIndex fail ({code}): {e}")
        return None


def fetch_index(code: str, days: int = 600) -> Optional[pd.DataFrame]:
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f'{code}_daily.csv')
    if os.path.exists(csv_path):
        age = (time.time() - os.path.getmtime(csv_path)) / 86400
        if age <= CACHE_MAX_AGE_DAYS:
            df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
            if len(df) > 0:
                return df[['close']]
    df = _fetch_csindex(code, days, csv_path)
    if df is not None and len(df) > 0:
        return df[['close']]
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        if len(df) > 0:
            return df[['close']]
    return None


# ============================================================
# Backtest engine
# ============================================================

def backtest(cb_w: pd.Series, eq_w: pd.Series, safe_w: pd.Series,
             bench_w: pd.Series,
             signal_type: str, ma_period: int,
             safe_name: str = 'credit',
             txn_bps: int = 5) -> Dict:
    """
    signal_type:
      'cb_mom'    - CB index > MA → hold CB
      'eq_mom'    - Equity > MA → hold CB
      'ratio'     - CB/EQ ratio < MA(ratio) → hold CB (ratio低=溢价低=便宜)
      'dual'      - eq_mom AND ratio both bullish → hold CB
      'ratio_inv' - CB/EQ ratio > MA(ratio) → hold CB (ratio高=CB强势)
    """
    # Align
    common_start = max(s.index[0] for s in [cb_w, eq_w, safe_w, bench_w])
    common_end = min(s.index[-1] for s in [cb_w, eq_w, safe_w, bench_w])
    cb = cb_w.loc[common_start:common_end]
    eq = eq_w.loc[common_start:common_end]
    safe = safe_w.loc[common_start:common_end]
    bench = bench_w.loc[common_start:common_end]

    cb_ret = cb.pct_change()
    safe_ret = safe.pct_change()

    # Compute signal
    if signal_type == 'cb_mom':
        ma = cb.rolling(ma_period).mean()
        hold_cb = cb > ma
    elif signal_type == 'eq_mom':
        ma = eq.rolling(ma_period).mean()
        hold_cb = eq > ma
    elif signal_type == 'ratio':
        ratio = cb / eq
        ma = ratio.rolling(ma_period).mean()
        hold_cb = ratio < ma  # ratio低=溢价压缩=便宜→买入CB
    elif signal_type == 'ratio_inv':
        ratio = cb / eq
        ma = ratio.rolling(ma_period).mean()
        hold_cb = ratio > ma  # ratio高=CB相对强势→趋势跟随
    elif signal_type == 'dual':
        eq_ma = eq.rolling(ma_period).mean()
        ratio = cb / eq
        ratio_ma = ratio.rolling(ma_period).mean()
        hold_cb = (eq > eq_ma) & (ratio < ratio_ma)
    elif signal_type == 'eq_mom_ratio_filter':
        eq_ma = eq.rolling(ma_period).mean()
        ratio = cb / eq
        ratio_ma = ratio.rolling(ma_period * 2).mean()  # longer MA for ratio
        # Hold CB when equity bullish OR when ratio is compressed (cheap)
        hold_cb = (eq > eq_ma) | (ratio < ratio_ma * 0.98)
    else:
        raise ValueError(f"Unknown signal: {signal_type}")

    txn_cost = txn_bps / 10000
    pv = [1.0]
    prev = None
    trades = 0
    weekly_rets = []
    start_i = max(ma_period, ma_period * 2 if signal_type == 'eq_mom_ratio_filter' else ma_period)

    for i in range(1, len(cb)):
        if i < start_i:
            weekly_rets.append(0.0)
            pv.append(pv[-1])
            continue

        curr = 'cb' if hold_cb.iloc[i - 1] else 'safe'
        cost = 0.0
        if prev is not None and curr != prev:
            cost = 2 * txn_cost
            trades += 1

        r = cb_ret.iloc[i] if curr == 'cb' else safe_ret.iloc[i]
        if pd.isna(r):
            r = 0.0
        r -= cost
        weekly_rets.append(r)
        pv.append(pv[-1] * (1 + r))
        prev = curr

    idx = cb.index
    pv_s = pd.Series(pv, index=idx)
    wr_s = pd.Series(weekly_rets, index=idx[1:])

    years = (idx[-1] - idx[0]).days / 365.25
    total_ret = pv_s.iloc[-1] / pv_s.iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    rm = pv_s.cummax()
    dd = (pv_s - rm) / rm
    mdd = dd.min()
    sharpe = (wr_s.mean() * 52 - 0.025) / (wr_s.std() * np.sqrt(52)) if wr_s.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    bm_norm = bench / bench.iloc[0]
    bm_total = bm_norm.iloc[-1] - 1
    bm_cagr = (1 + bm_total) ** (1 / years) - 1 if years > 0 else 0

    # CB buy-and-hold
    cb_norm = cb / cb.iloc[0]
    cb_total = cb_norm.iloc[-1] - 1
    cb_cagr = (1 + cb_total) ** (1 / years) - 1 if years > 0 else 0
    cb_rm = cb_norm.cummax()
    cb_mdd = ((cb_norm - cb_rm) / cb_rm).min()

    cb_pct = hold_cb.iloc[start_i:].mean() * 100
    tpy = trades / years if years > 0 else 0

    annual = {}
    for y in range(idx[0].year, idx[-1].year + 1):
        yd = pv_s[pv_s.index.year == y]
        if len(yd) >= 2:
            annual[y] = yd.iloc[-1] / yd.iloc[0] - 1

    return {
        'signal': signal_type, 'ma': ma_period, 'safe': safe_name,
        'years': round(years, 1),
        'cagr': round(cagr * 100, 2),
        'mdd': round(mdd * 100, 2),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'cb_buy_hold_cagr': round(cb_cagr * 100, 2),
        'cb_buy_hold_mdd': round(cb_mdd * 100, 2),
        'bench_cagr': round(bm_cagr * 100, 2),
        'trades': trades, 'tpy': round(tpy, 1),
        'cb_pct': round(cb_pct, 1),
        'annual': {str(k): round(v * 100, 1) for k, v in annual.items()},
        'pv': pv_s,
    }


# ============================================================
# Grid search
# ============================================================

def run_grid(cb_w, eq_w, credit_w, short_w, bench_w):
    signals = ['cb_mom', 'eq_mom', 'ratio', 'ratio_inv', 'dual', 'eq_mom_ratio_filter']
    mas = [8, 13, 16, 20, 26, 30, 40, 52]
    safes = [('credit', credit_w), ('short', short_w)]

    results = []
    print(f"\n{'='*100}")
    print(f"网格搜索: 信号类型 x MA周期 x 防守资产")
    print(f"{'='*100}")
    print(f"{'信号':>20} {'MA':>4} {'防守':>6} {'CAGR%':>7} {'MDD%':>7} {'Sharpe':>7} {'Calmar':>7} "
          f"{'交易/年':>7} {'CB%':>5} {'CB持有CAGR':>10} {'CB MDD':>7}")
    print(f"{'-'*100}")

    for sig in signals:
        for ma in mas:
            for safe_name, safe_w in safes:
                try:
                    r = backtest(cb_w, eq_w, safe_w, bench_w, sig, ma, safe_name)
                    results.append(r)
                    print(f"{sig:>20} {ma:>4} {safe_name:>6} {r['cagr']:>7.2f} {r['mdd']:>7.2f} "
                          f"{r['sharpe']:>7.3f} {r['calmar']:>7.3f} {r['tpy']:>7.1f} {r['cb_pct']:>5.1f} "
                          f"{r['cb_buy_hold_cagr']:>10.2f} {r['cb_buy_hold_mdd']:>7.2f}")
                except Exception as e:
                    print(f"{sig:>20} {ma:>4} {safe_name:>6} ERROR: {e}")

    # Top results
    print(f"\n{'='*80}")
    print("=== Top 10 by Calmar (风险调整收益最优) ===")
    top = sorted(results, key=lambda x: x['calmar'], reverse=True)[:10]
    for i, r in enumerate(top):
        print(f"  {i+1:>2}. {r['signal']:>20} MA={r['ma']:>2} safe={r['safe']:>6}: "
              f"CAGR={r['cagr']:>6.2f}% MDD={r['mdd']:>6.2f}% "
              f"Calmar={r['calmar']:>6.3f} Sharpe={r['sharpe']:>6.3f} trades/yr={r['tpy']:.1f}")

    print(f"\n=== Top 10 by CAGR (绝对收益最高) ===")
    top_cagr = sorted(results, key=lambda x: x['cagr'], reverse=True)[:10]
    for i, r in enumerate(top_cagr):
        print(f"  {i+1:>2}. {r['signal']:>20} MA={r['ma']:>2} safe={r['safe']:>6}: "
              f"CAGR={r['cagr']:>6.2f}% MDD={r['mdd']:>6.2f}% "
              f"Calmar={r['calmar']:>6.3f} trades/yr={r['tpy']:.1f}")

    print(f"\n=== Top 5 by Sharpe ===")
    top_sharpe = sorted(results, key=lambda x: x['sharpe'], reverse=True)[:5]
    for i, r in enumerate(top_sharpe):
        print(f"  {i+1:>2}. {r['signal']:>20} MA={r['ma']:>2} safe={r['safe']:>6}: "
              f"CAGR={r['cagr']:>6.2f}% Sharpe={r['sharpe']:>6.3f}")

    return results, top[0]


# ============================================================
# Signal generation (live)
# ============================================================

def generate_signal(params: Dict = None) -> Dict:
    """Generate live signal with given or default params"""
    if params is None:
        # Load best params from backtest
        bp = os.path.join(DATA_DIR, 'strategy_a_cb_backtest.json')
        if os.path.exists(bp):
            with open(bp) as f:
                params = json.load(f)
        else:
            params = {'signal': 'eq_mom', 'ma': 20, 'safe': 'credit'}

    sig_type = params['signal']
    ma_period = params['ma']
    safe_key = 'CREDIT' if params['safe'] == 'credit' else 'SHORT'

    print("=" * 60)
    print("策略A: 可转债溢价择时 - 周度信号")
    print("=" * 60)

    cb_df = fetch_index(INDICES['CB']['code'], 600)
    eq_df = fetch_index(INDICES['EQ']['code'], 600)
    safe_df = fetch_index(INDICES[safe_key]['code'], 600)

    if any(d is None for d in [cb_df, eq_df, safe_df]):
        return {'error': '数据加载失败'}

    cb_w = cb_df['close'].resample('W-FRI').last().dropna()
    eq_w = eq_df['close'].resample('W-FRI').last().dropna()

    # Compute signal
    if sig_type == 'cb_mom':
        ma = cb_w.rolling(ma_period).mean()
        is_bull = cb_w.iloc[-1] > ma.iloc[-1]
        sig_desc = f"转债指数 {'>' if is_bull else '<'} {ma_period}周MA"
    elif sig_type == 'eq_mom':
        ma = eq_w.rolling(ma_period).mean()
        is_bull = eq_w.iloc[-1] > ma.iloc[-1]
        sig_desc = f"沪深300 {eq_w.iloc[-1]:.0f} {'>' if is_bull else '<'} {ma_period}周MA {ma.iloc[-1]:.0f}"
    elif sig_type in ('ratio', 'ratio_inv'):
        ratio = cb_w / eq_w
        ma = ratio.rolling(ma_period).mean()
        if sig_type == 'ratio':
            is_bull = ratio.iloc[-1] < ma.iloc[-1]
        else:
            is_bull = ratio.iloc[-1] > ma.iloc[-1]
        sig_desc = f"转债/沪深300比值 {ratio.iloc[-1]:.4f} {'<' if sig_type == 'ratio' else '>'} MA {ma.iloc[-1]:.4f}"
    elif sig_type == 'dual':
        eq_ma = eq_w.rolling(ma_period).mean()
        ratio = cb_w / eq_w
        ratio_ma = ratio.rolling(ma_period).mean()
        is_bull = (eq_w.iloc[-1] > eq_ma.iloc[-1]) and (ratio.iloc[-1] < ratio_ma.iloc[-1])
        sig_desc = f"双重信号: 沪深300{'上' if eq_w.iloc[-1] > eq_ma.iloc[-1] else '下'}穿MA + 比值{'低' if ratio.iloc[-1] < ratio_ma.iloc[-1] else '高'}于MA"
    elif sig_type == 'eq_mom_ratio_filter':
        eq_ma = eq_w.rolling(ma_period).mean()
        ratio = cb_w / eq_w
        ratio_ma = ratio.rolling(ma_period * 2).mean()
        is_bull = (eq_w.iloc[-1] > eq_ma.iloc[-1]) or (ratio.iloc[-1] < ratio_ma.iloc[-1] * 0.98)
        sig_desc = f"股市动量+溢价过滤"
    else:
        is_bull = True
        sig_desc = "unknown"

    safe_info = INDICES[safe_key]
    cb_info = INDICES['CB']
    holding = cb_info if is_bull else safe_info

    signal = {
        'strategy': '策略A - 可转债溢价择时',
        'date': str(cb_w.index[-1].date()),
        'signal_type': sig_type,
        'signal_desc': sig_desc,
        'regime': '看多转债' if is_bull else '防守(持纯债)',
        'holding': holding['name'],
        'etf': holding.get('etf', ''),
        'etf_name': holding.get('etf_name', ''),
        'ma_period': ma_period,
    }

    if '--json' not in sys.argv:
        print(f"\n📊 信号日期: {signal['date']}")
        print(f"信号类型: {sig_type} (MA={ma_period}周)")
        print(f"信号判断: {sig_desc}")
        print(f"\n{'📈' if is_bull else '🛡️'} 状态: {signal['regime']}")
        print(f"🎯 持仓: {signal['holding']} ({signal['etf']} {signal['etf_name']}) → 100%")

    return signal


# ============================================================
def main():
    if '--backtest' in sys.argv:
        print("=" * 60)
        print("策略A: 可转债溢价择时 - 完整回测 + 网格搜索")
        print("=" * 60)

        days = 5500
        print(f"\n加载数据 (~{days // 365}年)...")
        data = {}
        for key, info in INDICES.items():
            print(f"  {info['name']} ({info['code']})...")
            df = fetch_index(info['code'], days)
            if df is None:
                print(f"  ERROR: {info['name']}")
                sys.exit(1)
            data[key] = df['close'].resample('W-FRI').last().dropna()
            print(f"    {data[key].index[0].date()} ~ {data[key].index[-1].date()} ({len(data[key])}周)")
            time.sleep(0.3)

        results, best = run_grid(data['CB'], data['EQ'], data['CREDIT'], data['SHORT'], data['BENCH'])

        print(f"\n{'='*60}")
        print(f"最优策略详情: {best['signal']} MA={best['ma']} safe={best['safe']}")
        print(f"{'='*60}")
        print(f"  CAGR:       {best['cagr']}%")
        print(f"  MDD:        {best['mdd']}%")
        print(f"  Sharpe:     {best['sharpe']}")
        print(f"  Calmar:     {best['calmar']}")
        print(f"  交易次数:   {best['trades']} ({best['tpy']}/年)")
        print(f"  转债持有:   {best['cb_pct']}%")
        print(f"\n  对比:")
        print(f"  转债买入持有: CAGR={best['cb_buy_hold_cagr']}% MDD={best['cb_buy_hold_mdd']}%")
        print(f"  中证全债:     CAGR={best['bench_cagr']}%")
        print(f"\n  年度收益:")
        for y, r in sorted(best['annual'].items()):
            print(f"    {y}: {r:+.1f}%")

        # Save
        save = {k: v for k, v in best.items() if k != 'pv'}
        path = os.path.join(DATA_DIR, 'strategy_a_cb_backtest.json')
        with open(path, 'w') as f:
            json.dump(save, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {path}")

    else:
        signal = generate_signal()
        if '--json' in sys.argv:
            print(json.dumps(signal, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
