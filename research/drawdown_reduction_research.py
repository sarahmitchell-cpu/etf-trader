#!/usr/bin/env python3
"""
Drawdown Reduction Research for MA Timing Strategies
研究降低MA择时策略最大回撤的方法

测试方法:
1. 基准: 纯MA60 (close > MA60 → 满仓)
2. 双均线: MA_fast 上穿 MA_slow → 做多
3. 移动止损: 持仓期间从高点回撤X%则平仓，等新的MA信号再入场
4. 波动率缩放: 高波动时减仓 (用ATR或realized vol)
5. 双因子: MA趋势 + RSI过滤 (RSI<30不卖, RSI>70不买)
6. 多资产分散: 300成长 + 中证500 + 红利 等权组合

目标: MaxDD < -25% 同时尽量保持CAGR和Sharpe
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json
import os
import time
import requests
from datetime import datetime


# ============================================================
# 数据获取
# ============================================================

def fetch_csi_tr(code, name):
    """从CSI API获取全收益指数"""
    url = 'https://www.csindex.com.cn/csindex-home/perf/index-perf'
    params = {'indexCode': code, 'startDate': '20050101',
              'endDate': datetime.now().strftime('%Y%m%d')}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
        'Referer': 'https://www.csindex.com.cn/'
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        data = r.json()
        if str(data.get('code')) != '200' or not data.get('data'):
            return None
        items = data['data']
        df = pd.DataFrame(items)
        df['date'] = pd.to_datetime(df['tradeDate'], format='%Y%m%d')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df[['date', 'close']].dropna().set_index('date').sort_index()
        df = df[df['close'] > 0]
        df = df[~df.index.duplicated(keep='first')]
        print(f"  {name}({code}): {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} rows")
        return df
    except Exception as e:
        print(f"  {name}({code}): error {e}")
        return None


def fetch_akshare(code, name):
    """从akshare获取价格指数"""
    try:
        import akshare as ak
        df = ak.index_zh_a_hist(symbol=code, period='daily',
                                start_date='20050101', end_date='20260328')
        if df is None or len(df) == 0:
            return None
        df['date'] = pd.to_datetime(df['日期'])
        df['close'] = pd.to_numeric(df['收盘'], errors='coerce')
        df = df[['date', 'close']].dropna().set_index('date').sort_index()
        df = df[df['close'] > 0]
        df = df[~df.index.duplicated(keep='first')]
        print(f"  {name}({code}): {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} rows")
        return df
    except Exception as e:
        print(f"  {name}({code}): error {e}")
        return None


# ============================================================
# 回测指标计算
# ============================================================

def calc_metrics(returns, bh_returns=None):
    """计算回测指标"""
    if len(returns) < 252:
        return None
    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1
    years = len(returns) / 252
    if years < 1:
        return None
    cagr = (1 + total_ret) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    monthly = returns.resample('ME').sum()
    win_rate = (monthly > 0).mean() * 100
    annual = returns.resample('YE').sum()
    annual_dict = {str(idx.year): round(val * 100, 1) for idx, val in annual.items()}

    result = {
        'cagr': round(cagr * 100, 2),
        'vol': round(vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'max_dd': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'win_rate': round(win_rate, 1),
        'total_ret': round(total_ret * 100, 1),
        'years': round(years, 1),
        'annual': annual_dict,
    }

    if bh_returns is not None:
        bh_cum = (1 + bh_returns).cumprod()
        bh_total = bh_cum.iloc[-1] - 1
        bh_cagr = (1 + bh_total) ** (1 / years) - 1
        bh_peak = bh_cum.cummax()
        bh_max_dd = ((bh_cum - bh_peak) / bh_peak).min()
        result['bh_cagr'] = round(bh_cagr * 100, 2)
        result['bh_max_dd'] = round(bh_max_dd * 100, 2)

    return result


# ============================================================
# 策略1: 纯MA (基准)
# ============================================================

def strategy_pure_ma(df, ma_period, txn_cost_bps=8):
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['ret'] = d['close'].pct_change()
    d['position'] = (d['close'] > d['ma']).astype(float)
    d['signal'] = d['position'].shift(1)
    d = d.dropna(subset=['signal', 'ret'])
    d['strat_ret'] = d['ret'] * d['signal']
    txn = txn_cost_bps / 10000
    d['strat_ret'] -= d['signal'].diff().abs() * txn
    m = calc_metrics(d['strat_ret'], d['ret'])
    if m:
        m['trades'] = int((d['signal'].diff().abs() > 0.01).sum())
    return m


# ============================================================
# 策略2: 双均线交叉
# ============================================================

def strategy_dual_ma(df, fast_ma, slow_ma, txn_cost_bps=8):
    d = df.copy()
    d['ma_fast'] = d['close'].rolling(fast_ma).mean()
    d['ma_slow'] = d['close'].rolling(slow_ma).mean()
    d['ret'] = d['close'].pct_change()
    d['position'] = (d['ma_fast'] > d['ma_slow']).astype(float)
    d['signal'] = d['position'].shift(1)
    d = d.dropna(subset=['signal', 'ret'])
    d['strat_ret'] = d['ret'] * d['signal']
    txn = txn_cost_bps / 10000
    d['strat_ret'] -= d['signal'].diff().abs() * txn
    m = calc_metrics(d['strat_ret'], d['ret'])
    if m:
        m['trades'] = int((d['signal'].diff().abs() > 0.01).sum())
    return m


# ============================================================
# 策略3: MA + 移动止损
# ============================================================

def strategy_ma_trailing_stop(df, ma_period, stop_pct, txn_cost_bps=8):
    """MA趋势 + 移动止损: 持仓期间从高点回撤stop_pct则平仓，等MA重新金叉再入场"""
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['ret'] = d['close'].pct_change()
    d = d.dropna(subset=['ma'])

    position = pd.Series(0.0, index=d.index)
    entry_high = 0.0
    stopped_out = False

    for i in range(1, len(d)):
        close = d['close'].iloc[i]
        ma = d['ma'].iloc[i]
        prev_close = d['close'].iloc[i-1]
        prev_ma = d['ma'].iloc[i-1]

        if position.iloc[i-1] == 1.0:
            # 持仓中: 更新高点, 检查止损
            entry_high = max(entry_high, close)
            drawdown = (close - entry_high) / entry_high
            if drawdown < -stop_pct:
                position.iloc[i] = 0.0  # 止损出场
                stopped_out = True
            else:
                if close < ma:
                    position.iloc[i] = 0.0  # MA跌破也出场
                else:
                    position.iloc[i] = 1.0
        else:
            # 空仓: 等MA信号入场
            if close > ma and not stopped_out:
                position.iloc[i] = 1.0
                entry_high = close
            elif close > ma and stopped_out:
                # 止损后要求价格重新站上MA才能入场
                if prev_close <= prev_ma and close > ma:
                    # 重新金叉
                    position.iloc[i] = 1.0
                    entry_high = close
                    stopped_out = False
                else:
                    # 还在MA上方但是被止损了，等重新金叉
                    position.iloc[i] = 0.0
            else:
                position.iloc[i] = 0.0
                stopped_out = False  # 跌破MA后重置止损状态

    d['signal'] = position.shift(1)
    d = d.dropna(subset=['signal', 'ret'])
    d['strat_ret'] = d['ret'] * d['signal']
    txn = txn_cost_bps / 10000
    d['strat_ret'] -= d['signal'].diff().abs() * txn
    m = calc_metrics(d['strat_ret'], d['ret'])
    if m:
        m['trades'] = int((d['signal'].diff().abs() > 0.01).sum())
    return m


# ============================================================
# 策略4: MA + 波动率缩放
# ============================================================

def strategy_ma_vol_scaling(df, ma_period, vol_window=20, vol_target=0.15, txn_cost_bps=8):
    """MA趋势 + 波动率缩放: 高波动减仓"""
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['ret'] = d['close'].pct_change()
    d['realized_vol'] = d['ret'].rolling(vol_window).std() * np.sqrt(252)

    # MA信号
    d['trend'] = (d['close'] > d['ma']).astype(float)

    # 波动率缩放: position = min(1, vol_target / realized_vol)
    d['vol_scale'] = (vol_target / d['realized_vol']).clip(0, 1)
    d['position'] = d['trend'] * d['vol_scale']
    d['signal'] = d['position'].shift(1)

    d = d.dropna(subset=['signal', 'ret'])
    d['strat_ret'] = d['ret'] * d['signal']
    txn = txn_cost_bps / 10000
    d['strat_ret'] -= d['signal'].diff().abs() * txn
    m = calc_metrics(d['strat_ret'], d['ret'])
    if m:
        m['trades'] = int((d['signal'].diff().abs() > 0.01).sum())
    return m


# ============================================================
# 策略5: MA + RSI过滤
# ============================================================

def strategy_ma_rsi(df, ma_period, rsi_period=14, rsi_sell=75, rsi_buy=30, txn_cost_bps=8):
    """MA趋势 + RSI: 趋势向上且RSI不极端高时持仓"""
    d = df.copy()
    d['ma'] = d['close'].rolling(ma_period).mean()
    d['ret'] = d['close'].pct_change()

    # RSI
    delta = d['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss
    d['rsi'] = 100 - (100 / (1 + rs))

    # 逐日仓位: MA向上 + RSI不超卖
    position = pd.Series(0.0, index=d.index)
    for i in range(1, len(d)):
        if pd.isna(d['ma'].iloc[i]) or pd.isna(d['rsi'].iloc[i]):
            continue
        trend_up = d['close'].iloc[i] > d['ma'].iloc[i]
        rsi = d['rsi'].iloc[i]

        if trend_up:
            if rsi > rsi_sell:
                position.iloc[i] = 0.5  # RSI过热减仓
            else:
                position.iloc[i] = 1.0
        else:
            if rsi < rsi_buy:
                position.iloc[i] = 0.3  # RSI超卖可以小仓抄底
            else:
                position.iloc[i] = 0.0

    d['signal'] = position.shift(1)
    d = d.dropna(subset=['signal', 'ret'])
    d['strat_ret'] = d['ret'] * d['signal']
    txn = txn_cost_bps / 10000
    d['strat_ret'] -= d['signal'].diff().abs() * txn
    m = calc_metrics(d['strat_ret'], d['ret'])
    if m:
        m['trades'] = int((d['signal'].diff().abs() > 0.01).sum())
    return m


# ============================================================
# 策略6: 多资产分散 + MA择时
# ============================================================

def strategy_multi_asset_ma(data_dict, ma_period, txn_cost_bps=8):
    """多资产等权 + 各自MA择时"""
    all_rets = []
    for name, df in data_dict.items():
        d = df.copy()
        d['ma'] = d['close'].rolling(ma_period).mean()
        d['ret'] = d['close'].pct_change()
        d['position'] = (d['close'] > d['ma']).astype(float)
        d['signal'] = d['position'].shift(1)
        d = d.dropna(subset=['signal', 'ret'])
        d['strat_ret'] = d['ret'] * d['signal']
        txn = txn_cost_bps / 10000
        d['strat_ret'] -= d['signal'].diff().abs() * txn
        all_rets.append(d['strat_ret'].rename(name))

    if not all_rets:
        return None

    combined = pd.concat(all_rets, axis=1).dropna()
    if len(combined) < 252:
        return None

    # 等权平均
    portfolio_ret = combined.mean(axis=1)

    # BH benchmark: 等权买入持有
    bh_rets = []
    for name, df in data_dict.items():
        d = df.copy()
        d['ret'] = d['close'].pct_change()
        bh_rets.append(d['ret'].rename(name))
    bh_combined = pd.concat(bh_rets, axis=1).dropna()
    bh_ret = bh_combined.mean(axis=1)
    # Align
    common_idx = portfolio_ret.index.intersection(bh_ret.index)
    portfolio_ret = portfolio_ret.loc[common_idx]
    bh_ret = bh_ret.loc[common_idx]

    m = calc_metrics(portfolio_ret, bh_ret)
    if m:
        m['trades'] = 'N/A (multi-asset)'
        m['n_assets'] = len(data_dict)
    return m


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("  Drawdown Reduction Research")
    print("  目标: MaxDD < -25% 同时保持CAGR和Sharpe")
    print("=" * 80)

    # 1. Fetch data
    print("\n[1] Fetching data...")

    # Primary: 300成长 TR (CSI API)
    df_300g_tr = fetch_csi_tr('H00918', '300成长TR')

    # Fallback to akshare if CSI fails
    df_300g = fetch_akshare('000918', '300成长')
    df_500 = fetch_akshare('000905', '中证500')
    df_div = fetch_akshare('000922', '中证红利')

    # Use TR if available, otherwise price index
    primary = df_300g_tr if df_300g_tr is not None else df_300g
    primary_name = '300成长TR' if df_300g_tr is not None else '300成长'

    if primary is None:
        print("ERROR: Failed to fetch primary data")
        return

    # 2. Test all strategies
    print(f"\n[2] Running backtests on {primary_name}...")
    results = []

    # 2.1 基准: 纯MA
    print("\n  --- 基准: 纯MA ---")
    for ma in [40, 60, 80, 120]:
        m = strategy_pure_ma(primary, ma)
        if m:
            m['strategy'] = f'PureMA{ma}'
            m['description'] = f'纯MA{ma}择时'
            results.append(m)
            print(f"    MA{ma}: CAGR={m['cagr']:.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:.1f}% Trades={m['trades']}")

    # 2.2 双均线
    print("\n  --- 双均线交叉 ---")
    for fast, slow in [(10, 60), (20, 60), (20, 120), (40, 120), (40, 200), (60, 200)]:
        m = strategy_dual_ma(primary, fast, slow)
        if m:
            m['strategy'] = f'DualMA_{fast}_{slow}'
            m['description'] = f'双均线 MA{fast}/MA{slow}'
            results.append(m)
            print(f"    MA{fast}/MA{slow}: CAGR={m['cagr']:.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:.1f}% Trades={m['trades']}")

    # 2.3 MA + 移动止损
    print("\n  --- MA + 移动止损 ---")
    for ma in [60, 80, 120]:
        for stop in [0.05, 0.08, 0.10, 0.12, 0.15]:
            m = strategy_ma_trailing_stop(primary, ma, stop)
            if m:
                m['strategy'] = f'MA{ma}_Stop{int(stop*100)}pct'
                m['description'] = f'MA{ma}+{int(stop*100)}%移动止损'
                results.append(m)
                print(f"    MA{ma}+Stop{int(stop*100)}%: CAGR={m['cagr']:.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:.1f}% Trades={m['trades']}")

    # 2.4 MA + 波动率缩放
    print("\n  --- MA + 波动率缩放 ---")
    for ma in [60, 80, 120]:
        for vol_target in [0.10, 0.12, 0.15, 0.20]:
            m = strategy_ma_vol_scaling(primary, ma, vol_target=vol_target)
            if m:
                m['strategy'] = f'MA{ma}_Vol{int(vol_target*100)}'
                m['description'] = f'MA{ma}+Vol目标{int(vol_target*100)}%'
                results.append(m)
                print(f"    MA{ma}+Vol{int(vol_target*100)}%: CAGR={m['cagr']:.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:.1f}% Trades={m['trades']}")

    # 2.5 MA + RSI
    print("\n  --- MA + RSI过滤 ---")
    for ma in [60, 80, 120]:
        for rsi_sell in [70, 75, 80]:
            m = strategy_ma_rsi(primary, ma, rsi_sell=rsi_sell, rsi_buy=30)
            if m:
                m['strategy'] = f'MA{ma}_RSI{rsi_sell}'
                m['description'] = f'MA{ma}+RSI卖{rsi_sell}'
                results.append(m)
                print(f"    MA{ma}+RSI{rsi_sell}: CAGR={m['cagr']:.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:.1f}% Trades={m['trades']}")

    # 2.6 多资产分散
    print("\n  --- 多资产分散 ---")
    multi_assets = {}
    if df_300g is not None:
        multi_assets['300成长'] = df_300g
    if df_500 is not None:
        multi_assets['中证500'] = df_500
    if df_div is not None:
        multi_assets['中证红利'] = df_div

    if len(multi_assets) >= 2:
        for ma in [60, 80, 120]:
            # 2资产: 300成长 + 中证红利
            if '300成长' in multi_assets and '中证红利' in multi_assets:
                combo2 = {'300成长': multi_assets['300成长'], '中证红利': multi_assets['中证红利']}
                m = strategy_multi_asset_ma(combo2, ma)
                if m:
                    m['strategy'] = f'Multi2_MA{ma}'
                    m['description'] = f'300成长+红利 等权 MA{ma}'
                    results.append(m)
                    print(f"    300成长+红利 MA{ma}: CAGR={m['cagr']:.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:.1f}%")

            # 3资产: 全部
            if len(multi_assets) >= 3:
                m = strategy_multi_asset_ma(multi_assets, ma)
                if m:
                    m['strategy'] = f'Multi3_MA{ma}'
                    m['description'] = f'300成长+500+红利 等权 MA{ma}'
                    results.append(m)
                    print(f"    3资产等权 MA{ma}: CAGR={m['cagr']:.1f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:.1f}%")

    # 3. Summary
    print(f"\n\n{'='*100}")
    print(f"  TOTAL: {len(results)} strategies tested")
    print(f"{'='*100}")

    # Sort by Sharpe
    results.sort(key=lambda x: x['sharpe'], reverse=True)

    print(f"\n  {'Rank':<5} {'Strategy':<30} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'Vol':>7} {'Trades':>7}")
    print(f"  {'-'*5} {'-'*30} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
    for i, r in enumerate(results):
        trades = str(r.get('trades', '?'))
        print(f"  {i+1:<5} {r['strategy']:<30} {r['cagr']:>6.1f}% {r['sharpe']:>7.3f} {r['max_dd']:>7.1f}% "
              f"{r['calmar']:>7.3f} {r['vol']:>6.1f}% {trades:>7}")

    # Filter: MaxDD > -25%
    low_dd = [r for r in results if r['max_dd'] > -25]
    print(f"\n\n{'='*100}")
    print(f"  FILTERED: MaxDD > -25% ({len(low_dd)} strategies)")
    print(f"{'='*100}")
    if low_dd:
        low_dd.sort(key=lambda x: x['sharpe'], reverse=True)
        print(f"\n  {'Rank':<5} {'Strategy':<30} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7}")
        print(f"  {'-'*5} {'-'*30} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
        for i, r in enumerate(low_dd):
            print(f"  {i+1:<5} {r['strategy']:<30} {r['cagr']:>6.1f}% {r['sharpe']:>7.3f} {r['max_dd']:>7.1f}% {r['calmar']:>7.3f}")
    else:
        print("  No strategies with MaxDD > -25%")

    # Filter: MaxDD > -20%
    very_low_dd = [r for r in results if r['max_dd'] > -20]
    print(f"\n  ULTRA LOW DD: MaxDD > -20% ({len(very_low_dd)} strategies)")
    if very_low_dd:
        very_low_dd.sort(key=lambda x: x['sharpe'], reverse=True)
        for i, r in enumerate(very_low_dd):
            print(f"  {i+1:<5} {r['strategy']:<30} {r['cagr']:>6.1f}% {r['sharpe']:>7.3f} {r['max_dd']:>7.1f}% {r['calmar']:>7.3f}")

    # Best by Calmar (CAGR/MaxDD)
    results.sort(key=lambda x: x['calmar'], reverse=True)
    print(f"\n\n{'='*100}")
    print(f"  TOP 10 by Calmar (CAGR/MaxDD) — best risk-adjusted:")
    print(f"{'='*100}")
    print(f"\n  {'Rank':<5} {'Strategy':<30} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7}")
    print(f"  {'-'*5} {'-'*30} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1:<5} {r['strategy']:<30} {r['cagr']:>6.1f}% {r['sharpe']:>7.3f} {r['max_dd']:>7.1f}% {r['calmar']:>7.3f}")

    # Best annual returns for top strategies
    print(f"\n\n{'='*100}")
    print(f"  Top 3 by Calmar — Annual Returns:")
    for r in results[:3]:
        print(f"\n  {r['strategy']} ({r['description']}):")
        print(f"    CAGR={r['cagr']:.1f}% Sharpe={r['sharpe']:.3f} MaxDD={r['max_dd']:.1f}% Calmar={r['calmar']:.3f}")
        for yr, ret in sorted(r['annual'].items()):
            bar = '+' * max(0, int(ret / 3)) if ret > 0 else '-' * max(0, int(-ret / 3))
            print(f"    {yr}: {ret:>+7.1f}% {bar}")

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'drawdown_reduction_results.json')
    # Remove 'annual' for cleaner JSON
    save_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != 'annual'}
        save_results.append(sr)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
