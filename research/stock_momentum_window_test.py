#!/usr/bin/env python3
"""
动量窗口对比测试: 低换手 + 不同动量窗口 (8W/3M/6M/12M)
CSI800 Top20 月度调仓, 含30bps交易成本, 历史成分股(无幸存者偏差)

测试目标: 找到最优动量回看期
"""
import sys, os, time, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'strategies'))
from strategy_m_stock_factor import (
    get_historical_constituents, fetch_stock_akshare,
    get_month_ends, get_constituents_at_date, get_csi800_at_date,
    build_composite_signal, TC_ONE_SIDE, PRICE_CACHE_DIR, CACHE_DIR
)

# ============================================================
# Custom factor computation with configurable momentum window
# ============================================================
def compute_factors_custom(code, df, as_of_date, val_data, mom_days, skip_days=22):
    """
    计算因子, 支持自定义动量窗口。

    Args:
        mom_days: 动量回看总天数 (e.g. 264=12M, 132=6M, 66=3M, 40=8W)
        skip_days: 跳过最近N天 (默认22=1个月, 避免短期反转)
    """
    hist = df[df.index <= as_of_date]
    n = len(hist)

    min_days = max(mom_days + skip_days, 120)
    if n < min_days:
        return None

    if (as_of_date - hist.index[-1]).days > 5:
        return None

    last = hist.iloc[-1]
    rec = {
        'code': code,
        'close': last['close'],
    }

    # 换手率
    rec['turn_20d'] = hist['turnover'].iloc[-20:].mean()

    # 自定义动量 (skip最近skip_days天)
    total_lookback = mom_days + skip_days
    if n >= total_lookback:
        rec['mom_custom'] = hist['close'].iloc[-skip_days] / hist['close'].iloc[-total_lookback] - 1

    return rec


def run_momentum_backtest(all_data, val_data, month_ends, hs300_const, zz500_const,
                          mom_days, skip_days, top_n, label):
    """运行单个动量窗口的回测"""
    factors = {
        'turn_20d': (1.0, True),     # 低换手
        'mom_custom': (1.0, False),  # 动量
    }

    nav = 1.0
    nav_history = []
    prev_holdings = set()
    rebal_count = 0
    total_turnover = 0
    rebal_events = 0
    monthly_rets = []

    for i in range(len(month_ends) - 1):
        dt = month_ends[i]
        dt_next = month_ends[i + 1]

        eligible = get_csi800_at_date(hs300_const, zz500_const, dt)
        if not eligible:
            nav_history.append({'date': dt, 'nav': nav})
            rebal_count += 1
            continue

        do_rebal = (rebal_count % 1 == 0)  # 月度调仓
        rebal_count += 1

        if do_rebal:
            records = []
            for code in eligible:
                if code not in all_data:
                    continue
                rec = compute_factors_custom(code, all_data[code], dt, val_data,
                                             mom_days, skip_days)
                if rec:
                    records.append(rec)

            if len(records) < top_n:
                nav_history.append({'date': dt, 'nav': nav})
                continue

            cur = pd.DataFrame(records)
            composite = build_composite_signal(cur, factors)
            if composite is None or composite.notna().sum() < top_n:
                nav_history.append({'date': dt, 'nav': nav})
                continue

            cur['signal'] = composite.values
            cur = cur.dropna(subset=['signal'])
            top = cur.nsmallest(top_n, 'signal')
            new_holdings = set(top['code'].tolist())
            rebal_events += 1
        else:
            new_holdings = prev_holdings.copy()

        # 前向收益
        fwd_rets = []
        for code in new_holdings:
            if code not in all_data:
                continue
            df = all_data[code]
            hist_cur = df[df.index <= dt]
            hist_nxt = df[df.index <= dt_next]
            if len(hist_cur) == 0 or len(hist_nxt) == 0:
                continue
            p_cur = hist_cur.iloc[-1]['close']
            p_nxt = hist_nxt.iloc[-1]['close']
            if p_cur > 0:
                fwd_rets.append(p_nxt / p_cur - 1)

        if not fwd_rets:
            nav_history.append({'date': dt, 'nav': nav})
            continue

        fwd_arr = np.array(fwd_rets)
        p01, p99 = np.percentile(fwd_arr, [1, 99])
        fwd_arr = np.clip(fwd_arr, p01, p99)
        port_ret = fwd_arr.mean()

        # 交易成本
        if do_rebal and prev_holdings:
            turnover = len(new_holdings - prev_holdings) / max(len(new_holdings), 1)
            tc_cost = turnover * TC_ONE_SIDE * 2
            total_turnover += turnover
        elif not prev_holdings:
            tc_cost = TC_ONE_SIDE
            total_turnover += 1.0
        else:
            tc_cost = 0

        nav *= (1 + port_ret - tc_cost)
        nav_history.append({'date': dt, 'nav': nav})
        monthly_rets.append(port_ret - tc_cost)
        prev_holdings = new_holdings

    if not nav_history or len(nav_history) < 12:
        return None

    nav_df = pd.DataFrame(nav_history).set_index('date')
    total_days = (nav_df.index[-1] - nav_df.index[0]).days
    years = total_days / 365.25
    if years < 1:
        return None

    cagr = (nav_df['nav'].iloc[-1]) ** (1 / years) - 1
    monthly_arr = np.array(monthly_rets)
    ann_vol = monthly_arr.std() * np.sqrt(12)
    sharpe = (cagr / ann_vol) if ann_vol > 0 else 0

    # Max drawdown
    running_max = nav_df['nav'].cummax()
    drawdown = (nav_df['nav'] / running_max - 1)
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

    # 胜率
    win_rate = (monthly_arr > 0).sum() / len(monthly_arr) * 100 if len(monthly_arr) > 0 else 0

    # 年度收益
    nav_df['year'] = nav_df.index.year
    yearly = {}
    for yr, grp in nav_df.groupby('year'):
        if len(grp) >= 2:
            yr_ret = grp['nav'].iloc[-1] / grp['nav'].iloc[0] - 1
            yearly[yr] = round(yr_ret * 100, 1)

    avg_turnover = total_turnover / rebal_events * 100 if rebal_events > 0 else 0

    return {
        'label': label,
        'CAGR': cagr * 100,
        'AnnVol': ann_vol * 100,
        'Sharpe': sharpe,
        'MaxDD': max_dd * 100,
        'Calmar': calmar,
        'WinRate': win_rate,
        'AvgTurnover': avg_turnover,
        'yearly': yearly,
    }


def main():
    t_start = time.time()

    print("=" * 70)
    print("动量窗口敏感性分析: LowTurn + Mom(X) CSI800 Top20 月度")
    print("=" * 70)

    # 加载数据
    print("\n加载历史成分股...")
    hs300_const = get_historical_constituents('hs300', 2016, 2026)
    zz500_const = get_historical_constituents('zz500', 2016, 2026)

    all_codes_ever = set()
    for codes in hs300_const.values():
        all_codes_ever |= codes
    for codes in zz500_const.values():
        all_codes_ever |= codes

    print(f"加载 {len(all_codes_ever)} 只股票数据...")
    all_data = {}
    for code in sorted(all_codes_ever):
        df = fetch_stock_akshare(code)
        if df is not None and len(df) > 60:
            all_data[code] = df
    print(f"成功加载: {len(all_data)} stocks")

    val_data = {}
    for code in all_data.keys():
        for cache_dir in [PRICE_CACHE_DIR, CACHE_DIR]:
            cache_file = os.path.join(cache_dir, f'{code}_val.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    val_data[code] = pickle.load(f)
                break

    month_ends = get_month_ends(all_data)
    print(f"月末日期: {len(month_ends)}, {month_ends[0].strftime('%Y-%m')} ~ {month_ends[-1].strftime('%Y-%m')}")

    # ============================================================
    # 动量窗口测试矩阵
    # ============================================================
    # (label, mom_days, skip_days, description)
    configs = [
        # 标准skip-1-month系列
        ('8W_skip1M',   40, 22, '8周动量(skip1月): t-62 to t-22'),
        ('3M_skip1M',   44, 22, '3月动量(skip1月): t-66 to t-22'),  # 3M=66 days total, minus 22 skip = 44 signal days
        ('6M_skip1M',  110, 22, '6月动量(skip1月): t-132 to t-22'),
        ('9M_skip1M',  176, 22, '9月动量(skip1月): t-198 to t-22'),
        ('12M_skip1M', 242, 22, '12月动量(skip1月): t-264 to t-22'),
        # 无skip系列 (含短期反转效应)
        ('8W_raw',      40,  0, '8周原始动量(无skip): t-40 to t-0'),
        ('3M_raw',      66,  0, '3月原始动量(无skip): t-66 to t-0'),
    ]

    # 修正: mom_days应该是从skip点到lookback点的距离
    # mom_6m: close[-22]/close[-132]-1, 所以 mom_days=132-22=110, skip=22, total=132
    # mom_12m: close[-22]/close[-264]-1, 所以 mom_days=264-22=242, skip=22, total=264
    # mom_3m: close[-22]/close[-66]-1, 所以 mom_days=66-22=44, skip=22, total=66
    # mom_8w: close[-22]/close[-62]-1, 8周=40天, skip=22, total=62
    # 但实际上 mom_days 在我的函数里是 total lookback - skip
    # compute_factors_custom: total_lookback = mom_days + skip_days

    # 重新定义: mom_days = 净动量天数 (不含skip)
    configs = [
        # skip-1-month (标准学术方法, 避免短期反转)
        ('8W(skip1M)',   40, 22, '8周净动量'),
        ('3M(skip1M)',   44, 22, '约3月净动量'),
        ('6M(skip1M)',  110, 22, '6月净动量 [V3已测]'),
        ('9M(skip1M)',  176, 22, '9月净动量'),
        ('12M(skip1M)', 242, 22, '12月净动量 [V3已测]'),
        # 无skip (原始动量, 含短期反转)
        ('8W(raw)',      40,  0, '8周原始动量'),
        ('3M(raw)',      66,  0, '3月原始动量'),
        ('6M(raw)',     132,  0, '6月原始动量'),
    ]

    top_n = 20
    results = []

    for label, mom_days, skip_days, desc in configs:
        print(f"\n--- 测试 {label}: {desc} (mom={mom_days}d, skip={skip_days}d, total={mom_days+skip_days}d) ---")
        r = run_momentum_backtest(
            all_data, val_data, month_ends,
            hs300_const, zz500_const,
            mom_days=mom_days,
            skip_days=skip_days,
            top_n=top_n,
            label=label,
        )
        if r:
            results.append(r)
            print(f"  CAGR={r['CAGR']:.1f}%, Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']:.1f}%, "
                  f"Calmar={r['Calmar']:.3f}, Turnover={r['AvgTurnover']:.0f}%, WinRate={r['WinRate']:.0f}%")
            print(f"  年度: {r['yearly']}")
        else:
            print(f"  FAILED")

    # ============================================================
    # 汇总表格
    # ============================================================
    print("\n" + "=" * 90)
    print(f"汇总: LowTurn + Mom(X) CSI800 Top{top_n} 月度调仓 (含30bps TC)")
    print("=" * 90)

    results.sort(key=lambda x: x['Sharpe'], reverse=True)

    hdr = f"{'动量窗口':<16s} {'CAGR':>7s} {'Vol':>6s} {'Sharpe':>7s} {'MaxDD':>7s} {'Calmar':>7s} {'换手':>6s} {'胜率':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['label']:<16s} {r['CAGR']:>6.1f}% {r['AnnVol']:>5.1f}% {r['Sharpe']:>7.3f} "
              f"{r['MaxDD']:>6.1f}% {r['Calmar']:>7.3f} {r['AvgTurnover']:>5.0f}% {r['WinRate']:>4.0f}%")

    print(f"\n年度收益对比:")
    for r in results:
        yr_str = '  '.join([f"{y}:{v:+.0f}%" for y, v in sorted(r['yearly'].items())])
        print(f"  {r['label']:<16s} {yr_str}")

    print(f"\n总耗时: {time.time()-t_start:.0f}s")
    print("\n结论: Sharpe越高=风险调整收益越优, Calmar越高=回撤补偿越好")


if __name__ == '__main__':
    main()
