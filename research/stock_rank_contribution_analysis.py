#!/usr/bin/env python3
"""
Top20 排位贡献分析:
1. 按composite排名分组(#1-5, #6-10, #11-15, #16-20), 比较各组前向收益
2. 按市值大小分组, 比较贡献
3. 分析排名靠前的股票是否真的贡献更多收益

CSI800 低换手+12M动量 Top20 月度调仓
"""
import sys, os, time, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/claw/etf-trader/research')
from stock_strategy_live import (
    get_historical_constituents, fetch_stock_akshare,
    get_month_ends, get_constituents_at_date, get_csi800_at_date,
    build_composite_signal, compute_factors_single,
    TC_ONE_SIDE, PRICE_CACHE_DIR, CACHE_DIR
)

FACTORS = {
    'turn_20d': (1.0, True),
    'mom_12m':  (1.0, False),
}
TOP_N = 20
BACKTEST_START = '2016-01-31'


def load_all_data():
    """Load cached stock data"""
    cache_file = os.path.join(CACHE_DIR, 'all_stock_data.pkl')
    if os.path.exists(cache_file) and time.time() - os.path.getmtime(cache_file) < 86400 * 3:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


def get_market_cap_proxy(code, df, as_of_date):
    """Use amount (turnover value) as market cap proxy, or total shares * price"""
    hist = df[df.index <= as_of_date]
    if len(hist) < 20:
        return np.nan
    # Use 20-day average daily amount as a rough cap proxy
    avg_amount = hist['amount'].iloc[-20:].mean()
    return avg_amount


def run_rank_analysis():
    import baostock as bs
    bs.login()

    print("=" * 70)
    print("Top20 排位贡献 & 市值分布分析")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    all_data = load_all_data()
    if all_data is None:
        print("Loading stock data from cache files...")
        all_data = {}
        cache_dirs = [PRICE_CACHE_DIR, CACHE_DIR]
        loaded_codes = set()
        for cdir in cache_dirs:
            if not os.path.exists(cdir):
                continue
            for fn in os.listdir(cdir):
                if fn.endswith('_ak.pkl'):
                    code = fn.replace('_ak.pkl', '')
                    if code in loaded_codes:
                        continue
                    fpath = os.path.join(cdir, fn)
                    try:
                        with open(fpath, 'rb') as f:
                            df = pickle.load(f)
                        if df is not None and len(df) > 100:
                            all_data[code] = df
                            loaded_codes.add(code)
                    except:
                        pass
        print(f"  Loaded {len(all_data)} stocks")

    hs300_const = get_historical_constituents('hs300')
    zz500_const = get_historical_constituents('zz500')

    month_ends = get_month_ends(all_data)
    month_ends = [d for d in month_ends if d >= pd.Timestamp(BACKTEST_START)]
    print(f"  Rebalance dates: {len(month_ends)}")

    # Track per-rank returns
    # rank_returns[rank] = list of monthly returns for that rank position
    rank_returns = {r: [] for r in range(1, TOP_N + 1)}
    # quartile tracking
    q_returns = {'Q1(1-5)': [], 'Q2(6-10)': [], 'Q3(11-15)': [], 'Q4(16-20)': []}
    # market cap tracking
    cap_data = []  # list of (date, rank, code, amount_proxy, fwd_ret)

    print("\n[2] Running rank-aware backtest...")
    for i in range(len(month_ends) - 1):
        dt = month_ends[i]
        dt_next = month_ends[i + 1]

        eligible = get_csi800_at_date(hs300_const, zz500_const, dt)
        if not eligible:
            continue

        # Compute factors
        records = []
        for code in eligible:
            if code not in all_data:
                continue
            rec = compute_factors_single(code, all_data[code], dt, None)
            if rec:
                records.append(rec)

        if len(records) < TOP_N:
            continue

        cur = pd.DataFrame(records)
        composite = build_composite_signal(cur, FACTORS)
        if composite is None or composite.notna().sum() < TOP_N:
            continue

        cur['signal'] = composite.values
        cur = cur.dropna(subset=['signal'])
        top = cur.nsmallest(TOP_N, 'signal').copy()
        top = top.reset_index(drop=True)
        top['rank'] = range(1, len(top) + 1)

        # Compute forward returns for each stock
        for _, row in top.iterrows():
            code = row['code']
            rank = row['rank']
            if code not in all_data:
                continue

            df = all_data[code]
            hist_cur = df[df.index <= dt]
            hist_nxt = df[df.index <= dt_next]
            if len(hist_cur) == 0 or len(hist_nxt) == 0:
                continue

            p_cur = hist_cur.iloc[-1]['close']
            p_nxt = hist_nxt.iloc[-1]['close']
            if p_cur <= 0:
                continue

            fwd_ret = p_nxt / p_cur - 1

            # Track by exact rank
            rank_returns[rank].append(fwd_ret)

            # Track by quartile
            if rank <= 5:
                q_returns['Q1(1-5)'].append(fwd_ret)
            elif rank <= 10:
                q_returns['Q2(6-10)'].append(fwd_ret)
            elif rank <= 15:
                q_returns['Q3(11-15)'].append(fwd_ret)
            else:
                q_returns['Q4(16-20)'].append(fwd_ret)

            # Market cap proxy
            cap_proxy = get_market_cap_proxy(code, all_data[code], dt)
            cap_data.append({
                'date': dt, 'rank': rank, 'code': code,
                'amount_20d': cap_proxy, 'fwd_ret': fwd_ret,
                'turn_20d': row.get('turn_20d', np.nan),
                'mom_12m': row.get('mom_12m', np.nan),
                'composite': row['signal'],
            })

    bs.logout()

    # ============================================================
    # Analysis 1: Per-rank average return
    # ============================================================
    print("\n" + "=" * 70)
    print("分析1: 每个排名位置的平均月收益")
    print("=" * 70)
    print(f"{'Rank':>6} {'AvgRet%':>10} {'Median%':>10} {'StdDev%':>10} {'WinRate%':>10} {'N':>6}")
    print("-" * 54)

    rank_summary = []
    for r in range(1, TOP_N + 1):
        rets = np.array(rank_returns[r])
        if len(rets) == 0:
            continue
        avg = rets.mean() * 100
        med = np.median(rets) * 100
        std = rets.std() * 100
        wr = (rets > 0).sum() / len(rets) * 100
        print(f"{'#' + str(r):>6} {avg:>10.2f} {med:>10.2f} {std:>10.2f} {wr:>10.1f} {len(rets):>6}")
        rank_summary.append({'rank': r, 'avg_ret': avg, 'median_ret': med, 'std': std, 'win_rate': wr, 'n': len(rets)})

    # ============================================================
    # Analysis 2: Quartile comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("分析2: 四分位组对比 (每组5只股)")
    print("=" * 70)
    print(f"{'Group':>12} {'AvgRet%':>10} {'Median%':>10} {'Ann.Ret%':>10} {'Sharpe':>8} {'WinRate%':>10} {'N':>6}")
    print("-" * 68)

    q_summary = {}
    for q_name, rets in q_returns.items():
        rets = np.array(rets)
        if len(rets) == 0:
            continue
        avg = rets.mean() * 100
        med = np.median(rets) * 100
        ann_ret = ((1 + rets.mean()) ** 12 - 1) * 100
        ann_vol = rets.std() * np.sqrt(12)
        sharpe = (ann_ret / 100 - 0.025) / ann_vol if ann_vol > 0 else 0
        wr = (rets > 0).sum() / len(rets) * 100
        print(f"{q_name:>12} {avg:>10.2f} {med:>10.2f} {ann_ret:>10.1f} {sharpe:>8.3f} {wr:>10.1f} {len(rets):>6}")
        q_summary[q_name] = {'avg_ret': avg, 'ann_ret': ann_ret, 'sharpe': sharpe}

    # ============================================================
    # Analysis 3: Top 10 vs Bottom 10
    # ============================================================
    print("\n" + "=" * 70)
    print("分析3: Top10 vs Bottom10 (排名1-10 vs 11-20)")
    print("=" * 70)

    top10_rets = q_returns['Q1(1-5)'] + q_returns['Q2(6-10)']
    bot10_rets = q_returns['Q3(11-15)'] + q_returns['Q4(16-20)']

    for name, rets in [('Top10(#1-10)', top10_rets), ('Bot10(#11-20)', bot10_rets)]:
        arr = np.array(rets)
        avg = arr.mean() * 100
        ann_ret = ((1 + arr.mean()) ** 12 - 1) * 100
        ann_vol = arr.std() * np.sqrt(12)
        sharpe = (ann_ret / 100 - 0.025) / ann_vol if ann_vol > 0 else 0
        wr = (arr > 0).sum() / len(arr) * 100
        print(f"  {name}: AvgMo={avg:.2f}%, Ann={ann_ret:.1f}%, Sharpe={sharpe:.3f}, WinRate={wr:.1f}%")

    # ============================================================
    # Analysis 4: Market cap distribution
    # ============================================================
    print("\n" + "=" * 70)
    print("分析4: 市值(成交额代理)分布 & 对收益的影响")
    print("=" * 70)

    cap_df = pd.DataFrame(cap_data)
    if len(cap_df) > 0 and 'amount_20d' in cap_df.columns:
        # Group by cap quintile within each month
        cap_df['cap_quintile'] = cap_df.groupby('date')['amount_20d'].transform(
            lambda x: pd.qcut(x, 5, labels=['小1', '小2', '中3', '大4', '大5'], duplicates='drop')
        )

        print("\n市值五分位 vs 平均月收益:")
        print(f"{'Cap Group':>10} {'AvgRet%':>10} {'Count':>8}")
        print("-" * 30)
        for q in ['小1', '小2', '中3', '大4', '大5']:
            sub = cap_df[cap_df['cap_quintile'] == q]
            if len(sub) > 0:
                avg = sub['fwd_ret'].mean() * 100
                print(f"{q:>10} {avg:>10.2f} {len(sub):>8}")

        # Overall cap stats
        print(f"\nTop20整体20日均成交额(万元):")
        print(f"  均值: {cap_df['amount_20d'].mean() / 10000:.0f}万")
        print(f"  中位数: {cap_df['amount_20d'].median() / 10000:.0f}万")
        print(f"  P25: {cap_df['amount_20d'].quantile(0.25) / 10000:.0f}万")
        print(f"  P75: {cap_df['amount_20d'].quantile(0.75) / 10000:.0f}万")

        # Rank vs Cap correlation
        corr = cap_df[['rank', 'amount_20d']].corr().iloc[0, 1]
        print(f"\n排名 vs 成交额 相关系数: {corr:.3f}")
        # Cap vs Return correlation
        corr2 = cap_df[['amount_20d', 'fwd_ret']].corr().iloc[0, 1]
        print(f"成交额 vs 前向收益 相关系数: {corr2:.3f}")

    # ============================================================
    # Analysis 5: Rank monotonicity check (is rank #1 truly better?)
    # ============================================================
    print("\n" + "=" * 70)
    print("分析5: 排名单调性检验 (Rank IC)")
    print("=" * 70)

    if len(cap_df) > 0:
        # For each month, compute rank-return correlation
        rank_ics = []
        for dt, grp in cap_df.groupby('date'):
            if len(grp) >= 10:
                ic = grp['rank'].corr(grp['fwd_ret'])
                rank_ics.append(ic)

        if rank_ics:
            rank_ics = np.array(rank_ics)
            icir = rank_ics.mean() / rank_ics.std() if rank_ics.std() > 0 else 0
            print(f"  Rank-Return IC (monthly avg): {rank_ics.mean():.4f}")
            print(f"  Rank-Return ICIR: {icir:.3f}")
            print(f"  IC > 0 比例: {(rank_ics > 0).sum() / len(rank_ics) * 100:.1f}%")
            print(f"  (IC < 0 表示排名越靠前收益越高, 符合预期)")

        # Also check composite signal vs return
        sig_ics = []
        for dt, grp in cap_df.groupby('date'):
            if len(grp) >= 10:
                ic = grp['composite'].corr(grp['fwd_ret'])
                sig_ics.append(ic)
        if sig_ics:
            sig_ics = np.array(sig_ics)
            print(f"\n  Composite Signal-Return IC: {sig_ics.mean():.4f}")
            print(f"  Signal ICIR: {sig_ics.mean() / sig_ics.std():.3f}" if sig_ics.std() > 0 else "")

    # ============================================================
    # Analysis 6: Would a weighted portfolio (by rank) do better?
    # ============================================================
    print("\n" + "=" * 70)
    print("分析6: 等权 vs 排名加权 vs Top10 对比")
    print("=" * 70)

    # Reconstruct monthly portfolio returns under different schemes
    # Equal weight (current), rank-weighted, top10 only
    ew_monthly = []
    rw_monthly = []
    t10_monthly = []

    for i in range(len(month_ends) - 1):
        dt = month_ends[i]
        sub = cap_df[cap_df['date'] == dt]
        if len(sub) < TOP_N:
            continue

        rets = sub.sort_values('rank')['fwd_ret'].values

        # Equal weight
        ew_ret = np.mean(rets)
        ew_monthly.append(ew_ret)

        # Rank weighted: rank 1 gets weight 20, rank 20 gets weight 1
        weights = np.array([TOP_N + 1 - r for r in range(1, TOP_N + 1)])
        weights = weights / weights.sum()
        rw_ret = np.sum(weights * rets[:TOP_N])
        rw_monthly.append(rw_ret)

        # Top 10 only
        t10_ret = np.mean(rets[:10])
        t10_monthly.append(t10_ret)

    for name, rets in [('EqualWeight(Top20)', ew_monthly),
                        ('RankWeight(Top20)', rw_monthly),
                        ('Top10Only', t10_monthly)]:
        arr = np.array(rets)
        if len(arr) < 12:
            continue
        ann_ret = ((1 + arr.mean()) ** 12 - 1) * 100
        ann_vol = arr.std() * np.sqrt(12)
        sharpe = (ann_ret / 100 - 0.025) / ann_vol if ann_vol > 0 else 0
        cum = np.cumprod(1 + arr)
        dd = cum / np.maximum.accumulate(cum) - 1
        max_dd = dd.min() * 100
        wr = (arr > 0).sum() / len(arr) * 100
        print(f"  {name:>20}: CAGR={ann_ret:.1f}%, Sharpe={sharpe:.3f}, MaxDD={max_dd:.1f}%, WinRate={wr:.1f}%")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return rank_summary, q_summary, cap_df


if __name__ == '__main__':
    run_rank_analysis()
