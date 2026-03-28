#!/usr/bin/env python3
"""
动量+成长 双因子策略研究
股票池: 沪深300 + 中证500 (中大盘)
类似创成长指数，但扩展到全市场中大盘

成长因子 (从已有PE/PB/PS推导):
  1. EPS增长率 = EPS_TTM变化率 (EPS = Price/PE_TTM)
  2. Revenue增长率 = Revenue/Share变化率 (Rev = Price/PS_TTM)
  3. ROE变化 = (1/PB) / (1/PE) 变化趋势
  4. 综合成长 = 多因子加权

动量因子:
  1. 价格动量 (N周涨幅, skip 1周)
  2. 52周新高距离

组合方式:
  - 纯成长
  - 纯动量
  - 成长+动量 (不同权重)
  - 等权基准

数据源: SQLite (已缓存的周线+日度估值)
回测: 2021~2026 (估值数据从2020-06开始, 需要半年计算成长)
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import os
import sys
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DB_PATH = os.path.join(DATA_DIR, 'stock_data.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

# ============================================================
# 1. Load data
# ============================================================
def load_all_data():
    conn = get_db()

    # Get CSI300 + CSI500 constituents
    stocks_300 = conn.execute(
        "SELECT DISTINCT stock_code FROM index_constituents WHERE index_code='000300'"
    ).fetchall()
    stocks_500 = conn.execute(
        "SELECT DISTINCT stock_code FROM index_constituents WHERE index_code='000905'"
    ).fetchall()
    all_codes = list(set([r[0] for r in stocks_300] + [r[0] for r in stocks_500]))
    print(f"Stock pool: CSI300({len(stocks_300)}) + CSI500({len(stocks_500)}) = {len(all_codes)} unique")

    # Weekly prices
    weekly = pd.read_sql(
        "SELECT code, date, close, volume FROM stock_weekly WHERE date >= '2020-01-01'",
        conn
    )
    weekly['date'] = pd.to_datetime(weekly['date'])
    weekly = weekly[weekly['code'].isin(all_codes)]

    price_pivot = weekly.pivot(index='date', columns='code', values='close').sort_index()
    volume_pivot = weekly.pivot(index='date', columns='code', values='volume').sort_index()
    print(f"Weekly prices: {price_pivot.shape[0]} weeks x {price_pivot.shape[1]} stocks")

    # Daily valuations -> resample to weekly (use Friday/last day of week)
    val = pd.read_sql(
        "SELECT code, date, pe_ttm, pb_mrq, ps_ttm FROM stock_daily_valuation WHERE date >= '2020-01-01'",
        conn
    )
    val['date'] = pd.to_datetime(val['date'])
    val = val[val['code'].isin(all_codes)]

    # For each stock, resample to weekly (last value per week)
    val['week'] = val['date'].dt.to_period('W').dt.end_time.dt.normalize()
    val_weekly = val.groupby(['code', 'week']).last().reset_index()

    pe_pivot = val_weekly.pivot(index='week', columns='code', values='pe_ttm').sort_index()
    pb_pivot = val_weekly.pivot(index='week', columns='code', values='pb_mrq').sort_index()
    ps_pivot = val_weekly.pivot(index='week', columns='code', values='ps_ttm').sort_index()

    # Align indices - use weekly price dates
    # Map each price date to nearest valuation week
    common_codes = list(set(price_pivot.columns) & set(pe_pivot.columns))
    print(f"Stocks with both price and valuation data: {len(common_codes)}")

    # Reindex valuation to price dates using forward fill
    pe_aligned = pe_pivot.reindex(price_pivot.index, method='ffill')[common_codes]
    pb_aligned = pb_pivot.reindex(price_pivot.index, method='ffill')[common_codes]
    ps_aligned = ps_pivot.reindex(price_pivot.index, method='ffill')[common_codes]
    price_df = price_pivot[common_codes]
    vol_df = volume_pivot[common_codes] if set(common_codes).issubset(volume_pivot.columns) else None

    conn.close()
    return price_df, pe_aligned, pb_aligned, ps_aligned, vol_df, common_codes

# ============================================================
# 2. Compute growth factors
# ============================================================
def compute_growth_factors(price_df, pe_df, pb_df, ps_df, lookback_quarters=4):
    """
    Derive growth factors from price + valuation multiples.

    EPS_TTM = Price / PE_TTM  -> EPS growth = EPS now / EPS N quarters ago - 1
    Rev_TTM = Price / PS_TTM  -> Rev growth = Rev now / Rev N quarters ago - 1
    ROE = EPS/BPS = (Price/PE) / (Price/PB) = PB/PE

    lookback_quarters: how many quarters back (4 = YoY, 2 = HoH)
    """
    weeks_back = lookback_quarters * 13  # ~13 weeks per quarter

    # EPS = Price / PE (handle negative PE by setting EPS to NaN)
    pe_clean = pe_df.replace(0, np.nan)
    pe_clean = pe_clean.where(pe_clean > 0)  # Only positive PE (profitable companies)
    eps = price_df / pe_clean

    # Revenue per share = Price / PS
    ps_clean = ps_df.replace(0, np.nan)
    ps_clean = ps_clean.where(ps_clean > 0)
    rev = price_df / ps_clean

    # ROE proxy = PB / PE (= EPS/BPS)
    roe = pb_df / pe_clean

    # Growth rates (YoY or specified lookback)
    eps_growth = eps / eps.shift(weeks_back) - 1
    rev_growth = rev / rev.shift(weeks_back) - 1
    roe_change = roe - roe.shift(weeks_back)  # Absolute change in ROE

    # Clean extreme values
    for df in [eps_growth, rev_growth, roe_change]:
        df.clip(-5, 50, inplace=True)  # Cap at -500% to +5000%

    return eps_growth, rev_growth, roe_change

# ============================================================
# 3. Compute momentum factors
# ============================================================
def compute_momentum(price_df, lookback=12, skip=1):
    """Price momentum: return over lookback weeks, skipping most recent skip weeks"""
    ret = price_df.shift(skip) / price_df.shift(lookback + skip) - 1
    return ret

def compute_high52w(price_df):
    """Distance from 52-week high (0 = at high, -0.5 = 50% below high)"""
    high_52w = price_df.rolling(52, min_periods=26).max()
    dist = price_df / high_52w - 1  # Negative = below high
    return dist

# ============================================================
# 4. Ranking and signal construction
# ============================================================
def rank_pct(df):
    """Cross-sectional percentile rank (0 to 1, higher = better)"""
    return df.rank(axis=1, pct=True)

def composite_signal(growth_rank, mom_rank, growth_weight=0.5):
    """Combine growth and momentum ranks"""
    mom_weight = 1.0 - growth_weight
    return growth_weight * growth_rank + mom_weight * mom_rank

# ============================================================
# 5. Backtest engine
# ============================================================
def backtest(price_df, signal_df, top_n=15, rebal_weeks=4, start_date='2021-01-01'):
    """
    Long-only top_n stocks by signal, equal weight, rebalance every rebal_weeks.
    Returns portfolio value series.
    """
    dates = price_df.index[price_df.index >= pd.Timestamp(start_date)]
    if len(dates) < 10:
        return None

    portfolio_value = [1.0]
    current_holdings = []
    rebal_counter = 0

    for i in range(len(dates) - 1):
        dt = dates[i]
        dt_next = dates[i + 1]

        # Rebalance?
        if rebal_counter == 0 or len(current_holdings) == 0:
            sig = signal_df.loc[dt].dropna()
            if len(sig) < top_n:
                # Not enough signals, hold cash
                portfolio_value.append(portfolio_value[-1])
                rebal_counter = (rebal_counter + 1) % rebal_weeks
                continue
            top_stocks = sig.nlargest(top_n).index.tolist()
            current_holdings = top_stocks
            rebal_counter = 1
        else:
            rebal_counter = (rebal_counter + 1) % rebal_weeks

        # Calculate returns
        if len(current_holdings) == 0:
            portfolio_value.append(portfolio_value[-1])
            continue

        p_now = price_df.loc[dt, current_holdings]
        p_next = price_df.loc[dt_next, current_holdings]
        valid = p_now.notna() & p_next.notna() & (p_now > 0)

        if valid.sum() == 0:
            portfolio_value.append(portfolio_value[-1])
            continue

        rets = (p_next[valid] / p_now[valid] - 1).mean()
        portfolio_value.append(portfolio_value[-1] * (1 + rets))

    idx = dates[:len(portfolio_value)]
    return pd.Series(portfolio_value, index=idx)

def calc_metrics(pv):
    """Calculate CAGR, MDD, Sharpe, Calmar"""
    if pv is None or len(pv) < 20:
        return None

    years = (pv.index[-1] - pv.index[0]).days / 365.25
    if years < 0.5:
        return None

    total_ret = pv.iloc[-1] / pv.iloc[0]
    cagr = total_ret ** (1/years) - 1

    # MDD
    cummax = pv.cummax()
    dd = pv / cummax - 1
    mdd = dd.min()

    # Sharpe (weekly -> annualized)
    weekly_rets = pv.pct_change().dropna()
    if weekly_rets.std() == 0:
        sharpe = 0
    else:
        sharpe = weekly_rets.mean() / weekly_rets.std() * np.sqrt(52)

    calmar = cagr / abs(mdd) if mdd != 0 else 0

    # Annual returns
    annual = {}
    for year in range(pv.index[0].year, pv.index[-1].year + 1):
        yr_data = pv[pv.index.year == year]
        if len(yr_data) >= 2:
            annual[year] = yr_data.iloc[-1] / yr_data.iloc[0] - 1

    return {
        'cagr': round(cagr * 100, 1),
        'mdd': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'annual': {str(k): round(v*100, 1) for k, v in annual.items()}
    }

# ============================================================
# 6. Main research
# ============================================================
def main():
    print("=" * 60)
    print("动量+成长 双因子策略研究")
    print("股票池: 沪深300 + 中证500")
    print("=" * 60)

    # Load data
    price_df, pe_df, pb_df, ps_df, vol_df, codes = load_all_data()

    # Compute factors
    print("\n--- Computing growth factors ---")
    # YoY growth (4 quarters = 52 weeks)
    eps_g_yoy, rev_g_yoy, roe_c_yoy = compute_growth_factors(price_df, pe_df, pb_df, ps_df, lookback_quarters=4)
    # HoH growth (2 quarters = 26 weeks)
    eps_g_hoh, rev_g_hoh, roe_c_hoh = compute_growth_factors(price_df, pe_df, pb_df, ps_df, lookback_quarters=2)

    print("--- Computing momentum factors ---")
    mom_12w = compute_momentum(price_df, lookback=12, skip=1)
    mom_20w = compute_momentum(price_df, lookback=20, skip=1)
    high52 = compute_high52w(price_df)

    # Rank all factors
    print("--- Ranking factors ---")
    # Growth ranks (higher growth = better)
    r_eps_yoy = rank_pct(eps_g_yoy)
    r_rev_yoy = rank_pct(rev_g_yoy)
    r_roe_yoy = rank_pct(roe_c_yoy)
    r_eps_hoh = rank_pct(eps_g_hoh)
    r_rev_hoh = rank_pct(rev_g_hoh)

    # Composite growth ranks
    r_growth_yoy = (r_eps_yoy + r_rev_yoy + r_roe_yoy) / 3  # Equal weight 3 factors
    r_growth_ep_rev = (r_eps_yoy + r_rev_yoy) / 2  # EPS + Revenue only
    r_growth_hoh = (r_eps_hoh + r_rev_hoh) / 2

    # Momentum ranks
    r_mom12 = rank_pct(mom_12w)
    r_mom20 = rank_pct(mom_20w)
    r_high52 = rank_pct(high52)

    # ============================================================
    # Define all strategy configs to test
    # ============================================================
    strategies = {}

    # --- Pure momentum baselines ---
    strategies['纯动量_12w'] = r_mom12
    strategies['纯动量_20w'] = r_mom20
    strategies['52周新高'] = r_high52

    # --- Pure growth ---
    strategies['纯成长_YoY(EPS+Rev+ROE)'] = r_growth_yoy
    strategies['纯成长_YoY(EPS+Rev)'] = r_growth_ep_rev
    strategies['纯成长_HoH(EPS+Rev)'] = r_growth_hoh
    strategies['纯EPS成长_YoY'] = r_eps_yoy
    strategies['纯营收成长_YoY'] = r_rev_yoy

    # --- Momentum + Growth combos ---
    for gw in [0.3, 0.4, 0.5, 0.6, 0.7]:
        mw = 1 - gw
        label_gw = int(gw * 10)
        label_mw = int(mw * 10)

        # Mom12 + Growth YoY
        strategies[f'动量12w+成长YoY_G{label_gw}M{label_mw}'] = composite_signal(r_growth_yoy, r_mom12, gw)
        # Mom12 + Growth EP+Rev
        strategies[f'动量12w+EP成长_G{label_gw}M{label_mw}'] = composite_signal(r_growth_ep_rev, r_mom12, gw)
        # Mom20 + Growth YoY
        strategies[f'动量20w+成长YoY_G{label_gw}M{label_mw}'] = composite_signal(r_growth_yoy, r_mom20, gw)
        # Mom12 + HoH Growth
        strategies[f'动量12w+成长HoH_G{label_gw}M{label_mw}'] = composite_signal(r_growth_hoh, r_mom12, gw)
        # 52w High + Growth
        strategies[f'52wH+成长YoY_G{label_gw}M{label_mw}'] = composite_signal(r_growth_yoy, r_high52, gw)

    # --- Special: momentum + single growth factor ---
    for gw in [0.4, 0.5, 0.6]:
        label_gw = int(gw * 10)
        label_mw = int((1-gw) * 10)
        strategies[f'动量12w+EPS成长_G{label_gw}M{label_mw}'] = composite_signal(r_eps_yoy, r_mom12, gw)
        strategies[f'动量12w+营收成长_G{label_gw}M{label_mw}'] = composite_signal(r_rev_yoy, r_mom12, gw)

    print(f"\nTotal strategies: {len(strategies)}")

    # ============================================================
    # Run backtests
    # ============================================================
    configs = [
        {'top_n': 10, 'rebal_weeks': 2},
        {'top_n': 10, 'rebal_weeks': 4},
        {'top_n': 15, 'rebal_weeks': 2},
        {'top_n': 15, 'rebal_weeks': 4},
        {'top_n': 20, 'rebal_weeks': 4},
        {'top_n': 30, 'rebal_weeks': 4},
    ]

    # Also test equal-weight benchmark
    print("\n--- Running backtests (2021~2026) ---")

    all_results = []
    total_tests = len(strategies) * len(configs)
    test_num = 0

    for strat_name, signal in strategies.items():
        for cfg in configs:
            test_num += 1
            if test_num % 50 == 0:
                print(f"  Progress: {test_num}/{total_tests}...")

            pv = backtest(price_df, signal, top_n=cfg['top_n'],
                         rebal_weeks=cfg['rebal_weeks'], start_date='2021-01-01')
            metrics = calc_metrics(pv)
            if metrics:
                result = {
                    'strategy': strat_name,
                    'top_n': cfg['top_n'],
                    'rebal_weeks': cfg['rebal_weeks'],
                    **metrics
                }
                all_results.append(result)

    # Equal weight benchmark
    eq_signal = pd.DataFrame(1.0, index=price_df.index, columns=price_df.columns)
    # For equal weight, use all stocks
    for cfg in [{'top_n': 700, 'rebal_weeks': 4}]:
        pv = backtest(price_df, eq_signal, top_n=700, rebal_weeks=4, start_date='2021-01-01')
        metrics = calc_metrics(pv)
        if metrics:
            all_results.append({
                'strategy': '等权持有全部',
                'top_n': 700,
                'rebal_weeks': 4,
                **metrics
            })

    # ============================================================
    # Analyze results
    # ============================================================
    df = pd.DataFrame(all_results)

    if len(df) == 0:
        print("ERROR: No results!")
        return

    print(f"\n{'='*80}")
    print(f"Total results: {len(df)}")
    print(f"{'='*80}")

    # Best by Sharpe
    print("\n=== TOP 20 by Sharpe ===")
    top_sharpe = df.nlargest(20, 'sharpe')
    for _, r in top_sharpe.iterrows():
        print(f"  {r['strategy']:40s} Top{r['top_n']:2d} R{r['rebal_weeks']}w | "
              f"CAGR={r['cagr']:6.1f}% MDD={r['mdd']:6.1f}% Sharpe={r['sharpe']:.3f} Calmar={r['calmar']:.3f}")

    # Best by Calmar
    print("\n=== TOP 20 by Calmar ===")
    top_calmar = df.nlargest(20, 'calmar')
    for _, r in top_calmar.iterrows():
        print(f"  {r['strategy']:40s} Top{r['top_n']:2d} R{r['rebal_weeks']}w | "
              f"CAGR={r['cagr']:6.1f}% MDD={r['mdd']:6.1f}% Sharpe={r['sharpe']:.3f} Calmar={r['calmar']:.3f}")

    # Best by CAGR
    print("\n=== TOP 20 by CAGR ===")
    top_cagr = df.nlargest(20, 'cagr')
    for _, r in top_cagr.iterrows():
        print(f"  {r['strategy']:40s} Top{r['top_n']:2d} R{r['rebal_weeks']}w | "
              f"CAGR={r['cagr']:6.1f}% MDD={r['mdd']:6.1f}% Sharpe={r['sharpe']:.3f} Calmar={r['calmar']:.3f}")

    # Compare pure momentum vs pure growth vs combo
    print("\n=== Category comparison (best Sharpe per category) ===")
    categories = {
        '纯动量': df[df['strategy'].str.startswith('纯动量') | df['strategy'].str.startswith('52周')],
        '纯成长': df[df['strategy'].str.startswith('纯成长') | df['strategy'].str.startswith('纯EPS') | df['strategy'].str.startswith('纯营收')],
        '动量+成长': df[df['strategy'].str.contains(r'\+')],
        '等权基准': df[df['strategy'] == '等权持有全部'],
    }

    for cat_name, cat_df in categories.items():
        if len(cat_df) == 0:
            continue
        best = cat_df.loc[cat_df['sharpe'].idxmax()]
        print(f"\n  [{cat_name}] Best Sharpe:")
        print(f"    {best['strategy']} Top{best['top_n']} R{best['rebal_weeks']}w")
        print(f"    CAGR={best['cagr']:.1f}% MDD={best['mdd']:.1f}% Sharpe={best['sharpe']:.3f} Calmar={best['calmar']:.3f}")
        if 'annual' in best and best['annual']:
            ann_str = ' | '.join([f"{y}:{v:+.1f}%" for y, v in sorted(best['annual'].items())])
            print(f"    Annual: {ann_str}")

        # Also show best Calmar
        best_c = cat_df.loc[cat_df['calmar'].idxmax()]
        if best_c.name != best.name:
            print(f"  [{cat_name}] Best Calmar:")
            print(f"    {best_c['strategy']} Top{best_c['top_n']} R{best_c['rebal_weeks']}w")
            print(f"    CAGR={best_c['cagr']:.1f}% MDD={best_c['mdd']:.1f}% Sharpe={best_c['sharpe']:.3f} Calmar={best_c['calmar']:.3f}")

    # Specific comparison: same config, different signals
    print("\n=== Head-to-head: Top15 R4w (monthly rebalance) ===")
    h2h = df[(df['top_n'] == 15) & (df['rebal_weeks'] == 4)].sort_values('sharpe', ascending=False)
    for _, r in h2h.head(25).iterrows():
        ann_str = ''
        if r.get('annual'):
            ann_str = ' | '.join([f"{y}:{v:+.1f}%" for y, v in sorted(r['annual'].items())])
        print(f"  {r['strategy']:40s} CAGR={r['cagr']:6.1f}% MDD={r['mdd']:6.1f}% "
              f"Sharpe={r['sharpe']:.3f} Calmar={r['calmar']:.3f}  {ann_str}")

    # Save results
    output_path = os.path.join(DATA_DIR, 'momentum_growth_research.json')
    # Convert annual dict properly
    results_serializable = []
    for r in all_results:
        r_copy = dict(r)
        results_serializable.append(r_copy)

    with open(output_path, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'pool': 'CSI300+CSI500',
            'backtest_period': '2021~2026',
            'total_strategies': len(strategies),
            'total_configs': len(configs),
            'total_results': len(all_results),
            'results': results_serializable
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 60)
    print("DONE")

if __name__ == '__main__':
    main()
