#!/usr/bin/env python3
"""
Factor Timing Strategy Research v2
3 strategies: Macro Timing, Sentiment Timing, ETF Rotation
Data: baostock (index prices), akshare (macro)
Period: 2015-2026, IS: 2015-2022, OOS: 2023-2026
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import akshare as ak
import baostock as bs
import json, re, traceback
from datetime import datetime

# ============================================================
# DATA
# ============================================================

def get_index_price(code, start='2012-01-01', end='2026-03-25'):
    """Get index daily price from baostock (INDEX codes like sh.000300)"""
    rs = bs.query_history_k_data_plus(
        code, "date,open,high,low,close,volume,amount",
        start_date=start, end_date=end, frequency="d"
    )
    data = []
    while (rs.error_code == '0') & rs.next():
        data.append(rs.get_row_data())
    df = pd.DataFrame(data, columns=rs.fields)
    for c in ['open','high','low','close','volume','amount']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df = df[df['close'] > 0]
    return df

def get_macro_data():
    """Get M1, M2, bond yields"""
    results = {}

    # M1, M2
    try:
        print("  Fetching M1/M2...")
        df = ak.macro_china_money_supply()
        # Parse "2026年02月份" -> datetime
        def parse_cn_date(s):
            m = re.match(r'(\d{4})年(\d{2})月份?', str(s))
            if m:
                return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-01")
            return pd.NaT
        df['date'] = df['月份'].apply(parse_cn_date)
        df = df.dropna(subset=['date']).set_index('date').sort_index()
        df['m1_yoy'] = pd.to_numeric(df['货币(M1)-同比增长'], errors='coerce')
        df['m2_yoy'] = pd.to_numeric(df['货币和准货币(M2)-同比增长'], errors='coerce')
        df['m1_m2_scissors'] = df['m1_yoy'] - df['m2_yoy']
        results['money'] = df[['m1_yoy', 'm2_yoy', 'm1_m2_scissors']].dropna()
        print(f"    OK: {len(results['money'])} rows, {results['money'].index[0]} to {results['money'].index[-1]}")
    except Exception as e:
        print(f"    M1/M2 error: {e}")

    # Bond yields
    try:
        print("  Fetching bond yields...")
        df = ak.bond_zh_us_rate(start_date="20100101")
        df['date'] = pd.to_datetime(df['日期'])
        df = df.set_index('date').sort_index()
        df['yield_10y'] = pd.to_numeric(df['中国国债收益率10年'], errors='coerce')
        df['yield_2y'] = pd.to_numeric(df['中国国债收益率2年'], errors='coerce')
        df['term_spread'] = df['yield_10y'] - df['yield_2y']
        results['bond'] = df[['yield_10y', 'yield_2y', 'term_spread']].dropna()
        print(f"    OK: {len(results['bond'])} rows, {results['bond'].index[0]} to {results['bond'].index[-1]}")
    except Exception as e:
        print(f"    Bond yield error: {e}")

    return results

# ============================================================
# BACKTEST ENGINE
# ============================================================

def calc_metrics(strat_ret, name):
    """Calculate performance metrics from daily return series"""
    if len(strat_ret) < 60:
        return None
    cum = (1 + strat_ret).cumprod()
    total = cum.iloc[-1] - 1
    years = len(strat_ret) / 252
    cagr = (1 + total) ** (1/max(years, 0.01)) - 1
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe = (strat_ret.mean() * 252) / ann_vol if ann_vol > 0 else 0
    peak = cum.cummax()
    mdd = ((cum - peak) / peak).min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    return {
        'strategy': name,
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'mdd': round(mdd * 100, 2),
        'calmar': round(calmar, 3),
        'years': round(years, 1),
        'start': str(strat_ret.index[0].date()),
        'end': str(strat_ret.index[-1].date()),
    }

def timing_backtest(price_df, signal_series, name, cost_bps=10):
    """Timing backtest: signal=1 long, signal=0 cash. Next-day execution."""
    common = price_df.index.intersection(signal_series.index)
    if len(common) < 100:
        return []

    price = price_df.loc[common]
    signal = signal_series.loc[common].shift(1).fillna(0)  # next-day execution
    ret = price['close'].pct_change()
    cost = cost_bps / 10000
    turnover = signal.diff().abs().fillna(0)
    strat_ret = (signal * ret - turnover * cost).dropna()

    # Buy-and-hold
    bh_ret = ret.dropna()

    results = []
    split = pd.Timestamp('2023-01-01')

    for period, sr in [('FULL', strat_ret),
                        ('IS', strat_ret[strat_ret.index < split]),
                        ('OOS', strat_ret[strat_ret.index >= split])]:
        r = calc_metrics(sr, f"{name} [{period}]")
        if r:
            # Add in-market % and trades
            sig = signal.loc[sr.index]
            r['pct_in_market'] = round(sig.mean() * 100, 1)
            r['trades'] = int(turnover.loc[sr.index].sum() / 2)

            # B&H comparison
            bh = bh_ret.loc[sr.index]
            bh_m = calc_metrics(bh, 'bh')
            if bh_m:
                r['bh_cagr'] = bh_m['cagr']
                r['bh_sharpe'] = bh_m['sharpe']
                r['bh_mdd'] = bh_m['mdd']
            results.append(r)

    return results

# ============================================================
# STRATEGY 1: MACRO TIMING
# ============================================================

def run_macro_timing(idx_prices, macro_data):
    print("\n" + "="*60)
    print("STRATEGY 1: MACRO TIMING (宏观择时)")
    print("="*60)

    all_results = []

    for idx_name, price_df in idx_prices.items():
        print(f"\n  --- {idx_name} ---")

        # 1a. M1-M2 Scissors threshold
        if 'money' in macro_data:
            money = macro_data['money'].resample('D').ffill()

            for thresh in [-3, -2, -1, 0, 1, 2, 3]:
                sig = (money['m1_m2_scissors'] > thresh).astype(float)
                sig.name = 'signal'
                r = timing_backtest(price_df, sig, f"M1M2>{thresh} | {idx_name}")
                all_results.extend(r)

            # M1-M2 rising (3-month momentum)
            money_m = macro_data['money'].copy()
            money_m['mom3'] = money_m['m1_m2_scissors'] - money_m['m1_m2_scissors'].shift(3)
            money_m['mom6'] = money_m['m1_m2_scissors'] - money_m['m1_m2_scissors'].shift(6)
            md = money_m.resample('D').ffill()

            for col, label in [('mom3', '3m'), ('mom6', '6m')]:
                sig = (md[col] > 0).astype(float)
                sig.name = 'signal'
                r = timing_backtest(price_df, sig, f"M1M2_Rising({label}) | {idx_name}")
                all_results.extend(r)

            # M1 YoY level
            for thresh in [0, 3, 5, 8]:
                sig = (money['m1_yoy'] > thresh).astype(float)
                sig.name = 'signal'
                r = timing_backtest(price_df, sig, f"M1>{thresh}% | {idx_name}")
                all_results.extend(r)

        # 1b. Term Spread
        if 'bond' in macro_data:
            bond = macro_data['bond'].copy()

            # Level threshold
            for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
                sig = (bond['term_spread'] > thresh).astype(float)
                sig.name = 'signal'
                r = timing_backtest(price_df, sig, f"TS>{thresh} | {idx_name}")
                all_results.extend(r)

            # MA crossover
            for window in [20, 60, 120, 250]:
                ma = bond['term_spread'].rolling(window).mean()
                sig = (bond['term_spread'] > ma).astype(float)
                sig.name = 'signal'
                r = timing_backtest(price_df, sig, f"TS>MA{window} | {idx_name}")
                all_results.extend(r)

            # 10Y yield level (low yield = loose policy = bullish)
            for thresh in [2.5, 2.8, 3.0, 3.2, 3.5]:
                sig = (bond['yield_10y'] < thresh).astype(float)
                sig.name = 'signal'
                r = timing_backtest(price_df, sig, f"10Y<{thresh} | {idx_name}")
                all_results.extend(r)

            # 10Y yield MA (declining = easing cycle)
            for window in [60, 120, 250]:
                ma = bond['yield_10y'].rolling(window).mean()
                sig = (bond['yield_10y'] < ma).astype(float)
                sig.name = 'signal'
                r = timing_backtest(price_df, sig, f"10Y<MA{window} | {idx_name}")
                all_results.extend(r)

        # 1c. Combined: M1-M2 + Term Spread
        if 'money' in macro_data and 'bond' in macro_data:
            money_d = macro_data['money'].resample('D').ffill()
            bond = macro_data['bond']
            combined = money_d[['m1_m2_scissors']].join(bond[['term_spread', 'yield_10y']], how='inner')

            for m_thresh in [-1, 0, 1]:
                for t_thresh in [0.3, 0.5]:
                    # Both bullish
                    sig = ((combined['m1_m2_scissors'] > m_thresh) &
                           (combined['term_spread'] > t_thresh)).astype(float)
                    sig.name = 'signal'
                    r = timing_backtest(price_df, sig, f"M1M2>{m_thresh}&TS>{t_thresh} | {idx_name}")
                    all_results.extend(r)

                    # Either bullish
                    sig = ((combined['m1_m2_scissors'] > m_thresh) |
                           (combined['term_spread'] > t_thresh)).astype(float)
                    sig.name = 'signal'
                    r = timing_backtest(price_df, sig, f"M1M2>{m_thresh}|TS>{t_thresh} | {idx_name}")
                    all_results.extend(r)

            # M1-M2 rising + yield declining
            money_m = macro_data['money'].copy()
            money_m['m1m2_rising'] = (money_m['m1_m2_scissors'] > money_m['m1_m2_scissors'].shift(3))
            md = money_m[['m1m2_rising']].resample('D').ffill()

            for w in [60, 120]:
                yield_ma = bond['yield_10y'].rolling(w).mean()
                yield_down = (bond['yield_10y'] < yield_ma)
                comb2 = md[['m1m2_rising']].join(yield_down.rename('yield_down'), how='inner')

                sig = (comb2['m1m2_rising'] & comb2['yield_down']).astype(float)
                sig.name = 'signal'
                r = timing_backtest(price_df, sig, f"M1M2_Up&10Y<MA{w} | {idx_name}")
                all_results.extend(r)

                sig = (comb2['m1m2_rising'] | comb2['yield_down']).astype(float)
                sig.name = 'signal'
                r = timing_backtest(price_df, sig, f"M1M2_Up|10Y<MA{w} | {idx_name}")
                all_results.extend(r)

    print(f"\n  Total macro timing results: {len(all_results)}")
    return all_results

# ============================================================
# STRATEGY 2: SENTIMENT TIMING
# ============================================================

def run_sentiment_timing(idx_prices, macro_data):
    """
    Sentiment timing using price-derived and bond-based indicators:
    - Equity-Bond Yield Gap proxy: use rolling PE estimate from price/earnings
    - Since we can't get reliable PE history, use inverse of rolling return as proxy
    - Also: price-based sentiment (RSI, distance from MA, new highs ratio)
    """
    print("\n" + "="*60)
    print("STRATEGY 2: SENTIMENT TIMING (情绪择时)")
    print("="*60)

    all_results = []

    for idx_name, price_df in idx_prices.items():
        print(f"\n  --- {idx_name} ---")
        close = price_df['close']

        # 2a. Equity-Bond Yield Gap proxy
        # Use 1/(P/10Y-avg-P) as earnings yield proxy, compare with bond yield
        # Simpler: use index close / 252d MA as "valuation" proxy
        if 'bond' in macro_data:
            bond = macro_data['bond']
            # Align
            both = pd.DataFrame({'close': close}).join(bond[['yield_10y']], how='inner')

            if len(both) > 504:
                # Proxy: when index is below long-term MA AND bond yield is low -> cheap
                for price_ma in [120, 250, 500]:
                    ma = both['close'].rolling(price_ma).mean()
                    price_below_ma = both['close'] < ma  # relatively cheap

                    for y_thresh in [2.5, 3.0, 3.5]:
                        yield_low = both['yield_10y'] < y_thresh  # loose policy

                        # Buy when cheap + loose policy
                        sig = (price_below_ma & yield_low).astype(float)
                        sig.name = 'signal'
                        r = timing_backtest(price_df, sig, f"Below_MA{price_ma}&10Y<{y_thresh} | {idx_name}")
                        all_results.extend(r)

                # Earnings yield proxy: use dividend yield proxy = 1/PE_rolling
                # Approximate: PE ~ price / trailing_earnings ~ price / (price * avg_return)
                # Actually just use: price percentile as valuation proxy
                for window in [504, 756]:  # 2y, 3y
                    pct_rank = both['close'].rolling(window).apply(lambda x: (x[-1] > x).mean(), raw=True)

                    # "Equity cheap" when in bottom percentile
                    for cheap_pct in [0.3, 0.4, 0.5]:
                        sig = (pct_rank < cheap_pct).astype(float)
                        sig.name = 'signal'
                        label = f"{window//252}y"
                        r = timing_backtest(price_df, sig, f"PriceRank<P{int(cheap_pct*100)}({label}) | {idx_name}")
                        all_results.extend(r)

                    # Combined: cheap + yield declining
                    for w in [60, 120]:
                        yield_ma = both['yield_10y'].rolling(w).mean()
                        yield_down = both['yield_10y'] < yield_ma

                        sig = ((pct_rank < 0.4) & yield_down).astype(float)
                        sig.name = 'signal'
                        r = timing_backtest(price_df, sig, f"Cheap40({window//252}y)&10YDown{w} | {idx_name}")
                        all_results.extend(r)

        # 2b. Price-based sentiment indicators
        # RSI
        for period in [14, 30, 60]:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - 100 / (1 + rs)

            # Oversold = buy, overbought = sell
            for low, high in [(30, 70), (20, 80), (40, 60)]:
                # State machine: buy when RSI crosses above low, sell when crosses above high
                position = pd.Series(0.0, index=rsi.index)
                in_pos = False
                for i in range(1, len(rsi)):
                    if pd.isna(rsi.iloc[i]):
                        position.iloc[i] = 0
                        continue
                    if not in_pos and rsi.iloc[i] < low:
                        in_pos = True
                    elif in_pos and rsi.iloc[i] > high:
                        in_pos = False
                    position.iloc[i] = 1.0 if in_pos else 0.0

                position.name = 'signal'
                r = timing_backtest(price_df, position, f"RSI{period}({low},{high}) | {idx_name}")
                all_results.extend(r)

        # 2c. Distance from MA (mean reversion)
        for ma_window in [20, 60, 120, 250]:
            ma = close.rolling(ma_window).mean()
            dist = (close - ma) / ma * 100  # % distance

            # Buy when below MA by threshold
            for below_thresh in [-3, -5, -8, -10, -15]:
                sig = (dist < below_thresh).astype(float)
                sig.name = 'signal'
                r = timing_backtest(price_df, sig, f"Dist_MA{ma_window}<{below_thresh}% | {idx_name}")
                all_results.extend(r)

            # Buy when above MA (trend following)
            sig = (dist > 0).astype(float)
            sig.name = 'signal'
            r = timing_backtest(price_df, sig, f"Above_MA{ma_window} | {idx_name}")
            all_results.extend(r)

        # 2d. Volatility regime
        for vol_window in [20, 60]:
            vol = close.pct_change().rolling(vol_window).std() * np.sqrt(252) * 100
            vol_ma = vol.rolling(120).mean()

            # Low vol = risk-on
            sig = (vol < vol_ma).astype(float)
            sig.name = 'signal'
            r = timing_backtest(price_df, sig, f"Vol{vol_window}<MA120 | {idx_name}")
            all_results.extend(r)

            # High vol + oversold = buy
            for vol_pct in [70, 80, 90]:
                vol_high = vol.rolling(504).apply(lambda x: (x[-1] > x).mean(), raw=True)
                ma250 = close.rolling(250).mean()
                sig = ((vol_high > vol_pct/100) & (close < ma250)).astype(float)
                sig.name = 'signal'
                r = timing_backtest(price_df, sig, f"HighVol{vol_pct}&BelowMA250 | {idx_name}")
                all_results.extend(r)

    print(f"\n  Total sentiment timing results: {len(all_results)}")
    return all_results

# ============================================================
# STRATEGY 3: ETF ROTATION
# ============================================================

def run_etf_rotation(idx_prices):
    print("\n" + "="*60)
    print("STRATEGY 3: ETF ROTATION (ETF轮动)")
    print("="*60)

    all_results = []
    names = list(idx_prices.keys())

    if len(names) < 2:
        print("  Need >= 2 indices")
        return all_results

    # Build price matrix
    close_dict = {n: df['close'] for n, df in idx_prices.items()}
    pm = pd.DataFrame(close_dict).dropna()
    ret = pm.pct_change()
    print(f"  Price matrix: {len(pm)} days x {len(names)} indices")
    print(f"  Period: {pm.index[0].date()} to {pm.index[-1].date()}")

    cost = 10 / 10000
    split = pd.Timestamp('2023-01-01')

    for lookback in [5, 10, 20, 60, 120, 250]:
        mom = pm.pct_change(lookback)
        vol = ret.rolling(max(lookback, 20)).std() * np.sqrt(252)
        risk_adj_mom = mom / vol

        for rebal, rb_name in [('W', 'Wk'), ('M', 'Mo')]:
            rebal_dates = sorted(set(mom.resample(rebal).last().index) & set(mom.index))

            for strat_type, score_matrix in [
                ('Mom', mom),
                ('RAMom', risk_adj_mom)
            ]:
                for top_n in [1, 2]:
                    if top_n >= len(names):
                        continue

                    # Build signals
                    signals = {n: pd.Series(0.0, index=pm.index) for n in names}

                    for i in range(len(rebal_dates) - 1):
                        rd = rebal_dates[i]
                        next_rd = rebal_dates[i + 1]

                        scores = score_matrix.loc[rd].dropna()
                        if len(scores) < top_n:
                            continue

                        top = scores.nlargest(top_n).index.tolist()
                        mask = (pm.index > rd) & (pm.index <= next_rd)
                        weight = 1.0 / top_n
                        for t in top:
                            signals[t][mask] = weight

                    # Calculate returns
                    sr = pd.Series(0.0, index=ret.index)
                    turnover = pd.Series(0.0, index=ret.index)
                    for n in names:
                        sr += signals[n] * ret[n]
                        turnover += signals[n].diff().abs()
                    sr -= turnover * cost
                    sr = sr.dropna()

                    for period, sub in [('FULL', sr), ('IS', sr[sr.index < split]), ('OOS', sr[sr.index >= split])]:
                        r = calc_metrics(sub, f"{strat_type}{lookback}d_Top{top_n}_{rb_name} [{period}]")
                        if r:
                            r['pct_in_market'] = 100.0
                            all_results.append(r)

    # Dual Momentum (absolute + relative)
    for lookback in [20, 60, 120, 250]:
        abs_mom = pm.pct_change(lookback)

        for rebal, rb_name in [('W', 'Wk'), ('M', 'Mo')]:
            rebal_dates = sorted(set(abs_mom.resample(rebal).last().index) & set(abs_mom.index))

            signals = {n: pd.Series(0.0, index=pm.index) for n in names}
            cash = pd.Series(0.0, index=pm.index)

            for i in range(len(rebal_dates) - 1):
                rd = rebal_dates[i]
                next_rd = rebal_dates[i + 1]

                scores = abs_mom.loc[rd].dropna()
                if len(scores) == 0:
                    continue

                top1 = scores.idxmax()
                mask = (pm.index > rd) & (pm.index <= next_rd)

                if scores[top1] > 0:
                    signals[top1][mask] = 1.0
                else:
                    cash[mask] = 1.0

            sr = pd.Series(0.0, index=ret.index)
            turnover = pd.Series(0.0, index=ret.index)
            for n in names:
                sr += signals[n] * ret[n]
                turnover += signals[n].diff().abs()
            sr -= turnover * cost
            sr = sr.dropna()

            pct_in = round((1 - cash.mean()) * 100, 1)

            for period, sub in [('FULL', sr), ('IS', sr[sr.index < split]), ('OOS', sr[sr.index >= split])]:
                r = calc_metrics(sub, f"DualMom{lookback}d_{rb_name} [{period}]")
                if r:
                    r['pct_in_market'] = pct_in
                    all_results.append(r)

    # Equal weight benchmark
    ew_ret = ret.mean(axis=1).dropna()
    for period, sub in [('FULL', ew_ret), ('IS', ew_ret[ew_ret.index < split]), ('OOS', ew_ret[ew_ret.index >= split])]:
        r = calc_metrics(sub, f"EqualWeight [{period}]")
        if r:
            r['pct_in_market'] = 100.0
            all_results.append(r)

    print(f"\n  Total rotation results: {len(all_results)}")
    return all_results

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("FACTOR TIMING STRATEGY RESEARCH")
    print(f"Started: {datetime.now()}")
    print("="*60)

    # Login baostock
    bs.login()

    # Fetch index prices (use INDEX codes, not ETF codes)
    print("\n[1/4] Fetching index prices...")
    idx_map = {
        '沪深300': 'sh.000300',
        '中证500': 'sh.000905',
        '创业板': 'sz.399006',
        '中证1000': 'sz.399673',
    }

    idx_prices = {}
    for name, code in idx_map.items():
        try:
            df = get_index_price(code, start='2012-01-01')
            if len(df) > 500:
                idx_prices[name] = df
                print(f"  {name}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
        except Exception as e:
            print(f"  {name} error: {e}")

    bs.logout()

    # Fetch macro data
    print("\n[2/4] Fetching macro data...")
    macro_data = get_macro_data()

    # Run strategies
    print("\n[3/4] Running backtests...")

    r1 = run_macro_timing(idx_prices, macro_data)
    r2 = run_sentiment_timing(idx_prices, macro_data)
    r3 = run_etf_rotation(idx_prices)

    all_results = r1 + r2 + r3

    # Save
    print("\n[4/4] Saving results...")
    df = pd.DataFrame(all_results)
    csv_path = '/Users/claw/etf-trader/data/factor_timing_research.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} results to {csv_path}")

    if len(df) == 0:
        print("  WARNING: No results generated!")
        return

    # Analysis
    print("\n" + "="*60)
    print("RESULTS ANALYSIS")
    print("="*60)

    oos = df[df['strategy'].str.contains('OOS', na=False)]
    is_df = df[df['strategy'].str.contains('\\[IS\\]', na=False)]

    if len(oos) > 0:
        print(f"\n--- OOS Top 20 by Sharpe (total {len(oos)} OOS results) ---")
        top = oos.nlargest(20, 'sharpe')
        for _, r in top.iterrows():
            bh = f" (B&H: {r.get('bh_sharpe','?')})" if 'bh_sharpe' in r else ""
            print(f"  {r['strategy']}: Sharpe={r['sharpe']}, CAGR={r['cagr']}%, MDD={r['mdd']}%, Calmar={r['calmar']}{bh}")

        print(f"\n--- OOS Top 10 by Calmar ---")
        top_c = oos.nlargest(10, 'calmar')
        for _, r in top_c.iterrows():
            print(f"  {r['strategy']}: Calmar={r['calmar']}, Sharpe={r['sharpe']}, CAGR={r['cagr']}%, MDD={r['mdd']}%")

    # Summary by strategy type
    print(f"\n--- Best OOS by Category ---")
    categories = {
        'M1M2 Level': 'M1M2>',
        'M1M2 Momentum': 'M1M2_Rising|M1M2_Up',
        'M1 Level': 'M1>',
        'Term Spread Level': 'TS>\\d',
        'Term Spread MA': 'TS>MA',
        '10Y Yield Level': '10Y<\\d',
        '10Y Yield MA': '10Y<MA',
        'Combined Macro': 'M1M2.*&|M1M2.*\\|',
        'Price Rank': 'PriceRank',
        'Below MA + Bond': 'Below_MA.*&10Y',
        'RSI': 'RSI',
        'Distance from MA': 'Dist_MA',
        'Above MA': 'Above_MA',
        'Vol Regime': 'Vol\\d',
        'Momentum Rotation': '^Mom\\d',
        'Risk-Adj Rotation': '^RAMom',
        'Dual Momentum': '^DualMom',
    }

    for cat, pattern in categories.items():
        subset = oos[oos['strategy'].str.contains(pattern, na=False, regex=True)]
        if len(subset) > 0:
            best = subset.loc[subset['sharpe'].idxmax()]
            print(f"  {cat}: Sharpe={best['sharpe']}, CAGR={best['cagr']}%, MDD={best['mdd']}%")
            print(f"    -> {best['strategy']}")

    # Strategies that beat buy-and-hold in OOS
    if 'bh_sharpe' in oos.columns:
        beats_bh = oos[oos['sharpe'] > oos['bh_sharpe']]
        print(f"\n--- Strategies beating B&H in OOS: {len(beats_bh)}/{len(oos)} ---")
        if len(beats_bh) > 0:
            top_beat = beats_bh.nlargest(10, 'sharpe')
            for _, r in top_beat.iterrows():
                excess = round(r['sharpe'] - r['bh_sharpe'], 3)
                print(f"  {r['strategy']}: Sharpe={r['sharpe']} vs B&H={r['bh_sharpe']} (+{excess})")

    # IS-OOS consistency check
    print(f"\n--- IS→OOS Consistency (strategies with both periods) ---")
    is_strats = set(is_df['strategy'].str.replace(' \\[IS\\]', '', regex=True))
    oos_strats = set(oos['strategy'].str.replace(' \\[OOS\\]', '', regex=True))
    common_strats = is_strats & oos_strats

    consistent = []
    for s in common_strats:
        is_r = is_df[is_df['strategy'] == f"{s} [IS]"]
        oos_r = oos[oos['strategy'] == f"{s} [OOS]"]
        if len(is_r) > 0 and len(oos_r) > 0:
            is_sharpe = is_r.iloc[0]['sharpe']
            oos_sharpe = oos_r.iloc[0]['sharpe']
            if is_sharpe > 0.3 and oos_sharpe > 0.3:
                decay = (oos_sharpe - is_sharpe) / max(abs(is_sharpe), 0.01) * 100
                consistent.append({
                    'strategy': s,
                    'is_sharpe': is_sharpe,
                    'oos_sharpe': oos_sharpe,
                    'decay_pct': round(decay, 1),
                    'oos_cagr': oos_r.iloc[0]['cagr'],
                    'oos_mdd': oos_r.iloc[0]['mdd'],
                })

    if consistent:
        consistent_df = pd.DataFrame(consistent).sort_values('oos_sharpe', ascending=False)
        print(f"  Found {len(consistent_df)} strategies with IS Sharpe>0.3 AND OOS Sharpe>0.3:")
        for _, r in consistent_df.head(15).iterrows():
            print(f"  {r['strategy']}: IS={r['is_sharpe']} -> OOS={r['oos_sharpe']} ({r['decay_pct']:+.1f}%), CAGR={r['oos_cagr']}%, MDD={r['oos_mdd']}%")

    # Save summary JSON
    summary = {
        'total_results': len(df),
        'oos_results': len(oos),
        'top_oos_sharpe': oos.nlargest(10, 'sharpe')[['strategy','sharpe','cagr','mdd','calmar']].to_dict('records') if len(oos) > 0 else [],
        'top_oos_calmar': oos.nlargest(10, 'calmar')[['strategy','sharpe','cagr','mdd','calmar']].to_dict('records') if len(oos) > 0 else [],
        'consistent_strategies': consistent[:10] if consistent else [],
    }
    json_path = '/Users/claw/etf-trader/data/factor_timing_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved to {json_path}")

    print(f"\nCompleted: {datetime.now()}")

if __name__ == '__main__':
    main()
