#!/usr/bin/env python3
"""
行业景气度与拥挤度信号研究 (Sector Prosperity & Crowding Signal Research)

研究内容:
  1. 景气度信号 (Prosperity/Momentum):
     - 中长期动量 (3M/6M/12M return)
     - 动量加速度 (momentum acceleration)
     - 相对强度 (relative strength vs market)
     - 趋势强度 (MA deviation)
  2. 拥挤度信号 (Crowding):
     - 成交量异常度 (volume z-score vs historical)
     - 换手率变化 (turnover change)
     - 量价背离 (price-volume divergence)
     - 波动率异常 (volatility spike)
  3. 组合策略:
     - 景气度单因子
     - 拥挤度单因子
     - 景气度+拥挤度双因子
     - 各信号与MA择时叠加

数据: 中证行业指数 via akshare (含成交量)
基准: 沪深300 / 等权基准
回测期: ~2012~2026
周度调仓, 15bps单边交易成本

Author: Sarah Mitchell / VisionClaw
Date: 2026-03-29
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import akshare as ak
import json, os, sys, time
from datetime import datetime

DATA_DIR = '/Users/claw/etf-trader/data'

# ============================================================
# 1. SECTOR INDEX DEFINITIONS
# ============================================================

SECTOR_INDICES = {
    '全指消费': '000990',
    '全指金融': '000991',
    '全指信息': '000992',
    '全指医药': '000993',
    '全指能源': '000994',
    '全指材料': '000995',
    '全指工业': '000996',
    '全指可选': '000997',
    '全指公用': '000998',
    '全指电信': '000999',
    '中证军工': '930633',
    '证券公司': '399975',
    '中证银行': '399986',
}

MARKET_INDEX = '000300'  # 沪深300
RISK_FREE_RATE = 0.025
TC_ONE_SIDE = 0.0015  # 15bps per side

# ============================================================
# 2. DATA FETCHING (with volume)
# ============================================================

def fetch_index_data(code, name, start='20120101', end='20260401'):
    """Fetch daily index data with volume via akshare."""
    try:
        df = ak.index_zh_a_hist(symbol=code, period='daily',
                                start_date=start, end_date=end)
        if df is None or len(df) < 100:
            return None
        df = df.rename(columns={
            '日期': 'date', '收盘': 'close', '成交量': 'volume',
            '成交额': 'amount', '开盘': 'open', '最高': 'high', '最低': 'low'
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        for c in ['close', 'volume', 'amount', 'open', 'high', 'low']:
            if c in df.columns:
                df[c] = df[c].astype(float)
        df = df.set_index('date')
        return df
    except Exception as e:
        print(f"  ERROR fetching {code} ({name}): {e}")
        return None


def load_all_data():
    """Load all sector + market data."""
    print("=== Loading data ===")

    # Market
    print(f"  Fetching market: 沪深300 ({MARKET_INDEX})")
    mkt_df = fetch_index_data(MARKET_INDEX, 'market')
    if mkt_df is None:
        raise RuntimeError("Cannot fetch market data")
    time.sleep(0.5)

    # Sectors
    sector_close = {}
    sector_volume = {}
    sector_amount = {}

    for name, code in SECTOR_INDICES.items():
        print(f"  Fetching {name} ({code})...")
        df = fetch_index_data(code, name)
        if df is not None:
            sector_close[name] = df['close']
            if 'volume' in df.columns:
                sector_volume[name] = df['volume']
            if 'amount' in df.columns:
                sector_amount[name] = df['amount']
        time.sleep(0.5)

    close_df = pd.DataFrame(sector_close)
    vol_df = pd.DataFrame(sector_volume)
    amt_df = pd.DataFrame(sector_amount)

    # Align with market using outer join
    mkt_close = mkt_df[['close']].rename(columns={'close': 'market_close'})
    mkt_vol = mkt_df[['volume']].rename(columns={'volume': 'market_volume'}) if 'volume' in mkt_df.columns else None

    # Merge
    all_close = close_df.join(mkt_close, how='outer')
    all_close = all_close.sort_index().dropna(subset=['market_close'])

    all_vol = vol_df.join(mkt_close[[]].rename(columns={}), how='outer') if not vol_df.empty else vol_df
    all_amt = amt_df.join(mkt_close[[]].rename(columns={}), how='outer') if not amt_df.empty else amt_df

    sectors = [c for c in close_df.columns if c in all_close.columns]
    print(f"\n  Loaded {len(sectors)} sectors, date range: {all_close.index[0].date()} ~ {all_close.index[-1].date()}")
    print(f"  Total trading days: {len(all_close)}")

    return all_close, vol_df, amt_df, sectors


# ============================================================
# 3. SIGNAL COMPUTATION
# ============================================================

def calc_returns(close_df, sectors):
    """Calculate daily returns for all sectors."""
    ret_df = close_df[sectors].pct_change()
    mkt_ret = close_df['market_close'].pct_change()
    return ret_df, mkt_ret


def calc_momentum(close_df, sectors, window):
    """N-day momentum (return over past N days)."""
    mom = close_df[sectors].pct_change(window)
    return mom


def calc_momentum_acceleration(close_df, sectors, short_window=20, long_window=60):
    """Momentum acceleration: short-term mom minus long-term mom."""
    mom_short = calc_momentum(close_df, sectors, short_window)
    mom_long = calc_momentum(close_df, sectors, long_window)
    return mom_short - mom_long


def calc_relative_strength(close_df, sectors, window=60):
    """Relative strength vs market over N days."""
    sec_mom = close_df[sectors].pct_change(window)
    mkt_mom = close_df['market_close'].pct_change(window)
    rs = sec_mom.subtract(mkt_mom, axis=0)
    return rs


def calc_ma_deviation(close_df, sectors, window=60):
    """Price deviation from MA (trend strength)."""
    ma = close_df[sectors].rolling(window).mean()
    dev = (close_df[sectors] - ma) / ma
    return dev


def calc_volume_zscore(vol_df, sectors, window=60):
    """Volume z-score: current volume vs historical distribution."""
    available = [s for s in sectors if s in vol_df.columns]
    if not available:
        return pd.DataFrame()
    vol = vol_df[available]
    vol_ma = vol.rolling(window).mean()
    vol_std = vol.rolling(window).std()
    zscore = (vol - vol_ma) / vol_std
    return zscore


def calc_amount_zscore(amt_df, sectors, window=60):
    """Turnover amount z-score."""
    available = [s for s in sectors if s in amt_df.columns]
    if not available:
        return pd.DataFrame()
    amt = amt_df[available]
    amt_ma = amt.rolling(window).mean()
    amt_std = amt.rolling(window).std()
    zscore = (amt - amt_ma) / amt_std
    return zscore


def calc_volume_price_divergence(close_df, vol_df, sectors, window=20):
    """Volume-price divergence: price going up but volume declining, or vice versa.
    Negative = bearish divergence (price up, volume down).
    """
    available = [s for s in sectors if s in vol_df.columns]
    if not available:
        return pd.DataFrame()
    price_mom = close_df[available].pct_change(window)
    vol_mom = vol_df[available].pct_change(window)
    # Correlation over rolling window
    div = pd.DataFrame(index=close_df.index)
    for s in available:
        div[s] = close_df[s].pct_change().rolling(window).corr(vol_df[s].pct_change())
    return div


def calc_volatility_zscore(close_df, sectors, vol_window=20, zscore_window=120):
    """Volatility z-score: current vol vs historical distribution."""
    ret = close_df[sectors].pct_change()
    rolling_vol = ret.rolling(vol_window).std() * np.sqrt(252)
    vol_ma = rolling_vol.rolling(zscore_window).mean()
    vol_std = rolling_vol.rolling(zscore_window).std()
    zscore = (rolling_vol - vol_ma) / vol_std
    return zscore


# ============================================================
# 4. STRATEGY ENGINE
# ============================================================

def resample_weekly(df):
    """Resample to weekly (Friday)."""
    return df.resample('W-FRI').last()


def backtest_strategy(close_df, sectors, signal_df, n_hold=3,
                      direction='top', rebal_freq='weekly',
                      market_regime=None, tc=TC_ONE_SIDE):
    """
    Generic ranking-based strategy backtester.

    signal_df: DataFrame with signal values for each sector
    n_hold: number of sectors to hold
    direction: 'top' = buy highest signal, 'bottom' = buy lowest signal
    market_regime: optional Series (1=bull, 0=bear) for regime filtering

    Returns: dict with performance metrics
    """
    # Resample to weekly
    weekly_close = resample_weekly(close_df[sectors])
    weekly_signal = resample_weekly(signal_df)
    if market_regime is not None:
        weekly_regime = resample_weekly(market_regime)

    # Align
    common_idx = weekly_close.dropna(how='all').index.intersection(
        weekly_signal.dropna(how='all').index
    )
    if len(common_idx) < 52:
        return None

    weekly_close = weekly_close.loc[common_idx]
    weekly_signal = weekly_signal.loc[common_idx]
    weekly_ret = weekly_close.pct_change()

    nav = [1.0]
    prev_holdings = set()

    for i in range(1, len(common_idx)):
        sig = weekly_signal.iloc[i-1]  # signal from previous week
        valid_sigs = sig.dropna()

        if len(valid_sigs) < n_hold:
            # Not enough signals, hold cash
            nav.append(nav[-1])
            prev_holdings = set()
            continue

        # Apply regime filter if provided
        if market_regime is not None:
            regime_val = weekly_regime.iloc[i-1] if i-1 < len(weekly_regime) else 1
            if regime_val == 0:
                # Bear market: go to cash
                nav.append(nav[-1])
                prev_holdings = set()
                continue

        # Rank and select
        if direction == 'top':
            ranked = valid_sigs.nlargest(n_hold)
        else:
            ranked = valid_sigs.nsmallest(n_hold)

        holdings = set(ranked.index)

        # Calculate return
        rets = weekly_ret.iloc[i]
        port_ret = rets[list(holdings)].mean()

        # Transaction cost
        turnover = len(holdings.symmetric_difference(prev_holdings)) / max(len(holdings) + len(prev_holdings), 1)
        cost = turnover * tc * 2  # buy + sell

        nav.append(nav[-1] * (1 + port_ret - cost))
        prev_holdings = holdings

    nav = pd.Series(nav, index=common_idx)
    return calc_metrics(nav)


def backtest_long_short(close_df, sectors, signal_df, n_hold=3, tc=TC_ONE_SIDE):
    """
    Long-short strategy: long top N, short bottom N.
    Returns combined L/S performance.
    """
    weekly_close = resample_weekly(close_df[sectors])
    weekly_signal = resample_weekly(signal_df)

    common_idx = weekly_close.dropna(how='all').index.intersection(
        weekly_signal.dropna(how='all').index
    )
    if len(common_idx) < 52:
        return None

    weekly_close = weekly_close.loc[common_idx]
    weekly_signal = weekly_signal.loc[common_idx]
    weekly_ret = weekly_close.pct_change()

    nav = [1.0]
    prev_long = set()
    prev_short = set()

    for i in range(1, len(common_idx)):
        sig = weekly_signal.iloc[i-1]
        valid_sigs = sig.dropna()

        if len(valid_sigs) < n_hold * 2:
            nav.append(nav[-1])
            prev_long = set()
            prev_short = set()
            continue

        longs = set(valid_sigs.nlargest(n_hold).index)
        shorts = set(valid_sigs.nsmallest(n_hold).index)

        rets = weekly_ret.iloc[i]
        long_ret = rets[list(longs)].mean()
        short_ret = rets[list(shorts)].mean()
        ls_ret = (long_ret - short_ret) / 2  # 50/50 allocation

        # TC
        long_to = len(longs.symmetric_difference(prev_long)) / max(len(longs) + len(prev_long), 1)
        short_to = len(shorts.symmetric_difference(prev_short)) / max(len(shorts) + len(prev_short), 1)
        cost = (long_to + short_to) * tc

        nav.append(nav[-1] * (1 + ls_ret - cost))
        prev_long = longs
        prev_short = shorts

    nav = pd.Series(nav, index=common_idx)
    return calc_metrics(nav)


def calc_metrics(nav):
    """Calculate performance metrics from NAV series."""
    if nav.iloc[-1] <= 0:
        return None
    total_days = (nav.index[-1] - nav.index[0]).days
    years = total_days / 365.25
    if years < 1:
        return None

    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1

    weekly_ret = nav.pct_change().dropna()
    ann_vol = weekly_ret.std() * np.sqrt(52)
    sharpe = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    drawdown = nav / nav.cummax() - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        'CAGR': round(cagr * 100, 2),
        'AnnVol': round(ann_vol * 100, 2),
        'Sharpe': round(sharpe, 3),
        'MaxDD': round(max_dd * 100, 2),
        'Calmar': round(calmar, 3),
        'Years': round(years, 1),
        'StartDate': str(nav.index[0].date()),
        'EndDate': str(nav.index[-1].date()),
    }


def detect_regime(close_df, ma_period=60):
    """Market regime: 1=bull (close > MA), 0=bear."""
    mkt = close_df['market_close']
    ma = mkt.rolling(ma_period).mean()
    regime = (mkt > ma).astype(int)
    return regime


# ============================================================
# 5. RUN ALL STRATEGIES
# ============================================================

def run_all(close_df, vol_df, amt_df, sectors):
    """Run all strategy variants and collect results."""
    results = []
    ret_df, mkt_ret = calc_returns(close_df, sectors)

    # Pre-compute regimes
    regime_60 = detect_regime(close_df, 60)
    regime_120 = detect_regime(close_df, 120)

    # ---- BASELINE ----
    print("\n=== Baseline Strategies ===")

    # Equal weight
    weekly_close = resample_weekly(close_df[sectors])
    weekly_ret = weekly_close.pct_change()
    ew_nav = (1 + weekly_ret.mean(axis=1)).cumprod()
    ew_nav.iloc[0] = 1.0
    m = calc_metrics(ew_nav)
    if m:
        m['Strategy'] = 'EqualWeight'
        m['Signal'] = 'none'
        m['Params'] = 'baseline'
        results.append(m)
        print(f"  EqualWeight: CAGR={m['CAGR']}%, Sharpe={m['Sharpe']}, MaxDD={m['MaxDD']}%")

    # Market (300)
    mkt_nav = close_df['market_close'] / close_df['market_close'].dropna().iloc[0]
    mkt_m = calc_metrics(resample_weekly(mkt_nav))
    if mkt_m:
        mkt_m['Strategy'] = 'CSI300_BH'
        mkt_m['Signal'] = 'none'
        mkt_m['Params'] = 'baseline'
        results.append(mkt_m)
        print(f"  CSI300 BH: CAGR={mkt_m['CAGR']}%, Sharpe={mkt_m['Sharpe']}, MaxDD={mkt_m['MaxDD']}%")

    # ---- PROSPERITY / MOMENTUM SIGNALS ----
    print("\n=== Prosperity (Momentum) Signals ===")

    momentum_configs = [
        ('MOM_20', 20),
        ('MOM_60', 60),
        ('MOM_120', 120),
        ('MOM_250', 250),
    ]

    for mom_name, mom_window in momentum_configs:
        print(f"\n  Signal: {mom_name}")
        sig = calc_momentum(close_df, sectors, mom_window)

        for n_hold in [2, 3, 4]:
            for direction in ['top', 'bottom']:
                strat_name = f'PROSP_{mom_name}_{"TOP" if direction == "top" else "BOT"}{n_hold}'
                m = backtest_strategy(close_df, sectors, sig, n_hold=n_hold, direction=direction)
                if m:
                    m['Strategy'] = strat_name
                    m['Signal'] = 'momentum'
                    m['Params'] = f'window={mom_window},n={n_hold},dir={direction}'
                    results.append(m)
                    print(f"    {strat_name}: CAGR={m['CAGR']}%, Sharpe={m['Sharpe']}, MaxDD={m['MaxDD']}%")

        # Long-short
        for n_hold in [2, 3]:
            strat_name = f'PROSP_{mom_name}_LS{n_hold}'
            m = backtest_long_short(close_df, sectors, sig, n_hold=n_hold)
            if m:
                m['Strategy'] = strat_name
                m['Signal'] = 'momentum_ls'
                m['Params'] = f'window={mom_window},n={n_hold}'
                results.append(m)
                print(f"    {strat_name}: CAGR={m['CAGR']}%, Sharpe={m['Sharpe']}, MaxDD={m['MaxDD']}%")

    # ---- MOMENTUM ACCELERATION ----
    print("\n=== Momentum Acceleration Signals ===")
    accel_configs = [
        ('ACCEL_20_60', 20, 60),
        ('ACCEL_20_120', 20, 120),
        ('ACCEL_60_120', 60, 120),
        ('ACCEL_60_250', 60, 250),
    ]

    for accel_name, short_w, long_w in accel_configs:
        print(f"\n  Signal: {accel_name}")
        sig = calc_momentum_acceleration(close_df, sectors, short_w, long_w)

        for n_hold in [2, 3]:
            for direction in ['top', 'bottom']:
                strat_name = f'PROSP_{accel_name}_{"TOP" if direction == "top" else "BOT"}{n_hold}'
                m = backtest_strategy(close_df, sectors, sig, n_hold=n_hold, direction=direction)
                if m:
                    m['Strategy'] = strat_name
                    m['Signal'] = 'mom_acceleration'
                    m['Params'] = f'short={short_w},long={long_w},n={n_hold},dir={direction}'
                    results.append(m)
                    print(f"    {strat_name}: CAGR={m['CAGR']}%, Sharpe={m['Sharpe']}, MaxDD={m['MaxDD']}%")

    # ---- RELATIVE STRENGTH ----
    print("\n=== Relative Strength Signals ===")
    rs_configs = [('RS_20', 20), ('RS_60', 60), ('RS_120', 120)]

    for rs_name, rs_window in rs_configs:
        print(f"\n  Signal: {rs_name}")
        sig = calc_relative_strength(close_df, sectors, rs_window)

        for n_hold in [2, 3]:
            for direction in ['top', 'bottom']:
                strat_name = f'PROSP_{rs_name}_{"TOP" if direction == "top" else "BOT"}{n_hold}'
                m = backtest_strategy(close_df, sectors, sig, n_hold=n_hold, direction=direction)
                if m:
                    m['Strategy'] = strat_name
                    m['Signal'] = 'relative_strength'
                    m['Params'] = f'window={rs_window},n={n_hold},dir={direction}'
                    results.append(m)
                    print(f"    {strat_name}: CAGR={m['CAGR']}%, Sharpe={m['Sharpe']}, MaxDD={m['MaxDD']}%")

    # ---- MA DEVIATION (TREND STRENGTH) ----
    print("\n=== MA Deviation (Trend Strength) Signals ===")
    madev_configs = [('MADEV_20', 20), ('MADEV_60', 60), ('MADEV_120', 120)]

    for md_name, md_window in madev_configs:
        print(f"\n  Signal: {md_name}")
        sig = calc_ma_deviation(close_df, sectors, md_window)

        for n_hold in [2, 3]:
            for direction in ['top', 'bottom']:
                strat_name = f'PROSP_{md_name}_{"TOP" if direction == "top" else "BOT"}{n_hold}'
                m = backtest_strategy(close_df, sectors, sig, n_hold=n_hold, direction=direction)
                if m:
                    m['Strategy'] = strat_name
                    m['Signal'] = 'ma_deviation'
                    m['Params'] = f'window={md_window},n={n_hold},dir={direction}'
                    results.append(m)
                    print(f"    {strat_name}: CAGR={m['CAGR']}%, Sharpe={m['Sharpe']}, MaxDD={m['MaxDD']}%")

    # ---- CROWDING SIGNALS ----
    print("\n=== Crowding Signals ===")

    # Volume z-score
    for vol_window in [20, 60, 120]:
        vol_name = f'VOLZ_{vol_window}'
        print(f"\n  Signal: {vol_name}")
        sig = calc_volume_zscore(vol_df, sectors, vol_window)
        if sig.empty:
            print("    No volume data, skip")
            continue

        # Low crowding (buy neglected) = bottom volume z-score
        # High crowding (avoid) = top volume z-score
        for n_hold in [2, 3]:
            for direction, label in [('bottom', 'LOW'), ('top', 'HIGH')]:
                strat_name = f'CROWD_{vol_name}_{label}{n_hold}'
                m = backtest_strategy(close_df, sectors, sig, n_hold=n_hold, direction=direction)
                if m:
                    m['Strategy'] = strat_name
                    m['Signal'] = 'volume_zscore'
                    m['Params'] = f'window={vol_window},n={n_hold},dir={direction}'
                    results.append(m)
                    print(f"    {strat_name}: CAGR={m['CAGR']}%, Sharpe={m['Sharpe']}, MaxDD={m['MaxDD']}%")

    # Amount z-score
    for amt_window in [20, 60, 120]:
        amt_name = f'AMTZ_{amt_window}'
        print(f"\n  Signal: {amt_name}")
        sig = calc_amount_zscore(amt_df, sectors, amt_window)
        if sig.empty:
            print("    No amount data, skip")
            continue

        for n_hold in [2, 3]:
            for direction, label in [('bottom', 'LOW'), ('top', 'HIGH')]:
                strat_name = f'CROWD_{amt_name}_{label}{n_hold}'
                m = backtest_strategy(close_df, sectors, sig, n_hold=n_hold, direction=direction)
                if m:
                    m['Strategy'] = strat_name
                    m['Signal'] = 'amount_zscore'
                    m['Params'] = f'window={amt_window},n={n_hold},dir={direction}'
                    results.append(m)
                    print(f"    {strat_name}: CAGR={m['CAGR']}%, Sharpe={m['Sharpe']}, MaxDD={m['MaxDD']}%")

    # Volume-price divergence
    for div_window in [20, 60]:
        div_name = f'VPDIV_{div_window}'
        print(f"\n  Signal: {div_name}")
        sig = calc_volume_price_divergence(close_df, vol_df, sectors, div_window)
        if sig.empty:
            print("    No data, skip")
            continue

        for n_hold in [2, 3]:
            # Low divergence (negative = bearish divergence, avoid)
            # High divergence (positive = healthy confirmation, buy)
            for direction, label in [('bottom', 'NEGDIV'), ('top', 'POSDIV')]:
                strat_name = f'CROWD_{div_name}_{label}{n_hold}'
                m = backtest_strategy(close_df, sectors, sig, n_hold=n_hold, direction=direction)
                if m:
                    m['Strategy'] = strat_name
                    m['Signal'] = 'vol_price_divergence'
                    m['Params'] = f'window={div_window},n={n_hold},dir={direction}'
                    results.append(m)
                    print(f"    {strat_name}: CAGR={m['CAGR']}%, Sharpe={m['Sharpe']}, MaxDD={m['MaxDD']}%")

    # Volatility z-score (crowding proxy: high vol = crowded/panic)
    for vol_w, zscore_w in [(20, 120), (20, 250)]:
        vz_name = f'VOLAZ_{vol_w}_{zscore_w}'
        print(f"\n  Signal: {vz_name}")
        sig = calc_volatility_zscore(close_df, sectors, vol_w, zscore_w)

        for n_hold in [2, 3]:
            for direction, label in [('bottom', 'LOWVOL'), ('top', 'HIGHVOL')]:
                strat_name = f'CROWD_{vz_name}_{label}{n_hold}'
                m = backtest_strategy(close_df, sectors, sig, n_hold=n_hold, direction=direction)
                if m:
                    m['Strategy'] = strat_name
                    m['Signal'] = 'volatility_zscore'
                    m['Params'] = f'vol_w={vol_w},zscore_w={zscore_w},n={n_hold},dir={direction}'
                    results.append(m)
                    print(f"    {strat_name}: CAGR={m['CAGR']}%, Sharpe={m['Sharpe']}, MaxDD={m['MaxDD']}%")

    # ---- COMBO: PROSPERITY + CROWDING ----
    print("\n=== Combo: Prosperity + Crowding ===")

    # Best momentum signals combined with anti-crowding
    for mom_w in [60, 120]:
        for crowd_type, crowd_func, crowd_args in [
            ('VOLZ60', calc_volume_zscore, (vol_df, sectors, 60)),
            ('AMTZ60', calc_amount_zscore, (amt_df, sectors, 60)),
            ('VOLAZ20_120', calc_volatility_zscore, (close_df, sectors, 20, 120)),
        ]:
            combo_name = f'COMBO_MOM{mom_w}_{crowd_type}'
            print(f"\n  Signal: {combo_name}")

            mom_sig = calc_momentum(close_df, sectors, mom_w)
            crowd_sig = crowd_func(*crowd_args)

            if crowd_sig.empty:
                print("    No crowding data, skip")
                continue

            # Normalize both signals to z-scores
            mom_rank = mom_sig.rank(axis=1, pct=True)
            crowd_rank = crowd_sig.rank(axis=1, pct=True)

            # High prosperity + low crowding
            # combo = mom_rank - crowd_rank  (high mom, low crowd = good)
            common_cols = [c for c in mom_rank.columns if c in crowd_rank.columns]
            combo_sig = mom_rank[common_cols] - crowd_rank[common_cols]

            for n_hold in [2, 3, 4]:
                strat_name = f'{combo_name}_TOP{n_hold}'
                m = backtest_strategy(close_df, sectors, combo_sig, n_hold=n_hold, direction='top')
                if m:
                    m['Strategy'] = strat_name
                    m['Signal'] = 'combo_mom_crowd'
                    m['Params'] = f'mom_w={mom_w},crowd={crowd_type},n={n_hold}'
                    results.append(m)
                    print(f"    {strat_name}: CAGR={m['CAGR']}%, Sharpe={m['Sharpe']}, MaxDD={m['MaxDD']}%")

    # ---- COMBO WITH MARKET REGIME ----
    print("\n=== Prosperity + Regime Timing ===")

    for mom_w in [60, 120]:
        for regime_name, regime in [('MA60', regime_60), ('MA120', regime_120)]:
            sig = calc_momentum(close_df, sectors, mom_w)

            for n_hold in [2, 3]:
                strat_name = f'PROSP_MOM{mom_w}_{regime_name}_TOP{n_hold}'
                m = backtest_strategy(close_df, sectors, sig, n_hold=n_hold,
                                     direction='top', market_regime=regime)
                if m:
                    m['Strategy'] = strat_name
                    m['Signal'] = 'momentum_regime'
                    m['Params'] = f'mom_w={mom_w},regime={regime_name},n={n_hold}'
                    results.append(m)
                    print(f"    {strat_name}: CAGR={m['CAGR']}%, Sharpe={m['Sharpe']}, MaxDD={m['MaxDD']}%")

    # ---- REVERSAL + ANTI-CROWDING ----
    print("\n=== Reversal + Anti-Crowding ===")

    for rev_w in [20, 60]:
        for crowd_type, crowd_func, crowd_args in [
            ('VOLZ60', calc_volume_zscore, (vol_df, sectors, 60)),
            ('AMTZ60', calc_amount_zscore, (amt_df, sectors, 60)),
        ]:
            combo_name = f'COMBO_REV{rev_w}_{crowd_type}'
            print(f"\n  Signal: {combo_name}")

            rev_sig = -calc_momentum(close_df, sectors, rev_w)  # negative = reversal
            crowd_sig = crowd_func(*crowd_args)

            if crowd_sig.empty:
                continue

            rev_rank = rev_sig.rank(axis=1, pct=True)
            crowd_rank = crowd_sig.rank(axis=1, pct=True)

            common_cols = [c for c in rev_rank.columns if c in crowd_rank.columns]
            # High reversal (beaten down) + low crowding (neglected) = good
            combo_sig = rev_rank[common_cols] - crowd_rank[common_cols]

            for n_hold in [2, 3]:
                strat_name = f'{combo_name}_TOP{n_hold}'
                m = backtest_strategy(close_df, sectors, combo_sig, n_hold=n_hold, direction='top')
                if m:
                    m['Strategy'] = strat_name
                    m['Signal'] = 'combo_rev_crowd'
                    m['Params'] = f'rev_w={rev_w},crowd={crowd_type},n={n_hold}'
                    results.append(m)
                    print(f"    {strat_name}: CAGR={m['CAGR']}%, Sharpe={m['Sharpe']}, MaxDD={m['MaxDD']}%")

    # ---- RELATIVE STRENGTH + ANTI-CROWDING ----
    print("\n=== Relative Strength + Anti-Crowding ===")

    for rs_w in [60, 120]:
        for crowd_type, crowd_func, crowd_args in [
            ('VOLZ60', calc_volume_zscore, (vol_df, sectors, 60)),
        ]:
            combo_name = f'COMBO_RS{rs_w}_{crowd_type}'
            print(f"\n  Signal: {combo_name}")

            rs_sig = calc_relative_strength(close_df, sectors, rs_w)
            crowd_sig = crowd_func(*crowd_args)
            if crowd_sig.empty:
                continue

            rs_rank = rs_sig.rank(axis=1, pct=True)
            crowd_rank = crowd_sig.rank(axis=1, pct=True)

            common_cols = [c for c in rs_rank.columns if c in crowd_rank.columns]
            combo_sig = rs_rank[common_cols] - crowd_rank[common_cols]

            for n_hold in [2, 3]:
                strat_name = f'{combo_name}_TOP{n_hold}'
                m = backtest_strategy(close_df, sectors, combo_sig, n_hold=n_hold, direction='top')
                if m:
                    m['Strategy'] = strat_name
                    m['Signal'] = 'combo_rs_crowd'
                    m['Params'] = f'rs_w={rs_w},crowd={crowd_type},n={n_hold}'
                    results.append(m)
                    print(f"    {strat_name}: CAGR={m['CAGR']}%, Sharpe={m['Sharpe']}, MaxDD={m['MaxDD']}%")

    return results


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("行业景气度与拥挤度信号研究")
    print("=" * 70)

    close_df, vol_df, amt_df, sectors = load_all_data()
    results = run_all(close_df, vol_df, amt_df, sectors)

    # Sort by Sharpe
    results.sort(key=lambda x: x.get('Sharpe', 0), reverse=True)

    # Save results
    out_path = os.path.join(DATA_DIR, 'sector_prosperity_crowding_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n\nSaved {len(results)} results to {out_path}")

    # Print top 20
    print("\n" + "=" * 70)
    print("TOP 20 STRATEGIES BY SHARPE")
    print("=" * 70)
    print(f"{'Rank':>4s}  {'Strategy':<45s}  {'CAGR':>7s}  {'Sharpe':>7s}  {'MaxDD':>7s}  {'Calmar':>7s}")
    print("-" * 90)
    for i, r in enumerate(results[:20]):
        print(f"{i+1:4d}  {r['Strategy']:<45s}  {r['CAGR']:>6.1f}%  {r['Sharpe']:>7.3f}  {r['MaxDD']:>6.1f}%  {r['Calmar']:>7.3f}")

    # Print bottom 5 for reference
    print("\n--- Bottom 5 ---")
    for r in results[-5:]:
        print(f"  {r['Strategy']:<45s}  {r['CAGR']:>6.1f}%  {r['Sharpe']:>7.3f}  {r['MaxDD']:>6.1f}%")

    # Summary by signal type
    print("\n" + "=" * 70)
    print("SUMMARY BY SIGNAL TYPE (avg Sharpe)")
    print("=" * 70)
    signal_types = {}
    for r in results:
        sig = r['Signal']
        if sig not in signal_types:
            signal_types[sig] = []
        signal_types[sig].append(r['Sharpe'])

    for sig, sharpes in sorted(signal_types.items(), key=lambda x: np.mean(x[1]), reverse=True):
        print(f"  {sig:<25s}  n={len(sharpes):3d}  avg_sharpe={np.mean(sharpes):.3f}  max_sharpe={max(sharpes):.3f}")


if __name__ == '__main__':
    main()
