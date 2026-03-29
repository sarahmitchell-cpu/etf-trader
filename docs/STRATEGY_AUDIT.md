# Strategy Audit Report
**Date**: 2026-03-29
**Auditor**: Sarah Mitchell / VisionClaw

---

## Executive Summary

Comprehensive code audit of all strategies in the etf-trader repository.
Reviewed: code logic, data sources, look-ahead bias, survivorship bias, Sharpe formulas, and backtest integrity.

**Bugs Fixed**: 2 (Sharpe calculation errors in Strategy E and L)
**Bias Warnings Added**: 3 files (Strategy E, E_v2, stock_data_common.py)
**Critical Issues**: Strategy E/E_v2 survivorship bias (28 hand-picked stocks)

---

## Strategy Overview

| ID | Name | Type | Universe | Data Source | Rebal | Bias Status |
|----|------|------|----------|-------------|-------|-------------|
| A | Bond Diversified Hold | Passive | 4 Bond Indices | CSIndex | Quarterly | CLEAN |
| B | Enhanced Factor ETF | Momentum+Regime | 4 Factor ETFs | CSIndex TR | Weekly | MINOR (selection bias noted) |
| C | Cross-border ETF Momentum | Top1 Momentum | 3 HK ETFs | Yahoo Finance | Weekly | MINOR (short backtest) |
| D | CSI300 Low Volatility | Low Vol Factor | CSI300 Constituents | baostock | Bi-weekly | CLEAN |
| E V3 | Value Reversal+Quality | Contrarian Value | CSI300 Constituents | baostock | Monthly | CLEAN (V3 fixed) |
| E_v2 | Enhanced Value | Multi-factor Value | 28 Hand-picked Stocks | Yahoo+akshare | Monthly | SURVIVORSHIP BIAS (legacy) |
| F | US QDII Momentum | Momentum+Regime | NQ100/SP500 (CNY) | CSIndex | Weekly | CLEAN |
| G | CSI500 Low Vol+Low PB | Multi-factor | CSI500 Constituents | baostock | Bi-weekly | CLEAN (results revised) |
| H | Index Dip-Buying | Event-driven | Star50 | akshare/local CSV | Event | CLEAN (IS/OOS validated) |
| L | MA60 Trend Timing | Trend Following | 300 Growth TR | CSIndex | Daily | CLEAN |
| Stock | Individual Stock Picking | Multi-factor | CSI800 Constituents | akshare+baostock | Monthly | CLEAN |

---

## Detailed Findings

### Strategy A: Bond Diversified Hold
- **Logic**: Equal-weight 4 bond indices (10Y Treasury, Municipal, Credit, Short-term), quarterly rebalance with 5% drift threshold
- **Data**: CSIndex bond indices (credible, includes coupon income)
- **Biases**: None. Passive strategy, no factor selection, no stock picking
- **Code Quality**: Clean, well-documented
- **Backtest**: CAGR=4.0%, MDD=-2.5%, Sharpe=1.544 (2011-2026, ~15 years)
- **Verdict**: PASS

### Strategy B: Enhanced Factor ETF
- **Logic**: 4-factor momentum rotation (Cash Flow, Value, Dividend Low Vol, GEM Growth) with dual-MA regime filter and vol scaling
- **Data**: CSIndex Total Return indices (credible, includes dividends)
- **Biases**: Factor selection bias acknowledged in comments. 4 factors were chosen based on historical performance. No survivorship bias (indices don't die)
- **Look-ahead**: Properly handled — T-1 data for decisions, T period returns
- **Code Quality**: Clean, extensive documentation, vol scaling implemented correctly
- **Issue**: CSIndex API often blocked (WAF), only ~85 weeks of data currently available. Backtest period too short for reliable results
- **Verdict**: PASS (with selection bias caveat)

### Strategy C: Cross-border ETF Momentum (HK)
- **Logic**: Top1 4-week momentum from 3 HK-tracking ETFs (Tech/Medical/High Dividend)
- **Data**: Yahoo Finance (reliable for ETFs)
- **Biases**: Small universe (3 ETFs), short backtest (~4 years)
- **Look-ahead**: Properly handled — T decision, T+1 return
- **Code Quality**: Clean, simple and effective
- **Backtest**: CAGR=17.0%, MDD=-23.3%, Sharpe=0.613 (2022-2026)
- **Verdict**: PASS (limited by short history)

### Strategy D: CSI300 Low Volatility
- **Logic**: Select 10 lowest-volatility stocks from CSI300, with sector cap (max 3/sector)
- **Data**: baostock historical constituent lists + weekly prices
- **Biases**: Survivorship bias ELIMINATED via real historical constituent lists from baostock
- **Look-ahead**: Factors computed strictly from past data
- **Code Quality**: Clean, proper constituent mask implementation
- **Backtest**: CAGR=13.4%, MDD=-8.5%, Sharpe=0.755, 12M rolling win rate=98.8% (2022-2026)
- **Verdict**: PASS

### Strategy E V3: Value Reversal + Quality (FIXED)
- **Logic**: Contrarian (negative momentum) + low volatility from CSI300 historical constituents
- **Data**: baostock (historical constituents + weekly prices + industry classification)
- **V3 Fix**: Migrated from 28 hand-picked stocks to CSI300 real historical constituents
- **BUG FIXED**: Sharpe formula corrected (was missing RF subtraction)
- **Survivorship Bias**: ELIMINATED in V3 via baostock historical constituent lists
- **Impact of Fix**: CAGR 22.0%→9.7%, Sharpe 0.82→0.40, confirming bias inflated returns by >50%
- **Backtest V3**: CAGR=9.7%, MDD=-28.7%, Sharpe=0.397 (2021-2026, 4.7 years)
- **Verdict**: PASS (V3). The strategy is mediocre after bias removal — reversal/contrarian doesn't work well in CSI300

### Strategy E_v2: Enhanced Value (PE/PB + Reversal + Quality + Earnings)
- **Logic**: Same 28-stock pool, adds PE/PB percentile and ROE proxy factors
- **Data**: Yahoo Finance + akshare PE/PB (Baidu Stock)
- **SURVIVORSHIP BIAS**: Same as Strategy E
- **Sharpe**: Correctly implemented (uses rf=2.5%)
- **Look-ahead in PE/PB**: Properly handled — `mask = df.index <= as_of_date` on line 175
- **Verdict**: FAIL (survivorship bias). Same warning as E

### Strategy F: US QDII Momentum
- **Logic**: Top1 momentum between NQ100 and SP500 (CNY-denominated) + safe haven (short-term bonds), with dual-MA regime + drawdown stop-loss
- **Data**: CSIndex (CNY-denominated US indices, credible, includes FX effects)
- **Biases**: Only 2 US indices — minimal selection bias. Drawdown stop-loss parameters could be overfit
- **Look-ahead**: Allocation at time i uses data up to i-1 (correct)
- **Backtest**: CAGR=13.6%, MDD=-17.8%, Sharpe=0.765 (2011-2026)
- **Verdict**: PASS

### Strategy G: CSI500 Low Vol + Low PB
- **Logic**: 50% low volatility + 50% low PB from CSI500 constituents
- **Data**: baostock historical constituents + PE/PB fundamentals
- **Biases**: Survivorship bias ELIMINATED via baostock historical CSI500 constituent lists. PB data coverage ~400/928 stocks (baostock limitation, not a look-ahead issue)
- **Code Quality**: Clean, proper z-score normalization within cross-sections
- **Backtest (fresh 2026-03-30)**: CAGR=4.4%, MDD=-15.8%, Sharpe=0.188 (2022-2026)
  - Previous cached result (CAGR=13.5%, Sharpe=0.794) was from stale/different data
  - Fresh run with updated baostock data shows significantly weaker performance
  - 12M rolling win rate=75.2% (was 97.5% with old data)
- **Verdict**: PASS (code is clean, but performance is weak)

### Strategy H V2: Index Dip-Buying
- **Logic**: 6 variants of dip-buying/rally-chasing on Star50 (most volatile A-share index)
- **Data**: akshare/yfinance/local CSV
- **Biases**: V2 properly uses next-day open entry (no look-ahead). IS/OOS validation framework (2020-2022 IS, 2023-2026 OOS)
- **Concerns**: Small sample (12-24 trades per variant), high OOS decay (-36% to -70%), but all 5 PASS variants maintain positive OOS Sharpe
- **Transaction costs**: 10bps round-trip, conservative
- **Backtest**: H1: CAGR=13.9%, Sharpe=1.04; H3: CAGR=13.4%, Sharpe=1.07 (best variants)
- **Verdict**: PASS (with small-sample caveat)

### Strategy L: MA60 Trend Timing
- **Logic**: Pure MA60 trend following on 300 Growth Total Return index
- **Data**: CSIndex total return index (credible, long history from 2005)
- **BUG FIXED**: Sharpe formula was `cagr/vol` — should be `(cagr-rf)/vol`. Fixed to `(cagr-0.025)/vol`
- **Look-ahead**: Signal properly shifted by 1 day (`d['signal'] = d['position'].shift(1)`)
- **Backtest**: CAGR=16.7%, MDD=-31.7%, Sharpe=0.487 (2005-2026, ~20 years)
- **Verdict**: PASS

### Individual Stock Strategy (research/stock_strategy_live.py)
- **Logic**: Low turnover + 12M momentum (skip-1M) on CSI800, Top20 monthly rebalance
- **Data**: akshare prices + baostock historical CSI800 constituents
- **Biases**: Survivorship bias ELIMINATED. Skip-1M avoids short-term reversal contamination
- **Backtest**: CAGR=21.4%, MDD=-25.4%, Sharpe=0.966 (2016-2026, 10 years)
- **Verdict**: PASS

---

## Bugs Fixed This Audit

1. **strategy_e_weekly_signal.py line 274**: Sharpe missing RF subtraction
   - Old: `wr.mean() / wr.std() * np.sqrt(52)`
   - New: `(wr.mean() * 52 - 0.025) / (wr.std() * np.sqrt(52))`

2. **strategy_l_weekly_signal.py line 233**: Sharpe using CAGR/vol instead of excess return
   - Old: `cagr / vol`
   - New: `(cagr - 0.025) / vol`

## Bias Warnings Added

3. **strategy_e_weekly_signal.py**: Added survivorship bias warning in docstring
4. **strategy_e_v2_weekly_signal.py**: Added survivorship bias warning in docstring
5. **stock_data_common.py**: Added survivorship bias warning above STOCK_POOL definition

---

## Backtest Results Summary (Latest Run 2026-03-29)

| Strategy | Period | CAGR | MDD | Sharpe | Calmar | Notes |
|----------|--------|------|-----|--------|--------|-------|
| A Bond | 2011-2026 (15y) | 4.0% | -2.5% | 1.544 | 1.544 | Passive, very stable |
| B Factor ETF | 2025-2026 (0.8y) | 61.1%* | -6.5% | 2.929* | 9.4* | *Short period, unreliable |
| C HK ETF | 2022-2026 (4.2y) | 17.0% | -23.3% | 0.613 | 0.727 | |
| D CSI300 LV | 2022-2026 (4y) | 13.4% | -8.5% | 0.755 | 1.578 | No survivorship bias |
| E V3 Value | 2021-2026 (4.7y) | 9.7% | -28.7% | 0.397 | 0.338 | Fixed! Was 22% with bias |
| F US QDII | 2011-2026 (15y) | 13.6% | -17.8% | 0.765 | 0.765 | |
| G CSI500 LV+PB | 2022-2026 (4y) | 4.4% | -15.8% | 0.188 | 0.278 | Fresh data, weaker than cached |
| H Dip-Buy | 2020-2026 (5y) | 8-16% | -3~-23% | 0.68-1.13 | varies | 6 variants |
| L MA60 Trend | 2005-2026 (20y) | 16.7% | -31.7% | 0.487 | 0.487 | Longest backtest |
| Stock LT+Mom | 2016-2026 (10y) | 21.4% | -25.4% | 0.966 | 0.843 | No survivorship bias |

---

## Recommendations

1. **Strategy E/E_v2**: Consider migrating to baostock historical CSI300/CSI800 constituents (like Strategy D/G/Stock) to eliminate survivorship bias. Current 28-stock pool should be deprecated for backtesting purposes.

2. **Strategy B**: Need longer data history. CSIndex API frequently blocked — consider adding more backup data sources or pre-downloading historical data.

3. **Strategy H**: Monitor trade count. With only 12-24 trades over 5 years, statistical significance is limited. Consider expanding to more indices.

4. **Code Structure**: Research files in `/research/` are properly separated from live signal files. Consider archiving older research versions (v1, v2) to reduce clutter.

5. **Sharpe Consistency**: All strategies now use rf=2.5% consistently. Verify this matches current risk-free rate periodically.
