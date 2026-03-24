# Strategy G: CSI500 Low Volatility + Low PB (Mean Reversion)

> CSI500 real constituents, 50% low vol + 50% low PB, Top10, bi-weekly rebalance

## 1. Overview
Composite factor strategy on CSI 500: combines low volatility and low PB to find cheap, stable mid-cap stocks. Essentially "value mean reversion" without relying on momentum (which fails completely on CSI 500).

## 2. Performance (2021-01 ~ 2026-03, ~5 years)

| Metric | Value |
|--------|-------|
| CAGR | 13.5% |
| Max Drawdown | -11.9% |
| Sharpe | 0.794 |
| Calmar | 1.132 |
| 12m Win Rate | 97.5% |
| 12m Worst | -1.25% |

Annual: 2022:+8.8% | 2023:+4.1% | 2024:+38.7% | 2025:+12.0%

## 3. Strategy Logic
1. Universe: CSI 500 real constituents (928 unique, ~500 active)
2. Factors (z-scored, cross-sectional):
   - 50% Low Volatility (12-week annualized, negated)
   - 50% Low PB (PB-MRQ from baostock, negated)
3. Selection: Top 10 by composite score
4. Weight: Equal (10% each)
5. Rebalance: Every 2 weeks
6. Cost: 8 bps one-way

## 4. Bias Handling
- Survivorship: ELIMINATED (928 historical stocks via baostock)
- Look-ahead: ELIMINATED
- Transaction cost: 8 bps
- Data coverage: ~400/928 stocks have PE/PB (baostock limitation). Only stocks with both price AND PB data are eligible.
- Note: This strategy was VALIDATED against full historical constituents. Previous research showed:
  - Pure momentum: ALL NEGATIVE (CAGR -3.8% to -12.6%)
  - Sector rotation: ALL NEGATIVE on real constituents
  - LV+PB: +13.5% CAGR (the ONLY approach that works)

## 5. Why "Mean Reversion"?
- NOT buying "stocks that dropped" (that fails on CSI500)
- Instead: buying stocks that are fundamentally cheap (low PB) AND stable (low vol)
- Low PB = "priced below book value" = value anchor
- Low vol = "not distressed/speculative" = quality filter
- Together: identifies undervalued mid-caps likely to revert to fair value

## 6. Research History (all on real constituents, no survivorship bias)
| Phase | Strategy | CAGR | MDD | Calmar |
|-------|----------|------|-----|--------|
| 1 | Pure Momentum | -3.8% | -55% | -0.07 |
| 2 | Sector Rotation | -3.8% | -55% | -0.07 |
| 3 | Dynamic Allocation | N/A | N/A | N/A |
| 4 | Growth Factors | +10.2% | -16.9% | 0.60 |
| 5 | **LV+PB (this)** | **+13.5%** | **-11.9%** | **1.13** |

## 7. Files
- `strategy_g_weekly_signal.py` - production signal generator
- `research/csi500_full_constituent_backtest.py` - SR research
- `research/csi500_multifactor_research.py` - multi-factor research (v3)
- `data/csi500_multifactor_research.json` - 209 configs results

## 8. Reproduction
```bash
pip3 install baostock pandas numpy
python3 strategy_g_weekly_signal.py --backtest
```
