# Strategy D v2: CSI300 Low Volatility

> CSI300 real constituents, 20-week low volatility, Top10 equal-weight, bi-weekly rebalance

## 1. Overview
Select the 10 lowest-volatility stocks from the CSI 300 index using real historical constituents (no survivorship bias).

**Replaces**: Strategy D v1 (28 hand-picked leader stocks with momentum)
**Reason**: v1 had survivorship bias from manually selected stock pool; v2 uses baostock historical constituent data.

## 2. Performance (2022-01 ~ 2026-03, ~4 years)

| Metric | Value |
|--------|-------|
| CAGR | 13.9% |
| Max Drawdown | -8.5% |
| Sharpe | 0.955 |
| Calmar | 1.636 |
| 12m Win Rate | 98.8% |

Annual: 2022:-1.5% | 2023:+12.0% | 2024:+46.6% | 2025:+5.2%

## 3. Strategy Logic
1. Universe: CSI 300 real constituents (445 unique, ~300 active)
2. Factor: 20-week rolling volatility (lower = better)
3. Selection: Top 10 lowest volatility
4. Weight: Equal (10% each)
5. Rebalance: Every 2 weeks
6. Cost: 8 bps one-way

## 4. Bias Handling
- Survivorship: ELIMINATED (baostock historical constituents)
- Look-ahead: ELIMINATED (factors from past data only)
- Transaction cost: 8 bps applied to turnover
- Data: baostock weekly close, back-adjusted

## 5. Files
- `strategy_d_weekly_signal.py` - production signal generator
- `research/csi300_multifactor_research.py` - research (209 configs)
- `data/csi300_multifactor_research.json` - research results

## 6. Reproduction
```bash
pip3 install baostock pandas numpy
python3 strategy_d_weekly_signal.py --backtest
```
