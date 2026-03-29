#!/usr/bin/env python3
"""
Strategy H V2: Index Dip-Buying & Rally-Chasing (指数超跌买入/追涨买入)
=======================================================================
V2 fixes (2026-03-24, addressing audit report):
  - Entry at next-day open price (no look-ahead bias)
  - 10bps round-trip transaction costs
  - Data from local CSV (reproducible) + akshare/baostock (live signals)
  - Variants selected from IS data only, blind OOS validation
  - Minimum 5 OOS trades required to validate

6 variants selected via IS (2020-2022) + neighborhood robustness,
  then blind-tested on OOS (2023-2026):

  PASS (5 variants, all 科创50):
  - H1: 超跌 3日跌>7% → 持4日     Full Sharpe=1.04  CAGR=13.9%  OOS Sharpe=0.52
  - H2: 追涨 5日涨>7% → 持20日    Full Sharpe=0.68  CAGR=15.9%  OOS Sharpe=0.32
  - H3: 超跌 3日跌>8% → 持4日     Full Sharpe=1.07  CAGR=13.4%  OOS Sharpe=0.65
  - H4: 超跌 4日跌>7% → 持3日     Full Sharpe=1.03  CAGR=13.7%  OOS Sharpe=0.58
  - H5: 超跌 3日跌>7% → 持1日     Full Sharpe=0.90  CAGR=8.6%   OOS Sharpe=0.37

  BORDERLINE (1 variant, OOS trades=4, threshold=5):
  - H6: 超跌 6日跌>10% → 持3日(SL-3%)  Full Sharpe=1.13  CAGR=12.8%  OOS Sharpe=0.87

  NOTE: All validated variants are on 科创50 (most volatile A-share index).
  This is the honest result of the methodology — dip-buying works best on
  volatile indices. See docs/strategy_h.md for full methodology and limitations.

Usage:
  python3 strategy_h_weekly_signal.py              # Check today's signals
  python3 strategy_h_weekly_signal.py --json        # JSON only
  python3 strategy_h_weekly_signal.py --backtest    # Full backtest all variants
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
STRATEGY_DATA_DIR = os.path.join(DATA_DIR, 'strategy_h')
os.makedirs(STRATEGY_DATA_DIR, exist_ok=True)

RISK_FREE_RATE = 0.02
TRANSACTION_COST_BPS = 10  # 10bps round-trip
COST_RATE = TRANSACTION_COST_BPS / 10000  # 0.001

# ============================================================
# 6 Validated Variants (V2: IS selection + blind OOS validation)
# ============================================================
VARIANTS = {
    'H1': {
        'name': 'H1: 科创50 超跌 3日跌>7% 持4日',
        'index_name': '科创50',
        'direction': 'dip',
        'cum_days': 3,
        'threshold_pct': 7,
        'hold_days': 4,
        'stop_loss_pct': None,
        'validation': {
            'is_sharpe': 1.54, 'oos_sharpe': 0.52, 'oos_decay': -66,
            'full_sharpe': 1.04, 'full_cagr': 13.9, 'full_mdd': -6.6,
            'full_trades': 18, 'full_win_rate': 72,
            'neighbor_pos_rate': 100, 'status': 'PASS',
        },
    },
    'H2': {
        'name': 'H2: 科创50 追涨 5日涨>7% 持20日',
        'index_name': '科创50',
        'direction': 'rally',
        'cum_days': 5,
        'threshold_pct': 7,
        'hold_days': 20,
        'stop_loss_pct': None,
        'validation': {
            'is_sharpe': 1.06, 'oos_sharpe': 0.32, 'oos_decay': -70,
            'full_sharpe': 0.68, 'full_cagr': 15.9, 'full_mdd': -23.4,
            'full_trades': 21, 'full_win_rate': 57,
            'neighbor_pos_rate': 100, 'status': 'PASS',
        },
    },
    'H3': {
        'name': 'H3: 科创50 超跌 3日跌>8% 持4日',
        'index_name': '科创50',
        'direction': 'dip',
        'cum_days': 3,
        'threshold_pct': 8,
        'hold_days': 4,
        'stop_loss_pct': None,
        'validation': {
            'is_sharpe': 1.49, 'oos_sharpe': 0.65, 'oos_decay': -56,
            'full_sharpe': 1.07, 'full_cagr': 13.4, 'full_mdd': -5.5,
            'full_trades': 12, 'full_win_rate': 83,
            'neighbor_pos_rate': 100, 'status': 'PASS',
        },
    },
    'H4': {
        'name': 'H4: 科创50 超跌 4日跌>7% 持3日',
        'index_name': '科创50',
        'direction': 'dip',
        'cum_days': 4,
        'threshold_pct': 7,
        'hold_days': 3,
        'stop_loss_pct': None,
        'validation': {
            'is_sharpe': 1.41, 'oos_sharpe': 0.58, 'oos_decay': -59,
            'full_sharpe': 1.03, 'full_cagr': 13.7, 'full_mdd': -8.9,
            'full_trades': 24, 'full_win_rate': 75,
            'neighbor_pos_rate': 100, 'status': 'PASS',
        },
    },
    'H5': {
        'name': 'H5: 科创50 超跌 3日跌>7% 持1日',
        'index_name': '科创50',
        'direction': 'dip',
        'cum_days': 3,
        'threshold_pct': 7,
        'hold_days': 1,
        'stop_loss_pct': None,
        'validation': {
            'is_sharpe': 1.26, 'oos_sharpe': 0.37, 'oos_decay': -70,
            'full_sharpe': 0.90, 'full_cagr': 8.6, 'full_mdd': -3.0,
            'full_trades': 19, 'full_win_rate': 74,
            'neighbor_pos_rate': 100, 'status': 'PASS',
        },
    },
    'H6': {
        'name': 'H6: 科创50 超跌 6日跌>10% 持3日(SL-3%)',
        'index_name': '科创50',
        'direction': 'dip',
        'cum_days': 6,
        'threshold_pct': 10,
        'hold_days': 3,
        'stop_loss_pct': -3,
        'validation': {
            'is_sharpe': 1.37, 'oos_sharpe': 0.87, 'oos_decay': -36,
            'full_sharpe': 1.13, 'full_cagr': 12.8, 'full_mdd': -3.6,
            'full_trades': 12, 'full_win_rate': 83,
            'neighbor_pos_rate': 100, 'status': 'BORDERLINE (4 OOS trades, min=5)',
        },
    },
}


# ============================================================
# Data Download (for live signal checking)
# ============================================================
def download_latest_kc50(lookback_days=60):
    """Download recent 科创50 data for signal checking.

    Tries multiple sources for reliability:
    1. akshare (stock_zh_index_daily for 科创50 index)
    2. yfinance (588000.SS fallback)
    """
    # Try akshare first
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol='sh000688')
        if df is not None and len(df) > lookback_days:
            df = df.sort_values('date').tail(lookback_days)
            return df['close'].values, df['open'].values
    except Exception:
        pass

    # Fallback to yfinance
    try:
        import yfinance as yf
        df = yf.download('588000.SS', period=f'{lookback_days}d', progress=False)
        if df is not None and len(df) > 0:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df['Close'].dropna().values, df['Open'].dropna().values
    except Exception:
        pass

    # Fallback to local CSV
    csv_path = os.path.join(STRATEGY_DATA_DIR, 'star50.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df = df.sort_values('date').tail(lookback_days)
        return df['close'].values, df['open'].values

    return None, None


# ============================================================
# Signal Generation
# ============================================================
def check_signal(closes, cum_days, threshold_pct, direction):
    """
    Check if the latest bar triggers a buy signal.

    For dip: cum_days cumulative decline >= threshold_pct%
    For rally: cum_days cumulative rally >= threshold_pct%

    Returns: (triggered: bool, actual_change_pct: float)
    """
    if len(closes) < cum_days + 1:
        return False, 0.0

    current = closes[-1]
    past = closes[-(cum_days + 1)]
    if past <= 0:
        return False, 0.0
    change_pct = (current / past - 1) * 100

    if direction == 'dip':
        triggered = change_pct <= -threshold_pct
    else:  # rally
        triggered = change_pct >= threshold_pct

    return triggered, change_pct


# ============================================================
# Backtest Engine V2 (next-day open entry + transaction costs)
# ============================================================
def backtest_variant(closes, opens, variant):
    """
    Run backtest for a single variant.

    V2 fixes:
    - Entry at NEXT-DAY OPEN after signal (no look-ahead bias)
    - 10bps round-trip transaction costs (5bps each way)
    """
    n = len(closes)
    cum_days = variant['cum_days']
    threshold_pct = variant['threshold_pct']
    hold_days = variant['hold_days']
    stop_loss_pct = variant['stop_loss_pct']
    direction = variant['direction']

    # Pre-compute cumulative returns
    cum_ret = np.full(n, np.nan)
    for d in range(cum_days, n):
        if closes[d - cum_days] > 0:
            cum_ret[d] = (closes[d] / closes[d - cum_days] - 1) * 100

    # Generate signals (at close of day i)
    if direction == 'dip':
        signals = np.array([not np.isnan(cr) and cr <= -threshold_pct for cr in cum_ret])
    else:
        signals = np.array([not np.isnan(cr) and cr >= threshold_pct for cr in cum_ret])

    # Backtest with next-day open entry
    nav = np.ones(n)
    position = False
    entry_price = 0.0
    entry_idx = 0
    hold_count = 0
    trades = []

    for i in range(1, n):
        if position:
            daily_ret = closes[i] / closes[i - 1] - 1
            nav[i] = nav[i - 1] * (1 + daily_ret)
            hold_count += 1

            current_trade_ret = closes[i] / entry_price - 1
            hit_stop = (stop_loss_pct is not None and current_trade_ret * 100 <= stop_loss_pct)

            if hold_count >= hold_days or hit_stop:
                exit_price = closes[i]
                net_ret = (exit_price / entry_price - 1) - COST_RATE
                nav[i] = nav[i] * (1 - COST_RATE)  # exit cost

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'net_return': net_ret,
                    'hold_days': hold_count,
                    'stop_loss': hit_stop,
                })
                position = False
        else:
            nav[i] = nav[i - 1]
            # Signal at day i-1 close -> enter at day i open
            if signals[i - 1] and not np.isnan(opens[i]) and opens[i] > 0:
                position = True
                entry_price = opens[i]
                entry_idx = i
                hold_count = 0
                nav[i] = nav[i] * (1 - COST_RATE)  # entry cost
                # Entry day return: open to close
                if closes[i] > 0:
                    nav[i] = nav[i] * (1 + (closes[i] / opens[i] - 1))

    # Force close
    if position:
        net_ret = (closes[-1] / entry_price - 1) - COST_RATE
        nav[-1] = nav[-1] * (1 - COST_RATE)
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': n - 1,
            'net_return': net_ret,
            'hold_days': hold_count,
            'stop_loss': False,
        })

    return nav, trades


def calc_metrics(nav, trades, n_days):
    """Calculate performance metrics."""
    if n_days < 2:
        return None

    n_years = n_days / 252
    if n_years <= 0:
        return None

    annualized_return = (nav[-1] / nav[0]) ** (1 / n_years) - 1
    cummax = np.maximum.accumulate(nav)
    drawdown = (nav - cummax) / cummax
    max_drawdown = np.min(drawdown)

    daily_ret = np.diff(nav) / nav[:-1]
    std = np.std(daily_ret)
    sharpe = (np.mean(daily_ret) - RISK_FREE_RATE / 252) / std * np.sqrt(252) if std > 0 else 0
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    n_trades = len(trades)
    if n_trades > 0:
        rets = [t['net_return'] for t in trades]
        win_rate = sum(1 for r in rets if r > 0) / n_trades
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        plr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    else:
        win_rate = plr = 0

    return {
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'calmar': calmar,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_loss_ratio': plr,
    }


# ============================================================
# Signal Checking
# ============================================================
def check_all_signals():
    """Check all variants for today's signal."""
    print("=" * 70)
    print(f"Strategy H V2 Signal Check - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Engine: next-day open entry + {TRANSACTION_COST_BPS}bps costs")
    print("=" * 70)

    closes, opens = download_latest_kc50()
    if closes is None:
        print("  ERROR: Failed to download 科创50 data from all sources")
        return []

    print(f"  科创50: {len(closes)} bars, latest close={closes[-1]:.2f}")
    print()

    signals = []
    for vid, v in VARIANTS.items():
        triggered, change = check_signal(
            closes, v['cum_days'], v['threshold_pct'], v['direction']
        )

        direction_zh = '跌' if v['direction'] == 'dip' else '涨'
        sl_str = f"止损{v['stop_loss_pct']}%" if v['stop_loss_pct'] else '不止损'
        status = "BUY SIGNAL!" if triggered else "no signal"

        print(f"  {vid}: {v['cum_days']}日{direction_zh}: "
              f"{change:+.2f}% (阈值{'-' if v['direction']=='dip' else '+'}{v['threshold_pct']}%) "
              f"-> {status}")

        if triggered:
            signals.append({
                'variant': vid,
                'name': v['name'],
                'index': '科创50',
                'action': 'BUY (next-day open)',
                'hold_days': v['hold_days'],
                'stop_loss': v['stop_loss_pct'],
                'change_pct': round(change, 2),
            })

    print()
    if signals:
        print(f"{'='*70}")
        print(f"ACTIVE SIGNALS: {len(signals)}")
        print(f"{'='*70}")
        for s in signals:
            sl = f"止损{s['stop_loss']}%" if s['stop_loss'] else '不止损'
            print(f"  {s['variant']}: 科创50 -> BUY at next-day OPEN, 持{s['hold_days']}日, {sl}")
    else:
        print("No active signals today.")

    return signals


# ============================================================
# Full Backtest
# ============================================================
def run_backtest():
    """Run full backtest for all variants using local CSV data."""
    print("=" * 70)
    print("Strategy H V2 Full Backtest")
    print(f"  Engine: next-day open entry + {TRANSACTION_COST_BPS}bps costs")
    print("=" * 70)

    # Load 科创50 data from CSV
    csv_path = os.path.join(STRATEGY_DATA_DIR, 'star50.csv')
    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found")
        print("  Run the data download script first")
        return []

    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    for col in ['open', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close', 'open'])

    closes = df['close'].values
    opens = df['open'].values
    n_days = len(closes)
    n_years = n_days / 252

    print(f"\n  科创50: {n_days} trading days ({df['date'].iloc[0].strftime('%Y-%m-%d')} ~ "
          f"{df['date'].iloc[-1].strftime('%Y-%m-%d')}, {n_years:.1f} years)\n")

    results = []
    for vid, v in VARIANTS.items():
        print(f"\n{'='*50}")
        print(f"{vid}: {v['name']}")
        print(f"{'='*50}")

        nav, trades = backtest_variant(closes, opens, v)
        metrics = calc_metrics(nav, trades, n_days)

        if metrics:
            print(f"  CAGR:      {metrics['annualized_return']*100:>8.1f}%")
            print(f"  MDD:       {metrics['max_drawdown']*100:>8.1f}%")
            print(f"  Sharpe:    {metrics['sharpe']:>8.2f}")
            print(f"  Calmar:    {metrics['calmar']:>8.2f}")
            print(f"  Win Rate:  {metrics['win_rate']*100:>8.0f}%")
            print(f"  Trades:    {metrics['n_trades']:>8d}")
            print(f"  P/L Ratio: {metrics['profit_loss_ratio']:>8.2f}")
            print(f"  Status:    {v['validation']['status']}")

            results.append({
                'variant': vid,
                'index': '科创50',
                **metrics,
            })

    # Buy-and-hold benchmark
    print(f"\n{'='*50}")
    print("Buy & Hold Benchmark")
    print(f"{'='*50}")
    bh_cagr = (closes[-1] / closes[0]) ** (1 / n_years) - 1
    bh_cummax = np.maximum.accumulate(closes)
    bh_dd = np.min((closes - bh_cummax) / bh_cummax)
    print(f"  科创50: CAGR={bh_cagr*100:.1f}% MDD={bh_dd*100:.1f}%")

    # Save results
    if results:
        result_path = os.path.join(DATA_DIR, 'strategy_h_backtest.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nResults saved to {result_path}")

    return results


# ============================================================
# JSON Signal Output
# ============================================================
def output_json():
    """Output signals as JSON for integration."""
    signals = check_all_signals()
    result = {
        'strategy': 'H',
        'version': 'V2 (next-day open + 10bps costs)',
        'name': 'Index Dip-Buying & Rally-Chasing',
        'timestamp': datetime.now().isoformat(),
        'signals': signals,
        'variants': {k: {
            'name': v['name'],
            'direction': v['direction'],
            'cum_days': v['cum_days'],
            'threshold_pct': v['threshold_pct'],
            'hold_days': v['hold_days'],
            'stop_loss_pct': v['stop_loss_pct'],
            'oos_status': v['validation']['status'],
        } for k, v in VARIANTS.items()},
    }
    signal_path = os.path.join(DATA_DIR, 'strategy_h_latest_signal.json')
    with open(signal_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSignal saved to {signal_path}")
    return result


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    if '--backtest' in sys.argv:
        run_backtest()
    elif '--json' in sys.argv:
        output_json()
    else:
        check_all_signals()
