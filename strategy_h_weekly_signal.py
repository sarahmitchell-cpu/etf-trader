#!/usr/bin/env python3
"""
Strategy H: Index Dip-Buying & Rally-Chasing (指数超跌买入/追涨买入)

8 variants selected from exhaustive parameter search (V5):
  - H1: 科创50 超跌 3日跌>7% → 持19日           Sharpe=1.42  CAGR=21.7%
  - H2: 沪深300 超跌 8日跌>4% → 持11日(SL-3%)    Sharpe=1.33  CAGR=15.5%
  - H3: 科创50 追涨 5日涨>6% → 持14日(SL-5%)     Sharpe=1.00  CAGR=22.3%
  - H4: 科创50 追涨 1日涨>3% → 持2日             Sharpe=0.95  CAGR=16.3%
  - H5: 恒生指数 超跌 5日跌>5% → 持6日(SL-7%)    Sharpe=0.61  CAGR=8.0%
  - H6: 上证50ETF 超跌 6日跌>4% → 持4日          Sharpe=0.51  CAGR=7.5%
  - H7: 通用 超跌 6日跌>6% → 持4日 (6指数覆盖)   avg Sharpe=0.42
  - H8: 沪深300 追涨 1日涨>2% → 持2日            Sharpe=0.85  CAGR=7.2%

Research backtest (V5): 200,094+ strategies across 6-8 indices, cum_days 1-10,
  threshold 2-20%, hold_days 1-50, stop-loss None/-3/-5/-7%.

Usage:
  python3 strategy_h_weekly_signal.py              # Check today's signals
  python3 strategy_h_weekly_signal.py --json        # JSON only
  python3 strategy_h_weekly_signal.py --backtest    # Full backtest all variants
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

RISK_FREE_RATE = 0.02

# ============================================================
# 8 Strategy Variants
# ============================================================
VARIANTS = {
    'H1': {
        'name': 'H1: 科创50 超跌 3日跌>7% 持19日',
        'ticker': '588000.SS',
        'index_name': '科创50ETF',
        'direction': 'dip',      # 超跌买入
        'cum_days': 3,           # 累计天数
        'threshold_pct': 7,      # 跌幅阈值%
        'hold_days': 19,         # 持有天数
        'stop_loss_pct': None,   # 止损%（None=不止损）
        'backtest': {'sharpe': 1.42, 'cagr': 21.7, 'mdd': -9.6, 'win_rate': 75, 'n_trades': 12},
    },
    'H2': {
        'name': 'H2: 沪深300 超跌 8日跌>4% 持11日(SL-3%)',
        'ticker': '000300.SS',
        'index_name': '沪深300',
        'direction': 'dip',
        'cum_days': 8,
        'threshold_pct': 4,
        'hold_days': 11,
        'stop_loss_pct': -3,
        'backtest': {'sharpe': 1.33, 'cagr': 15.5, 'mdd': -7.7, 'win_rate': 76, 'n_trades': 29},
    },
    'H3': {
        'name': 'H3: 科创50 追涨 5日涨>6% 持14日(SL-5%)',
        'ticker': '588000.SS',
        'index_name': '科创50ETF',
        'direction': 'rally',    # 追涨买入
        'cum_days': 5,
        'threshold_pct': 6,
        'hold_days': 14,
        'stop_loss_pct': -5,
        'backtest': {'sharpe': 1.00, 'cagr': 22.3, 'mdd': -19.0, 'win_rate': 79, 'n_trades': 19},
    },
    'H4': {
        'name': 'H4: 科创50 追涨 1日涨>3% 持2日',
        'ticker': '588000.SS',
        'index_name': '科创50ETF',
        'direction': 'rally',
        'cum_days': 1,
        'threshold_pct': 3,
        'hold_days': 2,
        'stop_loss_pct': None,
        'backtest': {'sharpe': 0.95, 'cagr': 16.3, 'mdd': -6.0, 'win_rate': 69, 'n_trades': 42},
    },
    'H5': {
        'name': 'H5: 恒生指数 超跌 5日跌>5% 持6日(SL-7%)',
        'ticker': '^HSI',
        'index_name': '恒生指数',
        'direction': 'dip',
        'cum_days': 5,
        'threshold_pct': 5,
        'hold_days': 6,
        'stop_loss_pct': -7,
        'backtest': {'sharpe': 0.61, 'cagr': 8.0, 'mdd': -14.1, 'win_rate': 61, 'n_trades': 59},
    },
    'H6': {
        'name': 'H6: 上证50ETF 超跌 6日跌>4% 持4日',
        'ticker': '510050.SS',
        'index_name': '上证50ETF',
        'direction': 'dip',
        'cum_days': 6,
        'threshold_pct': 4,
        'hold_days': 4,
        'stop_loss_pct': None,
        'backtest': {'sharpe': 0.51, 'cagr': 7.5, 'mdd': -20.3, 'win_rate': 54, 'n_trades': 81},
    },
    'H7': {
        'name': 'H7: 通用 超跌 6日跌>6% 持4日 (多指数)',
        'ticker': None,  # applies to all indices
        'index_name': '通用(6指数)',
        'direction': 'dip',
        'cum_days': 6,
        'threshold_pct': 6,
        'hold_days': 4,
        'stop_loss_pct': None,
        'backtest': {'sharpe': 0.42, 'cagr': 5.1, 'mdd': -15.0, 'win_rate': 55, 'n_trades': 'varies'},
    },
    'H8': {
        'name': 'H8: 沪深300 追涨 1日涨>2% 持2日',
        'ticker': '000300.SS',
        'index_name': '沪深300',
        'direction': 'rally',
        'cum_days': 1,
        'threshold_pct': 2,
        'hold_days': 2,
        'stop_loss_pct': None,
        'backtest': {'sharpe': 0.85, 'cagr': 7.2, 'mdd': -3.1, 'win_rate': 65, 'n_trades': 31},
    },
}

# All tickers needed for signal checking
ALL_TICKERS = {
    '科创50ETF': '588000.SS',
    '沪深300': '000300.SS',
    '上证50ETF': '510050.SS',
    '恒生指数': '^HSI',
    '国企指数': '^HSCE',
    'H股ETF': '510900.SS',
}


# ============================================================
# Signal Generation
# ============================================================
def check_signal(closes, cum_days, threshold_pct, direction):
    """
    Check if the latest bar triggers a buy signal.

    For dip: cum_days日累计跌幅 >= threshold_pct%
    For rally: cum_days日累计涨幅 >= threshold_pct%

    Returns: (triggered: bool, actual_change_pct: float)
    """
    if len(closes) < cum_days + 1:
        return False, 0.0

    current = closes[-1]
    past = closes[-(cum_days + 1)]
    change_pct = (current / past - 1) * 100

    if direction == 'dip':
        triggered = change_pct <= -threshold_pct
    else:  # rally
        triggered = change_pct >= threshold_pct

    return triggered, change_pct


def download_latest(ticker, lookback_days=60):
    """Download recent data for signal checking."""
    try:
        df = yf.download(ticker, period=f'{lookback_days}d', progress=False)
        if df is not None and len(df) > 0:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df['Close'].dropna().values
    except Exception as e:
        print(f"  Download failed for {ticker}: {e}")
    return None


# ============================================================
# Backtest Engine
# ============================================================
def backtest_variant(closes, variant):
    """Run backtest for a single variant on given close prices."""
    n = len(closes)
    cum_days = variant['cum_days']
    threshold_pct = variant['threshold_pct']
    hold_days = variant['hold_days']
    stop_loss_pct = variant['stop_loss_pct']
    direction = variant['direction']

    # Pre-compute cumulative returns
    cum_ret = np.zeros(n)
    for d in range(cum_days, n):
        cum_ret[d] = (closes[d] / closes[d - cum_days] - 1) * 100

    # Generate signals
    if direction == 'dip':
        signals = cum_ret <= -threshold_pct
    else:
        signals = cum_ret >= threshold_pct
    signals[:cum_days] = False

    # Backtest
    nav = np.ones(n)
    position = False
    entry_price = 0.0
    hold_count = 0
    trades = []

    for i in range(1, n):
        if position:
            daily_ret = closes[i] / closes[i - 1] - 1
            nav[i] = nav[i - 1] * (1 + daily_ret)
            hold_count += 1

            current_trade_ret = closes[i] / entry_price - 1
            hit_stop = False
            if stop_loss_pct is not None and current_trade_ret * 100 <= stop_loss_pct:
                hit_stop = True

            if hold_count >= hold_days or hit_stop:
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'return': current_trade_ret,
                    'hold_days': hold_count,
                    'stop_loss': hit_stop,
                })
                position = False
        else:
            nav[i] = nav[i - 1]
            if signals[i]:
                position = True
                entry_price = closes[i]
                entry_idx = i
                hold_count = 0

    # Force close
    if position:
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': n - 1,
            'return': closes[-1] / entry_price - 1,
            'hold_days': hold_count,
            'stop_loss': False,
        })

    return nav, trades


def calc_metrics(nav, trades, n_days):
    """Calculate performance metrics."""
    if n_days < 2:
        return None

    n_years = n_days / 252
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
        rets = [t['return'] for t in trades]
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
    """Check all 8 variants for today's signal."""
    print("=" * 70)
    print(f"Strategy H Signal Check - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Download data for all needed tickers
    price_cache = {}
    tickers_needed = set()
    for vid, v in VARIANTS.items():
        if v['ticker']:
            tickers_needed.add((v['index_name'], v['ticker']))
        else:
            # H7 universal - check all
            for name, ticker in ALL_TICKERS.items():
                tickers_needed.add((name, ticker))

    for name, ticker in tickers_needed:
        closes = download_latest(ticker)
        if closes is not None and len(closes) > 0:
            price_cache[name] = closes
            print(f"  {name}: {len(closes)} bars, latest={closes[-1]:.2f}")
        else:
            print(f"  {name}: FAILED to download")

    print()

    signals = []
    for vid, v in VARIANTS.items():
        if v['ticker']:
            # Single-index variant
            idx_name = v['index_name']
            if idx_name not in price_cache:
                print(f"  {vid}: {v['name']} -> NO DATA")
                continue

            closes = price_cache[idx_name]
            triggered, change = check_signal(
                closes, v['cum_days'], v['threshold_pct'], v['direction']
            )

            direction_zh = '跌' if v['direction'] == 'dip' else '涨'
            sl_str = f"止损{v['stop_loss_pct']}%" if v['stop_loss_pct'] else '不止损'
            status = "BUY SIGNAL!" if triggered else "no signal"

            print(f"  {vid}: {v['index_name']} {v['cum_days']}日{direction_zh}: "
                  f"{change:+.2f}% (阈值{'-' if v['direction']=='dip' else '+'}{v['threshold_pct']}%) "
                  f"-> {status}")

            if triggered:
                signals.append({
                    'variant': vid,
                    'name': v['name'],
                    'index': idx_name,
                    'action': 'BUY',
                    'hold_days': v['hold_days'],
                    'stop_loss': v['stop_loss_pct'],
                    'change_pct': round(change, 2),
                })
        else:
            # H7 universal - check all indices
            print(f"  {vid}: {v['name']}")
            for idx_name, closes in price_cache.items():
                triggered, change = check_signal(
                    closes, v['cum_days'], v['threshold_pct'], v['direction']
                )
                direction_zh = '跌' if v['direction'] == 'dip' else '涨'
                status = "BUY!" if triggered else "-"
                print(f"      {idx_name}: {v['cum_days']}日{direction_zh} {change:+.2f}% -> {status}")
                if triggered:
                    signals.append({
                        'variant': vid,
                        'name': v['name'],
                        'index': idx_name,
                        'action': 'BUY',
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
            print(f"  {s['variant']}: {s['index']} -> BUY, 持{s['hold_days']}日, {sl}")
    else:
        print("No active signals today.")

    return signals


# ============================================================
# Full Backtest
# ============================================================
def run_backtest():
    """Run full backtest for all 8 variants."""
    print("=" * 70)
    print("Strategy H Full Backtest")
    print("=" * 70)

    START_DATE = '2015-01-01'
    END_DATE = datetime.now().strftime('%Y-%m-%d')

    # Download data
    data = {}
    for name, ticker in ALL_TICKERS.items():
        print(f"Downloading {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if df is not None and len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                closes = df['Close'].dropna().values
                data[name] = closes
                print(f"  -> {len(closes)} trading days")
            else:
                print(f"  -> insufficient data, skipped")
        except Exception as e:
            print(f"  -> failed: {e}")

    print(f"\nLoaded {len(data)} indices\n")

    # Backtest each variant
    results = []
    for vid, v in VARIANTS.items():
        print(f"\n{'='*50}")
        print(f"{vid}: {v['name']}")
        print(f"{'='*50}")

        if v['ticker']:
            # Single-index variant
            idx_name = v['index_name']
            if idx_name not in data:
                print(f"  -> No data for {idx_name}")
                continue

            closes = data[idx_name]
            nav, trades = backtest_variant(closes, v)
            metrics = calc_metrics(nav, trades, len(closes))

            if metrics:
                print(f"  CAGR:      {metrics['annualized_return']*100:>8.1f}%")
                print(f"  MDD:       {metrics['max_drawdown']*100:>8.1f}%")
                print(f"  Sharpe:    {metrics['sharpe']:>8.2f}")
                print(f"  Calmar:    {metrics['calmar']:>8.2f}")
                print(f"  Win Rate:  {metrics['win_rate']*100:>8.0f}%")
                print(f"  Trades:    {metrics['n_trades']:>8d}")
                print(f"  P/L Ratio: {metrics['profit_loss_ratio']:>8.2f}")

                results.append({
                    'variant': vid,
                    'index': idx_name,
                    **metrics,
                })
        else:
            # H7 universal - test on all indices
            for idx_name, closes in data.items():
                nav, trades = backtest_variant(closes, v)
                metrics = calc_metrics(nav, trades, len(closes))

                if metrics and metrics['n_trades'] >= 3:
                    print(f"  {idx_name}: CAGR={metrics['annualized_return']*100:.1f}% "
                          f"MDD={metrics['max_drawdown']*100:.1f}% "
                          f"Sharpe={metrics['sharpe']:.2f} "
                          f"Trades={metrics['n_trades']}")
                    results.append({
                        'variant': vid,
                        'index': idx_name,
                        **metrics,
                    })

    # Buy-and-hold benchmark
    print(f"\n{'='*50}")
    print("Buy & Hold Benchmark")
    print(f"{'='*50}")
    for idx_name, closes in data.items():
        n_years = len(closes) / 252
        bh_cagr = (closes[-1] / closes[0]) ** (1 / n_years) - 1
        bh_cummax = np.maximum.accumulate(closes)
        bh_dd = np.min((closes - bh_cummax) / bh_cummax)
        print(f"  {idx_name}: CAGR={bh_cagr*100:.1f}% MDD={bh_dd*100:.1f}%")

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
