#!/usr/bin/env python3
"""
Strategy Q: 红利低波DY-国债利差择时
===================================
标的: 中证红利低波动全收益指数 (H20269) / ETF: 512890
逻辑: 当红利低波的估算股息率(DY)显著高于10年期国债收益率时持有,
      当利差收窄至阈值以下时卖出转货基。
信号: DY(45%分红率) - 10Y国债收益率
买入: 利差 > 3.3 (网格搜索最优参数)
卖出: 利差 < 0.0
空仓期假设2%年化货基收益。

回测结果 (2014-01 ~ 2026-04, 12年):
  年化: 17.6%
  最大回撤: 16.9%
  Calmar: 1.04
  交易次数: 5次
  当前信号: 持有 (利差3.27, 高于卖出阈值)

对应ETF: 512890 (红利低波ETF), 563020 (红利低波100ETF)
"""

import json
import urllib.request
import datetime
import sys
import os

# Add parent dir to path for lib imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Parameters ──────────────────────────────────────────────
PAYOUT_RATIO = 0.45       # Estimated dividend payout ratio
BUY_THRESHOLD = 3.3       # Buy when DY-BY spread > this (grid-search optimal)
SELL_THRESHOLD = 0.0      # Sell when DY-BY spread < this
CASH_RETURN = 0.02        # Annual return when in cash (money market)
TR_INDEX = 'H20269'       # Total return index (for NAV tracking)
PE_INDEX = 'H30269'       # Net return index (has PE/PEG data)
START_DATE = '20140101'


def fetch_url(url, headers=None):
    hdrs = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode('utf-8', errors='replace')


def fetch_csi_index(code, start=START_DATE, end=None):
    """Fetch daily index data from CSIndex official API."""
    if end is None:
        end = datetime.date.today().strftime('%Y%m%d')
    url = (f"https://www.csindex.com.cn/csindex-home/perf/index-perf"
           f"?indexCode={code}&startDate={start}&endDate={end}")
    data = fetch_url(url, {'Referer': 'https://www.csindex.com.cn/'})
    return json.loads(data).get('data', [])


def fetch_bond_yield():
    """Fetch 10Y China government bond yield from East Money."""
    bond = {}
    for page in range(1, 60):
        url = (f"https://datacenter-web.eastmoney.com/api/data/v1/get?"
               f"reportName=RPTA_WEB_TREASURYYIELD&columns=ALL&"
               f"pageNumber={page}&pageSize=500&"
               f"sortColumns=SOLAR_DATE&sortTypes=-1")
        raw = fetch_url(url)
        obj = json.loads(raw)
        if not obj.get('result') or not obj['result'].get('data'):
            break
        for row in obj['result']['data']:
            dt_str = row.get('SOLAR_DATE', '')[:10]
            y10 = row.get('EMM00166469')
            if dt_str and y10 is not None:
                dt_key = dt_str.replace('-', '')
                bond[dt_key] = float(y10)
        if page >= obj['result'].get('pages', 1):
            break
    return bond


def compute_signals(pe_data, bond_data, payout=PAYOUT_RATIO):
    """Compute DY-BY spread signals."""
    signals = {}
    for d in sorted(set(pe_data.keys()) & set(bond_data.keys())):
        pe = pe_data[d]
        if pe <= 0:
            continue
        ep = (1.0 / pe) * 100  # Earnings yield %
        dy = ep * payout        # Estimated dividend yield %
        by = bond_data[d]       # 10Y bond yield %
        signals[d] = dy - by
    return signals


def backtest(tr_prices, signals, buy_t=BUY_THRESHOLD, sell_t=SELL_THRESHOLD,
             cash_return=CASH_RETURN):
    """
    Run backtest with hysteresis thresholds.
    Returns dict with equity curve and stats.
    """
    common = sorted(set(tr_prices.keys()) & set(signals.keys()))
    if len(common) < 10:
        return None

    holding = False
    cash = tr_prices[common[0]]  # Start with cash equal to day-1 index value
    shares = 0
    trades = 0
    trade_log = []

    equity = []  # (date, value)
    peak = 0
    max_dd = 0

    for i, d in enumerate(common):
        spread = signals[d]
        p = tr_prices[d]

        # Signal logic
        if not holding and spread > buy_t:
            shares = cash / p
            cash = 0
            holding = True
            trades += 1
            trade_log.append((d, 'BUY', p, spread))
        elif holding and spread < sell_t:
            cash = shares * p
            shares = 0
            holding = False
            trades += 1
            trade_log.append((d, 'SELL', p, spread))

        # Portfolio value
        val = shares * p if holding else cash
        if not holding:
            cash *= (1 + cash_return / 250)
            val = cash

        equity.append((d, val))
        peak = max(peak, val)
        dd = (peak - val) / peak * 100
        max_dd = max(max_dd, dd)

    # Final stats
    start_val = equity[0][1]
    end_val = equity[-1][1]
    years = len(common) / 250
    total_ret = (end_val / start_val - 1) * 100
    ann_ret = ((end_val / start_val) ** (1 / years) - 1) * 100
    calmar = ann_ret / max_dd if max_dd > 0 else 999

    return {
        'equity': equity,
        'ann_ret': ann_ret,
        'total_ret': total_ret,
        'max_dd': max_dd,
        'calmar': calmar,
        'trades': trades,
        'trade_log': trade_log,
        'holding': holding,
        'start_date': common[0],
        'end_date': common[-1],
        'years': years,
        'current_spread': signals[common[-1]],
    }


def generate_equity_csv(equity, filepath):
    """Save equity curve to CSV for charting."""
    with open(filepath, 'w') as f:
        f.write('date,value\n')
        for d, v in equity:
            f.write(f'{d},{v:.2f}\n')


def main():
    today = datetime.date.today().strftime('%Y%m%d')

    print("=" * 60)
    print("Strategy Q: 红利低波DY-国债利差择时")
    print("=" * 60)

    # Fetch data
    print("\n[1/3] Fetching index data...")
    tr_raw = fetch_csi_index(TR_INDEX, START_DATE, today)
    pe_raw = fetch_csi_index(PE_INDEX, START_DATE, today)
    print(f"  H20269 (Total Return): {len(tr_raw)} days")
    print(f"  H30269 (Net Return/PE): {len(pe_raw)} days")

    tr_prices = {d['tradeDate']: float(d['close'])
                 for d in tr_raw if d.get('close') is not None}
    pe_data = {d['tradeDate']: float(d['peg'])
               for d in pe_raw if d.get('peg') is not None and float(d['peg']) > 0}

    print("\n[2/3] Fetching 10Y bond yield...")
    bond = fetch_bond_yield()
    print(f"  Bond yield data: {len(bond)} days")

    # Compute signals
    signals = compute_signals(pe_data, bond)
    print(f"  Signal data: {len(signals)} days")

    # Run backtest
    print("\n[3/3] Running backtest...")
    result = backtest(tr_prices, signals)

    if result is None:
        print("ERROR: Not enough data for backtest")
        return

    # Print results
    print(f"\n{'─' * 50}")
    print(f"回测期间: {result['start_date']} ~ {result['end_date']} ({result['years']:.1f}年)")
    print(f"年化收益: {result['ann_ret']:.1f}%")
    print(f"总收益:   {result['total_ret']:.1f}%")
    print(f"最大回撤: {result['max_dd']:.1f}%")
    print(f"Calmar:   {result['calmar']:.2f}")
    print(f"交易次数: {result['trades']}次")
    print(f"{'─' * 50}")

    # Buy and hold comparison
    eq = result['equity']
    bh_start = tr_prices[eq[0][0]]
    bh_end = tr_prices[eq[-1][0]]
    bh_ann = ((bh_end / bh_start) ** (1 / result['years']) - 1) * 100
    print(f"\n买入持有: 年化{bh_ann:.1f}%")
    print(f"超额收益: {result['ann_ret'] - bh_ann:+.1f}%/年")

    # Current signal
    print(f"\n{'─' * 50}")
    print(f"当前信号:")
    print(f"  DY-BY利差: {result['current_spread']:.2f}")
    print(f"  买入阈值:  {BUY_THRESHOLD}")
    print(f"  卖出阈值:  {SELL_THRESHOLD}")
    status = "持有" if result['holding'] else "空仓"
    print(f"  当前状态:  {status}")

    if result['current_spread'] > BUY_THRESHOLD:
        print(f"  建议: 维持持有，利差充足")
    elif result['current_spread'] > SELL_THRESHOLD:
        print(f"  建议: 持有但关注，利差收窄中")
    else:
        print(f"  建议: 空仓/卖出，利差不足")

    # Trade log
    print(f"\n{'─' * 50}")
    print("交易记录:")
    for d, action, price, spread in result['trade_log']:
        print(f"  {d[:4]}-{d[4:6]}-{d[6:]} {action:4s} 指数={price:.1f} 利差={spread:.2f}")

    # Save equity curve
    eq_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data', 'strategy_q_equity.csv')
    os.makedirs(os.path.dirname(eq_path), exist_ok=True)
    generate_equity_csv(result['equity'], eq_path)
    print(f"\n净值曲线已保存: {eq_path}")

    return result


if __name__ == '__main__':
    main()
