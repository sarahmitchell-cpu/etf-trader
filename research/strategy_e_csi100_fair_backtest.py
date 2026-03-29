#!/usr/bin/env python3
"""
Strategy E - CSI 100 Fair Backtest (Survivorship-Bias-Free)
Uses historical constituent data from fund quarterly reports.
At each rebalancing date, uses the ACTUAL constituents at that time.
"""

import json, os, time, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import akshare as ak

warnings.filterwarnings('ignore')

DATA_DIR = "/Users/claw/etf-trader/data"
HIST_FILE = os.path.join(DATA_DIR, "csi100_historical_constituents.json")
CACHE_DIR = os.path.join(DATA_DIR, "price_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Strategy E parameters
VALUE_LOOKBACK = 10   # weeks for reversal signal
VOL_LOOKBACK = 12     # weeks for volatility
VOL_WEIGHT = 0.2      # weight for low-vol factor
REBAL_FREQ = 4        # weeks between rebalances
TXN_COST_BPS = 8      # transaction cost in bps
SECTOR_MAX = 1        # max stocks per sector
MAX_DD_FILTER = -0.20 # skip stocks with drawdown worse than this

# Sector classification by stock name keywords
SECTOR_MAP = {
    '银行': '银行', '农业银行': '银行', '工商银行': '银行', '建设银行': '银行',
    '招商银行': '银行', '兴业银行': '银行', '民生银行': '银行', '光大银行': '银行',
    '平安银行': '银行', '浦发银行': '银行', '中信银行': '银行', '华夏银行': '银行',
    '交通银行': '银行', '邮储银行': '银行', '北京银行': '银行', '上海银行': '银行',
    '宁波银行': '银行',
    '保险': '保险', '中国平安': '保险', '中国人寿': '保险', '中国太保': '保险',
    '新华保险': '保险', '中国人保': '保险',
    '券商': '券商', '中信证券': '券商', '华泰证券': '券商', '国泰君安': '券商',
    '中信建投': '券商', '招商证券': '券商', '中金公司': '券商', '国信证券': '券商',
    '申万宏源': '券商', '海通证券': '券商', '东方财富': '券商', '光大证券': '券商',
    '茅台': '白酒', '五粮液': '白酒', '泸州老窖': '白酒', '洋河': '白酒', '汾酒': '白酒',
    '药': '医药', '医': '医药', '恒瑞': '医药', '迈瑞': '医药', '片仔癀': '医药',
    '爱美客': '医药', '云南白药': '医药', '沃森': '医药', '科伦': '医药', '莱士': '医药',
    '联影': '医药',
    '锂': '新能源', '宁德时代': '新能源', '亿纬': '新能源', '阳光电源': '新能源',
    '天合光能': '新能源', '隆基': '新能源', 'TCL中环': '新能源',
    '芯': '半导体', '中芯': '半导体', '北方华创': '半导体', '兆易创新': '半导体',
    '澜起': '半导体', '中微': '半导体', '海光': '半导体', '寒武纪': '半导体',
    '长电科技': '半导体', '圣邦': '半导体', '沪电': '半导体',
    '中国移动': '电信', '中国电信': '电信', '中国联通': '电信',
    '石油': '能源', '石化': '能源', '中国海油': '能源', '中国神华': '能源',
    '煤': '能源', '宝丰能源': '能源',
    '电力': '电力', '长江电力': '电力', '三峡': '电力', '核电': '电力', '电建': '电力',
    '汽车': '汽车', '比亚迪': '汽车', '长城汽车': '汽车', '上汽': '汽车',
    '地产': '地产', '万科': '地产', '保利': '地产', '招商蛇口': '地产',
    '钢': '钢铁', '铝': '钢铁', '铜': '有色', '黄金': '有色', '紫金': '有色',
    '稀土': '有色', '锂业': '有色', '钴': '有色', '钼': '有色', '盐湖': '有色',
    '中金黄金': '有色',
    '美的': '家电', '格力': '家电', '海尔': '家电',
    '科大讯飞': 'AI', '金山办公': 'AI', '中际旭创': 'AI',
    '军工': '军工', '航发': '军工', '中航': '军工',
    '中国船舶': '军工', '中国重工': '军工',
    '建筑': '建筑', '中国建筑': '建筑', '中国中铁': '建筑', '中国铁建': '建筑',
    '徐工': '机械', '潍柴': '机械', '三一': '机械',
}

def assign_sector(name):
    for kw, sector in SECTOR_MAP.items():
        if kw in name:
            return sector
    return name[:2]  # fallback: first 2 chars


def load_historical_constituents():
    with open(HIST_FILE, 'r') as f:
        data = json.load(f)
    # Convert to date-keyed dict
    result = {}
    for date_str, info in data.items():
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        result[dt] = info
    return result


def get_constituents_for_date(hist_data, target_date):
    """Get the constituent list that was effective at target_date."""
    dates = sorted(hist_data.keys())
    # Find the most recent constituent list before or on target_date
    effective = None
    for dt in dates:
        if dt <= target_date:
            effective = dt
        else:
            break
    if effective is None:
        effective = dates[0]  # use earliest if before all data
    return hist_data[effective]


def fetch_weekly_price(code, name, start='20200101', end='20260401'):
    """Fetch weekly price data with caching. Uses East Money API."""
    cache_file = os.path.join(CACHE_DIR, f"{code}_weekly.pkl")
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 7 * 86400:  # 7 day cache
            try:
                return pd.read_pickle(cache_file)
            except:
                pass

    try:
        # Use East Money API (stock_zh_a_hist) - more reliable
        df = ak.stock_zh_a_hist(symbol=str(code).zfill(6), period="weekly",
                                start_date=start, end_date=end, adjust="hfq")
        if df is None or df.empty:
            return None
        # EM API returns: 日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
        df['date'] = pd.to_datetime(df['日期'])
        df = df.set_index('date').sort_index()
        df = df[['收盘']].rename(columns={'收盘': 'close'})
        df.to_pickle(cache_file)
        return df
    except Exception as e:
        return None


def compute_signals(weekly_prices, date_idx, value_lb=VALUE_LOOKBACK, vol_lb=VOL_LOOKBACK):
    """Compute reversal + low-vol signals for each stock at a given date index."""
    signals = {}
    for code, df in weekly_prices.items():
        if df is None or len(df) < max(value_lb, vol_lb) + 1:
            continue
        # Get data up to date_idx
        mask = df.index <= date_idx
        sub = df[mask]
        if len(sub) < max(value_lb, vol_lb) + 1:
            continue

        close = sub['close'].values
        # Reversal signal: negative past return = buy losers
        if close[-value_lb - 1] <= 0:
            continue
        past_ret = close[-1] / close[-value_lb - 1] - 1

        # Recent drawdown filter
        peak = np.max(close[-vol_lb:])
        if peak <= 0:
            continue
        dd = close[-1] / peak - 1
        if dd < MAX_DD_FILTER:
            continue

        # Volatility: lower = better
        rets = np.diff(np.log(close[-vol_lb - 1:]))
        vol = np.std(rets) if len(rets) > 1 else 999

        signals[code] = {
            'past_ret': past_ret,
            'vol': vol,
            'dd': dd,
            'price': close[-1],
        }
    return signals


def select_stocks(signals, names, top_n=8, vol_weight=VOL_WEIGHT, sector_max=SECTOR_MAX):
    """Rank and select top N stocks by combined reversal + low-vol score."""
    if len(signals) < top_n:
        return []

    codes = list(signals.keys())
    past_rets = np.array([signals[c]['past_ret'] for c in codes])
    vols = np.array([signals[c]['vol'] for c in codes])

    # Rank: lower past return = higher reversal score (buy losers)
    ret_rank = np.argsort(np.argsort(past_rets))  # ascending: lowest ret gets rank 0
    # Rank: lower vol = higher score
    vol_rank = np.argsort(np.argsort(vols))  # ascending: lowest vol gets rank 0

    n = len(codes)
    # Combined score: lower rank = better (reversal: buy losers, vol: buy low vol)
    combined = (1 - vol_weight) * ret_rank / n + vol_weight * vol_rank / n

    # Sort by combined score (ascending = best)
    order = np.argsort(combined)

    selected = []
    sector_count = {}
    for idx in order:
        if len(selected) >= top_n:
            break
        code = codes[idx]
        name = names.get(code, code)
        sector = assign_sector(name)
        if sector_max > 0 and sector_count.get(sector, 0) >= sector_max:
            continue
        selected.append(code)
        sector_count[sector] = sector_count.get(sector, 0) + 1

    return selected


def run_fair_backtest(top_n=8, start_date='2021-01-01', end_date='2026-03-28'):
    """Run Strategy E backtest with historical constituents (no survivorship bias)."""
    hist_data = load_historical_constituents()

    # Collect ALL unique stock codes across all periods
    all_codes = set()
    all_names = {}
    for dt, info in hist_data.items():
        for code in info['codes']:
            all_codes.add(code)
            all_names[code] = info['names'].get(code, code)

    print(f"Total unique stocks across all periods: {len(all_codes)}")

    # Fetch price data for all stocks
    print("Fetching price data...")
    weekly_prices = {}
    failed = []
    for i, code in enumerate(sorted(all_codes)):
        name = all_names.get(code, code)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(all_codes)}] {code} {name}")
        df = fetch_weekly_price(code, name)
        if df is not None and len(df) > 20:
            weekly_prices[code] = df
        else:
            failed.append(code)
        time.sleep(0.05)

    print(f"Loaded {len(weekly_prices)} stocks, {len(failed)} failed")
    if failed:
        print(f"  Failed: {', '.join(failed[:10])}")

    # Generate weekly rebalancing dates
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    all_fridays = pd.date_range(start_dt, end_dt, freq='W-FRI')

    # Run backtest
    portfolio_value = 1.0
    portfolio_values = [portfolio_value]
    portfolio_dates = [all_fridays[0]]
    holdings = []
    rebal_count = 0
    week_count = 0

    for i in range(1, len(all_fridays)):
        date = all_fridays[i]
        prev_date = all_fridays[i - 1]
        week_count += 1

        # Calculate return for current holdings
        if holdings:
            week_rets = []
            for code in holdings:
                if code in weekly_prices:
                    df = weekly_prices[code]
                    prev_prices = df[df.index <= prev_date]
                    curr_prices = df[df.index <= date]
                    if len(prev_prices) > 0 and len(curr_prices) > 0:
                        ret = curr_prices['close'].iloc[-1] / prev_prices['close'].iloc[-1] - 1
                        week_rets.append(ret)
            if week_rets:
                avg_ret = np.mean(week_rets)
                portfolio_value *= (1 + avg_ret)

        # Rebalance every REBAL_FREQ weeks
        if week_count % REBAL_FREQ == 0:
            # Get constituents for this date
            constituents = get_constituents_for_date(hist_data, date)
            pool_codes = [c for c in constituents['codes'] if c in weekly_prices]
            pool_names = constituents['names']

            # Filter to stocks in pool that have enough data
            pool_prices = {c: weekly_prices[c] for c in pool_codes}

            # Compute signals only for current pool
            signals = compute_signals(pool_prices, date)

            if signals:
                new_holdings = select_stocks(signals, pool_names, top_n=top_n)

                # Transaction cost
                if holdings:
                    turnover = len(set(new_holdings) - set(holdings)) / max(len(new_holdings), 1)
                    txn_cost = turnover * TXN_COST_BPS / 10000 * 2  # buy + sell
                    portfolio_value *= (1 - txn_cost)

                holdings = new_holdings
                rebal_count += 1

        portfolio_values.append(portfolio_value)
        portfolio_dates.append(date)

    # Calculate metrics
    pv = np.array(portfolio_values)
    dates_arr = np.array(portfolio_dates)
    years = (dates_arr[-1] - dates_arr[0]).days / 365.25

    total_ret = pv[-1] / pv[0] - 1
    cagr = (pv[-1] / pv[0]) ** (1 / years) - 1 if years > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(pv)
    dd = pv / peak - 1
    mdd = np.min(dd)

    # Sharpe (weekly returns annualized)
    weekly_rets = np.diff(pv) / pv[:-1]
    sharpe = np.mean(weekly_rets) / np.std(weekly_rets) * np.sqrt(52) if np.std(weekly_rets) > 0 else 0

    calmar = cagr / abs(mdd) if mdd != 0 else 0

    return {
        'top_n': top_n,
        'cagr': round(cagr * 100, 1),
        'mdd': round(mdd * 100, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'total_ret': round(total_ret * 100, 1),
        'years': round(years, 1),
        'rebal_count': rebal_count,
        'final_holdings': holdings,
    }


def main():
    print("=" * 60)
    print("Strategy E - CSI 100 Fair Backtest (No Survivorship Bias)")
    print("=" * 60)
    print(f"Parameters: value_lb={VALUE_LOOKBACK}, vol_lb={VOL_LOOKBACK}, "
          f"vol_wt={VOL_WEIGHT}, rebal={REBAL_FREQ}w, sector_max={SECTOR_MAX}")
    print()

    results = []
    for top_n in [4, 5, 6, 8, 10]:
        print(f"\n--- Running Top{top_n} ---")
        r = run_fair_backtest(top_n=top_n)
        results.append(r)
        print(f"  CAGR: {r['cagr']}%, MDD: {r['mdd']}%, Sharpe: {r['sharpe']}, Calmar: {r['calmar']}")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Fair Backtest vs Previous Survivorship-Biased)")
    print("=" * 60)
    print(f"{'Config':<20} {'CAGR%':>6} {'MDD%':>7} {'Sharpe':>7} {'Calmar':>7}")
    print("-" * 50)
    for r in results:
        label = f"CSI100_Fair_Top{r['top_n']}"
        print(f"{label:<20} {r['cagr']:>6} {r['mdd']:>7} {r['sharpe']:>7} {r['calmar']:>7}")

    # Save results
    output_file = os.path.join(DATA_DIR, "strategy_e_csi100_fair_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
