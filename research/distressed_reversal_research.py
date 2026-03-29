#!/usr/bin/env python3
"""
大盘股困境反转研究
Distressed Reversal Strategy Research for A-share Large Caps

目标：
1. 找到过去10年成功困境反转的大盘股案例
2. 分析反转前的财务数据特征
3. 提炼可量化的信号指标
4. 设计监控框架

定义：
- 大盘股：市值200亿以上 或 沪深300/中证500成分股
- 困境：连续2+季度利润同比下降50%+，或年度亏损
- 反转：股价从低点上涨100%+ 或 利润恢复至困境前水平

数据源: baostock(价格/财务), akshare(辅助)

Author: Sarah Mitchell / VisionClaw
Date: 2026-03-28
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import baostock as bs
import json, os, traceback, time
from datetime import datetime, timedelta

DATA_DIR = '/Users/claw/etf-trader/data'

# ============================================================
# 1. DATA FUNCTIONS
# ============================================================

def get_csi300_500_constituents():
    """Get current CSI300 + CSI500 constituents"""
    print("[1] Getting index constituents...")
    all_stocks = {}

    for idx_code, idx_name in [('sh.000300', '沪深300'), ('sh.000905', '中证500')]:
        rs = bs.query_hs300_stocks() if '300' in idx_code else bs.query_zz500_stocks()
        data = []
        while (rs.error_code == '0') and rs.next():
            data.append(rs.get_row_data())
        if data:
            df = pd.DataFrame(data, columns=rs.fields)
            for _, row in df.iterrows():
                code = row['code'] if 'code' in row else row.get('code', '')
                name = row['code_name'] if 'code_name' in row else ''
                all_stocks[code] = name
            print(f"  {idx_name}: {len(data)} stocks")

    print(f"  Total unique: {len(all_stocks)} stocks")
    return all_stocks


def get_stock_price(code, start='2014-01-01', end='2026-03-28'):
    """Get daily stock price"""
    rs = bs.query_history_k_data_plus(
        code, "date,open,high,low,close,volume,amount",
        start_date=start, end_date=end, frequency="d",
        adjustflag="2"  # 前复权
    )
    data = []
    while (rs.error_code == '0') and rs.next():
        data.append(rs.get_row_data())
    if not data:
        return None
    df = pd.DataFrame(data, columns=['date','open','high','low','close','volume','amount'])
    for c in ['open','high','low','close','volume','amount']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df = df[df['close'] > 0]
    return df


def get_quarterly_financials(code):
    """Get quarterly profit/growth data from baostock"""
    results = []

    for year in range(2014, 2027):
        for quarter in [1, 2, 3, 4]:
            rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
            data = []
            while (rs.error_code == '0') and rs.next():
                data.append(rs.get_row_data())
            if data:
                for row in data:
                    results.append(row)

    if not results:
        return None

    # Get field names from last query
    df = pd.DataFrame(results, columns=rs.fields if rs.fields else
                      ['code','pubDate','statDate','roeAvg','npMargin','gpMargin',
                       'netProfit','epsTTM','MBRevenue','totalShare','liqaShare'])

    # Convert numeric columns
    for c in df.columns:
        if c not in ['code', 'pubDate', 'statDate']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['statDate'] = pd.to_datetime(df['statDate'], errors='coerce')
    df = df.dropna(subset=['statDate']).sort_values('statDate')

    return df


def get_growth_data(code):
    """Get quarterly growth data from baostock"""
    results = []

    for year in range(2014, 2027):
        for quarter in [1, 2, 3, 4]:
            rs = bs.query_growth_data(code=code, year=year, quarter=quarter)
            data = []
            while (rs.error_code == '0') and rs.next():
                data.append(rs.get_row_data())
            if data:
                for row in data:
                    results.append(row)

    if not results:
        return None

    df = pd.DataFrame(results, columns=rs.fields if rs.fields else
                      ['code','pubDate','statDate','YOYEquity','YOYAsset',
                       'YOYNI','YOYEPSBasic','YOYPNI'])

    for c in df.columns:
        if c not in ['code', 'pubDate', 'statDate']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['statDate'] = pd.to_datetime(df['statDate'], errors='coerce')
    df = df.dropna(subset=['statDate']).sort_values('statDate')

    return df


def get_cash_flow_data(code):
    """Get quarterly cash flow data from baostock"""
    results = []

    for year in range(2014, 2027):
        for quarter in [1, 2, 3, 4]:
            rs = bs.query_cash_flow_data(code=code, year=year, quarter=quarter)
            data = []
            while (rs.error_code == '0') and rs.next():
                data.append(rs.get_row_data())
            if data:
                for row in data:
                    results.append(row)

    if not results:
        return None

    df = pd.DataFrame(results, columns=rs.fields if rs.fields else
                      ['code','pubDate','statDate','CAToAsset','NCAToAsset',
                       'tangibleAssetToAsset','ebitToInterest','CFOToOR',
                       'CFOToNP','CFOToGr'])

    for c in df.columns:
        if c not in ['code', 'pubDate', 'statDate']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['statDate'] = pd.to_datetime(df['statDate'], errors='coerce')
    df = df.dropna(subset=['statDate']).sort_values('statDate')

    return df


# ============================================================
# 2. IDENTIFY DISTRESSED REVERSAL CASES
# ============================================================

def identify_distressed_stocks(stocks_dict, max_stocks=200):
    """
    Screen for stocks that experienced distress + subsequent recovery.

    Distress criteria:
    - Stock price dropped 50%+ from peak
    - Net profit YoY growth < -50% for 2+ quarters

    Recovery criteria:
    - Stock price recovered 80%+ from trough
    - Or net profit turned positive after loss
    """
    print("\n[2] Screening for distressed reversal candidates...")

    candidates = []
    stock_list = list(stocks_dict.keys())[:max_stocks]

    for i, code in enumerate(stock_list):
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(stock_list)}...")

        try:
            # Get price data
            price_df = get_stock_price(code, start='2014-01-01')
            if price_df is None or len(price_df) < 500:
                continue

            close = price_df['close']

            # Find significant drawdowns (50%+ from rolling 1-year high)
            rolling_max = close.rolling(252, min_periods=60).max()
            drawdown = close / rolling_max - 1

            # Find periods where drawdown exceeded -50%
            distress_mask = drawdown < -0.50
            if not distress_mask.any():
                continue

            # Find drawdown troughs
            # Group consecutive distress periods
            distress_periods = []
            in_distress = False
            start_date = None

            for date, is_distressed in distress_mask.items():
                if is_distressed and not in_distress:
                    in_distress = True
                    start_date = date
                elif not is_distressed and in_distress:
                    in_distress = False
                    # Find the trough in this period
                    period_prices = close[start_date:date]
                    if len(period_prices) > 0:
                        trough_date = period_prices.idxmin()
                        trough_price = period_prices.min()

                        # Check recovery: price rose 80%+ from trough within 2 years
                        future_prices = close[trough_date:trough_date + pd.Timedelta(days=730)]
                        if len(future_prices) > 60:
                            max_recovery = future_prices.max() / trough_price - 1
                            if max_recovery > 0.80:
                                # Find pre-distress peak
                                pre_peak = rolling_max.loc[start_date] if start_date in rolling_max.index else close[:start_date].max()
                                peak_to_trough = trough_price / pre_peak - 1 if pre_peak > 0 else 0

                                recovery_date = future_prices.idxmax()

                                distress_periods.append({
                                    'code': code,
                                    'name': stocks_dict.get(code, ''),
                                    'distress_start': start_date.strftime('%Y-%m-%d'),
                                    'trough_date': trough_date.strftime('%Y-%m-%d'),
                                    'recovery_date': recovery_date.strftime('%Y-%m-%d'),
                                    'peak_to_trough': round(peak_to_trough * 100, 1),
                                    'trough_to_recovery': round(max_recovery * 100, 1),
                                    'trough_price': round(trough_price, 2),
                                    'recovery_months': round((recovery_date - trough_date).days / 30, 1),
                                })

            if distress_periods:
                candidates.extend(distress_periods)

        except Exception as e:
            continue

    print(f"  Found {len(candidates)} distressed reversal episodes")
    return candidates


# ============================================================
# 3. ANALYZE FINANCIAL PATTERNS
# ============================================================

def analyze_reversal_financials(candidates, top_n=30):
    """
    For top reversal cases, analyze financial data around the trough.
    Look for common patterns before the reversal.
    """
    print(f"\n[3] Analyzing financials for top {top_n} reversal cases...")

    # Sort by recovery magnitude
    candidates_sorted = sorted(candidates, key=lambda x: x['trough_to_recovery'], reverse=True)[:top_n]

    detailed_cases = []

    for i, case in enumerate(candidates_sorted):
        code = case['code']
        trough_date = pd.Timestamp(case['trough_date'])

        print(f"  [{i+1}/{top_n}] {case['name']} ({code}): "
              f"跌{case['peak_to_trough']:.0f}% → 涨{case['trough_to_recovery']:.0f}%, "
              f"谷底{case['trough_date']}")

        try:
            # Get profit data
            profit_df = get_quarterly_financials(code)
            growth_df = get_growth_data(code)
            cashflow_df = get_cash_flow_data(code)

            analysis = {**case}

            # Analyze profit around trough (4 quarters before, 4 quarters after)
            if profit_df is not None and len(profit_df) > 4:
                # Find quarters around trough
                pre_trough = profit_df[profit_df['statDate'] <= trough_date].tail(6)
                post_trough = profit_df[profit_df['statDate'] > trough_date].head(6)

                if len(pre_trough) > 0:
                    analysis['pre_trough_netProfit'] = pre_trough['netProfit'].tolist()
                    analysis['pre_trough_roeAvg'] = pre_trough['roeAvg'].tolist()
                    analysis['pre_trough_gpMargin'] = pre_trough['gpMargin'].tolist()
                    analysis['pre_trough_npMargin'] = pre_trough['npMargin'].tolist()
                    analysis['pre_trough_dates'] = [str(d.date()) for d in pre_trough['statDate']]

                if len(post_trough) > 0:
                    analysis['post_trough_netProfit'] = post_trough['netProfit'].tolist()
                    analysis['post_trough_roeAvg'] = post_trough['roeAvg'].tolist()
                    analysis['post_trough_gpMargin'] = post_trough['gpMargin'].tolist()
                    analysis['post_trough_dates'] = [str(d.date()) for d in post_trough['statDate']]

            # Analyze growth data
            if growth_df is not None and len(growth_df) > 4:
                pre_g = growth_df[growth_df['statDate'] <= trough_date].tail(6)
                post_g = growth_df[growth_df['statDate'] > trough_date].head(6)

                if len(pre_g) > 0:
                    analysis['pre_trough_YOYNI'] = pre_g['YOYNI'].tolist()  # Net income YoY
                    analysis['pre_trough_YOYPNI'] = pre_g['YOYPNI'].tolist()  # Parent NI YoY

                if len(post_g) > 0:
                    analysis['post_trough_YOYNI'] = post_g['YOYNI'].tolist()

                # Key signal: YoY growth inflection
                all_g = growth_df.sort_values('statDate')
                trough_idx = all_g[all_g['statDate'] <= trough_date].index
                if len(trough_idx) > 0:
                    last_pre = trough_idx[-1]
                    pos = all_g.index.get_loc(last_pre)

                    # Check if growth was declining then started improving
                    if pos >= 2:
                        recent_growth = all_g['YOYNI'].iloc[max(0,pos-3):pos+1].tolist()
                        analysis['growth_trend_before_trough'] = recent_growth

                        # Growth inflection: negative but improving
                        if len(recent_growth) >= 2:
                            analysis['growth_inflection'] = (
                                recent_growth[-1] is not None and
                                recent_growth[-2] is not None and
                                not np.isnan(recent_growth[-1]) and
                                not np.isnan(recent_growth[-2]) and
                                recent_growth[-1] > recent_growth[-2]
                            )

            # Analyze cash flow
            if cashflow_df is not None and len(cashflow_df) > 2:
                pre_cf = cashflow_df[cashflow_df['statDate'] <= trough_date].tail(4)
                if len(pre_cf) > 0:
                    analysis['pre_trough_CFOToNP'] = pre_cf['CFOToNP'].tolist()
                    analysis['pre_trough_CFOToOR'] = pre_cf['CFOToOR'].tolist()

            detailed_cases.append(analysis)

        except Exception as e:
            print(f"    Error: {e}")
            detailed_cases.append({**case, 'error': str(e)})

    return detailed_cases


def extract_common_patterns(detailed_cases):
    """Extract common financial patterns from successful reversal cases"""
    print(f"\n[4] Extracting common patterns from {len(detailed_cases)} cases...")

    patterns = {
        'total_cases': len(detailed_cases),
        'avg_peak_to_trough': 0,
        'avg_trough_to_recovery': 0,
        'avg_recovery_months': 0,
        'profit_declining_before_trough': 0,
        'profit_improving_at_trough': 0,
        'growth_inflection_before_trough': 0,
        'positive_cashflow_during_distress': 0,
        'margin_bottoming_before_trough': 0,
        'cases_with_data': 0,
    }

    valid_cases = [c for c in detailed_cases if 'error' not in c]
    patterns['cases_with_data'] = len(valid_cases)

    if not valid_cases:
        return patterns

    patterns['avg_peak_to_trough'] = round(np.mean([c['peak_to_trough'] for c in valid_cases]), 1)
    patterns['avg_trough_to_recovery'] = round(np.mean([c['trough_to_recovery'] for c in valid_cases]), 1)
    patterns['avg_recovery_months'] = round(np.mean([c['recovery_months'] for c in valid_cases]), 1)

    for case in valid_cases:
        # Check if net profit was declining before trough
        if 'pre_trough_netProfit' in case:
            profits = [p for p in case['pre_trough_netProfit'] if p is not None and not np.isnan(p)]
            if len(profits) >= 2 and profits[-1] < profits[0]:
                patterns['profit_declining_before_trough'] += 1

            # Check if profit was already improving at trough (less bad)
            if len(profits) >= 3:
                if profits[-1] > profits[-2]:  # Last quarter before trough improved
                    patterns['profit_improving_at_trough'] += 1

        # Growth inflection
        if case.get('growth_inflection'):
            patterns['growth_inflection_before_trough'] += 1

        # Positive cashflow during distress
        if 'pre_trough_CFOToNP' in case:
            cfo = [c for c in case['pre_trough_CFOToNP'] if c is not None and not np.isnan(c)]
            if any(c > 0 for c in cfo):
                patterns['positive_cashflow_during_distress'] += 1

        # Gross margin bottoming
        if 'pre_trough_gpMargin' in case:
            margins = [m for m in case['pre_trough_gpMargin'] if m is not None and not np.isnan(m)]
            if len(margins) >= 3 and margins[-1] > margins[-2]:
                patterns['margin_bottoming_before_trough'] += 1

    # Convert to percentages
    n = patterns['cases_with_data']
    if n > 0:
        patterns['pct_profit_declining'] = round(patterns['profit_declining_before_trough'] / n * 100, 1)
        patterns['pct_profit_improving'] = round(patterns['profit_improving_at_trough'] / n * 100, 1)
        patterns['pct_growth_inflection'] = round(patterns['growth_inflection_before_trough'] / n * 100, 1)
        patterns['pct_positive_cashflow'] = round(patterns['positive_cashflow_during_distress'] / n * 100, 1)
        patterns['pct_margin_bottoming'] = round(patterns['margin_bottoming_before_trough'] / n * 100, 1)

    return patterns


def generate_report(candidates, detailed_cases, patterns):
    """Generate comprehensive report"""
    print("\n[5] Generating report...")

    report = []
    report.append("=" * 70)
    report.append("A股大盘股困境反转研究报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 70)

    report.append(f"\n总计发现 {len(candidates)} 个困境反转案例")
    report.append(f"详细分析 {len(detailed_cases)} 个案例")

    # Top cases by recovery
    report.append("\n" + "=" * 60)
    report.append("TOP 30 困境反转案例 (按反弹幅度排序)")
    report.append("=" * 60)

    sorted_cases = sorted(detailed_cases, key=lambda x: x['trough_to_recovery'], reverse=True)

    report.append(f"{'名称':<12} {'代码':<12} {'跌幅':>6} {'涨幅':>6} {'谷底日期':<12} {'恢复月数':>8}")
    report.append("-" * 60)

    for case in sorted_cases[:30]:
        name = case.get('name', '')[:10]
        report.append(
            f"{name:<12} {case['code']:<12} {case['peak_to_trough']:>5.0f}% "
            f"{case['trough_to_recovery']:>5.0f}% {case['trough_date']:<12} "
            f"{case['recovery_months']:>7.1f}"
        )

    # Pattern analysis
    report.append("\n" + "=" * 60)
    report.append("困境反转共性分析")
    report.append("=" * 60)

    n = patterns.get('cases_with_data', 0)
    report.append(f"分析样本: {n} 个案例")
    report.append(f"平均最大跌幅: {patterns['avg_peak_to_trough']:.1f}%")
    report.append(f"平均反弹幅度: {patterns['avg_trough_to_recovery']:.1f}%")
    report.append(f"平均恢复时间: {patterns['avg_recovery_months']:.1f} 个月")

    report.append(f"\n财务信号特征 (反转前):")
    report.append(f"  利润在下降中: {patterns.get('pct_profit_declining', 0):.1f}%")
    report.append(f"  利润已开始改善(减亏): {patterns.get('pct_profit_improving', 0):.1f}%")
    report.append(f"  增速拐点出现: {patterns.get('pct_growth_inflection', 0):.1f}%")
    report.append(f"  经营现金流仍为正: {patterns.get('pct_positive_cashflow', 0):.1f}%")
    report.append(f"  毛利率见底回升: {patterns.get('pct_margin_bottoming', 0):.1f}%")

    # Distilled signals
    report.append("\n" + "=" * 60)
    report.append("可量化的困境反转信号")
    report.append("=" * 60)
    report.append("""
信号1: 价格信号 - 必要条件
  - 股价从1年高点下跌50%+
  - 市值仍在200亿以上(确保是大盘股)
  - PB < 历史25分位

信号2: 利润拐点 - 核心信号
  - 连续2季度净利润同比下降50%+（确认困境）
  - 最新一季净利润同比降幅收窄（拐点信号）
  - 或：净利润环比改善（Q-on-Q好转）

信号3: 毛利率触底 - 领先信号
  - 毛利率连续下降后出现回升
  - 毛利率回升通常领先净利润拐点1-2个季度

信号4: 经营现金流 - 质量信号
  - 净利润下降但经营现金流仍为正
  - 说明主业仍有造血能力，困境是暂时的
  - 经营现金流/营收比率稳定

信号5: 费用率下降 - 管理层行动
  - 管理费用率/销售费用率开始下降
  - 说明管理层在积极应对困境

综合评分: 同时满足3个以上信号的股票值得关注
""")

    report_text = "\n".join(report)

    # Save report
    report_path = os.path.join(DATA_DIR, 'distressed_reversal_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Save detailed data
    json_path = os.path.join(DATA_DIR, 'distressed_reversal_results.json')
    output = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'total_candidates': len(candidates),
        'analyzed_cases': len(detailed_cases),
        'patterns': patterns,
        'all_candidates': candidates,
        'detailed_cases': detailed_cases,
    }

    # Clean NaN for JSON
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(clean_for_json(output), f, ensure_ascii=False, indent=2)

    print(f"\n  Report saved to {report_path}")
    print(f"  Data saved to {json_path}")
    print("\n" + report_text)

    return output


# ============================================================
# 4. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("A股大盘股困境反转研究")
    print("=" * 70)

    bs.login()

    try:
        # Step 1: Get constituents
        stocks = get_csi300_500_constituents()

        # Step 2: Screen for distressed reversals
        candidates = identify_distressed_stocks(stocks, max_stocks=300)

        if not candidates:
            print("No distressed reversal candidates found!")
            return

        # Sort by recovery
        candidates = sorted(candidates, key=lambda x: x['trough_to_recovery'], reverse=True)

        print(f"\nTop 10 reversals:")
        for c in candidates[:10]:
            print(f"  {c['name']} ({c['code']}): 跌{c['peak_to_trough']:.0f}% → 涨{c['trough_to_recovery']:.0f}%, "
                  f"谷底{c['trough_date']}, 恢复{c['recovery_months']:.0f}月")

        # Step 3: Detailed financial analysis on top cases
        detailed = analyze_reversal_financials(candidates, top_n=30)

        # Step 4: Extract patterns
        patterns = extract_common_patterns(detailed)

        print(f"\n共性分析:")
        print(f"  利润已下降: {patterns.get('pct_profit_declining', 0):.1f}%")
        print(f"  利润开始改善: {patterns.get('pct_profit_improving', 0):.1f}%")
        print(f"  增速拐点: {patterns.get('pct_growth_inflection', 0):.1f}%")
        print(f"  现金流正: {patterns.get('pct_positive_cashflow', 0):.1f}%")
        print(f"  毛利率回升: {patterns.get('pct_margin_bottoming', 0):.1f}%")

        # Step 5: Generate report
        output = generate_report(candidates, detailed, patterns)

    finally:
        bs.logout()

    return output


if __name__ == '__main__':
    main()
