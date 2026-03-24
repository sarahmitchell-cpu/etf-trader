#!/usr/bin/env python3
"""
A-Share 10x Potential Stock Screening
Filter: Market cap 100亿-1000亿
Criteria: Growth, Valuation, Profitability, Industry prospects
"""

import baostock as bs
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from collections import defaultdict

# ============================================================
# STEP 1: Get all A-share stocks and filter by market cap
# ============================================================

def get_all_stocks():
    """Get all A-share stocks"""
    lg = bs.login()

    # Get stock list
    rs = bs.query_stock_basic(code_name="", code="")
    stocks = []
    while rs.next():
        row = rs.get_row_data()
        # row: code, code_name, ipoDate, outDate, type, status
        if row[4] == '1' and row[5] == '1':  # type=stock, status=listed
            stocks.append({
                'code': row[0],
                'name': row[1],
                'ipo_date': row[2],
            })

    bs.logout()
    print(f"Total listed stocks: {len(stocks)}")
    return stocks


def get_market_cap_and_filter(stocks):
    """Get latest market cap and filter 100亿-1000亿"""
    lg = bs.login()

    filtered = []
    total = len(stocks)

    for i, s in enumerate(stocks):
        if i % 200 == 0:
            print(f"  Fetching market cap: {i}/{total}...")

        code = s['code']
        # Get latest daily data with market cap proxy
        # Use close * total shares to estimate market cap
        rs = bs.query_history_k_data_plus(
            code, 'date,close,volume,amount,turn,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST',
            start_date='2026-03-17', end_date='2026-03-28',
            frequency='d', adjustflag='3'  # unadjusted for market cap calc
        )

        latest = None
        while rs.next():
            latest = rs.get_row_data()

        if latest is None:
            continue

        try:
            close = float(latest[1])
            pe = float(latest[5]) if latest[5] else 0
            pb = float(latest[6]) if latest[6] else 0
            ps = float(latest[7]) if latest[7] else 0
            is_st = latest[9]

            if is_st == '1':
                continue

        except (ValueError, IndexError):
            continue

        # Get total shares from profit data
        rs2 = bs.query_profit_data(code=code, year=2024, quarter=4)
        total_share = 0
        net_profit = 0
        revenue = 0
        roe = 0
        eps_ttm = 0
        while rs2.next():
            row2 = rs2.get_row_data()
            try:
                total_share = float(row2[9]) if row2[9] else 0
                net_profit = float(row2[6]) if row2[6] else 0
                revenue = float(row2[8]) if row2[8] else 0
                roe = float(row2[3]) if row2[3] else 0
                eps_ttm = float(row2[7]) if row2[7] else 0
            except Exception:
                pass

        if total_share <= 0:
            continue

        market_cap = close * total_share / 1e8  # in 亿元

        if 100 <= market_cap <= 1000:
            s['close'] = close
            s['market_cap'] = round(market_cap, 1)
            s['pe'] = round(pe, 1) if pe > 0 else None
            s['pb'] = round(pb, 2) if pb > 0 else None
            s['ps'] = round(ps, 2) if ps > 0 else None
            s['net_profit'] = net_profit
            s['revenue'] = revenue
            s['roe'] = round(roe * 100, 2) if roe else 0  # percentage
            s['eps_ttm'] = eps_ttm
            s['total_share'] = total_share
            filtered.append(s)

        if i % 100 == 99:
            time.sleep(0.5)

    bs.logout()
    print(f"Filtered {len(filtered)} stocks with market cap 100亿-1000亿")
    return filtered


def get_growth_data(stocks):
    """Fetch growth data for filtered stocks"""
    lg = bs.login()

    total = len(stocks)
    for i, s in enumerate(stocks):
        if i % 50 == 0:
            print(f"  Fetching growth data: {i}/{total}...")

        code = s['code']

        # Get growth for recent quarters
        growth_ni_list = []
        growth_rev_list = []
        roe_list = []

        for year in [2022, 2023, 2024]:
            for quarter in [4]:  # Annual data
                rs = bs.query_growth_data(code=code, year=year, quarter=quarter)
                while rs.next():
                    row = rs.get_row_data()
                    try:
                        yoy_ni = float(row[5]) if row[5] else None
                        growth_ni_list.append((year, yoy_ni))
                    except Exception:
                        pass

                rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
                while rs.next():
                    row = rs.get_row_data()
                    try:
                        roe_val = float(row[3]) if row[3] else None
                        rev = float(row[8]) if row[8] else None
                        roe_list.append((year, roe_val))
                        if rev:
                            growth_rev_list.append((year, rev))
                    except Exception:
                        pass

        # Also get latest quarters (2024Q1-Q4, 2025Q1-Q3 if available)
        for year in [2024, 2025]:
            for quarter in [1, 2, 3, 4]:
                rs = bs.query_growth_data(code=code, year=year, quarter=quarter)
                while rs.next():
                    row = rs.get_row_data()
                    try:
                        yoy_ni = float(row[5]) if row[5] else None
                        if yoy_ni is not None:
                            growth_ni_list.append((f"{year}Q{quarter}", yoy_ni))
                    except Exception:
                        pass

        # Compute growth metrics
        s['growth_ni_history'] = growth_ni_list

        # Revenue CAGR (3-year if available)
        if len(growth_rev_list) >= 2:
            rev_sorted = sorted(growth_rev_list, key=lambda x: x[0])
            first_rev = rev_sorted[0][1]
            last_rev = rev_sorted[-1][1]
            years_span = rev_sorted[-1][0] - rev_sorted[0][0]
            if first_rev > 0 and last_rev > 0 and years_span > 0:
                s['rev_cagr'] = round(((last_rev / first_rev) ** (1 / years_span) - 1) * 100, 1)
            else:
                s['rev_cagr'] = None
        else:
            s['rev_cagr'] = None

        # Average ROE
        roe_vals = [r[1] for r in roe_list if r[1] is not None and r[1] > 0]
        s['avg_roe'] = round(np.mean(roe_vals) * 100, 2) if roe_vals else 0

        # Latest NI growth
        ni_vals = [(k, v) for k, v in growth_ni_list if v is not None]
        if ni_vals:
            s['latest_ni_growth'] = round(ni_vals[-1][1] * 100, 1)
        else:
            s['latest_ni_growth'] = None

        # NI growth trend (accelerating?)
        annual_ni = [(k, v) for k, v in growth_ni_list if isinstance(k, int) and v is not None]
        if len(annual_ni) >= 2:
            annual_ni.sort(key=lambda x: x[0])
            s['ni_growth_trend'] = 'accelerating' if annual_ni[-1][1] > annual_ni[-2][1] else 'decelerating'
        else:
            s['ni_growth_trend'] = 'unknown'

        if i % 50 == 49:
            time.sleep(0.5)

    bs.logout()
    return stocks


def get_industry_classification(stocks):
    """Get industry for all stocks"""
    lg = bs.login()

    for i, s in enumerate(stocks):
        rs = bs.query_stock_industry(code=s['code'])
        while rs.next():
            row = rs.get_row_data()
            if len(row) >= 4:
                s['industry'] = row[3]
                s['classification'] = row[4] if len(row) > 4 else ''
                break
        if 'industry' not in s:
            s['industry'] = '其他'

    bs.logout()
    return stocks


def get_price_performance(stocks):
    """Get price performance for momentum signals"""
    lg = bs.login()

    total = len(stocks)
    for i, s in enumerate(stocks):
        if i % 100 == 0:
            print(f"  Fetching price performance: {i}/{total}...")

        code = s['code']
        rs = bs.query_history_k_data_plus(
            code, 'date,close',
            start_date='2024-03-01', end_date='2026-03-28',
            frequency='w', adjustflag='1'  # adjusted
        )

        prices = []
        while rs.next():
            row = rs.get_row_data()
            try:
                prices.append(float(row[1]))
            except Exception:
                continue

        if len(prices) >= 52:
            s['ret_1y'] = round((prices[-1] / prices[-52] - 1) * 100, 1)
            s['ret_2y'] = round((prices[-1] / prices[0] - 1) * 100, 1) if len(prices) >= 100 else None

            # Volatility
            rets = np.diff(prices) / prices[:-1]
            s['volatility'] = round(np.std(rets[-52:]) * np.sqrt(52) * 100, 1)

            # Distance from 52-week high
            high_52w = max(prices[-52:])
            s['from_52w_high'] = round((prices[-1] / high_52w - 1) * 100, 1)
        else:
            s['ret_1y'] = None
            s['ret_2y'] = None
            s['volatility'] = None
            s['from_52w_high'] = None

        if i % 100 == 99:
            time.sleep(0.5)

    bs.logout()
    return stocks


def score_and_rank(stocks):
    """Comprehensive scoring for 10x potential"""

    scored = []
    for s in stocks:
        score = 0
        reasons = []
        penalties = []

        mc = s.get('market_cap', 0)
        pe = s.get('pe')
        pb = s.get('pb')
        roe = s.get('avg_roe', 0)
        rev_cagr = s.get('rev_cagr')
        ni_growth = s.get('latest_ni_growth')
        ret_1y = s.get('ret_1y')
        vol = s.get('volatility')
        industry = s.get('industry', '')
        trend = s.get('ni_growth_trend', 'unknown')
        from_high = s.get('from_52w_high')

        # === GROWTH SCORE (max 40) ===
        # Net income growth
        if ni_growth is not None:
            if ni_growth > 100:
                score += 15
                reasons.append(f"净利润爆发增长{ni_growth}%")
            elif ni_growth > 50:
                score += 12
                reasons.append(f"净利润高增长{ni_growth}%")
            elif ni_growth > 30:
                score += 8
                reasons.append(f"净利润增长{ni_growth}%")
            elif ni_growth > 15:
                score += 5
                reasons.append(f"净利润稳定增长{ni_growth}%")
            elif ni_growth < -20:
                score -= 5
                penalties.append(f"净利润下滑{ni_growth}%")

        # Revenue CAGR
        if rev_cagr is not None:
            if rev_cagr > 30:
                score += 10
                reasons.append(f"营收CAGR={rev_cagr}%")
            elif rev_cagr > 15:
                score += 6
                reasons.append(f"营收CAGR={rev_cagr}%")
            elif rev_cagr > 5:
                score += 3
            elif rev_cagr < -10:
                score -= 3
                penalties.append(f"营收萎缩{rev_cagr}%")

        # Growth acceleration
        if trend == 'accelerating':
            score += 8
            reasons.append("增长加速")
        elif trend == 'decelerating':
            score -= 2

        # ROE (max 5)
        if roe > 20:
            score += 5
            reasons.append(f"高ROE={roe}%")
        elif roe > 15:
            score += 3
        elif roe > 10:
            score += 1
        elif roe < 3:
            score -= 3
            penalties.append(f"低ROE={roe}%")

        # === VALUATION SCORE (max 20) ===
        # For 10x potential, we want growth at reasonable price (GARP), not necessarily cheap
        if pe is not None and pe > 0:
            if ni_growth and ni_growth > 0:
                peg = pe / ni_growth
                if peg < 0.5:
                    score += 10
                    reasons.append(f"PEG极低={peg:.1f}")
                elif peg < 1.0:
                    score += 7
                    reasons.append(f"PEG合理={peg:.1f}")
                elif peg < 1.5:
                    score += 4
                elif peg > 3:
                    score -= 3
                    penalties.append(f"PEG过高={peg:.1f}")
            else:
                if pe < 15:
                    score += 5
                    reasons.append(f"低PE={pe}")
                elif pe < 30:
                    score += 2
                elif pe > 100:
                    score -= 5
                    penalties.append(f"高PE={pe}")

        if pb is not None and pb > 0:
            if pb < 2:
                score += 3
            elif pb < 4:
                score += 1
            elif pb > 10:
                score -= 2

        # === MARKET CAP SCORE (max 10) ===
        # Smaller cap within range has more room to grow 10x
        if mc < 200:
            score += 10
            reasons.append(f"市值{mc}亿，空间大")
        elif mc < 400:
            score += 7
            reasons.append(f"市值{mc}亿")
        elif mc < 600:
            score += 4
        else:
            score += 1

        # === INDUSTRY SCORE (max 15) ===
        high_growth_industries = [
            '半导体', '新能源', '人工智能', 'AI', '芯片', '光伏', '锂电',
            '军工', '航天', '生物医药', '创新药', '医疗器械',
            '机器人', '自动驾驶', '云计算', '数据', '算力',
            '新材料', '碳纤维',
        ]
        moderate_growth = [
            '电子', '计算机', '通信', '传媒', '软件',
            '汽车', '新能源汽车', '消费电子',
            '医药', '化工',
        ]
        low_growth = [
            '银行', '保险', '房地产', '煤炭', '钢铁',
            '建筑', '公用事业', '交通运输',
        ]

        industry_boost = 0
        for keyword in high_growth_industries:
            if keyword in industry:
                industry_boost = 15
                reasons.append(f"高景气行业:{industry}")
                break
        if industry_boost == 0:
            for keyword in moderate_growth:
                if keyword in industry:
                    industry_boost = 8
                    reasons.append(f"成长行业:{industry}")
                    break
        if industry_boost == 0:
            for keyword in low_growth:
                if keyword in industry:
                    industry_boost = -5
                    penalties.append(f"低增长行业:{industry}")
                    break
            if industry_boost == 0:
                industry_boost = 3  # neutral
        score += industry_boost

        # === MOMENTUM SCORE (max 10) ===
        if ret_1y is not None:
            if ret_1y > 50:
                score += 5  # good momentum but might be overextended
            elif ret_1y > 20:
                score += 8
                reasons.append(f"强势上涨{ret_1y}%")
            elif ret_1y > 0:
                score += 4
            elif ret_1y < -30:
                score += 2  # deep value potential
                reasons.append(f"深度回调{ret_1y}%")

        # Distance from 52-week high
        if from_high is not None:
            if -15 < from_high < 0:
                score += 3  # near highs, strong
            elif from_high < -40:
                score += 1  # too beaten down

        # === PENALTIES ===
        # IPO too recent (less than 3 years)
        ipo = s.get('ipo_date', '')
        if ipo > '2023-01-01':
            score -= 5
            penalties.append("上市不足3年")

        s['score'] = score
        s['reasons'] = reasons
        s['penalties'] = penalties
        scored.append(s)

    scored.sort(key=lambda x: -x['score'])
    return scored


def main():
    t0 = time.time()
    print("=" * 60)
    print("A股10倍潜力股筛选（市值100亿-1000亿）")
    print("=" * 60)

    # 1. Get all stocks
    print("\n[1/5] 获取全部A股...")
    all_stocks = get_all_stocks()

    # 2. Filter by market cap
    print("\n[2/5] 筛选市值100亿-1000亿...")
    filtered = get_market_cap_and_filter(all_stocks)

    # 3. Get industry
    print("\n[3/5] 获取行业分类...")
    filtered = get_industry_classification(filtered)

    # 4. Get growth data
    print("\n[4/5] 获取成长数据...")
    filtered = get_growth_data(filtered)

    # 5. Get price performance
    print("\n[5/5] 获取价格表现...")
    filtered = get_price_performance(filtered)

    print(f"\n数据获取完成: {time.time()-t0:.0f}s")

    # Score and rank
    print("\n综合评分...")
    ranked = score_and_rank(filtered)

    # Industry distribution
    print(f"\n{'='*60}")
    print(f"行业分布 ({len(ranked)} stocks):")
    ind_count = defaultdict(int)
    for s in ranked:
        ind_count[s.get('industry', '其他')] += 1
    for ind, cnt in sorted(ind_count.items(), key=lambda x: -x[1])[:20]:
        print(f"  {ind}: {cnt}")

    # Top 30
    print(f"\n{'='*60}")
    print("TOP 30 - 未来三年10倍潜力股:")
    print(f"{'='*60}")

    for i, s in enumerate(ranked[:30]):
        print(f"\n{i+1}. {s['name']} ({s['code']}) - 综合得分: {s['score']}")
        print(f"   市值: {s['market_cap']}亿 | PE: {s.get('pe','N/A')} | PB: {s.get('pb','N/A')}")
        print(f"   ROE: {s.get('avg_roe',0)}% | 营收CAGR: {s.get('rev_cagr','N/A')}%")
        print(f"   最新净利润增长: {s.get('latest_ni_growth','N/A')}% | 增长趋势: {s.get('ni_growth_trend','N/A')}")
        print(f"   1年涨幅: {s.get('ret_1y','N/A')}% | 波动率: {s.get('volatility','N/A')}%")
        print(f"   行业: {s.get('industry','')}")
        if s['reasons']:
            print(f"   优势: {'; '.join(s['reasons'])}")
        if s['penalties']:
            print(f"   风险: {'; '.join(s['penalties'])}")

    # Save results
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_stocks_screened': len(all_stocks),
        'filtered_by_market_cap': len(filtered),
        'top_30': [{
            'rank': i + 1,
            'code': s['code'],
            'name': s['name'],
            'score': s['score'],
            'market_cap': s['market_cap'],
            'pe': s.get('pe'),
            'pb': s.get('pb'),
            'roe': s.get('avg_roe', 0),
            'rev_cagr': s.get('rev_cagr'),
            'latest_ni_growth': s.get('latest_ni_growth'),
            'ni_growth_trend': s.get('ni_growth_trend'),
            'ret_1y': s.get('ret_1y'),
            'volatility': s.get('volatility'),
            'industry': s.get('industry', ''),
            'reasons': s['reasons'],
            'penalties': s['penalties'],
        } for i, s in enumerate(ranked[:30])]
    }

    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(out_dir, 'data', 'astock_10x_screening.json')
    import os
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n保存至: {out_path}")
    print(f"总耗时: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
