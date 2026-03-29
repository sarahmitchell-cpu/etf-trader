#!/usr/bin/env python3
"""
A股三年五倍潜力股筛选器
- 全A股范围（主板+创业板+科创板+北交所）
- 市值筛选：50-300亿
- 目标：找出未来3年有5倍潜力的标的
- 数据源：East Money API
"""

import json
import requests
import time
import sys

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Referer': 'https://quote.eastmoney.com/',
}

def fetch_all_stocks():
    """Fetch all A-share stocks with key metrics from East Money API"""
    all_stocks = []
    # fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81 covers all boards
    # m:0+t:6 = 沪A, m:0+t:80 = 创业板, m:1+t:2 = 深A, m:1+t:23 = 科创板, m:0+t:81 = 北交所
    page = 1
    page_size = 200

    while True:
        url = (
            f"https://push2.eastmoney.com/api/qt/clist/get?"
            f"pn={page}&pz={page_size}&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281"
            f"&fltt=2&invt=2&dect=1&wbp2u=|0|0|0|web"
            f"&fid=f20&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81"
            f"&fields=f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f37,f38,f39,f40,f41,f45,f46,f48,f49,f50,f51,f52,f57,f61,f62,f100,f102,f103,f104,f105,f112,f113,f114,f115,f128,f136,f152,f162,f167,f168,f170,f171"
        )

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            data = resp.json()
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break

        if not data or not data.get('data') or not data['data'].get('diff'):
            break

        stocks = data['data']['diff']
        if not stocks:
            break

        all_stocks.extend(stocks)
        total = data['data'].get('total', 0)
        print(f"  Fetched page {page}, got {len(stocks)} stocks (total so far: {len(all_stocks)}/{total})")

        if len(all_stocks) >= total:
            break
        page += 1
        time.sleep(0.3)

    return all_stocks

def fetch_stock_detail(code, market):
    """Fetch detailed financials for a single stock"""
    secid = f"{market}.{code}"
    url = (
        f"https://push2.eastmoney.com/api/qt/stock/get?"
        f"secid={secid}&ut=bd1d9ddb04089700cf9c27f6f7426281"
        f"&fields=f57,f58,f84,f85,f116,f117,f162,f163,f167,f168,f169,f170,f171,f173,f177,f187,f188,f189,f190,f191,f192,f193,f194,f195"
        f"&invt=2"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        return resp.json().get('data', {})
    except:
        return {}

def fetch_revenue_growth(code, market):
    """Fetch quarterly revenue/profit growth from East Money"""
    secid = f"{market}.{code}"
    url = (
        f"https://datacenter-web.eastmoney.com/api/data/v1/get?"
        f"reportName=RPT_LICO_FN_CPD&columns=SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,"
        f"REPORT_DATE,BASIC_EPS,WEIGHTAVG_ROE,YSTZ,SJLTZ,BPS,MGJYXJJE,XSMLL,YYZSR,PARENT_NETPROFIT,DJDYYZSR,DJDKCFJLR"
        f"&quoteColumns=&filter=(SECURITY_CODE%3D%22{code}%22)"
        f"&pageNumber=1&pageSize=8&sortTypes=-1&sortColumns=REPORT_DATE"
        f"&token=894050c76af8597a853f5b408b759f5d"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        data = resp.json()
        if data.get('result') and data['result'].get('data'):
            return data['result']['data']
    except:
        pass
    return []

def screen_stocks():
    """Main screening logic"""
    print("=" * 80)
    print("A股三年五倍潜力股筛选")
    print("条件：市值50-300亿，高成长，行业前景好")
    print("=" * 80)

    # Step 1: Fetch all stocks
    print("\n[1/4] 获取全A股数据...")
    all_stocks = fetch_all_stocks()
    print(f"  共获取 {len(all_stocks)} 只股票")

    # Step 2: Filter by market cap (50-300亿)
    print("\n[2/4] 按市值筛选 (50-300亿)...")
    filtered = []
    for s in all_stocks:
        mcap = s.get('f20')  # 总市值
        if mcap and isinstance(mcap, (int, float)) and mcap != '-':
            mcap_yi = mcap / 1e8  # 转为亿
            if 50 <= mcap_yi <= 300:
                filtered.append(s)

    print(f"  市值50-300亿共 {len(filtered)} 只")

    # Step 3: Apply growth filters
    print("\n[3/4] 应用成长性筛选...")
    candidates = []
    for s in filtered:
        code = s.get('f12', '')
        name = s.get('f14', '')
        mcap_yi = s.get('f20', 0) / 1e8
        pe = s.get('f9', '-')  # PE(动)
        pb = s.get('f23', '-')  # PB
        roe = s.get('f37', '-')  # ROE
        rev_yoy = s.get('f40', '-')  # 营收同比增长
        profit_yoy = s.get('f41', '-')  # 净利润同比增长
        gross_margin = s.get('f49', '-')  # 毛利率
        turnover = s.get('f8', '-')  # 换手率
        price = s.get('f2', '-')  # 最新价
        industry = s.get('f100', '-')  # 行业

        # Skip ST stocks
        if 'ST' in str(name) or '*ST' in str(name):
            continue

        # Skip if price is suspended
        if price == '-' or not price:
            continue

        # Basic quality filters:
        # 1. PE > 0 (profitable) and PE < 200 (not crazy expensive)
        if not isinstance(pe, (int, float)) or pe <= 0 or pe > 200:
            continue

        # 2. Revenue growth > 15% or Profit growth > 20%
        rev_ok = isinstance(rev_yoy, (int, float)) and rev_yoy > 15
        profit_ok = isinstance(profit_yoy, (int, float)) and profit_yoy > 20
        if not (rev_ok or profit_ok):
            continue

        # 3. ROE > 5% (minimum profitability)
        if isinstance(roe, (int, float)) and roe < 5:
            continue

        # Score calculation for ranking
        score = 0

        # Revenue growth score (max 30)
        if isinstance(rev_yoy, (int, float)):
            if rev_yoy > 100: score += 30
            elif rev_yoy > 60: score += 25
            elif rev_yoy > 40: score += 20
            elif rev_yoy > 25: score += 15
            elif rev_yoy > 15: score += 10

        # Profit growth score (max 30)
        if isinstance(profit_yoy, (int, float)):
            if profit_yoy > 200: score += 30
            elif profit_yoy > 100: score += 25
            elif profit_yoy > 60: score += 20
            elif profit_yoy > 35: score += 15
            elif profit_yoy > 20: score += 10

        # ROE score (max 15)
        if isinstance(roe, (int, float)):
            if roe > 25: score += 15
            elif roe > 18: score += 12
            elif roe > 12: score += 9
            elif roe > 8: score += 6
            elif roe > 5: score += 3

        # Gross margin score (max 10)
        if isinstance(gross_margin, (int, float)):
            if gross_margin > 60: score += 10
            elif gross_margin > 45: score += 8
            elif gross_margin > 30: score += 6
            elif gross_margin > 20: score += 4

        # Market cap sweet spot (smaller = more room to grow, max 15)
        if mcap_yi < 80: score += 15
        elif mcap_yi < 120: score += 12
        elif mcap_yi < 180: score += 9
        elif mcap_yi < 250: score += 6
        else: score += 3

        # PE reasonableness (max 10 - lower PE with high growth = better)
        if isinstance(pe, (int, float)) and isinstance(profit_yoy, (int, float)) and profit_yoy > 0:
            peg = pe / profit_yoy
            if peg < 0.5: score += 10
            elif peg < 0.8: score += 8
            elif peg < 1.0: score += 6
            elif peg < 1.5: score += 4
            elif peg < 2.0: score += 2

        candidates.append({
            'code': code,
            'name': name,
            'market': s.get('f13', 0),
            'mcap_yi': round(mcap_yi, 1),
            'price': price,
            'pe': round(pe, 1) if isinstance(pe, (int, float)) else pe,
            'pb': round(pb, 2) if isinstance(pb, (int, float)) else pb,
            'roe': round(roe, 1) if isinstance(roe, (int, float)) else roe,
            'rev_yoy': round(rev_yoy, 1) if isinstance(rev_yoy, (int, float)) else rev_yoy,
            'profit_yoy': round(profit_yoy, 1) if isinstance(profit_yoy, (int, float)) else profit_yoy,
            'gross_margin': round(gross_margin, 1) if isinstance(gross_margin, (int, float)) else gross_margin,
            'industry': industry,
            'score': score,
        })

    # Sort by score descending
    candidates.sort(key=lambda x: x['score'], reverse=True)
    print(f"  通过成长性筛选: {len(candidates)} 只")

    # Step 4: Output top candidates
    print(f"\n[4/4] Top 50 候选股票 (按综合得分排序)")
    print("-" * 120)
    print(f"{'排名':>4} {'代码':<8} {'名称':<10} {'行业':<12} {'市值(亿)':>8} {'PE':>7} {'ROE%':>7} {'营收增%':>8} {'利润增%':>8} {'毛利率%':>8} {'得分':>5}")
    print("-" * 120)

    top50 = candidates[:50]
    for i, c in enumerate(top50, 1):
        print(f"{i:>4} {c['code']:<8} {c['name']:<10} {str(c['industry']):<12} {c['mcap_yi']:>8} {c['pe']:>7} {str(c['roe']):>7} {str(c['rev_yoy']):>8} {str(c['profit_yoy']):>8} {str(c['gross_margin']):>8} {c['score']:>5}")

    # Save full results
    output = {
        'screening_date': '2026-03-25',
        'criteria': {
            'market_cap': '50-300亿',
            'target': '3年5倍',
            'pe_range': '0-200',
            'min_rev_growth': '15% or min_profit_growth 20%',
            'min_roe': '5%',
        },
        'total_stocks': len(all_stocks),
        'after_mcap_filter': len(filtered),
        'after_growth_filter': len(candidates),
        'top50': top50,
        'all_candidates': candidates,
    }

    output_path = '/Users/claw/etf-trader/data/astock_3y5x_screening.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")

    return top50, candidates

if __name__ == '__main__':
    top50, all_candidates = screen_stocks()

    # Print summary by industry
    print("\n\n行业分布 (Top50):")
    print("-" * 40)
    industry_count = {}
    for c in top50:
        ind = c.get('industry', '未知')
        industry_count[ind] = industry_count.get(ind, 0) + 1
    for ind, cnt in sorted(industry_count.items(), key=lambda x: -x[1]):
        print(f"  {ind}: {cnt}只")
