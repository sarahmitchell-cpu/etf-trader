#!/usr/bin/env python3
"""
A-Share 10x Potential Stock Screening (Fast Version)
Uses akshare for bulk data retrieval
Market cap 100亿-1000亿
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import akshare as ak
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from collections import defaultdict

DATA_DIR = '/Users/claw/etf-trader/data'


def main():
    t0 = time.time()
    print("=" * 60)
    print("A股10倍潜力股筛选（市值100亿-1000亿）")
    print("=" * 60)

    # 1. Get all A-share stocks with market data
    print("\n[1/4] 获取全A股实时行情...")
    try:
        # Get all A-share real-time quotes (includes market cap)
        df_sh = ak.stock_sh_a_spot_em()
        df_sz = ak.stock_sz_a_spot_em()
        df_all = pd.concat([df_sh, df_sz], ignore_index=True)
        print(f"  上海: {len(df_sh)}, 深圳: {len(df_sz)}, 合计: {len(df_all)}")
    except Exception as e:
        print(f"  Error with spot data: {e}")
        print("  Trying alternative...")
        df_all = ak.stock_zh_a_spot_em()
        print(f"  Got {len(df_all)} stocks")

    print(f"  Columns: {list(df_all.columns)}")
    print(f"  Sample row:\n{df_all.iloc[0]}")

    # Filter by market cap
    # Find market cap column
    mc_col = None
    for col in df_all.columns:
        if '总市值' in col or '流通市值' in col:
            mc_col = col
            break

    if mc_col is None:
        print(f"  WARNING: Market cap column not found. Available: {list(df_all.columns)}")
        # Try to get it another way
        print("  Using stock_zh_a_spot_em instead...")
        df_all = ak.stock_zh_a_spot_em()
        print(f"  Columns: {list(df_all.columns)}")
        for col in df_all.columns:
            if '总市值' in col:
                mc_col = col
                break

    if mc_col is None:
        print("ERROR: Cannot find market cap column!")
        return

    print(f"\n  Using market cap column: '{mc_col}'")

    # Convert to numeric and filter
    df_all[mc_col] = pd.to_numeric(df_all[mc_col], errors='coerce')
    df_all['market_cap_yi'] = df_all[mc_col] / 1e8  # convert to 亿

    # Filter 100-1000亿
    df_filtered = df_all[(df_all['market_cap_yi'] >= 100) & (df_all['market_cap_yi'] <= 1000)].copy()
    print(f"\n  市值100-1000亿: {len(df_filtered)} stocks")

    # Remove ST stocks
    name_col = None
    for col in df_filtered.columns:
        if '名称' in col:
            name_col = col
            break

    if name_col:
        df_filtered = df_filtered[~df_filtered[name_col].str.contains('ST|退', na=False)]
        print(f"  去除ST/退市: {len(df_filtered)} stocks")

    code_col = None
    for col in df_filtered.columns:
        if '代码' in col:
            code_col = col
            break

    print(f"\n[2/4] 获取财务指标...")

    # Get PE, PB etc from the data we already have
    pe_col = None
    pb_col = None
    for col in df_filtered.columns:
        if '市盈率' in col and 'TTM' not in col.upper() and pe_col is None:
            pe_col = col
        if '市盈率' in col and 'TTM' in col.upper():
            pe_col = col  # prefer TTM
        if '市净率' in col:
            pb_col = col

    print(f"  PE column: {pe_col}, PB column: {pb_col}")

    # Get financial data via akshare
    # Try to get key financial metrics
    results = []
    total = len(df_filtered)
    failed = []

    for i, (_, row) in enumerate(df_filtered.iterrows()):
        if i % 100 == 0:
            print(f"  Processing: {i}/{total} ({time.time()-t0:.0f}s)")

        code = str(row[code_col]) if code_col else ''
        name = str(row[name_col]) if name_col else ''
        mc = row['market_cap_yi']
        pe = pd.to_numeric(row.get(pe_col, np.nan), errors='coerce') if pe_col else np.nan
        pb = pd.to_numeric(row.get(pb_col, np.nan), errors='coerce') if pb_col else np.nan

        # Get price change columns
        change_col = None
        for col in df_filtered.columns:
            if '涨跌幅' in col:
                change_col = col
                break

        change_pct = pd.to_numeric(row.get(change_col, 0), errors='coerce') if change_col else 0

        # Try to get financial indicators
        try:
            fin = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
            if fin is not None and len(fin) > 0:
                # Extract key metrics
                latest = fin.iloc[0]
                stock_info = {
                    'code': code,
                    'name': name,
                    'market_cap': round(mc, 1),
                    'pe': round(pe, 1) if pd.notna(pe) and pe > 0 else None,
                    'pb': round(pb, 2) if pd.notna(pb) and pb > 0 else None,
                    'financial_data': latest.to_dict() if len(latest) > 0 else {},
                }
                results.append(stock_info)
            else:
                results.append({
                    'code': code, 'name': name, 'market_cap': round(mc, 1),
                    'pe': round(pe, 1) if pd.notna(pe) and pe > 0 else None,
                    'pb': round(pb, 2) if pd.notna(pb) and pb > 0 else None,
                    'financial_data': {},
                })
        except Exception as e:
            results.append({
                'code': code, 'name': name, 'market_cap': round(mc, 1),
                'pe': round(pe, 1) if pd.notna(pe) and pe > 0 else None,
                'pb': round(pb, 2) if pd.notna(pb) and pb > 0 else None,
                'financial_data': {},
            })
            failed.append(code)

        if i % 20 == 19:
            time.sleep(0.5)

    print(f"\n  Got data for {len(results)} stocks, {len(failed)} financial data failures")

    # 3. Get growth data using baostock (faster for specific metrics)
    print(f"\n[3/4] 获取成长数据(baostock)...")
    import baostock as bs
    bs.login()

    for i, s in enumerate(results):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(results)}")

        code = s['code']
        # Convert to baostock format
        if code.startswith('6'):
            bs_code = f'sh.{code}'
        else:
            bs_code = f'sz.{code}'

        # Get 2024 growth
        rs = bs.query_growth_data(code=bs_code, year=2024, quarter=4)
        while rs.next():
            row = rs.get_row_data()
            try:
                s['yoy_ni'] = round(float(row[5]) * 100, 1) if row[5] else None
                s['yoy_pni'] = round(float(row[7]) * 100, 1) if row[7] else None
            except:
                pass

        # Get 2023 growth for trend
        rs = bs.query_growth_data(code=bs_code, year=2023, quarter=4)
        while rs.next():
            row = rs.get_row_data()
            try:
                s['yoy_ni_2023'] = round(float(row[5]) * 100, 1) if row[5] else None
            except:
                pass

        # Get ROE and revenue
        rs = bs.query_profit_data(code=bs_code, year=2024, quarter=4)
        while rs.next():
            row = rs.get_row_data()
            try:
                s['roe'] = round(float(row[3]) * 100, 1) if row[3] else None
                s['revenue'] = float(row[8]) if row[8] else None
            except:
                pass

        rs = bs.query_profit_data(code=bs_code, year=2023, quarter=4)
        while rs.next():
            row = rs.get_row_data()
            try:
                s['revenue_2023'] = float(row[8]) if row[8] else None
            except:
                pass

        # Get industry
        rs = bs.query_stock_industry(code=bs_code)
        while rs.next():
            row = rs.get_row_data()
            if len(row) >= 4:
                s['industry'] = row[3]
                break

    bs.logout()

    # Revenue growth
    for s in results:
        rev = s.get('revenue')
        rev_prev = s.get('revenue_2023')
        if rev and rev_prev and rev_prev > 0:
            s['rev_growth'] = round((rev / rev_prev - 1) * 100, 1)
        else:
            s['rev_growth'] = None

    print(f"\n[4/4] 综合评分...")

    # Score each stock
    for s in results:
        score = 0
        reasons = []
        penalties = []

        mc = s.get('market_cap', 0)
        pe = s.get('pe')
        pb = s.get('pb')
        roe = s.get('roe')
        ni_growth = s.get('yoy_ni')
        ni_growth_prev = s.get('yoy_ni_2023')
        rev_growth = s.get('rev_growth')
        industry = s.get('industry', '')

        # === GROWTH (max 40) ===
        if ni_growth is not None:
            if ni_growth > 100:
                score += 15; reasons.append(f"净利润爆增{ni_growth}%")
            elif ni_growth > 50:
                score += 12; reasons.append(f"净利润高增{ni_growth}%")
            elif ni_growth > 30:
                score += 8; reasons.append(f"净利润增{ni_growth}%")
            elif ni_growth > 15:
                score += 5
            elif ni_growth < -20:
                score -= 5; penalties.append(f"净利润降{ni_growth}%")

        if rev_growth is not None:
            if rev_growth > 30:
                score += 10; reasons.append(f"营收增{rev_growth}%")
            elif rev_growth > 15:
                score += 6
            elif rev_growth > 5:
                score += 3
            elif rev_growth < -10:
                score -= 3

        # Growth acceleration
        if ni_growth is not None and ni_growth_prev is not None:
            if ni_growth > ni_growth_prev and ni_growth > 20:
                score += 8; reasons.append("利润增速加快")
            elif ni_growth < ni_growth_prev:
                score -= 2

        if roe is not None:
            if roe > 20:
                score += 5; reasons.append(f"ROE={roe}%")
            elif roe > 15:
                score += 3
            elif roe < 3:
                score -= 3

        # === VALUATION (max 20) ===
        if pe is not None and pe > 0 and ni_growth is not None and ni_growth > 0:
            peg = pe / ni_growth
            if peg < 0.5:
                score += 10; reasons.append(f"PEG={peg:.1f}极低")
            elif peg < 1.0:
                score += 7; reasons.append(f"PEG={peg:.1f}")
            elif peg < 1.5:
                score += 4
            elif peg > 3:
                score -= 3
        elif pe is not None:
            if 0 < pe < 15:
                score += 5
            elif pe > 100:
                score -= 5

        if pb is not None:
            if 0 < pb < 2:
                score += 3
            elif pb > 10:
                score -= 2

        # === MARKET CAP (max 10) ===
        if mc < 200:
            score += 10; reasons.append(f"市值{mc:.0f}亿空间大")
        elif mc < 400:
            score += 7
        elif mc < 600:
            score += 4
        else:
            score += 1

        # === INDUSTRY (max 15) ===
        high_growth = ['半导体', '新能源', '人工智能', 'AI', '芯片', '光伏', '锂电', '军工', '航天',
                      '生物医药', '创新药', '医疗器械', '机器人', '自动驾驶', '云计算', '算力',
                      '新材料', '碳纤维', '电子', '软件', '互联网']
        moderate = ['计算机', '通信', '传媒', '汽车', '消费电子', '医药', '化工', '电力设备', '有色金属']
        low = ['银行', '保险', '房地产', '煤炭', '钢铁', '建筑', '公用事业', '交通运输', '纺织', '造纸']

        ind_score = 3
        for k in high_growth:
            if k in industry:
                ind_score = 15; reasons.append(f"高景气:{industry}")
                break
        if ind_score == 3:
            for k in moderate:
                if k in industry:
                    ind_score = 8; reasons.append(f"成长行业:{industry}")
                    break
        if ind_score == 3:
            for k in low:
                if k in industry:
                    ind_score = -3; penalties.append(f"传统行业:{industry}")
                    break
        score += ind_score

        s['score'] = score
        s['reasons'] = reasons
        s['penalties'] = penalties

    # Sort by score
    results.sort(key=lambda x: -x['score'])

    # Print top 30
    print(f"\n{'='*60}")
    print(f"TOP 30 - 未来三年10倍潜力股")
    print(f"筛选范围: {len(results)}只（市值100-1000亿）")
    print(f"{'='*60}")

    for i, s in enumerate(results[:30]):
        print(f"\n{i+1}. {s['name']} ({s['code']}) 得分:{s['score']}")
        print(f"   市值:{s['market_cap']:.0f}亿 PE:{s.get('pe','N/A')} PB:{s.get('pb','N/A')} ROE:{s.get('roe','N/A')}%")
        ni = s.get('yoy_ni', 'N/A')
        rev = s.get('rev_growth', 'N/A')
        print(f"   净利润增长:{ni}% 营收增长:{rev}% 行业:{s.get('industry','')}")
        if s['reasons']:
            print(f"   亮点: {'; '.join(s['reasons'])}")
        if s['penalties']:
            print(f"   风险: {'; '.join(s['penalties'])}")

    # Industry distribution
    print(f"\nTop30行业分布:")
    ind_dist = defaultdict(int)
    for s in results[:30]:
        ind_dist[s.get('industry', '?')] += 1
    for ind, cnt in sorted(ind_dist.items(), key=lambda x: -x[1]):
        print(f"  {ind}: {cnt}")

    # Save
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_screened': len(df_filtered),
        'results_count': len(results),
        'top_30': [{
            'rank': i + 1,
            'code': s['code'],
            'name': s['name'],
            'score': s['score'],
            'market_cap': s['market_cap'],
            'pe': s.get('pe'),
            'pb': s.get('pb'),
            'roe': s.get('roe'),
            'yoy_ni': s.get('yoy_ni'),
            'rev_growth': s.get('rev_growth'),
            'industry': s.get('industry', ''),
            'reasons': s['reasons'],
            'penalties': s['penalties'],
        } for i, s in enumerate(results[:30])]
    }

    out_path = f'{DATA_DIR}/astock_10x_screening.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n保存至: {out_path}")
    print(f"总耗时: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
