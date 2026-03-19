#!/usr/bin/env python3
"""
行业ETF轮动策略 v1.1
基于申万一级行业月度PE百分位 + 趋势双维度评分
数据: akshare月报数据 (2000至今, 314个月报) + 申万指数价格

作者: Sarah Mitchell (VisionClaw)
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json, os, warnings
warnings.filterwarnings('ignore')

CACHE_DIR = '/tmp/etf-trader/sector_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# 申万一级行业 → 代表ETF 映射 (精简为流动性较好的ETF)
# ============================================================
SECTOR_ETF_MAP = {
    '801010': {'name': '农林牧渔', 'etf': '516550', 'etf_name': '农业ETF'},
    '801030': {'name': '基础化工', 'etf': '516020', 'etf_name': '化工ETF'},
    '801050': {'name': '有色金属', 'etf': '512400', 'etf_name': '有色ETF'},
    '801080': {'name': '电子',   'etf': '512480', 'etf_name': '科技ETF'},
    '801110': {'name': '家用电器', 'etf': '159996', 'etf_name': '家电ETF'},
    '801120': {'name': '食品饮料', 'etf': '515170', 'etf_name': '食品ETF'},
    '801150': {'name': '医药生物', 'etf': '512010', 'etf_name': '医药ETF'},
    '801160': {'name': '公用事业', 'etf': '516880', 'etf_name': '公用ETF'},
    '801180': {'name': '房地产', 'etf': '515000', 'etf_name': '地产ETF'},
    '801200': {'name': '商贸零售', 'etf': '515650', 'etf_name': '消费ETF'},
    '801720': {'name': '建筑材料', 'etf': '159745', 'etf_name': '建材ETF'},
    '801760': {'name': '传媒',   'etf': '512980', 'etf_name': '传媒ETF'},
    '801770': {'name': '通信',   'etf': '515880', 'etf_name': '通信ETF'},
    '801780': {'name': '银行',   'etf': '512800', 'etf_name': '银行ETF'},
    '801790': {'name': '非银金融', 'etf': '512070', 'etf_name': '证券ETF'},
    '801740': {'name': '国防军工', 'etf': '512660', 'etf_name': '军工ETF'},
    '801750': {'name': '计算机', 'etf': '512720', 'etf_name': '计算机ETF'},
    '801960': {'name': '石油石化', 'etf': '512050', 'etf_name': '能源ETF'},
    '801950': {'name': '煤炭',   'etf': '515220', 'etf_name': '煤炭ETF'},
    '801900': {'name': '电力设备', 'etf': '516160', 'etf_name': '新能源ETF'},
    '801890': {'name': '机械设备', 'etf': '159892', 'etf_name': '机械ETF'},
    '801040': {'name': '钢铁',   'etf': '515210', 'etf_name': '钢铁ETF'},
    '801880': {'name': '汽车',   'etf': '516110', 'etf_name': '新能源车ETF'},
}

# 核心底仓
SMART_BETA = [
    {'code': '512890', 'name': '红利低波ETF', 'type': 'dividend'},
    {'code': '159793', 'name': '自由现金流ETF', 'type': 'cashflow'},
    {'code': '510300', 'name': '沪深300ETF', 'type': 'broad'},
]


def load_monthly_pe_data(use_cache: bool = True, years: int = 15) -> pd.DataFrame:
    """
    加载申万一级行业月度PE数据
    缓存到本地，避免重复下载
    """
    cache_file = f"{CACHE_DIR}/sw_monthly_pe_cache.pkl"
    
    # Load existing cache
    if use_cache and os.path.exists(cache_file):
        df_cache = pd.read_pickle(cache_file)
        df_cache['date'] = pd.to_datetime(df_cache['date'])
        latest_cached = df_cache['date'].max()
        print(f"  缓存数据: {len(df_cache)} 行, 最新: {latest_cached.strftime('%Y-%m')}")
    else:
        df_cache = pd.DataFrame()
        latest_cached = None
    
    # Get available monthly dates
    all_dates = ak.index_analysis_week_month_sw()
    all_dates['date'] = pd.to_datetime(all_dates['date'])
    
    # Filter to needed range
    cutoff = datetime.now() - timedelta(days=years * 365)
    needed_dates = all_dates[all_dates['date'] >= cutoff].sort_values('date')
    
    # Only fetch dates not in cache
    if latest_cached is not None:
        needed_dates = needed_dates[needed_dates['date'] > latest_cached]
    
    if len(needed_dates) == 0:
        print("  缓存已是最新，无需下载")
        return df_cache
    
    print(f"  需要下载 {len(needed_dates)} 个月的数据...")
    
    new_records = []
    for i, row in enumerate(needed_dates.itertuples()):
        date_str = row.date.strftime('%Y%m%d')
        try:
            df_month = ak.index_analysis_monthly_sw(symbol='一级行业', date=date_str)
            if len(df_month) > 0:
                df_month['report_date'] = row.date
                new_records.append(df_month)
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(needed_dates)}")
        except Exception as e:
            pass  # Skip failed months
    
    if new_records:
        df_new = pd.concat(new_records, ignore_index=True)
        df_new['date'] = df_new['report_date']
        df_new['pe'] = pd.to_numeric(df_new.get('市盈率', pd.Series()), errors='coerce')
        df_new['pb'] = pd.to_numeric(df_new.get('市净率', pd.Series()), errors='coerce')
        df_new['close'] = pd.to_numeric(df_new.get('收盘指数', pd.Series()), errors='coerce')
        df_new['sector_code'] = df_new['指数代码'].astype(str)
        df_new['sector_name'] = df_new['指数名称']
        
        df_new = df_new[['date', 'sector_code', 'sector_name', 'pe', 'pb', 'close']].dropna(subset=['pe'])
        
        df_all = pd.concat([df_cache, df_new], ignore_index=True) if len(df_cache) > 0 else df_new
        df_all = df_all.drop_duplicates(['date', 'sector_code']).sort_values(['sector_code', 'date'])
        
        df_all.to_pickle(cache_file)
        print(f"  缓存已更新: {len(df_all)} 行")
        return df_all
    
    return df_cache


def fetch_sw_index_price(sector_code: str) -> pd.DataFrame:
    """获取申万行业指数历史价格"""
    try:
        df = ak.index_hist_sw(symbol=sector_code, period='day')
        if len(df) > 0:
            df.columns = [c.lower() for c in df.columns]
            # Find date and close columns
            date_col = next((c for c in df.columns if 'date' in c or '日期' in c), None)
            close_col = next((c for c in df.columns if 'close' in c or '收盘' in c), None)
            if date_col and close_col:
                result = df[[date_col, close_col]].copy()
                result.columns = ['date', 'close']
                result['date'] = pd.to_datetime(result['date'])
                result['close'] = pd.to_numeric(result['close'], errors='coerce')
                return result.dropna().sort_values('date')
    except Exception as e:
        pass
    return pd.DataFrame()


def calc_pe_percentile(pe_series: pd.Series) -> float:
    """计算当前PE在历史数据中的百分位"""
    if len(pe_series) < 12:
        return 50.0
    hist = pe_series.dropna()
    current = hist.iloc[-1]
    if current <= 0:
        return 50.0
    return float((hist < current).mean() * 100)


def calc_trend_score(price_series: pd.Series) -> float:
    """计算趋势评分 0-100 (基于月度价格)"""
    if len(price_series) < 6:
        return 50.0
    p = price_series.dropna()
    scores = []
    
    # MA3/12 相对位置 (月度)
    if len(p) >= 12:
        ma3 = p.rolling(3).mean().iloc[-1]
        ma12 = p.rolling(12).mean().iloc[-1]
        cur = p.iloc[-1]
        scores.extend([1 if cur > ma3 else 0, 1 if cur > ma12 else 0, 1 if ma3 > ma12 else 0])
    
    # 3个月涨跌幅
    if len(p) >= 3:
        ret3 = (p.iloc[-1] / p.iloc[-3] - 1) * 100
        scores.append(1 if ret3 > 0 else 0)
        scores.append(min(max(ret3 / 15 + 0.5, 0), 1))
    
    # 6个月涨跌幅
    if len(p) >= 6:
        ret6 = (p.iloc[-1] / p.iloc[-6] - 1) * 100
        scores.append(min(max(ret6 / 20 + 0.5, 0), 1))
    
    return float(np.mean(scores) * 100) if scores else 50.0


def score_sector(pe_pct: float, trend: float) -> tuple:
    """双维度评分"""
    pe_score = 100 - pe_pct  # 低估值=高分
    composite = pe_score * 0.55 + trend * 0.45
    
    if composite >= 68:
        return composite, '强烈买入', 0.15
    elif composite >= 55:
        return composite, '买入', 0.10
    elif composite >= 42:
        return composite, '持有', 0.05
    elif composite >= 28:
        return composite, '减仓', 0.02
    else:
        return composite, '回避', 0.0


def get_current_signals(df_pe: pd.DataFrame) -> list:
    """获取当前行业信号"""
    results = []
    
    for sector_code, info in SECTOR_ETF_MAP.items():
        sector_data = df_pe[df_pe['sector_code'] == sector_code].sort_values('date')
        if len(sector_data) < 12:
            continue
        
        current_pe = sector_data['pe'].iloc[-1]
        current_close = sector_data['close'].iloc[-1]
        
        if pd.isna(current_pe) or current_pe <= 0:
            continue
        
        pe_pct = calc_pe_percentile(sector_data['pe'])
        trend = calc_trend_score(sector_data['close'])
        composite, action, weight = score_sector(pe_pct, trend)
        
        results.append({
            'sector_code': sector_code,
            'sector_name': info['name'],
            'etf': info['etf'],
            'etf_name': info['etf_name'],
            'pe': round(current_pe, 1),
            'pe_percentile': round(pe_pct, 1),
            'trend_score': round(trend, 1),
            'composite_score': round(composite, 1),
            'action': action,
            'target_weight': weight,
        })
    
    results.sort(key=lambda x: -x['composite_score'])
    return results


def run_backtest(df_pe: pd.DataFrame, years: int = 15, top_n: int = 3) -> dict:
    """
    月度调仓回测: 每月选综合评分最高top_n个行业持有
    对比: 等权持有所有行业 (Buy-and-Hold)
    """
    print(f"\n🔄 运行{years}年行业轮动回测 (月度调仓, Top{top_n}行业)...")
    
    cutoff = datetime.now() - timedelta(days=years * 365)
    df = df_pe[df_pe['date'] >= cutoff].copy()
    
    all_months = sorted(df['date'].unique())
    valid_sectors = list(SECTOR_ETF_MAP.keys())
    
    # Create price pivot (monthly)
    price_pivot = df.pivot_table(index='date', columns='sector_code', values='close')
    pe_pivot = df.pivot_table(index='date', columns='sector_code', values='pe')
    
    strategy_value = 1.0
    bh_value = 1.0
    
    strategy_returns = []
    bh_returns = []
    dates_used = []
    current_positions = {}
    
    lookback = 60  # 5yr monthly lookback for PE percentile
    
    for i, rebal_month in enumerate(all_months):
        if i < lookback:
            continue
        
        # Score sectors using data up to this month
        hist_slice = pe_pivot[pe_pivot.index <= rebal_month]
        price_slice = price_pivot[price_pivot.index <= rebal_month]
        
        sector_scores = {}
        for code in valid_sectors:
            if code not in hist_slice.columns:
                continue
            pe_hist = hist_slice[code].dropna().tail(lookback)
            price_hist = price_slice[code].dropna().tail(lookback) if code in price_slice.columns else pd.Series()
            
            if len(pe_hist) < 12:
                continue
            
            current_pe = pe_hist.iloc[-1]
            if current_pe <= 0 or pd.isna(current_pe):
                continue
            
            pe_pct = float((pe_hist < current_pe).mean() * 100)
            trend = calc_trend_score(price_hist)
            composite, _, _ = score_sector(pe_pct, trend)
            sector_scores[code] = composite
        
        if not sector_scores:
            continue
        
        # Select top N
        selected = sorted(sector_scores.items(), key=lambda x: -x[1])[:top_n]
        new_positions = {s[0]: 1.0/top_n for s in selected}
        
        # Calculate return from prev month to this month
        if i > lookback and current_positions:
            prev_month = all_months[i-1]
            
            strat_ret = 0.0
            for code, weight in current_positions.items():
                if code in price_pivot.columns:
                    p_prev = price_pivot.loc[:prev_month, code].dropna()
                    p_cur = price_pivot.loc[:rebal_month, code].dropna()
                    if len(p_prev) > 0 and len(p_cur) > 0:
                        strat_ret += weight * (p_cur.iloc[-1] / p_prev.iloc[-1] - 1)
            
            bh_ret = 0.0
            valid_for_bh = [c for c in valid_sectors if c in price_pivot.columns]
            n_bh = len(valid_for_bh)
            for code in valid_for_bh:
                p_prev = price_pivot.loc[:prev_month, code].dropna()
                p_cur = price_pivot.loc[:rebal_month, code].dropna()
                if len(p_prev) > 0 and len(p_cur) > 0:
                    bh_ret += (p_cur.iloc[-1] / p_prev.iloc[-1] - 1) / n_bh
            
            strategy_value *= (1 + strat_ret)
            bh_value *= (1 + bh_ret)
            strategy_returns.append(strat_ret)
            bh_returns.append(bh_ret)
            dates_used.append(rebal_month)
        
        current_positions = new_positions
    
    if not strategy_returns:
        print("  数据不足，回测中止")
        return {}
    
    n_years = (dates_used[-1] - dates_used[0]).days / 365.25
    strat_annual = (strategy_value ** (1/n_years) - 1) * 100
    bh_annual = (bh_value ** (1/n_years) - 1) * 100
    
    def max_drawdown(ret_list):
        cum = pd.Series(ret_list).add(1).cumprod()
        roll_max = cum.cummax()
        return ((cum - roll_max) / roll_max).min() * 100
    
    strat_dd = max_drawdown(strategy_returns)
    bh_dd = max_drawdown(bh_returns)
    strat_sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10) * np.sqrt(12)
    bh_sharpe = np.mean(bh_returns) / (np.std(bh_returns) + 1e-10) * np.sqrt(12)
    
    start_str = dates_used[0].strftime('%Y-%m')
    end_str = dates_used[-1].strftime('%Y-%m')
    
    print(f"\n{'='*58}")
    print(f"📊 行业轮动策略回测结果 ({years}年, 月度调仓, Top{top_n})")
    print(f"   回测区间: {start_str} ~ {end_str} ({n_years:.1f}年)")
    print(f"{'='*58}")
    print(f"{'指标':<15} {'轮动策略':>12} {'等权持有':>12}")
    print(f"{'-'*40}")
    print(f"{'累计收益':<15} {(strategy_value-1)*100:>11.1f}% {(bh_value-1)*100:>11.1f}%")
    print(f"{'年化收益':<15} {strat_annual:>11.1f}% {bh_annual:>11.1f}%")
    print(f"{'最大回撤':<15} {strat_dd:>11.1f}% {bh_dd:>11.1f}%")
    print(f"{'夏普比率':<15} {strat_sharpe:>11.2f} {bh_sharpe:>11.2f}")
    print(f"{'超额年化':<15} {strat_annual-bh_annual:>+11.1f}%")
    
    return {
        'strategy_annual': round(strat_annual, 2),
        'bh_annual': round(bh_annual, 2),
        'strategy_dd': round(strat_dd, 2),
        'bh_dd': round(bh_dd, 2),
        'strategy_sharpe': round(strat_sharpe, 2),
        'bh_sharpe': round(bh_sharpe, 2),
        'total_return': round((strategy_value-1)*100, 1),
        'years': round(n_years, 1),
        'start': start_str, 'end': end_str,
    }


def print_signals_report(results: list):
    """打印当日行业信号报告"""
    print("\n" + "="*65)
    print("📈 行业ETF轮动策略 — 当日信号")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*65)
    print(f"{'':2}{'行业':<8} {'ETF':<8} {'PE':>6} {'PE%位':>6} {'趋势':>5} {'综合':>5} {'建议'}")
    print("-"*55)
    
    for i, r in enumerate(results[:15]):
        flag = "🟢" if r['action'] in ('强烈买入', '买入') else "🟡" if r['action'] == '持有' else "🔴"
        print(f"{flag} {r['sector_name']:<7} {r['etf']:<8} {r['pe']:>6.1f} {r['pe_percentile']:>5.1f}% "
              f"{r['trend_score']:>5.1f} {r['composite_score']:>5.1f} {r['action']}")
    
    print("\n🎯 推荐行业 (Top 3):")
    for r in results[:3]:
        print(f"  ✅ {r['sector_name']} ({r['etf_name']}) | PE={r['pe']} ({r['pe_percentile']:.0f}%位) | 趋势{r['trend_score']:.0f}")
    
    print("\n⛔ 回避行业 (Bottom 3):")
    avoids = [r for r in reversed(results) if r['action'] in ('回避', '减仓')][:3]
    for r in avoids:
        print(f"  ❌ {r['sector_name']} ({r['etf_name']}) | PE={r['pe']} ({r['pe_percentile']:.0f}%位)")


if __name__ == '__main__':
    import sys
    
    do_backtest = '--backtest' in sys.argv
    years = 15
    for arg in sys.argv:
        if arg.startswith('--years='):
            years = int(arg.split('=')[1])
    top_n = 3
    for arg in sys.argv:
        if arg.startswith('--top='):
            top_n = int(arg.split('=')[1])
    
    print(f"📥 加载申万行业月度PE数据 (最近{years}年)...")
    df_pe = load_monthly_pe_data(use_cache=True, years=years)
    
    if len(df_pe) == 0:
        print("❌ 数据加载失败")
        sys.exit(1)
    
    print(f"✅ 数据加载完成: {len(df_pe)} 行, {df_pe['sector_code'].nunique()} 个行业")
    print(f"   时间范围: {df_pe['date'].min().strftime('%Y-%m')} ~ {df_pe['date'].max().strftime('%Y-%m')}")
    
    # Current signals
    signals = get_current_signals(df_pe)
    print_signals_report(signals)
    
    if do_backtest:
        result = run_backtest(df_pe, years=years, top_n=top_n)
        if result:
            result_file = f'/tmp/etf-trader/sector_backtest_{years}yr.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 回测结果已保存: {result_file}")

