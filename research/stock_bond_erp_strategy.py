#!/usr/bin/env python3
"""
股债性价比（ERP）平衡策略研究
Equity Risk Premium = 指数盈利收益率(1/PE) - 10年期国债收益率

策略逻辑：
- ERP高 → 股票便宜，增配股票ETF
- ERP低 → 股票贵，增配债券ETF
- 覆盖A股主要宽基指数

数据源：
- PE数据：蛋卷基金API (danjuanfunds.com)
- 国债收益率：akshare
- 指数价格：akshare / baostock

Author: Sarah Mitchell / VisionClaw
Date: 2026-03-25
"""

import akshare as ak
import pandas as pd
import numpy as np
import requests
import json
import warnings
import traceback
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


# ============================================================
# 1. 数据获取
# ============================================================

def get_bond_yield_10y():
    """获取中国10年期国债收益率 (日频)"""
    print("[1/5] 获取10年期国债收益率...")
    try:
        df = ak.bond_zh_us_rate(start_date="20100101")
        df = df[['日期', '中国国债收益率10年']].copy()
        df.columns = ['date', 'bond_yield']
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['bond_yield'])
        df = df.set_index('date').sort_index()
        if df['bond_yield'].mean() > 1:
            df['bond_yield'] = df['bond_yield'] / 100.0
        print(f"  ✓ 国债收益率: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}条")
        return df
    except Exception as e:
        print(f"  ✗ 国债收益率获取失败: {e}")
        return None


def get_pe_history_danjuan(index_code, index_name):
    """从蛋卷基金获取指数PE历史数据 (周频)"""
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    url = f'https://danjuanfunds.com/djapi/index_eva/pe_history/{index_code}?day=all'

    try:
        r = requests.get(url, headers=headers, timeout=15)
        data = r.json()

        if 'data' not in data or not data['data']:
            return None

        pe_data = data['data'].get('index_eva_pe_growths', [])
        if not pe_data:
            return None

        df = pd.DataFrame(pe_data)
        df['date'] = pd.to_datetime(df['ts'], unit='ms')
        df = df[['date', 'pe']].set_index('date').sort_index()
        df = df[df['pe'] > 0]  # Filter invalid PE

        print(f"  ✓ {index_name}: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}条, PE: {df['pe'].min():.1f}~{df['pe'].max():.1f}")
        return df

    except Exception as e:
        print(f"  ✗ {index_name}: {e}")
        return None


def get_index_price(index_code_ak, index_name, start_date="2016-01-01"):
    """获取指数日线价格 (baostock)"""
    import baostock as bs

    bs_map = {
        '000300': 'sh.000300', '000905': 'sh.000905', '000852': 'sh.000852',
        '000016': 'sh.000016', '399006': 'sz.399006', '000001': 'sh.000001',
        '000906': 'sh.000906',
    }
    bs_code = bs_map.get(index_code_ak)
    if not bs_code:
        return None

    try:
        lg = bs.login()
        rs = bs.query_history_k_data_plus(
            bs_code, "date,close",
            start_date=start_date,
            end_date=datetime.now().strftime("%Y-%m-%d"),
            frequency="d", adjustflag="3"
        )
        data = []
        while rs.error_code == '0' and rs.next():
            data.append(rs.get_row_data())
        bs.logout()

        if not data:
            return None
        df = pd.DataFrame(data, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date'])
        df['close'] = df['close'].astype(float)
        df = df[['date', 'close']].set_index('date').sort_index()
        return df
    except Exception as e:
        print(f"  ✗ {index_name}价格获取失败: {e}")
        try:
            bs.logout()
        except:
            pass
        return None


def get_all_data():
    """获取所有需要的数据"""
    # Index mapping: danjuan_code -> (akshare_code, name)
    indices = {
        'SH000300': ('000300', '沪深300'),
        'SH000905': ('000905', '中证500'),
        'SH000852': ('000852', '中证1000'),
        'SH000016': ('000016', '上证50'),
        'SZ399006': ('399006', '创业板指'),
        'SH000001': ('000001', '上证指数'),
        'SH000906': ('000906', '中证800'),
    }

    print("[2/5] 获取各指数PE历史数据 (蛋卷基金)...")
    pe_data = {}
    for dj_code, (ak_code, name) in indices.items():
        df = get_pe_history_danjuan(dj_code, name)
        if df is not None and len(df) > 50:
            pe_data[ak_code] = {'name': name, 'pe': df, 'dj_code': dj_code}

    print(f"\n[3/5] 获取各指数日线价格...")
    for ak_code in list(pe_data.keys()):
        name = pe_data[ak_code]['name']
        price_df = get_index_price(ak_code, name)
        if price_df is not None and len(price_df) > 100:
            pe_data[ak_code]['price'] = price_df
            print(f"  ✓ {name}价格: {price_df.index[0].date()} ~ {price_df.index[-1].date()}, {len(price_df)}条")
        else:
            del pe_data[ak_code]
            print(f"  ✗ {name}: 价格数据不足，移除")

    return pe_data


# ============================================================
# 2. ERP计算与信号
# ============================================================

def build_erp_dataset(pe_weekly, bond_yield_daily, price_daily):
    """
    构建ERP数据集
    PE数据是周频，需要插值到日频
    """
    # Resample PE to daily (forward fill)
    pe_daily = pe_weekly.resample('D').ffill()

    # Align all data
    combined = pd.DataFrame({
        'pe': pe_daily['pe'],
        'bond_yield': bond_yield_daily['bond_yield'],
        'close': price_daily['close'],
    }).ffill().dropna()

    # Calculate ERP
    combined['earnings_yield'] = 1.0 / combined['pe']
    combined['erp'] = combined['earnings_yield'] - combined['bond_yield']

    # ERP percentile (rolling)
    for window in [750, 1250]:
        col = f'erp_pct_{window}'
        combined[col] = combined['erp'].rolling(window=window, min_periods=min(250, window//2)).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

    # ERP全历史分位
    combined['erp_pct_all'] = combined['erp'].expanding(min_periods=250).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    return combined


def generate_signals(erp_df, method, **kwargs):
    """生成股票配置权重信号"""

    if method == 'continuous_3y':
        # 连续权重: 根据3年滚动分位数线性映射到[0, 1]
        pct = erp_df['erp_pct_750']
        weight = pct.clip(0.1, 0.9)  # 最低10%股，最高90%股
        return weight

    elif method == 'continuous_5y':
        pct = erp_df['erp_pct_1250']
        weight = pct.clip(0.1, 0.9)
        return weight

    elif method == 'continuous_all':
        pct = erp_df['erp_pct_all']
        weight = pct.clip(0.1, 0.9)
        return weight

    elif method == 'binary_3y':
        # 二值: 分位>50% 全仓股; <50% 全仓债
        pct = erp_df['erp_pct_750']
        return (pct >= 0.5).astype(float)

    elif method == 'binary_3y_40':
        pct = erp_df['erp_pct_750']
        return (pct >= 0.4).astype(float)

    elif method == '3level_3y':
        # 三档: <30%→全债, 30-70%→50/50, >70%→全股
        pct = erp_df['erp_pct_750']
        w = pd.Series(0.5, index=pct.index)
        w[pct >= 0.7] = 1.0
        w[pct < 0.3] = 0.0
        return w

    elif method == '3level_5y':
        pct = erp_df['erp_pct_1250']
        w = pd.Series(0.5, index=pct.index)
        w[pct >= 0.7] = 1.0
        w[pct < 0.3] = 0.0
        return w

    elif method == '5level_3y':
        # 五档: 20%分位一档
        pct = erp_df['erp_pct_750']
        w = pd.Series(0.5, index=pct.index)
        w[pct >= 0.8] = 1.0
        w[(pct >= 0.6) & (pct < 0.8)] = 0.75
        w[(pct >= 0.4) & (pct < 0.6)] = 0.5
        w[(pct >= 0.2) & (pct < 0.4)] = 0.25
        w[pct < 0.2] = 0.0
        return w

    elif method == 'ma_cross':
        # ERP均线交叉
        ma_period = kwargs.get('ma_period', 60)
        erp = erp_df['erp']
        erp_ma = erp.rolling(window=ma_period, min_periods=30).mean()
        return (erp > erp_ma).astype(float)

    elif method == 'dual_ma':
        # 双均线: 快线>慢线 → 股票
        erp = erp_df['erp']
        fast = erp.rolling(window=20, min_periods=10).mean()
        slow = erp.rolling(window=120, min_periods=60).mean()
        return (fast > slow).astype(float)

    return None


# ============================================================
# 3. 回测引擎
# ============================================================

def backtest(close, stock_weight, bond_yield, rebal_freq='weekly', cost=0.001):
    """
    回测股债平衡策略

    股票端: 跟踪指数
    债券端: 使用10年国债收益率/252作为日收益近似
    """
    df = pd.DataFrame({
        'close': close,
        'weight': stock_weight,
        'bond_yield': bond_yield,
    }).ffill().dropna()

    if len(df) < 500:
        return None

    df['stock_ret'] = df['close'].pct_change()
    df['bond_ret'] = df['bond_yield'] / 252

    # Rebalance dates
    if rebal_freq == 'weekly':
        df['period'] = df.index.isocalendar().week.astype(int) + df.index.year * 100
    elif rebal_freq == 'monthly':
        df['period'] = df.index.month + df.index.year * 100
    rebal = df['period'] != df['period'].shift(1)

    # Simulate
    nav = np.ones(len(df))
    stock_nav = np.ones(len(df))
    bond_nav = np.ones(len(df))
    bal50_nav = np.ones(len(df))

    cur_w = 0.5
    n_trades = 0
    total_turnover = 0.0

    for i in range(1, len(df)):
        sr = df['stock_ret'].iloc[i]
        br = df['bond_ret'].iloc[i]
        if np.isnan(sr): sr = 0
        if np.isnan(br): br = 0

        # Portfolio
        port_ret = cur_w * sr + (1 - cur_w) * br

        # Rebalance
        if rebal.iloc[i]:
            target = df['weight'].iloc[i]
            if not np.isnan(target):
                change = abs(target - cur_w)
                if change > 0.01:
                    port_ret -= change * cost
                    n_trades += 1
                    total_turnover += change
                    cur_w = target

        nav[i] = nav[i-1] * (1 + port_ret)
        stock_nav[i] = stock_nav[i-1] * (1 + sr)
        bond_nav[i] = bond_nav[i-1] * (1 + br)
        bal50_nav[i] = bal50_nav[i-1] * (1 + 0.5*sr + 0.5*br)

    # Metrics
    def metrics(nav_arr, name):
        s = pd.Series(nav_arr, index=df.index)
        ret = s.pct_change().dropna()
        years = len(ret) / 252
        total = s.iloc[-1] / s.iloc[0] - 1
        ann_ret = (1 + total) ** (1/years) - 1 if years > 0 else 0
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        dd = (s - s.cummax()) / s.cummax()
        max_dd = dd.min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        return {
            'name': name,
            'total_ret': round(total*100, 1),
            'ann_ret': round(ann_ret*100, 2),
            'ann_vol': round(ann_vol*100, 2),
            'sharpe': round(sharpe, 3),
            'max_dd': round(max_dd*100, 2),
            'calmar': round(calmar, 3),
            'years': round(years, 1),
        }

    return {
        'strategy': metrics(nav, 'ERP策略'),
        'stock': metrics(stock_nav, '纯股票'),
        'bond': metrics(bond_nav, '纯债券'),
        'bal50': metrics(bal50_nav, '固定50/50'),
        'trades': n_trades,
        'avg_turnover': round(total_turnover / max(n_trades, 1) * 100, 1),
        'start_date': str(df.index[0].date()),
        'end_date': str(df.index[-1].date()),
    }


# ============================================================
# 4. 主研究流程
# ============================================================

def run_research():
    print("=" * 70)
    print("📊 股债性价比(ERP)平衡策略研究")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Get bond yield
    bond_df = get_bond_yield_10y()
    if bond_df is None:
        return None

    # Get all index data
    all_data = get_all_data()
    if not all_data:
        return None

    # Strategy list
    strategies = [
        ('连续权重_3年分位', 'continuous_3y', {}),
        ('连续权重_5年分位', 'continuous_5y', {}),
        ('连续权重_全历史', 'continuous_all', {}),
        ('二值_3年_50%', 'binary_3y', {}),
        ('二值_3年_40%', 'binary_3y_40', {}),
        ('三档_3年', '3level_3y', {}),
        ('三档_5年', '3level_5y', {}),
        ('五档_3年', '5level_3y', {}),
        ('均线交叉_60日', 'ma_cross', {'ma_period': 60}),
        ('均线交叉_120日', 'ma_cross', {'ma_period': 120}),
        ('双均线_20_120', 'dual_ma', {}),
    ]

    print(f"\n[4/5] 构建ERP数据集 & 回测...")
    print("=" * 70)

    results = {}

    for ak_code, info in all_data.items():
        name = info['name']
        print(f"\n{'─'*60}")
        print(f"📈 {name} ({ak_code})")
        print(f"{'─'*60}")

        try:
            erp_df = build_erp_dataset(info['pe'], bond_df, info['price'])
        except Exception as e:
            print(f"  构建ERP数据失败: {e}")
            traceback.print_exc()
            continue

        if len(erp_df) < 500:
            print(f"  数据不足 ({len(erp_df)}条)，跳过")
            continue

        # Current status
        latest = erp_df.iloc[-1]
        print(f"  数据范围: {erp_df.index[0].date()} ~ {erp_df.index[-1].date()} ({len(erp_df)}天)")
        print(f"  当前PE: {latest['pe']:.2f} | 盈利收益率: {latest['earnings_yield']*100:.2f}%")
        print(f"  当前国债: {latest['bond_yield']*100:.2f}% | ERP: {latest['erp']*100:.2f}%")

        if not np.isnan(latest.get('erp_pct_750', np.nan)):
            print(f"  ERP 3年分位: {latest['erp_pct_750']*100:.1f}%")
        if not np.isnan(latest.get('erp_pct_all', np.nan)):
            print(f"  ERP 全历史分位: {latest['erp_pct_all']*100:.1f}%")

        index_results = {}

        for strat_label, method, params in strategies:
            try:
                weight = generate_signals(erp_df, method, **params)
                if weight is None:
                    continue

                result = backtest(
                    erp_df['close'], weight, erp_df['bond_yield'],
                    rebal_freq='weekly', cost=0.001
                )

                if result:
                    s = result['strategy']
                    b = result['bal50']
                    stk = result['stock']

                    alpha_vs_50 = s['ann_ret'] - b['ann_ret']
                    dd_improve = stk['max_dd'] - s['max_dd']

                    print(f"  {strat_label:18s} | 年化{s['ann_ret']:+6.1f}% 夏普{s['sharpe']:5.2f} 回撤{s['max_dd']:6.1f}% | α vs50/50: {alpha_vs_50:+.1f}% | 回撤改善: {dd_improve:+.1f}%")

                    index_results[strat_label] = result

            except Exception as e:
                print(f"  {strat_label}: ERROR - {e}")

        # Store
        results[ak_code] = {
            'name': name,
            'results': index_results,
            'current': {
                'pe': round(latest['pe'], 2),
                'earnings_yield': round(latest['earnings_yield'] * 100, 2),
                'bond_yield': round(latest['bond_yield'] * 100, 2),
                'erp': round(latest['erp'] * 100, 2),
                'erp_pct_3y': round(latest.get('erp_pct_750', 0) * 100, 1) if not np.isnan(latest.get('erp_pct_750', np.nan)) else None,
                'erp_pct_all': round(latest.get('erp_pct_all', 0) * 100, 1) if not np.isnan(latest.get('erp_pct_all', np.nan)) else None,
            }
        }

    return results


def format_report(results):
    """格式化研究报告"""
    if not results:
        return "无结果"

    lines = []
    lines.append("=" * 60)
    lines.append("📊 股债性价比(ERP)平衡策略研究报告")
    lines.append(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)

    # 1. Current ERP Overview
    lines.append("\n🔍 各指数当前股债性价比:")
    lines.append("─" * 50)
    for code, info in results.items():
        c = info['current']
        pct_str = f"{c['erp_pct_3y']:.0f}%" if c['erp_pct_3y'] is not None else "N/A"
        lines.append(f"  {info['name']:8s} PE={c['pe']:6.1f} 盈利收益率={c['earnings_yield']:.2f}% 国债={c['bond_yield']:.2f}% ERP={c['erp']:+.2f}% 3年分位={pct_str}")

    # 2. Best strategy per index
    lines.append("\n🏆 各指数最优策略 (Sharpe最高):")
    lines.append("─" * 60)

    global_best = None
    global_best_sharpe = -999

    for code, info in results.items():
        if not info['results']:
            continue

        best_name = None
        best_sharpe = -999
        for strat_name, result in info['results'].items():
            if result['strategy']['sharpe'] > best_sharpe:
                best_sharpe = result['strategy']['sharpe']
                best_name = strat_name

        if best_name:
            r = info['results'][best_name]
            s = r['strategy']
            b = r['bal50']
            stk = r['stock']
            bnd = r['bond']

            lines.append(f"\n  {info['name']} ({code})")
            lines.append(f"    最优策略: {best_name}")
            lines.append(f"    ERP策略:  年化{s['ann_ret']:+.1f}% 夏普{s['sharpe']:.2f} 回撤{s['max_dd']:.1f}% Calmar{s['calmar']:.2f}")
            lines.append(f"    固定50/50: 年化{b['ann_ret']:+.1f}% 夏普{b['sharpe']:.2f} 回撤{b['max_dd']:.1f}%")
            lines.append(f"    纯股票:   年化{stk['ann_ret']:+.1f}% 夏普{stk['sharpe']:.2f} 回撤{stk['max_dd']:.1f}%")
            lines.append(f"    纯债券:   年化{bnd['ann_ret']:+.1f}% 夏普{bnd['sharpe']:.2f} 回撤{bnd['max_dd']:.1f}%")
            lines.append(f"    回测期: {r['start_date']} ~ {r['end_date']} ({s['years']}年)")
            lines.append(f"    调仓{r['trades']}次, 平均换仓{r['avg_turnover']}%")
            lines.append(f"    ⭐ vs固定50/50: 年化{s['ann_ret']-b['ann_ret']:+.1f}%, 夏普{s['sharpe']-b['sharpe']:+.2f}")
            lines.append(f"    ⭐ vs纯股票: 回撤改善{stk['max_dd']-s['max_dd']:+.1f}%")

            if best_sharpe > global_best_sharpe:
                global_best_sharpe = best_sharpe
                global_best = (code, info['name'], best_name, r)

    # 3. Global best
    if global_best:
        code, name, strat, r = global_best
        s = r['strategy']
        lines.append(f"\n{'='*60}")
        lines.append(f"🥇 全局最优组合: {name} + {strat}")
        lines.append(f"   年化{s['ann_ret']:+.1f}% | 夏普{s['sharpe']:.2f} | 回撤{s['max_dd']:.1f}% | Calmar{s['calmar']:.2f}")

    # 4. Current recommendations
    lines.append(f"\n{'='*60}")
    lines.append("💡 当前配置建议:")
    for code, info in results.items():
        c = info['current']
        pct = c.get('erp_pct_3y')
        if pct is None:
            advice = "⚪ 数据不足"
        elif pct >= 70:
            advice = "🟢 高性价比 → 建议增配股票 (ERP分位>70%)"
        elif pct >= 40:
            advice = "🟡 中等 → 维持均衡 (ERP分位40-70%)"
        else:
            advice = "🔴 低性价比 → 建议增配债券 (ERP分位<40%)"
        lines.append(f"  {info['name']:8s} ERP={c['erp']:+.2f}% 分位={pct if pct else 'N/A'}%  {advice}")

    # 5. Full comparison table
    lines.append(f"\n{'='*60}")
    lines.append("📋 全部策略对比:")
    lines.append(f"{'指数':8s} {'策略':18s} {'年化%':>7s} {'夏普':>6s} {'回撤%':>7s} {'Calmar':>7s} {'vs50/50':>8s}")
    lines.append("─" * 70)

    for code, info in results.items():
        for strat_name, result in info['results'].items():
            s = result['strategy']
            b = result['bal50']
            alpha = s['ann_ret'] - b['ann_ret']
            lines.append(f"{info['name']:8s} {strat_name:18s} {s['ann_ret']:7.1f} {s['sharpe']:6.2f} {s['max_dd']:7.1f} {s['calmar']:7.2f} {alpha:+8.1f}")

    return "\n".join(lines)


def save_results(results, report):
    """保存结果"""
    print(f"\n[5/5] 保存结果...")

    report_path = "/Users/claw/etf-trader/data/erp_strategy_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  ✓ 报告: {report_path}")

    # JSON
    json_data = {}
    for code, info in results.items():
        idx_json = {
            'name': info['name'],
            'current': info['current'],
            'strategies': {}
        }
        for strat_name, result in info['results'].items():
            idx_json['strategies'][strat_name] = {
                'strategy': result['strategy'],
                'stock': result['stock'],
                'bond': result['bond'],
                'bal50': result['bal50'],
                'trades': result['trades'],
                'start_date': result['start_date'],
                'end_date': result['end_date'],
            }
        json_data[code] = idx_json

    json_path = "/Users/claw/etf-trader/data/erp_strategy_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ JSON: {json_path}")

    return report_path, json_path


if __name__ == "__main__":
    results = run_research()
    if results:
        report = format_report(results)
        print("\n" + report)
        save_results(results, report)
    else:
        print("❌ 研究失败")
