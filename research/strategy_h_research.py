#!/usr/bin/env python3
"""
指数买入策略全面回测 V5 - 全参数覆盖
- 累计交易日: 1-10 (全覆盖)
- 持有天数: 1-50 (全覆盖)
- 止损: None, -3%, -5%, -7%
- 超跌 + 追涨
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ========== 配置 ==========
INDICES = {
    '沪深300': '000300.SS',
    '中证500ETF': '510500.SS',
    '上证50ETF': '510050.SS',
    '创业板ETF': '159915.SZ',
    '科创50ETF': '588000.SS',
    '恒生指数': '^HSI',
    '国企指数': '^HSCE',
    'H股ETF': '510900.SS',
}

CUM_DAYS_LIST = list(range(1, 11))  # 1-10 全覆盖
THRESHOLD_PCT_LIST = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
HOLD_DAYS_LIST = list(range(1, 51))  # 1-50 全覆盖
STOP_LOSS_LIST = [None, -3, -5, -7]  # 精简止损(V4发现-2%和-10%效果一般)

START_DATE = '2015-01-01'
END_DATE = '2026-03-24'
RISK_FREE_RATE = 0.02
MIN_TRADES = 3

# ========== 数据下载 ==========
def download_data():
    data = {}
    for name, ticker in INDICES.items():
        print(f"下载 {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if df is not None and len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[['Close']].dropna()
                df.columns = ['close']
                data[name] = df
                print(f"  -> {len(df)} 交易日")
            else:
                print(f"  -> 数据不足，跳过")
        except Exception as e:
            print(f"  -> 下载失败: {e}")
    return data

# ========== 回测引擎(优化版) ==========
def backtest_strategy(closes, dates, n, cum_returns, cum_days, threshold_pct, hold_days, direction='dip', stop_loss_pct=None):
    if n < cum_days + hold_days + 10:
        return None, []

    nav = np.ones(n)
    position = False
    entry_idx = 0
    entry_price = 0
    hold_count = 0
    trades = []

    for i in range(1, n):
        if position:
            daily_ret = closes[i] / closes[i-1] - 1
            nav[i] = nav[i-1] * (1 + daily_ret)
            hold_count += 1

            current_trade_ret = closes[i] / entry_price - 1
            hit_stop = False
            if stop_loss_pct is not None and current_trade_ret * 100 <= stop_loss_pct:
                hit_stop = True

            if hold_count >= hold_days or hit_stop:
                trades.append({
                    'return': closes[i] / entry_price - 1,
                    'hold_days': hold_count,
                    'stop_loss': hit_stop
                })
                position = False
        else:
            nav[i] = nav[i-1]
            cr = cum_returns[i]
            if not np.isnan(cr):
                trigger = False
                if direction == 'dip' and cr <= -threshold_pct:
                    trigger = True
                elif direction == 'rally' and cr >= threshold_pct:
                    trigger = True
                if trigger:
                    position = True
                    entry_idx = i
                    entry_price = closes[i]
                    hold_count = 0

    if position:
        trades.append({
            'return': closes[-1] / entry_price - 1,
            'hold_days': n - 1 - entry_idx,
            'stop_loss': False
        })

    return nav, trades

def calc_metrics(nav, trades, n_days):
    if nav is None or n_days < 2:
        return None

    total_return = nav[-1] / nav[0] - 1
    n_years = n_days / 252
    if n_years <= 0:
        return None
    annualized_return = (nav[-1] / nav[0]) ** (1 / n_years) - 1

    cummax = np.maximum.accumulate(nav)
    drawdown = (nav - cummax) / cummax
    max_drawdown = np.min(drawdown)

    # Daily returns for Sharpe
    daily_ret = np.diff(nav) / nav[:-1]
    std = np.std(daily_ret)
    if std > 0:
        sharpe = (np.mean(daily_ret) - RISK_FREE_RATE / 252) / std * np.sqrt(252)
    else:
        sharpe = 0

    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    n_trades = len(trades)
    if n_trades > 0:
        trade_returns = [t['return'] for t in trades]
        win_rate = sum(1 for r in trade_returns if r > 0) / n_trades
        avg_trade_return = np.mean(trade_returns)
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        stop_loss_trades = sum(1 for t in trades if t.get('stop_loss'))
        stop_loss_rate = stop_loss_trades / n_trades
    else:
        win_rate = avg_trade_return = profit_loss_ratio = stop_loss_rate = 0

    total_hold = sum(t['hold_days'] for t in trades)
    exposure = total_hold / n_days if n_days > 0 else 0

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_trade_return': avg_trade_return,
        'profit_loss_ratio': profit_loss_ratio,
        'stop_loss_rate': stop_loss_rate,
        'exposure': exposure,
    }

# ========== 主程序 ==========
def main():
    print("=" * 90)
    print("指数买入策略全面回测 V5 - 全参数覆盖")
    print(f"累计天数: {CUM_DAYS_LIST[0]}-{CUM_DAYS_LIST[-1]} ({len(CUM_DAYS_LIST)}个)")
    print(f"阈值: {THRESHOLD_PCT_LIST} ({len(THRESHOLD_PCT_LIST)}个)")
    print(f"持有天数: {HOLD_DAYS_LIST[0]}-{HOLD_DAYS_LIST[-1]} ({len(HOLD_DAYS_LIST)}个)")
    print(f"止损: {STOP_LOSS_LIST} ({len(STOP_LOSS_LIST)}个)")
    combos = len(CUM_DAYS_LIST) * len(THRESHOLD_PCT_LIST) * len(HOLD_DAYS_LIST) * len(STOP_LOSS_LIST) * 2
    print(f"每指数组合: {combos}, 总组合(×{len(INDICES)}指数): {combos * len(INDICES)}")
    print("=" * 90)

    data = download_data()
    print(f"\n成功下载 {len(data)} 个指数数据\n")
    if not data:
        return

    # 预计算所有指数的cum_returns
    precomputed = {}
    for idx_name, prices_df in data.items():
        closes = prices_df['close'].values
        dates = prices_df.index
        n = len(closes)
        cum_rets = {}
        for cd in CUM_DAYS_LIST:
            cr = np.full(n, np.nan)
            for i in range(cd, n):
                cr[i] = (closes[i] / closes[i - cd] - 1) * 100
            cum_rets[cd] = cr
        precomputed[idx_name] = {
            'closes': closes,
            'dates': dates,
            'n': n,
            'cum_returns': cum_rets
        }

    all_results = []
    count = 0
    total = combos * len(data)

    for direction_tag, direction in [('超跌买入', 'dip'), ('追涨买入', 'rally')]:
        print(f"\n=== 测试{direction_tag}策略 ===")
        for cum_days in CUM_DAYS_LIST:
            for threshold in THRESHOLD_PCT_LIST:
                for hold_days in HOLD_DAYS_LIST:
                    for stop_loss in STOP_LOSS_LIST:
                        for idx_name in data.keys():
                            count += 1
                            if count % 10000 == 0:
                                print(f"  进度: {count}/{total} ({count*100//total}%)...")

                            pc = precomputed[idx_name]
                            nav, trades = backtest_strategy(
                                pc['closes'], pc['dates'], pc['n'],
                                pc['cum_returns'][cum_days],
                                cum_days, threshold, hold_days,
                                direction, stop_loss
                            )
                            metrics = calc_metrics(nav, trades, pc['n'])
                            if metrics and metrics['n_trades'] >= MIN_TRADES:
                                sl_str = f"止损{stop_loss}%" if stop_loss is not None else "不止损"
                                if direction == 'dip':
                                    strat_str = f"{cum_days}日跌>{threshold}%→持{hold_days}日({sl_str})"
                                else:
                                    strat_str = f"{cum_days}日涨>{threshold}%→持{hold_days}日({sl_str})"

                                result = {
                                    'direction': direction_tag,
                                    'index': idx_name,
                                    'cum_days': cum_days,
                                    'threshold_pct': threshold,
                                    'hold_days': hold_days,
                                    'stop_loss_pct': stop_loss if stop_loss is not None else 0,
                                    'has_stop_loss': stop_loss is not None,
                                    'strategy': strat_str,
                                }
                                result.update(metrics)
                                all_results.append(result)

    df = pd.DataFrame(all_results)
    print(f"\n共产生 {len(df)} 条有效结果")
    print(f"  超跌: {len(df[df['direction']=='超跌买入'])} | 追涨: {len(df[df['direction']=='追涨买入'])}")
    print(f"  有止损: {len(df[df['has_stop_loss']==True])} | 无止损: {len(df[df['has_stop_loss']==False])}")

    if len(df) == 0:
        return

    # ============================================================
    # 报告
    # ============================================================

    def print_top(title, subset, sort_col, n=15):
        print(f"\n{'='*90}")
        print(f"【{title}】")
        print(f"{'='*90}")
        top = subset.nlargest(n, sort_col)
        for _, row in top.iterrows():
            d = "↓" if row['direction'] == '超跌买入' else "↑"
            sl = f"SL{row['stop_loss_pct']}%" if row['has_stop_loss'] else "无SL"
            print(f"  [{d}][{sl}] {row['index']} | {row['strategy']} | "
                  f"年化:{row['annualized_return']*100:.1f}% | "
                  f"回撤:{row['max_drawdown']*100:.1f}% | "
                  f"夏普:{row['sharpe_ratio']:.2f} | "
                  f"Calmar:{row['calmar_ratio']:.2f} | "
                  f"胜率:{row['win_rate']*100:.0f}% | "
                  f"{row['n_trades']}笔 | "
                  f"盈亏比:{row['profit_loss_ratio']:.2f}")

    # 全局TOP - 按年化(过滤交易次数>=5)
    freq_df = df[df['n_trades'] >= 5]
    print_top("全局 TOP 15 - 按年化收益 (>=5笔交易)", freq_df, 'annualized_return', 15)
    print_top("全局 TOP 15 - 按夏普比率 (>=5笔交易)", freq_df, 'sharpe_ratio', 15)

    # 过滤>=10笔交易的更可靠策略
    reliable_df = df[df['n_trades'] >= 10]
    print_top("可靠策略 TOP 15 - 按年化收益 (>=10笔交易)", reliable_df, 'annualized_return', 15)
    print_top("可靠策略 TOP 15 - 按夏普比率 (>=10笔交易)", reliable_df, 'sharpe_ratio', 15)

    # 分方向
    for dir_tag in ['超跌买入', '追涨买入']:
        dir_df = reliable_df[reliable_df['direction'] == dir_tag]
        tag = "超跌" if dir_tag == '超跌买入' else "追涨"
        print_top(f"{tag} TOP 15 - 按年化收益 (>=10笔)", dir_df, 'annualized_return', 15)
        print_top(f"{tag} TOP 15 - 按夏普 (>=10笔)", dir_df, 'sharpe_ratio', 15)

    # --- 止损 vs 不止损 配对分析 ---
    print(f"\n{'='*90}")
    print("【止损 vs 不止损 配对分析】")
    print(f"{'='*90}")

    no_sl = df[df['has_stop_loss'] == False]
    has_sl = df[df['has_stop_loss'] == True]
    merge_keys = ['direction', 'index', 'cum_days', 'threshold_pct', 'hold_days']

    no_sl_base = no_sl[merge_keys + ['annualized_return', 'max_drawdown', 'sharpe_ratio', 'calmar_ratio']].copy()
    no_sl_base = no_sl_base.rename(columns={
        'annualized_return': 'base_annual',
        'max_drawdown': 'base_dd',
        'sharpe_ratio': 'base_sharpe',
        'calmar_ratio': 'base_calmar'
    })

    for sl_pct in [-3, -5, -7]:
        sl_sub = has_sl[has_sl['stop_loss_pct'] == sl_pct]
        merged = sl_sub.merge(no_sl_base, on=merge_keys, how='inner')
        if len(merged) == 0:
            continue

        annual_imp = (merged['annualized_return'] > merged['base_annual']).mean()
        dd_imp = (merged['max_drawdown'] > merged['base_dd']).mean()
        sharpe_imp = (merged['sharpe_ratio'] > merged['base_sharpe']).mean()
        calmar_imp = (merged['calmar_ratio'] > merged['base_calmar']).mean()
        avg_dd_change = (merged['max_drawdown'] - merged['base_dd']).mean() * 100

        print(f"\n  止损{sl_pct}% vs 不止损 ({len(merged)}对):")
        print(f"    年化提升: {annual_imp*100:.1f}%策略 | 回撤改善: {dd_imp*100:.1f}%策略 (平均{avg_dd_change:+.2f}%)")
        print(f"    夏普提升: {sharpe_imp*100:.1f}%策略 | Calmar提升: {calmar_imp*100:.1f}%策略")

    # 止损最佳TOP
    merged_all = has_sl.merge(no_sl_base, on=merge_keys, how='inner')
    if len(merged_all) > 0:
        merged_all['calmar_imp'] = merged_all['calmar_ratio'] - merged_all['base_calmar']
        merged_all['sharpe_imp'] = merged_all['sharpe_ratio'] - merged_all['base_sharpe']

        print(f"\n  --- 止损Calmar提升最大 TOP 10 ---")
        top_imp = merged_all[merged_all['n_trades'] >= 5].nlargest(10, 'calmar_imp')
        for _, row in top_imp.iterrows():
            print(f"  {row['index']} | {row['strategy']}")
            print(f"    Calmar: {row['base_calmar']:.2f}->{row['calmar_ratio']:.2f}(+{row['calmar_imp']:.2f}) | "
                  f"回撤: {row['base_dd']*100:.1f}%->{row['max_drawdown']*100:.1f}% | "
                  f"年化: {row['base_annual']*100:.1f}%->{row['annualized_return']*100:.1f}%")

    # --- 各指数最佳 ---
    print(f"\n{'='*90}")
    print("【各指数最佳策略(>=5笔交易, 按夏普)】")
    print(f"{'='*90}")
    for idx_name in data.keys():
        idx_df = freq_df[freq_df['index'] == idx_name]
        if len(idx_df) == 0:
            continue
        print(f"\n  === {idx_name} ===")
        for dir_tag in ['超跌买入', '追涨买入']:
            sub = idx_df[idx_df['direction'] == dir_tag]
            if len(sub) == 0:
                continue
            tag = "↓" if dir_tag == '超跌买入' else "↑"

            best_sharpe = sub.nlargest(1, 'sharpe_ratio').iloc[0]
            best_annual = sub.nlargest(1, 'annualized_return').iloc[0]
            sl = f"SL{best_sharpe['stop_loss_pct']}%" if best_sharpe['has_stop_loss'] else "无SL"
            print(f"  {tag} 最高夏普[{sl}]: {best_sharpe['strategy']}")
            print(f"     夏普:{best_sharpe['sharpe_ratio']:.2f} | 年化:{best_sharpe['annualized_return']*100:.1f}% | "
                  f"回撤:{best_sharpe['max_drawdown']*100:.1f}% | 胜率:{best_sharpe['win_rate']*100:.0f}% | {best_sharpe['n_trades']}笔")

            if best_annual.name != best_sharpe.name:
                sl2 = f"SL{best_annual['stop_loss_pct']}%" if best_annual['has_stop_loss'] else "无SL"
                print(f"  {tag} 最高年化[{sl2}]: {best_annual['strategy']}")
                print(f"     年化:{best_annual['annualized_return']*100:.1f}% | 夏普:{best_annual['sharpe_ratio']:.2f} | "
                      f"回撤:{best_annual['max_drawdown']*100:.1f}% | 胜率:{best_annual['win_rate']*100:.0f}% | {best_annual['n_trades']}笔")

    # --- 跨指数通用 ---
    print(f"\n{'='*90}")
    print("【跨指数通用策略(>=4指数, >=5笔交易) - 按均夏普】")
    print(f"{'='*90}")

    for sl_label, subset in [('不止损', freq_df[freq_df['has_stop_loss']==False]),
                              ('有止损', freq_df[freq_df['has_stop_loss']==True])]:
        if len(subset) == 0:
            continue

        gk = ['direction', 'cum_days', 'threshold_pct', 'hold_days', 'strategy']
        if sl_label == '有止损':
            gk.append('stop_loss_pct')

        grouped = subset.groupby(gk).agg({
            'annualized_return': 'mean',
            'max_drawdown': 'mean',
            'sharpe_ratio': 'mean',
            'calmar_ratio': 'mean',
            'win_rate': 'mean',
            'n_trades': 'mean',
            'index': 'count'
        }).rename(columns={'index': 'n_indices'})
        grouped = grouped[grouped['n_indices'] >= 4].reset_index()

        if len(grouped) == 0:
            continue

        print(f"\n  --- {sl_label} TOP 10 (按均夏普) ---")
        for _, row in grouped.nlargest(10, 'sharpe_ratio').iterrows():
            print(f"  {row['strategy']} | "
                  f"覆盖{int(row['n_indices'])}指数 | "
                  f"均夏普:{row['sharpe_ratio']:.2f} | "
                  f"均年化:{row['annualized_return']*100:.1f}% | "
                  f"均回撤:{row['max_drawdown']*100:.1f}% | "
                  f"均Calmar:{row['calmar_ratio']:.2f} | "
                  f"均胜率:{row['win_rate']*100:.0f}%")

    # --- 持有天数敏感性分析 ---
    print(f"\n{'='*90}")
    print("【持有天数敏感性分析 - 按持有天数区间的平均表现】")
    print(f"{'='*90}")
    for dir_tag in ['超跌买入', '追涨买入']:
        dir_df = df[df['direction'] == dir_tag]
        tag = "超跌" if dir_tag == '超跌买入' else "追涨"
        print(f"\n  --- {tag} ---")
        print(f"  {'持有天数':<12} {'平均年化':>10} {'平均回撤':>10} {'平均夏普':>10} {'平均胜率':>10} {'策略数':>8}")
        for hd_range, hd_min, hd_max in [('1-5日', 1, 5), ('6-10日', 6, 10), ('11-20日', 11, 20),
                                           ('21-30日', 21, 30), ('31-40日', 31, 40), ('41-50日', 41, 50)]:
            sub = dir_df[(dir_df['hold_days'] >= hd_min) & (dir_df['hold_days'] <= hd_max)]
            if len(sub) > 0:
                print(f"  {hd_range:<12} {sub['annualized_return'].mean()*100:>9.2f}% "
                      f"{sub['max_drawdown'].mean()*100:>9.2f}% "
                      f"{sub['sharpe_ratio'].mean():>10.3f} "
                      f"{sub['win_rate'].mean()*100:>9.1f}% "
                      f"{len(sub):>8}")

    # --- 累计天数敏感性分析 ---
    print(f"\n{'='*90}")
    print("【累计天数敏感性分析】")
    print(f"{'='*90}")
    for dir_tag in ['超跌买入', '追涨买入']:
        dir_df = df[df['direction'] == dir_tag]
        tag = "超跌" if dir_tag == '超跌买入' else "追涨"
        print(f"\n  --- {tag} ---")
        print(f"  {'累计天数':<10} {'平均年化':>10} {'平均回撤':>10} {'平均夏普':>10} {'平均胜率':>10} {'策略数':>8}")
        for cd in CUM_DAYS_LIST:
            sub = dir_df[dir_df['cum_days'] == cd]
            if len(sub) > 0:
                print(f"  {cd:<10} {sub['annualized_return'].mean()*100:>9.2f}% "
                      f"{sub['max_drawdown'].mean()*100:>9.2f}% "
                      f"{sub['sharpe_ratio'].mean():>10.3f} "
                      f"{sub['win_rate'].mean()*100:>9.1f}% "
                      f"{len(sub):>8}")

    # --- 买入持有基准 ---
    print(f"\n{'='*90}")
    print("【买入持有基准】")
    print(f"{'='*90}")
    for idx_name, prices_df in data.items():
        closes = prices_df['close']
        bh_total = closes.iloc[-1] / closes.iloc[0] - 1
        n_years = len(closes) / 252
        bh_annual = (1 + bh_total) ** (1/n_years) - 1
        bh_cummax = closes.cummax()
        bh_dd = ((closes - bh_cummax) / bh_cummax).min()
        bh_daily = closes.pct_change().dropna()
        bh_sharpe = (bh_daily.mean() - RISK_FREE_RATE/252) / bh_daily.std() * np.sqrt(252) if bh_daily.std() > 0 else 0
        print(f"  {idx_name}: 年化 {bh_annual*100:.1f}% | 最大回撤 {bh_dd*100:.1f}% | 夏普 {bh_sharpe:.2f}")

    # 保存
    output_path = '/tmp/dip_rally_backtest_v5.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n完整结果已保存至: {output_path}")
    print(f"共 {len(df)} 条记录")

if __name__ == '__main__':
    main()
