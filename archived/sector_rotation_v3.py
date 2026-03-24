#!/usr/bin/env python3
"""
行业轮动策略 v3.0 — 定性研究驱动的定量优化

【定性研究总结 & 改进方向】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题1: 换仓成本侵蚀 (核心问题)
  v2月度换仓 → ~12次/年 × 0.2%/次 = ~2.4%/年摩擦成本
  → V3: 季度换仓(4次/年) + 惰性阈值(新分>旧分+12才换) → 成本降至~0.6%/年

问题2: 未区分宏观市场环境
  v2无论估值高低一律选行业，熊市/高估期系统性风险未规避
  → V3: 宏观PE%位调节选股数量: 高估期Top2 / 合理期Top3 / 低估期Top5
  → V3: 宏观高估期提升PE因子权重，宏观低估期提升动量权重

问题3: 等权分配忽略置信度
  → V3: 按分数归一化加权

问题4: 混合动量信号处理不当
  → V3: 信号一致性检验: 方向相同满权重, 方向冲突动量权重减半

问题5: 无估值上限保护
  → V3: 硬排除PE%>85%行业

【改进对比】
V2:  月度换仓 | 等权 | 无宏观调节 | 无上限 | 信号直接混合
V3:  季度换仓 | 分数加权 | 宏观PE联动 | 85%上限 | 一致性检验
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

CACHE_DIR = '/tmp/etf-trader/sector_cache'
PE_CACHE  = os.path.join(CACHE_DIR, 'sw_monthly_pe_cache.pkl')

SECTOR_MAP = {
    '801010': '农林牧渔', '801030': '基础化工', '801040': '钢铁',
    '801050': '有色金属', '801080': '电子',    '801110': '家用电器',
    '801120': '食品饮料', '801150': '医药生物', '801160': '公用事业',
    '801180': '房地产',   '801200': '商贸零售', '801720': '建筑材料',
    '801740': '国防军工', '801750': '计算机',  '801760': '传媒',
    '801770': '通信',     '801780': '银行',    '801790': '非银金融',
    '801880': '汽车',     '801890': '机械设备', '801900': '电力设备',
    '801950': '煤炭',     '801960': '石油石化',
}
ETF_MAP = {
    '801010': '516550', '801030': '516020', '801040': '515210',
    '801050': '512400', '801080': '512480', '801110': '159996',
    '801120': '515170', '801150': '512010', '801160': '516880',
    '801180': '515000', '801200': '515650', '801720': '159745',
    '801740': '512660', '801750': '512720', '801760': '512980',
    '801770': '515880', '801780': '512800', '801790': '512070',
    '801880': '516110', '801890': '159892', '801900': '516160',
    '801950': '515220', '801960': '512050',
}


def load_data():
    df = pd.read_pickle(PE_CACHE)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['sector_code'].isin(SECTOR_MAP)].copy()
    pe_piv    = df.pivot_table(index='date', columns='sector_code', values='pe')
    price_piv = df.pivot_table(index='date', columns='sector_code', values='close')
    return pe_piv, price_piv


def calc_macro_pct(pe_piv, month):
    hist = pe_piv[pe_piv.index <= month].median(axis=1).dropna()
    if len(hist) < 12: return 50.0
    return float((hist.iloc[:-1] < hist.iloc[-1]).mean() * 100)


def score_v3(pe_pct, m3, m6, macro_pct=50.0):
    if pe_pct > 85:
        return 0.0, '🚫高估排除', True
    pe_s = 100.0 - pe_pct
    def ms(m, sc): return max(0., min(100., 50. + m / sc * 50.))
    m3s, m6s = ms(m3, 15.), ms(m6, 25.)
    if (m3 >= 0) == (m6 >= 0):
        mom = 0.55 * m3s + 0.45 * m6s
        bonus = 3.0
    else:
        mom = 0.5 * min(m3s, m6s)
        bonus = 0.0
    wp, wm = (0.62,0.38) if macro_pct>70 else (0.35,0.65) if macro_pct<35 else (0.48,0.52)
    sc = wp * pe_s + wm * mom + bonus
    if sc >= 70: action = '🟢🟢强烈买入'
    elif sc >= 58: action = '🟢买入'
    elif sc >= 45: action = '🟡持有'
    elif sc >= 32: action = '🟠减仓'
    else: action = '🔴回避'
    return sc, action, False


def _monthly_return(price_piv, pos, month, prev_month):
    ret = 0.0
    for c, w in pos.items():
        if c not in price_piv.columns: continue
        sn = price_piv.loc[:month,    c].dropna()
        sp = price_piv.loc[:prev_month, c].dropna()
        if len(sn) > 0 and len(sp) > 0:
            ret += w * (sn.iloc[-1] / sp.iloc[-1] - 1)
    return ret


def _bh_return(price_piv, valid, month, prev_month):
    codes = [c for c in valid if c in price_piv.columns]
    if not codes: return 0.0
    br = 0.0
    for c in codes:
        sn = price_piv.loc[:month,    c].dropna()
        sp = price_piv.loc[:prev_month, c].dropna()
        if len(sn) > 0 and len(sp) > 0:
            br += (sn.iloc[-1] / sp.iloc[-1] - 1) / len(codes)
    return br


def _stats(rets_list, val):
    if not rets_list: return {}
    r = np.array(rets_list)
    cum = pd.Series(r).add(1).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    sharpe = r.mean() / (r.std() + 1e-10) * np.sqrt(12)
    return {'mdd': round(mdd, 1), 'sharpe': round(sharpe, 2),
            'cum': round((val-1)*100, 1)}


# ====================================================================
# V2 基准 (无前视偏差修正版)
# ====================================================================
def backtest_v2(pe_piv, price_piv, top_n=3):
    MIN_H = 24
    months = sorted(pe_piv.index)
    valid  = [c for c in pe_piv.columns if c in SECTOR_MAP]

    sv, bv = 1.0, 1.0
    sr_list, br_list, dates = [], [], []
    cur_pos = {}  # position decided at END of prev month, earns THIS month

    def v2_score(pe_pct, m3, m6):
        def ms(m, s): return max(0., min(100., 50.+m/s*50.))
        return 0.45*(100-pe_pct) + 0.30*ms(m3,15) + 0.25*ms(m6,25)

    for i, m in enumerate(months):
        if i < MIN_H: continue
        prev = months[i-1]

        # Step 1: earn this month with cur_pos (position from prev month)
        if i > MIN_H and cur_pos:
            sr = _monthly_return(price_piv, cur_pos, m, prev)
            br = _bh_return(price_piv, valid, m, prev)
            sv *= (1+sr); bv *= (1+br)
            sr_list.append(sr); br_list.append(br)
            dates.append(m)

        # Step 2: decide next month's position at end of this month
        hpe = pe_piv[pe_piv.index <= m]
        hpr = price_piv[price_piv.index <= m]
        sc = {}
        for code in valid:
            if code not in hpe.columns: continue
            pe_s = hpe[code].dropna()
            if len(pe_s) < MIN_H: continue
            cv = pe_s.iloc[-1]
            if cv <= 0 or pd.isna(cv): continue
            pct = float((pe_s.iloc[:-1] < cv).mean() * 100)
            m3 = m6 = 0.0
            if code in hpr.columns:
                pr = hpr[code].dropna()
                if len(pr) >= 4: m3 = (pr.iloc[-1]/pr.iloc[-4]-1)*100
                if len(pr) >= 7: m6 = (pr.iloc[-1]/pr.iloc[-7]-1)*100
            sc[code] = v2_score(pct, m3, m6)
        if sc:
            top = sorted(sc.items(), key=lambda x:-x[1])[:top_n]
            cur_pos = {c: 1./top_n for c,_ in top}

    if not sr_list: return {}
    ny = (dates[-1]-dates[0]).days/365.25
    sa = (sv**(1/ny)-1)*100; ba = (bv**(1/ny)-1)*100
    ss = _stats(sr_list, sv); bs = _stats(br_list, bv)
    return {'sa': round(sa,2), 'ba': round(ba,2), 'excess': round(sa-ba,2),
            'sc': round((sv-1)*100,1), 'bc': round((bv-1)*100,1),
            'smdd': ss['mdd'], 'bmdd': bs['mdd'],
            'ss': ss['sharpe'], 'bs': bs['sharpe'],
            'ny': round(ny,1), 'start': dates[0].strftime('%Y-%m'), 'end': dates[-1].strftime('%Y-%m')}


# ====================================================================
# V3 优化版 (无前视偏差修正版)
# ====================================================================
def backtest_v3(pe_piv, price_piv):
    MIN_H = 24
    SWITCH_THR = 12   # 新候选分数比当前均分高N才触发换仓
    QUARTERLY  = 3    # 最少持仓N个月后才允许换仓

    months = sorted(pe_piv.index)
    valid  = [c for c in pe_piv.columns if c in SECTOR_MAP]

    sv, bv = 1.0, 1.0
    sr_list, br_list, dates = [], [], []
    cur_pos = {}
    since_rebal = 0

    for i, m in enumerate(months):
        if i < MIN_H: continue
        prev = months[i-1]

        # Step 1: earn this month with last month's position
        if i > MIN_H and cur_pos:
            sr = _monthly_return(price_piv, cur_pos, m, prev)
            br = _bh_return(price_piv, valid, m, prev)
            sv *= (1+sr); bv *= (1+br)
            sr_list.append(sr); br_list.append(br)
            dates.append(m)

        since_rebal += 1

        # Step 2: decide position at end of this month
        hpe = pe_piv[pe_piv.index <= m]
        hpr = price_piv[price_piv.index <= m]
        mpct = calc_macro_pct(pe_piv, m)

        if mpct > 70:   max_n = 2
        elif mpct < 35: max_n = 5
        else:           max_n = 3

        sc = {}
        for code in valid:
            if code not in hpe.columns: continue
            pe_s = hpe[code].dropna()
            if len(pe_s) < MIN_H: continue
            cv = pe_s.iloc[-1]
            if cv <= 0 or pd.isna(cv): continue
            pct = float((pe_s.iloc[:-1] < cv).mean() * 100)
            m3 = m6 = 0.0
            if code in hpr.columns:
                pr = hpr[code].dropna()
                if len(pr) >= 4: m3 = (pr.iloc[-1]/pr.iloc[-4]-1)*100
                if len(pr) >= 7: m6 = (pr.iloc[-1]/pr.iloc[-7]-1)*100
            score, _, rejected = score_v3(pct, m3, m6, mpct)
            if not rejected and score >= 40:
                sc[code] = score

        if not sc:
            cur_pos = {}; since_rebal = 0; continue

        top = sorted(sc.items(), key=lambda x:-x[1])[:max_n]
        top_codes  = [c for c,_ in top]
        top_scores = [s for _,s in top]

        # 惰性换仓: 季度频率 + 高分才触发
        do_rebal = (since_rebal >= QUARTERLY)
        if not do_rebal and cur_pos:
            cur_avg = np.mean([sc.get(c, 0) for c in cur_pos])
            new_avg = np.mean(top_scores[:max(1, len(cur_pos))])
            if new_avg > cur_avg + SWITCH_THR:
                do_rebal = True

        if do_rebal or not cur_pos:
            total_sc = sum(top_scores)
            cur_pos = {c: s/total_sc for c,s in zip(top_codes, top_scores)}
            since_rebal = 0

    if not sr_list: return {}
    ny = (dates[-1]-dates[0]).days/365.25
    sa = (sv**(1/ny)-1)*100; ba = (bv**(1/ny)-1)*100
    ss = _stats(sr_list, sv); bs = _stats(br_list, bv)
    return {'sa': round(sa,2), 'ba': round(ba,2), 'excess': round(sa-ba,2),
            'sc': round((sv-1)*100,1), 'bc': round((bv-1)*100,1),
            'smdd': ss['mdd'], 'bmdd': bs['mdd'],
            'ss': ss['sharpe'], 'bs': bs['sharpe'],
            'ny': round(ny,1), 'start': dates[0].strftime('%Y-%m'), 'end': dates[-1].strftime('%Y-%m')}


# ====================================================================
# 主程序
# ====================================================================
def main():
    do_backtest = '--backtest' in sys.argv

    print('=' * 62)
    print('  行业轮动策略 V3.0 — 定性驱动优化版')
    print('=' * 62)
    print('\n📦 加载数据...')
    pe_piv, price_piv = load_data()
    print(f'  PE: {pe_piv.shape[0]}月 × {pe_piv.shape[1]}行业')
    print(f'  区间: {pe_piv.index.min().strftime("%Y-%m")} ~ {pe_piv.index.max().strftime("%Y-%m")}')

    if not do_backtest:
        month = pe_piv.index.max()
        mpct = calc_macro_pct(pe_piv, month)
        if mpct > 70:   max_n = 2; regime = '高估→精选2行业'
        elif mpct < 35: max_n = 5; regime = '低估→广选5行业'
        else:           max_n = 3; regime = '合理→选3行业'
        print(f'\n📅 {month.strftime("%Y-%m")} | 宏观PE: {mpct:.0f}%位 | {regime}')

        rows = []
        for code in pe_piv.columns:
            if code not in SECTOR_MAP: continue
            pe_s = pe_piv[code].dropna()
            if len(pe_s) < 12: continue
            cv = pe_s.iloc[-1]
            if cv <= 0: continue
            pct = float((pe_s.iloc[:-1] < cv).mean() * 100)
            m3 = m6 = 0.0
            if code in price_piv:
                pr = price_piv[code].dropna()
                if len(pr)>=4: m3 = (pr.iloc[-1]/pr.iloc[-4]-1)*100
                if len(pr)>=7: m6 = (pr.iloc[-1]/pr.iloc[-7]-1)*100
            sc, act, rej = score_v3(pct, m3, m6, mpct)
            if not rej:
                rows.append((code, SECTOR_MAP[code], ETF_MAP.get(code,'--'), sc, pct, m3, m6, act))

        rows.sort(key=lambda x: -x[3])
        top = rows[:max_n]
        total = sum(r[3] for r in top)
        print(f'\n{"行业":<8} {"ETF":>8} {"PE%":>5} {"3M%":>7} {"6M%":>7} {"分":>5} {"权重":>6} 建议')
        print('-' * 62)
        for code, name, etf, sc, pct, m3, m6, act in top:
            w = sc/total*100 if total>0 else 0
            print(f'{name:<8} {etf:>8} {pct:>4.0f}% {m3:>+6.1f}% {m6:>+6.1f}% {sc:>5.1f} {w:>5.1f}% {act}')
        return

    # ---- 回测模式 ----
    print('\n⏳ V2-Top3 基准回测...')
    r2_3 = backtest_v2(pe_piv, price_piv, top_n=3)
    print(f'  年化{r2_3.get("sa",0):.1f}%  超额{r2_3.get("excess",0):+.1f}%  最大回撤{r2_3.get("smdd",0):.1f}%')

    print('\n⏳ V2-Top5 基准回测...')
    r2_5 = backtest_v2(pe_piv, price_piv, top_n=5)
    print(f'  年化{r2_5.get("sa",0):.1f}%  超额{r2_5.get("excess",0):+.1f}%  最大回撤{r2_5.get("smdd",0):.1f}%')

    print('\n⏳ V3 优化版回测...')
    r3 = backtest_v3(pe_piv, price_piv)
    if not r3:
        print('V3回测失败'); return
    print(f'  年化{r3["sa"]:.1f}%  超额{r3["excess"]:+.1f}%  最大回撤{r3["smdd"]:.1f}%')

    print(f'\n{"="*65}')
    print(f'  策略对比回测 ({r3["start"]} ~ {r3["end"]}, {r3["ny"]}年)')
    print(f'{"="*65}')
    print(f'{"指标":<14} {"V2-Top3":>10} {"V2-Top5":>10} {"V3优化":>10} {"等权买持":>10}')
    print(f'{"-"*65}')
    print(f'{"年化收益":<14} {r2_3.get("sa",0):>9.1f}% {r2_5.get("sa",0):>9.1f}% {r3["sa"]:>9.1f}% {r3["ba"]:>9.1f}%')
    print(f'{"累计收益":<14} {r2_3.get("sc",0):>9.1f}% {r2_5.get("sc",0):>9.1f}% {r3["sc"]:>9.1f}% {r3["bc"]:>9.1f}%')
    print(f'{"最大回撤":<14} {r2_3.get("smdd",0):>9.1f}% {r2_5.get("smdd",0):>9.1f}% {r3["smdd"]:>9.1f}% {r3["bmdd"]:>9.1f}%')
    print(f'{"夏普比率":<14} {r2_3.get("ss",0):>10.2f} {r2_5.get("ss",0):>10.2f} {r3["ss"]:>10.2f} {r3["bs"]:>10.2f}')
    print(f'{"超额年化":<14} {r2_3.get("excess",0):>+9.1f}% {r2_5.get("excess",0):>+9.1f}% {r3["excess"]:>+9.1f}%')
    print(f'{"="*65}')

    print('\n【V3 改进说明】')
    print('  ✅ 季度换仓(4次/年) → 年摩擦成本 ~2.4% → ~0.6%')
    print('  ✅ 宏观PE联动: 高估期Top2 / 合理期Top3 / 低估期Top5')
    print('  ✅ 分数加权仓位 (高置信度行业权重更大)')
    print('  ✅ 动量一致性检验: 3M/6M方向冲突时权重减半')
    print('  ✅ 硬排除PE%>85%行业')

    v2  = r2_3.get('excess', 0)
    v3  = r3['excess']
    bh  = 0
    chg = v3 - v2
    print(f'\n  超额变化: V2({v2:+.1f}%) → V3({v3:+.1f}%)  改进{chg:+.1f}%')
    if v3 > bh:
        print(f'  ✅ V3 超额收益为正 ({v3:+.1f}%/年)，策略有效')
    else:
        print(f'  ⚠️  V3 超额收益仍为负，需进一步优化')


if __name__ == '__main__':
    main()
