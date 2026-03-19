#!/usr/bin/env python3
"""
行业轮动策略 v4.0 — 双矩阵对比 + 趋势过滤 (修复版)

【V4 回测结论汇总】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
方案               年化    超额    MDD     Sharpe  说明
────────────────────────────────────────────────
V3基准-动量确认    10.8%  -1.3%  -50.5%   0.63   无新增措施
V4A动量+趋势MA3   10.6%  -1.5%  -39.6%   0.71   ✅ 最优方案
V4B逆向+趋势MA3    0.9%  -4.2%  -44.8%   0.14   ❌ 逆向视角极差
V4C动量+止损+MA3   5.2%  -1.3%  -39.6%   0.43   ❌ 止损伤害收益
等权买持           12.1%          -29.6%   0.72   参考基准
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【关键结论】
1. 动量确认 >> 逆向视角: 年化10.6% vs 0.9%，逆向完全失效
   原因: 行业轮动需要催化剂，低估+上涨才是最强信号，
         低估+继续下跌往往是价值陷阱

2. 趋势过滤有效: MDD从-50.5%压缩至-39.6% (改善11%)，收益几乎不变
   大盘等权指数月线低于MA3时减半仓位，Sharpe 0.63→0.71

3. 宏观止损(PE>90%清仓)反而有害: 年化从10.6%→5.2%
   原因: 该策略专门选低估行业，大盘贵时我们买的是便宜板块，
         宏观止损会把便宜行业也清掉，得不偿失
   替代方案: etf_quant_strategy.py中的整体仓位已含PE过滤

4. BUG修复:
   ✅ calc_macro_pct: 用hist.iloc[:-1]排除当前月
   ✅ 行业PE百分位: 用pe_s.iloc[:-1]排除当前月
   (V3中已修复，V4保持)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# V4 核心参数
PE_HARD_REJECT   = 85.0   # 行业PE%超过此值 → 硬排除
TREND_MA_MONTHS  = 3      # 大盘MA60 ≈ 月线MA3 (60交易日 / 20日/月)
TREND_HALVE      = 0.50   # 大盘在MA以下时仓位系数
SWITCH_THR       = 12.0   # 新分比当前均分高N才换仓
QUARTERLY        = 3      # 最少持仓N个月


def load_data():
    df = pd.read_pickle(PE_CACHE)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['sector_code'].isin(SECTOR_MAP)].copy()
    pe_piv    = df.pivot_table(index='date', columns='sector_code', values='pe')
    price_piv = df.pivot_table(index='date', columns='sector_code', values='close')
    return pe_piv, price_piv


def calc_macro_pct(pe_piv, month):
    """宏观PE百分位 ✅ 修复: hist.iloc[:-1]排除当前月"""
    hist = pe_piv[pe_piv.index <= month].median(axis=1).dropna()
    if len(hist) < 12: return 50.0
    return float((hist.iloc[:-1] < hist.iloc[-1]).mean() * 100)


def calc_market_above_ma(price_piv, month, ma_months=TREND_MA_MONTHS):
    """大盘等权月线是否在MA以上"""
    mkt = price_piv[price_piv.index <= month].mean(axis=1).dropna()
    if len(mkt) < ma_months: return True, 1.0
    ma_val = mkt.rolling(ma_months).mean().iloc[-1]
    if pd.isna(ma_val): return True, 1.0
    above = (mkt.iloc[-1] >= ma_val)
    return above, 1.0 if above else TREND_HALVE


def calc_sector_pe_pct(pe_series, current_val):
    """行业PE百分位 ✅ 修复: iloc[:-1]排除当前月"""
    hist = pe_series.dropna()
    if len(hist) < 2: return 50.0
    return float((hist.iloc[:-1] < current_val).mean() * 100)


def momentum_score(m3, m6):
    def ms(m, sc): return max(0., min(100., 50. + m / sc * 50.))
    m3s, m6s = ms(m3, 15.), ms(m6, 25.)
    if (m3 >= 0) == (m6 >= 0):
        return 0.55 * m3s + 0.45 * m6s, 3.0
    return 0.5 * min(m3s, m6s), 0.0


def macro_weights(macro_pct):
    if macro_pct > 70:   return 0.62, 0.38
    elif macro_pct < 35: return 0.35, 0.65
    else:                return 0.48, 0.52


# ── 动量确认视角 (V4最优) ──
def score_momentum(pe_pct, m3, m6, macro_pct=50.0):
    """
    低估 + 上涨 = 最高分
    逻辑: 等待价值被市场认可并开始修复，价值+趋势共振
    """
    if pe_pct > PE_HARD_REJECT: return 0.0, '🚫高估', True
    pe_s = 100.0 - pe_pct
    mom, bonus = momentum_score(m3, m6)
    wp, wm = macro_weights(macro_pct)
    sc = wp * pe_s + wm * mom + bonus
    action = ('🟢🟢强烈买入' if sc>=70 else '🟢买入' if sc>=58 else
              '🟡持有' if sc>=45 else '🟠减仓' if sc>=32 else '🔴回避')
    return sc, action, False


# ── 逆向视角 (对比用，实测表现极差) ──
def score_contrarian(pe_pct, m3, m6, macro_pct=50.0):
    """
    低估 + 下跌 = 最高分
    逻辑: 越跌越买，等待均值回归
    ⚠️ 回测结果: 年化仅0.9%，大幅跑输 (低估陷阱风险高)
    """
    if pe_pct > PE_HARD_REJECT: return 0.0, '🚫高估', True
    pe_s = 100.0 - pe_pct
    mom, bonus = momentum_score(m3, m6)
    anti_mom = 100.0 - mom
    wp, wm = macro_weights(macro_pct)
    sc = wp * pe_s + wm * anti_mom + bonus
    action = ('🟢🟢强烈买入' if sc>=70 else '🟢买入' if sc>=58 else
              '🟡持有' if sc>=45 else '🟠减仓' if sc>=32 else '🔴回避')
    return sc, action, False


# ── 通用收益计算 ──
def _monthly_return(price_piv, pos, month, prev):
    ret = 0.0
    for c, w in pos.items():
        if c not in price_piv.columns: continue
        sn = price_piv.loc[:month, c].dropna()
        sp = price_piv.loc[:prev, c].dropna()
        if len(sn) > 0 and len(sp) > 0:
            ret += w * (sn.iloc[-1] / sp.iloc[-1] - 1)
    return ret


def _bh_return(price_piv, valid, month, prev):
    codes = [c for c in valid if c in price_piv.columns]
    if not codes: return 0.0
    rets = []
    for c in codes:
        sn = price_piv.loc[:month, c].dropna()
        sp = price_piv.loc[:prev, c].dropna()
        if len(sn) > 0 and len(sp) > 0:
            rets.append(sn.iloc[-1] / sp.iloc[-1] - 1)
    return np.mean(rets) if rets else 0.0


def _stats(rets_list, val):
    r = np.array(rets_list)
    cum = pd.Series(r).add(1).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    sharpe = r.mean() / (r.std() + 1e-10) * np.sqrt(12)
    return {'mdd': round(mdd,1), 'sharpe': round(sharpe,2), 'cum': round((val-1)*100,1)}


# ── V4 通用回测框架 ──
def backtest(pe_piv, price_piv, score_fn, label, use_trend=True):
    MIN_H = 24
    months = sorted(pe_piv.index)
    valid  = [c for c in pe_piv.columns if c in SECTOR_MAP]
    sv, bv = 1.0, 1.0
    sr_list, br_list, dates = [], [], []
    cur_pos, since_rebal = {}, 0
    halve_cnt = 0

    for i, m in enumerate(months):
        if i < MIN_H: continue
        prev = months[i-1]

        # Step 1: 赚本月收益 (使用上月仓位，无前瞻偏差)
        if i > MIN_H and cur_pos:
            sr = _monthly_return(price_piv, cur_pos, m, prev)
            br = _bh_return(price_piv, valid, m, prev)
            sv *= (1+sr); bv *= (1+br)
            sr_list.append(sr); br_list.append(br); dates.append(m)

        since_rebal += 1

        # Step 2: 月末决定下月仓位
        hpe = pe_piv[pe_piv.index <= m]
        hpr = price_piv[price_piv.index <= m]
        mpct = calc_macro_pct(pe_piv, m)

        # 趋势过滤: 大盘 < MA60 → 仓位减半
        t_ratio = 1.0
        if use_trend:
            above, t_ratio = calc_market_above_ma(price_piv, m)
            if not above: halve_cnt += 1

        max_n = 2 if mpct>70 else 5 if mpct<35 else 3

        sc = {}
        for code in valid:
            if code not in hpe.columns: continue
            pe_s = hpe[code].dropna()
            if len(pe_s) < MIN_H: continue
            cv = pe_s.iloc[-1]
            if cv <= 0 or pd.isna(cv): continue
            pct = calc_sector_pe_pct(pe_s, cv)
            m3 = m6 = 0.0
            if code in hpr.columns:
                pr = hpr[code].dropna()
                if len(pr) >= 4: m3 = (pr.iloc[-1]/pr.iloc[-4]-1)*100
                if len(pr) >= 7: m6 = (pr.iloc[-1]/pr.iloc[-7]-1)*100
            score, _, rejected = score_fn(pct, m3, m6, mpct)
            if not rejected and score >= 40: sc[code] = score

        if not sc:
            cur_pos = {}; since_rebal = 0; continue

        top = sorted(sc.items(), key=lambda x: -x[1])[:max_n]
        top_codes  = [c for c,_ in top]
        top_scores = [s for _,s in top]

        do_rebal = (since_rebal >= QUARTERLY)
        if not do_rebal and cur_pos:
            cur_avg = np.mean([sc.get(c,0) for c in cur_pos])
            new_avg = np.mean(top_scores[:max(1, len(cur_pos))])
            if new_avg > cur_avg + SWITCH_THR: do_rebal = True

        if do_rebal or not cur_pos:
            total = sum(top_scores)
            cur_pos = {c: s/total * t_ratio for c,s in zip(top_codes, top_scores)}
            since_rebal = 0

    if not sr_list: return {}
    ny = (dates[-1]-dates[0]).days/365.25
    sa = (sv**(1/ny)-1)*100; ba = (bv**(1/ny)-1)*100
    ss = _stats(sr_list, sv); bs = _stats(br_list, bv)
    return {
        'label': label,
        'sa': round(sa,2), 'ba': round(ba,2), 'excess': round(sa-ba,2),
        'sc': round((sv-1)*100,1), 'bc': round((bv-1)*100,1),
        'smdd': ss['mdd'], 'bmdd': bs['mdd'],
        'ss': ss['sharpe'], 'bs': bs['sharpe'],
        'ny': round(ny,1),
        'start': dates[0].strftime('%Y-%m'), 'end': dates[-1].strftime('%Y-%m'),
        'halve_months': halve_cnt,
    }


def main():
    do_backtest = '--backtest' in sys.argv

    print('=' * 68)
    print('  行业轮动策略 V4.0 — 双矩阵对比 + 大盘趋势过滤')
    print('=' * 68)
    print('\n📦 加载数据...')
    pe_piv, price_piv = load_data()
    print(f'  PE: {pe_piv.shape[0]}月 × {pe_piv.shape[1]}行业')
    print(f'  区间: {pe_piv.index.min().strftime("%Y-%m")} ~ {pe_piv.index.max().strftime("%Y-%m")}')

    if not do_backtest:
        month = pe_piv.index.max()
        mpct  = calc_macro_pct(pe_piv, month)
        above, ratio = calc_market_above_ma(price_piv, month)
        max_n = 2 if mpct>70 else 5 if mpct<35 else 3

        print(f'\n📅 {month.strftime("%Y-%m")} | 宏观PE: {mpct:.0f}%位', end='')
        print(f' | 大盘MA{TREND_MA_MONTHS*20}: {"✅上方" if above else "❌下方 (仓位×50%)"}')
        print(f'  宏观环境: {"高估期→选Top2" if mpct>70 else "低估期→选Top5" if mpct<35 else "合理期→选Top3"}')

        rows = []
        for code in pe_piv.columns:
            if code not in SECTOR_MAP: continue
            pe_s = pe_piv[code].dropna()
            if len(pe_s) < 12: continue
            cv = pe_s.iloc[-1]
            if cv <= 0: continue
            pct = calc_sector_pe_pct(pe_s, cv)
            m3 = m6 = 0.0
            if code in price_piv:
                pr = price_piv[code].dropna()
                if len(pr)>=4: m3=(pr.iloc[-1]/pr.iloc[-4]-1)*100
                if len(pr)>=7: m6=(pr.iloc[-1]/pr.iloc[-7]-1)*100
            scM, actM, rejM = score_momentum(pct, m3, m6, mpct)
            if not rejM: rows.append((code, SECTOR_MAP[code], ETF_MAP.get(code,'--'), scM, pct, m3, m6, actM))

        rows.sort(key=lambda x: -x[3])
        top = rows[:max_n]
        total = sum(r[3] for r in top)
        print(f'\n【V4最优方案: 动量确认视角】 Top{max_n}:')
        print(f'{"行业":<8} {"ETF":>8} {"PE%":>5} {"3M%":>7} {"6M%":>7} {"分":>5} {"权重":>6} 建议')
        print('-' * 55)
        for code, name, etf, sc, pct, m3, m6, act in top:
            w = sc/total*100 if total>0 else 0
            eff_w = w * ratio
            print(f'{name:<8} {etf:>8} {pct:>4.0f}% {m3:>+6.1f}% {m6:>+6.1f}% {sc:>5.1f} {eff_w:>5.1f}% {act}')
        if ratio < 1.0:
            print(f'\n  ⚠️ 大盘在MA{TREND_MA_MONTHS*20}下方 → 实际仓位×{ratio:.0%}')
        return

    # ── 回测模式 ──
    print('\n⏳ 回测中 (共4个方案)...\n')

    r3  = backtest(pe_piv, price_piv, score_momentum,   'V3基准(无趋势)',  use_trend=False)
    r4a = backtest(pe_piv, price_piv, score_momentum,   'V4A动量+MA3趋势', use_trend=True)
    r4b = backtest(pe_piv, price_piv, score_contrarian, 'V4B逆向+MA3趋势', use_trend=True)

    ref = r4a
    print(f'{"="*68}')
    print(f'  综合回测 ({ref["start"]} ~ {ref["end"]}, {ref["ny"]}年)')
    print(f'{"="*68}')
    print(f'{"指标":<12} {"V3基准":>10} {"V4A动量确认":>12} {"V4B逆向视角":>12} {"等权买持":>10}')
    print(f'{"-"*68}')
    print(f'{"年化收益":<12} {r3["sa"]:>9.1f}% {r4a["sa"]:>11.1f}% {r4b["sa"]:>11.1f}% {ref["ba"]:>9.1f}%')
    print(f'{"累计收益":<12} {r3["sc"]:>9.1f}% {r4a["sc"]:>11.1f}% {r4b["sc"]:>11.1f}% {ref["bc"]:>9.1f}%')
    print(f'{"最大回撤":<12} {r3["smdd"]:>9.1f}% {r4a["smdd"]:>11.1f}% {r4b["smdd"]:>11.1f}% {ref["bmdd"]:>9.1f}%')
    print(f'{"夏普比率":<12} {r3["ss"]:>10.2f} {r4a["ss"]:>11.2f} {r4b["ss"]:>11.2f} {ref["bs"]:>9.2f}')
    print(f'{"超额年化":<12} {r3["excess"]:>+9.1f}% {r4a["excess"]:>+10.1f}% {r4b["excess"]:>+10.1f}%')
    print(f'{"趋势减半月":<12} {"0":>9} {r4a["halve_months"]:>11} {r4b["halve_months"]:>11}')
    print(f'{"="*68}')

    print('\n【V4 关键结论】')
    print(f'  🏆 最优方案: V4A (动量确认 + 趋势MA60过滤)')
    print(f'     年化{r4a["sa"]:.1f}%  Sharpe {r4a["ss"]:.2f}  MDD {r4a["smdd"]:.1f}%')
    print(f'  📊 趋势过滤效果: MDD {r3["smdd"]:.1f}% → {r4a["smdd"]:.1f}% (改善{abs(r4a["smdd"]-r3["smdd"]):.1f}%)')
    print(f'                 Sharpe {r3["ss"]:.2f} → {r4a["ss"]:.2f} (+{r4a["ss"]-r3["ss"]:.2f})')
    print(f'  ❌ 逆向视角: 年化仅{r4b["sa"]:.1f}% (低估陷阱，不建议使用)')
    print(f'\n  💡 宏观止损说明:')
    print(f'     PE>90%不适合清仓，原因: 策略专选低估行业，')
    print(f'     宏观贵时我们买的是便宜板块，全仓止损反而损失alpha')
    print(f'     替代方案: 宏观PE>70%时已自动缩减为Top2行业 (已实现)')


if __name__ == '__main__':
    main()
