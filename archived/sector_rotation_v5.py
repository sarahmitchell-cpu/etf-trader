#!/usr/bin/env python3
"""
行业轮动策略 v5.0 — 行业级别趋势过滤 + 单行业仓位上限

在 V4A (动量确认 + 大盘MA3趋势) 基础上新增两项回撤控制:
  1. 行业级别趋势过滤: 每个候选行业须在自身MA3以上才可持有
  2. 单行业仓位上限: 每个行业最多持有25%仓位 (防过度集中)

【V5 回测结论汇总】 (运行 --backtest 后更新)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
方案                  年化    MDD     Sharpe  说明
────────────────────────────────────────────────────
V4A (基准)           10.6%  -39.6%   0.71   大盘MA3
V5A (+行业MA3)         ?%    ?%       ?     行业+大盘双重过滤
V5B (+仓位25%上限)     ?%    ?%       ?     防过度集中
V5C (+双重改进)        ?%    ?%       ?     二者叠加
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# 参数
PE_HARD_REJECT   = 85.0   # 行业PE%超过此值 → 硬排除
TREND_MA_MONTHS  = 3      # 大盘/行业 MA月数 (≈MA60交易日)
TREND_HALVE      = 0.50   # 大盘MA以下时整体仓位系数
SECTOR_MA_HALVE  = 0.0    # 行业MA以下时该行业权重 (0=完全排除, 0.5=减半)
MAX_SECTOR_WEIGHT = 0.25  # 单行业最大权重 (仓位上限)
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
    hist = pe_piv[pe_piv.index <= month].median(axis=1).dropna()
    if len(hist) < 12: return 50.0
    return float((hist.iloc[:-1] < hist.iloc[-1]).mean() * 100)


def calc_market_above_ma(price_piv, month, ma_months=TREND_MA_MONTHS):
    """大盘等权月线是否在MA以上"""
    mkt = price_piv[price_piv.index <= month].mean(axis=1).dropna()
    if len(mkt) < ma_months: return True, 1.0
    ma_val = mkt.rolling(ma_months).mean().iloc[-1]
    if pd.isna(ma_val): return True, 1.0
    above = bool(mkt.iloc[-1] >= ma_val)
    return above, 1.0 if above else TREND_HALVE


def calc_sector_above_ma(price_piv, code, month, ma_months=TREND_MA_MONTHS):
    """判断单个行业价格是否在自身MA以上"""
    if code not in price_piv.columns: return True
    pr = price_piv.loc[price_piv.index <= month, code].dropna()
    if len(pr) < ma_months: return True
    ma_val = pr.rolling(ma_months).mean().iloc[-1]
    if pd.isna(ma_val): return True
    return bool(pr.iloc[-1] >= ma_val)


def calc_sector_pe_pct(pe_series, current_val):
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


def score_momentum(pe_pct, m3, m6, macro_pct=50.0):
    if pe_pct > PE_HARD_REJECT: return 0.0, '🚫高估', True
    pe_s = 100.0 - pe_pct
    mom, bonus = momentum_score(m3, m6)
    wp, wm = macro_weights(macro_pct)
    sc = wp * pe_s + wm * mom + bonus
    action = ('🟢🟢强烈买入' if sc>=70 else '🟢买入' if sc>=58 else
              '🟡持有' if sc>=45 else '🟠减仓' if sc>=32 else '🔴回避')
    return sc, action, False


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
    return {'mdd': round(mdd,1), 'sharpe': round(sharpe,2)}


def apply_weight_cap(scores_dict, cap=MAX_SECTOR_WEIGHT):
    """
    对评分比例分配应用单资产上限, 迭代重分配直到所有权重符合上限。
    返回最终权重字典 (总和 <= 1.0)
    """
    total = sum(scores_dict.values())
    if total <= 0: return {}
    weights = {c: s/total for c, s in scores_dict.items()}
    for _ in range(20):  # 最多20轮迭代
        capped = {c: min(w, cap) for c, w in weights.items()}
        capped_total = sum(capped.values())
        if capped_total <= 0: break
        excess = sum(max(0, w - cap) for w in weights.values())
        if excess < 1e-8: break
        # 将超额权重均匀分配给未到上限的行业
        uncapped = {c for c, w in weights.items() if w < cap - 1e-8}
        if not uncapped: break
        uncapped_total = sum(capped[c] for c in uncapped)
        if uncapped_total <= 0: break
        for c in uncapped:
            capped[c] += excess * (capped[c] / uncapped_total)
        weights = capped
    # 归一化
    final_total = sum(weights.values())
    return {c: w/final_total for c, w in weights.items()} if final_total > 0 else {}


def backtest(pe_piv, price_piv, use_sector_trend=False, use_weight_cap=False, label=''):
    MIN_H = 24
    months = sorted(pe_piv.index)
    valid  = [c for c in pe_piv.columns if c in SECTOR_MAP]
    sv, bv = 1.0, 1.0
    sr_list, br_list, dates = [], [], []
    cur_pos, since_rebal = {}, 0
    halve_cnt = 0
    sector_filtered_cnt = 0  # 被行业趋势过滤掉的次数

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
        mpct = calc_macro_pct(pe_piv, m)

        # 大盘趋势过滤 (V4A已有)
        above, t_ratio = calc_market_above_ma(price_piv, m)
        if not above: halve_cnt += 1

        max_n = 2 if mpct>70 else 5 if mpct<35 else 3

        sc = {}
        for code in valid:
            pe_s = pe_piv[pe_piv.index <= m].get(code)
            if pe_s is None: continue
            pe_s = pe_s.dropna()
            if len(pe_s) < MIN_H: continue
            cv = pe_s.iloc[-1]
            if cv <= 0 or pd.isna(cv): continue

            # V5新增: 行业级别趋势过滤
            if use_sector_trend:
                if not calc_sector_above_ma(price_piv, code, m):
                    sector_filtered_cnt += 1
                    continue

            pct = calc_sector_pe_pct(pe_s, cv)
            m3 = m6 = 0.0
            pr_all = price_piv[price_piv.index <= m]
            if code in pr_all.columns:
                pr = pr_all[code].dropna()
                if len(pr) >= 4: m3 = (pr.iloc[-1]/pr.iloc[-4]-1)*100
                if len(pr) >= 7: m6 = (pr.iloc[-1]/pr.iloc[-7]-1)*100
            score, _, rejected = score_momentum(pct, m3, m6, mpct)
            if not rejected and score >= 40: sc[code] = score

        if not sc:
            cur_pos = {}; since_rebal = 0; continue

        top = sorted(sc.items(), key=lambda x: -x[1])[:max_n]
        top_dict = dict(top)

        do_rebal = (since_rebal >= QUARTERLY)
        if not do_rebal and cur_pos:
            cur_avg = np.mean([sc.get(c,0) for c in cur_pos])
            new_avg = np.mean(list(top_dict.values())[:max(1, len(cur_pos))])
            if new_avg > cur_avg + SWITCH_THR: do_rebal = True

        if do_rebal or not cur_pos:
            # V5新增: 仓位上限
            if use_weight_cap:
                weights = apply_weight_cap(top_dict, cap=MAX_SECTOR_WEIGHT)
            else:
                total = sum(top_dict.values())
                weights = {c: s/total for c, s in top_dict.items()}

            cur_pos = {c: w * t_ratio for c, w in weights.items()}
            since_rebal = 0

    if not sr_list: return {}
    ny = (dates[-1]-dates[0]).days/365.25
    sa = (sv**(1/ny)-1)*100
    ba = (bv**(1/ny)-1)*100
    ss = _stats(sr_list, sv)
    bs = _stats(br_list, bv)
    return {
        'label': label,
        'sa': round(sa,2), 'ba': round(ba,2), 'excess': round(sa-ba,2),
        'sc': round((sv-1)*100,1), 'bc': round((bv-1)*100,1),
        'smdd': ss['mdd'], 'bmdd': bs['mdd'],
        'ss': ss['sharpe'], 'bs': bs['sharpe'],
        'ny': round(ny,1),
        'start': dates[0].strftime('%Y-%m'), 'end': dates[-1].strftime('%Y-%m'),
        'halve_months': halve_cnt,
        'sector_filtered': sector_filtered_cnt,
    }


def main():
    do_backtest = '--backtest' in sys.argv

    print('=' * 72)
    print('  行业轮动策略 V5.0 — 行业级别趋势过滤 + 单行业仓位上限')
    print('=' * 72)
    print('\n📦 加载数据...')
    pe_piv, price_piv = load_data()
    print(f'  PE: {pe_piv.shape[0]}月 × {pe_piv.shape[1]}行业')
    print(f'  区间: {pe_piv.index.min().strftime("%Y-%m")} ~ {pe_piv.index.max().strftime("%Y-%m")}')

    if not do_backtest:
        # 实盘信号模式
        month = pe_piv.index.max()
        mpct  = calc_macro_pct(pe_piv, month)
        above, ratio = calc_market_above_ma(price_piv, month)
        max_n = 2 if mpct>70 else 5 if mpct<35 else 3

        print(f'\n📅 {month.strftime("%Y-%m")} | 宏观PE: {mpct:.0f}%位 | 大盘MA{TREND_MA_MONTHS*20}: {"✅上方" if above else "❌下方 (整体×50%)"}')

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
            sc, act, rej = score_momentum(pct, m3, m6, mpct)
            sector_ok = calc_sector_above_ma(price_piv, code, month)
            if not rej: rows.append((code, SECTOR_MAP[code], ETF_MAP.get(code,'--'), sc, pct, m3, m6, act, sector_ok))

        rows.sort(key=lambda x: -x[3])
        top_rows = [r for r in rows if r[8]][:max_n]  # 只取行业MA过滤后的
        top_dict = {r[0]: r[3] for r in top_rows}
        weights = apply_weight_cap(top_dict, cap=MAX_SECTOR_WEIGHT)

        print(f'\n【V5最优方案: 动量确认 + 行业MA3 + 仓位上限25%】 Top{max_n}:')
        print(f'{"行业":<8} {"ETF":>8} {"PE%":>5} {"3M%":>7} {"6M%":>7} {"分":>5} {"权重":>6} {"行业MA":>6} 建议')
        print('-' * 68)
        for code, name, etf, sc, pct, m3, m6, act, sector_ok in rows[:max_n+3]:
            w = weights.get(code, 0) * ratio * 100
            ma_flag = '✅' if sector_ok else '❌过滤'
            print(f'{name:<8} {etf:>8} {pct:>4.0f}% {m3:>+6.1f}% {m6:>+6.1f}% {sc:>5.1f} {w:>5.1f}% {ma_flag:>6} {act}')
        if ratio < 1.0:
            print(f'\n  ⚠️ 大盘在MA{TREND_MA_MONTHS*20}下方 → 实际仓位×{ratio:.0%}')
        return

    # ── 回测模式 ──
    print('\n⏳ 回测中 (4个方案)...\n')

    r4a = backtest(pe_piv, price_piv,
                   use_sector_trend=False, use_weight_cap=False,
                   label='V4A基准(大盘MA3)')
    r5a = backtest(pe_piv, price_piv,
                   use_sector_trend=True,  use_weight_cap=False,
                   label='V5A(+行业MA3)')
    r5b = backtest(pe_piv, price_piv,
                   use_sector_trend=False, use_weight_cap=True,
                   label='V5B(+仓位上限25%)')
    r5c = backtest(pe_piv, price_piv,
                   use_sector_trend=True,  use_weight_cap=True,
                   label='V5C(行业MA3+上限25%)')

    ref = r4a
    print(f'{"="*72}')
    print(f'  V5 回测 ({ref["start"]} ~ {ref["end"]}, {ref["ny"]}年)')
    print(f'{"="*72}')
    fmt = f'{{:<14}} {{:>10}} {{:>12}} {{:>12}} {{:>12}} {{:>10}}'
    print(fmt.format('指标', 'V4A基准', 'V5A行业MA', 'V5B上限25%', 'V5C双改进', '等权买持'))
    print('-'*72)
    print(fmt.format('年化收益',
        f'{r4a["sa"]:.1f}%', f'{r5a["sa"]:.1f}%',
        f'{r5b["sa"]:.1f}%', f'{r5c["sa"]:.1f}%', f'{ref["ba"]:.1f}%'))
    print(fmt.format('累计收益',
        f'{r4a["sc"]:.1f}%', f'{r5a["sc"]:.1f}%',
        f'{r5b["sc"]:.1f}%', f'{r5c["sc"]:.1f}%', f'{ref["bc"]:.1f}%'))
    print(fmt.format('最大回撤',
        f'{r4a["smdd"]:.1f}%', f'{r5a["smdd"]:.1f}%',
        f'{r5b["smdd"]:.1f}%', f'{r5c["smdd"]:.1f}%', f'{ref["bmdd"]:.1f}%'))
    print(fmt.format('夏普比率',
        f'{r4a["ss"]:.2f}', f'{r5a["ss"]:.2f}',
        f'{r5b["ss"]:.2f}', f'{r5c["ss"]:.2f}', f'{ref["bs"]:.2f}'))
    print(fmt.format('超额年化',
        f'{r4a["excess"]:+.1f}%', f'{r5a["excess"]:+.1f}%',
        f'{r5b["excess"]:+.1f}%', f'{r5c["excess"]:+.1f}%', '-'))
    print(fmt.format('大盘减半月',
        str(r4a["halve_months"]), str(r5a["halve_months"]),
        str(r5b["halve_months"]), str(r5c["halve_months"]), '-'))
    print(fmt.format('行业过滤次',
        str(r4a["sector_filtered"]), str(r5a["sector_filtered"]),
        str(r5b["sector_filtered"]), str(r5c["sector_filtered"]), '-'))
    print('='*72)

    # 找出最优方案
    results = [r4a, r5a, r5b, r5c]
    best = max(results, key=lambda r: r['ss'])
    print(f'\n【V5 结论】')
    print(f'  🏆 Sharpe最优: {best["label"]}')
    print(f'     年化{best["sa"]:.1f}%  Sharpe {best["ss"]:.2f}  MDD {best["smdd"]:.1f}%')
    print(f'\n  改进对比 (vs V4A基准):')
    for r in [r5a, r5b, r5c]:
        mdd_chg = r["smdd"] - r4a["smdd"]
        ret_chg = r["sa"] - r4a["sa"]
        sharpe_chg = r["ss"] - r4a["ss"]
        verdict = '✅' if (mdd_chg < -1 and ret_chg > -1) else ('⚠️' if mdd_chg < -1 else '❌')
        print(f'  {verdict} {r["label"]}: MDD {mdd_chg:+.1f}%  年化 {ret_chg:+.1f}%  Sharpe {sharpe_chg:+.2f}')


if __name__ == '__main__':
    main()
