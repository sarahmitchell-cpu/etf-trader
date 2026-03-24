#!/usr/bin/env python3
"""
行业轮动策略 v6.0 — 全特征升级版

在 V5A (大盘+行业双重MA3趋势过滤) 基础上新增六项改进:
  1. 交易成本模型   : 0.2%/次 (佣金+冲击成本+滑点), 按实际换手计算
  2. 动态止损       : 组合净值从峰值回撤>8%触发, 减仓至50%
  3. 跨资产防御     : 防御仓位配置短融ETF (~3.6%/年), 而非空仓
  4. 困境反转动量   : 三层评分(强动量/改善中/下行), 解决A股PE-动量矛盾
  5. 景气度代理因子 : 动量加速度 + 相对市场强度 (从价格数据构造)
  6. 动态因子权重   : 按宏观PE分位自动切换 PE/动量/景气 三因子权重

【V6 回测方案对比】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
V5A     : 行业MA3 (基准)
V6A     : V5A + 交易成本 (显示成本拖累)
V6B     : V6A + 动态止损 + 跨资产防御 (显示风控改进)
V6C     : V6B + 困境反转 + 动态权重 + 景气度 (完整V6)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用法:
  python3 sector_rotation_v6.py           # 实盘信号模式
  python3 sector_rotation_v6.py --backtest # 4方案回测对比
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

# ────────────────────────────────────────────────
# 参数配置
# ────────────────────────────────────────────────
PE_HARD_REJECT    = 85.0   # 行业PE%超过此值 → 硬排除
TREND_MA_MONTHS   = 3      # 大盘/行业 MA月数
TREND_HALVE       = 0.50   # 大盘MA以下时整体仓位系数
MAX_SECTOR_WEIGHT = 0.40   # 单行业最大权重 (V6放宽至40%, 因因子更丰富)
SWITCH_THR        = 12.0   # 新分比当前均分高N才换仓
QUARTERLY         = 3      # 最少持仓N个月

# V6 新增参数
COST_RATE         = 0.002  # 交易成本 0.2%/次 (佣金+冲击+滑点)
DD_STOP_THRESHOLD = 0.08   # 组合回撤>8%触发减仓保护
DD_RECOVER        = 0.03   # 从止损触发线反弹>3%恢复
BOND_MONTHLY_RET  = 0.003  # 防御仓位收益率 (短融ETF≈3.6%/年)


# ────────────────────────────────────────────────
# 数据加载
# ────────────────────────────────────────────────
def load_data():
    df = pd.read_pickle(PE_CACHE)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['sector_code'].isin(SECTOR_MAP)].copy()
    pe_piv    = df.pivot_table(index='date', columns='sector_code', values='pe')
    price_piv = df.pivot_table(index='date', columns='sector_code', values='close')
    return pe_piv, price_piv


# ────────────────────────────────────────────────
# 宏观与趋势函数
# ────────────────────────────────────────────────
def calc_macro_pct(pe_piv, month):
    """市场整体PE百分位 (等权中位数)"""
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
    """行业PE历史百分位 (无前瞻偏差: 排除当前值)"""
    hist = pe_series.dropna()
    if len(hist) < 2: return 50.0
    return float((hist.iloc[:-1] < current_val).mean() * 100)


# ────────────────────────────────────────────────
# V6 新增: 景气度代理因子
# ────────────────────────────────────────────────
def calc_prosperity(price_piv, code, month):
    """
    景气度代理因子 (从价格数据构造):
    Factor 1: 动量加速度 = 1M收益 - 3M收益/3  (正 = 动量加速)
    Factor 2: 相对市场强度 = 行业1M - 市场1M   (正 = 跑赢大盘)

    返回 0-100 的评分, 50为中性
    """
    if code not in price_piv.columns: return 50.0
    pr  = price_piv.loc[price_piv.index <= month, code].dropna()
    mkt = price_piv.loc[price_piv.index <= month].mean(axis=1).dropna()
    if len(pr) < 4: return 50.0

    m1_s = (pr.iloc[-1] / pr.iloc[-2] - 1) * 100 if len(pr) >= 2 else 0.0
    m3_s = (pr.iloc[-1] / pr.iloc[-4] - 1) * 100 if len(pr) >= 4 else 0.0
    m1_m = (mkt.iloc[-1] / mkt.iloc[-2] - 1) * 100 if len(mkt) >= 2 else 0.0

    # 动量加速度: 1M vs 3M月均化
    accel   = m1_s - m3_s / 3.0        # unit: % per month
    # 相对强度: 超额1M收益
    rel_str = m1_s - m1_m

    # 映射到 [0,100]: 基准50, ±10%贡献±25分
    score = 50.0
    score += float(np.clip(accel   * 2.5, -25.0, 25.0))
    score += float(np.clip(rel_str * 2.5, -25.0, 25.0))
    return float(np.clip(score, 0.0, 100.0))


# ────────────────────────────────────────────────
# V6 新增: 三层动量评分 (解决PE-动量矛盾)
# ────────────────────────────────────────────────
def momentum_score_v6(m1, m3, m6):
    """
    三层动量评分:
    - 强动量 (3M≥0 且 6M≥0): 100%权重, 一致性加成
    - 改善中 (3M<0 但 1M > 3M/3): 70%权重 ← 困境反转信号
    - 持续下行 (1M/3M/6M均差):  30%权重 (不完全排除低估值)

    A股特点: 低PE板块常伴随弱动量。引入"改善中"层可以捕捉
    估值低洼 + 动量触底的布局机会。
    """
    def ms(val, scale): return max(0., min(100., 50. + val / scale * 50.))

    m1s = ms(m1, 8.)    # 1M动量得分
    m3s = ms(m3, 15.)   # 3M动量得分
    m6s = ms(m6, 25.)   # 6M动量得分

    if m3 >= 0 and m6 >= 0:
        # ── 层1: 强动量 ──
        # 3M/6M方向一致, 全权重评分 + 一致性奖励
        base  = 0.50 * m3s + 0.35 * m6s + 0.15 * m1s
        bonus = 3.0   # 一致性加成 (保留V4/V5逻辑)
        return base, bonus, 'STRONG'

    elif m1 >= 0 and m3 < 0:
        # ── 层2: 困境反转 ──
        # 1M已经转正, 但3M仍为负 → 真正的底部翻转信号
        # 给予80%权重 (信号强于纯弱动量, 但弱于强动量)
        base  = 0.45 * m1s + 0.20 * m3s + 0.35 * m6s
        return base * 0.80, 1.0, 'TURNING'  # 小额一致性奖励

    else:
        # ── 层3: 持续下行 ──
        # 保留50%权重: 极端低估+弱动量在A股历史上仍有布局价值
        base  = 0.40 * m3s + 0.60 * m6s
        return base * 0.50, 0.0, 'WEAK'


# ────────────────────────────────────────────────
# V6 新增: 动态因子权重
# ────────────────────────────────────────────────
def dynamic_weights_v6(macro_pct):
    """
    按宏观PE分位动态调整三因子权重:

    市场高估 (PE>65%): 动量主导 → 追涨有效, 价值陷阱风险
    市场低估 (PE<40%): 估值主导 → 价值回归有效, 动量信号失真
    均衡区间          : 均衡权重

    返回: (w_pe, w_momentum, w_prosperity)
    三者之和 = 0.85 (余0.15留给一致性奖励/惩罚)
    """
    if macro_pct > 65:
        # 高估区: 动量为主, PE权重低 (追涨有效, 避免价值陷阱)
        return 0.25, 0.55, 0.10
    elif macro_pct < 40:
        # 低估区: 价值为主, 动量辅助 (价值回归有效)
        return 0.50, 0.35, 0.05
    else:
        # 均衡区: 靠近V5A权重, 加入少量景气度
        return 0.42, 0.48, 0.05


# ────────────────────────────────────────────────
# V6 综合评分
# ────────────────────────────────────────────────
def score_v6(pe_pct, m1, m3, m6, prosperity, macro_pct=50.0):
    """
    V6综合评分 = 动态权重 × [PE得分 + 动量得分 + 景气度得分] + 一致性奖励

    pe_pct    : 行业PE历史百分位 (0-100)
    m1/m3/m6  : 1/3/6月价格涨跌幅 (%)
    prosperity: 景气度代理评分 (0-100)
    macro_pct : 宏观市场PE百分位
    """
    if pe_pct > PE_HARD_REJECT:
        return 0.0, '🚫高估', True

    pe_s = 100.0 - pe_pct  # 低PE → 高PE得分

    mom_score, mom_bonus, mom_tier = momentum_score_v6(m1, m3, m6)
    wp, wm, wpr = dynamic_weights_v6(macro_pct)

    sc = wp * pe_s + wm * mom_score + wpr * prosperity + mom_bonus

    # 动量层级调整标签
    tier_icon = {'STRONG': '🟢', 'TURNING': '🔄', 'WEAK': '⚠️'}.get(mom_tier, '')
    action = (f'{tier_icon}🟢强买' if sc>=70 else
              f'{tier_icon}🟢买入' if sc>=58 else
              f'{tier_icon}🟡持有' if sc>=45 else
              f'{tier_icon}🟠减仓' if sc>=32 else
              f'{tier_icon}🔴回避')
    return sc, action, False


# ────────────────────────────────────────────────
# 回测辅助函数
# ────────────────────────────────────────────────
def _monthly_return(price_piv, pos, month, prev, bond_ratio=0.0):
    """
    计算月收益:
    - pos: {code: weight} 权益仓位 (weights sum ≤ 1.0)
    - bond_ratio: 防御仓位比例 (权益仓位外), 获得BOND_MONTHLY_RET收益
    """
    ret = 0.0
    for c, w in pos.items():
        if c not in price_piv.columns: continue
        sn = price_piv.loc[:month, c].dropna()
        sp = price_piv.loc[:prev,  c].dropna()
        if len(sn) > 0 and len(sp) > 0:
            ret += w * (sn.iloc[-1] / sp.iloc[-1] - 1)
    # 防御仓位收益 (短融ETF)
    if bond_ratio > 0:
        ret += bond_ratio * BOND_MONTHLY_RET
    return ret


def _bh_return(price_piv, valid, month, prev):
    codes = [c for c in valid if c in price_piv.columns]
    if not codes: return 0.0
    rets = []
    for c in codes:
        sn = price_piv.loc[:month, c].dropna()
        sp = price_piv.loc[:prev,  c].dropna()
        if len(sn) > 0 and len(sp) > 0:
            rets.append(sn.iloc[-1] / sp.iloc[-1] - 1)
    return float(np.mean(rets)) if rets else 0.0


def _stats(rets_list):
    r = np.array(rets_list)
    cum = pd.Series(r).add(1).cumprod()
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min() * 100)
    sharpe = float(r.mean() / (r.std() + 1e-10) * np.sqrt(12))
    return {'mdd': round(mdd, 1), 'sharpe': round(sharpe, 2)}


def calc_trade_cost(old_pos, new_pos):
    """
    计算换仓交易成本:
    cost = sum(|new_weight - old_weight|) * COST_RATE
    """
    all_codes = set(old_pos) | set(new_pos)
    turnover = sum(abs(new_pos.get(c, 0) - old_pos.get(c, 0)) for c in all_codes)
    return turnover * COST_RATE


def apply_weight_cap(scores_dict, cap=MAX_SECTOR_WEIGHT):
    """按上限迭代重分配权重"""
    total = sum(scores_dict.values())
    if total <= 0: return {}
    weights = {c: s / total for c, s in scores_dict.items()}
    for _ in range(20):
        excess  = sum(max(0, w - cap) for w in weights.values())
        if excess < 1e-8: break
        uncapped = {c for c, w in weights.items() if w < cap - 1e-8}
        if not uncapped: break
        capped   = {c: min(w, cap) for c, w in weights.items()}
        unc_sum  = sum(capped[c] for c in uncapped)
        if unc_sum <= 0: break
        for c in uncapped:
            capped[c] += excess * (capped[c] / unc_sum)
        weights = capped
    ft = sum(weights.values())
    return {c: w / ft for c, w in weights.items()} if ft > 0 else {}


# ────────────────────────────────────────────────
# 核心回测函数
# ────────────────────────────────────────────────
def backtest(pe_piv, price_piv,
             use_costs=False,
             use_dd_stop=False,
             use_bond_defense=False,
             use_v6_scoring=False,
             label=''):
    """
    统一回测框架, 通过开关控制V6各特性:
    use_costs       : 开启交易成本扣除
    use_dd_stop     : 开启动态回撤止损
    use_bond_defense: 防御仓位配置债券而非空仓
    use_v6_scoring  : 使用V6三层动量+景气度+动态权重评分
    """
    MIN_H = 24
    months = sorted(pe_piv.index)
    valid  = [c for c in pe_piv.columns if c in SECTOR_MAP]

    sv, bv = 1.0, 1.0
    sr_list, br_list, dates = [], [], []
    cur_pos, since_rebal = {}, 0

    # 状态变量
    halve_cnt      = 0
    sector_flt_cnt = 0
    total_cost     = 0.0
    rebal_cnt      = 0

    # 动态止损状态
    port_peak        = 1.0
    in_dd_stop       = False
    dd_trigger_level = 1.0

    for i, m in enumerate(months):
        if i < MIN_H: continue
        prev = months[i - 1]

        # ── Step 1: 结算本月收益 (使用上月仓位, 无前瞻偏差) ──
        if i > MIN_H and cur_pos:
            equity_sum = sum(cur_pos.values())
            bond_ratio = (1.0 - equity_sum) if use_bond_defense else 0.0
            sr = _monthly_return(price_piv, cur_pos, m, prev, bond_ratio)
            br = _bh_return(price_piv, valid, m, prev)
            sv *= (1 + sr)
            bv *= (1 + br)
            sr_list.append(sr)
            br_list.append(br)
            dates.append(m)

            # 更新峰值与止损状态
            if use_dd_stop:
                port_peak = max(port_peak, sv)
                dd_now    = (sv - port_peak) / port_peak
                if not in_dd_stop and dd_now < -DD_STOP_THRESHOLD:
                    in_dd_stop       = True
                    dd_trigger_level = sv
                elif in_dd_stop and sv > dd_trigger_level * (1 + DD_RECOVER):
                    in_dd_stop = False

        since_rebal += 1

        # ── Step 2: 月末决策: 确定下月仓位 ──
        mpct = calc_macro_pct(pe_piv, m)

        # 大盘趋势 (V5已有)
        above, t_ratio = calc_market_above_ma(price_piv, m)
        if not above: halve_cnt += 1

        # 动态止损叠加大盘过滤
        if use_dd_stop and in_dd_stop:
            t_ratio = min(t_ratio, 0.50)  # 止损期间最多50%仓位

        max_n = 2 if mpct > 70 else 5 if mpct < 35 else 3

        sc = {}
        for code in valid:
            pe_s = pe_piv[pe_piv.index <= m].get(code)
            if pe_s is None: continue
            pe_s = pe_s.dropna()
            if len(pe_s) < MIN_H: continue
            cv = pe_s.iloc[-1]
            if cv <= 0 or pd.isna(cv): continue

            # 行业趋势过滤 (V5已有, V6保留)
            if not calc_sector_above_ma(price_piv, code, m):
                sector_flt_cnt += 1
                continue

            pct = calc_sector_pe_pct(pe_s, cv)

            # 计算动量
            pr_all = price_piv[price_piv.index <= m]
            m1 = m3 = m6 = 0.0
            if code in pr_all.columns:
                pr = pr_all[code].dropna()
                if len(pr) >= 2: m1 = (pr.iloc[-1] / pr.iloc[-2] - 1) * 100
                if len(pr) >= 4: m3 = (pr.iloc[-1] / pr.iloc[-4] - 1) * 100
                if len(pr) >= 7: m6 = (pr.iloc[-1] / pr.iloc[-7] - 1) * 100

            if use_v6_scoring:
                # V6: 三层动量 + 景气度 + 动态权重
                pros = calc_prosperity(price_piv, code, m)
                score, _, rejected = score_v6(pct, m1, m3, m6, pros, mpct)
            else:
                # V5A评分 (基准)
                def ms_v5(v, sc): return max(0., min(100., 50. + v / sc * 50.))
                m3s, m6s = ms_v5(m3, 15.), ms_v5(m6, 25.)
                if (m3 >= 0) == (m6 >= 0):
                    mom_sc, bonus = 0.55 * m3s + 0.45 * m6s, 3.0
                else:
                    mom_sc, bonus = 0.5 * min(m3s, m6s), 0.0
                wp = 0.62 if mpct > 70 else (0.35 if mpct < 35 else 0.48)
                wm = 1 - wp
                pe_s_v = 100.0 - pct
                score = wp * pe_s_v + wm * mom_sc + bonus
                rejected = (pct > PE_HARD_REJECT)

            if not rejected and score >= 40:
                sc[code] = score

        if not sc:
            # 全空仓
            if use_costs and cur_pos:
                cost = calc_trade_cost(cur_pos, {})
                sv  *= (1 - cost)
                total_cost += cost
            cur_pos = {}
            since_rebal = 0
            continue

        top      = sorted(sc.items(), key=lambda x: -x[1])[:max_n]
        top_dict = dict(top)

        do_rebal = (since_rebal >= QUARTERLY)
        if not do_rebal and cur_pos:
            cur_avg = float(np.mean([sc.get(c, 0) for c in cur_pos]))
            new_avg = float(np.mean(list(top_dict.values())[:max(1, len(cur_pos))]))
            if new_avg > cur_avg + SWITCH_THR:
                do_rebal = True

        if do_rebal or not cur_pos:
            weights  = apply_weight_cap(top_dict, cap=MAX_SECTOR_WEIGHT)
            new_pos  = {c: w * t_ratio for c, w in weights.items()}

            # 交易成本
            if use_costs:
                cost = calc_trade_cost(cur_pos, new_pos)
                sv  *= (1 - cost)
                total_cost += cost
                rebal_cnt  += 1

            cur_pos      = new_pos
            since_rebal  = 0
        else:
            # 惰性换仓: 保持当前持仓不变
            # 大盘信号调整只在下次正式换仓时生效, 避免频繁交易成本
            pass

    if not sr_list: return {}
    ny  = (dates[-1] - dates[0]).days / 365.25
    sa  = (sv ** (1 / ny) - 1) * 100
    ba  = (bv ** (1 / ny) - 1) * 100
    ss  = _stats(sr_list)
    bs  = _stats(br_list)
    return {
        'label':          label,
        'sa':             round(sa, 2),
        'ba':             round(ba, 2),
        'excess':         round(sa - ba, 2),
        'sc':             round((sv - 1) * 100, 1),
        'bc':             round((bv - 1) * 100, 1),
        'smdd':           ss['mdd'],
        'bmdd':           bs['mdd'],
        'ss':             ss['sharpe'],
        'bs':             bs['sharpe'],
        'ny':             round(ny, 1),
        'start':          dates[0].strftime('%Y-%m'),
        'end':            dates[-1].strftime('%Y-%m'),
        'halve_months':   halve_cnt,
        'sector_filtered':sector_flt_cnt,
        'total_cost_pct': round(total_cost * 100, 2),
        'rebal_cnt':      rebal_cnt,
    }


# ────────────────────────────────────────────────
# 主程序
# ────────────────────────────────────────────────
def main():
    do_backtest = '--backtest' in sys.argv

    print('=' * 76)
    print('  行业轮动策略 V6.0 — 交易成本 + 动态止损 + 跨资产防御 + 困境反转')
    print('=' * 76)
    print('\n📦 加载数据...')
    pe_piv, price_piv = load_data()
    print(f'  PE: {pe_piv.shape[0]}月 × {pe_piv.shape[1]}行业')
    print(f'  区间: {pe_piv.index.min().strftime("%Y-%m")} ~ {pe_piv.index.max().strftime("%Y-%m")}')

    if not do_backtest:
        # ── 实盘信号模式 ──
        month = pe_piv.index.max()
        mpct  = calc_macro_pct(pe_piv, month)
        above, ratio = calc_market_above_ma(price_piv, month)
        max_n = 2 if mpct > 70 else 5 if mpct < 35 else 3

        print(f'\n📅 {month.strftime("%Y-%m")} | 宏观PE: {mpct:.0f}%位',
              f'| 大盘MA{TREND_MA_MONTHS*20}: {"✅上方" if above else "❌下方(×50%)"}')

        rows = []
        for code in sorted(pe_piv.columns):
            if code not in SECTOR_MAP: continue
            pe_s = pe_piv[code].dropna()
            if len(pe_s) < 12: continue
            cv = pe_s.iloc[-1]
            if cv <= 0: continue

            pct  = calc_sector_pe_pct(pe_s, cv)
            m1 = m3 = m6 = 0.0
            if code in price_piv:
                pr = price_piv[code].dropna()
                if len(pr) >= 2: m1 = (pr.iloc[-1]/pr.iloc[-2]-1)*100
                if len(pr) >= 4: m3 = (pr.iloc[-1]/pr.iloc[-4]-1)*100
                if len(pr) >= 7: m6 = (pr.iloc[-1]/pr.iloc[-7]-1)*100

            pros = calc_prosperity(price_piv, code, month)
            sc, act, rej = score_v6(pct, m1, m3, m6, pros, mpct)
            _, _, mom_tier = momentum_score_v6(m1, m3, m6)
            sector_ok = calc_sector_above_ma(price_piv, code, month)
            if not rej:
                rows.append((code, SECTOR_MAP[code], ETF_MAP.get(code,'--'),
                             sc, pct, m1, m3, m6, pros, act, mom_tier, sector_ok))

        rows.sort(key=lambda x: -x[3])
        top_rows = [r for r in rows if r[11]][:max_n]
        top_dict = {r[0]: r[3] for r in top_rows}
        weights  = apply_weight_cap(top_dict, cap=MAX_SECTOR_WEIGHT)

        print(f'\n【V6最优信号 Top{max_n}】(宏观PE:{mpct:.0f}%位)')
        hdr = f'{"行业":<8} {"ETF":>8} {"PE%":>5} {"1M%":>6} {"3M%":>7} {"6M%":>7} {"景气":>5} {"分":>5} {"权重":>6} {"层级":>5} 建议'
        print(hdr)
        print('-' * 82)
        for code, name, etf, sc, pct, m1, m3, m6, pros, act, tier, ok in rows[:max_n+4]:
            w    = weights.get(code, 0) * ratio * 100
            tstr = {'STRONG':'强','TURNING':'转','WEAK':'弱'}.get(tier,'?')
            flag = '' if ok else '❌'
            print(f'{name:<8} {etf:>8} {pct:>4.0f}% {m1:>+5.1f}% {m3:>+6.1f}% {m6:>+6.1f}%'
                  f' {pros:>4.0f} {sc:>5.1f} {w:>5.1f}%{flag:>3} {tstr:>5} {act}')
        if ratio < 1.0:
            print(f'\n  ⚠️ 大盘MA{TREND_MA_MONTHS*20}下方 → 仓位×{ratio:.0%}',
                  '(防御仓位配置短融ETF)' if True else '')
        return

    # ── 回测模式: 4方案对比 ──
    print('\n⏳ 回测中 (4个方案, 请稍候)...\n')

    r_v5a = backtest(pe_piv, price_piv,
                     use_costs=False, use_dd_stop=False,
                     use_bond_defense=False, use_v6_scoring=False,
                     label='V5A(基准)')

    r_v6a = backtest(pe_piv, price_piv,
                     use_costs=True,  use_dd_stop=False,
                     use_bond_defense=False, use_v6_scoring=False,
                     label='V6A(+成本)')

    r_v6b = backtest(pe_piv, price_piv,
                     use_costs=True,  use_dd_stop=True,
                     use_bond_defense=True,  use_v6_scoring=False,
                     label='V6B(+止损+债券)')

    r_v6c = backtest(pe_piv, price_piv,
                     use_costs=True,  use_dd_stop=True,
                     use_bond_defense=True,  use_v6_scoring=True,
                     label='V6C(全特征)')

    # ── 打印结果表 ──
    ref = r_v5a
    print('=' * 76)
    print(f'  V6 回测 ({ref["start"]} ~ {ref["end"]}, {ref["ny"]}年)')
    print('=' * 76)
    fmt = '{:<17} {:>10} {:>12} {:>12} {:>12} {:>10}'
    print(fmt.format('指标', 'V5A基准', 'V6A+成本', 'V6B+止损债券', 'V6C全特征', '等权买持'))
    print('-' * 76)

    def row(name, fn):
        return fmt.format(name,
            fn(r_v5a), fn(r_v6a), fn(r_v6b), fn(r_v6c), fn(ref, bh=True))

    def _r(r, key, bh=False):
        k = ('b'+key[1:]) if bh else key
        v = r.get(k if bh else key, r.get(key, '-'))
        return str(v) if isinstance(v, str) else f'{v}'

    # 手动打印各行
    def p(name, key, fmt_fn):
        vals = [fmt_fn(r.get(key, 0)) for r in [r_v5a, r_v6a, r_v6b, r_v6c]]
        bh   = fmt_fn(ref.get('b'+key[1:], ref.get(key, 0)) if key.startswith('s') else ref.get(key, 0))
        # 特殊处理 buy-hold keys
        if key == 'sa':   bh = fmt_fn(ref['ba'])
        elif key == 'sc': bh = fmt_fn(ref['bc'])
        elif key == 'smdd': bh = fmt_fn(ref['bmdd'])
        elif key == 'ss':   bh = fmt_fn(ref['bs'])
        else:               bh = '-'
        print(fmt.format(name, *vals, bh))

    p('年化收益',  'sa',   lambda v: f'{v:.1f}%')
    p('累计收益',  'sc',   lambda v: f'{v:.1f}%')
    p('最大回撤',  'smdd', lambda v: f'{v:.1f}%')
    p('夏普比率',  'ss',   lambda v: f'{v:.2f}')
    p('超额年化',  'excess', lambda v: f'{v:+.1f}%')

    # 额外统计
    print(fmt.format('累计成本%',
        '-',
        f'{r_v6a.get("total_cost_pct",0):.2f}%',
        f'{r_v6b.get("total_cost_pct",0):.2f}%',
        f'{r_v6c.get("total_cost_pct",0):.2f}%',
        '-'))
    print(fmt.format('换仓次数',
        f'{r_v5a.get("rebal_cnt",0)}',
        f'{r_v6a.get("rebal_cnt",0)}',
        f'{r_v6b.get("rebal_cnt",0)}',
        f'{r_v6c.get("rebal_cnt",0)}',
        '-'))
    print(fmt.format('行业过滤次',
        str(r_v5a['sector_filtered']),
        str(r_v6a['sector_filtered']),
        str(r_v6b['sector_filtered']),
        str(r_v6c['sector_filtered']),
        '-'))
    print('=' * 76)

    # ── 结论 ──
    results = [r_v5a, r_v6a, r_v6b, r_v6c]
    best_sharpe = max(results, key=lambda r: r['ss'])
    best_mdd    = min(results, key=lambda r: r['smdd'])  # less negative = better

    print('\n【V6 结论】')
    print(f'  🏆 Sharpe最优 : {best_sharpe["label"]}')
    print(f'     年化{best_sharpe["sa"]:.1f}%  Sharpe {best_sharpe["ss"]:.2f}  MDD {best_sharpe["smdd"]:.1f}%')
    print(f'  🛡️  MDD最小   : {best_mdd["label"]}')
    print(f'     年化{best_mdd["sa"]:.1f}%  Sharpe {best_mdd["ss"]:.2f}  MDD {best_mdd["smdd"]:.1f}%')

    print('\n  改进对比 (vs V5A基准):')
    for r in [r_v6a, r_v6b, r_v6c]:
        mdd_chg    = r['smdd'] - r_v5a['smdd']   # 正数 = MDD更差
        ret_chg    = r['sa']   - r_v5a['sa']
        sharpe_chg = r['ss']   - r_v5a['ss']
        # 判断改进: MDD减小(mdd_chg>0, 因MDD是负数, +代表变好) 且 年化不明显下滑
        verdict = ('✅' if (mdd_chg > 1 and ret_chg > -2) else
                   ('⚠️' if mdd_chg > 0.5 else '❌'))
        cost_info = f' [累计成本{r.get("total_cost_pct",0):.2f}%]' if r.get('total_cost_pct') else ''
        print(f'  {verdict} {r["label"]}: MDD {mdd_chg:+.1f}%  年化 {ret_chg:+.1f}%  Sharpe {sharpe_chg:+.2f}{cost_info}')

    print('\n  注: 交易成本对年化收益的实际拖累 =',
          f'{r_v6a["sa"] - r_v5a["sa"]:+.1f}% (V6A vs V5A)')
    print('  注: A股PE-动量矛盾改善效果 = 困境反转层触发时PE低估板块可入选')


if __name__ == '__main__':
    main()
