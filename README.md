# ETF & 个股量化策略体系

> 10套独立策略 · A股/港股/美股/债券全覆盖 · 无前瞻偏差 · 自动化信号推送

一套完整的量化交易策略体系，覆盖债券、A股ETF、港股ETF、美股QDII、指数择时和个股因子选股。

---

## 策略总览

| 策略 | 名称 | 标的 | 核心逻辑 | CAGR | MDD | Sharpe | 调仓频率 |
|------|------|------|---------|------|-----|--------|---------|
| **A** | 债券分散持有 | 4只债券ETF | 等权分散+季度再平衡 | 4.1% | -2.1% | 1.54 | 季度 |
| **B** | 因子ETF轮动 | 4只A股Smart Beta ETF | 动量排名+Regime+Vol缩放 | 15.6% | -18.3% | 1.157 | 每周 |
| **C** | 跨境ETF纯动量 | 3只恒生系ETF | Top1纯动量，100%持仓 | 17.0% | -23.3% | 0.61 | 每周 |
| **D** | CSI300低波动 | 沪深300成分股 | 20周波动率最低TOP10 | 13.9% | -8.5% | 0.76 | 双周 |
| **E** | 逆向价值+质量 | CSI100成分股 | 10w反转+低波质量过滤 | 9.7% | -17.9% | 0.49 | 月度 |
| **F** | 美股QDII动量 | 纳指/标普QDII ETF | 双MA趋势+回撤锁定 | 13.0% | -18.9% | 0.77 | 每周 |
| **G** | CSI500低波价值 | 中证500成分股 | 50%低波+50%低PB | 4.4% | -15.8% | 0.19 | 双周 |
| **H** | 指数超跌买入 | 科创50指数 | 超跌/追涨信号+定期持有 | 13.9% | -6.6% | 1.04 | 事件驱动 |
| **L** | 300成长MA60趋势 | 沪深300成长指数 | 纯MA60趋势跟踪 | 18.9% | -31.7% | 0.49 | 日频信号 |

> 另有**个股因子策略**（CSI800 低换手+12M动量 Top20，CAGR=21.4%，Sharpe=0.97），见 `research/stock_strategy_live.py`

### 策略互补关系

```
               +-- 策略A (债券分散持有)          <-- 防守底仓，极低波动
               |
               +-- 策略B (因子ETF轮动)          <-- A股SmartBeta，Regime自适应
               |
               +-- 策略C (跨境ETF动量)           <-- 港股独立配置
               |
               +-- 策略D (CSI300低波) --+
               |                        +--  成分股策略，数据无后视偏差
资金分配 ------+-- 策略G (CSI500低波) --+
               |
               +-- 策略E (逆向价值)              <-- CSI100中选低估股 (审计后修正)
               |
               +-- 策略F (美股QDII)              <-- 美股配置，趋势跟踪
               |
               +-- 策略H (指数超跌买入)          <-- 科创50事件驱动，低仓位占用
               |
               +-- 策略L (300成长MA60)           <-- 大盘成长趋势择时，极简
```

---

## 快速开始

### 环境要求

```bash
Python 3.8+
pip3 install -r requirements.txt
```

- `baostock`: 策略D和G需要（获取沪深300/中证500历史成分股数据）
- `akshare`: 策略E v2和个股因子策略需要（获取PE/PB基本面数据）
- 其他策略仅需 `pandas numpy requests`

### 运行信号

```bash
git clone https://github.com/sarahmitchell-cpu/etf-trader.git
cd etf-trader

# 策略A - 债券分散持有
python3 strategies/strategy_a_weekly_signal.py

# 策略B - 因子ETF轮动
python3 strategies/strategy_b_weekly_signal.py

# 策略C - 跨境ETF纯动量
python3 strategies/strategy_c_weekly_signal.py

# 策略D - CSI300低波动（双周）
python3 strategies/strategy_d_weekly_signal.py

# 策略E - 逆向价值+质量（月度）
python3 strategies/strategy_e_weekly_signal.py

# 策略F - 美股QDII动量
python3 strategies/strategy_f_weekly_signal.py

# 策略G - CSI500低波价值（双周）
python3 strategies/strategy_g_weekly_signal.py

# 策略H - 科创50超跌买入（事件驱动）
python3 strategies/strategy_h_weekly_signal.py

# 策略L - 300成长MA60趋势择时
python3 strategies/strategy_l_weekly_signal.py

# 个股因子策略 - CSI800 低换手+动量 Top20
python3 research/stock_strategy_live.py
```

### 运行回测

```bash
# 各策略均支持 --backtest 参数
python3 strategies/strategy_a_weekly_signal.py --backtest
python3 strategies/strategy_b_weekly_signal.py --backtest
python3 strategies/strategy_d_weekly_signal.py --backtest
python3 strategies/strategy_g_weekly_signal.py --backtest
python3 strategies/strategy_h_weekly_signal.py --backtest
python3 strategies/strategy_l_weekly_signal.py --backtest
# ...
```

### JSON输出

```bash
# 各策略均支持 --json 输出结构化信号
python3 strategies/strategy_a_weekly_signal.py --json
python3 strategies/strategy_b_weekly_signal.py --json
python3 strategies/strategy_h_weekly_signal.py --json
# ...
```

---

## 策略详细说明

每个策略都有独立的详细文档，位于 `docs/` 目录。以下是概要说明。

---

### 策略A: 债券分散持有

> 详细文档: [docs/strategy_a.md](docs/strategy_a.md)

**逻辑**: 在4只债券ETF中等权分散持有（各25%），每季度再平衡。纯被动策略，目标获取债券市场平均收益。

**ETF池**:

| 指数 | ETF | 代码 | 定位 |
|------|-----|------|------|
| 十年国债 (H11009) | 国泰十年国债ETF | 511260 | 长久期利率债 |
| 城投债 (H11015) | 海富通城投债ETF | 511220 | 中久期信用债 |
| 信用债 (H11073) | 平安公司债ETF | 511030 | 中短久期信用债 |
| 短债 (H11006) | 海富通短融ETF | 511360 | 短久期现金替代 |

**参数**: `weights=[25%, 25%, 25%, 25%], rebalance=quarterly, txn_cost=5bp`

**回测 (15年)**: CAGR 4.1%, MDD -2.1%, Sharpe 1.54, Calmar 1.97

---

### 策略B: 4因子ETF轮动

> 详细文档: [docs/strategy_b.md](docs/strategy_b.md)

**逻辑**: 4只SmartBeta ETF按4周动量排名 -> 50/30/20/0集中配置 -> Regime判断（牛/熊/过渡）调整仓位 -> 波动率缩放动态调节。

**ETF池**: 红利低波(515080)、自由现金流(159201)、国信价值(512040)、创业成长(159967)

**参数**: `momentum_lookback=4, vol_target=15%, regime: MA13w/MA40w, vol_scale_cap=1.3`

**回测 (11年)**: CAGR 15.6%, MDD -18.3%, Sharpe 1.157

**杠杆风险提示**: vol_scale_cap=1.3意味着低波动市场中仓位可达130%，需要融资。如不使用杠杆，建议将vol_scale_cap设为1.0。

---

### 策略C: 跨境ETF纯动量

> 详细文档: [docs/strategy_c.md](docs/strategy_c.md)

**逻辑**: 3只恒生系ETF取4周动量最强的1只，100%持仓。极致简洁，无趋势过滤、无Regime择时。

**ETF池**: 恒生高股息(159726)、恒生科技(513180)、恒生医疗(513060)

**回测 (4.2年)**: CAGR 17.0%, MDD -23.3%, Sharpe 0.61

---

### 策略D: CSI300低波动TOP10

> 详细文档: [docs/strategy_d.md](docs/strategy_d.md)

**逻辑**: 从沪深300**当期真实成分股**中，选20周年化波动率最低的10只，等权配置，双周调仓。使用baostock获取历史成分股数据，完全消除生存偏差。

**回测 (~4.2年, 2022-01~2026-03)**: CAGR 13.9%, MDD -8.5%, Sharpe 0.76

---

### 策略E: 逆向价值+质量

> 详细文档: [docs/strategy_e.md](docs/strategy_e.md)

**逻辑**: 从CSI100**当期真实成分股**中，选10周跌幅最大的（逆向）x 低波质量过滤，取Top4等权配置。

**回测 (审计修正后, 无幸存者偏差)**: CAGR 9.7%, MDD -17.9%, Sharpe 0.49

**偏差处理**: V3版本使用baostock CSI100历史成分股，消除幸存者偏差。原版28只龙头股的22% CAGR被证实>50%来自偏差膨胀。

---

### 策略F: 美股QDII动量轮动

> 详细文档: [docs/strategy_f.md](docs/strategy_f.md)

**逻辑**: 纳指100 vs 标普500按4周动量排名，Top1持100%。纳指低于20周MA时进入防守，全仓短债ETF。

**回测 (15年)**: CAGR 13.0%, MDD -18.9%, Sharpe 0.77

---

### 策略G: CSI500低波价值

> 详细文档: [docs/strategy_g.md](docs/strategy_g.md)

**逻辑**: 从中证500**当期真实成分股**中，综合50%低波动+50%低PB因子打分，选Top10等权配置。

**回测 (4年, 2022-01~2026-03, 新鲜数据)**: CAGR 4.4%, MDD -15.8%, Sharpe 0.19

**注意**: 原缓存数据显示CAGR=13.5%/Sharpe=0.79，刷新baostock数据后大幅下降。策略表现弱于预期。

---

### 策略H: 指数超跌买入/追涨买入

> 详细文档: [docs/strategy_h.md](docs/strategy_h.md)

**逻辑**: 系统化捕捉短期价格极端——超跌后买入（均值回归）或大涨后追入（动量延续），持有固定天数后退出。

**6个验证通过的子变体（全部为科创50）**:

| 变体 | 方向 | 信号 | 持有 | Sharpe | CAGR | MDD | 胜率 |
|------|------|------|------|--------|------|-----|------|
| H1 | 超跌 | 3日跌>7% | 4日 | 1.04 | 13.9% | -6.6% | 72% |
| H2 | 追涨 | 5日涨>7% | 20日 | 0.68 | 15.9% | -23.4% | 57% |
| H3 | 超跌 | 3日跌>8% | 4日 | 1.07 | 13.4% | -5.5% | 83% |
| H4 | 超跌 | 4日跌>7% | 3日 | 1.03 | 13.7% | -8.9% | 75% |
| H5 | 超跌 | 3日跌>7% | 1日 | 0.90 | 8.6% | -3.0% | 74% |
| H6 | 超跌 | 6日跌>10% | 3日(SL-3%) | 1.13 | 12.8% | -3.6% | 83% |

---

### 策略L: 300成长纯MA60趋势择时

> 详细文档: [docs/strategy_l.md](docs/strategy_l.md)

**逻辑**: 极简趋势跟踪——沪深300成长全收益指数收盘价在MA60上方满仓，下方空仓。

**回测 (~20.5年, 2005~2026)**: CAGR 18.9%, MDD -31.7%, Sharpe 0.49

---

### 个股因子策略: CSI800 低换手+12M动量

> 代码: `research/stock_strategy_live.py`

**逻辑**: 从CSI800历史真实成分股中，用低换手率+12M动量(skip1M)多因子选股，Top20等权，月度调仓。

**回测 (2016-01~2026-03, ~10年)**: CAGR 21.4%, Sharpe 0.97, MDD -25.4%

---

## 目录结构

```
etf-trader/
├── README.md                              # 策略总览与复现说明
├── requirements.txt                       # Python依赖
├── .gitignore
│
├── strategies/                            # 生产策略 (每周信号生成)
│   ├── strategy_a_weekly_signal.py        # A: 债券分散持有
│   ├── strategy_b_weekly_signal.py        # B: 因子ETF轮动 (v6)
│   ├── strategy_c_weekly_signal.py        # C: 跨境ETF纯动量
│   ├── strategy_d_weekly_signal.py        # D: CSI300低波动TOP10
│   ├── strategy_e_weekly_signal.py        # E: 逆向价值+质量 (V3无偏差)
│   ├── strategy_f_weekly_signal.py        # F: 美股QDII动量
│   ├── strategy_g_weekly_signal.py        # G: CSI500低波价值
│   ├── strategy_h_weekly_signal.py        # H: 科创50超跌买入
│   ├── strategy_j_growth_momentum.py      # J: 成长动量
│   ├── strategy_k_growth_value_rotation.py # K: 成长价值轮动
│   ├── strategy_l_weekly_signal.py        # L: 300成长MA60趋势
│   └── panic_buy_signal.py               # 恐慌买入信号
│
├── lib/                                   # 共享库模块
│   └── stock_data_common.py               # D/E/G共用数据获取模块
│
├── research/                              # 研究 & 回测脚本 (66个)
│   ├── stock_strategy_live.py             # 个股因子策略 (CSI800)
│   ├── stock_factor_research_v*.py        # 个股因子研究系列
│   ├── strategy_*_research.py             # 各策略研究脚本
│   ├── csi500_*.py                        # CSI500因子研究系列
│   ├── factor_timing_*.py                 # 因子择时研究
│   ├── strategy_momentum_*.py             # 动量策略研究
│   └── ...                                # 更多研究脚本
│
├── tools/                                 # 工具脚本
│   ├── generate_position_report_ppt.py    # PPT建仓报告生成器
│   └── download_missing_stocks.py         # 缺失股票数据下载
│
├── docs/                                  # 详细策略文档
│   ├── STRATEGY_AUDIT.md                  # 策略审计报告
│   ├── strategy_a.md ~ strategy_l.md      # 各策略详细文档
│   └── strategy_d_v1.md                   # D v1旧版文档
│
├── data/                                  # 数据缓存 (大部分gitignored)
│   ├── *_daily.csv / *_weekly.csv         # 指数/个股历史行情
│   ├── *_backtest*.json                   # 回测结果
│   ├── *_latest_signal.json               # 最新信号
│   ├── baostock_cache/                    # 历史成分股缓存
│   └── fundamental_cache/                 # 基本面数据缓存
│
└── archived/                              # 废弃代码存档
    ├── sector_rotation_v*.py              # 旧版行业轮动
    └── *.archived                         # 其他废弃脚本
```

---

## 关键设计原则

### 1. 无前瞻偏差 (No Look-Ahead Bias)

- **策略D/G**: 使用baostock历史成分股API，回测每期仅使用该期真实成分股
- **策略E V3**: 使用baostock CSI100历史成分股，消除幸存者偏差
- **策略H V2**: 信号当日收盘检测，次日开盘价入场
- **策略L**: 信号T日生成，T+1执行（`shift(1)`）
- **所有策略**: T期数据做决策 -> T+1期收益计算

### 2. 包含交易成本

| 策略 | 单边成本 | 说明 |
|------|---------|------|
| A | 5bp | 债券ETF低成本 |
| B | 10bp | A股ETF（佣金+冲击） |
| C | 8bp | 跨境ETF |
| D/G | 8bp | A股个股 |
| E | 8bp | A股个股 |
| F | 15bp | QDII ETF（含汇兑） |
| H | 10bp(双边) | 指数ETF |
| L | 8bp | A股ETF |
| 个股因子 | 30bp | A股个股（含冲击成本） |

### 3. 审计验证 (2026-03-29)

全部11个策略经过独立审计，详见 [docs/STRATEGY_AUDIT.md](docs/STRATEGY_AUDIT.md)。
主要发现:
- 策略E: 原28只龙头存在严重幸存者偏差（CAGR从22%降至9.7%）
- 策略G: 刷新baostock数据后表现大幅下降（CAGR从13.5%降至4.4%）
- 推荐策略: A(Sharpe=1.54), D(0.76), 个股低换手+动量(0.97)

---

## 已知局限性

- **策略B**: `vol_scale_cap=1.3`允许仓位达130%（隐含杠杆），需融资账户支持
- **策略C**: 仅4.2年数据，统计意义有限；无止损/回撤保护机制
- **策略D**: 回测仅~4.2年，统计检验力不足
- **策略E**: V3修正幸存者偏差后CAGR仅9.7%，表现一般
- **策略F**: 15bp交易成本较高，QDII ETF折溢价风险
- **策略G**: 刷新数据后Sharpe仅0.19，建议谨慎使用
- **策略H**: 全部变体集中在科创50，依赖高波动特性
- **策略L**: 震荡市频繁交易；MDD -31.7%仍较大

## 风险提示

- 本项目为个人量化研究工具，**不构成任何投资建议**
- 历史回测不代表未来收益
- 请根据自身风险承受能力决定是否参考

---

## 版本历史

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2026-03-30 | v5.0 | 代码库重构：strategies/lib/research/tools/目录划分；全策略审计完成；修正E/G偏差 |
| 2026-03-29 | v4.0 | 新增策略H(指数超跌)/L(300成长MA60), 个股因子策略; 全策略文档完善 |
| 2026-03-24 | v3.0 | 新增策略A/F/G，策略D改为CSI300低波TOP10，7策略模拟盘启动 |
| 2026-03-22 | v2.0 | 新增策略C/D/E，整理仓库结构 |
| 2026-03-18 | v1.0 | 初版：宏观定仓+行业轮动+策略B |

---

*由 VisionClaw AI 代理维护 · Sarah Mitchell · sarahmitchell@visionclaw.dev*
