# ETF & 个股量化策略体系

> 7套独立策略 · A股/港股/美股/债券全覆盖 · 无前瞻偏差 · 自动化信号推送

一套完整的量化交易策略体系，覆盖债券、A股ETF、港股ETF、美股QDII和个股因子策略。

---

## 策略总览

| 策略 | 名称 | 标的 | 核心逻辑 | CAGR | MDD | Calmar | 调仓频率 |
|------|------|------|---------|------|-----|--------|---------|
| **A** | 纯债ETF轮动 | 4只债券ETF | 动量排名+集中配置 | 4.1% | -2.1% | 1.97 | 每周 |
| **B** | 因子ETF轮动 | 4只A股Smart Beta ETF | 动量排名+Regime+Vol缩放 | 15.6% | -18.3% | 0.85 | 每周 |
| **C** | 跨境ETF纯动量 | 3只恒生系ETF | Top1纯动量，100%持仓 | 17.0% | -23.3% | 0.73 | 每周 |
| **D** | CSI300低波动 | 沪深300成分股 | 20周波动率最低TOP10 | 13.9% | -8.5% | 1.64 | 双周 |
| **E** | 逆向价值+质量 | 28只A股龙头 | 10w反转+低波质量过滤 | 22.0% | -17.6% | 1.25 | 月度 |
| **F** | 美股QDII动量 | 纳指/标普QDII ETF | 双MA趋势+回撤锁定 | 13.0% | -18.9% | 0.69 | 每周 |
| **G** | CSI500低波价值 | 中证500成分股 | 50%低波+50%低PB | 13.5% | -11.9% | 1.13 | 双周 |

### 策略互补关系

```
               ┌── 策略A (纯债轮动)         ← 防守底仓，极低波动
               │
               ├── 策略B (因子ETF轮动)      ← A股SmartBeta，Regime自适应
               │
               ├── 策略C (跨境ETF动量)      ← 港股独立配置
资金分配 ──────┤
               ├── 策略D (CSI300低波) ─┐
               │                       ├── 成分股策略，数据无后视偏差
               ├── 策略G (CSI500低波) ─┘
               │
               ├── 策略E (逆向价值)         ← 28只龙头中选低估股（有后视偏差风险）
               │
               └── 策略F (美股QDII)         ← 美股配置，趋势跟踪
```

---

## 快速开始

### 环境要求

```bash
Python 3.8+
pip3 install pandas numpy requests baostock
```

- `baostock`: 策略D和G需要（获取沪深300/中证500历史成分股数据）
- 其他策略仅需 `pandas numpy requests`

### 运行信号

```bash
git clone https://github.com/sarahmitchell-cpu/etf-trader.git
cd etf-trader

# 策略A - 纯债ETF周度信号
python3 strategy_a_weekly_signal.py

# 策略B - 因子ETF周度信号
python3 strategy_b_weekly_signal.py

# 策略C - 跨境ETF周度信号
python3 strategy_c_weekly_signal.py

# 策略D - CSI300低波动信号（双周）
python3 strategy_d_weekly_signal.py

# 策略E - 逆向价值信号（月度）
python3 strategy_e_weekly_signal.py

# 策略F - 美股QDII周度信号
python3 strategy_f_weekly_signal.py

# 策略G - CSI500低波价值信号（双周）
python3 strategy_g_weekly_signal.py
```

### 运行回测

```bash
# 各策略均支持 --backtest 参数
python3 strategy_b_weekly_signal.py --backtest
python3 strategy_d_weekly_signal.py --backtest
python3 strategy_g_weekly_signal.py --backtest
# ...
```

### JSON输出

```bash
# 各策略均支持 --json 输出结构化信号
python3 strategy_a_weekly_signal.py --json
python3 strategy_b_weekly_signal.py --json
# ...
```

---

## 策略详细说明

### 策略A: 纯债ETF动量轮动

**逻辑**：在4只债券ETF中按8周动量排名，集中配置前3只（50/30/20权重）。可选MA防守信号（当前关闭）。

**ETF池**：
| 指数 | ETF | 代码 |
|------|-----|------|
| 十年国债 (H11009) | 国泰十年国债ETF | 511260 |
| 城投债 (H11015) | 海富通城投债ETF | 511220 |
| 信用债 (H11073) | 平安公司债ETF | 511030 |
| 短债 (H11006) | 海富通短融ETF | 511360 |

**参数**: `momentum_lookback=8, weights=[0.50, 0.30, 0.20, 0.00], txn_cost=5bp`

**回测 (15年)**: CAGR 4.1%, MDD -2.1%, Sharpe 1.045, Calmar 1.97

### 策略B: 4因子ETF轮动

**逻辑**：4只SmartBeta ETF按8周动量排名 → 50/30/20/0集中配置 → Regime判断（牛/熊/过渡）调整仓位 → 波动率缩放动态调节。

**ETF池**：红利低波(515080)、自由现金流(159201)、国信价值(512040)、创业成长(159967)

**参数**: `momentum_lookback=8, vol_target=15%, regime: MA13w/MA40w`

**回测 (11年)**: CAGR 15.6%, MDD -18.3%, Sharpe 1.157

### 策略C: 跨境ETF纯动量

**逻辑**：3只恒生系ETF取4周动量最强的1只，100%持仓。

**ETF池**：恒生高股息(159726)、恒生科技(513180)、恒生医疗(513060)

**回测 (4.2年)**: CAGR 17.0%, MDD -23.3%, Sharpe 0.70

### 策略D: CSI300低波动TOP10

**逻辑**：从沪深300当期真实成分股中，选20周年化波动率最低的10只，等权配置，双周调仓。使用baostock获取历史成分股数据，消除生存偏差。

**参数**: `vol_lookback=20w, top_n=10, txn_cost=8bp`

**回测 (15年+)**: CAGR 13.9%, MDD -8.5%, Calmar 1.64

### 策略E: 逆向价值+质量

**逻辑**：在28只A股龙头中，选10周跌幅最大的（逆向）× 低波质量过滤，取Top4等权配置。过滤4周跌幅>20%的暴跌股。

**参数**: `value_lookback=10w, quality_weight=0.2, top_n=4, txn_cost=8bp`

**回测**: CAGR 22.0%, MDD -17.6%, Sharpe 0.937 (⚠ 有后视偏差风险)

### 策略F: 美股QDII动量轮动

**逻辑**：纳指100 vs 标普500按4周动量排名。双MA趋势判断（MA8w/MA20w），纳指低于MA20w时进入防守，全仓短债ETF。回撤超-8%锁定8周。

**ETF池**：纳指100(159941)、标普500(513500)、短债(511360)

**参数**: `momentum_lookback=4, ma_fast=8, ma_slow=20, dd_limit=-8%, dd_lockout=8w`

**回测 (15年)**: CAGR 13.0%, MDD -18.9%, Sharpe 0.809

### 策略G: CSI500低波价值

**逻辑**：从中证500当期真实成分股中，综合50%低波动+50%低PB因子打分，选Top10等权配置。使用baostock历史成分股，消除生存偏差。

**参数**: `vol_lookback=12w, factor_weights={low_vol:0.5, low_pb:0.5}, top_n=10, txn_cost=8bp`

**回测 (15年+)**: CAGR 13.5%, MDD -11.9%, Calmar 1.13

---

## 复现说明

### 数据源

| 策略 | 数据源 | 说明 |
|------|-------|------|
| A | CSIndex API | 中证指数公司官方债券指数日线 |
| B | CSIndex API (主) / 东方财富 (备) | 全收益指数日线 |
| C | Yahoo Finance API | v8 chart endpoint，日线 |
| D | baostock + Yahoo Finance | baostock获取成分股，Yahoo获取行情 |
| E | Yahoo Finance API | v8 chart endpoint，周线 |
| F | CSIndex API | 中证纳斯达克100/标普500指数 |
| G | baostock | 成分股+行情+基本面(PB) |

**缓存策略**：所有数据缓存3天，过期自动刷新。API不可用时使用过期缓存兜底。

### 完整复现步骤

```bash
# 1. 克隆代码
git clone https://github.com/sarahmitchell-cpu/etf-trader.git
cd etf-trader

# 2. 安装依赖
pip3 install pandas numpy requests baostock

# 3. 运行回测（自动下载数据）
python3 strategy_a_weekly_signal.py --backtest  # 策略A
python3 strategy_b_weekly_signal.py --backtest  # 策略B
python3 strategy_c_weekly_signal.py --backtest  # 策略C
python3 strategy_d_weekly_signal.py --backtest  # 策略D
python3 strategy_e_weekly_signal.py --backtest  # 策略E
python3 strategy_f_weekly_signal.py --backtest  # 策略F
python3 strategy_g_weekly_signal.py --backtest  # 策略G

# 4. 生成JSON信号
python3 strategy_b_weekly_signal.py --json > signal_b.json

# 5. 策略D/G首次运行需baostock下载成分股数据，约需5分钟
```

### 回测数据

`data/` 目录包含预缓存的历史数据和回测结果：
- `*_daily.csv` / `*_weekly.csv`: 指数/个股历史行情
- `*_backtest.json` / `*_backtest_result.json`: 回测结果
- `*_latest_signal.json`: 最新信号
- `baostock_cache/`: 沪深300/中证500历史成分股缓存
- `fundamental_cache/`: 基本面数据缓存(PB等)

---

## 目录结构

```
etf-trader/
│
├── README.md                          # 策略总览与复现说明
│
├── 策略信号脚本 (7套策略)
│   ├── strategy_a_weekly_signal.py    # A: 纯债ETF轮动
│   ├── strategy_b_weekly_signal.py    # B: 因子ETF轮动 (v6)
│   ├── strategy_c_weekly_signal.py    # C: 跨境ETF纯动量
│   ├── strategy_d_weekly_signal.py    # D: CSI300低波动TOP10
│   ├── strategy_e_weekly_signal.py    # E: 逆向价值+质量
│   ├── strategy_f_weekly_signal.py    # F: 美股QDII动量
│   ├── strategy_g_weekly_signal.py    # G: CSI500低波价值
│   └── stock_data_common.py          # D/E共用数据模块
│
├── docs/                              # 详细策略文档
│   ├── strategy_b.md
│   ├── strategy_c.md
│   ├── strategy_d.md
│   ├── strategy_d_v1.md              # D v1 (旧版28龙头)
│   ├── strategy_e.md
│   └── strategy_g.md
│
├── research/                          # 研究&优化脚本
│   ├── strategy_d_research.py
│   ├── csi300_multifactor_research.py
│   ├── csi500_*.py                   # CSI500因子研究系列
│   ├── strategy_de_optimizer*.py
│   └── astock_10x_*.py              # A股筛选研究
│
├── data/                              # 数据缓存 (自动更新)
│   ├── *_daily.csv                    # 指数日线数据
│   ├── *_weekly.csv                   # 个股周线数据
│   ├── baostock_cache/                # 成分股历史数据
│   ├── fundamental_cache/             # 基本面数据
│   ├── *_latest_signal.json           # 各策略最新信号
│   └── *_backtest*.json               # 回测结果
│
├── archived/                          # 历史版本/废弃脚本
│   └── (sector_rotation_v*.py etc.)
│
└── generate_position_report_ppt.py    # PPT建仓报告生成器
```

---

## 关键设计原则

### 1. 无前瞻偏差 (No Look-Ahead Bias)

- **策略D/G**：使用baostock历史成分股API，回测每期仅使用该期真实成分股，完全消除生存偏差
- **所有策略**：T周数据做决策 → T+1周收益计算

### 2. 包含交易成本

| 策略 | 单边成本 | 说明 |
|------|---------|------|
| A | 5bp | 债券ETF低成本 |
| B | 10bp | A股ETF（佣金+冲击） |
| C | 8bp | 跨境ETF |
| D/E | 8bp | A股个股 |
| F | 15bp | QDII ETF（含汇兑） |
| G | 8bp | A股个股 |

### 3. Walk-Forward验证

策略D和E经过5折walk-forward交叉验证。

---

## 模拟盘

所有7策略均设模拟盘，每策略100万人民币，2026-03-24开盘价建仓。

模拟盘代码：`/sim-portfolio/`（独立仓库）

---

## 已知局限性

- **策略C**：仅4.2年数据，2025港股大涨贡献大部分收益，统计意义有限
- **策略E**：28只龙头是当前视角选出的，存在后视选择偏差
- **策略D/G**：使用baostock成分股数据已消除大部分生存偏差，但不排除数据源本身的微小误差
- **策略F**：15bp交易成本较高，QDII ETF折溢价风险未完全建模

## 风险提示

- 本项目为个人量化研究工具，**不构成任何投资建议**
- 历史回测不代表未来收益
- 请根据自身风险承受能力决定是否参考

---

## 版本历史

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2026-03-24 | v3.0 | 新增策略A(纯债)/F(美股QDII)/G(CSI500低波价值)，策略D改为CSI300低波TOP10，7策略模拟盘启动 |
| 2026-03-22 | v2.2 | 修复B vol scaling，抽取D/E公共模块，重跑全部回测 |
| 2026-03-22 | v2.0 | 新增策略C/D/E，整理仓库结构，添加详细文档 |
| 2026-03-21 | v1.5 | 策略D+E完成5轮优化，walk-forward验证 |
| 2026-03-18 | v1.0 | 初版：宏观定仓+行业轮动+策略B |

---

*由 VisionClaw AI 代理维护 · Sarah Mitchell · sarahmitchell@visionclaw.dev*
