# light_performance 插件设计文档（含 Simulator 设计）

> 文档位置：`docs/light_performance/simulator_design.md`
> 需求来源：`docs/light_performance/HCCLSIM/需求描述`
> 参考实现：`src/plugin/checker/src/framework/task_graph_generator_v3/`（成图原理）、
> `docs/light_performance/HCCLSIM/hccl_sim/`（仿真器原型，**仿真模块的权威基准**）
> 对齐遗留项与待匹配清单：第 10 章
> db 任务 / CCU 指令 → JSON 任务类型映射总表：第 11 章

---

## 1. 需求背景与文档说明

### 1.1 端到端需求

将 light_performance（轻量性能仿真工具）作为一个独立插件集成到当前项目，端到端完成如下流程：

```
hccl-vm 流程执行 ──生成──> task 序列 / 指令等 db 文件
        │
        ▼
light_performance 插件读取 db 文件 ──生成──> 性能仿真所需任务图
        │
        ▼
性能仿真模块根据任务图进行性能仿真
        │
        ▼
输出性能 profiling json 文件
```

### 1.2 子需求分解与章节映射

| 子需求 | 内容 | 状态 | 本文章节 |
|--------|------|------|---------|
| 子需求 1 | 插件工程化：命令行、编译安装、完整插件架构 | 命令行与安装设计已细化（@estimate 一次性执行通道） | 第 3 章 |
| 子需求 2 | 与 hccl-vm 集成：db 文件读取、任务图生成 | **本次详细设计** | 第 4、5 章 |
| 子需求 3 | CCU 模式任务图生成（SQE 队列 + 微码指令序列 → 任务图） | **本次详细设计** | 第 6 章 |
| 子需求 4 | 性能仿真模块（任务图解析、性能仿真） | **本次已按 HCCLSIM 基准对齐收编** | 第 7 章 |
| 子需求 5 | 性能 profiling json 输出（性能指标、任务图） | 概要设计 | 第 8 章 |

> 第 10 章记录本次对齐中的**字段缺口、类型不匹配与其他冲突点**（db 成图路径
> 与 HCCLSIM 仿真模型之间的差异），供后续匹配参考。第 11 章 11.4 节为 v3 采集
> profInfo（`HcclTaskProfInfo` 三字段）与 JSON 显示类型的对照及切换目标态。

### 1.3 本次交付边界

本次需求仅聚焦于 **db 文件的读取解析以及成图模块**：

- **输入**：hccl_vm_install/data 目录下的 db 文件（格式与数据参考 `hccl_vm_install/data/hccl_vm_data.db`）。
- **输出**：performance 模块（hccl_sim）所需的任务图 `DependencyGraph` 和 `SimConfig`。
- **参考代码**：
  - 成图基本逻辑：`src/plugin/checker/src/framework/task_graph_generator_v3/task_graph_generator_v3.cc/h`；
  - CCU 微码指令转任务图：`src/plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/`；
  - AIV 模式成图：暂不考虑。
- **成图范围（核心设计前提）**：db 中为**单 rank 任务数据**，本插件做**单 rank 成图**——
  这是与 checker（多 rank 联合成图）的主要区别之一。原成图过程中的**跨 rank 任务边
  （post-wait 关系）全部忽略**：只关心单 rank 内的 post-wait 边，跨 rank 不考虑
  （详见 5.1、5.4、6.6 节）。
- **与 checker 的关系**：light_performance 成图模块输出结构与 checker 不完全相同，但基本成图与
  指令转换原理相同；前者不需要 checker 后续的内存冲突校验 / 语义校验等操作，因此可做适当简化
  （差异对照见第 5.8 节、第 6.7 节）。

---

## 2. 插件总体设计

### 2.1 端到端数据流与模块划分

```
┌────────────────────────── light_performance 插件 ──────────────────────────┐
│                                                                           │
│  ┌───────────┐   ┌───────────────┐   ┌───────────┐   ┌────────────────┐  │
│  │ db_reader  │──>│ graph_builder │──>│ simulator │──>│ profiling      │  │
│  │ (第 4 章)  │   │ (第 5、6 章)  │   │ (第 7 章) │   │ (第 8 章)      │  │
│  └───────────┘   └───────────────┘   └───────────┘   └────────────────┘  │
│   sqlite 读取       SQE→Task 映射      simpy 离散事件     json + trace    │
│   BLOB 解码         notify 配对        端口带宽共享                        │
│   算子分组          CCU 微码展开       HBM 动态共享                        │
│                                        eid→端口解析                       │
│  输入: data/hccl_vm_data.db (+ data/hccl_sim.db, ranktable.json)          │
│  输出: (SimConfig, DependencyGraph, TopoResolver) → makespan →            │
│        perf_profiling.json (+ graph dump: DOT/JSON/HTML)                  │
└───────────────────────────────────────────────────────────────────────────┘
```

各模块间的数据契约：

```
db_reader     ──OpExecutionGroup(中间数据模型, 4.5 节)──────────>  graph_builder
graph_builder ──(SimConfig, DependencyGraph)─────────────────>  simulator   [本次需求的输出]
simulator     ──result dict(tasks含start/end + makespan + ports)──>  profiling/报告
```

### 2.2 插件目录结构（源码位置：`src/plugin/light_performance`）

```
src/plugin/light_performance/
├── __init__.py
├── __main__.py             # 独立 CLI 入口（第 3.3 节）
├── main.py                 # @estimate 插件一次性执行入口（第 3.1/3.2 节）
├── manifest.json           # 插件清单（name=estimate；随包安装）
├── pipeline.py             # 共享 pipeline：db → 图 → 仿真 → perf_profiling.json
│                           #   （run_perf_sim.py 与 main.py 共用，3.1 节）
├── db_reader.py            # 子需求 2：db 读取与 BLOB 解码（第 4 章）
├── sqe_group_merge.py      # SQE 批合并预处理：同 (stream, sqeGroupId) 批内 UDMA 数据任务并 1 条（5.1.1 节）
├── graph_builder.py        # 子需求 2：SQE → DependencyGraph（第 5 章）
├── graph_dumper.py         # 任务图 dump：DOT + JSON + HTML
├── ccu_graph_builder.py    # 子需求 3：CCU 微码 → 任务图（第 6 章）
├── register_model.py       # CCU 寄存器/CKE 执行模型（第 6.4 节）
├── sim_config.json         # 缺省时延/带宽配置（第 7.2 节，单位 us/GBps；随包安装）
├── bw_profile.json         # 缺省带宽利用率配置（随包安装）
└── simulator/              # 子需求 4：仿真模块（自 HCCLSIM 平移，第 7 章）
    ├── config.py           #   SimConfig（JSON 单位 us/GBps）
    ├── task_model.py       #   Task / TaskType / TRANSFER_TYPES / INVALID
    ├── dependency_graph.py #   DependencyGraph（int 流 id）
    ├── task_timing.py      #   任务时延公式（单点维护）
    ├── bandwidth_profile.py#   带宽利用率表
    ├── topo_resolver.py    #   eid→ports 拓扑解析（ranktable.json）
    ├── input_builder.py    #   JSON 任务文件构建 + load_config
    ├── simulator.py        #   HcclSimulator / NetworkPort / _BandwidthManager
    ├── report.py           #   文本报告 + Chrome trace
    └── __main__.py         #   hccl_sim 兼容 CLI（独立于 db 路径）
```

> `simulator/` 子包从 `docs/light_performance/HCCLSIM/hccl_sim/` 平移，接口以
> HCCLSIM 为基准（第 7 章）。仓库根目录的 `dump_graph.py` 为一键 dump 入口
> （db → DOT/JSON/HTML，可附带仿真时刻）。
> `manifest.json` 与 `main.py`（一次性执行入口，第 3 章）直接置于本包目录内，
> 安装时整目录成为 `plugin/light_performance` 插件目录
> （安装布局见 3.1 节，CMake 规则见 3.4 节）。

### 2.3 与现有插件的关系

| 插件 | 职责 | 与 light_performance 的关系 |
|------|------|---------------------------|
| checker | 通信正确性校验（死锁/内存冲突/语义） | 复用其成图原理；主要区别：checker 为**多 rank 联合成图**，light_performance 为**单 rank 成图**（跨 rank 不建配对边，远端同步任务保留、按固定远端时延近似），且不做校验类后处理 |
| runner | 虚拟运行时（逐任务模拟执行数据面） | 同样消费 opTask/ccuInstr；light_performance 只读不写、不推进 isDone |
| insight | 可视化 | light_performance 的 profiling json / chrome trace 可被其消费（后续） |

> **插件 tag 与目录/模块名**：本插件安装 tag 为 `estimate`（宿主 `@estimate` 命令通道
> 与模式门禁按 tag 硬编码，见 3.1 节），而插件目录与 Python 包名均为
> `light_performance`（`plugin/light_performance`）——tag 对应"宿主路由身份"，
> 目录/包名对应"代码组织身份"。

---

## 3. 插件工程化设计（子需求 1）

### 3.1 插件标识与 manifest.json

**插件 tag 固定为 `estimate`**（而非模块名 light_performance）：宿主 CLI 对该插件做了
硬编码门禁与路径约定——

- `hccl-vm plugin run @estimate` 的执行通道在 `src/cmd/cmd_base_utils.cc:599-680`
  （`RunUserPlugin`）中对 `estimate` 特判：仅 estimate 模式（`sim::IsEstimateMode()`）可用，
  插件目录固定为 `<installRoot>/plugin/light_performance`（tag 与目录名解耦），
  直接读取其 `manifest.json` 并同步执行 `entry` 命令（详见 3.2 节）；
- 模式互斥：`@checker` 在 estimate 模式不可用（`cmd_base_utils.cc:618-624`），
  `@estimate` 在非 estimate 模式不可用——正确性校验与性能仿真分属两种运行模式的收尾动作；
- 插件扫描/常驻模型（`src/plugin/hccl_plugin_manager.cc:94-169`、`src/plugin/hccl_plugin.cc`）
  按 manifest 的 `name` 字段匹配 tag（扫描 `<installRoot>/plugin`，深度 ≤ 2）。

manifest.json（源码与安装均在包目录内：`src/plugin/light_performance/manifest.json`
→ `plugin/light_performance/manifest.json`）：

```json
{
    "name": "estimate",
    "version": "1.0.0",
    "entry": "python3 main.py",
    "dependency": {
        "min_core_version": "1.0.0"
    },
    "setting": {
        "db": "../../data/hccl_vm_data.db",
        "device": null,
        "iter": null,
        "sim_config": "sim_config.json",
        "bw_profile": "bw_profile.json",
        "ranktable": "",
        "out_dir": "data",
        "per_op_mode": true,
        "enable_trace": true,
        "dump_graph": false,
        "max_trace_events": 0
    }
}
```

- `name`：插件 tag，宿主按它路由（`hccl-vm plugin run @estimate`）。
- `entry`：插件目录内的启动命令。@estimate 走一次性同步执行通道（3.2 节），
  执行时 cwd 即插件目录，相对路径以插件目录为基准。
- `setting`：插件私有配置键值，由 main.py 自行读取（仿照 checker 的 `SettingManager`
  自管配置的思路，Python 侧直接 json 解析）。路径类配置相对插件目录（即包目录）解析：
  `db` 缺省 `<installRoot>/data/hccl_vm_data.db`（与 checker 的
  `Loader::LoadOpTaskFile` 同源）；`sim_config`/`bw_profile` 为第 7.2 节格式的
  时延/带宽与利用率配置，缺省取与 manifest 同目录、随包安装的
  `sim_config.json`/`bw_profile.json`；
  `ranktable` 缺省取 db 同目录 `ranktable.json`；
  `device`/`iter` 对应 CLI 的 `--device`/`--iter`；`out_dir` 为输出根目录
  （其下按运行时间戳建子目录）；`per_op_mode` 当前仅支持逐算子（false 时告警
  并按逐算子执行）；`enable_trace`/`dump_graph`/`max_trace_events` 控制附加产物。

源码与安装布局（插件目录即包目录，无嵌套）：

```
源码 = 安装产物结构 (src/plugin/light_performance → hccl_vm_install/plugin/light_performance):
plugin(light_performance)/            # 插件目录 = light_performance 包目录
├── manifest.json                     # 插件清单 (name=estimate)
├── main.py                           # @estimate 一次性执行入口 (复用 pipeline.py)
├── __init__.py / __main__.py / db_reader.py / graph_builder.py / pipeline.py / ...
├── sim_config.json                   # 缺省时延/带宽（第 7.2 节）
├── bw_profile.json                   # 带宽利用率
├── simulator/                        # 仿真子包（第 7 章）
└── data/                             # 运行期输出（perf_profiling.json 等，第 8 章;
                                      #   仅安装树内产生, 源码树无此目录）
```

> main.py 位于包目录内但以脚本方式执行（`python3 main.py`）：其将包的**父目录**
> （`plugin/`，源码树为 `src/plugin/`）加入 `sys.path` 后再
> `from light_performance.pipeline import ...`——包内相对导入不受影响，
> 源码树与安装树布局一致。
> 仓库根目录 `run_perf_sim.py`（手动入口）与 `main.py` 共用同一 pipeline 实现：
> main.py 在其基础上增加"安装树内路径约定 + manifest setting 读取"。
> 源码树调试仍用 run_perf_sim.py（3.3 节），安装树内由 `hccl-vm plugin run @estimate`
> 触发 main.py。

### 3.2 调用链路与进程模型（@estimate 一次性执行）

宿主侧仅需一处路径适配——`@estimate` 的一次性执行通道本已实现
（`cmd_base_utils.cc:599-680`），本次将特判分支中的插件目录由 `plugin/estimate`
改为 `plugin/light_performance`（tag 仍为 `estimate`，目录即包目录）：

```
hccl-vm plugin run @estimate
    │
    ▼
RunUserPlugin("@estimate")                                  # subcmd_plugin.cc:55-57 路由
    ├─ 门禁: 须处于 estimate 模式, 否则报错退出              # cmd_base_utils.cc:611-617
    ├─ 读取 <installRoot>/plugin/light_performance/manifest.json  # 缺文件/缺 entry 字段 → 报错
    └─ system("cd <installRoot>/plugin/light_performance && python3 main.py")   # 同步阻塞执行
       └─ 退出码非 0 → HCCL_VM_ERROR, 命令整体返回失败
```

**长耗时命令的客户端等待策略**（`cmd_base_utils.cc` 的 `StartHostClient`/`ServerListen`）：
会话内 `hccl-vm` 经 POSIX 消息队列发给 host，**host 在 `ParseCommand` 返回（即命令执行完毕）
后才回执**；CCU MS 模式微码展开后任务量可达 10^5 级（实测 192,005 任务、软件仿真约 54s），
远超普通命令的 5s 回执上限。为此：

- 客户端识别 `plugin run @estimate` 后采用长等待上限（缺省 30 分钟，环境变量
  `HVM_CLIENT_RESP_TIMEOUT_MS` 可覆盖），分片等待期间每 30s 向终端打印
  "命令仍在 host 端执行中"提示（首次 5s），超时打印明确告警；其余命令维持 5s 上限。
- 回执携带原命令串（`ACK:<cmd>`），客户端发送命令前先排空回执队列、且只接受与本命令
  严格匹配的回执——避免误读此前超时命令的迟到回执而提前返回（同一命令中断重试时尤甚）。
- host 在仿真起止处向终端打印 `[HVM] @estimate 性能仿真开始/结束` 标记（stdout，随报告
  输出）；该机制保证**提示符在仿真真正结束后才返回**，不再产生"仿真已结束"的误解。

main.py（插件侧）执行流程：

```
RunEstimate():
    setting   ← 解析 manifest.json 的 setting 段（路径相对插件目录解析）
    db        ← setting.db（缺省 <installRoot>/data/hccl_vm_data.db, 经 __file__ 推导 ../../data）
    config    ← setting.sim_config / bw_profile（缺失时用内置默认值, 第 7.2 节）
    ranktable ← db 同目录 ranktable.json（_find_ranktable 缺省规则; 与宿主 InitHvmCommEnv
                写入的 data/ranktable.json 一致, cmd_base_utils.cc:437-450, 无需配置）
    groups    ← db_reader.load_op_executions(db)            # 第 4 章
    topo      ← TopoResolver(ranktable)                     # 第 7.6 节
    for group in groups:                                    # 逐算子（5.1 节）
        config, dep_graph = graph_builder.build(group)      # 第 5、6 章
        result = HcclSimulator(config, dep_graph, topo).run()   # 第 7 章
    profiling.dump(<plugin>/light_performance/data/<时间戳>/perf_profiling.json)   # 第 8 章
```

与 checker 常驻进程模型（stdin 单行 JSON 命令循环）的对照与选择依据：

| 维度 | checker（常驻模型） | estimate（一次性模型） |
|------|--------------------|----------------------|
| 拉起方式 | `plugin install` 时 fork+pipe 常驻（`hccl_plugin.cc:74-156`） | `plugin run` 时 `cd && entry` 同步执行（`cmd_base_utils.cc:655-660`） |
| 触发协议 | stdin JSON：`{"action":"status","payload":{"status":"finish","data_id":...}}` | 无协议：entry 即启动，输入路径全部走约定 + manifest setting |
| data_id | 消息携带（宿主 `DumpData` 生成） | 不传递，输出目录以运行时间戳充当（10.3 节 #13） |
| 生命周期 | stop 指令/管道 EOF/宿主退出时回收 | 进程自然退出，退出码即执行结果 |
| 适配原因 | 校验需由宿主在流程结束时触发，且可能多次运行 | 仿真为离线批处理：消费已落盘的 db，一次跑完即退 |

> **演进方向**（本期不实施）：若后续需要"流程结束由宿主自动触发仿真"（常驻模型），
> main.py 增加 stdin 命令循环即可复用同一 pipeline——宿主的消息通道
> （`SendMessageToPlugin`）与常驻拉起（`RegisterPlugin`）均已就绪，届时只需将
> `RunUserPlugin` 中 @estimate 从特判分支切回通用消息路径。

数据契约（安装树内的固定约定，均与现有机制对齐）：

| 路径 | 来源 | 说明 |
|------|------|------|
| `data/hccl_vm_data.db` | 宿主流程执行期间写入 | 仿真主输入；estimate 模式产出**单 rank 任务数据**（幻影对端），与 4.6/5.1 节单 rank 成图前提天然对齐 |
| `data/ranktable.json` | `InitHvmCommEnv` 写入 | eid→端口解析（7.6 节）；`_find_ranktable` 缺省取 db 同目录，无需显式配置 |
| `plugin/light_performance/sim_config.json`、`bw_profile.json` | CMake 随包目录安装（3.4 节） | 第 7.2 节格式（单位 us/GBps） |
| `plugin/light_performance/data/<时间戳>/` | main.py 运行时创建 | 输出目录。**不得落在 `data/` 下**：宿主每次 start 会 `remove_all("data")`（`src/main.cc:32`），data 目录内容不跨会话保留 |

### 3.3 命令行设计

宿主集成模式（`plugin run` 即仿真入口，取代手动执行 run_perf_sim.py）：

```bash
# ① 以评估模式启动环境（二选一）
hccl-vm start <topo> --estimate             # 交互模式, PS1 显示 (estimate)$>
hccl-vm run <topo> --estimate <算例命令>     # one-shot 模式
# （已启动的会话内也可 hccl-vm mode estimate 切换, subcmd_mode.cc）

# ② 运行业务 → 产出 data/hccl_vm_data.db（单 rank 任务清单）

# ③ 一键性能仿真（仅 estimate 模式可用, 3.2 节门禁）
hccl-vm plugin run @estimate
# 产物: plugin/light_performance/data/<时间戳>/perf_profiling.json (+ trace / graph dump)
```

- 一次性执行通道不透传命令行参数；仿真参数（目标设备、opIter 过滤、trace/dump 开关等）
  经 manifest `setting` 段配置（3.1 节），改配置后重跑 `plugin run @estimate` 即生效。
- `plugin install @estimate` 亦可执行（走 `RegisterPlugin` 常驻拉起通道），main.py 一次性
  跑完即退出属可接受的退化行为（10.3 节 #14）；常规使用无需 install。

独立调试模式（不依赖宿主，便于离线分析 db；已实现）：

```bash
# 仿真 + 报告（可选 trace 导出、任务图 dump）
python3 -m light_performance \
    --db hccl_vm_install/data/hccl_vm_data.db \
    [--config sim_config.json] [--bw-profile bw_profile.json] \
    [--ranktable ranktable.json] [--device 1] [--iter 0] \
    [--trace trace.json] [--dump-dir graph_dump] \
    [--max-trace-events 0] [-v] [-s]

# 一键仿真（run_perf_sim.py, 输出 perf_profiling.json; 与 main.py 共用 pipeline）
python3 run_perf_sim.py hccl_vm_install/data/hccl_vm_data.db \
    [-o perf_profiling.json] [--config ...] [--bw-profile ...] [--trace-dir ...] [--dump-dir ...]

# 一键任务图 dump（DOT + JSON，可选 HTML；--no-sim 跳过仿真）
python3 dump_graph.py hccl_vm_install/data/hccl_vm_data.db [--html] [--no-sim]
```

| 参数 | 说明 |
|------|------|
| `--db` | hccl_vm_data.db 路径（必填） |
| `--config` | 时延/带宽配置（第 7.2 节格式，单位 us/GBps；缺省用内置默认值） |
| `--bw-profile` | 带宽利用率配置（缺省利用率为 100%） |
| `--ranktable` | 拓扑文件，用于 eid → 物理端口解析（缺省取 db 同目录 `ranktable.json`；存在走网任务且无拓扑时**跳过仿真仅 dump**） |
| `--device` | 单 rank 成图的目标设备（deviceId）；缺省取 opDetails 中首个设备，仅当 db 含多设备数据时需要 |
| `--iter` | 仅仿真指定 opIter 的算子（缺省全部） |
| `--trace` | 导出 Chrome tracing JSON（多算子时按 `_<commId>_<opIter>` 分文件） |
| `--dump-dir` | 任务图 dump 目录（每算子输出 .dot/.json/.html，含仿真时刻） |
| `-v` / `-s` | 任务级明细 / 仅汇总 |

### 3.4 编译与安装设计（hccl_vm_install/plugin/light_performance）

在 `src/plugin/CMakeLists.txt` 新增 install 规则（仿照 insight 的目录安装先例；
Python 插件无需编译目标。插件目录即包目录，manifest.json/main.py/缺省配置均在
包目录内，单条 DIRECTORY 规则即可携带全部内容）：

```cmake
# estimate 性能仿真插件: 插件目录即 light_performance 包目录, 整目录安装
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/light_performance"
    DESTINATION "plugin"
    FILE_PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
    PATTERN "__pycache__" EXCLUDE
    PATTERN "data" EXCLUDE)      # 运行期输出目录不进安装包
```

与现有插件安装方式的对照：

| 插件 | 安装方式 | 产物 |
|------|---------|------|
| checker | `install(TARGETS checker DESTINATION plugin/checker)` + `install(FILES manifest.json)`（`src/plugin/checker/CMakeLists.txt:67-82`） | 编译产物 + manifest |
| insight | `install(DIRECTORY insight DESTINATION plugin/visualization PATTERN frontend EXCLUDE)`（`src/plugin/CMakeLists.txt:26-39`） | Python 目录 + manifest |
| **estimate（本次）** | `install(DIRECTORY light_performance DESTINATION plugin)`（单规则；manifest/main.py/缺省配置均在包目录内随包安装） | `plugin/light_performance/`：Python 包 + manifest（插件目录与包目录合一，无嵌套） |

构建与验证：`bash build.sh`（内部为 `cmake && make && make install`，安装前缀默认为
仓库内 `hccl_vm_install/`，`CMakeLists.txt:89-92`）后检查安装产物：

```
hccl_vm_install/plugin/light_performance/
├── manifest.json
├── main.py
├── __init__.py / __main__.py / db_reader.py / graph_builder.py / pipeline.py / ...
├── sim_config.json / bw_profile.json
└── simulator/ ...
```

- 打包路径：`build.sh --pkg` 会将整个 `hccl_vm_install` 打成 tar 包，estimate 插件随之分发。
- 依赖项：python3（≥3.8，含 sqlite3 标准库）与 **simpy**（离散事件仿真引擎，当前环境
  4.1.2，`simulator/simulator.py` import）；部署环境需预装（insight 插件同样依赖 python3，
  约定一致），见 10.3 节 #16。

---

## 4. db 文件读取与解析（子需求 2 · 上半，本次重点）

### 4.1 db 定位与打开

- 路径固定为 `<installRoot>/data/hccl_vm_data.db`（与 checker 的 `Loader::LoadOpTaskFile` 一致）。
- 只读打开：Python `sqlite3` 标准库，`mode=ro` URI，避免与运行中的 hccl-vm 写入冲突：
  ```python
  conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
  ```
- 辅助库 `hccl_sim.db`（同目录）：仅用于把 `opDetails.commId` 解析为通信域身份
  （`Communicator` 表，`comm_id[128] + comm_hash` 唯一标识），可选。

### 4.2 表清单与关联模型

hccl_vm_data.db 由 `OpDbOps::InitSchema()` 建表（`src/runnerdb/modeldb/db_hccl_op_db_ops.cc:76-184`），
light_performance 消费的表如下：

| 表 | 主键/关联 | 内容 | 用途 |
|----|----------|------|------|
| `opDetails` | `id`；`pid`→opTask 表名 | 每算子每 rank 一行：pid/deviceId/rankId/commId/opIter/syncIter/streamId(主流)/root/rankSize/srcRank/dstRank + `opDetail` BLOB（算子参数）+ `algorithmName`（算法选择结果，DFX 劫持/日志双路径截取回填，见下方注） | 算子分组、主流识别、算子参数、算法选择结果 |
| `opMemInfo` | `opDetailId` → opDetails.id | input/output/ccl 三段内存的地址与大小 | 方向判定（第 5.5 节）、数据量校验 |
| `opTask_P_<pid>` | `opDetailId` → opDetails.id | SQE 任务队列：deviceId/streamId/taskType/isDone + `opTaskMeta` BLOB | **成图主输入** |
| `ccuChannels` | `(deviceId, dieId, channelId)` | CCU 通道：src/dst 的 die/device/rank + leid/reid(16B EID) + protocol + jettyNum + jettyId 数组 | CCU 通道 → 对端/jetty 映射（第 6 章） |
| `ccuInstrRes` | `(deviceId, dieId)` | 每 die 的微码指令空间：instrCount + instrSpace BLOB（32K×32B） | CCU 微码源（第 6 章） |
| `ccuInstr` | `ccuInstrResId` → ccuInstrRes.id | 每次微码下发一行（startId/instrInfoSize） | 仅用于感知"有新微码下发"，解码不依赖 |
| `syncRecords` | — | host 侧 stream 同步记录 | 本次不用 |
| `JettyMaps` | `opDetailId` | 按算子记录的 eid 映射 | 备用（ccuChannels 覆盖时不用） |

> 表结构定义见 `include/runnerdb/sim_op_db_types.h`；写入方语义见 `src/runnerdb/tracedb/db_sim_op_db_ops.cc`。

> **algorithmName 列的来源（算法选择结果截取）**：双路径截取，均 last-wins 回填
> `sim::UpdateOpDetailAlgName`，互为备份、写入同值。
>
> 1. **DFX 劫持桩（主路径）**：业务在算法选择完成后逐算子调用
>    `HcclDfxRegOpInfoByCommId(commName, &HcclDfxOpInfoCompat)`（hccl 仓
>    op_common.cc），其 `algTag` 携带所选算法名（格式
>    `{opType_commName}_{algName}_{host|device}[_{dataType}...]`，图模式
>    `Graph_{algName}_{...}`）。Level=2 模式（默认）仅挂 level0+level2、
>    level1 的 dfx_stub 不在进程内，`src/proxy/level2/hcomm_api_stub.cc` 以
>    level2 强符号抢占解析（preload 序先于业务经 python 插件链加载的真实
>    libhcomm.so），解析 algTag 中的算法名回填后经
>    `DlsymRealWithFallback`（RTLD_NEXT + libhcomm.so SONAME 回退 + dladdr
>    防自调）**转发真实实现，透传返回值，不改调用语义**。结构体布局按
>    hccl_res_dl.h 复刻（`static_assert` 锁 algTag 偏移 68），内容校验
>    （commId 前缀 + host/device 后缀）过滤 op-begin 等无算法名的注册、
>    兜底业务头文件漂移。Level=1 模式下本桩被 level1 dfx_stub 遮蔽（行为
>    同旧版），截取退回日志路径。
> 2. **日志截取（备份路径）**：工具即业务日志底层接口的实现者（业务
>    `DlogRecord` weak 符号解析到 `src/proxy/level2/log_stub.cc`）。level0 桩
>    `RecordOpDbInfo` 先插入 opDetails 行（置 runnerdb 全局 `g_currOpDetailId`）再调用
>    真实业务，业务内选择器输出的 INFO 级算法选择日志（`[HcclGetAlgRes] algName=` /
>    `SelectMinCost: selected algName=` / 旧选择器 `the selected algo type is ` 三类标记，
>    同一算子多次匹配后写覆盖）经 `DlogPrintStub` 入口匹配提取。截取在消息进入工具日志输出之前
>    执行（免疫 `log_config.yaml` 级别），proxy 层同时提供 `acllogCheckDebugLevel`
>    恒放行 stub（免疫业务侧日志级别门槛）。
>
> 旧库由 `OpDbOps::InitSchema()` 自动
> `ALTER TABLE` 补列；插件读取对无列旧库回退空值。

**算子分组模型**：一个算子实例 = 同一 `(commId, opIter)` 下 rankSize 条 opDetails（各 rank 一条），
每条挂一份 opMemInfo 和一组任务（`WHERE opDetailId=?`，表名由该行 `pid` 列推导为 `opTask_P_<pid>`，
参考 checker `QueryCompositeOpDetailByOpIter`）。

### 4.3 opTask 动态表与多 pid 处理

- 每个写任务进程各建一张表 `"opTask_P_" + getpid()`；AICPU 设备侧进程用 `getppid()` 与 host 同表。
- **发现全部任务表**（必须枚举，多 pid 是常态；早期样例 db 即有 `opTask_P_477776`/`opTask_P_477780` 两张，当前样例为单张）：
  ```sql
  SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'opTask_P_%'
  ```
- 消费顺序：同一算子内按 `id` 自增序（= SQE 下发顺序，这是 notify 先到先得配对的基础）。

### 4.4 BLOB 解码规范（pack(1)，小端）

所有 BLOB 均为 C 结构体原样 memcpy（无序列化框架），Python 侧用 `struct` 模块解码。
以下布局均已用 `hccl_vm_install/data/hccl_vm_data.db` 实测验证（v5 采集起样例
opTaskMeta 长度 190B，历史样例 189B/188B）。

#### 4.4.1 opTaskMeta（190 字节）

结构体定义与写入方共享：`include/sim_profiling_defs.h`（profiling 对账信息单源定义）
及 `src/plugin/checker/header/internal/task_meta_defs.h`（HcclTaskMetaData）。

v5 版采集结构（较 v4 的 189B 版本）：`sqeGroupId` 后新增 `taskSrc`（1 字节）——
任务来源标记（`HcclTaskSrc`：0=默认 SQE/aclrt 原生路径，1=DPU 即 host CPU RoCE
`ibv_dpu_stub` 采集路径），供性能仿真 DPU 序列合并 pass 按来源精确切分
（5.1.2 节），taskData 起始偏移 69 → 70。

v4 版采集结构（较 v3 的 188B 版本）：`jettyId` 后新增 `sqeGroupId`（1 字节）——UDMA
SQE 批分组标记（一次 doorbell 内全部 WQE 同组），供性能仿真入口按批合并
（5.1.1 节），taskData 起始偏移 68 → 69。组号语义：**环形递增门铃批号 1..255**
（255 回绕，采集侧 `device_sqe_parse_stub.cc` 维护全局 `g_currSqeGroupId`，
`% 255 + 1`）——批号在 255 窗口内唯一，其他算子的门铃虽不在本算子任务列表中
可见、但同样消耗批号，故列表内同流同号必为同一门铃；**0 保留为"未打标"**
（HcclTaskMetaData 构造默认），计数器永不取 0，合并入口将 `sqeGroupId <= 0`
一律视为不参与（5.1.1 节）。

v3 版采集结构（较 v2 的 185B 版本）：`taskType` 后新增 `HcclTaskProfInfo` 子结构（3 连续字节，
checker/runner 作为不透明整体透传）：`profTaskType`（对齐 hcomm TaskParamType 值域 0~22）与
`linkType`/`transportType`（对齐 hcomm DfxLinkType 值域 0~9 / DfxTransportType 值域 0~4，
用于与 profiling 数据对账），taskData 起始偏移 65 → 68。

> **实现状态**：`db_reader.py` 按 BLOB 长度自动识别布局——`>= 190B` 按 v5
> （`<BBBBQIQQIBB16s16s`，taskSrc 解码入 `HcclTaskMetaData.src_type`；该分支
> **必须先于 v4 判定**，否则 190B 会先命中 `>= 189` 的 v4 分支导致 eid/taskData
> 偏移错位），`>= 189B` 按 v4
> （`<BBBBQIQQIB16s16s`，sqeGroupId 解码入 `RawTask.sqe_group_id`，src_type=0），
> `188B` 按 v3（`<BBBBQIQQI16s16s`，sqe_group_id 填 -1 不参与合并），
> `185~187B` 按旧 v2（`<BQIQQI16s16s`，profInfo 填 0xFF=UNKNOWN）向后兼容，
> `< 185B` 丢弃告警。profInfo 的消费（`task_type` 显示名 + link/transport）
> **已落地**（#18，第 11.4 节）。

v2 版采集结构（较早期 169B 版本）：头部新增 `lcEid[16]`（显式本端 EID），
taskData 各成员曾补充 `srcRankId/dstRankId` 字段。

> **rank 字段回退：srcRankId/dstRankId 已废弃（现行约定）**：taskData 各成员曾补充的
> `srcRankId/dstRankId` 字段已回退删除，统一改用 `srcDeviceId/dstDeviceId`——采集端
> （estimate 模式，`device_sqe_parse_stub.cc` 等）把通信域 rankId 直接写入
> `srcDeviceId/dstDeviceId` 字段，即 **deviceId 字段承载 rankId**（对端 rank 未解析
> 时为 `0xFFFFFFFF`）。`db_reader.py` 解包后直接以 `src_rank_id/dst_rank_id` 命名
> 消费（读 `srcDeviceId` 当 `srcRankId` 使用）。此为硬切换：字段回退前采集的旧
> BLOB（rank/device 双字段布局）不再支持。

```
头部 70B:  struct.unpack_from("<BBBBQIQQIBB16s16s", blob, 0)
┌──────┬──────┬──────────────┬────────┬───────────┬──────────────────────┐
│ 偏移 │ 长度 │ 字段          │ 类型   │ 哨兵值     │ 说明                  │
├──────┼──────┼──────────────┼────────┼───────────┼──────────────────────┤
│ 0    │ 1    │ taskType     │ u8 枚举│ —         │ 与表列 taskType 冗余一致 │
│ 1~3  │ 3    │ profInfo     │ 子结构 │ 0xFF      │ HcclTaskProfInfo:      │
│      │  1   │ .profTaskType│ u8 枚举│ 0xFF      │ hcomm TaskParamType 对齐 │
│      │  1   │ .linkType    │ u8 枚举│ 0xFF      │ hcomm DfxLinkType 对齐 │
│      │  1   │ .transportType│u8 枚举│ 0xFF      │ hcomm DfxTransportType 对齐 │
│ 4    │ 8    │ commId       │ u64    │ —         │ 通信域 ID              │
│ 12   │ 4    │ rankId       │ u32    │ 0xFFFFFFFF│ 缺省时经 device→rank 回填 │
│ 16   │ 8    │ deviceId     │ u64    │ —         │ 物理设备 ID            │
│ 24   │ 8    │ streamId     │ u64    │ —         │ 流 ID                 │
│ 32   │ 4    │ jettyId      │ u32    │ 0xFFFFFFFF│ 无效 = 不走网           │
│ 36   │ 1    │ sqeGroupId   │ u8     │ 0         │ 门铃批号 1..255 环形递增（v4 新增，5.1.1 节）；0 = 未打标│
│ 37   │ 1    │ taskSrc      │ u8 枚举│ 0         │ 任务来源（v5 新增，11.4.6 节）：0=默认, 1=DPU(host ibv)│
│ 38   │ 16   │ rmEid[16]    │ bytes  │ 全 0      │ 远端 EID              │
│ 54   │ 16   │ lcEid[16]    │ bytes  │ 全 0      │ 本端 EID（v2 新增）     │
│ 70   │ 120  │ taskData     │ union  │ 按 taskType 解释，见下          │
└──────┴──────┴──────────────┴────────┴───────────┴──────────────────────┘
```

taskData union 内部布局（相对偏移 69）：

| taskType | 成员 | 格式串 | 字段（按序） |
|----------|------|--------|------|
| 0/1（notify） | `NotifyTask` | `<IQIBB` | srcDeviceId(承载rankId)、notifyId、dstDeviceId(承载rankId)、notifyCount、protocol |
| 3（transMem） | `TransMemTask` | `<IQIQQB` | srcDeviceId(承载rankId)、srcOffset、dstDeviceId(承载rankId)、dstOffset、len、protocol |
| 2（reduce） | `ReduceTask` | `<IQIQQBBBB` | srcDeviceId(承载rankId)、srcOffset、dstDeviceId(承载rankId)、dstOffset、dataCount(实为字节 size)、dataType、reduceOp、protocol、resv |
| 4（ccu） | `CcuTask` | `<BBHHHII` + `<13Q` | dieId、missionId、timeout、instStartId、instCnt、key、argSize、args[13] |
| 5（aiv） | `AivTask` | `<Q` | launchIdx |
| 6（sync） | `SyncStreamTask` | `<Q` | syncIdx |
| 7（writeWithNotify） | `WriteWithNotifyTask` | `<IQIQQQBBBBB` | srcDeviceId(承载rankId)、srcOffset、dstDeviceId(承载rankId)、dstOffset、len、notifyId、isReduce、dataType、reduceOp、protocol、notifyCount |

> `len` 与 `dataCount` 均为**字节 size**（reduce 的字段名有历史原因，实际按 size 使用）。
> `WriteWithNotifyTask.len` 同为字节 size；`isReduce=1` 表示写载荷为规约数据
> （`dataType`/`reduceOp` 此时有效），映射为单一 `Write_With_Notify` 任务（5.2 节）。

#### 4.4.2 taskType 枚举全表

`HccLTaskMetaType`（`include/sim_common_defs.h:184-193`）：

| 值 | 枚举 | light_performance 处理 |
|----|------|----------------------|
| 0 | `NOTIFY_WAIT` | → local_wait / remote_wait（第 5.2 节） |
| 1 | `NOTIFY_RECORD` | → local_record / remote_record |
| 2 | `REDUCE` | → memcpy / local_copy（规约计算近似，第 5.7 节） |
| 3 | `MEM_CPY` | → memcpy / local_copy（按 jettyId 判定） |
| 4 | `CCU_GRAPH` | → CCU 子图展开（第 6 章） |
| 5 | `AIV_GRAPH` | 暂不支持，跳过并告警 |
| 6 | `SYNC_STREAM` | → 跳过不生成节点（第 5.6 节） |
| 7 | `WRITE_WITH_NOTIFY` | → 单任务映射 `Write_With_Notify`（网络写 + 附带 notify，不拆分，第 5.2 节） |

#### 4.4.3 opDetails.opDetail（28 字节）

`OpDetails`（`task_meta_defs.h:141-165`）：`<HHH` 头 + union：

```
opType u16@0 | dataType u16@2 | reduceType u16@4
opV1: count u64@6 | dataType u16@14 | reduceOpType u32@16          (合计 28B)
opV2: sendCount u64@6 | sendDataType u16@14 | recvCount u64@16
      | recvDataType u16@24 | extInfo u16@26                       (AllToallV 类)
```

opType 即 `HcclCMDType`（ALLREDUCE=2、ALLGATHER=6、REDUCE_SCATTER=7、ALLTOALL=10 等，
完整枚举见 `src/plugin/checker/header/external/hccl_types.h:162-189`），用于 profiling 报告的算子标注。
`opExtInfo` BLOB（可选）存放 V 算子 counts/displs、AllToallV 发送矩阵等，本次仅透传。

V1/V2 布局按 opType 区分（level0 桩写入侧约定，`hccl_op_stub.cc`）：
**V2 仅 AllToAll(10)/AllToAllV(8)/AllToAllVC(9)**（`BuildOpDetailsV2`，sendCount/recvCount
为本 rank 发送/接收总元素数）；**AllGatherV(16)/ReduceScatterV(17) 为 V1 + opExtInfo**
（count = 本 rank 侧元素数）。db_reader 解码时按 opType 选布局，V2 算子解出收发
两侧元素数与类型（`send_count/recv_count/send_data_type/recv_data_type`），
`data_size/data_type` 沿用发送侧保持旧消费方兼容、`reduce_op` 置 0。

#### 4.4.4 ccuChannels 的 BLOB

| 字段 | 格式 | 说明 |
|------|------|------|
| `leid` / `reid` | 各 16B 原始字节 | 源/目的端点 EID（`URMA_EID_LEN=16`） |
| `jettyId` | `jettyNum × u32` 小端 | 连续 jetty 号数组（C 结构预留 `u32[64]`；样例实测 jettyNum=1、jettyId=(1024,)） |
| `protocol` | int 列 | CommProtocol：HCCS=0/TCP=1/ROCE=2/UB_CTP=3/UB_TP=4/PCIE=5/SIO=6 |

注意：`srcRankId/dstRankId` 两列实际填的是 device_id（通道建立早于通信域，见写入方
`hccp_ccu_stub.cc:401-405` 注释），rank 语义需经 opDetails 修正。

#### 4.4.5 ccuInstrRes 的 instrSpace

- `instrSpace` = 一个 die 的**完整微码指令空间快照**：`32K 条 × 32B = 1 MiB`
  （`CCU_INSTRUCTION_NUM=32*1024`、`CCU_INSTRUCTION_SIZE=32`，`include/sim_common_defs.h:35-36`）。
- 第 i 条指令位于字节偏移 `i*32`；`instrCount = max(历史值, startId+instrCnt)`，**允许空洞**。
- 读取时只取前 `instrCount` 条；CCU 任务的 `instStartId/instCnt` 直接作下标区间使用。
- `ccuInstr` 表仅为运行感知用（runner 轮询 `COUNT(ccuInstr)+COUNT(ccuInstrRes)` 变化），
  light_performance 解码不依赖该表。

### 4.5 中间数据模型（db_reader 输出）

```python
@dataclass
class OpExecutionGroup:          # 一个算子实例（单 rank 视角：仅目标设备的任务数据）
    comm_id: int
    op_iter: int
    op_type: int                 # HcclCMDType
    rank_size: int
    main_stream_ids: dict[int, int]      # deviceId → 主流 streamId（opDetails.streamId）
    tasks: list[RawTask]         # 该算子全部 SQE，全局有序
    mem_info: dict[int, MemInfo]         # deviceId → input/output/ccl 区间
    channels: dict[tuple, ChannelInfo]   # (srcDev, srcDie, channelId) → 通道信息
    instr_space: dict[tuple, list[bytes]]# (deviceId, dieId) → 32B 指令列表
    algorithm_name: str          # 算法选择结果（opDetails.algorithmName，DFX/日志双路径截取回填；旧库空串）
    send_count / recv_count: int # 收发两侧元素数（V2 布局算子；V1 与 count 一致）
    send_data_type / recv_data_type: int # 收发两侧数据类型（同上）

@dataclass
class RawTask:
    global_id: int               # 跨 opTask_P_* 表的全局序（按 表 pid 数值 + 表内 id 排序）
    op_detail_id: int
    device_id: int
    stream_id: int
    meta: HcclTaskMetaData       # 4.4.1 解码结果（dataclass 形态）
```

`meta` 含 v2 新增字段：头部 `lc_eid`（显式本端 EID，5.5 节本端 eid 解析优先使用）；
NotifyTaskData/TransMemTaskData 均携带 `src_rank_id/dst_rank_id`
（estimate 采集约定下解包自 deviceId 字段承载的 rankId 值，见 4.4.1 节 rank 字段回退）。

### 4.6 数据完整性校验

对每个 `OpExecutionGroup`：

1. **单 rank 过滤（设计前提）**：性能插件只消费**单 rank 任务数据**（样例 db 仅 device 1，
   属设计常态而非数据缺失）。若 db 含多设备数据，按目标 deviceId（`--rank`，缺省取首个设备）
   过滤 opDetails 及其任务；`rankSize` 仅作为元数据透传，不要求 `len(opDetails 行数) == rankSize`。
2. **notify 对端**（无回填）：
    - estimate 采集约定（`device_sqe_parse_stub.cc`，两类任务一致）：notify 的
      `srcDeviceId`（承载 rankId）= record 侧、`dstDeviceId`（承载 rankId）= wait 侧，
      均由发射端显式填写；对端 rank 未解析时为 `0xFFFFFFFF`。
    - 早期版本的 `dstDeviceId == 0` 回填兜底已随 rank 字段回退删除（0 哨兵与合法
      rank 0 冲突，且发射端恒显式填写后无回填必要）。
    - `WRITE_WITH_NOTIFY` (7) 携带的 notify 面向**写对端**（wait 侧不在本 db，单 rank
      仿真前提），不参与本地配对；其 `dstDeviceId`（承载 rankId）即写目的端。
    - 注：runner/checker 消费侧（`DecodeAndExpandTaskMeta`）在解码边界将 7 拆分为
      [写, NOTIFY_RECORD] 两个子任务；light_performance **不拆分**——单任务映射
      `Write_With_Notify`（5.2 节），保证仿真任务与 db 任务一一对应。
3. **解码长度校验**：`len(opTaskMeta) >= 185`（按长度自适应 v4/v3/v2 布局，4.4.1 节）、
   `len(opDetail) >= 28`、
   `len(instrSpace) >= instrCount*32`，不满足则丢弃该行并告警
   （checker 为直接 memcpy，此处更宽松）。

### 4.7 解码实例（实测 hccl_vm_data.db）

样例 db（REDUCE_SCATTER，opType=7，dataType=4/FP32，count=8，rankSize=2，devType=6/950）
实测解码结果：

- `opTaskMeta` 长度恒为 188B（v3 采集样例；v4 采集起为 189B，4.4.1 节）；taskType 分布：4（CCU_GRAPH）与 6（SYNC_STREAM）交替。
- opDetails 仅 device 1（rank 0）→ 单 rank 数据（设计常态）；主流 stream 2。
- 任务序列（每 opIter 主流）：

```
stream 2 (主流):  CCU_GRAPH(die=1, mission=0, instStartId=14, instCnt=100, argSize=13)
                  SYNC_STREAM(syncIdx=n)          # 微码下发 + 流同步
```

- CCU 任务的 args 与 data 目录 `sqe_info_device_1_die_1_mission_0_startId_14.txt`
  逐项一致（args[0]=281475405602881、args[1]=281475405602945 …）；
  `args[0]/args[1]` 实测即 opMemInfo 的 inputAddr/outputAddr。
- `ccuInstrRes`（dev 1, die 1）instrCount=114 = instStartId(14) + instCnt(100)。
- 早期含 MEM_CPY/NOTIFY 的 169B 样例中实测：MEM_CPY 的 srcOffset/dstOffset 均可落入
  opMemInfo 的 input/output/ccl 区间（`input ⊃ output`，ReduceScatter 的输出是输入的
  前 1/rankSize 段）；带 jettyId 的网络拷贝，其 offset 也表现为"本端区间地址"
  （通信域内存对称布局，对端 buffer 与本端 VA 相同），因此**方向判定不能只看地址**
  （见 5.5 节）。该结论对 v2 数据同样适用。

#### 4.7.1 v2 AICPU 模式样例（opTask_P_118800）

REDUCE_SCATTER、rankSize=8、count=32768（FP32，output=128KB）、76 任务、
流 2（主流）/6（调度流）/7..13（每对端一条流）。与 CCU 模式样例的关键差异：

- taskType 分布：NOTIFY_WAIT×30、NOTIFY_RECORD×30、MEM_CPY×8（7 走网 write
  jet=1..7 + 1 本地）、REDUCE×7（本地）、SYNC_STREAM×1；无 CCU_GRAPH，`ccuChannels` 为空。
- **方向全部由 rank 维度判定**（5.5 节规则 1）：7 条走网 MEM_CPY 均为
  `srcRank=0/dstRank=1..7`（write），任务偏移与 opMemInfo 区间不相交（区间推断失效），
  `direction_inferred=0`。
- **notifyId 跨 rank 复用**：如 `RECORD(nid=30, dst=dev2)`（wait 侧在对端，跳过）与
  本地 `WAIT(nid=30)`（record 侧在对端、不在本 db）成对出现，两轮各 7×2 个；
  本地配对 16 对（nid 3/13 主↔调度流、15..27 调度流→对端流、6..12 对端流→调度流），
  未配对本地 WAIT 14 个 → 按 `remote_wait_delay` 计费（5.4 节）。
- lcEid/rmEid 与 ranktable.json 逐字匹配（`…0040…01010001` → port `1/0` 等），
  走网任务端口全部可解析。

### 4.8 eid→rank 索引（`build_rank_index`，对端 rank 解析）

对端 rank 解析（5.5 节 SQE 回退、6.7 节 CCU 主路径）共用一个 eid(hex) → rank_id
索引，构建优先级与依据：

1. **RankEidMap 表（主路径）**：runnerdb `hccl_sim.db`（与 hccl_vm_data.db 同目录）
   的 `RankEidMap` 表。estimate 模式 Communicator 表只有本 rank 一行，对端 rank
   以该表为准（`sim_models.h` 注释；宿主先例 `GetRmtRankIdByEid`，
   `device_sqe_parse_stub.cc:889`）。行 BLOB 布局与 `sim::RankEidMap` 一致：
   `id(u64) + rankId(u32) + eid(16B) + pad(4B)`；eid 为"十六进制串字节序"，
   与 `ccuChannels.leid/reid` 同源同序，Python 侧按 `bytes.hex()` 直接相等匹配。
   该表由 ranktable.json 解析写入（`topo_ascend_cluster_parser.cc:229`），
   mock-comm 场景随通信域初始化整体重写，故为权威。
2. **ranktable.json（回退）**：RankEidMap 空/`hccl_sim.db` 缺失/无该表时，展开
   `rank_list[].rank_id ↔ level_list[].rank_addr_list[].addr`（addr_type=EID）
   构建同构索引。实测两路径索引完全一致（8 rank × 9 eid = 72 项）。

只读打开（`file:…?mode=ro` URI）；读取失败（含表不存在）降级回退而非报错。
注意 **`ccuChannels.srcRankId/dstRankId` 不能直接当通信域 rank 用**：写入侧
（`hccp_ccu_stub.cc:401-405`）明确填的是全局 endpoint deviceId（通道建立早于
communicator，同一通道可被多域复用），与 opDetails.rankId 编号系存在偏移，
checker 正因此弃用该字段（`g_allRankChannelInfo` 只存 dstDeviceId/dstDieId）。

---

## 5. SQE 任务图生成（子需求 2 · 下半，本次重点）

### 5.1 总体流水线

```
OpExecutionGroup
   │
   ▼
⓪ SQE 批合并（sqe_group_merge.py，5.1.1 节）: 同 (stream, sqeGroupId) 批内 UDMA 数据任务并成 1 条
   │
   ▼
⓪' DPU 序列合并（sqe_group_merge.py，5.1.2 节）: DPU 大消息 N-1 前块 + 尾块 WRITE_WITH_NOTIFY 并成 1 条
   │
   ▼
① 逐条 SQE → Task 节点（5.2 映射表；CCU_GRAPH 交第 6 章展开）
   │     stream_id: int（SQE 流直用原始 streamId, 6.5 节）
   ▼
② stream 串行链: 按 (deviceId, streamId) 分组、global_id 升序 → dep_graph.add_stream_chain
   ▼
 ③ notify 配对: 跨 rank 任务不参与配对（节点保留）, 仅本地 notify 先到先得 → dep_graph.add_match（5.4 节）
   ▼
④ 网络任务参数补全: 方向推断 / eid 解析（5.5 节）
   ▼
⑤ CCU 子图展开（第 6 章）: CCU_GRAPH 占位 → 微码指令解释 → 内部节点 + CKE 配对
   ▼
⑥ 输出组装: (SimConfig, DependencyGraph) → HcclSimulator(config, dep, topo)
```

### 5.1.1 SQE 批合并（成图前预处理）

采集侧 UDMA 一次 doorbell（SQE）可携带多个 WQE，任务仍逐 WQE 落库，并以
`sqeGroupId` 标记批归属（环形递增门铃批号 1..255、255 回绕、永不取 0；0 = 构造
默认"未打标"，4.4.1 节）。成图前按 **(streamId, sqeGroupId)** 切批——切批键显式
含流，把批内全部数据任务（MEM_CPY/REDUCE/WRITE_WITH_NOTIFY）合并为单条，对齐
hcomm 批量搬运 profiling 的 `ExecProfilingAll` 口径——三个入口（`pipeline.py`/
`__main__.py`/`dump_graph.py`）统一调用，保证仿真与 dump 口径一致：

- **按流分组的隔离保证**：切批键含 streamId，**跨流任务永不并入同批**（一个门铃
  本就只属一条流，单门铃批不会被键拆开）；组号（门铃批号）在 255 窗口内唯一
  ——其他算子的门铃虽不在本算子任务列表中可见、但同样消耗批号，故**同流同号必为
  同一门铃**，不跨门铃错并（多算子 db 安全）。
- **描述信息**（类型/地址/端点/notifyId/规约参数/profInfo）取批内**最后一个**数据
  任务，**长度取全批累加**——与 `ExecProfilingAll(loc[insNum-1], rmt[insNum-1],
  totalSize, transferOp[insNum-1])` 逐字段一致；trace/dump 显示名同理取最后任务的
  op 名（11.4.1 节）。批内混合 write/read 方向或不同 dataType 时同样随最后任务
  （单 SQE 批内通常同构，混合为罕见场景）。
- **不参与合并**：notify/SDMA/同步任务（notifyId 唯一、配对依赖顺序；本地任务
  原样保留、不影响批边界）；未打标/无组号任务（`sqeGroupId <= 0`：0 = 未打标，
  -1 = v3 及更早 BLOB 无此字段）原样保留并**切断当前批**（无法判定批归属，保守
  结束当前批）；UDMA 非数据任务（inline 写 notify）仅参与切批，批内夹杂的排在
  合并任务之后（notify 因果）。
- 合并任务占批内第一个数据任务的槽位（批首），切批键含 stream 保证批内任务同
  stream 同 jetty，stream 串行链（5.3 节）与 notify 配对（5.4 节）的相对顺序不受
  影响；本地（非 UDMA）任务不参与合并且不重置批边界（原位保留，不影响批连续性）。

### 5.1.2 DPU 序列合并（成图前预处理）

DPU（host CPU RoCE）路径的大消息 WriteWithNotify 序列合并，在 5.1.1 之后执行
（`pipeline.py` 统一调用）。规则与语义见 11.4.6 节：按 DPU 流序扫描
**src/dst 双地址连续**的 MEM_CPY 链（仅 `src_type==1` 的 DPU 来源数据任务参与，
其他任务原样保留不切断链），由地址连续的 WRITE_WITH_NOTIFY 收尾合并为单条——
**收尾任务是天然终止符**，两次独立的 WriteWithNotify 调用不可能跨并（每次调用的
尾块必然关闭当前链），误并仅剩"独立纯写恰与 WriteWithNotify 起始地址相接"一种
理论场景（语义等价于一段连续数据+收尾通知，仿真结果无差异）。无收尾的链不合并
（独立纯写保持逐条 `RDMASend`）；合并任务占收尾任务槽位（notify 因果），被并入
的前块从原位移除、未合并任务保持原始流序。

**单 rank 成图策略（设计前提，与 checker 的主要区别）**：db 中只有单 rank 任务数据，
DependencyGraph 只描述该 rank 的任务及其依赖。**跨 rank 的配对边（post-wait 关系）
全部不建（任务节点保留，按固定远端时延近似）**：

- SQE 级：waitDev ≠ 本设备的 RECORD（跨 rank post）、recordDev ≠ 本设备的 WAIT（跨 rank wait）
  **保留任务节点**——按 stream 顺序正常成图（src≠dst 表达远端、走固定远端时延），但**不建立
  任何配对边**——只关心单 rank 内的 post-wait 配对（5.4 节）。
- CCU 级：post 到远端 CKE 的指令、依赖远端 CKE 置位的 wait **同样生成任务节点**
  （Notify_Record/Notify_Wait，src=本端、dst=INVALID → 固定远端时延），但不建立
  跨 rank 配对边（6.4/6.6 节）。
- **保留跨 rank 的数据搬运**（direction=write/read 的 MEMCPY）：它们是本 rank 视角网络
  带宽占用的主体，忽略的只是同步依赖边，不忽略传输本身。
- makespan 语义：单 rank 视角的算子端到端耗时 = 本 rank 内计算/拷贝 + 本端发起的网络传输
  + 本 rank 内同步等待；远端何时就绪不建模（等价于假设远端即时就绪）。

**仿真粒度**：默认逐算子（每个 `(commId, opIter)` 一个 DependencyGraph、一次仿真、一条记录），
与 checker 的逐算子校验口径一致；`--all-in-one` 可选全量模式（同一 stream 上多算子天然串行）。

### 5.2 taskType → hccl_sim Task 映射

对齐 HCCLSIM 新版任务模型（`TaskType`: `NOTIFY_RECORD`/`NOTIFY_WAIT`/`REDUCE_INLINE`/
`MEMCPY`/`UB_INLINE_WRITE`/`UB_WRITE_OR_READ`/`WRITE_WITH_NOTIFY`，详见 7.3 节）：

| db taskType | 判定条件 | hccl_sim TaskType | Task 参数 | 时延模型 |
|-------------|---------|-------------------|-----------|---------|
| NOTIFY_RECORD (1) | waitRank == 本端 rank | `NOTIFY_RECORD` | notify_id=int, src=dst=local_rank | 固定 + 触发 Event |
| NOTIFY_RECORD (1) | waitRank != 本端 rank | `NOTIFY_RECORD`（跨 rank post：节点保留，不配对） | notify_id=int, src=local_rank, dst=对端 rank（未解析退 INVALID） | 固定 remote_record_delay，不触发 Event |
| NOTIFY_WAIT (0) | recordRank == 本端 rank | `NOTIFY_WAIT` | 同上 | 等待 Event |
| NOTIFY_WAIT (0) | recordRank != 本端 rank | `NOTIFY_WAIT`（跨 rank wait：节点保留，不配对） | notify_id=int, src=对端 rank（未解析退 INVALID）, dst=local_rank | 固定 remote_wait_delay |
| MEM_CPY (3) | jettyId == INVALID | `MEMCPY` | size=len, src/dst_rank=taskData 端点 rank（未填退本端） | HBM 动态共享带宽 |
| MEM_CPY (3) | jettyId 有效, write 方向 | `UB_WRITE_OR_READ` | size=len, src_rank=local, dst_rank=对端 rank, src/dst_eid, jetty_id, protocol, is_write=True | 端口 write 方向带宽共享 |
| MEM_CPY (3) | jettyId 有效, read 方向 | `UB_WRITE_OR_READ` | size=len, src_rank=对端 rank, dst_rank=local, src/dst_eid, jetty_id, protocol, is_write=False | 端口 read 方向带宽共享 |
| REDUCE (2) | jettyId == INVALID | `REDUCE_INLINE` | size=dataCount, data_type=任务级 dataType 枚举名, src/dst_rank 同 MEMCPY 本地分支 | HBM 动态共享带宽 |
| REDUCE (2) | jettyId 有效 | 同 MEM_CPY 走网分支（5.7 节近似） | 同上（含 data_type） | 同上 |
| WRITE_WITH_NOTIFY (7) | 恒 write 方向（本端 src → 远端 dst，无方向推断） | `WRITE_WITH_NOTIFY`（**单任务，不拆分**） | size=len, notify_id=notifyId, src_rank=local, dst_rank=对端 rank（未解析经 eid→rank 索引回退）, src/dst_eid, jetty_id, protocol, is_write=True；isReduce=1 时 data_type=dataType 枚举名 | 端口 write 方向带宽共享 + 写落地后 notify 固定 record 时延（remote） |
| CCU_GRAPH (4) | — | （展开，第 6 章） | — | — |
| AIV_GRAPH (5) | — | 跳过 | — | 告警 |
| SYNC_STREAM (6) | — | **跳过**（host 侧流同步点，5.6 节） | — | — |

> waitRank/recordRank 为 taskData 端点 rankId（deviceId 字段承载，src=record 侧、
> dst=wait 侧，4.4.1/4.6 节），仅用于**判定本地/跨 rank**：跨 rank 的 notify 任务在
> 映射阶段保留节点但不进入配对（5.1 节策略），
> 配对只在本地 notify 之间进行。notify 的本地/远端在新模型中以 `src_rank == dst_rank` 表达
> （`is_local_notify`）。
> **WRITE_WITH_NOTIFY (7) 单任务映射**（与 db 任务一一对应，不拆分为写 + NOTIFY_RECORD
> 两个子任务）：写方向恒为 write（发射端约定 src 本端 → dst 对端，无 5.5 节方向推断），
> 走端口 write 方向带宽共享；写数据落地后向对端发 notify，按固定 record 时延计费
> （对端 wait 不在本 db，单 rank 仿真前提，同跨 rank record 语义；自写 src==dst 边缘
> 场景按 local record 时延）；`notifyId` 随任务保留（trace/dump 显示），
> `isReduce=1` 时仅影响 `data_type` 显示（同 REDUCE 分支的枚举名映射）。
> **走网任务 rank 直接消费 taskData**（deviceId 字段承载 rankId；发射端约定：
> write=src 本端/dst 对端，read=src 对端/dst 本端）：write 填 `dst_rank=对端 rank`、
> read 填 `src_rank=对端 rank`；对端 rank 未解析（`0xFFFFFFFF` 或等于本端）时退 INVALID
> （10.1 节）。方向以 `is_write` 显式携带，仿真器不再依赖 `src_rank==0` 启发式
> （10.3 节 #1 已修复；task.txt 旧输入路径仍走启发式）。
> 本端 eid 优先取 SQE 显式 lcEid（v2 采集新增），缺省经 jettyId → ccuChannels 通道
> leid 推导（5.5 节、10.1 节）。
> 本表 TaskType 为内部时延模型分类；trace/dump 的任务显示名已切换为
> profInfo.profTaskType 的 op 名（11.4.1 节对照表，10.3 节 #18 已落地），
> 与本表 TaskType 的对应以 11.4.1 节"仿真 TaskType"列为准。

### 5.3 stream 串行链构建

- 按 `(deviceId, streamId)` 分组，组内按 `global_id` 升序（= SQE 下发顺序）构成链。
- 仿真器为每条链启动一个 simpy 进程串行执行（第 7 章）——**不需要 checker 的 mainStart 虚拟
  起点与主流特判**：仿真关注端到端 makespan，各链从 0 时刻并行启动即可；
  从流首任务的 WAIT 自然形成与主流的实际同步。

### 5.4 notify 配对算法（仅本地）

**跨 rank 的 post-wait 边全部忽略**（5.1 节策略）：配对前先按 4.6 节对端回填结果分类，
跨 rank 的 notify 任务（waitDev ≠ 本设备的 RECORD、recordDev ≠ 本设备的 WAIT）在 5.2 节
映射阶段保留节点（按 stream 顺序成图），但不进入配对。因此配对键从 checker 的三元组简化为单一 notifyId：

```
checker 语义:   配对 key = (recordDeviceId, waitDeviceId, notifyId)   ← 多 rank
本插件:         先过滤跨 rank → 配对 key = notifyId                    ← 单 rank
```

与 checker 的 `AddLocalNotifyEdges`（下发顺序模拟 + 先到先得消费，
`task_graph_generator_v3.cc:847-946`）原理一致，light_performance 简化为
**两遍扫描 + 先到先得**：

```
输入: 某算子的全部本地 notify 任务（跨 rank 节点保留但不参与配对）
第一遍: 按 global_id 升序扫描
    RECORD → pending_records[notifyId].append(task_id)                        # 队列
    WAIT   → if pending_records[notifyId] 非空:
                 rec = 队首弹出; dep_graph.add_match(wait_id, rec_id)          # 先到先得
             else:
                 unmatched_waits.append(wait_id)                               # 等"更晚"的 record
第二遍: 处理 unmatched_waits（WAIT 先于 RECORD 出现的情形）
    按 global_id 序重放: WAIT 未匹配时挂起，RECORD 到来后与最早的挂起 WAIT 配对
    （配对成功的 RECORD 同步从 pending_records 移除，保证诊断统计准确）
收尾: 仍残留的 WAIT → 属正常（见下）；仍残留的 RECORD → 告警，正常数据不应出现
```

说明：

- db 的 id 顺序即下发顺序，且 stream 串行链已保证"WAIT 之后同一流的后继任务不会先执行"，
  因此无需 checker 的完整队列模拟与死锁检测（死锁在仿真中以 makespan 异常/任务未执行的形式暴露）。
- 同一 notifyId 多个 RECORD/WAIT（如早期 169B 样例中 n11/n15 在 stream 6 上重复出现、主流两轮迭代）
  由先到先得规则自然配对，**notifyId 允许复用**。
- **未配对本地 WAIT 属正常语义**：v2 AICPU 样例（4.7 节）中 notifyId 存在跨 rank 复用——
  本地 `RECORD(nid=30, dst=dev2)`（wait 侧在对端，保留节点但不配对）与本地 `WAIT(nid=30)`
  （record 侧在对端、不在本 db）成对出现；该 WAIT 在仿真中按 `remote_wait_delay` 固定时延计费
  （7.6 节），告警仅为提示。
- 早期 169B 样例中的跨 rank 对：`RECORD(n18, s1→d2)`（通知 dev2）与 `WAIT(n18)`（回填 recordDev=2）
  均保留节点、不建跨 rank 配对边：按 stream 顺序成图，`src≠dst` 走固定远端时延；
  stream 8 上的数据搬运任务（MEMCPY(jet=3) 等）不受影响。

### 5.5 网络任务的方向与 eid 解析

**方向判定**——走网数据任务统一映射为 `UB_WRITE_OR_READ`，方向决定端口计费
方向（is_write）与 rank 端点取值。taskData 端点 rankId 由 deviceId 字段承载
（4.4.1 节 rank 字段回退），发射端约定（`device_sqe_parse_stub.cc`）：
write = src 本端/dst 对端，read = src 对端/dst 本端：

| 优先级 | 规则 | 依据 |
|--------|------|------|
| 1 | srcRank == 本端且 dstRank ≠ 本端 → write；dstRank == 本端且 srcRank ≠ 本端 → read | 采集端点 rankId（deviceId 字段承载，4.4.1 节），实测样例（4.7 节）全部由此判定，无推断 |
| 2 | rank 未区分（相等或均未填）时按内存区间：dst 落入本端 output 区间 → read；src 落入本端 input 区间 → write | 实测样例归纳（4.7 节）：任务偏移与 opMemInfo 区间不相交时失效 |
| 3 | 兜底：默认 write，诊断统计 `direction_inferred` 计数 | — |

> 走网任务在 Task 上显式携带 `is_write`（10.3 节 #1 已修复；CCU 路径 6.7 节同步修复），
> 仿真器据此选择 write/read 计费方向与端口查询 eid（`src_eid`/`dst_eid`），不再依赖
> `src_rank==0` 启发式；task.txt 旧输入路径无该字段，退回启发式判定。
> `WRITE_WITH_NOTIFY` (7) **不参与方向判定**：语义恒为 write（本端 src → 远端 dst），
> 成图时直接置 `is_write=True`。

**eid 解析**——网络任务经 eid 查 ranktable 得物理端口（`TopoResolver.get_ports_by_eid`）：

```
SQE:        本端 eid 优先取 SQE 显式 lcEid(v2 新增); 缺省经 jettyId
            ──ccuChannels(本设备出向通道)──> leid(本端 eid)
            对端优先取通道 reid, 缺省退回 SQE rmEid; 通道缺失时本端 eid 退化为任一本地通道 leid
CCU 微码:   channelId → channels[(dev,die,channelId)] → leid/reid + jettyId 列表
            通道缺失时本端 eid 优先取该 mission SQE 的 lcEid, 退化为同 die 任一通道 leid
write 方向: 仿真器查 task.src_eid → ports; read 方向查 task.dst_eid → ports
```

- `protocol`：通道 protocol 列 → HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO（CommProtocol 名，
  仅显示用）；`link_type`/`transport_type`：任务 profInfo（DfxLinkType/
  ProfTransportType 名，11.4.2/11.4.3 节，`_add_task` 统一填充，与 msprof L1 对账）。
- 本端 eid 无法解析时告警 + `eid_unresolved` 计数；存在走网任务而无 ranktable 时
  仿真整体跳过（TopoResolver 对缺失 eid 直接抛错，见 10.3 节）。

**对端 rank 解析**——SQE 走网任务的对端 rank（`_peer_rank`）：taskData 端点 rankId
（deviceId 字段承载，发射端已按 `GetRmtRankIdByEid` 解析）优先；未解析
（UINT32_MAX/本端值）时经 eid→rank 索引（4.8 节）按对端 eid 回退，仍无则
INVALID + `rank_unresolved` 计数。

### 5.6 SYNC_STREAM 处理

host 侧 stream 同步点（样例主流上每轮迭代出现一次），表示 host 等待本流完成，
**不代表设备侧任务**：不生成任务节点、不占用时间（makespan 即流链末任务结束时间）。

- 诊断统计 `skipped_sync` 计数；
- 曾以"未配对本地 `NOTIFY_RECORD` 固定时延占位"近似，因会在流尾引入虚假
  `local_record_delay`（且 trace 中呈现为 notify_id=invalid 的伪 notify 任务）已移除。

### 5.7 REDUCE 近似

规约 = 网络传输 + 计算。当前模型以数据搬运近似：

- 走网（jettyId 有效）：按 `UB_WRITE_OR_READ` 计端口带宽时延
  （方向同 5.5 节判定）。
- 不走网（jettyId INVALID）：映射为 `REDUCE_INLINE`，按 HBM 带宽动态共享计延
  （`size / hbm_bandwidth`）；计算时延暂未单列（后续可按 `dataType×dataCount` 标定，
  见 10.1 节 data_type 字段缺口）。

### 5.8 与 checker 成图的对照与简化

| checker 环节 | light_performance | 理由 |
|--------------|-------------------|------|
| mainStart 虚拟节点 + 主流入边特判 | 去除 | 仿真只需 makespan，链首 WAIT 已表达同步 |
| AddLocalNotifyEdges 队列模拟 + 死锁检测（GRAPH_DEADLOCK） | 两遍扫描先到先得，不做死锁检测 | 无校验诉求；异常以仿真结果形式暴露 |
| AddInterRankNotifyEdges 跨 rank 配对（全图 BFS 模拟） | **完全去除**：跨 rank post-wait 配对边全部忽略（notify 任务节点保留，按 stream 顺序成图） | 单 rank 成图（核心区别） |
| MemSlice 地址解析 + 内存布局表 | 仅做方向判定的区间归属（可选） | 仿真只需 len，不需精确内存模型 |
| CheckSlaveTaskQueue 从流结构校验 | 去除 | 非校验工具 |
| 内存冲突/语义校验/同步冲突/可达性/单任务检查 | 全部去除 | 非校验工具 |
| CompactSyncNodes（SYNC_STREAM 合并） | 跳过不生成节点（host 侧同步点，非设备侧任务） | makespan 即流链末任务结束时间（5.6 节） |
| AIV 子图展开 | 不支持（告警跳过） | 需求明确暂不考虑 |
| CCU 子图：queue 并行化改造、loop merge、资源冲突校验 | queue 并行化简化实现（7.4.1 节：组装期骨架/单元切分 + fork/完成集汇接与挂接 extra 边 + `#v` 虚拟流）；loop merge 同等实现（6.5 节）；资源冲突校验去除 | 并行化消除队列串行化失真（7.4.1 节）；免校验，节点数可控 |

### 5.9 输出组装

```python
def build(group: OpExecutionGroup, sim_config: SimConfig | None = None
          ) -> tuple[SimConfig, DependencyGraph]:
    dep = DependencyGraph()
    # ① SQE → Task（CCU_GRAPH 暂存占位）
    # ② stream 链（int 流 id）
    # ③ notify 配对（add_match）
    # ④ 方向/eid 补全
    # ⑤ CCU 展开（ccu_graph_builder.expand_ccu_graph(..., local_rank, alloc_stream)）
    return sim_config, dep
```

- `SimConfig` 来源：`--config` / manifest setting 指定的 JSON（第 7.2 节格式，单位
  us/GBps），叠加带宽利用率配置。
- 拓扑：`TopoResolver` 由 CLI 层构建（ranktable 定位见 3.3 节）并传入仿真器。
- 输出即刻可送 `HcclSimulator(config, dep_graph, topo).run()`（返回 dict，见 7.1 节）。

---

## 6. CCU 模式任务图生成（子需求 3，本次重点）

### 6.1 CCU 数据在 db 中的组织

```
opTask_P_<pid> 中 taskType=4 (CCU_GRAPH) 的 SQE
    └─ taskData.ccu: {dieId, missionId, timeout, instStartId, instCnt, key, argSize, args[13]}
           │  instStartId/instCnt 界定指令执行窗口 [instStartId, instStartId+instCnt)
           │  args[13] 逐个被 LOAD 类指令消费（每读满 13 个 arg 推进 sqeIndex）
           ▼
ccuInstrRes (deviceId, dieId) → instrSpace: 32K × 32B 微码指令空间
ccuChannels (deviceId, dieId, channelId) → {dstDeviceId, dstDieId, leid/reid, jettyId[]}
    └─ 通道寄存器值 → 对端位置与 jetty
```

- SQE 按 `(dieId, missionId, queue)`（单设备成图，deviceId 恒定；queue 即 SQE 流）
  归并为派发上下文（checker 的 `CcuMissionKey` + 最新节点追加语义，
  `task_meta_translator_v3.h:47-62`）：同键 SQE 仅当**与前一条连续**（前条
  `instCnt==13` 且 `instStartId` 首尾相接，`IsContiguousCcuSqe`，
  `task_meta_translator_v3.cc:213-231`）时并入同一上下文，否则**新建上下文**——
  同一 mission 的多次 dispatch（指令窗口回绕重发，典型如"头 SQE(13 条 Load
  SqeArgs) + 体 SQE"连发两遍）各自独立展开：每个上下文完整重放自己的指令窗口
  （含 Load SqeArgs 头部，用**本 dispatch 的 args** 重新初始化循环计数等寄存器），
  避免后一 dispatch 沿用前次残留的寄存器状态导致数据搬运丢失（10.3 节 #20）。
  - **args 消费**（checker `CcuGraphStateV3::GetSqe` 语义）：上下文内 SQE 按派发序
    排列，args 逐槽顺序消费——每读满 13 个 arg（读到末槽 `sqeArgsId==12`）游标推进
    到下一 SQE；典型布局下头 SQE 的 13 条 Load SqeArgs 消费头 SQE 的 args[0..12]，
    体 SQE 窗口内的后续 Load 指令消费体 SQE 的 args。
  - **寄存器/CKE 位图跨上下文全局共享**（checker `AllRankParamRecorder::Global()`
    语义）：后一 dispatch 的 wait 可由前一 dispatch 遗留的 CKE 位/残留 post 记账
    消解，与 checker 多图共享 recorder 的行为一致。
  - **同 queue 的派发上下文严格按派发序串行推进**（checker queue 语义：单个
    queue 严格按指令顺序，同流后一 TaskCcuGraph 须等前一完成）；跨 queue 轮询
    联合推进（6.6 节）。
- 子图挂接点：CCU_GRAPH SQE 在 stream 链中的位置即子图入口；展开后该占位被移除
  （任务图中不引入任何占位节点），子图经 extra 边与 SQE 前驱/后继任务挂接（7.4.1 节）。

### 6.2 微码指令格式（32B 定长）

与 `${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode` 同源
（本仓参考副本：`ai_skills/skills/convert-ccu-microcode/references/ccu_microcode_v1.h`）：

```
CcuInstr = CcuInstrHeader (2B) + union { V1(A5) | V2(A6) | V3 } (30B)

CcuInstrHeader 位域 (小端 u16):
    code     : 11 bit   (header & 0x7FF)      指令码
    type     :  4 bit   ((header >> 11) & 0xF) 指令大类
    reserved :  1 bit

指令大类:  LOAD_TYPE=0x0 | CTRL_TYPE=0x1 | TRANS_TYPE=0x2 | REDUCE_TYPE=0x3
```

芯片代次 → 指令集版本映射（checker `GetCcuInstrVersion`，`ccu_task_transform_v3.cc:175-184`）：
`Ascend950 → A5/V1`，`Ascend960 → A6/V2`，其余 devType 回退 A5；由 Device 的 soc_version
（hccl_sim.db）或 `opDetails.devType` 推导（`expand_ccu_graph` 按 `dev_type` 选择
`_MissionContext`/`_MissionContextA6` 解释器）。**两代指令集均已实现**（A6 移植自
checker `ccu_task_transform_instruct_a6_v3.cc`，11.3.3 节）；A5/A6 指令集差异概览见 6.8 节。

### 6.3 指令类型 → Task 映射

#### 6.3.1 总表

| 指令大类 | 代表指令 | light_performance 行为 | 产出 Task |
|----------|---------|----------------------|-----------|
| LOAD（寄存器写） | LoadSqeArgsToXn/GSA、LoadImd、LoadX、算逻（A6 Add/Sub/…） | 更新寄存器模型，数据面**不产生节点**；A6 LOAD 类普遍携带 `setCKE` 尾部字段（mask 非零 → 本端 `NOTIFY_RECORD`，checker `ProcessSetMask` 语义）；A6 Load/Store（HBM↔Xn 数据面）不建模 HBM 内容，寄存器不回填（仅保留 setCKE 同步语义，告警留痕） | —（A6 setCKE 非零时 Notify_Record） |
| CTRL: SetCKE/SetCKBit | 置 CKE 位 | 本端 → `NOTIFY_RECORD`（登记 CKE 记账，带 mask）；远端目标（跨 rank post，SyncCKE 等）→ 远端 `NOTIFY_RECORD`（不配对，6.6 节） | Notify_Record |
| CTRL: Wait/Jmp/Fence | 等 CKE 位 / 跳转 / 栅栏 | CKE wait（A5 内嵌 waitCKE 字段）: 位不足则暂停该 queue 指针；位齐生成 `NOTIFY_WAIT`（按贡献位拆分，带 mask）；依赖远端位的 wait → 远端 `NOTIFY_WAIT`（不配对，固定时延，6.6 节）。A6 独立 Wait 指令（Xn 值比较自旋，checker `TransformWaitInstr`）: 满足直通无节点；未满足 = 依赖对端远端寄存器写（SyncWtX/TransMem notify）→ 顶层按停滞消解生成远端 `NOTIFY_WAIT`（固定 `remote_wait_delay`），循环体内直接生成节点继续。Jmp: A5 按 `Xn[conditionXnId] != expectData` 求值（checker `TransformJmpInstrA5` 语义）；A6 支持 6 种条件码（==/!=/>/>=/</<=，≥6 无条件）+ 相对/绝对两模式（jumpMode，目标 `Xn[relTarInstrXnId]`）；目标限当前指令窗口内，越界告警直通；循环体内直通；带步数预算防御后向跳转死循环。Fence（仅 A6）: checker 不支持，告警直通（队列串行模型下顺序天然保证） | Notify_Wait |
| CTRL: ClearCKE | wait + 清位 | wait 语义；A6 按 checker `ClearWaitMask` 语义清除本次消费的 wait 位（`clearMask != 0` 为 checker 不支持场景，告警 + 宽容清除目标位） | Notify_Wait |
| CTRL: Loop/LoopGroup | 循环 | loopCnt 来自寄存器，**合并展开**（6.5 节，失败回退全展开）；A6 Loop `mode==1` 为条件展开（`wishCKEBit == Xp.loopCtxId + 1024` 命中才执行，恒走全展开路径） | 无（体内指令按本表映射；合并路径下数据任务 size × 总迭代数） |
| TRANS（搬运） | LocMem↔LocMS/RmtMem/RmtMS（A5）、本地 4 码 + 统一 TransMem dmaOpCode（A6） | 按角色映射（6.3.2） | `Memcpy`/`Ub_Write_Or_Read` |
| TRANS: Sync（跨 rank 同步） | SyncCKE/GSA/Xn（A5）、SyncXnWtx/SyncAtx（A6） | 远端置位/通知 → 远端 `NOTIFY_RECORD`（不配对，6.6 节）；值同步为远端寄存器写，不建节点 | Notify_Record（远端） |
| REDUCE | Add/Max/Min | 同 SQE REDUCE 近似（5.7 节）；MS 为 rank 本地 buffer，恒本地 HBM 规约 | `Reduce_Inline` |

#### 6.3.2 TRANS 传输角色判定（与 checker `ccu_loop_merge_v3.cc:73-144` 一致）

按 src/dst 的设备归属：

| src | dst | 角色 | hccl_sim Task |
|-----|-----|------|---------------|
| 本端 | 本端 | LOCAL_COPY | `Memcpy`（HBM 动态共享） |
| 本端 | 远端 | WRITE | `Ub_Write_Or_Read`（src_rank=local, dst_rank=对端, is_write=True） |
| 远端 | 本端 | READ | `Ub_Write_Or_Read`（src_rank=对端, dst_rank=local, is_write=False） |

> 跨 die/卡的读写两向统一映射 `Ub_Write_Or_Read`，方向经 `is_write`/trace
> `direction` 字段表达，端口按 write/read 方向独立计费；对端 rank 经通道
> reid → eid→rank 索引解析（未解析为 INVALID）。为保留微码可区分的缓冲区
> 信息，CCU 转换的搬运任务携带 `src_buf_type`/`dst_buf_type`（mem/ms，按
> 指令源/目的操作数判定），SQE/CLI 路径同名任务无该信息，输出无效值 N/A。

> 跨 rank 的**数据搬运**（WRITE/READ 角色）保留——它是网络带宽占用的主体；
> 忽略的只是跨 rank 的**同步边**（post-wait，见 6.4/6.6 节）。

关键字段来源：

- `size`：长度寄存器（A5 `lengthXnId` 直读 / A6 `xlId` 经 `GetXnId`——bit15=1
  时寄存器号按循环 `xnIdOffset` 偏移）→ 查寄存器模型。
- 对端位置：A6 通道经**寄存器取值**（`Xn[GetXnId(xcId)]` 的值即 channelId）→
  `channels[(dev,die,channelId)]` → `{dstDeviceId, dstDieId}`；A5 通道为指令
  立即数字段。A5 MS 类指令 `msId / 0x8000` = rank、`msId % 0x8000` = MS 偏移（MS 页 4KB）——**仅作
  寻址编码**：MS 是 rank 可直接访问的本地 buffer（checker `GenSliceFromMs` +
  `AddLocalBatchReduce` 对全部 msId 恒本地），msId 高位**不作为**本地/远端任务类型
  判据；跨 rank 数据移动由 Rmt* 角色指令（走通道）表达。A6 TransMem 的
  `msIdMode==1` 标记 MS 端（write 的源 / read 的目的为本地 MS，
  `src/dst_buf_type=ms`）。
- eid：通道 leid/reid（hex 字符串）；本端 eid 优先取该 mission SQE 的 `lcEid`
  （v2 新增），缺省取同 die 通道 leid 近似。
- A6 `TransMem` 的 `dmaOpCode`（已实现，checker `TransformTransMemInstr`）：
  3=写、5=写带 notify（notify 值写远端 Xn/CKE，目标类型单 rank 不可解析 →
  按远端 record 近似）、6=读、0x1A=远端原子计数（无数据面，按远端 record
  近似）、0x19（压缩写+原子）及其余不支持（告警跳过数据面，setCKE 本端
  post 仍执行保持同步骨架）；`udfEnable`（走网规约）按普通 UB 搬运计费
  （5.7 节近似，对端规约不建模）；`udfType != 0` 告警跳过数据面。

### 6.4 执行模型：寄存器 + CKE 记账（register_model.py）

对 checker `AllRankParamRecorder`（`ccu_all_rank_param_recorder_v3.h:27-110`）的 Python 简化：

```python
class CcuRegisterModel:
    xn:    dict[(dev, die, id)] -> int      # 通用寄存器（长度/循环计数等）
    cke:   dict[(dev, die, ckeId)] -> int   # CKE 位图（置位/清位）
    gsa:   dict[(dev, die, id)] -> int      # A5 GSA 寄存器（A6 无 GSA，仅 xn/cke）
    seen_post: dict[key, list[PostRecord]]  # CKE post 记账（配对用）

    key = (waitDeviceId, dieId, ckeId)
    PostRecord = {record_device_id, cke_mask, remaining_mask, task_id, is_local}
```

- LOAD/算逻指令在此模型上"执行"（读写寄存器）；未初始化读取 → 告警并按 0 处理
  （checker 为报错退出，此处宽容处理以适配不完整数据）。A6 算逻指令
  （Add/Sub/Mul/And/Or/Not/Xor/Shl/Shr/Popcnt）按 64 位回绕语义执行（Mul 取
  低 32 位乘积、移位量按 0x3F 截断、算术移位按有符号解释）；LoadX/StoreX 的
  间接寻址目标寄存器号越界（>16383）告警跳过写入（checker 报错中止）。
- **A6 循环内寄存器号偏移**（checker `GetXnId`/`UpdateXnId`）：载荷中寄存器号
  字段 bit15=1 时，展开区内（`curLoopIdx >= expandOffset`）按
  `xnIdOffset × curExpandCnt` 偏移寄存器号；bit15=0 直用原始号。CKE/MS id 的
  循环偏移两代同规则（`UpdateId`）。
- **CKE 配对规则**（与 checker `GenWaitNode`，`ccu_task_transform_v3.cc:56-121` 一致）：
  ```
  post 与 wait 匹配 ⇔ post.waitDeviceId == wait.waitDeviceId
                   && post.dieId == wait.dieId
                   && post.ckeId == wait.ckeId
                   && (post.ckeMask & wait.ckeMask) != 0        # 位有交集
  匹配后: 一条 wait 指令聚合为至多 1 个本地 wait 节点 (mask = 全部贡献
          post 的位或) + 1 个远端 wait 节点 (依赖远端置位的位)
          post.remaining_mask &= ~wait.ckeMask                  # 一个 post 的位可被多个 wait 分次消费
          扣空 → 从 seen_post 移除
  节点依赖: 聚合 wait 节点与全部贡献 post 建边 —— 首个贡献 post 走
          notify 配对 (add_match, Event 语义), 其余经 extra 边
          (child=wait, parent=post, "等待全部父完成" 纯时序约束)
  ```
- **CKE 记账只登记本 rank 的 post**（单 rank 成图）：跨 rank post 生成远端
  `Notify_Record` 节点（src=本端、dst=INVALID → 固定远端时延）但不登记
  `seen_post`（置位动作发生在对端 rank 的 CKE）；依赖远端置位的 wait 位生成
  远端 `Notify_Wait` 节点（固定时延，不配对），见 6.6 节。
- 位不满足时该 queue 指针暂停（等待本 rank 其他 queue 的 post 解锁）——多 queue 联合推进
  见 6.6 节算法；同 queue 存在多个派发上下文（同 mission 多次 dispatch，6.1 节）时，
  后一上下文须等前一上下文完成才开始推进。
- 收尾校验：`seen_post` 残留 → 告警并计入诊断统计（checker 报 GRAPH_UNMATCHED，此处降级）。

### 6.5 queue 模型与 loop 处理

- **queue**：`CcuSqeParam` 按 SQE 所属 stream 分组（checker 为 ccuParams[queueId]），
  每个 queue 一条串行链。新版任务模型 `stream_id` 为 **int**：
  SQE 流直用原始 streamId；CCU queue 链经分配器取
  `alloc_stream("ccu_d{die}_q{queue}")`（基址 `1<<20` 起顺序分配，同名复用以
  合并同 queue 多 mission），逻辑名保留在 `stream_labels` 供 dump/报告显示。
  同一 queue 内指令严格按序（checker 语义：`ccu_task_transform_v3.cc:189`，
  "单个 queue 严格按指令顺序推进；跨 queue 的协同只通过生成出来的 wait/post 边表达"）。
- **组装阶段队列并行化切分**（7.4.1 节）：解释执行完成后，每条 CCU 队列链再切分为
  骨架（wait / 游离 set / 远端同步节点——游离 set 指不紧随数据任务的 record 节点）
  与异步数据搬运单元两类：
  骨架留在 queue 串行链上表达时序框架，异步单元改挂虚拟流
  `ccu_d{die}_q{queue}#v{n}`（与该 queue 同一基址空间顺序分配），跨 mission 复用、
  mission 内不复用，经 fork extra 边与骨架衔接——串行链只约束骨架与单元内部，
  消除"整条队列逐任务串行"的失真。
- **loop 展开**（合并优先，对齐 checker `ccu_loop_merge_v3.cc`）：
  - `Loop`/`LoopGroup` 指令的迭代数来自寄存器，在解释执行时已知：
    - A5：LoopGroup 参数在 Xn（LoopNum[61:55]/RepeatNum[54:48]/NoRepeatNum[47:41]）
      与 Xm（gsaOffset/MSOffset/ckeOffset），Loop 迭代数在 Xn[xmId] 低 13 位；
    - A6：LoopGroup 参数改由 **Xn/Xm/Xp 三寄存器**承载（checker
      `LoopGroupXnA6/XmA6/XpA6`）——Xn：`loopInsCnt:10 | expandOffset:9 |
      expandCnt:9`，Xm：`ckOffset:10 | msOffset:11 | gsaOffset:32`，
      Xp：`xnIdOffset:32`（GetXnId 偏移量，6.4 节）；Loop 的迭代数在
      `Xn[GetXnId(xmId)]` 低 13 位、GSA 步长在 `Xn[xnId]` 低 32 位、上下文 id
      在 `Xn[xpId]` 低 9 位；`mode==1` 为**条件 Loop**——仅当指令
      `wishCKEBit == loopCtxId + 1024`（`A6_LOOP_BOARD_ENTRY`）时展开，否则
      本展开跳过（checker `ProcessLoopIns`），且条件 Loop 恒走全展开路径
      （checker `CollectLoopExpandA6` 拒绝合并）。
    LoopGroup 内每个 Loop 先尝试**合并路径**（checker `TryProcessMergedLoopA5/A6`
    → `LoopIterationMerge` / `LoopExpandMerge` 语义），校验不过再回退按迭代全展开。
  - **合并前置校验**（任一不满足即 WARN 回退，与 checker "Skip merged-loop
    optimization" 同语义）：
    1. 循环体仅含可合并指令：
       - A5：简单 TRANS / SYNC TRANS（SyncCKE/GSA/Xn）/ REDUCE / ClearCKE
         （`CollectLoopCkeOpsA5` 规则；SetCKE、LOAD、LOOP、JMP、
         `setRmtCKEMask != 0` 的 TransMem、`clearMask != 0` 的 ClearCKE 均拒绝）；
       - A6（`CollectLoopCkeOpsA6` + `Collect*LoopInstrA6` 规则）：本地 TRANS
         4 码 / TransMem（`udfType==0` 且 `dmaOpCode ∈ {3,6}`——5 写带 notify
         每迭代触发远端 post，拒绝；通道须可解析，`IsExistRemoteDieInfo`）/
         REDUCE（`udfEnable` 的 TransMem 走网规约同样按 trans 槽位折叠）/
         ClearCKE（`clearMask==0`）；SetCKBit、LOAD、LOOP、JMP、Wait 均拒绝；
    2. wait/set 掩码在循环体内**配平**（`ConsumeProducedMask`/`ProduceMask`：
       每个 wait 的位须由更早的 set 产出，序列结束时全部产出的位被耗尽）；
    3. 跨展开 CKE/MS 资源无冲突（`CheckResourceConflict`：不同展开经
       ckOffset/msOffset 偏移后的 CKE/MS 编号不得复用；A6 的 MS 冲突检查覆盖
       本地搬运/TransMem `msIdMode` 的 MS 端与规约 msId 列表）；
    4. 搬运/规约长度 > 0（`ValidateLoopLen`）且展开数 × 迭代数 > 0；
    5. **A6 附加**：loop 体内 wait CKE ID 不得复用（`CheckA6LoopWaitCkeIdUnique`
       硬件约束，checker 报错中止、此处降级回退全展开）；各展开的迭代数/
       长度/通道取值一致（checker 各展开 `loopCnt` 相等 + `InstrMerge`/
       `MemMerge` 可合并性的性能侧近似——寄存器号经 xnIdOffset 偏移后可能
       指向不同寄存器）。
  - **合并发射**（`EmitMergedLoopInstrsV3` 语义）：循环体按 sample 展开
    （curExpandCnt=0）执行**一次**，逐槽位 wait → 数据任务 → set → 清 wait 位；
    数据任务 size = 单次长度 × (展开数 × 迭代数)（时序等价折叠，与 checker
    AddBatchTransMem/AddBatchReduce 的批量任务一致）；其余展开的 CKE 寄存器
    位图按 or/clear 幂等性重放，保证循环后寄存器状态与全展开一致（仅
    PostRecord 任务记账不重复建立）。发射窗口内的全部任务（数据任务与折叠后
    的 notify 任务）携带 `loop_cnt`（展开后的 loop 数 = 扩展 N 次后的 N+1）与
    `loop_iter_cnt`（单个 loop 自循环次数），总迭代数 = `loop_cnt` ×
    `loop_iter_cnt`，保留合并前循环细节供后续仿真使用；直行指令、回退全展开与
    SQE/CLI 路径的任务两字段为 0（无效，trace 统一输出 N/A）。
  - **量化效果**（REDUCE_SCATTER 实测 db，128 展开 × 50 迭代 × 13 指令）：
    任务数 192,005 → 66（数据任务 10 个、本地 post/wait 若干、远端同步节点
    35 个——远端 post 21 + 远端 wait 14，6.6 节），成图 + 仿真端到端
    ~55 s → ~6 ms，makespan 70,113.650 us → 2,753.711 us（本节合并消除
    12.8 万次任务级 notify 开销与 6.4 万次小搬运的 per-task 时延；队列并行化
    与远端同步节点计入后的最终值，7.4.1 节）。
  - A6 LoopGroup 配置校验（`loopInsCnt≤512`、`expandCnt≤511`、
    `expandOffset≤511` 且 `expandOffset≤loopInsCnt`，checker
    `TransformLoopGroupInstr`）不满足时告警跳过整个 LoopGroup（防御异常
    寄存器值导致的失控展开）；"loop 内 waitCKE 不得重复 CKE ID"校验见上
    合并前置校验 #5（`ccu_task_transform_instruct_a6_v3.cc:1126-1146`）。

### 6.5.1 合并 loop 流水线并发恢复（loop 标签 + 特征链消解）

**问题——折叠伪串行（合并对 notify 结构非时序等价）**：6.5 节合并对**数据任务**是
时序等价折叠（size × 总迭代数，带宽受限场景总耗时不变），但对循环体内的 CKE
wait/set 握手**不是**。真实微码执行时，LoopGroup 的 N 份展开并发发射，每份展开
内部再按 M 次内层迭代流水推进（典型每次迭代 4KB 量级），循环体内的
`TransMemToMs → setCKE → waitCKE → TransLocMsToRmtMem / TransLocMsToLocMem`
是**逐迭代的细粒度握手**：第 k+1 迭代的 HBM→MS 预取（memcpy 引擎）与第 k 迭代
的走网写 + 本地拷出（通道 / memcpy 引擎）重叠，稳态吞吐 ≈ max(各相位总量)。
合并折叠后，逐迭代握手退化为**单对** wait/record 节点，语义变为"前段全部数据
（× N × M 迭代）完成 → 后段才可开始"的**全量屏障**——流水线并发被折叠成串行。

以 `hccl_vm_install/data/hccl_vm_data.db` 首个 loop 模板为例（微码 0–10：
`TransMemToLocMs(4KB, set 0:1) → waitCKE(0:1) → 7× TransLocMsToRmtMem(各 4KB,
set 0:1..0x40) → TransLocMsToLocMem(4KB, set 0x80) → waitCKE(0:ff)`；LoopGroup
128 展开 × 256 内层迭代 = 32768 次 × 4KB = 128MB/任务）：

```
当前成图 (7.4.1 并行化后;  AA = loopgroup 前置任务代表, 即队列骨架中
                            loop 发射窗口前的最后一个节点):
  AA -> memcpy(mem->ms) -> record(0:1)                     ← 预取单元 (128MB, 95.95us)
  record(0:1) →[配对]→ wait(0:1) → 7× ub_write + memcpy(ms->mem)
                                    ↑ 全量串行: 后段 8 个单元 fork 自 wait(0:1),
                                      经配对边等预取 128MB 全部完成
  多个 { record(0:xx) / record(0x80) -> wait(0:ff) }       ← loop 末屏障 (保留)
```

- 真实执行：稳态 ≈ max(预取 95.95, 7 端口写 2526.48) us；
- 当前成图：串行 95.95 + 2526.48 us。
- **量化**（样例 db ALLGATHER，本节修复前后实测）：makespan **2657.52 →
  2561.54 us**。消除 95.95 us 串行预取相位；预取与本地拷出并发后经 HBM 动态共享
  各 191.79 us（流水线稳态的正确表达，7.5 节），7 端口走网写 2526.48 us 成为唯一
  关键路径。误差上界：两相位分属不同引擎（HBM 预取 vs 走网写）且时长相当时，
  loop 体耗时被高估接近 **2×**（串行 t1+t2 vs 稳态 max(t1,t2)）；同引擎相位
  （如纯 HBM 的预取+拷出）串行化误差被 HBM 动态共享吸收（总量不变）；
  误差随合并 loop 个数线性累加。

**正确性前提——合并校验不变量**：6.5 节合并前置校验 #2（wait/set 掩码体内
配平）保证合并 loop 体内**全部 wait 位由体内更早的 set 产出且末尾耗尽**——即
合并发射窗口内的全部 CKE 依赖均为 intra-loop 逐迭代流水线握手，不存在跨 loop
的真实数据依赖（否则校验失败回退全展开）。因此消解"体内中间 wait 及其消费的
set"不破坏 loop 与外部的同步：**loop 末 wait**（发射窗口内最后一个 wait，其后
无同 loop 数据任务）与 **mission 完成集**（骨架尾 + 全部单元尾，7.4.1 节）继续
承载 loop 后指令 / SQE 后继 / 同 queue 后一 mission 的队列级串行语义。

#### 方案（loop 标签 + 特征链识别 + 删除重接，迭代至收敛）

1. **loop 标签**：`_MissionContext` 维护当前发射窗口的 `_loop_tag`，命名
   `loopgroup_{i}_loop_{j}`（i = 派发上下文内 LoopGroup 指令执行序号，0 起；
   j = 组内 cur_loop_idx，0 起；loop 体内嵌 LoopGroup 时各自按发射窗口赋值，
   不构成层级）。`_emit_merged_loop` 窗口内置位、`add_task` 缺省继承到新字段
   `Task.loop_tag`（与 `loop_cnt` 同模式）。**仅合并路径打标**；全展开回退 /
   SQE / CLI / 直行指令为空串（trace 输出 N/A）。标签同时服务消解范围限定
   （跨 loop / 跨 mission 不误删）与 trace/dump 溯源。
2. **特征链识别**（发射期记录 + 组装前判定，免成图结果反查）：`_gen_waits`
   配对成功时将 (wait_task_id → 全部贡献 post task_id) 记入
   `ctx.wait_posts`（聚合 wait 的 match + extra 全部贡献者，6.4 节）。组装阶段
   （`expand_ccu_graph` 组装循环内、骨架/单元切分**之前**）对 ctx.nodes 序列
   扫描，命中特征链须同时满足：
   - W 为本地已配对 `Notify_Wait` 且 `loop_tag == T`（合并窗口内不存在远端
     wait——校验 #2 保证体内 wait 位全部本地产出）；
   - W 的全部贡献 post R* 均为 `loop_tag == T` 的 `Notify_Record`，且各自在
     ctx.nodes 中**紧随同标签数据任务 M\***（单元尾 record，即被折叠的生产者
     任务，如 TransMemToMs 折叠的 memcpy；数据任务 ⊆ `_QUEUE_DATA_TYPES`，
     不限于 Ub_Write_Or_Read——Memcpy/Reduce_Inline 同样存在本伪串行）；
   - **W 之后（ctx.nodes 序）存在同标签数据任务**（mid-body 判定；loop 末
     wait 其后无同 loop 数据任务，天然豁免；后一 loop 的数据任务标签不同，
     亦不误判）；
   - R* 均不紧随非数据节点（游离 set / sync 槽位的本地置位**不命中**——其
     等待的是远端同步等真实依赖，禁止消解）。
3. **删除与重接**（对 ctx.nodes 剪枝，而非组装后图手术）：
   - **W 恒删**：移出 ctx.nodes 与 dep 任务表，撤销其配对边（`remove_match`）
     与聚合 extra 边（`extra_edges[W]`）——这是打破屏障的本质操作；
   - **R\* 视消费者而定**：R 不出现在其他 wait 的 `wait_posts` 值中 → 连同
     删除（样例 db 的 set24 仅被 wt25 消费，直接删除）；否则**保留 R 只删
     W**（其保留边不受影响，loop 末 wait 照常等待——时序等价、语义无损的
     保守分支；post 位被多个 wait 部分消费时走此分支）；
   - **后继单元重接自动完成**：剪枝后组装逻辑重扫 ctx.nodes，原 fork 自 W
     的数据单元自动 fork 到 W 的前一骨架节点（即 loopgroup 前置任务 AA）；
     W 为骨架头（无前骨架）时该单元转 entry unit，入口依赖由 SQE 挂接边
     表达（7.4.1 节）。**无需任何边重接的图手术**——fork 父 / 骨架链 /
     完成集由既有组装规则基于剪枝后序列自然生成。
   - 寄存器 / CKE 记账**零改动**：配对消耗在发射期已完成，剪枝仅是图结构
     操作；loop 后指令的 wait 匹配沿用的记账结果不受影响（被删 post 的位
     已被 W 消费，无残留等待方）。
4. **迭代与统计**：重复 2–3 直至无新匹配（处理体内多级链，如
   预取→wait→中间段→wait→末段：删除首层后次层暴露为新的链头）；
   `CcuBuildStats` 新增 `pipeline_chains_broken`，summary 行以 `pb=N` 呈现。

修复后样例任务图（对照本节开头；被删节点为 record(0:1)/wait(0:1)，原
"wait(0:1) → ub_write" 的门控由 AA 承接，时序等价）：

```
AA -> wait(0:ff)                                        ← loop 末屏障 (骨架)
AA -> memcpy(mem->ms)                                   ← 预取单元 (record 已删)
AA -> 多个 { ub_write_or_read -> record(0:xx) }          ← 7 路写, 与预取并发
AA -> memcpy(ms->mem) -> record(0x80)                   ← 拷出单元, 与预取并发
多个 { record(0:xx) / record(0x80) -> wait(0:ff) }      ← loop 末等全部数据完成
```

#### 实现细节（代码落点）

- `simulator/task_model.py`：`Task` 增加 `loop_tag` slot（默认空串），
  `to_trace_args` 输出 `loop_tag`（空串 → `N/A`，与 `loop_cnt` 同口径）；
- `ccu_graph_builder.py`：
  - `_MissionContext.__init__`：`_loop_tag = ""`、`_lg_seq = 0`、`wait_posts = {}`；
  - `_exec_loop_group`：进入时记 `lg_tag = f"loopgroup_{self._lg_seq}_loop_{
    cur_loop_idx}"`（透传给 `_emit_merged_loop`），LoopGroup 处理完后
    `self._lg_seq += 1`；
  - `_emit_merged_loop`：窗口内置 `self._loop_tag = lg_tag`（`finally` 复位，
    与 `_loop_cnt` 同模式）；
  - `_gen_waits`：配对成功后 `self.wait_posts[wait_task.task_id] =
    [p.task_id for p in posts]`；
  - `expand_ccu_graph` 组装循环开头调用 `_break_pipeline_chains(ctx, dep)`
    （返回删除链数，累入 `stats.pipeline_chains_broken`）；
- `graph_dumper.py` / `simulator/report.py`：dump `nodes[]` 与 trace args
  增补 `loop_tag`；DOT/HTML 图例同步（可选，随 loop_cnt 呈现）。

trace 呈现（`export_trace`，仅显示层、不影响仿真）：虚拟从流上携带
`loop_tag` 的任务，其显示行改挂本流的 loop 标签行（行名即标签，如
`loopgroup_0_loop_0`），表明其为 loop 内指令转换而来；同一从流的其余
任务保持原显示行——跨 mission 复用的混合从流因此拆为原行 + 标签行。
显示行按 `(stream_id, loop_tag)` 划分，不同虚拟流的任务不合行（保留
从流间任务划分）；主流任务（含 loop 末 wait）一律保持原行；任务
`args.stream_id` 保留真实流号。


```
_break_pipeline_chains(ctx, dep):
    broken = 0; changed = True
    while changed:
        changed = False
        for W in [n for n in ctx.nodes if 类型 == Notify_Wait 且 n.loop_tag]:
            posts = ctx.wait_posts.get(W.task_id);  T = W.loop_tag
            if not posts:                              continue   # 未配对/远端
            if 任意 R ∈ posts: R.loop_tag != T:        continue   # 跨 loop 来源
            if 任意 R ∈ posts: R 的前一 ctx.nodes 节点不是同标签数据任务: continue
                                                                # 游离 set/sync
            if W 之后无同标签数据任务:                   continue   # loop 末屏障
            for R in posts:                                        # --- 命中 ---
                if R ∈ 其他 wait 的 wait_posts 值: continue        # 有保留消费者
                dep 删除 R; ctx.nodes 移除 R                       # 单元尾改由 M 承接
            dep.remove_match(W); dep.extra_edges 弹出 W
            dep 删除 W; ctx.nodes 移除 W
            broken += 1; changed = True; break                     # 重扫
    return broken
```

无环保证：剪枝只删节点与边、不添边，且删除的 W/R 均为发射窗口内节点，
fork/汇合边由组装期基于剪枝后序列重算（7.4.1 节无环论证继续成立）；
`_drop_cyclic_matches`（5.4 节）与仿真器运行期死锁检测兜底不变。

#### 方案评估（优缺点）与备选

**优点**：

1. 不动 6.5 节合并语义与 checker 对齐逻辑——事后消解，风险隔离；
2. 组装前对 ctx.nodes 剪枝（而非对成图结果图手术）：fork 父 / 骨架链 /
   完成集由既有组装规则自然生成，规避"删节点后边重接"的图手术正确性负担
   （组装后手术需手工重接 fork 边、骨架链与完成集，易错）；
3. 发射期记录 wait→posts 配对，识别不依赖成图结果的依赖形状——对 7.4.1
   组装规则演化（fork 父选择、单元切分调整）不敏感；
4. loop 标签限定消解范围（同 loop 同 mission），杜绝跨 loop / 跨 mission
   误删；标签同时反哺 trace/dump 溯源；
5. 删除节点而非仅删配对边：避免保留一个语义悬空的未配对本地 wait（虽然其
   实测走 `local_wait_delay` = 30ns、时序无差，但 trace 统计口径干净）。

**缺点 / 风险**：

1. 特征链是结构启发式，识别条件与发射 / 组装逻辑存在耦合，需随两者演进
   维护（已通过"发射期记录 + 组装前剪枝"把耦合面压缩到 ctx.nodes 序列一层）；
2. `Task` 模型扩字段，trace/dump/JSON 呈现需同步（11 章口径）；
3. "完全并发 + 带宽管理器仲裁"是流水线**稳态**近似：忽略流水线填充/排空
   延迟（单迭代 4KB 量级，亚微秒，可忽略）与逐迭代握手的**速率**开销
   （若 CKE 引擎吞吐为瓶颈需专门速率模型，当前折叠 / 删除均按单次计，
   与修复前口径一致，登记 10.3 节 #21）；
4. 全展开回退路径（`mf > 0`）不经此消解——其逐迭代 wait/record 为真实 4KB
   粒度握手，但迭代间重叠同样被队列链串行化；该路径节点数大、当前样例未
   触发，作为已知残留（10.3 节 #21）。

**备选方案对比**：

| 备选 | 思路 | 优点 | 缺点 | 结论 |
|------|------|------|------|------|
| B. 发射期直判（免标签 / 免识别） | `_emit_merged_loop` 按 slot 结构直接判定 mid-body wait（其后仍有数据槽位）并跳过发射其 wait/配对 | 实现最简，无 Task 字段扩展；判定信息（slot 序）在手，无启发式 | 丢失 trace 中握手节点与 loop 溯源；跳过发射改变寄存器配对记账（loop 末 wait 吸收预取 set，时序无害但与 checker 行为差异扩大）；游离 set / sync 场景需另加排除条件 | 可行简化变体；若不需 trace 呈现可选 |
| C. 分段流水线建模 | 数据任务折叠为 K 段（8~16），段间保留 wait/record，显式建模填充/排空 | 含流水线填充效应，更精细 | 节点数 × K、K 值无物理依据；稳态下与完全并发仅差首尾一段 | 过度设计，不取 |
| D. 仿真器软依赖边 | 新边类型"流水线边"（child 在 parent 完成 X% 后开始） | 图结构不变 | 仿真器语义复杂化、X 无物理依据；且串行链结构（单元 fork 自 wait）仍需成图层修改 | 否决 |

结论：采用原方案（标签 + 特征链 + 删除），按上述第 2/3 步的实现位置与安全
条件强化（发射期记录 + 组装前剪枝 + mid-body 判定 + 消费者保守分支）。

**实现状态（已落地）**：`Task.loop_tag` 字段（task_model / trace args /
dump `nodes[]` / input_builder 回路）+ `DependencyGraph.remove_task()`
（连带清理 extra 边与 notify 配对）+ `_break_pipeline_chains()`（组装循环
开头调用，统计 `pipeline_chains_broken`，summary 呈现 `pb=N`）+ trace
loop 从流改名呈现（`export_trace`，见上）。验证：
样例 ALLGATHER db 跨引擎场景 makespan 2657.521 → 2561.541 us（56 → 54
任务，pb=1，9 单元全部 fork 重接前一骨架节点 rwt22，与手工原型逐项一致）；
归档库回归——同引擎相位场景（mv→mv 均 HBM）makespan 不变（零误差场景，
理论预期）、SQE-only 库 pb=0 不受影响。`loop_tag` 作为任务持久属性保留，
供后续性能仿真（如逐 loop 带宽/速率建模，10.3 节 #21）消费。

### 6.6 成图算法（伪代码）

```
expand_ccu(device_id, die_id, sqe_params, instr_space, dep_graph, reg_model):
    SQE 按连续段归并为派发上下文 (6.1 节: 非连续 = 新 dispatch, 独立窗口)
    同 queue 上下文按派发序串行; 各 queue 取首上下文入活跃集
    while 活跃集非空:
        progress = False
        for q in 活跃集的各 queue 当前上下文:        # 跨 queue 轮询
            while ptr[q] 未到窗口尾:
                instr = decode(instr_space[ptr[q]])
                if instr 是 WAIT 且 reg_model.cke 位不足:
                    break                              # 本 queue 暂停, 轮到其他 queue
                nodes = transform(instr, reg_model)   # 6.3 节映射; LOAD 仅改寄存器
                for node in nodes: 挂到 queue q 的链尾
                if instr 是 SET_CKE: reg_model.seen_post 登记
                if instr 是 WAIT:     与 seen_post 配对 → 聚合 wait 节点 (6.4 节)
                if instr 是 LOOP:     合并路径 (6.5 节校验) 或按迭代重复解释循环体
                ptr[q] += 1; progress = True
            if ptr[q] 到窗口尾: 当前上下文完成, 同 queue 后一上下文入活跃集
        if not progress:
            # 停滞 = 存在依赖远端 CKE 的 wait (跨 rank, 单 rank 图中不可见)
            对停滞 queue: 生成远端 Notify_Wait 节点 (notify_id+mask, 固定时延),
                再重放该指令其余部分 (本地 set / 数据搬运), 继续推进
    SQE 流挂接 (7.4.1 节): 占位自 SQE 流移除 (任务图不引入占位节点);
        入口边 (SQE 前驱 → 队列骨架头) / 出口边 (mission 完成集 → SQE 后继)
        经 extra 边表达
    组装切分 (7.4.1 节): 各 mission 队列序列 → 骨架 (留 queue 链) + 异步单元
        (挂 #v 虚拟流, stream_id 改指虚拟流), fork (前驱骨架→单元头) /
        完成集汇合 (骨架尾+全部单元尾 → SQE 后继 / 同 queue 后一 mission 入口)
        extra 边衔接
```

- **跨 rank post-wait 不建边、但保留同步任务节点**（单 rank 成图，与 checker 的
  主要区别）：checker 的 `GenerateRawCcuGraphsV3`（`ccu_task_transform_v3.cc:1275-1406`）
  让多 rank 的 CCU 图交替推进、跨 rank post 相互解锁；本插件只有单 rank 指令数据：
  - post 到远端 CKE / 远端寄存器通知的指令 → 生成远端 `Notify_Record` 节点
    （src=本端、dst=INVALID → 固定 `remote_record_delay`，不登记本地 CKE 记账、
    不参与配对）：
    - A5：SyncCKE 的 `rmtCKEId/locCKEMask`、SyncGSA/SyncXn 的
      `setRmtCKEId/setRmtCKEMask`、TransLocMSToRmtMS 的 `setRmtCKE`（与 checker
      `AddPost` 的触发条件一致）；
    - A6：TransMem `dmaOpCode=5` 的写后 notify 与 `0x1A` 远端原子计数、
      SyncXnWtx 的 `notifyValid=1` notify、SyncAtx 的原子同步——远端目标类型
      （Xn/CKE）由对端地址映射解析（checker `GetXnAndTypeIdByAddr`），单 rank
      不可得，一律按远端 record 近似（mask 取通知值低 16 位仅供 trace 显示）。
    SyncGSA/SyncXn（A5）与 SyncXnWtx 主写（A6）的值同步仅为远端寄存器写
    （checker 在对端图上建节点/写状态），本端不单独生成任务。
   - wait 所需的位无法由本 rank post 满足（停滞检测发现）→ 该 wait 依赖远端 → 生成
     远端 `Notify_Wait` 节点（固定 `remote_wait_delay`，不配对），并重放指令其余
     部分；循环体内的未满足 wait 同样生成远端 wait 节点后继续。部分满足时：已满足
     位与本地 post 正常配对（聚合为 1 个本地 wait 节点，6.4 节），未满足位生成一个
     远端 wait 节点。A6 独立 Wait 指令（Xn 值比较）未满足时同机制：顶层停滞 →
     停滞消解生成远端 `Notify_Wait`（任务 id 标签 `xwt`）后直通。
   - 任务图中不凭空产生任何节点：全部节点来自原始 db 数据或微码指令转换
     （挂接依赖经 extra 边表达，见 7.4.1 节）。
- 本 rank 内的 post-wait 正常配对：同一 device 的多个 queue 之间、多个 die 的 CKE
  均在本地图内，按 6.4 节规则建边。

### 6.7 与 checker CCU 成图的简化对照

**CCU 任务的 rank/eid/direction 语义**：

- **通道类走网任务**（write/read）：本端 rank 取 opDetails.rankId；对端 rank 经
  通道对端 eid（reid）查 eid→rank 索引（4.8 节），查不到填 INVALID +
  `rank_unresolved` 计数。等价于 checker `GetRemoteRankByChannel`
  （`g_allRankChannelInfo`: channelId → 对端），但以 RankEidMap 的通信域 rank
  替代 checker 的物理 dstDeviceId。
- **规约任务**（REDUCE 微码）：MS 为 rank 本地 buffer（checker 对全部 msId 恒
  `GenSliceFromMs` + `AddLocalBatchReduce`），规约恒为本地 HBM 操作，src/dst 均填
  本端 rank；跨 rank 数据已在先行的 Rmt* 角色搬运中计费，规约不再重复走网。
- **本地任务**（本地搬运 Memcpy、本地规约 Reduce_Inline）：src/dst 均填本端 rank。
- **is_write**：CCU 产出的 UB 任务均显式携带（write=True / read=False），
  不依赖仿真器 rank 启发式。
- **本端 rank 自校验**：通道 leid ↔ 索引解出的 rank 若与 opDetails.rankId 不一致，
  以 opDetails 为准并告警（eid 编号系错位防护，如 deviceId 与 rankId 存在偏移）。

| checker 环节 | light_performance | 理由 |
|--------------|-------------------|------|
| 两阶段（raw 成图 → queue 并行化 → 回接） | 单阶段解释执行 + 组装期队列并行化切分（7.4.1 节：`ApplyCcuQueueParallelizationV3` 简化移植，骨架/单元切分 + fork/汇接与挂接 extra 边 + `#v` 虚拟流） | 免校验，无需保留可改造的 raw 中间形态；并行化在组装阶段一次完成（复用粒度等差异见 7.4.1 节） |
| Loop 合并（BatchTransMem/BatchReduce、资源冲突检查、producedMasks 配平） | **同等实现**（6.5 节：前置校验 + 折叠发射 + 回退全展开；内存切片合并不需要——仿真只关心数据量与时序） | 对齐 checker 语义消除节点膨胀 |
| SQE 归并为 TaskCcuGraph（`IsContiguousCcuSqe` 连续才并入 + `CcuMissionKey` 最新节点追加） | **同等实现**（6.1 节：连续段切分派发上下文，各上下文独立窗口重放 + 顺序消费 args；寄存器/CKE 跨上下文共享，同 queue 上下文按派发序串行） | 对齐 checker 多次 dispatch（窗口回绕重发）语义，曾因归并为单一上下文丢失第二次 dispatch 数据搬运（10.3 节 #20 已修复） |
| GetRemoteRankByChannel（channelId → dstDeviceId/dstDieId，经 Communicator 表映射 rank） | 对端 eid（reid）→ RankEidMap/ranktable 索引 → 通信域 rank（4.8/6.7 节） | estimate 模式 Communicator 表仅本 rank 一行；ccuChannels.rankId 字段为 endpoint deviceId 不可用 |
| AllRankParamRecorder 完整地址空间映射（XN/CKE/PFE/CHANNEL/JETTY/MISSION/MS …）+ HBM 数据内容 + 远端地址→寄存器反查（`GetXnAndTypeIdByAddr`） | 仅保留 xn/cke/gsa/通道 四类；A6 Load/Store 的 HBM 数据面与远端目标反查不建模（6.4 节、`_MissionContextA6` 偏差清单） | 仿真只需长度与对端；数据内容/远端寄存器号不影响时序 |
| A6 checker 报错中止场景（未支持 dmaOpCode/dstType、clearType!=1、clearMask!=0、LoopGroup 配置非法、LoadX/StoreX 间接号越界等） | 降级为告警 + 跳过数据面（同步语义仍执行）或跳过指令 | 性能工具容忍不完整/异常数据，保持队列骨架完整 |
| CcuTraceInfo 溯源、ccu_convert_config.json 生成 | 去除 | 校验/重写工具专属 |
| CheckAllPostMatch 报错 | 降级为告警 + 诊断统计 | 适应单 rank 数据 |
| 跨 rank post 解锁他人 wait（多 rank 联合推进） | **不建跨 rank 边**：跨 rank post/wait 生成远端同步节点（固定时延），对端何时就绪以固定时延近似 | 单 rank 成图（核心区别） |

### 6.8 A5 与 A6 指令集差异概览

| 维度 | A5（V1/Ascend950） | A6（V2/Ascend960） |
|------|--------------------|--------------------|
| LOAD 类 | 7 条：含 GSA 系 | 18 条：统一 X 寄存器 + LoadX/StoreX/ClearX/Nop + 算逻（Add/Sub/Mul/And/Or/Not/Xor/Shl/Shr/Popcnt） |
| CTRL 类 | 5 条：Loop/LoopGroup/SetCKE/ClearCKE/Jmp | 7 条：+ SetCKBit/Wait/Fence |
| TRANS 类 | 12 条：Mem↔MS 全组合 + SyncCKE/GSA/Xn | 7 条：本地 4 种 + 统一 TransMem（dmaOpCode 区分）+ SyncWtX/SyncAtX |
| REDUCE 类 | Add/Max/Min（msId 列表寻址，≤8 MS） | ReduceAdd/Max/Min（同一结构，无内嵌 waitCKE） |
| Loop 寄存器 | v1.loop{xnId, startInstrId, endInstrId}，loopCnt 在 Xn | v2.loop{xpId(上下文), xnId, xmId(迭代数), mode}，mode==1 条件展开（6.5 节） |
| LoopGroup 参数 | Xn：LoopNum[61:55]/RepeatNum[54:48]/NoRepeatNum[47:41]；Xm：gsaOffset/MSOffset/ckeOffset | Xn：loopInsCnt:10/expandOffset:9/expandCnt:9；Xm：ckOffset:10/msOffset:11/gsaOffset:32；Xp：xnIdOffset:32 |
| 寄存器号寻址 | 载荷字段直用 | `GetXnId`：bit15=1 时按循环 xnIdOffset 偏移（6.4 节） |
| 内嵌同步字段 | TRANS/REDUCE 普遍携带 waitCKE + setCKE + clearType | 本地 TRANS/REDUCE 仅 setCKE（wait 由独立 SetCKBit/ClearCKE/Wait 指令承担） |
| 通道/长度取值 | 通道为指令立即数字段 | 通道/长度均为 Xn 寄存器值（`Xn[xcId]`/`Xn[xlId]`） |

两代指令集均已成图落地（A6 解释器 `_MissionContextA6` 复用 A5 的任务发射/
CKE 配对/推进/合并发射基础设施，仅覆写指令解码与 loop 组织）；A6 指令 →
任务类型总表见 11.3.3 节，与 checker 的既定偏差清单见 `_MissionContextA6`
类 docstring 与 6.7 节对照表。

---

## 7. 性能仿真模块（子需求 4，按 HCCLSIM 基准收编）

> 本章为 `simulator/` 子包（自 `docs/light_performance/HCCLSIM/hccl_sim/` 平移）的
> 设计说明，接口以 HCCLSIM 为权威基准。与旧版的差异及遗留项见第 10 章。

### 7.1 总览

```python
sim = HcclSimulator(config, dep_graph, topo_resolver)
result = sim.run(trace_filepath=None, max_trace_events=0)   # → dict
```

仿真器接收三个参数：

| 参数 | 类型 | 内容 | 构建方 |
|------|------|------|--------|
| `config` | `SimConfig` | 时延、带宽、利用率配置 | 调用方 |
| `dep_graph` | `DependencyGraph` | 任务节点 + 依赖关系 | `graph_builder`（db 路径）/ `input_builder`（JSON 路径） |
| `topo_resolver` | `TopoResolver` | eid → 物理端口解析（ranktable.json） | 调用方 |

`run()` 返回结果 dict（不再返回裸 makespan）：

```python
{
    "makespan": float,        # 端到端耗时 (秒)
    "makespan_us": float,     # (微秒)
    "trace_file": str|None,   # trace 输出路径
    "trace_events": int,      # trace 事件数
    "sim_time_s": float,      # 软件运行耗时 (秒)
    "tasks": list[Task],      # 全部任务 (含 start/end_time)
    "tasks_by_stream": dict,  # int 流 id → [Task] (链序)
    "ports": dict,            # tuple(ports) → NetworkPort
}
```

仿真器不接收任务序列列表，不推断依赖——全部从 `dep_graph` 获取。

### 7.2 参数一：`config: SimConfig`

**单位约定**：内部时延=秒、带宽=B/s；**JSON 配置单位时延=us、带宽=GBps**
（`from_dict`/`to_dict` 自动换算）。端口标识为 `"die_id/port_number"` 字符串
（如 `"1/0"`），与 ranktable.json 一致。

#### 字段定义

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `local_record_delay` | float | `1e-6` | 本地 record 固定耗时；JSON 值单位 us |
| `remote_record_delay` | float | `1e-6` | 远端 record 固定耗时；JSON 值单位 us |
| `remote_wait_delay` | float | `1e-6` | 远端 wait / 未匹配本地 wait 固定耗时；JSON 值单位 us |
| `port_bandwidth` | dict | `{}` | `"die/port"` → 带宽（B/s）。一个 eid 可对应多端口，按端口逐个配置，总带宽求和 |
| `default_port_bandwidth` | float | `53.125e9` | 未配置端口默认带宽（425 Gbps = 53.125 GBps）；JSON 值单位 GBps |
| `hbm_bandwidth` | float | `1.4e12` | HBM 带宽，用于 Reduce_Inline/Memcpy；JSON 值单位 GBps |
| `bw_profile_set` | BandwidthProfileSet? | `None` | 带宽利用率配置集，None 时利用率为 100% |

#### 端口带宽配置格式

```json
{"port_bandwidth": {"1/0": 53.125, "0/7": 53.125}}
```

#### 带宽利用率配置 (`bw_profile_set`)

```
BandwidthProfileSet
├── profile_low:   BandwidthProfile  ← 共用端口并发数 < 8 时使用
├── profile_high:  BandwidthProfile  ← 共用端口并发数 ≥ 8 时使用
└── THRESHOLD: 8
```

每个 `BandwidthProfile` 是 数据量(32B~128MB) → 利用率(0~1) 的映射表，查询时向下取整到
不大于该数据量的最大档位（floor 匹配）。

JSON 格式：
```json
{
  "profile_low":  {"32": 0.30, "64": 0.35, "...": 0.90, "134217728": 0.95},
  "profile_high": {"32": 0.20, "64": 0.25, "...": 0.84, "134217728": 0.85}
}
```

### 7.3 参数二：`dep_graph: DependencyGraph`

#### 四层内容

```
DependencyGraph
├── tasks: dict[str, Task]                       ← 任务节点 (带全部参数)
├── stream_chains: dict[int, list[str]]          ← 串行依赖 (int 流 id)
├── notify_matches: dict[str, str]               ← notify 依赖 (wait_task_id → record_task_id)
└── extra_edges: dict[str, list[str]]            ← 额外依赖边 (child_task_id → [parent_task_id, ...])
                                                  ← 来源: CCU 队列并行化 fork/汇接边与 SQE 挂接边 (7.4.1 节)
```

#### 第一层：任务节点 (`Task`)

每个 Task 携带执行所需的全部参数（无效值统一用 `INVALID = 0xFFFFFFFF`，
eid 为空串，字符串属性为 `""`）：

| 字段 | 类型 | 适用任务类型 | 仿真器用途 |
|------|------|------------|-----------|
| `task_id` | str | 全部 | 依赖图索引 |
| `task_type` | TaskType | 全部 | 决定时延模型（内部分类，7.3 节） |
| `task_name` | str | 显示用 | trace/dump 显示名：profInfo.profTaskType 的 op 名（`_add_task` 统一填充，11.4.1 节）；空时 `display_name` 回退 `task_type.value`（CCU 子任务/CLI 输入） |
| `stream_id` | **int** | 全部 | 串行链归属 |
| `notify_id` | int? | Notify_Record/Wait/Write_With_Notify | 依赖配对标识（CCU 路径 = cke id）；Write_With_Notify 仅显示（其 notify 面向对端，不参与配对） |
| `mask` | int | Notify_Record/Wait | notify 位掩码（CCU 路径）：notify_id + mask 唯一标识一个 notify 动作；wait 节点携带聚合 mask（一条 wait 指令全部贡献 post 的消费位之或，6.4 节）；SQE 路径与 0=未填 |
| `src_rank` / `dst_rank` | int? | notify/传输 | 本地判定（src==dst）；读方向本端判定 |
| `size` | int | Memcpy/Reduce_Inline/Ub_*/Write_With_Notify | 计算时延 + 查利用率 |
| `port` | str/list? | （保留字段） | 端口由 eid 实时解析，db 路径不填 |
| `src_eid` / `dst_eid` | str | Ub_*/Write_With_Notify | eid → ranktable → 物理端口 |
| `transport_type` / `link_type` | str | 显示用 | msprof 对账维度（profInfo transportType/linkType 名，11.4.2/11.4.3 节，`_add_task` 统一填充；CCU 子任务不填） |
| `data_type` | str | 显示用 | trace/报告标注 |
| `protocol` | str | 显示用 | 通道 CommProtocol 名（11.4.2 节注；无通道信息时空） |
| `jetty_id` | int? | Ub_*/Write_With_Notify | trace 标注 |
| `is_write` | bool? | 走网任务 | 计费方向与端口查询 eid 选择（10.3 节 #1） |
| `start_time` / `end_time` | float | 全部 | 仿真器填充（-1=未执行） |

7 种 TaskType：

| TaskType | 时延模型 | 说明 |
|----------|---------|------|
| `NOTIFY_RECORD` | 固定时延（本地另触发 Event） | 本地 = `is_local_notify`(src==dst)；未配对本地 record 仅固定时延 |
| `NOTIFY_WAIT` | 等 Event / 固定时延 | matched→等 record 的 Event；未匹配/远端→固定 `remote_wait_delay` |
| `REDUCE_INLINE` | HBM 动态共享 | `size / hbm_bandwidth`，与其他 HBM 任务共享带宽 |
| `MEMCPY` | HBM 动态共享 | 同上（HBM 拷贝，非网络传输） |
| `UB_INLINE_WRITE` | 固定时延（同 Notify_Record：local/remote record_delay） | notify inline 写；SQE 成图路径不产出，仅旧输入路径（task.txt）使用 |
| `UB_WRITE_OR_READ` | 端口带宽共享（write/read 两方向独立计费） | 网络数据收发；`is_write` 由成图显式携带（SQE 走网任务两方向统一映射此类型） |
| `WRITE_WITH_NOTIFY` | 端口 write 方向共享 + 写后 notify 固定时延 | 网络写 + 附带 notify 单任务，与 db `WRITE_WITH_NOTIFY` (7) 一一对应；notify 按远端 record 固定时延（对端 wait 不在本 db） |

#### 第二层：串行依赖 (`stream_chains`)

```
int 流 id → [task_id_0, task_id_1, ..., task_id_n]
```

- 列表顺序 = 执行顺序：task[i] 完成后才能执行 task[i+1]
- 每个 stream_id 启动一个 simpy 进程，按链顺序 `yield` 执行
- 不同 stream 之间天然并行
- db 路径的流 id 编址：SQE 流直用原始 streamId；CCU 队列链与虚拟流自基址 `1<<20`
  分配（6.5 节，虚拟流逻辑名带 `#v{n}` 后缀），逻辑名经 `stream_labels` 保留；
  `GraphBuilder.stream_meta` 在此之上给出 real/ccu/virtual 三类分类与虚拟流的
  parent_sid（7.4.1 节），供报告/trace/dump 标注

#### 第三层：notify 依赖 (`notify_matches`)

```
wait_task_id → record_task_id
```

显式 1:1 配对：哪个 wait 等哪个 record。同一 notify_id 多个 record 时只有一个
与 wait 配对，未配对 record 走固定时延。

#### 第四层：额外依赖边 (`extra_edges`)

```
child_task_id → [parent_task_id, ...]
```

- 语义：child 开始执行前须等待**全部** parent 完成（纯时序约束，不承载
  record/wait 事件语义——事件配对仍由 `notify_matches` 表达）。
- 当前来源：CCU 队列并行化的 fork 边（前驱骨架 → 单元头）、mission 完成集
  汇合边（骨架尾 + 单元尾 → SQE 后继 / 同 queue 后一 mission 入口）、CCU 子图
  在 SQE 流上的挂接边（入口/出口，见 7.4.1 节），以及聚合 wait 节点的多父边
  （wait → 全部贡献 post，6.4 节——首个贡献 post 走 `notify_matches`，其余走此层）。
- 写入侧 `add_edge` 自动去重（同一 child-parent 对只记一条）；仿真器经
  `get_extra_parents` / `extra_parent_ids` 查询。
- 无环性：上述边恒沿原指令顺序"向前"连（骨架在单元前、单元在汇合点前、
  占位位置前后挂接、post 先于消费它的 wait 指令登记），成图方向保证无环，
  并计入 `_drop_cyclic_matches` 的环检测兜底（5.4 节）。

### 7.4 仿真器消费流程

```python
def run(self, trace_filepath=None, max_trace_events=0):
    # 1. 取出全部任务节点
    tasks = list(dep_graph.tasks.values())

    # 2. 创建同步原语: 仅 matched wait 创建 Event
    for wait_task_id in dep_graph.matched_wait_ids:
        notify_events[wait_task_id] = env.event()
    for parent_id in dep_graph.extra_parent_ids:  # extra 边的全部父任务
        task_done_events[parent_id] = env.event() # 须在流进程启动前预建:
                                                  # 父先完成、子后建 Event 会永久等待 (7.4.1 节)

    # 3. HBM 带宽管理器 (Reduce_Inline + Memcpy 共享, 利用率恒 1.0)
    hbm_mgr = _BandwidthManager(env, config.hbm_bandwidth, 空利用率表)

    # 4. 每个 stream 启动一个 simpy 进程, 按 chain 顺序串行执行
    for task in chain:
        for pid in dep_graph.get_extra_parents(task.task_id):
            yield task_done_events[pid]           # extra 边 (7.4.1 节), start_time 不含等待
        if NOTIFY_RECORD:
            yield timeout(calc_notify_record_delay(task, config))
            if 本地且 matched: 触发匹配 wait 的 Event
        elif NOTIFY_WAIT:
            if 本地且 matched: yield notify_events[task_id]   # 等 Event
            else:              yield timeout(remote_wait_delay)
        elif REDUCE_INLINE / MEMCPY:
            yield hbm_mgr.send(task_id, size)                 # HBM 动态共享
        elif UB_INLINE_WRITE / UB_WRITE_OR_READ / WRITE_WITH_NOTIFY:
            is_write = task.is_write  # 显式方向; 旧输入路径 None 退回启发式
            eid = src_eid if is_write else dst_eid
            ports = topo_resolver.get_ports_by_eid(eid)       # 缺失则抛错
            port = 查/建 NetworkPort(ports, 总带宽)
            yield port.send(task_id, size, is_write)          # 端口带宽共享
            if WRITE_WITH_NOTIFY:
                yield timeout(calc_write_with_notify_delay(task, config))
                                                           # 写落地后 notify:
                                                           # 固定 record 时延
                                                           # (对端 wait 不在本 db)
        task.end_time = env.now
        if task_id in task_done_events: task_done_events[task_id].succeed(env.now)
    # 5. makespan = max(end_time); 组装结果 dict (含 trace 导出)
```

时延公式统一收敛在 `task_timing.py` 单点维护。

#### 7.4.1 CCU 队列并行化与虚拟流（extra 边）

**问题**——第 6.5/6.6 节产出的 CCU 队列链是逐任务串行流：同一 queue 的全部
数据搬运、wait、set 按序执行。真实硬件上 queue 的异步搬运单元并发执行，
串行化造成三重失真：

| 失真 | 机理 |
|------|------|
| makespan 高估 | 异步搬运被骨架（wait/set）逐个隔开，无法重叠 |
| 利用率档位失真 | 同一 queue 的并发搬运退化为逐个执行，`_BandwidthManager` 观测的端口并发数恒为 1，永远查 profile_low 档（7.5 节） |
| 端口峰值并发统计失真 | `peak_concurrency` 恒为 1，无法反映并发共享带宽行为（8.2 节端口级指标） |

**checker 参照**——`ApplyCcuQueueParallelizationV3`（`ccu_task_transform_v3.cc:1052`）：
raw 成图后二次遍历改造，`CollectAsyncBoundRecordNodesV3` 收集异步搬运节点，
`QueueIdReuseAllocatorV3` 按 `[startNodeId, afterNode]` 区间复用虚拟 queue id，
fork/join 边回接骨架。

**简化移植**（`ccu_graph_builder.py` 组装循环，组装阶段一次完成，无 raw
中间形态）——对每个 mission 的队列节点序列做切分：

1. **单元识别**：`_QUEUE_DATA_TYPES`（`Memcpy`/`Ub_Inline_Write`/
   `Ub_Write_Or_Read`/`Reduce_Inline`）数据任务 + 紧随的连续
   `NOTIFY_RECORD` 链 = 一个**异步单元**；其余节点（wait / 游离 set /
   远端同步节点）为**骨架**，留在 queue 串行链——队列链即骨架本身，
   不再引入任何首尾占位节点。
2. **fork 边**：`add_edge(单元头, 前一骨架节点)`——单元开始前须等队列推进
   到其前驱骨架（相邻单元共享同一 fork 父节点）；骨架之前出现的单元
   （mission 首节点即为数据任务）无骨架前驱，其入口依赖改由挂接边表达。
3. **完成集汇合**：mission 完成集 = 骨架尾 + 全部单元尾，经 extra 边汇合到
   ——同 queue 后一 mission 的入口（骨架头 + 骨架前单元头，队列串行语义；
   骨架尾之间的顺序已由 queue 链串行覆盖）以及 SQE 流后继任务（出口边，
   graph_builder 挂接阶段添加）。
4. **虚拟流改挂**：每 queue 一组计数器/复用池（`vstream_seq`/
   `vstream_free`/`vstream_busy`），单元任务的 `stream_id` 改指
   `ccu_d{die}_q{queue}#v{n}`（同一 `1<<20` 基址空间），单元链登记入
   `result.virtual_chains` 并 `add_stream_chain`——串行链仅约束骨架与
   单元内部任务顺序。
5. **SQE 流挂接**（`graph_builder._expand_ccu` 消费 `result.attach`）：
   CCU 占位自 SQE 流链移除（任务图中不产生任何占位/凭空节点），
   入口边 = SQE 前驱任务（含紧邻的前序 CCU mission 完成集）→ 骨架头与
   骨架前单元头；出口边 = mission 完成集 → SQE 后继任务。全部经 extra 边
   表达，时序语义为：SQE 前驱（及前序 mission 完成集）完成后子图才可开始、
   子图 mission 完成集全部结束后 SQE 后继任务才可开始。
   全部任务为占位的 SQE 流移除空链。

**CKE 时序保留**：切分不动 `add_match` 配对——record 落入哪个单元、wait
留在骨架都不影响事件配对（wait 经 notify_events 跨流等待）。实测样例中
收尾 memcpy 单元的 fork 父 wt28 恰在前一 read 单元尾部 set 的记账 wait 上，
跨流事件 + fork 边共同保证"读完成 → 收尾拷贝"的原时序。**例外**：合并 loop
发射窗口内的 mid-body wait/record 配对为折叠伪串行屏障（逐迭代流水线握手被
折叠成全量屏障），经 6.5.1 节特征链消解在组装前剪除——本节切分基于剪枝后
序列，骨架/单元/fork 结构不受影响。

**与 checker 的复用规则差异**：

| 维度 | checker (V3) | light_performance |
|------|--------------|-------------------|
| 复用粒度 | afterNode 区间（join 后继节点）级 | mission 级（进入新 mission 时前序全部释放） |
| mission 内复用 | 允许（DAG 显式边，无伪依赖） | 不复用（同 mission 单元彼此并发，链式模型下复用同流会引入伪串行） |
| 残留严格性 | GRAPH_UNMATCHED 报错退出 | 容忍（性能工具，6.4 节降级告警） |

跨 mission 复用的安全性：复用同一虚拟流的单元在该流链内串行（后一 mission
的单元链接在前一 mission 单元链之后），时间轴无重叠。

**无环保证**：fork（骨架 → 单元头）与完成集汇合边（单元尾 → SQE 后继 /
后一 mission 入口）恒沿原指令顺序"向前"连接，切分不改变原序；
`_drop_cyclic_matches` 的环检测同时计入 extra 边（5.4 节）作兜底；仿真器
另有运行期死锁检测（任务 `end_time<0` 时报错退出）。

**统计与标记**：`CcuBuildStats` 携带并行化（`parallel_units`/`virtual_streams`/
`fork_edges`/`join_edges`——join_edges 现统计完成集汇合与 SQE 挂接边）、远端同步
（`local_posts`/`remote_posts`/`wait_matched`/`remote_waits`）等计数，
summary 行以 `pu/vs/lp/rp/wm/rw` 呈现；
`GraphBuilder.stream_meta` 按逻辑流名分类——`s{n}` → real、含 `#` →
virtual（附 parent_sid）、其余 → ccu，供三处消费：报告（`Stream 数：
N (含虚拟 M)`，虚拟流行标注 `vir_stream {n} (label)`，n 按分配顺序从 0
编号，超过 64 条流折叠明细）、trace（虚拟流 thread_name `vir_stream
{n} ({label})` + 任务 args `virtual_stream: true`）、dump（streams[] 附
kind/virtual/parent_sid、
extra_edges 列表；DOT 紫色虚线 cluster + 蓝色点线 extra 边，HTML 徽标
+ 点线弧）。stream_meta 缺省（simulator 独立 CLI 旧输入）时不标注，向后兼容。

**量化效果**（样例 db REDUCE_SCATTER iter0，66 任务）：队列并行化使 makespan
20,404.20 → 5,149.36 us（≈3.96x）；后续两项修正进一步降至 **2,753.71 us**：
① 远端 MS 规约修正（6.3.2 节，规约恒本地 HBM）——原第二相位 128MB 规约按
网络读计费 ~2526 us，修正后为 HBM ~96 us；② 跨 rank 同步节点补全（6.6 节）——
14 个依赖远端置位的 wait 此前被跳过（读单元提前至 t≈0 发射），修正后按固定
远端时延仿真。10 个异步单元改挂 10 条虚拟流（fork 边 10、挂接/汇合边按
SQE 邻居与 queue 串行情况生成）——首相位 8 单元并发（7 个走网读单元各
~2526 us、分属 7 个独立端口 `1/0`..`1/6` + 1 个本地 memcpy ~96 us），随后
规约 ~96 us 与收尾 memcpy ~96 us 串行衔接。注意单端口 read 峰值并发仍为
1——收益来自**跨端口重叠**，而非单端口并发提升。

### 7.5 带宽共享机制（`_BandwidthManager`）

#### 通用管理器（端口与 HBM 复用同一模型）

- **单管理进程模型**：新传输加入时 `interrupt` 管理进程，管理进程按当前并发数 N
  均分带宽：`bw_per = 总带宽 / N`。
- **利用率查询基准为 `min_remaining`**：同一组并发任务以"最小剩余数据量"查
  利用率表（并发 <8 用 profile_low，≥8 用 profile_high），组内同基准。
- 有效带宽 = `(总带宽 / N) × 利用率(min_remaining, N)`；计算最早完成时间，
  timeout 结算进度；有加入/完成时重算。
- **HBM 动态共享**：`Reduce_Inline` 与 `Memcpy` 共用一个 HBM 管理器
  （利用率恒 1.0）——多个 HBM 任务并发时均分 HBM 带宽（与旧版每任务独立
  `size / hbm_bandwidth` 不同，见 10.3 节）。

#### 双方向独立

每个 `NetworkPort` 含 write 和 read 两个独立管理器，带宽独立互不干扰：
配置端口带宽 B 时，write 方向可用 B，同时 read 方向也可用 B。

#### 多端口聚合

一个 eid 对应多个物理端口时，`port_key = tuple(port_ids)` 作为逻辑端口，
总带宽 = 各端口带宽之和（`get_ports_total_bandwidth`）。

### 7.6 参数三：`topo_resolver: TopoResolver`

- 输入 `ranktable.json`（`TopoResolver(ranktable_path=...)`），提供
  **eid（16B hex 字符串）→ 物理端口列表（`["die/port", ...]`）** 解析。
- 传输任务路由：write 方向查 `task.src_eid`，read 方向查 `task.dst_eid`。
- **eid 查不到端口时直接抛错**（无 jettyId 兜底、无默认端口）——因此 db 路径
  在"存在走网任务但无 ranktable"时整体跳过仿真（3.3 节），db 成图对无法解析
  eid 的任务计数告警（5.5 节）。
- 端口带宽按解析出的每个端口逐个查 `SimConfig.port_bandwidth` 并求和。

### 7.7 构建方式与 CLI

所有输入方式统一构建 `(config, dep_graph)`：

| 输入源 | 构建函数 | 返回 | 状态 |
|--------|---------|------|------|
| **hccl-vm db（本次新增）** | `graph_builder.build(group)` | `(config, dep_graph)` | 第 5、6 章，主输入 |
| JSON 任务文件 | `load_dep_graph(filepath)` + `load_config(...)` | `(config, dep_graph)` | 保留（hccl_sim 兼容） |
| task.txt / 算法生成等旧输入源 | — | — | **已随 HCCLSIM 版移除**（见 10.3 节） |

构建流程统一为：
1. 创建 `Task` 节点，调用 `dep_graph.add_task(task)`
2. 按 stream 内任务顺序调用 `dep_graph.add_stream_chain(stream_id, task_ids)`
3. 按 notify 配对结果调用 `dep_graph.add_match(wait_task_id, record_task_id)`

CLI 入口：

```bash
# db 输入 (light_performance 主入口, 详见 3.3 节)
python3 -m light_performance --db data/hccl_vm_data.db \
    --config sim_config.json --bw-profile bw_profile.json --ranktable ranktable.json -v

# 一键任务图 dump (仓库根目录)
python3 dump_graph.py data/hccl_vm_data.db --html

# hccl_sim 兼容入口 (JSON 任务文件)
python -m simulator --tasks tasks.json --config sim_config.json \
    --bw-profile bw_profile.json --ranktable ranktable.json -v --trace trace.json
```

`--config` 覆盖各模式的默认时延/带宽（单位 us/GBps）；`--bw-profile` 独立加载
利用率配置；报告与 trace 由 `report.py` 的 `generate_report` / `export_trace`
从 `run()` 结果 dict 生成。

trace 文件为 Chrome Trace Event 格式：`ts`/`dur` 值单位为 **us**（格式规范约定），
文件顶层携带 `"displayTimeUnit": "ns"` 指定查看器（chrome://tracing 等）的
Start/Wall Duration **显示**单位为 ns（缺省 ms）。

`export_trace` 的显示适配约定：

- **tid 顺序编号**：thread 行按 stream id 升序映射为 tid `0,1,2,...`（原始
  streamId 2/6..13 直接作 tid 时，按字符串排序的查看器会把 10..13 排到 2 之前）；
  真实 stream id 保留在 thread_name 标签与 `args.stream_id`。
- **端点取整**：`ts`/`end` 分别取整后作差得 `dur`——`ts` 与 `dur` 各自独立取整
  会产生 0.001us 伪重叠，同 tid 重叠切片会被查看器嵌套成子行（换行显示）。
- **虚拟流标注**（7.4.1 节）：db 路径经 `export_trace(..., stream_meta=...)`
  传入流元信息，虚拟流 thread_name 改为 `vir_stream {n} ({label})`（n 按
  sid 升序即分配顺序从 0 编号，摆脱 2^20 基址起的大数值流号；真实流号见
  行内任务 `args.stream_id`）、其任务 args 追加 `virtual_stream: true`；
  real/ccu 流名不变（ccu 队列链的逻辑名仅出现在文本报告行）。stream_meta
  缺省（simulator 独立 CLI 旧输入）时不标注，向后兼容。
- **notify 任务 args 携带 mask**：CCU 路径的 `Notify_Record`/`Notify_Wait`
  任务 args 追加 `mask: "0x…"`（notify_id + mask 唯一标识一个 notify 动作，
  7.3 节）；SQE 路径与 mask=0 不输出该字段。
- **算子综合信息首行（op_summary）**：pipeline 逐算子调用
  `export_trace(..., op_summary=...)` 时，每个算子（多算子合并导出时为每个
  pid 进程组）以 tid 1 + `thread_sort_index` 0 保留 `op_summary` 首行，任务行
  tid 与 sort_index 顺延（仅呈现层；tid 避开 0——部分查看器如 Perfetto 将
  tid 0 视作 idle 线程，`thread_sort_index` 为 Chrome Trace 标准行序控制），
  汇总事件置于事件数组最前，三重保证 op_summary 恒为第一行。首行放置一条
  横跨算子起止的 `X` 汇总切片（起始时间与持续时间即切片自身 ts/dur，取
  任务最早 start 到最晚 end），行标签   `"<op_name> | <algorithm_name>"`，
  args 携带：`algorithm_name`（算法选择结果，opDetails.algorithmName 日志
  截取回填（DFX 主路径，4.2 节注））、`data_type`/`data_count`（元素数，V2 布局算子为
  收发两侧元素数与类型，其余为 count 与单侧类型）、`task_count`（成图后
  任务总数，CCU 模式即微码指令翻译成 task 图后的数量）、以及 op_type/
  comm_id/op_iter/rank/rank_size/input_bytes/output_bytes（opMemInfo 的
  inputSize/outputSize，采集时由业务 API 参数计算）上下文；汇总事件不计入
  `max_events`。

### 7.8 对齐适配完成情况

旧版（本仓库改前）与 HCCLSIM 基准的差异已在本次对齐中消化：

| # | 项 | 处理 |
|---|----|------|
| 1 | 任务模型 6 类型 → 新 6 类型（本地/远端 record/wait 合并为 Notify_*，LOCAL_COPY 拆分 Memcpy/Reduce_Inline，网络传输改 Ub_*；后续随 db 新增 WRITE_WITH_NOTIFY 扩为 7 类型，10.3 节 #19） | `task_model.py` 按 HCCLSIM 替换 |
| 2 | stream_id str → **int**；notify_id str → **int**（INVALID=0xFFFFFFFF） | `dependency_graph.py`/成图侧同步 |
| 3 | 端口由 `port` int 字段指定 → **eid 实时解析**（write 查 src_eid / read 查 dst_eid） | `topo_resolver.py` + 成图侧填 eid |
| 4 | 单位制：JSON 时延 s → **us**、带宽 B/s → **GBps**、端口键 int → **"die/port"** | `config.py` from_dict 换算 |
| 5 | HBM 时延 `size/bw` 独立 → **动态共享**（Reduce_Inline+Memcpy 共用管理器） | `_BandwidthManager` 复用 |
| 6 | `run()` 返回 makespan → **结果 dict**；仿真器两参 → **三参**（+topo_resolver） | `simulator.py` |
| 7 | 时延公式分散 → `task_timing.py` 单点 | 已收编 |

遗留未匹配项（含 `src_rank==0` 硬编码、eid 无兜底、广播配对只触发首个 wait 等）
统一移入第 10 章跟踪。

---

## 8. 性能 profiling 输出（子需求 5，概要）

### 8.1 输出位置与文件

```
<installRoot>/plugin/light_performance/data/<时间戳>/
├── perf_profiling.json      # 主产物：性能指标 + 任务图摘要
├── trace_<commId>_<opIter>.json   # (可选) Chrome tracing, chrome://tracing 可视化
└── light_performance.log    # 运行日志
```

- 目录命名：`hccl-vm plugin run @estimate` 一次性执行通道不传递 data_id（3.2 节），
  以运行时间戳充当；独立调试模式（run_perf_sim.py）按 `-o` 指定输出路径。
- 输出固定在插件数据目录而非 `data/`：宿主每次 start 会清空 `data/`（`src/main.cc:32`），
  插件目录下的产物可跨会话保留；`archive.sh` 归档仅覆盖 logs/ 与 data/，插件产物
  不参与自动归档（10.3 节 #15）。

### 8.2 指标定义（草案）

| 层级 | 指标 | 来源 |
|------|------|------|
| 算子级 | opType/opIter/rankSize/dataSize、makespan、软件仿真耗时 | opDetails + `sim.run()` 结果 dict |
| 算子级 | 算子综合信息首行（algorithm_name/data_type/data_count，起止与时长由切片 ts/dur 承载，7.5 节 op_summary） | opDetails.algorithmName（DFX/日志双路径截取）+ 仿真任务时间轴 |
| 算子级 | 总传输量、有效带宽利用率（总传输量 / (makespan×端口带宽)） | 汇总 result["tasks"] |
| Stream 级 | 各 stream 起止时间、耗时、任务数 + 虚拟流标注（7.4.1 节） | result["tasks_by_stream"] + stream_meta |
| 图级 | 异步单元数、虚拟流数、fork 边数、完成集/挂接边数（7.4.1 节）；本地/远端 post 数（lp/rp）、本地 wait 配对数（wm）、远端 wait 数（rw） | CcuBuildStats（parallel_units / virtual_streams / fork_edges / join_edges / local_posts / remote_posts / wait_matched / remote_waits，summary 行 `lp/rp/wm/rw`） |
| 端口级 | 各 port 峰值并发、write/read 传输量与平均带宽 | result["ports"]（NetworkPort） |
| 任务级 | 每任务 start/end/duration、传输任务平均带宽 | result["tasks"]（现有 trace 逻辑） |

### 8.3 perf_profiling.json 格式草案

```json
{
  "meta": {"data_id": "1234", "db": "...", "sim_config": {...}, "generated_at": "..."},
  "ops": [
    {
      "comm_id": 1, "op_iter": 0, "op_type": 7, "op_name": "REDUCE_SCATTER",
      "rank_size": 4, "data_size": 33554432,
      "makespan_us": 15230.5,
      "total_bytes": 522190080,
      "streams": [{"stream_id": 2, "label": "s2", "kind": "real", "start_us": 0.0, "end_us": 15230.5, "task_count": 30}],
      "ports": [{"ports": ["1/0"], "write_bytes": 272629760, "read_bytes": 0,
                 "peak_concurrency": 3, "avg_bw_Gbps": 143.2}],
      "graph": {"task_count": 40, "stream_count": 5, "notify_match_count": 44,
                "cross_rank_notify": 40,
                "virtual_stream_count": 10, "extra_edge_count": 20,
                "dump_files": ["graph_comm1_iter0_dev1.dot"],
                "trace_file": "trace_1_0.json"}
    }
  ]
}
```

> `cross_rank_notify` 为保留的跨 rank notify 节点数量（诊断信息）：这些节点按 stream 顺序
> 正常成图、走固定远端时延，但不参与本地配对（无跨 rank record-wait 边）；该计数用于排查
> 数据/分类异常（如本地 notify 被误判为跨 rank）。`kind`/`virtual_stream_count`/
> `extra_edge_count` 为 CCU 队列并行化（7.4.1 节）新增字段：流分类 real/ccu/virtual、
> 虚拟流数量与 extra 边（fork/汇接/挂接）数量。

---

## 9. 附录

### 9.1 checker 成图模块 → light_performance 成图模块总对照

| 方面 | checker（C++） | light_performance（Python） |
|------|---------------|---------------------------|
| 目标 | 正确性校验 DAG（mainStart 可达、支持各类检查） | 仿真 DependencyGraph（三层：tasks / stream_chains / notify_matches） |
| 输入 | db（经 StorageManager/sim::Query） | 同一 db（sqlite3 直读 + struct 解码） |
| 成图范围 | **多 rank 联合成图**（全 rank 任务进同一 DAG） | **单 rank 成图**（跨 rank 不建配对边，远端同步任务保留） |
| SQE 翻译 | TaskMetaTranslatorV3（含 MemSlice 地址解析） | 5.2 节映射表（仅取 len/jettyId/notify 字段） |
| WRITE_WITH_NOTIFY 消费 | 解码边界拆分：`DecodeAndExpandTaskMeta` 展开为 [写, NOTIFY_RECORD] 两个子任务 | **不拆分**：单任务映射 `Write_With_Notify`（5.2 节），与 db 任务一一对应，trace/dump/统计粒度一致 |
| 依赖边 | stream 链 + 本地/跨 rank notify 边（队列模拟） | stream 链 + 仅本地 notify 先到先得（两遍扫描）+ extra 边（CCU fork/汇接/挂接，7.4.1 节） |
| CCU | 两阶段展开 + queue 并行化 + loop 合并 + CKE 记账 | 单阶段解释执行 + 组装期队列并行化（7.4.1 节）+ loop 合并（6.5 节）+ CKE 记账（第 6 章） |
| 校验后处理 | 内存冲突/语义/同步冲突/可达性/单任务检查 | 无 |
| 输出 | 校验结论 + dump | (SimConfig, DependencyGraph) → makespan → profiling json |

### 9.2 术语表

| 术语 | 说明 |
|------|------|
| SQE | Submit Queue Element，AICPU/SDMA 侧提交的任务单元，即 opTask 表的一行 |
| WQE | Work Queue Element（UB/DB 模式），本次不涉及 |
| notify | AICPU 侧事件同步原语，record 置位 / wait 等待，按 (recordDev, waitDev, notifyId) 配对 |
| CKE | CCU 侧事件同步寄存器（位图），post 置位 / wait 等待，按 (waitDev, dieId, ckeId, mask) 配对 |
| jetty | 网卡（通信引擎）端口编号，决定 memcpy 的带宽归属 |
| die | 芯片裸片，CCU 微码空间与 CKE 均按 (device, die) 组织 |
| mission | CCU 一次任务下发（instStartId/instCnt 界定指令窗口，args[13] 传参） |
| makespan | 任务图端到端耗时（最晚任务结束时间） |
| 主流/从流 | opDetails.streamId 标记的算子主流；其余为从流（首 WAIT 尾 RECORD 结构） |

---

## 10. HCCLSIM 对齐遗留项与待匹配清单

> 本章记录本次"db 成图路径 + 设计文档"对齐 HCCLSIM 基准过程中确认的**字段缺口、
> 任务类型不匹配与其他冲突点**。当前实现均已按"缺省填无效值 / 近似映射"落地，
> 此处留档供后续与采集侧、仿真模块联合匹配时逐项消化。

### 10.1 Task 字段可得性表（db 成图路径）

新模型 Task 字段在 db 成图路径的来源与缺省值（无效值约定：int 字段 `INVALID=0xFFFFFFFF`，
eid 空串，其余字符串 `""`）：

| Task 字段 | db 来源 | 可得性 | 缺省/近似值 | 后续匹配路径 |
|-----------|--------|--------|------------|-------------|
| `task_id` / `stream_id` | 全局序 / streamId+合成基址 | 可得 | — | — |
| `notify_id` | opTaskMeta `NotifyTask.notifyId` | 可得 | — | — |
| `size` | MEM_CPY `len` / REDUCE `dataCount` / CCU 长度寄存器 | 可得 | — | — |
| `jetty_id` | SQE 表列 / ccuChannels.jettyId 数组 | 可得 | INVALID（不走网任务） | — |
| `link_type` | opTaskMeta `HcclTaskProfInfo.linkType`（v3，枚举名映射 11.4.2 节） | 可得（v3） | ""（v2 旧版 BLOB 无 profInfo 时显示 `Unknown`） | —（11.4 节已消费） |
| `protocol` | ccuChannels.protocol（CommProtocol 枚举名映射） | 可得 | ""（无通道信息时，trace 显示 `N/A`） | —（自 `link_type` 迁入，11.1 节方案 a） |
| `src_eid`（本端，write） | SQE `lcEid`（v2 采集新增，4.4.1 节）优先；缺省 jettyId → 本设备出向通道 `leid` 推导 | 可得（v2）/近似（回退） | 空→任一本地通道 leid 兜底 | —（v2 已补充显式字段） |
| `dst_eid`（对端） | 通道 `reid` 优先，退回 SQE `rmEid` | 近似 | ""（无通道信息的任务） | 采集侧补充对端显式 eid |
| `src_rank` / `dst_rank` | taskData 端点 rankId（deviceId 字段承载，4.4.1 节 rank 字段回退）直接消费：write=src 本端/dst 对端，read=src 对端/dst 本端；CCU/未解析经 eid→rank 索引回退（4.8 节） | 可得/索引回退 | 显式值与索引均未解析时 INVALID + `rank_unresolved` 计数 | —（5.2/5.5/6.7 节） |
| `transport_type` | opTaskMeta `HcclTaskProfInfo.transportType`（v3，枚举名映射 11.4.3 节） | 可得（v3） | ""（v2 旧版 BLOB 无 profInfo 时显示 `Unknown`） | —（11.4 节已消费） |
| `data_type` | REDUCE 任务级 `dataType`（v2 新增，枚举名映射 int8..bfp16）；MEM_CPY 任务级无来源 | 部分可得（v2） | REDUCE 填枚举名；MEM_CPY "" | MEM_CPY 采集侧补充 |
| `port` | （保留字段） | 不填 | None | 仿真器经 eid 实时解析端口（7.6 节） |

**指令级缺口（CCU 微码展开）**：通道类走网任务的对端 rank 已由 eid→rank 索引解析
（4.8/6.7 节）；规约任务恒本地（MS 为 rank 本地 buffer，6.3.2 节），无对端 eid
需求。CCU 产出的 UB 任务均显式携带 `is_write`（6.7 节），方向判定不依赖 rank
填值。

### 10.2 任务类型不匹配表

| db taskType / 旧类型 | 新模型 TaskType | 不匹配处理 |
|---------------------|----------------|-----------|
| SYNC_STREAM (taskType=6) | **无对应类型** | 跳过不生成节点（host 侧流同步点，非设备侧任务，5.6 节）；统计 `skipped_sync` |
| AIV_GRAPH (taskType=5) | 无 | 跳过 + 告警（新旧模型一致，均不支持） |
| WRITE_WITH_NOTIFY (taskType=7，db 新增) | `WRITE_WITH_NOTIFY`（新增，第 7 种） | 单任务映射（**不拆分**为写 + NOTIFY_RECORD）：写恒 write 方向走端口带宽共享，附带 notify 按固定 record 时延（5.2 节，10.3 节 #19 已支持） |
| 旧 `REMOTE_RECORD` / `REMOTE_WAIT` | 已合并入 `NOTIFY_RECORD` / `NOTIFY_WAIT`（src≠dst 表达远端） | db 单 rank 路径的跨 rank notify 以 `src≠dst` 生成远端节点（5.2 节，保留节点不配对） |
| 旧 `MEMCPY`（网络传输） | 语义变更：新 `Memcpy` = HBM 拷贝 | 网络传输改由 `Ub_Write_Or_Read`（write/read 方向区分）表达；`Ub_Inline_Write` 仅保留 notify inline 写语义（固定时延） |
| 旧 `LOCAL_COPY` | 拆分为 `Memcpy`（纯拷贝）/ `Reduce_Inline`（规约） | REDUCE 不走网 → Reduce_Inline |
| 旧 `MemcpyDirection` 枚举（direction 字段） | 已删除 | 方向由任务类型 + rank/eid 填值表达（5.5 节） |
| 旧 `sync_stream_delay` 配置项 | 已随类型缺口取消 | 并入 local_record_delay 复用 |

### 10.3 其他冲突点（后续匹配）

1. ~~**`src_rank == 0` 判本端硬编码**（`simulator.py` TRANSFER 分支）：读方向任务若
   对端恰为 rank 0 且显式填写，会被误判为 write。~~ **已修复**：Task 新增显式
    `is_write` 字段（db 成图路径填写，task.txt 旧路径为 None 退回启发式），仿真器
    优先消费；taskData 的对端 rankId（deviceId 字段承载）同时直接填入 src/dst_rank（5.2/5.5 节）。
2. **TopoResolver 无兜底**：eid 在 ranktable 查不到直接抛错（无 jettyId→端口映射、
   无默认端口）。db 路径以"存在走网任务且无 ranktable → 跳过仿真仅 dump"规避
   （3.3 节）；后续可考虑接入 ccuChannels/JettyMaps 构建兜底映射。
3. **广播配对只触发首个 wait**（matched record 的 `break`）：多个 wait 配对同一
   record 时，后续 wait 的 Event 不被触发、end_time 保持 -1（任务静默丢失）。
4. **`is_matched_record` 线性扫描**：`dependency_graph.py` 对 matches 做 O(R×M)
   遍历，大规模 db（数万任务）有性能风险；建议维护 record→wait 反向索引（同时
   解决 #3）。
5. **SimConfig 单位制与端口键变更**：JSON 时延 s→**us**、带宽 B/s→**GBps**、
    端口键 int→**"die/port"** 字符串；旧配置文件不兼容（本次已同步样例
    `src/plugin/light_performance/sim_config.json`）。
6. **HBM 动态共享语义变化**：旧版每任务独立 `size / hbm_bandwidth`，新版
   Reduce_Inline+Memcpy 并发时均分 HBM 带宽（利用率恒 1.0 的管理器）；结果口径
   有差异。
7. **利用率查询基准 `min_remaining`**：同组并发任务以最小剩余数据量查利用率档位，
   与旧版"各任务按自身 data_size 查询"不同——组内小任务会拖低大任务的利用率档。
8. **旧输入源移除**：task.txt 解析、AllToAll 生成器、内置 demo 已随 HCCLSIM 版
   从 input_builder 移除；如需日志回放调试需另行恢复。
9. **TopoResolver 无 net_type 过滤**：同一 eid 多网络类型条目时无优先级规则。
10. **profiling json 未实现**（子需求 5）：当前输出为任务图 dump（DOT/JSON/HTML）
    + 文本报告 + Chrome trace；perf_profiling.json 格式见第 8 章草案。
11. ~~**local_rank 回退约定**：opDetails.rankId 缺失时按 `deviceId−1` 回退
    （沿用 task.txt 路径约定）；deviceId 与 rank 编号无必然关系，ranktable 可
    精确化。~~ 对端 rank 已由 eid→rank 索引（RankEidMap 主路径 + ranktable
    回退，4.8 节）解析，不再依赖编号约定；本端仍以 opDetails.rankId 为准
    （通道 leid 交叉校验，6.7 节）。
12. ~~**CCU queue 并行度未建模**：CCU 队列链为独立合成流并行执行，真实硬件的
    queue 并发资源约束未建模（checker 侧有 queue 并行化改造，本插件 6.7 节已简化）。~~
    **已实现队列并行化**（7.4.1 节：组装期骨架/单元切分 + fork/汇接/挂接 extra 边 +
    `#v` 虚拟流跨 mission 复用）。仍不建模：真实硬件 queue 并发资源上限——
    虚拟流数量不设限（checker V3 同样不设限），后续可按芯片代次封顶。
13. **@estimate 一次性通道不传递 data_id**：`plugin run @estimate` 特判分支直接执行
    entry（`cmd_base_utils.cc:626-663`），不经过通用消息路径的 `DumpData`（其生成的
    `data_id` 当前为固定值 "1234"），故输出目录以运行时间戳命名（8.1 节）；后续若
    切换常驻模型可对齐 data_id 语义。
14. **`plugin install @estimate` 语义退化**：install 走 `RegisterPlugin` 常驻拉起
    （fork+pipe），main.py 一次性执行完即在 stdin EOF 后退出，MonitorThread 回收——
    行为无害但与 install 的常驻语义不符；如需真正常驻须实现 stdin 命令循环
    （3.2 节演进方向）。
15. **插件产物不参与自动归档**：`archive.sh` 仅归档 `logs/` 与 `data/`，
    `plugin/light_performance/data/` 的 perf_profiling.json 等产物不进入 archive 目录
    （好处是不受 start 时 `remove_all("data")` 影响）；如需归档须扩展 archive.sh。
16. **simpy 部署依赖**：目标环境需预装 simpy（未随 hccl_vm_install 分发离线 wheel），
    main.py 启动时 import 失败应给出明确报错与安装提示（3.4 节）。
17. ~~**db_reader.py 未同步 v3 采集布局**：C 侧 `HcclTaskMetaData` 已扩至 188B（taskType
    后新增 `HcclTaskProfInfo` 3 字节，4.4.1 节），`db_reader.py` 仍按 185B
    （`TASK_META_LEN=185`、`<BQIQQI16s16s`）解码——长度闸门 `len(blob) >= 185` 拦不住
    188B 的 BLOB，读取 v3 采集的 db 会**字段整体错位**（commId 起）。需同步为
    188B/`<BBBBQIQQI16s16s` 并解码 profInfo 三字段（值域/名称见 11.4 节）。~~
    **已同步**：`db_reader.py` 按 BLOB 长度自动识别 v3（188B/`<BBBBQIQQI16s16s`，
    profInfo 三字段入 `HcclTaskMetaData`）与旧 v2（185B）布局，见 4.4.1 节实现状态。
18. ~~**JSON 任务类型未消费 profInfo**（余项缩小）：trace args 的
    `transport_type`/`link_type` 已切换 profInfo 驱动、CommProtocol 迁移至新字段
    `protocol`（11.4.2/11.4.3 节，11.1 节方案 a 落地）；**余项**为 `task_type`
    字段——当前仍由 TaskType 枚举 + jettyId/方向条件推导（11.2 节），与采集侧
    profInfo 的 hcomm op 名存在**语义错位**（如 `Ub_Inline_Write` 双向指代不同
    任务，11.4.4 节）；切换为 `profInfo.profTaskType` 驱动后可与 msprof timeline
    op 名一一对应，目标态对照（含各值与仿真 TaskType 的对应）见 11.4.1 节。~~
    **已落地（全部余项收尾）**：`Task` 新增 `task_name` 显示名（`task_model.py`，
    `display_name` = `task_name or task_type.value`），`graph_builder._add_task`
    统一注入 `prof_task_type_name(profInfo.profTaskType)`；trace 事件名/`cat`
    （`report.py export_trace`）、`args.task_type`（`to_trace_args`）、verbose
    明细与 dump（DOT/JSON/HTML 图例）全部切换 op 名。内部 `TaskType` 退役为
    时延模型分类（11.4.5 节 #3），时延/配对/计费零改动。v2 旧版 BLOB
    （profInfo=0xFF）显示 `Unknown`（与 link/transport 同口径）；CCU 微码子任务
    无 profInfo，回退内部名（11.3 节）。错位消除对照见 11.4.4 节实施记录。
19. ~~**WRITE_WITH_NOTIFY (taskType=7) 解码缺失**：`db_reader.py` 的
    `HccLTaskMetaType` IntEnum 仅 0~6，`decode_op_task_meta` 遇 7 直接 raise 丢弃
    （计入 warning）。~~ **已支持（单任务映射）**：`db_reader.py` 解码
    `WriteWithNotifyTask`（4.4.1 节）并单任务透传，`graph_builder` 映射为
    `WRITE_WITH_NOTIFY` TaskType（trace 显示名 `Write_With_Notify`，5.2 节），
    与 db 任务**一一对应**；时延 = 端口 write 方向带宽共享 + 写落地后 notify
    固定 record 时延（7.3/7.4 节）。**不采用拆分策略**（checker/runner 侧的
    "拆为 [写, NOTIFY_RECORD] 两子任务"、以及本插件早期的解码边界拆分均已
    废弃）：拆分会拆散任务粒度，使 trace/dump/统计与 db 任务无法一一对应。
    采集侧 profInfo=TASK_WRITE_(REDUCE_)WITH_NOTIFY（op 名 `Write_With_Notify` /
    `Reduce_With_Notify`，hcomm 源码后者为错拼 `Reuce_With_Notify`，见 11.4.5），
    两种 profInfo 值均映射为同一 `Write_With_Notify` 任务（isReduce 仅影响
    data_type 显示）。
 20. ~~**同一 mission 多次 dispatch 的 SQE 归并为单一上下文**：CCU 展开把同
     `(dieId, missionId, queue)` 的全部 SQE 并入一个派发上下文（单个指令窗口
     [首 instStartId, 末尾)，单次 SQE args 消费）——不满足 checker
     `IsContiguousCcuSqe` 的"连续才并入"约束（`task_meta_translator_v3.cc:213-231`）。
     典型受害场景：指令窗口回绕重发（头 SQE 13 条 Load SqeArgs + 体 SQE 连发两遍，
     第二遍用新 args），旧实现下第二遍 dispatch：① 窗口起点按头 SQE 计，末段
     JMP 越过整段数据搬运指令；② SQE args 已被首遍消费殆尽，后续 Load 取 0 值
     （GSA 地址全 0）；③ 循环计数/CKE 残留首遍状态——第二次 dispatch 的
     Ub/Memcpy/本地 post 整体丢失，网络数据量仅为 checker 的 3/10。~~
     **已修复（连续段切分）**：按 `instStartId`/`instCnt` 连续性切分为多个派发
     上下文（6.1 节），各上下文独立窗口重放（含 Load SqeArgs 头部）+ 独立顺序
     消费 args（`GetSqe` 语义）；寄存器/CKE 位图跨上下文共享（`AllRankParamRecorder`
     语义），同 queue 上下文按派发序串行，多段任务 id 以 `_s{序号}` 后缀消歧
     （`ccu_graph_builder.expand_ccu_graph`）。修复后与 checker 单遍图逐流对齐：
     record 本地/远端 54/54、wait 本地/远端 27/36、Ub 39/39、Memcpy 6/6、
     网络+本地数据量 1088MB（2333 验证环境 ALLGATHER dev1）。
     同根因的**wait 节点发射偏差**一并修复：旧实现按"每贡献 post 一个 wait 节点"
     拆分（一条 wait 指令可产出 N 个节点，与 checker `GenWaitNode` 的"每指令至多
     1 本地 + 1 远端节点"不符，本环境表现为本地 wait 48 vs 27）；现聚合为单节点
      （mask = 全部贡献 post 的位或，首个贡献 post 走 notify 配对、其余经 extra 边，
      6.4 节），修复后本地 wait 27/27 对齐。
 21. **合并 loop 折叠伪串行的残留缺口**：6.5.1 节特征链消解覆盖合并路径的
      mid-body wait/record（折叠伪串行屏障），但两处残留：① 全展开回退路径
      （`mf > 0`）的逐迭代 wait/record 为真实 4KB 粒度握手，其迭代间重叠同样
      被队列链串行化（节点数大、当前样例未触发，暂不处理）；② 逐迭代握手的
      **速率**开销未建模——折叠与消解均按单次计（CKE 引擎吞吐若为瓶颈需专门
      速率模型，如 ns/次 × 32768 次量级评估）。

---

## 11. db 任务类型与 JSON 产物任务类型映射总表

> 本章为类型映射速查表，回答三个问题：
> ① db 数据库中的任务（`opTask_P_<pid>` 表 taskType 列）最终转换为 JSON 产物中的
> 哪些任务类型（11.1、11.2 节）；② CCU 模式下微码指令与 JSON 任务类型的对应
> 关系（11.3 节）；③ 采集侧 profiling 对账信息（`HcclTaskProfInfo`）与 JSON
> 显示类型的对应关系及目标态（11.4 节）。实现依据：`src/plugin/light_performance/
> graph_builder.py`（SQE 映射，`_map_tasks`/`_map_transfer`）、
> `src/plugin/light_performance/ccu_graph_builder.py`（CCU 指令映射，`transform`
> 系列）、`src/plugin/light_performance/simulator/task_model.py`（TaskType 枚举）。
> 与 5.2 节（SQE 映射 + 时延模型）、6.3 节（指令大类映射）同源，此处按
> "db → JSON"的对应关系重新组织，并补充逆向索引。
>
> **profInfo 背景**：v3 采集起，db 任务的 opTaskMeta BLOB 携带 `HcclTaskProfInfo`
> 子结构（4.4.1 节，单源定义于 `include/sim_profiling_defs.h`，checker/runner 作为
> 不透明整体透传），三个字段 `profTaskType`/`linkType`/`transportType` 的值域与
> 名称逐一对齐 hcomm `TaskParamType`/`DfxLinkType`/`DfxTransportType`——即
> msprof timeline op 名与 L1 明细 linkType/transportType 字段。本章 11.1~11.3 描述
> **当前 TaskType 实现**（尚未消费 profInfo），11.4 给出三字段完整对照与切换
> 目标态（待办见 10.3 节 #17~#18）。

### 11.1 JSON 产物中的任务类型载体

当前实现的 JSON 产物共三类，其中的任务类型均取自 `TaskType` 枚举值
（字符串形式，`task_model.py:20-26`）：

| JSON 产物 | 格式 | 任务类型所在字段 | 生成模块 |
|-----------|------|----------------|---------|
| `perf_profiling.json`（主产物，8.1 节） | Chrome Trace Event | `traceEvents[].name` / `.cat` / `.args.task_type` | `simulator/report.py`（`export_trace`，事件名即任务显示名：profInfo op 名，11.4 节） |
| `trace_comm<commId>_iter<opIter>.json`（可选单算子 trace） | Chrome Trace Event | 同上 | 同上 |
| 任务图 dump `graph_comm<commId>_iter<opIter>_dev<dev>.json`（可选，`--dump-dir`） | `light_perf.graph_dump/2` | `nodes[].type` | `graph_dumper.py` |

> dump JSON 的 notify 节点在 mask 非零时携带 `nodes[].mask`（7.3 节 Task.mask），
> HTML 视图 tooltip 同步展示；`perf_profiling.json`/trace 的任务 args 亦含
> `mask`（7.7 节）。

JSON 中出现的任务类型全集（7 种）：

| TaskType 值（JSON 中呈现） | 时延模型 | 语义 |
|--------------------------|---------|------|
| `Notify_Record` | 固定时延（本地另触发 Event） | notify 置位（record / CKE post） |
| `Notify_Wait` | 等 Event / 固定时延 | notify 等待（wait / CKE wait） |
| `Reduce_Inline` | HBM 动态共享 | 本地规约（计算近似为搬运，5.7 节） |
| `Memcpy` | HBM 动态共享 | 本地拷贝（HBM，非网络） |
| `Ub_Inline_Write` | 固定时延（同 Notify_Record） | notify inline 写（仅旧输入路径） |
| `Ub_Write_Or_Read` | 端口带宽共享（write/read 方向独立计费） | 网络数据收发（SQE 走网任务两方向统一映射） |
| `Write_With_Notify` | 端口 write 方向带宽共享 + 写后 notify 固定时延 | 网络写 + 附带 notify（与 db `WRITE_WITH_NOTIFY` 一一对应，5.2 节） |

> 8.3 节的 `perf_profiling.json` 结构化格式（meta/ops 数组）为草案，当前主产物
> 实为 Chrome tracing 格式（10.3 节 #10）；无论哪种格式，任务类型均为上述
> 7 种 TaskType 值，本章映射表对其统一适用。

**trace args 中的类型相关字段（现状与 profInfo 消费）**：Task 的 `to_trace_args`
（`task_model.py`）输出四个类型相关字段；四字段均已切换 profInfo 驱动
（10.3 节 #18 全部落地，11.4 节）：

| trace args 字段 | 当前来源 | 状态 |
|----------------|---------|--------------------------|
| `task_type` | `profInfo.profTaskType` 的 op 名（`prof_task_type_name`，`graph_builder._add_task` 统一填充为 `Task.task_name`；`display_name` 空时回退内部 TaskType 名） | **已落地**（与 msprof timeline op 名一一对应，11.4.1 节；v2 旧版 BLOB 为 `Unknown`） |
| `transport_type` | `profInfo.transportType` 名（`ProfTransportTypeName`，11.4.3 节，`graph_builder._add_task` 统一填充） | **已落地** |
| `link_type` | `profInfo.linkType` 名（`ProfLinkTypeName`，11.4.2 节，`graph_builder._add_task` 统一填充）——**注意与 `protocol`（CommProtocol）是两个不同概念** | **已落地**（方案 a：覆盖为 DfxLinkType 名） |
| `protocol`（新增字段） | 通道 `CommProtocol` 枚举名（HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO，经 `graph_builder._resolve_eids` / CCU `_channel_eids` 填充） | **已落地**（CommProtocol 自 `link_type` 迁入，无通道信息时 `N/A`） |

> `link_type` 的覆盖（方案 a）已实施：语义与 msprof L1 明细字段名对齐，避免同名异义；
> 现有 CommProtocol 信息不丢失，移入 `protocol` 字段。CCU 微码子任务为 simulator
> 语义产物、无 profInfo（11.3 节），其 `link_type`/`transport_type` 为 `N/A`，
> `protocol` 保留通道 CommProtocol 名。

### 11.2 SQE 路径：db taskType → JSON 任务类型映射总表

正向映射（db 任务 → JSON 任务显示名）。"JSON 任务类型"列为 trace/dump 显示的
op 名（已切换 `profInfo.profTaskType` 驱动，10.3 节 #18 落地）；"内部 TaskType"
（时延模型分类，决定计费/配对）保持 11.2 原判定条件（jettyId/方向/recordDev），
与 op 名的对应见 11.4.1 节"仿真 TaskType"列：

| db taskType（opTask.taskType 列） | 判定条件 | JSON 任务类型（op 名） | 采集侧 profInfo.profTaskType (op 名) | 说明 |
|----------------------------------|---------|--------------|--------------------------------------|------|
| `NOTIFY_WAIT` (0) | recordDev == 本设备 | `Notify_Wait` | `TASK_NOTIFY_WAIT` → `Notify_Wait` ✓ | 本地 wait：参与先到先得配对（5.4 节），等待 Event |
| `NOTIFY_WAIT` (0) | recordDev != 本设备 | `Notify_Wait` | `TASK_NOTIFY_WAIT` → `Notify_Wait` ✓ | 跨 rank wait：节点保留不配对，src≠dst → 固定 `remote_wait_delay` |
| `NOTIFY_RECORD` (1) | waitDev == 本设备（SDMA/本地 notify） | `Notify_Record` | `TASK_NOTIFY_RECORD` → `Notify_Record` ✓ | 本地 record：参与配对，触发 Event |
| `NOTIFY_RECORD` (1) | waitDev != 本设备（SDMA/本地 notify） | `Notify_Record` | `TASK_NOTIFY_RECORD` → `Notify_Record` ✓ | 跨 rank post：节点保留不配对，src≠dst → 固定 `remote_record_delay` |
| `NOTIFY_RECORD` (1) | UBDMA WQE `inlineEn==1`（notify inline 写） | `Ub_Inline_Write` | `TASK_UB_INLINE_WRITE` → `Ub_Inline_Write` ✓ | notify inline 写（名字已归还，11.4.4 节）；内部 TaskType 仍为 Notify_Record（固定 record 时延） |
| `REDUCE` (2) | jettyId == INVALID | `Reduce_Inline` | `TASK_REDUCE_INLINE` → `Reduce_Inline` ✓ | 本地规约：HBM 动态共享计费 |
| `REDUCE` (2) | jettyId 有效 + write 方向 | `Ub_Reduce_Inline` | `TASK_UB_REDUCE_INLINE` → `Ub_Reduce_Inline` ✓ | 走网规约近似（5.7 节），内部 TaskType=Ub_Inline_Write，端口 write 方向计费 |
| `REDUCE` (2) | jettyId 有效 + read 方向 | `Ub_Reduce_Inline` | `TASK_UB_REDUCE_INLINE` → `Ub_Reduce_Inline` ✓ | 同上，内部 TaskType=Ub_Write_Or_Read，read 方向计费 |
| `MEM_CPY` (3) | jettyId == INVALID（同设备） | `Memcpy` | `TASK_SDMA` → `Memcpy` ✓ | 本地拷贝：HBM 动态共享计费；profInfo.linkType=ONCHIP |
| `MEM_CPY` (3) | jettyId == INVALID（跨设备 SDMA SQE） | `Memcpy` | `TASK_SDMA` → `Memcpy` ✓ | 跨片 SDMA 搬运；同为 Memcpy，差异由 profInfo.linkType=PCIE / transportType=SDMA 表达（11.4 节） |
| `MEM_CPY` (3) | jettyId 有效 + write 方向 | `Ub_Write_Or_Read` | `TASK_UB` → `Ub_Write_Or_Read` ✓ | 网络发送（方向不编码进类型名，由 direction/is_write 表达）；内部 TaskType=Ub_Inline_Write |
| `MEM_CPY` (3) | jettyId 有效 + read 方向 | `Ub_Write_Or_Read` | `TASK_UB` → `Ub_Write_Or_Read` ✓ | 网络接收；内部 TaskType=Ub_Write_Or_Read |
| `WRITE_WITH_NOTIFY` (7) | 恒 write 方向（无方向推断，5.5 节） | `Write_With_Notify`（单任务，不拆分）；`udfFlag==1` 时 `Reduce_With_Notify` | `TASK_WRITE_WITH_NOTIFY` → `Write_With_Notify`；`udfFlag==1` 时 `TASK_WRITE_REDUCE_WITH_NOTIFY` → `Reduce_With_Notify`（hcomm 源码为错拼 `Reuce_With_Notify`，11.4.5 节） | 写+notify 组合任务（`WriteWithNotifyTask`，4.4.1 节）：端口 write 方向带宽共享 + 写后 notify 固定 record 时延；两种 profInfo 值均映射为同一内部 WRITE_WITH_NOTIFY（时延模型一致，显示名与 isReduce 区分） |
| `CCU_GRAPH` (4) | — | （占位，不直接产出） | `TASK_CCU` → `Ccu` | 展开为 CCU 子图，最终产物见 11.3 节 |
| `AIV_GRAPH` (5) | — | 无（跳过） | `TASK_AIV` → `AivKernel` | 告警 + `skipped_aiv` 计数（暂不支持） |
| `SYNC_STREAM` (6) | — | 无（跳过） | `UNKNOWN` → `Unknown` | host 侧流同步点，非设备侧任务（5.6 节），hcomm profiling 无对应类型 |

> 跨 rank 判定（waitDev/recordDev）见 4.6 节；走网方向判定（write/read）见 5.5 节。

逆向索引（JSON 任务显示名 ← 全部来源，含 CCU 路径与 profInfo 维度；SQE 路径
显示名 = profInfo op 名，CCU 路径显示名 = 内部 TaskType 名，11.3 节）：

| JSON 任务显示名 | SQE 来源（db taskType） | 采集侧 profInfo 来源（op 名） | CCU 来源（微码指令，11.3 节） |
|--------------|------------------------|------------------------------|------------------------------|
| `Notify_Record` | `NOTIFY_RECORD` (1)，本地/跨 rank（SDMA/本地 notify） | `Notify_Record`（TASK_NOTIFY_RECORD ✓） | SetCKE/SetCKBit 本端 post；TRANS/REDUCE/LOAD 类指令的内嵌 `setCKE` 字段（A6 LOAD 类亦携带）；跨 rank 置位/通知（A5 Sync 系 setRmtCKE；A6 TransMem notify/0x1A 原子、SyncWtX notify、SyncAtx——远端近似）；子图挂接 launch-record / completion-record |
| `Notify_Wait` | `NOTIFY_WAIT` (0)，本地/跨 rank | `Notify_Wait` ✓ | 各类指令的内嵌 `waitCKE` 字段；ClearCKE；A6 独立 Wait 指令（Xn 值比较，未满足 → 远端 wait）；子图挂接 completion-wait / launch-wait |
| `Reduce_Inline` | `REDUCE` (2)，jettyId 无效 | `Reduce_Inline` ✓ | REDUCE 类指令（msId 列表全本地） |
| `Memcpy` | `MEM_CPY` (3)，jettyId 无效（含跨片 SDMA） | `Memcpy` ✓ | TRANS 本地搬运指令（A5 4 条码：0x0/0x2/0x5/0xA；A6 4 条码：0x0/0x2/0x5/0x6） |
| `Ub_Inline_Write` | `NOTIFY_RECORD` (1) + `inlineEn==1`（notify inline 写） | `Ub_Inline_Write`（TASK_UB_INLINE_WRITE ✓，名字已归还） | —（CCU 走网搬运统一映射 `Ub_Write_Or_Read`，11.3.1） |
| `Ub_Write_Or_Read` | `MEM_CPY` (3) / `REDUCE` (2)，jettyId 有效（write/read 方向） | `Ub_Write_Or_Read`（TASK_UB ✓，方向由 direction/is_write 表达） | TRANS 走网搬运指令（A5 write 角色 3 条码：0x3/0x7/0x9、read 角色 4 条码：0x1/0x4/0x6/0x8；A6 统一 TransMem 0x10：dmaOpCode 3/5=write、6=read，含 udfEnable 走网规约近似）；REDUCE 类含远端 MS |
| `Ub_Reduce_Inline` | `REDUCE` (2)，jettyId 有效（走网规约） | `Ub_Reduce_Inline`（TASK_UB_REDUCE_INLINE ✓） | — |
| `Write_With_Notify` | `WRITE_WITH_NOTIFY` (7)，isReduce=0 | `Write_With_Notify`（TASK_WRITE_WITH_NOTIFY ✓） | — |
| `Reduce_With_Notify` | `WRITE_WITH_NOTIFY` (7)，udfFlag==1 | `Reduce_With_Notify`（TASK_WRITE_REDUCE_WITH_NOTIFY ✓） | — |

### 11.3 CCU 路径：微码指令 → JSON 任务类型映射

db 中 `CCU_GRAPH` (4) 的 SQE 本身只是占位（`CcuPlaceholder`，6.1 节），最终
JSON 任务由 `ccuInstrRes` 指令空间中的微码解释产生（第 6 章）。**A5（V1，
Ascend950，devType=6）与 A6（V2，Ascend960，devType=8）两代指令集均已实现**
（按 `opDetails.devType` 选择，其余 devType 回退 A5）：A5 总表见 11.3.1 节，
A6 总表见 11.3.3 节。

> profInfo 对应：CCU 占位 SQE 的 `HcclTaskProfInfo` 整体为
> `TASK_CCU`→`Ccu`（linkType/transportType 为 UNKNOWN，微码无链路维度）。展开后的
> 微码子任务（`Memcpy`/`Ub_Write_Or_Read`/`Notify_*` 等）是
> simulator 的语义产物，**无对应 ProfTaskParamType**，显示名回退内部 TaskType 名
> （10.3 节 #18 实施时的既定取舍，11.4.4 节）；msprof 侧 CCU 任务经
> 微码级明细（`CcuProfilingInfo`：CCU_TASK/WAITCKE/LOOPGROUP 三类，随 TASK_CCU
> 上报）表达同等粒度信息，对账时以整 SQE 的 `Ccu` 为对齐锚点。

#### 11.3.1 A5 指令总表（指令 → JSON 任务类型）

微码指令 = 2B 头部（type:4bit 大类 + code:11bit 指令码，6.2 节）+ 30B 载荷：

| 指令大类 (type) | 指令 (code) | JSON 任务类型 | 说明 |
|----------------|------------|--------------|------|
| LOAD (0x0) | LoadSqeArgsToGSA (0x0)、LoadSqeArgsToXn (0x1)、LoadImdToGSA (0x2)、LoadImdToXn (0x3)、LoadGsaXn (0x4)、LoadGsaGsa (0x5)、LoadXx (0x6) | **无**（不生成节点） | 仅读写寄存器模型（xn/gsa），供后续指令取长度、循环计数等 |
| CTRL (0x1) | SetCKE (0x2) | `Notify_Record` | 本端 CKE post（ckeId 作 notify_id、ckeMask 作 mask，src=dst=本端）；内嵌 `waitCKE` 字段可叠加产出 `Notify_Wait` |
| CTRL (0x1) | ClearCKE (0x4) | `Notify_Wait` | wait 语义（内嵌 `waitCKE` 字段）；清位动作本身不产节点 |
| CTRL (0x1) | Loop (0x0) | **无** | 须由 LoopGroup 触发；单独出现属异常，告警跳过 |
| CTRL (0x1) | LoopGroup (0x1) | （无直接节点） | 循环控制：合并/展开循环体（6.5 节），体内指令按本表映射；合并路径下数据任务 size × 总迭代数，且发射的任务携带 loop_cnt（展开后 loop 数）/ loop_iter_cnt（单 loop 自循环次数），总迭代数 = 两字段之积 |
| CTRL (0x1) | Jmp (0x5) | **无** | 条件跳转：`Xn[conditionXnId] != expectData` 时 pc 跳至 `Xn[dstInstrXnId]` 所指指令（checker `TransformJmpInstrA5` 语义，目标限当前指令窗口内，越界告警直通）；循环体内直通（合并前置校验已拒绝含 JMP 的循环体）；带步数预算防御后向跳转死循环 |
| TRANS (0x2) | LocMemToLocMs (0x0)、LocMsToLocMem (0x2)、LocMsToLocMs (0x5)、LocMemToLocMem (0xA) | `Memcpy`（src/dst_buf_type 标注 mem/ms） | 本地搬运（LOCAL_COPY 角色，6.3.2 节），HBM 动态共享 |
| TRANS (0x2) | LocMsToRmtMem (0x3)、LocMemToRmtMem (0x9)、LocMsToRmtMs (0x7)、RmtMemToLocMs (0x1)、RmtMsToLocMem (0x4)、RmtMsToLocMs (0x6)、RmtMemToLocMem (0x8) | `Ub_Write_Or_Read`（is_write 区分方向, src/dst_buf_type 标注 mem/ms） | 跨 die/卡搬运（write/read 角色统一类型，6.3.2 节）；对端 rank 经通道 reid → eid→rank 索引解析（4.8/6.7 节） |
| TRANS (0x2) | SyncCKE (0xB)、SyncGSA (0xC)、SyncXN (0xD) | 本端 post 部分 → `Notify_Record`；远端 post 部分 → `Notify_Record`（远端，不配对） | 远端置位字段（SyncCKE 的 `rmtCKEId/locCKEMask`、SyncGSA/SyncXN 的 `setRmtCKEId/setRmtCKEMask`）生成远端 record 节点（src=本端、dst=INVALID → 固定 `remote_record_delay`，不登记本地记账），计入 `remote_posts`（6.6 节）；SyncGSA/SyncXN 携带第二个远端 post 时同样生成；内嵌 `waitCKE` 可产出 `Notify_Wait` |
| REDUCE (0x3) | Add (0x0)、Max (0x1)、Min (0x2) | `Reduce_Inline` | 规约计算近似为搬运（5.7 节）；MS 恒为 rank 本地 buffer（msId 高位仅寻址编码，6.3.2 节），本地 HBM 计费 |
| （子图挂接） | —（成图阶段合成，非微码） | 无（不产生任务节点） | 挂接依赖全部经 extra 边表达（7.4.1 节）：入口边 SQE 前驱→骨架头/骨架前单元头、出口边 mission 完成集→SQE 后继；任务图中不凭空产生任何节点 |

#### 11.3.2 内嵌同步字段（跨指令通用）

TRANS / REDUCE / SetCKE / Sync 类指令的载荷普遍携带同步字段，与主语义
**叠加产出** JSON 任务：

| 载荷字段 | JSON 任务类型 | 行为 |
|---------|--------------|------|
| `waitCKEId` / `waitCKEMask` | `Notify_Wait` | 位已满足时与本端 post 记账配对（聚合为单节点：mask=全部贡献 post 消费位之或，首个 post add_match、其余 extra 边，10.3 节 #20）；位不足：顶层指令阻塞本 queue（停滞检测发现后生成远端 wait 节点并重放指令其余部分，`remote_waits`）；循环体内无法阻塞，直接生成远端 wait 节点继续 |
| `setCKEId` / `setCKEMask` | `Notify_Record` | 本端 CKE post，登记 seen_post 记账（6.4 节） |
| `clearType` 位 / SetCKE 的清位标志 | 无（仅寄存器清位） | wait 完成后清位，不产生节点 |

> 因此一条 TRANS 指令可产出 1 个数据搬运任务（`Memcpy`/`Ub_Write_Or_Read`），
> 并按内嵌字段叠加产出若干 `Notify_Wait`（按位配对，
> 0..n 个）与 1 个 `Notify_Record`。合并循环（6.5 节）下每个槽位同样保留
> wait/set 任务，仅数据任务折叠为 size × 总迭代数。
>
> **A6 差异**：内嵌 `waitCKE` 仅存在于 SetCKBit/ClearCKE（CTRL 类）；本地
> TRANS/REDUCE 与 TransMem/Sync 类只携带尾部 `setCKEId/setCKEMask`（@26/28，
> 叠加产出 `Notify_Record`）；等待语义由独立 Wait 指令（Xn 值比较）承担。

#### 11.3.3 A6（Ascend960）指令总表（指令 → JSON 任务类型）

devType=960（`DEV_TYPE_960`）对应 V2/A6 指令集，**已实现**（移植自 checker
`ccu_task_transform_instruct_a6_v3.cc`，解释器 `_MissionContextA6`）。微码指令
= 2B 头部（与 A5 同构）+ 30B V2 载荷（`ccu_microcode_v1.h` CcuV2 结构体，
pack(1)）；寄存器号普遍经 `GetXnId`（bit15 寻址模式，6.4 节）。

| 指令大类 (type) | 指令 (code) | JSON 任务类型 | 说明 |
|----------------|------------|--------------|------|
| LOAD (0x0) | LoadSqeArgsToX (0x1)、LoadImdToX (0x2)、LoadX (0x6)、StoreX (0x7)、ClearX (0x8)、Nop (0x9)、算逻 Add (0xD)/Sub (0xE)/Mul (0xF)/And (0x10)/Or (0x11)/Not (0x12)/Xor (0x13)/Shl (0x14)/Shr (0x15)/Popcnt (0x16) | 数据面**无**；`setCKEMask != 0` 时叠加 `Notify_Record` | 统一 Xn 寄存器读写（无 GSA）：SQE args 顺序消费（6.1 节）、立即数装载、间接读写（LoadX/StoreX，oMode 0=立即数偏移/1=寄存器偏移）、区间清零（ClearX）、64 位回绕算逻（Mul 取低 32 位乘积、移位量 0x3F 截断、Shl/Shr 支持逻辑/算术/循环三型）；除 Nop 外均携带尾部 setCKE 字段（checker `ProcessSetMask`，mask 非零 → 本端 post 节点） |
| LOAD (0x0) | Load (0xA)、Store (0xB) | 数据面**无**；setCKE 叠加 `Notify_Record` | HBM ↔ Xn 数据面：HBM 内容不建模（checker 经 AllRankParamRecorder 读写数据），寄存器不回填，告警留痕；仅保留同步语义 |
| CTRL (0x1) | SetCKBit (0x2) | `Notify_Record`（+ 内嵌 `waitCKE` 叠加 `Notify_Wait`） | wait → 本端 post → clearType==1 时清 wait 位（checker 仅支持 clearType=1，其余告警跳过清位） |
| CTRL (0x1) | ClearCKE (0x4) | `Notify_Wait`（内嵌 `waitCKE`） | wait 后按 checker `ClearWaitMask` 语义清除消费的 wait 位；`clearMask != 0` 为 checker 不支持场景（告警 + 宽容清除目标位） |
| CTRL (0x1) | Loop (0x0) | **无** | 须由 LoopGroup 触发；单独出现告警跳过。`mode==1` 条件 Loop：`wishCKEBit == Xp.loopCtxId + 1024` 命中才展开（6.5 节） |
| CTRL (0x1) | LoopGroup (0x1) | （无直接节点） | 循环控制：参数取自 Xn/Xm/Xp 三寄存器（loopInsCnt:10/expandOffset:9/expandCnt:9 等，6.5 节）；合并/展开同 A5（合并校验含 A6 附加规则：wait CKE ID 不复用、跨展开迭代数/长度/通道一致）；配置非法（loopInsCnt>512 等）告警跳过 |
| CTRL (0x1) | Jmp (0x5) | **无** | 条件跳转：conditionType 0~5（==/!=/>/>=/</<=）、≥6 无条件；jumpMode 0=相对 pc（模 0x10000）/1=绝对指令号，目标取 `Xn[relTarInstrXnId]`；限当前指令窗口，越界告警直通；循环体内直通；步数预算防御死循环 |
| CTRL (0x1) | Wait (0x7) | 未满足时远端 `Notify_Wait` | Xn 值比较自旋（conditionType 同 Jmp，缺省按等于）：满足直通无节点；未满足 = 依赖对端远端寄存器写 → 顶层停滞消解/循环体内直接生成远端 wait 节点（固定 `remote_wait_delay`，任务 id 标签 `xwt`，计入 `remote_waits`） |
| CTRL (0x1) | Fence (0x8) | **无** | checker 不支持（报错中止）；此处告警直通（队列串行模型下顺序天然保证） |
| TRANS (0x2) | TransLocMemToLocMS (0x0)、TransLocMSToLocMem (0x2)、TransLocMSToLocMS (0x5)、TransLocMemToLocMem (0x6) | `Memcpy`（src/dst_buf_type 标注 mem/ms） | 本地搬运（长度 `Xn[GetXnId(xlId)]`）；无内嵌 waitCKE，尾部 setCKE 叠加 `Notify_Record` |
| TRANS (0x2) | TransMem (0x10) | `Ub_Write_Or_Read`（dmaOpCode 3/5=write、6=read）；notify/原子 → 远端 `Notify_Record` | 统一走网搬运：通道 = `Xn[GetXnId(xcId)]` 寄存器值 → 对端 rank/eid 解析同 A5（4.8/6.7 节）；`msIdMode==1` 时 write 源 / read 目的为本地 MS（buf_type=ms）；dmaOpCode 5 的写后 notify 与 0x1A 远端原子计数（无数据面）按远端 record 近似（mask=通知值低 16 位）；`udfEnable` 走网规约按普通 UB 搬运计费（5.7 节近似）；0x19 及其余 dmaOpCode / `udfType != 0` 告警跳过数据面（setCKE post 仍执行） |
| TRANS (0x2) | SyncXnWtx (0xD)、SyncAtx (0xE) | 远端 `Notify_Record`（近似）；SyncAtx 另有 setCKE → 本端 `Notify_Record` | 跨 rank 寄存器同步：SyncWtX 的 `notifyValid==1` notify 与 SyncAtx 的原子写按远端 record 近似（远端目标类型单 rank 不可解析，6.6 节）；主值同步为远端寄存器写不建节点（A5 SyncXn 同口径）；SyncWtX 尾部 setCKE 字段 checker 不消费（本端不 post）；通道不可解析告警跳过（`eid_unresolved`） |
| REDUCE (0x3) | ReduceAdd (0x0)、ReduceMax (0x1)、ReduceMin (0x2) | `Reduce_Inline` | 本地 MS 批量规约（msId[8] 列表 + `Xn[XnIdLength]` 长度）；无内嵌 waitCKE，尾部 setCKE 叠加 `Notify_Record`；MS 恒本地 HBM 计费（6.3.2 节） |
| （子图挂接） | —（成图阶段合成，非微码） | 无（不产生任务节点） | 同 A5（7.4.1 节 extra 边挂接） |

> A6 与 checker 的既定偏差（HBM 数据面、远端目标类型解析、checker 报错场景的
> 宽容降级等）汇总见 `_MissionContextA6` 类 docstring 与 6.7 节对照表；
> loop 合并/流水线并发恢复（6.5/6.5.1 节）、队列并行化（7.4.1 节）、SQE 归并
> 与停滞消解（6.1/6.6 节）等框架机制两代共用。

### 11.4 HcclTaskProfInfo 三字段对照（profInfo → JSON 显示）

v3 采集起 db 任务携带 `HcclTaskProfInfo`（4.4.1 节，单源定义
`include/sim_profiling_defs.h`）。本节给出三字段"值 → 枚举 → JSON 显示字符串"
的完整对照及各值与仿真 TaskType（时延模型分类，7.3 节）的对应关系；
C++ 侧名称函数（`ProfTaskParamTypeName`/`ProfLinkTypeName`/
`ProfTransportTypeName`）为显示字符串的唯一权威来源。

> **实现状态**：`task_type`/`transport_type`/`link_type`/`protocol` 四字段均已按
> 本节对照落地（Python 侧显示名映射于 `db_reader.py` 的 `PROF_*_NAMES`，与 C++
> 名称函数逐字符一致，11.4.5 节；`graph_builder._add_task` 统一填充——`task_type`
> 经 `Task.task_name`/`display_name`，10.3 节 #18 已收尾）。

#### 11.4.1 profTaskType（ProfTaskParamType，0~22 → msprof timeline op 名）

本表补充各值与仿真 TaskType（`task_model.py`，时延模型分类）的对应关系——此前
db 任务与仿真任务类型经 db taskType 列匹配（11.2 节），profInfo 消费后以本表
`profTaskType → 仿真 TaskType` 为第二匹配维度（判定条件不变，同一 db taskType
按 jettyId/方向分流到不同 TaskType 的行内注明）：

| 值 | 枚举 | JSON 显示字符串（op 名） | 仿真 TaskType（时延模型分类） | light_perf 出现场景 |
|---|---|---|---|---|
| 0 | TASK_SDMA | `Memcpy` | `Memcpy`（jettyId 无效，HBM 计费）/ `Ub_Inline_Write`·`Ub_Write_Or_Read`（jettyId 有效按方向，端口计费） | 本地拷贝、跨片 SDMA SQE（linkType 区分，11.4.2）✓ |
| 1 | TASK_RDMA | `RDMASend` | `Memcpy`（jettyId 无效，HBM 计费近似） | DPU host ibv 普通读写（`RDMASend`，hcomm UbMemTransport::Write/Read 同口径）✓ |
| 2 | TASK_REDUCE_INLINE | `Reduce_Inline` | `Reduce_Inline`（jettyId 无效，HBM 计费）/ `Ub_Inline_Write`·`Ub_Write_Or_Read`（走网近似，5.7 节，端口计费） | 本地规约 ✓ |
| 3 | TASK_REDUCE_TBE | `Reduce_TBE` | —（不出现） | 910 代预留，不出现 |
| 4 | TASK_NOTIFY_RECORD | `Notify_Record` | `Notify_Record` | SDMA/本地 notify record ✓ |
| 5 | TASK_NOTIFY_WAIT | `Notify_Wait` | `Notify_Wait` | notify wait ✓ |
| 6/7 | TASK_SEND_NOTIFY / TASK_SEND_PAYLOAD | `Send_Notify` / `Send_Payload` | —（不出现） | hcomm 预留，不出现 |
| 8 | TASK_WRITE_WITH_NOTIFY | `Write_With_Notify` | `Write_With_Notify` | WRITE_WITH_NOTIFY (7) 整任务（单任务映射，10.3 节 #19 已支持）✓ |
| 9 | TASK_WRITE_REDUCE_WITH_NOTIFY | `Reduce_With_Notify`（工具侧修正拼写；hcomm 源码/msprof 为错拼 `Reuce_With_Notify`，11.4.5 节） | `Write_With_Notify`（isReduce=1 的规约写载荷，data_type 区分显示） | WRITE_WITH_NOTIFY (7) isReduce=1 |
| 10 | TASK_CCU | `Ccu` | （CCU 占位整 SQE，展开为第 6 章子图，11.3 节） | CCU 占位整 SQE ✓ |
| 11/12 | TASK_AICPU_KERNEL / TASK_AICPU_REDUCE | `AicpuKernel` / `Aicpu_Reduce` | —（不消费） | AICPU kernel 下发（走 device 进程，当前无采集落点/不消费） |
| 13 | TASK_AIV | `AivKernel` | —（跳过） | AIV kernel 任务（跳过） |
| 14 | TASK_UB_INLINE_WRITE | `Ub_Inline_Write` | `Notify_Record`（notify 固定 record 时延；显示名已归还 op 名 ✓） | UBDMA `inlineEn==1` 的 notify inline 写 |
| 15 | TASK_UB_REDUCE_INLINE | `Ub_Reduce_Inline` | `Ub_Inline_Write`（write）/ `Ub_Write_Or_Read`（read）（走网规约近似，端口计费） | UBDMA `udfFlag==1` 走网规约 |
| 16 | TASK_UB | `Ub_Write_Or_Read` | `Ub_Inline_Write`（write 方向）/ `Ub_Write_Or_Read`（read 方向）（方向由 direction/is_write 表达） | UBDMA 普通读写 WQE（write/read 两方向同值） |
| 17~22 | TASK_DPU_KERNEL / THREAD_FENCE / CHANNEL_FENCE / INLINE_WRITE / NOTIFY_WAIT / WRITE_WITH_NOTIFY | `DpuKernel` / `Dpu_ThreadFence` / `Dpu_ChannelFence` / `Dpu_Notify_Record` / `Dpu_Notify_Wait` / `Dpu_Write_With_Notify` | `Dpu_Notify_Record`→`Notify_Record`、`Dpu_Notify_Wait`→`Notify_Wait`（跨 rank 固定时延）、`Dpu_Write_With_Notify`→`Write_With_Notify`（单任务端口计费，11.4.6 节）；`DpuKernel`/Fence 类当前采集路径不产生 | host CPU RoCE（ibv_dpu_stub）任务 ✓（11.4.6 节） |
| 0xFF | UNKNOWN | `Unknown` | —（SYNC_STREAM 等已跳过，不出节点） | SYNC_STREAM、level1 通道 API 任务、v2 旧版 BLOB 等未标注来源 |

> `task_type` 已切换为 op 名（10.3 节 #18 已收尾），本表"仿真 TaskType"列现为
> 内部时延模型分类索引（11.4.5 节 #3）；✓ 行 op 名与 TaskType 值一致。
> SQE 批合并（5.1.1 节）后，合并任务的 op 名及 link/transport 取批内**最后一个**
> 数据任务的 profInfo（描述随最后任务、长度取全批累加），v3 及更早 BLOB 无
> sqeGroupId 不合并、逐任务显示。

#### 11.4.2 linkType（ProfLinkType，0~9 → hcomm DfxLinkType 枚举名）

| 值 | 枚举 | JSON 显示字符串 | light_perf 出现场景 |
|---|---|---|---|
| 0 | LINK_ONCHIP | `ONCHIP` | 本地 Memcpy/Reduce_Inline/Notify_* |
| 1 | LINK_HCCS | `HCCS` | 预留（当前采集路径不产生） |
| 2 | LINK_PCIE | `PCIE` | 跨片 SDMA SQE（同设备判定为本地 ONCHIP，4.4.1 节采集规则） |
| 3 | LINK_ROCE | `ROCE` | host CPU RoCE（DPU，ibv_dpu_stub）任务 ✓（11.4.6 节） |
| 4/5/6 | LINK_SIO / HCCS_SW / STANDARD_ROCE | `SIO` / `HCCS_SW` / `STANDARD_ROCE` | 预留 |
| 7 | LINK_UB | `UB` | 走网任务（UBDMA WQE 全类） |
| 8 | LINK_UBoE | `UBoE` | UBoE 通道（采集侧暂不区分 UB/UBoE，暂不出现） |
| 9 | LINK_RESERVED | `RESERVED` | 预留 |
| 0xFF | UNKNOWN | `Unknown` | CCU/AIV/SYNC_STREAM、level1 通道 API 任务 |

> **实现状态**：trace args 的 `link_type` 字段已按本表填充（`prof_link_type_name`，
> `graph_builder._add_task` 统一注入，11.1 节方案 a）；本表与原 `link_type` 所载
> `CommProtocol` 通道协议（HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO）为**两个不同
> 概念**：CommProtocol 描述通道建链协议，DfxLinkType 描述任务执行链路（profiling
> 维度）。CommProtocol 名已迁移至 `protocol` 字段（无通道信息时 `N/A`）。

#### 11.4.3 transportType（ProfTransportType，0~4 → msprof L1 明细 transportType 值域）

| 值 | 枚举 | JSON 显示字符串 | light_perf 出现场景 |
|---|---|---|---|
| 0 | TRANSPORT_SDMA | `SDMA` | 跨片 SDMA SQE |
| 1 | TRANSPORT_RDMA | `RDMA` | 预留（910 代 RDMA 语义） |
| 2 | TRANSPORT_LOCAL | `LOCAL` | 本地任务（Memcpy/Reduce_Inline/Notify_*） |
| 3 | TRANSPORT_UB | `UB` | 走网任务（**时延模型锚点**：端口带宽共享计费，5.5 节） |
| 4 | TRANSPORT_ROCE | `ROCE` | host CPU RoCE（DPU，ibv_dpu_stub）任务 ✓（11.4.6 节） |
| 0xFF | UNKNOWN | `Unknown` | 未标注任务 |

> **实现状态**：trace args 的 `transport_type` 字段已按本表填充
> （`prof_transport_type_name`，`graph_builder._add_task` 统一注入）。
> 同名 `Memcpy` 任务的本地/跨片差异由 linkType+transportType 表达：
> ONCHIP+LOCAL（本地）vs PCIE+SDMA（跨片）——trace args 现可无损区分这两类
> 此前在 JSON 中不可分的任务。

#### 11.4.4 Ub_Inline_Write 双向语义错位与切换实施记录

**历史问题**：切换前 JSON 的 `Ub_Inline_Write`（走网 write 方向发送）与 hcomm
profiling 的 `Ub_Inline_Write`（TASK_UB_INLINE_WRITE，notify inline 写）**指代
不同的任务**，且互为反向错位：

| 任务（采集语义） | db taskType | ProfTaskParamType (op 名) | 切换前 JSON 类型 |
|---|---|---|---|
| UBDMA 普通写 WQE（走网发送，write 方向） | MEM_CPY (3) | `TASK_UB` → `Ub_Write_Or_Read` | `Ub_Inline_Write` ← 名字错位 |
| UBDMA 走网规约（write/read 方向） | REDUCE (2) | `TASK_UB_REDUCE_INLINE` → `Ub_Reduce_Inline` | `Ub_Inline_Write` / `Ub_Write_Or_Read` ← 名字错位 |
| UBDMA inline 写 notify（`inlineEn==1`） | NOTIFY_RECORD (1) | `TASK_UB_INLINE_WRITE` → `Ub_Inline_Write` | `Notify_Record` ← 名字错位 |

**已实施（10.3 节 #18 收尾）**——trace/dump 显示名切换为 `profInfo.profTaskType`
的 op 名（`Task.task_name`，`graph_builder._add_task` 统一注入），三处错位全部消除：
1. 走网数据任务（TASK_UB）write/read 统一显示 `Ub_Write_Or_Read`；**方向不再编码
   进类型名**，由 trace args 现有 `direction`（write/read）与 `is_write` 字段
   表达——时延模型不变，仍按端口 write/read 方向带宽共享分别计费（5.5 节）。
2. `Ub_Inline_Write` 名字**归还**给 notify inline 写任务（db taskType=
   NOTIFY_RECORD + inlineEn==1），与 msprof timeline 一一对应。
3. 走网规约显示 `Ub_Reduce_Inline`（TASK_UB_REDUCE_INLINE），保留规约语义可辨。
4. `Notify_Record`/`Notify_Wait`/`Reduce_Inline`/`Memcpy` 四类原本已与 op 名一致，
   切换无变化（`Memcpy` 对应 TASK_SDMA）。

**实施取舍（既定决策）**：
- v2 旧版 BLOB（无 profInfo，profTaskType=0xFF）显示 `Unknown`——与
  `link_type`/`transport_type` 的 v2 行为同口径（严格语义，不做内部名回退）。
- CCU 微码子任务（11.3 节）无 profInfo，显示名回退内部 TaskType 名——simulator
  语义产物无对账需求，维持现状（走网搬运统一 `Ub_Write_Or_Read`，方向由
  `direction`/`is_write` 表达，11.3.1 节）。
- 内部 `TaskType` 枚举退役为时延模型分类（11.4.5 节 #3），仿真/配对/计费零改动。

#### 11.4.5 字符串一致性约定（对账基础）

1. **Python 侧显示字符串必须与 C++ 侧三个名称函数输出逐字符一致**
   （`sim_profiling_defs.h` 单源），包括 hcomm 源码的既有拼写
   （`Ub_Write_Or_Read`、`Dpu_Notify_Record` 等）——这是与 msprof timeline /
   L1 明细逐字段对账的基础。
   **唯一例外**：`TASK_WRITE_REDUCE_WITH_NOTIFY` 的 op 名工具侧统一为修正拼写
   `Reduce_With_Notify`（hcomm 源码两处均为错拼 `Reuce_With_Notify`：
   `profiling_common.h:42` 与 `dfx_profiling_handler_lite.cc:67`，msprof 原始
   数据显示的是错拼版本）；跨系统对账遇到该字符串时需做
   `Reuce_With_Notify ↔ Reduce_With_Notify` 的差异映射。
2. trace args 字段级切换（**已全部落地**，10.3 节 #18 收尾）：
   `transport_type` ← transportType 名、`link_type` ← linkType 名
   （`graph_builder._add_task` 统一注入，值域表见 11.4.2/11.4.3）、CommProtocol
   迁移至新字段 `protocol`（方案 a，11.1 节）；`task_type` ← profTaskType op 名
   （`Task.task_name`/`display_name`，11.4.4 节实施记录）。
 3. `TaskType` 枚举已退役为内部时延模型分类（`task_name` 承担显示职责），
    trace 事件名（`traceEvents[].name`）与 `args.task_type` 均已切换为 op 名；
    CCU 微码子任务/CLI 输入无 profInfo，`display_name` 回退内部名。

#### 11.4.6 DPU（host CPU RoCE）任务采集与合并

host DPU 传输（hcomm `HostCpuRoceChannel`，pod 组网跨节点路径）的 ibv 劫持任务
（`ibv_dpu_stub.cc`）此前不标 profInfo（显示 `Unknown`）、WriteWithNotify 拆两条
任务、对端解析回环到本端。本节为该路径的对齐实施记录。

**profInfo 标注**（对齐 hcomm `host_cpu_roce_channel.cc` DFX 口径，link/transport
恒 ROCE/ROCE）：

| ibv 操作 | db taskType | profTaskType（op 名） |
|---|---|---|
| 纯 imm notify（len=0 WRITE_WITH_IMM / hybrid） | NOTIFY_RECORD | `TASK_DPU_INLINE_WRITE` → `Dpu_Notify_Record` |
| 写+imm（len>0 WRITE_WITH_IMM） | **WRITE_WITH_NOTIFY (7) 单任务** | `TASK_DPU_WRITE_WITH_NOTIFY` → `Dpu_Write_With_Notify` |
| 纯 RDMA 写/读（无 imm） | MEM_CPY | `TASK_RDMA` → `RDMASend`（对齐 hcomm `UbMemTransport::Write/Read` 先例） |
| poll_cq 配对 | NOTIFY_WAIT | `TASK_DPU_NOTIFY_WAIT` → `Dpu_Notify_Wait` |

**WriteWithNotify 合并单任务**：整操作只产一条 taskType=7 任务
（`WriteWithNotifyTaskData`：src/dst rank、offsets、len、notifyId、protocol），
对齐 hcomm 整操作只上报一条 `Dpu_Write_With_Notify` 的口径；`DpuPendingNotify`
配对机制不变。仿真侧 `_map_write_with_notify` 走端口 write 方向带宽共享
（本端 RoCE eid → ranktable `d2h` 端口；`d2h` 未配置端口带宽时回退
`default_port_bandwidth`）。

**真实对端解析（幻影回声问题，socket 时序绑定）**：estimate 模式 socket 层构造
幻影对端（本端 Send 镜像回放为 Recv，`hccp_ra_socket_stub.cc`），QP 握手交换回的
"远端 gid"实为本端自身，`RaTypicalQpModify` 的 gid→IP 解析回到本端。修复
（零字节读取，仅依赖 API 调用时序）：
- `RaGetSockets` 由 `SocketInfoT.remoteIp`（hcomm 链路配置，client/server 角色
  均携带）解析真实对端 EndPoint 存入 FdInfo（peerDeviceId/peerEid/localEid）；
- `RaGetQpAttr` 锚点（hccp_stub，仅 estimate 分支内执行）记录待绑 QP
  （`DpuRecordPendingBindQp`，`bindAttempted` 防 Modify 重复查 attr 后误绑）；
- 同线程下一次 `RaSocketSend`（即 hcomm ExchangeData 的 attr 打包→Send 时序）经
  `BindDpuPendingQps` 将 QP 绑定到该 socket 真实对端：`DpuQpInfo` 新增
  `peerRankId`/`lcEid`（对端 rank 由 FdInfo 的 peerEid 查 `RankEidMap`，estimate
  通信域表只有本 rank；lcEid 供 WRITE_WITH_NOTIFY 端口计费）。
`RaTypicalQpModify` 维持 `remoteDeviceId` 的回环平面语义
（`DpuPendingNotify` 配对依赖，estimate 下=本端），任务元数据一律取绑定字段。

**NotifyWait 精确合成（hcomm 原语拦截）**：正确语义为"对端 post 插入的 pending
才给我生成 NotifyWait 用"，幻影下对端不运行、该通知无真实来源，且 hcomm
`NotifyWait` 对 poll 结果做 imm 硬校验（`wc.imm_data == dpuNotifyId`，不匹配立即
报错）。实现（`hcomm_api_stub.cc` 拦截 5 个 hcomm 北向原语，真身指针 static
一次性 dlsym 解析，**正常模式第一行短路纯转发，零开销**）：
- post 类原语（`HcommChannelNotifyRecordOnThread` / `HcommWriteWithNotify(±Nbi)
  OnThread`，入参携带 `remoteNotifyIdx`）→ `InsertDpuPendingNotifyEstimate`
  记录 `channelSlots[channel][idx] = (imm, notifyId)` 槽位映射——回声下
  `remoteDpuNotifyIds_ == localDpuNotifyIds_`，post 侧 idx→imm 映射即 wait 侧
  期望映射（同 idx 同槽位值）；
- wait 类原语（`HcommChannelNotifyWaitOnThread(±WithDefaultTimeout)`，入参携带
  `localNotifyIdx`）→ `ConsumeDpuPendingNotifyAndEmit` 精确路径：期望
  imm = `channelSlots[channel][idx]` **精确已知**，直接合成"虚拟对端 post"构造
  wc 返回，imm 必过 hcomm 硬校验；NotifyWait 节点记录的是本地真实等待事件，
  虚拟化的仅是"通知到达"这一外部事实（与 socket 回声虚拟化"对端应答"同构）；
- 设计约束：评估模式的 wait 消费**只走精确路径**——所有 wait 入口必须拦截
  （单卡幻影下 FIFO 回环消费不可靠，收发不对称时 imm 错配/超时，已随旧机制移除）；
  未拦截入口 → `pollCq::miss` **ERROR**（每 qpn+idx 组合去重一次，防 busy-loop
  刷爆日志）→ hcomm 超时暴露，据此在 `hcomm_api_stub.cc` 补拦截。

**taskSrc 来源标记（BLOB v5）**：`HcclTaskMetaData` 新增 `taskSrc`
（0=默认，1=DPU），DPU stub 发射的任务置 1，供 Python 侧按来源精确切分
（不依赖 transport==ROCE 等链路语义判据，AICPU RoCE 等未来来源不受影响）。

**DPU 序列合并 pass（5.1.2 节，`sqe_group_merge.py::merge_dpu_write_sequences`）**：
hcomm `WriteWithNotify` 对 len > maxMsgSize 的大消息拆为 N-1 个普通 RDMA_WRITE +
1 个收尾 WRITE_WITH_IMM（整操作仍只报一条）；采集侧逐 WR 落库为 N-1 条 MEM_CPY +
1 条 WRITE_WITH_NOTIFY。pass 按 DPU 流序扫描 src/dst 双地址连续的 MEM_CPY 链，
由地址连续的 WRITE_WITH_NOTIFY 收尾合并为单条（len 全链累加、描述随收尾任务含
notifyId、占收尾槽位）；无收尾的链不合并（独立纯写保持逐条 `RDMASend`）；
仅 `src_type==1` 的数据任务参与，其他任务原样保留不切断链。

> **验证**（222 组网 8MB all_reduce）：首轮（20260921）15 个 `Unknown` → 0
> （`Dpu_Notify_Record` ×5 / `Dpu_Notify_Wait` ×9 / `Dpu_Write_With_Notify` ×4，
> link=ROCE），对端 rank=2/4/6（真实跨 pod 对端），`eid_unresolved=0`，
> WRITE_WITH_NOTIFY 进入 `d2h` 端口计费。终版（20260922，原语拦截后）：
> estimate（-np 1）makespan 166.596us、原语拦截 16 次、精确合成 11 次、miss=0；
> normal（-np 8）checker All Success（原语纯转发，拦截日志 0 条）；单机 8p
> （118）SQE 路径回归无变化。
