# CCU Executor Debug 工具设计文档

## 1. 概述

### 1.1 目标

为 CCU Executor 模块构建一个调试回放工具，通过采集指令执行过程中的 trace 数据，在前端 UI 上实现单步回放，展示每条指令执行时的：
- 指令本身的信息（类型、参数、所属 CCU）
- 指令执行过程中的 debug 信息（关键变量值、执行路径）
- CCU 资源的变化（XN/GSA/CKE/MS/Channel 的前后对比）

### 1.2 代码现状分析

**CCU Executor 模块架构** (`src/plugin/solver/virtual_runtime/ccu_executor/`)：

```
CcuExecutorBase (抽象基类)
├── Parser()          // 解析指令字段
├── Run()             // 执行指令逻辑
├── Process()         // 资源操作（部分指令实现）
└── Describe()        // 生成指令描述字符串

CcuSimulator (执行引擎)
├── Execute()         // 主循环：串行执行指令
├── ExecuteInstr()    // 单条指令执行
├── ExecuteLoop()     // Loop 循环执行
└── ExecuteLoopGroup()// LoopGroup 展开执行

CcuResourceManager (资源管理 - 单例)
├── XN[]    // 通用寄存器 (V1:3072, V2:4096)
├── GSA[]   // 地址寄存器 (V1:3072, V2:4096)
├── CKE[]   // 同步信号   (1024)
├── MS[]    // Memory Slice 4K (1536)
├── Channel[] // 通信通道 (128)
└── InstrSpace // 指令空间
```

**指令分类（4大类）**：

| 类型 | 子类 (V1) | 子类 (V2新增) | 操作的资源 |
|------|-----------|---------------|-----------|
| **Load** | LoadSqeArgsToGsa, LoadSqeArgsToXn, LoadImdToGsa, LoadImdToXn, LoadGsaXn, LoadGsaGsa, LoadXX | LoadImdToX, LoadX, StoreX, ClearX, Nop, Load, Store, Add, Sub, Mul, And, Or, Not, Xor, Shl, Shr, Popcnt | XN, GSA |
| **Trans** | LocMem↔LocMem, LocMem↔LocMS, LocMem↔RmtMem, LocMS↔LocMS, LocMS↔RmtMS, LocMS↔RmtMem, RmtMem↔LocMem, RmtMS↔LocMem, RmtMS↔LocMS, SyncCke, SyncGsa, SyncXn | TransMem, SyncXnWt, SyncAt | MS, GSA, XN, Channel, CKE |
| **Control** | Loop, LoopGroup, SetCke, ClearCke, Jump | Wait, Fence | CKE, LoopEngine, XN |
| **Reduce** | ReduceAdd, ReduceMax, ReduceMin | (同V1) | MS, CKE |

**资源归属关系**：
```
Rank[0..N]
  └── CCU[0..1]  (die0, die1)
       ├── XN[0..max]
       ├── GSA[0..max]
       ├── CKE[0..1023]
       ├── MS[0..1535]  (each 4KB)
       ├── Channel[0..127]
       └── InstrSpace (指令列表)
```

---

## 2. 回放方案分析：日志回放 vs Trace 回放

### 2.1 方案对比

| 维度 | 日志回放 (Log Replay) | Trace 回放 (Structured Trace) |
|------|----------------------|-------------------------------|
| **数据采集** | 拦截 `HCCL_VM_DEBUG/INFO/TRACE` 日志输出 | 在执行引擎中插桩，采集结构化数据 |
| **数据格式** | 非结构化文本字符串 | JSON/MsgPack 结构化数据 |
| **信息完整性** | 仅包含日志中打印的信息，存在遗漏 | 可采集完整资源快照，不遗漏 |
| **解析复杂度** | 需要正则表达式解析，脆弱且维护成本高 | 直接反序列化，无解析成本 |
| **存储大小** | 文本冗长，体积大 | 二进制编码紧凑，可增量记录 |
| **回放精度** | 无法精确恢复资源状态 | 可精确恢复到任意指令点的完整状态 |
| **前端集成** | 需要额外的日志解析服务 | 前端直接加载 JSON/MsgPack 渲染 |
| **性能开销** | 低（仅文本输出） | 中等（需序列化+快照） |
| **扩展性** | 新增指令需更新正则 | 新增指令只需扩展 trace 结构 |
| **业界参考** | GDB textual trace | Chrome DevTools Timeline, LLVM Execution Trace, rr debugger |

### 2.2 结论：采用 Trace 回放方案

**理由**：
1. **精确性**：CCU 调试的核心需求是观察资源变化，日志无法保证完整采集 XN/GSA/CKE/MS 的值
2. **前端友好**：HVRM Insight 已有完善的数据加载管线（Worker + JSON），结构化 trace 数据可直接对接
3. **增量记录**：仅记录变化的资源字段（delta），而非全量快照，控制存储大小
4. **单步回放**：trace 数据天然支持按指令序号定位，日志回放则难以实现精确的单步控制

---

## 3. Trace 数据结构设计

### 3.1 设计原则

参考业界做法：
- **Chrome DevTools Trace Event Format**：分层结构，每个事件包含 category、name、timestamp、args
- **LLVM Execution Trace**：指令级 trace + 寄存器快照
- **rr (Record and Replay) debugger**：事件流 + 关键检查点

结合本地代码特点：
- 指令天然分为 4 大类（Load/Trans/Control/Reduce），每类操作不同资源
- CCU 资源按 die 独立，trace 需关联 (rankId, dieId)
- 指令串行执行，trace 天然有序

### 3.2 总体架构

#### 3.2.1 执行模型回顾（代码实际逻辑）

CCU 执行并非 per-CCU 独立运行，而是**全局交错调度**。核心调度逻辑在 `SequentialExecutor::Execute()` 中：

```
while (HasTask()) {                          // 外层轮次循环（round）
    for (rank in allRanks) {                 // 遍历所有 rank
        for (stream in rankStreams) {        // 遍历每个 rank 的每个流
            while (stream 有任务) {
                ret = ExecuteOneTask(stream.front());
                if (ret == HOLD_CMD) break;  // waitCKE 卡住 → 跳到下一个 stream
                stream.pop();                // 任务完成 → 出队
            }
        }
    }
}
```

当某个 CCU 的 SQE 执行到 waitCKE 指令且条件不满足时，`TaskCcuGraph()` 返回 `HOLD_CMD`，
该 SQE 任务**不出队**，外层循环继续处理下一个 CCU。等另一 CCU 执行 SetCke 设置了对应的 CKE 后，
下一轮外层循环重试该 SQE 时 CKE 条件满足，继续执行。

关键事实：
- **CCU 资源独立**：每个 CCU (rankId, dieId) 拥有独立的 XN/GSA/CKE/MS/指令空间
- **一个 CCU 一个 Simulator**：同一个 (rankId, dieId) 上的多个 SQE 复用同一个 `CcuSimulator` 实例（通过 `Init()` 重置指令指针）
- **SQE 描述执行范围**：每个 SQE 任务包含 missionId、起始指令 ID、指令数量、入参列表
- **全局交错执行**：实际执行顺序是 R0:D0 → R0:D1 → R1:D0 → R1:D1 → R0:D0 → ...（所有 CCU 轮流执行）

#### 3.2.2 Trace 总体架构

Trace 必须是**全局的**，记录所有 CCU 的交错执行序列，而非 per-CCU 独立记录。

```
CcuTraceRun (一次完整的虚拟运行时执行，覆盖所有 rank 和 CCU)
├── runMetadata              // 运行级元信息（算子名、rank 数、CCU 版本等）
├── ccuRegistry[]            // CCU 注册表：所有参与执行的 CCU (rankId, dieId)
│   └── CcuIdentity           //   每个 CCU 的标识和初始资源快照
├── sqeTaskRegistry[]        // SQE 任务注册表：所有 SQE 任务（被 entry 引用）
│   └── CcuSqeTask            //   missionId、指令范围、入参、所属 CCU
│
│  ┌─────────── 静态配置层（per CCU，初始化后不变）───────────┐
├── instrSpaces[]            // 指令空间（per CCU，纯静态快照）
│   └── CcuInstrSpace         //   rankId + dieId + instructions[instrId, describe]
├── channelSpaces[]          // Channel 映射表（per CCU，纯静态快照）
│   └── CcuChannelSpace       //   rankId + dieId + channels[channelId, remoteRankId, remoteDieId]
│  └──────────────────────────────────────────────────────────┘
│
│  ┌─────────── 动态执行层（运行时变化）──────────────────┐
├── globalEntries[]          // 全局指令执行序列（按实际交错顺序排列）
│   └── CcuTraceEntry         //   每条 entry 通过 (rankId, dieId, instrId) 索引指令空间
│       ├── (rankId, dieId,    //   三元组索引指令空间中的指令描述
│       │    instrId)           //
│       ├── sqeRef              // → sqeTaskRegistry 中的 SQE 任务
│       ├── globalSeqId         //   全局执行序号（反映实际交错顺序）
│       ├── execRound           //   外层 while(HasTask()) 的轮次号
│       ├── resourceDelta       //   本 CCU 资源变化（XN/GSA/CKE/MS，不含 Channel）
│       ├── crossCcuChanges     //   跨 CCU 资源变更
│       ├── waitInfo            //   CKE Wait 自旋信息
│       ├── errorInfo           //   执行失败信息
│       └── detail.args         //   指令专属动态参数（运行时值，同一指令不同轮次可能不同）
├── ccuFinalSnapshots{}      // 每个 CCU 的最终资源快照 (key: rankId_dieId)
│  └──────────────────────────────────────────────────────────┘
└── runSummary               // 运行级摘要统计
```

**静态/动态分离设计**：

| 数据 | 存储位置 | 特性 | 说明 |
|------|---------|------|------|
| 指令描述 (Describe) | `instrSpaces[]` | **静态** | 每条指令只存一份，不随 Loop 迭代变化 |
| 指令类别/名称 | `instrSpaces[]` | **静态** | 由指令字节决定，固定不变 |
| Channel 映射表 | `channelSpaces[]` | **静态** | InitChannelInfo 初始化后不变，与 instrSpaces 并列存放 |
| 运行时参数 | `detail.args` | **动态** | Loop offset 后的实际值、SQE 参数值等 |
| 资源变化 | `resourceDelta` | **动态** | 每次执行的资源 before/after（不含 Channel） |
| 执行上下文 | `context` | **动态** | Loop 轮次、偏移量等 |

**同一指令在 Loop 中执行多遍**：指令空间只有 1 条记录，但 trace 产生 N 条 entry，每条有不同的 `detail.args`、`resourceDelta`、`context`。

### 3.3 数据结构详细定义

#### 3.3.1 顶层：CcuTraceRun、CCU 注册表、SQE 注册表

```cpp
// ===== 一次完整的虚拟运行时执行 =====
// 覆盖所有 rank、所有 CCU、所有 SQE 的全局执行过程
struct CcuTraceRun {
    CcuRunMetadata runMetadata;                        // 运行级元信息
    std::vector<CcuIdentity> ccuRegistry;              // CCU 注册表
    std::vector<CcuSqeTask> sqeTaskRegistry;           // SQE 任务注册表
    
    // ===== 静态配置层（per CCU，初始化后不变）=====
    std::vector<CcuInstrSpace> instrSpaces;            // 指令空间（per CCU，纯静态快照）
    std::vector<CcuChannelSpace> channelSpaces;        // Channel 映射表（per CCU，纯静态快照）
    
    // ===== 动态执行层（运行时变化）=====
    std::vector<CcuTraceEntry> globalEntries;          // 全局指令执行序列（交错有序）
    std::vector<CcuTraceNonCcuEntry> nonCcuEntries;    // 非 CCU 任务记录
    std::map<std::string, CcuResourceSnapshot> ccuFinalSnapshots;  // 每个 CCU 的最终快照（仅动态资源）
    CcuRunSummary runSummary;                          // 运行级摘要统计
};

// ===== 运行级元信息 =====
struct CcuRunMetadata {
    uint32_t traceFormatVersion;                       // trace 格式版本号（当前 = 1）
    std::string algorithmName;                         // 算子名称（如 "AllReduce"）
    std::string operatorName;                          // 操作名称（如 "RingAllReduce_Step1"）
    uint32_t rankSize;                                 // rank 总数
    uint32_t diePerRank;                               // 每个 rank 的 die 数（固定 = 2）
    RunnerCcuVersion ccuVersion;                       // CCU 微码版本
    uint64_t runTimestampNs;                           // 运行开始时间戳
    uint32_t totalSqeTaskCount;                        // SQE 任务总数
    uint32_t totalExecRounds;                          // 外层 while(HasTask()) 的总轮次数
};

// ===== CCU 标识（注册表条目） =====
// 每个参与执行的 CCU 在注册表中有且仅有一条记录
struct CcuIdentity {
    int32_t rankId;                                    // rank 编号
    uint16_t dieId;                                    // die 编号（0 或 1）
    RunnerCcuVersion ccuVersion;                       // CCU 微码版本
    CcuResourceSnapshot initialSnapshot;               // 该 CCU 的初始资源快照
    std::string ccuKey() const {                       // 唯一标识 key: "rankId_dieId"
        return std::to_string(rankId) + "_" + std::to_string(dieId);
    }
};

// ===== SQE 任务（注册表条目） =====
// 每个 SQE 任务在注册表中有且仅有一条记录，被 CcuTraceEntry 通过 sqeTaskId 引用
struct CcuSqeTask {
    uint32_t sqeTaskId;                                // SQE 任务 ID（在 Run 内唯一，从 0 递增）
    int32_t rankId;                                    // 所属 rank
    uint16_t dieId;                                    // 所属 die
    uint8_t missionId;                                 // SQE 任务编号（CcuTask.missionId）
    uint16_t instStartId;                              // 起始指令 ID（CcuTask.instStartId）
    uint16_t instCnt;                                  // 指令数量（CcuTask.instCnt）
    uint32_t key;                                      // 任务 key（CcuTask.key）
    std::vector<uint64_t> args;                        // SQE 入参列表（CcuTask.args[]，最多 13 个）
    uint64_t simulatorPtr;                             // CcuSimulator 实例指针（同一 CCU 的多个 SQE 共享同一个）
    uint32_t firstExecRound;                           // 首次被执行的外层轮次号
};

// ===== 运行级摘要 =====
struct CcuRunSummary {
    uint32_t totalInstrExecuted;                       // 总执行指令数
    uint32_t totalFailedInstr;                         // 失败指令数
    uint32_t totalCkeWaitSpins;                        // CKE 等待自旋总次数
    uint32_t totalHoldEvents;                          // HOLD 事件总数（waitCKE 导致的中断）
    std::map<std::string, uint32_t> instrCountByCategory;  // 按指令类别统计
    std::map<std::string, uint32_t> instrCountByCcu;       // 按 CCU 统计（key: "rankId_dieId"）
    uint64_t totalMsBytesTransferred;                  // MS 搬运总字节数
};
```

#### 3.3.2 指令空间：CcuInstrSpace（per CCU，纯静态快照）

指令空间按 CCU 独立存储，包含该 CCU 的所有指令的静态信息（指令 ID + 预计算的 Describe() 输出）。
指令空间与 trace 动态参数**完全解耦**：trace entry 仅通过 `(rankId, dieId, instrId)` 三元组索引指令空间。

```cpp
// 指令空间中的单条指令（纯静态）
struct CcuInstrSpaceEntry {
    uint16_t instrId;                        // 指令 ID（在指令空间中的索引）
    std::string instrDescribe;               // 预计算的 Describe() 输出（静态，不含运行时值）
};

// 单个 CCU 的指令空间
struct CcuInstrSpace {
    int32_t rankId;                          // 所属 rank
    uint16_t dieId;                          // 所属 die
    std::vector<CcuInstrSpaceEntry> instructions;  // 该 CCU 的所有指令
};
```

**采集时机**：trace 开始前，遍历每个 CCU 的指令空间，对所有指令调用 `Parser()` + `Describe()`，将结果缓存。

**前端使用**：前端加载指令空间后，左侧栏显示完整指令列表（纯静态），回放时通过 `(rankId, dieId, instrId)` 高亮对应行。

#### 3.3.3 公共层：CcuTraceEntry

每条 entry 记录**一个 CCU 上的一条指令执行**，通过 `(rankId, dieId, instrId)` 三元组索引指令空间获取指令描述。

```cpp
// 单条指令的 trace 条目（全局时间线中的一个点）
struct CcuTraceEntry {
    // === 全局定位信息 ===
    uint32_t globalSeqId;                   // 全局执行序号（从 0 递增，反映实际交错顺序）
    uint32_t execRound;                     // 外层 while(HasTask()) 的轮次号（从 0 递增）

    // === CCU 归属 + 指令索引（三元组索引指令空间） ===
    int32_t rankId;                         // 所属 rank
    uint16_t dieId;                         // 所属 die
    uint32_t instrId;                       // 指令 ID，与 (rankId, dieId) 一起索引 instrSpaces
    // 通过 (rankId, dieId, instrId) 可在 instrSpaces 中找到指令描述

    // === SQE 归属 ===
    uint32_t sqeTaskId;                     // 引用 sqeTaskRegistry 中的 CcuSqeTask

    // === 指令类别（仅用于统计聚合，1 字节） ===
    CcuInstrCategory category;              // 指令大类: Load/Trans/Control/Reduce

    // === 执行状态 ===
    CcuExecState execState;                 // 执行后的状态

    // === 执行上下文 ===
    CcuExecutionContext context;            // Loop/Jump 上下文信息

    // === 资源变化 (Delta) ===
    CcuResourceDelta resourceDelta;         // 本条指令引起的本 CCU 资源变化
    CcuCrossCcuChanges crossCcuChanges;     // 本条指令引起的跨 CCU 资源变化（如有）

    // === CKE Wait 自旋信息 ===
    CcuWaitInfo waitInfo;                   // Wait 自旋合并信息（仅在 waitCKE 时有效）

    // === 执行失败信息 ===
    CcuErrorInfo errorInfo;                 // 失败信息（仅 execState == EXEC_FAIL 时有效）

    // === 指令专属细节 (多态) ===
    std::unique_ptr<CcuInstrTraceDetail> detail;  // 指令特有的 debug 信息
};

// 指令分类枚举
enum class CcuInstrCategory : uint8_t {
    LOAD = 0,       // 加载/存储/算术
    TRANS = 1,      // 数据搬运
    CONTROL = 2,    // 控制流
    REDUCE = 3,     // 归约运算
};

// 执行上下文（Loop/Jump 相关，不含 SQE 信息 —— SQE 通过 sqeTaskId 引用获取）
// Loop 内的 GSA 地址偏移计算公式（来自 CcuExecutorBase::UpdateAddress）：
//   finalOffset = (loopExtendNum * gsaOffset << addrExpandCoef)
//               + (curLoopCnt * iterStepGSA << addrExpandCoef)
struct CcuExecutionContext {
    bool inLoop;                            // 是否在 Loop 中执行
    uint16_t loopRound;                     // 当前 Loop 迭代轮次
    uint16_t loopExtendIndex;               // Loop 展开索引

    // === Loop 内偏移参数（仅在 inLoop=true 时有效） ===
    // GSA 地址偏移参数
    uint32_t gsaOffset;                     // GSA 基础偏移系数（GetGSAOffset）
    uint64_t iterStepGSA;                   // 每轮迭代的 GSA 步长（GetLoopIterStepGSA）
    uint32_t loopExtendNum;                 // 展开数（GetLoopExtendNum）
    uint32_t curLoopCnt;                    // 当前迭代计数（GetCurLoopCnt）
    uint64_t gsaAddrOffset;                 // 计算后的最终 GSA 地址偏移值
    uint16_t addrExpandCoef;                // 地址扩展系数（UpdateAddress 的第 2 个参数）

    // 资源 ID 偏移参数
    uint16_t msOffset;                      // MS ID 偏移（GetLoopMsOffset）
    uint16_t ckeOffset;                     // CKE ID 偏移（GetLoopCKEOffset）
    uint16_t xnIdOffset;                    // XN ID 偏移（GetLoopXnIdOffset）
};

// CKE Wait 自旋合并信息
// 当一条指令因 CKE 条件不满足而自旋等待时，多次 ExecuteInstr 调用
// 会被合并为一条 CcuTraceEntry，而非产生多条重复记录。
struct CcuWaitInfo {
    bool hadWait;                           // 是否发生过 CKE 等待
    uint32_t waitRetryCount;                // 自旋等待的次数
    uint16_t waitCKEId;                     // 等待的 CKE ID
    uint16_t waitCKEMask;                   // 等待的 CKE mask
    uint16_t ckeValueOnFirstCheck;          // 首次检查时的 CKE 值
    uint16_t ckeValueOnPass;                // 最终通过时的 CKE 值
    uint64_t waitDurationNs;                // 等待总耗时（纳秒）
};

// 执行失败信息
struct CcuErrorInfo {
    bool hasError;                          // 是否有错误
    std::string errorMessage;              // 错误描述（取自 HCCL_VM_ERROR 日志）
    std::string failPhase;                  // 失败阶段: "Parser" / "Run" / "Process"
    CcuResourceSnapshot failSnapshot;       // 失败时的完整资源快照（便于事后分析）
};

// 跨 CCU 资源变更
// 当指令操作远端 CCU 资源时（如 SetRmtCKE、TransLocMSToRmtMS），
// 记录对远端 CCU 的影响。
struct CcuCrossCcuChanges {
    bool hasCrossCcuChange;                 // 是否有跨 CCU 变更
    std::vector<CcuRemoteCkeChange> remoteCkeChanges;   // 远端 CKE 变更
    std::vector<CcuRemoteMsChange> remoteMsChanges;     // 远端 MS 变更
    std::vector<CcuRemoteMemChange> remoteMemChanges;   // 远端内存变更
};

struct CcuRemoteCkeChange {
    int32_t remoteRankId;
    uint16_t remoteDieId;
    uint16_t ckeId;
    uint16_t valueBefore;
    uint16_t valueAfter;
};

struct CcuRemoteMsChange {
    int32_t remoteRankId;
    uint16_t remoteDieId;
    uint16_t msId;
    uint64_t offset;
    uint32_t length;
    std::vector<uint8_t> dataAfter;         // 仅记录 after（远端 before 由远端 trace 覆盖）
};

struct CcuRemoteMemChange {
    int32_t remoteRankId;
    uint64_t remoteAddr;
    uint64_t length;
    std::vector<uint8_t> dataAfter;
};
```

#### 3.3.4 资源层：CcuChannelSpace & CcuResourceSnapshot & CcuResourceDelta

```cpp
// ===== Channel 空间（per CCU，纯静态快照）=====
// Channel 映射表与指令空间一样，是 CCU 独有的静态资源：
//   - InitChannelInfo() 初始化后不再变化
//   - 指令执行时只读取（通过 GetRmtCcu()），从不修改
//   - 与 XN/GSA/CKE/MS 等动态资源分离存放
struct CcuChannelSpace {
    int32_t rankId;
    uint16_t dieId;
    std::vector<CcuChannelRecord> channels;  // channelId → (remoteRankId, remoteDieId)
};

// CCU 资源快照（仅包含动态资源：XN/GSA/CKE/MS）
// 注：Channel 已移至 CcuChannelSpace（静态配置，与 CcuInstrSpace 并列）
struct CcuResourceSnapshot {
    std::vector<CcuXnRecord> xnRecords;      // XN 寄存器快照
    std::vector<CcuGsaRecord> gsaRecords;    // GSA 寄存器快照
    std::vector<CcuCkeRecord> ckeRecords;    // CKE 信号快照
    std::vector<CcuMsRecord> msRecords;      // MS 内存切片快照（可选，体积大）
};

// 资源变化增量（仅记录变化的部分）
// 注：Channel 表是静态配置，不会变化，因此不包含 channelChanges
struct CcuResourceDelta {
    std::vector<CcuXnChange> xnChanges;      // XN 变化列表
    std::vector<CcuGsaChange> gsaChanges;    // GSA 变化列表
    std::vector<CcuCkeChange> ckeChanges;    // CKE 变化列表
    std::vector<CcuMsChange> msChanges;      // MS 变化列表
};

// ========== 资源记录（快照用） ==========

struct CcuXnRecord {
    uint16_t id;
    uint64_t value;
};

struct CcuGsaRecord {
    uint16_t id;
    uint64_t value;
};

struct CcuCkeRecord {
    uint16_t id;
    uint16_t value;
};

struct CcuMsRecord {
    uint16_t id;
    std::vector<uint8_t> data;  // 4KB 数据
};

struct CcuChannelRecord {
    uint16_t channelId;
    int32_t remoteRankId;
    uint16_t remoteDieId;
};

// ========== 资源变化（Delta 用） ==========

struct CcuXnChange {
    uint16_t id;
    uint64_t valueBefore;
    uint64_t valueAfter;
};

struct CcuGsaChange {
    uint16_t id;
    uint64_t valueBefore;
    uint64_t valueAfter;
};

struct CcuCkeChange {
    uint16_t id;
    uint16_t valueBefore;
    uint16_t valueAfter;
};

struct CcuMsChange {
    uint16_t id;
    uint64_t offset;              // 变化起始偏移
    uint32_t length;              // 变化长度
    std::vector<uint8_t> dataBefore;  // 变化前数据
    std::vector<uint8_t> dataAfter;   // 变化后数据
};

// 注：Channel 表是静态配置，不会变化，因此无 CcuChannelChange 结构
```

#### 3.3.5 专属层：CcuInstrTraceDetail（仅含运行时动态参数）

每种指令类型特有的**运行时动态参数**。由于代码中有 30+ 种指令类型（V1 约 20 种 + V2 约 10+ 种），
且不同指令的参数差异较大，采用**扁平 key-value map** 而非多态继承，以降低开发和维护成本。

**关键区分**：`CcuInstrTraceDetail.args` **只存储运行时动态参数**，即同一指令在不同 Loop 轮次或不同 SQE 任务中可能不同的值。
静态信息（如指令描述、固定参数 ID）存储在指令空间中，不在此处重复。

```cpp
// 指令专属 trace 细节（仅含运行时动态参数）
struct CcuInstrTraceDetail {
    std::string typeName;                     // 指令类型名称（如 "LoadSqeArgsToXn"）

    // 运行时动态参数（key-value 形式）
    // 仅记录随 Loop 迭代或 SQE 任务变化的值
    // 示例：
    //   LoadSqeArgsToXn:
    //     {"sqeArgValue": "0x1000"}           // SQE 参数值随任务不同可能不同
    //
    //   TransLocMSToLocMem (Loop 内):
    //     {"resolvedLocMSId": "15",           // offset 后的实际 MS ID
    //      "resolvedLocGSAId": "103",         // offset 后的实际 GSA ID
    //      "resolvedLocXnId": "52"}           // offset 后的实际 XN ID
    //
    //   ReduceAdd:
    //     {"reduceOp": "Add", "dataType": "FP32"}
    std::map<std::string, std::string> args;
    // 注意：不包含 describeText（指令描述存储在指令空间中，通过 instrId 索引）
};
```

**静态 vs 动态分工**：
| 信息类型 | 存储位置 | 特性 | 说明 |
|----------|---------|------|------|
| 指令描述 | `instrSpaces[].instructions[].instrDescribe` | **静态** | 预计算的 Describe() 输出 |
| 指令类别/名称 | `instrSpaces[].instructions[]` | **静态** | 由指令字节决定 |
| CKE 等待信息 | `CcuTraceEntry.waitInfo` | **动态** | waitCKEId/mask、自旋次数等 |
| CKE 资源变化 | `CcuResourceDelta.ckeChanges` | **动态** | CKE before/after 值 |
| XN/GSA/MS 变化 | `CcuResourceDelta.xnChanges` 等 | **动态** | 资源 before/after 值 |
| Loop 偏移上下文 | `CcuExecutionContext` | **动态** | loopRound、offset 等 |
| 指令专属动态参数 | `CcuInstrTraceDetail.args` | **动态** | 运行时解析后的值（offset 后、SQE 参数等） |

### 3.4 数据序列化格式

采用 **JSON** 作为序列化格式（与 Insight 工具现有数据格式一致），后续可升级为 MsgPack/FlatBuffers 以减小体积。

JSON Schema 概要：
```json
{
    "runMetadata": {
        "traceFormatVersion": 1,
        "algorithmName": "AllReduce",
        "operatorName": "RingAllReduce_Step1",
        "rankSize": 2,
        "diePerRank": 2,
        "ccuVersion": "CCU_V2",
        "runTimestampNs": 1717000000000000000,
        "totalSqeTaskCount": 4,
        "totalExecRounds": 3
    },
    "ccuRegistry": [
        {
            "rankId": 0, "dieId": 0, "ccuVersion": "CCU_V2",
            "initialSnapshot": {
                "xnRecords": [{"id": 0, "value": "0x0"}, ...],
                "gsaRecords": [{"id": 0, "value": "0x7f0000"}, ...],
                "ckeRecords": [{"id": 0, "value": 0}, ...]
            }
        },
        { "rankId": 0, "dieId": 1, ... },
        { "rankId": 1, "dieId": 0, ... },
        { "rankId": 1, "dieId": 1, ... }
    ],
    "sqeTaskRegistry": [
        {
            "sqeTaskId": 0, "rankId": 0, "dieId": 0,
            "missionId": 3, "instStartId": 0, "instCnt": 50, "key": 1,
            "args": ["0x1000", "0x2000", ...],
            "simulatorPtr": "0x7f3a0000", "firstExecRound": 0
        },
        { "sqeTaskId": 1, "rankId": 0, "dieId": 1, ... }
    ],
    "instrSpaces": [
        {
            "rankId": 0, "dieId": 0,
            "instructions": [
                {"instrId": 0, "instrDescribe": "[Simulation Execute] locCcu[0:0], Load SqeArg[0] to Xn[5]"},
                {"instrId": 1, "instrDescribe": "ParseTransLocMemToLocMemInstr Wait CKE[0:0000]..."},
                {"instrId": 2, "instrDescribe": "[Simulation Execute] Wait CKE[0:0000], Set CKE[3:0001]..."},
                ...
            ]
        },
        { "rankId": 0, "dieId": 1, "instructions": [...] },
        { "rankId": 1, "dieId": 0, "instructions": [...] },
        { "rankId": 1, "dieId": 1, "instructions": [...] }
    ],
    "channelSpaces": [
        {
            "rankId": 0, "dieId": 0,
            "channels": [
                {"channelId": 0, "remoteRankId": 1, "remoteDieId": 0},
                {"channelId": 1, "remoteRankId": 2, "remoteDieId": 1},
                ...
            ]
        },
        { "rankId": 0, "dieId": 1, "channels": [...] },
        { "rankId": 1, "dieId": 0, "channels": [...] },
        { "rankId": 1, "dieId": 1, "channels": [...] }
    ],
    "globalEntries": [
        {
            "globalSeqId": 0,
            "execRound": 0,
            "rankId": 0,
            "dieId": 0,
            "instrId": 0,
            "sqeTaskId": 0,
            "category": "Load",
            "execState": "EXEC_NORMAL_INSTR",
            "context": {
                "inLoop": false,
                "loopRound": 0,
                "loopExtendIndex": 0,
                "gsaOffset": 0,
                "iterStepGSA": 0,
                "loopExtendNum": 0,
                "curLoopCnt": 0,
                "gsaAddrOffset": 0,
                "addrExpandCoef": 0,
                "msOffset": 0,
                "ckeOffset": 0,
                "xnIdOffset": 0
            },
            "resourceDelta": {
                "xnChanges": [
                    {"id": 5, "valueBefore": "0x0", "valueAfter": "0x1000"}
                ],
                "gsaChanges": [],
                "ckeChanges": [],
                "msChanges": []
            },
            "detail": {
                "typeName": "LoadSqeArgsToXn",
                "args": {"sqeArgValue": "0x1000"}
            }
        },
        {
            "globalSeqId": 1,
            "execRound": 0,
            "rankId": 0,
            "dieId": 1,
            "instrId": 0,
            ...
        }
    ],
    "ccuFinalSnapshots": {
        "0_0": { ... },
        "0_1": { ... }
    },
    "runSummary": {
        "totalInstrExecuted": 1280,
        "totalFailedInstr": 0,
        "totalCkeWaitSpins": 42,
        "totalHoldEvents": 3,
        "instrCountByCategory": {"Load": 400, "Trans": 500, "Control": 300, "Reduce": 80},
        "instrCountByCcu": {"0_0": 320, "0_1": 320, "1_0": 320, "1_1": 320}
    }
}
```

---

## 4. 采集插桩点设计

### 4.1 插桩位置

在 `CcuSimulator::ExecuteInstr()` 中，指令执行前后各插入一次采集点。**关键：CKE Wait 自旋合并**。

由于 `WaitCkeProcess` 中 CKE 条件不满足时，会设置 `waitCKE_=true` 导致 `ExecuteInstr` 返回 false，上层 `Execute()` 重新调用同一条指令。如果不做合并，一条 Wait 指令可能产生数百条重复 trace entry。

**解决方案**：采集器检测连续的同一条指令重复执行（CKE Wait 自旋），将其合并为一条 trace entry，并记录自旋次数。

```cpp
bool CcuSimulator::ExecuteInstr(uint16_t curInstrId)
{
    auto &ccuResMgr = CcuResourceManager::GetInstance();
    auto instrData = ccuResMgr.GetInstrData(rankId_, dieId_);

    // ① 采集执行前快照
    auto snapshotBefore = CcuTraceCollector::CaptureResourceSnapshot(rankId_, dieId_);

    auto executor = CcuExecutorFactory::MakeCcuExecutorInstance(...);
    executor->Parser();
    executor->Run();

    // ② CKE Wait 自旋检测与合并
    if (waitCKE_) {
        // 本条指令因 CKE 等待而返回 false，采集器记录一次自旋
        CcuTraceCollector::RecordWaitSpin(rankId_, dieId_, curInstrId,
                                           snapshotBefore);
        // 不记录 trace entry，等 CKE 满足后再记录最终合并结果
        return false;
    }

    // ③ CKE 通过（或本条指令不涉及 CKE 等待），采集执行后快照
    auto snapshotAfter = CcuTraceCollector::CaptureResourceSnapshot(rankId_, dieId_);
    auto detail = executor->CollectTraceDetail();

    // ④ 获取跨 CCU 变更（由 CcuResourceManager 在指令执行过程中实时拦截记录，见 4.4 节）
    auto crossChanges = CcuTraceCollector::ConsumeCrossCcuChanges(rankId_, dieId_);

    // ⑤ 合并 CKE Wait 自旋信息（如果之前有自旋），记录最终 trace entry
    auto waitInfo = CcuTraceCollector::FinalizeWaitInfo(rankId_, dieId_, curInstrId,
                                                         snapshotBefore);

    // ⑥ 检测执行失败
    auto errorInfo = CcuTraceCollector::CaptureErrorInfo(rankId_, dieId_, executor.get());

    // ⑦ 计算 delta 并记录 trace entry（含 wait、error、cross-ccu 信息）
    // globalSeqId、execRound、sqeTaskId 由 SequentialExecutor 层传递给 CcuTraceCollector（见 4.3 节）
    auto globalCtx = CcuTraceCollector::GetCurrentGlobalContext();
    CcuTraceCollector::RecordEntry(rankId_, dieId_, curInstrId,
                                    globalCtx.globalSeqId,
                                    globalCtx.execRound,
                                    globalCtx.currentSqeTaskId,
                                    snapshotBefore, snapshotAfter,
                                    detail, waitInfo, errorInfo, crossChanges);

    UpdateLoopStatus();
    return true;
}
```

### 4.2 CcuExecutorBase 扩展

在基类中新增 `CollectTraceDetail()` 虚方法，各子类按需实现：

```cpp
class CcuExecutorBase {
public:
    // 新增：采集指令专属 trace 细节（扁平 key-value 结构）
    // 默认实现：仅包含 typeName 和 Describe() 输出
    virtual CcuInstrTraceDetail CollectTraceDetail() {
        CcuInstrTraceDetail detail;
        detail.typeName = "Unknown";
        detail.describeText = Describe();
        return detail;
    }
};
```

各 Executor 子类实现示例（填充指令特有参数到 args map 中）：

```cpp
// TransLocMemToLocMemExecutor
CcuInstrTraceDetail TransLocMemToLocMemExecutor::CollectTraceDetail() {
    CcuInstrTraceDetail detail;
    detail.typeName = "TransLocMemToLocMem";
    detail.describeText = Describe();
    detail.args = {
        {"srcAddr", std::to_string(srcLocAddr_)},
        {"dstAddr", std::to_string(dstLocAddr_)},
        {"transLength", std::to_string(transLength_)},
    };
    // 注：CKE 同步信息已在 CcuWaitInfo 和 CcuResourceDelta 中记录，无需重复
    return detail;
}

// ReduceAddExecutor
CcuInstrTraceDetail ReduceAddExecutor::CollectTraceDetail() {
    CcuInstrTraceDetail detail;
    detail.typeName = "ReduceAdd";
    detail.describeText = Describe();
    detail.args = {
        {"count", std::to_string(count_)},
        {"castEn", std::to_string(castEn_)},
        {"dataType", std::to_string(dataType_)},
        {"msList", ParseMSList()},
    };
    return detail;
}
```

### 4.3 全局上下文管理

`globalSeqId`、`execRound`、`currentSqeTaskId` 等全局上下文信息只能在 `SequentialExecutor` 层获取，需要从上向下传递给 `CcuTraceCollector`。

```cpp
// SequentialExecutor::Execute() 中管理全局上下文
while (HasTask()) {
    CcuTraceCollector::BeginRound(execRound);    // 设置当前轮次

    for (rankTasks : allRankTaskQueues_) {
        for (streamTasks : rankTasks) {
            while (!streamTasks.empty()) {
                auto task = streamTasks.front();
                if (task.taskType == CCU_GRAPH) {
                    // 注册 SQE 任务并获取 sqeTaskId
                    uint32_t sqeTaskId = CcuTraceCollector::RegisterSqeTask(
                        task.rankId, task.dieId, task.missionId,
                        task.instStartId, task.instCnt, task.args);
                    CcuTraceCollector::SetCurrentSqeTaskId(sqeTaskId);
                }
                CcuTraceCollector::BeginGlobalStep();  // 原子递增 globalSeqId
                auto ret = ExecuteOneTask(task);
                if (ret == HOLD_CMD) break;
                streamTasks.pop();
            }
        }
    }
    execRound++;
}
```

`CcuSimulator::ExecuteInstr()` 中通过 `CcuTraceCollector::GetCurrentGlobalContext()` 读取这些值。

### 4.4 跨 CCU 变更采集（ResourceManager 拦截）

跨 CCU 变更（如 `SetRmtCKESignal`、`TransLocMSToRmtMS` 等）在 Executor 的 `Process()` 中发生，操作远端 CCU 的资源。相比事后对比远端 CCU 状态，更可靠的方式是在 `CcuResourceManager` 的 Update 方法中直接拦截：

```cpp
// CcuResourceManager::UpdateCkeValue() 中拦截跨 CCU 变更
void CcuResourceManager::UpdateCkeValue(int rankId, int dieId, uint16_t ckeId, uint16_t value) {
    uint16_t oldValue = GetCkeValue(rankId, dieId, ckeId);

    // 检测是否为跨 CCU 操作
    auto [execRank, execDie] = CcuTraceCollector::GetCurrentExecutingCcu();
    if (rankId != execRank || dieId != execDie) {
        CcuTraceCollector::RecordCrossCcuCkeChange(
            rankId, dieId, ckeId, oldValue, value, execRank, execDie);
    }

    // 执行实际更新
    ckeData_[rankId][dieId][ckeId] = value;
}
```

同理适用于 `UpdateXnValue()`、`UpdateGsaValue()`、`TransMSToMS()` 等涉及远端 CCU 的方法。

拦截的跨 CCU 变更记录在 `CcuTraceCollector` 的缓冲区中，由 `CcuSimulator::ExecuteInstr()` 在记录 trace entry 时通过 `ConsumeCrossCcuChanges()` 消费。

### 4.5 资源快照采集

```cpp
class CcuTraceCollector {
public:
    static CcuResourceSnapshot CaptureResourceSnapshot(int rankId, int dieId);
    static CcuResourceDelta ComputeDelta(const CcuResourceSnapshot& before,
                                          const CcuResourceSnapshot& after);
    static void RecordEntry(int rankId, int dieId, uint16_t instrId,
                            const CcuResourceSnapshot& before,
                            const CcuResourceSnapshot& after,
                            std::unique_ptr<CcuInstrTraceDetail> detail);
    static void StartSession(int rankId, int dieId, RunnerCcuVersion version);
    static void EndSession(int rankId, int dieId);
    static void DumpToFile(const std::string& outputPath);

    // 全局上下文管理（由 SequentialExecutor 层调用）
    static void BeginRound(uint32_t execRound);
    static void BeginGlobalStep();
    static uint32_t RegisterSqeTask(int rankId, int dieId, uint8_t missionId,
                                     uint16_t instStartId, uint16_t instCnt,
                                     const std::vector<uint64_t>& args);
    static void SetCurrentSqeTaskId(uint32_t sqeTaskId);
    static CcuGlobalContext GetCurrentGlobalContext();

    // 跨 CCU 变更拦截（由 CcuResourceManager 调用）
    static void RecordCrossCcuCkeChange(int rankId, int dieId, uint16_t ckeId,
                                         uint16_t oldValue, uint16_t newValue,
                                         int execRank, int execDie);
    static std::pair<int,int> GetCurrentExecutingCcu();
    static CcuCrossCcuChanges ConsumeCrossCcuChanges(int rankId, int dieId);
};
```

### 4.6 性能优化策略

#### 4.6.1 采集端优化

1. **按需启用**：通过环境变量 `HCCLVM_ENABLE_CCU_TRACE=1` 控制是否采集
2. **增量记录**：仅记录非零变化的资源字段（delta），不存储未变化的资源
3. **MS 数据分级采集**（通过 `HCCLVM_CCU_TRACE_MS_LEVEL` 控制）：
   - `none`（默认）：不采集 MS 数据，仅记录 MS ID 和长度
   - `digest`：采集 MS 数据的哈希摘要（16 字节），用于完整性校验
   - `range`：仅记录被修改的偏移区间（offset + length + before/after）
   - `full`：记录完整 4KB（仅在需要数据对比时使用）
4. **延迟写入**：先内存缓存，执行结束后批量写入文件
5. **可选全量快照**：通过 `HCCLVM_CCU_TRACE_FULL_SNAPSHOT=1` 启用全量快照模式

#### 4.6.2 Loop 折叠与采样

Loop 执行可能导致 trace 数据爆炸（一个 Loop 1000 轮 × N 条指令 = 数千条 entry）。采用折叠 + 采样策略控制数据量：

```
环境变量控制：
  HCCLVM_CCU_TRACE_LOOP_MODE=fold|sample|full

fold 模式（默认）：
  将同一 Loop 的多轮迭代折叠为一条摘要记录：
  ┌─────────────────────────────────────────────────┐
  │ CcuLoopFoldEntry                                │
  │ ├── loopInstrId: 5                              │
  │ ├── totalRounds: 1000                           │
  │ ├── firstRoundEntries: [...]   // 首轮完整 trace │
  │ ├── lastRoundEntries: [...]    // 末轮完整 trace │
  │ ├── resourceStats:                              │
  │ │   ├── xnWriteCount: {5: 1000, 8: 500}         │
  │ │   ├── ckeToggleCount: {3: 2000}               │
  │ │   └── avgMsBytesPerRound: 4096                │
  │ └── anomalyRounds: [42, 87]  // 异常轮次序号     │
  └─────────────────────────────────────────────────┘
  前端默认显示折叠视图，点击"展开"可查看任意轮次详情。

sample 模式：
  仅记录首轮 + 末轮 + 每隔 N 轮采样（N 由 HCCLVM_CCU_TRACE_SAMPLE_RATE 控制，默认 10）

full 模式：
  记录所有轮次（仅用于小规模调试，数据量大）
```

异常轮次检测规则：
- 该轮指令数与首轮不一致
- 该轮资源变化模式与首轮显著不同（delta 字段集合不同）
- 该轮包含 EXEC_FAIL
- 该轮 CKE Wait 次数异常（超过首轮 2 倍以上）

#### 4.6.3 前端渲染优化

当 trace entry 数量超过 1 万条时，前端需要特殊处理：

1. **虚拟滚动**：指令列表使用 virtual scroll（如 `vue-virtual-scroller`），仅渲染可见行
2. **懒加载**：初始只加载 entry 的摘要信息（seqId/instrId/name/category），点击时才加载完整 detail
3. **分页加载**：大数据集按 1000 条/页分块加载，前端分页浏览
4. **资源热图降采样**：资源概览面板中，当资源数量过多时按区间分组聚合显示

### 4.7 非 CCU 任务 Trace 记录

`SequentialExecutor::Execute()` 中除了 `CCU_GRAPH` 外，还有 `NOTIFY_RECORD/NOTIFY_WAIT/REDUCE/MEM_CPY/AIV_GRAPH` 等任务类型。`NotifyWait` 也会返回 `HOLD_CMD`，影响全局调度流程。如果不在 trace 中记录，前端回放时会出现"CCU 指令突然跳到下一轮"的现象。

在 `globalEntries[]` 中增加非 CCU 任务的简化记录：

```cpp
struct CcuTraceNonCcuEntry {
    uint32_t globalSeqId;                   // 全局执行序号
    uint32_t execRound;                     // 外层调度轮次
    int32_t rankId;                         // 所属 rank
    HccLTaskMetaType taskType;              // 任务类型: NOTIFY_RECORD/NOTIFY_WAIT/REDUCE/MEM_CPY 等
    HcclVmResult execResult;                // 执行结果: SUCCESS 或 HOLD_CMD
    std::string description;                // 任务描述（便于前端展示）
};
```

前端在回放时可显示调度事件，如"Round 1: Rank1 NotifyWait 阻塞 → 跳过"，帮助用户理解 CCU 切换的原因。

### 4.8 MVP 分阶段落地策略

**第一阶段（MVP）**：

最小可用的 trace 采集，快速验证前端回放流程：

1. 采集 `Describe()` 输出 + resource delta（通过 ResourceManager 拦截变更）
2. `CcuInstrTraceDetail` 的 `args` 为空，`describeText` 直接使用 `executor->Describe()` 的返回值
3. 不实现每个 Executor 的 `CollectTraceDetail()` 定制版本，使用基类默认实现
4. 前端仅展示指令列表面板 + 资源变化面板 + 指令描述面板
5. 不支持 Loop 折叠/采样、跨 CCU 变更、非 CCU 任务记录

**第二阶段（增强）**：

1. 为高频/关键指令类型实现 `CollectTraceDetail()`（如 Trans、Reduce、SyncCke）
2. 前端展示完整的 key-value 参数表格
3. 支持 Loop 折叠/采样
4. 支持跨 CCU 变更记录和非 CCU 任务记录

**第三阶段（完善）**：

1. 逐步覆盖所有 30+ 种指令类型的 `CollectTraceDetail()`
2. 支持搜索/过滤、数据依赖追溯、Trace Diff 对比
3. 支持断点功能
4. 可选升级为 msgpack/flatbuffers 二进制格式

---

## 5. 前端集成方案

### 5.1 现有 HVRM Insight 架构

```
App.vue
├── AppTopNav.vue          // 顶部导航栏
├── DashboardPage.vue      // 总览页
├── MemViewPage.vue        // 关联分析页
└── AnalyticPage.vue       // 报错诊断页
```

技术栈：Vue 3 + Element Plus + Vite

### 5.2 新增页签：CCU Trace Replay

在 `App.vue` 中新增第四个页签：

```javascript
const pages = [
  { id: 'dashboard', label: '总览' },
  { id: 'mem-view', label: '关联' },
  { id: 'analytic', label: '报错' },
  { id: 'ccu-trace', label: 'CCU调试' },  // 新增
]
```

### 5.3 CCU Trace Replay 页面设计

#### 整体布局

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  CCU Trace Replay                                                            │
├────────────┬─────────────────────────────────────────────────────────────────┤
│            │  ┌─ Trace 进度条 ─────────────────────────────────────────────┐  │
│  左侧栏     │  │  Trace: Rank [0 ▼]  Die [0 ▼]    #4/1280  Round 0        │  │
│            │  │  ──────●──────○────────○────────○────────○────────  0.3%   │  │
│  ────────  │  └───────────────────────────────────────────────────────────┘  │
│            │  ┌───────────────────────────────────────────────────────────┐  │
│  回放       │  │         指令列表面板  (Rank0:Die0 指令空间)                 │  │
│  控制器     │  │  BP | instrId | 指令描述 (Describe)         | 执行状态     │  │
│            │  │   ○ | 0       | [Load] LoadImd to Xn[5]...  | ✓ #0 R0     │  │
│  [⏮][⏪]   │  │   ● | 1       | [Trans] LocMem→LocMem...    | ✓ #4 R0     │  │
│  [▶/⏸]    │  │   ○ | 2       | [SetCke] Set CKE[3:0001]    | ✓ #7 R0     │  │
│  [⏩][⏭]   │  │ ▶ ○ | 3       | [WaitCKE] Wait CKE[0:0000]..| ← 当前 #12  │  │
│  [⇥ 下一断点]│  │   ○ | 4       | [Trans] LocMem→RmtMem...    |             │  │
│            │  │   ○ | 5       | [Reduce] Add MS[0:1,1:2,2:3]|             │  │
│  ────────  │  │   ...                                                     │  │
│            │  │  ▶ = 当前回放位置  ● = 断点  ○ = 无断点                      │  │
│  当前CCU:   │  │  执行状态: ✓=已执行 ←当前=当前步骤 HOLD=CKE阻塞 FAIL=失败   │  │
│  Rank0:Die0 │  └───────────────────────────────────────────────────────────┘  │
│            │  ┌─────────────────────────┬─────────────────────────────────┐  │
│  ────────  │  │   资源变化面板            │   指令细节面板                    │  │
│            │  │                         │                                 │  │
│  SQE 任务   │  │  XN Changes             │  指令: WaitCKE                   │  │
│  (当前指令  │  │  Xn[5]: 0→0x1000        │  CCU: Rank0:Die0  Round: 0       │  │
│   所属)     │  │  Xn[8]: 0x400→0         │                                 │  │
│            │  │                         │  执行上下文:                     │  │
│  mission: 3 │  │  GSA Changes            │  Loop Round: 0                   │  │
│  sim*:      │  │  (none)                 │  Extend Index: 0                 │  │
│  0x7f3a..   │  │                         │  GSA Offset: 0x4000              │  │
│  args[0..12]│  │  CKE Changes            │                                 │  │
│  0x1000..   │  │  CKE[3]: 0→1            │  CKE Wait 信息:                  │  │
│            │  │                         │  waitCKE: CKE[0]&0x1 expect=1    │  │
│  ────────  │  │  MS Changes             │  等待自旋: 5 轮 (首次CKE=0)       │  │
│            │  │  MS[0]: @0x0 +4KB       │  最终通过: CKE=1                 │  │
│  执行轨迹   │  │                         │                                 │  │
│  摘要       │  └─────────────────────────┴─────────────────────────────────┘  │
│            │  ┌───────────────────────────────────────────────────────────┐  │
│  已执行 3条  │  │              CCU 资源概览面板                              │  │
│  Round 0   │  │  XN[0..31]  | GSA[0..31] | CKE[0..15] | MS[0..7]       │  │
│            │  │  (展示当前 CCU 即 Rank0:Die0 的累积资源状态)               │  │
│  ────────  │  └───────────────────────────────────────────────────────────┘  │
│            │  ┌───────────────────────────────────────────────────────────┐  │
│  断点       │  │              断点命中提示栏 (仅命中时显示)                  │  │
│  管理       │  │  ⚠ 断点命中: Rank0:Die0 instrId=1 (TransLocMemToLocMem)    │  │
│            │  │  Round 0  missionId=3                                      │  │
│  [+ 添加]  │  │  [继续播放 ▶]  [单步 ⏭]  [跳到下一断点 ⇥]                  │  │
│  [清除全部] │  └───────────────────────────────────────────────────────────┘  │
│            │                                                                 │
│  ┌────────┐│                                                                 │
│  │BP列表   ││                                                                 │
│  │● R0:D0  ││                                                                 │
│  │  id=1   ││                                                                 │
│  │● R1:D0  ││                                                                 │
│  │  id=5   ││                                                                 │
│  │ [✏][🗑]││                                                                 │
│  └────────┘│                                                                 │
└────────────┴─────────────────────────────────────────────────────────────────┘
```

#### 指令列表面板说明

指令列表共 4 列：

| 列 | 内容 | 说明 |
|---|------|------|
| **BP** | ○ / ● / ▶ | 断点标记 + 当前回放位置（`▶` 标记在当前行） |
| **instrId** | 指令 ID | CCU 指令空间中的索引 |
| **指令描述** | `Describe()` 输出 | 包含完整的指令解析信息（参数、地址、寄存器 ID 等） |
| **执行状态** | 指令执行状态 | 展示该指令在当前回放进度下的执行信息 |

**执行状态列含义**：

| 显示 | 含义 |
|------|------|
| （空白） | 尚未执行 |
| `✓ #N RN` | 已执行，#N 为全局序号，RN 为 Round 号 |
| `← 当前 #N` | 当前回放步骤停留在此指令 |
| `HOLD` | waitCKE 阻塞（CKE 条件不满足，等待中），**橙色高亮** |
| `FAIL` | 执行失败，**红色高亮** |

**类型列/SQE 列删除原因**：
- 类型（Load/Trans/Control/Reduce）已包含在 Describe() 输出中，无需独立列
- SQE 是任务级别信息而非指令级别信息，放在左侧栏的 SQE 任务区展示

#### CCU 指令空间视图与自动切换

指令列表面板始终只展示**一个 CCU 的指令空间**（由顶部 Rank/Die 下拉框决定）。trace 回放时，指令列表跟随 trace 执行记录自动切换和定位：

```
trace 回放推进到下一条 entry
       │
       ▼
  entry 的 (rankId, dieId) == 当前显示的 CCU？
       │                              │
       │ 是（同一 CCU 内连续执行）      │ 否（切换到其他 CCU）
       ▼                              ▼
  在当前指令列表中               ① 自动切换顶部 Rank/Die 下拉框
  ▶ 标记移到对应行               ② 指令列表切换到新 CCU 的指令空间
                                 ③ ▶ 标记移到新 CCU 中对应的行
                                 ④ 状态栏显示 "CCU 切换: R0:D0 → R1:D0"
```

**当前回放位置标记**：

`▶` 标记始终显示在 BP 列，标识当前回放步骤所在的行。与断点标记叠加显示（如 `▶●` 表示当前行同时是断点）。

**断点设置**：

用户在当前 CCU 的指令列表中点击 BP 列圆圈设置断点。断点本质上是 `(rankId, dieId, instrId)` 三元组，与当前显示的 CCU 绑定。

#### 核心组件

| 组件 | 功能 |
|------|------|
| `CcuTraceCcuSelector` | **CCU 选择器**：Rank/Die 下拉框，选择当前显示的 CCU 指令空间 |
| `CcuTraceProgressBar` | **Trace 进度条**：显示全局进度、当前 globalSeqId、Round 号 |
| `CcuTraceInstrList` | **指令列表面板**：4 列（BP/instrId/Describe/执行状态），回放时自动高亮/切换 |
| `CcuTraceExecSummary` | **执行轨迹摘要**：左侧栏展示当前 CCU 的已执行数、Round |
| `CcuTraceSqeInfo` | **SQE 任务信息**：左侧栏展示当前指令所属 SQE 的 missionId、simulator 指针、args[0..12] 参数列表 |
| `CcuTracePlaybackControl` | 回放控制器（单步前进/后退/自动播放/跳到下一断点） |
| `CcuTraceBreakpointPanel` | **断点管理面板**：列出所有断点（按 rankId/dieId/instrId 标识），支持增删改 |
| `CcuTraceBreakpointHitBar` | **断点命中提示栏**：命中断点时弹出，含 CCU/SQE/Round 信息，提供 Continue/Step/Skip |
| `CcuTraceResourceDelta` | 资源变化面板，展示当前步骤的 delta |
| `CcuTraceInstrDetail` | 指令细节面板，展示指令特有的 debug 信息 |
| `CcuTraceResourceOverview` | CCU 资源概览，展示当前 CCU 的累积资源状态 |

#### 断点功能设计（前端回放断点）

##### 断点类型

| 类型 | 说明 | 设置方式 |
|------|------|---------|
| **行断点** | 到达指定 globalSeqId 时暂停 | 点击指令列表 BP 列的圆圈 |
| **名称断点** | 所有指定名称的指令处暂停 | 断点管理面板添加 |
| **类别断点** | 所有指定类别（Load/Trans/Control/Reduce）的指令处暂停 | 断点管理面板添加 |
| **资源变更断点** | 指定资源被修改的指令处暂停（如 "Xn[5] 被写入时断点"） | 断点管理面板添加 |
| **轮次断点** | 外层调度进入指定 Round 时暂停（如 "Round 3 的首条指令"） | 断点管理面板添加 |

##### 断点状态

- **○** 无断点（默认）
- **●** 已启用断点
- **⊘** 已禁用断点（保留配置但不触发）
- **▶●** 当前步骤命中断点（高亮 + 弹出命中提示栏）

##### 自动播放遇到断点的行为

```
用户点击 [▶ 自动播放]
       │
       ▼
  逐条推进 ────────► 每步检查：当前 entry 是否命中断点？
       │                              │
       │ 否                           │ 是
       ▼                              ▼
  继续下一步               暂停播放，高亮当前行
                                  │
                          弹出断点命中提示栏：
                          "⚠ 断点命中: Rank0:Die0 instrId=1 (指令名)"
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
            [继续播放 ▶]    [单步 ⏭]       [跳到下一断点 ⇥]
             恢复自动播放   前进一步后暂停    快进到下一个断点处
```

##### 断点管理面板交互

左侧栏底部的断点管理区域展示当前所有断点，支持：
1. **快速添加**：点击指令列表 BP 列，或点击 `[+ 添加]` 打开对话框
2. **启用/禁用**：点击断点条目前的开关
3. **跳转定位**：点击断点条目，指令列表自动滚动到对应行
4. **编辑/删除**：每条断点右侧提供 `[✏ 编辑]` 和 `[🗑 删除]` 按钮
5. **批量操作**：`[清除全部]` 一键移除所有断点

#### 交互逻辑

1. **CCU 选择** → 通过顶部 Rank/Die 下拉框选择当前显示的 CCU 指令空间，指令列表切换到对应 CCU
2. **回放推进** → 点击回放控制按钮，全局推进（globalSeqId 递增）：
   - 若当前 entry 属于已选 CCU → `▶` 标记移到对应 instrId 行，执行状态列更新
   - 若当前 entry 属于其他 CCU → 自动切换 CCU 选择器，指令列表切换到新 CCU 并标记对应行
3. **CCU 切换提示** → 自动切换 CCU 时，状态栏短暂显示 "CCU 切换: R0:D0 → R1:D0"，帮助用户理解调度跳转
4. **自动播放模式** → 按设定速度逐条推进全局序列，模拟实际交错执行过程；CCU 随 trace 自动切换
5. **Round 分隔** → 当 execRound 变化时，左侧执行轨迹摘要区显示 Round 变化（如 "Round 0 → Round 1"）
6. **HOLD 状态** → waitCKE 阻塞的指令在执行状态列显示橙色 `HOLD` 标记；指令细节面板展示等待的 CKE ID、mask、自旋次数
7. **点击指令行** → 联动更新：资源变化面板（展示该指令的 delta）、指令细节面板、CCU 资源概览（更新为当前 CCU 状态）
8. **点击 BP 列圆圈** → 切换该行的断点标记（○ ↔ ●），断点三元组 `(rankId, dieId, instrId)` 自动绑定当前 CCU；断点管理面板同步更新
9. **自动播放遇到断点** → 暂停并弹出命中提示栏（含 CCU/Round 信息），可选继续/单步/跳到下一断点
10. **点击 [⇥ 下一断点]** → 快进到下一个断点位置；若断点在其他 CCU，自动切换 CCU 选择器
11. **执行轨迹摘要** → 左侧栏实时展示当前 CCU 的已执行指令数、当前 Round、当前 SQE 等摘要信息
12. **SQE 信息联动** → 指令细节面板中展示当前指令所属 SQE 的 missionId、instStartId、instCnt、args 列表
13. **资源概览** → 展示当前 CCU 的累积资源状态（XN/GSA/CKE/MS），随回放实时更新
14. **搜索指令** → 在当前 CCU 的指令列表中按指令名、描述、instrId 搜索
15. **追溯资源来源** → 点击资源变化面板中的某个变更值，自动高亮上一次修改该资源的指令（在同一 CCU 内追溯）
16. **跨 CCU 跳转** → 点击 crossCcuChanges 中的远端变更，自动切换 CCU 选择器到对应 CCU 并定位到关联指令
17. **失败快速定位** → 点击"跳转到首个失败"按钮，自动切换到失败指令所属的 CCU，定位到第一条 execState=EXEC_FAIL 的指令，红色高亮显示
18. **手动切换 CCU** → 用户可随时通过下拉框手动切换 CCU，指令列表切换到对应 CCU 的指令空间；回放位置不变，但高亮该 CCU 中最近执行的指令

### 5.4 数据加载

复用 Insight 现有的 Worker 数据加载管线：

```javascript
// 新增数据加载工具
// src/utils/ccuTraceData.js
export async function loadCcuTrace(datasetPath, rankId, dieId) {
    const response = await fetch(`/api/ccu-trace/${datasetPath}/rank${rankId}_die${dieId}.json`);
    return await response.json();
}
```

后端新增 API 端点：

```python
# server.py 新增
@app.route('/api/ccu-trace/<dataset>/<file>')
def ccu_trace_data(dataset, file):
    trace_path = os.path.join(DATA_DIR, dataset, 'ccu_trace', file)
    return send_file(trace_path, mimetype='application/json')
```

---

## 6. 文件组织

### 6.1 后端（C++）

```
src/plugin/solver/virtual_runtime/ccu_executor/trace/
├── ccu_trace_types.h          // trace 数据结构定义
├── ccu_trace_collector.h      // trace 采集器声明
├── ccu_trace_collector.cc     // trace 采集器实现
├── ccu_trace_serializer.h     // JSON 序列化声明
└── ccu_trace_serializer.cc    // JSON 序列化实现
```

### 6.2 前端（Vue）

```
src/plugin/solver/virtual_runtime/insight/frontendV3/src/
├── pages/
│   └── CcuTracePage.vue              // CCU Trace 回放页
├── components/ccu-trace/
│   ├── CcuTraceCcuSelector.vue       // CCU 选择器（Rank/Die 下拉框）
│   ├── CcuTraceProgressBar.vue       // Trace 进度条
│   ├── CcuTraceInstrList.vue         // 指令列表面板（当前 CCU 指令空间）
│   ├── CcuTraceExecSummary.vue       // 执行轨迹摘要（左侧栏）
│   ├── CcuTracePlaybackControl.vue   // 回放控制
│   ├── CcuTraceSqeInfo.vue           // SQE 任务信息（左侧栏）
│   ├── CcuTraceResourceDelta.vue     // 资源变化面板
│   ├── CcuTraceInstrDetail.vue       // 指令细节面板
│   ├── CcuTraceResourceOverview.vue  // CCU 资源概览（当前 CCU）
│   ├── CcuTraceBreakpointPanel.vue   // 断点管理面板
│   └── CcuTraceBreakpointHitBar.vue  // 断点命中提示栏
└── utils/
    └── ccuTraceData.js               // 数据加载 + 索引构建工具
```

---

## 7. 四方关系总结

本设计的核心是将 **指令**、**CCU**、**SQE 任务** 和 **CCU 资源** 四者通过 trace 数据结构紧密关联：

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CcuTraceEntry                                │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │ 全局定位  │  │ 指令信息  │  │  资源变化     │  │  指令专属细节  │   │
│  │          │  │          │  │  (Delta)     │  │  (Detail)     │   │
│  │globalSeq │  │ instrId  │  │ xnChanges[]  │  │ CcuLoadTrace  │   │
│  │execRound │  │ category │  │ gsaChanges[] │  │ CcuTransTrace │   │
│  │rankId    │  │ name     │  │ ckeChanges[] │  │ CcuLoopTrace  │   │
│  │dieId     │  │ describe │  │ msChanges[]  │  │ CcuReduceTrace│   │
│  │sqeTaskId─┤──│          │  │ crossCcuChgs │  │ ...           │   │
│  └──────────┘  └──────────┘  └──────────────┘  └───────────────┘   │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  SQE 任务 (sqeTaskRegistry 中查找)                             │   │
│  │  missionId, instStartId, instCnt, args[], simulatorPtr        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  执行上下文 (Loop/Jump 状态)                                   │   │
│  │  inLoop, loopRound, gsaOffset, msOffset, ckeOffset, ...      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

**关键设计决策**：
1. **全局交错时间线**：所有 CCU 的指令按实际执行顺序交错排列在 `globalEntries[]` 中，`globalSeqId` 反映真实调度顺序
2. **CCU 注册表**：每个 CCU (rankId, dieId) 在 `ccuRegistry` 中有唯一条目，包含初始资源快照；entry 通过 rankId+dieId 关联
3. **SQE 注册表**：每个 SQE 任务在 `sqeTaskRegistry` 中有唯一条目；entry 通过 `sqeTaskId` 引用，避免冗余
4. **Delta 而非全量快照**：每条 trace 只记录变化部分，初始和最终快照用于恢复完整状态
5. **分层多态**：公共层处理所有指令共享的信息，专属层通过多态支持每种指令的特有细节
6. **调度轮次**：`execRound` 记录外层 `while(HasTask())` 的轮次，帮助理解交错调度的时序

---

## 8. 断点功能设计

### 8.1 可行性分析

断点功能完全可行，且可以在两个独立层面实现：

| 层面 | 名称 | 触发时机 | 复杂度 | 用途 |
|------|------|---------|--------|------|
| **前端回放断点** | Offline Breakpoint | 回放已采集的 trace 数据时 | 低 | 自动播放到某指令时暂停，供用户查看细节 |
| **后端执行断点** | Live Breakpoint | CCU 指令实际执行/录制过程中 | 高 | 实时暂停模拟器执行，交互式审查当前状态后继续 |

两者可独立使用，也可联合：先用后端断点精确录制关键区间的 trace，再用前端断点反复回放分析。

### 8.2 断点类型

```cpp
// 断点类型枚举（断点匹配策略，本质都是定位到具体的 rankId+dieId+instrId）
enum class CcuBreakpointType : uint8_t {
    INSTR_ID,           // 指令序号断点：到达指定 instrId 时触发
    INSTR_NAME,         // 指令名称断点：到达指定名称的指令时触发（如 "ReduceAdd"）
    CATEGORY,           // 指令类别断点：到达指定类别时触发（如所有 Trans 类指令）
    RESOURCE_CHANGE,    // 资源变更断点：指定资源被修改的指令处触发（如 Xn[5] 被写入）
    CONDITION,          // 条件断点：资源满足条件表达式的指令处触发（如 Xn[5] == 0x1000）
    ITERATION,          // 迭代断点：Loop 循环到第 N 轮时的指令处触发
    ROUND,              // 调度轮次断点：外层 while(HasTask()) 进入指定轮次时触发（如第 3 轮所有指令）
};

// 断点配置
// 断点本质上只跟指令有关，核心三元组：(rankId, dieId, instrId)
struct CcuBreakpointConfig {
    uint32_t bpId;                      // 断点唯一标识
    CcuBreakpointType type;             // 断点匹配策略类型
    bool enabled;                       // 是否启用
    bool oneShot;                       // 一次性断点（触发后自动禁用）

    // === 断点核心定位信息 ===
    int32_t rankId;                     // 目标 rank（-1 表示所有 rank）
    int32_t dieId;                      // 目标 die（-1 表示所有 die）
    uint16_t instrId;                   // 目标指令 ID

    // === 匹配策略附加参数（根据 type 选用） ===
    std::string targetInstrName;        // INSTR_NAME: 目标指令名称
    CcuInstrCategory targetCategory;    // CATEGORY: 目标指令类别
    std::string targetResourceType;     // RESOURCE_CHANGE: 资源类型 "XN"/"GSA"/"CKE"/"MS"
    uint16_t targetResourceId;          // RESOURCE_CHANGE: 资源 ID
    std::string conditionExpr;          // CONDITION: 条件表达式（如 "XN[5] == 0x1000"）
    uint16_t loopRound;                 // ITERATION: 目标 Loop 迭代轮次
    uint32_t targetExecRound;           // ROUND: 目标外层调度轮次号

    // === 触发计数 ===
    uint32_t hitCount;                  // 已命中次数
    uint32_t skipCount;                 // 前 N 次命中跳过（用于 "在第 3 次循环时断点"）
};

// 断点命中信息
struct CcuBreakpointHit {
    uint32_t bpId;                      // 命中的断点 ID
    uint32_t globalSeqId;               // 当前指令的全局执行序号
    uint32_t execRound;                 // 当前外层调度轮次
    uint16_t instrId;                   // 当前指令 ID
    std::string instrName;              // 当前指令名称
    int32_t rankId;                     // 所在 rank
    int32_t dieId;                      // 所在 die
    uint32_t sqeTaskId;                 // 所属 SQE 任务 ID（引用 sqeTaskRegistry）
    CcuResourceSnapshot snapshot;       // 命中时的资源快照
    CcuExecutionContext context;        // 命中时的执行上下文
};
```

**断点与 SQE 的关系说明**：

断点只关注"在哪个 CCU 的哪条指令处停下"，不涉及 SQE。SQE 信息通过 `sqeTaskRegistry` 注册表管理，每条 `CcuTraceEntry` 通过 `sqeTaskId` 引用对应的 SQE 任务。

当用户回放跳转到某条指令时：
1. **指令细节面板**的 SQE 信息区自动展示该指令所属 SQE 的 missionId、args、指令范围等
2. **SQE 注册表面板**（左侧栏）自动高亮对应的 SQE 条目
3. 同一 CCU 上的多个 SQE 共享同一个 CcuSimulator 实例（`simulatorPtr` 相同），SQE 之间通过 `(sqeTaskId, missionId)` 区分

### 8.3 方案一：前端回放断点（Offline Breakpoint）

**原理**：纯前端实现，不涉及后端修改。用户在前端指令列表中点击某行设置断点标记，自动播放模式下到达该指令时暂停。

**实现方式**：

```javascript
// CcuTracePlaybackControl.vue 中断点管理
const breakpoints = ref(new Set());  // 存储断点的 seqId 集合

function toggleBreakpoint(seqId) {
    if (breakpoints.value.has(seqId)) {
        breakpoints.value.delete(seqId);
    } else {
        breakpoints.value.add(seqId);
    }
}

// 自动播放逻辑
function autoPlay() {
    const timer = setInterval(() => {
        currentStep.value++;
        // 遇到断点则暂停
        if (breakpoints.value.has(currentStep.value)) {
            clearInterval(timer);
            isPlaying.value = false;
            notifyBreakpointHit(currentStep.value);
        }
        // 播放结束
        if (currentStep.value >= entries.length - 1) {
            clearInterval(timer);
            isPlaying.value = false;
        }
    }, playSpeed.value);
}
```

**支持的断点类型**：
- 按 seqId 断点（最常用：点击指令行设置）
- 按指令名称断点（如 "所有 ReduceAdd 指令处断点"）
- 按指令类别断点（如 "所有 Trans 类指令处断点"）
- 按资源变更断点（如 "Xn[5] 被修改的指令处断点"，需扫描 delta 数据）

**前端断点 UI 交互**：

```
指令列表面板：
  seqId | instrId | 名称        | 类型   | 断点 | 状态
  0     | 0       | LoadImd     | Load   |      | ✓
  1     | 1       | TransMM     | Trans  | 🔴   | ✓     ← 点击行号区域切换断点
  2     | 2       | SetCke      | Ctrl   |      | ✓
  ▶ 3   | 3       | ReduceAdd   | Reduce |      | ← 当前（被断点命中后停在此处）
  4     | 4       | Loop        | Ctrl   |      |
  ...
```

### 8.4 方案二：后端执行断点（Live Breakpoint）

**原理**：在 CCU 指令实际执行过程中，当执行到断点指定的指令时，暂停模拟器执行线程，通过 WebSocket 通知前端，前端展示当前状态。用户确认后通过 WebSocket 发送继续指令，恢复执行。目前方案不实现方案二。

#### 8.4.1 架构

```
┌──────────────────┐      WebSocket        ┌─────────────────────────┐
│   前端 (Vue)      │ ◄──────────────────► │  后端 Debug Server       │
│                  │   /ws/ccu-debug       │  (嵌入在 hccl-vm 进程中) │
│  断点管理面板     │                       │                         │
│  实时状态查看     │                       │  CcuBreakpointManager   │
│  Continue/Step   │                       │    ├── 断点配置表        │
│  按钮            │                       │    ├── 条件求值器        │
└──────────────────┘                       │    └── 状态快照采集      │
                                           │                         │
                                           │  CcuSimulator           │
                                           │    └── ExecuteInstr()   │
                                           │        └── CheckBreakpoint() │
                                           │            └── 命中? → 暂停线程 │
                                           └─────────────────────────┘
```

#### 8.4.2 CcuBreakpointManager

```cpp
class CcuBreakpointManager {
public:
    static CcuBreakpointManager& GetInstance();

    // 断点管理
    uint32_t AddBreakpoint(const CcuBreakpointConfig& config);
    void RemoveBreakpoint(uint32_t bpId);
    void EnableBreakpoint(uint32_t bpId, bool enabled);
    void ClearAllBreakpoints();
    std::vector<CcuBreakpointConfig> GetAllBreakpoints() const;

    // 执行时检查（在 ExecuteInstr 中调用）
    // 返回 true 表示需要暂停（命中断点）
    bool CheckBreakpoint(int rankId, int dieId, uint16_t instrId,
                         uint32_t seqId, const CcuExecutionContext& ctx);

    // 暂停与恢复控制
    void RequestPause();          // 外部请求暂停（用户点击暂停按钮）
    void RequestContinue();       // 请求继续执行
    void RequestStepOver();       // 请求单步执行一条指令后暂停
    void RequestStepIntoLoop();   // 请求进入 Loop 内部单步
    bool IsPaused() const;

    // 命中信息（供 Debug Server 读取后发送给前端）
    CcuBreakpointHit GetLastHitInfo() const;

private:
    // 条件表达式求值
    bool EvaluateCondition(const std::string& expr, int rankId, int dieId);

    std::mutex bpMutex_;
    std::map<uint32_t, CcuBreakpointConfig> breakpoints_;
    std::atomic<bool> paused_{false};
    std::atomic<bool> stepMode_{false};        // 单步模式
    std::condition_variable pauseCV_;           // 用于线程暂停/恢复
    std::mutex pauseMutex_;
    CcuBreakpointHit lastHitInfo_;
    uint32_t nextBpId_{1};
};
```

#### 8.4.3 在 CcuSimulator 中集成断点检查

```cpp
bool CcuSimulator::ExecuteInstr(uint16_t curInstrId)
{
    auto& bpMgr = CcuBreakpointManager::GetInstance();
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    auto instrData = ccuResMgr.GetInstrData(rankId_, dieId_);

    // ① 断点检查（执行前）
    if (bpMgr.IsEnabled() && bpMgr.CheckBreakpoint(
            rankId_, dieId_, curInstrId, seqId_, GetExecContext())) {
        // 命中断点 → 暂停线程，等待用户操作
        // CheckBreakpoint 内部会采集快照并填充 lastHitInfo_
        // Debug Server 通过 WebSocket 通知前端
        bpMgr.WaitWhilePaused();  // 阻塞直到用户发送 Continue/Step
    }

    auto executor = CcuExecutorFactory::MakeCcuExecutorInstance(...);
    executor->Parser();
    executor->Run();

    // ② 单步模式：执行完一条后暂停
    if (bpMgr.IsStepMode()) {
        bpMgr.RequestPause();
        bpMgr.WaitWhilePaused();
    }

    // ③ trace 采集（如有）
    if (CcuTraceCollector::IsEnabled()) {
        // ... 原有 trace 采集逻辑 ...
    }

    UpdateLoopStatus();
    return true;
}
```

#### 8.4.4 线程暂停/恢复机制

```cpp
// CcuBreakpointManager 中的线程同步
void CcuBreakpointManager::WaitWhilePaused() {
    std::unique_lock<std::mutex> lock(pauseMutex_);
    pauseCV_.wait(lock, [this] { return !paused_.load(); });
}

void CcuBreakpointManager::RequestPause() {
    paused_.store(true);
}

void CcuBreakpointManager::RequestContinue() {
    stepMode_.store(false);
    paused_.store(false);
    pauseCV_.notify_all();
}

void CcuBreakpointManager::RequestStepOver() {
    stepMode_.store(true);
    paused_.store(false);
    pauseCV_.notify_all();  // 释放执行线程，执行一条后再次暂停
}
```

#### 8.4.5 WebSocket Debug Server 协议

后端嵌入一个轻量 WebSocket 服务（仅在 `HCCLVM_ENABLE_CCU_DEBUG=1` 时启动），与前端实时通信：

```
消息方向          消息类型                  内容
─────────────────────────────────────────────────────────
Server → Client   breakpoint_hit          { bpId, seqId, instrId, instrName,
                                            rankId, dieId, snapshot, context }
Server → Client   execution_paused        { reason: "breakpoint"|"step"|"user",
                                            currentSeqId, instrId }
Server → Client   execution_resumed       { }
Server → Client   resource_snapshot       { rankId, dieId, snapshot }

Client → Server   set_breakpoint          { CcuBreakpointConfig }
Client → Server   remove_breakpoint       { bpId }
Client → Server   continue                { }
Client → Server   step_over               { }
Client → Server   pause                   { }
Client → Server   get_snapshot            { rankId, dieId }
Client → Server   get_instr_info          { rankId, dieId, instrId }
```

#### 8.4.6 条件断点表达式求值

支持简单的资源条件表达式，用于条件断点：

```
语法示例：
  XN[5] == 0x1000                 // Xn 寄存器 5 的值等于 0x1000
  GSA[3] > 0x7F0000               // GSA 寄存器 3 大于指定值
  CKE[0] & 0x0001 != 0            // CKE 信号 0 的第 0 位为 1
  MS[10][0x100:0x10] == 0xFF      // MS 10 偏移 0x100 处 16 字节等于指定值
  XN[5] == 0x1000 && GSA[3] > 0  // 组合条件
```

实现方式：轻量递归下降解析器，直接读取 `CcuResourceManager` 中的资源值求值。

### 8.5 前端断点面板 UI 设计

```
┌─────────────────────────────────────────────────────────────────────────┐
│  断点管理面板                                                             │
├─────────────────────────────────────────────────────────────────────────┤
│  [+ 添加断点]  [清除全部]  [全部启用 ✓]                                   │
│                                                                         │
│  ┌──────┬──────────────────────────────────┬──────┬────────┬──────────┐ │
│  │ 状态 │ 断点条件                          │ 命中  │ oneShot│ 操作    │ │
│  ├──────┼──────────────────────────────────┼──────┼────────┼──────────┤ │
│  │ ✓   │ instrId == 42 @ Rank0/Die0       │ 0    │        │ [✏][🗑] │ │
│  │ ✓   │ name == "ReduceAdd" @ All        │ 3    │        │ [✏][🗑] │ │
│  │     │ XN[5] == 0x1000 @ Rank0/Die1     │ 0    │ ✓     │ [✏][🗑] │ │
│  │ ✓   │ category == Trans @ Rank0/Die0   │ 12   │        │ [✏][🗑] │ │
│  │ ✓   │ loopRound == 3 @ Rank0/Die0      │ 1    │        │ [✏][🗑] │ │
│  └──────┴──────────────────────────────────┴──────┴────────┴──────────┘ │
│                                                                         │
│  ┌─── 添加断点对话框 ───────────────────────────────────────────────┐   │
│  │  断点类型: [指令序号 ▼]                                           │   │
│  │  Rank: [0 ▼]  Die: [0 ▼]  InstrId: [42]                         │   │
│  │  跳过次数: [0]   ☐ 一次性断点                                     │   │
│  │                                              [取消]  [确定]       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.6 两种模式的协作流程

```
用户操作流程：

阶段1：录制（可选，使用后端断点）
  ① 设置后端断点：instrId=42, Rank0/Die0
  ② 启动 hccl-vm 执行算子
  ③ CCU 执行到指令 42 → 自动暂停
  ④ 前端显示实时状态：资源快照、指令细节
  ⑤ 用户审查后点击 [Continue]
  ⑥ 执行完成，trace 数据自动保存

阶段2：回放分析（使用前端断点）
  ⑦ 前端加载 trace 数据
  ⑧ 在指令列表中设置前端断点（感兴趣的指令）
  ⑨ 点击 [▶ 自动播放]
  ⑩ 播放到断点指令时自动暂停
  ⑪ 用户查看资源变化、指令细节
  ⑫ 点击 [▶ 继续播放] 或 [⏭ 跳到下一断点]
```

### 8.7 文件组织（断点功能新增）

```
src/plugin/solver/virtual_runtime/ccu_executor/trace/
├── ccu_trace_types.h              // (已有) trace 数据结构
├── ccu_trace_collector.h/cc       // (已有) trace 采集器
├── ccu_trace_serializer.h/cc      // (已有) JSON 序列化
├── ccu_breakpoint_types.h         // (新增) 断点数据结构
├── ccu_breakpoint_manager.h       // (新增) 断点管理器声明
├── ccu_breakpoint_manager.cc      // (新增) 断点管理器实现
├── ccu_condition_evaluator.h      // (新增) 条件表达式求值器
├── ccu_condition_evaluator.cc     // (新增)
├── ccu_debug_server.h             // (新增) WebSocket 调试服务
└── ccu_debug_server.cc            // (新增)

前端新增：
src/plugin/solver/virtual_runtime/insight/frontendV3/src/
├── components/ccu-trace/
│   ├── CcuTraceBreakpointPanel.vue    // (新增) 断点管理面板
│   ├── CcuTraceBreakpointDialog.vue   // (新增) 添加/编辑断点对话框
│   └── CcuTraceLiveControl.vue        // (新增) 实时调试控制(Continue/Step/Pause)
├── composables/
│   └── useCcuDebugWebSocket.js        // (新增) WebSocket 通信 composable
└── utils/
    └── ccuBreakpointUtils.js          // (新增) 断点前端工具函数
```

---

## 9. 高级分析功能

### 9.1 搜索与多维过滤

在大量 trace 数据中快速定位问题指令，提供以下搜索和过滤能力：

#### 9.1.1 文本搜索

在指令列表顶部提供搜索框，支持实时过滤：
- 按指令名称搜索（如输入 "TransLocMem" 匹配所有本地内存搬运指令）
- 按指令描述搜索（匹配 `Describe()` 输出中的关键字）
- 按 seqId 搜索（如输入 "#42" 跳转到第 42 条指令）

#### 9.1.2 多维筛选器

提供下拉筛选器，支持多维度组合过滤：

| 筛选维度 | 选项 | 说明 |
|---------|------|------|
| **指令类别** | Load / Trans / Control / Reduce / 全部 | 按 4 大类过滤 |
| **执行状态** | 正常 / 失败 / CKE等待 / 全部 | 按执行结果过滤 |
| **Loop 上下文** | 非 Loop / Loop 首轮 / Loop 中间轮 / Loop 末轮 / 全部 | 按 Loop 位置过滤 |
| **资源变更类型** | 修改了 XN / 修改了 GSA / 修改了 CKE / 修改了 MS / 跨 CCU / 全部 | 按变更的资源类型过滤 |
| **CKE Wait** | 有 Wait / 无 Wait / 全部 | 是否包含 CKE 等待 |
| **SQE 范围** | SQE 0 / SQE 1 / ... / 全部 | 按 SQE session 过滤 |

筛选器状态可保存为"筛选预设"，方便反复使用。

#### 9.1.3 搜索结果统计

筛选后在列表底部显示统计信息：
```
显示 42 / 1280 条指令  |  失败: 2  |  CKE等待: 5  |  跨CCU: 8
```

### 9.2 数据依赖追溯

帮助用户理解"某个资源的值是从哪里来的"，建立指令间的数据流关系。

#### 9.2.1 追溯来源（Trace Backward）

用户在资源变化面板中点击某个变更值，自动定位到上一次修改该资源的指令：

```
用户操作：点击 "Xn[5]: 0x400 → 0x1000" 旁的 [追溯] 按钮

系统行为：
  1. 从当前 seqId 向前搜索，找到最后一条修改 Xn[5] 的指令
  2. 指令列表自动滚动到该指令并高亮
  3. 资源变化面板联动更新
  4. 在两个指令之间绘制虚线箭头（可选，在指令列表中显示）
```

实现原理：
```javascript
// 构建资源写入索引（在数据加载时预处理）
const resourceWriteIndex = {
    'XN:5': [
        { seqId: 0, valueAfter: '0x400' },
        { seqId: 15, valueAfter: '0x1000' },
        { seqId: 42, valueAfter: '0x2000' },
    ],
    'GSA:3': [ ... ],
    'CKE:0': [ ... ],
};

// 追溯：找到当前 seqId 之前最后一个写入该资源的 entry
function traceBackward(resourceType, resourceId, currentSeqId) {
    const key = `${resourceType}:${resourceId}`;
    const writes = resourceWriteIndex[key] || [];
    return writes.filter(w => w.seqId < currentSeqId).pop();
}
```

#### 9.2.2 追溯影响（Trace Forward）

反向操作：查找当前指令修改的资源被后续哪些指令读取：

```
用户操作：点击 "Xn[5]: 0 → 0x400" 旁的 [影响] 按钮

系统行为：
  1. 从当前 seqId 向后搜索，找到所有读取 Xn[5] 的指令
  2. 在指令列表中高亮这些指令（黄色背景）
  3. 统计：Xn[5]=0x400 被后续 N 条指令读取，直到 seqId=M 被覆写
```

#### 9.2.3 CKE 同步关系可视化

CKE 是 CCU 指令间同步的核心机制。提供专门的 CKE 关系视图：

```
CKE[3] 同步关系时间线：

seqId=10  SetCke   CKE[3]: 0→1  (设置)
  ↓ (等待 CKE[3]&0x1 == 0x1)
seqId=25  TransMM  waitCKE[3:0001] ✓ 通过
  ↓
seqId=26  ClearCke CKE[3]: 1→0  (清除)
  ↓ (等待 CKE[3]&0x1 == 0x1)
seqId=40  TransMM  waitCKE[3:0001] ⚠ 自旋 14 轮后通过
```

### 9.3 Trace Diff 对比

对比两次运行的 trace 数据，快速定位差异点。典型场景：同一算子在正确/错误场景下的对比。

#### 9.3.1 对比模式

```
┌────────────────────────────────────────────────────────────────────────┐
│  Trace Diff 模式                                                       │
├──────────────────────────┬─────────────────────────────────────────────┤
│  Run A (正确)             │  Run B (错误)                              │
│  rank0_die0_trace.json   │  rank0_die0_trace.json                     │
│                          │                                             │
│  seqId | 名称    | 状态   │  seqId | 名称    | 状态   | Diff           │
│  0     | LoadImd | ✓     │  0     | LoadImd | ✓     | = (一致)        │
│  1     | TransMM | ✓     │  1     | TransMM | ✓     | ✗ delta不同     │
│  2     | SetCke  | ✓     │  2     | SetCke  | ✓     | =               │
│  3     | Reduce  | ✓     │  3     | Reduce  | FAIL  | ✗ B失败         │
│  ...                    │  ...                                       │
│                          │                                             │
│  差异摘要:                                       │
│  首个分歧点: seqId=1 (TransLocMemToLocMem)                             │
│  差异类型: Xn[5] 值不同 (A:0x1000 vs B:0x0)                           │
│  总差异数: 3 / 128                                                    │
└──────────────────────────┴─────────────────────────────────────────────┘
```

#### 9.3.2 Diff 检测规则

| 对比项 | 一致条件 | 差异标记 |
|-------|---------|---------|
| 指令序列 | 相同 instrId + instrName | 指令缺失/多出 → 红色 |
| 资源 Delta | 相同的 xnChanges/gsaChanges/ckeChanges | 值不同 → 黄色 |
| 执行状态 | 相同 execState | A 成功 B 失败 → 红色 |
| CKE Wait | 相同 waitRetryCount（容差 ±1） | 等待次数差异大 → 橙色 |
| 跨 CCU 变更 | 相同的 remoteCkeChanges | 远端操作不一致 → 紫色 |

#### 9.3.3 Diff 数据来源

Diff 数据由前端在加载两个 trace 文件后实时计算，不需要后端支持：

```javascript
// 前端 Diff 计算
function computeTraceDiff(traceA, traceB) {
    const diffs = [];
    const maxLen = Math.max(traceA.entries.length, traceB.entries.length);

    for (let i = 0; i < maxLen; i++) {
        const entryA = traceA.entries[i];
        const entryB = traceB.entries[i];

        if (!entryA) { diffs.push({ type: 'missing_in_a', entryB }); continue; }
        if (!entryB) { diffs.push({ type: 'missing_in_b', entryA }); continue; }

        if (entryA.instrId !== entryB.instrId ||
            entryA.instrName !== entryB.instrName) {
            diffs.push({ type: 'instr_mismatch', entryA, entryB });
        }

        if (!isEqual(entryA.resourceDelta, entryB.resourceDelta)) {
            diffs.push({ type: 'delta_diff', entryA, entryB,
                         deltaDiff: diffDeltas(entryA.resourceDelta, entryB.resourceDelta) });
        }

        if (entryA.execState !== entryB.execState) {
            diffs.push({ type: 'state_diff', entryA, entryB });
        }
    }
    return diffs;
}
```

---

## 10. 版本兼容策略

### 10.1 Trace 格式版本管理

`CcuRunMetadata.traceFormatVersion` 标识 trace 数据格式的版本号。版本号遵循语义化版本：

| 版本 | 含义 | 变更示例 |
|------|------|---------|
| 1.0 | 初始版本 | 当前设计 |
| 1.x | 向后兼容扩展 | 新增可选字段（如新指令类型的 Detail） |
| 2.0 | 不兼容变更 | 数据结构重组（如 delta 格式变更） |

### 10.2 CCU 微码版本兼容

CCU V1 和 V2 的资源规格不同：

| 资源 | V1 规格 | V2 规格 |
|------|--------|--------|
| XN | 3072 | 4096 |
| GSA | 3072 | 4096 |
| CKE | 1024 | 1024 |
| MS | 1536 | 1536 |
| 指令类型 | 4大类 ~20种 | 4大类 ~35种 |

**兼容策略**：
1. `CcuRunMetadata.ccuVersion` 标识本次 trace 的微码版本
2. 前端根据 ccuVersion 动态调整资源显示范围（V1 显示 3072 个 XN，V2 显示 4096 个）
3. 前端根据 ccuVersion 显示对应的指令类型名称（V2 有 Add/Sub/Mul 等算术指令，V1 没有）
4. Trace Diff 支持跨版本对比（V1 trace vs V2 trace），但标记资源规格差异

### 10.3 前端向后兼容

前端代码通过版本分发器处理不同版本的 trace 数据：

```javascript
function loadTrace(data) {
    const version = data.runMetadata?.traceFormatVersion || 1;

    switch (version) {
        case 1:
            return parseTraceV1(data);
        case 2:
            return parseTraceV2(data);
        default:
            console.warn(`Unknown trace version ${version}, trying V1 parser`);
            return parseTraceV1(data);
    }
}
```

对于向后兼容的扩展（1.x），新增字段提供默认值：

```javascript
function parseTraceEntry(raw) {
    return {
        seqId: raw.seqId,
        instrId: raw.instrId,
        // ... 必要字段
        waitInfo: raw.waitInfo || { hadWait: false, waitRetryCount: 0 },  // 默认值
        errorInfo: raw.errorInfo || { hasError: false },                   // 默认值
        crossCcuChanges: raw.crossCcuChanges || { hasCrossCcuChange: false }, // 默认值
    };
}
```

---

## 11. 异常场景下的 Trace 可靠性保障

### 11.1 问题分析

当前 trace dump 流程存在**致命缺陷**：trace 数据在运行期间全部驻留在内存中（`m_traceRun.globalEntries` 的 `std::vector`），仅在 `SequentialExecutor::Execute()` 的**最末尾**才一次性序列化写入磁盘。

```
Execute() {
    // trace 初始化
    while (HasTask()) {       // ← 如果这里崩溃
        ExecuteOneTask();     //   ← 或者这里崩溃
    }
    // ===== Trace 落盘 =====  //   ← 永远不会执行到
    traceCollector.EndRun();
    DumpToFile(...);
}
```

**三种异常场景分析**：

| 场景 | 触发位置 | trace 能否落盘 | 原因 |
|------|---------|:---:|------|
| **段错误 (SIGSEGV)** | 指令 `Process()` 中指针越界 | **不能** | 进程直接终止，不会执行到 dump 代码 |
| **越界访问** | `GetXnValue()` 等数组操作 | **不能** | 触发 crash/UB 时同上 |
| **C++ 异常 (throw)** | 指令执行中抛出异常 | **不能** | 调用链无 try/catch，穿透到顶层触发 `std::terminate` |

**根本原因（4个缺陷）**：

1. **trace 数据全在内存**：所有 `CcuTraceEntry` 存储在 `m_traceRun.globalEntries`（`std::vector`），运行期间从不写磁盘
2. **无信号处理机制**：全局搜索 `signal`/`sigaction`/`SIGSEGV`/`SIGABRT`/`atexit`/`set_terminate` 均无结果，崩溃时无机会触发 dump
3. **无异常捕获**：`Execute()` → `ExecuteOneTask()` → `TaskCcuGraph()` → `CcuSimulator::Execute()` → `ExecuteInstr()` 整条链路没有 try/catch
4. **无增量落盘/checkpoint**：没有运行期间定期写入磁盘的机制

### 11.2 解决方案

采用 **方案 1（信号处理）+ 方案 3（增量落盘）** 的组合策略。

#### 11.2.1 方案 1：注册信号处理函数（针对段错误/abort）

在 `SequentialExecutor::Execute()` 中注册 SIGSEGV/SIGABRT 信号处理器，进程崩溃时紧急 dump 已采集的 trace 数据：

```cpp
#include <signal.h>
#include <execinfo.h>

static const char* g_crashDumpPath = nullptr;

static void CrashDumpHandler(int sig) {
    // 1. 恢复默认信号处理，防止 dump 过程中再次触发信号导致无限递归
    signal(sig, SIG_DFL);

    // 2. 紧急 dump 已采集的 trace 数据
    auto& collector = CcuTrace::CcuTraceCollector::GetInstance();
    if (collector.IsEnabled() && g_crashDumpPath != nullptr) {
        collector.EndRun();
        auto traceRun = collector.GetTraceRun();
        CcuTrace::CcuTraceSerializer::DumpToFile(traceRun, g_crashDumpPath);
    }

    // 3. 打印调用栈辅助定位
    void* frames[64];
    int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);

    // 4. 重新触发默认处理（core dump）
    raise(sig);
}

// 在 Execute() 中注册
g_crashDumpPath = crashOutputPath.c_str();
signal(SIGSEGV, CrashDumpHandler);
signal(SIGABRT, CrashDumpHandler);
```

**async-signal-safety 说明**：
严格来说，signal handler 中使用 `std::ostringstream`（DumpToFile 内部用到）不是 async-signal-safe 的。但实践中通常可行，因为崩溃时不太可能恰好在 malloc 内部。即使 dump 失败，进程也不会比原来更糟（原来就是 100% 丢失）。如需严格安全，可后续改为 `write()` 系统调用直接输出二进制格式。

#### 11.2.2 方案 3：增量落盘（定期 checkpoint）

在主调度循环中定期将已采集的 trace 数据序列化写入磁盘，即使进程被 `kill -9` 或 OOM killer 杀掉，磁盘上仍有最近一次 checkpoint 的 trace 数据。

```cpp
const uint32_t TRACE_FLUSH_INTERVAL = 100;  // 每 100 条指令 flush 一次
uint32_t instrCountSinceFlush = 0;

while (HasTask()) {
    // ... 执行逻辑 ...
    
    // 增量落盘
    instrCountSinceFlush++;
    if (traceCollector.IsEnabled() && instrCountSinceFlush >= TRACE_FLUSH_INTERVAL) {
        traceCollector.IncrementalDump();
        instrCountSinceFlush = 0;
    }
}
```

**CcuTraceCollector 新增 `IncrementalDump()` 接口**：

```cpp
void CcuTraceCollector::IncrementalDump() {
    if (!m_enabled || m_outputPath.empty()) return;
    auto traceRun = GetTraceRun();  // 获取当前快照（含锁保护）
    CcuTraceSerializer::DumpToFile(traceRun, m_outputPath);
}
```

**性能影响评估**：
- 每 100 条指令 flush 一次，序列化开销约 1-5ms（取决于已采集数据量）
- 总指令数通常 < 10000，flush 次数 < 100，总额外开销 < 500ms
- 相对于指令执行本身的开销，影响可接受
- 可通过 `HCCLVM_TRACE_FLUSH_INTERVAL` 环境变量调整间隔

#### 11.2.3 覆盖范围总结

| 层 | 机制 | 覆盖场景 |
|----|------|---------|
| **信号处理** (方案 1) | `signal(SIGSEGV/SIGABRT, handler)` | 段错误、abort、栈溢出 |
| **增量落盘** (方案 3) | 定期 flush | 所有场景（含 kill -9、OOM、断电） |

两者组合确保：
- **正常退出**：最终 dump 完整 trace
- **段错误/abort**：信号处理器紧急 dump（文件名标记 `_crash`）
- **kill -9 / OOM / 断电**：磁盘上有最近一次增量 checkpoint 的 trace

### 11.3 文件命名规则

| 场景 | 文件名 | 说明 |
|------|--------|------|
| 正常完成 | `hccl_trace_output.json` | 完整 trace |
| 增量 checkpoint | `hccl_trace_output.json` | 覆盖写入，始终保留最新状态 |
| 崩溃 dump | `hccl_trace_crash_dump.json` | 崩溃时刻的 trace 快照 |
