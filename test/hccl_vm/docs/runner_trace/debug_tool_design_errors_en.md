# CCU Executor Debug 工具设计文档 — 可行性审视与优化报告

基于 `src/plugin/solver/virtual_runtime/ccu_executor/` 代码与 `runner_debug/debug_tool_design_doc.md` 设计方案的全面审视。

---

## 一、可行性问题（必须修正）

### P0-1: CKE Wait 流程与插桩设计不匹配

**严重程度**: P0（方案可行性错误）

**问题描述**:

文档 4.1 节的插桩伪代码假设 `waitCKE_` 标志在 `ExecuteInstr()` 返回之前就能检测，并在之后计算 `snapshotBefore → snapshotAfter` 的 delta。但实际代码流程：

```
WaitCkeProcess() → CKE 不满足 → ccuSimulator_->SetWaitCKEFlag(true) → return (不调用 Process())
→ UpdateLoopStatus() 返回 false → ExecuteInstr() 返回 false → Execute() 返回 false
```

关键：CKE 不满足时，`Run()` 内部的 `WaitCkeProcess` 发现 CKE 不满足就设置 waitCKE 标志后直接 return，**不调用 Process()，不修改任何资源**。因此：
- 执行前快照 = 执行后快照（delta 为空）
- 不应该捕获 snapshotAfter 和计算 delta

**代码依据**:

`CcuExecutorBase.cc` 第 162-172 行：
```cpp
if ((waitCKE & waitCKEMask) == waitCKEMask) {
    Process(ccuResMgr);      // CKE 满足 → 执行 Process
} else {
    ccuSimulator_->SetWaitCKEFlag(true);  // CKE 不满足 → 设置标志，不执行 Process
    return;
}
```

**修正建议**:

waitCKE 不满足时，只记录关键自旋字段，不计算 delta，不创建完整 trace entry：
```cpp
// waitCKE 时只记录自旋信息
CcuTraceCollector::RecordWaitSpin(rankId_, dieId_, curInstrId,
                                    waitCKEId, waitCKEMask, actualCKEValue);
// 不捕获 snapshotAfter，不计算 delta
// 不创建完整 CcuTraceEntry
return false;
```

直到 CKE 通过后才创建完整 entry（合并之前的自旋信息）。

---

### P0-2: globalSeqId/execRound 的维护位置应在 SequentialExecutor

**严重程度**: P0（方案可行性错误）

**问题描述**:

文档假设 `CcuTraceCollector` 在 `CcuSimulator::ExecuteInstr()` 内部维护 `globalSeqId/execRound/currentSqeTaskId`。但 `CcuSimulator` 只能看到本 CCU 的执行，无法感知全局调度轮次和 CCU 切换。

`CcuSimulator::ExecuteInstr()` 在单个 CCU 内执行，不知道：
- 当前外层 `while(HasTask())` 的第几轮（`execRound`）
- 全局执行序号（`globalSeqId`）
- 当前执行的是哪个 SQE 任务（`sqeTaskId`）

这些信息只有在 `SequentialExecutor::Execute()` 和 `TaskCcuGraph()` 的上层调用链中才能获取。

**代码依据**:

`hccl_task_sequential_execute.cc` 第 51-72 行 — 全局调度循环：
```cpp
while (HasTask()) {
    uint32_t rankId = 0;
    for (auto& rankTasks : allRankTaskQueues_) {  // 遍历所有 rank
        for (auto& streamTasks : rankTasks) {     // 遍历每个 stream
            while (!streamTasks.empty()) {
                auto task = streamTasks.front();
                auto ret = ExecuteOneTask(task);   // ← globalSeqId 在此递增
                if (ret == HCCL_SIM_VRT_HOLD_CMD) break;
                streamTasks.pop();
            }
        }
    }
}
```

`hccl_task_thread.cc` 第 135-148 行 — TaskCcuGraph：
```cpp
auto simulator = ccuResMgr.InitSimulator(rankId, dieId, instrStartId, endInstrId, instCnt);
// sqeTaskId 在此可知（从 task 参数获取）
if (simulator->Execute() == false) {
    return HCCL_SIM_VRT_HOLD_CMD;
}
```

**修正建议**:

全局上下文由 `SequentialExecutor` 管理，通过参数传递：
```cpp
// SequentialExecutor::Execute() 中
while (HasTask()) {
    uint32_t execRound = 0;
    for (rankTasks : allRankTaskQueues_) {
        for (streamTasks : rankTasks) {
            while (!streamTasks.empty()) {
                CcuTraceCollector::BeginGlobalStep(execRound);
                // globalSeqId 在 ExecuteOneTask 内递增
                auto ret = ExecuteOneTask(task);
                ...
            }
        }
    }
    execRound++;
}
```

`TaskCcuGraph()` 中将 sqeTaskId 传递给 `CcuSimulator`：
```cpp
uint32_t sqeTaskId = CcuTraceCollector::RegisterSqeTask(rankId, dieId, task);
CcuSimulator::SetCurrentSqeTaskId(sqeTaskId);  // Simulator 内部使用
```

---

### P0-3: 全量 Snapshot 采集性能开销过大

**严重程度**: P0（性能瓶颈）

**问题描述**:

文档 4.1 中每条指令前后各做一次 `CaptureResourceSnapshot()`（全量读取所有资源），开销极大：

- 单个 snapshot：4096×8(XN) + 4096×8(GSA) + 1024×2(CKE) ≈ 67KB
- 每条指令前后各一次 ≈ 134KB
- 10000 条指令 ≈ 1.34GB（仅 snapshot 数据）

文档 4.5 提到了增量采集优化，但 4.1 的伪代码仍然是全量 snapshot + diff 计算的流程，与优化方案矛盾。

**业界做法参考**:

- **rr debugger**: 检查点（checkpoint）+ 增量记录模式，只在关键节点做全量快照
- **Chrome DevTools Timeline**: 事件流格式，每个 event 只包含变化的 args
- **Perfetto (Android trace)**: protobuf 二进制增量格式

**修正建议**:

改为 ResourceManager 拦截变更 + 初始快照 + 定期 checkpoint：

1. **初始快照**：Run 开始时做一次全量（存储在 `CcuIdentity.initialSnapshot` 中）
2. **变更拦截**：在 `CcuResourceManager::UpdateXnValue/UpdateGsaValue/UpdateCkeValue` 等方法中直接拦截变更，记录 delta：
```cpp
void CcuResourceManager::UpdateXnValue(int rankId, int dieId, uint16_t xnId, uint64_t value) {
    uint64_t oldValue = GetXnValue(rankId, dieId, xnId);
    if (oldValue != value) {
        CcuTraceCollector::RecordXnDelta(rankId, dieId, xnId, oldValue, value);
    }
    xnData_[rankId][dieId][xnId] = value;
}
```
3. **定期 checkpoint**：每 N 条指令做一次全量快照（N 可配置，默认 1000），用于快速恢复中间状态
4. **恢复任意点状态**：从最近的 checkpoint + 累加后续 delta

---

## 二、设计优化点（建议改进）

### P1-1: Loop 内偏移公式不完整

**严重程度**: P1（信息缺失）

**问题描述**:

代码中 `UpdateAddress()` 的偏移计算比文档中 `CcuExecutionContext.gsaOffset` 更复杂：
```cpp
uint64_t CcuExecutorBase::UpdateAddress(uint64_t addr, uint16_t addrExpandCoef) {
    return addr + ((ccuSimulator_->GetLoopExtendNum() * ccuSimulator_->GetGSAOffset()) << addrExpandCoef)
        + ((ccuSimulator_->GetCurLoopCnt() * ccuSimulator_->GetLoopIterStepGSA()) << addrExpandCoef);
}
```

涉及 5 个参数：`addr`（原始地址）、`addrExpandCoef`（扩展系数）、`loopExtendNum`、`gsaOffset`、`curLoopCnt`、`loopIterStepGSA`。

当前 `CcuExecutionContext` 只有一个简单的 `gsaOffset` 字段，无法完整恢复 Loop 内 GSA 地址偏移的计算过程。

**修正建议**:

`CcuExecutionContext` 增加完整的偏移参数：
```cpp
struct CcuExecutionContext {
    bool inLoop;
    uint16_t loopRound;          // 当前迭代轮次
    uint16_t loopExtendIndex;    // 展开索引

    // Loop 内偏移参数（仅在 inLoop=true 时有效）
    uint64_t gsaAddrOffset;      // = extendIndex * gsaOffset + curLoopCnt * iterStepGSA（计算后的最终偏移）
    uint64_t gsaOffset;          // GSA 基础偏移系数（GetGSAOffset）
    uint64_t iterStepGSA;        // 每轮迭代的 GSA 步长（GetLoopIterStepGSA）
    uint32_t curLoopCnt;         // 当前迭代计数（GetCurLoopCnt）
    uint32_t loopExtendNum;      // 展开数（GetLoopExtendNum）
    uint16_t addrExpandCoef;     // 地址扩展系数

    uint16_t msOffset;           // MS ID 偏移（GetLoopMsOffset）
    uint16_t ckeOffset;          // CKE ID 偏移（GetLoopCKEOffset）
    uint16_t xnIdOffset;         // XN ID 偏移（GetLoopXnIdOffset）
};
```

---

### P1-2: 跨 CCU 变更应在 ResourceManager 中拦截

**严重程度**: P1（采集精度）

**问题描述**:

文档 4.1 中跨 CCU 变更的采集点在 `CcuSimulator::ExecuteInstr()` 中事后调用 `CaptureCrossCcuChanges()`。但实际跨 CCU 操作发生在 Executor 的 `Process()` 方法中（如 `SyncCkeExecutor::Process()` 调用 `SetRmtCKESignal()`）。

事后采集的方式需要对比远端 CCU 的前后状态，这意味着需要在指令执行前后各做一次远端 CCU 的快照——开销大且可能遗漏（远端 CCU 可能被其他代码同时修改）。

**修正建议**:

在 `CcuResourceManager::UpdateCkeValue()` 中直接拦截跨 CCU 变更：
```cpp
void CcuResourceManager::UpdateCkeValue(int rankId, int dieId, uint16_t ckeId, uint16_t value) {
    uint16_t oldValue = GetCkeValue(rankId, dieId, ckeId);
    // 检测是否为跨 CCU 操作
    auto [execRank, execDie] = CcuTraceCollector::GetCurrentExecutingCcu();
    if (rankId != execRank || dieId != execDie) {
        CcuTraceCollector::RecordCrossCcuCkeChange(
            rankId, dieId, ckeId, oldValue, value, execRank, execDie);
    }
    ckeData_[rankId][dieId][ckeId] = value;
}
```

同理适用于 `UpdateXnValue()`（TransXnToRmt 等操作）、`UpdateGsaValue()`、`TransMSToMS()` 等。

---

### P1-3: CcuInstrTraceDetail 多态过度设计

**严重程度**: P1（架构优化）

**问题描述**:

当前设计了 **13 个** `CcuInstrTraceDetail` 子类（Load/Arith/Trans/Sync/Loop/LoopGroup/Cke/Jump/Wait/Fence/Reduce/...），但存在以下问题：

1. **与公共层重复**：`CcuTransTraceDetail` 的 `waitCKEId/setCKEId` 与 `CcuTraceEntry.waitInfo` 重复；`CcuCkeTraceDetail` 的 `ckeId/ckeValueBefore/ckeValueAfter` 与 `resourceDelta.ckeChanges` 重复
2. **子类数量膨胀**：代码中有 30+ 种指令类型（V1 约 20 种 + V2 约 10+ 种），每种都需要一个子类
3. **前端渲染复杂**：需要根据 typeName 分发不同渲染模板
4. **新增指令成本高**：每新增一种指令类型都需要新增一个子类结构体 + CollectTraceDetail 实现

**业界做法参考**:

- **Chrome DevTools Trace Event Format**: 扁平的 `args` 字段（key-value map），不使用多态继承
- **LLVM Execution Trace**: 扁平的"附加信息"字段
- **GDB MI (Machine Interface)**: key-value 属性列表

**修正建议**:

将 `CcuInstrTraceDetail` 简化为扁平结构：
```cpp
struct CcuInstrTraceDetail {
    std::string typeName;                     // 如 "TransLocMemToLocMem"
    std::map<std::string, std::string> args;  // 指令特有参数的 key-value 表
    // 示例:
    //   Trans 类: {"srcAddr": "0x7f0000", "dstAddr": "0x7f1000", "length": "4KB", "channelId": "0"}
    //   Reduce 类: {"op": "Add", "msList": "[0,1,2,3]", "dataType": "FP32", "castEn": "0"}
    //   CKE 类: {"ckeOp": "Set", "ckeId": "3", "ckeMask": "0x0001", "isRemote": "true"}
};
```

优点：
- 不需要为每种指令定义子类
- 新增指令时只需在 `CollectTraceDetail()` 中填充 args
- 前端渲染逻辑统一（key-value 表格）
- `Describe()` 的输出可直接作为 args 来源

---

### P1-4: JSON 序列化格式与大数据量的矛盾

**严重程度**: P1（性能优化）

**问题描述**:

文档 3.4 采用 JSON 作为序列化格式，但 JSON 的弱点：
- 每个 uint64_t 的 XN/GSA 值需要字符串表示（`"0x7f3a0000"`），8 字节变 ~16 字节
- 大量冗余的 key 名称重复（`"xnChanges"` 在每条 entry 中重复出现）
- 10000 条指令的 JSON 文件可达 50MB+
- 前端加载和解析大 JSON 文件耗时严重

**业界做法参考**:

| 工具 | 格式 | 特点 |
|------|------|------|
| Chrome DevTools | JSON 流式（每行一个 event） | 支持逐行解析 |
| Perfetto (Android) | protobuf 二进制 | 前端 WASM 解析，体积小 |
| rr debugger | 自定义二进制 | 高效记录/回放 |
| Intel VTune | 自定义二进制 + SQLite | 按需查询 |

**修正建议**:

分层序列化策略：
- **采集端**：msgpack/flatbuffers 二进制格式，体积小 5-10 倍，写入速度快
- **前端加载**：通过 Worker 将二进制解码为前端索引结构
- **导出/调试**：可选导出为 JSON（便于人类阅读和调试）
- JSON Schema 中 `metadata.rankId/dieId` 应删除（改为 `rankSize/diePerRank`，与全局模型一致）

---

## 三、设计优化点（建议改进，优先级较低）

### P2-1: Describe() 可作为 MVP 的 trace 信息来源

**严重程度**: P2（落地策略）

**问题描述**:

代码中每个 Executor 都实现了 `Describe()` 方法，返回格式化字符串如：
```
"[Simulation Execute] Wait CKE[3:0001], Sync LocCKE[5:00ff] To rmtCKE[8:00ff]
 Use Channel[0], Set CKE[10:0001], clearType[1]"
```

这些字符串已经包含了指令的关键参数信息。当前设计要求每个 Executor 新增 `CollectTraceDetail()` 虚方法并实现，工作量较大（30+ 种指令类型）。

**修正建议**:

分阶段落地策略：

**第一阶段（MVP）**:
- 只采集 `Describe()` 输出 + resource delta
- `instrDescribe` 字段直接使用 `executor->Describe()` 的返回值
- 不实现 `CollectTraceDetail()`，detail 字段为空或仅含 typeName

**第二阶段（增强）**:
- 为高频/关键指令类型实现 `CollectTraceDetail()`（如 Trans、Reduce、SyncCke）
- detail 采用扁平 key-value map（见 P1-3）
- 逐步覆盖所有指令类型

这样可以快速落地 MVP，降低初始开发成本。

---

### P2-2: Notify 等非 CCU 任务不在 trace 中

**严重程度**: P2（信息缺失）

**问题描述**:

`SequentialExecutor::Execute()` 中除了 `CCU_GRAPH` 外，还有 `NOTIFY_RECORD/NOTIFY_WAIT/REDUCE/MEM_CPY/AIV_GRAPH` 等任务类型。`NotifyWait` 也会返回 `HOLD_CMD`，与 CCU 的 waitCKE 有类似的阻塞行为。

当前 trace 只覆盖 `CCU_GRAPH` 类型任务。但 Notify 的 Record/Wait 会影响执行流程：
- `NotifyWait` 阻塞会导致外层循环跳过该 stream
- `NotifyRecord` 解除阻塞后，下一轮该 stream 的 CCU 任务才能继续执行

如果不在 trace 中体现，前端回放时会出现"CCU 指令突然跳到下一轮"的现象，无法解释中间发生了什么。

**修正建议**:

在 `globalEntries[]` 中增加非 CCU 任务类型的简化记录：
```cpp
struct CcuTraceNonCcuEntry {
    uint32_t globalSeqId;
    uint32_t execRound;
    int32_t rankId;
    HccLTaskMetaType taskType;    // NOTIFY_RECORD, NOTIFY_WAIT, REDUCE, MEM_CPY...
    HcclVmResult execResult;      // SUCCESS 或 HOLD_CMD
    std::string description;      // 任务描述（便于前端展示）
};
```

前端在 CCU 切换提示时可显示："Round 1: Rank1 NotifyWait 阻塞，跳过"。

---

### P2-3: JSON Schema 中 metadata 模型不一致

**严重程度**: P2（模型一致性）

**问题描述**:

文档 3.4 的 JSON Schema 示例中 `metadata` 仍包含旧设计的 `rankId`/`dieId` 字段（per-CCU 模型），但 3.3.1 的 `CcuRunMetadata` 已改为 `rankSize`/`diePerRank`（全局模型）。两处不一致。

此外，JSON Schema 示例中使用了 `"seqId"` 而非新设计的 `"globalSeqId"`，缺少 `"execRound"`/`"sqeTaskId"`/`"ccuRegistry"`/`"sqeTaskRegistry"` 等新字段。

**修正建议**:

更新 JSON Schema 示例，与 3.3.1-3.3.2 的数据结构定义保持一致：
- 删除 `metadata.rankId`/`metadata.dieId`
- 增加 `metadata.rankSize`/`metadata.diePerRank`/`metadata.totalExecRounds`
- 增加 `ccuRegistry[]` 和 `sqeTaskRegistry[]`
- entry 中 `seqId` 改为 `globalSeqId`
- 增加 `execRound`/`sqeTaskId`/`rankId`/`dieId` 字段

---

## 四、汇总表

| 优先级 | 编号 | 问题 | 类型 | 修正要点 |
|--------|------|------|------|---------|
| P0 | P0-1 | CKE Wait 时不应计算 delta | 可行性错误 | waitCKE 时只记录关键字段，CKE 通过后才创建完整 entry |
| P0 | P0-2 | globalSeqId/execRound 维护位置错误 | 可行性错误 | 全局上下文由 SequentialExecutor 管理，传参给 Simulator |
| P0 | P0-3 | 全量 snapshot 性能瓶颈 | 性能瓶颈 | ResourceManager 拦截变更 + 初始快照 + 定期 checkpoint |
| P1 | P1-1 | Loop 偏移公式不完整 | 信息缺失 | 补充 iterStepGSA/curLoopCnt/extendNum/addrExpandCoef |
| P1 | P1-2 | 跨 CCU 变更采集时机 | 采集精度 | ResourceManager 内部拦截跨 CCU 变更 |
| P1 | P1-3 | Detail 多态过度设计 | 架构优化 | 简化为扁平 key-value map |
| P1 | P1-4 | JSON 格式不适合大数据量 | 性能优化 | 采集端 msgpack/flatbuffers，前端 WASM 解析 |
| P2 | P2-1 | Describe() 可作为 MVP 来源 | 落地策略 | 第一阶段只做 Describe() + delta |
| P2 | P2-2 | Notify 等非 CCU 任务缺失 | 信息缺失 | 增加简化记录 |
| P2 | P2-3 | JSON Schema 模型不一致 | 模型一致性 | 与 3.3.1-3.3.2 保持一致 |

---

## 五、修正优先级建议

**第一步（必须）**: 修正 P0-1、P0-2、P0-3，否则方案无法落地

**第二步（建议）**: 修正 P1-1、P1-2、P1-3，提高方案质量和实用性

**第三步（可选）**: 修正 P1-4、P2-1、P2-2、P2-3，完善细节和用户体验
