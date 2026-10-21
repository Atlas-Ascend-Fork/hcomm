# CCU Executor Debug Tool Design Document — Feasibility Review and Optimization Report

Comprehensive review based on code in `src/plugin/solver/virtual_runtime/ccu_executor/` and the design proposal in `runner_debug/debug_tool_design_doc.md`.

---

## I. Feasibility Issues (Must Fix)

### P0-1: CKE Wait Flow Does Not Match Instrumentation Design

**Severity**: P0 (proposal feasibility error)

**Problem Description**:

The instrumentation pseudocode in Section 4.1 assumes that the `waitCKE_` flag can be detected before `ExecuteInstr()` returns, and that the delta between `snapshotBefore` and `snapshotAfter` is computed afterward. However, the actual code flow is:

```
WaitCkeProcess() → CKE not satisfied → ccuSimulator_->SetWaitCKEFlag(true) → return (Process() not called)
→ UpdateLoopStatus() returns false → ExecuteInstr() returns false → Execute() returns false
```

Key point: when CKE is not satisfied, `WaitCkeProcess` inside `Run()` sets the waitCKE flag and returns directly. **Process() is not called and no resources are modified**. Therefore:
- Pre-execution snapshot equals post-execution snapshot (delta is empty)
- The system should not capture snapshotAfter or compute delta

**Code Evidence**:

`CcuExecutorBase.cc` lines 162-172:
```cpp
if ((waitCKE & waitCKEMask) == waitCKEMask) {
    Process(ccuResMgr);      // CKE satisfied → execute Process
} else {
    ccuSimulator_->SetWaitCKEFlag(true);  // CKE not satisfied → set flag, do not execute Process
    return;
}
```

**Fix Recommendation**:

When waitCKE is not satisfied, record only key spin fields. Do not compute delta. Do not create a full trace entry:
```cpp
// During waitCKE, record only spin information
CcuTraceCollector::RecordWaitSpin(rankId_, dieId_, curInstrId,
                                     waitCKEId, waitCKEMask, actualCKEValue);
// Do not capture snapshotAfter; do not compute delta
// Do not create a full CcuTraceEntry
return false;
```

Create the full entry only after CKE passes (merging previous spin information).

---

### P0-2: globalSeqId/execRound Maintenance Should Reside in SequentialExecutor

**Severity**: P0 (proposal feasibility error)

**Problem Description**:

The document assumes that `CcuTraceCollector` maintains `globalSeqId/execRound/currentSqeTaskId` inside `CcuSimulator::ExecuteInstr()`. However, `CcuSimulator` can only see execution within its own CCU. It cannot perceive global scheduling rounds or CCU switching.

`CcuSimulator::ExecuteInstr()` executes within a single CCU and does not know:
- Which iteration the outer `while(HasTask())` loop is on (`execRound`)
- The global execution sequence number (`globalSeqId`)
- Which SQE task is currently executing (`sqeTaskId`)

This information is available only in the upper call chain of `SequentialExecutor::Execute()` and `TaskCcuGraph()`.

**Code Evidence**:

`hccl_task_sequential_execute.cc` lines 51-72 — global scheduling loop:
```cpp
while (HasTask()) {
    uint32_t rankId = 0;
    for (auto& rankTasks : allRankTaskQueues_) {  // iterate over all ranks
        for (auto& streamTasks : rankTasks) {     // iterate over each stream
            while (!streamTasks.empty()) {
                auto task = streamTasks.front();
                auto ret = ExecuteOneTask(task);   // ← globalSeqId increments here
                if (ret == HCCL_SIM_VRT_HOLD_CMD) break;
                streamTasks.pop();
            }
        }
    }
}
```

`hccl_task_thread.cc` lines 135-148 — TaskCcuGraph:
```cpp
auto simulator = ccuResMgr.InitSimulator(rankId, dieId, instrStartId, endInstrId, instCnt);
// sqeTaskId is available here (obtained from task parameters)
if (simulator->Execute() == false) {
    return HCCL_SIM_VRT_HOLD_CMD;
}
```

**Fix Recommendation**:

The global context is managed by `SequentialExecutor` and passed through parameters:
```cpp
// Inside SequentialExecutor::Execute()
while (HasTask()) {
    uint32_t execRound = 0;
    for (rankTasks : allRankTaskQueues_) {
        for (streamTasks : rankTasks) {
            while (!streamTasks.empty()) {
                CcuTraceCollector::BeginGlobalStep(execRound);
                // globalSeqId increments inside ExecuteOneTask
                auto ret = ExecuteOneTask(task);
                ...
            }
        }
    }
    execRound++;
}
```

Pass sqeTaskId to `CcuSimulator` inside `TaskCcuGraph()`:
```cpp
uint32_t sqeTaskId = CcuTraceCollector::RegisterSqeTask(rankId, dieId, task);
CcuSimulator::SetCurrentSqeTaskId(sqeTaskId);  // Used internally by Simulator
```

---

### P0-3: Full Snapshot Collection Has Excessive Performance Overhead

**Severity**: P0 (performance bottleneck)

**Problem Description**:

Section 4.1 calls `CaptureResourceSnapshot()` (full read of all resources) once before and once after every instruction. The overhead is extremely high:

- Single snapshot: 4096x8(XN) + 4096x8(GSA) + 1024x2(CKE) ≈ 67KB
- Two snapshots per instruction ≈ 134KB
- 10000 instructions ≈ 1.34GB (snapshot data only)

Section 4.5 mentions incremental collection optimization, but the pseudocode in 4.1 still uses the full snapshot + diff computation flow, which contradicts the optimization proposal.

**Industry Practice Reference**:

- **rr debugger**: checkpoint + incremental recording mode; full snapshots only at key nodes
- **Chrome DevTools Timeline**: event stream format; each event contains only changed args
- **Perfetto (Android trace)**: protobuf binary incremental format

**Fix Recommendation**:

Switch to ResourceManager change interception + initial snapshot + periodic checkpoints:

1. **Initial snapshot**: perform one full capture at the start of Run (stored in `CcuIdentity.initialSnapshot`)
2. **Change interception**: intercept changes directly in `CcuResourceManager::UpdateXnValue/UpdateGsaValue/UpdateCkeValue` and record deltas:
```cpp
void CcuResourceManager::UpdateXnValue(int rankId, int dieId, uint16_t xnId, uint64_t value) {
    uint64_t oldValue = GetXnValue(rankId, dieId, xnId);
    if (oldValue != value) {
        CcuTraceCollector::RecordXnDelta(rankId, dieId, xnId, oldValue, value);
    }
    xnData_[rankId][dieId][xnId] = value;
}
```
3. **Periodic checkpoint**: take a full snapshot every N instructions (N is configurable, default 1000) for fast intermediate state recovery
4. **Recover arbitrary point state**: start from the nearest checkpoint and accumulate subsequent deltas

---

## II. Design Optimization Points (Recommended Improvements)

### P1-1: Loop Offset Formula Is Incomplete

**Severity**: P1 (missing information)

**Problem Description**:

The offset calculation in `UpdateAddress()` is more complex than the `CcuExecutionContext.gsaOffset` described in the document:
```cpp
uint64_t CcuExecutorBase::UpdateAddress(uint64_t addr, uint16_t addrExpandCoef) {
    return addr + ((ccuSimulator_->GetLoopExtendNum() * ccuSimulator_->GetGSAOffset()) << addrExpandCoef)
        + ((ccuSimulator_->GetCurLoopCnt() * ccuSimulator_->GetLoopIterStepGSA()) << addrExpandCoef);
}
```

Five parameters are involved: `addr` (original address), `addrExpandCoef` (expansion coefficient), `loopExtendNum`, `gsaOffset`, `curLoopCnt`, `loopIterStepGSA`.

The current `CcuExecutionContext` has only a simple `gsaOffset` field. It cannot fully reconstruct the GSA address offset calculation inside a Loop.

**Fix Recommendation**:

Add complete offset parameters to `CcuExecutionContext`:
```cpp
struct CcuExecutionContext {
    bool inLoop;
    uint16_t loopRound;          // current iteration round
    uint16_t loopExtendIndex;    // unroll index

    // Loop offset parameters (valid only when inLoop=true)
    uint64_t gsaAddrOffset;      // = extendIndex * gsaOffset + curLoopCnt * iterStepGSA (computed final offset)
    uint64_t gsaOffset;          // GSA base offset coefficient (GetGSAOffset)
    uint64_t iterStepGSA;        // GSA step per iteration round (GetLoopIterStepGSA)
    uint32_t curLoopCnt;         // current iteration count (GetCurLoopCnt)
    uint32_t loopExtendNum;      // unroll count (GetLoopExtendNum)
    uint16_t addrExpandCoef;     // address expansion coefficient

    uint16_t msOffset;           // MS ID offset (GetLoopMsOffset)
    uint16_t ckeOffset;          // CKE ID offset (GetLoopCKEOffset)
    uint16_t xnIdOffset;         // XN ID offset (GetLoopXnIdOffset)
};
```

---

### P1-2: Cross-CCU Changes Should Be Intercepted in ResourceManager

**Severity**: P1 (collection accuracy)

**Problem Description**:

Section 4.1 places the cross-CCU change collection point in `CcuSimulator::ExecuteInstr()` as a post-hoc call to `CaptureCrossCcuChanges()`. However, cross-CCU operations actually occur in the Executor `Process()` method (for example, `SyncCkeExecutor::Process()` calls `SetRmtCKESignal()`).

Post-hoc collection requires comparing the remote CCU state before and after. This means taking a remote CCU snapshot both before and after instruction execution. The overhead is high and omissions are possible (the remote CCU may be modified concurrently by other code).

**Fix Recommendation**:

Intercept cross-CCU changes directly in `CcuResourceManager::UpdateCkeValue()`:
```cpp
void CcuResourceManager::UpdateCkeValue(int rankId, int dieId, uint16_t ckeId, uint16_t value) {
    uint16_t oldValue = GetCkeValue(rankId, dieId, ckeId);
    // Detect whether this is a cross-CCU operation
    auto [execRank, execDie] = CcuTraceCollector::GetCurrentExecutingCcu();
    if (rankId != execRank || dieId != execDie) {
        CcuTraceCollector::RecordCrossCcuCkeChange(
            rankId, dieId, ckeId, oldValue, value, execRank, execDie);
    }
    ckeData_[rankId][dieId][ckeId] = value;
}
```

Apply the same approach to `UpdateXnValue()` (operations like TransXnToRmt), `UpdateGsaValue()`, `TransMSToMS()`, and so on.

---

### P1-3: CcuInstrTraceDetail Polymorphism Is Over-Engineered

**Severity**: P1 (architecture optimization)

**Problem Description**:

The current design defines **13** `CcuInstrTraceDetail` subclasses (Load/Arith/Trans/Sync/Loop/LoopGroup/Cke/Jump/Wait/Fence/Reduce/...), but the following issues exist:

1. **Duplication with common layer**: `CcuTransTraceDetail` has `waitCKEId/setCKEId` that overlap with `CcuTraceEntry.waitInfo`; `CkuCkeTraceDetail` has `ckeId/ckeValueBefore/ckeValueAfter` that overlap with `resourceDelta.ckeChanges`
2. **Subclass count bloat**: the code contains 30+ instruction types (V1 approximately 20 types + V2 approximately 10+ types). Each type needs its own subclass.
3. **Complex frontend rendering**: the frontend must dispatch different rendering templates based on typeName
4. **High cost for new instructions**: every new instruction type requires a new subclass struct + CollectTraceDetail implementation

**Industry Practice Reference**:

- **Chrome DevTools Trace Event Format**: flat `args` field (key-value map); no polymorphic inheritance
- **LLVM Execution Trace**: flat "additional info" field
- **GDB MI (Machine Interface)**: key-value attribute list

**Fix Recommendation**:

Simplify `CcuInstrTraceDetail` to a flat structure:
```cpp
struct CcuInstrTraceDetail {
    std::string typeName;                     // for example "TransLocMemToLocMem"
    std::map<std::string, std::string> args;  // instruction-specific parameter key-value table
    // Examples:
    //   Trans type: {"srcAddr": "0x7f0000", "dstAddr": "0x7f1000", "length": "4KB", "channelId": "0"}
    //   Reduce type: {"op": "Add", "msList": "[0,1,2,3]", "dataType": "FP32", "castEn": "0"}
    //   CKE type: {"ckeOp": "Set", "ckeId": "3", "ckeMask": "0x0001", "isRemote": "true"}
};
```

Advantages:
- No need to define a subclass for each instruction type
- For new instructions, only populate args in `CollectTraceDetail()`
- Frontend rendering logic is unified (key-value table)
- Output from `Describe()` can serve directly as the args source

---

### P1-4: JSON Serialization Format Conflicts with Large Data Volumes

**Severity**: P1 (performance optimization)

**Problem Description**:

Section 3.4 uses JSON as the serialization format, but JSON has weaknesses:
- Each uint64_t XN/GSA value requires string representation (`"0x7f3a0000"`); 8 bytes become approximately 16 bytes
- Massive redundant key name repetition (`"xnChanges"` repeats in every entry)
- A JSON file for 10000 instructions can reach 50MB+
- Frontend loading and parsing of large JSON files is extremely slow

**Industry Practice Reference**:

| Tool | Format | Characteristics |
|------|------|------|
| Chrome DevTools | JSON streaming (one event per line) | Supports line-by-line parsing |
| Perfetto (Android) | protobuf binary | Frontend WASM parsing; small size |
| rr debugger | custom binary | Efficient recording/playback |
| Intel VTune | custom binary + SQLite | On-demand queries |

**Fix Recommendation**:

Layered serialization strategy:
- **Collection side**: msgpack/flatbuffers binary format; 5-10x smaller size; faster write speed
- **Frontend loading**: decode binary into frontend index structures through Worker
- **Export/debug**: optional export to JSON (for human readability and debugging)
- Remove `metadata.rankId/dieId` from JSON Schema (replace with `rankSize/diePerRank` to align with the global model)

---

## III. Design Optimization Points (Recommended Improvements, Lower Priority)

### P2-1: Describe() Can Serve as MVP Trace Information Source

**Severity**: P2 (implementation strategy)

**Problem Description**:

Every Executor in the code implements a `Describe()` method that returns formatted strings such as:
```
"[Simulation Execute] Wait CKE[3:0001], Sync LocCKE[5:00ff] To rmtCKE[8:00ff]
 Use Channel[0], Set CKE[10:0001], clearType[1]"
```

These strings already contain key parameter information for the instruction. The current design requires each Executor to add a new `CollectTraceDetail()` virtual method and implement it. The workload is significant (30+ instruction types).

**Fix Recommendation**:

Phased implementation strategy:

**Phase 1 (MVP)**:
- Collect only `Describe()` output + resource delta
- Set the `instrDescribe` field directly from the `executor->Describe()` return value
- Do not implement `CollectTraceDetail()`; leave the detail field empty or containing only typeName

**Phase 2 (Enhancement)**:
- Implement `CollectTraceDetail()` for high-frequency/critical instruction types (for example Trans, Reduce, SyncCke)
- Use flat key-value map for detail (see P1-3)
- Progressively cover all instruction types

This approach enables rapid MVP delivery and reduces initial development cost.

---

### P2-2: Notify and Other Non-CCU Tasks Are Missing from Trace

**Severity**: P2 (missing information)

**Problem Description**:

`SequentialExecutor::Execute()` handles not only `CCU_GRAPH` but also `NOTIFY_RECORD/NOTIFY_WAIT/REDUCE/MEM_CPY/AIV_GRAPH` and other task types. `NotifyWait` also returns `HOLD_CMD` and exhibits blocking behavior similar to CCU waitCKE.

The current trace covers only `CCU_GRAPH` type tasks. However, Notify Record/Wait affects the execution flow:
- `NotifyWait` blocking causes the outer loop to skip that stream
- After `NotifyRecord` unblocks, the CCU task on that stream can continue in the next round

If the trace does not reflect this, the frontend replay will show "CCU instructions suddenly jump to the next round" with no explanation of what happened in between.

**Fix Recommendation**:

Add simplified records for non-CCU task types in `globalEntries[]`:
```cpp
struct CcuTraceNonCcuEntry {
    uint32_t globalSeqId;
    uint32_t execRound;
    int32_t rankId;
    HccLTaskMetaType taskType;    // NOTIFY_RECORD, NOTIFY_WAIT, REDUCE, MEM_CPY...
    HcclVmResult execResult;      // SUCCESS or HOLD_CMD
    std::string description;      // task description (for frontend display)
};
```

The frontend can display during CCU switching prompts: "Round 1: Rank1 NotifyWait blocked, skipped".

---

### P2-3: JSON Schema metadata Model Is Inconsistent

**Severity**: P2 (model consistency)

**Problem Description**:

The JSON Schema example in Section 3.4 still includes `rankId`/`dieId` fields in `metadata` (per-CCU model), but `CcuRunMetadata` in Section 3.3.1 has already switched to `rankSize`/`diePerRank` (global model). The two locations are inconsistent.

Additionally, the JSON Schema example uses `"seqId"` instead of the new design's `"globalSeqId"`. It is missing new fields such as `"execRound"`/`"sqeTaskId"`/`"ccuRegistry"`/`"sqeTaskRegistry"`.

**Fix Recommendation**:

Update the JSON Schema example to align with the data structure definitions in Sections 3.3.1-3.3.2:
- Remove `metadata.rankId`/`metadata.dieId`
- Add `metadata.rankSize`/`metadata.diePerRank`/`metadata.totalExecRounds`
- Add `ccuRegistry[]` and `sqeTaskRegistry[]`
- Change `seqId` to `globalSeqId` in entries
- Add `execRound`/`sqeTaskId`/`rankId`/`dieId` fields

---

## IV. Summary Table

| Priority | ID | Issue | Type | Fix Summary |
|--------|------|------|------|---------|
| P0 | P0-1 | Delta should not be computed during CKE Wait | Feasibility error | Record only key fields during waitCKE; create full entry only after CKE passes |
| P0 | P0-2 | globalSeqId/execRound maintenance location is wrong | Feasibility error | Global context managed by SequentialExecutor; passed to Simulator as parameters |
| P0 | P0-3 | Full snapshot performance bottleneck | Performance bottleneck | ResourceManager change interception + initial snapshot + periodic checkpoints |
| P1 | P1-1 | Loop offset formula is incomplete | Missing information | Add iterStepGSA/curLoopCnt/extendNum/addrExpandCoef |
| P1 | P1-2 | Cross-CCU change collection timing | Collection accuracy | Intercept cross-CCU changes inside ResourceManager |
| P1 | P1-3 | Detail polymorphism is over-engineered | Architecture optimization | Simplify to flat key-value map |
| P1 | P1-4 | JSON format unsuitable for large data volumes | Performance optimization | Use msgpack/flatbuffers on collection side; WASM parsing on frontend |
| P2 | P2-1 | Describe() can serve as MVP source | Implementation strategy | Phase 1: only Describe() + delta |
| P2 | P2-2 | Notify and other non-CCU tasks are missing | Missing information | Add simplified records |
| P2 | P2-3 | JSON Schema model inconsistency | Model consistency | Align with Sections 3.3.1-3.3.2 |

---

## V. Fix Priority Recommendations

**Step 1 (mandatory)**: Fix P0-1, P0-2, P0-3; otherwise the proposal cannot be implemented

**Step 2 (recommended)**: Fix P1-1, P1-2, P1-3 to improve proposal quality and practicality

**Step 3 (optional)**: Fix P1-4, P2-1, P2-2, P2-3 to refine details and user experience
