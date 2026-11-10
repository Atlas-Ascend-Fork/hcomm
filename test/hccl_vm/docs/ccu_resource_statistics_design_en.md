# CCU Resource Statistics Feature Design

> Feature objective: After the checker finishes graph construction, for each operator invocation (op) x device(+die) granularity,
> collect all CCU resources used by that invocation (channel, instruction space, Xn/GSA/CKE/MS registers, loop, mission);
> during graph construction, data is collected as dfx; after all graph construction completes, a CSV statistical report is extracted from dfx.
>
> Status: Design phase (D1-D7 implemented; revised to V6 based on review comments)
> Affected modules: `src/plugin/checker`
>
> V4 key decisions:
> 1. **Resource collection runs only in the single-operator graph construction phase; statistics are disabled during the big-graph graph construction phase**. The two paths are fully isolated.
> 2. **Statistics are collected only per operator invocation; no run-level/device total aggregation is performed**. When an operator processes different data volumes, it uses different CCU resources, so run-level aggregation is meaningless for analyzing operator resource usage.
>
> V5 revision: **The primary channel metric was changed from "delta" to "translation-phase reference collection"**. Channels are persistently reused across ops. Under the hccl_test batch-invocation scenario, the delta metric for subsequent ops is almost always empty (channelNum=0 appears widely), which contradicts the per-op statistics premise. The reference metric is promoted to the primary metric for phase 1, and delta is demoted to a reference (see 3.1).
>
> V6 revision: **CSV now displays sections per operator, with section headers carrying operator information** (operator type, algorithm name, data type, element count, input/output data size, reduce type, rankSize). Data rows are simplified to 14 columns (see 4.2). Topology mapping is corrected to the `Device.id -> server_id -> CommConfigServer.serverKey` primary path (D7 revealed deviceId and physical_id metric misalignment; see 4.3).

---

## 1. Background and Requirements

### 1.1 Requirements Description

1. After the checker finishes graph construction, collect statistics for all resources used by the CCU.
2. **Statistics scope (V4 revision)**: The unit of statistics is **each operator invocation**. Each invocation (`OpExecutionKey{commName, commHash, opIter}`) produces a separate statistics report. Reason: When an operator processes different data volumes, the CCU resources it uses also differ (for example, hccl_test invokes in batches of 64B/128B/.../1G; the loop expansion count and register usage of each invocation differ). Run-level aggregated statistics are meaningless for analyzing operator resource usage. **Device total-level reports are not produced**. Composite samples (one run invoking multiple operators) are also distinguished by operator.
3. **Collection scope decision (V4)**: Resource collection runs **only in the single-operator graph construction phase** (`enableNewChecker` path, `CheckOneOp -> GenAndCheckGraphV3`). **The big-graph graph construction phase (BigGraphCheckerV3) disables statistics**. The two paths are fully isolated. This eliminates duplicate counting and operator attribution issues inside big-graph windows when both checkers are enabled at the same time. Cost: Pure big-graph runs (`enableNewChecker=off`) produce no statistics data, which must be declared with a WARN (see section 6 risk list).
4. Statistics dimension: Within each operator invocation, statistics are by device + ccu. Index label: `superPodId + serverId + phyDevId` locates the device, then `dieId` locates the ccu.
5. Statistics resource items:
   - **channel**: Obtained from the checker plugin's channel table; records the deduplicated channel count.
   - **Instruction space**: CCU instruction space is 32K entries x 32B = 1MB; collects the number of generated instructions.
   - **Registers**:
     - A5: Xn, GSA, CKE, MS;
     - A6: Xn, CKE, MS (A6 does not have GSA);
     - **CKE statistics granularity follows ckeId; no mask/bit information is included**;
     - **A6 introduces register address addressing; all resources must first convert addresses to ids before statistics are collected**;
   - **loop**: CCU concurrent execution unit. Collects only the **static expansion instance count** (excluding runtime iteration count).
     Computed from LoopGroup instructions: Let loop template count be `n`, expansion offset be `o` (`o in [0, n-1]`), and expansion count be `m`. Then loop count = **`n + (n - o) * m`**.
   - **mission**: Statistics (added in V2);
   - **Explicitly not collected**: queue, loopCtxId, jetty (jetty is collected elsewhere), halfRTT (not handled for now).
6. Two-level collection/report structure: During graph construction, statistics are collected and stored as dfx; after all checker graph construction completes, dfx is extracted to produce a per-op granularity CSV report (see section 4).

### 1.2 Correspondence with Existing Code (Verified)

| Requirement | Code Basis |
|---|---|
| deviceId equals phyDevId | `operation_data_types.h:36`: "Physical device ID (globally unique)" |
| Topology label source | checker reads directly from the DB: `TopoMetaConfig` table (`include/runtime_state/sim_models.h:23-26`); checker already has direct Db access precedents (`dag_graphviz_dump.cc:648` reads the Communicator table) |
| Register write centralized entry | `AllRankParamRecorder::SetXn/SetGSA/SetCKE` (`ccu_all_rank_param_recorder_v3.cc:84/90/96`), all carry `(deviceId, dieId)` |
| A6 address -> id conversion | `GetXnAndTypeIdByAddr / GetXnIdByAddr / GetMSIdByAddr` (recorder .cc:279/377/406); address range constants .cc:259-269. When the checker accesses resources, it already performs address -> id conversion uniformly. Statistics can reuse the same conversion. |
| channel table | `m_allRankChannelInfo: map<DeviceId, map<dieId, map<channelId, RemoteDieInfo>>>` (`storage_manager.h:289`), loaded by `Loader::GetCcuChannelInfo` + `EnsureChannelInfo` (`storage_manager.cc:582`) |
| Instruction space | `CcuInstrResTab{deviceId, dieId, instrCount, instrSpace[32K][32]}` (`operation_data_types.h:117-122`), decoded by `LoadHcclVmInstrData` (`storage_manager.cc:765`); the 32K upper-bound check already exists (.cc:771) |
| mission / instruction range | `CcuTaskParam{dieId, missionId, instStartId, instCnt}`; `CcuGraphStateV3::InitInstrInfo` already computes `missionStartInstrId / missionInstrCount` (`ccu_task_common_v3.cc:307-336`) |
| loop expansion formula | A6: `ccu_task_transform_instruct_a6_v3.cc:1703` `expandTimes = (curLoopIdx < expandOffset) ? 1 : (expandCnt + 1)`. Total expansion = `o + (n-o)*(m+1)` = `n + (n-o)*m`, **consistent with the requirements formula**; A5 is isomorphic (a5_v3.cc:2056); no nesting (`CheckLoopGroupNotSupport` guarantees this) |
| LoopGroup field widths | A6: `loopInsCnt(10bit) / expandOffset(9bit) / expandCnt(9bit)`; A5: all 7bit |
| MS usage convergence point | `GenSliceFromMs(msId, len, slice)` (`ccu_task_common_v3.cc:1071`): All A5/A6 MS usage points (normal instructions + loop expansion instructions; about 30+ call sites) call this function with the final msId after `UpdateMSId` offset |
| Existing dfx pattern | `src/dfx/dag_graphviz_dump`: controlled by the `enableDagGraphvizDump` switch; collects during graph construction and writes to disk at end |

---

## 2. Overall Architecture

**Collection occurs only in the single-operator graph construction path**. The collector resets per run and buckets by operator invocation (`BeginOp/EndOp`). After all operator graph construction completes, it generates a per-op granularity CSV in one pass (**displayed in operator sections**, each section containing an operator information header + the operator's (dev,die) data rows). No cross-op aggregation is performed.

```
         Single-operator phase (collection)         Big-graph phase           Post-construction (report)
+-----------------------------------------+   +------------------+   +------------------------------+
| RunChecker -> iterate op -> CheckOneOp:  |   | ProcessBigGraph   |   | CcuResourceStatsReporter(dfx) |
|                                         |   |                  |   |  - DB reads topology; builds  |
|  auto g = BeginOp(opKey) <--- gate       |   |  statistics off  |   |    deviceId->(superPod,server) |
|  (RAII guard; destructor calls EndOp;   |   |  (no BeginOp,    |   |  - per-op bucket -> sectioned  |
|   must precede LoadHcclVmSynthesisData) |   |   hooks no-op,   |   |    ccu_resource_stats.csv     |
|  (1) Has/GetChannelInfo ref collection  |   |   zero collect/  |   |    (one section per op: header|
|  (2) SetXn/SetGSA/SetCKE hooks          |   |   zero overhead) |   |     + (dev,die) data rows)    |
|      (A6 address->id first)             |   +------------------+   |  - summary log                |
|  (3) GenSliceFromMs records msId        |                           |  - dumps to DumpV3Manager dir |
|  (4) TransformLoopGroupInstr key log    |                           +------------------------------+
|  (5) InitInstrInfo records mission/range|
|  (6) SetCurrentOpInfo records op info   |
|  ~guard -> EndOp() seals bucket         |
|      (on any exit path)                 |
|                                         |
|  CcuResourceStatsCollector (dfx, singleton)|
|  map<OpKey, map<dev, map<die, Stats>>>  |
+-----------------------------------------+
```

- **Collector**: New `src/plugin/checker/src/dfx/ccu_resource_stats_dfx.h/.cc`, singleton.
  Run-level reset point: start of `RunChecker` (near main.cc:508) -- **not** the per-op Reset inside `CheckOneOp` (main.cc:221; that is the recorder's reset point and happens to be the same location as this feature's `BeginOp`).
- **Gate mechanism (key to big-graph isolation)**: All hooks (Set*/GenSliceFromMs/TransformLoopGroupInstr/InitInstrInfo) reside in code paths shared by single-operator and big-graph construction. They are driven uniformly by the "current operator context":
   - Inside `CheckOneOp` (after main.cc:221 Reset and before `LoadHcclVmSynthesisData`), call `collector.BeginOp(opKey)`;
   - **(Mandatory) `BeginOp` must return a RAII guard; the guard destructor executes `EndOp()` to seal the bucket**. Reason: `CheckOneOp` has multiple early-return failure paths after BeginOp (rank load / PrepareOpParam / FinalizeOpGroup / taskMeta load failures, main.cc:250-301). A bare `EndOp()` call on the failure path is inevitably missed, causing the operator context to hang and the next op's collected data to pollute the previous bucket (bucket contamination). RAII guarantees every exit path (including exception/CHK_RET short circuits) seals the bucket correctly.
   - Hook entry check: If no current operator context exists, return immediately (no-op).
   - The big-graph path (`ProcessBigGraph`) never calls `BeginOp`, so it naturally collects nothing. Even when both checkers are enabled simultaneously, no duplicate counting occurs.
- **per-op isolation correctness basis**: In single-operator mode, `AllRankParamRecorder::Reset()` runs before each op (main.cc:221). Register state does not leak across ops. Xn/GSA/CKE/MS recorded in an op bucket all belong to that op, with no "read/write attribution" ambiguity. Channels are collected by actual translation-phase references (wrapped by `Has/GetChannelInfo`; see 3.1), naturally belonging to the current op.
- **Reporter**: Generated uniformly at the end of `RunChecker` (after the single-operator result summary table; near main.cc:594-654).
- **Switch**: `CheckerSettings` adds `enableCcuResourceStatsDump` (`manifest.json`; disabled by default), aligning with the `enableDagGraphvizDump` pattern (`setting_manager.h:21`). When the switch is on but `enableNewChecker=off`, print a WARN (no collection path; the report is empty).

### 2.1 Statistics Data Structure

```cpp
// Inside dfx; one per (opKey, deviceId, dieId); set types are used for deduplication
struct CcuResStats {
    // ---- channel (reference metric, deduplicated) ----
    std::set<uint32_t> channelIds;       // channelIds actually referenced during translation (channelNum = size(); primary metric)
    std::set<uint32_t> newChannelIds;    // Delta metric: channelIds newly introduced by this op (reference; not in the main CSV column)

    // ---- instruction space ----
    uint32_t instrResidentNum = 0;       // CcuInstrResTab.instrCount (run-level snapshot; reference column; not per-op metric)
    std::set<uint32_t> opInstrIds;       // Union of this op's SQE instruction ranges (instructionNum = size(); primary metric)

    // ---- registers (logged at write time; A6 address->id first; deduplicated) ----
    std::set<uint16_t> xnIds;            // xnNum = size(); also records xnMaxId peak
    std::set<uint16_t> gsaIds;           // A5 only; A6 outputs NA
    std::set<uint16_t> ckeIds;           // ckeNum = size(); granularity is ckeId; no mask
    std::set<uint16_t> msIds;            // msNum = size()

    // ---- loop (idempotent logging by key) ----
    struct LoopGroupKey {              // (missionId, instStartId, loopGroup instruction position)
        uint16_t missionId;
        uint32_t instStartId;          // SQE identifier: the same LoopGroup instruction can be referenced by SQEs with different args
        uint32_t loopGroupInstrId;
    };
    struct LoopGroupRecord {
        uint32_t n;                    // loopInsCnt (template count)
        uint32_t expanded;             // n + (n-o)*m (static expansion instance count)
    };
    std::map<LoopGroupKey, LoopGroupRecord> loopGroups;
    // loopGroupNum = loopGroups.size()
    // loopTemplateNum = sum record.n
    // loopNum = sum record.expanded

    // ---- mission (deduplicated) ----
    std::set<uint16_t> missionIds;       // missionNum = size()
};

// Operator identifier
struct OpKey {  // = OpExecutionKey
    std::string commName;
    uint64_t commHash;
    uint32_t opIter;
};

// Operator display info (V6: included in CSV section header; CheckOneOp calls SetCurrentOpInfo after CheckerParam is ready to bucket it)
struct OpInfo {
    std::string collectiveType;  // HcclCmdTypeToString(cmdType)
    std::string algName;         // OpDetailTab.algorithmName (first non-empty across ranks; NA if empty)
    std::string dataType;        // HcclDataTypeToString(dataType)
    uint64_t elementCount;       // dataCount
    uint64_t inputBytes;         // CalcInputOutputSize, rank0 view
    uint64_t outputBytes;
    std::string reduceType;      // Only for reduce-class operators; NA for others
    uint32_t rankSize;
};
```

Aggregation containers: `std::map<OpKey, std::map<DeviceId, std::map<uint32_t /*dieId*/, CcuResStats>>>`
+ `std::map<OpKey, OpInfo>`.

**No cross-op aggregation (V4 decision)**: The report outputs each op bucket directly, with no device total merge layer.

> Deduplication semantics: The metric is "after this operator invocation finishes graph construction, which resource ids have been used". Within an op bucket, deduplication is by id set (the same resource written/read/cleared multiple times counts only once; not accumulated by instruction count; see 3.3 general principle 1). Loop uses idempotent key logging (once instruction space + SQE parameters are determined, the expansion result is determined, independent of expansion count; see 3.5). Resource reuse between different ops is **not merged**. Each invocation's resource usage is presented independently, which is exactly the analysis goal of this feature (different data volumes -> different resource usage).

---

## 3. Collection Details for Each Resource Item

### 3.1 channel (channelNum)

- Data source: `g_allRankChannelInfo[dev][die]` (`storage_manager.h:289`).
- **Primary metric (V5 revision): Reference metric**: Collects "the set of channelIds actually referenced during this operator invocation's instruction translation". Reason: Channels are persistently reused across ops. The delta metric (see below) is almost always empty for subsequent ops under the V4 core scenario (hccl_test batch invocations under the same communicator), with channelNum=0 appearing widely, making the data unusable. **Implementation (landed)**: Add unified wrappers `HasChannelInfo(dev, die, channelId)` (existence check) and `GetChannelInfo(dev, die, channelId)` (value retrieval; returns `const RemoteDieInfo&`), replacing 32 existing direct `g_allRankChannelInfo` lookups (a5_v3.cc 22 sites, a6_v3.cc 10 sites; verified all concentrated in the two translation files with uniform pattern; A6 `IsExistRemoteDieInfo` preserves find semantics and logs internally). When the wrapper hits an actually existing channel, it calls `collector.RecordChannel(dev, die, channelId)` into the current op bucket. Reference points are all in the serial translation phase, reusing the same context mechanism from 3.4 plan A, with consistent gating. Note: On the graph node side, only `CcuNotify.channelId` (task_def_v3.h:216, synchronization-class node) carries a channel; data movement nodes do not. The "traverse nodes after graph construction" path has incomplete coverage and is excluded.
- **Reference metric (demoted; optional column): Delta metric**: At `BeginOp`, record a channel set snapshot per (dev,die); at `EndOp`, compute the delta to obtain "channelIds newly introduced by this op". It is valuable only when analyzing "which op first introduced a channel", and is not included in the main CSV column (kept in dfx).
- halfRTT-related resources: **Not collected** (the review explicitly said not to handle this for now).

### 3.2 Instruction Space (instructionNum)

- **Primary metric (per-op; included in CSV)**: For each SQE of this op, the `[instStartId, instStartId+instCnt)` range is unioned per (dev,die) (hook point `CcuGraphStateV3::InitInstrInfo`, ccu_task_common_v3.cc:307; already has `missionStartInstrId/missionInstrCount`), inserted into the current op bucket `opInstrIds`. `instructionNum = opInstrIds.size()`. Semantics: "the number of instructions used by this operator invocation on this CCU". Note: `missionInstrCount` is uint16 (force-cast at ccu_task_common_v3.cc:329), and single-mission instruction scale is far less than 64K, so truncation does not affect statistics accuracy (confirmed by review; no handling needed).
- **Reference metric (optional column)**: `CcuInstrResTab.instrCount` per (dev,die) (`LoadHcclVmInstrData`, `storage_manager.cc:765-795`, run-level complete resource snapshot). Semantics: "the total number of instructions generated within the CCU's 1M instruction space", used to cross-check the 32K upper-bound occupancy (`instrResidentNum / 32768`). This value is the same across all op rows within the same run.
- The difference between the two metrics can reveal instructions resident on this CCU that are not referenced by this op (including instruction segments from other ops).
- Implementation optimization (non-blocking): `opInstrIds` uses `set<uint32_t>`, up to 32K elements per bucket per die. When op count x die count is large, interval merge storage can replace it; total length is computed at report time.

### 3.3 Registers Xn / GSA / CKE

**General principles:**
1. **Deduplicate by id set; do not accumulate by instruction (confirmed by review)**. The metric is "after this operator invocation finishes, which resource ids have been used". The same resource repeatedly operated by multiple instructions (write, read, clear) counts as 1: for example, `SetXn(xn[0])` followed by `ClearX(xn[0])` counts xnNum as 1, not 2. The `std::set<id>` structure satisfies this naturally. Hooks only `insert(id)` and are unrelated to instruction count.
   - ClearX metric (review settled): A6 `ClearX` calls `SetXn(i, 0)` per id in the range (a6_v3.cc:502). After set deduplication, no double counting occurs. **Any id touched by ClearX is counted as used** (it does not need to have been actually written before).
   - ClearX anomaly detection (optional extension; not in phase 1): Clearing an id that was never written may indicate an algorithm orchestration issue (misaligned with SetXn, or redundant cleanup). Implementation cost is low (verified a6_v3.cc:485-510). Before `TransformClearXInstr` clears the `[xnId, xmId]` range, check whether each id already exists in `curXn[dev][die]` (this map is filled only by SetXn; the recorder needs a `HasXn` query interface). Non-existence indicates suspected misalignment/redundancy. Two caveats: (1) When `xnId > xmId`, the semantics is clear to `MAX_LOADX_STOREX_ID_NUM(16383)`. Full clear hits many unwritten ids. WARN must be aggregated per instruction into a count summary (such as "ClearX [a,b]: k ids never written"), and per-id printing is forbidden. (2) At the end of ClearX, `ProcessSetMask -> SetCKE` is also called. The ckeIds it touches follow the "touch means collect" metric and enter ckeIds normally. **This phase only performs statistics, not monitoring**. This item is listed as optional for phase 2.
2. **Log at write time; prohibit end-state snapshots**. A6 `ClearX`, CKE wait bit clear (`ClearWaitMask`), and `ClearCKE` erase the end state. The snapshot method misses "used then cleared" resources. Logging at write time + set deduplication simultaneously satisfy "no misses" and "no duplicates".
3. **A6 always converts address -> id before statistics**. A6 instruction operands are register addresses (CKE/PFE/CHANNEL/JETTY/MISSION/LOOP/MS segments; recorder .cc:259-269). **Segment classification is done in the caller**: Translation code first goes through `GetXnAndTypeIdByAddr / GetXnIdByAddr` for conversion, then dispatches by `CcuComponerntType` before calling SetXn/SetCKE (verified: a6_v3.cc:2312-2317 branches by type). Therefore, the `SetXn` hook naturally receives only the XN segment id; the hook itself needs no further classification.
4. **The id after loop expansion offset is the true occupancy**. Values after `UpdateXnId/UpdateCKEId` (a6_v3.cc:205/218; bit15 offset logic a6:272) are logged (the offset write still goes through Set* in the end; no direct map bypass was verified).
5. **per-op attribution is unambiguous (V4)**. In single-operator mode, `AllRankParamRecorder::Reset()` runs before each op (main.cc:221). Register state does not leak across ops. Hooks write uniformly to the current op bucket set by `BeginOp`. Records in the op bucket are "resources written/occupied by this operator invocation". Reading values left by other ops does not exist in single-operator mode. Cross-device writes (remote post/movement) carry the real target (dev,die) in `SetXn/SetCKE(rmtDeviceId, rmtDieId, ...)` (a6_v3.cc:2314-2317/2385/2454/2534), correctly logging to the peer row.

| Resource | Hook Point | Record Content |
|---|---|---|
| Xn | `SetXn` (recorder .cc:84) | `xnIds.insert(id)`; also records `xnMaxId` peak. A6 segment classification is done in the caller; the hook receives the XN segment id directly (see general principle 3) |
| GSA (A5 only) | `SetGSA` (.cc:90) | `gsaIds.insert(id)`; A6 report outputs `NA` (do not output 0, to avoid misreading as "unused") |
| CKE | `SetCKE` (.cc:96) | `ckeIds.insert(ckeId)`. **Granularity is ckeId; no mask/bit information** |

### 3.4 MS (Plan for Review)

Current state: The recorder has no MS registration table, and msId does not go through Set* hooks. After code review, **`GenSliceFromMs(msId, len, slice)` (`ccu_task_common_v3.cc:1071`) is the single convergence function for all A5/A6 MS usage points**: Normal movement/reduction instructions and loop expansion instructions (about 30+ call sites, a5_v3.cc:834-1866; A6 is isomorphic) all call it with the final msId after `UpdateMSId` offset. A6 address-form MS has already been converted to id by `GetMSIdByAddr` (recorder .cc:279). Therefore, only the problem "this function lacks (dev,die) context" needs to be solved. Three candidate plans:

| Plan | Approach | Change Scope | Pros/Cons |
|---|---|---|---|
| **A. Context injection (recommended)** | Reuse the "current operator context" gate from section 2: After `BeginOp`, the instruction translation entry (`TransformInstrQue`/`ProcessCcuNode`, which knows current dev/die via `curCcuTask->GetDeviceId()/GetDieId()`) writes `(opKey, dev, die)` into the collector's current context (a plain member is sufficient; CCU expansion is single-threaded serial). `GenSliceFromMs` reads the context internally and calls `RecordMs(...)` | 2 entry points + 1 function body; **zero changes at call sites** | Covers all MS usage paths (including loop merge/expansion); shares the same context mechanism with the collection gate; depends on the "translation-phase serial" assumption (verified: ccu_graph_generator_v3 has no threads; multithreading exists only in the memory conflict check phase). Must be documented in code comments. |
| B. Explicit parameter passing | Add `(deviceId, dieId)` parameters to `GenSliceFromMs` and pass them at all call sites | 1 function signature + 30+ call sites mechanically modified | No implicit state; most robust. Large change scope; both A5/A6 files need modification. |
| C. Post-graph node traversal | After graph construction, traverse TransMem/Reduce/Batch nodes' MS slices (`BufferType::MS`, `offset = msId*4096` reverse-derive msId; loop merge nodes have an additional `msIds` set), aggregate by node `TaskPosition.deviceId` + die | Independent traversal code | Non-invasive to the translation layer. However, it must cover all node types and depends on graph shape (after merge/parallelization rewrites), with missed-collection risk. |

**Recommendation: Select A** (A+C cross-validation as fallback). Outstanding confirmation item: Whether the msId produced by A6 `GetMSIdByAddr`'s 4 CCUA bank x interleave 8 addressing (recorder .cc:266-277) is a die-global index or a bank-local index must be aligned with hardware specifications and noted in the report.

### 3.5 loop (loopNum)

- **Formula (verified consistent with code)**: `loopNum = n + (n - o) * m`
  - `n = loopInsCnt` (loop template count within the LoopGroup)
  - `o = expandOffset` (the first o loops are not expanded; each counts as 1 copy)
  - `m = expandCnt` (the remaining n-o loops each expand to m+1 copies)
  - Code basis: a6_v3.cc:1703-1727.
- **Collects only the static expansion instance count; does not include the runtime iteration count loopIterCnt** (confirmed by review).
- **Hook point**: Where `TransformLoopGroupInstr` finishes parsing the LoopGroup configuration (A6: a6_v3.cc:1654; A5: a5_v3.cc:2056).
- **Idempotent logging (V4 semantics)**: Collection runs only in the single-operator path, and each op is expanded exactly once. Simple accumulation would already be sufficient. Key-based idempotent logging is retained as a **defense mechanism**:
  1. If collection is enabled on the big-graph path in the future, or op retry/repeat expansion logic is introduced, the count result remains independent of expansion count.
  2. Same-key value conflicts can serve as data anomaly probes.
  Rule: Use `(dev, die, missionId, instStartId, loopGroupInstrId)` as key to write into `map<LoopGroupKey, LoopGroupRecord>` (see 2.1):
   - Duplicate logging hitting the same key with the same value -> automatic deduplication;
   - `instStartId` (SQE identifier) must be in the key: The same LoopGroup instruction can be referenced by SQEs with different args. At this time, n/o/m may differ. This is a "truly different call" and should be counted separately.
   - `loopGroupInstrId` takes the absolute id of the LoopGroup instruction in the instruction space (`startLoopInstrId`; available on site). missionId/instStartId are obtained through `curCcuTask` (queId -> current SQE's `CcuTaskParam`; when multiple SQEs exist in a queue, take the current SQE pointed to by `ccuParamIndexs[queId]`).
   - Defense: When the same key is logged twice with inconsistent values (should not happen in theory), print WARN and keep the first value.
- Report aggregation: `loopGroupNum = loopGroups.size()`, `loopTemplateNum = sum record.n`, `loopNum = sum record.expanded`.
- This layer is not affected by loop merge optimization (`TryProcessMergedLoopA6` / `EmitMergedLoopInstrsV3`) or queue parallelization rewrite (`ApplyCcuQueueParallelizationV3`). **Deriving from final graph nodes is prohibited** (batch merge nodes distort the expansion count).
- No nested loops (`CheckLoopGroupNotSupport` guarantees this); the formula needs no recursion.

### 3.6 mission (missionNum)

- Hook point: `CcuGraphStateV3::InitInstrInfo` (ccu_task_common_v3.cc:307) or the SQE translation entry. Collect `CcuTaskParam.missionId` into the current op bucket's `missionIds` set (deduplicated per (dev,die)).
- Side product: `missionStartInstrId/missionInstrCount` is available at the same point, supporting the 3.2 per-op instruction range metric.

### 3.7 Explicitly Not Collected Items

| Resource | Reason |
|---|---|
| queue | Review explicitly stated not needed |
| loopCtxId (xp) | Review explicitly stated not needed |
| jetty | Other statistics channels already exist |
| halfRTT (wishCnt/totalCnt XN) | Not handled for now |
| PFE / `TransDep.reserve*` reserved resources | Not in phase 1. If needed later, collect using the same address -> id method as 3.3 |

---

## 4. Statistics Report (CSV; V6: Single Per-Op Sectioned Report)

### 4.1 Generation Timing and Location

- Timing: End of `RunChecker`, generated in one pass after all operator graph construction/verification completes.
- Location: Reuses the `DumpV3Manager` output directory; single file `ccu_resource_stats.csv`. **Sectioned by operator** (one section per operator invocation: two `#` header lines + 14-column data rows). Satisfies "each operator invocation counted separately" (hccl_test batch invocations produce one section per call; composite samples with multiple operators are distinguished by section).
- **No run-level/device total aggregated report is generated** (V4 decision: Aggregated data is meaningless for analyzing operator resource usage).
- Simultaneously outputs a summary log (style aligned with `[CHECKER_RUN_SUMMARY]`).

### 4.2 CSV Format Definition (V6: Displayed in Operator Sections + Operator Info Section Header)

Statistics are already isolated per operator invocation. Display is also **sectioned by operator**: Each operator invocation gets one section; sections are separated by blank lines. Each section header is two `#` comment lines (operator identifier + operator display info). **Fields in section header lines use `"; "` as separator instead of comma** (V6.1): This ensures the entire line occupies only the first column in Excel without being split. The remaining cells in the header row are empty, and Excel automatically overflows rightward to display the full operator information. The `#` line prefix is retained so tools like pandas can skip with `comment='#'`. Within a section are the column header + the operator's (dev,die) data rows (14 columns = 5 topology/device columns + 9 resource columns; the operator identifier is moved to the section header and not repeated per row).

```
# ===== op[0]: commName=hccl_world_group; commHash=0; opIter=0 =====
# opInfo: type=AllReduce; alg=CcuMSAllReduceSoleMesh; dataType=INT32; elements=262144; inputBytes=1048576; outputBytes=1048576; reduceType=SUM; rankSize=4
superPodId,serverId,boardId,phyDevId,dieId,channelNum,instructionNum,loopTemplateNum,loopNum,xnNum,gsaNum,ckeNum,msNum,missionNum
0,0,0,1,1,3,235,2,128,69,31,5,256,1
0,0,0,2,1,3,235,2,128,69,31,5,256,1

# ===== op[1]: commName=hccl_world_group; commHash=0; opIter=1 =====
# opInfo: type=AllReduce; alg=CcuMSAllReduceSoleMesh; dataType=INT32; elements=1074791; ...
...
```

**Section header operator info fields** (source `CheckerParam` + `OpDetailTab.algorithmName`; `SetCurrentOpInfo` buckets them into the op after CheckerParam is ready inside CheckOneOp):

| Field | Description | Source |
|---|---|---|
| type | Operator type | `HcclCmdTypeToString(cmdType)` |
| alg | Algorithm name | `OpDetailTab.algorithmName` (first non-empty across ranks; NA if empty) |
| dataType / elements | Data type / element count | `HcclDataTypeToString(dataType)` / `dataCount` |
| inputBytes / outputBytes | Input/output data size (**rank0 view**; for rank-sensitive operators like send/recv/alltoall, myRank=0 is used) | `CalcInputOutputSize` (check_utils) |
| reduceType | reduce type; output only for ALLREDUCE/REDUCE/REDUCE_SCATTER/REDUCE_SCATTER_V; `NA` for others | `HcclReduceOpToString(reduceType)` |
| rankSize | Number of ranks in the communicator | `CheckerParam.rankSize` |

**Intra-section data columns** (14 columns):

| # | Column | Description | Source |
|---|---|---|---|
| 1 | superPodId | Super node Id | DB topology mapping (4.3) |
| 2 | serverId | Server Id | DB topology mapping |
| 3 | boardId | Board Id (board sequence number in POD form; always 0 in server form) | DB topology mapping |
| 4 | phyDevId | Physical device Id (= checker deviceId; globally unique) | Collection container key |
| 5 | dieId | die Id (locates CCU) | Collection container key |
| 6 | channelNum | Number of channels actually referenced by this operator invocation (reference metric; deduplicated) | 3.1 |
| 7 | instructionNum | Number of instructions used by this operator invocation (union of SQE ranges) | 3.2 |
| 8 | loopTemplateNum | Sum of n (loop template count) | 3.5 |
| 9 | loopNum | Sum of (n + (n-o)*m) (static expansion instance count) | 3.5 |
| 10 | xnNum | Deduplicated Xn register count | 3.3 |
| 11 | gsaNum | Deduplicated GSA count; A6 outputs `NA` | 3.3 |
| 12 | ckeNum | Deduplicated ckeId count (excluding mask) | 3.3 |
| 13 | msNum | Deduplicated MS count | 3.4 |
| 14 | missionNum | Deduplicated mission count | 3.6 |

- Section ordering: By (opIter, commName, commHash). Intra-section row ordering: By (superPodId, serverId, boardId, phyDevId, dieId).
- Row coverage: Each section outputs only the (dev,die) pairs actually involved in that op. Ops where a single rank had no task and was skipped (main.cc:228) produce no section.
- Extended data retained in dfx but not yet in CSV: `xnMaxId` (peak watermark), `loopGroupNum`, `instrResidentNum`, channel delta metric set (newChannelIds). If analysis later requires them, columns can be added without changing the collection layer.

### 4.3 superPodId / serverId Mapping (Implemented: checker reads DB directly)

Implementation steps (`dfx/ccu_resource_stats_topo.cc`):

1. Read the `TopoMetaConfig` table (`sim::runtime::Db::GetOneByPred<TopoMetaConfig>`; the checker already has Db access precedent: `dag_graphviz_dump.cc:648`). The table entries store topo yaml filenames; `"ranktable"` indicates ranktable.json mode.
2. Call `sim::runtime::ParseCommConfigData(fileName, nullptr, commConfig)` (exported by runtime_state; already linked by checker; yaml and ranktable.json dual-mode unified handling) to obtain `commConfig.servers[]` (including podId/serverId/serverKey/phyIds).
3. **Primary path (D7 actual-test correction)**: Query the `Device` table to build the `deviceId(=Device.id) -> server_id -> CommConfigServer.serverKey -> (podId, serverId)` mapping. **Pitfall**: The checker's deviceId (Device.id; actually 1-based) and the topology phyIds (Device.physical_id; 0-based) have misaligned metrics. Matching deviceId directly against phyIds causes labels to be misaligned or missing (D7 first-round actual test dev4 -> -1 is exactly this cause).
4. Fallback path: Direct match on `phyIds` (deployment scenarios where deviceId == phyDevId). emplace does not overwrite the primary path.
5. Failure at any step (missing table entry / unreachable file / parse failure) -> empty map + WARN. Both columns output `-1`. Report generation is not blocked.

---

## 5. Feasibility Conclusion

**The plan is feasible**:

1. **Collection occurs only in the single-operator graph construction phase** (V4 decision): This path executes `CheckOneOp -> GenAndCheckGraphV3` independently per op, with recorder Reset before each op (main.cc:221). per-op statistics boundaries are naturally clean. Registers do not leak across ops. Channels are collected by translation-phase reference attribution (V5). The answer to "can operator invocations be counted separately" is **yes, and both typical scenarios are covered**:
   - hccl_test batch invocations: Each invocation has an increasing opIter -> independent OpExecutionKey -> independent statistics bucket.
   - Composite samples with multiple operators: Different operators have different keys (commName/commHash/opIter triple is unique) -> independent statistics buckets. The distinguishing basis is the first-class identity of the input data model, not heuristic splitting.
2. **The big-graph path does not collect**. Through the "BeginOp context gate", zero collection and zero overhead are achieved. Even when both checkers are enabled simultaneously, no duplicate counting occurs. Difficulties such as shared state inside big-graph windows and TaskPosition attribution are all resolved.
3. Register writes have centralized entry points (SetXn/SetGSA/SetCKE). A6 address -> id conversion is already uniformly implemented in the checker, and statistics can reuse it. MS has a single convergence point `GenSliceFromMs` (3.4 plan A has zero call-site changes).
4. The loop formula is consistent with the code implementation (a6_v3.cc:1703). The hook point is in the translation layer, unaffected by merge/parallelization rewrites. Key-based idempotent logging serves as a defense mechanism to ensure the count is independent of expansion count.
5. Topology labels are reachable through direct DB reads. The dfx collection + end-of-run report pattern has an existing example (dag_graphviz_dump).
6. **Known cost**: Pure big-graph runs (`enableNewChecker=off` and `enableBigGraphChecker=on`) produce no statistics data. The statistics feature explicitly depends on the single-operator checker being enabled. When switches conflict, a WARN is printed (see section 6 risk #4).

---

## 6. Risk and Outstanding Issue List (V5 Revision)

| # | Risk / Gap | Level | Countermeasure |
|---|---|---|---|
| 1 | The topology parsing utility library (src/cmd / src/runtime_state) is not linked by checker. Also, `TopoMetaConfig` stores only the yaml filename/`"ranktable"` marker. The absolute path resolution for the topology file under the checker process runtime environment has not been verified. | Medium | Sink into a shared library or implement equivalent logic inside checker. Verify file path reachability before implementation. Fallback to -1 on failure. |
| 2 | `ccu_resource_base_addr_` is currently a mock hard-coded value (storage_manager.cc:757-763; only 2 die slots). This affects A6 address -> id conversion. When dieId>=2, `GetXnAndTypeIdByAddr/GetMSIdByAddr` directly reports an error. A6 multi-die statistics fail instead of degrade. | Medium | Statistics reuse the checker's existing conversion path (same metric as check logic; no new error introduced). On conversion failure, log to the UNKNOWN bucket + WARN; do not abort. Formalizing the base addr is a pre-existing issue and auto-takes-effect when that issue is fixed. |
| 3 | MS statistics plan awaits review decision (3.4 A/B/C). A6 msId's bank/interleave numbering metric must align with hardware specifications. | Medium | Plan A is recommended. Once the metric is confirmed, note it in the report. |
| 4 | Pure big-graph runs produce no statistics data (cost of the V4 collection scope decision). | Medium | When the statistics switch is on but `enableNewChecker=off`, print a WARN. Document dependency in the report header. If big-graph statistics are needed later, extend phase 2 by TaskPosition.operatorId attribution (key-based idempotent logging already provides a fallback against double counting). |
| 5 | The primary instructionNum metric is the per-op SQE range union. instrResidentNum (run-level snapshot) is a reference value. The semantics of both must be described clearly in usage instructions. | Low | Defined in 3.2/4.2; noted in report header comments. |
| 6 | Whether to output a partial report on check failure early exit. | Low | Recommendation: Still output (data is the collected portion); mark partial in the log. Failed op buckets are retained and listed in the summary. |
| 7 | Set/GenSliceFromMs are high-frequency paths; collection hooks have performance overhead. | Low | set insertion + switch default off + first-line short-circuit return when no context exists. During collection, the context is a plain member with no locks. |
| 8 | Reset/gate points are error-prone: The collector's run-level reset must hook near main.cc:508. `BeginOp` must hook inside CheckOneOp after the recorder Reset (main.cc:221). (The reference metric newChannelIds delta snapshot must precede `LoadHcclVmSynthesisData`, main.cc:244.) | **High** | **`BeginOp` must return a RAII guard (destructor = EndOp)**: CheckOneOp has multiple early-return failure paths after BeginOp (main.cc:250-301). A bare EndOp on the failure path is inevitably missed, causing context hang and next-op bucket contamination. RAII covers all exit paths (see section 2 gate mechanism). Unit tests must cover the "op fails midway -> next op bucket not contaminated" scenario. |
| 9 | Dual checker in the same run stacked (main.cc:536-560): V4 big-graph path has zero collection, architecturally eliminating double counting. The remaining risk is someone mistakenly adding BeginOp on the big-graph path in the future. | Low | 3.5 key-based idempotent logging fallback. Big-graph path code comments state "collection prohibited". On same-key value inconsistency, WARN and keep the first value. |
| 10 | Multi-die scenario: A5 device dieId value range differs from A6. Report rows must be generated based on actually observed (dev,die) pairs. | Low | Follow the actual keys in the collection container. |
| 11 | Op failure rerun/retry scenarios (if introduced in the future) cause multiple writes to the same op bucket. | Low | On BeginOp, if the bucket already exists, WARN and choose overwrite or merge (decided at implementation time). |
| 12 | ~~Channel delta metric boundary~~ (V5 resolved): Under batch-invocation scenarios, the delta metric for subsequent ops is always empty, contradicting the per-op premise. | Closed | 3.1 primary metric changed to translation-phase reference collection (`Has/GetChannelInfo` wrappers replace 32 direct lookups). Delta is demoted to a reference metric (newChannelIds; not in main CSV column). |
| 13 | Channel reference collection needs to replace 32 direct `g_allRankChannelInfo` lookups (a5_v3.cc 22 sites, a6_v3.cc 10 sites). The changes are mechanical but numerous. Missing a site causes under-counting. | Low | Reference pattern is uniform (`g_allRankChannelInfo[dev][die]...`). Use grep to verify zero remaining direct lookup sites. Wrapper functions enforce the logging path. Unit tests compare against a golden channel set. |

---

## 7. Implementation Breakdown (Suggested)

1. **M1 dfx collection framework**: `ccu_resource_stats_dfx` (OpKey-bucketed container + run-level Reset hook + `BeginOp` returns RAII guard (destructor = EndOp, covering all CheckOneOp early-return failure paths) + switch + switch conflict WARN).
2. **M2 register collection**: SetXn/SetGSA/SetCKE hooks (short-circuit when no context; A6 address -> id classification then log).
3. **M3 MS collection**: Implement `RecordMs` per the review-selected plan (recommend 3.4-A context injection; shares context with M1 gate).
4. **M4 loop/mission/instruction range collection**: TransformLoopGroupInstr uses key-based idempotent logging (including same-key value conflict defense). InitInstrInfo collects mission/per-op instruction ranges.
5. **M5 channel collection**: Add unified `HasChannelInfo/GetChannelInfo` wrappers (embedding `RecordChannel` reference collection); replace 32 direct lookups in the a5/a6 translation files. BeginOp/EndOp delta logic is retained as the reference metric (newChannelIds).
6. **M6 topology mapping + CSV reporter**: DB reads TopoMetaConfig; `ParseCommConfigData` parses; Device.server_id -> serverKey reverse lookup. Per-op bucket generates a single sectioned `ccu_resource_stats.csv` (section header operator info + 13 data columns; see 4.2) + summary log.
7. **M7 tests**: loop formula unit tests (o=0, o=n-1, m=0 boundaries). Multi-op bucket/deduplication scenarios (hccl_test batch simulation, **focus on verifying channelNum reference metric is non-zero and stable across ops under batch invocations**). Channel reference collection against golden set (verify all 32 direct lookup replacements have no omissions). **Op midway failure -> RAII seals bucket -> next op bucket not contaminated**. A5/A6 dual-version golden CSV. Dual checker simultaneous enable zero-collection verification.
