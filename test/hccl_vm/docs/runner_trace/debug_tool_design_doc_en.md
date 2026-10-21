# CCU Executor Debug Tool Design Document

## 1. Overview

### 1.1 Objective

Build a debug replay tool for the CCU Executor module. The tool collects trace data during instruction execution and provides step-by-step replay on a front-end UI. Each instruction displays the following information:
- Instruction information (type, parameters, owning CCU)
- Debug information during instruction execution (key variable values, execution path)
- CCU resource changes (before and after comparison of XN/GSA/CKE/MS/Channel)

### 1.2 Current Code Analysis

**CCU Executor Module Architecture** (`src/plugin/solver/virtual_runtime/ccu_executor/`):

```
CcuExecutorBase (abstract base class)
├── Parser()          // Parse instruction fields
├── Run()             // Execute instruction logic
├── Process()         // Resource operations (implemented by some instructions)
└── Describe()        // Generate instruction description string

CcuSimulator (execution engine)
├── Execute()         // Main loop: serial instruction execution
├── ExecuteInstr()    // Single instruction execution
├── ExecuteLoop()     // Loop repeated execution
└── ExecuteLoopGroup()// LoopGroup expanded execution

CcuResourceManager (resource management - singleton)
├── XN[]    // General-purpose registers (V1:3072, V2:4096)
├── GSA[]   // Address registers (V1:3072, V2:4096)
├── CKE[]   // Synchronization signals   (1024)
├── MS[]    // Memory Slice 4K (1536)
├── Channel[] // Communication channels (128)
└── InstrSpace // Instruction space
```

**Instruction Categories (4 major types)**:

| Type | Subtypes (V1) | Subtypes (V2 new) | Operated Resources |
|------|-----------|---------------|-----------|
| **Load** | LoadSqeArgsToGsa, LoadSqeArgsToXn, LoadImdToGsa, LoadImdToXn, LoadGsaXn, LoadGsaGsa, LoadXX | LoadImdToX, LoadX, StoreX, ClearX, Nop, Load, Store, Add, Sub, Mul, And, Or, Not, Xor, Shl, Shr, Popcnt | XN, GSA |
| **Trans** | LocMem↔LocMem, LocMem↔LocMS, LocMem↔RmtMem, LocMS↔LocMS, LocMS↔RmtMS, LocMS↔RmtMem, RmtMem↔LocMem, RmtMS↔LocMem, RmtMS↔LocMS, SyncCke, SyncGsa, SyncXn | TransMem, SyncXnWt, SyncAt | MS, GSA, XN, Channel, CKE |
| **Control** | Loop, LoopGroup, SetCke, ClearCke, Jump | Wait, Fence | CKE, LoopEngine, XN |
| **Reduce** | ReduceAdd, ReduceMax, ReduceMin | (same as V1) | MS, CKE |

**Resource Ownership**:
```
Rank[0..N]
  └── CCU[0..1]  (die0, die1)
       ├── XN[0..max]
       ├── GSA[0..max]
       ├── CKE[0..1023]
       ├── MS[0..1535]  (each 4KB)
       ├── Channel[0..127]
       └── InstrSpace (instruction list)
```

---

## 2. Replay Approach Analysis: Log Replay vs Trace Replay

### 2.1 Approach Comparison

| Dimension | Log Replay | Trace Replay (Structured Trace) |
|------|----------------------|-------------------------------|
| **Data Collection** | Intercept `HCCL_VM_DEBUG/INFO/TRACE` log output | Instrument the execution engine and collect structured data |
| **Data Format** | Unstructured text strings | JSON/MsgPack structured data |
| **Information Completeness** | Only includes information printed in logs; omissions occur | Collect complete resource snapshots without omissions |
| **Parsing Complexity** | Require regular expression parsing; fragile and costly to maintain | Deserialize directly; no parsing cost |
| **Storage Size** | Verbose text; large size | Compact binary encoding; incremental recording supported |
| **Replay Precision** | Cannot precisely restore resource states | Precisely restore the complete state at any instruction point |
| **Front-end Integration** | Require an additional log parsing service | The front end loads JSON/MsgPack and renders directly |
| **Performance Overhead** | Low (text output only) | Medium (requires serialization and snapshots) |
| **Extensibility** | New instructions require regular expression updates | New instructions only require trace structure extension |
| **Industry References** | GDB textual trace | Chrome DevTools Timeline, LLVM Execution Trace, rr debugger |

### 2.2 Conclusion: Adopt Trace Replay Approach

**Rationale**:
1. **Precision**: The core requirement of CCU debugging is observing resource changes. Logs cannot guarantee complete collection of XN/GSA/CKE/MS values.
2. **Front-end Friendly**: HVRM Insight already has a mature data loading pipeline (Worker + JSON). Structured trace data integrates directly.
3. **Incremental Recording**: Record only changed resource fields (delta) instead of full snapshots. This controls storage size.
4. **Step-by-step Replay**: Trace data natively supports locating by instruction sequence number. Log replay cannot achieve precise step-by-step control.

---

## 3. Trace Data Structure Design

### 3.1 Design Principles

Reference industry practices:
- **Chrome DevTools Trace Event Format**: Layered structure. Each event includes category, name, timestamp, and args.
- **LLVM Execution Trace**: Instruction-level trace with register snapshots.
- **rr (Record and Replay) debugger**: Event stream with key checkpoints.

Consider local code characteristics:
- Instructions naturally fall into 4 major categories (Load/Trans/Control/Reduce). Each category operates on different resources.
- CCU resources are independent per die. Trace data must associate (rankId, dieId).
- Instructions execute serially. Trace data is naturally ordered.

### 3.2 Overall Architecture

#### 3.2.1 Execution Model Review (Actual Code Logic)

CCU execution does not run independently per CCU. Instead, the system uses **global interleaved scheduling**. The core scheduling logic resides in `SequentialExecutor::Execute()`:

```
while (HasTask()) {                          // Outer round loop
    for (rank in allRanks) {                 // Iterate all ranks
        for (stream in rankStreams) {        // Iterate each stream of each rank
            while (stream has tasks) {
                ret = ExecuteOneTask(stream.front());
                if (ret == HOLD_CMD) break;  // waitCKE blocks -> skip to next stream
                stream.pop();                // Task complete -> dequeue
            }
        }
    }
}
```

When a CCU SQE reaches a waitCKE instruction and the CKE condition is not met, `TaskCcuGraph()` returns `HOLD_CMD`. The SQE task **remains in the queue**. The outer loop continues to process the next CCU. After another CCU executes SetCke and sets the corresponding CKE, the next outer loop iteration retries this SQE. The CKE condition is then met and execution continues.

Key facts:
- **CCU resources are independent**: Each CCU (rankId, dieId) owns independent XN/GSA/CKE/MS/instruction space.
- **One CCU has one Simulator**: Multiple SQEs on the same (rankId, dieId) reuse the same `CcuSimulator` instance (the system resets the instruction pointer through `Init()`).
- **SQE describes an execution range**: Each SQE task contains missionId, start instruction ID, instruction count, and input parameter list.
- **Global interleaved execution**: The actual execution order is R0:D0 -> R0:D1 -> R1:D0 -> R1:D1 -> R0:D0 -> ... (all CCUs take turns).

#### 3.2.2 Trace Overall Architecture

Trace must be **global**. Trace records the interleaved execution sequence of all CCUs, not per-CCU independent records.

```
CcuTraceRun (one complete virtual runtime execution, covering all ranks and CCUs)
├── runMetadata              // Run-level metadata (operator name, rank count, CCU version, etc.)
├── ccuRegistry[]            // CCU registry: all CCUs participating in execution (rankId, dieId)
│   └── CcuIdentity           //   Identity and initial resource snapshot of each CCU
├── sqeTaskRegistry[]        // SQE task registry: all SQE tasks (referenced by entries)
│   └── CcuSqeTask            //   missionId, instruction range, input parameters, owning CCU
│
│  ┌─────────── Static Configuration Layer (per CCU, unchanged after initialization) ───────────┐
├── instrSpaces[]            // Instruction space (per CCU, pure static snapshot)
│   └── CcuInstrSpace         //   rankId + dieId + instructions[instrId, describe]
├── channelSpaces[]          // Channel mapping table (per CCU, pure static snapshot)
│   └── CcuChannelSpace       //   rankId + dieId + channels[channelId, remoteRankId, remoteDieId]
│  └──────────────────────────────────────────────────────────┘
│
│  ┌─────────── Dynamic Execution Layer (changes at runtime) ──────────────────┐
├── globalEntries[]          // Global instruction execution sequence (ordered by actual interleaving)
│   └── CcuTraceEntry         //   Each entry indexes the instruction space through (rankId, dieId, instrId)
│       ├── (rankId, dieId,    //   Triple indexes the instruction description in the instruction space
│       │    instrId)           //
│       ├── sqeRef              // -> SQE task in sqeTaskRegistry
│       ├── globalSeqId         //   Global execution sequence number (reflects actual interleaving order)
│       ├── execRound           //   Round number of the outer while(HasTask()) loop
│       ├── resourceDelta       //   Resource changes of this CCU (XN/GSA/CKE/MS, excluding Channel)
│       ├── crossCcuChanges     //   Cross-CCU resource changes
│       ├── waitInfo            //   CKE Wait spin information
│       ├── errorInfo           //   Execution failure information
│       └── detail.args         //   Instruction-specific dynamic parameters (runtime values; may differ across rounds for the same instruction)
├── ccuFinalSnapshots{}      // Final resource snapshot of each CCU (key: rankId_dieId)
│  └──────────────────────────────────────────────────────────┘
└── runSummary               // Run-level summary statistics
```

**Static/Dynamic Separation Design**:

| Data | Storage Location | Property | Description |
|------|---------|------|------|
| Instruction Description (Describe) | `instrSpaces[]` | **Static** | Store only one copy per instruction. Does not change across Loop iterations. |
| Instruction Category/Name | `instrSpaces[]` | **Static** | Determined by instruction bytes. Fixed and unchanging. |
| Channel Mapping Table | `channelSpaces[]` | **Static** | Unchanged after InitChannelInfo initialization. Stored alongside instrSpaces. |
| Runtime Parameters | `detail.args` | **Dynamic** | Actual values after Loop offset, SQE parameter values, etc. |
| Resource Changes | `resourceDelta` | **Dynamic** | Resource before/after for each execution (excluding Channel). |
| Execution Context | `context` | **Dynamic** | Loop round, offset, etc. |

**The same instruction executes multiple times in a Loop**: The instruction space contains only 1 record. The trace produces N entries. Each entry has different `detail.args`, `resourceDelta`, and `context`.

### 3.3 Data Structure Detailed Definitions

#### 3.3.1 Top Level: CcuTraceRun, CCU Registry, SQE Registry

```cpp
// ===== One complete virtual runtime execution =====
// Covers all ranks, all CCUs, and all SQEs in the global execution process
struct CcuTraceRun {
    CcuRunMetadata runMetadata;                        // Run-level metadata
    std::vector<CcuIdentity> ccuRegistry;              // CCU registry
    std::vector<CcuSqeTask> sqeTaskRegistry;           // SQE task registry
    
    // ===== Static configuration layer (per CCU, unchanged after initialization) =====
    std::vector<CcuInstrSpace> instrSpaces;            // Instruction space (per CCU, pure static snapshot)
    std::vector<CcuChannelSpace> channelSpaces;        // Channel mapping table (per CCU, pure static snapshot)
    
    // ===== Dynamic execution layer (changes at runtime) =====
    std::vector<CcuTraceEntry> globalEntries;          // Global instruction execution sequence (interleaved and ordered)
    std::vector<CcuTraceNonCcuEntry> nonCcuEntries;    // Non-CCU task records
    std::map<std::string, CcuResourceSnapshot> ccuFinalSnapshots;  // Final snapshot of each CCU (dynamic resources only)
    CcuRunSummary runSummary;                          // Run-level summary statistics
};

// ===== Run-level metadata =====
struct CcuRunMetadata {
    uint32_t traceFormatVersion;                       // Trace format version number (current = 1)
    std::string algorithmName;                         // Operator name (for example, "AllReduce")
    std::string operatorName;                          // Operation name (for example, "RingAllReduce_Step1")
    uint32_t rankSize;                                 // Total rank count
    uint32_t diePerRank;                               // Die count per rank (fixed = 2)
    RunnerCcuVersion ccuVersion;                       // CCU microcode version
    uint64_t runTimestampNs;                           // Run start timestamp
    uint32_t totalSqeTaskCount;                        // Total SQE task count
    uint32_t totalExecRounds;                          // Total round count of outer while(HasTask()) loop
};

// ===== CCU identity (registry entry) =====
// Each participating CCU has exactly one record in the registry
struct CcuIdentity {
    int32_t rankId;                                    // Rank number
    uint16_t dieId;                                    // Die number (0 or 1)
    RunnerCcuVersion ccuVersion;                       // CCU microcode version
    CcuResourceSnapshot initialSnapshot;               // Initial resource snapshot of this CCU
    std::string ccuKey() const {                       // Unique identifier key: "rankId_dieId"
        return std::to_string(rankId) + "_" + std::to_string(dieId);
    }
};

// ===== SQE task (registry entry) =====
// Each SQE task has exactly one record in the registry. CcuTraceEntry references it through sqeTaskId.
struct CcuSqeTask {
    uint32_t sqeTaskId;                                // SQE task ID (unique within Run, incrementing from 0)
    int32_t rankId;                                    // Owning rank
    uint16_t dieId;                                    // Owning die
    uint8_t missionId;                                 // SQE mission number (CcuTask.missionId)
    uint16_t instStartId;                              // Start instruction ID (CcuTask.instStartId)
    uint16_t instCnt;                                  // Instruction count (CcuTask.instCnt)
    uint32_t key;                                      // Task key (CcuTask.key)
    std::vector<uint64_t> args;                        // SQE input parameter list (CcuTask.args[], up to 13)
    uint64_t simulatorPtr;                             // CcuSimulator instance pointer (multiple SQEs on the same CCU share the same instance)
    uint32_t firstExecRound;                           // First outer round number when this task was executed
};

// ===== Run-level summary =====
struct CcuRunSummary {
    uint32_t totalInstrExecuted;                       // Total executed instruction count
    uint32_t totalFailedInstr;                         // Failed instruction count
    uint32_t totalCkeWaitSpins;                        // Total CKE wait spin count
    uint32_t totalHoldEvents;                          // Total HOLD event count (interruptions caused by waitCKE)
    std::map<std::string, uint32_t> instrCountByCategory;  // Statistics by instruction category
    std::map<std::string, uint32_t> instrCountByCcu;       // Statistics by CCU (key: "rankId_dieId")
    uint64_t totalMsBytesTransferred;                  // Total MS bytes transferred
};
```

#### 3.3.2 Instruction Space: CcuInstrSpace (per CCU, pure static snapshot)

The instruction space stores instructions independently per CCU. It contains static information for all instructions of this CCU (instruction ID + precomputed Describe() output).
The instruction space is **fully decoupled** from trace dynamic parameters. Trace entries index the instruction space only through the `(rankId, dieId, instrId)` triple.

```cpp
// Single instruction in the instruction space (pure static)
struct CcuInstrSpaceEntry {
    uint16_t instrId;                        // Instruction ID (index in the instruction space)
    std::string instrDescribe;               // Precomputed Describe() output (static, no runtime values)
};

// Instruction space of a single CCU
struct CcuInstrSpace {
    int32_t rankId;                          // Owning rank
    uint16_t dieId;                          // Owning die
    std::vector<CcuInstrSpaceEntry> instructions;  // All instructions of this CCU
};
```

**Collection timing**: Before trace starts, iterate the instruction space of each CCU. Call `Parser()` + `Describe()` for all instructions and cache the results.

**Front-end usage**: After the front end loads the instruction space, the left sidebar displays the complete instruction list (pure static). During replay, the system highlights the corresponding row through `(rankId, dieId, instrId)`.

#### 3.3.3 Common Layer: CcuTraceEntry

Each entry records **one instruction execution on one CCU**. The system retrieves the instruction description by indexing the instruction space through the `(rankId, dieId, instrId)` triple.

```cpp
// Trace entry for a single instruction (one point on the global timeline)
struct CcuTraceEntry {
    // === Global positioning information ===
    uint32_t globalSeqId;                   // Global execution sequence number (incrementing from 0, reflects actual interleaving order)
    uint32_t execRound;                     // Round number of outer while(HasTask()) loop (incrementing from 0)

    // === CCU ownership + instruction index (triple indexes the instruction space) ===
    int32_t rankId;                         // Owning rank
    uint16_t dieId;                         // Owning die
    uint32_t instrId;                       // Instruction ID; together with (rankId, dieId) indexes instrSpaces
    // The system locates the instruction description in instrSpaces using (rankId, dieId, instrId)

    // === SQE ownership ===
    uint32_t sqeTaskId;                     // References CcuSqeTask in sqeTaskRegistry

    // === Instruction category (for statistical aggregation only, 1 byte) ===
    CcuInstrCategory category;              // Instruction major category: Load/Trans/Control/Reduce

    // === Execution state ===
    CcuExecState execState;                 // State after execution

    // === Execution context ===
    CcuExecutionContext context;            // Loop/Jump context information

    // === Resource changes (Delta) ===
    CcuResourceDelta resourceDelta;         // Resource changes on this CCU caused by this instruction
    CcuCrossCcuChanges crossCcuChanges;     // Cross-CCU resource changes caused by this instruction (if any)

    // === CKE Wait spin information ===
    CcuWaitInfo waitInfo;                   // Wait spin merge information (valid only during waitCKE)

    // === Execution failure information ===
    CcuErrorInfo errorInfo;                 // Failure information (valid only when execState == EXEC_FAIL)

    // === Instruction-specific details (polymorphic) ===
    std::unique_ptr<CcuInstrTraceDetail> detail;  // Instruction-specific debug information
};

// Instruction category enumeration
enum class CcuInstrCategory : uint8_t {
    LOAD = 0,       // Load/store/arithmetic
    TRANS = 1,      // Data transfer
    CONTROL = 2,    // Control flow
    REDUCE = 3,     // Reduce operations
};

// Execution context (Loop/Jump related, excluding SQE information -- SQE is referenced through sqeTaskId)
// GSA address offset calculation formula inside Loop (from CcuExecutorBase::UpdateAddress):
//   finalOffset = (loopExtendNum * gsaOffset << addrExpandCoef)
//               + (curLoopCnt * iterStepGSA << addrExpandCoef)
struct CcuExecutionContext {
    bool inLoop;                            // Whether executing inside a Loop
    uint16_t loopRound;                     // Current Loop iteration round
    uint16_t loopExtendIndex;               // Loop expansion index

    // === Loop offset parameters (valid only when inLoop=true) ===
    // GSA address offset parameters
    uint32_t gsaOffset;                     // GSA base offset coefficient (GetGSAOffset)
    uint64_t iterStepGSA;                   // GSA step per iteration round (GetLoopIterStepGSA)
    uint32_t loopExtendNum;                 // Expansion count (GetLoopExtendNum)
    uint32_t curLoopCnt;                    // Current iteration count (GetCurLoopCnt)
    uint64_t gsaAddrOffset;                 // Computed final GSA address offset value
    uint16_t addrExpandCoef;                // Address expansion coefficient (2nd parameter of UpdateAddress)

    // Resource ID offset parameters
    uint16_t msOffset;                      // MS ID offset (GetLoopMsOffset)
    uint16_t ckeOffset;                     // CKE ID offset (GetLoopCKEOffset)
    uint16_t xnIdOffset;                    // XN ID offset (GetLoopXnIdOffset)
};

// CKE Wait spin merge information
// When an instruction spins waiting because the CKE condition is not met, multiple ExecuteInstr calls
// are merged into one CcuTraceEntry instead of producing multiple duplicate records.
struct CcuWaitInfo {
    bool hadWait;                           // Whether CKE waiting occurred
    uint32_t waitRetryCount;                // Number of spin wait retries
    uint16_t waitCKEId;                     // CKE ID being waited on
    uint16_t waitCKEMask;                   // CKE mask being waited on
    uint16_t ckeValueOnFirstCheck;          // CKE value at first check
    uint16_t ckeValueOnPass;                // CKE value when condition is finally met
    uint64_t waitDurationNs;                // Total wait duration (nanoseconds)
};

// Execution failure information
struct CcuErrorInfo {
    bool hasError;                          // Whether an error occurred
    std::string errorMessage;              // Error description (from HCCL_VM_ERROR log)
    std::string failPhase;                  // Failure phase: "Parser" / "Run" / "Process"
    CcuResourceSnapshot failSnapshot;       // Complete resource snapshot at failure time (for post-mortem analysis)
};

// Cross-CCU resource changes
// When an instruction operates on remote CCU resources (such as SetRmtCKE, TransLocMSToRmtMS),
// record the impact on the remote CCU.
struct CcuCrossCcuChanges {
    bool hasCrossCcuChange;                 // Whether cross-CCU changes exist
    std::vector<CcuRemoteCkeChange> remoteCkeChanges;   // Remote CKE changes
    std::vector<CcuRemoteMsChange> remoteMsChanges;     // Remote MS changes
    std::vector<CcuRemoteMemChange> remoteMemChanges;   // Remote memory changes
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
    std::vector<uint8_t> dataAfter;         // Record after only (remote before is covered by the remote trace)
};

struct CcuRemoteMemChange {
    int32_t remoteRankId;
    uint64_t remoteAddr;
    uint64_t length;
    std::vector<uint8_t> dataAfter;
};
```

#### 3.3.4 Resource Layer: CcuChannelSpace & CcuResourceSnapshot & CcuResourceDelta

```cpp
// ===== Channel space (per CCU, pure static snapshot) =====
// The Channel mapping table, like the instruction space, is a CCU-specific static resource:
//   - Unchanged after InitChannelInfo() initialization
//   - Instructions only read it (through GetRmtCcu()) during execution, never modify it
//   - Stored separately from dynamic resources such as XN/GSA/CKE/MS
struct CcuChannelSpace {
    int32_t rankId;
    uint16_t dieId;
    std::vector<CcuChannelRecord> channels;  // channelId -> (remoteRankId, remoteDieId)
};

// CCU resource snapshot (includes dynamic resources only: XN/GSA/CKE/MS)
// Note: Channel has been moved to CcuChannelSpace (static configuration, alongside CcuInstrSpace)
struct CcuResourceSnapshot {
    std::vector<CcuXnRecord> xnRecords;      // XN register snapshot
    std::vector<CcuGsaRecord> gsaRecords;    // GSA register snapshot
    std::vector<CcuCkeRecord> ckeRecords;    // CKE signal snapshot
    std::vector<CcuMsRecord> msRecords;      // MS memory slice snapshot (optional, large size)
};

// Resource change delta (records only the changed portion)
// Note: The Channel table is static configuration and does not change, so channelChanges is not included
struct CcuResourceDelta {
    std::vector<CcuXnChange> xnChanges;      // XN change list
    std::vector<CcuGsaChange> gsaChanges;    // GSA change list
    std::vector<CcuCkeChange> ckeChanges;    // CKE change list
    std::vector<CcuMsChange> msChanges;      // MS change list
};

// ========== Resource records (for snapshots) ==========

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
    std::vector<uint8_t> data;  // 4KB data
};

struct CcuChannelRecord {
    uint16_t channelId;
    int32_t remoteRankId;
    uint16_t remoteDieId;
};

// ========== Resource changes (for Delta) ==========

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
    uint64_t offset;              // Change start offset
    uint32_t length;              // Change length
    std::vector<uint8_t> dataBefore;  // Data before change
    std::vector<uint8_t> dataAfter;   // Data after change
};

// Note: The Channel table is static configuration and does not change, so no CcuChannelChange structure exists
```

#### 3.3.5 Detail Layer: CcuInstrTraceDetail (runtime dynamic parameters only)

Each instruction type has its own **runtime dynamic parameters**. The code contains 30+ instruction types (approximately 20 in V1 + approximately 10+ in V2).
Parameters differ significantly across instruction types. The design uses a **flat key-value map** instead of polymorphic inheritance to reduce development and maintenance cost.

**Key distinction**: `CcuInstrTraceDetail.args` **stores only runtime dynamic parameters**, meaning values that may differ for the same instruction across different Loop rounds or different SQE tasks.
Static information (such as instruction descriptions and fixed parameter IDs) is stored in the instruction space and is not duplicated here.

```cpp
// Instruction-specific trace detail (runtime dynamic parameters only)
struct CcuInstrTraceDetail {
    std::string typeName;                     // Instruction type name (for example, "LoadSqeArgsToXn")

    // Runtime dynamic parameters (key-value form)
    // Records only values that change with Loop iterations or SQE tasks
    // Examples:
    //   LoadSqeArgsToXn:
    //     {"sqeArgValue": "0x1000"}           // SQE parameter value may differ across tasks
    //
    //   TransLocMSToLocMem (inside Loop):
    //     {"resolvedLocMSId": "15",           // Actual MS ID after offset
    //      "resolvedLocGSAId": "103",         // Actual GSA ID after offset
    //      "resolvedLocXnId": "52"}           // Actual XN ID after offset
    //
    //   ReduceAdd:
    //     {"reduceOp": "Add", "dataType": "FP32"}
    std::map<std::string, std::string> args;
    // Note: does not include describeText (instruction descriptions are stored in the instruction space, indexed by instrId)
};
```

**Static vs Dynamic Division**:
| Information Type | Storage Location | Property | Description |
|----------|---------|------|------|
| Instruction Description | `instrSpaces[].instructions[].instrDescribe` | **Static** | Precomputed Describe() output |
| Instruction Category/Name | `instrSpaces[].instructions[]` | **Static** | Determined by instruction bytes |
| CKE Wait Information | `CcuTraceEntry.waitInfo` | **Dynamic** | waitCKEId/mask, spin count, etc. |
| CKE Resource Changes | `CcuResourceDelta.ckeChanges` | **Dynamic** | CKE before/after values |
| XN/GSA/MS Changes | `CcuResourceDelta.xnChanges` etc. | **Dynamic** | Resource before/after values |
| Loop Offset Context | `CcuExecutionContext` | **Dynamic** | loopRound, offset, etc. |
| Instruction-specific Dynamic Parameters | `CcuInstrTraceDetail.args` | **Dynamic** | Values after runtime resolution (after offset, SQE parameters, etc.) |

### 3.4 Data Serialization Format

Use **JSON** as the serialization format (consistent with the existing data format of the Insight tool). The format can later be upgraded to MsgPack/FlatBuffers to reduce size.

JSON Schema overview:
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

## 4. Collection Instrumentation Point Design

### 4.1 Instrumentation Locations

Insert one collection point before and one after instruction execution in `CcuSimulator::ExecuteInstr()`. **Key point: CKE Wait spin merging**.

In `WaitCkeProcess`, when the CKE condition is not met, the system sets `waitCKE_=true` and `ExecuteInstr` returns false. The upper-level `Execute()` calls the same instruction again. Without merging, one Wait instruction could produce hundreds of duplicate trace entries.

**Solution**: The collector detects consecutive repeated execution of the same instruction (CKE Wait spin) and merges it into one trace entry. The collector records the spin count.

```cpp
bool CcuSimulator::ExecuteInstr(uint16_t curInstrId)
{
    auto &ccuResMgr = CcuResourceManager::GetInstance();
    auto instrData = ccuResMgr.GetInstrData(rankId_, dieId_);

    // (1) Capture pre-execution snapshot
    auto snapshotBefore = CcuTraceCollector::CaptureResourceSnapshot(rankId_, dieId_);

    auto executor = CcuExecutorFactory::MakeCcuExecutorInstance(...);
    executor->Parser();
    executor->Run();

    // (2) CKE Wait spin detection and merging
    if (waitCKE_) {
        // This instruction returns false due to CKE waiting. The collector records one spin.
        CcuTraceCollector::RecordWaitSpin(rankId_, dieId_, curInstrId,
                                           snapshotBefore);
        // Do not record a trace entry. Wait until CKE is met, then record the final merged result.
        return false;
    }

    // (3) CKE condition met (or this instruction does not involve CKE waiting). Capture post-execution snapshot.
    auto snapshotAfter = CcuTraceCollector::CaptureResourceSnapshot(rankId_, dieId_);
    auto detail = executor->CollectTraceDetail();

    // (4) Retrieve cross-CCU changes (CcuResourceManager intercepts and records these in real time during instruction execution, see Section 4.4)
    auto crossChanges = CcuTraceCollector::ConsumeCrossCcuChanges(rankId_, dieId_);

    // (5) Merge CKE Wait spin information (if spins occurred before). Record the final trace entry.
    auto waitInfo = CcuTraceCollector::FinalizeWaitInfo(rankId_, dieId_, curInstrId,
                                                         snapshotBefore);

    // (6) Detect execution failure
    auto errorInfo = CcuTraceCollector::CaptureErrorInfo(rankId_, dieId_, executor.get());

    // (7) Compute delta and record trace entry (including wait, error, and cross-CCU information)
    // SequentialExecutor passes globalSeqId, execRound, and sqeTaskId to CcuTraceCollector (see Section 4.3)
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

### 4.2 CcuExecutorBase Extension

Add a new `CollectTraceDetail()` virtual method to the base class. Each subclass implements it as needed:

```cpp
class CcuExecutorBase {
public:
    // New: collect instruction-specific trace detail (flat key-value structure)
    // Default implementation: includes only typeName and Describe() output
    virtual CcuInstrTraceDetail CollectTraceDetail() {
        CcuInstrTraceDetail detail;
        detail.typeName = "Unknown";
        detail.describeText = Describe();
        return detail;
    }
};
```

Implementation examples for each Executor subclass (populate instruction-specific parameters into the args map):

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
    // Note: CKE synchronization information is already recorded in CcuWaitInfo and CcuResourceDelta. No duplication needed.
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

### 4.3 Global Context Management

Global context information such as `globalSeqId`, `execRound`, and `currentSqeTaskId` is available only at the `SequentialExecutor` level. The system must pass it down to `CcuTraceCollector`.

```cpp
// SequentialExecutor::Execute() manages global context
while (HasTask()) {
    CcuTraceCollector::BeginRound(execRound);    // Set current round

    for (rankTasks : allRankTaskQueues_) {
        for (streamTasks : rankTasks) {
            while (!streamTasks.empty()) {
                auto task = streamTasks.front();
                if (task.taskType == CCU_GRAPH) {
                    // Register SQE task and obtain sqeTaskId
                    uint32_t sqeTaskId = CcuTraceCollector::RegisterSqeTask(
                        task.rankId, task.dieId, task.missionId,
                        task.instStartId, task.instCnt, task.args);
                    CcuTraceCollector::SetCurrentSqeTaskId(sqeTaskId);
                }
                CcuTraceCollector::BeginGlobalStep();  // Atomically increment globalSeqId
                auto ret = ExecuteOneTask(task);
                if (ret == HOLD_CMD) break;
                streamTasks.pop();
            }
        }
    }
    execRound++;
}
```

`CcuSimulator::ExecuteInstr()` reads these values through `CcuTraceCollector::GetCurrentGlobalContext()`.

### 4.4 Cross-CCU Change Collection (ResourceManager Interception)

Cross-CCU changes (such as `SetRmtCKESignal`, `TransLocMSToRmtMS`, etc.) occur in the `Process()` method of the Executor and operate on remote CCU resources. Intercepting directly in `CcuResourceManager` Update methods is more reliable than comparing remote CCU states after the fact:

```cpp
// CcuResourceManager::UpdateCkeValue() intercepts cross-CCU changes
void CcuResourceManager::UpdateCkeValue(int rankId, int dieId, uint16_t ckeId, uint16_t value) {
    uint16_t oldValue = GetCkeValue(rankId, dieId, ckeId);

    // Detect whether this is a cross-CCU operation
    auto [execRank, execDie] = CcuTraceCollector::GetCurrentExecutingCcu();
    if (rankId != execRank || dieId != execDie) {
        CcuTraceCollector::RecordCrossCcuCkeChange(
            rankId, dieId, ckeId, oldValue, value, execRank, execDie);
    }

    // Perform the actual update
    ckeData_[rankId][dieId][ckeId] = value;
}
```

The same approach applies to `UpdateXnValue()`, `UpdateGsaValue()`, `TransMSToMS()`, and other methods involving remote CCUs.

The collector buffers intercepted cross-CCU changes. `CcuSimulator::ExecuteInstr()` consumes them through `ConsumeCrossCcuChanges()` when recording trace entries.

### 4.5 Resource Snapshot Collection

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

    // Global context management (called by SequentialExecutor layer)
    static void BeginRound(uint32_t execRound);
    static void BeginGlobalStep();
    static uint32_t RegisterSqeTask(int rankId, int dieId, uint8_t missionId,
                                     uint16_t instStartId, uint16_t instCnt,
                                     const std::vector<uint64_t>& args);
    static void SetCurrentSqeTaskId(uint32_t sqeTaskId);
    static CcuGlobalContext GetCurrentGlobalContext();

    // Cross-CCU change interception (called by CcuResourceManager)
    static void RecordCrossCcuCkeChange(int rankId, int dieId, uint16_t ckeId,
                                         uint16_t oldValue, uint16_t newValue,
                                         int execRank, int execDie);
    static std::pair<int,int> GetCurrentExecutingCcu();
    static CcuCrossCcuChanges ConsumeCrossCcuChanges(int rankId, int dieId);
};
```

### 4.6 Performance Optimization Strategies

#### 4.6.1 Collection-side Optimization

1. **Enable on demand**: Control collection through the environment variable `HCCLVM_ENABLE_CCU_TRACE=1`.
2. **Incremental recording**: Record only resource fields with non-zero changes (delta). Do not store unchanged resources.
3. **Tiered MS data collection** (controlled through `HCCLVM_CCU_TRACE_MS_LEVEL`):
   - `none` (default): Do not collect MS data. Record only MS ID and length.
   - `digest`: Collect hash digests of MS data (16 bytes) for integrity verification.
   - `range`: Record only modified offset ranges (offset + length + before/after).
   - `full`: Record complete 4KB (use only when data comparison is needed).
4. **Deferred writing**: Buffer in memory first. Write to file in batch after execution completes.
5. **Optional full snapshots**: Enable full snapshot mode through `HCCLVM_CCU_TRACE_FULL_SNAPSHOT=1`.

#### 4.6.2 Loop Folding and Sampling

Loop execution can cause trace data explosion (one Loop with 1000 rounds x N instructions = thousands of entries). Use folding + sampling strategies to control data volume:

```
Environment variable control:
  HCCLVM_CCU_TRACE_LOOP_MODE=fold|sample|full

fold mode (default):
  Fold multiple iterations of the same Loop into one summary record:
  +--------------------------------------------------+
  | CcuLoopFoldEntry                                |
  | +-- loopInstrId: 5                              |
  | +-- totalRounds: 1000                           |
  | +-- firstRoundEntries: [...]   // Complete first-round trace |
  | +-- lastRoundEntries: [...]    // Complete last-round trace  |
  | +-- resourceStats:                              |
  | |   +-- xnWriteCount: {5: 1000, 8: 500}         |
  | |   +-- ckeToggleCount: {3: 2000}               |
  | |   +-- avgMsBytesPerRound: 4096                |
  | +-- anomalyRounds: [42, 87]  // Anomaly round numbers |
  +--------------------------------------------------+
  The front end displays a folded view by default. Click "Expand" to view details of any round.

sample mode:
  Record only the first round + last round + sampling every N rounds (N is controlled through HCCLVM_CCU_TRACE_SAMPLE_RATE, default 10)

full mode:
  Record all rounds (use only for small-scale debugging; large data volume)
```

Anomaly round detection rules:
- The instruction count in this round does not match the first round.
- The resource change pattern in this round differs significantly from the first round (different delta field sets).
- This round contains EXEC_FAIL.
- The CKE Wait count in this round is abnormally high (more than 2x the first round).

#### 4.6.3 Front-end Rendering Optimization

When the trace entry count exceeds 10,000, the front end requires special handling:

1. **Virtual scrolling**: Use virtual scroll for the instruction list (for example, `vue-virtual-scroller`). Render only visible rows.
2. **Lazy loading**: Load only entry summary information initially (seqId/instrId/name/category). Load the complete detail only when the user clicks.
3. **Paginated loading**: For large datasets, load in chunks of 1000 entries per page. The front end provides paginated browsing.
4. **Resource heatmap downsampling**: In the resource overview panel, group and aggregate by range when resource counts are too high.

### 4.7 Non-CCU Task Trace Records

`SequentialExecutor::Execute()` handles task types beyond `CCU_GRAPH`, including `NOTIFY_RECORD/NOTIFY_WAIT/REDUCE/MEM_CPY/AIV_GRAPH`. `NotifyWait` also returns `HOLD_CMD` and affects the global scheduling flow. Without recording these in the trace, the front-end replay would show "CCU instructions suddenly jump to the next round."

Add simplified records for non-CCU tasks in `globalEntries[]`:

```cpp
struct CcuTraceNonCcuEntry {
    uint32_t globalSeqId;                   // Global execution sequence number
    uint32_t execRound;                     // Outer scheduling round
    int32_t rankId;                         // Owning rank
    HccLTaskMetaType taskType;              // Task type: NOTIFY_RECORD/NOTIFY_WAIT/REDUCE/MEM_CPY etc.
    HcclVmResult execResult;                // Execution result: SUCCESS or HOLD_CMD
    std::string description;                // Task description (for front-end display)
};
```

The front end can display scheduling events during replay, such as "Round 1: Rank1 NotifyWait blocked -> skipped". This helps the user understand the reason for CCU switching.

### 4.8 MVP Phased Implementation Strategy

**Phase 1 (MVP)**:

Minimal viable trace collection to quickly validate the front-end replay workflow:

1. Collect `Describe()` output + resource delta (intercept changes through ResourceManager).
2. `CcuInstrTraceDetail` `args` is empty. `describeText` uses the return value of `executor->Describe()` directly.
3. Do not implement custom `CollectTraceDetail()` for each Executor. Use the base class default implementation.
4. The front end displays only the instruction list panel + resource change panel + instruction description panel.
5. Do not support Loop folding/sampling, cross-CCU changes, or non-CCU task records.

**Phase 2 (Enhancement)**:

1. Implement `CollectTraceDetail()` for high-frequency/critical instruction types (for example, Trans, Reduce, SyncCke).
2. The front end displays complete key-value parameter tables.
3. Support Loop folding/sampling.
4. Support cross-CCU change records and non-CCU task records.

**Phase 3 (Completion)**:

1. Progressively cover `CollectTraceDetail()` for all 30+ instruction types.
2. Support search/filtering, data dependency tracing, and Trace Diff comparison.
3. Support breakpoint functionality.
4. Optionally upgrade to msgpack/flatbuffers binary format.

---

## 5. Front-end Integration Plan

### 5.1 Existing HVRM Insight Architecture

```
App.vue
+-- AppTopNav.vue          // Top navigation bar
+-- DashboardPage.vue      // Overview page
+-- MemViewPage.vue        // Correlation analysis page
+-- AnalyticPage.vue       // Error diagnosis page
```

Technology stack: Vue 3 + Element Plus + Vite

### 5.2 New Tab: CCU Trace Replay

Add a fourth tab in `App.vue`:

```javascript
const pages = [
  { id: 'dashboard', label: 'Overview' },
  { id: 'mem-view', label: 'Correlation' },
  { id: 'analytic', label: 'Error' },
  { id: 'ccu-trace', label: 'CCU Debug' },  // New
]
```

### 5.3 CCU Trace Replay Page Design

#### Overall Layout

```
+-----------------------------------------------------------------------------+
|  CCU Trace Replay                                                            |
+------------+-----------------------------------------------------------------+
|            |  +-- Trace progress bar -------------------------------------+  |
|  Left      |  |  Trace: Rank [0 v]  Die [0 v]    #4/1280  Round 0        |  |
|  sidebar   |  |  ------*------o--------o--------o--------o--------  0.3% |  |
|            |  +-----------------------------------------------------------+  |
|  --------  |  +-----------------------------------------------------------+  |
|            |  |         Instruction List Panel  (Rank0:Die0 instruction space) |  |
|  Replay    |  |  BP | instrId | Instruction Description (Describe) | Exec State|  |
|            |  |   o | 0       | [Load] LoadImd to Xn[5]...  | v #0 R0     |  |
|  controls  |  |   * | 1       | [Trans] LocMem->LocMem...    | v #4 R0     |  |
|            |  |   o | 2       | [SetCke] Set CKE[3:0001]    | v #7 R0     |  |
|  [<<][<]   |  | > o | 3       | [WaitCKE] Wait CKE[0:0000]..| <- current #12|  |
|  [>/||]    |  |   o | 4       | [Trans] LocMem->RmtMem...    |             |  |
|  [>][>>]   |  |   o | 5       | [Reduce] Add MS[0:1,1:2,2:3]|             |  |
|  [Next BP] |  |   ...                                                     |  |
|            |  |  > = current replay position  * = breakpoint  o = no BP      |  |
|  --------  |  |  Exec state: v=executed <-current=current step HOLD=CKE blocked FAIL=failed |  |
|            |  +-----------------------------------------------------------+  |
|  Current   |  +--------------------------+----------------------------------+  |
|  CCU:      |  |   Resource Change Panel  |   Instruction Detail Panel      |  |
|  Rank0:Die0|  |                          |                                 |  |
|            |  |                          |                                 |  |
│            │  │                         │                                 │  │
│  SQE Task   │  │  XN Changes             │  Instruction: WaitCKE          │  │
│  (Current   │  │  Xn[5]: 0→0x1000        │  CCU: Rank0:Die0  Round: 0     │  │
│   Instr     │  │  Xn[8]: 0x400→0         │                                │  │
│   Owner)    │  │                         │  Execution Context:            │  │
│            │  │  GSA Changes            │  Loop Round: 0                  │  │
│  mission: 3 │  │  (none)                 │  Extend Index: 0                │  │
│  sim*:      │  │                         │  GSA Offset: 0x4000             │  │
│  0x7f3a..   │  │  CKE Changes            │                                │  │
│  args[0..12]│  │  CKE[3]: 0→1            │  CKE Wait Info:                │  │
│  0x1000..   │  │                         │  waitCKE: CKE[0]&0x1 expect=1   │  │
│            │  │  MS Changes             │  Spin wait: 5 rounds (initial   │  │
│            │  │  MS[0]: @0x0 +4KB       │  CKE=0)                         │  │
│  Execution  │  │                         │  Final pass: CKE=1              │  │
│  Trace      │  │                         │                                │  │
│  Summary    │  └─────────────────────────┴─────────────────────────────────┘  │
│            │  ┌───────────────────────────────────────────────────────────┐  │
│  Executed   │  │              CCU Resource Overview Panel                  │  │
│  3 instrs   │  │  XN[0..31]  | GSA[0..31] | CKE[0..15] | MS[0..7]       │  │
│  Round 0    │  │  (Shows accumulated resource state of current CCU,       │  │
│            │  │   i.e., Rank0:Die0)                                       │  │
│  ────────  │  └───────────────────────────────────────────────────────────┘  │
│            │  ┌───────────────────────────────────────────────────────────┐  │
│  Breakpoint │  │         Breakpoint Hit Banner (shown only on hit)         │  │
│  Management │  │  ⚠ Breakpoint hit: Rank0:Die0 instrId=1                  │  │
│            │  │  (TransLocMemToLocMem)                                    │  │
│  [+ Add]    │  │  Round 0  missionId=3                                    │  │
│  [Clear All]│  │  [Continue ▶]  [Step ⏭]  [Skip to Next BP ⇥]            │  │
│            │  └───────────────────────────────────────────────────────────┘  │
│            │                                                                 │
│  ┌────────┐│                                                                 │
│  │BP List  ││                                                                 │
│  │● R0:D0  ││                                                                 │
│  │  id=1   ││                                                                 │
│  │● R1:D0  ││                                                                 │
│  │  id=5   ││                                                                 │
│  │ [✏][🗑]││                                                                 │
│  └────────┘│                                                                 │
└────────────┴─────────────────────────────────────────────────────────────────┘
```

#### Instruction List Panel Description

The instruction list contains 4 columns:

| Column | Content | Description |
|---|------|------|
| **BP** | ○ / ● / ▶ | Breakpoint marker + current playback position (`▶` marks the current line) |
| **instrId** | Instruction ID | Index in the CCU instruction space |
| **Instruction Description** | `Describe()` output | Contains full instruction parsing information (parameters, addresses, register IDs, etc.) |
| **Execution Status** | Instruction execution status | Shows execution information for this instruction at the current playback progress |

**Execution Status Column Meanings**:

| Display | Meaning |
|------|------|
| (blank) | Not yet executed |
| `✓ #N RN` | Executed; #N is the global sequence number, RN is the Round number |
| `← Current #N` | The current playback step is paused at this instruction |
| `HOLD` | waitCKE blocked (CKE condition not met, waiting), **orange highlight** |
| `FAIL` | Execution failed, **red highlight** |

**Reasons for Removing Type Column and SQE Column**:
- The type (Load/Trans/Control/Reduce) is already included in the Describe() output; no separate column is needed
- SQE is task-level information rather than instruction-level information; display it in the SQE task area on the left sidebar

#### CCU Instruction Space View and Automatic Switching

The instruction list panel always shows the instruction space of **one CCU** (determined by the Rank/Die dropdown at the top). During trace playback, the instruction list automatically switches and locates entries following the trace execution records:

```
Trace playback advances to the next entry
        │
        ▼
   Does the entry (rankId, dieId) match the currently displayed CCU?
        │                              │
        │ Yes (sequential execution    │ No (switch to a different CCU)
        │ within the same CCU)         │
        ▼                              ▼
   In the current instruction      ① Automatically switch the top Rank/Die
   list, move the ▶ marker          dropdown
   to the corresponding row        ② Switch the instruction list to the new
                                    CCU instruction space
                                   ③ Move the ▶ marker to the corresponding
                                    row in the new CCU
                                   ④ The status bar displays "CCU switch:
                                    R0:D0 → R1:D0"
```

**Current Playback Position Marker**:

The `▶` marker always appears in the BP column and identifies the line where the current playback step resides. The marker overlays with breakpoint markers (for example, `▶●` indicates the current line is also a breakpoint).

**Breakpoint Setting**:

The user clicks the circle in the BP column of the instruction list for the current CCU to set a breakpoint. A breakpoint is essentially a `(rankId, dieId, instrId)` tuple bound to the currently displayed CCU.

#### Core Components

| Component | Function |
|------|------|
| `CcuTraceCcuSelector` | **CCU Selector**: Rank/Die dropdown; selects the currently displayed CCU instruction space |
| `CcuTraceProgressBar` | **Trace Progress Bar**: Displays global progress, current globalSeqId, and Round number |
| `CcuTraceInstrList` | **Instruction List Panel**: 4 columns (BP/instrId/Describe/Execution Status); auto-highlights and switches during playback |
| `CcuTraceExecSummary` | **Execution Trace Summary**: Shows executed count and Round for the current CCU in the left sidebar |
| `CcuTraceSqeInfo` | **SQE Task Information**: Shows the missionId, simulator pointer, and args[0..12] parameter list of the SQE that owns the current instruction in the left sidebar |
| `CcuTracePlaybackControl` | Playback controller (step forward/backward/auto-play/skip to next breakpoint) |
| `CcuTraceBreakpointPanel` | **Breakpoint Management Panel**: Lists all breakpoints (identified by rankId/dieId/instrId); supports add, remove, and edit |
| `CcuTraceBreakpointHitBar` | **Breakpoint Hit Banner**: Appears when a breakpoint is hit; contains CCU/SQE/Round information; provides Continue/Step/Skip options |
| `CcuTraceResourceDelta` | Resource change panel; shows the delta for the current step |
| `CcuTraceInstrDetail` | Instruction detail panel; shows instruction-specific debug information |
| `CcuTraceResourceOverview` | CCU resource overview; shows the accumulated resource state of the current CCU |

#### Breakpoint Feature Design (Frontend Playback Breakpoints)

##### Breakpoint Types

| Type | Description | Setting Method |
|------|------|---------|
| **Line Breakpoint** | Pause when the specified globalSeqId is reached | Click the circle in the BP column of the instruction list |
| **Name Breakpoint** | Pause at all instructions with the specified name | Add through the breakpoint management panel |
| **Category Breakpoint** | Pause at all instructions of the specified category (Load/Trans/Control/Reduce) | Add through the breakpoint management panel |
| **Resource Change Breakpoint** | Pause at the instruction that modifies the specified resource (for example, "break when Xn[5] is written") | Add through the breakpoint management panel |
| **Round Breakpoint** | Pause when the outer scheduler enters the specified Round (for example, "first instruction of Round 3") | Add through the breakpoint management panel |

##### Breakpoint States

- **○** No breakpoint (default)
- **●** Breakpoint enabled
- **⊘** Breakpoint disabled (configuration retained but not triggered)
- **▶●** Current step hit the breakpoint (highlighted + hit banner appears)

##### Auto-Play Behavior Upon Breakpoint Hit

```
User clicks [▶ Auto-Play]
        │
        ▼
   Advance step by step ────► Check each step: does the current entry hit a breakpoint?
        │                              │
        │ No                           │ Yes
        ▼                              ▼
   Continue to next step     Pause playback, highlight the current line
                                   │
                           Display breakpoint hit banner:
                           "⚠ Breakpoint hit: Rank0:Die0 instrId=1 (instruction name)"
                                   │
                   ┌───────────────┼───────────────┐
                   ▼               ▼               ▼
             [Continue ▶]    [Step ⏭]      [Skip to Next BP ⇥]
             Resume auto-play  Advance one    Fast-forward to the
                               step then pause next breakpoint
```

##### Breakpoint Management Panel Interaction

The breakpoint management area at the bottom of the left sidebar shows all current breakpoints and supports:
1. **Quick Add**: Click the BP column in the instruction list, or click `[+ Add]` to open the dialog
2. **Enable/Disable**: Click the toggle switch on a breakpoint entry
3. **Jump to Location**: Click a breakpoint entry; the instruction list scrolls to the corresponding row automatically
4. **Edit/Delete**: Each breakpoint row provides `[✏ Edit]` and `[🗑 Delete]` buttons on the right side
5. **Batch Operations**: `[Clear All]` removes all breakpoints with one click

#### Interaction Logic

1. **CCU Selection** → Select the currently displayed CCU instruction space through the top Rank/Die dropdown; the instruction list switches to the corresponding CCU
2. **Playback Advance** → Click the playback control button to advance globally (globalSeqId increments):
   - If the current entry belongs to the selected CCU → the `▶` marker moves to the corresponding instrId row; the execution status column updates
   - If the current entry belongs to a different CCU → the CCU selector switches automatically; the instruction list switches to the new CCU and marks the corresponding row
3. **CCU Switch Notification** → When the CCU switches automatically, the status bar briefly displays "CCU switch: R0:D0 → R1:D0" to help the user understand the scheduling jump
4. **Auto-Play Mode** → Advance the global sequence step by step at the configured speed to simulate the actual interleaved execution process; the CCU switches automatically following the trace
5. **Round Separation** → When execRound changes, the execution trace summary area on the left displays the Round change (for example, "Round 0 → Round 1")
6. **HOLD State** → Instructions blocked by waitCKE display an orange `HOLD` marker in the execution status column; the instruction detail panel shows the waiting CKE ID, mask, and spin count
7. **Click Instruction Row** → Linked updates occur: resource delta panel (shows the delta for that instruction), instruction detail panel, and CCU resource overview (updates to the current CCU state)
8. **Click BP Column Circle** → Toggle the breakpoint marker for that row (○ ↔ ●); the breakpoint tuple `(rankId, dieId, instrId)` binds to the current CCU automatically; the breakpoint management panel updates in sync
9. **Auto-Play Hits Breakpoint** → Pause and display the hit banner (containing CCU/Round information); the user can choose continue/step/skip to next breakpoint
10. **Click [⇥ Next Breakpoint]** → Fast-forward to the next breakpoint position; if the breakpoint is in a different CCU, the CCU selector switches automatically
11. **Execution Trace Summary** → The left sidebar displays the executed instruction count, current Round, and current SQE summary for the current CCU in real time
12. **SQE Information Linkage** → The instruction detail panel shows the missionId, instStartId, instCnt, and args list of the SQE that owns the current instruction
13. **Resource Overview** → Display the accumulated resource state (XN/GSA/CKE/MS) of the current CCU; update in real time during playback
14. **Search Instruction** → Search by instruction name, description, or instrId in the instruction list of the current CCU
15. **Trace Resource Origin** → Click a changed value in the resource delta panel to highlight the previous instruction that modified that resource automatically (traced within the same CCU)
16. **Cross-CCU Jump** → Click a remote change in crossCcuChanges to switch the CCU selector to the corresponding CCU and locate the associated instruction automatically
17. **Failure Quick Locate** → Click the "Jump to First Failure" button to switch to the CCU that owns the failed instruction automatically, locate the first instruction with execState=EXEC_FAIL, and display it with red highlighting
18. **Manual CCU Switch** → The user can switch the CCU manually through the dropdown at any time; the instruction list switches to the instruction space of the corresponding CCU; the playback position remains unchanged, but the most recently executed instruction in that CCU is highlighted

### 5.4 Data Loading

Reuse the existing Worker data loading pipeline in Insight:

```javascript
// New data loading utility
// src/utils/ccuTraceData.js
export async function loadCcuTrace(datasetPath, rankId, dieId) {
    const response = await fetch(`/api/ccu-trace/${datasetPath}/rank${rankId}_die${dieId}.json`);
    return await response.json();
}
```

Add a new backend API endpoint:

```python
# server.py additions
@app.route('/api/ccu-trace/<dataset>/<file>')
def ccu_trace_data(dataset, file):
    trace_path = os.path.join(DATA_DIR, dataset, 'ccu_trace', file)
    return send_file(trace_path, mimetype='application/json')
```

---

## 6. File Organization

### 6.1 Backend (C++)

```
src/plugin/solver/virtual_runtime/ccu_executor/trace/
├── ccu_trace_types.h          // Trace data structure definitions
├── ccu_trace_collector.h      // Trace collector declaration
├── ccu_trace_collector.cc     // Trace collector implementation
├── ccu_trace_serializer.h     // JSON serialization declaration
└── ccu_trace_serializer.cc    // JSON serialization implementation
```

### 6.2 Frontend (Vue)

```
src/plugin/solver/virtual_runtime/insight/frontendV3/src/
├── pages/
│   └── CcuTracePage.vue              // CCU Trace playback page
├── components/ccu-trace/
│   ├── CcuTraceCcuSelector.vue       // CCU selector (Rank/Die dropdown)
│   ├── CcuTraceProgressBar.vue       // Trace progress bar
│   ├── CcuTraceInstrList.vue         // Instruction list panel (current CCU instruction space)
│   ├── CcuTraceExecSummary.vue       // Execution trace summary (left sidebar)
│   ├── CcuTracePlaybackControl.vue   // Playback control
│   ├── CcuTraceSqeInfo.vue           // SQE task information (left sidebar)
│   ├── CcuTraceResourceDelta.vue     // Resource delta panel
│   ├── CcuTraceInstrDetail.vue       // Instruction detail panel
│   ├── CcuTraceResourceOverview.vue  // CCU resource overview (current CCU)
│   ├── CcuTraceBreakpointPanel.vue   // Breakpoint management panel
│   └── CcuTraceBreakpointHitBar.vue  // Breakpoint hit banner
└── utils/
    └── ccuTraceData.js               // Data loading + index building utility
```

---

## 7. Four-Way Relationship Summary

The core of this design tightly links **instructions**, **CCUs**, **SQE tasks**, and **CCU resources** through the trace data structure:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CcuTraceEntry                                │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │ Global    │  │ Instr    │  │  Resource    │  │  Instr-Specific│   │
│  │ Positioning│  │ Info     │  │  Changes     │  │  Details       │   │
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
│  │  SQE Task (looked up in sqeTaskRegistry)                      │   │
│  │  missionId, instStartId, instCnt, args[], simulatorPtr        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Execution Context (Loop/Jump State)                          │   │
│  │  inLoop, loopRound, gsaOffset, msOffset, ckeOffset, ...      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

**Key Design Decisions**:
1. **Global Interleaved Timeline**: Instructions from all CCUs are interleaved in actual execution order within `globalEntries[]`; `globalSeqId` reflects the real scheduling order
2. **CCU Registry**: Each CCU (rankId, dieId) has a unique entry in `ccuRegistry` containing the initial resource snapshot; entries are linked through rankId+dieId
3. **SQE Registry**: Each SQE task has a unique entry in `sqeTaskRegistry`; entries reference it through `sqeTaskId` to avoid redundancy
4. **Delta Instead of Full Snapshots**: Each trace entry records only changes; initial and final snapshots are used to reconstruct the full state
5. **Layered Polymorphism**: The common layer handles information shared by all instructions; the specialized layer supports instruction-specific details through polymorphism
6. **Scheduling Rounds**: `execRound` records the iteration of the outer `while(HasTask())` loop to help understand the timing of interleaved scheduling

---

## 8. Breakpoint Feature Design

### 8.1 Feasibility Analysis

The breakpoint feature is fully feasible and can be implemented at two independent levels:

| Level | Name | Trigger Timing | Complexity | Purpose |
|------|------|---------|--------|------|
| **Frontend Playback Breakpoint** | Offline Breakpoint | During playback of collected trace data | Low | Pause auto-play at a specific instruction for the user to inspect details |
| **Backend Execution Breakpoint** | Live Breakpoint | During actual CCU instruction execution/recording | High | Pause the simulator execution in real time; review the current state interactively before continuing |

The two levels can be used independently or together: first use backend breakpoints to record trace data for critical intervals precisely, then use frontend breakpoints to replay and analyze repeatedly.

### 8.2 Breakpoint Types

```cpp
// Breakpoint type enumeration (breakpoint matching strategies; all ultimately locate a specific rankId+dieId+instrId)
enum class CcuBreakpointType : uint8_t {
    INSTR_ID,           // Instruction ID breakpoint: trigger when the specified instrId is reached
    INSTR_NAME,         // Instruction name breakpoint: trigger when an instruction with the specified name is reached (for example, "ReduceAdd")
    CATEGORY,           // Instruction category breakpoint: trigger when the specified category is reached (for example, all Trans category instructions)
    RESOURCE_CHANGE,    // Resource change breakpoint: trigger at the instruction that modifies the specified resource (for example, Xn[5] is written)
    CONDITION,          // Conditional breakpoint: trigger at the instruction where the resource satisfies a condition expression (for example, Xn[5] == 0x1000)
    ITERATION,          // Iteration breakpoint: trigger at the instruction during the Nth Loop iteration
    ROUND,              // Scheduling round breakpoint: trigger when the outer while(HasTask()) enters the specified round (for example, all instructions in round 3)
};

// Breakpoint configuration
// A breakpoint relates only to instructions; the core tuple is: (rankId, dieId, instrId)
struct CcuBreakpointConfig {
    uint32_t bpId;                      // Unique breakpoint identifier
    CcuBreakpointType type;             // Breakpoint matching strategy type
    bool enabled;                       // Whether the breakpoint is enabled
    bool oneShot;                       // One-shot breakpoint (auto-disables after trigger)

    // === Core Breakpoint Positioning Information ===
    int32_t rankId;                     // Target rank (-1 means all ranks)
    int32_t dieId;                      // Target die (-1 means all dies)
    uint16_t instrId;                   // Target instruction ID

    // === Additional Matching Strategy Parameters (selected based on type) ===
    std::string targetInstrName;        // INSTR_NAME: target instruction name
    CcuInstrCategory targetCategory;    // CATEGORY: target instruction category
    std::string targetResourceType;     // RESOURCE_CHANGE: resource type "XN"/"GSA"/"CKE"/"MS"
    uint16_t targetResourceId;          // RESOURCE_CHANGE: resource ID
    std::string conditionExpr;          // CONDITION: condition expression (for example, "XN[5] == 0x1000")
    uint16_t loopRound;                 // ITERATION: target Loop iteration round
    uint32_t targetExecRound;           // ROUND: target outer scheduling round number

    // === Hit Counting ===
    uint32_t hitCount;                  // Number of times hit
    uint32_t skipCount;                 // Skip the first N hits (for "break at the 3rd loop iteration")
};

// Breakpoint hit information
struct CcuBreakpointHit {
    uint32_t bpId;                      // ID of the hit breakpoint
    uint32_t globalSeqId;               // Global execution sequence number of the current instruction
    uint32_t execRound;                 // Current outer scheduling round
    uint16_t instrId;                   // Current instruction ID
    std::string instrName;              // Current instruction name
    int32_t rankId;                     // Rank where the hit occurred
    int32_t dieId;                      // Die where the hit occurred
    uint32_t sqeTaskId;                 // SQE task ID that owns this instruction (references sqeTaskRegistry)
    CcuResourceSnapshot snapshot;       // Resource snapshot at the time of the hit
    CcuExecutionContext context;        // Execution context at the time of the hit
};
```

**Relationship Between Breakpoints and SQEs**:

A breakpoint only concerns "at which instruction on which CCU to stop" and does not involve SQEs. The SQE information is managed through the `sqeTaskRegistry`; each `CcuTraceEntry` references the corresponding SQE task through `sqeTaskId`.

When the user jumps to an instruction during playback:
1. The SQE information area in the **instruction detail panel** displays the missionId, args, and instruction range of the SQE that owns this instruction automatically
2. The **SQE registry panel** (left sidebar) highlights the corresponding SQE entry automatically
3. Multiple SQEs on the same CCU share the same CcuSimulator instance (`simulatorPtr` is identical); SQEs are distinguished by `(sqeTaskId, missionId)`

### 8.3 Option 1: Frontend Playback Breakpoints (Offline Breakpoint)

**Principle**: This is a pure frontend implementation with no backend modifications. The user clicks a row in the frontend instruction list to set a breakpoint marker; auto-play pauses when it reaches that instruction.

**Implementation**:

```javascript
// Breakpoint management in CcuTracePlaybackControl.vue
const breakpoints = ref(new Set());  // Set storing breakpoint seqIds

function toggleBreakpoint(seqId) {
    if (breakpoints.value.has(seqId)) {
        breakpoints.value.delete(seqId);
    } else {
        breakpoints.value.add(seqId);
    }
}

// Auto-play logic
function autoPlay() {
    const timer = setInterval(() => {
        currentStep.value++;
        // Pause when a breakpoint is hit
        if (breakpoints.value.has(currentStep.value)) {
            clearInterval(timer);
            isPlaying.value = false;
            notifyBreakpointHit(currentStep.value);
        }
        // Playback complete
        if (currentStep.value >= entries.length - 1) {
            clearInterval(timer);
            isPlaying.value = false;
        }
    }, playSpeed.value);
}
```

**Supported Breakpoint Types**:
- Break by seqId (most common: click the instruction row to set)
- Break by instruction name (for example, "break at all ReduceAdd instructions")
- Break by instruction category (for example, "break at all Trans category instructions")
- Break by resource change (for example, "break at the instruction that modifies Xn[5]"; requires scanning delta data)

**Frontend Breakpoint UI Interaction**:

```
Instruction List Panel:
  seqId | instrId | Name        | Type   | BP   | Status
  0     | 0       | LoadImd     | Load   |      | ✓
  1     | 1       | TransMM     | Trans  | 🔴   | ✓     ← Click the line number area to toggle breakpoint
  2     | 2       | SetCke      | Ctrl   |      | ✓
  ▶ 3   | 3       | ReduceAdd   | Reduce |      | ← Current (paused here after breakpoint hit)
  4     | 4       | Loop        | Ctrl   |      |
  ...
```

### 8.4 Option 2: Backend Execution Breakpoints (Live Breakpoint)

**Principle**: During actual CCU instruction execution, when the instruction specified by the breakpoint is reached, the simulator execution thread pauses. The backend notifies the frontend through WebSocket. The frontend displays the current state. After the user confirms, the frontend sends a continue command through WebSocket to resume execution. This option is not implemented in the current plan.

#### 8.4.1 Architecture

```
┌──────────────────┐      WebSocket        ┌─────────────────────────┐
│  Frontend (Vue)   │ ◄──────────────────► │  Backend Debug Server    │
│                  │   /ws/ccu-debug       │  (embedded in hccl-vm    │
│  Breakpoint Mgmt │                       │   process)               │
│  Panel           │                       │                         │
│  Real-time Status│                       │  CcuBreakpointManager   │
│  Continue/Step   │                       │    ├── Breakpoint config │
│  Buttons         │                       │    ├── Condition evaluator│
└──────────────────┘                       │    └── State snapshot    │
                                           │        collection        │
                                           │                         │
                                           │  CcuSimulator           │
                                           │    └── ExecuteInstr()   │
                                           │        └── CheckBreakpoint() │
                                           │            └── Hit? → Pause thread │
                                           └─────────────────────────┘
```

#### 8.4.2 CcuBreakpointManager

```cpp
class CcuBreakpointManager {
public:
    static CcuBreakpointManager& GetInstance();

    // Breakpoint management
    uint32_t AddBreakpoint(const CcuBreakpointConfig& config);
    void RemoveBreakpoint(uint32_t bpId);
    void EnableBreakpoint(uint32_t bpId, bool enabled);
    void ClearAllBreakpoints();
    std::vector<CcuBreakpointConfig> GetAllBreakpoints() const;

    // Runtime check (called within ExecuteInstr)
    // Returns true if pause is needed (breakpoint hit)
    bool CheckBreakpoint(int rankId, int dieId, uint16_t instrId,
                         uint32_t seqId, const CcuExecutionContext& ctx);

    // Pause and resume control
    void RequestPause();          // External pause request (user clicks pause button)
    void RequestContinue();       // Request to continue execution
    void RequestStepOver();       // Request to execute one instruction then pause
    void RequestStepIntoLoop();   // Request to step into Loop body
    bool IsPaused() const;

    // Hit information (for Debug Server to read and send to frontend)
    CcuBreakpointHit GetLastHitInfo() const;

private:
    // Condition expression evaluation
    bool EvaluateCondition(const std::string& expr, int rankId, int dieId);

    std::mutex bpMutex_;
    std::map<uint32_t, CcuBreakpointConfig> breakpoints_;
    std::atomic<bool> paused_{false};
    std::atomic<bool> stepMode_{false};        // Step mode
    std::condition_variable pauseCV_;           // For thread pause/resume
    std::mutex pauseMutex_;
    CcuBreakpointHit lastHitInfo_;
    uint32_t nextBpId_{1};
};
```

#### 8.4.3 Integrating Breakpoint Checks in CcuSimulator

```cpp
bool CcuSimulator::ExecuteInstr(uint16_t curInstrId)
{
    auto& bpMgr = CcuBreakpointManager::GetInstance();
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    auto instrData = ccuResMgr.GetInstrData(rankId_, dieId_);

    // ① Breakpoint check (before execution)
    if (bpMgr.IsEnabled() && bpMgr.CheckBreakpoint(
            rankId_, dieId_, curInstrId, seqId_, GetExecContext())) {
        // Breakpoint hit → pause thread, wait for user action
        // CheckBreakpoint collects the snapshot and populates lastHitInfo_ internally
        // Debug Server notifies the frontend through WebSocket
        bpMgr.WaitWhilePaused();  // Block until user sends Continue/Step
    }

    auto executor = CcuExecutorFactory::MakeCcuExecutorInstance(...);
    executor->Parser();
    executor->Run();

    // ② Step mode: pause after executing one instruction
    if (bpMgr.IsStepMode()) {
        bpMgr.RequestPause();
        bpMgr.WaitWhilePaused();
    }

    // ③ Trace collection (if enabled)
    if (CcuTraceCollector::IsEnabled()) {
        // ... original trace collection logic ...
    }

    UpdateLoopStatus();
    return true;
}
```

#### 8.4.4 Thread Pause/Resume Mechanism

```cpp
// Thread synchronization in CcuBreakpointManager
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
    pauseCV_.notify_all();  // Release the execution thread; it pauses again after one instruction
}
```

#### 8.4.5 WebSocket Debug Server Protocol

The backend embeds a lightweight WebSocket server (started only when `HCCLVM_ENABLE_CCU_DEBUG=1`) to communicate with the frontend in real time:

```
Message Direction  Message Type              Content
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

#### 8.4.6 Conditional Breakpoint Expression Evaluation

The system supports simple resource condition expressions for conditional breakpoints:

```
Syntax examples:
  XN[5] == 0x1000                 // Xn register 5 value equals 0x1000
  GSA[3] > 0x7F0000               // GSA register 3 is greater than the specified value
  CKE[0] & 0x0001 != 0            // CKE signal 0 bit 0 is 1
  MS[10][0x100:0x10] == 0xFF      // MS 10 at offset 0x100, 16 bytes equal the specified value
  XN[5] == 0x1000 && GSA[3] > 0  // Combined condition
```

Implementation: a lightweight recursive descent parser that reads resource values directly from `CcuResourceManager` for evaluation.

### 8.5 Frontend Breakpoint Panel UI Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Breakpoint Management Panel                                            │
├─────────────────────────────────────────────────────────────────────────┤
│  [+ Add Breakpoint]  [Clear All]  [Enable All ✓]                        │
│                                                                         │
│  ┌──────┬──────────────────────────────────┬──────┬────────┬──────────┐ │
│  │Status│ Breakpoint Condition             │ Hits │OneShot │ Actions  │ │
│  ├──────┼──────────────────────────────────┼──────┼────────┼──────────┤ │
│  │ ✓   │ instrId == 42 @ Rank0/Die0       │ 0    │        │ [✏][🗑] │ │
│  │ ✓   │ name == "ReduceAdd" @ All        │ 3    │        │ [✏][🗑] │ │
│  │     │ XN[5] == 0x1000 @ Rank0/Die1     │ 0    │ ✓     │ [✏][🗑] │ │
│  │ ✓   │ category == Trans @ Rank0/Die0   │ 12   │        │ [✏][🗑] │ │
│  │ ✓   │ loopRound == 3 @ Rank0/Die0      │ 1    │        │ [✏][🗑] │ │
│  └──────┴──────────────────────────────────┴──────┴────────┴──────────┘ │
│                                                                         │
│  ┌─── Add Breakpoint Dialog ────────────────────────────────────────┐   │
│  │  Breakpoint Type: [Instruction ID ▼]                             │   │
│  │  Rank: [0 ▼]  Die: [0 ▼]  InstrId: [42]                         │   │
│  │  Skip Count: [0]   ☐ One-shot breakpoint                         │   │
│  │                                              [Cancel]  [OK]       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.6 Collaboration Workflow of the Two Modes

```
User workflow:

Phase 1: Recording (optional, uses backend breakpoints)
  ① Set a backend breakpoint: instrId=42, Rank0/Die0
  ② Start hccl-vm to execute the operator
  ③ CCU reaches instruction 42 → pauses automatically
  ④ Frontend displays real-time state: resource snapshot, instruction details
  ⑤ User reviews then clicks [Continue]
  ⑥ Execution completes; trace data is saved automatically

Phase 2: Playback Analysis (uses frontend breakpoints)
  ⑦ Frontend loads the trace data
  ⑧ Set frontend breakpoints in the instruction list (instructions of interest)
  ⑨ Click [▶ Auto-Play]
  ⑩ Playback pauses automatically when a breakpoint instruction is reached
  ⑪ User inspects resource changes and instruction details
  ⑫ Click [▶ Continue] or [⏭ Skip to Next Breakpoint]
```

### 8.7 File Organization (Breakpoint Feature Additions)

```
src/plugin/solver/virtual_runtime/ccu_executor/trace/
├── ccu_trace_types.h              // (existing) trace data structures
├── ccu_trace_collector.h/cc       // (existing) trace collector
├── ccu_trace_serializer.h/cc      // (existing) JSON serialization
├── ccu_breakpoint_types.h         // (new) breakpoint data structures
├── ccu_breakpoint_manager.h       // (new) breakpoint manager declaration
├── ccu_breakpoint_manager.cc      // (new) breakpoint manager implementation
├── ccu_condition_evaluator.h      // (new) condition expression evaluator
├── ccu_condition_evaluator.cc     // (new)
├── ccu_debug_server.h             // (new) WebSocket debug server
└── ccu_debug_server.cc            // (new)

Frontend additions:
src/plugin/solver/virtual_runtime/insight/frontendV3/src/
├── components/ccu-trace/
│   ├── CcuTraceBreakpointPanel.vue    // (new) breakpoint management panel
│   ├── CcuTraceBreakpointDialog.vue   // (new) add/edit breakpoint dialog
│   └── CcuTraceLiveControl.vue        // (new) live debug control (Continue/Step/Pause)
├── composables/
│   └── useCcuDebugWebSocket.js        // (new) WebSocket communication composable
└── utils/
    └── ccuBreakpointUtils.js          // (new) breakpoint frontend utility functions
```

---

## 9. Advanced Analysis Features

### 9.1 Search and Multi-Dimensional Filtering

Locate problem instructions quickly in large trace datasets. The system provides the following search and filtering capabilities:

#### 9.1.1 Text Search

A search box at the top of the instruction list supports real-time filtering:
- Search by instruction name (for example, type "TransLocMem" to match all local memory transfer instructions)
- Search by instruction description (match keywords in the `Describe()` output)
- Search by seqId (for example, type "#42" to jump to the 42nd instruction)

#### 9.1.2 Multi-Dimensional Filters

Dropdown filters support multi-dimensional combined filtering:

| Filter Dimension | Options | Description |
|---------|------|------|
| **Instruction Category** | Load / Trans / Control / Reduce / All | Filter by the 4 major categories |
| **Execution Status** | Normal / Failed / CKE Wait / All | Filter by execution result |
| **Loop Context** | Non-Loop / Loop First Round / Loop Middle Round / Loop Last Round / All | Filter by Loop position |
| **Resource Change Type** | Modified XN / Modified GSA / Modified CKE / Modified MS / Cross-CCU / All | Filter by the type of changed resource |
| **CKE Wait** | Has Wait / No Wait / All | Whether CKE wait is present |
| **SQE Range** | SQE 0 / SQE 1 / ... / All | Filter by SQE session |

Filter states can be saved as "filter presets" for convenient reuse.

#### 9.1.3 Search Result Statistics

After filtering, statistics are displayed at the bottom of the list:
```
Showing 42 / 1280 instructions  |  Failed: 2  |  CKE Wait: 5  |  Cross-CCU: 8
```

### 9.2 Data Dependency Tracing

This feature helps the user understand "where a resource value came from" and establishes data flow relationships between instructions.

#### 9.2.1 Trace Origin (Trace Backward)

The user clicks a changed value in the resource delta panel to locate the previous instruction that modified that resource automatically:

```
User action: Click the [Trace] button next to "Xn[5]: 0x400 → 0x1000"

System behavior:
  1. Search backward from the current seqId to find the last instruction that modified Xn[5]
  2. The instruction list scrolls to that instruction and highlights it automatically
  3. The resource delta panel updates in sync
  4. Draw a dashed arrow between the two instructions (optional, displayed in the instruction list)
```

Implementation principle:
```javascript
// Build resource write index (preprocessed during data loading)
const resourceWriteIndex = {
    'XN:5': [
        { seqId: 0, valueAfter: '0x400' },
        { seqId: 15, valueAfter: '0x1000' },
        { seqId: 42, valueAfter: '0x2000' },
    ],
    'GSA:3': [ ... ],
    'CKE:0': [ ... ],
};

// Trace backward: find the last entry that wrote to this resource before the current seqId
function traceBackward(resourceType, resourceId, currentSeqId) {
    const key = `${resourceType}:${resourceId}`;
    const writes = resourceWriteIndex[key] || [];
    return writes.filter(w => w.seqId < currentSeqId).pop();
}
```

#### 9.2.2 Trace Impact (Trace Forward)

The reverse operation: find which subsequent instructions read the resource that the current instruction modified:

```
User action: Click the [Impact] button next to "Xn[5]: 0 → 0x400"

System behavior:
  1. Search forward from the current seqId to find all instructions that read Xn[5]
  2. Highlight these instructions in the instruction list (yellow background)
  3. Statistics: Xn[5]=0x400 was read by N subsequent instructions until seqId=M where it was overwritten
```

#### 9.2.3 CKE Synchronization Relationship Visualization

CKE is the core synchronization mechanism between CCU instructions. The system provides a dedicated CKE relationship view:

```
CKE[3] Synchronization Timeline:

seqId=10  SetCke   CKE[3]: 0→1  (set)
  ↓ (waiting for CKE[3]&0x1 == 0x1)
seqId=25  TransMM  waitCKE[3:0001] ✓ passed
  ↓
seqId=26  ClearCke CKE[3]: 1→0  (clear)
  ↓ (waiting for CKE[3]&0x1 == 0x1)
seqId=40  TransMM  waitCKE[3:0001] ⚠ passed after 14 spin rounds
```

### 9.3 Trace Diff Comparison

Compare trace data from two runs to locate differences quickly. A typical scenario: compare the same operator between a correct run and an incorrect run.

#### 9.3.1 Comparison Mode

```
┌────────────────────────────────────────────────────────────────────────┐
│  Trace Diff Mode                                                       │
├──────────────────────────┬─────────────────────────────────────────────┤
│  Run A (Correct)          │  Run B (Incorrect)                         │
│  rank0_die0_trace.json   │  rank0_die0_trace.json                     │
│                          │                                             │
│  seqId | Name    | Status│  seqId | Name    | Status | Diff           │
│  0     | LoadImd | ✓    │  0     | LoadImd | ✓     | = (identical)   │
│  1     | TransMM | ✓    │  1     | TransMM | ✓     | ✗ delta differs │
│  2     | SetCke  | ✓    │  2     | SetCke  | ✓     | =               │
│  3     | Reduce  | ✓    │  3     | Reduce  | FAIL  | ✗ B failed      │
│  ...                    │  ...                                       │
│                          │                                             │
│  Diff Summary:                                   │
│  First divergence: seqId=1 (TransLocMemToLocMem)                      │
│  Diff type: Xn[5] value differs (A:0x1000 vs B:0x0)                  │
│  Total diffs: 3 / 128                                                  │
└──────────────────────────┴─────────────────────────────────────────────┘
```

#### 9.3.2 Diff Detection Rules

| Comparison Item | Match Condition | Diff Marker |
|-------|---------|---------|
| Instruction Sequence | Same instrId + instrName | Missing/extra instruction → red |
| Resource Delta | Same xnChanges/gsaChanges/ckeChanges | Different values → yellow |
| Execution Status | Same execState | A succeeded, B failed → red |
| CKE Wait | Same waitRetryCount (tolerance ±1) | Large wait count difference → orange |
| Cross-CCU Changes | Same remoteCkeChanges | Remote operations inconsistent → purple |

#### 9.3.3 Diff Data Source

The frontend computes diff data in real time after loading two trace files. No backend support is required:

```javascript
// Frontend diff computation
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

## 10. Version Compatibility Strategy

### 10.1 Trace Format Version Management

`CcuRunMetadata.traceFormatVersion` identifies the version number of the trace data format. Version numbers follow semantic versioning:

| Version | Meaning | Change Example |
|------|------|---------|
| 1.0 | Initial version | Current design |
| 1.x | Backward-compatible extensions | New optional fields added (for example, Detail for new instruction types) |
| 2.0 | Incompatible changes | Data structure reorganization (for example, delta format changes) |

### 10.2 CCU Microcode Version Compatibility

CCU V1 and V2 have different resource specifications:

| Resource | V1 Specification | V2 Specification |
|------|--------|--------|
| XN | 3072 | 4096 |
| GSA | 3072 | 4096 |
| CKE | 1024 | 1024 |
| MS | 1536 | 1536 |
| Instruction Types | 4 categories ~20 types | 4 categories ~35 types |

**Compatibility Strategy**:
1. `CcuRunMetadata.ccuVersion` identifies the microcode version for this trace
2. The frontend adjusts the resource display range dynamically based on ccuVersion (V1 shows 3072 XNs, V2 shows 4096)
3. The frontend displays the corresponding instruction type names based on ccuVersion (V2 has Add/Sub/Mul arithmetic instructions; V1 does not)
4. Trace Diff supports cross-version comparison (V1 trace vs V2 trace) but marks resource specification differences

### 10.3 Frontend Backward Compatibility

The frontend code handles trace data of different versions through a version dispatcher:

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

For backward-compatible extensions (1.x), new fields receive default values:

```javascript
function parseTraceEntry(raw) {
    return {
        seqId: raw.seqId,
        instrId: raw.instrId,
        // ... required fields
        waitInfo: raw.waitInfo || { hadWait: false, waitRetryCount: 0 },  // default value
        errorInfo: raw.errorInfo || { hasError: false },                   // default value
        crossCcuChanges: raw.crossCcuChanges || { hasCrossCcuChange: false }, // default value
    };
}
```

---

## 11. Trace Reliability Under Exception Scenarios

### 11.1 Problem Analysis

The current trace dump flow has a **critical defect**: all trace data resides in memory during execution (the `std::vector` of `m_traceRun.globalEntries`), and the system serializes and writes it to disk only at the **very end** of `SequentialExecutor::Execute()`.

```
Execute() {
    // trace initialization
    while (HasTask()) {       // ← If a crash occurs here
        ExecuteOneTask();     //   ← Or here
    }
    // ===== Trace Dump =====  //   ← Will never be reached
    traceCollector.EndRun();
    DumpToFile(...);
}
```

**Three Exception Scenario Analysis**:

| Scenario | Trigger Location | Can Trace Be Dumped? | Reason |
|------|---------|:---:|------|
| **Segmentation Fault (SIGSEGV)** | Pointer out of bounds in instruction `Process()` | **No** | The process terminates directly; the dump code is never reached |
| **Out-of-Bounds Access** | Array operations such as `GetXnValue()` | **No** | Triggers crash/UB same as above |
| **C++ Exception (throw)** | Exception thrown during instruction execution | **No** | No try/catch in the call chain; the exception propagates to the top level and triggers `std::terminate` |

**Root Causes (4 Defects)**:

1. **All trace data is in memory**: All `CcuTraceEntry` objects are stored in `m_traceRun.globalEntries` (`std::vector`); the system never writes to disk during execution
2. **No signal handling mechanism**: A global search for `signal`/`sigaction`/`SIGSEGV`/`SIGABRT`/`atexit`/`set_terminate` returns no results; there is no opportunity to trigger a dump at crash time
3. **No exception catching**: The entire chain `Execute()` → `ExecuteOneTask()` → `TaskCcuGraph()` → `CcuSimulator::Execute()` → `ExecuteInstr()` has no try/catch
4. **No incremental dump/checkpoint**: No mechanism exists for periodic disk writes during execution

### 11.2 Solution

The design adopts a combined strategy of **Option 1 (signal handling) + Option 3 (incremental dump)**.

#### 11.2.1 Option 1: Register Signal Handlers (For Segfaults/Abort)

Register SIGSEGV/SIGABRT signal handlers in `SequentialExecutor::Execute()` to perform an emergency dump of collected trace data when the process crashes:

```cpp
#include <signal.h>
#include <execinfo.h>

static const char* g_crashDumpPath = nullptr;

static void CrashDumpHandler(int sig) {
    // 1. Restore default signal handling to prevent infinite recursion
    //    if a signal triggers again during the dump
    signal(sig, SIG_DFL);

    // 2. Emergency dump of collected trace data
    auto& collector = CcuTrace::CcuTraceCollector::GetInstance();
    if (collector.IsEnabled() && g_crashDumpPath != nullptr) {
        collector.EndRun();
        auto traceRun = collector.GetTraceRun();
        CcuTrace::CcuTraceSerializer::DumpToFile(traceRun, g_crashDumpPath);
    }

    // 3. Print call stack for debugging
    void* frames[64];
    int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);

    // 4. Re-trigger default handling (core dump)
    raise(sig);
}

// Register in Execute()
g_crashDumpPath = crashOutputPath.c_str();
signal(SIGSEGV, CrashDumpHandler);
signal(SIGABRT, CrashDumpHandler);
```

**Async-Signal-Safety Note**:
Strictly speaking, using `std::ostringstream` (used internally by DumpToFile) in a signal handler is not async-signal-safe. However, it is usually workable in practice because a crash is unlikely to occur inside malloc. Even if the dump fails, the process is no worse off than before (previously, 100% of data was lost). For strict safety, a future improvement could use the `write()` system call to output binary format directly.

#### 11.2.2 Option 3: Incremental Dump (Periodic Checkpoint)

Periodically serialize and write collected trace data to disk in the main scheduling loop. Even if the process is killed by `kill -9` or the OOM killer, the disk still contains the trace data from the most recent checkpoint.

```cpp
const uint32_t TRACE_FLUSH_INTERVAL = 100;  // Flush every 100 instructions
uint32_t instrCountSinceFlush = 0;

while (HasTask()) {
    // ... execution logic ...
    
    // Incremental dump
    instrCountSinceFlush++;
    if (traceCollector.IsEnabled() && instrCountSinceFlush >= TRACE_FLUSH_INTERVAL) {
        traceCollector.IncrementalDump();
        instrCountSinceFlush = 0;
    }
}
```

**New `IncrementalDump()` Interface in CcuTraceCollector**:

```cpp
void CcuTraceCollector::IncrementalDump() {
    if (!m_enabled || m_outputPath.empty()) return;
    auto traceRun = GetTraceRun();  // Get current snapshot (with lock protection)
    CcuTraceSerializer::DumpToFile(traceRun, m_outputPath);
}
```

**Performance Impact Assessment**:
- Flushing every 100 instructions incurs approximately 1-5ms serialization overhead (depending on the collected data volume)
- The total instruction count is typically < 10000; the flush count is < 100; the total additional overhead is < 500ms
- Compared to the overhead of instruction execution itself, the impact is acceptable
- The interval can be adjusted through the `HCCLVM_TRACE_FLUSH_INTERVAL` environment variable

#### 11.2.3 Coverage Summary

| Layer | Mechanism | Covered Scenarios |
|----|------|---------|
| **Signal Handling** (Option 1) | `signal(SIGSEGV/SIGABRT, handler)` | Segmentation faults, abort, stack overflow |
| **Incremental Dump** (Option 3) | Periodic flush | All scenarios (including kill -9, OOM, power failure) |

The combination of both ensures:
- **Normal exit**: The final dump writes the complete trace
- **Segmentation fault/abort**: The signal handler performs an emergency dump (filename marked with `_crash`)
- **kill -9 / OOM / power failure**: The disk contains the trace from the most recent incremental checkpoint

### 11.3 File Naming Rules

| Scenario | Filename | Description |
|------|--------|------|
| Normal completion | `hccl_trace_output.json` | Complete trace |
| Incremental checkpoint | `hccl_trace_output.json` | Overwritten; always retains the latest state |
| Crash dump | `hccl_trace_crash_dump.json` | Trace snapshot at the time of the crash |
