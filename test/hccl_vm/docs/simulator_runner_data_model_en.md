# HCCL Simulator Runner Data Model Design

> **Naming and Reference Conventions**: Entity name = struct name in `include/runnerdb/sim_models.h` (CamelCase, implementation is authoritative); field name = field name in that file (snake_case). Database table names are found in `include/runnerdb/db_sim_sqlite_db.h`; some differ from struct names: `VirtualMemBlock`→table `VirMem`, `PhyMemBlock`→table `PhMem`, `IpcMemWhiteList`→table `IpMemWhiteList`. Legacy kebab-case names not yet replaced section by section (such as `phy-mem-id`) map to implementation field names by replacing `-` with `_`. Non-`sim_models.h` entities (such as ACL-side `KernelLaunchCfg`) retain their real field names; do not force snake_case conversion.
>
> **Interface Name Convention**: `aclrt*` in this document refers to ACL public interfaces (`runtime/include/external/acl/acl_rt.h`). For brevity, §7 flowcharts and interface tables omit the `acl` prefix (`rtLaunchKernel` means `aclrtLaunchKernel`). This notation is identical in form to the RTS layer's real names `rt*` (`runtime/pkg_inc/runtime/runtime/kernel.h`). All references follow the ACL layer; ACL type names (such as `aclrtLaunchKernelCfg`, `aclrtLaunchKernelAttr`) retain full names.

## Table of Contents

- [1. Software-Hardware Resource Interaction Modeling](#1-software-hardware-resource-interaction-modeling)
- [2. Device, Context, Stream and Other Hardware Resource Relationship Modeling](#2-device-context-stream-and-other-hardware-resource-relationship-modeling)
  - [2.1 Basic Device Relationship Modeling](#21-basic-device-relationship-modeling)
    - [2.1.1 Hierarchy Structure Overview](#211-hierarchy-structure-overview)
    - [2.1.2 Network Communication Resources](#212-network-communication-resources)
    - [2.1.3 Key Relationship Description](#213-key-relationship-description)
    - [2.1.4 Corresponding Interface Mapping Table](#214-corresponding-interface-mapping-table)
  - [2.2 Basic Memory Management Relationship Modeling](#22-basic-memory-management-relationship-modeling)
    - [2.2.1 Key Relationship Description (Basic Memory Management)](#221-key-relationship-description-basic-memory-management-relationship-modeling)
    - [2.2.2 Corresponding Interface Mapping Table (Basic Memory Management)](#222-corresponding-interface-mapping-table-basic-memory-management-relationship-modeling)
- [3. Data Task Model Extension on Basic Model](#3-data-task-model-extension-on-basic-model)
  - [3.1 Data / Task Flow Modeling](#31-data--task-flow-modeling)
  - [3.2 CCU Resource Modeling](#32-ccu-resource-modeling)
    - [3.2.1 Corresponding Interface Mapping Table (CCU Resource Modeling)](#321-corresponding-interface-mapping-table-ccu-resource-modeling)
    - [3.2.2 CCU Resource Lifecycle Description](#322-ccu-resource-lifecycle-description)
  - [3.3 Asynchronous / Synchronous Execution Modeling](#33-asynchronous--synchronous-execution-modeling)
    - [3.3.1 Notify Resource Management](#331-notify-resource-management)
    - [3.3.2 Notify Synchronization Control](#332-notify-synchronization-control)
    - [3.3.3 Event Resource Management](#333-event-resource-management)
    - [3.3.4 Event Flow Control](#334-event-flow-control)
- [4. Communicator Modeling](#4-communicator-modeling)
  - [4.1 Communicator Core Concepts](#41-communicator-core-concepts)
    - [4.1.1 Communicator Basic Definition](#411-communicator-basic-definition)
    - [4.1.2 Control Plane: Socket Communication](#412-control-plane-socket-communication)
    - [4.1.3 Data Plane: RDMA Communication](#413-data-plane-rdma-communication)
    - [4.1.4 Data Plane: UB Unified Bus](#414-data-plane-ub-unified-bus)
    - [4.1.5 Asynchronous Request Management](#415-asynchronous-request-management)
    - [4.1.6 Communicator Architecture Summary](#416-communicator-architecture-summary)
    - [4.1.7 Key Entity Comparison Table](#417-key-entity-comparison-table)
    - [4.1.8 HCCP Interface Classification](#418-hccp-interface-classification)
- [5. Callback and Report Relationship Modeling](#5-callback-and-report-relationship-modeling)
  - [5.1 Key Relationship Description (Callback and Report)](#51-key-relationship-description-callback-and-report-relationship-modeling)
    - [5.1.1 Corresponding Interface Mapping Table (Callback and Report)](#511-corresponding-interface-mapping-table-callback-and-report-relationship-modeling)
- [6. Basic `Device` Model Fine-Grained Low-Level Extension](#6-basic-device-model-fine-grained-low-level-extension)
  - [6.1 Key Relationship Description (Basic Device Model Extension)](#61-key-relationship-description-basic-device-model-fine-grained-low-level-extension)
    - [6.1.1 Corresponding Interface Mapping Table (Basic Device Model Extension)](#611-corresponding-interface-mapping-table-basic-device-model-fine-grained-low-level-extension)
- [7. Kernel Runtime Relationship Modeling](#7-kernel-runtime-relationship-modeling)
  - [7.1 Key Relationship Description (Kernel Runtime)](#71-key-relationship-description-kernel-runtime-relationship-modeling)
    - [7.1.1 Corresponding Interface Mapping Table (Kernel Runtime Relationship Modeling)](#711-corresponding-interface-mapping-table-kernel-runtime-relationship-modeling)

## Related Documents

- [`hccl_simulator.md`](./hccl_simulator_en.md): HCCL Simulator Requirements Analysis (upstream requirements for this document)
- [`checker_quick_intro.md`](./checker_quick_intro_en.md): Checker Quick Start Guide (basic concepts and processing flow)
- [`hccl_vm_binary_file_format.md`](./hccl_vm_binary_file_format_en.md): HCCL VM Binary Stream File Format Specification (database file format)
- [`header_dependency.md`](./header_dependency_en.md): CheckerL2 Third-Party Header Dependency List
- [`insightV3_guide.md`](./insightV3_guide_en.md): HVRM Insight V3 Usage Guide
- [`ranktable_rankid_device_conflict_analysis.md`](./ranktable_rankid_device_conflict_analysis_en.md): ranktable rankId and Device Physical ID Mapping Conflict Analysis

## 1. Software-Hardware Resource Interaction Modeling

Data flow interaction diagram.

```mermaid

sequenceDiagram
    participant HostThread as Host CPU (Runner)
    participant StreamQueue as Stream (Memory Queue)
    participant DeviceScheduler as Device Hardware (TS)
    participant EventMem as Event Status (Memory)

    Note over HostThread: 1. aclrtRecordEvent(evt1)
    HostThread->>StreamQueue: Push CMD: [Write Event1=Done]

    Note over HostThread: 2. aclrtStreamWaitEvent(evt1)
    HostThread->>StreamQueue: Push CMD: [Wait Event1==Done]

    Note over HostThread: 3. aclrtLaunchKernel(MatMul)
    HostThread->>StreamQueue: Push CMD: [Execute MatMul]

    Note over DeviceScheduler: Asynchronous Execution Phase (Device Side)

    StreamQueue->>DeviceScheduler: Pop CMD: [Write Event1]
    DeviceScheduler->>EventMem: Update Status to DONE

    StreamQueue->>DeviceScheduler: Pop CMD: [Wait Event1]
    DeviceScheduler->>EventMem: Check Status?
    Note right of DeviceScheduler: Found DONE, pass!<br/>(If NotReady, hardware spins here)

    StreamQueue->>DeviceScheduler: Pop CMD: [MatMul]
    DeviceScheduler->>DeviceScheduler: Start AI Core Computing...
```

Everything pushed into a Stream is executed by Device hardware.

## 2. Device, Context, Stream and Other Hardware Resource Relationship Modeling

Relationships among Device, Context, Stream, and user host threads. Communicator-side resources (Socket/RDMA/UB interfaces) are described in [§4 Communicator Modeling](#4-communicator-modeling).

```mermaid
graph TD
    subgraph Server1[Server 1]
        Host1
        Device
        Device2
    end

    subgraph Server2[Server 2]
        Host3
        Host2
        Device3
        Device4

    end

    subgraph Host1[Host1]
        Runner1[runner<br>user thread 1]
        Runner2[runner<br>user thread 2]
    end

    subgraph Host2[Host2]
        Runner3[Runner...]
    end

    subgraph Host3[Device CPU<br>edge/embedded: atlas 500]
        embedRunner[Embed Runner...]
    end

    subgraph Device[Device 1]
        Context1
    end

    subgraph Context1[run-Context1]
        Stream1
    end

    subgraph Stream1[ctx-Stream1]
        TaskKernel
    end

    subgraph TaskKernel[Task/Kernel]

    end

    subgraph Device2[Device 2]
        Context2
    end

    subgraph Context2[run-Context2]
        Stream2
    end

    subgraph Stream2[ctx-Stream2]
        Kernel
    end

    subgraph Kernel[Kernel]
    end

    subgraph Device3[Device 3]
        ctxM[...]
    end

    subgraph Device4[Device 4]
        ctxN[...]
    end

    Runner1-->Device
    Runner2-->Device
    Runner2-->Device2
    Runner3-->Device3
    embedRunner-->Device4

```

### 2.1 Basic Device Relationship Modeling

#### 2.1.1 Hierarchy Structure Overview

```mermaid
erDiagram
    %% ==========================================
    %% Layer 1: Physical Topology Layer (Server -> Host/Device)
    %% 🧠 Process-private candidates: Context / Stream / DeviceConnection (only src/proxy accesses; cmd only dumps)
    %% ==========================================
    Server {
        typ id PK
        typ server_id
        typ pod_id
        typ version
        typ hardware_type
        typ used_dev_num
    }
    Host {
        typ id PK
        typ host_id
        typ server_id FK
        typ ip_addr
        typ arch
    }
    Device {
        typ id PK
        typ device_id
        typ server_id FK
        typ user_id "🚧 field not in implementation"
        typ logic_id "usable device index; 🚧 field not in implementation"
        typ physical_id
        typ ccu_die_num "910D dual die; 🚧 field not in implementation"
        typ super_device_id
        typ overflow_mode
        typ status
        typ soc_version "A3"
        typ max_stream_cnt "1984; 🚧 field not in implementation"
    }
    %% Implementation has no ccu_die_num (die count see §3.2); max_stream_cnt is a capability value (aclrt_stream_stub.cc:287)
    Server ||--|{ Host : contains
    Server ||--o{ Device : contains

    %% ==========================================
    %% Layer 2: Process and Context Layer (Runner -> Context -> Stream)
    %% ==========================================
    Runner {
        typ id PK
        typ run_id
        typ host_id FK
        typ pid
        typ thread_id
        typ timeout_config_ms
        typ current_ctx_id FK
    }
    Context {
        typ id PK
        typ ctx_id
        typ run_id FK
        typ thread_id "🚧 field not in implementation"
        typ device_id FK
        typ is_default
        typ ref_cnt
        typ float_overflow_addr
        typ capture_mode
    }
    %% Context implementation has no thread_id (thread field is Runner.thread_id); ref_cnt is the implementation's reference count
    Stream {
        typ id PK
        typ stream_id
        typ ctx_id FK
        typ sq_base_addr
        typ is_primary_default
        typ is_other_default
        typ priority
        typ schedule_strategy
        typ failure_mode
        typ user_tag
        typ overflow_switch
        typ activated
        typ capture_status
        typ task_complete_status
    }
    Host ||--o{ Runner : runs
    Runner ||--o{ Context : "creates/owns"
    Runner |o--o{ Context : "current activates"
    Context }o--|| Device : binds
    Context ||--|{ Stream : owns
    Device ||..|{ Stream : "hardware constraint"

    %% ==========================================
    %% Layer 3: Device Internal Resource Layer (Port/EndPoint/Ccu)
    %% ==========================================
    Port {
        typ id PK
        typ port_id
        typ device_id FK
        typ die_id "die Id"
        typ status "0:unused/1:used"
        typ name "0/0, 0/1"
    }
    Rank {
        typ id PK
        typ device_id FK
        typ rank_id
        typ comm_id
    }
    EndPoint {
        typ id PK
        typ endpoint_id
        typ rank_id FK "🚧 field not in implementation"
        typ device_id FK
        typ func_id
        typ die_id
        typ addr "IP address/EID; 🚧 field not in implementation"
        typ type "0-EID/1-IPV4/2-IPV6"
        typ eid "16-byte EID"
        typ ip_addr "64-byte IP"
        typ status "🚧 field not in implementation"
        typ is_uboe "🚧 field not in implementation"
    }
    Ccu {
        typ id PK
        typ ccu_id
        typ device_id FK
        typ resource_addr
        typ die_id
        typ status
    }
    DeviceConnection {
        typ id PK
        typ connection_id
        typ src_dev_id FK
        typ dst_dev_id FK
        typ link_type
        typ access_by_remote
    }
    Device ||--o{ Port : "has"
    Device ||--o{ Rank : "has"
    Device ||--o{ EndPoint : "has"
    Device ||--o{ Ccu : "1:2 dual die"
    Device ||--o{ DeviceConnection : "peer access"

    %% Legend: 🟥 unimplemented (no entity with same name in repo) 🟨 in-process/dump access only  No color = has cross-process consumer
    classDef unimplemented fill:#ffe0e0,stroke:#c0392b,stroke-width:2px,stroke-dasharray:5 3
    classDef inproc fill:#fff6d6,stroke:#d4a017,stroke-width:2px
    class Rank unimplemented
    class Runner,Context,Stream,DeviceConnection inproc
```

#### 2.1.2 Network Communication Resources

```mermaid
erDiagram
    %% ==========================================
    %% Network Topology and Connection Layer
    %% CcuChannel: only cmd dump access, no writer in repo (empty table)
    %% ==========================================
    EndPoint {
        typ id PK
        typ endpoint_id
        typ device_id FK
        typ rank_id FK "🚧 field not in implementation"
        typ func_id "used by ccu"
        typ die_id
        typ addr "IP address/EID; 🚧 field not in implementation"
        typ type "0-EID/1-IPV4/2-IPV6"
        typ eid "16-byte EID"
        typ ip_addr "64-byte IP"
        typ status "🚧 field not in implementation"
        typ is_uboe "🚧 field not in implementation"
    }
    Port {
        typ id PK
        typ port_id
        typ device_id FK
        typ die_id "die Id"
        typ name "0/0, 0/1"
    }
    EndPointPortMapping {
        typ id PK
        typ port_id FK
        typ endpoint_id FK
        typ net_layer
    }
    %% Physical connections defined by topo.json
    Link {
        typ id PK
        typ link_id
        typ local_endpoint_id FK
        typ remote_endpoint_id FK
        typ net_layer "network layer"
        typ type "connection type"
        typ protocols "protocols[8], corresponds to design LinkProtocolMapping"
    }
    LinkProtocolMapping {
        typ link_protocol_mapping_id PK
        typ link_id FK
        typ protocol "UB_CTP/UB_MEM/..."
    }

    EndPointPair {
        typ id PK
        typ local_endpoint_id FK "implementation spelling local_enpoint_id"
        typ remote_endpoint_id FK "implementation spelling remote_enpoint_id"
        typ tp_type "transport type"
    }
    CcuChannel {
        typ id PK
        typ ccu_channel_id
        typ channel_id FK   "business allocation"
        typ local_endpoint_id FK
        typ remote_endpoint_id FK
        typ protocol "communication protocol"
        typ jetty_start "jetty start Id"
        typ jetty_num "jetty count"
    }

    EndPointPortMapping }|--|| Port : maps
    EndPointPortMapping }|--|| EndPoint : maps
    Link ||--o{ EndPoint : connects
    Link ||--|{ LinkProtocolMapping : "supports"
    EndPointPair ||--|{ EndPoint : mapping
    EndPointPair }o..|| Link : "based-on"
    CcuChannel ||--|{ EndPoint : uses
    CcuChannel }o..|| Link : "based-on"

    classDef unimplemented fill:#ffe0e0,stroke:#c0392b,stroke-width:2px,stroke-dasharray:5 3
    classDef inproc fill:#fff6d6,stroke:#d4a017,stroke-width:2px
    class LinkProtocolMapping unimplemented
    class CcuChannel inproc
```

> `LinkProtocolMapping` is not an independent struct/table in the implementation. Instead, it is the `Link.protocols[8]` array field (see `include/runnerdb/sim_models.h`). Treat it as an array field during troubleshooting or table creation; do not split it into an independent table per this diagram.

#### 2.1.3 Key Relationship Description

##### Layer 1: Physical Topology Layer

| Relationship | Meaning | Description/Notes |
| --- | --- | --- |
| Server → Host | One Server may contain multiple Host instances | For example, multi-CPU or virtualized environments |
| Server → Device | One Server contains multiple AI devices | Corresponds to /dev/davinci0, /dev/davinci1, and so on |

##### Layer 2: Process and Context Layer

| Relationship | Meaning | Description/Notes |
| --- | --- | --- |
| Host → Runner | Each host machine has multiple application threads (Runner) | Each Runner can create multiple Contexts |
| Runner → Context | A thread creates or switches to different Contexts | Uses aclrtCreateContext() and aclrtSetCurrentContext() |
| Context → Device | Context binds to a Device | Cannot cross devices once created |
| Context → Stream | Each Context can create multiple Streams | Corresponds to aclrtCreateStream() |
| Runner ↔ Context (current) | Current context activation state | aclrtGetCurrentContext(), aclrtSetCurrentContext() |
| Device .. Stream | Resource upper limit constraint | Stream count is limited by hardware (`max_stream_cnt`, source `src/proxy/level2/aclrt_stream_stub.cc:244,287`) |

##### Layer 3: Device Internal Resource Layer

| Relationship | Meaning | Description/Notes |
| --- | --- | --- |
| Device → Port | A device contains multiple communication ports | Used for network topology connections, format such as "0/0, 0/1" |
| Device → Rank | A device is associated with a communication Rank | Rank is the participant node identifier in a communicator |
| Device → EndPoint | A device contains multiple endpoints | IP address/EID addressing identifiers for network communication |
| Device → Ccu | A device contains multiple CCU units | 910D has a dual-die architecture, one CCU per die |
| Device → DeviceConnection | Inter-device communication channels | aclrtDeviceCanAccessPeer(), aclrtDeviceEnablePeerAccess() |

##### Network Communication Resource Layer

| Relationship | Meaning | Description/Notes |
| --- | --- | --- |
| EndPoint ↔ Port | Endpoint-to-port mapping | Implements many-to-many mapping through EndPointPortMapping |
| Link → EndPoint | Physical connection associates endpoints | Defined by topo.json, describes physical topology |
| Link → LinkProtocolMapping | Protocols supported by a connection | One Link can support multiple protocols (UB_CTP/UB_MEM, and so on) |

#### 2.1.4 Corresponding Interface Mapping Table

##### Basic and Device Layer (Device / Server)

| Entity/Property                   | Key API Interfaces                                                                                          |
| --------------------------- | ---------------------------------------------------------------------------------------------------- |
| Server/Host                 | `aclInit`, `aclFinalize`                                                                             |
| Runner.pid                  | `rtDeviceGetBareTgid`                                                                                |
| Device                      | `rtGetDeviceCount`                                                                                   |
| Device.id             | `rtSetDevice`,`rtResetDevice`,`rtGetDevice`,`rtsGetLogicDevIdByPhyDevId`                             |
| Device.id          | `rtGetPhyDevIdByLogicDevId`                                                                          |
| Device.name             | `rtGetSocName`                                                                                       |
| Device.mode        | `rtSetDeviceSatMode`,`rtGetDeviceSatMode`                                                            |
| DeviceConnection            | `rtGetDevicesTopo`,`rtDeviceDisablePeerAccess`,`rtDeviceEnablePeerAccess`,`rtDevicePeerAccessStatus` |
| Context.id          | `rtCreateContext`,`rtDestroyContext`                                                                 |
| Context.default          | `rtSetCurrentContext`,`rtGetCurrentContext`                                                          |
| Context.addr | `rtCtxGetFloatOverflowAddr`                                                                          |
| Stream table                    | `aclrtGetStreamAvailableNum`                                                                         |
| Stream.id            | `rtCreateStream`,`rtCreateStreamWithConfig`,`rtDestroyStream`,`rtDestroyStreamForce`                 |
| Stream.status | `rtSynchronizeStream`, `rtSynchronizeStreamWithTimeout`                                              |
| Stream.activated            | `rtStreamStop`                                                                                       |
| Stream.mode         | `rtSetStreamAttribute`,`rtGetStreamAttribute`                                                        |

### 2.2 Basic Memory Management Relationship Modeling

Local memory blocks and remote memory import (`RaCtxLmemRegister` / `RaCtxRmemImport`) corresponding interfaces are described in [§4.1.4 Data Plane: UB Unified Bus](#414-data-plane-ub-unified-bus).

```mermaid
erDiagram
    %% 🧠 Process-private candidates: IpcMemRecord / IpcMemWhiteList / FdMemWhiteList (only src/proxy accesses)
    %% FdMemRecord: only cmd dump, no writer (empty table)
    PhyMemBlock ||--o{ VirtualMemBlock : "physical to virtual mapping"
    PhyMemBlock ||--o{ FdMemRecord : "file descriptor mapping"
    PhyMemBlock {
        typ id PK "auto-increment ID"
        typ phy_mem_id
        typ device_id FK "0,1... or -1(host)"
        typ size
        typ type
        typ ref_count
        typ name "64 bytes"
        typ is_freed
    }

    VirtualMemBlock {
        typ id PK "auto-increment ID"
        typ start_ptr "per-card virtual address"
        typ size
        typ ctx_id FK
        typ phy_mem_id FK
        typ owner_pid "creating process"
        typ src_type ""
        typ policy
        typ dev_mapped_ptr "device-side mapped address"
        typ is_dev_access
        typ device_id FK
        typ rank_id FK
    }

    VirtualMemBlock ||--o{ IpcMemRecord : "shared memory registration"
    IpcMemRecord {
        typ id PK
        typ ipc_id
        typ vir_mem_id FK
        typ phy_mem_id FK "🚧 field not in implementation"
        typ name_or_key "🚧 field not in implementation"
        typ create_pid
        typ offset
    }

    IpcMemRecord ||--o{ IpcMemWhiteList : "process whitelist"
    IpcMemWhiteList {
        typ id PK
        typ name_or_key "carries original design ipc_id: stores IpcMemRecord index"
        typ pid
        typ create_pid
    }

    FdMemRecord {
        typ id PK "database key (hardcoded id)"
        typ fd
        typ phy_mem_id FK
        typ name "🚧 field not in implementation"
        typ type "🚧 field not in implementation"
        typ vir_mem_id FK
        typ create_pid
    }

    FdMemRecord ||--o{ FdMemWhiteList : "process whitelist"
    FdMemWhiteList {
        typ id PK
        typ name_or_key "carries original design fd_id: stores shareableHandle"
        typ pid
        typ create_pid
    }

    %%VirtualMemBlock ||--o{ MemMapRecord : "mapping relationship"
    %%MemMapRecord {
    %%    typ ptr FK
    %%    typ phy_mem_id FK
    %%}

    classDef inproc fill:#fff6d6,stroke:#d4a017,stroke-width:2px
    class IpcMemRecord,IpcMemWhiteList,FdMemRecord,FdMemWhiteList inproc
```

> **Database Table Names**: Entities in this diagram map to tables registered in `include/runnerdb/db_sim_sqlite_db.h`. Use table names when querying or modifying the database — `PhyMemBlock`→`PhMem`, `VirtualMemBlock`→`VirMem`, `IpcMemWhiteList`→`IpMemWhiteList`; field names correspond to `include/runnerdb/sim_models.h`.

#### 2.2.1 Key Relationship Description (Basic Memory Management Relationship Modeling)

1. **Physical Memory Central Role**.
   `PhyMemBlock` serves as the base entity, associating with all other entities through `phy_mem_id`, reflecting the Ascend "physical memory pooling" design philosophy.
2. **Three-Layer Mapping System**:

   - Physical→Virtual (`VirtualMemBlock`)
   - Physical→IPC shared (`IpcMemRecord`)
   - Physical→File descriptor (`fdMemRecord`)
3. **Security Control**:
   `IpcMemWhiteList` implements Huawei HCCS (Huawei Collective Communication Service) secure sharing through process PID whitelists.
4. **Special Mapping Types**:
   `MemMapRecord` records dual virtual address mapping scenarios (such as mappings produced by `aclrtMapMem`), supporting Ascend NPU zero-copy data transfers.

#### 2.2.2 Corresponding Interface Mapping Table (Basic Memory Management Relationship Modeling)

| Entity                           | Key Management Interfaces                                                                                                                                    |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| PhyMemBlock                    | `rtMallocPhysical`, `rtFreePhysical`                                                                                                            |
| VirtualMemBlock            | `rtMallocWithCfg`,`rtMallocForTaskScheduler`,`rtMallocHostWithCfg`,`rtFree`,`rtReserveMemAddress`,`ReleaseMemAddress`, `rtMapMem`, `rtUnmapMem` |
| VirtualMemBlock.ctx_id | `rtPointerGetAttributes`                                                                                                                        |
| FdMemRecord.fd                 | `rtMemExportToShareableHandle`, `rtMemImportFromShareableHandle`                                                                                |
| FdMemWhiteList.pid             | `rtMemSetPidToShareableHandle`                                                                                                                  |
| IpcMemRecord.name_or_key       | `rtIpcMemGetExportKey`                                                                                                                          |
| IpcMemRecord.id            | `rtIpcMemImportByKey`,`IpcMemClose`                                                                                                             |
| IpcMemWhiteList.pid            | `rtIpcMemSetImportPid`                                                                                                                          |
| MemMapRecord                   | `rtHostRegister`, `rtHostUnRegister`                                                                                                            |

## 3. Data Task Model Extension on Basic Model

### 3.1 Data / Task Flow Modeling

Task placement relationships on Streams; the Task family is carried by `Task` / `EventSyncTask` in the implementation. For synchronization primitives, see [§3.3 Asynchronous / Synchronous Execution Modeling](#33-asynchronous--synchronous-execution-modeling).

```mermaid
erDiagram
    Context {
        typ id PK
        typ ctx_id
        typ run_id FK
    }

    Stream {
        typ id PK
        typ stream_id
        typ ctx_id FK
        typ activated
        typ state "Running/Idle; 🚧 field not in implementation"
    }
    %% Implementation has no state field; run state is expressed by activated/capture_status/task_complete_status

    %% Stream contains an ordered task list
    Context ||--o{ Stream : "manages/submits"
    Stream ||--o{ Task : "queues [1..*]"

    Task {
        typ id PK
        typ task_id
        typ stream_id FK
        typ seq_number "auto-increment within stream"
        typ type "Kernel/Memcpy/Callback"
        typ cid
    }

    %% Various specific Task types (logical inheritance)
    MemcpyTask {
        typ task_id FK
        typ src_addr
        typ dst_addr
        typ size
    }

    %% Inheritance logical expression (Task has multiple types)
    %% 🧠 Process-private candidates: Context / Stream / Task (only src/proxy accesses)
    %% MemcpyTask is discriminated by Task.type (implementation only creates Task table)
    Task ||--|{ MemcpyTask : "is a"

    %% MemcpyTask addresses should be addressable in VirtualMemBlock
    MemcpyTask }o..|{ VirtualMemBlock : "Range Constraint"
    VirtualMemBlock {
        typ id PK
        typ start_ptr
        typ ctx_id FK
    }

    VirtualMemBlock }o--|| Context : "belongs to"

    classDef unimplemented fill:#ffe0e0,stroke:#c0392b,stroke-width:2px,stroke-dasharray:5 3
    classDef inproc fill:#fff6d6,stroke:#d4a017,stroke-width:2px
    class MemcpyTask unimplemented
    class Context,Stream,Task inproc

```

### 3.2 CCU Resource Modeling

One NPU device contains 2 CCUs, for die0 and die1 respectively. CCU die-level hardware synchronization resources (Notify) are described in [§3.3.1 Notify Resource Management](#331-notify-resource-management); UB-side context interfaces are described in [§4.1.4 Data Plane: UB Unified Bus](#414-data-plane-ub-unified-bus).

```mermaid
graph RL
    subgraph DavidDevice0[David 0]
        direction RL
        Memory0[Memory]
        David0Die0[Die0_ccu]
        David0Die1[Die1_ccu]
    end

    subgraph David0Die0[Die0_ccu]

        CcuBuf00[CcuBuf]
        Variable00[Variable]
        Notify00[Notify]
        CompletedEvent00[CompletedEvent]
        Local/Rmt-Addr00[Local/Rmt-Addr]
    end

    subgraph David0Die1[Die1_ccu]

        CcuBuf01[CcuBuf]
        Variable01[Variable]
        Notify01[Notify]
        CompletedEvent01[CompletedEvent]
        Local/Rmt-Addr01[Local/Rmt-Addr]
    end

    David0Die0---Memory0
    David0Die1---Memory0

```

#### 3.2.1 Corresponding Interface Mapping Table (CCU Resource Modeling)

| Entity               | Key Management Interfaces                                             |
| ------------------ | -------------------------------------------------------- |
| CcuBuf             | `rtCcuBufAlloc`, `rtCcuBufFree`, `rtCcuBufGetAddr`       |
| Variable           | `rtVariableCreate`, `rtVariableDestroy`, `rtVariableSet` |
| Notify             | `rtCreateNotify`, `rtDestroyNotify`                      |
| CompletedEvent     | `rtCreateEvent`, `rtDestroyEvent`                        |
| Local/Rmt-Addr     | `rtGetDeviceLocalAddr`, `rtGetDeviceRemoteAddr`          |
| CCU Resource Query        | `rtGetCcudieInfo`, `rtGetCcudieNum`                      |

#### 3.2.2 CCU Resource Lifecycle Description

**CCU Initialization Flow**:

1. Two CCUs (die0/die1) initialize automatically when the Device starts.
2. Each CCU allocates independent CcuBuf, Variable, and Notify resource pools.
3. CompletedEvent notifies task completion status.

**Resource Constraints**:

- CcuBuf count per CCU is limited (depends on Device.version)
- Variable stores shared variables during communication.
- Notify provides cross-CCU synchronization notification mechanisms.
- Local/Rmt-Addr handles address translation for cross-die communication.

> Note: The Notify inside CCU here is a die-level hardware synchronization resource (hcomm-side concept). It is not the same entity as the `Notify` managed by `rtCreateNotify` in §3.3.1 (Device/Context level, database table `sim::Notify`: `create_ctx_id` + `device_notify_seq`).

### 3.3 Asynchronous / Synchronous Execution Modeling

#### 3.3.1 [Notify Resource Management](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/appdevg/acldevg/aclcppdevg_000524.html)

```mermaid
erDiagram
    %% 🧠 Process-private candidates: Context / IpcNotify / IpcNotifyVistorList (only src/proxy accesses)
    Device ||..o{ Notify : "Hardware Limit"
    Device ||--o{ Context : "referred by"
    Device {
        typ id PK
        typ device_id
        typ device_type "A3; 🚧 field not in implementation"
        typ max_notify_cnt "8192; 🚧 field not in implementation"
    }
    Context {
        typ id PK
        typ ctx_id
        typ device_id FK
    }

    %% IpcNotify: IPC shared notification from same source as Notify, not separately modeled in implementation
    Notify o|--|| Context : "record"
    Notify {
        typ id PK
        typ notify_id
        typ create_ctx_id FK
        typ device_notify_seq "0~8191"
        typ value "notify read/write register"
    }

    IpcNotify {
        typ id PK
        typ ipc_id
        typ notify_id FK
        typ name_or_key
        typ create_pid
    }

    Notify ||--o| IpcNotify : "is a"
    IpcNotify ||--o{ IpcNotifyVistorList : "has"
    IpcNotifyVistorList {
        typ id PK
        typ ipc_id FK
        typ visitor_pid
    }

    classDef inproc fill:#fff6d6,stroke:#d4a017,stroke-width:2px
    class Context,IpcNotify,IpcNotifyVistorList inproc
```

##### Key Relationship Description ([Notify Resource Management](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/appdevg/acldevg/aclcppdevg_000524.html))

**Notify and Device Hardware Constraints**:

- Each Device has a Notify count upper limit of `max_notify_cnt` (for example, 8192 for A3 chips, source `src/plugin/runner/runner_utils/device_resource.h:22 MAX_NOTIFY_NUM`)
- `device_notify_seq` is the physical index of a Notify within a Device (0~8191)
- Notify creation must specify the owning Context; the Context is bound to a specific Device.

**Notify IPC Sharing Mechanism**:

- `IpcNotify` allows cross-process sharing of Notify instances.
- `name_or_key` is the sharing identifier, obtained through `rtNotifyGetExportKey`.
- Other processes import and use it through `rtNotifyImportByKey`.
- `IpcNotifyVistorList` records the PIDs of processes authorized to access this Notify.

**Notify State Management**:

- The `value` field maps to hardware registers for read/write state.
- `rtWaitAndResetNotify` waits for Notify to become Ready and resets it.
- Notify is used for inter-Stream synchronization and cross-process synchronization scenarios.

##### Corresponding Interface Mapping Table ([Notify Resource Management](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/appdevg/acldevg/aclcppdevg_000524.html))

| Entity                       | Key Management Interfaces                                        |
| -------------------------- | --------------------------------------------------- |
| Notify.id           | `rtCreateNotify`,`rtDestroyNotify`,`rtGetNotifyId` |
| Notify.value               | `lrtWaitAndResetNotify`, `rtWaitAndResetNotify`     |
| IpcNotify.name_or_key      | `rtNotifyGetExportKey`,`rtNotifyImportByKey`        |
| IpcNotifyVistorList.ipc_id | `rtNotifySetImportPid`                              |

#### 3.3.2 Notify Synchronization Control

```mermaid
erDiagram
    Context {
        typ id PK
        typ ctx_id
        typ run_id FK
    }

    Stream {
        typ id PK
        typ stream_id
        typ ctx_id FK
        typ activated
        typ state "Running/Idle; 🚧 field not in implementation"
    }
    %% Implementation has no state field; run state is expressed by activated/capture_status/task_complete_status

    %% Stream contains an ordered task list
    Context ||--o{ Stream : "manages/submits"
    Stream ||--o{ Task : "queues [1..*]"

    Task {
        typ id PK
        typ task_id
        typ stream_id FK
        typ seq_number "auto-increment within stream"
        typ type "Notify"
    }

    %% Various specific Task types (logical inheritance)
    NotifyRecordTask {
        typ id PK
        typ notify_id FK

    }

    NotifyWaitTask {
        typ id PK
        typ notify_id FK
    }

    NotifyRecordTask }o..|| Notify : "use"
    NotifyWaitTask }o..|| Notify : "use"
    Notify {
        typ id PK
        typ notify_id
        typ value
    }
    %% Inheritance logical expression (Task has multiple types)
    %% 🧠 Process-private candidates: Context / Stream / Task (only src/proxy accesses)
    %% NotifyRecordTask/NotifyWaitTask are discriminated by Task.type (implementation only creates Task table)
    Task ||--|{ NotifyRecordTask : "is a"
    Task ||--|{ NotifyWaitTask : "is a"

    classDef inproc fill:#fff6d6,stroke:#d4a017,stroke-width:2px
    class Context,Stream,Task,NotifyRecordTask,NotifyWaitTask inproc
```

##### Key Relationship Description (Notify Synchronization Control)

**Notify Task Types**:

- `NotifyRecordTask`: Sets the Notify state to Ready, indicating an event has completed.
- `NotifyWaitTask`: Waits for the Notify state to become Ready, implementing inter-Stream synchronization.

**Task Execution Order**:

- NotifyRecordTask executes in StreamA, setting Notify to Ready.
- NotifyWaitTask executes in StreamB, waiting for the same Notify.
- StreamB subsequent tasks can execute only after Notify becomes Ready.

**Cross-Stream Synchronization Example**:

```text
StreamA: Task1 -> NotifyRecordTask(notify_id=1) -> Task2
StreamB: NotifyWaitTask(notify_id=1) -> Task3
// Task3 must wait for Task1 to complete before executing
```

##### Corresponding Interface Mapping Table (Notify Synchronization Control)

| Entity             | Key Management Interfaces            |
| ---------------- | ----------------------- |
| NotifyRecordTask | `rtRecordNotify`        |
| NotifyWaitTask   | `lrtWaitAndResetNotify` |

#### 3.3.3 Event Resource Management

```mermaid
erDiagram
    %% 🧠 Process-private candidates: Event (only src/proxy accesses)
    Device ||..|{ Event : "Hardware Limit"
    Device ||--o{ Context : "refered by"
    Device {
        typ id PK
        typ device_id
        typ device_type "A3; 🚧 field not in implementation"
        typ max_event_cnt "65535; 🚧 field not in implementation"
    }

    Event }o..|| Context : "created by"
    Event {
        typ id PK
        typ event_id
        typ create_ctx_id FK
        typ event_flag
        typ device_res_seq "0~65535"
        typ created_time
        typ status
    }

    classDef inproc fill:#fff6d6,stroke:#d4a017,stroke-width:2px
    class Event inproc
```

##### Key Relationship Description (Event Resource Management)

**Event and Device Hardware Constraints**:

- Each Device has an Event count upper limit of `max_event_cnt` (for example, 65535 for A3 chips, source `src/proxy/level2/aclrt_event_stub.cc:191,220`)
- `device_res_seq` is the physical index of an Event within a Device (0~65535)
- Event creation must specify the owning Context; the Context is bound to a specific Device.

**Event and Context Relationship**:

- `create_ctx_id` records the creating Context of an Event.
- Events can be shared across multiple Streams but must belong to the same Context.
- Cross-Context Event sharing requires IPC mechanisms (similar to Notify)

**Event State Management**:

- The `status` field represents the current Event state: NotRecorded/Recorded/Completed.
- `event_flag` controls Event behavior (such as auto-reset)
- `created_time` is used for performance statistics.

##### Corresponding Interface Mapping Table (Event Resource Management)

| Entity           | Key Management Interfaces                                                             |
| -------------- | ------------------------------------------------------------------------ |
| Event.id | `rtCreateEvent`, `rtCreateEventWithFlag`,`rtDestroyEvent`,`rtGetEventId` |
| Event.status   | `rtRecordEvent`,`rtQueryEventStatus`                                     |

#### 3.3.4 Event Flow Control

```mermaid
erDiagram
    Context {
        typ id PK
        typ ctx_id
        typ run_id FK
    }

    Stream {
        typ id PK
        typ stream_id
        typ ctx_id FK
        typ activated
        typ state "Running/Idle; 🚧 field not in implementation"
    }
    %% Implementation has no state field; run state is expressed by activated/capture_status/task_complete_status

    %% Stream contains an ordered task list
    Context ||--o{ Stream : "manages/submits"
    Stream ||--o{ Task : "queues [1..*]"

    Task {
        typ id PK "auto-increment"
        typ task_id
        typ stream_id FK
        typ seq_number "auto-increment within stream"
        typ type "EVENT"
    }

    %% Various specific Task types (logical inheritance)
    EventTask {
        typ task_id FK
        typ event_id FK
        typ execute_time_ms
        typ finish_time_ms
        typ first_capture_taskid FK
    }

    EventRICaptureTask {
        typ updated_time
    }

    EventSyncTask {
        typ id PK
        typ event_id FK "🚧 field not in implementation"
        typ execute_time_ms
        typ finish_time_ms
        typ op_timeout_s
    }
    EventRecordTask {
        typ event_id FK
        typ execute_time_ms
        typ finish_time_ms
    }
    EventWaitTask {
        typ event_id FK
        typ execute_time_ms
        typ finish_time_ms
    }
    EventTimeTask {
        typ event_id FK
        typ execute_time_ms
    }
    EventTraceTask {
        typ event_id FK
        typ start_task_id FK
    }

    %% 🧠 Process-private candidates: Context / Stream / Task (only src/proxy accesses)
    %% Inheritance logical expression (Task has multiple types): discriminated by Task.type, implementation only creates Task table
    Task ||--|{ EventTask : "is a "
    EventTask ||--|{ EventRICaptureTask : "is a EXTERNAL"
    EventTask ||--|{ EventSyncTask : "is a EX"
    EventTask ||--|{ EventTimeTask : "is a EX"
    EventTask ||--|{ EventTraceTask : "is a EX"
    EventSyncTask ||--|{ EventRecordTask : "is a EX"
    EventSyncTask ||--|{ EventWaitTask : "is a EX"

    classDef unimplemented fill:#ffe0e0,stroke:#c0392b,stroke-width:2px,stroke-dasharray:5 3
    classDef inproc fill:#fff6d6,stroke:#d4a017,stroke-width:2px
    class EventTask,EventRICaptureTask,EventRecordTask,EventWaitTask,EventTimeTask,EventTraceTask unimplemented
    class Context,Stream,Task,EventSyncTask inproc
```

##### Key Relationship Description (Event Flow Control)

**Event Task Type Classification**:

- `EventRecordTask`: Sets the Event state to Recorded/Completed.
- `EventWaitTask`: Waits for the Event state to become Completed.
- `EventSyncTask`: Synchronously waits for Event completion (blocking call)
- `EventRICaptureTask`: Special recording task in RI Capture mode.
- `EventTimeTask`: Records timestamp-related tasks.
- `EventTraceTask`: Task records for performance tracing.

**Event Task Inheritance Hierarchy**:

- `EventTask` is the base class, containing task-id, event-id, execute-time, finish-time. 🚧 (Implementation `EventSyncTask` has no `task_id`/`event_id`; Event identity is `Event.id`; time field implementation names are `execute_time_ms`/`finish_time_ms`)
- `EventSyncTask` inherits EventTask and adds the op_timeout_s timeout parameter.
- `EventRecordTask` and `EventWaitTask` inherit EventSyncTask.
- `EventRICaptureTask`, `EventTimeTask`, and `EventTraceTask` directly inherit EventTask.

**Event Execution Flow**:

```text
StreamA: KernelTask -> EventRecordTask(event_id=1)
StreamB: EventWaitTask(event_id=1) -> KernelTask2
// StreamB KernelTask2 must wait for StreamA KernelTask to complete
```

**Task Tracing Relationships**:

- `first_capture_task_id` records the first Capture Task ID.
- `EventTraceTask.id` associates the tracing start task.
- EventRecordTask implements cross-Stream synchronization through EventWaitTask mapping.

##### Corresponding Interface Mapping Table (Event Flow Control)

| Entity            | Key Management Interfaces                                         |
| --------------- | ---------------------------------------------------- |
| EventTask       | `rtRecordEvent`, `rtResetEvent`,`rtSynchronizeEvent` |
| EventRecordTask | `rtRecordEvent`, `rtResetEvent`                      |
| EventWaitTask   | `rtStreamWaitEvent`, `rtQueryEventWaitStatus`        |
| EventTimeTask   | `rtResetEvent`, `rtRecordEvent`                      |
| EventTraceTask  | `rtResetEvent`, `rtRecordEvent`                      |

## 4. Communicator Modeling

Cross-machine communication involves multi-communicator hybrid task orchestration.
A communicator is essentially a multi-card network topology maintained by HCCL at the framework layer through the link-building capability provided by Rdma_Agent, stored in the host process. The interface list for each plane is described in [§4.1.8 HCCP Interface Classification](#418-hccp-interface-classification).

### 4.1 Communicator Core Concepts

A Communicator is the basic abstraction of HCCL collective communication. Each communicator defines a group of participating Ranks and their topology relationships. Rank / EndPoint entity modeling is described in [§2.1.2 Network Communication Resources](#212-network-communication-resources).

#### 4.1.1 Communicator Basic Definition

```mermaid
erDiagram
    %% ==========================================
    %% Communicator Definition (HCCL Communicator)
    %% ==========================================
    Communicator {
        typ id PK
        typ comm_id
        typ run_id FK "owning Runner process; 🚧 field not in implementation"
        typ world_size "total Rank count; 🚧 field not in implementation (implementation name rank_size)"
        typ my_rank "current Rank; 🚧 field not in implementation (implementation name rank_id)"
        typ color "sub-communicator color identifier; 🚧 field not in implementation"
        typ new_comm_id FK "derived new communicator; 🚧 field not in implementation"
        typ rank_size "implementation field name (corresponds to world_size)"
        typ rank_id "implementation field name (corresponds to my_rank)"
        typ device_id FK
        typ comm_hash
        typ deterministic
        typ op_expansion_mode
        typ rdma_traffic_class
        typ rdma_service_level
        typ sym_win_addr
        typ sym_win_size
        typ sym_win_registered
        typ status
    }
    Rank {
        typ rank_id PK "Rank number (0~world-size-1)"
        typ device_id FK "bound device"
        typ comm_id FK "owning communicator"
    }
    Runner ||--o{ Communicator : "creates/holds"
    Communicator }o--|| Device : "binds"
    Rank }o--|| Device : "binds"
    Communicator ||--o{ Communicator : "derives(MPI_Comm_split)"

    classDef unimplemented fill:#ffe0e0,stroke:#c0392b,stroke-width:2px,stroke-dasharray:5 3
    class Rank unimplemented
```

#### 4.1.2 Control Plane: Socket Communication

```mermaid
erDiagram
    %% ==========================================
    %% Socket Communication (RaSocket Series Interfaces)
    %% 🧠 Process-private candidates: RaSocketPair (only src/proxy accesses; RaSocket in same section is 🧊)
    %% 🧊 RaSocket table: zero references in repo (only registered, no read/write) — does not need database persistence
    %% ==========================================
    Device ||--o{ RaSocket : creates
    %% 🧊 Zero-reference table: only has struct and registration, no database read/write
    RaSocket {
        typ id PK "database key (hardcoded id)"
        typ device_id
        typ role "0:server,1:client"
        typ state "0: inited 1:listened"
        typ endpoint_id "ip id"
        typ slot_idx "slot index"
    }
    RaSocketPair {
        typ id PK "database key (hardcoded id)"
        typ server_id FK
        typ client_id FK
        typ ref_cnt
        typ port
        typ tag_hash
        typ buf_status "0: buffer pending, 1: buffer ready"
        typ slot_idx "slot index"
    }
    RaSocket ||--o| RaSocketPair : "participates in connection"
    RaSocketPair }o--|| CommMemSlot : "associated communication memory(slot_idx)"

    %% store layer communication memory pool: mmap shared memory divided by slots (store_sim_comm_memory_manager.h), not a RunnerDB registered table
    CommMemSlot {
        typ slot_idx PK "slot index (RaSocketPair.slot_idx; AllocSlot on connect, CloseSlot on disconnect)"
        typ ref_cnt "reference count (same slot can be shared by multiple connection pairs, AddRef)"
        typ c2s_size "client→server bytes written (Send/Recv)"
        typ s2c_size "server→client bytes written (Send/Recv)"
    }

    %% Socket Event Management (Epoll mechanism)
    RaSocketEvent {
        typ event_handle PK "event handle"
        typ max_events "maximum event count"
        typ timeout "timeout(ms)"
    }
    RaEpoll {
        typ epoll_id PK "Epoll ID"
        typ event_handle FK "associated event handle"
        typ socket_handle FK "monitored Socket handle"
        typ events "event types of interest"
    }
    RaSocketEvent ||--o{ RaEpoll : "manages"
    RaSocket ||--o{ RaEpoll : "monitored"

    classDef unimplemented fill:#ffe0e0,stroke:#c0392b,stroke-width:2px,stroke-dasharray:5 3
    classDef inproc fill:#fff6d6,stroke:#d4a017,stroke-width:2px
    class CommMemSlot,RaSocketEvent,RaEpoll unimplemented
    class RaSocket,RaSocketPair inproc
```

#### 4.1.3 Data Plane: RDMA Communication

```mermaid
erDiagram
    %% ==========================================
    %% RDMA Devices and Resources (RaRdev, RaQp, RaMr Interfaces)
    %% 🧠 Process-private candidates: RaQP / RaCQ / RaCQE / RaMR / RaDevice (only src/proxy accesses; RaQP has peer filtering)
    %% ==========================================
    Device ||--|{ RaDevice : "has virtual NIC"
    RaDevice {
        typ id PK "RDMA device handle"
        typ rdev_handle
        typ device_id FK "associated NPU Device"
        typ mac_addr "MAC address"
        typ ip_addr "IP address; 🚧 field not in implementation"
        typ state "UP/DOWN"
        typ port_num "physical port number; 🚧 field not in implementation"
        typ link_speed "link rate; 🚧 field not in implementation"
        typ mtu "maximum transmission unit; 🚧 field not in implementation"
        typ endpoint_id FK "local endpoint (implementation field)"
    }

    %% RDMA Core: QP (Queue Pair)
    RaQP {
        typ id PK "QP handle"
        typ qp_handle
        typ ra_dev_id FK "owning RaDevice"
        typ qp_num "QPN(Queue Pair Number)"
        typ type "RC/UC/UD"
        typ state "RESET/INIT/RTR/RTS/SQD/SQE/Error"
        typ peer_qpn "peer QPN"
        typ send_cq_handle FK "send completion queue"
        typ recv_cq_handle FK "receive completion queue"
        typ srq_handle FK "shared receive queue(optional); 🚧 field not in implementation"
        typ taJettyId
        typ mode "jetty mode 0: URMA, 2: CCU, 3: Normal"
        typ peer_qp_id
        typ perr_lid
        typ pid
    }
    RaDevice ||--o{ RaQP : "owns QP"
    RaQP ||--|| RaCQ : "send_cq"
    RaQP ||--|| RaCQ : "recv_cq"
    RaQP |o--o| RaQP : "logical link"
    RaQP ||--o| RaSRQ : "uses shared RQ"

    %% Completion Queue CQ
    RaCQ {
        typ id PK "CQ handle"
        typ cq_handle
        typ ra_dev_id FK "owning RaDevice"
        typ cqn "CQN"
        typ size "queue depth"
        typ policy "CQ completion policy; 🚧 field not in implementation"
    }
    RaCQE {
        typ id PK "CQE ID"
        typ cqe_id "🚧 field not in implementation"
        typ cq_handle FK "owning CQ"
        typ wr_id "Work Request ID"
        typ status "SUCCESS/FLUSH_ERR/..."
        typ opcode "SEND/RECV/READ/WRITE; 🚧 field not in implementation"
        typ byte_len "transfer byte count; 🚧 field not in implementation"
    }
    RaDevice ||--o{ RaCQ : "owns CQ"
    RaCQ ||--o{ RaCQE : "contains"

    %% Memory Registration MR
    RaMR {
        typ id PK "MR handle"
        typ mr_handle
        typ ra_dev_handle FK "owning RaDevice; 🚧 field not in implementation"
        typ local_key "Local Key"
        typ remote_key "Remote Key"
        typ addr "start address"
        typ length "memory length(bytes)"
        typ access "access permissions; 🚧 field not in implementation"
        typ vptr_id
    }
    RaDevice ||--o{ RaMR : "registers memory"
    RaMR ||--|{ VirtualMemBlock : "maps to virtual memory"

    %% Shared Receive Queue SRQ
    RaSRQ {
        typ srq_handle PK "SRQ handle"
        typ ra_dev_handle FK "owning RaDevice"
        typ srq_num "SRQN"
        typ max_wr "maximum WR count"
        typ max_sge "maximum SGE count"
    }
    RaDevice ||--o{ RaSRQ : "owns SRQ"

    %% NDA Direct Access
    RaNdaCQ {
        typ nda_cq_handle PK "NDA CQ handle"
        typ rdma_handle FK "owning RDMA handle"
        typ cqn "CQN"
        typ depth "queue depth"
    }
    RaNdaQP {
        typ nda_qp_handle PK "NDA QP handle"
        typ rdma_handle FK "owning RDMA handle"
        typ qp_num "QPN"
        typ nda_cq_handle FK "associated NDA CQ"
    }
    RaDevice ||--o{ RaNdaCQ : "creates NDA CQ"
    RaDevice ||--o{ RaNdaQP : "creates NDA QP"
    RaNdaQP ||--|| RaNdaCQ : "uses"

    classDef unimplemented fill:#ffe0e0,stroke:#c0392b,stroke-width:2px,stroke-dasharray:5 3
    classDef inproc fill:#fff6d6,stroke:#d4a017,stroke-width:2px
    class RaSRQ,RaNdaCQ,RaNdaQP unimplemented
    class RaDevice,RaQP,RaCQ,RaCQE,RaMR inproc
```

#### 4.1.4 Data Plane: UB Unified Bus

```mermaid
erDiagram
    %% ==========================================
    %% UB Context (RaContext Series Interfaces)
    %% 🧠 Process-private candidates: RaChan / RaLmem / RaRmem / RaTokenId / RaTp (only src/proxy accesses)
    %% RaCr / RaJfc: only cmd dump access, no writer in repo (empty tables)
    %% ==========================================
    Device ||--o{ RaContext : creates
    RaContext {
        typ id PK "UB context handle"
        typ ctx_handle
        typ device_id FK "associated device ID"
        typ mode "mode:RDMA/UB/UB_PLUS"
        typ endpoint_id FK "local endpoint,EID mapping"
        typ max_jetty_num "maximum Jetty count"
        typ max_jfc_num "maximum JFC count"
        typ eidIndex
    }

    EndPointPair {
        typ id PK
        typ local_endpoint_id FK "implementation spelling local_enpoint_id"
        typ remote_endpoint_id FK "implementation spelling remote_enpoint_id"
        typ tp_type "transport type"
    }

    %% UB Core Resources
    RaContext ||--o{ RaJetty : "creates Jetty"
    RaContext ||--o{ RaJfc : "creates JFC"
    RaContext ||--o{ RaLmem : "registers local memory"
    RaContext ||--o{ RaRmem : "imports remote memory"
    RaContext ||--o{ RaTp : "manages transport paths"
    RaContext ||--o{ RaTokenId : "allocates TokenID"
    RaContext ||--o{ RaChan : "creates channel"
    RaContext ||--o{ EndPointPair : "associates EndPointPair"

    %% Jetty (QP equivalent)
    RaJetty {
        typ id PK "Jetty handle"
        typ jetty_handle
        typ ctx_handle FK "owning UB context"
        typ jetty_id "Jetty ID"
        typ mode "URMA_NORMAL/CACHE_LOCK_DWQE/CCU/..."
        typ sqDepth "send queue depth"
        typ rqDepth "receive queue depth"
        typ state "RESET/READY/SUSPENDED/ERROR"
        typ peer_jetty_handle FK "peer Jetty"
        typ peer_endpoint_id FK "peer EndPoint"
        typ send_cq_handle FK "send completion queue"
        typ recv_cq_handle FK "receive completion queue"
        typ sqBuffer "buffer address for WQE dispatch"
        typ sqBufType
        typ type
        typ pid
        typ dieId
    }
    RaJetty ||--o| RaJfc : "send_jfc"
    RaJetty ||--o| RaJfc : "recv_jfc"
    RaJetty |o--o| RaJetty : "logical binding"

    %% JFC (CQ equivalent)
    RaJfc {
        typ id PK "JFC handle"
        typ jfc_handle
        typ ctx_handle FK "owning UB context"
        typ jfc_id "JFC ID"
        typ depth "queue depth"
        typ mode "NORMAL/STARS_POLL/CCU_POLL"
        typ policy "completion policy"
    }
    RaJfc ||--o{ RaCr : "contains"

    %% Completion Request
    RaCr {
        typ id PK "completion request ID"
        typ cr_id
        typ jfc_handle FK "owning JFC"
        typ status "SUCCESS/FLUSH_ERR/..."
        typ opcode "SEND/RECV/READ/WRITE"
        typ byte_len "transfer byte count"
        typ user_ctx "user context"
    }

    %% Local Memory Registration
    RaLmem {
        typ id PK "local memory handle"
        typ lmem_handle
        typ ctx_handle FK "owning UB context"
        typ addr "memory address"
        typ size "memory size(bytes)"
        typ mem_key "memory key"
        typ token_id FK "associated TokenID"
    }
    RaLmem ||--|{ VirtualMemBlock : "maps"

    %% Remote Memory Import
    RaRmem {
        typ id PK "remote memory handle"
        typ rmem_handle
        typ ctx_handle FK "owning UB context"
        typ remote_key "remote memory key"
        typ target_seg_handle FK "target segment handle"
        typ remote_eid "remote EID"
    }

    %% Transport Path
    RaTp {
        typ id PK "transport path handle"
        typ tp_handle
        typ ctx_handle FK "owning UB context"
        typ tp_type "RTP/CTP/UTP"
        typ tpn "transport path number"
        typ speed "link rate"
        typ status "UP/DOWN"
    }
    RaJetty ||--o{ RaTp : "uses"

    %% TokenID
    RaTokenId {
        typ id PK "Token handle"
        typ token_handle
        typ ctx_handle FK "owning UB context"
        typ token_id "Token ID"
        typ ref_count "reference count"
    }

    RaChan {
        typ id PK "channel handle"
        typ chan_handle
        typ ctx_handle FK "owning UB context"
        typ chan_id "channel ID; implementation spelling chann_id"
        typ mode "channel mode"
    }

    classDef inproc fill:#fff6d6,stroke:#d4a017,stroke-width:2px
    class RaJfc,RaCr,RaLmem,RaRmem,RaTp,RaTokenId,RaChan inproc
```

#### 4.1.5 Asynchronous Request Management

```mermaid
erDiagram
    AsyncRequest {
        typ req_handle PK "async request handle"
        typ req_type "CONNECT/LISTEN/CLOSE/QP_CREATE/..."
        typ status "PENDING/COMPLETED/FAILED"
        typ submit_time "submit time"
        typ complete_time "complete time"
    }

    classDef unimplemented fill:#ffe0e0,stroke:#c0392b,stroke-width:2px,stroke-dasharray:5 3
    class AsyncRequest unimplemented
```

> Note: `AsyncRequest` is an internal implementation object and does not need modeling currently (no corresponding struct or database table).

#### 4.1.6 Communicator Architecture Summary

```text
┌─────────────────────────────────────────────────────────────────┐
│                    HCCL Communicator Architecture                │
├─────────────────────────────────────────────────────────────────┤
│  Application Layer                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Communicator                                             │   │
│  │  └── Rank[0..N] (participating nodes, each bound to a Device) │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Control Plane (link building/handshake)                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  RaSocket (Socket Communication)                          │   │
│  │  ├── RaSocketPair (connection pair)                       │   │
│  │  └── RaEpoll (event monitoring)                           │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Data Plane (data transfer)                                      │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │  RDMA (traditional)   │    │  UB (unified bus)    │            │
│  │  ├── RaDevice        │    │  ├── RaContext      │            │
│  │  ├── RaQP (queue pair)│    │  ├── RaJetty (QP)   │            │
│  │  ├── RaCQ (completion)│    │  ├── RaJfc (CQ)     │            │
│  │  ├── RaMR (memory reg)│    │  ├── RaLmem/Rmem    │            │
│  │  └── RaSRQ (shared RQ)│    │  └── RaTp (transport)│            │
│  └─────────────────────┘    └─────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.1.7 Key Entity Comparison Table

| Concept | RDMA Mode | UB Mode | Description |
| --- | --- | --- | --- |
| Context | RaDevice | RaContext | Device/context handle |
| Queue Pair | RaQP | RaJetty | Data transfer channel |
| Completion Queue | RaCQ | RaJfc | Completion notification |
| Completion Element | RaCQE | RaCr | Completion status |
| Local Memory | RaMR | RaLmem | Memory registration |
| Remote Memory | - | RaRmem | Remote memory import |
| Transport Path | - | RaTp | Physical path management |
| Security Token | - | RaTokenId | Access control |

#### 4.1.8 HCCP Interface Classification

> Table "Implementation" column: paths omit the common prefix `src/proxy/level2/`; `only hcomm/` indicates no implementation in this repo (implementation is in `hcomm/src/base_comm/resources/hccp/inc/network/hccp_nda.h`); 🚧 indicates not modeled.

```text
HCCP Network API
├── Control Plane (Socket Communication)
|   ├── Initialization: RaSocketInit/RaSocketDeinit (Socket)
│   ├── Connection Management: RaSocketBatchConnect/Close/Abort
│   ├── Listen Management: RaSocketListenStart/Stop
│   ├── Data Send/Recv: RaSocketSend/Recv
│   ├── Status Query: RaGetSockets
│   └── Event Management: RaEpollCtlAdd/Mod/Del
├── Data Plane - RDMA
|   ├── Initialization: RaRdevInit/RaRdevDeinit (RDMA device)
│   ├── QP Management: RaQpCreate/Destroy/ConnectAsync
│   ├── CQ Management: RaCqCreate/Destroy
│   ├── MR Management: RaMrReg/Dereg
│   ├── Work Requests: RaSendWr/RaRecvWrlist
│   └── Completion Polling: RaPollCq
├── Data Plane - UB
|   ├── Initialization: RaCtxInit/RaCtxDeinit (unified context)
│   ├── Jetty Management: RaCtxQpCreate/Destroy/Import/Bind
│   ├── JFC Management: RaCtxCqCreate/Destroy
│   ├── Memory Management: RaCtxLmemRegister/RmemImport
│   ├── Token Management: RaCtxTokenIdAlloc/Free
│   └── Work Requests: RaBatchSendWr
├── Asynchronous Operations
│   ├── RaSocketBatchConnectAsync
│   ├── RaCtxQpCreateAsync/DestroyAsync
│   └── RaGetAsyncReqResult
├── Network Diagnostics
|   ├── RaPingInit/RaPingDeinit (Ping)
│   ├── RaPingTargetAdd/Del
│   ├── RaPingTaskStart/Stop
│   └── RaPingGetResults
└── TLV Messages
|   ├── RaTlvInit/RaTlvDeinit (TLV)
    └── RaTlvRequest
```

##### Socket Communication Interfaces

| Interface                   | Function           | Key Parameters                                                                 | Implementation |
| ---------------------- | -------------- | ------------------------------------------------------------------------ | -------- |
| `RaSocketInit`         | Socket initialization    | `mode`, `rdevInfo`, `socketHandle`                                      | `hccp_ra_socket_stub.cc:66` |
| `RaSocketDeinit`       | Socket de-initialization | `socketHandle`                                                           | `hccp_ra_socket_stub.cc:110` |
| `RaSocketBatchConnect` | Batch connect       | `SocketConnectInfoT[]`, `num`                                            | `hccp_ra_socket_stub.cc:138` |
| `RaSocketBatchClose`   | Batch close       | `SocketCloseInfoT[]`, `num`                                              | `hccp_ra_socket_stub.cc:254` |
| `RaSocketBatchAbort`   | Batch abort       | `SocketConnectInfoT[]`, `num`                                            | `hccp_ra_socket_stub.cc:279` |
| `RaSocketListenStart`  | Start listening       | `SocketListenInfoT[]`, `num`                                             | `hccp_ra_socket_stub.cc:118` |
| `RaSocketListenStop`   | Stop listening       | `SocketListenInfoT[]`, `num`                                             | `hccp_ra_socket_stub.cc:128` |
| `RaGetSockets`         | Get Socket status | `role`, `SocketInfoT[]`, `num`, `connectedNum`                           | `hccp_ra_socket_stub.cc:188` |
| `RaSocketSend`         | Send data       | `fdHandle`, `data`, `size`, `sentSize`                                   | `hccp_ra_socket_stub.cc:286` |
| `RaSocketRecv`         | Receive data       | `fdHandle`, `data`, `size`, `receivedSize`                               | `hccp_ra_socket_stub.cc:298` |
| `RaEpollCtlAdd`        | Add Epoll event  | `fdHandle`, `event`                                                      | `hccp_ra_socket_stub.cc:313` |
| `RaEpollCtlMod`        | Modify Epoll event  | `fdHandle`, `event`                                                      | `hccp_ra_socket_stub.cc:320` |
| `RaEpollCtlDel`        | Delete Epoll event  | `fdHandle`                                                               | `hccp_ra_socket_stub.cc:327` |
| `RaCreateEventHandle`  | Create event handle   | `eventHandle`                                                            | `hccp_stub.cc:486` |
| `RaWaitEventHandle`    | Wait for event       | `eventHandle`, `SocketEventInfoT[]`, `timeout`, `maxevents`, `eventsNum` | `hccp_stub.cc:498` |
| `RaDestroyEventHandle` | Destroy event handle   | `eventHandle`                                                            | `hccp_stub.cc:505` |
| `RaSocketWhiteListAdd` | Add whitelist     | `socketHandle`, `SocketWlistInfoT[]`, `num`                              | `hccp_stub.cc:1605` |
| `RaSocketWhiteListDel` | Delete whitelist     | `socketHandle`, `SocketWlistInfoT[]`, `num`                              | `hccp_stub.cc:1611` |

###### RDMA Operation Interfaces

| Interface                  | Function           | Key Parameters                                                                | Implementation |
| --------------------- | -------------- | ----------------------------------------------------------------------- | -------- |
| `RaRdevInit`           | RDMA device initialization       | `mode`, `notifyType`, `rdevInfo`, `rdmaHandle`      | `hccp_stub.cc:976` |
| `RaRdevInitV2`         | RDMA device initialization(extended) | `RdevInitInfo`, `rdevInfo`, `rdmaHandle`            | `hccp_stub.cc:948` |
| `RaRdevInitWithBackup` | Initialization with backup       | `initInfo`, `rdevInfo`, `backupRdevInfo`            | `hccp_stub.cc:796` |
| `RaRdevDeinit`         | RDMA device de-initialization     | `rdmaHandle`, `notifyType`                          | `hccp_stub.cc:983` |
| `RaQpCreate`          | Create QP         | `rdevHandle`, `flag`, `qpMode`, `qpHandle`                              | `hccp_stub.cc:1049` |
| `RaQpCreateWithAttrs` | Create QP(with attributes) | `rdevHandle`, `QpExtAttrs`, `qpHandle`                                  | `hccp_stub.cc:1085` |
| `RaAiQpCreate`        | Create AI QP      | `rdevHandle`, `QpExtAttrs`, `AiQpInfo`, `qpHandle`                      | `hccp_stub.cc:1126` |
| `RaLoopbackQpCreate`  | Create loopback QP     | `rdevHandle`, `LoopbackQpPair`, `qpHandle`                              | `hccp_stub.cc:522` |
| `RaTypicalQpCreate`   | Create typical QP     | `rdevHandle`, `flag`, `qpMode`, `TypicalQp`, `qpHandle`                 | `hccp_stub.cc:1132` |
| `RaQpDestroy`         | Destroy QP         | `qpHandle`                                                              | `hccp_stub.cc:1217` |
| `RaQpConnectAsync`    | Connect QP asynchronously     | `qpHandle`, `fdHandle`                                                  | `hccp_stub.cc:587` |
| `RaGetQpStatus`       | Get QP status     | `qpHandle`, `status`                                                    | `hccp_stub.cc:803` |
| `RaTypicalQpModify`   | Modify typical QP     | `qpHandle`, `localQpInfo`, `remoteQpInfo`                               | `hccp_stub.cc:1483` |
| `RaMrReg`             | Register MR         | `qpHandle`, `MrInfoT`                                                   | `hccp_stub.cc:324` |
| `RaMrDereg`           | Deregister MR         | `qpHandle`, `MrInfoT`                                                   | `hccp_stub.cc:353` |
| `RaRegisterMr`        | Register MR independently     | `rdmaHandle`, `MrInfoT`, `mrHandle`                                     | `hccp_stub.cc:376` |
| `RaDeregisterMr`      | Deregister MR independently     | `rdmaHandle`, `mrHandle`                                                | `hccp_stub.cc:407` |
| `RaRemapMr`           | Remap MR       | `rdmaHandle`, `MemRemapInfo[]`, `num`                                   | `hccp_stub.cc:401` |
| `RaGetNotifyMrInfo`   | Get notify MR info | `rdevHandle`, `MrInfoT`                                                 | `hccp_stub.cc:918` |
| `RaSendWr`            | Send work request   | `qpHandle`, `SendWr`, `SendWrRsp`                                       | `hccp_stub.cc:426` |
| `RaSendWrV2`          | Send work request V2 | `qpHandle`, `SendWrV2`, `SendWrRsp`                                     | `hccp_stub.cc:607` |
| `RaSendWrlist`        | Batch send       | `qpHandle`, `SendWrlistData[]`, `SendWrRsp[]`, `sendNum`, `completeNum` | `hccp_stub.cc:822` |
| `RaRecvWrlist`        | Batch receive       | `qpHandle`, `RecvWrlistData`, `recvNum`, `completeNum`                  | `hccp_stub.cc:669` |
| `RaPollCq`            | Poll CQ         | `qpHandle`, `isSendCq`, `numEntries`, `wc`                              | `hccp_stub.cc:630` |
| `RaCqCreate`          | Create CQ         | `rdevHandle`, `CqAttr`                                                  | `hccp_stub.cc:991` |
| `RaCqDestroy`         | Destroy CQ         | `rdevHandle`, `CqAttr`                                                  | `hccp_stub.cc:1025` |
| `RaCreateSrq`         | Create SRQ        | `rdmaHandle`, `SrqAttr`                                                 | `hccp_stub.cc:474` |
| `RaDestroySrq`        | Destroy SRQ        | `rdmaHandle`, `SrqAttr`                                                 | `hccp_stub.cc:480` |
| `RaSetQpAttrQos`      | Set QP QoS     | `qpHandle`, `QosAttr`                                                   | `hccp_stub.cc:450` |
| `RaSetQpAttrTimeout`  | Set QP timeout     | `qpHandle`, `timeout`                                                   | `hccp_stub.cc:456` |
| `RaSetQpAttrRetryCnt` | Set QP retry count | `qpHandle`, `retryCnt`                                                  | `hccp_stub.cc:462` |
| `RaGetQpAttr`         | Get QP attributes     | `qpHandle`, `QpAttr`                                                    | `hccp_stub.cc:1562` |
| `RaGetQpContext`      | Get QP context   | `qpHandle`, `qp`, `sendCq`, `recvCq`                                    | `hccp_stub.cc:696` |

###### UB Unified Bus Interfaces

| Interface                    | Function             | Key Parameters                                                         | Implementation |
| ----------------------- | ---------------- | ---------------------------------------------------------------- | -------- |
| `RaCtxInit`            | Context initialization         | `CtxInitCfg`, `CtxInitAttr`, `ctxHandle`            | `hccp_stub.cc:1138` |
| `RaCtxDeinit`          | Context de-initialization       | `ctxHandle`                                         | `hccp_stub.cc:1175` |
| `RaGetDevEidInfoNum`    | Get EID count      | `RaInfo`, `num`                                                  | `hccp_ccu_stub.cc:575` |
| `RaGetDevEidInfoList`   | Get EID list      | `RaInfo`, `HccpDevEidInfo[]`, `num`                              | `hccp_ccu_stub.cc:586` |
| `RaGetEidByIp`          | Get EID by IP    | `ctxHandle`, `IpInfo[]`, `HccpEid[]`, `num`                      | `hccp_stub.cc:2306` |
| `RaGetDevBaseAttr`      | Get device attributes     | `ctxHandle`, `DevBaseAttr`                                       | `hccp_stub.cc:1182` |
| `RaCtxGetAsyncEvents`   | Get async events     | `ctxHandle`, `AsyncEvent[]`, `num`                               | `hccp_ccu_stub.cc:553` |
| `RaCtxTokenIdAlloc`     | Allocate TokenID      | `ctxHandle`, `HccpTokenId`, `tokenIdHandle`                      | `hccp_stub.cc:1885` |
| `RaCtxTokenIdFree`      | Free TokenID      | `ctxHandle`, `tokenIdHandle`                                     | `hccp_stub.cc:2036` |
| `RaCtxLmemRegister`     | Register local memory     | `ctxHandle`, `MrRegInfoT`, `lmemHandle`                          | `hccp_stub.cc:2047` |
| `RaCtxLmemUnregister`   | Unregister local memory     | `ctxHandle`, `lmemHandle`                                        | `hccp_stub.cc:1906` |
| `RaCtxRmemImport`       | Import remote memory     | `ctxHandle`, `MrImportInfoT`, `rmemHandle`                       | `hccp_stub.cc:1968` |
| `RaCtxRmemUnimport`     | Unimport remote memory | `ctxHandle`, `rmemHandle`                                        | `hccp_stub.cc:1919` |
| `RaCtxChanCreate`       | Create channel         | `ctxHandle`, `ChanInfoT`, `chanHandle`                           | `hccp_stub.cc:1993` |
| `RaCtxChanDestroy`      | Destroy channel         | `ctxHandle`, `chanHandle`                                        | `hccp_stub.cc:2005` |
| `RaCtxCqCreate`         | Create CQ           | `ctxHandle`, `CqInfoT`, `cqHandle`                               | `hccp_stub.cc:2142` |
| `RaCtxCqDestroy`        | Destroy CQ           | `ctxHandle`, `cqHandle`                                          | `hccp_stub.cc:2148` |
| `RaCtxQpCreate`         | Create QP/Jetty     | `ctxHandle`, `QpCreateAttr`, `QpCreateInfo`, `qpHandle`          | `hccp_stub.cc:1254` |
| `RaCtxQpQueryBatch`     | Batch query QP       | `qpHandle[]`, `JettyAttr[]`, `num`                               | `hccp_stub.cc:2106` |
| `RaCtxQpDestroy`        | Destroy QP/Jetty     | `qpHandle`                                                       | `hccp_stub.cc:2012` |
| `RaCtxQpImport`         | Import Jetty        | `ctxHandle`, `QpImportInfoT`, `remQpHandle`                      | `hccp_stub.cc:1321` |
| `RaCtxQpUnimport`       | Unimport Jetty    | `ctxHandle`, `remQpHandle`                                       | `hccp_stub.cc:1425` |
| `RaCtxQpBind`           | Bind Jetty        | `qpHandle`, `remQpHandle`                                        | `hccp_stub.cc:1394` |
| `RaCtxQpUnbind`         | Unbind Jetty        | `qpHandle`                                                       | `hccp_stub.cc:2112` |
| `RaBatchSendWr`         | Batch send         | `qpHandle`, `SendWrData[]`, `SendWrResp[]`, `num`, `completeNum` | `hccp_stub.cc:2135` |
| `RaCtxUpdateCi`         | Update CI           | `qpHandle`, `ci`                                                 | `hccp_stub.cc:2154` |
| `RaCtxGetAuxInfo`       | Get auxiliary info     | `ctxHandle`, `HccpAuxInfoIn`, `HccpAuxInfoOut`                   | `hccp_stub.cc:2270` |
| `RaCtxGetCrErrInfoList` | Get CR errors       | `ctxHandle`, `CrErrInfo[]`, `num`                                | `hccp_stub.cc:2276` |

###### Asynchronous Operation Interfaces

| Interface                        | Function           | Key Parameters                                                             | Implementation |
| --------------------------- | -------------- | -------------------------------------------------------------------- | -------- |
| `RaGetAsyncReqResult`       | Get async result   | `reqHandle`, `reqResult`                                             | `hccp_ra_socket_stub.cc:341` |
| `RaSocketBatchConnectAsync` | Async batch connect   | `SocketConnectInfoT[]`, `num`, `reqHandle`                           | `hccp_ra_socket_stub.cc:348` |
| `RaSocketListenStartAsync`  | Async start listening   | `SocketListenInfoT[]`, `num`, `reqHandle`                            | `hccp_ra_socket_stub.cc:354` |
| `RaSocketListenStopAsync`   | Async stop listening   | `SocketListenInfoT[]`, `num`, `reqHandle`                            | `hccp_ra_socket_stub.cc:362` |
| `RaSocketBatchCloseAsync`   | Async batch close   | `SocketCloseInfoT[]`, `num`, `reqHandle`                             | `hccp_ra_socket_stub.cc:370` |
| `RaSocketSendAsync`         | Async send       | `fdHandle`, `data`, `size`, `sentSize`, `reqHandle`                  | `hccp_ra_socket_stub.cc:377` |
| `RaSocketRecvAsync`         | Async receive       | `fdHandle`, `data`, `size`, `receivedSize`, `reqHandle`              | `hccp_ra_socket_stub.cc:384` |
| `RaCtxLmemRegisterAsync`    | Async register memory   | `ctxHandle`, `MrRegInfoT`, `lmemHandle`, `reqHandle`                 | `hccp_stub.cc:2075` |
| `RaCtxLmemUnregisterAsync`  | Async unregister memory   | `ctxHandle`, `lmemHandle`, `reqHandle`                               | `hccp_stub.cc:1937` |
| `RaCtxQpCreateAsync`        | Async create QP     | `ctxHandle`, `QpCreateAttr`, `QpCreateInfo`, `qpHandle`, `reqHandle` | `hccp_stub.cc:2239` |
| `RaCtxQpDestroyAsync`       | Async destroy QP     | `qpHandle`, `reqHandle`                                              | `hccp_stub.cc:1943` |
| `RaCtxQpDestroyBatchAsync`  | Async batch destroy   | `ctxHandle`, `qpHandle[]`, `num`, `reqHandle`                        | `hccp_stub.cc:1958` |
| `RaCtxQpImportAsync`        | Async import Jetty  | `ctxHandle`, `QpImportInfoT`, `remQpHandle`, `reqHandle`             | `hccp_stub.cc:1378` |
| `RaGetTpInfoListAsync`      | Async get TP info | `ctxHandle`, `GetTpCfg`, `HccpTpInfo[]`, `num`, `reqHandle`          | `hccp_stub.cc:2082` |
| `RaGetEidByIpAsync`         | Async get EID    | `ctxHandle`, `IpInfo[]`, `HccpEid[]`, `num`, `reqHandle`             | `hccp_stub.cc:2091` |
| `RaGetTpAttrAsync`          | Async get TP attributes | `ctxHandle`, `tpHandle`, `attrBitmap`, `TpAttr`, `reqHandle`         | `hccp_stub.cc:2098` |
| `RaSetTpAttrAsync`          | Async set TP attributes | `ctxHandle`, `tpHandle`, `attrBitmap`, `TpAttr`, `reqHandle`         | `hccp_stub.cc:2263` |

###### Network Diagnostics Interfaces

| Interface               | Function         | Key Parameters                                     | Implementation |
| ------------------ | ------------ | -------------------------------------------- | -------- |
| `RaPingInit`       | Ping initialization    | `PingInitAttr`, `PingInitInfo`, `pingHandle`| `hccp_stub.cc:1843` |
| `RaPingDeinit`     | Ping de-initialization  | `pingHandle`                                | `hccp_stub.cc:1849` |
| `RaPingTargetAdd`  | Add Ping target | `pingHandle`, `PingTargetInfo[]`, `num`      | `hccp_stub.cc:1855` |
| `RaPingTargetDel`  | Delete Ping target | `pingHandle`, `PingTargetCommInfo[]`, `num`  | `hccp_stub.cc:1873` |
| `RaPingTaskStart`  | Start Ping task | `pingHandle`, `PingTaskAttr`                 | `hccp_stub.cc:1861` |
| `RaPingTaskStop`   | Stop Ping task | `pingHandle`                                 | `hccp_stub.cc:1879` |
| `RaPingGetResults` | Get Ping results | `pingHandle`, `PingTargetResult[]`, `num`    | `hccp_stub.cc:1867` |

###### TLV Message Interfaces

| Interface           | Function        | Key Parameters                                      | Implementation |
| -------------- | ----------- | --------------------------------------------- | -------- |
| `RaTlvInit`    | TLV initialization   | `TlvInitInfo`, `bufferSize`, `tlvHandle`      | `hccp_stub.cc:1670` |
| `RaTlvDeinit`  | TLV de-initialization | `tlvHandle`                                   | `hccp_stub.cc:1680` |
| `RaTlvRequest` | TLV request processing | `tlvHandle`, `moduleType`, `TlvMsg`, `TlvMsg` | `hccp_stub.cc:1821` |

###### NDA (Network Direct Access) Interfaces

| Interface                 | Function             | Key Parameters                                               | Implementation |
| -------------------- | ---------------- | ------------------------------------------------------ | -------- |
| `RaNdaGetDirectFlag` | Get direct access flag | `rdmaHandle`, `directFlag`                             | `only hcomm/` |
| `RaNdaCqCreate`      | Create NDA CQ       | `rdmaHandle`, `NdaCqInitAttr`, `NdaCqInfo`, `cqHandle` | `only hcomm/` |
| `RaNdaCqDestroy`     | Destroy NDA CQ       | `rdmaHandle`, `cqHandle`                               | `only hcomm/` |
| `RaNdaQpCreate`      | Create NDA QP       | `rdmaHandle`, `NdaQpInitAttr`, `NdaQpInfo`, `qpHandle` | `only hcomm/` |

###### General Query Interfaces

| Interface                     | Function           | Key Parameters                                       | Implementation |
| ------------------------ | -------------- | ---------------------------------------------- | -------- |
| `RaGetIfnum`             | Get interface count   | `RaGetIfattr`, `num`                           | `hccp_stub.cc:1622` |
| `RaGetIfaddrs`           | Get interface addresses   | `RaGetIfattr`, `InterfaceInfo[]`, `num`        | `hccp_stub.cc:1640` |
| `RaSocketGetVnicIpInfos` | Get virtual NIC IP | `phyId`, `IdType`, `ids[]`, `num`, `IpInfo[]`  | `hccp_stub.cc:99` |
| `RaGetTlsEnable`         | Get TLS status    | `RaInfo`, `tlsEnable`                          | `hccp_stub.cc:936` |
| `RaGetHccnCfg`           | Get HCCN config   | `RaInfo`, `HccnCfgKey`, `value`, `valueLen`    | `hccp_stub.cc:942` |
| `RaGetInterfaceVersion`  | Get interface version   | `phyId`, `interfaceOpcode`, `interfaceVersion` | `hccp_stub.cc:130` |
| `RaRdevGetHandle`        | Get Rdev handle   | `phyId`, `rdmaHandle`                          | `hccp_stub.cc:759` |
| `RaRdevGetSupportLite`   | Get Lite support   | `rdmaHandle`, `supportLite`                    | `hccp_stub.cc:147` |
| `RaSaveSnapshot`         | Save snapshot       | `RaInfo`, `SaveSnapshotAction`                 | `hccp_stub.cc:785` |
| `RaRestoreSnapshot`      | Restore snapshot       | `RaInfo`                                       | `hccp_stub.cc:791` |
| `RaGetSecRandom`         | Get secure random | `RaInfo`, `value`                              | `hccp_stub.cc:1900` |

##### Key Relationship Description (HCCP Interface Classification)

**Communicator Hierarchy Structure**:

- `Communicator` is the core abstraction of HCCL collective communication, defining a group of participating Ranks.
- `Rank` is a participating node in a communicator; each Rank is bound to a specific Device.
- Parent communicators derive child communicators (for example, `color` attribute implements grouping)

**Control Plane and Data Plane Separation**:

- **Control Plane (Socket)**: Used for link building, QP information exchange, and control signaling, based on TCP protocol.
- **Data Plane (RDMA/UB)**: Used for high-performance data transfer, based on RDMA Verbs or UB protocol.

**RDMA Resource Hierarchy**:

- `RaDevice` is the virtual NIC abstraction; one Device can create multiple RaDevices.
- `RaQP` is a queue pair containing a Send Queue and a Receive Queue.
- `RaCQ` is a completion queue for polling WR completion status.
- `RaMR` is memory registration, mapping virtual memory to RDMA-accessible physical memory.
- `RaSRQ` is a shared receive queue; multiple QPs can share the same SRQ to improve resource utilization.

**UB Resource Hierarchy**:

- `RaContext` is the UB unified context, replacing the RaDevice device abstraction.
- `RaJetty` is the QP equivalent, supporting multiple modes (URMA_NORMAL/CCU, and so on)
- `RaJfc` is the CQ equivalent for completion request management.
- `RaLmem/RaRmem` are local/remote memory management, replacing RaMR.
- `RaTp` is transport path management, supporting three types: RTP/CTP/UTP.
- `RaTokenId` is a secure communication token for cross-process memory access control.

**Entity Association Key Points**:

1. `RaSocketPair` requires two `RaSocket` instances (client and server) to establish a connection.
2. `RaQP.state` must undergo RESET→INIT→RTR→RTS state transitions for normal communication.
3. `RaMR.lkey` is for local access; `RaMR.rkey` is for remote RDMA access.
4. `RaJetty` establishes logical connections with peer Jettys through `Bind` operations.
5. `RaLmem` must be registered before `RaRmem` can import and access from the peer.

**NDA (Network Direct Access) Mechanism**:

- `RaNdaQP` and `RaNdaCQ` are QP/CQ variants for network direct access.
- NDA mode allows bypassing parts of the protocol stack to reduce latency.
- `RaNdaGetDirectFlag` checks whether a device supports NDA mode.

**Asynchronous Request Management**:

- `AsyncRequest` uniformly manages request handles for all asynchronous operations.
- Asynchronous operations include: connect, listen, QP create/destroy, memory registration, and so on.
- `RaGetAsyncReqResult` polls asynchronous operation results.

**Socket Event Mechanism**:

- `RaSocketEvent` is the handle for the event waiting mechanism.
- `RaEpoll` implements a Linux Epoll-like event monitoring mechanism.
- Supports adding, modifying, and deleting monitored Socket events.

**Network Interfaces and Configuration**:

- `InterfaceInfo` describes network interface attributes such as IP/MAC/MTU.
- `HccnConfig` stores HCCN network configuration key-value pairs.
- `Snapshot` supports saving and restoring device state.

##### Control Plane vs Data Plane

| Dimension         | Control Plane                                                      | Data Plane                                   |
| ------------ | ------------------------------------------------------------- | ------------------------------------------ |
| **Core Function** | Link building, QP info exchange, control signaling                                    | Data transfer, RDMA operations                         |
| **Key Entities** | RaSocket 🧊, SocketConnection, RaEpoll                           | RaQP, RaCQ, RaMR, RaJetty                  |
| **Key Interfaces** | `RaSocketBatchConnect`, `RaSocketListenStart`, `RaGetSockets` | `RaSendWr`, `RaPollCq`, `RaQpConnectAsync` |
| **Communication** | TCP Socket                                                    | RDMA Verbs / UB                            |

##### RDMA Mode vs UB Mode

| Comparison       | RDMA Mode         | UB Mode                 |
| ------------ | ---------------- | ---------------------- |
| **Device Abstraction** | RaDevice         | RaContext              |
| **Queue Pair**   | RaQP (QP)        | RaJetty (Jetty)        |
| **Completion Queue** | RaCQ (CQ)        | RaJfc (JFC)            |
| **Memory Registration** | RaMR (lkey/rkey) | RaLmem/RaRmem (MemKey) |
| **Address ID** | IP + GID         | EID (Endpoint ID)      |
| **Transport Path** | QPN + GID        | RaTp (TPN)             |

##### Key Points Description

1. **RaContext/RaDevice**: Unified context entity supporting both RDMA and UB modes.
2. **RaJetty/RaJfc/RaQP/RaCQ**: QP/CQ equivalents in UB mode.
3. **RaLmem/RaRmem/RaMR**: Local/remote memory management in UB mode.
4. **RaTp**: UB transport path management.
5. **RaTokenId**: Secure communication token.

##### Key Attribute Supplement

- **QP Mode**: `NOR`(normal), `GDR_TMPL`(template), `OP`(operation), `GDR_ASYN`(async GDR)
- **Transport Mode**: `RC`(reliable connection), `RM`(reliable message, UB only)
- **Jetty Mode**: `URMA_NORMAL`, `CACHE_LOCK_DWQE`, `CCU`, `USER_CTL_NORMAL`
- **JFC Mode**: `NORMAL`, `STARS_POLL`, `CCU_POLL`

##### Corresponding Interface Mapping Table (HCCP Interface Classification)

| Entity             | Init Interfaces                   | Create Interfaces                                                 | Operation Interfaces                                   | Destroy/Cleanup Interfaces                               |
| --------------- | ---------------------------- | -------------------------------------------------------- | ------------------------------------------ | ------------------------------------------ |
| **Communicator** | -                            | `HcclCommInitRankInfo`, `HcclCommInitClusterInfo`        | `HcclGetRankId`, `HcclGetRankSize`         | `HcclCommDestroy`                          |
| **Rank**         | -                            | -                                          | -                   | -                        |
| **RaSocket**     | `RaSocketInit`               | `RaSocketBatchConnect`                                   | `RaSocketSend`, `RaSocketRecv`, `RaGetSockets` | `RaSocketDeinit`, `RaSocketBatchClose`, `RaSocketBatchAbort` |
| **RaSocketPair** | -                            | `RaSocketBatchConnect`                                   | `RaGetSockets`                             | `RaSocketBatchClose`                       |
| **RaSocketEvent** | `RaCreateEventHandle`       | -                                                        | `RaWaitEventHandle`                        | `RaDestroyEventHandle`                     |
| **RaEpoll**      | -                            | `RaEpollCtlAdd`                                          | `RaEpollCtlMod`                            | `RaEpollCtlDel`                            |
| **RaDevice**     | `RaRdevInit`, `RaRdevInitV2`, `RaRdevInitWithBackup` | -                                                        | `RaRdevGetHandle`, `RaRdevGetSupportLite`  | `RaRdevDeinit`                             |
| **RaQP**         | -                            | `RaQpCreate`, `RaQpCreateWithAttrs`, `RaTypicalQpCreate`, `RaAiQpCreate`, `RaLoopbackQpCreate` | `RaQpConnectAsync`, `RaSendWr`, `RaSendWrV2`, `RaSendWrlist`, `RaRecvWrlist`, `RaPollCq`, `RaGetQpStatus`, `RaTypicalQpModify` | `RaQpDestroy`                              |
| **RaCQ**         | -                            | `RaCqCreate`                                             | `RaPollCq`                                 | `RaCqDestroy`                              |
| **RaSRQ**        | -                            | `RaCreateSrq`                                            | `RaModifySrq`                              | `RaDestroySrq`                             |
| **RaMR**         | -                            | `RaMrReg`, `RaRegisterMr`                                | `RaRemapMr`, `RaGetNotifyMrInfo`           | `RaMrDereg`, `RaDeregisterMr`              |
| **RaCQE**        | -                            | -                                                        | `RaPollCq`                                 | -                                          |
| **RaQPAttr**     | -                            | -                                                        | `RaGetQpAttr`, `RaSetQpAttrQos`, `RaSetQpAttrTimeout`, `RaSetQpAttrRetryCnt`, `RaGetQpContext` | -                                          |
| **RaNdaQP**      | -                            | `RaNdaQpCreate`                                          | -                                          | -                                          |
| **RaNdaCQ**      | -                            | `RaNdaCqCreate`                                          | -                                          | `RaNdaCqDestroy`                           |
| **RaContext**    | `RaCtxInit`                  | -                                                        | `RaGetDevBaseAttr`, `RaGetDevEidInfoList`, `RaGetDevEidInfoNum`, `RaGetEidByIp`, `RaCtxGetAsyncEvents` | `RaCtxDeinit`                              |
| **RaJetty**      | -                            | `RaCtxQpCreate`                                          |  `RaBatchSendWr`, `RaCtxUpdateCi`, `RaCtxQpQueryBatch` | `RaCtxQpDestroy`        |
| **EndPointPair**      | -                            | -                                         | `RaCtxQpBind`, `RaCtxQpUnbind`, `RaCtxQpImport`, `RaCtxQpUnimport` | - |
| **RaJfc**        | -                            | `RaCtxCqCreate`                                          | `RaCtxGetAuxInfo`, `RaCtxGetCrErrInfoList` | `RaCtxCqDestroy`                           |
| **RaCr**         | -                            | -                                                        | `RaCtxGetAuxInfo`                          | -                                          |
| **RaLmem**       | -                            | `RaCtxLmemRegister`                                      | -                                          | `RaCtxLmemUnregister`                      |
| **RaRmem**       | -                            | `RaCtxRmemImport`                                        | -                                          | `RaCtxRmemUnimport`                        |
| **RaTp**         | -                            | `RaGetTpInfoListAsync`                                   | `RaGetTpAttrAsync`, `RaSetTpAttrAsync`     | -                                          |
| **RaTokenId**    | -                            | `RaCtxTokenIdAlloc`                                      | -                                          | `RaCtxTokenIdFree`                         |
| **RaChan**       | -                            | `RaCtxChanCreate`                                        | -                                          | `RaCtxChanDestroy`                         |
| **RaPing**       | `RaPingInit`                 | `RaPingTargetAdd`                                        | `RaPingTaskStart`, `RaPingGetResults`      | `RaPingDeinit`, `RaPingTargetDel`          |
| **RaTlv**        | `RaTlvInit`                  | -                                                        | `RaTlvRequest`                             | `RaTlvDeinit`                              |
| **AsyncRequest** | -                            | `RaSocketBatchConnectAsync`, `RaCtxQpCreateAsync`, `RaCtxQpDestroyAsync`, `RaCtxQpDestroyBatchAsync`, `RaCtxQpImportAsync`, `RaCtxLmemRegisterAsync`, `RaSocketListenStartAsync`, `RaSocketListenStopAsync`, `RaSocketBatchCloseAsync`, `RaSocketSendAsync`, `RaSocketRecvAsync`, `RaGetTpInfoListAsync`, `RaGetEidByIpAsync`, `RaGetTpAttrAsync`, `RaSetTpAttrAsync` | `RaGetAsyncReqResult` | - |
| **InterfaceInfo** | -                           | -                                                        | `RaGetIfnum`, `RaGetIfaddrs`, `RaSocketGetVnicIpInfos` | -                                          |
| **HccnConfig**   | -                            | -                                                        | `RaGetHccnCfg`, `RaGetTlsEnable`, `RaGetInterfaceVersion`, `RaGetSecRandom` | -                                          |
| **Snapshot**     | -                            | -                                                        | `RaSaveSnapshot`, `RaRestoreSnapshot`      | -                                          |

##### Entity Property and Interface Mapping Supplement Table

| Entity Property                    | Corresponding Interface                                                                 |
| -------------------------- | ------------------------------------------------------------------------ |
| RaSocket.state             | `RaGetSockets` returns state                                                  |
| RaSocket.list 🚧        | `RaSocketWhiteListAdd`, `RaSocketWhiteListDel`                           |
| RaQP.qp_num/peer_qpn/perr_lid     | `RaQpConnectAsync` exchanges peer info                                          |
| RaQP.mode/type/state                  | `RaGetQpAttr`, `RaSetQpAttrQos`, `RaSetQpAttrTimeout`, `RaSetQpAttrRetryCnt` |
| RaQP.send_cq_handle/recv_cq_handle               | `RaGetQpContext` returns QP send_cq and recv_cq                                |
| RaCQE.wr_id                | `RaSendWr`, `RaSendWrV2`, `RaSendWrlist`, `RaRecvWrlist` sets             |
| RaCQE.status               | `RaPollCq` returns                                                          |
| RaJetty.state              | `RaCtxQpQueryBatch` returns                                                 |
| RaJetty.jetty_id/peer_jetty_handle  | `RaCtxQpBind`, `RaCtxQpImport` sets                                      |
| RaMR.local_key/remote_key                | `RaMrReg` parameter setting                                                       |
| RaLmem.id            | `RaCtxTokenIdAlloc` pre-allocation                                               |
| RaTp.tp_type               | `RaGetTpInfoListAsync` returns                                              |
| RaContext.endpoint_id        | `RaGetDevEidInfoList`, `RaGetEidByIp` gets                               |
| RaContext.mode             | `RaGetDevBaseAttr` returns                                                  |
| AsyncRequest.status 🚧        | `RaGetAsyncReqResult` returns                                               |
| InterfaceInfo.* 🚧            | `RaGetIfnum`, `RaGetIfaddrs`, `RaSocketGetVnicIpInfos` gets              |
| HccnConfig.value 🚧           | `RaGetHccnCfg` gets                                                      |
| RaNdaQP.flag 🚧    | `RaNdaGetDirectFlag` checks NDA support                                         |
| Snapshot.data 🚧              | `RaSaveSnapshot` saves, `RaRestoreSnapshot` restores                          |
| RaSocketEvent.events 🚧       | `RaEpollCtlAdd`, `RaEpollCtlMod`, `RaEpollCtlDel` manages                   |
| RaDevice.flag 🚧       | `RaNdaGetDirectFlag` checks NDA support                                         |

> Note: Field names in the table follow `include/runnerdb/sim_models.h`; 🚧 indicates the field is not in the implementation or the entity is not modeled; 🧠 indicates only this process accesses it (candidate for process-private memory); 🧊 indicates a database table is registered but has zero database calls in the repo (zombie table). Classification basis is described in the "Database Persistence Necessity Classification" subsection.
>
> **ER Diagram Coloring** (scanned by `struct`/`class`/`enum` definitions across the repo + verified by real database call directory ownership):
> - 🟥 `unimplemented`: No entity with the same name in the repo; pure design or conceptual entities (30, see 31 locations in diagrams).
> - 🟨 `inproc`: Only this process/dump accesses; database tables are accessed only by `src/proxy` stubs or `cmd` dumps at runtime, with no cross-process/cross-component consumers (33).
> - No color: `src/device_arm`/`src/plugin`/`src/store`/`src/topo` and other external components have real database calls (15).

##### Database Persistence Coverage Comparison (`include/runnerdb/db_sim_sqlite_db.h`)

The database registration contains 64 tables total. Comparing with entities in this document reveals three categories of differences:

- **Table name ≠ struct name**: `VirMem`(VirtualMemBlock), `PhMem`(PhyMemBlock), `IpMemWhiteList`(IpcMemWhiteList).
- **In database but not modeled in this document (16 tables)**: `HcclBuffer`, `HcclChannel`, `HcclEngineCtx` 🧊, `HcclMem`, `HcclThread`, `HcommEndpoint`, `HcommMemReg`, `DpuDeviceInfo`, `DpuPendingNotify`, `CommunicatorDestroySync`, `SimModelData`, `MemoryLayout`, `RunModeConfig`, `TopoMetaConfig`, `Plugin`, `RaTlv`.
- **In this document but not registered in database (30)**: `AsyncRequest`, `Rank`, `ReportChannel`, `RaEpoll`, `RaSRQ`, `RaNdaCQ`, `RaNdaQP`, `RaSocketEvent`, `CommMemSlot` (store layer mmap communication pool slot, not a database table), `LinkProtocolMapping` (implemented as `Link.protocols[8]` array field), `KernelBinary`/`KernelBinaryHandle`/`KernelFuncHandle`/`KernelFuncArgsHandle`/`KernelFuncArgsParamHandle`/`KernelLaunchCfg` (ACL-side configuration types), `CPU`/`Scalar`/`CcuInternal`/`Cube`/`Vector`/`HybridComputeDie` (conceptual entities), `MemcpyTask`/`CallbackTask`/`EventTask`/`EventRICaptureTask`/`EventRecordTask`/`EventWaitTask`/`EventTimeTask`/`EventTraceTask` (carried by `Task`/`EventSyncTask` in implementation).

##### Database Persistence Necessity Classification (Process-Private Candidates)

The determination criterion is **whether another process/component makes real database API calls at runtime** (`Add`/`Get*`/`Update*`/`Delete*` `<sim::X>`). Occurrences of an entity name in other directories do not count — the original criterion incorrectly counted `cmd` `PrintTable` dumps and local variables of same-named structs as runtime usage. This section corrects that.

- **C1 Cross-Process Shared**: `src/device_arm` (independent aarch64 `device` executable, separately links `db_sim_sqlite_db.cc`), `src/plugin/runner`, `src/plugin/checker`, `src/store` (shm channel), `src/plugin/hccl_plugin_manager.cc` have real database calls ⇒ Must persist.
- **C2 Topology/Hardware Facts**: Written by `src/topo/topo_ascend_cluster_parser.cc`, shared across processes ⇒ Must persist.
- **C3 This Process Only**: Only `src/proxy` has real database calls; `cmd` only does `PrintTable` dumps ⇒ Can be process-private.
- **C4 No Runtime Access**: Only `cmd` calls (dumps + `DeleteAll` resets) or zero calls across repo ⇒ Should not persist.

| Category | Table Count | Tables |
| --- | --- | --- |
| A Must Persist (C1/C2) | 19 | `Ccu`, `Communicator`, `Device`, `DpuDeviceInfo`, `EndPoint`, `EndPointPair`, `EndPointPortMapping`, `HcclChannel`, `HcclThread`, `Host`, `Link`, `Notify`, `PhyMemBlock`, `Plugin`, `Port`, `RaContext`, `RaJetty`, `Server`, `VirtualMemBlock` |
| B 🧠 Can Be Process-Private (C3) | 29 | `Context`, `DeviceConnection`, `DeviceStatus`, `DpuPendingNotify`, `Event`, `FdMemWhiteList`, `HcclBuffer`, `HcclMem`, `HcommEndpoint`, `HcommMemReg`, `IpcMemRecord`, `IpcMemWhiteList`, `IpcNotify`, `IpcNotifyVistorList`, `RaCQ`, `RaCQE`, `RaChan`, `RaDevice`, `RaLmem`, `RaMR`, `RaQP`, `RaRmem`, `RaSocketPair`, `RaTlv`, `RaTokenId`, `RaTp`, `SimModelData`, `Stream`, `Task` |
| C Cmd Access Only (C4) | 9 | `CcuChannel`, `EventSyncTask`, `FdMemRecord`, `MemoryLayout`, `NotifyRecordTask`, `NotifyWaitTask`, `RaCr`, `RaJfc`, `RunModeConfig` |
| D Ops Layer Only (caller TBD) | 5 | `CommunicatorDestroySync`, `ComputeDie`, `Runner`, `TaskSchedulerDevice`, `TopoMetaConfig` |
| E 🧊 Zero DB Calls (Zombie Registrations) | 2 | `HcclEngineCtx`: data lives only in `src/proxy/level1/level1_proxy_common.h` `unordered_map<uint64_t, sim::HcclEngineCtx>`, zero DB layer calls; `cmd` only `dlsym`s its `HcclEngineCtxResetAll`. `RaSocket`: only has struct definition and registration; actually in use is `RaSocketPair`/`RaSocketEvent`. |

> 23 tables "persisted only because dump makes them visible": `Context`, `Stream`, `Task`, `Event`, `Runner`, `Plugin`, `SimModelData`, `DeviceStatus`, `DeviceConnection`, `HcclBuffer`, `HcclMem`, `HcommEndpoint`, `HcommMemReg` — 13 tables only have `src/proxy` writes and `cmd` reads, where `cmd` only does `PrintTable` (`src/cmd/cmd_table_utils.cc`); additionally `CcuChannel`, `ComputeDie`, `CommunicatorDestroySync`, `EventSyncTask`, `MemoryLayout`, `NotifyRecordTask`, `NotifyWaitTask`, `RunModeConfig`, `TaskSchedulerDevice`, `TopoMetaConfig` — 10 tables only referenced by `cmd` (writers on runner side, to be verified). If later changed to "export on process exit", these 23 tables can be converted to process-private entirely.

## 5. Callback and Report Relationship Modeling

Callbacks (`CallbackTask`) and report channels (`ReportChannel`) are consumed by Host Runner threads. The Task family definition is described in [§3.1 Data / Task Flow Modeling](#31-data--task-flow-modeling).

```mermaid
erDiagram
    Runner ||--o{ Context: "creates"
    Runner {
        typ id PK
        typ run_id
    }

    Context {
        typ id PK
        typ ctx_id
        typ run_id FK
    }

    Stream {
        int stream_id PK
        int ctx_id FK
        int activated
        string state "Running/Idle; 🚧 field not in implementation"
    }
    %% Implementation has no state field; run state is expressed by activated/capture_status/task_complete_status

    %% Stream contains an ordered task list
    Context ||--o{ Stream : "manages/submits"
    Stream ||--o{ Task : "queues [1..*]"

    Task {
        int task_id PK
        int stream_id FK
        typ seq_number "auto-increment within stream"
        string type "Kernel/Memcpy/Callback"
    }

    %% Various specific Task types (logical inheritance)
    CallbackTask {
        typ report_id FK
        typ callback_fn
        typ user_data
    }

    %% Inheritance logical expression (Task has multiple types)
    %% 🧠 Process-private candidates: Context / Stream / Task (only src/proxy accesses)
    %% CallbackTask is discriminated by Task.type (implementation only creates Task table)
    Task ||--|{ CallbackTask : "is a"

    ReportChannel {
        typ report_id
        typ stream_id
        typ run_id
    }

    CallbackTask }o..|| ReportChannel : "push"
    ReportChannel ||--o{ Runner : "trigger and called by"

    classDef unimplemented fill:#ffe0e0,stroke:#c0392b,stroke-width:2px,stroke-dasharray:5 3
    classDef inproc fill:#fff6d6,stroke:#d4a017,stroke-width:2px
    class CallbackTask,ReportChannel unimplemented
    class Runner,Context,Stream,Task inproc

```

### 5.1 Key Relationship Description (Callback and Report Relationship Modeling)

When the **device** reaches a CallbackTask, it triggers the Host Runner thread to execute the callback.

#### 5.1.1 Corresponding Interface Mapping Table (Callback and Report Relationship Modeling)

| Entity          | Key Management Interfaces                                                          |
| ------------- | --------------------------------------------------------------------- |
| CallbackTask  | `rtSetExceptionInfoCallback`, `rtLaunchCallback`,`rtSynchronizeEvent` |
| ReportChannel | `rtSubscribeReport`, `rtUnSubscribeReport`                            |
| Runner        | `rtProcessReport`                                                     |

## 6. Basic `Device` Model Fine-Grained Low-Level Extension

Fine-grained low-level scheduling entities that refine the Device hierarchy from [§2.1.1 Hierarchy Structure Overview](#211-hierarchy-structure-overview) (`Die` / `SuperPod` / `Plane` and other conceptual entities have no corresponding structs in this repo implementation).

```mermaid
erDiagram

    %% 🧠 Process-private candidates: DeviceStatus (only src/proxy accesses)
    Device ||--|| DeviceStatus : "has a"
    Device ||--|{ TaskSchedulerDevice : "has "
    Device {
        typ id PK
        typ device_id
    }
    DeviceStatus {
        typ id PK
        typ device_id FK
        typ overflow_status
        typ synchronize_strategy
        typ synchronize_timeout
        typ capability_mask
        typ run_by_host
        typ ts_core
        typ online_status
    }

    %% Scalar/CcuInternal/CPU are discriminated by TaskSchedulerDevice.type (conceptual entities, not separate tables)
    TaskSchedulerDevice ||--|| Scalar :"is a"
    TaskSchedulerDevice ||--|| CcuInternal :"is a"
    TaskSchedulerDevice ||--|| CPU :"is a"
    TaskSchedulerDevice {
        typ id PK
        typ ts_id
        typ device_id FK
        typ type "Scalar"
    }

   CPU {
       %% Conceptual entity: no independent fields, discriminated by ComputeDie.type
   }

   Scalar ||--|| ComputeDie :"schedule"
   Scalar {
       %% Conceptual entity: no independent fields, discriminated by ComputeDie.type
   }

   CcuInternal {
        typ id PK
        typ ccu_id
        typ ts_id FK
        typ version "v1/v2"
        typ xn_num
        typ cke_num
        typ ms_num
        typ channel_num
    }

    %% Cube/Vector/HybridComputeDie are discriminated by ComputeDie.type (implementation only creates ComputeDie table)
    ComputeDie ||--|| Cube :"is a"
    ComputeDie ||--|| Vector :"is a"
    ComputeDie ||--|| HybridComputeDie :"is a (vector+cube)"
    ComputeDie {
        typ id PK
        typ compute_id
        typ ts_id FK
        typ type
    }
    Cube {
        %% Conceptual entity: no independent fields, discriminated by ComputeDie.type
    }
    Vector {
        %% Conceptual entity: no independent fields, discriminated by ComputeDie.type
    }
    HybridComputeDie {
        %% Conceptual entity: no independent fields, discriminated by ComputeDie.type
    }

    classDef unimplemented fill:#ffe0e0,stroke:#c0392b,stroke-width:2px,stroke-dasharray:5 3
    classDef inproc fill:#fff6d6,stroke:#d4a017,stroke-width:2px
    class CPU,Scalar,CcuInternal,Cube,Vector,HybridComputeDie unimplemented
    class DeviceStatus,TaskSchedulerDevice,ComputeDie inproc
```

### 6.1 Key Relationship Description (Basic `Device` Model Fine-Grained Low-Level Extension)

**Device Scheduler Hierarchy**:

- `TaskSchedulerDevice` is the device scheduler abstraction; one Device can contain multiple schedulers.
- Scheduler types include: `Scalar` (scalar processor), `CCU` (collective communication unit), `CPU` (AI CPU)
- The CCU entity block is named `CcuInternal` to distinguish it from the `Ccu` (CCU resource allocation table) modeled in §2.1.1; the `type` field takes values `Scalar`/`CCU`/`CPU`.
- Different scheduler types handle different computation task types.

**ComputeDie Compute Units**:

- `ComputeDie` is the compute unit abstraction; inheritance relationships represent compute unit types.
- `Vector`: Vector compute unit, handles vector operations.
- `Cube`: Cube compute unit, handles matrix operations.
- `HybridComputeDie`: Hybrid compute unit, supports both Vector and Cube. 🚧 (Implementation does not have this type; discriminated by `ComputeDie.type`; zero hits in this repo and `hcomm/`)

**CCU Internal Structure**:

- `xn_num`: XN node count (cross-node communication)
- `cke_num`: CKE engine count (Checksum engine)
- `ms_num`: MS module count (Memory Scheduler)
- `channel_num`: Communication channel count.
- `version`: CCU version (v1/v2, determines functional differences)

**DeviceStatus State Management**:

- `overflow_status`: Overflow state.
- `synchronize_strategy`: Synchronization strategy configuration.
- `synchronize_timeout`: Synchronization timeout setting.
- `capability_mask`: Device capability mask.
- `run_by_host`: Whether running in Host mode.
- `ts_core`: Scheduler core count.
- `online_status`: Device online state.

**Scalar Scheduling Relationship**:

- `Scalar` scheduler manages `ComputeDie` execution.
- Different ComputeDie types correspond to different computation workloads.

#### 6.1.1 Corresponding Interface Mapping Table (Basic Device Model Fine-Grained Low-Level Extension)

| Entity                | Key Management Interfaces                       |
| ------------------- | ---------------------------------- |
| TaskSchedulerDevice | `rtGetDeviceInfo`, `rtSetTsDevice` |
| DeviceStatus        | `rtGetRunMode`                     |

## 7. Kernel Runtime Relationship Modeling

Kernel-side entities are submitted through ACL interfaces and carried by Task / EventSyncTask (see [§3.1 Data / Task Flow Modeling](#31-data--task-flow-modeling)). Corresponding interfaces for each entity are described in [§7.1.1 Corresponding Interface Mapping Table](#711-corresponding-interface-mapping-table-kernel-runtime-relationship-modeling).

```mermaid
erDiagram
    KernelBinary {
        typ id PK
        typ file
        typ create_pid
    }

    KernelBinary  ||--|| KernelBinaryHandle :"loaded"
    KernelBinaryHandle  ||--o{ KernelFuncHandle :"contains"
    KernelBinaryHandle {
        typ handle_id PK
        typ kernel_id FK
    }

    KernelFuncHandle  ||--o{ KernelFuncArgsHandle :"has a"
    KernelFuncHandle  }o--|| Task :"Called "
    KernelFuncHandle  }o--|| KernelLaunchCfg :"launch config"
    %% KernelLaunchCfg = ACL-side aclrtLaunchKernelCfg (numAttrs/attrs[]), not a sim model, not persisted
    KernelFuncHandle {
        typ handle_id PK
        typ binary_id FK
        typ func_name
        typ kernel_name
        typ aic_addr
        typ aiv_addr
    }

    KernelFuncArgsHandle  ||--o{ KernelFuncArgsParamHandle :"append"
    KernelFuncArgsHandle {
        typ args_handle_id PK
        typ func_id FK
        typ args_size
        typ type "device/host"
    }

    KernelFuncArgsParamHandle {
        typ args_param_id PK
        typ args_id FK
        typ param_size
        typ is_place_holder
    }

    KernelLaunchCfg {
        typ numAttrs
        typ attrs "aclrtLaunchKernelAttr[]: id + value"
    }

    classDef unimplemented fill:#ffe0e0,stroke:#c0392b,stroke-width:2px,stroke-dasharray:5 3
    class KernelBinary,KernelBinaryHandle,KernelFuncHandle,KernelFuncArgsHandle,KernelFuncArgsParamHandle,KernelLaunchCfg unimplemented
```

### 7.1 Key Relationship Description (Kernel Runtime Relationship Modeling)

**KernelBinary Loading Flow**:

1. `KernelBinary` stores .o/.so binary file paths and the creating process PID.
2. `rtBinaryLoadFromFile` or `rtBinaryLoadFromData` loads the binary into device memory.
3. After loading, a `KernelBinaryHandle` is generated containing handle_id and kernel_id.
4. `KernelFuncHandle` represents specific functions in the binary, containing func_name, kernel_name, and address information.

**Kernel Function Call Relationships**:

- `aic_addr` is the AI Core function address.
- `aiv_addr` is the AI Vector function address.
- `KernelFuncArgsHandle` stores function parameter information.
- `KernelFuncArgsParamHandle` records parameter size and whether it is a placeholder.
- `KernelLaunchCfg` configures Kernel launch parameters (block/grid, and so on) 🚧 (not a `sim_models.h` entity, not persisted; on the implementation side it is the ACL type `aclrtLaunchKernelCfg`: `numAttrs` + `attrs[]`, with 9 attribute IDs — `SCHEM_MODE`(1)/`DYN_UBUF_SIZE`(2)/`ENGINE_TYPE`(3)/`BLOCKDIM_OFFSET`(4)/`BLOCK_TASK_PREFETCH`(5)/`DATA_DUMP`(6)/`TIMEOUT`(7)/`TIMEOUT_US`(8)/`ENABLE_PROFILING`(9); authoritative definition at `runtime/include/external/acl/acl_rt.h:527-541` (`aclrtLaunchKernelAttrId`); `LOCAL_MEMORY_SIZE` is a deprecated alias of `DYN_UBUF_SIZE` (same value 2, `acl_rt.h:529-532`); `TIMEOUT` and `TIMEOUT_US` cannot be carried simultaneously (`acl_rt.h:538`); this repo stub recognizes only the first 8, the 9th is not implemented, see `src/proxy/level2/aclrt_kernel_stub.cc:564,1101`)

**Kernel Lifecycle**:

```text
rtCreateBinary -> rtBinaryLoad -> rtBinaryGetFunction
                -> rtLaunchKernel(funcHandle, argsHandle, cfg)
                -> rtBinaryUnLoad -> rtDestroyBinary
```

**Parameter Management Mechanism**:

- args_handle_id supports both device and host types.
- is_place_holder identifies whether a parameter is a placeholder (deferred binding)
- param_size records individual parameter size.

#### 7.1.1 Corresponding Interface Mapping Table (Kernel Runtime Relationship Modeling)

| Entity               | Key Management Interfaces                                                                                                    |
| ------------------ | --------------------------------------------------------------------------------------------------------------- |
| KernelBinary       | `rtCreateBinary`, `rtDestroyBinary`                                                                             |
| KernelBinaryHandle | `rtBinaryLoad`, `rtBinaryUnLoad`,`rtBinaryLoadFromFile`,`rtBinaryLoadFromData`                                  |
| KernelFuncHandle   | `rtBinaryGetFunction`, `rtBinaryGetFunctionByEntry`,`rtGetFunctionAddr`,`rtGetFunctionName`,`rtRegisterCpuFunc` |
| KernelLaunchCfg 🚧 | `rtLaunchKernelWithConfig` (ACL-side type, not a sim model)                                                       |

## 8. Model Loading Relationship Modeling
