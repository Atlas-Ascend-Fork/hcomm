# Analysis of Mapping Conflict Between rankId and Device Physical ID in Ranktable

## Problem Overview

### Symptom

In a cluster topology configuration with multiple servers where some servers contain multiple devices, the hcomm service initialization phase reports a `CheckRankGraphAddrs` failure:

```
[CommunicatorImpl][CheckRankGraphAddrs]the ip address
  IpAddress[...c002:0001...] of ranktable in rank 1 is error!
[CommunicatorImpl][CheckRankGraphAddrs]the ip address
  IpAddress[...c002:000b...] of ranktable in rank 0 is error!
```

The debug log reveals the key clue:

```
ZHF-DEBUG: peer deviceId is 0, devPhyId is 1   ← rank 1: ranktable says device 0, but the process actually runs on device 1
ZHF-DEBUG: peer deviceId is 1, devPhyId is 0   ← rank 2: ranktable says device 1, but the process actually runs on device 0
```

### Trigger Conditions

The problem is triggered only when the following conditions are met:

```
At least one server satisfies:
  1. The number of devices D_s on that server is greater than 1
  2. Other servers before this server have already consumed some rankIds
     (that is, the rankIds on this server do not start from 0 in a contiguous sequence)
```

This is equivalent to: **The YAML topology configuration contains a server with more than 1 device, and that server is not the only server that owns ranks.**

| Configuration Type | Example | Triggered | Reason |
|---------|------|---------|------|
| All servers have 1 device each | (1,1), (1,1,1,1) | Not triggered | `rank % 1 = 0` always holds; sequential allocation is equivalent to modulo operation |
| Single server with multiple devices | (2), (4), (8) | Not triggered | Only one server exists; `rank % D = rank`; sequential allocation is equivalent to modulo operation |
| Multiple servers, some with multiple devices | (1,2), (2,4) | **Triggered** | On a multi-device server, `rank % D_s != sequential value` |
| Multiple servers, all with multiple devices (uniform) | (2,2), (2,2,2,2), (8x8) | **Triggered** | Same as above |

---

## Key Concept Definitions

### Two Types of rankId

This document involves two different rankId concepts, which must be strictly distinguished:

| Concept | Code Name | Definition | Producer | Allocation Method |
|------|------|------|--------|---------|
| ranktable rankId | **hcclRankid** | The value of the `rank_id` field in ranktable.json | This tool `CreateRankTableFile` | Incremented from 0 in the traversal order of superPod -> server -> device |
| MPI rankId | **mpiRankid** | The number assigned to each process by MPI | MPI runtime (`MPI_Comm_rank`) | Determined by the MPI scheduling strategy; typically 0, 1, 2, and so on |

> **Implicit convention**: hcclRankid and mpiRankid are numerically equal (both increment as 0, 1, 2, and so on). This is the prerequisite for the entire system to function correctly.

### Device Identification System

| Concept | Definition | Relationship in This System |
|------|------|-----------------|
| `physical_id` (phyDevId) | The physical identifier of a device; fixed and unchanging | Defined by the cluster topology file |
| `logic_id` (logicDevId) | The logical identifier of a device; mappable | In this simulator, `logic_id = physical_id` (numerically equal) |
| `device_id` (DB key) | The primary key of the Device table | Auto-incremented by the internal database |
| `local_id` | The logical device identifier recorded in ranktable.json | Server form: `srcDevPhyId`; POD form: `boardIdx * devNumPerBoard + srcDevPhyId` |
| `serverKey` | The unique identifier of a server | `GetServerKeyById(superPodId, serverId, boardId)` |

> **POD form (4-layer topology: board → pod → supernode → cluster) conventions**:
> The term "server" in the table above refers to the **leaf node (host node)**. In the server form, the leaf node is the server itself;
> in the POD form, the leaf node is the **board** (each board occupies one `Server` row in the DB, distinguished by `board_id`). Therefore:
> - `physical_id` and `device_id` in the ranktable are **physical IDs within a board** (for example, `[0,7]`,
>   rather than `[0,63]` within the POD before the refactoring), which is consistent with real hardware;
>   `local_id` is the **POD-global ID** (`= board_id × number of cards per board + physical ID within the board`,
>   consistent with the local_id value range in the POD-level topo.json);
> - **D (`serverMeta.size()`) in all formulas in this document = the number of cards assigned to the communicator on that board** (for example, 8),
>   so the application side calls `aclrtSetDevice(rank % 8)` and `aclrtGetDeviceCount()` returns 8;
> - `serverKey` must be located by the triple `(superPodId, serverId, boardId)`.

### Involved Software Layers

```
┌─────────────────────────────────────────────────────────┐
│  Application layer (hccl_test / sample programs)         │
│  MPI_Comm_rank -> mpiRankid                              │
│  aclrtSetDevice(mpiRankid % deviceCount)                 │
│  HcclCommInitClusterInfoConfig(ranktable, mpiRankid,...) │
└──────────────────────┬──────────────────────────────────┘
                       │ Call
┌──────────────────────▼──────────────────────────────────┐
│  hcomm service layer (libhccl_v2.so)                     │
│  CommunicatorImpl::CheckRankGraphAddrs()                │
│  RankGraphBuilder::BuildFromRankTable()                  │
└──────────────────────┬──────────────────────────────────┘
                       │ Call
┌──────────────────────▼──────────────────────────────────┐
│  This project (hccl-vm simulator)                        │
│  AscendClusterTopoParser::CreateRankTableFile()           │
│  aclrtSetDevice (stub) / RaGetDevEidInfoList (stub)      │
└─────────────────────────────────────────────────────────┘
```

---

## Problem Reproduction Environment

### Topology Configuration

The topology used for testing: 1 superPod, 2 servers (rack0 has 1 card, rack1 has 2 cards), 3 ranks in total.

```
superPod0
├── rack0 (server0): [phyDev0]              ← 1 device
└── rack1 (server1): [phyDev0, phyDev1]     ← 2 devices
```

### Process and Device Allocation (Measured from Logs)

| PID | mpiRankid | `aclrtSetDevice(rankId % D)` | logicDevId | phyDevId | Server |
|-----|-----------|------------------------------|------------|----------|-------------|
| 13311 | 0 | `0 % 1 = 0` | 0 | 0 | rack0 (1 card) |
| 13312 | 1 | `1 % 2 = 1` | 1 | 1 | rack1 (2 cards) |
| 13313 | 2 | `2 % 2 = 0` | 0 | 0 | rack1 (2 cards) |

> Log source: `log.txt:79,83,85`
> ```
> [aclrtSetDevice] Init Rank: rankId=1, deviceKey=10, logicDevId=1, phyDevId=1 totalRanks=3
> [aclrtSetDevice] Init Rank: rankId=2, deviceKey=9, logicDevId=0, phyDevId=0 totalRanks=3
> [aclrtSetDevice] Init Rank: rankId=0, deviceKey=1, logicDevId=0, phyDevId=0 totalRanks=3
> ```

Key point: rack1 has 2 devices. `mpiRankid % 2` yields 1 for rank 1 and 0 for rank 2, **which is exactly the reverse of the sequential traversal order [0, 1]**.

### Comparison of ranktable.json Before and After the Modification

| rank_id | Old code `rankId % D` | Old code device_id | New code `deviceIdx` | New code device_id | Actual phyDevId in application |
|---------|---------------------|-----------------|-------------------|-----------------|------------------|
| 0 | `0 % 1 = 0` | 0 | 0 | 0 | 0 |
| 1 | `1 % 2 = 1` | **1** | 1 | **0** | **1** |
| 2 | `2 % 2 = 0` | **0** | 2 | **1** | **0** |

- **Old code** (`rankId % D`): The device_id in the ranktable **matches** the actual phyDevId in the application -> verification passes
- **New code** (`deviceIdx`): The device_id values for rank 1 and rank 2 are swapped -> **conflict**

> The ranktable generated by the old code is in the archive file `hccl_vm_install/archive/20260815_094326/data/ranktable.json`
> The ranktable generated by the new code is in `hccl_vm_install/data/ranktable.json`

---

## CheckRankGraphAddrs Verification Logic Details

### Function Location and Signature

```
File: hcomm/src/legacy/ascend950/framework/communicator/communicator_impl.cc
Function: void CommunicatorImpl::CheckRankGraphAddrs() const
Lines: 1291-1338
```

### Verification Flow

This function obtains the EID through two separate paths and then compares the two results for consistency:

```
                      CheckRankGraphAddrs
                      /                  \
                Path A                   Path B
           (Actual hardware EID)       (EID claimed by ranktable)
                |                            |
    Find in ranktable where         rankGraph->GetPeer(myRank)
    rank.deviceId==devPhyId           (look up peers_ table by rank_id)
                |                            |
    RaGetDevEidInfoList                peer->GetIfaces()
    (phyId=devPhyId)                  Get the EID address of the interface
                |                            |
    localEidSet (EID set)          interface->GetAddr().GetEid()
                \                            /
                 \                          /
                  → localEidSet.count(interface EID) == 0 ?
                       Not in set → THROW exception
```

### Path A: Actual Hardware EID

```cpp
// communicator_impl.cc:1306-1316
for (auto& rank : ranktableInfo->ranks) {
    if (rank.deviceId == devPhyId) {                    // Select entry: deviceId == physical device of this process
        HRaInfo info(HrtNetworkMode::HDC, rank.deviceId);
        std::vector<HrtDevEidInfo> localEidInfos = HrtRaGetDevEidInfoList(info);  // Query hardware
        for (auto& eidInfo : localEidInfos) {
            localEidSet.insert(eidInfo.ipAddress.GetEid());  // Store only EID/IP values
        }
        break;  // Call once, then exit
    }
}
```

**Key characteristics**:
- The `RaGetDevEidInfoList` interface only recognizes `phyId`; it does not recognize rankId
- The `devPhyId` passed in is the physical device on which the process actually runs
- In this path, ranktable.json is only used to "filter entries where `deviceId == devPhyId`"; the EID values come from the hardware query
- Although the `chipId` field is populated (`hccp_ccu_stub.cc:573`), `CheckRankGraphAddrs` **does not read chipId**; EID comparison matches only by the original IP value

### Path B: EID Claimed by Ranktable

```cpp
// communicator_impl.cc:1322-1328
const auto& peer = rankGraph->GetPeer(myRank);   // Look up peers_ by rank_id == myRank
const auto& interfaces = peer->GetIfaces();
for (auto& interface : interfaces) {
    // Check whether the EID of the interface is in localEidSet
    if (interface->GetPos() == AddrPosition::DEVICE
        && protocols.count(LinkProtocol::PCIE) == 0
        && protocols.count(LinkProtocol::UBOE) == 0
        && localEidSet.count(interface->GetAddr().GetEid()) == 0) {
        THROW<InvalidParamsException>(...);  // EID mismatch; report error
    }
}
```

**Key characteristics**:
- `GetPeer(myRank)` directly uses `myRank` as the key to look up the `peers_` table
- The interface EID of the peer originates from the `level_list` of the entry where `rank_id == myRank` in ranktable.json
- **This path is entirely determined by the content of ranktable.json**, that is, it is affected by lines 450/451

### Acquisition of devPhyId and myRank

```cpp
// communicator_impl.cc:1256-1269 (InitCommonData)
myRank = commParams.myRank;                              // ← From rank parameter = mpiRankid
devLogicId = HrtGetDevice();                              // ← Logical device already set by the current process
devPhyId = HrtGetDevicePhyIdByIndex(devLogicId);         // ← Physical device corresponding to this logical device
HCCL_INFO("ZHF-DEBUG: devLogicId is %u, devPhyId is %u", devLogicId, devPhyId);
```

---

## Source and Ownership of myRank

### Complete Propagation Chain

```
┌─ Application layer ────────────────────────────────────────────────────┐
│                                                                        │
│  MPI_Comm_rank(MPI_COMM_WORLD, &procRank)    // procRank=mpiRankid     │
│  uint32_t devId = (uint32_t)procRank;                                   │
│                                                                        │
│  // devId is used for two purposes simultaneously:                      │
│  aclrtSetDevice(devId);                      // Select device           │
│  HcclCommInitClusterInfoConfig(rtk, devId, &config, &comm);  // Pass rank │
│                           ↑                                            │
│                      rank parameter = mpiRankid                         │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
┌─ hcomm entry point (op_base_v2.cc) ───────────────────────────────────▼┐
│                                                                        │
│  HcclCommInitClusterInfoConfigV2(clusterInfo, uint32_t rank, ...)      │
│  // op_base_v2.cc:455                                                  │
│  → ParseJsonAndCreateComm(data, rank, ...)                              │
│  // op_base_v2.cc:501                                                  │
│  → CreateCommConfig(rank, config, comm, ranktableM)                    │
│  // op_base_v2.cc:318                                                  │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
┌─ CommParams construction (op_base_v2.cc:146-152) ────────────────────▼┐
│                                                                        │
│  Hccl::CommParams commParams{                                          │
│      commId,                                                           │
│      static_cast<Hccl::RankId>(rank),   // ← myRank = mpiRankid        │
│      0,                                                                │
│      static_cast<Hccl::RankId>(rank),   // rankInParentComm also = rank │
│      ...                                                               │
│  };                                                                    │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
┌─ CommunicatorImpl (communicator_impl.cc:1261) ───────────────────────▼┐
│                                                                        │
│  void CommunicatorImpl::InitCommonData(const CommParams& params) {     │
│      myRank = commParams.myRank;    // ← Final destination: mpiRankid  │
│      devLogicId = HrtGetDevice();                                      │
│      devPhyId = HrtGetDevicePhyIdByIndex(devLogicId);                  │
│  }                                                                     │
└────────────────────────────────────────────────────────────────────────┘
```

### Conclusion

**`myRank` is mpiRankid.** It is passed in through the `rank` parameter of `HcclCommInitClusterInfoConfig`, and its origin is the application's `MPI_Comm_rank`.

### Construction and Query of the peers_ Table

```cpp
// rank_graph_builder.cc:320-328 (BuildFromRankTable)
for (const auto& rankInfo : rankTable_->ranks) {
    RankId rankId = rankInfo.rankId;           // ← hcclRankid (rank_id from ranktable.json)
    auto peer = make_shared<NetInstance::Peer>(
        rankId, rankInfo.localId, rankInfo.replacedLocalId,
        rankInfo.deviceId, ...);               // ← The deviceId of Peer comes from ranktable.json
    peers_.emplace(rankId, peer);              // ← peers_ uses hcclRankid as the key
}
```

```cpp
// communicator_impl.cc:1322 (CheckRankGraphAddrs)
const auto& peer = rankGraph->GetPeer(myRank);  // GetPeer(peers_.at(myRank))
//        ↑ Uses mpiRankid to look up the hcclRankid table
```

**The lookup succeeds** purely because hcclRankid == mpiRankid (both increment numerically as 0, 1, 2, and so on).

### Source of the Peer deviceId

```
Entry in ranktable.json where rank_id==myRank
    ↓ (Read by BuildFromRankTable)
rankInfo.deviceId
    ↓ (Passed to the Peer constructor)
peer->deviceId_ = rankInfo.deviceId
    ↓ (Returned by GetDeviceId)
peer->GetDeviceId() == device_id of the rank entry in the ranktable
```

This is the source of `peer deviceId` in the logs. It equals the `device_id` field of the entry where `rank_id == myRank` in ranktable.json.

---

## How the Two EID Paths Relate to rankId

### Overview Diagram

```
                          mpiRankid (= myRank)
                         /                   \
                        /                     \
               ┌─ Path A ─┐              ┌─ Path B ──────────┐
               │          │              │                    │
    App calls  │ aclrtSetDevice           │ GetPeer(myRank)   │
    (app       │   (mpiRankid % D)        │   = peers_[myRank] │
    layer)     │          │              │   = entry in       │
               │          │              │     ranktable where │
               │          │              │     rank_id==myRank │
               │          │              │          │          │
    Query      │ (serverKey,logic_id)    │     interface EID   │
    Device     │   → device              │     (addr in        │
    table to   │   → phyDevId           │      level_list)    │
    get        │          │              │          │          │
    phyDevId   │          │              │          │          │
               │          │              │          │          │
    Query      │ RaGetDevEidInfoList      │          │          │
    hardware   │   (phyDevId)            │          │          │
    EID        │          │              │          │          │
               │          │              │          │          │
               ▼          ▼              ▼          ▼          │
          localEidSet (EID set)    interface EID (single EID)  │
               \          /              \          /          │
                \        /                \        /           │
                 → Compare: localEidSet.count(interface EID) ←─┘
```

### rankId -> Device Mapping Formulas for the Two Paths

| Path | rankId -> Device Mapping Formula | Formula Location | Affected by Lines 450/451 |
|------|--------------------------|-------------|---------------------|
| Path A (hardware) | `server.devices[mpiRankid % D]` | Hardcoded in application layer (`aclrtSetDevice`) | **No**; the application is fixed and unchanging |
| Path B (ranktable) | `server.devices[logicDevId]`, where logicDevId is determined by lines 450/451 | `CreateRankTableFile` | **Yes** |

### Formula Consistency Analysis

#### Old code (line 450): `logicDevId = rankId % serverMeta.size()`

```cpp
uint32_t logicDevId = rankId % serverMeta.size();  // Consistent with the application's rank % D formula
```

| rankId | Path A: `rankId % D` -> device | Path B (line 450): `rankId % D` -> device | Consistent |
|--------|-------------------------------|-------------------------------------|---------|
| 0 | `0 % 1 = 0` -> phyDev0 | `0 % 1 = 0` -> phyDev0 | Consistent |
| 1 | `1 % 2 = 1` -> phyDev1 | `1 % 2 = 1` -> phyDev1 | Consistent |
| 2 | `2 % 2 = 0` -> phyDev0 | `2 % 2 = 0` -> phyDev0 | Consistent |

-> Both sides use the same mapping -> EIDs match -> verification passes

#### New code (line 451): `logicDevId = deviceIdx`

```cpp
uint32_t logicDevId = deviceIdx;  // Sequential allocation 0, 1, 2, and so on; may differ from the application's % D formula
```

| rankId | Path A: `rankId % D` -> device | Path B (line 451): `deviceIdx` -> device | Consistent |
|--------|-------------------------------|-------------------------------------|---------|
| 0 | `0 % 1 = 0` -> phyDev0 | `0` -> phyDev0 | Consistent |
| 1 | `1 % 2 = 1` -> phyDev1 | `0` -> phyDev0 | **Inconsistent** |
| 2 | `2 % 2 = 0` -> phyDev0 | `1` -> phyDev1 | **Inconsistent** |

-> The devices for rank 1 and rank 2 are swapped -> EIDs do not match -> verification fails

### Detailed EID Comparison Failure Process (Using rank 1 as an Example)

```
rank 1 (PID 13312):
  ┌─ Path A (actual hardware EID) ─────────────────────────────────┐
  │ app: aclrtSetDevice(1 % 2 = 1) -> logicDevId=1                │
  │ Device table: logic_id=1 -> phyDevId=1                         │
  │ Entry in ranktable where device_id==1 -> RaGetDevEidInfoList(1) │
  │ localEidSet = {192.2.0.11, 192.2.0.18}  (real IPs of device 1) │
  └────────────────────────────────────────────────────────────────┘

  ┌─ Path B (EID claimed by ranktable) ───────────────────────────┐
  │ GetPeer(myRank=1) -> peers_[1]                                │
  │   = entry in ranktable where rank_id=1                         │
  │ New code: device_id=0 for this entry (should be 1)            │
  │ peer->GetDeviceId() = 0  (ranktable says device 0)            │
  │ interface EID = {192.2.0.1, 192.2.0.8} (IPs of device 0)      │
  └────────────────────────────────────────────────────────────────┘

  Compare: 192.2.0.1 in {192.2.0.11, 192.2.0.18} ?
           -> Not in set -> THROW "ip address ... is error!"
```

Log corroboration:
- `log.txt:1062`: `peer deviceId is 0, devPhyId is 1` (peer goes through Path B and gets deviceId=0, but the process actually runs on devPhyId=1)
- `log.txt:1267`: `the ip address ...c002:0001... of ranktable in rank 1 is error!`

---

## Root Cause

### One-Sentence Summary

**Both EID acquisition paths use rankId as the anchor point, but they map to devices through different formulas.** Lines 450/451 only changed the formula on Path B (the ranktable side); the formula on Path A (the application side) remains fixed and unchanging. When the number of devices on a server is greater than 1 and the server is not the first server, the two formulas produce different results, causing the same rankId to point to different devices on the two sides, which leads to EID comparison failure.

### rankId Association on Path A (Indirect)

The `RaGetDevEidInfoList(devPhyId)` interface itself does not recognize rankId; it only recognizes phyDevId. However, `devPhyId` is selected by the application using `aclrtSetDevice(mpiRankid % D)`. This step binds mpiRankid into the process:

```
mpiRankid → aclrtSetDevice(mpiRankid % D) → Device table → devPhyId → RaGetDevEidInfoList(devPhyId)
```

The ranktable.json **does not participate** in device selection on this path (it is only used to filter entries where `deviceId == devPhyId` to determine the phyId parameter).

### rankId Association on Path B (Direct)

The interface EID of the peer is obtained by looking up the ranktable using `GetPeer(myRank)`, where `myRank` is mpiRankid:

```
mpiRankid(=myRank) → GetPeer(myRank) → peers_[myRank] → entry in ranktable where rank_id==myRank → interface EID
```

The ranktable.json **entirely determines** which device EID corresponds to this rankId on this path.

### Consistency Requirement

Since both paths start from mpiRankid, the rank -> device formulas on both sides must be consistent:

```
Path A formula:  device = server.devices[mpiRankid % D]     (hardcoded in application; cannot be changed)
Path B formula:  device = server.devices[logicDevId]         (determined by lines 450/451)
```

- **Line 450** `logicDevId = rankId % serverMeta.size()` -> Consistent with Path A -> Passes
- **Line 451** `logicDevId = deviceIdx` -> Deviates from Path A -> Fails

### Complete Mapping Chain Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Consistency Requirement                           │
│                                                                         │
│  The rank → device mapping must be consistent across all levels:        │
│                                                                         │
│  1. Application aclrtSetDevice:    logicDevId = mpiRankid % D_s         │
│  2. ranktable.json device_id: Must equal the phyDevId actually          │
│     selected by the application                                        │
│  3. ranktable.json local_id:  Must equal the logicDevId actually        │
│     selected by the application                                        │
│  4. EIDs in ranktable.json: Must equal the real EIDs corresponding     │
│     to that phyDevId                                                    │
│                                                                         │
│  Any inconsistency at any level causes CheckRankGraphAddrs to report    │
│  an error                                                               │
└─────────────────────────────────────────────────────────────────────────┘

Mapping chain:

  mpiRankid (assigned by MPI)
      │
      ├──→ [Application layer] aclrtSetDevice(mpiRankid % D)
      │         │
      │         ▼
      │    Device table (logic_id) → physical_id → devPhyId
      │         │                              │
      │         │              ┌───────────────┘
      │         │              │
      │         │              ▼
      │         │    RaGetDevEidInfoList(devPhyId)
      │         │              │
      │         │              ▼
      │         │    localEidSet (actual hardware EID)
      │         │
      │    (Simultaneously, mpiRankid is passed as the rank parameter
      │     into hcomm)
      │         │
      │         ▼
      │    myRank = mpiRankid
      │         │
      │         ▼
      │    rankGraph->GetPeer(myRank)
      │         │
      │         ▼
      │    peers_[myRank] ← Entry in ranktable.json where rank_id==myRank
      │         │
      │         ▼
      │    Peer interface EID (EID claimed by ranktable)
      │         │
      └─────────┘
                │
                ▼
         localEidSet.count(interface EID) == 0 ?
              │
         Mismatch → Report error
```

---

## Modification Analysis

### Modified Code

```
File: src/topo/topo_ascend_cluster_parser.cc
Function: AscendClusterTopoParser::CreateRankTableFile
Lines: 449-456

Before modification (line 450):
    for (uint32_t deviceIdx = 0; deviceIdx < serverMeta.size(); ++deviceIdx) {
        uint32_t logicDevId = rankId % serverMeta.size();   // ← Old code
        PhyDeviceId phyDevId = serverMeta[logicDevId];
        ...
        rankId++;
    }

After modification (line 451):
    for (uint32_t deviceIdx = 0; deviceIdx < serverMeta.size(); ++deviceIdx) {
        uint32_t logicDevId = deviceIdx;                    // ← New code
        PhyDeviceId phyDevId = serverMeta[logicDevId];
        ...
        rankId++;
    }
```

### Impact of the Modification

This modification changes the `device_id` and `local_id` corresponding to each `rank_id` in `ranktable.json`:

| rank_id | Old: device_idx from `rankId % D` | New: `deviceIdx` | Same? |
|---------|-----------------------------------|-----------------|-------|
| 0 (rack0) | 0 | 0 | Same |
| 1 (rack1) | 1 | 0 | **Different** |
| 2 (rack1) | 0 | 1 | **Different** |

When all servers have exactly 1 device or when there is only a single server, the two formulas are equivalent (`rankId % D` within the first server is the same as `deviceIdx`), and the modification does not cause problems. However, when there are multiple servers and a non-first server has multiple devices, the two formulas differ.

### Setting of device_id and local_id in BuildRankEntry

```cpp
// topo_ascend_cluster_parser.cc:413-416
json AscendClusterTopoParser::BuildRankEntry(const Server &server, int srcDevPhyId, uint32_t boardIdx, ...)
{
    json rankEntry;
    const int devNumPerBoard = server.GetDeviceCount();
    rankEntry["device_id"] = srcDevPhyId;                                          // ← Board-local physical ID
    rankEntry["local_id"] = static_cast<int>(boardIdx) * devNumPerBoard + srcDevPhyId;  // ← POD-global ID
    rankEntry["rank_id"] = rankId;
    ...
}
```

In this simulator, `logic_id = physical_id`, so `device_id` and `logic_id` have the same value. In server form, `local_id = device_id` (both are the physical ID within the server). In POD form, `local_id = board_id × cards per board + physical ID within the board` (POD-global ID). `BuildRankEntry` itself has no problem. The real issue is that the `phyDevId` passed into `BuildRankEntry` is taken from `serverMeta[logicDevId]`, and the calculation method for `logicDevId` (lines 450/451) determines which physical device is assigned to that rank.

---

## Fix Solution

### Core Fix

**Restore the formula on line 450** so that the rank -> device mapping in the ranktable is consistent with the application's `aclrtSetDevice(rank % D)`:

```cpp
uint32_t logicDevId = rankId % serverMeta.size();
```

### Data Flow Verification After the Fix

Verify using the test topology (rack0=1 card, rack1=2 cards):

```
Step 1: Application selects devices (aclrtSetDevice)
  rank 0: 0 % 1 = 0 → rack0 logicDevId=0 → phyDevId=0  (IP=192.1.0.8)
  rank 1: 1 % 2 = 1 → rack1 logicDevId=1 → phyDevId=1  (IP=192.2.0.18)
  rank 2: 2 % 2 = 0 → rack1 logicDevId=0 → phyDevId=0  (IP=192.2.0.1)

Step 2: ranktable.json generation (rankId % D)
  rank 0: device_id=0, local_id=0  (phyDev0)
  rank 1: device_id=1, local_id=1  (phyDev1)
  rank 2: device_id=0, local_id=0  (phyDev0)

Step 3: CheckRankGraphAddrs verification
  rank 1: devPhyId=1 → localEidSet={192.2.0.11, 192.2.0.18}
          peer(rank_id=1) → device_id=1 → interface EID={192.2.0.18}
          192.2.0.18 in localEidSet → Match ✓
  rank 2: devPhyId=0 → localEidSet={192.2.0.1, 192.2.0.8}
          peer(rank_id=2) → device_id=0 → interface EID={192.2.0.1}
          192.2.0.1 in localEidSet → Match ✓
```

All mapping chains are consistent, and verification passes.

---

## Key Code Location Index

### This Project (CheckerL2_8620) Code

| File | Lines | Function/Description |
|------|------|---------|
| `src/topo/topo_ascend_cluster_parser.cc` | 424-477 | `CreateRankTableFile` - Generate ranktable.json |
| `src/topo/topo_ascend_cluster_parser.cc` | 449-456 | **Modification point** - logicDevId calculation method (line 450 old / line 451 new) |
| `src/topo/topo_ascend_cluster_parser.cc` | 409-422 | `BuildRankEntry` - Set device_id/local_id/rank_id |
| `src/topo/topo_ascend_cluster_parser.cc` | 200-272 | `InitDynamicModelData` - Initialize Device/Rank tables |
| `src/topo/topo_ascend_cluster_parser.cc` | 202, 234 | Comments explaining that the application uses `aclrtSetDevice(rank % deviceCount)` |
| `src/proxy/level2/aclrt_device_stub.cc` | 69-128 | `aclrtSetDevice` - Stub implementation for application device selection |
| `src/proxy/level2/aclrt_device_stub.cc` | 87-88 | Query the Device table by `(serverKey, logic_id)` |
| `src/proxy/level2/aclrt_device_stub.cc` | 110-122 | Create Rank table record; print logicDevId/phyDevId |
| `src/proxy/level2/hccl_comm_stub.cc` | 83-111 | `SimGetDeviceComm` - Call `aclrtSetDevice` |
| `src/proxy/level2/hccl_comm_stub.cc` | 130-131 | Call `SimGetDeviceComm` in `HcclCommInitAll` |
| `src/proxy/level2/hccp_ccu_stub.cc` | 528-541 | `GetAllUsedEndPoint` - Query EndPoint by phyDevId |
| `src/proxy/level2/hccp_ccu_stub.cc` | 554-587 | `RaGetDevEidInfoList` - Return EID list by phyId |
| `src/proxy/level2/hccp_ccu_stub.cc` | 573 | `chipId = info.phyId` (has a TODO comment, but unrelated to this problem) |

### hcomm Service Code

| File | Lines | Function/Description |
|------|------|---------|
| `communicator_impl.cc` | 1256-1269 | `InitCommonData` - Set myRank, devLogicId, devPhyId |
| `communicator_impl.cc` | 1291-1338 | `CheckRankGraphAddrs` - EID verification logic |
| `communicator_impl.cc` | 1306-1316 | Path A: Filter by `deviceId==devPhyId` and call `RaGetDevEidInfoList` |
| `communicator_impl.cc` | 1322-1328 | Path B: `GetPeer(myRank)` to get interface EID and compare |
| `communicator_impl.cc` | 1323 | ZHF-DEBUG log: `peer deviceId` vs `devPhyId` |
| `communicator_impl.cc` | 1353-1371 | `InitRankGraph` - Call bridge to build rankGraph |
| `communicator_impl.cc` | 1364 | `bridge->buildFromString(ranktableM, topoPath, myRank, ...)` |
| `op_base_v2.cc` | 455-514 | `HcclCommInitClusterInfoConfigV2` - Entry function |
| `op_base_v2.cc` | 131-152 | `CreateCommConfig` - Construct CommParams (myRank=rank) |
| `op_base_v2.cc` | 311-322 | `ParseJsonAndCreateComm` - Parse JSON and create communicator |
| `rank_graph_builder.cc` | 315-353 | `BuildFromRankTable` - Traverse ranktable to build peers_ |
| `rank_graph_builder.cc` | 323-328 | `peers_.emplace(rankId, peer)` - Use hcclRankid as the key |
| `rank_graph_builder.cc` | 22, 46 | `Build` method accepts the `myRank` parameter |

### Application Sample Code

| File | Lines | Description |
|------|------|------|
| `hcomm/examples/01_communicators/02_one_device_per_process_rank_table/main.cc` | 101-105 | MPI initialization; obtain procRank |
| Same as above | 106, 111 | `devId = procRank`, `aclrtSetDevice(devId)` |
| Same as above | 130 | `HcclCommInitClusterInfoConfig(rankTableFile, devId, ...)` |

### Key Log Files

| File | Key Lines | Description |
|------|--------|------|
| `hccl_vm_install/bin/log.txt` | 79, 83, 85 | rankId/logicDevId/phyDevId printed by `aclrtSetDevice` |
| Same as above | 187, 219, 299 | devLogicId/devPhyId printed by `InitCommonData` |
| Same as above | 1023, 1062, 1094 | ZHF-DEBUG: `peer deviceId` vs `devPhyId` |
| Same as above | 1267, 1309 | `CheckRankGraphAddrs` error messages |

### ranktable.json Files

| File | Description |
|------|------|
| `hccl_vm_install/data/ranktable.json` | Generated by the current run (new code; has problems) |
| `hccl_vm_install/archive/20260815_094326/data/ranktable.json` | Archive generated by old code (correct) |
| `hccl_vm_install/archive/20260815_094636/data/ranktable.json` | Archive from another run |

---

## Excluded Items

The following factors have been analyzed and confirmed **not to be** the root cause of this problem:

### chipId = info.phyId in RaGetDevEidInfoList

```cpp
// hccp_ccu_stub.cc:573
infoList[idx].chipId = info.phyId; // todo: single server, logic id equals rank id, but this is problematic for multi-server.
```

This location does have a potential hidden risk (as noted by the TODO comment), but `CheckRankGraphAddrs` **does not read the chipId field at all**. EID comparison matches only by the original IP/EID value (`localEidSet.count(interface->GetAddr().GetEid())`). Therefore, the chipId assignment issue is unrelated to this error.

### device_id and local_id Both Set to phyDevId in BuildRankEntry

```cpp
// topo_ascend_cluster_parser.cc:414-415
rankEntry["device_id"] = srcDevPhyId;
rankEntry["local_id"] = srcDevPhyId;
```

Since `logic_id = physical_id` (numerically equal) in this simulator, setting both to the same value is not a problem. The issue is that the `srcDevPhyId` (that is, `phyDevId`) passed in is taken from `serverMeta[logicDevId]`, and the calculation method for `logicDevId` (lines 450/451) is the root cause.

### Whether hcomm Filters Peers by Server

`CheckRankGraphAddrs` does not iterate over all peers, nor does it filter by server. It only retrieves the single local peer returned by `GetPeer(myRank)` and checks the interface EID of that peer. There is no cross-server peer comparison issue.

---

## Summary

| Dimension | Content |
|------|------|
| Problem essence | The two EID acquisition paths use inconsistent rankId -> device mapping formulas |
| Path A | Actual hardware EID; devPhyId is determined by the application's `aclrtSetDevice(mpiRankid % D)`; does not go through the ranktable |
| Path B | EID claimed by ranktable; obtained from the ranktable through `GetPeer(myRank=mpiRankid)` |
| Conflict condition | When the number of devices on a server is greater than 1 and the server is not the first server, `rankId % D != deviceIdx` |
| Root cause | Line 451 changed `rankId % D` to `deviceIdx`, breaking the consistency between Path B and Path A |
| myRank ownership | mpiRankid (from `MPI_Comm_rank`, passed in through the rank parameter of `HcclCommInitClusterInfoConfig`) |
| Fix method | Restore line 450: `uint32_t logicDevId = rankId % serverMeta.size();` |
