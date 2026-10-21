# Analysis of rankId and Device Physical ID Mapping Conflict in Ranktable

## 1 Problem Overview

### 1.1 Symptoms

In a cluster topology with multiple servers where some servers contain multiple devices, the hcomm service initialization phase reports a `CheckRankGraphAddrs` failure:

```
[CommunicatorImpl][CheckRankGraphAddrs]the ip address
  IpAddress[...c002:0001...] of ranktable in rank 1 is error!
[CommunicatorImpl][CheckRankGraphAddrs]the ip address
  IpAddress[...c002:000b...] of ranktable in rank 0 is error!
```

Debug logs reveal critical clues:

```
ZHF-DEBUG: peer deviceId is 0, devPhyId is 1   ← rank 1: ranktable says device 0, actually runs on device 1
ZHF-DEBUG: peer deviceId is 1, devPhyId is 0   ← rank 2: ranktable says device 1, actually runs on device 0
```

### 1.2 Trigger Conditions

The issue triggers only when the following conditions are met:

```
At least one server satisfies:
  1. The device count D_s of that server is greater than 1
  2. Other servers before that server have already consumed some rankIds
     (that is, rankIds on that server do not start from 0 with consecutive increments)
```

This is equivalent to: **The YAML topology contains a server with device count greater than 1, and that server is not the only server owning ranks.**

| Configuration Type | Sample | Triggers | Reason |
|---------|------|---------|------|
| Every server has 1 device | (1,1), (1,1,1,1) | No | `rank % 1 = 0` always holds, sequential allocation equals modulo operation |
| Single server with multiple devices | (2), (4), (8) | No | Only one server exists, `rank % D = rank`, sequential allocation equals modulo operation |
| Multiple servers, some with multiple devices | (1,2), (2,4) | **Yes** | On multi-device servers, `rank % D_s != sequential value` |
| Multiple servers, all with multiple devices (uniform) | (2,2), (2,2,2,2), (8x8) | **Yes** | Same as above |

---

## 2 Key Concept Definitions

### 2.1 Two Types of rankId

This document involves two different rankId concepts that must be strictly distinguished:

| Concept | Code Name | Definition | Producer | Allocation Method |
|------|------|------|--------|---------|
| ranktable rankId | **hcclRankid** | Value of the `rank_id` field in ranktable.json | This tool `CreateRankTableFile` | Traverses superPod -> server -> device order, increments from 0 |
| MPI rankId | **mpiRankid** | Number assigned by MPI to each process | MPI runtime (`MPI_Comm_rank`) | Determined by MPI scheduling strategy, typically 0,1,2,... |

> **Implicit Convention**: hcclRankid and mpiRankid are numerically equal (both increment as 0,1,2,...). This is the prerequisite for the entire system to work correctly.

### 2.2 Device Identification System

| Concept | Definition | Relationship in This System |
|------|------|-----------------|
| `physical_id` (phyDevId) | Physical number of the device, fixed permanently | Defined by the cluster topology file |
| `logic_id` (logicDevId) | Logical number of the device, mappable | In this simulator, `logic_id = physical_id` (numerically equal) |
| `device_id` (DB key) | Device table primary key | Auto-incremented by internal database |
| `local_id` | Device logical number recorded in ranktable.json | Set to `srcDevPhyId` in `BuildRankEntry` |
| `serverKey` | Unique identifier of the server | `GetServerKeyById(superPodId, serverId)` |

### 2.3 Software Layers Involved

```
┌─────────────────────────────────────────────────────────┐
│  Application Layer (hccl_test / sample programs)          │
│  MPI_Comm_rank -> mpiRankid                              │
│  aclrtSetDevice(mpiRankid % deviceCount)                 │
│  HcclCommInitClusterInfoConfig(ranktable, mpiRankid,...) │
└──────────────────────┬──────────────────────────────────┘
                       │ calls
┌──────────────────────▼──────────────────────────────────┐
│  hcomm Service Layer (libhccl_v2.so)                     │
│  CommunicatorImpl::CheckRankGraphAddrs()                │
│  RankGraphBuilder::BuildFromRankTable()                  │
└──────────────────────┬──────────────────────────────────┘
                       │ calls
┌──────────────────────▼──────────────────────────────────┐
│  This Project (hccl-vm simulator)                        │
│  AscendClusterTopoParser::CreateRankTableFile()           │
│  aclrtSetDevice (stub) / RaGetDevEidInfoList (stub)      │
└─────────────────────────────────────────────────────────┘
```

---

## 3 Problem Reproduction Environment

### 3.1 Topology Configuration

Test topology: 1 superPod, 2 servers (rack0 has 1 card, rack1 has 2 cards), 3 ranks total.

```
superPod0
├── rack0 (server0): [phyDev0]              ← 1 device
└── rack1 (server1): [phyDev0, phyDev1]     ← 2 devices
```

### 3.2 Process and Device Allocation (Measured from Logs)

| PID | mpiRankid | `aclrtSetDevice(rankId % D)` | logicDevId | phyDevId | Server Location |
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

Key point: rack1 has 2 devices. `mpiRankid % 2` yields 1 for rank 1 and 0 for rank 2, **which is the exact reverse of sequential traversal [0,1]**.

### 3.3 Ranktable.json Comparison Before and After the Change

| rank_id | Old Code `rankId % D` | Old Code device_id | New Code `deviceIdx` | New Code device_id | Actual Application phyDevId |
|---------|---------------------|-----------------|-------------------|-----------------|------------------|
| 0 | `0 % 1 = 0` | 0 | 0 | 0 | 0 |
| 1 | `1 % 2 = 1` | **1** | 1 | **0** | **1** |
| 2 | `2 % 2 = 0` | **0** | 2 | **1** | **0** |

- **Old code** (`rankId % D`): ranktable device_id **matches** actual application phyDevId -> validation passes
- **New code** (`deviceIdx`): device_id values for rank 1 and rank 2 are swapped -> **conflict**

> For the ranktable generated by old code, see archive file `hccl_vm_install/archive/20260815_094326/data/ranktable.json`
> For the ranktable generated by new code, see `hccl_vm_install/data/ranktable.json`

---

## 4 CheckRankGraphAddrs Validation Logic Details

### 4.1 Function Location and Signature

```
File: hcomm/src/legacy/ascend950/framework/communicator/communicator_impl.cc
Function: void CommunicatorImpl::CheckRankGraphAddrs() const
Lines: 1291-1338
```

### 4.2 Validation Flow

The function retrieves EIDs through two separate paths and then compares them for consistency:

```
                      CheckRankGraphAddrs
                      /                  \
                Path A                   Path B
          (actual hardware EID)          (EID claimed by ranktable)
                |                            |
    Find entry in ranktable          rankGraph->GetPeer(myRank)
    where rank.deviceId==devPhyId      (look up peers_ table by rank_id)
                |                            |
    RaGetDevEidInfoList                peer->GetIfaces()
    (phyId=devPhyId)                  Get EID address of interfaces
                |                            |
    localEidSet (EID set)            interface->GetAddr().GetEid()
                \                            /
                 \                          /
                  → localEidSet.count(interface EID) == 0 ?
                       Not in set → THROW exception
```

### 4.3 Path A: Actual Hardware EID

```cpp
// communicator_impl.cc:1306-1316
for (auto& rank : ranktableInfo->ranks) {
    if (rank.deviceId == devPhyId) {                    // Select entry: deviceId == physical device of current process
        HRaInfo info(HrtNetworkMode::HDC, rank.deviceId);
        std::vector<HrtDevEidInfo> localEidInfos = HrtRaGetDevEidInfoList(info);  // Query hardware
        for (auto& eidInfo : localEidInfos) {
            localEidSet.insert(eidInfo.ipAddress.GetEid());  // Store only EID/IP values
        }
        break;  // Call only once, then exit
    }
}
```

**Key Characteristics**:
- The `RaGetDevEidInfoList` interface recognizes only `phyId`, not rankId
- The passed `devPhyId` is the physical device on which the process actually runs
- In this path, ranktable.json is used only to "filter entries where `deviceId == devPhyId`". EID values come from hardware queries
- The `chipId` field is populated (`hccp_ccu_stub.cc:573`), but `CheckRankGraphAddrs` **does not read chipId**. EID comparison matches only by raw IP value

### 4.4 Path B: EID Claimed by Ranktable

```cpp
// communicator_impl.cc:1322-1328
const auto& peer = rankGraph->GetPeer(myRank);   // Look up peers_ by rank_id == myRank
const auto& interfaces = peer->GetIfaces();
for (auto& interface : interfaces) {
    // Check if the interface EID is in localEidSet
    if (interface->GetPos() == AddrPosition::DEVICE
        && protocols.count(LinkProtocol::PCIE) == 0
        && protocols.count(LinkProtocol::UBOE) == 0
        && localEidSet.count(interface->GetAddr().GetEid()) == 0) {
        THROW<InvalidParamsException>(...);  // EID mismatch, report error
    }
}
```

**Key Characteristics**:
- `GetPeer(myRank)` directly uses `myRank` as key to look up the `peers_` table
- The interface EID of the peer originates from the `level_list` of the entry where `rank_id == myRank` in ranktable.json
- **This path is entirely determined by ranktable.json content**, that is, affected by lines 450/451

### 4.5 Acquisition of devPhyId and myRank

```cpp
// communicator_impl.cc:1256-1269 (InitCommonData)
myRank = commParams.myRank;                              // ← From rank parameter = mpiRankid
devLogicId = HrtGetDevice();                              // ← Logical device already set by current process
devPhyId = HrtGetDevicePhyIdByIndex(devLogicId);         // ← Physical device corresponding to that logical device
HCCL_INFO("ZHF-DEBUG: devLogicId is %u, devPhyId is %u", devLogicId, devPhyId);
```

---

## 5 Source and Ownership of myRank

### 5.1 Complete Propagation Chain

```
┌─ Application Layer ───────────────────────────────────────────────────┐
│                                                                       │
│  MPI_Comm_rank(MPI_COMM_WORLD, &procRank)    // procRank=mpiRankid    │
│  uint32_t devId = (uint32_t)procRank;                                 │
│                                                                       │
│  // devId serves two purposes simultaneously:                         │
│  aclrtSetDevice(devId);                      // select device         │
│  HcclCommInitClusterInfoConfig(rtk, devId, &config, &comm);  // pass rank │
│                           ↑                                           │
│                      rank parameter = mpiRankid                        │
└──────────────────────────┬────────────────────────────────────────────┘
                           │
┌─ hcomm Entry (op_base_v2.cc) ────────────────────────────────────────▼┐
│                                                                       │
│  HcclCommInitClusterInfoConfigV2(clusterInfo, uint32_t rank, ...)     │
│  // op_base_v2.cc:455                                                 │
│  → ParseJsonAndCreateComm(data, rank, ...)                            │
│  // op_base_v2.cc:501                                                 │
│  → CreateCommConfig(rank, config, comm, ranktableM)                   │
│  // op_base_v2.cc:318                                                 │
└──────────────────────────┬────────────────────────────────────────────┘
                           │
┌─ CommParams Construction (op_base_v2.cc:146-152) ────────────────────▼┐
│                                                                       │
│  Hccl::CommParams commParams{                                         │
│      commId,                                                          │
│      static_cast<Hccl::RankId>(rank),   // ← myRank = mpiRankid      │
│      0,                                                               │
│      static_cast<Hccl::RankId>(rank),   // rankInParentComm = rank    │
│      ...                                                              │
│  };                                                                   │
└──────────────────────────┬────────────────────────────────────────────┘
                           │
┌─ CommunicatorImpl (communicator_impl.cc:1261) ──────────────────────▼┐
│                                                                       │
│  void CommunicatorImpl::InitCommonData(const CommParams& params) {    │
│      myRank = commParams.myRank;    // ← Final destination: mpiRankid │
│      devLogicId = HrtGetDevice();                                     │
│      devPhyId = HrtGetDevicePhyIdByIndex(devLogicId);                 │
│  }                                                                    │
└───────────────────────────────────────────────────────────────────────┘
```

### 5.2 Conclusion

**`myRank` is mpiRankid.** It is passed through the `rank` parameter of `HcclCommInitClusterInfoConfig`, originating from the application `MPI_Comm_rank`.

### 5.3 Construction and Query of the peers_ Table

```cpp
// rank_graph_builder.cc:320-328 (BuildFromRankTable)
for (const auto& rankInfo : rankTable_->ranks) {
    RankId rankId = rankInfo.rankId;           // ← hcclRankid (rank_id from ranktable.json)
    auto peer = make_shared<NetInstance::Peer>(
        rankId, rankInfo.localId, rankInfo.replacedLocalId,
        rankInfo.deviceId, ...);               // ← Peer deviceId comes from ranktable.json
    peers_.emplace(rankId, peer);              // ← peers_ uses hcclRankid as key
}
```

```cpp
// communicator_impl.cc:1322 (CheckRankGraphAddrs)
const auto& peer = rankGraph->GetPeer(myRank);  // GetPeer(peers_.at(myRank))
//        ↑ Uses mpiRankid to look up hcclRankid table
```

**The lookup succeeds** purely because hcclRankid == mpiRankid (both increment numerically as 0,1,2,...).

### 5.4 Source of Peer deviceId

```
Entry in ranktable.json where rank_id==myRank
    ↓ (BuildFromRankTable reads it)
rankInfo.deviceId
    ↓ (Passed to Peer constructor)
peer->deviceId_ = rankInfo.deviceId
    ↓ (GetDeviceId returns it)
peer->GetDeviceId() == device_id of that rank entry in ranktable
```

This is the source of `peer deviceId` in logs. It equals the `device_id` field of the entry where `rank_id == myRank` in ranktable.json.

---

## 6 How the Two EID Paths Relate to rankId

### 6.1 Overview Diagram

```
                          mpiRankid (= myRank)
                         /                   \
                        /                     \
               ┌─ Path A ─┐              ┌─ Path B ──────────┐
               │          │              │                    │
    App calls  │ aclrtSetDevice           │ GetPeer(myRank)   │
    (app layer)│   (mpiRankid % D)        │   = peers_[myRank] │
               │          │              │   = entry in       │
               │          │              │     ranktable where │
               │          │              │     rank_id==myRank │
               │          │              │          │          │
    Look up    │ (serverKey,logic_id)    │     interface EID   │
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

### 6.2 rankId -> Device Mapping Formulas for Both Paths

| Path | rankId -> Device Mapping Formula | Formula Location | Affected by Lines 450/451 |
|------|--------------------------|-------------|---------------------|
| Path A (hardware) | `server.devices[mpiRankid % D]` | Application hardcoded (`aclrtSetDevice`) | **No**, application is fixed |
| Path B (ranktable) | `server.devices[logicDevId]`, logicDevId determined by lines 450/451 | `CreateRankTableFile` | **Yes** |

### 6.3 Formula Consistency Analysis

#### Old Code (Line 450): `logicDevId = rankId % serverMeta.size()`

```cpp
uint32_t logicDevId = rankId % serverMeta.size();  // Same formula as application rank % D
```

| rankId | Path A: `rankId % D` -> device | Path B(450): `rankId % D` -> device | Consistent |
|--------|-------------------------------|-------------------------------------|---------|
| 0 | `0 % 1 = 0` -> phyDev0 | `0 % 1 = 0` -> phyDev0 | Consistent |
| 1 | `1 % 2 = 1` -> phyDev1 | `1 % 2 = 1` -> phyDev1 | Consistent |
| 2 | `2 % 2 = 0` -> phyDev0 | `2 % 2 = 0` -> phyDev0 | Consistent |

-> Both sides use the same mapping -> EIDs match -> validation passes

#### New Code (Line 451): `logicDevId = deviceIdx`

```cpp
uint32_t logicDevId = deviceIdx;  // Sequential allocation 0,1,2,..., may differ from application % D formula
```

| rankId | Path A: `rankId % D` -> device | Path B(451): `deviceIdx` -> device | Consistent |
|--------|-------------------------------|-------------------------------------|---------|
| 0 | `0 % 1 = 0` -> phyDev0 | `0` -> phyDev0 | Consistent |
| 1 | `1 % 2 = 1` -> phyDev1 | `0` -> phyDev0 | **Inconsistent** |
| 2 | `2 % 2 = 0` -> phyDev0 | `1` -> phyDev1 | **Inconsistent** |

-> Devices for rank 1 and rank 2 are swapped -> EIDs do not match -> validation fails

### 6.4 Specific EID Comparison Failure Process (Using rank 1 as Example)

```
rank 1 (PID 13312):
  ┌─ Path A (actual hardware EID) ────────────────────────────────┐
  │ app: aclrtSetDevice(1 % 2 = 1) -> logicDevId=1               │
  │ Device table: logic_id=1 -> phyDevId=1                        │
  │ Entry in ranktable where device_id==1 -> RaGetDevEidInfoList(1) │
  │ localEidSet = {192.2.0.11, 192.2.0.18}  (real IPs of device 1) │
  └───────────────────────────────────────────────────────────────┘

  ┌─ Path B (EID claimed by ranktable) ──────────────────────────┐
  │ GetPeer(myRank=1) -> peers_[1]                               │
  │   = entry in ranktable where rank_id=1                        │
  │ New code: that entry has device_id=0 (should be 1)           │
  │ peer->GetDeviceId() = 0  (ranktable says device 0)           │
  │ interface EID = {192.2.0.1, 192.2.0.8} (IPs of device 0)     │
  └───────────────────────────────────────────────────────────────┘

  Compare: 192.2.0.1 in {192.2.0.11, 192.2.0.18} ?
        -> Not in set -> THROW "ip address ... is error!"
```

Log confirmation:
- `log.txt:1062`: `peer deviceId is 0, devPhyId is 1` (peer gets deviceId=0 through Path B, actually runs on devPhyId=1)
- `log.txt:1267`: `the ip address ...c002:0001... of ranktable in rank 1 is error!`

---

## 7 Root Cause

### 7.1 One-Sentence Summary

**Both EID retrieval paths use rankId as the anchor point, but map to devices through different formulas.** Lines 450/451 changed only the formula on Path B (ranktable side). The formula on Path A (application side) remains fixed. When the server device count is greater than 1 and the server is not the first server, the two formulas produce different results, causing the same rankId to point to different devices on each side and EID comparison to fail.

### 7.2 rankId Association on Path A (Indirect)

The `RaGetDevEidInfoList(devPhyId)` interface itself recognizes only phyDevId, not rankId. However, `devPhyId` is selected by the application using `aclrtSetDevice(mpiRankid % D)`, which binds mpiRankid into the process:

```
mpiRankid → aclrtSetDevice(mpiRankid % D) → Device table → devPhyId → RaGetDevEidInfoList(devPhyId)
```

ranktable.json **does not participate** in device selection on this path. It is used only to filter entries where `deviceId == devPhyId` to determine the phyId parameter.

### 7.3 rankId Association on Path B (Direct)

The interface EID of the peer is obtained by looking up the ranktable using `GetPeer(myRank)`, where `myRank` is mpiRankid:

```
mpiRankid(=myRank) → GetPeer(myRank) → peers_[myRank] → entry in ranktable where rank_id==myRank → interface EID
```

ranktable.json **entirely determines** which device EID corresponds to that rankId on this path.

### 7.4 Consistency Requirement

Since both paths start from mpiRankid, the rank -> device formulas on both sides must be consistent:

```
Path A formula:  device = server.devices[mpiRankid % D]     (application hardcoded, cannot change)
Path B formula:  device = server.devices[logicDevId]         (determined by lines 450/451)
```

- **Line 450** `logicDevId = rankId % serverMeta.size()` -> Consistent with Path A -> passes
- **Line 451** `logicDevId = deviceIdx` -> Diverges from Path A -> fails

### 7.5 Complete Mapping Chain Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Consistency Requirement                            │
│                                                                         │
│  rank → device mapping must be consistent across all layers:            │
│                                                                         │
│  1. Application aclrtSetDevice:    logicDevId = mpiRankid % D_s         │
│  2. ranktable.json device_id: must equal actual phyDevId chosen by app  │
│  3. ranktable.json local_id:  must equal actual logicDevId chosen by app│
│  4. EIDs in ranktable.json: must equal real EIDs for that phyDevId      │
│                                                                         │
│  Any inconsistency causes CheckRankGraphAddrs to report an error         │
└─────────────────────────────────────────────────────────────────────────┘

Mapping chain:

  mpiRankid (allocated by MPI)
      │
      ├──→ [Application Layer] aclrtSetDevice(mpiRankid % D)
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
      │         │    localEidSet (actual hardware EIDs)
      │         │
      │    (mpiRankid also passed as rank parameter to hcomm)
      │         │
      │         ▼
      │    myRank = mpiRankid
      │         │
      │         ▼
      │    rankGraph->GetPeer(myRank)
      │         │
      │         ▼
      │    peers_[myRank] ← entry in ranktable.json where rank_id==myRank
      │         │
      │         ▼
      │    peer interface EID (EID claimed by ranktable)
      │         │
      └─────────┘
                │
                ▼
         localEidSet.count(interface EID) == 0 ?
              │
         Mismatch → report error
```

---

## 8 Change Point Analysis

### 8.1 Modified Code

```
File: src/topo/topo_ascend_cluster_parser.cc
Function: AscendClusterTopoParser::CreateRankTableFile
Lines: 449-456

Before change (line 450):
    for (uint32_t deviceIdx = 0; deviceIdx < serverMeta.size(); ++deviceIdx) {
        uint32_t logicDevId = rankId % serverMeta.size();   // ← Old code
        PhyDeviceId phyDevId = serverMeta[logicDevId];
        ...
        rankId++;
    }

After change (line 451):
    for (uint32_t deviceIdx = 0; deviceIdx < serverMeta.size(); ++deviceIdx) {
        uint32_t logicDevId = deviceIdx;                    // ← New code
        PhyDeviceId phyDevId = serverMeta[logicDevId];
        ...
        rankId++;
    }
```

### 8.2 Impact of the Change

This change alters the `device_id` and `local_id` corresponding to each `rank_id` in `ranktable.json`:

| rank_id | Old: `rankId % D` corresponding device_idx | New: `deviceIdx` | Same? |
|---------|-----------------------------------|-----------------|-------|
| 0 (rack0) | 0 | 0 | Same |
| 1 (rack1) | 1 | 0 | **Different** |
| 2 (rack1) | 0 | 1 | **Different** |

When every server has exactly 1 device or only a single server exists, the two formulas are equivalent (`rankId % D` within the first server equals `deviceIdx`). The change causes no issue. However, with multiple servers where a non-first server has multiple devices, the two formulas differ.

### 8.3 device_id and local_id Settings in BuildRankEntry

```cpp
// topo_ascend_cluster_parser.cc:413-416
json AscendClusterTopoParser::BuildRankEntry(const Server &server, int srcDevPhyId, ...)
{
    json rankEntry;
    rankEntry["device_id"] = srcDevPhyId;    // ← Set to physical device ID
    rankEntry["local_id"] = srcDevPhyId;     // ← Also set to physical device ID
    rankEntry["rank_id"] = rankId;
    ...
}
```

Since `logic_id = physical_id` in this simulator, `device_id` and `local_id` have the same value. `BuildRankEntry` itself has no problem. The real issue is that the `phyDevId` passed into `BuildRankEntry` comes from `serverMeta[logicDevId]`, and the calculation method of `logicDevId` (lines 450/451) determines which physical device is assigned to that rank.

---

## 9 Fix

### 9.1 Core Fix

**Restore the formula on line 450** so that the rank -> device mapping in the ranktable matches the application `aclrtSetDevice(rank % D)`:

```cpp
uint32_t logicDevId = rankId % serverMeta.size();
```

### 9.2 Data Flow Verification After Fix

Verify with the test topology (rack0=1 card, rack1=2 cards):

```
Step 1: Application selects device (aclrtSetDevice)
  rank 0: 0 % 1 = 0 → rack0 logicDevId=0 → phyDevId=0  (IP=192.1.0.8)
  rank 1: 1 % 2 = 1 → rack1 logicDevId=1 → phyDevId=1  (IP=192.2.0.18)
  rank 2: 2 % 2 = 0 → rack1 logicDevId=0 → phyDevId=0  (IP=192.2.0.1)

Step 2: ranktable.json generation (rankId % D)
  rank 0: device_id=0, local_id=0  (phyDev0)
  rank 1: device_id=1, local_id=1  (phyDev1)
  rank 2: device_id=0, local_id=0  (phyDev0)

Step 3: CheckRankGraphAddrs validation
  rank 1: devPhyId=1 → localEidSet={192.2.0.11, 192.2.0.18}
          peer(rank_id=1) → device_id=1 → interface EID={192.2.0.18}
          192.2.0.18 in localEidSet → match ✓
  rank 2: devPhyId=0 → localEidSet={192.2.0.1, 192.2.0.8}
          peer(rank_id=2) → device_id=0 → interface EID={192.2.0.1}
          192.2.0.1 in localEidSet → match ✓
```

All mapping chains are consistent. Validation passes.

---

## 10 Key Code Location Index

### 10.1 This Project (CheckerL2_8620) Code

| File | Lines | Function/Description |
|------|------|---------|
| `src/topo/topo_ascend_cluster_parser.cc` | 424-477 | `CreateRankTableFile` - Generate ranktable.json |
| `src/topo/topo_ascend_cluster_parser.cc` | 449-456 | **Change point** - logicDevId calculation (450 old / 451 new) |
| `src/topo/topo_ascend_cluster_parser.cc` | 409-422 | `BuildRankEntry` - Set device_id/local_id/rank_id |
| `src/topo/topo_ascend_cluster_parser.cc` | 200-272 | `InitDynamicModelData` - Initialize Device/Rank tables |
| `src/topo/topo_ascend_cluster_parser.cc` | 202, 234 | Comments explain application uses `aclrtSetDevice(rank % deviceCount)` |
| `src/proxy/level2/aclrt_device_stub.cc` | 69-128 | `aclrtSetDevice` - Stub implementation for device selection by application |
| `src/proxy/level2/aclrt_device_stub.cc` | 87-88 | Look up Device table by `(serverKey, logic_id)` |
| `src/proxy/level2/aclrt_device_stub.cc` | 110-122 | Create Rank table record, print logicDevId/phyDevId |
| `src/proxy/level2/hccl_comm_stub.cc` | 83-111 | `SimGetDeviceComm` - Call `aclrtSetDevice` |
| `src/proxy/level2/hccl_comm_stub.cc` | 130-131 | `HcclCommInitAll` calls `SimGetDeviceComm` |
| `src/proxy/level2/hccp_ccu_stub.cc` | 528-541 | `GetAllUsedEndPoint` - Look up EndPoint by phyDevId |
| `src/proxy/level2/hccp_ccu_stub.cc` | 554-587 | `RaGetDevEidInfoList` - Return EID list by phyId |
| `src/proxy/level2/hccp_ccu_stub.cc` | 573 | `chipId = info.phyId` (has TODO comment, unrelated to this issue) |

### 10.2 hcomm Service Code

| File | Lines | Function/Description |
|------|------|---------|
| `communicator_impl.cc` | 1256-1269 | `InitCommonData` - Set myRank, devLogicId, devPhyId |
| `communicator_impl.cc` | 1291-1338 | `CheckRankGraphAddrs` - EID validation logic |
| `communicator_impl.cc` | 1306-1316 | Path A: Filter by `deviceId==devPhyId` and call `RaGetDevEidInfoList` |
| `communicator_impl.cc` | 1322-1328 | Path B: `GetPeer(myRank)` retrieves interface EID and compares |
| `communicator_impl.cc` | 1323 | ZHF-DEBUG log: `peer deviceId` vs `devPhyId` |
| `communicator_impl.cc` | 1353-1371 | `InitRankGraph` - Call bridge to build rankGraph |
| `communicator_impl.cc` | 1364 | `bridge->buildFromString(ranktableM, topoPath, myRank, ...)` |
| `op_base_v2.cc` | 455-514 | `HcclCommInitClusterInfoConfigV2` - Entry function |
| `op_base_v2.cc` | 131-152 | `CreateCommConfig` - Construct CommParams (myRank=rank) |
| `op_base_v2.cc` | 311-322 | `ParseJsonAndCreateComm` - Parse JSON and create communicator |
| `rank_graph_builder.cc` | 315-353 | `BuildFromRankTable` - Traverse ranktable to build peers_ |
| `rank_graph_builder.cc` | 323-328 | `peers_.emplace(rankId, peer)` - Use hcclRankid as key |
| `rank_graph_builder.cc` | 22, 46 | `Build` method receives `myRank` parameter |

### 10.3 Application Sample Code

| File | Lines | Description |
|------|------|------|
| `hcomm/examples/01_communicators/02_one_device_per_process_rank_table/main.cc` | 101-105 | MPI initialization, obtain procRank |
| Same as above | 106, 111 | `devId = procRank`, `aclrtSetDevice(devId)` |
| Same as above | 130 | `HcclCommInitClusterInfoConfig(rankTableFile, devId, ...)` |

### 10.4 Key Log Files

| File | Key Lines | Description |
|------|--------|------|
| `hccl_vm_install/bin/log.txt` | 79, 83, 85 | `aclrtSetDevice` prints rankId/logicDevId/phyDevId |
| Same as above | 187, 219, 299 | `InitCommonData` prints devLogicId/devPhyId |
| Same as above | 1023, 1062, 1094 | ZHF-DEBUG: `peer deviceId` vs `devPhyId` |
| Same as above | 1267, 1309 | `CheckRankGraphAddrs` error messages |

### 10.5 ranktable.json Files

| File | Description |
|------|------|
| `hccl_vm_install/data/ranktable.json` | Generated by current run (new code, has issue) |
| `hccl_vm_install/archive/20260815_094326/data/ranktable.json` | Archive generated by old code (correct) |
| `hccl_vm_install/archive/20260815_094636/data/ranktable.json` | Archive from another run |

---

## 11 Excluded Items

The following factors are confirmed **not** to be the root cause of this issue after analysis:

### 11.1 chipId = info.phyId in RaGetDevEidInfoList

```cpp
// hccp_ccu_stub.cc:573
infoList[idx].chipId = info.phyId; // todo: single server, logic id equals rank id, but multi-server has issues here.
```

This location does have a potential risk (TODO comment). However, `CheckRankGraphAddrs` **does not read the chipId field at all**. EID comparison matches only by raw IP/EID value (`localEidSet.count(interface->GetAddr().GetEid())`). Therefore, the chipId assignment issue is unrelated to this error.

### 11.2 device_id and local_id Both Set to phyDevId in BuildRankEntry

```cpp
// topo_ascend_cluster_parser.cc:414-415
rankEntry["device_id"] = srcDevPhyId;
rankEntry["local_id"] = srcDevPhyId;
```

Since `logic_id = physical_id` (numerically equal) in this simulator, setting both to the same value is fine. The problem is that the passed `srcDevPhyId` (that is, `phyDevId`) comes from `serverMeta[logicDevId]`, and the calculation method of `logicDevId` (lines 450/451) is the root cause.

### 11.3 Whether hcomm Filters Peers by Server

`CheckRankGraphAddrs` does not iterate over all peers, nor does it filter by server. It only takes the single local peer returned by `GetPeer(myRank)` and checks the interface EIDs of that peer. No cross-server peer comparison issue exists.

---

## 12 Summary

| Dimension | Content |
|------|------|
| Problem Essence | The two EID retrieval paths use inconsistent rankId -> device mapping formulas |
| Path A | Actual hardware EID, devPhyId determined by application `aclrtSetDevice(mpiRankid % D)`, does not go through ranktable |
| Path B | EID claimed by ranktable, obtained from ranktable through `GetPeer(myRank=mpiRankid)` |
| Conflict Condition | When server device count is greater than 1 and server is not the first server, `rankId % D != deviceIdx` |
| Root Cause | Line 451 changed `rankId % D` to `deviceIdx`, breaking consistency between Path B and Path A |
| myRank Ownership | mpiRankid (from `MPI_Comm_rank`, passed through the rank parameter of `HcclCommInitClusterInfoConfig`) |
| Fix Method | Restore line 450: `uint32_t logicDevId = rankId % serverMeta.size();` |