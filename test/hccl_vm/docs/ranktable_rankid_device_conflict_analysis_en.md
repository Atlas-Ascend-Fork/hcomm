# Ranktable 中 rankId 与 Device 物理ID 映射冲突问题分析

## 1 问题概述

### 1.1 现象

在多 server 且部分 server 含多个 device 的集群拓扑配置下，hcomm 业务初始化阶段报错 `CheckRankGraphAddrs` 失败：

```
[CommunicatorImpl][CheckRankGraphAddrs]the ip address
  IpAddress[...c002:0001...] of ranktable in rank 1 is error!
[CommunicatorImpl][CheckRankGraphAddrs]the ip address
  IpAddress[...c002:000b...] of ranktable in rank 0 is error!
```

调试日志显示关键线索：

```
ZHF-DEBUG: peer deviceId is 0, devPhyId is 1   ← rank 1: ranktable 说在 device 0，实际跑在 device 1
ZHF-DEBUG: peer deviceId is 1, devPhyId is 0   ← rank 2: ranktable 说在 device 1，实际跑在 device 0
```

### 1.2 触发条件

仅当满足以下条件时触发：

```
存在至少一个 server 满足：
  1. 该 server 的 device 数量 D_s > 1
  2. 该 server 之前已有其他 server 消耗了部分 rankId
     （即该 server 上的 rankId 不从 0 开始连续递增）
```

等价于：**YAML 拓扑配置中存在 device 数 > 1 的 server，且该 server 不是唯一拥有 rank 的 server。**

| 配置类型 | 示例 | 是否触发 | 原因 |
|---------|------|---------|------|
| 所有 server 各 1 个 device | (1,1), (1,1,1,1) | 不触发 | `rank % 1 = 0` 恒成立，顺序分配与模运算等价 |
| 单 server 多 device | (2), (4), (8) | 不触发 | 只有一个 server，`rank % D = rank`，顺序分配与模运算等价 |
| 多 server，部分多 device | (1,2), (2,4) | **触发** | 多 device 的 server 上 `rank % D_s != 顺序值` |
| 多 server，全部多 device（均匀） | (2,2), (2,2,2,2), (8x8) | **触发** | 同上 |

---

## 2 关键概念定义

### 2.1 两种 rankId

本文涉及两个不同的 rankId 概念，必须严格区分：

| 概念 | 代号 | 定义 | 产生方 | 分配方式 |
|------|------|------|--------|---------|
| ranktable rankId | **hcclRankid** | ranktable.json 中 `rank_id` 字段的值 | 本工具 `CreateRankTableFile` | 按 superPod -> server -> device 遍历顺序，从 0 递增 |
| MPI rankId | **mpiRankid** | MPI 给每个进程分配的编号 | MPI 运行时 (`MPI_Comm_rank`) | 由 MPI 调度策略决定，通常为 0,1,2,... |

> **隐含约定**：hcclRankid 与 mpiRankid 在数值上相等（都是 0,1,2,... 递增），这是整个系统正常工作的前提。

### 2.2 设备标识体系

| 概念 | 定义 | 在本系统中的关系 |
|------|------|-----------------|
| `physical_id` (phyDevId) | Device 的物理编号，固定不变 | 由集群拓扑文件定义 |
| `logic_id` (logicDevId) | Device 的逻辑编号，可映射 | 本 simulator 中 `logic_id = physical_id`（数值相等） |
| `device_id` (DB key) | Device 表主键 | 内部数据库自增分配 |
| `local_id` | ranktable.json 中记录的 device 逻辑编号 | `BuildRankEntry` 中设为 `srcDevPhyId` |
| `serverKey` | Server 的唯一标识 | `GetServerKeyById(superPodId, serverId)` |

### 2.3 涉及的软件层级

```
┌─────────────────────────────────────────────────────────┐
│  应用层 (hccl_test / 示例程序)                            │
│  MPI_Comm_rank -> mpiRankid                              │
│  aclrtSetDevice(mpiRankid % deviceCount)                 │
│  HcclCommInitClusterInfoConfig(ranktable, mpiRankid,...) │
└──────────────────────┬──────────────────────────────────┘
                       │ 调用
┌──────────────────────▼──────────────────────────────────┐
│  hcomm 业务层 (libhccl_v2.so)                            │
│  CommunicatorImpl::CheckRankGraphAddrs()                │
│  RankGraphBuilder::BuildFromRankTable()                  │
└──────────────────────┬──────────────────────────────────┘
                       │ 调用
┌──────────────────────▼──────────────────────────────────┐
│  本项目 (hccl-vm simulator)                              │
│  AscendClusterTopoParser::CreateRankTableFile()           │
│  aclrtSetDevice (stub) / RaGetDevEidInfoList (stub)      │
└─────────────────────────────────────────────────────────┘
```

---

## 3 问题复现环境

### 3.1 拓扑配置

测试用拓扑：1 个 superPod，2 个 server（rack0 有 1 卡，rack1 有 2 卡），共 3 个 rank。

```
superPod0
├── rack0 (server0): [phyDev0]              ← 1 个 device
└── rack1 (server1): [phyDev0, phyDev1]     ← 2 个 device
```

### 3.2 进程与设备分配（日志实测）

| PID | mpiRankid | `aclrtSetDevice(rankId % D)` | logicDevId | phyDevId | 所在 server |
|-----|-----------|------------------------------|------------|----------|-------------|
| 13311 | 0 | `0 % 1 = 0` | 0 | 0 | rack0 (1卡) |
| 13312 | 1 | `1 % 2 = 1` | 1 | 1 | rack1 (2卡) |
| 13313 | 2 | `2 % 2 = 0` | 0 | 0 | rack1 (2卡) |

> 日志来源：`log.txt:79,83,85`
> ```
> [aclrtSetDevice] Init Rank: rankId=1, deviceKey=10, logicDevId=1, phyDevId=1 totalRanks=3
> [aclrtSetDevice] Init Rank: rankId=2, deviceKey=9, logicDevId=0, phyDevId=0 totalRanks=3
> [aclrtSetDevice] Init Rank: rankId=0, deviceKey=1, logicDevId=0, phyDevId=0 totalRanks=3
> ```

关键点：rack1 有 2 个 device，`mpiRankid % 2` 对 rank 1 得 1、对 rank 2 得 0，**与顺序遍历 [0,1] 恰好相反**。

### 3.3 修改前后的 ranktable.json 对比

| rank_id | 旧代码 `rankId % D` | 旧代码 device_id | 新代码 `deviceIdx` | 新代码 device_id | 应用实际 phyDevId |
|---------|---------------------|-----------------|-------------------|-----------------|------------------|
| 0 | `0 % 1 = 0` | 0 | 0 | 0 | 0 |
| 1 | `1 % 2 = 1` | **1** | 1 | **0** | **1** |
| 2 | `2 % 2 = 0` | **0** | 2 | **1** | **0** |

- **旧代码**（`rankId % D`）：ranktable 的 device_id 与应用实际 phyDevId **一致** -> 通过校验
- **新代码**（`deviceIdx`）：rank 1 和 rank 2 的 device_id 被对调 -> **冲突**

> 旧代码生成的 ranktable 见归档文件 `hccl_vm_install/archive/20260815_094326/data/ranktable.json`
> 新代码生成的 ranktable 见 `hccl_vm_install/data/ranktable.json`

---

## 4 CheckRankGraphAddrs 校验逻辑详解

### 4.1 函数位置与签名

```
文件: hcomm/src/legacy/ascend950/framework/communicator/communicator_impl.cc
函数: void CommunicatorImpl::CheckRankGraphAddrs() const
行号: 1291-1338
```

### 4.2 校验流程

该函数通过两条路径分别获取 EID，然后比对二者是否一致：

```
                     CheckRankGraphAddrs
                     /                  \
               路径 A                   路径 B
          (硬件真实 EID)              (ranktable声称的EID)
               |                            |
   在ranktable中找                  rankGraph->GetPeer(myRank)
   rank.deviceId==devPhyId            (按rank_id查peers_表)
               |                            |
   RaGetDevEidInfoList                peer->GetIfaces()
   (phyId=devPhyId)                  取接口的EID地址
               |                            |
   localEidSet (EID集合)         interface->GetAddr().GetEid()
               \                            /
                \                          /
                 → localEidSet.count(interface EID) == 0 ?
                      不在集合中 → THROW 异常
```

### 4.3 路径 A：硬件真实 EID

```cpp
// communicator_impl.cc:1306-1316
for (auto& rank : ranktableInfo->ranks) {
    if (rank.deviceId == devPhyId) {                    // 选条目：deviceId == 本进程物理设备
        HRaInfo info(HrtNetworkMode::HDC, rank.deviceId);
        std::vector<HrtDevEidInfo> localEidInfos = HrtRaGetDevEidInfoList(info);  // 查硬件
        for (auto& eidInfo : localEidInfos) {
            localEidSet.insert(eidInfo.ipAddress.GetEid());  // 只存 EID/IP 值
        }
        break;  // 只调用一次，然后退出
    }
}
```

**关键特征**：
- `RaGetDevEidInfoList` 接口本身只认 `phyId`，不认 rankId
- 传入的 `devPhyId` 是进程实际运行的物理设备
- 该路径中 ranktable.json 仅用于"按 `deviceId == devPhyId` 筛选条目"，EID 值来自硬件查询
- `chipId` 字段虽被填充（`hccp_ccu_stub.cc:573`），但 `CheckRankGraphAddrs` **不读取 chipId**，EID 比对仅按 IP 原值匹配

### 4.4 路径 B：ranktable 声称的 EID

```cpp
// communicator_impl.cc:1322-1328
const auto& peer = rankGraph->GetPeer(myRank);   // 按 rank_id == myRank 查 peers_
const auto& interfaces = peer->GetIfaces();
for (auto& interface : interfaces) {
    // 检查 interface 的 EID 是否在 localEidSet 中
    if (interface->GetPos() == AddrPosition::DEVICE
        && protocols.count(LinkProtocol::PCIE) == 0
        && protocols.count(LinkProtocol::UBOE) == 0
        && localEidSet.count(interface->GetAddr().GetEid()) == 0) {
        THROW<InvalidParamsException>(...);  // EID 不匹配，报错
    }
}
```

**关键特征**：
- `GetPeer(myRank)` 直接用 `myRank` 作为 key 查 `peers_` 表
- peer 的接口 EID 来源于 ranktable.json 中 `rank_id == myRank` 条目的 `level_list`
- **该路径完全由 ranktable.json 的内容决定**，即受 450/451 行影响

### 4.5 devPhyId 与 myRank 的获取

```cpp
// communicator_impl.cc:1256-1269 (InitCommonData)
myRank = commParams.myRank;                              // ← 来自 rank 参数 = mpiRankid
devLogicId = HrtGetDevice();                              // ← 当前进程已 set 的逻辑设备
devPhyId = HrtGetDevicePhyIdByIndex(devLogicId);         // ← 该逻辑设备对应的物理设备
HCCL_INFO("ZHF-DEBUG: devLogicId is %u, devPhyId is %u", devLogicId, devPhyId);
```

---

## 5 myRank 的来源与归属

### 5.1 完整传递链路

```
┌─ 应用层 ──────────────────────────────────────────────────────────┐
│                                                                  │
│  MPI_Comm_rank(MPI_COMM_WORLD, &procRank)    // procRank=mpiRankid │
│  uint32_t devId = (uint32_t)procRank;                             │
│                                                                  │
│  // devId 同时用于两个目的：                                      │
│  aclrtSetDevice(devId);                      // 选设备             │
│  HcclCommInitClusterInfoConfig(rtk, devId, &config, &comm);  // 传 rank │
│                           ↑                                      │
│                      rank 参数 = mpiRankid                         │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌─ hcomm 入口 (op_base_v2.cc) ──────────────────────────────────────▼┐
│                                                                   │
│  HcclCommInitClusterInfoConfigV2(clusterInfo, uint32_t rank, ...)  │
│  // op_base_v2.cc:455                                             │
│  → ParseJsonAndCreateComm(data, rank, ...)                         │
│  // op_base_v2.cc:501                                             │
│  → CreateCommConfig(rank, config, comm, ranktableM)               │
│  // op_base_v2.cc:318                                             │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌─ CommParams 构造 (op_base_v2.cc:146-152) ─────────────────────────▼┐
│                                                                    │
│  Hccl::CommParams commParams{                                      │
│      commId,                                                       │
│      static_cast<Hccl::RankId>(rank),   // ← myRank = mpiRankid     │
│      0,                                                            │
│      static_cast<Hccl::RankId>(rank),   // rankInParentComm 也 = rank │
│      ...                                                           │
│  };                                                                │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
┌─ CommunicatorImpl (communicator_impl.cc:1261) ────────────────────▼┐
│                                                                    │
│  void CommunicatorImpl::InitCommonData(const CommParams& params) { │
│      myRank = commParams.myRank;    // ← 最终落地：mpiRankid         │
│      devLogicId = HrtGetDevice();                                  │
│      devPhyId = HrtGetDevicePhyIdByIndex(devLogicId);               │
│  }                                                                 │
└────────────────────────────────────────────────────────────────────┘
```

### 5.2 结论

**`myRank` 是 mpiRankid。** 它通过 `HcclCommInitClusterInfoConfig` 的 `rank` 参数传入，源头是应用的 `MPI_Comm_rank`。

### 5.3 peers_ 表的构建与查询

```cpp
// rank_graph_builder.cc:320-328 (BuildFromRankTable)
for (const auto& rankInfo : rankTable_->ranks) {
    RankId rankId = rankInfo.rankId;           // ← hcclRankid (ranktable.json 的 rank_id)
    auto peer = make_shared<NetInstance::Peer>(
        rankId, rankInfo.localId, rankInfo.replacedLocalId,
        rankInfo.deviceId, ...);               // ← Peer 的 deviceId 来自 ranktable.json
    peers_.emplace(rankId, peer);              // ← peers_ 以 hcclRankid 为 key
}
```

```cpp
// communicator_impl.cc:1322 (CheckRankGraphAddrs)
const auto& peer = rankGraph->GetPeer(myRank);  // GetPeer(peers_.at(myRank))
//        ↑ 用 mpiRankid 查 hcclRankid 表
```

**能查到**，纯粹是因为 hcclRankid == mpiRankid（数值上都是 0,1,2,... 递增）。

### 5.4 Peer 的 deviceId 来源

```
ranktable.json 的 rank_id==myRank 条目
    ↓ (BuildFromRankTable 读取)
rankInfo.deviceId
    ↓ (Peer 构造函数传入)
peer->deviceId_ = rankInfo.deviceId
    ↓ (GetDeviceId 返回)
peer->GetDeviceId() == ranktable 中该 rank 条目的 device_id
```

这就是日志中 `peer deviceId` 的来源——它等于 ranktable.json 中 `rank_id == myRank` 条目的 `device_id` 字段。

---

## 6 两条 EID 路径如何与 rankId 关联

### 6.1 总览图

```
                         mpiRankid (= myRank)
                        /                   \
                       /                     \
              ┌─ 路径 A ─┐              ┌─ 路径 B ──────────┐
              │          │              │                    │
   app 调用    │ aclrtSetDevice           │ GetPeer(myRank)   │
   (应用层)   │   (mpiRankid % D)        │   = peers_[myRank] │
              │          │              │   = ranktable中     │
              │          │              │     rank_id==myRank │
              │          │              │     的条目           │
              │          │              │          │          │
   查 Device  │ (serverKey,logic_id)    │     interface EID   │
   表得到     │   → device              │     (level_list中   │
   phyDevId   │   → phyDevId           │      的 addr)       │
              │          │              │          │          │
   查硬件     │ RaGetDevEidInfoList      │          │          │
   EID        │   (phyDevId)            │          │          │
              │          │              │          │          │
              ▼          ▼              ▼          ▼          │
         localEidSet (EID集合)    interface EID (单个EID)     │
              \          /              \          /          │
               \        /                \        /           │
                → 比对: localEidSet.count(interface EID) ←────┘
```

### 6.2 两条路径的 rankId -> device 映射公式

| 路径 | rankId -> device 映射公式 | 公式所在位置 | 是否受 450/451 行影响 |
|------|--------------------------|-------------|---------------------|
| 路径 A (硬件) | `server.devices[mpiRankid % D]` | 应用层硬编码 (`aclrtSetDevice`) | **否**，应用固定不变 |
| 路径 B (ranktable) | `server.devices[logicDevId]`，logicDevId 由 450/451 决定 | `CreateRankTableFile` | **是** |

### 6.3 公式一致性分析

#### 旧代码（450 行）：`logicDevId = rankId % serverMeta.size()`

```cpp
uint32_t logicDevId = rankId % serverMeta.size();  // 与应用的 rank % D 公式一致
```

| rankId | 路径 A: `rankId % D` -> device | 路径 B(450): `rankId % D` -> device | 是否一致 |
|--------|-------------------------------|-------------------------------------|---------|
| 0 | `0 % 1 = 0` -> phyDev0 | `0 % 1 = 0` -> phyDev0 | 一致 |
| 1 | `1 % 2 = 1` -> phyDev1 | `1 % 2 = 1` -> phyDev1 | 一致 |
| 2 | `2 % 2 = 0` -> phyDev0 | `2 % 2 = 0` -> phyDev0 | 一致 |

-> 两侧映射相同 -> EID 匹配 -> 校验通过

#### 新代码（451 行）：`logicDevId = deviceIdx`

```cpp
uint32_t logicDevId = deviceIdx;  // 顺序分配 0,1,2,...，与应用的 % D 公式可能不同
```

| rankId | 路径 A: `rankId % D` -> device | 路径 B(451): `deviceIdx` -> device | 是否一致 |
|--------|-------------------------------|-------------------------------------|---------|
| 0 | `0 % 1 = 0` -> phyDev0 | `0` -> phyDev0 | 一致 |
| 1 | `1 % 2 = 1` -> phyDev1 | `0` -> phyDev0 | **不一致** |
| 2 | `2 % 2 = 0` -> phyDev0 | `1` -> phyDev1 | **不一致** |

-> rank 1 和 rank 2 的 device 被对调 -> EID 不匹配 -> 校验失败

### 6.4 具体 EID 比对失败过程（以 rank 1 为例）

```
rank 1 (PID 13312):
  ┌─ 路径 A (硬件真实 EID) ──────────────────────────────────┐
  │ app: aclrtSetDevice(1 % 2 = 1) -> logicDevId=1           │
  │ Device表: logic_id=1 -> phyDevId=1                        │
  │ ranktable中 device_id==1 的条目 -> RaGetDevEidInfoList(1) │
  │ localEidSet = {192.2.0.11, 192.2.0.18}  (device 1的真实IP) │
  └───────────────────────────────────────────────────────────┘

  ┌─ 路径 B (ranktable声称的EID) ────────────────────────────┐
  │ GetPeer(myRank=1) -> peers_[1]                           │
  │   = ranktable中 rank_id=1 的条目                          │
  │ 新代码: 该条目 device_id=0 (应为1)                        │
  │ peer->GetDeviceId() = 0  (ranktable说在device 0)          │
  │ interface EID = {192.2.0.1, 192.2.0.8} (device 0的IP)     │
  └───────────────────────────────────────────────────────────┘

  比对: 192.2.0.1 in {192.2.0.11, 192.2.0.18} ?
        -> 不在集合中 -> THROW "ip address ... is error!"
```

日志印证：
- `log.txt:1062`: `peer deviceId is 0, devPhyId is 1` (peer走路径B得deviceId=0，实际跑在devPhyId=1)
- `log.txt:1267`: `the ip address ...c002:0001... of ranktable in rank 1 is error!`

---

## 7 根本原因

### 7.1 一句话总结

**两条 EID 获取路径都以 rankId 为锚点，但通过不同的公式映射到 device。** 450/451 行只改了路径 B 的公式（ranktable 侧），路径 A 的公式（应用侧）固定不变。当 server 设备数 > 1 且非首 server 时，两条公式产生不同结果，导致同一 rankId 在两侧指向不同 device，EID 比对失败。

### 7.2 路径 A 的 rankId 关联（间接）

`RaGetDevEidInfoList(devPhyId)` 接口本身不认 rankId，只认 phyDevId。但 `devPhyId` 是应用用 `aclrtSetDevice(mpiRankid % D)` 选出来的，这一步把 mpiRankid 绑了进去：

```
mpiRankid → aclrtSetDevice(mpiRankid % D) → Device表 → devPhyId → RaGetDevEidInfoList(devPhyId)
```

ranktable.json 在此路径中**不参与**设备选择（仅用于按 `deviceId == devPhyId` 筛选条目以确定 phyId 参数）。

### 7.3 路径 B 的 rankId 关联（直接）

peer 的接口 EID 是用 `GetPeer(myRank)` 从 ranktable 查表得到的，`myRank` 即 mpiRankid：

```
mpiRankid(=myRank) → GetPeer(myRank) → peers_[myRank] → ranktable中rank_id==myRank条目 → interface EID
```

ranktable.json 在此路径中**完全决定**该 rankId 对应哪个 device 的 EID。

### 7.4 一致性要求

由于两条路径都以 mpiRankid 为起点，要求两侧的 rank -> device 公式一致：

```
路径 A 公式:  device = server.devices[mpiRankid % D]     (应用硬编码，不可改)
路径 B 公式:  device = server.devices[logicDevId]         (由 450/451 行决定)
```

- **450 行** `logicDevId = rankId % serverMeta.size()` -> 与路径 A 一致 -> 通过
- **451 行** `logicDevId = deviceIdx` -> 与路径 A 背离 -> 失败

### 7.5 完整映射链路图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        一致性要求                                        │
│                                                                         │
│  所有层级的 rank → device 映射必须一致:                                   │
│                                                                         │
│  1. 应用 aclrtSetDevice:    logicDevId = mpiRankid % D_s                │
│  2. ranktable.json device_id: 必须等于应用实际选择的 phyDevId             │
│  3. ranktable.json local_id:  必须等于应用实际选择的 logicDevId           │
│  4. ranktable.json 中的 EID: 必须等于该 phyDevId 对应的真实 EID           │
│                                                                         │
│  任何一层不一致都会导致 CheckRankGraphAddrs 报错                          │
└─────────────────────────────────────────────────────────────────────────┘

映射链路:

  mpiRankid (MPI分配)
      │
      ├──→ [应用层] aclrtSetDevice(mpiRankid % D)
      │         │
      │         ▼
      │    Device表 (logic_id) → physical_id → devPhyId
      │         │                              │
      │         │              ┌───────────────┘
      │         │              │
      │         │              ▼
      │         │    RaGetDevEidInfoList(devPhyId)
      │         │              │
      │         │              ▼
      │         │    localEidSet (硬件真实EID)
      │         │
      │    (同时 mpiRankid 作为 rank 参数传入 hcomm)
      │         │
      │         ▼
      │    myRank = mpiRankid
      │         │
      │         ▼
      │    rankGraph->GetPeer(myRank)
      │         │
      │         ▼
      │    peers_[myRank] ← ranktable.json 中 rank_id==myRank 的条目
      │         │
      │         ▼
      │    peer 接口 EID (ranktable声称的EID)
      │         │
      └─────────┘
                │
                ▼
         localEidSet.count(interface EID) == 0 ?
              │
         不匹配 → 报错
```

---

## 8 修改点分析

### 8.1 被修改的代码

```
文件: src/topo/topo_ascend_cluster_parser.cc
函数: AscendClusterTopoParser::CreateRankTableFile
行号: 449-456

修改前 (450行):
    for (uint32_t deviceIdx = 0; deviceIdx < serverMeta.size(); ++deviceIdx) {
        uint32_t logicDevId = rankId % serverMeta.size();   // ← 旧代码
        PhyDeviceId phyDevId = serverMeta[logicDevId];
        ...
        rankId++;
    }

修改后 (451行):
    for (uint32_t deviceIdx = 0; deviceIdx < serverMeta.size(); ++deviceIdx) {
        uint32_t logicDevId = deviceIdx;                    // ← 新代码
        PhyDeviceId phyDevId = serverMeta[logicDevId];
        ...
        rankId++;
    }
```

### 8.2 修改的影响

该修改改变了 `ranktable.json` 中每个 `rank_id` 对应的 `device_id` 和 `local_id`：

| rank_id | 旧: `rankId % D` 对应的 device_idx | 新: `deviceIdx` | 相同? |
|---------|-----------------------------------|-----------------|-------|
| 0 (rack0) | 0 | 0 | 相同 |
| 1 (rack1) | 1 | 0 | **不同** |
| 2 (rack1) | 0 | 1 | **不同** |

当 server 设备数均为 1 或仅有单个 server 时，两个公式等价（`rankId % D` 在首 server 内与 `deviceIdx` 相同），修改不会产生问题。但多 server 且非首 server 有多 device 时，二者不同。

### 8.3 BuildRankEntry 中 device_id 与 local_id 的设置

```cpp
// topo_ascend_cluster_parser.cc:413-416
json AscendClusterTopoParser::BuildRankEntry(const Server &server, int srcDevPhyId, ...)
{
    json rankEntry;
    rankEntry["device_id"] = srcDevPhyId;    // ← 设为物理设备ID
    rankEntry["local_id"] = srcDevPhyId;     // ← 也设为物理设备ID
    rankEntry["rank_id"] = rankId;
    ...
}
```

由于本 simulator 中 `logic_id = physical_id`，`device_id` 和 `local_id` 数值相同，`BuildRankEntry` 本身没有问题。真正的问题在于传入 `BuildRankEntry` 的 `phyDevId` 取自 `serverMeta[logicDevId]`，而 `logicDevId` 的计算方式（450/451 行）决定了哪个物理设备被分配给该 rank。

---

## 9 修复方案

### 9.1 核心修复

**恢复 450 行的公式**，使 ranktable 的 rank -> device 映射与应用 `aclrtSetDevice(rank % D)` 一致：

```cpp
uint32_t logicDevId = rankId % serverMeta.size();
```

### 9.2 修复后的数据流验证

以测试拓扑（rack0=1卡, rack1=2卡）验证：

```
第一步: 应用选设备 (aclrtSetDevice)
  rank 0: 0 % 1 = 0 → rack0 logicDevId=0 → phyDevId=0  (IP=192.1.0.8)
  rank 1: 1 % 2 = 1 → rack1 logicDevId=1 → phyDevId=1  (IP=192.2.0.18)
  rank 2: 2 % 2 = 0 → rack1 logicDevId=0 → phyDevId=0  (IP=192.2.0.1)

第二步: ranktable.json 生成 (rankId % D)
  rank 0: device_id=0, local_id=0  (phyDev0)
  rank 1: device_id=1, local_id=1  (phyDev1)
  rank 2: device_id=0, local_id=0  (phyDev0)

第三步: CheckRankGraphAddrs 校验
  rank 1: devPhyId=1 → localEidSet={192.2.0.11, 192.2.0.18}
          peer(rank_id=1) → device_id=1 → interface EID={192.2.0.18}
          192.2.0.18 in localEidSet → 匹配 ✓
  rank 2: devPhyId=0 → localEidSet={192.2.0.1, 192.2.0.8}
          peer(rank_id=2) → device_id=0 → interface EID={192.2.0.1}
          192.2.0.1 in localEidSet → 匹配 ✓
```

所有映射链路一致，校验通过。

---

## 10 关键代码位置索引

### 10.1 本项目 (CheckerL2_8620) 代码

| 文件 | 行号 | 函数/说明 |
|------|------|---------|
| `src/topo/topo_ascend_cluster_parser.cc` | 424-477 | `CreateRankTableFile` - 生成 ranktable.json |
| `src/topo/topo_ascend_cluster_parser.cc` | 449-456 | **修改点** - logicDevId 计算方式 (450 旧/451 新) |
| `src/topo/topo_ascend_cluster_parser.cc` | 409-422 | `BuildRankEntry` - 设置 device_id/local_id/rank_id |
| `src/topo/topo_ascend_cluster_parser.cc` | 200-272 | `InitDynamicModelData` - 初始化 Device/Rank 表 |
| `src/topo/topo_ascend_cluster_parser.cc` | 202, 234 | 注释说明应用用 `aclrtSetDevice(rank % deviceCount)` |
| `src/proxy/level2/aclrt_device_stub.cc` | 69-128 | `aclrtSetDevice` - 应用选设备的 stub 实现 |
| `src/proxy/level2/aclrt_device_stub.cc` | 87-88 | 按 `(serverKey, logic_id)` 查 Device 表 |
| `src/proxy/level2/aclrt_device_stub.cc` | 110-122 | 创建 Rank 表记录，打印 logicDevId/phyDevId |
| `src/proxy/level2/hccl_comm_stub.cc` | 83-111 | `SimGetDeviceComm` - 调用 `aclrtSetDevice` |
| `src/proxy/level2/hccl_comm_stub.cc` | 130-131 | `HcclCommInitAll` 中调用 `SimGetDeviceComm` |
| `src/proxy/level2/hccp_ccu_stub.cc` | 528-541 | `GetAllUsedEndPoint` - 按 phyDevId 查 EndPoint |
| `src/proxy/level2/hccp_ccu_stub.cc` | 554-587 | `RaGetDevEidInfoList` - 按 phyId 返回 EID 列表 |
| `src/proxy/level2/hccp_ccu_stub.cc` | 573 | `chipId = info.phyId` (有 TODO 注释，但与本问题无关) |

### 10.2 hcomm 业务代码

| 文件 | 行号 | 函数/说明 |
|------|------|---------|
| `communicator_impl.cc` | 1256-1269 | `InitCommonData` - 设置 myRank, devLogicId, devPhyId |
| `communicator_impl.cc` | 1291-1338 | `CheckRankGraphAddrs` - EID 校验逻辑 |
| `communicator_impl.cc` | 1306-1316 | 路径 A: 按 `deviceId==devPhyId` 查并调 `RaGetDevEidInfoList` |
| `communicator_impl.cc` | 1322-1328 | 路径 B: `GetPeer(myRank)` 取 interface EID 并比对 |
| `communicator_impl.cc` | 1323 | ZHF-DEBUG 日志: `peer deviceId` vs `devPhyId` |
| `communicator_impl.cc` | 1353-1371 | `InitRankGraph` - 调 bridge 构建 rankGraph |
| `communicator_impl.cc` | 1364 | `bridge->buildFromString(ranktableM, topoPath, myRank, ...)` |
| `op_base_v2.cc` | 455-514 | `HcclCommInitClusterInfoConfigV2` - 入口函数 |
| `op_base_v2.cc` | 131-152 | `CreateCommConfig` - 构造 CommParams (myRank=rank) |
| `op_base_v2.cc` | 311-322 | `ParseJsonAndCreateComm` - 解析 JSON 并创建通信域 |
| `rank_graph_builder.cc` | 315-353 | `BuildFromRankTable` - 遍历 ranktable 构建 peers_ |
| `rank_graph_builder.cc` | 323-328 | `peers_.emplace(rankId, peer)` - 以 hcclRankid 为 key |
| `rank_graph_builder.cc` | 22, 46 | `Build` 方法接收 `myRank` 参数 |

### 10.3 应用示例代码

| 文件 | 行号 | 说明 |
|------|------|------|
| `hcomm/examples/01_communicators/02_one_device_per_process_rank_table/main.cc` | 101-105 | MPI 初始化，获取 procRank |
| 同上 | 106, 111 | `devId = procRank`, `aclrtSetDevice(devId)` |
| 同上 | 130 | `HcclCommInitClusterInfoConfig(rankTableFile, devId, ...)` |

### 10.4 关键日志文件

| 文件 | 关键行 | 说明 |
|------|--------|------|
| `hccl_vm_install/bin/log.txt` | 79, 83, 85 | `aclrtSetDevice` 打印的 rankId/logicDevId/phyDevId |
| 同上 | 187, 219, 299 | `InitCommonData` 打印的 devLogicId/devPhyId |
| 同上 | 1023, 1062, 1094 | ZHF-DEBUG: `peer deviceId` vs `devPhyId` |
| 同上 | 1267, 1309 | `CheckRankGraphAddrs` 报错信息 |

### 10.5 ranktable.json 文件

| 文件 | 说明 |
|------|------|
| `hccl_vm_install/data/ranktable.json` | 当前运行生成的（新代码，有问题） |
| `hccl_vm_install/archive/20260815_094326/data/ranktable.json` | 旧代码生成的归档（正确） |
| `hccl_vm_install/archive/20260815_094636/data/ranktable.json` | 另一次运行的归档 |

---

## 11 排除项

以下因素经分析确认**不是**本次问题的根因：

### 11.1 RaGetDevEidInfoList 中 chipId = info.phyId

```cpp
// hccp_ccu_stub.cc:573
infoList[idx].chipId = info.phyId; // todo: 单server, logic id与rank id相等，但多server此处有问题。
```

该处确实有潜在隐患（TODO 注释），但 `CheckRankGraphAddrs` **完全不读取 chipId 字段**，EID 比对仅按 IP/EID 原值匹配（`localEidSet.count(interface->GetAddr().GetEid())`）。因此 chipId 的赋值问题与本次报错无关。

### 11.2 BuildRankEntry 中 device_id 与 local_id 同时设为 phyDevId

```cpp
// topo_ascend_cluster_parser.cc:414-415
rankEntry["device_id"] = srcDevPhyId;
rankEntry["local_id"] = srcDevPhyId;
```

由于本 simulator 中 `logic_id = physical_id`（数值相等），二者设为相同值没有问题。问题在于传入的 `srcDevPhyId`（即 `phyDevId`）取自 `serverMeta[logicDevId]`，而 `logicDevId` 的计算方式（450/451 行）才是根因。

### 11.3 hcomm 是否按 server 过滤 peer

`CheckRankGraphAddrs` 不遍历所有 peer，也不按 server 过滤。它只取 `GetPeer(myRank)` 返回的单一本地 peer，检查该 peer 的接口 EID。不存在跨 server peer 比对的问题。

---

## 12 总结

| 维度 | 内容 |
|------|------|
| 问题本质 | 两条 EID 获取路径对 rankId -> device 的映射公式不一致 |
| 路径 A | 硬件真实 EID，devPhyId 由应用 `aclrtSetDevice(mpiRankid % D)` 决定，不经过 ranktable |
| 路径 B | ranktable 声称的 EID，通过 `GetPeer(myRank=mpiRankid)` 从 ranktable 查得 |
| 冲突条件 | server 设备数 > 1 且非首 server 时，`rankId % D != deviceIdx` |
| 根因 | 451 行将 `rankId % D` 改为 `deviceIdx`，破坏了路径 B 与路径 A 的一致性 |
| myRank 归属 | mpiRankid（来自 `MPI_Comm_rank`，经 `HcclCommInitClusterInfoConfig` 的 rank 参数传入） |
| 修复方法 | 恢复 450 行：`uint32_t logicDevId = rankId % serverMeta.size();` |
