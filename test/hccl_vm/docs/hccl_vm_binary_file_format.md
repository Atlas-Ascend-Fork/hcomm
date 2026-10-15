# HCCL VM 二进制流文件格式规范

## 文档信息

| 项目     | 内容                           |
|----------|--------------------------------|
| 版本     | v2.0                           |
| 更新日期 | 2026-09-17                     |
| 适用场景 | HCCL VM 仿真数据导出与读取     |
| 作者     | HCCL VM Team                   |

> **v2.0 变更说明**：综合数据文件（`*_hcclvm_syn_data.bin`）**version 保持 1，文件头不做版本变更**
> （本特性不做新旧文件兼容性区分）。`ChannelData` 二进制布局与历史版本**一致（未新增字段，176 字节）**，
> 但 AICPU 模式 dump **语义**变更以支持多 channel（`HCCL_UB_MULTI_CHANNEL_NUM`）特性：
> jetty 精确归属到（本端 eid, 对端 eid）链路 + **每 eid 对一条聚合记录**
> （`jettyNum` 为该 eid 对的 jetty 个数，`jettyId[0..jettyNum)` 携带全部本端 jetty，
> 读取方需在数组内查找 `task.jettyId`，不能再假定 `jettyNum` 恒为 1 或直接取 `jettyId[0]`）。
> 任务/微码/flag 文件格式不变。

---

## 1. 概述

`DumpData` 接口用于将 HCCL VM 仿真运行时的数据导出为二进制流文件，共生成 **3 种文件**：

| 序号 | 文件名格式                        | 魔数         | 用途                           |
|:----:|-----------------------------------|--------------|--------------------------------|
|  1   | `{dataId}_hcclvm_syn_data.bin`    | `0x48564D44` | 综合数据（模型、通道、内存布局）|
|  2   | `{dataId}_hcclvm_instr_data.bin`  | `0x434D4349` | 微码指令数据                   |
|  3   | `{dataId}_hcclvm_task_data.bin`   | `0x48565444` | 任务元数据                     |

> **说明**：`{dataId}` 为时间戳格式的唯一标识，例如 `20250115_143052_3847`。

---

## 2. 通用文件头

所有二进制文件均以 **20 字节** 的文件头开始。

### 2.1 文件头结构

| 偏移量 | 字节数 | 类型    | 字段名      | 说明                       |
|:------:|:------:|---------|-------------|----------------------------|
|   0    |   4    | uint32  | magic       | 魔数，标识文件类型         |
|   4    |   2    | uint16  | version     | 版本号（当前为 1）            |
|   6    |   2    | uint16  | header_size | 文件头大小（20 字节）      |
|   8    |   4    | uint32  | flags       | 标志位（保留）             |
|   12   |   4    | uint32  | count       | 数据条目数量               |
|   16   |   4    | uint32  | checksum    | 校验和（保留）             |

### 2.2 C/C++ 结构定义

```cpp
#pragma pack(push, 1)
struct FileHeader {
    uint32_t magic;        // 魔数
    uint16_t version;      // 版本号
    uint16_t header_size;  // 文件头大小
    uint32_t flags;        // 标志位
    uint32_t count;        // 数据条目数量
    uint32_t checksum;     // 校验和
};
#pragma pack(pop)
```

---

## 3. 综合数据文件 (`*_hcclvm_syn_data.bin`)

### 3.1 整体结构

| 序号 | 数据块                           | 大小     | 说明                                         |
|:----:|----------------------------------|----------|----------------------------------------------|
|  1   | FileHeader                       | 20 Bytes | 文件头                                       |
|  2   | ModelInfo                        | 变长     | 模型信息（含 ModelInfoCommInner、VDataDesTag、All2AllDataDesTag） |
|  3   | ChannelInfo（CCU/AICPU 共用）    | 变长     | AIV 模式无此数据块                           |
|  4   | MemLayoutInfo                    | 变长     | 内存布局信息                                 |

**op_expansion_mode 取值说明：**

| 值 | 模式    | 使用的数据结构 |
|:--:|---------|----------------|
|  0 | CCU     | ChannelInfo    |
|  1 | AICPU   | ChannelInfo（多 channel 语义见 3.3.3） |

### 3.2 ModelInfo 结构

#### 3.2.1 ModelInfoCommInner

**大小**：36 字节。

| 偏移量 | 字节数 | 类型    | 字段名            | 说明             |
|:------:|:------:|---------|-------------------|------------------|
|   0    |   4    | uint32  | src_rank          | 源 Rank ID       |
|   4    |   4    | uint32  | dst_rank          | 目标 Rank ID     |
|   8    |   4    | uint32  | root              | 根节点 Rank      |
|   12   |   4    | uint32  | rank_size         | Rank 总数        |
|   16   |   2    | uint16  | chip_type         | 芯片类型         |
|   18   |   2    | uint16  | op_type           | 操作类型         |
|   20   |   2    | uint16  | reduce_op         | 归约操作类型     |
|   22   |   2    | uint16  | data_type         | 数据类型         |
|   24   |   8    | uint64  | data_count        | 数据元素个数     |
|   32   |   4    | uint32  | op_expansion_mode | 扩展模式（见枚举）|
|   36   |   8    | uint64  | ccu0_resource_base_addr | die0 ccu资源基址 |
|   44   |   8    | uint64  | ccu1_resource_base_addr | die1 ccu资源基址 |

#### 3.2.2 VDataDesTag

**用途**：ReduceScatterV / AllGatherV 操作。

**大小**：变长。

| 偏移量  | 字节数     | 类型     | 字段名  | 说明               |
|:-------:|:----------:|----------|---------|--------------------|
|    0    |     2      | uint16   | dataType| 数据类型           |
|    2    |     4      | uint32   | count   | Rank 数量          |
|    6    | 8 × count  | uint64[] | displs  | 各 Rank 数据偏移量 |
|  6+8×n  | 8 × count  | uint64[] | counts  | 各 Rank 数据大小   |

#### 3.2.3 All2AllDataDesTag

**用途**：All2All / All2AllV 操作。

**大小**：变长。

| 偏移量 | 字节数     | 类型     | 字段名          | 说明                   |
|:------:|:----------:|----------|-----------------|------------------------|
|   0    |     2      | uint16   | sendType        | 发送数据类型           |
|   2    |     2      | uint16   | recvType        | 接收数据类型           |
|   4    |     8      | uint64   | sendCount       | 发送数据量             |
|   12   |     8      | uint64   | recvCount       | 接收数据量             |
|   20   |     4      | uint32   | count           | 矩阵大小 (= rankSize²) |
|   24   | 8 × count  | uint64[] | sendCountMatrix | 发送矩阵               |

### 3.3 ChannelInfo 结构

**适用条件**：`op_expansion_mode = 0`（CCU 模式）或 `1`（AICPU 模式）——两种模式均使用 ChannelData 定长结构；AIV 模式不写。

#### 3.3.1 ChannelInfo 头部

| 偏移量 | 字节数 | 类型          | 字段名 | 说明         |
|:------:|:------:|---------------|--------|--------------|
|   0    |   4    | uint32        | count  | 通道数量     |
|   4    | 变长   | ChannelData[] | data   | 通道数据数组 |

#### 3.3.2 ChannelData

**大小**：176 字节（布局与历史版本一致）。

| 偏移量 | 字节数 | 类型       | 字段名       | 说明          |
|:------:|:------:|------------|-----------|---------------|
|   0    |   2    | uint16     | channelId | CCU 模式: 通道 ID；AICPU 模式: EndPointPair 主键（eid 对标识） |
|   2    |   1    | uint8      | srcDieId  | 源 Die ID     |
|   3    |   1    | uint8      | dstDieId  | 目标 Die ID   |
|   4    |   4    | uint32     | srcRank   | 源 Rank ID    |
|   8    |   4    | uint32     | dstRank   | 目标 Rank ID  |
|   12   |   16   | uint8[16]  | leid      | 本端 EID      |
|   28   |   16   | uint8[16]  | reid      | 远端 EID      |
|   44   |   2    | uint16     | protocol  | 协议类型      |
|   46   |   2    | uint16     | jettyNum  | Jetty 数量    |
|   48   |   128  | uint32[32] | jettyId   | Jetty ID 数组 |

#### 3.3.3 多 channel 语义（AICPU 模式）

`HCCL_UB_MULTI_CHANNEL_NUM=N` 特性下，同一对 rank（同一对 eid）之间存在 N 条并行 channel，
AICPU 模式 dump 规则：

1. **每 eid 对一条聚合 ChannelData 记录**：同一 `channelId`（eid 对）的 N 条 channel 输出 **1 条**记录，
   `jettyNum=N`，`jettyId[0..jettyNum)` 为该 eid 对上**全部本端（发送侧）jetty**，
   按**本端 jetty 创建序**排列（数组下标即 pair 内 channel 序号）；
2. 读取方将任务元数据中的 `task.jettyId` 在 `jettyId[0..jettyNum)` **数组内查找**，
   即得任务经由哪个 jetty 从本端哪个 eid（`leid`）发往对端哪个 eid（`reid`）；
3. `jettyId[0..jettyNum)` **精确归属**到该 eid 对（由 RA 层 `RaCtxQpImport` 回填归属信息），
   不含该本端 eid 上通往其它对端的 jetty。

CCU 模式维持"每 channel 一条记录、`jettyId[]` 携带该 channel 的 jetty 列表"的既有语义。

### 3.4 JettyInfo 结构（未使用，保留历史说明）

`JettyData` 为早期设计中 AICPU 模式的独立结构，**当前实现未使用**：AICPU 模式实际同样写
ChannelData（见 3.3）。本节保留仅作历史格式参考，读取方不应按 JettyData 解析 AICPU 数据。

#### 3.4.1 JettyInfo 头部

| 偏移量 | 字节数 | 类型       | 字段名 | 说明         |
|:------:|:------:|------------|--------|--------------|
|   0    |   4    | uint32     | count  | Jetty 数量   |
|   4    | 变长   | JettyData[]| data   | Jetty 数据数组|

#### 3.4.2 JettyData

**大小**：48 字节。

| 偏移量 | 字节数 | 类型      | 字段名   | 说明         |
|:------:|:------:|-----------|----------|--------------|
|   0    |   4    | uint32    | jettyId  | Jetty ID     |
|   4    |   1    | uint8     | srcDieId | 源 Die ID    |
|   5    |   1    | uint8     | dstDieId | 目标 Die ID  |
|   6    |   2    | uint16    | protocol | 协议类型     |
|   8    |   4    | uint32    | srcRank  | 源 Rank ID   |
|   12   |   4    | uint32    | dstRank  | 目标 Rank ID |
|   16   |  16    | uint8[16] | leid     | 本端 EID     |
|   32   |  16    | uint8[16] | reid     | 远端 EID     |

### 3.5 MemLayoutInfo 结构

#### 3.5.1 MemLayoutInfo 头部

| 偏移量 | 字节数 | 类型            | 字段名 | 说明         |
|:------:|:------:|-----------------|--------|--------------|
|   0    |   4    | uint32          | count  | 内存块数量   |
|   4    | 变长   | MemLayoutData[] | data   | 内存布局数组 |

#### 3.5.2 MemLayoutData

**大小**：32 字节。

| 偏移量 | 字节数 | 类型   | 字段名       | 说明               |
|:------:|:------:|--------|--------------|--------------------|
|   0    |   4    | uint32 | rank_id      | Rank ID            |
|   4    |   1    | uint8  | buffer_type  | 缓冲区类型（见枚举）|
|   5    |   1    | uint8  | reserved     | 保留               |
|   6    |   8    | uint64 | start_addr   | 物理起始地址       |
|   14   |   8    | uint64 | size         | 块大小             |
|   22   |   8    | uint64 | global_offset| 该类型总偏移       |

---

## 4. 微码指令文件 (`*_hcclvm_instr_data.bin`)

### 4.1 整体结构

| 序号 | 数据块                     | 大小     | 说明                           |
|:----:|----------------------------|----------|--------------------------------|
|  1   | FileHeader                 | 20 Bytes | 文件头                         |
|  2   | MicrocodeInstrInner × count| 变长     | 微码指令数据（含 Desc 和 Instr）|

### 4.2 MicrocodeInstrDesc

**大小**：8 字节。

| 偏移量 | 字节数 | 类型   | 字段名   | 说明     |
|:------:|:------:|--------|----------|----------|
|   0    |   4    | uint32 | rank_id  | Rank ID  |
|   4    |   1    | uint8  | die_id   | Die ID   |
|   5    |   1    | uint8  | reserved | 保留     |
|   6    |   2    | uint16 | count    | 指令数量 |

### 4.3 微码指令数据

每个 MicrocodeInstrInner 包含：

| 数据块                  | 大小              | 说明             |
|-------------------------|-------------------|------------------|
| MicrocodeInstrDesc      | 8 Bytes           | 指令描述符       |
| CcuInstr[desc.count]    | 32 × count Bytes  | 微码指令数组     |

---

## 5. 任务元数据文件 (`*_hcclvm_task_data.bin`)

### 5.1 整体结构

| 序号 | 数据块                  | 大小     | 说明     |
|:----:|-------------------------|----------|----------|
|  1   | FileHeader              | 20 Bytes | 文件头   |
|  2   | HcclTaskMetaData × count| 变长     | 任务列表 |

### 5.2 HcclTaskMetaData 结构

**大小**：188 字节（pack(1)，含联合体；偏移 1~3 为 `HcclTaskProfInfo` 子结构，checker/runner 作为不透明整体透传）

| 偏移量 | 字节数 | 类型            | 字段名   | 说明                                       |
|:------:|:------:|:----------------|----------|--------------------------------------------|
|   0    |   1    | int8            | taskType | 任务类型（见枚举）                         |
|   1    |   3    | HcclTaskProfInfo| profInfo | profiling 对账信息子结构，见下表          |
|   4    |   8    | uint64          | commId   | 通信域 ID                                  |
|   12   |   4    | uint32          | rankId   | 任务所属 Rank ID                           |
|   16   |   8    | uint64          | deviceId | 任务所属 Device ID                         |
|   24   |   8    | uint64          | streamId | Stream ID                                  |
|   32   |   4    | uint32          | jettyId  | Jetty ID                                   |
|   36   |   16   | uint8[16]       | rmEid    | 对端 EID（WQE 原始字节序）                 |
|   52   |   16   | uint8[16]       | lcEid    | 本端 EID（WQE 字节序，见下方说明）         |
|   68   |   120  | union           | taskData | 任务数据（根据类型解析）                   |

`HcclTaskProfInfo` 子结构（偏移 1~3，定义于 `include/sim_profiling_defs.h`，单源管理）：

| 偏移量 | 字节数 | 类型               | 字段名        | 说明                        |
|:------:|:------:|:-------------------|---------------|-----------------------------|
|   1    |   1    | uint8              | profTaskType  | 对齐 hcomm TaskParamType    |
|   2    |   1    | uint8              | linkType      | 对齐 hcomm DfxLinkType      |
|   3    |   1    | uint8              | transportType | 对齐 hcomm DfxTransportType |

**EID 字段说明**：

- `rmEid` 与 `lcEid` 统一采用 **ranktable.json 十六进制串字节序**（与 EndPoint.eid /
  RankEidMap 的入库序一致），可与 DB 表 / ranktable 的 EID 串直接比对，二者亦可直接互比。
- `rmEid` 来源：WQE 中的 `rmtEid` 为 `uint32[4]` 小端字存储（内存 16 字节与 ranktable
  串整体相反），拷入任务时做整 16 字节翻转。
- `lcEid` 来源：由 `jettyId → RaJetty → RaContext → EndPoint.eid` 查询本端 EID 直拷，
  天然为该序。
- `lcEid` 仅设备侧 SQE 解析（`device_sqe_parse_stub.cc`）生成的带 jettyId 的任务填充；
  其余任务保持全 0。
- **rank 约定（estimate 采集）**：taskData 的 `srcDeviceId`/`dstDeviceId` 字段承载
  通信域 rank（原 `srcRankId`/`dstRankId` 字段已回退删除，值合并入 deviceId 字段）；
  为 `UINT32_MAX`（0xFFFFFFFF）时表示未解析到对应 rank（如非 ranktable/EID 模式），
  0 为合法 rank 值。normal 模式（checker 链路）下 deviceId 字段仍为真实 Device ID，
  两种语义按运行模式隔离（estimate 模式禁用 checker）。

### 5.3 taskData 联合体

联合体大小 120 字节（取最大成员 CcuTask）。

#### 5.3.1 TransMemTask

**适用任务类型**：`MEM_CPY (3)`。

**大小**：33 字节。

| 偏移量 | 字节数 | 类型   | 字段名      | 说明                                  |
|:------:|:------:|--------|-------------|---------------------------------------|
|   0    |   4    | uint32 | srcDeviceId | 源端点（estimate: rank / normal: Device ID） |
|   4    |   8    | uint64 | srcOffset   | 源偏移地址                            |
|   12   |   4    | uint32 | dstDeviceId | 目标端点（estimate: rank / normal: Device ID） |
|   16   |   8    | uint64 | dstOffset   | 目标偏移地址                          |
|   24   |   8    | uint64 | len         | 数据长度                              |
|   32   |   1    | uint8  | protocol    | 协议类型                              |

#### 5.3.2 ReduceTask

**适用任务类型**：`REDUCE (2)`。

**大小**：36 字节。

| 偏移量 | 字节数 | 类型   | 字段名      | 说明                                  |
|:------:|:------:|--------|-------------|---------------------------------------|
|   0    |   4    | uint32 | srcDeviceId | 源端点（estimate: rank / normal: Device ID） |
|   4    |   8    | uint64 | srcOffset   | 源偏移地址                            |
|   12   |   4    | uint32 | dstDeviceId | 目标端点（estimate: rank / normal: Device ID） |
|   16   |   8    | uint64 | dstOffset   | 目标偏移地址                          |
|   24   |   8    | uint64 | dataCount   | 数据元素个数                          |
|   32   |   1    | uint8  | dataType    | 数据类型                              |
|   33   |   1    | uint8  | reduceOp    | 归约操作类型                          |
|   34   |   1    | uint8  | protocol    | 协议类型                              |
|   35   |   1    | uint8  | resv        | 保留                                  |

#### 5.3.3 NotifyTask

**适用任务类型**：`NOTIFY_WAIT (0)` / `NOTIFY_RECORD (1)`。

**大小**：18 字节。

| 偏移量 | 字节数 | 类型   | 字段名      | 说明                                        |
|:------:|:------:|--------|-------------|---------------------------------------------|
|   0    |   4    | uint32 | srcDeviceId | 记录方端点（estimate: rank / normal: Device ID） |
|   4    |   8    | uint64 | notifyId    | 通知 ID                                     |
|   12   |   4    | uint32 | dstDeviceId | 等待方端点（estimate: rank / normal: Device ID） |
|   16   |   1    | uint8  | notifyCount | 通知计数                                    |
|   17   |   1    | uint8  | protocol    | 协议类型                                    |

---

## 6. 枚举定义

### 6.1 任务类型 (HccLTaskMetaType)

| 值 | 名称          | 说明       |
|:--:|---------------|------------|
|  0 | NOTIFY_WAIT   | 等待通知   |
|  1 | NOTIFY_RECORD | 记录通知   |
|  2 | REDUCE        | 归约操作   |
|  3 | MEM_CPY       | 内存拷贝   |
|  4 | CCU_GRAPH     | CCU 图执行 |
|  5 | AIV_GRAPH     | AIV 图执行 |
|  6 | EVENT_WAIT    | 事件等待   |
|  7 | EVENT_RECORD  | 事件记录   |

### 6.2 协议类型 (ProtocolType)

| 值 | 名称    | 说明                         |
|:--:|---------|------------------------------|
|  0 | HCCS    | 高速片间通信                 |
|  1 | ROCE    | RDMA over Converged Ethernet |
|  2 | PCIE    | PCIe 通信                    |
|  3 | SIO     | Socket I/O                   |
|  4 | UBC_CTP | UBC CTP 协议                 |
|  5 | UBC_TP  | UBC TP 协议                  |
|  6 | UB_MEM  | UB 内存协议                  |

### 6.3 缓冲区类型 (BufferType)

| 值 | 名称     | 说明       |
|:--:|----------|------------|
|  0 | INPUT    | 输入缓冲区 |
|  1 | OUTPUT   | 输出缓冲区 |
|  2 | CCL      | 通信缓冲区 |
|  3 | RESERVED | 保留       |

### 6.4 扩展模式 (SimOpExpansionMode)

| 值 | 名称                            | 说明                      |
|:--:|---------------------------------|---------------------------|
|  0 | SIM_OP_EXPANSION_MODE_CCU       | CCU 模式，使用 ChannelInfo |
|  1 | SIM_OP_EXPANSION_MODE_AICPU     | AICPU 模式，使用 ChannelInfo |

---

## 7. C/C++ 读取示例

### 7.1 结构定义

```cpp
#include <cstdint>
#include <vector>
#include <cstring>

#pragma pack(push, 1)

// ============================================================================
// 文件头
// ============================================================================
struct FileHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint32_t flags;
    uint32_t count;
    uint32_t checksum;
};

// ============================================================================
// 模型信息
// ============================================================================
struct ModelInfoCommInner {
    uint32_t src_rank;
    uint32_t dst_rank;
    uint32_t root;
    uint32_t rank_size;
    uint16_t chip_type;
    uint16_t op_type;
    uint16_t reduce_op;
    uint16_t data_type;
    uint64_t data_count;
    uint32_t op_expansion_mode;
    uint64_t ccu0_resource_base_addr;
    uint64_t ccu1_resource_base_addr;
};

// ============================================================================
// 通道数据 (CCU / AICPU 模式共用)
// ============================================================================
struct ChannelData {
    uint16_t channelId;      // CCU: 通道ID; AICPU: EndPointPair主键(eid对标识)
    uint8_t  srcDieId;
    uint8_t  dstDieId;
    uint32_t srcRank;
    uint32_t dstRank;
    uint8_t  leid[16];
    uint8_t  reid[16];
    uint16_t protocol;
    uint16_t jettyNum;       // AICPU: 该eid对的本端jetty个数(多channel时可大于1); CCU: 该channel的jetty个数
    uint32_t jettyId[32];    // AICPU: 该eid对全部本端(发送)jetty, 按创建序排列, task.jettyId在数组内查找
};

// ============================================================================
// Jetty 数据（未使用的历史结构；AICPU 实际使用 ChannelData，见上方）
// ============================================================================
struct JettyData {
    uint32_t jettyId;
    uint8_t  srcDieId;
    uint8_t  dstDieId;
    uint16_t protocol;
    uint32_t srcRank;
    uint32_t dstRank;
    uint8_t  leid[16];
    uint8_t  reid[16];
};

// ============================================================================
// 内存布局
// ============================================================================
struct MemLayoutData {
    uint32_t rank_id;
    uint8_t  buffer_type;
    uint8_t  reserved;
    uint64_t start_addr;
    uint64_t size;
    uint64_t global_offset;
};

// ============================================================================
// 任务类型枚举
// ============================================================================
enum class HccLTaskMetaType : int8_t {
    NOTIFY_WAIT        = 0,
    NOTIFY_RECORD      = 1,
    REDUCE             = 2,
    MEM_CPY            = 3,
    CCU_GRAPH          = 4,
    AIV_GRAPH          = 5,
    SYNC_STREAM        = 6,
    WRITE_WITH_NOTIFY  = 7,
    INVALID            = 8
};

// ============================================================================
// profiling 任务类型枚举值域与 hcomm TaskParamType 一一对齐
// ============================================================================
enum class ProfTaskParamType : uint8_t {
    TASK_SDMA                    = 0,   // Memcpy
    TASK_RDMA                    = 1,   // RDMASend
    TASK_REDUCE_INLINE           = 2,   // Reduce_Inline
    TASK_REDUCE_TBE              = 3,   // Reduce_TBE
    TASK_NOTIFY_RECORD           = 4,   // Notify_Record
    TASK_NOTIFY_WAIT             = 5,   // Notify_Wait
    TASK_SEND_NOTIFY             = 6,   // Send_Notify
    TASK_SEND_PAYLOAD            = 7,   // Send_Payload
    TASK_WRITE_WITH_NOTIFY       = 8,   // Write_With_Notify
    TASK_WRITE_REDUCE_WITH_NOTIFY = 9,  // Reduce_With_Notify
    TASK_CCU                     = 10,  // Ccu
    TASK_AICPU_KERNEL            = 11,  // AicpuKernel
    TASK_AICPU_REDUCE            = 12,  // Aicpu_Reduce
    TASK_AIV                     = 13,  // AivKernel
    TASK_UB_INLINE_WRITE         = 14,  // Ub_Inline_Write
    TASK_UB_REDUCE_INLINE        = 15,  // Ub_Reduce_Inline
    TASK_UB                      = 16,  // Ub_Write_Or_Read
    TASK_DPU_KERNEL              = 17,  // DpuKernel
    TASK_DPU_THREAD_FENCE        = 18,  // Dpu_ThreadFence
    TASK_DPU_CHANNEL_FENCE       = 19,  // Dpu_ChannelFence
    TASK_DPU_INLINE_WRITE        = 20,  // Dpu_Notify_Record
    TASK_DPU_NOTIFY_WAIT         = 21,  // Dpu_Notify_Wait
    TASK_DPU_WRITE_WITH_NOTIFY   = 22,  // Dpu_Write_With_Notify
    UNKNOWN                      = 0xFF
};

// ============================================================================
// profiling 链路类型枚举值域与 hcomm DfxLinkType 一一对齐
// ============================================================================
enum class ProfLinkType : uint8_t {
    LINK_ONCHIP        = 0,   // 片上/本地
    LINK_HCCS          = 1,   // HCCS
    LINK_PCIE          = 2,   // PCIe(跨片 SDMA 通道)
    LINK_ROCE          = 3,   // RoCE
    LINK_SIO           = 4,   // SIO
    LINK_HCCS_SW       = 5,   // HCCS_SW
    LINK_STANDARD_ROCE = 6,   // 标准RoCE
    LINK_UB            = 7,   // UB
    LINK_UBoE          = 8,   // UB over Ethernet
    LINK_RESERVED      = 9,   // 预留
    UNKNOWN            = 0xFF
};

// ============================================================================
// profiling 传输类型枚举值域与 hcomm DfxTransportType/SimpleTaskType 一一对齐
// ============================================================================
enum class ProfTransportType : uint8_t {
    TRANSPORT_SDMA   = 0,   // 跨片 SDMA 通道搬运
    TRANSPORT_RDMA   = 1,   // RDMA
    TRANSPORT_LOCAL  = 2,   // 本地搬运/本地算子
    TRANSPORT_UB     = 3,   // UB 传输
    TRANSPORT_ROCE   = 4,   // RoCE 传输
    UNKNOWN          = 0xFF
};

// ============================================================================
// 任务数据联合体 (联合体大小 120 字节, 取最大成员 CcuTask)
// ============================================================================
struct TransMemTask {
    uint32_t srcDeviceId;
    uint64_t srcOffset;
    uint32_t dstDeviceId;
    uint64_t dstOffset;
    uint64_t len;
    uint8_t  protocol;
};

struct ReduceTask {
    uint32_t srcDeviceId;
    uint64_t srcOffset;
    uint32_t dstDeviceId;
    uint64_t dstOffset;
    uint64_t dataCount;
    uint8_t  dataType;
    uint8_t  reduceOp;
    uint8_t  protocol;
    uint8_t  resv;
};

struct NotifyTask {
    uint32_t srcDeviceId;
    uint64_t notifyId;
    uint32_t dstDeviceId;
    uint8_t  notifyCount;
    uint8_t  protocol;
};

struct CcuTask {
    uint8_t  dieId;
    uint8_t  missionId;
    uint16_t timeout;
    uint16_t instStartId;
    uint16_t instCnt;
    uint32_t key;
    uint32_t argSize;
    uint64_t args[13];
};

struct AivTask {
    uint64_t launchIdx;
};

struct SyncStreamTask {
    uint64_t syncIdx;
};

struct WriteWithNotifyTask {
    uint32_t srcDeviceId;
    uint64_t srcOffset;
    uint32_t dstDeviceId;
    uint64_t dstOffset;
    uint64_t len;                // MEM_CPY: 传输长度; REDUCE: dataCount
    uint64_t notifyId;
    uint8_t  isReduce;           // 0: 写部分为MEM_CPY; 1: 写部分为REDUCE
    uint8_t  dataType;
    uint8_t  reduceOp;
    uint8_t  protocol;
    uint8_t  notifyCount;
};

// ============================================================================
// 任务 profiling 对账信息子结构 (3 字节, checker/runner 作为不透明整体透传)
// 值域对齐 hcomm TaskParamType/DfxLinkType/DfxTransportType
// ============================================================================
struct HcclTaskProfInfo {
    ProfTaskParamType  profTaskType;   // 对应 msprof timeline op 名
    ProfLinkType       linkType;       // 物理链路类型
    ProfTransportType  transportType;  // 数据面传输方式
};

// ============================================================================
// 任务元数据 (taskType 后含 HcclTaskProfInfo 子结构, sizeof 189 字节)
// ============================================================================
struct HcclTaskMetaData {
    HccLTaskMetaType taskType;
    HcclTaskProfInfo profInfo;         // profiling 对账信息(整体 3 字节)
    uint64_t commId;
    uint32_t rankId;
    uint64_t deviceId;
    uint64_t streamId;
    uint32_t jettyId;
    uint8_t  sqeGroupId;               // UDMA SQE批分组标记, 环形递增1~255
    uint8_t  rmEid[16];
    uint8_t  lcEid[16];
    union {
        TransMemTask  transMem;
        ReduceTask    reduce;
        NotifyTask    notify;
        CcuTask       ccu;
        AivTask       aiv;
        SyncStreamTask syncStreamTask;
        WriteWithNotifyTask writeWithNotify;
    } taskData;
};

#pragma pack(pop)
```

### 7.2 读取综合数据文件示例

```cpp
#include <cstdio>
#include <iostream>

// 魔数定义
constexpr uint32_t HCCLVM_SYN_FILE_MAGIC = 0x48564D44;

bool ReadSynthesisData(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // ===== Step 1: 读取文件头 =====
    FileHeader header;
    if (fread(&header, sizeof(FileHeader), 1, fp) != 1) {
        std::cerr << "Failed to read header" << std::endl;
        fclose(fp);
        return false;
    }

    // 验证魔数
    if (header.magic != HCCLVM_SYN_FILE_MAGIC) {
        std::cerr << "Invalid magic number: 0x" << std::hex << header.magic << std::endl;
        fclose(fp);
        return false;
    }

    std::cout << "File version: " << header.version << std::endl;
    std::cout << "Data count: " << header.count << std::endl;

    // ===== Step 2: 读取模型信息 =====
    ModelInfoCommInner modelComm;
    fread(&modelComm, sizeof(ModelInfoCommInner), 1, fp);
    
    std::cout << "\n[Model Info]" << std::endl;
    std::cout << "  Rank Size: " << modelComm.rank_size << std::endl;
    std::cout << "  Op Type: " << modelComm.op_type << std::endl;
    std::cout << "  Data Count: " << modelComm.data_count << std::endl;
    std::cout << "  Expansion Mode: " << modelComm.op_expansion_mode << std::endl;

    // ===== Step 3: 读取 VDataDesTag =====
    uint16_t vDataType;
    uint32_t vDataCount;
    fread(&vDataType, sizeof(uint16_t), 1, fp);
    fread(&vDataCount, sizeof(uint32_t), 1, fp);
    
    if (vDataCount > 0) {
        std::vector<uint64_t> displs(vDataCount);
        std::vector<uint64_t> counts(vDataCount);
        fread(displs.data(), sizeof(uint64_t), vDataCount, fp);
        fread(counts.data(), sizeof(uint64_t), vDataCount, fp);
        std::cout << "\n[VDataDes] Count: " << vDataCount << std::endl;
    }

    // ===== Step 4: 读取 All2AllDataDesTag =====
    uint16_t sendType, recvType;
    uint64_t sendCount, recvCount;
    uint32_t matrixCount;
    
    fread(&sendType, sizeof(uint16_t), 1, fp);
    fread(&recvType, sizeof(uint16_t), 1, fp);
    fread(&sendCount, sizeof(uint64_t), 1, fp);
    fread(&recvCount, sizeof(uint64_t), 1, fp);
    fread(&matrixCount, sizeof(uint32_t), 1, fp);
    
    if (matrixCount > 0) {
        std::vector<uint64_t> sendCountMatrix(matrixCount);
        fread(sendCountMatrix.data(), sizeof(uint64_t), matrixCount, fp);
        std::cout << "\n[All2All] Matrix Count: " << matrixCount << std::endl;
    }

    // ===== Step 5: 读取 ChannelInfo（CCU 与 AICPU 模式均为 ChannelData 定长结构；AIV 模式无此段） =====
    if (modelComm.op_expansion_mode != 2) {
        uint32_t channelCount;
        fread(&channelCount, sizeof(uint32_t), 1, fp);
        
        std::vector<ChannelData> channels(channelCount);
        fread(channels.data(), sizeof(ChannelData), channelCount, fp);
        
        std::cout << "\n[ChannelInfo] Count: " << channelCount << std::endl;
        for (size_t i = 0; i < channels.size() && i < 3; ++i) {
            std::cout << "  Channel[" << i << "]: ID=" << channels[i].channelId
                      << ", srcRank=" << channels[i].srcRank
                      << ", dstRank=" << channels[i].dstRank
                      << ", jettyNum=" << channels[i].jettyNum;
            for (uint16_t j = 0; j < channels[i].jettyNum; ++j) {
                std::cout << ", jetty[" << j << "]=" << channels[i].jettyId[j];
            }
            std::cout << std::endl;
        }
    }

    // ===== Step 6: 读取内存布局 =====
    uint32_t memCount;
    fread(&memCount, sizeof(uint32_t), 1, fp);
    
    std::vector<MemLayoutData> memLayouts(memCount);
    fread(memLayouts.data(), sizeof(MemLayoutData), memCount, fp);
    
    std::cout << "\n[MemLayout] Count: " << memCount << std::endl;
    for (size_t i = 0; i < memLayouts.size() && i < 3; ++i) {
        std::cout << "  Mem[" << i << "]: rank=" << memLayouts[i].rank_id
                  << ", type=" << (int)memLayouts[i].buffer_type
                  << ", size=" << memLayouts[i].size << std::endl;
    }

    fclose(fp);
    std::cout << "\nRead synthesis data success!" << std::endl;
    return true;
}
```

### 7.3 读取任务元数据文件示例

```cpp
constexpr uint32_t HCCLVM_TASK_FILE_MAGIC = 0x48565444;

bool ReadTaskMetaData(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // ===== Step 1: 读取文件头 =====
    FileHeader header;
    fread(&header, sizeof(FileHeader), 1, fp);

    if (header.magic != HCCLVM_TASK_FILE_MAGIC) {
        std::cerr << "Invalid magic number" << std::endl;
        fclose(fp);
        return false;
    }

    std::cout << "Task count: " << header.count << std::endl;

    // ===== Step 2: 读取所有任务 =====
    std::vector<HcclTaskMetaData> tasks(header.count);
    fread(tasks.data(), sizeof(HcclTaskMetaData), header.count, fp);

    // ===== Step 3: 解析并打印任务 =====
    std::cout << "\n[Task List]" << std::endl;
    for (size_t i = 0; i < tasks.size(); ++i) {
        const auto& task = tasks[i];
        std::cout << "Task[" << i << "] ";
        
        switch (task.taskType) {
            case HccLTaskMetaType::MEM_CPY:
                std::cout << "MEM_CPY: "
                          << "src=" << task.taskData.transMem.srcDeviceId
                          << ", dst=" << task.taskData.transMem.dstDeviceId
                          << ", len=" << task.taskData.transMem.len;
                break;
                
            case HccLTaskMetaType::REDUCE:
                std::cout << "REDUCE: "
                          << "src=" << task.taskData.reduce.srcDeviceId
                          << ", dst=" << task.taskData.reduce.dstDeviceId
                          << ", count=" << task.taskData.reduce.dataCount;
                break;
                
            case HccLTaskMetaType::NOTIFY_WAIT:
                std::cout << "NOTIFY_WAIT: "
                          << "notifyId=" << task.taskData.notify.notifyId;
                break;
                
            case HccLTaskMetaType::NOTIFY_RECORD:
                std::cout << "NOTIFY_RECORD: "
                          << "notifyId=" << task.taskData.notify.notifyId;
                break;
                
            default:
                std::cout << "UNKNOWN: type=" << static_cast<int>(task.taskType);
                break;
        }
        std::cout << std::endl;
    }

    fclose(fp);
    std::cout << "\nRead task meta data success!" << std::endl;
    return true;
}
```

---

## 8. 注意事项

| 序号 | 注意事项          | 说明                                                            |
|:----:|-------------------|-----------------------------------------------------------------|
|  1   | 字节序            | 所有多字节字段均采用**小端序**（Little-Endian）                 |
|  2   | 内存对齐          | 结构体使用 `#pragma pack(1)` 进行 **1 字节对齐**                |
|  3   | 变长字段读取      | 先读取 `count` 字段，再根据 count 值读取相应数量的数据          |
|  4   | 模式判断          | CCU 与 AICPU 模式均读取 ChannelInfo（ChannelData 定长结构）；AIV 模式无此段；JettyData 为未使用的历史结构 |
|  5   | 联合体解析        | 任务元数据中的 `taskData` 需根据 `taskType` 选择正确的结构体    |
|  6   | 魔数验证          | 读取文件时务必先验证魔数，确保文件类型正确                      |

---

## 附录 A：魔数速查表

| 文件类型   | 魔数值       | ASCII  | 说明                     |
|-----------|--------------|--------|--------------------------|
| 综合数据   | `0x48564D44` | "HVMD" | Hccl VM Data             |
| 微码指令   | `0x434D4349` | "CMCI" | CCU Microcode Instruction|
| 任务元数据 | `0x48565444` | "HVTM" | Hccl VM Task Metadata    |

---

## 附录 B：文件读取流程

**Step 1**：打开二进制文件。

**Step 2**：读取 FileHeader (20 Bytes)

**Step 3**：验证魔数 (magic)

- 魔数不匹配 → 返回错误，关闭文件。
- 魔数匹配 → 继续。

**Step 4**：根据 count 读取数据内容。

**Step 5**：关闭文件。

---

## 附录 C：联系方式

如有疑问，请联系 HCCL VM Team。

---

## 文档结束
