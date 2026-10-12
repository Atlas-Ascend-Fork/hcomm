/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_MODEL_DEFS_H
#define SIM_MODEL_DEFS_H

#include <cstdint>
#include <functional>

#include "sim_common_defs.h"

using namespace HcclSim;

namespace sim::runtime {

typedef struct {
    uint64_t id;         // PK
    char file_name[128]; // topo_meta yaml文件名(不含.yaml后缀); "ranktable"表示ranktable.json模式
} TopoMetaConfig;

typedef struct {
    uint64_t id; // PK
    uint64_t pod_id;
    uint32_t server_id;
    uint32_t used_dev_num;
    char version[16];
    char hardware_type[128];
} Server;

typedef struct {
    uint64_t id;  // PK
    uint8_t mode; // 0=normal,1=check-only
} RunModeConfig;

typedef struct {
    uint64_t id; // PK
    char tag[64];
} Plugin;

typedef struct {
    uint64_t id; // PK
    uint64_t server_id;
    char ip_addr[40]; // ip地址
    uint8_t arch;
} Host;

typedef struct {
    uint64_t id;      // PK
    uint64_t host_id; // FK
    uint64_t pid;
    uint64_t thread_id;
    uint64_t timeout_config_ms;
    uint64_t current_ctx_id; // FK
} Runner;

// Communicator.status: 通信域成员销毁状态(level0 销毁同步使用)
enum CommStatus : uint8_t {
    COMM_STATUS_ACTIVE = 0,    // 未销毁
    COMM_STATUS_DESTROYED = 1, // 已调用 HcclCommDestroy
};

typedef struct {
    uint64_t id; // PK（通信域锚点成员：同一 (comm_id, comm_hash) 下 id 最小者作为通信域标识）
    char comm_id[128];  // 通信域名称，与 comm_hash 共同唯一标识通信域
    uint32_t rank_size; // 通信域总 Rank 数
    uint32_t rank_id;   // 当前线程在通信域中的 Rank
    uint64_t device_id; // FK -> Device.id
    uint64_t comm_hash; // 通信域成员hash值，用于区分同名多通信域
    // Level1 control-plane state is kept on the same communicator member row.
    uint32_t deterministic;
    uint8_t op_expansion_mode;
    uint32_t rdma_traffic_class;
    uint32_t rdma_service_level;
    uint64_t sym_win_addr;
    uint64_t sym_win_size;
    uint8_t sym_win_registered;
    uint8_t status; // CommStatus: 销毁同步状态
} Communicator;

// level 1 专用通信域清理同步机制
typedef struct {
    uint64_t id;
    char comm_id[128];
    uint64_t comm_hash;
    uint32_t rank_size;
    uint32_t rank_id;
} CommunicatorDestroySync;

typedef struct {
    uint64_t id;        // PK
    uint32_t server_id; // FK
    uint32_t user_id{0xFFFF};
    uint32_t logic_id{0xFFFF};
    uint32_t physical_id;
    uint32_t super_device_id;
    uint32_t overflow_mode;
    // 完整SoC型号名(如"Ascend950DT_95A1"，16字符)需16+1字节，32对齐hal侧MAX_CHIP_NAME
    char soc_version[32];
    uint32_t status; // 0 可用， 1不可用
} Device;

enum TsDevType { TS_DEV_TYPE_SCALAR = 0, TS_DEV_TYPE_CPU = 1, TS_DEV_TYPE_CCU = 2 };

typedef struct {
    uint64_t id;        // PK
    uint64_t device_id; // FK
    uint8_t type;       // 0:Scalar, 1:CCU, 2:CPU
} TaskSchedulerDevice;

enum ComputeDieType { COMPUTE_TYPE_CUBE = 0, COMPUTE_TYPE_VECTOR = 1, COMPUTE_TYPE_HYBRID = 2 };

typedef struct {
    uint64_t id;    // PK
    uint64_t ts_id; // FK
    uint8_t type;   // 0:Cube 1:Vector 2: HybridCompute
} ComputeDie;

typedef struct {
    uint64_t id;        // PK
    uint64_t device_id; // FK
    uint8_t overflow_status;
    uint8_t synchronize_strategy;
    uint8_t synchronize_timeout;
    uint8_t capability_mask;
    uint8_t run_by_host;
    uint8_t ts_core;
    uint8_t online_status;
} DeviceStatus;

typedef struct {
    uint64_t id;        // PK
    uint64_t device_id; // FK
    uint64_t die_id;    // FK
    char name[128];     // 对应topo文件中的port_id: 0/0 ~ 0/8
    uint8_t status;     // 0: 未使用 1: 已使用
} Port;

typedef struct CcuTag {
    uint64_t id;            // PK
    uint64_t device_id;     // FK
    uint64_t resource_addr; // ccu资源地址，区分不同die的资源：die0: 0x123123123; die1: 0x456456456
    uint8_t die_id;
    uint8_t status;
} Ccu;

typedef struct DeviceConnectionTag {
    uint64_t id;         // PK
    uint64_t src_dev_id; // FK
    uint64_t dst_dev_id; // FK
    uint8_t link_type;
    uint8_t access_by_remote;
} DeviceConnection;

typedef struct {
    uint64_t id; // PK
    uint32_t device_id;
    uint16_t func_id;
    uint8_t die_id;
    uint16_t type; // 地址类型: 0 - EID, 1 - IPV4, 2 - IPV6
    uint8_t eid[16];
    char ip_addr[64];
    uint8_t status{0};
    bool is_uboe{false};
} EndPoint;

typedef struct {
    uint64_t id; // PK
    uint64_t local_enpoint_id;
    uint64_t remote_enpoint_id;
    uint8_t tp_type;
} EndPointPair;

typedef struct {
    uint64_t id; // PK
    uint64_t port_id;
    uint64_t endpoint_id;
    uint8_t net_layer;
} EndPointPortMapping;

typedef struct {
    uint64_t id; // PK
    uint64_t local_endpoint_id;
    uint64_t remote_endpoint_id;
    uint8_t net_layer;    // 网络层级: 0/1/2/3
    uint8_t type;         // 0: peer2peer, 1: peer2net
    uint8_t protocols[8]; // 0: UB_CTP, 1: UB_MEM, 2: UB_TP, ...
} Link;

typedef struct {
    uint64_t id; // PK
    uint16_t channel_id;
    uint64_t local_endpoint_id;
    uint64_t remote_endpoint_id;
    uint8_t protocol;
    uint32_t jetty_start;
    uint32_t jetty_num;
} CcuChannel;

typedef struct {
    uint64_t id; // PK
    uint64_t run_id;
    uint64_t device_id;
    uint8_t is_default;
    uint32_t ref_cnt;
    uint64_t float_overflow_addr;
    uint8_t capture_mode;
} Context;

typedef struct {
    uint64_t id;     // PK
    uint64_t ctx_id; // 上下文ID (FK)
    uint64_t sq_base_addr;
    uint8_t is_primary_default;
    uint8_t is_other_default;
    uint8_t priority;
    uint8_t schedule_strategy;
    uint8_t failure_mode;
    uint32_t user_tag;
    uint8_t overflow_switch;
    uint8_t activated;
    uint8_t capture_status;
    uint8_t task_complete_status;
} Stream;

typedef struct {
    uint64_t id; // PK
    uint64_t stream_id;
    uint64_t cid;
    uint64_t seq_number;
    uint8_t type;
} Task;

typedef struct {
    uint64_t id; // PK
    uint64_t execute_time_ms;
    uint64_t finish_time_ms;
    uint32_t op_timeout_s;
} EventSyncTask;

typedef struct {
    uint64_t id; // PK
    uint64_t create_ctx_id;
    uint64_t device_notify_seq;
    uint8_t value;
} Notify;

typedef struct {
    uint64_t id; // PK
    uint64_t notify_id;
    uint8_t name_or_key[16];
    uint8_t create_pid;
} IpcNotify;

typedef struct {
    uint64_t id;     // PK
    uint64_t ipc_id; // FK
    uint64_t visitor_pid;
} IpcNotifyVistorList;

typedef struct {
    uint64_t id;        // PK
    uint64_t notify_id; // FK
} NotifyRecordTask;

typedef struct {
    uint64_t id;        // PK
    uint64_t notify_id; // FK
} NotifyWaitTask;

typedef struct {
    uint64_t id;            // PK
    uint64_t create_ctx_id; // FK
    uint64_t event_flag;
    uint64_t device_res_seq;
    uint64_t created_time;
    uint8_t status;
} Event;

typedef struct {
    uint64_t id;
    uint64_t device_id;
    char name[64];
    uint64_t size;
    uint8_t type;
    uint64_t ref_count;
    uint8_t is_freed; // 0: using, 1: freed
} PhyMemBlock;

enum VirMemType { VIR_MEM_TYPE_HOST, VIR_MEM_TYPE_DEV };

typedef struct {
    uint64_t id;
    uint64_t start_ptr;      // 按照卡分配的虚拟编址的地址
    uint64_t dev_mapped_ptr; // device进程打开共享内存后的地址
    uint8_t is_dev_access;   // 0: dev不可直接访问, 1: dev可以直接访问
    uint64_t host_ptr{0};    // host进程打开共享内存后的地址
    uint64_t size;
    uint64_t ctx_id;
    uint64_t device_id; // Device表主键
    uint64_t rank_id;   // 通信域RankId
    uint64_t phy_mem_id;
    uint64_t owner_pid; // host查找根据pid
    uint8_t src_type;   // 0: host, 1: device
    uint8_t policy;
    // 派生区间端点（编码时按基址+大小计算入库，解码忽略）：
    // 仅作为地址包含查询（基址 <= 地址 < 基址+大小）上界下推的成员锚点，不承载业务赋值语义。
    uint64_t start_end_ptr;
    uint64_t dev_mapped_end_ptr;
    // 软删标记（2026-09-17 拆卸竞态修复，业务层语义）：拆卸期快 rank 宿主
    // 删除本 rank 的 VMB 行成功后 16.7s+，慢 rank 设备进程的迟到地址解析按
    // 合同得到 NOT_FOUND（FR1/R3K4 取证 43/48 次，删除回执全部成功）——
    // 释放路径改为 Update 置本字段=1（行保留，业务侧 aclrt_memory_stub.cc），
    // 迟到解析仍可命中。业务构造行为零初始化（=0 在用），不感知本字段；
    // 与 PhyMemBlock.is_freed 同名同语义（Phy 为历史预留，本轮一并启用）。
    uint8_t is_freed; // 0: using, 1: freed(已标记释放)
} VirtualMemBlock;

typedef struct {
    uint64_t id;
    uint64_t vir_mem_id;
    uint64_t offset;
    uint8_t create_pid;
} IpcMemRecord;

typedef struct {
    uint64_t id;
    uint64_t name_or_key;
    uint64_t pid;
    uint8_t create_pid;
} IpcMemWhiteList;

typedef struct {
    uint64_t id;
    uint64_t vir_mem_id;
    uint64_t phy_mem_id;
    uint8_t create_pid;
} FdMemRecord;

typedef struct {
    uint64_t id;
    uint64_t name_or_key;
    uint64_t pid;
    uint8_t create_pid;
} FdMemWhiteList;

typedef struct {
    uint64_t id;
    uint64_t server_id; // FK
    uint64_t client_id; // FK
    uint32_t ref_cnt;
    uint32_t port;
    uint64_t tag_hash;
    uint8_t buf_status; // 0: buffer pending, 1: buffer ready
    uint32_t slot_idx;  // 槽位索引
} RaSocketPair;

// todo: 数据建模？
typedef struct {
    uint64_t id;      // PK
    uint32_t rank_id; // todo: 后续可能需要跟device关联
    uint64_t base_addr;
    uint8_t buf_type;
    uint8_t reserved;
    uint64_t size;
    uint64_t global_offset;
} MemoryLayout;

// ReduceScatterV AllGatherV使用
struct VDataDesTagInner {
    uint16_t dataType;   // 数据类型
    uint32_t count{0};   // rank size
    uint64_t displs[64]; // 每个rank的数据在sendBuf中的偏移量（单位为dataType）
    uint64_t counts[64]; // 每个rank在sendBuf中的数据size，第i个元素表示需要向rank i发送/接受的数据量
};

struct All2AllDataDesTagInner {
    uint16_t sendType;              // 发送数据的数据类型
    uint16_t recvType;              // 接收数据的数据类型
    uint64_t sendCount;             // 发送数据量 (All2All)
    uint64_t recvCount;             // 接收数据量 (All2All)
    uint32_t count{0};              // count = rankSize * rankSize
    uint64_t sendCountMatrix[4096]; // (All2AllVC) sendCountMatrix[i * ranksize + j] 代表rank i发送到rank j的count参数
};

enum SimOpExpansionMode {
    SIM_OP_EXPANSION_MODE_CCU = 0,
    SIM_OP_EXPANSION_MODE_AICPU = 1,
    SIM_OP_EXPANSION_MODE_AIV = 2,
    SIM_OP_EXPANSION_MODE_RESERVED = 255
};

typedef struct SimModelDataTag {
    uint64_t id; // PK
    uint32_t rank_id;
    uint32_t src_rank;
    uint32_t dst_rank;
    uint32_t root;
    uint32_t rank_size;
    uint16_t chip_type;
    uint16_t op_type;
    uint16_t reduce_op;
    uint16_t data_type;
    uint64_t data_count;
    uint8_t op_expansion_mode;
    uint64_t ccu0_resource_base_addr;
    uint64_t ccu1_resource_base_addr;
    VDataDesTagInner vDataDes;
    All2AllDataDesTagInner all2AllDataDes;
} SimModelData;

typedef struct {
    uint64_t id; // PK
    uint32_t device_id;
    uint8_t mac_addr[14];
    uint8_t state;
    uint64_t endpoint_id; // FK local EID
} RaDevice;

typedef struct {
    uint64_t id;        // PK
    uint64_t ra_dev_id; // FK
    uint64_t send_cq_handle;
    uint64_t recv_cq_handle;
    uint32_t qp_num;
    uint8_t type;
    uint8_t state; // 0: RESET, 1: INIT, 2: RTR, 3:RTS
    uint64_t taJettyId;
    uint32_t mode; // jetty mode 0: URMA, 2: CCU, 3: Normal
    uint64_t peer_qp_id;
    uint32_t peer_qpn;
    uint32_t perr_lid;
    uint64_t pid;
} RaQP;

typedef struct {
    uint64_t id;        // PK
    uint64_t ra_dev_id; // FK
    uint32_t cqn;
    uint32_t size;
} RaCQ;

typedef struct {
    uint64_t id; // PK
    uint64_t cq_handle;
    uint32_t wr_id;
    uint8_t status; // 0: success, 1: flush_err
} RaCQE;

typedef struct {
    uint64_t id; // PK
    uint64_t local_key;
    uint64_t remote_key;
    uint64_t vptr_id;
    uint64_t length;
    uint64_t addr;
} RaMR;

typedef struct {
    uint64_t id;        // PK
    uint64_t device_id; // FK
    int mode;
    uint64_t endpoint_id; // FK local EID
    uint32_t eidIndex;
    uint64_t max_jetty_num;
    uint64_t max_jfc_num;
} RaContext;

typedef struct {
    uint64_t id;         // PK
    uint64_t ctx_handle; // FK
    uint32_t chann_id;
    uint32_t mode;
} RaChan;

typedef struct {
    uint64_t id;         // PK
    uint64_t ctx_handle; // FK
    uint32_t token_id;
    uint64_t ref_count;
} RaTokenId;

typedef struct {
    uint64_t id;         // PK
    uint64_t ctx_handle; // FK
    uint8_t tp_type;
    uint64_t tpn;
    uint64_t speed;
    uint8_t status;
} RaTp;

typedef struct {
    uint64_t id;         // PK
    uint64_t ctx_handle; // FK
    uint64_t remote_key;
    uint64_t target_seg_handle;
    uint64_t remote_eid;
} RaRmem;

typedef struct {
    uint64_t id;         // PK
    uint64_t ctx_handle; // FK
    uint64_t addr;
    uint64_t size;
    uint64_t mem_key;
    uint64_t token_id;
    // 注册时所属 RaContext 的 endpoint_id 快照：Import 直接取用，免每次
    // 按远端 ctx_handle 二次查询 RaContext（1024p RaRmem 30 万+行的热点）。
    uint64_t endpoint_id;
} RaLmem;

typedef struct {
    uint64_t id;         // PK
    uint64_t ctx_handle; // FK
    uint64_t send_cq_handle;
    uint64_t recv_cq_handle;
    uint32_t sqDepth;
    uint32_t rqDepth;
    uint64_t sqBuffer; // jetty下发wqe对应的buffer地址
    uint8_t sqBufType;
    uint8_t type;
    uint32_t jetty_id; // jetty 硬件编号，写入 doorbell SQE 的 16 位字段（AICPU 与 CCU 均使用；行主键仅作句柄，不写入
                       // SQE 字段）
    uint32_t dieId;
    uint32_t rank_id;
    uint8_t state; // 0: RESET, 1: INIT, 2: RTR, 3:RTS
    uint32_t mode; // jetty mode 0: URMA, 2: CCU, 3: Normal
    uint64_t peer_jetty_handle;
    uint64_t peer_endpoint_id; // 本jetty所在channel的对端EndPoint id(FK EndPoint.id)
    uint64_t pid;
} RaJetty;

typedef struct {
    uint64_t id;         // PK
    uint64_t ctx_handle; // FK
    uint64_t jfc_id;
    uint64_t depth;
    uint32_t mode; // jetty mode 0: URMA, 2: CCU, 3: Normal
    uint64_t policy;
} RaJfc;

typedef struct {
    uint64_t id;         // PK
    uint64_t jfc_handle; // FK
    uint64_t user_ctx;
    uint64_t status;
    uint32_t opcode;
    uint64_t byte_len;
} RaCr;

typedef struct {
    uint64_t id; // PK
    uint32_t physical_id;
} RaTlv;

// 北向打桩接口——HCCL_BUFFER表
typedef struct {
    uint64_t id;     // buffer的ID，PK
    uint64_t commId; // 所属的通信域FK
    uint64_t addr;   // 内存地址
    uint64_t size;   // 内存的大小
} HcclBuffer;

// 北向打桩接口——HCCL_Thread表
typedef struct {
    uint64_t id;           // thread的ID，PK
    uint64_t commId;       // 所属的通信域FK
    uint8_t engine;        // 通信引起engine
    uint16_t notifyNum;    // Notify的数量
    uint32_t notifyId[40]; // 对应的NofifyId
    uint64_t streamId;     // 关联的流id
} HcclThread;

// 北向打桩接口——HCCL_Channel表
typedef struct {
    uint64_t id;           // channel的ID，PK
    uint64_t commId;       // 所属的通信域FK
    uint64_t endpoint_id;  // 所属的北向 HcommEndpoint 逻辑外键，0 表示未绑定
    uint8_t engine;        // 通信引起engine
    uint64_t remoteRankId; // 对端rankId
    uint16_t notifyNum;    // Notify的数量
    uint64_t notifyId[64]; // 对应的NofifyId
    bool status;           // 连接状态
} HcclChannel;

// 北向打桩接口——HComm Endpoint 与南向拓扑 Endpoint 的绑定状态
enum HcommEndpointBindState : uint8_t {
    HCOMM_ENDPOINT_UNRESOLVED = 0,     // 尚未找到唯一的南向 Endpoint
    HCOMM_ENDPOINT_BOUND = 1,          // 已绑定唯一的南向 Endpoint
    HCOMM_ENDPOINT_NOT_APPLICABLE = 2, // 当前位置或协议不需要南向 Endpoint
};

// 北向打桩接口——HComm Endpoint 表
typedef struct {
    uint64_t id;                // EndpointHandle 对应的主键
    uint64_t runner_id;         // 创建该 Endpoint 的 Runner 逻辑外键
    uint64_t south_endpoint_id; // 南向 sim::EndPoint 逻辑外键，0 表示未绑定或不适用
    int32_t protocol;           // CommProtocol
    int32_t addr_type;          // CommAddrType
    uint8_t addr[36];           // CommAddr 中的原始地址内容
    int32_t loc_type;           // EndpointLocType
    uint8_t loc[60];            // EndpointLoc 中的原始位置内容
    uint8_t extension[52];      // EndpointDesc 预留扩展内容
    uint8_t bind_state;         // HcommEndpointBindState
} HcommEndpoint;

// 北向打桩接口——HComm 内存记录类型
enum HcommMemRecordKind : uint8_t {
    HCOMM_MEM_LOCAL_REGISTERED = 0, // HcommMemReg 创建的本地注册内存
    HCOMM_MEM_REMOTE_IMPORTED = 1,  // HcommMemImport 创建的远端导入内存
};

// 北向打桩接口——HComm 内存资源表
typedef struct {
    uint64_t id;                 // HcommMemHandle 对应的主键
    uint64_t endpoint_id;        // 所属的北向 HcommEndpoint 逻辑外键
    uint64_t runner_id;          // 创建该记录的 Runner 逻辑外键
    uint64_t source_endpoint_id; // 导入记录的源端 HcommEndpoint 主键，本地记录为 0
    uint64_t source_mem_id;      // 导入记录的源端本地内存主键，本地记录为 0
    uint64_t descriptor_key;     // 导出描述符的稳定校验键
    uint64_t addr;               // CommMem.addr 的整数形式
    uint64_t size;               // CommMem.size
    int32_t mem_type;            // CommMemType
    uint8_t record_kind;         // HcommMemRecordKind
    char mem_tag[255];           // 本地注册内存标签，导入记录为空
} HcommMemReg;

// 北向打桩接口——HCCL_EngineCtx表
typedef struct {
    uint64_t id;     // engineCtx的ID，PK
    uint64_t commId; // 所属的通信域FK
    uint64_t ctxTag; //
    uint64_t engine; // 对端rankId
    uint64_t size;   // 大小
    uint64_t addr;   // 地址
} HcclEngineCtx;

// 北向打桩接口——HCCLMem表
typedef struct {
    uint64_t id;           // engineCtx的ID，PK
    uint64_t commId;       // 所属的通信域FK
    uint64_t memTag;       // 标签哈希值，用于快速查找
    uint64_t memType;      // 内存类型 CommMemType
    uint64_t addr;         // 地址
    uint64_t size;         // 大小
    char mem_tag_str[255]; // 原始标签字符串，供 GetRemoteMems 返回
} HcclMem;

typedef struct {
    uint64_t id;        // PK
    uint32_t device_id; // NPU-DPU 交互所属设备
    uint64_t stream_id; // NPU-DPU 交互的流
} DpuDeviceInfo;

typedef struct {
    uint64_t id{0};
    uint32_t dstDeviceId{0};
    uint64_t notifyId{0};
    uint32_t srcDeviceId{0};
    uint32_t immData{0};
    uint8_t consumed{0}; // 0: 未被消费, 1: 已被消费
} DpuPendingNotify;

// 北向打桩接口——CCU Level1 交换资源表（跨 rank 同步值，per-channel 交换区，**按 op+轮次隔离**）
// 语义：hcomm CcuTransport::INIT_XN_NUM/INIT_CKE_NUM(4) —— 每 channel 固定预留
//       XN/CKE 0..3；PreSync 时写方把值写到对端坐标，读方（NotifyWait / 首次使用处）查询。
// 逻辑键：(opIter, round, srcRank, targetRank, connOrdinal, dieId, resType, resId)；
//       opIter = 算子迭代（OpDetailTab.opIter，两侧进程按同一序列自增，天然跨 rank 对齐），
//       round = 写方 CCU launch 序号。**一轮一行、不覆盖** —— 读方按自己所在 op+轮次精确取，
//       可免疫两进程异步漂移（写方领先时不会被"未来轮次/未来 op 的值"污染）。
//       op 维度不可省：round 是进程内全局 launch 计数，表跨 op 累积，若不带 op 则后一 op 的
//       行会覆盖/裁掉前一 op 尚未被读走的行，落后 rank 只能取到别的 op 的地址（现象：
//       checker 报 MEM_CPY 目标地址落不到本 op 内存布局）。
typedef struct {
    uint64_t id{0};          // PK
    uint32_t opIter{0};      // 算子迭代（跨 rank 一致的 op 身份）
    uint32_t round{0};       // 轮次（= 写方 CCU launch 序号）
    uint32_t targetRank{0};  // 被写入方 rank（= 读方 rank）
    uint32_t connOrdinal{0}; // 同 peer 多 channel 连接序号（multi-jetty；单连接=0）
    uint32_t dieId{0};       // 资源所在 die
    uint32_t resType{0};     // 0=XN(Var), 1=CKE
    uint32_t resId{0};       // 0..3（交换区槽位）
    uint64_t value{0};       // 交换值（地址/token/CKE 位置位）
    uint32_t srcRank{0};     // 写方 rank（排障）
    uint64_t ownerPid{0};    // 写方进程（排障）
} CcuSyncResTab;

} // namespace sim::runtime
#endif
