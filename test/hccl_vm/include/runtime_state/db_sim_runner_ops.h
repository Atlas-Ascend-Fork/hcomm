/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * for the full text of the License.
 */

// 文件职责：RunnerDB（RUNNER_DB）的统一逻辑 DDL 目录、薄兼容接口与 Runner
// 领域操作声明。 65 张表在一个 master catalog 中平铺登记；每张表只有一个紧凑
// Table entry， 字段/属性/约束/索引全部写在同一 entry，公共生成器从这里派生
// RecordSchema/字段映射/manifest/fingerprint 与关系 mapping 输入。
// 短宏（Field/PrimaryKey/Index 等）只在本 DDL 区局部定义并立即
// #undef，不污染包含方。 DDL
// 区是唯一事实源：表结构与约束只在此声明，不另设第二份 schema 清单。
// sim::runtime::Db 为薄兼容接口：只一对一转发现有 Storage 一次性原子接口
// （Add/GetById/GetOneByPred/GetByPred/Count/Update/Delete/DeleteAll），不是新的数据库抽象，
// 也不绑定任何表或字段；SQLite 与 MariaDB 都经同一 Storage 合同执行。
// 普通业务代码的数据库边界：只经 sim::runtime::Db
// 兼容接口、Condition/Assignment DSL
// 或本文件声明的稳定一行业务函数访问数据库，不直接 AcquireTable/Storage CRUD/
// session/backend/SQL 类型；Storage 原子合同只有实现边界文件
// （runtime_state 两个 db_sim_runner_*.cc、operation_data_ops.cc 与 storage/**
// 内部） 可以直接使用。本文件不包含 session/backend/engine/SQL 逻辑。
#ifndef _SIM_RUNNER_OPS_H_
#define _SIM_RUNNER_OPS_H_

#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "runtime_state/sim_models.h"
#include "storage/database_result.h"
#include "storage/query_expression.h"
#include "storage/query_spec.h"
#include "storage/storage_catalog.h"
#include "storage/table_access.h"

// ==== RunnerDB 唯一 DDL 区（master catalog；短别名区段结束立即 #undef） ====
#define Field HCCL_VM_DDL_FIELD
#define Version HCCL_VM_DDL_VERSION
#define PrimaryKey HCCL_VM_DDL_PRIMARY_KEY
#define Unique HCCL_VM_DDL_UNIQUE
#define Index HCCL_VM_DDL_INDEX
#define ForeignKey HCCL_VM_DDL_FOREIGN_KEY
#define Scope Ddl::ScopeKind
#define AutoIncrement Ddl::AutoIncrement
#define Text Ddl::Text
#define Blob Ddl::Blob
#define Derived Ddl::Derived
#define OnDelete Ddl::OnDelete

// clang-format off
// RunnerDB 表目录（65 条目巨宏）：字段/索引的紧凑排版为本区块的固定手写格式，
// 关闭 clang-format 以避免自动对齐产生超长填充并污染 git diff（见文件头 DDL 区说明）。
#define HCCL_VM_RUNNER_DATABASE(Table) \
    Table(sim::runtime::Server, (Version(3), Scope::Static, \
        Field(id, AutoIncrement), Field(pod_id), Field(server_id), Field(used_dev_num), Field(version, Text), \
        Field(hardware_type, Text), \
        PrimaryKey(id), \
            Index(pod_id, server_id))) \
    Table(sim::runtime::Host, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(server_id), Field(ip_addr, Text), Field(arch), \
        PrimaryKey(id), \
            Index(server_id))) \
    Table(sim::runtime::Runner, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(host_id), Field(pid), Field(thread_id), Field(timeout_config_ms), \
        Field(current_ctx_id), \
        PrimaryKey(id))) \
    Table(sim::runtime::Device, (Version(3), Scope::Static, \
        Field(id, AutoIncrement), Field(server_id), Field(user_id), Field(logic_id), Field(physical_id), \
        Field(super_device_id), Field(overflow_mode), Field(soc_version, Text), Field(status), \
        PrimaryKey(id), \
            Index(server_id, physical_id), Index(physical_id), \
            Index(logic_id), Index(server_id, logic_id), Index(status, server_id))) \
    Table(sim::runtime::DeviceStatus, (Version(3), Scope::Static, \
        Field(id, AutoIncrement), Field(device_id), Field(overflow_status), Field(synchronize_strategy), \
        Field(synchronize_timeout), Field(capability_mask), Field(run_by_host), Field(ts_core), \
        Field(online_status), \
        PrimaryKey(id), \
            Index(device_id), Index(synchronize_strategy))) \
    Table(sim::runtime::Context, (Version(3), Scope::Static, \
        Field(id, AutoIncrement), Field(run_id), Field(device_id), Field(is_default), Field(ref_cnt), \
        Field(float_overflow_addr), Field(capture_mode), \
        PrimaryKey(id), \
            Index(device_id, is_default))) \
    Table(sim::runtime::Stream, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(ctx_id), Field(sq_base_addr), Field(is_primary_default), \
        Field(is_other_default), Field(priority), Field(schedule_strategy), Field(failure_mode), \
        Field(user_tag), Field(overflow_switch), Field(activated), Field(capture_status), \
        Field(task_complete_status), \
        PrimaryKey(id), \
            Index(ctx_id, is_primary_default))) \
    Table(sim::runtime::Notify, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(create_ctx_id), Field(device_notify_seq), Field(value), \
        PrimaryKey(id))) \
    Table(sim::runtime::Event, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(create_ctx_id), Field(event_flag), Field(device_res_seq), \
        Field(created_time), Field(status), \
        PrimaryKey(id))) \
    Table(sim::runtime::VirtualMemBlock, (Version(3), Scope::Static, \
        Field(id, AutoIncrement), Field(start_ptr), Field(dev_mapped_ptr), Field(is_dev_access), \
        Field(host_ptr), Field(size), Field(ctx_id), Field(device_id), Field(rank_id), Field(phy_mem_id), \
        Field(owner_pid), Field(src_type), Field(policy), \
        Field(start_end_ptr, Derived), Field(dev_mapped_end_ptr, Derived), \
        Field(is_freed), \
        PrimaryKey(id), \
            Index(device_id, src_type, dev_mapped_ptr), Index(rank_id, src_type, dev_mapped_ptr), \
            Index(device_id, src_type, start_ptr), Index(src_type, start_ptr), \
            Index(dev_mapped_ptr), Index(start_ptr), \
            Index(dev_mapped_end_ptr), Index(start_end_ptr), \
            Index(device_id, src_type, dev_mapped_ptr, dev_mapped_end_ptr), \
            Index(rank_id, src_type, dev_mapped_ptr, dev_mapped_end_ptr), \
            Index(device_id, src_type, start_ptr, start_end_ptr), \
            Index(src_type, start_ptr, start_end_ptr), \
            /* 无 device_id/rank_id 前导的设备映射地址包含查询（IsDeviceAddress 形状：
               src_type 等值 + dev_mapped_ptr<=addr<dev_mapped_end_ptr，查询条件不变）：
               三列普通联合索引形态（MariaDB FORCE INDEX 选择按谓词形状匹配）。 */ \
            Index(src_type, dev_mapped_ptr, dev_mapped_end_ptr), \
            /* (device_id, is_freed) purge 复合索引（2026-09-20；先例 PhyMemBlock \
               2026-09-17 同形索引）：释放路径改标记（Update is_freed=1）后，\
               插入前 purge（Delete And(Eq(device_id), Eq(is_freed,1))，1024p \
               实测观测一度约 4096 次/秒）在 MariaDB 默认计划下无覆盖索引走\
               全表扫描；复合索引使 purge 谓词完全被前缀覆盖。仅新增索引：\
               不改 purge 调用频率与 is_freed 生命周期语义。索引变更混入表\
               指纹，旧布局由装配期显式拒绝，沿用 367dc4517 惯例不升 Version */ \
            Index(device_id, is_freed))) \
    Table(sim::runtime::PhyMemBlock, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(device_id), Field(name, Text), Field(size), Field(type), \
        Field(ref_count), Field(is_freed), \
        PrimaryKey(id), \
            /* (device_id, is_freed) 复合二级索引（2026-09-17 is_freed 软删轮）：
               释放路径改标记（Update is_freed=1）后，插入前 purge（Delete
               And(Eq(device_id), Eq(is_freed,1))）若无索引将全表扫描
               （128p 实测 purge 重试耗尽、单次 10-18s、负载 135 死亡螺旋：
               标记行被并发标记持续失效+标记行累积放大扫描）；复合索引使
               purge 谓词完全被前缀覆盖——稳态（零标记行）索引扫描即空结果。
               索引变更混入表指纹，旧布局由装配期显式拒绝，
               沿用 367dc4517 惯例不升 Version */ \
            Index(device_id, is_freed))) \
    Table(sim::runtime::IpcMemRecord, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(vir_mem_id), Field(offset), Field(create_pid), \
        PrimaryKey(id))) \
    Table(sim::runtime::IpcMemWhiteList, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(name_or_key), Field(pid), Field(create_pid), \
        PrimaryKey(id))) \
    Table(sim::runtime::FdMemRecord, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(vir_mem_id), Field(phy_mem_id), Field(create_pid), \
        PrimaryKey(id))) \
    Table(sim::runtime::FdMemWhiteList, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(name_or_key), Field(pid), Field(create_pid), \
        PrimaryKey(id))) \
    Table(sim::runtime::Task, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(stream_id), Field(cid), Field(seq_number), Field(type), \
        PrimaryKey(id))) \
    Table(sim::runtime::EventSyncTask, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(execute_time_ms), Field(finish_time_ms), Field(op_timeout_s), \
        PrimaryKey(id))) \
    Table(sim::runtime::Port, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(device_id), Field(die_id), Field(name, Text), Field(status), \
        PrimaryKey(id))) \
    Table(sim::runtime::Ccu, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(device_id), Field(resource_addr), Field(die_id), Field(status), \
        PrimaryKey(id), \
            Index(device_id, die_id))) \
    Table(sim::runtime::DeviceConnection, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(src_dev_id), Field(dst_dev_id), Field(link_type), \
        Field(access_by_remote), \
        PrimaryKey(id), \
            Index(src_dev_id, dst_dev_id))) \
    Table(sim::runtime::EndPoint, (Version(4), Scope::Static, \
        Field(id, AutoIncrement), Field(device_id), Field(func_id), Field(die_id), Field(type), \
        Field(eid, Blob), Field(ip_addr, Text), Field(status), Field(is_uboe), \
        PrimaryKey(id), \
            Index(ip_addr), Index(eid), Index(device_id, status))) \
    Table(sim::runtime::EndPointPair, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(local_enpoint_id), Field(remote_enpoint_id), Field(tp_type), \
        PrimaryKey(id), \
            Index(local_enpoint_id, remote_enpoint_id))) \
    Table(sim::runtime::EndPointPortMapping, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(port_id), Field(endpoint_id), Field(net_layer), \
        PrimaryKey(id))) \
    Table(sim::runtime::Link, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(local_endpoint_id), Field(remote_endpoint_id), Field(net_layer), \
        Field(type), Field(protocols, Blob), \
        PrimaryKey(id))) \
    Table(sim::runtime::CcuChannel, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(channel_id), Field(local_endpoint_id), Field(remote_endpoint_id), \
        Field(protocol), Field(jetty_start), Field(jetty_num), \
        PrimaryKey(id))) \
    Table(sim::runtime::TaskSchedulerDevice, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(device_id), Field(type), \
        PrimaryKey(id), \
            Index(device_id, type))) \
    Table(sim::runtime::ComputeDie, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(ts_id), Field(type), \
        PrimaryKey(id), \
            Index(ts_id, type))) \
    Table(sim::runtime::RaSocketPair, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(server_id), Field(client_id), Field(ref_cnt), Field(port), \
        Field(tag_hash), Field(buf_status), Field(slot_idx), \
        PrimaryKey(id), \
            Index(tag_hash), \
            Index(client_id, server_id, tag_hash, port), Index(client_id, server_id, tag_hash, buf_status))) \
    Table(sim::runtime::RaDevice, (Version(3), Scope::Static, \
        Field(id, AutoIncrement), Field(device_id), Field(mac_addr, Blob), Field(state), \
        Field(endpoint_id), \
        PrimaryKey(id), \
            Index(device_id))) \
    Table(sim::runtime::RaQP, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(ra_dev_id), Field(send_cq_handle), Field(recv_cq_handle), \
        Field(qp_num), Field(type), Field(state), Field(taJettyId), Field(mode), Field(peer_qp_id), \
        Field(peer_qpn), Field(perr_lid), Field(pid), \
        PrimaryKey(id))) \
    Table(sim::runtime::RaCQ, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(ra_dev_id), Field(cqn), Field(size), \
        PrimaryKey(id))) \
    Table(sim::runtime::RaCQE, (Version(3), Scope::Static, \
        Field(id, AutoIncrement), Field(cq_handle), Field(wr_id), Field(status), \
        PrimaryKey(id), \
            Index(cq_handle, status))) \
    Table(sim::runtime::RaMR, (Version(3), Scope::Static, \
        Field(id, AutoIncrement), Field(local_key), Field(remote_key), Field(vptr_id), Field(length), \
        Field(addr), \
        PrimaryKey(id), \
            Index(local_key), Index(vptr_id))) \
    Table(sim::runtime::MemoryLayout, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(rank_id), Field(base_addr), Field(buf_type), Field(reserved), \
        Field(size), Field(global_offset), \
        PrimaryKey(id))) \
    Table(sim::runtime::SimModelData, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(rank_id), Field(src_rank), Field(dst_rank), Field(root), \
        Field(rank_size), Field(chip_type), Field(op_type), Field(reduce_op), Field(data_type), \
        Field(data_count), Field(op_expansion_mode), Field(ccu0_resource_base_addr), \
        Field(ccu1_resource_base_addr), Field(vDataDes, Blob), Field(all2AllDataDes, Blob), \
        PrimaryKey(id))) \
    Table(sim::runtime::IpcNotify, (Version(3), Scope::Static, \
        Field(id, AutoIncrement), Field(notify_id), Field(name_or_key, Text), Field(create_pid), \
        PrimaryKey(id), \
            Index(notify_id), Index(name_or_key))) \
    Table(sim::runtime::IpcNotifyVistorList, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(ipc_id), Field(visitor_pid), \
        PrimaryKey(id))) \
    Table(sim::runtime::NotifyRecordTask, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(notify_id), \
        PrimaryKey(id))) \
    Table(sim::runtime::NotifyWaitTask, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(notify_id), \
        PrimaryKey(id))) \
    Table(sim::runtime::RaContext, (Version(3), Scope::Static, \
        Field(id, AutoIncrement), Field(device_id), Field(mode), Field(endpoint_id), Field(eidIndex), \
        Field(max_jetty_num), Field(max_jfc_num), \
        PrimaryKey(id), \
            Index(endpoint_id))) \
    Table(sim::runtime::RaChan, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(ctx_handle), Field(chann_id), Field(mode), \
        PrimaryKey(id))) \
    Table(sim::runtime::RaTokenId, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(ctx_handle), Field(token_id), Field(ref_count), \
        PrimaryKey(id))) \
    Table(sim::runtime::RaTp, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(ctx_handle), Field(tp_type), Field(tpn), Field(speed), \
        Field(status), \
        PrimaryKey(id))) \
    Table(sim::runtime::RaRmem, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(ctx_handle), Field(remote_key), Field(target_seg_handle), \
        Field(remote_eid), \
        PrimaryKey(id))) \
    Table(sim::runtime::RaLmem, (Version(3), Scope::Static, \
        Field(id, AutoIncrement), Field(ctx_handle), Field(addr), Field(size), Field(mem_key), \
        Field(token_id), Field(endpoint_id), \
        PrimaryKey(id))) \
    Table(sim::runtime::RaJetty, (Version(5), Scope::Static, \
        Field(id, AutoIncrement), Field(ctx_handle), Field(send_cq_handle), Field(recv_cq_handle), \
        Field(sqDepth), Field(rqDepth), Field(sqBuffer), Field(sqBufType), Field(type), Field(jetty_id), Field(dieId), \
        Field(rank_id), Field(state), Field(mode), Field(peer_jetty_handle), Field(peer_endpoint_id), Field(pid), \
        PrimaryKey(id), \
            Index(ctx_handle, mode), Index(jetty_id, pid, dieId))) \
    Table(sim::runtime::RaJfc, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(ctx_handle), Field(jfc_id), Field(depth), Field(mode), \
        Field(policy), \
        PrimaryKey(id))) \
    Table(sim::runtime::RaCr, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(jfc_handle), Field(user_ctx), Field(status), Field(opcode), \
        Field(byte_len), \
        PrimaryKey(id))) \
    Table(sim::runtime::RaTlv, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(physical_id), \
        PrimaryKey(id))) \
    Table(sim::runtime::RunModeConfig, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(mode), \
        PrimaryKey(id))) \
    Table(sim::runtime::RankEidMap, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(rank_id), Field(eid, Blob), \
        PrimaryKey(id), \
            Index(eid))) \
    Table(sim::runtime::Communicator, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(comm_id, Text), Field(rank_size), Field(rank_id), \
        Field(device_id), Field(comm_hash), Field(deterministic), Field(op_expansion_mode), \
        Field(rdma_traffic_class), Field(rdma_service_level), Field(sym_win_addr), \
        Field(sym_win_size), Field(sym_win_registered), Field(status), Field(dpu_mem_addr), Field(dpu_mem_size), \
        PrimaryKey(id), \
            Index(comm_id, comm_hash), Index(device_id), \
            Index(comm_id, comm_hash, device_id))) \
    Table(sim::runtime::CommunicatorDestroySync, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(comm_id, Text), Field(comm_hash), Field(rank_size), \
        Field(rank_id), \
        PrimaryKey(id), \
            Index(comm_id, comm_hash))) \
    Table(sim::runtime::TopoMetaConfig, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(file_name, Text), \
        PrimaryKey(id))) \
    Table(sim::runtime::Plugin, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(tag, Text), \
        PrimaryKey(id))) \
    Table(sim::runtime::HcclBuffer, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(commId), Field(addr), Field(size), \
        PrimaryKey(id), \
            Index(commId))) \
    Table(sim::runtime::HcclThread, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(commId), Field(engine), Field(notifyNum), \
        Field(notifyId, Blob), Field(streamId), \
        PrimaryKey(id), \
            Index(commId))) \
    Table(sim::runtime::HcclChannel, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(commId), Field(endpoint_id), Field(engine), \
        Field(remoteRankId), Field(notifyNum), Field(notifyId, Blob), Field(status), \
        PrimaryKey(id), \
            Index(commId), Index(endpoint_id))) \
    Table(sim::runtime::HcommEndpoint, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(runner_id), Field(south_endpoint_id), Field(protocol), \
        Field(addr_type), Field(addr, Blob), Field(loc_type), Field(loc, Blob), \
        Field(extension, Blob), Field(bind_state), \
        PrimaryKey(id), \
            Index(runner_id))) \
    Table(sim::runtime::HcommMemReg, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(endpoint_id), Field(runner_id), Field(source_endpoint_id), \
        Field(source_mem_id), Field(descriptor_key), Field(addr), Field(size), Field(mem_type), \
        Field(record_kind), Field(mem_tag, Text), \
        PrimaryKey(id), \
            Index(endpoint_id), Index(runner_id))) \
    Table(sim::runtime::HcclEngineCtx, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(commId), Field(ctxTag), Field(engine), Field(size), \
        Field(addr), \
        PrimaryKey(id), \
            Index(commId))) \
    Table(sim::runtime::HcclMem, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(commId), Field(memTag), Field(memType), Field(addr), \
        Field(size), Field(mem_tag_str, Text), \
        PrimaryKey(id), \
            Index(commId), Index(memTag))) \
    Table(sim::runtime::DpuDeviceInfo, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(device_id), Field(stream_id), \
        PrimaryKey(id), \
            Index(device_id))) \
    Table(sim::runtime::DpuPendingNotify, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(dstDeviceId), Field(notifyId), Field(srcDeviceId), \
        Field(immData), Field(consumed), \
        PrimaryKey(id), \
            Index(dstDeviceId))) \
    Table(sim::runtime::CcuSyncResTab, (Version(2), Scope::Static, \
        Field(id, AutoIncrement), Field(opIter), Field(round), Field(srcRank), Field(targetRank), \
        Field(connOrdinal), Field(dieId), Field(resType), Field(resId), \
        Field(value), Field(ownerPid), \
        PrimaryKey(id)))

// 发布 Runner 逻辑库目录：展开全部 65 个表条目并派生只读 manifest（表数量/
// fingerprint/类型擦除通道均由 catalog 派生，不另写第二份清单）。
#define HCCL_VM_DDL_CURRENT_DATABASE "RUNNER_DB"
HCCL_VM_DEFINE_DATABASE(kRunnerDatabase, RUNNER_DB, HCCL_VM_RUNNER_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE
#undef HCCL_VM_RUNNER_DATABASE

#undef OnDelete
#undef Derived
#undef Blob
#undef Text
#undef AutoIncrement
#undef Scope
#undef ForeignKey
#undef Index
#undef Unique
#undef PrimaryKey
#undef Version
#undef Field
// clang-format on

namespace HcclSim::Storage::Ddl {
// VirtualMemBlock 记录语义 hook：两个区间端点为派生列（base+size），
// 编码时计算并入库供包含查询下推；解码忽略（基址/大小为权威）。
template <>
struct RecordCodecHooks<sim::runtime::VirtualMemBlock> : DefaultRecordCodecHooks<sim::runtime::VirtualMemBlock> {
    /**
     * @brief 计算派生列的入库值（两个区间端点 = 基址 + size）。
     * @param record  输入：待编码的 VirtualMemBlock 记录。
     * @param fieldId 输入：字段编号（13=start_end_ptr，14=dev_mapped_end_ptr）。
     * @return 派生列的值；非派生字段返回 std::nullopt。
     */
    static std::optional<DbScalar> EncodeDerivedField(const sim::runtime::VirtualMemBlock& record, uint16_t fieldId)
    {
        if (fieldId == 13) { // start_end_ptr = start_ptr + size
            return DbScalar{record.start_ptr + record.size};
        }
        if (fieldId == 14) { // dev_mapped_end_ptr = dev_mapped_ptr + size
            return DbScalar{record.dev_mapped_ptr + record.size};
        }
        return std::nullopt;
    }

    /**
     * @brief 判断字段在解码（读取）时是否忽略（派生列不回写记录，基址/大小为权威）。
     * @param fieldId 输入：字段编号（13/14 为两个派生区间端点）。
     * @return true 表示解码时跳过该字段。
     */
    static bool IgnoreDecodedField(uint16_t fieldId) { return fieldId == 13 || fieldId == 14; }
};
} // namespace HcclSim::Storage::Ddl

namespace sim::runtime {
// ==== 薄兼容接口分区（sim::runtime::Db；一对一转发 Storage 一次性原子接口）
// ==== 恢复旧 RunnerDB 的
// Add/GetById/GetOneByPred/GetByPred/Count/Update/Delete/DeleteAll
// 使用习惯。固定映射：Add->Insert、GetById->SelectOne(Eq(&T::id, id))、
// GetOneByPred->SelectOne(固定 id ASC)、GetByPred->Select(固定 id ASC)、
// Count->Count、Update->Update、Delete->Delete、DeleteAll->Clear。
// 每个接口只转调一次现有原子接口：不缓存、不重试、不改事务边界、不吞错误、不写日志、
// 不安装会话也不选择数据库/后端。Pred 表示数据库可下推的类型化 Condition<T>，
// 不是迁移前的 std::function/lambda 内存谓词（全表 SELECT 后逐行 C++
// 过滤不回归）。 GetOneByPred/GetByPred 只接收 Condition<T>：默认排序（主键 id
// 升序）与单行/多行语义 是接口内部固定合同，调用方不传 OrderBy/Limit
// 等任何子句；需要自定义排序或 数量限制时直接使用
// HcclSim::Storage::SelectOne/Select 原子接口。 Update
// 完美转发变更（事务句柄与返回类型不变）；DeleteAll 必须显式给出 ClearMode。
// 需要事务或同一张表多次操作复用句柄的复合流程由实现边界文件（runtime_state
// 两个 db_sim_runner_*.cc、operation_data_ops.cc）持有
// AcquireTable<T>()/BeginTransaction()； 普通业务代码不获取表句柄、不开启事务。
namespace Db {
    /// 新增单条记录（一对一转发 Insert<T>；自增主键写回 record）。
    template <typename T>
    HcclSim::Storage::DbResult<HcclSim::Storage::PrimaryKey<T>> Add(T& record)
    {
        return HcclSim::Storage::Insert<T>(record);
    }

    /// 按主键取单条记录（一对一转发 SelectOne<T>(Eq(&T::id, id))；未命中
    /// NOT_FOUND）。
    template <typename T>
    HcclSim::Storage::DbResult<T> GetById(uint64_t id)
    {
        return HcclSim::Storage::SelectOne<T>(HcclSim::Storage::Eq(&T::id, id));
    }

    /// 条件查询单条记录（兼容入口；固定主键 id 升序取首条匹配，未命中 NOT_FOUND）。
    /// 只接收类型化 Condition<T>（Pred 为可下推数据库端的条件，不是 std::function
    /// 谓词）； OrderBy(id ASC) 是接口内部固定语义，不接受 OrderBy/Limit
    /// 等子句参数； 内部只调用一次
    /// Storage::SelectOne<T>，不加载全表、不在内存筛选。
    template <typename T>
    HcclSim::Storage::DbResult<T> GetOneByPred(const HcclSim::Storage::Condition<T>& where)
    {
        return HcclSim::Storage::SelectOne<T>(
            where, HcclSim::Storage::OrderBy(&T::id, HcclSim::Storage::SortDirection::ASC));
    }

    /// 条件查询多行（兼容入口；返回全部匹配记录，固定主键 id 升序）。
    /// 只接收类型化 Condition<T>，不接受 OrderBy/Limit 等子句参数；全表读取传
    /// HcclSim::Storage::All<T>()；内部只调用一次
    /// Storage::Select<T>，不加载全表后筛选。
    template <typename T>
    HcclSim::Storage::DbResult<std::vector<T>> GetByPred(const HcclSim::Storage::Condition<T>& where)
    {
        return HcclSim::Storage::Select<T>(
            where, HcclSim::Storage::OrderBy(&T::id, HcclSim::Storage::SortDirection::ASC));
    }

    /// 条件计数（一对一转发 Count<T>；数据库端聚合，不加载行集）。
    template <typename T>
    HcclSim::Storage::DbResult<uint64_t> Count(const HcclSim::Storage::Condition<T>& where)
    {
        return HcclSim::Storage::Count<T>(where);
    }

    /// 条件更新（一对一转发 Update<T>；变更完美转发，Transaction&
    /// 与即时返回类型不变）。
    template <typename T, typename... Changes>
    auto Update(const HcclSim::Storage::Condition<T>& where, Changes&&... changes)
        -> decltype(HcclSim::Storage::Update<T>(where, std::forward<Changes>(changes)...))
    {
        return HcclSim::Storage::Update<T>(where, std::forward<Changes>(changes)...);
    }

    /// 条件删除（一对一转发 Delete<T>；affectedRows 原样返回）。
    template <typename T>
    HcclSim::Storage::DbResult<uint64_t> Delete(const HcclSim::Storage::Condition<T>& where)
    {
        return HcclSim::Storage::Delete<T>(where);
    }

    /// 清空全表（一对一转发 Clear<T>；ClearMode 必须显式给出，无隐藏默认值）。
    template <typename T>
    HcclSim::Storage::DbResult<void> DeleteAll(HcclSim::Storage::ClearMode mode)
    {
        return HcclSim::Storage::Clear<T>(mode);
    }
} // namespace Db

// ==== Device / Server / Topology 分区 ====

uint64_t GetServerKeyById(uint32_t superPodIdx, uint32_t serverIdx);

// ==== Context / Stream / Task 分区 ====

void SetLastStreamIdTls(uint64_t streamId);
void SetLastTaskIdTls(uint64_t taskId);

uint64_t GetLastStreamIdTls();
uint64_t GetLastTaskIdTls();

/// 通信域 rank 总数（取自当前线程 Communicator 表行的 rank_size；
/// Rank 表已随 CheckerL2 基线移除；未设置当前通信域或查询失败按 0 返回）。
uint32_t GetRankSize();

/// host 总数（迁移前一行业务函数；内部一次数据库端 COUNT 聚合，
/// 不加载行集；失败按 0 返回）。
uint32_t GetHostSize();

// ==== Runtime State / Lifecycle 分区 ====

bool GetCurrRunnerTls(uint64_t serverKey, Runner& runner);
bool SetCurrCtxTls(uint64_t ctx);
uint64_t GetCurrDeviceId();
uint64_t GetCurrDeviceKey();
uint64_t GetDeviceIdByCtxId(uint64_t ctxId);

void SetTsDevice(int tsId);

// 设置运行模式：清空运行模式单行表（重置自增序列）后写入新模式
// （start/run 命令重复的 Clear + Insert 复合流程的领域封装）。
// mode 取 RUN_MODE_* 常量（normal / check-only / estimate）。
bool SetRunMode(uint8_t mode);
} // namespace sim::runtime
#endif
