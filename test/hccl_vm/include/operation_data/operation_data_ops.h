/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：OperationData（OP_DATA_DB）的统一逻辑 DDL 目录与 OpDB 领域操作声明。
// 8 张表在一个 master catalog 中登记（OpTaskTab 为 PID 动态 scope 表）；
// 每张表只有一个紧凑 Table entry，字段/属性/约束/索引/外键写在同一 entry，
// 公共生成器从这里派生全部表契约产物。短宏区段结束立即 #undef。
// 领域声明与 DDL 分区共存于本文件（唯一声明事实源，不设第三个声明入口）。
#ifndef _OPERATION_DATA_OPS_H_
#define _OPERATION_DATA_OPS_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "operation_data/operation_data_types.h"
#include "storage/storage_catalog.h"

// ==== OpDB 唯一 DDL 区（master catalog；短别名区段结束立即 #undef） ====
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

#define HCCL_VM_OPDATA_DATABASE(Table)                                                                                            \
    /* (deviceId,id)：设备维度最新算子定位（deviceId 过滤 + id 倒序取首条），                                \
       服务 QueryLatestOpDetailStreamByDeviceId；id 命中索引列，streamId 回表 */                                       \
    Table(                                                                                                                        \
        sim::operation::OpDetailTab,                                                                                              \
        (Version(1), Scope::Static, Field(id, AutoIncrement), Field(pid), Field(deviceId), Field(rankId),                         \
         Field(commId), Field(opIter), Field(syncIter), Field(streamId), Field(root), Field(opExpansionMode),                     \
         Field(devType), Field(rankSize), Field(srcRank), Field(dstRank), Field(opDetail), Field(opExtInfo),                      \
         Field(algorithmName, Text), PrimaryKey(id), Index(pid, syncIter), Index(commId, deviceId, id),                           \
         Index(syncIter), Index(deviceId, id)))                                                                                   \
        Table(                                                                                                                    \
            sim::operation::OpMemInfoTab,                                                                                         \
            (Version(1), Scope::Static, Field(id, AutoIncrement), Field(opDetailId), Field(inputAddr),                            \
             Field(inputSize), Field(outputAddr), Field(outputSize), Field(cclAddr), Field(cclSize), PrimaryKey(id),              \
             Index(opDetailId), ForeignKey(opDetailId, sim::operation::OpDetailTab::id, OnDelete::Cascade)))                      \
            Table(                                                                                                                \
                sim::operation::SyncRecordTab,                                                                                    \
                (Version(1), Scope::Static, Field(id, AutoIncrement), Field(pid), Field(rankId), Field(rankSize),                 \
                 Field(syncIter), Field(streamId), Field(status), PrimaryKey(id), Index(status)))                                 \
                Table(                                                                                                            \
                    sim::operation::CcuChannelTab,                                                                                \
                    (Version(1), Scope::Static, Field(id, AutoIncrement), Field(channelId), Field(srcDieId),                      \
                     Field(dstDieId), Field(srcDeviceId), Field(dstDeviceId), Field(srcRankId), Field(dstRankId),                 \
                     Field(leid, Blob), Field(reid, Blob), Field(protocol), Field(jettyNum), Field(jettyId, Blob),                \
                     PrimaryKey(id), Index(channelId)))                                                                           \
                    Table(                                                                                                        \
                        sim::operation::JettyMapTab,                                                                              \
                        (Version(1), Scope::Static, Field(id, AutoIncrement), Field(opDetailId), Field(srcDieId),                 \
                         Field(dstDieId), Field(srcRankId), Field(dstRankId), Field(leid, Blob), Field(reid, Blob),               \
                         Field(protocol), PrimaryKey(id), Index(opDetailId), Index(srcDieId, dstDieId)))                          \
                        Table(                                                                                                    \
                            sim::operation::CcuInstrResTab,                                                                       \
                            (Version(1), Scope::Static, Field(id, AutoIncrement), Field(deviceId), Field(dieId),                  \
                             Field(instrCount), Field(instrSpace, Blob), PrimaryKey(id), Index(deviceId, dieId)))                 \
                            Table(                                                                                                \
                                sim::operation::CcuInstrTab,                                                                      \
                                (Version(1), Scope::Static, Field(id, AutoIncrement), Field(ccuInstrResId),                       \
                                 Field(startId), Field(instrInfoSize), PrimaryKey(id),                                            \
                                 ForeignKey(                                                                                      \
                                     ccuInstrResId, sim::operation::CcuInstrResTab::id,                                           \
                                     OnDelete::Cascade))) /* (streamId,isDone)：任务完成度聚合（streamId 过滤 + isDone \
                                                             匹配计数）的 覆盖索引；独立 isDone                      \
                                                             索引仍服务只按完成态过滤的查询 */                     \
        Table(                                                                                                                    \
            sim::operation::OpTaskTab,                                                                                            \
            (Version(1), Scope::DynamicPid, Field(id, AutoIncrement), Field(pid), Field(opDetailId), Field(deviceId),             \
             Field(streamId), Field(taskType), Field(isDone), Field(optaskMeta), PrimaryKey(id), Index(opDetailId),               \
             Index(streamId, isDone), Index(isDone)))                                                                             \
            Table(                                                                                                                \
                sim::operation::HalfRTTTab, (Version(1), Scope::Static, Field(id, AutoIncrement), Field(deviceId),                \
                                             Field(dieId), Field(wishCntXnIdBegin), Field(wishCntXnIdEnd),                        \
                                             Field(totalCntId), PrimaryKey(id), Index(deviceId, dieId)))

// 发布 OperationData 逻辑库目录：展开全部 9 个表条目并派生只读 manifest
// （表数量/fingerprint/类型擦除通道均由 catalog 派生，不另写第二份清单）。
#define HCCL_VM_DDL_CURRENT_DATABASE "OP_DATA_DB"
HCCL_VM_DEFINE_DATABASE(kOpDataDatabase, OP_DATA_DB, HCCL_VM_OPDATA_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE
#undef HCCL_VM_OPDATA_DATABASE

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

namespace HcclSim::Storage::Ddl {
// CcuChannelTab 记录语义 hook：jettyId 只编码 jettyNum 个有效元素（有效数组长度
// 由同记录另一字段决定），不是固定数组全部容量。
template <>
struct RecordCodecHooks<sim::operation::CcuChannelTab> : DefaultRecordCodecHooks<sim::operation::CcuChannelTab> {
    static int64_t EffectiveBytesLength(const sim::operation::CcuChannelTab& record, uint16_t fieldId)
    {
        if (fieldId != 10) {
            return -1;
        }
        return static_cast<int64_t>(record.jettyNum) * static_cast<int64_t>(sizeof(record.jettyId[0]));
    }
};
} // namespace HcclSim::Storage::Ddl

// ==== OpDB 领域操作声明区（与实现文件分区顺序一致） ====
// 本域已有语义完整的业务接口（InsertOpDetail/Query.../Update... 等），不再建立
// 第二套 Db 薄门面；业务优先使用这些领域接口，简单或特殊场景仍可直接使用
// HcclSim::Storage 原子接口。后端、会话、lease、SQL 均不可见。
namespace sim::operation {
// 进程当前 OpDetail 序号（实现文件定义；device 进程经头文件声明读写，替代
// device 侧本地的 extern 声明——命名空间随本域收敛为 sim::operation）。
extern uint32_t g_currOpDetailId;
// 进程当前 OpMem 序号（实现文件定义；aclrt model 重放路径经头文件声明读写）。
extern uint32_t g_currOpMemId;

/**
 * @brief 当前进程正在处理的算子迭代（= 最近一次 InsertOpDetail 分配的 opIter）。
 *
 * 用途：CCU 交换表（CcuSyncResTab）需要跨 rank 一致的 op 身份来隔离不同算子的交换值。
 *       opIter 由 InsertOpDetail 按 (commName, commHash) 自增，两侧进程按同一算子序列
 *       推进时天然对齐，可直接当 op 身份使用。
 *
 * @return 最新分配的 opIter；尚无算子插入时为 0。
 */
uint32_t GetCurOpIter();

// ==== Stream Sync 迭代分区（纯进程内状态，不落库） ====
// 迁移前语义：以 streamId 为键维护各流独立的 sync 迭代计数。
void AddSyncStreamIter(uint64_t streamId);
void RemoveSyncStreamIter(uint64_t streamId);
bool NextStreamSyncIdx(uint64_t streamId, uint64_t& syncIdx);

// ==== Op Detail / Memory 分区 ====
int InsertOpDetail(OpDetailTab& rec);
int InsertOpMem(OpMemInfoTab& rec);
// OpDetail 与 OpMem 同事务原子提交（多表流程）。
int InsertOpDetailAndMem(OpDetailTab& detail, OpMemInfoTab& mem);
// 算法选择结果回填（日志截取）: 同一算子多次匹配时后写覆盖 (last-wins)，与最终生效算法一致
int UpdateOpDetailAlgName(const std::string& algName);
int QueryNewestOpDeatailIdByPid(uint64_t pid, uint32_t& OpDetailId);
// 同 syncIter 下目标 rank 的最新算子内存信息（组合查询）。
int QueryCurrentOpMemInfoByRank(uint32_t rankId, OpMemInfoTab& out);
// 按通信域成员 + 设备维度定位最新算子内存信息（迁移前 JOIN opDetails+opMemInfo 语义）。
int QueryCurrentOpMemInfo(uint64_t commId, uint32_t deviceId, OpMemInfoTab& out);
// 由 opDetailId 定位对应算子内存信息（图模式录制/重放复制内存上下文用）。
int QueryOpMemInfoByOpDetailId(uint32_t opDetailId, OpMemInfoTab& out);
int UpdateOpMemCclBuffer(uint64_t cclAddr, uint64_t cclSize);
int UpdateOpExpansionMode(uint8_t mode);
// 把指定算子的主流 streamId 回写（graph 模式流绑定）。
int UpdateOpDetailMainStream(uint32_t opDetailId, uint64_t streamId);
int QueryLatestOpExpansionMode();
int QueryCompositeOpDetailBySyncIter(uint32_t syncIter, std::map<uint32_t, std::vector<CompositeOpDetail>>& detail);
// 算子执行索引：全部 opDetails 的执行身份快照（id/deviceId/rankId/commId/opIter/rankSize）。
int QueryOpExecutionIndexEntries(std::vector<OpExecutionIndexEntry>& out);
// 算子明细身份解析：由 opDetailId 反查通信域成员/rank/设备。
int QueryOpDetailIdentity(uint32_t opDetailId, uint64_t& commId, uint32_t& rankId, uint32_t& deviceId);
// 按主键取完整算子明细行（含 opDetail/opExtInfo BLOB；图模式录制复制用）。
int QueryOpDetailById(uint32_t opDetailId, OpDetailTab& out);
// 设备维度最新算子明细（graph 模式主流发现；空表 WARN 不 ERROR）。
int QueryLatestOpDetailByDeviceId(uint32_t deviceId, OpDetailTab& out);
// 设备维度最新算子的主流摘要：只取 id/streamId 两列（不读 opDetail/opExtInfo
// BLOB）；排序（id 降序取首条）与空表/失败的 -1 + WARN 处理同完整行接口。
int QueryLatestOpDetailStreamByDeviceId(uint32_t deviceId, uint32_t& opDetailId, uint64_t& streamId);
// 按通信域 (commName, commHash) 解析全部成员，汇总各成员在指定 opIter 的复合算子明细
// （detail + memInfo + 动态 opTask 表任务），上游 QueryCompositeOpDetailByOpIter 语义。
int QueryCompositeOpDetailByOpIter(
    const std::string& commName, uint64_t commHash, uint32_t opIter, std::vector<CompositeOpDetail>& details);

// ==== Sync Record 分区 ====
int InsertSyncRecord(SyncRecordTab& rec);
int QuerySyncRecordAll(std::vector<SyncRecordTab>& out);
int QuerySyncRecordByStatus(uint8_t status, std::vector<SyncRecordTab>& out);
// 全部状态更新在同一事务内提交（批量流程）。
int UpdateSyncRecordStatus(std::vector<SyncRecordTab>& syncRecord);

// ==== CCU / Jetty 分区 ====
int InsertCcuChannel(CcuChannelTab& rec);
int InsertJettyMap(JettyMapTab& rec);
int QueryCcuChannelAll(std::vector<CcuChannelTab>& out);
int QueryJettyMapAll(std::vector<JettyMapTab>& out);
int InsertCcuInstrRes(CcuInstrResTab& rec);
int InsertCcuInstr(CcuInstrTab& rec);
// 按 device+die 读改写整块指令空间，不存在则插入（状态机入口）。
// mergedInstrSpace/totalInstrCount 非空时返回合并后的完整指令空间与总指令数：
// 更新路径取 max(已存指令数, startId + instrCount)，插入路径为 startId + instrCount。
int UpdateAndInsertByCcuId(
    uint64_t& ccuId, uint32_t deviceId, uint32_t dieId, uint32_t startId, uint32_t instrCount, uint32_t instrOffset,
    uint32_t instrInfoSize, const void* instrInfo, std::vector<uint8_t>* mergedInstrSpace = nullptr,
    uint32_t* totalInstrCount = nullptr);
int QueryCcuInstrResAll(std::vector<CcuInstrResTab>& out);
// 指令/通道资源元数据计数（轮询感知微码下发与资源行补齐的中间状态）。
int QueryCcuResourceMetaCount(uint32_t& instrLoadCnt, uint32_t& channelCnt);
// 0.5RTT 半程往返数据表。
int InsertHalfRTT(HalfRTTTab& rec);
int QueryHalfRTTAll(std::vector<HalfRTTTab>& out);

// ==== Dynamic Task 分区 ====
// 动态 PID scope 表：host 用 pid、AICPU 模式 device 进程用 ppid 解析物理表。
int InsertOpTask(OpTaskTab& rec, bool isDevice = false);
// 按当前进程 pid 解析动态表，按 streamId 查任务（迁移前 getpid() 语义）。
int QueryOpTasksByStreamId(uint64_t streamId, std::vector<OpTaskTab>& tasks);
// 同一聚合语句返回 stream 上任务总数与已完成数（isDone 为 true 的行数）：
// 统计在数据库端完成，不加载任务行（optaskMeta BLOB 不回传）。仅成功后
// 写入两个输出，失败输出不构成有效完成度。
int QueryOpTaskCompletionByStreamId(uint64_t streamId, uint64_t& total, uint64_t& done);
// 把指定任务在动态 opTask_P_<pid> 表中标记为完成（isDone=1）。
int FinishOpTask(const OpTaskTab& task);
// 枚举 OP_DATA_DB 全部动态 opTask 物理表名（迁移前 sqlite_master LIKE 'opTask_P_%' 语义）。
int QueryOpTaskTabNames(std::vector<std::string>& names);
// 汇总全部动态 opTask 表记录（filterDone 时仅未完成任务；迁移前 QueryAllOpTasks 语义）。
int QueryAllOpTasks(std::vector<OpTaskTab>& tasks, bool filterDone = false);

// ==== Lifecycle / Maintenance 分区 ====
// 确保进程存储会话已安装（解析安装根下的基线库路径并完成建表；幂等）。
// 迁移前语义：Host start/run 与 device 子进程在首次业务访问前的 fail-fast 建连时机。
int EnsureProcessStorageSession();

// 领域维护接口：换库（加载备份文件）、生成 OP_DATA_DB 物理备份、
// 清空全部静态操作数据表（保留自增序列）、销毁全部动态任务表。
int SetDbConfig(DBConfig& config);
int BackupOperationData(const std::string& destinationPath);
int ClearStaticData();
int DropDynamicData();
} // namespace sim::operation

#endif
