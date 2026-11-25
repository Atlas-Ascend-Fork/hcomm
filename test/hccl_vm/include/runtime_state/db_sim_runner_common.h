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
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * https://www.huawei.com/en/software/cann/license
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef _SIM_RUNNER_COMMOM_H_
#define _SIM_RUNNER_COMMOM_H_

#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl_base.h"
#include "runtime_state/db_sim_communicator.h"
#include "runtime_state/sim_models.h"
#include "sim_common_defs.h"
#include "sim_ip_address.h"

namespace sim {
namespace devspec {
    struct DeviceSpec;
}
} // namespace sim

namespace sim::runtime {
extern thread_local uint64_t g_cur_device_key;
// 当前线程正在执行算子的本地 Communicator 表行 id; 它不是 HcclComm 句柄, 也不是整个通信域共用的 id.
extern thread_local uint64_t g_cur_comm_key;
// 本设备 DPU 线程的虚拟流槽位号：Launch 孵化时分配并快照传递（与 g_cur_comm_key 同机制）
extern thread_local uint32_t g_dpuThreadSlot;

// 通信域叶子节点信息: 由TopoMetaConfig表项指向的topo_meta配置解析生成, 按rankId分配顺序排列。
// 叶子节点 = server形态的server / POD形态的board(每board一个独立host节点, 对应一行Server表记录)
struct CommConfigServer {
    uint64_t serverKey{0};           // Server表主键(叶子节点)
    uint32_t podId{0};               // 超级节点id
    uint32_t serverId{0};            // server id(POD形态下为POD id)
    uint32_t boardId{0};             // board id(server形态恒为0)
    uint32_t devNum{0};              // 通信域内该叶子节点的device数
    uint32_t rankOffset{0};          // 该叶子节点首个rankId(前面叶子节点的device累计数)
    std::vector<PhyDeviceId> phyIds; // 叶子节点内按rank顺序的物理device id(POD形态即board内物理id)
};

// 进程级通信域配置缓存(TopoMetaConfig指向的配置解析结果)
struct CommConfigData {
    std::string fileName;                  // topo_meta yaml文件名; "ranktable"表示ranktable.json模式
    std::vector<CommConfigServer> servers; // 按(podId, serverId, boardId)升序, 与rankId分配顺序一致
};

bool GetOrInsertCommunicator(
    const char* commName, uint32_t rankSize, uint32_t rankId, uint64_t deviceId, uint64_t commHash, uint64_t& commId);
bool WaitCommunicatorReady(const char* commName, uint64_t commHash, uint32_t rankSize);
// estimate 专用: 单通信域防护。estimate 模式当前仅支持单通信域场景（多通信域下
// rankid+commid 与物理卡的对应关系、性能仿真支持等均有致命问题待解决）：同一 rank
// 已注册其他通信域（commName/commHash 不同）时拒绝再次注册并明确提示用户；
// 同一通信域重复初始化（幂等）放行。非 estimate 模式直接放行。
bool EnsureEstimateSingleCommDomain(const char* commName, uint64_t commHash, uint32_t rankId);
// level 1 专用: 通信域销毁栅栏, 全部 rank 到齐才放行
bool WaitCommunicatorDestroyReady(uint64_t commId);
// level 0 专用: 避免快卡提前销毁, 标记本 rank 已调用销毁接口并等待同通信域全部成员到达后再返回
bool MarkAndSyncCommunicatorDestroy(uint64_t commId);

aclError GetDeviceByLogicId(uint32_t deviceId, Device& device);
aclError GetDeviceByCommRank(uint64_t commId, uint32_t rankId, Device& device);
aclError GetDeviceByPhysicalId(uint32_t deviceId, Device& device);
aclError GetServerByKey(uint64_t serverKey, Server& server);
aclError UpdateDeviceLogicId(uint64_t serverKey, uint32_t phyDevId, uint32_t logicDevId, uint32_t userId);
aclError UpdateSuperDeviceId(uint32_t logicDevId, uint32_t superDeviceId);
aclError GetCcuFromDeviceByDieId(uint64_t deviceKey, uint8_t dieId, Ccu& ccu);

/*==================== device 规格查询（设计文档 §4.8.3）====================
 * 数据源是 config/device/*.yaml，由 InitClusterStaticTopoData 落库为运行时产物。
 * ★ 消费方必须实现回退链：DB 未命中（表空 / mock-comm 路径不经落库，见 §4.8.7 / O9）
 *   → 回退 DeviceSpecRegistry 直接读 yaml。
 */
/// 按 spec 名（= yaml 文件名主干）查头信息。
aclError GetDeviceSpecByName(const std::string& name, DeviceSpec& out);
/// 按 device 主键查其绑定的 spec 头信息（经 DeviceSpecBinding）。
aclError GetDeviceSpecByDeviceKey(uint64_t deviceKey, DeviceSpec& out);
/// 取某 spec 的全部区域行（按 offset 升序）。
bool GetDeviceSpecRegions(uint64_t specId, std::vector<DeviceSpecRegion>& out);
/// 取某 spec 的全部 MS bank 窗口行（按 ccua_index 升序）。
bool GetDeviceSpecMsWindows(uint64_t specId, std::vector<DeviceSpecMsWindow>& out);
/// 取某 spec 的全部 die 偏移行（按 die_id 升序）。
bool GetDeviceSpecDies(uint64_t specId, std::vector<DeviceSpecDie>& out);
/// 取某 spec 的全部分区段行（按 offset 升序）。
bool GetDeviceSpecPartitions(uint64_t specId, std::vector<DeviceSpecPartition>& out);
/// 取全部已落库的 spec 头信息。
bool GetAllDeviceSpecs(std::vector<DeviceSpec>& out);

/**
 * @brief 取某 (device, die) 的 CCU 资源空间基址 —— **全项目唯一入口**。
 *
 * 取代此前散落在 8 处的 `0x123123123` / `0x456456456` 硬编码（设计文档 §2.2）。
 *
 * 解析顺序：
 *   ① DB `Ccu.resource_addr`（拓扑初始化时由 FillCcu 按 device 规格写入，per-rank per-die 唯一）
 *   ② 回退：按 `Device.soc_version` → DeviceSpecRegistry 读 yaml 推导
 *      （mock-comm 路径不经 InitClusterStaticTopoData，DB 无行，见 §4.8.7 / O9）
 *
 * @param deviceKey Device 表主键
 * @param dieId     die 序号
 * @param baseAddr  [out] CCU 资源空间基址，如 rank0 die0 = 0x8e00000000
 * @return 两条路径都拿不到时返回 false（调用方须显式处理，不得回退到硬编码常量）
 *
 * ★ 成功结果按 (deviceKey, dieId) 进程级缓存，重复调用不产生 SQL（见 ResetDeviceSpecCache）。
 */
bool GetCcuResourceBaseAddr(uint64_t deviceKey, uint32_t dieId, uint64_t& baseAddr);

/**
 * @brief 取某 device 绑定的 device 规格（yaml 权威源）—— CCU 资源规格的**唯一读取入口**。
 *
 * 取代此前散落在 proxy level1 / runner / checker 三侧、各自硬编码的容量表与区域偏移表
 * （`CCU_V1_CAPACITY`/`CCU_V2_CAPACITY`/`V1_*_OFFSET`/`XN_*_ADDR_A6`/`CCU_MS_START_ADDR` 等，
 * 设计文档 §2.5 / §4.5.4-4.5.6）。容量、区域偏移/步长、MS 交织几何全部由本规格的
 * regions/ms 派生，不再出现字面量。
 *
 * 解析顺序（与 GetCcuResourceBaseAddr 同源）：
 *   ① DB `DeviceSpecBinding` → `DeviceSpec`（拓扑初始化时由 PersistDeviceSpecs 落库）
 *   ② 回退：按 `Device.soc_version` → DeviceSpecRegistry 读 yaml（mock-comm 路径不落库）
 *
 * @param deviceKey Device 表主键
 * @return 命中返回规格指针（进程级常驻、只读，可安全长期持有）；两条路径都拿不到返回 nullptr，
 *         调用方须显式处理，**不得回退到任何硬编码常量**（设计文档 §2.2）。
 *
 * ★ 本函数与 GetCcuResourceBaseAddr 的结果均进程级缓存（见 ResetDeviceSpecCache 注释），
 *   重复调用不产生 SQL。
 */
const sim::devspec::DeviceSpec* GetDeviceSpec(uint64_t deviceKey);

/**
 * @brief 清空 device 规格解析缓存（Device.soc_version 与 (device,die) → CCU 基址）。
 *
 * 缓存前提：两者都是 `Scope::Static` 数据 —— 由 InitClusterStaticTopoData 在 start 阶段
 * 一次性写入，运行期不再变更；而 `Db::` 薄层按其合同**不缓存**（db_sim_runner_ops.h），
 * 每次 GetById/GetOneByPred 都是一条 SQL。规格解析又处在 CCU 的热点路径上
 * （proxy L1 每次资源分配 / 每次取址、A5 每条通道、checker 每卡启动解析），
 * 故在此按 key 常驻，把重复解析降为 0 次读库（设计文档 §4.5.5/§4.5.6 的"规格只解析一次"口径）。
 *
 * 需要显式清空的场景：① 同一进程内重新初始化拓扑；② 经 `hccl-vm table update` 改了
 * Device.soc_version；③ UT 重置 DB / DeviceSpecRegistry。
 */
void ResetDeviceSpecCache();

aclError GetContextByDevId(uint32_t deviceId, Context& context);
aclError GetPortByName(uint64_t serverKey, uint32_t phyDevId, const std::string& name, Port& port);
aclError GetEndPointByIpAddr(const std::string& ip, EndPoint& endPoint);
aclError GetEndPointByEid(const IpAddress& addr, EndPoint& endPoint);
// EID精确匹配失败时, 若为v4映射EID(高12字节全0+低4字节IPv4, UBOE IPv4链路地址派生),
// 按低32位IPv4回退匹配EndPoint.ip_addr(与真实驱动v4->EID解析语义一致)
aclError GetEndPointByEidOrIpV4(const IpAddress& addr, EndPoint& endPoint);
// 按EID查ranktable.json记录的rankId(RankEidMap表); estimate模式Communicator表无对端行, 对端rank以此为准
bool GetRankIdByEid(const IpAddress& addr, uint32_t& rankId);
aclError GetPortById(uint64_t portId, Port& port);

uint32_t GetAICpuCount(uint64_t deviceId);
uint32_t GetAICoreCount(uint64_t deviceId);
uint32_t GetVectorCoreCount(uint64_t deviceId);
uint32_t GetCubeCoreCount(uint64_t deviceId);
std::string GetHardwareTypeByDevice(uint64_t deviceId);
bool GetCommRankByDeviceId(uint64_t commId, uint64_t deviceId, uint32_t& rankId);
bool ResetAllDeviceLogicId();
bool GetRankIdByMPI(uint32_t& rankId, uint64_t& serverId);
uint64_t GetCurServerId();

// 公共解析函数: 解析topo_meta yaml(或ranktable模式下的data/ranktable.json)生成通信域server有序表.
// topoMeta非空时直接使用调用方已解析的结果, 避免重复读文件
bool ParseCommConfigData(const std::string& fileName, const TopoMeta* topoMeta, CommConfigData& data);
// mock-comm每次执行时调用: 覆盖TopoMetaConfig表项并刷新本进程通信域配置缓存
bool RefreshTopoMetaConfig(const std::string& fileName, const TopoMeta* topoMeta = nullptr);
// 通信域重置时清空进程缓存, 下次使用时按TopoMetaConfig表项重新加载
void ResetCommConfigData();
// 获取通信域配置中的总rank数，加载失败时返回0
uint32_t GetCommConfigTotalRankNum();
// rankId → 所属server的serverKey
bool GetServerKeyByRankId(uint32_t rankId, uint64_t& serverKey);
// serverKey + 物理device id → 通信域内device序号(与Device.logic_id一致)
aclError GetDevIndexByPhyId(uint64_t serverKey, uint32_t phyId, uint32_t& devIndex);

} // namespace sim::runtime

#endif // _SIM_RUNNER_COMMOM_H_
