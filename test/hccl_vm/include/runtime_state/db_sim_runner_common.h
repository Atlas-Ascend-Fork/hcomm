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

namespace sim::runtime {
extern thread_local uint64_t g_cur_device_key;
// 当前线程正在执行算子的本地 Communicator 表行 id; 它不是 HcclComm 句柄, 也不是整个通信域共用的 id.
extern thread_local uint64_t g_cur_comm_key;

// 通信域server信息: 由TopoMetaConfig表项指向的topo_meta配置解析生成, 按rankId分配顺序排列
struct CommConfigServer {
    uint64_t serverKey{0};           // Server表主键
    uint32_t podId{0};               // 超级节点id
    uint32_t serverId{0};            // server id
    uint32_t devNum{0};              // 通信域内该server的device数
    uint32_t rankOffset{0};          // 该server首个rankId(前面server的device累计数)
    std::vector<PhyDeviceId> phyIds; // server内按rank顺序的物理device id
};

// 进程级通信域配置缓存(TopoMetaConfig指向的配置解析结果)
struct CommConfigData {
    std::string fileName;                  // topo_meta yaml文件名; "ranktable"表示ranktable.json模式
    std::vector<CommConfigServer> servers; // 按(podId, serverId)升序, 与rankId分配顺序一致
};

// 进程级通信域配置缓存: mock-comm执行时刷新, 其余进程首次使用时加载
extern CommConfigData g_comm_config_data;

bool GetOrInsertCommunicator(
    const char* commName, uint32_t rankSize, uint32_t rankId, uint64_t deviceId, uint64_t commHash, uint64_t& commId);
bool WaitCommunicatorReady(const char* commName, uint64_t commHash, uint32_t rankSize);
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
aclError GetContextByDevId(uint32_t deviceId, Context& context);
aclError GetPortByName(uint64_t serverKey, uint32_t phyDevId, const std::string& name, Port& port);
aclError GetEndPointByIpAddr(const std::string& ip, EndPoint& endPoint);
aclError GetEndPointByEid(const IpAddress& addr, EndPoint& endPoint);
// EID精确匹配失败时, 若为v4映射EID(高12字节全0+低4字节IPv4, UBOE IPv4链路地址派生),
// 按低32位IPv4回退匹配EndPoint.ip_addr(与真实驱动v4->EID解析语义一致)
aclError GetEndPointByEidOrIpV4(const IpAddress& addr, EndPoint& endPoint);
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
// mock-comm每次执行时调用: 覆盖TopoMetaConfig表项并刷新本进程g_comm_config_data
bool RefreshTopoMetaConfig(const std::string& fileName, const TopoMeta* topoMeta = nullptr);
// 通信域重置时清空进程缓存, 下次使用时按TopoMetaConfig表项重新加载
void ResetCommConfigData();
// rankId → 所属server的serverKey
bool GetServerKeyByRankId(uint32_t rankId, uint64_t& serverKey);
// serverKey + 物理device id → 通信域内device序号(与Device.logic_id一致)
aclError GetDevIndexByPhyId(uint64_t serverKey, uint32_t phyId, uint32_t& devIndex);

} // namespace sim::runtime

#endif // _SIM_RUNNER_COMMOM_H_
