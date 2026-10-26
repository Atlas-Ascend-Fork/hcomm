/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "CCU_STUB"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <utility>
#include <unistd.h>
#include <vector>
#include "sim_ccu_channel_ctx.h"
#include "ccu_common.h"
#include "sim_ccu_jetty_ctx.h"
#include "ccu_microcode_v1.h"
#include "ccu_u_comm.h"
#include "dtype_common.h"
#include "store_sim_store_pub.h"
#include <cstring>
#include "sim_capacity_limits.h"
#include "sim_log.h"
#include "sim_common_api.h"
#include "sim_yaml_config.h"
#include "hccp_common.h"
#include "hccp_ctx.h"
#include "hccp_tlv.h"
#include "sim_ip_address.h"
#include "rt_external_kernel.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "sim_common_defs.h"
#include "graph_capture.h"
#include "store_dump_shm_data.h"
#include "storage/table_access.h"
#include "hccl_proxy_common.h"

using sim::runtime::g_cur_comm_key;
using sim::runtime::g_cur_device_key;

extern uint64_t g_cur_server_key;
extern thread_local std::string g_currentOpFastLaunchTag;

using namespace HcclSim;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int GetEnableCcuDie(channel_info_out* output, uint8_t dieId)
{
    HCCL_VM_INFO("dieId:{}.", dieId);
    output->data.data_info.data_array[0].dieinfo.enable_flag = 1;
    return 0;
}

void SetCcuV1ResourceBasicInfo(channel_info_out* output, uint8_t dieId, uint32_t devId)
{
    // todo: 后续根据dieId，设置不同的resourceAddr
    if (dieId == 0) {
        output->data.data_info.data_array[0].baseinfo.resourceAddr = (void*)0x123123123;
    } else {
        output->data.data_info.data_array[0].baseinfo.resourceAddr = (void*)0x456456456;
    }
    output->data.data_info.data_array[0].baseinfo.missionKey = 0;
    output->data.data_info.data_array[0].baseinfo.ms_id = 3; //
    uint32_t instructionNum = 0x8000;                        // Instruction 32k
    uint32_t missionNum = 16;                                // Mission ctx 16
    uint32_t loopEngineNum = 200;                            // Loop ctx 200
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap0
        = (instructionNum - 1) | ((missionNum - 1) << MOVE_TOW_BYTES) | ((loopEngineNum - 1) << MOVE_THREE_BYTES);
    uint32_t gsaNum = 3072; // GSA 3072
    uint32_t xnNum = 3072;  // Xn 3072
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap1 = ((xnNum - 1) << MOVE_TOW_BYTES) | (gsaNum - 1);
    uint32_t ckeNum = 1024; // Checlist Entry(CKE) 1024
    uint32_t msNum = 1536;  // MemorySlice(MS) 1536
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap2 = ((msNum - 1) << MOVE_TOW_BYTES) | (ckeNum - 1);
    uint32_t channelNum = 128; // Channel map 128 for v1
    uint32_t jettyNum = 128;   // Jetty context 128
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap3
        = ((jettyNum - 1) << MOVE_TOW_BYTES) | (channelNum - 1);
    uint32_t pfeNum = 16; // PFE 16 for v1
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap4 = (pfeNum - 1) & 0x000000FF;
}

void SetCcuV2ResourceBasicInfo(channel_info_out* output, uint8_t dieId, uint32_t devId)
{
    // todo: 后续根据dieId，设置不同的resourceAddr
    if (dieId == 0) {
        output->data.data_info.data_array[0].baseinfo.resourceAddr = (void*)0x123123123;
    } else {
        output->data.data_info.data_array[0].baseinfo.resourceAddr = (void*)0x456456456;
    }
    output->data.data_info.data_array[0].baseinfo.missionKey = 0;
    output->data.data_info.data_array[0].baseinfo.ms_id = 3; //
    uint32_t instructionNum = 0x8000;                        // Instruction 32k
    uint32_t missionNum = 16;                                // Mission ctx 16
    uint32_t loopEngineNum = 512;                            // Loop ctx 512
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap0
        = ((instructionNum - 1) & 0xFFFF) | (((missionNum - 1) & 0xF) << 16) | (((loopEngineNum - 1) & 0xFFF) << 20);
    uint32_t gsaNum = 0;    // GSA 0 for v2
    uint32_t xnNum = 16384; // Xn 16384
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap1 = ((xnNum - 1) << MOVE_TOW_BYTES);
    uint32_t ckeNum = 1024; // Checlist Entry(CKE) 1024
    uint32_t msNum = 1536;  // MemorySlice(MS) 1536
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap2 = ((msNum - 1) << MOVE_TOW_BYTES) | (ckeNum - 1);
    uint32_t channelNum = 1024; // Channel map 1024 for v2
    uint32_t jettyNum = 128;    // Jetty context 128
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap3
        = ((jettyNum - 1) << MOVE_TOW_BYTES) | (channelNum - 1);
    uint32_t pfeNum = 20; // PFE 20 for v2
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap4 = (pfeNum - 1) & 0x000000FF;
}

int SetCcuResourceBasicInfo(channel_info_out* output, uint8_t dieId, uint32_t devId)
{
    sim::runtime::Device device{};
    if (sim::runtime::GetDeviceByPhysicalId(devId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by logic id {} failed.", devId);
        return -1;
    }
    if (strncmp(device.soc_version, "Ascend950", strlen("Ascend950")) == 0) {
        SetCcuV1ResourceBasicInfo(output, dieId, devId);
    } else if (strncmp(device.soc_version, "Ascend960", strlen("Ascend960")) == 0) {
        SetCcuV2ResourceBasicInfo(output, dieId, devId);
    } else {
        HCCL_VM_ERROR("unknown device version: {} for devId: {}", device.soc_version, devId);
        return -1;
    }
    return 0;
}
extern void* GetRealPtrByAddr(const void* devPtr);

void DumpInstrDecToFile(uint32_t instrCnt, const hcomm::CcuRep::CcuInstr* instrData, const std::string& fileName)
{
    std::ofstream ofs(fileName, std::ios::out | std::ios::trunc);
    ofs << "ccu total instruction number: " << instrCnt << "\n";
    for (uint32_t idx = 0; idx < instrCnt; idx++) {
        ofs << "[InstrData][ " + std::to_string(idx) + "]" + hcomm::CcuRep::ParseInstr(&instrData[idx]) + "\n";
    }
    return;
}

void DumpCcuSqeToFile(
    uint32_t startId, uint32_t instrCnt, uint32_t argSize, uint64_t args[], const std::string& fileName)
{
    std::ofstream ofs(fileName, std::ios::out | std::ios::trunc);
    ofs << "ccu sqe info: startInstrId= " << startId << ", instrCnt= " << instrCnt << ", argSize= " << argSize << "\n";
    for (uint32_t idx = 0; idx < argSize; idx++) {
        ofs << "[SQE Arg][" << idx << "]: " << args[idx] << "\n";
    }
    return;
}

namespace fs = std::filesystem;

bool write_or_overwrite_in_cwd(const std::string& filename, const std::string& data)
{
    fs::path target = fs::path(filename).is_absolute() ? fs::path(filename) : fs::current_path() / filename;

    std::error_code ec;
    bool exists = fs::exists(target, ec);
    if (ec) {
        // 读取状态出错，视为失败
        return false;
    }

    std::ofstream ofs;
    if (exists) {
        ofs.open(target, std::ios::out | std::ios::app);
    } else {
        ofs.open(target, std::ios::out | std::ios::trunc);
    }

    if (!ofs.is_open()) {
        return false;
    }

    if (exists) {
        ofs << "\n\n";
    }

    ofs << data;
    ofs.flush();
    return true;
}

// input->op == Hccl::CcuOpcodeType::CCU_U_OP_SET_INSTRUCTION
int LoadMicrocodeInstructionStub(uint32_t devId, uint8_t dieId, const channel_info_in* input)
{
    HCCL_VM_INFO("enter LoadMicrocodeInstruction .....");
    if (input == nullptr || dieId >= DIE_NUM) {
        HCCL_VM_ERROR("invalid microcode input or die id: {}", dieId);
        return -1;
    }

    sim::runtime::Device device{};
    if (sim::runtime::GetDeviceByPhysicalId(devId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by logic id {} failed.", devId);
        return -1;
    }
    sim::runtime::Ccu ccu{};
    if (sim::runtime::GetCcuFromDeviceByDieId(device.id, dieId, ccu) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get ccu from device by die id {} failed.", dieId);
        return -1;
    }
    const auto deviceId = static_cast<uint32_t>(device.id);
    const uint32_t startId = input->offset_start;
    const uint32_t instrInfoSize = input->data.data_info.data_array_len;
    constexpr uint32_t kInstructionSize = sizeof(hcomm::CcuRep::CcuInstr);
    constexpr uint32_t kInstructionCapacity = HcclSim::CCU_INSTRUCTION_NUM;
    // 先按剩余容量校验，避免后续计算 startId * kInstructionSize 时发生 32 位溢出。
    if (instrInfoSize % kInstructionSize != 0 || startId > kInstructionCapacity
        || instrInfoSize / kInstructionSize > kInstructionCapacity - startId) {
        HCCL_VM_ERROR(
            "invalid microcode range: startId={}, byteSize={}, instructionCapacity={}", startId, instrInfoSize,
            kInstructionCapacity);
        return -1;
    }
    const uint32_t instrCnt = instrInfoSize / kInstructionSize;

    uint64_t ccuId = ccu.id;
    const uint32_t instrOffset = startId * kInstructionSize;
    auto ccuDataTmp = (ccu_data_type_union)(input->data.data_info.data_array[0]);
    auto instructionData
        = reinterpret_cast<hcomm::CcuRep::CcuInstr*>(GetRealPtrByAddr((void*)ccuDataTmp.insinfo.resourceAddr));
    if (instructionData == nullptr) {
        HCCL_VM_ERROR("get ccu instruction data by resourceAddr failed  addr:0x{:x}", ccuDataTmp.insinfo.resourceAddr);
        return -1;
    }
    std::vector<uint8_t> mergedInstrSpace;
    uint32_t totalInstrCount = 0;
    if (sim::operation::UpdateAndInsertByCcuId(
            ccuId, deviceId, dieId, startId, instrCnt, instrOffset, instrInfoSize, instructionData, &mergedInstrSpace,
            &totalInstrCount)
        != 0) {
        HCCL_VM_ERROR("update ccu failed");
        return -1;
    }

    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    std::ostringstream fileName;
    fileName << "mc_instr_info_device_" << deviceId << "_die_" << static_cast<uint32_t>(dieId) << ".txt";
    std::ostringstream mcData;
    mcData << "ccu total instruction number: " << instrCnt << "\n";
    if (strncmp(device.soc_version, "Ascend950", strlen("Ascend950")) == 0) {
        for (uint32_t idx = 0; idx < instrCnt; idx++) {
            mcData << "[InstrData][ " + std::to_string(startId + idx) + "]"
                          + hcomm::CcuRep::ParseInstr(&instructionData[idx]) + "\n";
        }
    } else if (strncmp(device.soc_version, "Ascend960", strlen("Ascend960")) == 0) {
        for (uint32_t idx = 0; idx < instrCnt; idx++) {
            mcData << "[InstrData][ " + std::to_string(startId + idx) + "]"
                          + hcomm::CcuRep::CcuV2::ParseInstrV2(&instructionData[idx]) + "\n";
        }
    } else {
        HCCL_VM_ERROR("not support device soc version: {:s}", device.soc_version);
        return HCCL_E_NOT_SUPPORT;
    }

    auto status = write_or_overwrite_in_cwd(InstallPath::ResolveToInstallRoot("data/" + fileName.str()), mcData.str());

    // rankId 仅用于 GenCaModelCcuInstr 的 dump 文件名；解析失败时保持 0xFFFF 无效值并跳过 dump，
    // 不影响后续 InsertCcuInstr(distribute 微码到 runner)。
    uint32_t rankId = 0xFFFF;
    if (sim::runtime::g_cur_comm_key == 0) {
        // 算子入口线程未设置通信域(本 stub 可能被 HCCL 内部线程调用)，直接跳过 dump，不调 GetCommRankByDeviceId。
        HCCL_VM_WARN("CCU GenCaModelCcuInstr dump skipped: g_cur_comm_key not set, deviceId={}", deviceId);
    } else if (!sim::runtime::GetCommRankByDeviceId(sim::runtime::g_cur_comm_key, deviceId, rankId)) {
        HCCL_VM_WARN(
            "CCU GenCaModelCcuInstr dump skipped: cannot resolve rankId, commKey={}, deviceId={}",
            sim::runtime::g_cur_comm_key, deviceId);
    } else {
        // 生成指令文件
        if (GenCaModelCcuInstr(
                reinterpret_cast<hcomm::CcuRep::CcuInstr*>(mergedInstrSpace.data()), totalInstrCount, rankId, dieId)
            != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("gen instr file failed");
            return -1;
        }
    }

    sim::operation::CcuInstrTab instr{};
    instr.id = 0;
    instr.ccuInstrResId = ccuId;
    instr.startId = startId;
    instr.instrInfoSize = instrInfoSize;
    if (sim::operation::InsertCcuInstr(instr) != 0) {
        HCCL_VM_ERROR("insert instr failed");
        return -1;
    }
    return 0;
}

uint64_t GetRemoteCcuVa(const ChannelCtxDataV1& chDataTmp)
{
    uint64_t dstVa = 0;
    dstVa |= (uint64_t)(chDataTmp.dstVaLow & MASK_VA_LOW);                         // 低 8 位
    dstVa |= ((uint64_t)(chDataTmp.dstVaMiddle & MASK_VA_MID) << SHIFT_8BITS);     // 位 8-23
    dstVa |= ((uint64_t)(chDataTmp.dstVaHigh & MASK_VA_HIGH) << SHIFT_24BITS);     // 位 24-39
    dstVa |= ((uint64_t)(chDataTmp.dstVaHigher & MASK_VA_HIGHER) << SHIFT_40BITS); // 位 40+

    return dstVa << REMOTE_CCU_VA_RIGHT_SHIFT_NUM;
}

int GetLocalEndPointByJetty(uint64_t jettyId, uint16_t dieId, sim::runtime::EndPoint& endPoint)
{
    // CCU 与 AICPU 使用不同编号域。必须显式拒绝 AICPU 编号：仅按编号/pid/dieId
    // 查询不会自动隔离模式，误传时可能读到同一进程创建的 AICPU 行。
    if (HcclSim::IsAicpuJettyHwIdInRange(jettyId)) {
        HCCL_VM_ERROR("AICPU jetty {} cannot be used for a CCU channel", jettyId);
        return -1;
    }
    auto localJetty = sim::runtime::Db::GetOneByPred<sim::runtime::RaJetty>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::RaJetty::jetty_id, jettyId),
        HcclSim::Storage::Eq(&sim::runtime::RaJetty::pid, getpid()),
        HcclSim::Storage::Eq(&sim::runtime::RaJetty::dieId, dieId)));
    if (!localJetty.ok()) {
        HCCL_VM_ERROR("can not find jetty {} die:{} in local jetty map", jettyId, dieId);
        return -1;
    }

    auto ctxRes = sim::runtime::Db::GetById<sim::runtime::RaContext>(localJetty.value->ctx_handle);
    if (!ctxRes.ok()) {
        HCCL_VM_ERROR("can not find context {} in local context map", localJetty.value->ctx_handle);
        return -1;
    }

    auto localEp = ctxRes.value->endpoint_id;
    auto endPointOpt = sim::runtime::Db::GetById<sim::runtime::EndPoint>(localEp);
    if (!endPointOpt.ok()) {
        HCCL_VM_ERROR("can not find endpoint:{:d}", localEp);
        return -1;
    }
    endPoint = *endPointOpt.value;
    return 0;
}

// 配置channel信息：input->op == Hccl::CcuOpcodeType::CCU_U_OP_SET_CHANNEL
int ConfigChannelInfo(channel_info_in* input, uint32_t deviceId)
{
    sim::runtime::Device locDevice{};
    if (sim::runtime::GetDeviceByPhysicalId(deviceId, locDevice) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by physic id {} failed.", deviceId);
        return -1;
    }
    uint8_t dieId = input->data.data_info.udie_idx;
    uint32_t chId = input->offset_start;

    // 配置channel信息：input->op == Hccl::CcuOpcodeType::CCU_U_OP_SET_CHANNEL
    ChannelCtxDataV1 chDataTmp;
    memcpy(&chDataTmp, input->data.data_info.data_array, sizeof(struct ChannelCtxDataV1));
    Eid eid{};
    for (uint32_t i = 0; i < URMA_EID_LEN; i++) {
        eid.raw[i] = chDataTmp.eidRaw[URMA_EID_LEN - i - 1];
    }

    // 判断eid是否为全0
    static constexpr uint8_t zeroEid[URMA_EID_LEN] = {0};
    if (memcmp(eid.raw, zeroEid, URMA_EID_LEN) == 0) {
        HCCL_VM_WARN("skip channel info. eid is zero");
        return 0;
    }

    IpAddress addr(eid);
    auto eidStr = addr.EidToHexString();

    sim::runtime::EndPoint rmtEndPoint{};
    if (sim::runtime::GetEndPointByEidOrIpV4(addr, rmtEndPoint) != 0) {
        HCCL_VM_ERROR("Get remote endpoint failed. eid:{}", eidStr.c_str());
        return -1;
    }

    HCCL_VM_INFO(
        "channel info: loc phyId: {:d}, loc devKey: {:d}, loc dieId: {:d}, chId: {:d}, rmt devKey: {:d}, rmt dieId: "
        "{:d}, rmt eid: {}",
        deviceId, locDevice.id, static_cast<uint32_t>(dieId), chId, rmtEndPoint.device_id,
        static_cast<uint32_t>(chDataTmp.ioDieId), eidStr.c_str());

    uint16_t srcJettyId{0};
    uint16_t srcDieId{0};
    uint16_t dstDieId{0};
    // 仅A5的ChannelCtxDataV1携带dstVa字段(A6的ChannelDataV2没有, 按V1解析会读到无效数据),
    // 因此只在A5下通过dstVa推导同设备跨die的目的dieId。
    bool dstDieFromVa{false};
    DevType devType;
    hrtGetDeviceType(devType);
    if (devType == DevType::DEV_TYPE_950) {
        srcJettyId = (uint16_t)((chDataTmp.startJettyIdHigh << 4) | chDataTmp.startJettyIdLow);
        srcDieId = chDataTmp.ioDieId;
        auto dstVa = GetRemoteCcuVa(chDataTmp);
        if (dstVa == 0x456000000) {
            dstDieId = 1;
            dstDieFromVa = true;
        } else if (dstVa == 0x123000000) {
            dstDieId = 0;
            dstDieFromVa = true;
        }
    } else if (devType == DevType::DEV_TYPE_960) {
        srcJettyId = chId / 8 + 1024;
        ChannelDataV2* chnV2 = (ChannelDataV2*)input->data.data_info.data_array;
        srcDieId = chnV2->ioDieId;
    }
    auto jettyNum = (uint16_t)((chDataTmp.jettyNumHigh << 4) | chDataTmp.jettyNumLow);
    // 根据endPointPair获取src eid
    sim::runtime::EndPoint localEndPoint{};
    if (GetLocalEndPointByJetty(srcJettyId, srcDieId, localEndPoint) != 0) {
        HCCL_VM_ERROR("can not find local endPoint by jettyId: {}", srcJettyId);
        return -1;
    }

    HCCL_VM_INFO(
        "add chn:{:d}, srcJetty:{:d},jettyNum:{:d}, srcAddr:{}, dstAddr:{}, {:d}<-->{:d}, dstDieId({:d} - {:d})", chId,
        srcJettyId, jettyNum, localEndPoint.ip_addr, rmtEndPoint.ip_addr, localEndPoint.id, rmtEndPoint.id, dstDieId,
        rmtEndPoint.die_id);

    sim::operation::CcuChannelTab ccuChannelTab{};
    ccuChannelTab.id = 0;
    ccuChannelTab.channelId = chId;
    ccuChannelTab.srcDieId = localEndPoint.die_id;
    if (locDevice.id == rmtEndPoint.device_id && dstDieFromVa) {
        ccuChannelTab.dstDieId = dstDieId; // 当对端deviceId和本地deviceId相同时，需要根据dstVa信息获取dstDieId
    } else {
        ccuChannelTab.dstDieId = rmtEndPoint.die_id;
    }

    ccuChannelTab.srcDeviceId = localEndPoint.device_id;
    ccuChannelTab.dstDeviceId = rmtEndPoint.device_id;
    // The channel table is indexed by the global endpoint device IDs.  Do not
    // resolve communicator ranks here: channel setup happens before a current
    // communicator exists and the same channel may be reused by many domains.
    ccuChannelTab.srcRankId = 0;
    ccuChannelTab.dstRankId = 0;
    memcpy(ccuChannelTab.leid, &localEndPoint.eid, sizeof(localEndPoint.eid));
    memcpy(ccuChannelTab.reid, &rmtEndPoint.eid, sizeof(rmtEndPoint.eid));
    ccuChannelTab.protocol = 0;
    ccuChannelTab.jettyNum = jettyNum + 1;
    if (ccuChannelTab.jettyNum > HcclSim::HCCL_VM_MAX_CHANNEL_JETTIES) {
        HCCL_VM_ERROR(
            "channel id:{} jettyNum:{} exceeds capacity:{}", chId, ccuChannelTab.jettyNum,
            HcclSim::HCCL_VM_MAX_CHANNEL_JETTIES);
        return -1;
    }
    for (uint32_t i = 0; i < ccuChannelTab.jettyNum; i++) {
        ccuChannelTab.jettyId[i] = srcJettyId + i;
    }
    auto ret = sim::operation::InsertCcuChannel(ccuChannelTab);
    if (ret != 0) {
        HCCL_VM_ERROR("insert ccu channel table failed for channel id: {}", chId);
        return -1;
    }

    return 0;
}

int ConfigJettyInfo(channel_info_in* input, uint32_t deviceId)
{
    HCCL_VM_INFO("Enter into config jetty info...");
    uint8_t dieId = input->data.data_info.udie_idx;
    uint32_t jettyNum = input->data.data_info.data_array_size;
    uint32_t startJettyCtxId = input->offset_start;

    std::vector<LocalJettyCtxData> jettyCtxData;
    jettyCtxData.resize(jettyNum);
    for (size_t i = 0; i < jettyNum; i++) {
        (void)memcpy(&jettyCtxData[i], &input->data.data_info.data_array[i], sizeof(LocalJettyCtxData));
    }

    for (auto& tmp : jettyCtxData) {
        HCCL_VM_DEBUG(
            "doorbellAddr: [3]0x{:04x}, [2]0x{:04x}, [1]0x{:04x}, [0]0x{:04x}",
            tmp.doorbellAddr[3], // 3: doorbell ��ַ����
            tmp.doorbellAddr[2], // 2: doorbell ��ַ����
            tmp.doorbellAddr[1], tmp.doorbellAddr[0]);

        // ��ȫ���⣺��ֹ��ӡtoken�����Ϣ
        HCCL_VM_DEBUG(
            "pfeIdx: 0x{:04x}, ioDieId: 0x{:04x}, doorbellAddrType: 0x{:04x}, tokenValueIsValid: 0x{:04x}",
            static_cast<uint16_t>(tmp.pfeIdx), static_cast<uint16_t>(tmp.ioDieId),
            static_cast<uint16_t>(tmp.doorbellAddrType), static_cast<uint16_t>(tmp.tokenValueIsValid));

        HCCL_VM_DEBUG(
            "sqeBasicBlockLeftShifts: 0x{:04x}, pi: 0x{:04x}, ci: 0x{:04x}, "
            "maxCi: 0x{:04x}, oooCqeCnt: 0x{:04x}, startWqeBasicBlockIdxLow: 0x{:04x}, "
            "startWqeBasicBlockIdxHigh: 0x{:04x}, doorbellSendState: 0x{:04x}",
            static_cast<uint16_t>(tmp.sqeBasicBlockLeftShifts), tmp.pi, tmp.ci, tmp.maxCi,
            static_cast<uint16_t>(tmp.oooCqeCnt), static_cast<uint16_t>(tmp.startWqeBasicBlockIdxLow),
            static_cast<uint16_t>(tmp.startWqeBasicBlockIdxHigh), static_cast<uint16_t>(tmp.doorbellSendState));
    }

    return 0;
}

int GetCcuVersion(channel_info_out* output, uint32_t deviceId)
{
    sim::runtime::Device locDevice{};
    if (sim::runtime::GetDeviceByPhysicalId(deviceId, locDevice) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by physic id {} failed.", deviceId);
        return -1;
    }
    if (strncmp(locDevice.soc_version, "Ascend950", strlen("Ascend950")) == 0) {
        output->data.data_info.data_array[0].version = static_cast<ccu_version_e>(CcuVersion::CCU_V1);
    } else if (strncmp(locDevice.soc_version, "Ascend960", strlen("Ascend960")) == 0) {
        output->data.data_info.data_array[0].version = static_cast<ccu_version_e>(CcuVersion::CCU_V2);
    } else {
        HCCL_VM_ERROR("unknown soc version: {}", locDevice.soc_version);
        return -1;
    }
    return 0;
}

int SetXnTotalCnt(channel_info_in* input, uint32_t deviceId, uint8_t dieId)
{
    uint32_t wishCntXnIdBegin = input->data.data_info.data_array[0].xn_total_cnt.flag_from_addr;
    uint32_t wishCntXnIdEnd = input->data.data_info.data_array[0].xn_total_cnt.flag_to_addr;
    uint32_t totalCntId = input->data.data_info.data_array[0].xn_total_cnt.total_addr;

    HCCL_VM_INFO("Enter into set xn total cnt. wishCntXnIdBegin: {}", wishCntXnIdBegin);
    if (wishCntXnIdBegin == 0XFFFF) {
        HCCL_VM_WARN("wish cnt xn id begin is 0XFFFF, skip it.");
        return 0;
    }

    sim::operation::HalfRTTTab halfRttTab{};
    halfRttTab.id = 0;
    halfRttTab.dieId = dieId;
    halfRttTab.deviceId = deviceId;
    halfRttTab.wishCntXnIdBegin = wishCntXnIdBegin;
    halfRttTab.wishCntXnIdEnd = wishCntXnIdEnd;
    halfRttTab.totalCntId = totalCntId;

    auto ret = sim::operation::InsertHalfRTT(halfRttTab);
    if (ret != 0) {
        HCCL_VM_ERROR(
            "insert halfRTT table failed for die id: {}, device id: {}, wish cnt xn id begin: {}, wish cnt xn id end: "
            "{}, total cnt id: {}",
            dieId, deviceId, wishCntXnIdBegin, wishCntXnIdEnd, totalCntId);
        return -1;
    }

    return 0;
}

int SimRaCustomChannel(void* tlvHandle, struct TlvMsg* sendMsg, struct TlvMsg* recvMsg)
{
    HCCL_VM_INFO("Enter into custom channel...");
    auto in = reinterpret_cast<struct channel_info_in*>(sendMsg->data);
    auto out = reinterpret_cast<struct channel_info_out*>(recvMsg->data);

    auto tlvId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tlvHandle));
    auto curTlv = sim::runtime::Db::GetById<sim::runtime::RaTlv>(tlvId);
    if (!curTlv.ok()) {
        HCCL_VM_ERROR("tlv id not found:{:d}", tlvId);
        return ACL_ERROR_INVALID_PARAM;
    }

    uint8_t dieId = in->data.data_info.udie_idx;
    uint32_t devId = curTlv.value->physical_id;

    switch (in->op) {
        case ccu_u_opcode_t::CCU_U_OP_GET_DIE_WORKING:
            return GetEnableCcuDie(out, dieId);
        case ccu_u_opcode_t::CCU_U_OP_GET_BASIC_INFO:
            return SetCcuResourceBasicInfo(out, dieId, devId);
        case ccu_u_opcode_t::CCU_U_OP_SET_INSTRUCTION:
            return LoadMicrocodeInstructionStub(devId, dieId, in);
        case ccu_u_opcode_t::CCU_U_OP_SET_CHANNEL:
            return ConfigChannelInfo(in, devId);
        case ccu_u_opcode_t::CCU_U_OP_SET_JETTY_CTX:
            return ConfigJettyInfo(in, devId);
        case ccu_u_opcode_t::CCU_U_OP_GET_VERSION:
            return GetCcuVersion(out, devId);
        case ccu_u_opcode_t::CCU_U_OP_SET_XN_TOTAL_CNT:
            return SetXnTotalCnt(in, devId, dieId);
        default:
            break;
    }

    return 0;
}

int RaCtxGetAsyncEvents(void* ctxHandle, struct AsyncEvent events[], unsigned int* num)
{
    sleep(1);
    *num = 0;
    return 0;
}

int GetAllUsedEndPoint(uint32_t phyDevId, std::vector<sim::runtime::EndPoint>& endPoints)
{
    sim::runtime::Device device{};
    if (sim::runtime::GetDeviceByPhysicalId(phyDevId, device) != 0) {
        HCCL_VM_ERROR("can not find device by id:{:d}", phyDevId);
        return -1;
    }

    auto deviceId = device.id;
    auto endPointsResult = sim::runtime::Db::GetByPred<sim::runtime::EndPoint>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::EndPoint::device_id, deviceId),
        HcclSim::Storage::Eq(&sim::runtime::EndPoint::status, 1)));
    if (!endPointsResult.ok() || !endPointsResult.value.has_value()) {
        return -1;
    }
    endPoints = std::move(*endPointsResult.value);
    return 0;
}

int RaGetDevEidInfoNum(struct RaInfo info, unsigned int* num)
{
    std::vector<sim::runtime::EndPoint> endPoints;
    if (GetAllUsedEndPoint(info.phyId, endPoints) != 0) {
        return -1;
    }
    HCCL_VM_INFO("return success {:d}", endPoints.size());
    *num = endPoints.size();
    return 0;
}

int RaGetDevEidInfoList(struct RaInfo info, struct HccpDevEidInfo infoList[], unsigned int* num)
{
    std::vector<sim::runtime::EndPoint> endPoints;
    if (GetAllUsedEndPoint(info.phyId, endPoints) != 0) {
        HCCL_VM_ERROR("get all EndPoint failed phyId:{:d}", info.phyId);
        return -1;
    }

    uint32_t rankId = 0xFFFF;
    uint64_t serverTmp = 0;
    if (!sim::runtime::GetRankIdByMPI(rankId, serverTmp)) {
        HCCL_VM_ERROR("get rankId by MPI fail serverKey:{:d}", serverTmp);
        return ACL_ERROR_INVALID_PARAM;
    }

    for (uint32_t idx = 0; idx < *num; idx++) {
        infoList[idx].type = 0;
        infoList[idx].eidIndex = 0;
        infoList[idx].funcId = endPoints[idx].func_id;
        infoList[idx].chipId = info.phyId; // todo: 单server, logic id与rank id相等，但多server此处有问题。
        infoList[idx].dieId = endPoints[idx].die_id;
        // 19: UBOE_DEV_FLAG_RIGHT_SHIFT; 仅当EndPoint对应port/port_group的protocols为UBOE时置位
        infoList[idx].devFeature = endPoints[idx].is_uboe ? (1u << 19) : 0;
        memcpy(infoList[idx].eid.raw, endPoints[idx].eid, sizeof(endPoints[idx].eid));

        Eid eid{};
        for (uint32_t i = 0; i < URMA_EID_LEN; i++) {
            eid.raw[i] = endPoints[idx].eid[i];
        }
        IpAddress addr(eid);
        auto eidStr = addr.EidToHexString();
        // 无 MPI 或仅定位 server 时可能保留 0xFFFF，日志将其显示为 unknown。
        HCCL_VM_INFO(
            "rankId:{}, eid: {}, phyId: {}, serverKey=: {}", rankId == 0xFFFF ? "unknown" : std::to_string(rankId),
            eidStr.c_str(), info.phyId, g_cur_server_key);
    }

    return 0;
}

// 图模式重放专用重发：只吃 GraphAction 固化的原料(taskMeta，含 virAddr args)。
// 与正常下发 rtCCULaunch 完全隔离、互不复用——正常下发怎么演进都不影响重放，重放有 bug 也不影响正常下发。
// 重放 = 重新走 index 分配 + GenCaModelCcuToml + sqe dump + InsertTaskToCollection，让 runner
// 重新跑 CCU 仿真、重新产数据面行为——与 AICPU 的 LaunchAicpuKernelRaw 同粒度。
// taskMeta 里的 ccu.args 必须是 virAddr（采集期 rtCCULaunch 已做 devAddr->virAddr 转换，
// 重放沿用固化结果，无需再转换）。
extern "C" void LaunchCcuKernelRaw(const HcclTaskMetaData& taskMeta)
{
    const uint32_t deviceId = static_cast<uint32_t>(taskMeta.deviceId);
    const uint8_t dieId = taskMeta.taskData.ccu.dieId;

    // 同一 (deviceId, dieId) 多次重放下发多个 SQE 任务，按调用顺序递增 index 作为 SQE 任务序号。
    // 与正常下发 rtCCULaunch 的序号空间独立，互不干扰。
    static std::mutex sqeIdxMutex;
    static std::map<std::pair<uint32_t, uint32_t>, uint32_t> sqeIdxMap;
    uint32_t index = 0;
    {
        std::lock_guard<std::mutex> lock(sqeIdxMutex);
        index = sqeIdxMap[std::make_pair(deviceId, static_cast<uint32_t>(dieId))]++;
    }

    // 从 taskMeta 还原 rtCcuTaskInfo_t（字段布局与 CcuTask 一致，args 已是 virAddr）
    rtCcuTaskInfo_t checkerTask{};
    memcpy(&checkerTask, &taskMeta.taskData.ccu, sizeof(rtCcuTaskInfo_t));

    // 生成toml文件（仅当采集期已解析出有效 rankId；文件名第三段使用 SQE 任务序号 index，保证同 device/die
    // 多任务不冲突）
    if (taskMeta.rankId != 0xFFFF
        && GenCaModelCcuToml(&checkerTask, taskMeta.rankId, index) != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("gen ccu toml file fail");
        return;
    }

    // 生成txt文件
    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    const uint32_t argCount = checkerTask.argSize > RT_CCU_SQE_ARGS_LEN ? RT_CCU_SQE_ARGS_LEN : checkerTask.argSize;
    std::ostringstream fileName;
    fileName << "sqe_info_device_" << deviceId << "_die_" << static_cast<uint32_t>(dieId) << "_mission_"
             << static_cast<uint32_t>(checkerTask.missionId) << "_startId_" << checkerTask.instStartId << "_sqe_"
             << index << ".txt";
    std::ostringstream sqeData;
    sqeData << "ccu sqe info: startInstrId= " << checkerTask.instStartId << ", instrCnt= " << checkerTask.instCnt
            << ", argSize= " << checkerTask.argSize << "\n";
    for (uint32_t idx = 0; idx < argCount; idx++) {
        sqeData << "[SQE Arg][" << idx << "]: " << checkerTask.args[idx] << "\n";
    }
    auto status = write_or_overwrite_in_cwd(InstallPath::ResolveToInstallRoot("data/" + fileName.str()), sqeData.str());
    HCCL_VM_INFO("write file {} success", fileName.str());

    HcclTaskMetaData taskMetaData = taskMeta;
    uint32_t outIndex = 0;
    auto ret = InsertTaskToCollection(&taskMetaData, &outIndex);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("insert task fail");
        return;
    }
}

int rtCCULaunch(rtCcuTaskInfo_t* taskInfo, rtStream_t const stream)
{
    if (taskInfo == nullptr) {
        HCCL_VM_ERROR("CCU task info is null");
        return -1;
    }

    uint64_t streamId = reinterpret_cast<uint64_t>(stream);

    uint32_t deviceId = (uint32_t)sim::runtime::GetCurrDeviceId();
    uint32_t dieId = static_cast<uint32_t>(taskInfo->dieId);

    // 同一 (deviceId, dieId) 可能下发多个 SQE 任务，按调用顺序递增 index 作为 SQE 任务序号
    static std::mutex sqeIdxMutex;
    static std::map<std::pair<uint32_t, uint32_t>, uint32_t> sqeIdxMap;
    uint32_t index = 0;
    {
        std::lock_guard<std::mutex> lock(sqeIdxMutex);
        index = sqeIdxMap[std::make_pair(deviceId, dieId)]++;
    }

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::CCU_GRAPH;
    taskMetaData.streamId = streamId;
    taskMetaData.deviceId = deviceId;
    taskMetaData.commId = g_cur_comm_key;
    // rankId 仅用于 GenCaModelCcuToml 的 dump 文件名；解析失败时保持 0xFFFF 无效值并跳过 dump，
    // 不影响后续 sqe txt 与 InsertTaskToCollection。
    uint32_t rankId = 0xFFFF;
    if (taskMetaData.commId == 0) {
        // 算子入口线程未设置通信域(本 stub 可能被 HCCL 内部线程调用)，直接跳过 dump，不调 GetCommRankByDeviceId。
        HCCL_VM_WARN("CCU GenCaModelCcuToml dump skipped: commId not set, deviceId={}", deviceId);
    } else if (!sim::runtime::GetCommRankByDeviceId(taskMetaData.commId, deviceId, rankId)) {
        HCCL_VM_WARN(
            "CCU GenCaModelCcuToml dump skipped: cannot resolve rankId, commId={}, deviceId={}", taskMetaData.commId,
            deviceId);
    }
    taskMetaData.rankId = rankId;

    // CCU SQE arguments are device addresses in the runtime launch API, while
    // checker memory layouts use the corresponding virtual start_ptr values.
    // Keep scalar arguments unchanged and translate only mapped addresses.
    rtCcuTaskInfo_t checkerTask = *taskInfo;
    const uint32_t argCount = checkerTask.argSize > RT_CCU_SQE_ARGS_LEN ? RT_CCU_SQE_ARGS_LEN : checkerTask.argSize;
    for (uint32_t argId = 0; argId < argCount; ++argId) {
        const uint64_t devAddr = checkerTask.args[argId];
        if (devAddr == 0) {
            continue;
        }
        const uint64_t virtualAddr = sim::TryGetVirPtrByDevPtr(devAddr);
        if (virtualAddr != 0) {
            HCCL_VM_INFO(
                "CCU arg converted: deviceId={}, dieId={}, argId={}, devAddr=0x{:x}, virAddr=0x{:x}", deviceId, dieId,
                argId, devAddr, virtualAddr);
            checkerTask.args[argId] = virtualAddr;
        }
    }
    memcpy(&taskMetaData.taskData.ccu, &checkerTask, sizeof(rtCcuTaskInfo_t));

    // 图模式采集分支：录制 CCU_LAUNCH action，不立即入库，保证重放时落在 START_SUB/END_SUB 之间。
    // taskMeta 已在上面完成 devAddr -> virAddr 转换，重放沿用采集期翻译结果即可。
    if (IsCapturingStream(streamId)) {
        // g_currOpDetailId 是进程级全局,由算子入口 RecordOpDbInfo 设置; CCU 的 rtCCULaunch 在 HCCL 内部
        // 若 g_cur_comm_key 为 0, 但 g_currOpDetailId 仍可读，直接取它即可;
        uint32_t opDetailId = sim::operation::g_currOpDetailId;
        RecordCcuLaunchAction(streamId, taskMetaData, opDetailId);
        HCCL_VM_INFO(
            "ccu captured to model(stream:{}), not executed, missionId[{}]", streamId,
            taskMetaData.taskData.ccu.missionId);
        return 0;
    }

    // 生成toml文件（仅当 rankId 有效；文件名第三段使用 SQE 任务序号 index，保证同 device/die 多任务不冲突）
    if (rankId != 0xFFFF && GenCaModelCcuToml(&checkerTask, rankId, index) != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("gen ccu toml file fail");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 生成txt文件
    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    std::ostringstream fileName;
    fileName << "sqe_info_device_" << deviceId << "_die_" << dieId << "_mission_"
             << static_cast<uint32_t>(taskInfo->missionId) << "_startId_" << taskInfo->instStartId << ".txt";
    std::ostringstream sqeData;
    sqeData << "ccu sqe info: startInstrId= " << checkerTask.instStartId << ", instrCnt= " << checkerTask.instCnt
            << ", argSize= " << checkerTask.argSize << "\n";
    for (uint32_t idx = 0; idx < argCount; idx++) {
        sqeData << "[SQE Arg][" << idx << "]: " << checkerTask.args[idx] << "\n";
    }
    auto status = write_or_overwrite_in_cwd(InstallPath::ResolveToInstallRoot("data/" + fileName.str()), sqeData.str());
    HCCL_VM_INFO("write file {} success", fileName.str());

    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("insert task fail");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    return 0;
}

#ifdef __cplusplus
}
#endif // __cplusplus
