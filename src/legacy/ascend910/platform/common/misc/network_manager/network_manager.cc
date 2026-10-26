/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "network_manager.h"
#include "externalinput_pub.h"
#include "device_capacity.h"
#include "adapter_tdt.h"
#include "adapter_rts.h"
#include "adapter_hccp.h"
#include "adapter_hal.h"
#include "adapter_error_manager.h"
#include "dlhal_function.h"
#include "adapter_hccp_common.h"
namespace hccl {

using namespace std;

constexpr u32 SOCKET_LISTEN_AUTO_INTERFACE = 4;
constexpr u32 SOCKET_LISTEN_AUTO_INTERFACE_VERSION = 3;

NetworkManager* NetworkManager::nmInstance[MAX_DEV_NUM] = {nullptr};
std::atomic<unsigned> NetworkManager::InitTool::initCount(0);

NetworkManager::InitTool::InitTool()
{
    if (initCount.load() == 0) {
        for (u32 i = 0; i < MAX_DEV_NUM; i++) {
            NetworkManager::nmInstance[i] = new (std::nothrow) NetworkManager;
            CHK_PRT_CONT(
                NetworkManager::nmInstance[i] == nullptr,
                HCCL_ERROR("[NetworkManager][InitTool] nmInstance[%u] new failed", i));
        }
    }
    ++initCount;
}

NetworkManager::InitTool::~InitTool()
{
    --initCount;
    if (initCount.load() == 0) {
        for (u32 i = 0; i < MAX_DEV_NUM; i++) {
            if (NetworkManager::nmInstance[i] != nullptr) {
                delete NetworkManager::nmInstance[i];
                NetworkManager::nmInstance[i] = nullptr;
            }
        }
    }
}

NetworkManager::NetworkManager()
    : deviceLogicId_(INVALID_INT),
      devicePhyId_(INVALID_UINT),
      isHostUseDevNic_(false),
      notifyType_(NO_USE)
{}

NetworkManager::~NetworkManager()
{
    Destroy();
    isRaDeInit_ = false;
}

NetworkManager& NetworkManager::GetInstance(s32 deviceLogicID)
{
    HCCL_INFO("NetworkManager::GetInstance deviceLogicID[%u].", deviceLogicID);
    if (deviceLogicID == HOST_DEVICE_ID) {
        nmInstance[DEFAULT_DEVICE_LOGIC_ID]->deviceLogicId_ = DEFAULT_DEVICE_LOGIC_ID;
        return *(nmInstance[DEFAULT_DEVICE_LOGIC_ID]);
    }

    if (static_cast<u32>(deviceLogicID) >= MAX_DEV_NUM || deviceLogicID <= HOST_DEVICE_ID) {
        HCCL_WARNING("[Get][Instance]deviceLogicID[%d] is invalid", deviceLogicID);
        nmInstance[DEFAULT_DEVICE_LOGIC_ID]->deviceLogicId_ = DEFAULT_DEVICE_LOGIC_ID;
        return *(nmInstance[DEFAULT_DEVICE_LOGIC_ID]);
    }
    nmInstance[deviceLogicID]->deviceLogicId_ = deviceLogicID;
    return *(nmInstance[deviceLogicID]);
}

HcclResult NetworkManager::TsdCapabilityGet(bool& supportMultiProcHCCP)
{
    int32_t type = TSD_CAPABILITY_MUTIPLE_HCCP;
    bool* resultPtr = &supportMultiProcHCCP;
    CHK_RET(hrtTsdCapabilityGet(deviceLogicId_, type, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(resultPtr))));
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::GetNicIp(uint32_t devicePhyId, HcclAddress** addr, uint32_t* len)
{
    vector<HcclIpAddress> tempIp;
    std::unique_lock<std::mutex> lock(memResMutex_);
    if (nicIpAddrs_.size() == 0) {
        CHK_RET(hrtRaGetDeviceIP(devicePhyId, tempIp));
        nicIpAddrs_.resize(tempIp.size());
        for (size_t i = 0; i < tempIp.size(); ++i) {
            CHK_RET(HcclIpAddressConvertHcclAddr(&nicIpAddrs_[i], &tempIp[i]));
        }
    }
    *addr = this->nicIpAddrs_.data();
    *len = this->nicIpAddrs_.size();
    lock.unlock();
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::TsdProcessOpen(bool hasBackup)
{
    s32 locaLogDevid = 0;
    hrtGetDevice(&locaLogDevid);
    if (locaLogDevid != deviceLogicId_) {
        hrtSetDevice(deviceLogicId_);
    }
    // 校验是否为新版本驱动，旧版本驱动不支持配置backupPhyId，报错返回
    s32 halAPIVersion = 0;
    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    CHK_RET(hrtHalGetAPIVersion(halAPIVersion));
    HCCL_INFO(
        "[%s]params: halAPIVersion[%d], BACKUP_DEVICE_LOG_DEV_VERSION[%d]", __func__, halAPIVersion,
        BACKUP_DEVICE_LOG_DEV_VERSION);
    if (halAPIVersion < BACKUP_DEVICE_LOG_DEV_VERSION) {
        HCCL_WARNING(
            "[%s]this package does not support obtaining backUp HCCP Log in PLOG. halAPIVersion[%d]", __func__,
            halAPIVersion);
    }
    isTsdProcessOpen_ = true;
    if (!hasBackup || halAPIVersion < BACKUP_DEVICE_LOG_DEV_VERSION) {
        std::string extPam("--hdcType=" + std::to_string(HDC_SERVICE_TYPE_RDMA_V2));
        rtNetServiceOpenArgs openArgs = {};
        rtProcExtParam extParam{};
        extParam.paramInfo = extPam.c_str();
        extParam.paramLen = extPam.size();
        openArgs.extParamList = &extParam;
        openArgs.extParamCnt = 1UL;
        // 根据pid粒度拉起hccp的rs进程
        CHK_RET(hrtOpenNetService(&openArgs));
        HCCL_INFO(
            "[%s]hrtOpenNetService success, subPid[%d], devicePhyId_[%u], deviceLogicId_[%d], hasBackup[%u]", __func__,
            subPid_, devicePhyId_, deviceLogicId_, hasBackup);
    } else {
        // 获取chip上另一个die的logicalID
        u32 deviceBackUpPhyId = 0;
        CHK_RET(hrtGetPairDevicePhyId(devicePhyId_, deviceBackUpPhyId));
        rtNetServiceOpenArgs openArgs = {};
        std::string extPams[TSD_OPEN_EXT_PARA_NUM]
            = {std::string("--hdcType=" + std::to_string(HDC_SERVICE_TYPE_RDMA_V2)),
               std::string("--backupPhyId=" + std::to_string(deviceBackUpPhyId))};
        rtProcExtParam extParams[TSD_OPEN_EXT_PARA_NUM]{};
        for (u32 i = 0; i < TSD_OPEN_EXT_PARA_NUM; i++) {
            extParams[i].paramInfo = extPams[i].c_str();
            extParams[i].paramLen = extPams[i].size();
        }
        openArgs.extParamList = extParams;
        openArgs.extParamCnt = TSD_OPEN_EXT_PARA_NUM;
        // 根据pid粒度拉起hccp的rs进程
        CHK_RET(hrtOpenNetService(&openArgs));
        HCCL_INFO(
            "[%s]hrtOpenNetService success, subPid[%u], "
            "devicePhyId_[%u], deviceLogicId_[%d], deviceBackUpPhyId[%u], hasBackup[%u]",
            __func__, subPid_, devicePhyId_, deviceLogicId_, deviceBackUpPhyId, hasBackup);
    }

    if (locaLogDevid != deviceLogicId_) {
        hrtSetDevice(locaLogDevid);
    }
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::PrepareInit(NICDeployment nicDeploy, u32 devicePhyId, s32& ref)
{
    HCCL_INFO("nicDeploy = [%u], devicePhyId = [%u] ", nicDeploy, devicePhyId);
    if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST) {
        ref = hostNicInitRef_.Ref();
    } else if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        ref = deviceNicInitRef_.Ref();
    } else {
        HCCL_ERROR(
            "[NetworkManager][PrepareInit]NetworkManager: init nic failed, nicPosition[%u] is not supported.",
            static_cast<u32>(nicDeploy));
        return HCCL_E_INTERNAL;
    }
    if (ref > 1) {
        HCCL_INFO("NetworkManager: init nic, nicPosition[%u] ref[%u], skip.", static_cast<u32>(nicDeploy), ref);
        return HCCL_SUCCESS;
    }

    if (devicePhyId != INVALID_UINT) {
        devicePhyId_ = devicePhyId;
    } else {
        // 初始化ra资源(dev信息带入逻辑ID)
        CHK_RET(hrtGetDevice(&deviceLogicId_));
        CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId_), devicePhyId_));
    }
    return HCCL_SUCCESS;
}
HcclResult NetworkManager::GetConfigAndRaInit(struct RaInitConfig& config, bool isHdcV2, NICDeployment nicDeploy)
{
    // DC场景网卡与进程在同一侧，需要设置为类似host网卡模式
    config.nicPosition = Is310PDevice() ? 0 : static_cast<u32>(nicDeploy);
    config.phyId = devicePhyId_;

    if (isHdcV2) {
        // 使用HDC_SERVICE_TYPE_RDMA_V2指定进程粒度
        config.hdcType = HDC_SERVICE_TYPE_RDMA_V2;
        HCCL_DEBUG(
            "[%s]hdcType is set to HDC_SERVICE_TYPE_RDMA_V2"
            "devicePhyId[%u], deviceLogicId_[%d]",
            __func__, devicePhyId_, deviceLogicId_);
    }
    HcclResult ret = HrtRaInit(&config);
    RPT_CALL_ERR(
        ret != HCCL_SUCCESS, "ra init failed,return[%d] devicePhyId_[%u], nicPosition[%u]", ret, devicePhyId_,
        static_cast<u32>(nicDeploy));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[NetworkManager][Init]errNo[0x%016llx] ra init failed,return[%d] devicePhyId_[%u], "
            "nicPosition[%u]",
            HCCL_ERROR_CODE(ret), ret, devicePhyId_, static_cast<u32>(nicDeploy)),
        HCCL_E_NETWORK);

    return HCCL_SUCCESS;
}

HcclResult NetworkManager::GetTsdOpen(NICDeployment nicDeploy, bool hasBackup, bool& supportMultiProcHCCP)
{
    if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && !Is310PDevice()) { // DC场景不需要拉起tsd
        CHK_RET(DlTdtFunction::GetInstance().DlTdtFunctionInit());
        CHK_RET(TsdCapabilityGet(supportMultiProcHCCP));
        HCCL_INFO("[NetworkManager][Init]supportMultiProcHCCP[%u], hasBackup[%u]", supportMultiProcHCCP, hasBackup);
        if (supportMultiProcHCCP || hasBackup) {
            // 根据pid粒度拉起hccp的rs进程
            CHK_RET(TsdProcessOpen(hasBackup));
            HCCL_INFO(
                "[NetworkManager][Init]open tsd by process success, devicePhyId[%u], deviceLogicId_[%d].", devicePhyId_,
                deviceLogicId_);
        } else {
            // device 网卡初始化前需要拉起 hccp.
            CHK_RET(hrtOpenTsd());
            HCCL_INFO(
                "[%s]NetworkManager open tsd success, devicePhyId[%u], deviceLogicId_[%d]", __func__, devicePhyId_,
                deviceLogicId_);
        }
    }
    return HCCL_SUCCESS;
}
/* init network resource */
HcclResult NetworkManager::Init(
    NICDeployment nicDeploy, [[maybe_unused]] bool enableWhitelistFlag, u32 devicePhyId, bool isHostUseDevNic,
    bool hasBackup)
{
    s32 ref = 0;
    if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST) {
        ref = hostNicInitRef_.Ref();
    } else if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        ref = deviceNicInitRef_.Ref();
    } else {
        HCCL_ERROR(
            "[NetworkManager][Init]NetworkManager: init nic failed, nicPosition[%u] is not supported.",
            static_cast<u32>(nicDeploy));
        return HCCL_E_INTERNAL;
    }
    if (ref > 1) {
        HCCL_INFO("NetworkManager: init nic, nicPosition[%u] ref[%u], skip.", static_cast<u32>(nicDeploy), ref);
        return HCCL_SUCCESS;
    }

    if (devicePhyId != INVALID_UINT) {
        devicePhyId_ = devicePhyId;
    } else {
        // 初始化ra资源(dev信息带入逻辑ID)
        CHK_RET(hrtGetDevice(&deviceLogicId_));
        CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId_), devicePhyId_));
    }
    HCCL_INFO(
        "NetworkManager: devicePhyId[%u], devicePhyId_[%u] deviceLogicId_[%u], hasBackup[%d]", devicePhyId,
        devicePhyId_, deviceLogicId_, hasBackup);

    bool supportMultiProcHCCP = false;
    if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && !Is310PDevice()) { // DC场景不需要拉起tsd
        CHK_RET(DlTdtFunction::GetInstance().DlTdtFunctionInit());
        CHK_RET(TsdCapabilityGet(supportMultiProcHCCP));
        HCCL_INFO("[NetworkManager][Init]supportMultiProcHCCP[%u], hasBackup[%u]", supportMultiProcHCCP, hasBackup);
        if (supportMultiProcHCCP || hasBackup) {
            // 根据pid粒度拉起hccp的rs进程
            CHK_RET(TsdProcessOpen(hasBackup));
            HCCL_INFO(
                "[NetworkManager][Init]open tsd by process success, devicePhyId[%u], deviceLogicId_[%d].", devicePhyId_,
                deviceLogicId_);
        } else {
            // device 网卡初始化前需要拉起 hccp.
            CHK_RET(hrtOpenTsd());
            HCCL_INFO(
                "[%s]NetworkManager open tsd success, devicePhyId[%u], deviceLogicId_[%d]", __func__, devicePhyId_,
                deviceLogicId_);
        }
    }

    struct RaInitConfig config = {DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE, false};

    // DC场景网卡与进程在同一侧，需要设置为类似host网卡模式
    config.nicPosition = Is310PDevice() ? 0 : static_cast<u32>(nicDeploy);
    config.phyId = devicePhyId_;
    HCCL_DEBUG(
        "[%s]config.phyId = %u, nicPosition[%u], hasBackup[%d], devicePhyId_[%u], deviceLogicId_[%d], "
        "devicePhyId[%u]",
        __func__, config.phyId, config.nicPosition, hasBackup, devicePhyId_, deviceLogicId_, devicePhyId);

    if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && !Is310PDevice() && (supportMultiProcHCCP || hasBackup)) {
        // 使用HDC_SERVICE_TYPE_RDMA_V2指定进程粒度
        config.hdcType = HDC_SERVICE_TYPE_RDMA_V2;
        HCCL_DEBUG(
            "[%s]hdcType is set to HDC_SERVICE_TYPE_RDMA_V2, hasBackup[%d], nicDeploy[%d], "
            "devicePhyId[%u], deviceLogicId_[%d]",
            __func__, hasBackup, nicDeploy, devicePhyId_, deviceLogicId_);
    }
    DevType devType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_910_93 && nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        isEnableHdcAsync_ = true;
        config.enableHdcAsync = true;
    }
    HCCL_INFO(
        "[%s]config.phyId[%u], config.nicPosition[%u], config.hdcType[%d], config.enableHdcAsync[%d]", __func__,
        config.phyId, config.nicPosition, config.hdcType, config.enableHdcAsync);
    HcclResult ret = HrtRaInit(&config);
    RPT_CALL_ERR(
        ret != HCCL_SUCCESS, "ra init failed,return[%d] devicePhyId_[%u], nicPosition[%u]", ret, devicePhyId_,
        static_cast<u32>(nicDeploy));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[NetworkManager][Init]errNo[0x%016llx] ra init failed,return[%d] devicePhyId_[%u], "
            "nicPosition[%u]",
            HCCL_ERROR_CODE(ret), ret, devicePhyId_, static_cast<u32>(nicDeploy)),
        HCCL_E_NETWORK);

    HCCL_INFO(
        "NetworkManager nicDeploy[%u] deviceLogicId_[%d] devicePhyId_[%u] init ra OK, nicSocketMap size[%u], "
        "isHostUseDevNic[%d]",
        static_cast<u32>(nicDeploy), deviceLogicId_, devicePhyId_, raResourceInfo_.nicSocketMap.size(),
        isHostUseDevNic);
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::InitV2(NICDeployment nicDeploy, bool isBackup, u32 devicePhyId, bool isHostUseDevNic)
{
    s32 ref = 0;
    CHK_RET(PrepareInit(nicDeploy, devicePhyId, ref));
    if (ref > 1) {
        HCCL_INFO("NetworkManager: initv2 nic, nicPosition[%u] ref[%u], skip.", static_cast<u32>(nicDeploy), ref);
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "NetworkManager InitV2: devicePhyId[%u], devicePhyId_[%u] deviceLogicId_[%u], isBackup[%d]", devicePhyId,
        devicePhyId_, deviceLogicId_, isBackup);

    bool supportMultiProcHCCP = false;
    CHK_RET(GetTsdOpen(nicDeploy, isBackup, supportMultiProcHCCP));

    struct RaInitConfig config = {DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE, false};
    bool isHdcV2
        = (nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && !Is310PDevice() && (supportMultiProcHCCP || isBackup));
    if (isHdcV2) {
        HCCL_DEBUG(
            "[%s]hdcType is set to HDC_SERVICE_TYPE_RDMA_V2, isBackup[%d], nicDeploy[%d], "
            "devicePhyId[%u], deviceLogicId_[%d]",
            __func__, isBackup, nicDeploy, devicePhyId_, deviceLogicId_);
    }
    CHK_RET(GetConfigAndRaInit(config, isHdcV2, nicDeploy));

    HCCL_INFO(
        "NetworkManager nicDeploy[%u] deviceLogicId_[%d] devicePhyId_[%u] init ra OK, nicSocketMap size[%u], "
        "isHostUseDevNic[%d]",
        static_cast<u32>(nicDeploy), deviceLogicId_, devicePhyId_, raResourceInfo_.nicSocketMap.size(),
        isHostUseDevNic);
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::HeterogStartListen(const HcclIpAddress& ipAddr, u32 port)
{
    HCCL_DEBUG("HeterogStartListen ipAddr[%s] port[%u]", ipAddr.GetReadableAddress(), port);
    SocketHandle nicSocketHandle = raResourceInfo_.nicSocketMap[ipAddr].nicSocketHandle;

    CHK_PRT_RET(
        port > MAX_PORT_ID || port < MIN_PORT_ID,
        HCCL_ERROR(
            "[NetworkManager][HeterogStartListen] port error[%u], valid range[%u, %u]", port, MIN_PORT_ID, MAX_PORT_ID),
        HCCL_E_NETWORK);

    if (nicSocketHandle != nullptr
        && raResourceInfo_.nicSocketMap[ipAddr].listenedPort.find(port)
               == raResourceInfo_.nicSocketMap[ipAddr].listenedPort.end()
        && IPPortListenRefMapHost_[ipAddr][port].Ref() == FIRST_LISTEN) {
        CHK_RET(StartListenSocket(nicSocketHandle, port));
        raResourceInfo_.nicSocketMap[ipAddr].listenedPort.insert(port);
        raResourceInfo_.hostNetSocketMap[ipAddr].listenedPort.insert(port);
    }
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::HeterogInit(u32 devId, const HcclIpAddress& ipAddr, u32 port)
{
    s32 ref = hostNicInitRef_.Ref();
    if (ref > 1) {
        HCCL_INFO("NetworkManager: heterog init nic, ref[%u], skip", ref);
        return HCCL_SUCCESS;
    }

    // 暂缺获取物理id的手段
    RaInitConfig config = {DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE, false};
    config.phyId = ((static_cast<s32>(devId) == HOST_DEVICE_ID) ? 0 : devId);
    config.nicPosition = static_cast<u32>(NICDeployment::NIC_DEPLOYMENT_HOST);
    HCCL_INFO("HeterogInit call HrtRaInit.");
    CHK_RET(HrtRaInit(&config));

    struct rdev nicRdevInfo = {};
    nicRdevInfo.phyId = devId;
    nicRdevInfo.family = ipAddr.GetFamily();
    nicRdevInfo.localIp.addr = ipAddr.GetBinaryAddress().addr;
    nicRdevInfo.localIp.addr6 = ipAddr.GetBinaryAddress().addr6;
    SocketHandle socketHandle = nullptr;
    HcclResult ret = hrtRaSocketInit(NETWORK_PEER_ONLINE, nicRdevInfo, socketHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HeterogInit]errNo[0x%016llx] ra socket init failed, ip[%s], return[%d]",
            HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ipAddr.GetReadableAddress(), ret);
        HrtRaDeInit(&config);
        return HCCL_E_TCP_CONNECT;
    }
    HCCL_INFO("ip[%s] socket init OK", ipAddr.GetReadableAddress());

    IpSocket ipSocketInfo;
    ipSocketInfo.nicSocketHandle = socketHandle;

    raResourceInfo_.nicSocketMap.insert(std::make_pair(ipAddr, ipSocketInfo));
    raResourceInfo_.hostNetSocketMap.insert(std::make_pair(ipAddr, ipSocketInfo));

    /*  device网卡初始化暂不考虑 */
    if (!GetExternalInputHcclIsTcpMode()) {
        ret = InitRdmaHandle(devId, ipAddr);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HeterogInit] InitRdmaHandle fail, ret[%d], destroy resource.", ret);
            hrtRaSocketDeInit(socketHandle);
            HrtRaDeInit(&config);
            return ret;
        }
    }

    ret = HeterogStartListen(ipAddr, port);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HeterogInit] HeterogStartListen fail, ret[%d], destroy resource.", ret);
        if (!GetExternalInputHcclIsTcpMode()) {
            auto& ipSock = raResourceInfo_.nicSocketMap[ipAddr];
            if (ipSock.nicRdmaHandle != nullptr) {
                (void)HrtRaRdmaDeInit(ipSock.nicRdmaHandle, NO_USE);
            }
        }
        hrtRaSocketDeInit(socketHandle);
        HrtRaDeInit(&config);
        raResourceInfo_.nicSocketMap.erase(ipAddr);
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::HeterogStopListen(const HcclIpAddress& ipAddr, u32 port, bool isNeedDeinit)
{
    HCCL_DEBUG("HeterogStopListen ipAddr[%s] port[%u]", ipAddr.GetReadableAddress(), port);
    SocketHandle nicSocketHandle = raResourceInfo_.nicSocketMap[ipAddr].nicSocketHandle;

    if (nicSocketHandle != nullptr && raResourceInfo_.nicSocketMap[ipAddr].listenedPort.size() > 0) {
        if (raResourceInfo_.nicSocketMap[ipAddr].listenedPort.find(port)
                != raResourceInfo_.nicSocketMap[ipAddr].listenedPort.end()
            && IPPortListenRefMapHost_[ipAddr][port].Unref() == LAST_RELEASE) {
            // 重复raInit时不用去停止监听，但是要处理引用计数，否则析构会出问题
            if (isRaInitRepeated_) {
                return HCCL_SUCCESS;
            }
            HCCL_INFO("HeterogStopListen socket listen stop, socket port[%u]", port);
            CHK_RET(StopListenSocket(nicSocketHandle, port));
            raResourceInfo_.nicSocketMap[ipAddr].listenedPort.erase(port);
            raResourceInfo_.hostNetSocketMap[ipAddr].listenedPort.erase(port);
        }

        if (isNeedDeinit) {
            CHK_RET(hrtRaSocketDeInit(nicSocketHandle));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::HeterogDeinit(u32 devId, const HcclIpAddress& ipAddr, u32 port)
{
    // 在用时跳过整个Deinit, 条目留给延迟拆除; 守卫须在Unref之前, 否则计数归零后
    // 进程退出时Destroy()的兜底失效, Ra资源泄漏
    if (MarkNicUserPendingStop(ipAddr, true, true)) {
        HCCL_INFO("[NetworkManager][HeterogDeinit]ip[%s] in use, defer teardown.", ipAddr.GetReadableAddress());
        return HCCL_SUCCESS;
    }
    s32 ref = hostNicInitRef_.Unref();
    if (ref > 0) {
        HCCL_INFO("NetworkManager: heterog deinit nic success, ref[%u], skip.", ref);
        return HCCL_SUCCESS;
    } else if (ref < 0) {
        HCCL_ERROR("[NetworkManager][DeInit]NetworkManager: heterog deinit nic failed, nic has already deinit.");
        return HCCL_E_INTERNAL;
    }

    CHK_RET(HeterogStopListen(ipAddr, port, true));

    SocketHandle nicRdmaHandle = raResourceInfo_.nicSocketMap[ipAddr].nicRdmaHandle;
    if (!GetExternalInputHcclIsTcpMode() && nicRdmaHandle != nullptr) {
        CHK_RET(HrtRaRdmaDeInit(nicRdmaHandle, NO_USE));
    }

    RaInitConfig config = {DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE, false};
    config.phyId = devId;
    config.nicPosition = static_cast<u32>(NICDeployment::NIC_DEPLOYMENT_HOST);

    HCCL_INFO("HeterogDeinit call HrtRaDeInit.");
    CHK_RET(HrtRaDeInit(&config));

    raResourceInfo_.nicSocketMap.erase(raResourceInfo_.nicSocketMap.find(ipAddr));
    raResourceInfo_.hostNetSocketMap.erase(raResourceInfo_.hostNetSocketMap.find(ipAddr));

    return HCCL_SUCCESS;
}

HcclResult NetworkManager::CloseHccpProcess()
{
    std::unique_lock<std::mutex> lock(hccpProcInfoMutex_);
    if (isTsdProcessOpen_ == true) {
        s32 locaLogDevid = 0;
        hrtGetDevice(&locaLogDevid);
        if (locaLogDevid != deviceLogicId_) {
            hrtSetDevice(deviceLogicId_);
        }
        CHK_RET(hrtCloseNetService());
        isTsdProcessOpen_ = false;
        if (locaLogDevid != deviceLogicId_) {
            hrtSetDevice(locaLogDevid);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::PrepareDeInit(s32& ref, NICDeployment nicDeploy)
{
    if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST) {
        ref = hostNicInitRef_.Unref();
    } else if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        ref = deviceNicInitRef_.Unref();
    } else {
        HCCL_ERROR(
            "[NetworkManager][PrepareDeInit]NetworkManager: deinit nic failed, nicPosition[%u] is not supported.",
            static_cast<u32>(nicDeploy));
        return HCCL_E_INTERNAL;
    }

    if (ref > 0) {
        HCCL_INFO(
            "NetworkManager: PrepareDeInit nic success, nicPosition[%u] ref[%u], skip.", static_cast<u32>(nicDeploy),
            ref);
        return HCCL_SUCCESS;
    } else if (ref < 0) {
        HCCL_ERROR(
            "[NetworkManager][PrepareDeInit]NetworkManager: deinit nic failed, nicPosition[%u] has already deinit.",
            nicDeploy);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::GetConfigAndRaDeinit(
    struct RaInitConfig& config, NICDeployment nicDeploy, bool& isMultiProc, bool hasBackup)
{
    if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && !Is310PDevice()) {
        CHK_RET(DlTdtFunction::GetInstance().DlTdtFunctionInit());
        bool supportMultiProcHCCP = false;
        CHK_RET(TsdCapabilityGet(supportMultiProcHCCP));
        if (supportMultiProcHCCP || hasBackup) {
            isMultiProc = true;
            config.hdcType = HDC_SERVICE_TYPE_RDMA_V2;
        }
    }

    HcclResult ret = HrtRaDeInit(&config);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[NetworkManager][DeInit]ra deinit failed. para: nicdeploy[%u], phyId[%u], ret[%u]", config.nicPosition,
            config.phyId, ret),
        ret);
    if (IsGeneralServer()) {
        isRaDeInit_ = true;
    }
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::DeInit(NICDeployment nicDeploy, [[maybe_unused]] bool resetDeviceFlag, bool hasBackup)
{
    s32 ref = 0;
    if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST) {
        ref = hostNicInitRef_.Unref();
    } else if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        ref = deviceNicInitRef_.Unref();
    } else {
        HCCL_ERROR(
            "[NetworkManager][DeInit]NetworkManager: deinit nic failed, nicPosition[%u] is not supported.",
            static_cast<u32>(nicDeploy));
        return HCCL_E_INTERNAL;
    }

    if (ref > 0) {
        HCCL_INFO(
            "NetworkManager: deinit nic success, nicPosition[%u] ref[%u], skip.", static_cast<u32>(nicDeploy), ref);
        return HCCL_SUCCESS;
    } else if (ref < 0) {
        HCCL_ERROR(
            "[NetworkManager][DeInit]NetworkManager: deinit nic failed, nicPosition[%u] has already deinit.",
            nicDeploy);
        return HCCL_E_INTERNAL;
    }

    struct RaInitConfig config = {DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE, isEnableHdcAsync_};
    config.nicPosition = Is310PDevice() ? 0 : static_cast<u32>(nicDeploy);
    config.phyId = devicePhyId_;

    bool isMultiProc = false;
    if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && !Is310PDevice()) {
        CHK_RET(DlTdtFunction::GetInstance().DlTdtFunctionInit());
        bool supportMultiProcHCCP = false;
        CHK_RET(TsdCapabilityGet(supportMultiProcHCCP));
        if (supportMultiProcHCCP || hasBackup) {
            isMultiProc = true;
            config.hdcType = HDC_SERVICE_TYPE_RDMA_V2;
        }
    }

    HcclResult ret = HrtRaDeInit(&config);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[NetworkManager][DeInit]ra deinit failed. para: nicdeploy[%u], phyId[%u], ret[%u]", config.nicPosition,
            config.phyId, ret),
        ret);
    if (IsGeneralServer()) {
        isRaDeInit_ = true;
    }

    if (isMultiProc) {
        CHK_RET(CloseHccpProcess());
        HCCL_INFO(
            "[%s]finish CloseHccpProcess, phyId[%u], hdcType[%d], nicDeployment[%d], hasBackup[%d]", __func__,
            config.phyId, config.hdcType, nicDeploy, hasBackup);
    }

    HCCL_INFO("NetworkManager: deinit nic success, nicPosition[%u] ref[%u].", static_cast<u32>(nicDeploy), ref);
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::DeInitV2(NICDeployment nicDeploy, bool isBackup, [[maybe_unused]] bool resetDeviceFlag)
{
    s32 ref = 0;
    CHK_RET(PrepareDeInit(ref, nicDeploy));
    if (ref > 0) {
        HCCL_INFO(
            "NetworkManager: DeInitV2 nic success, nicPosition[%u] ref[%u], skip.", static_cast<u32>(nicDeploy), ref);
        return HCCL_SUCCESS;
    }

    struct RaInitConfig config = {DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE, false};
    config.nicPosition = Is310PDevice() ? 0 : static_cast<u32>(nicDeploy);
    config.phyId = devicePhyId_;

    bool isMultiProc = false;
    CHK_RET(GetConfigAndRaDeinit(config, nicDeploy, isMultiProc, isBackup));

    if (isMultiProc) {
        CHK_RET(CloseHccpProcess());
        HCCL_INFO(
            "[%s]finish CloseHccpProcess, phyId[%u], hdcType[%d], nicDeployment[%d], isBackup[%d]", __func__,
            config.phyId, config.hdcType, nicDeploy, isBackup);
    }

    HCCL_INFO("NetworkManager: DeInitV2 success, nicPosition[%u] ref[%u].", static_cast<u32>(nicDeploy), ref);
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::StartVnic(HcclIpAddress localIp, u32& port, bool tagChkDis)
{
    CHK_PRT_RET(
        deviceNicInitRef_.Count() <= 0, HCCL_ERROR("[Start][Vnic]can't start vnic socket before init device nic!"),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(Is310PDevice(), HCCL_INFO("DC does not need vnic"), HCCL_SUCCESS);
    CHK_PRT_RET(
        port > MAX_PORT_ID, HCCL_ERROR("[Start][Vnic]invalid port id[%u], max port id[%u]", port, MAX_PORT_ID),
        HCCL_E_PARA);

    auto sockInfo = raResourceInfo_.vnicSocketMap.find(localIp);
    if (sockInfo == raResourceInfo_.vnicSocketMap.end()) {
        IpSocket tempSock;
        raResourceInfo_.vnicSocketMap.insert(std::make_pair(localIp, tempSock)); // 本IP占位
        HCCL_INFO("[Start][Vnic]device[%u] Start Vnic insert ip[%s]", devicePhyId_, localIp.GetReadableAddress());
    }

    IpSocket& sock = raResourceInfo_.vnicSocketMap[localIp];
    if (sock.listenedPort.size() == 0 && sock.nicSocketHandle == nullptr) {
        HcclResult ret = InitDeviceSocket(devicePhyId_, localIp, sock.nicSocketHandle);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Start][Vnic]errNo[0x%016llx] ra vnic init socket failed, devid[%u], ipAddr[%s], return[%d]",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, localIp.GetReadableAddress(), ret),
            ret);
    } else {
        HCCL_INFO(
            "[Start][Vnic]vnic socket has inited, ipAddr[%s] port[%u], skip init.", localIp.GetReadableAddress(), port);
    }

    if (sock.listenedPort.find(port) != sock.listenedPort.end()) {
        HCCL_WARNING("[Start][Vnic]ipAddr[%s] port[%u] is already listened.", localIp.GetReadableAddress(), port);
    } else {
        // 作为socket server端启动监听(虚拟网卡)
        bool isAutoPort = port == 0;
        HCCL_INFO("[Start][Vnic]trying to listen on ip[%s] port[%u].", localIp.GetReadableAddress(), port);
        CHK_RET(CheckAutoListenVersion(isAutoPort));
        HcclResult ret = StartListenSocket(sock.nicSocketHandle, port, tagChkDis); /* 只拉起1个vnic */
        CHK_PRT_RET(
            ret == HCCL_E_UNAVAIL,
            HCCL_INFO(
                "[Start][StartVnic]Could not start listening socket for IP [%s] and port [%u], tagChkDis[%u].",
                localIp.GetReadableAddress(), port, tagChkDis),
            ret);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Start][Vnic]errNo[0x%016llx] ra inner listen start failed, "
                "devid[%u], ip[%s], port[%u], return[%d]",
                HCCL_ERROR_CODE(ret), devicePhyId_, localIp.GetReadableAddress(), port, ret),
            ret);
        sock.listenedPort.insert(port);
        HCCL_RUN_INFO(
            "[Start][Vnic]Listen on ip[%s], port[%u] success, devPhyId[%u], devLogicId[%u], isAutoPort[%d]",
            localIp.GetReadableAddress(), port, devicePhyId_, deviceLogicId_, isAutoPort);
    }
    int refCount = IPPortListenRefMapVnicDevice_[localIp][port].Ref();
    HCCL_INFO(
        "NetworkManager devicePhyId_[%u] ip[%s] port[%u] start vnic OK. refCount[%d]", devicePhyId_,
        localIp.GetReadableAddress(), port, refCount);
    return HCCL_SUCCESS;
}

// 此处只进行socket的创建 不listen
HcclResult NetworkManager::CreateVnicSocketHandle(HcclIpAddress localIp)
{
    CHK_PRT_RET(
        deviceNicInitRef_.Count() == 0,
        HCCL_ERROR("[NetworkManager][CreateVnicSocketHandle]"
                   "can't start vnic socket before init device nic!"),
        HCCL_E_INTERNAL);
    OccupyIp(localIp, raResourceInfo_.vnicSocketMap);
    CHK_PRT_RET(Is310PDevice(), HCCL_INFO("DC does not need vnic"), HCCL_SUCCESS);

    IpSocket& sock = raResourceInfo_.vnicSocketMap[localIp];
    if (sock.listenedPort.size() == 0 && sock.nicSocketHandle == nullptr) {
        HcclResult ret = InitDeviceSocket(devicePhyId_, localIp, sock.nicSocketHandle);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[NetworkManager][CreateVnicSocketHandle]errNo[0x%016llx] ra vnic init socket failed, "
                "devid[%u], return[%d]",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, ret),
            ret);
    } else {
        HCCL_INFO(
            "[NetworkManager][CreateVnicSocketHandle] socket has inited, ipAddr[%s], skip.",
            localIp.GetReadableAddress());
    }
    HCCL_INFO(
        "[NetworkManager][CreateVnicSocketHandle] CreateVnicSocketHandle OK, ipAddr[%s].",
        localIp.GetReadableAddress());
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::StopVnic(const HcclIpAddress& localIp, u32 port)
{
    auto it = raResourceInfo_.vnicSocketMap.find(localIp);
    CHK_PRT_RET(
        it == raResourceInfo_.vnicSocketMap.end(),
        HCCL_ERROR("[Stop][Vnic]ip[%s] is not found in vnicSocketMap, port[%u].", localIp.GetReadableAddress(), port),
        HCCL_E_INTERNAL);
    IpSocket& ipSock = it->second;

    int count = IPPortListenRefMapVnicDevice_[localIp][port].Unref();
    CHK_PRT_RET(
        count > 0,
        HCCL_INFO("[Stop][Vnic]ip[%s] port[%u] ref[%d] skip stop.", localIp.GetReadableAddress(), port, count),
        HCCL_SUCCESS);
    CHK_PRT_RET(
        count < 0,
        HCCL_WARNING(
            "[Stop][Vnic]ip[%s] port[%u] devicePhyId_[%u] vnic stopped ERROR, refcount[%d].",
            localIp.GetReadableAddress(), port, devicePhyId_, count),
        HCCL_SUCCESS);

    // Stop Listen
    CHK_PRT_RET(
        ipSock.nicSocketHandle != nullptr && StopListenSocket(ipSock.nicSocketHandle, port) != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Stop][Vnic]errNo[0x%016llx] stop vnic socket failed,devid[%u], ip[%s], port[%u]",
            HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, localIp.GetReadableAddress(), port),
        HCCL_E_INTERNAL);
    ipSock.listenedPort.erase(port);

    // DeInit Socket
    if (ipSock.listenedPort.size() == 0 && ipSock.nicSocketHandle != nullptr) {
        if (MarkNicUserPendingStop(localIp, true, false)) {
            return HCCL_SUCCESS;
        }
        HcclResult ret = hrtRaSocketDeInit(ipSock.nicSocketHandle);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Stop][Vnic]errNo[0x%016llx] stop vnic socket failed,devid[%u], ip[%s], return[%d]",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, localIp.GetReadableAddress(), ret),
            HCCL_E_INTERNAL);
        raResourceInfo_.vnicSocketMap.erase(localIp);
    }
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::StopVnicSocketHandle(const HcclIpAddress& localIp)
{
    if (MarkNicUserPendingStop(localIp, true, false)) {
        return HCCL_SUCCESS;
    }
    auto it = raResourceInfo_.vnicSocketMap.find(localIp);
    CHK_PRT_RET(
        it == raResourceInfo_.vnicSocketMap.end(),
        HCCL_ERROR(
            "[NetworkManager][StopVnicSocketHandle]ip[%s] is not found in vnicSocketMap.",
            localIp.GetReadableAddress()),
        HCCL_E_INTERNAL);

    IpSocket& ipSock = it->second;
    // 关闭该ip下的全部端口的listen
    for (auto& port : ipSock.listenedPort) {
        if (ipSock.nicSocketHandle != nullptr && StopListenSocket(ipSock.nicSocketHandle, port) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[NetworkManager][StopVnicSocketHandle]errNo[0x%016llx] stop vnic listen failed, "
                "devid[%u], ip[%s], port[%u]",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, localIp.GetReadableAddress(), port);
            break;
        }
        IPPortListenRefMapVnicDevice_[localIp][port].Clear();
        HCCL_INFO(
            "[NetworkManager][StopVnicSocketHandle] ip[%s] stop listen port[%u]", localIp.GetReadableAddress(), port);
    }
    ipSock.listenedPort.clear();

    // 销毁socket
    CHK_PRT_RET(
        ipSock.nicSocketHandle != nullptr && hrtRaSocketDeInit(ipSock.nicSocketHandle) != HCCL_SUCCESS,
        HCCL_ERROR("[Stop][NicsSocket]VNIC socket deInit not successfully"), HCCL_E_NETWORK);
    ipSock.nicSocketHandle = nullptr;
    raResourceInfo_.vnicSocketMap.erase(localIp);

    HCCL_INFO(
        "[NetworkManager][StopVnicSocketHandle] devid[%u] ip[%s] stop vnic socket success", devicePhyId_,
        localIp.GetReadableAddress());
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::StartNic(
    const HcclIpAddress& ipAddr, u32& port, bool rdmaFlag, HcclIpAddress ipAddrBackup, bool tagChkDis)
{
    CHK_PRT_RET(
        deviceNicInitRef_.Count() == 0, HCCL_ERROR("[Start][Nic]can't start nic socket before init device nic!"),
        HCCL_E_INTERNAL);
    auto sockInfo = raResourceInfo_.nicSocketMap.find(ipAddr);
    if (sockInfo == raResourceInfo_.nicSocketMap.end()) {
        IpSocket tempSock;
        raResourceInfo_.nicSocketMap.insert(std::make_pair(ipAddr, tempSock)); // 本IP占位
        HCCL_INFO("device[%u] Start Nic insert ip[%s]", devicePhyId_, ipAddr.GetReadableAddress());
    }

    IpSocket& sock = raResourceInfo_.nicSocketMap[ipAddr];
    if (sock.listenedPort.size() == 0) {
        if (sock.nicSocketHandle == nullptr) {
            HcclResult ret = InitDeviceSocket(devicePhyId_, ipAddr, sock.nicSocketHandle);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Start][Nic]errNo[0x%016llx] ra nic init socket failed, devid[%u], return[%d]",
                    HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, ret),
                ret);
        }
    } else {
        HCCL_INFO("NetworkManager: socket has inited, ipAddr[%s] port[%u], skip.", ipAddr.GetReadableAddress(), port);
    }
    if (sock.nicRdmaHandle == nullptr && rdmaFlag) {
        NetworkMode netMode;
        GetNetworkMode(netMode);
        CHK_RET(GetNotifyType(notifyType_));
        HcclResult ret = InitRDMA(
            devicePhyId_, ipAddr, netMode, notifyType_, sock.nicRdmaHandle, isDisableLiteThread_, false, ipAddrBackup);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Start][Nic]errNo[0x%016llx] ra nic init rdma failed, devid[%u], return[%d]",
                HCCL_ERROR_CODE(HCCL_E_NETWORK), devicePhyId_, ret),
            HCCL_E_NETWORK);

        int supportLite;
        ret = HrtGetRdmaLiteStatus(sock.nicRdmaHandle, &supportLite);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Get][RdmaLiteStatus]errNo[0x%016llx] get rdma lite status failed, return[%d]",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ret),
            HCCL_E_TCP_CONNECT);
        isRdmaLiteEn_ = (supportLite == 1);
    } else {
        HCCL_INFO(
            "[Start][Nic] requesting not init rdma , or already init rdma, "
            "rdmaFlag[%u], ipAddr[%s], port[%u], skip.",
            rdmaFlag, ipAddr.GetReadableAddress(), port);
    }
    // 如果port id传入的值无效值0xFFFFFFFF, 不启动监听
    CHK_PRT_RET(port == MAX_VALUE_U32, HCCL_INFO("[Start][Nic] port id[%u], skip listen socket", port), HCCL_SUCCESS);
    CHK_PRT_RET(
        port > MAX_PORT_ID, HCCL_ERROR("[Start][Nic]invalid port id[%u], max port id[%u]", port, MAX_PORT_ID),
        HCCL_E_INTERNAL);
    if (sock.listenedPort.find(port) != sock.listenedPort.end()) {
        HCCL_WARNING("port[%u] is already listened.", port);
    } else {
        bool isAutoPort = port == 0;
        HCCL_INFO(
            "[Start][Nic]trying to listen on ip[%s] port[%u] tagChkDis[%u].", ipAddr.GetReadableAddress(), port,
            tagChkDis);
        CHK_RET(CheckAutoListenVersion(isAutoPort));
        HcclResult ret = StartListenSocket(sock.nicSocketHandle, port, tagChkDis);
        CHK_PRT_RET(
            ret == HCCL_E_UNAVAIL,
            HCCL_INFO(
                "[Start][Nic]Could not start listening socket for IP [%s] and port [%u].", ipAddr.GetReadableAddress(),
                port),
            ret);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Start][Nic]errNo[0x%016llx] ra inner listen start failed, "
                "devid[%u], ip[%s], port[%u], tagChkDis[%u], return[%d]",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, ipAddr.GetReadableAddress(), port, tagChkDis, ret),
            HCCL_E_TCP_CONNECT);
        HCCL_INFO("port[%u] listen start OK", port);
        sock.listenedPort.insert(port);
        HCCL_RUN_INFO(
            "[Start][Nic]Listen on ip[%s], port[%u], tagChkDis[%u] success, devPhyId[%u], devLogicId[%u], "
            "isAutoPort[%d]",
            ipAddr.GetReadableAddress(), port, tagChkDis, devicePhyId_, deviceLogicId_, isAutoPort);
    }
    int refCount = IPPortListenRefMapDevice_[ipAddr][port].Ref();
    HCCL_INFO("Nic ip[%s] port[%u] refcount is [%d]", ipAddr.GetReadableAddress(), port, refCount);
    return HCCL_SUCCESS;
}

void NetworkManager::SetDisableLiteThread(bool disable) { isDisableLiteThread_ = disable; }

HcclResult NetworkManager::CreateNicSocketHandle(const HcclIpAddress& ipAddr)
{
    CHK_PRT_RET(
        deviceNicInitRef_.Count() == 0,
        HCCL_ERROR("[NetworkManager][CreateNicSocketHandle]can't start nic socket before init device nic!"),
        HCCL_E_INTERNAL);
    OccupyIp(ipAddr, raResourceInfo_.nicSocketMap);

    IpSocket& sock = raResourceInfo_.nicSocketMap[ipAddr];
    if (sock.listenedPort.size() == 0) {
        if (sock.nicSocketHandle == nullptr) {
            HcclResult ret = InitDeviceSocket(devicePhyId_, ipAddr, sock.nicSocketHandle);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[NetworkManager][CreateNicSocketHandle]errNo[0x%016llx] ra nic init socket failed, devid[%u], "
                    "return[%d]",
                    HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, ret),
                ret);
        }
    } else {
        HCCL_INFO(
            "[NetworkManager][CreateNicSocketHandle] socket has inited, ipAddr[%s], skip.",
            ipAddr.GetReadableAddress());
    }
    HCCL_INFO(
        "[NetworkManager][CreateNicSocketHandle] CreateNicSocketHandle OK, ipAddr[%s].", ipAddr.GetReadableAddress());
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::CreateRdmaHandle(
    const HcclIpAddress& ipAddr, bool isBackup, NetworkMode netMode, NotifyTypeT notifyType,
    HcclNetDevDeployment netDevDeployment)
{
    //  device侧如果传入的ip是备份ip,那么给InitRDMA传递的两个主备ip(两个相同)都是备份ip hostrdma没有备份 不受影响
    HcclIpAddress ipAddrBackup(ipAddr.GetFamily(), ipAddr.GetBinaryAddress());
    if (!isBackup) {
        HCCL_INFO("[NetworkManager][CreateRdmaHandle] ipAddr[%s] is not backup", ipAddr.GetReadableAddress());
        ipAddrBackup.clear();
    }
    CHK_PRT_RET(
        (deviceNicInitRef_.Count() == 0 && hostNicInitRef_.Count() == 0),
        HCCL_ERROR("[NetworkManager][CreateRdmaHandle]can't start nic socket before init device nic!"),
        HCCL_E_INTERNAL);
    switch (netDevDeployment) {
        case HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE: {
            // ip占位
            OccupyIp(ipAddr, raResourceInfo_.nicSocketMap);
            // 初始化socket
            IpSocket& sock = raResourceInfo_.nicSocketMap[ipAddr];
            if (sock.nicRdmaHandle == nullptr) {
                notifyType_ = notifyType;
                HcclResult ret = InitRDMA(
                    devicePhyId_, ipAddr, netMode, notifyType_, sock.nicRdmaHandle, false, false, ipAddrBackup);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[NetworkManager][CreateRdmaHandle]errNo[0x%016llx] ra nic init rdma failed, devid[%u], "
                        "return[%d]",
                        HCCL_ERROR_CODE(HCCL_E_NETWORK), devicePhyId_, ret),
                    HCCL_E_NETWORK);
            } else {
                HCCL_INFO(
                    "[NetworkManager][CreateRdmaHandle] requesting not init rdma , or already init rdma, "
                    "ipAddr[%s], skip.",
                    ipAddr.GetReadableAddress());
            }
            // device-roce需要开启rdmalite
            RdmaSupportLite(sock.nicRdmaHandle);
            break;
        }
        case HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_HOST: {
            // 本ip占位
            OccupyIp(ipAddr, raResourceInfo_.hostNetSocketMap);
            // 初始化socket
            IpSocket& sock = raResourceInfo_.hostNetSocketMap[ipAddr];
            if (sock.nicRdmaHandle == nullptr) {
                notifyType_ = NOTIFY;
                HcclResult ret = InitRDMA(devicePhyId_, ipAddr, netMode, notifyType_, sock.nicRdmaHandle);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[Start][Nic]errNo[0x%016llx] ra nic init rdma failed, devid[%u], return[%d]",
                        HCCL_ERROR_CODE(HCCL_E_NETWORK), devicePhyId_, ret),
                    HCCL_E_NETWORK);
            } else {
                HCCL_INFO(
                    "[NetworkManager][CreateRdmaHandle] requesting not init rdma , or already init rdma, "
                    "ipAddr[%s], skip.",
                    ipAddr.GetReadableAddress());
            }
            // host rdma的额外占位
            auto temp = raResourceInfo_.nicSocketMap.find(ipAddr);
            if (temp != raResourceInfo_.nicSocketMap.end()) {
                HCCL_ERROR(
                    "[NetworkManager][CreateRdmaHandle] ipAddr[%s] has already occupied.", ipAddr.GetReadableAddress());
                return HCCL_E_INTERNAL;
            }
            raResourceInfo_.nicSocketMap.insert(std::make_pair(ipAddr, sock));
            break;
        }
        default: {
            HCCL_ERROR(
                "[NetworkManager][CreateRdmaHandle]this Deployment [%u] is not supported, please check the "
                "configuration.",
                netDevDeployment);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::RdmaSupportLite(RdmaHandle rdmaHandle)
{
    int supportLite;
    HcclResult ret = HrtGetRdmaLiteStatus(rdmaHandle, &supportLite);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Get][RdmaLiteStatus]errNo[0x%016llx] get rdma lite status failed, return[%d]",
            HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ret),
        HCCL_E_TCP_CONNECT);
    isRdmaLiteEn_ = (supportLite == 1);
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::StopRdmaHandle(const HcclIpAddress& ipAddr, HcclNetDevDeployment netDevDeployment)
{
    // rdma没有socket 没有listen

    // 销毁socket
    switch (netDevDeployment) {
        case HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE: {
            auto it = raResourceInfo_.nicSocketMap.find(ipAddr);
            if (it == raResourceInfo_.nicSocketMap.end()) {
                HCCL_WARNING(
                    "[Stop][NicsSocket]ip[%s] not found in nicSocketMap, may already cleaned.",
                    ipAddr.GetReadableAddress());
                return HCCL_SUCCESS;
            }
            IpSocket& ipSock = it->second;
            if (MarkNicUserPendingStop(ipAddr, false, true)) {
                return HCCL_SUCCESS;
            }
            RdmaHandle deinitRdmaHandle = ipSock.nicRdmaHandle;
            if (ipSock.nicRdmaHandle != nullptr && HrtRaRdmaDeInit(ipSock.nicRdmaHandle, notifyType_) != HCCL_SUCCESS) {
                HCCL_ERROR("[Stop][rdma]NIC rdev deInit not successfully, notifyType_[%d]", notifyType_);
                return HCCL_E_NETWORK;
            }
            ipSock.nicRdmaHandle = nullptr;
            // 两个map记录同一rdma handle: 一侧deinit后另一侧成为悬空指针, 此处同步置空, 防止二次deinit
            if (deinitRdmaHandle != nullptr) {
                auto hostIt = raResourceInfo_.hostNetSocketMap.find(ipAddr);
                if (hostIt != raResourceInfo_.hostNetSocketMap.end()
                    && hostIt->second.nicRdmaHandle == deinitRdmaHandle) {
                    hostIt->second.nicRdmaHandle = nullptr;
                }
            }
            // 如果该ip下既没有tcp也没有rdma那么移除该ip
            if (ipSock.nicSocketHandle == nullptr && ipSock.nicRdmaHandle == nullptr) {
                raResourceInfo_.nicSocketMap.erase(ipAddr);
            }
            break;
        }
        case HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_HOST: {
            auto it = raResourceInfo_.hostNetSocketMap.find(ipAddr);
            CHK_PRT_RET(
                it == raResourceInfo_.hostNetSocketMap.end(),
                HCCL_ERROR("[Stop][rdma]ip[%s] is not found in hostNetSocketMap.", ipAddr.GetReadableAddress()),
                HCCL_E_INTERNAL);
            IpSocket& ipSock = it->second;
            // 延迟拆除路径会调本函数, 同样需要in-use守卫
            if (MarkNicUserPendingStop(ipAddr, false, true)) {
                return HCCL_SUCCESS;
            }
            RdmaHandle deinitRdmaHandle = ipSock.nicRdmaHandle;
            if (ipSock.nicRdmaHandle != nullptr && HrtRaRdmaDeInit(ipSock.nicRdmaHandle, notifyType_) != HCCL_SUCCESS) {
                HCCL_ERROR("[Stop][rdma]NIC rdev deInit not successfully, notifyType_[%d]", notifyType_);
                return HCCL_E_NETWORK;
            }
            ipSock.nicRdmaHandle = nullptr;
            // 同步置空nicSocketMap内同一指针, 防止二次deinit; 仅当该ip的socket与rdma均已空时才移除,
            // 条目可能还持有Device侧未拆的socket, 提前erase会使延迟拆除查不到条目而泄漏
            auto nicIt = raResourceInfo_.nicSocketMap.find(ipAddr);
            if (nicIt != raResourceInfo_.nicSocketMap.end()) {
                if (deinitRdmaHandle != nullptr && nicIt->second.nicRdmaHandle == deinitRdmaHandle) {
                    nicIt->second.nicRdmaHandle = nullptr;
                }
                if (nicIt->second.nicSocketHandle == nullptr && nicIt->second.nicRdmaHandle == nullptr) {
                    raResourceInfo_.nicSocketMap.erase(ipAddr); // 移除额外的占位
                }
            }
            // 如果该ip下没有tcp也没有rdma那么移除该ip
            if (ipSock.nicSocketHandle == nullptr && ipSock.nicRdmaHandle == nullptr) {
                raResourceInfo_.hostNetSocketMap.erase(ipAddr);
            }
            break;
        }
        default: {
            HCCL_ERROR(
                "[NetworkManager][StopRdmaHandle]this Deployment [%u] is not supported, please check the "
                "configuration.",
                netDevDeployment);
            return HCCL_E_NOT_SUPPORT;
        }
    }

    HCCL_INFO(
        "[NetworkManager] [StopRdmaHandle] devicePhyId_[%u] StopRdmaHandle success ip [%s]", devicePhyId_,
        ipAddr.GetReadableAddress());
    return HCCL_SUCCESS; // stop socket。port 数清零时自动关闭socket
}

HcclResult NetworkManager::AddNicSocketUser(const HcclIpAddress& ipAddr)
{
    std::lock_guard<std::mutex> lock(nicSocketUserMutex_);
    s32 ref = nicSocketUserMap_[ipAddr].ref.Ref();
    HCCL_INFO("[NetworkManager][AddNicSocketUser]ip[%s] user ref[%d]", ipAddr.GetReadableAddress(), ref);
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::RemoveNicSocketUser(const HcclIpAddress& ipAddr)
{
    bool needDeferredStop = false;
    bool needRdmaDeferredStop = false;
    nicSocketUserMutex_.lock();
    auto it = nicSocketUserMap_.find(ipAddr);
    if (it == nicSocketUserMap_.end()) {
        nicSocketUserMutex_.unlock();
        return HCCL_SUCCESS;
    }
    int count = it->second.ref.Unref();
    if (count > 0) {
        nicSocketUserMutex_.unlock();
        return HCCL_SUCCESS;
    }
    if (count < 0) {
        // 下溢说明Add/Remove未严格配对: 负计数使条目永不清除且in-use守卫失效, 在用handle会被强拆;
        // Clear复位后条目正常清除, 拆除函数幂等, 重复进入无害
        HCCL_ERROR(
            "[NetworkManager][RemoveNicSocketUser]ip[%s] refcount underflow[%d], "
            "AddNicSocketUser/RemoveNicSocketUser not paired",
            ipAddr.GetReadableAddress(), count);
        it->second.ref.Clear();
    }
    needDeferredStop = it->second.pendingStop;
    needRdmaDeferredStop = it->second.pendingRdmaStop;
    // 条目保留至拆除完成后再移除, 否则拆除期间新接入的用户会失去in-use守卫保护
    nicSocketUserMutex_.unlock();

    HcclResult ret = HCCL_SUCCESS;
    if (needRdmaDeferredStop) {
        HCCL_INFO(
            "[NetworkManager][RemoveNicSocketUser]ip[%s] last user released, do deferred rdma stop",
            ipAddr.GetReadableAddress());
        // 两个map可能各持有真实handle, 都需拆除(原else分支只拆一侧会泄漏);
        // 失败只记录不中断, 返回值保留首个非SUCCESS
        auto nicIt = raResourceInfo_.nicSocketMap.find(ipAddr);
        if (nicIt != raResourceInfo_.nicSocketMap.end() && nicIt->second.nicRdmaHandle != nullptr) {
            HcclResult curRet = StopRdmaHandle(ipAddr, HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE);
            if (curRet != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[NetworkManager][RemoveNicSocketUser]deferred stop device rdma handle failed, ip[%s], "
                    "ret[%d]",
                    ipAddr.GetReadableAddress(), curRet);
                ret = (ret == HCCL_SUCCESS) ? curRet : ret;
            }
        }
        auto hostIt = raResourceInfo_.hostNetSocketMap.find(ipAddr);
        if (hostIt != raResourceInfo_.hostNetSocketMap.end() && hostIt->second.nicRdmaHandle != nullptr) {
            HcclResult curRet = StopRdmaHandle(ipAddr, HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_HOST);
            if (curRet != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[NetworkManager][RemoveNicSocketUser]deferred stop host rdma handle failed, ip[%s], "
                    "ret[%d]",
                    ipAddr.GetReadableAddress(), curRet);
                ret = (ret == HCCL_SUCCESS) ? curRet : ret;
            }
        }
    }
    if (needDeferredStop) {
        HCCL_INFO(
            "[NetworkManager][RemoveNicSocketUser]ip[%s] last user released, do deferred stop",
            ipAddr.GetReadableAddress());
        // 三个map各持有独立的socket handle, 都需拆除; 失败不中断, 返回首个非SUCCESS
        auto nicIt = raResourceInfo_.nicSocketMap.find(ipAddr);
        if (nicIt != raResourceInfo_.nicSocketMap.end() && nicIt->second.nicSocketHandle != nullptr) {
            HcclResult curRet = StopNicSocketHandle(ipAddr);
            if (curRet != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[NetworkManager][RemoveNicSocketUser]deferred stop device nic socket failed, ip[%s], "
                    "ret[%d]",
                    ipAddr.GetReadableAddress(), curRet);
                ret = (ret == HCCL_SUCCESS) ? curRet : ret;
            }
        }
        auto vnicIt = raResourceInfo_.vnicSocketMap.find(ipAddr);
        if (vnicIt != raResourceInfo_.vnicSocketMap.end() && vnicIt->second.nicSocketHandle != nullptr) {
            HcclResult curRet = StopVnicSocketHandle(ipAddr);
            if (curRet != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[NetworkManager][RemoveNicSocketUser]deferred stop vnic socket failed, ip[%s], ret[%d]",
                    ipAddr.GetReadableAddress(), curRet);
                ret = (ret == HCCL_SUCCESS) ? curRet : ret;
            }
        }
        auto hostIt = raResourceInfo_.hostNetSocketMap.find(ipAddr);
        if (hostIt != raResourceInfo_.hostNetSocketMap.end() && hostIt->second.nicSocketHandle != nullptr) {
            HcclResult curRet = StopHostSocketHandle(ipAddr);
            if (curRet != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[NetworkManager][RemoveNicSocketUser]deferred stop host socket failed, ip[%s], ret[%d]",
                    ipAddr.GetReadableAddress(), curRet);
                ret = (ret == HCCL_SUCCESS) ? curRet : ret;
            }
        }
    }

    // 拆除完成后移除条目; 若期间有新用户接入则保留, 由其释放时再处理
    {
        std::lock_guard<std::mutex> lock(nicSocketUserMutex_);
        auto eraseIt = nicSocketUserMap_.find(ipAddr);
        if (eraseIt != nicSocketUserMap_.end() && eraseIt->second.ref.Count() == 0) {
            nicSocketUserMap_.erase(eraseIt);
        }
    }
    return ret;
}

bool NetworkManager::MarkNicUserPendingStop(const HcclIpAddress& ipAddr, bool markSocket, bool markRdma)
{
    // 查询与置位须在同一把锁内完成, 避免间隙内最后一个使用者释放导致置位落空、延迟拆除丢失
    std::lock_guard<std::mutex> lock(nicSocketUserMutex_);
    auto it = nicSocketUserMap_.find(ipAddr);
    if (it != nicSocketUserMap_.end() && it->second.ref.Count() > 0) {
        if (markSocket) {
            it->second.pendingStop = true;
        }
        if (markRdma) {
            it->second.pendingRdmaStop = true;
        }
        HCCL_INFO(
            "[NetworkManager][MarkNicUserPendingStop]ip[%s] still in use ref[%d], markSocket[%d], markRdma[%d]",
            ipAddr.GetReadableAddress(), it->second.ref.Count(), markSocket, markRdma);
        return true;
    }
    return false;
}

HcclResult NetworkManager::StopNicSocketHandle(const HcclIpAddress& ipAddr)
{
    if (MarkNicUserPendingStop(ipAddr, true, false)) {
        return HCCL_SUCCESS;
    }
    auto it = raResourceInfo_.nicSocketMap.find(ipAddr);
    CHK_PRT_RET(
        it == raResourceInfo_.nicSocketMap.end(),
        HCCL_ERROR(
            "[NetworkManager][StopNicSocketHandle]ip[%s] is not found in nicSocketMap.", ipAddr.GetReadableAddress()),
        HCCL_E_INTERNAL);
    IpSocket& ipSock = it->second;
    HcclResult ret = HCCL_SUCCESS;
    // 关闭该ip下的全部端口的listen
    for (auto& port : ipSock.listenedPort) {
        ret = StopNicsSocketListen(ipAddr, port);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[NetworkManager][StopNicSocketHandle]errNo[0x%016llx] stop nic listen failed,devid[%u], ip[%s], "
                "port[%u], return[%d]",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, ipAddr.GetReadableAddress(), port, ret),
            HCCL_E_INTERNAL);
        ipSock.listenedPort.erase(port);
        IPPortListenRefMapDevice_[ipAddr][port].Clear(); // port计数更新
        HCCL_WARNING(
            "[NetworkManager][StopNicSocketHandle] ip [%s] stop listen port [%u] refcount is [%d]",
            ipAddr.GetReadableAddress(), port, IPPortListenRefMapDevice_[ipAddr][port].Count());
    }

    // 销毁socket
    SocketHandle deinitSocketHandle = ipSock.nicSocketHandle;
    if (ipSock.nicSocketHandle != nullptr && hrtRaSocketDeInit(ipSock.nicSocketHandle) != HCCL_SUCCESS) {
        HCCL_ERROR("[Stop][NicsSocket]NIC socket deInit not successfully");
        return HCCL_E_NETWORK;
    }
    ipSock.nicSocketHandle = nullptr;
    // 两个map记录同一socket handle(PsWorkerRaInit双写): 一侧deinit后同步置空另一侧, 防止二次释放
    if (deinitSocketHandle != nullptr) {
        auto hostIt = raResourceInfo_.hostNetSocketMap.find(ipAddr);
        if (hostIt != raResourceInfo_.hostNetSocketMap.end() && hostIt->second.nicSocketHandle == deinitSocketHandle) {
            hostIt->second.nicSocketHandle = nullptr;
        }
    }

    // 如果该ip下既没有tcp也没有rdma那么移除该ip
    if (ipSock.nicSocketHandle == nullptr && ipSock.nicRdmaHandle == nullptr) {
        raResourceInfo_.nicSocketMap.erase(ipAddr);
    }

    return HCCL_SUCCESS;
}

HcclResult NetworkManager::StopNic(const HcclIpAddress& ipAddr, u32 port)
{
    auto it = raResourceInfo_.nicSocketMap.find(ipAddr);
    CHK_PRT_RET(
        it == raResourceInfo_.nicSocketMap.end(),
        HCCL_ERROR("[Stop][Nic]ip[%s] is not found in nicSocketMap, port[%u].", ipAddr.GetReadableAddress(), port),
        HCCL_E_INTERNAL);
    IpSocket& ipSock = it->second;
    HcclResult ret = HCCL_SUCCESS;
    // 传入端口号为无效值0xFFFFFFFF，未启动监听，不需要stop listen
    if (port != MAX_VALUE_U32) {
        CHK_PRT_RET(
            IPPortListenRefMapDevice_[ipAddr][port].Unref() > 0,
            HCCL_INFO(
                "[Stop][Nic]ip[%s] port[%u] ref[%d] skip stop.", ipAddr.GetReadableAddress(), port,
                IPPortListenRefMapDevice_[ipAddr][port].Count()),
            HCCL_SUCCESS);
        ret = StopNicsSocketListen(ipAddr, port);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Stop][Nic]errNo[0x%016llx] stop nic socket failed,devid[%u], ip[%s], port[%u], return[%d]",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, ipAddr.GetReadableAddress(), port, ret),
            HCCL_E_INTERNAL);
        ipSock.listenedPort.erase(port);
    }

    if (ipSock.listenedPort.size() == 0) {
        ret = StopNicsSocket(ipAddr);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Stop][Nic]errNo[0x%016llx] stop nic socket failed,devid[%u], ip[%s], return[%d]",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, ipAddr.GetReadableAddress(), ret),
            HCCL_E_INTERNAL);
        // 纯rdma占位条目需保留, 否则延迟拆除查不到条目导致rdma handle泄漏
        if (ipSock.nicSocketHandle == nullptr && ipSock.nicRdmaHandle == nullptr) {
            raResourceInfo_.nicSocketMap.erase(ipAddr);
        }
    }
    return HCCL_SUCCESS;
}
HcclResult NetworkManager::StopAllDeviceNicSockets()
{
    HcclResult ret = HCCL_SUCCESS;
    for (auto& itSocket : raResourceInfo_.nicSocketMap) {
        std::set<u32> listenedPorts = itSocket.second.listenedPort;
        for (auto itPort : listenedPorts) {
            ret = StopNicsSocketListen(itSocket.first, itPort);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[Stop][AllDeviceNicSockets]errNo[0x%016llx] stop nic socket failed,devid[%u],ip[%s], "
                    "port[%u],return[%d]",
                    HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, itSocket.first.GetReadableAddress(), itPort,
                    ret);
                itSocket.second.listenedPort.erase(itPort);
                break;
            }
        }
        ret = StopNicsSocket(itSocket.first);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[Stop][AllDeviceNicSockets]errNo[0x%016llx] stop nic socket failed,devid[%u],ip[%s],return[%d]",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, itSocket.first.GetReadableAddress(), ret);
            if (itSocket.second.nicRdmaHandle != nullptr) {
                HCCL_ERROR("[Stop][AllDeviceNicSockets] itSocket.second.nicRdmaHandle is not nullptr.");
                (void)HrtRaRdmaDeInit(itSocket.second.nicRdmaHandle, notifyType_);
                itSocket.second.nicRdmaHandle = nullptr;
            }
            if (itSocket.second.nicSocketHandle != nullptr) {
                HCCL_ERROR("[Stop][AllDeviceNicSockets] itSocket.second.nicSocketHandle is not nullptr.");
                // socket handle须调hrtRaSocketDeInit, 误调rdma版本编译期无法发现
                (void)hrtRaSocketDeInit(itSocket.second.nicSocketHandle);
                itSocket.second.nicSocketHandle = nullptr;
            }
        }
    }

    // 在用(如心跳)条目保留, 由最后一个用户释放时延迟拆除, 无条件clear会泄漏handle
    for (auto it = raResourceInfo_.nicSocketMap.begin(); it != raResourceInfo_.nicSocketMap.end();) {
        if (MarkNicUserPendingStop(it->first, true, true)) {
            ++it;
            continue;
        }
        it = raResourceInfo_.nicSocketMap.erase(it);
    }
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::StopAllDeviceVnicSockets()
{
    HcclResult ret = HCCL_SUCCESS;

    // 在用(如心跳)的vnic条目保留, 由最后一个用户释放时延迟拆除; 强拆会触发use-after-free
    for (auto it = raResourceInfo_.vnicSocketMap.begin(); it != raResourceInfo_.vnicSocketMap.end();) {
        if (MarkNicUserPendingStop(it->first, true, true)) {
            ++it;
            continue;
        }
        HCCL_WARNING(
            "vnicSocketMap ip[%s] is not released when NetworkManager Destroy, force releasing",
            it->first.GetReadableAddress());
        if (it->second.nicSocketHandle != nullptr) {
            for (auto itPort : it->second.listenedPort) {
                ret = StopListenSocket(it->second.nicSocketHandle, itPort);
                if (ret != HCCL_SUCCESS) {
                    HCCL_ERROR(
                        "[Stop][AllDeviceVnicSockets]errNo[0x%016llx] stop vnic socket listen failed, "
                        "devid[%u], ip[%s], port[%u], return[%d]",
                        HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, it->first.GetReadableAddress(), itPort, ret);
                }
                IPPortListenRefMapVnicDevice_[it->first][itPort].Clear();
            }
            ret = hrtRaSocketDeInit(it->second.nicSocketHandle);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[Stop][AllDeviceVnicSockets]errNo[0x%016llx] deinit vnic socket failed, "
                    "devid[%u], ip[%s], return[%d]",
                    HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, it->first.GetReadableAddress(), ret);
            }
            it->second.nicSocketHandle = nullptr;
        }
        it = raResourceInfo_.vnicSocketMap.erase(it);
    }

    return HCCL_SUCCESS;
}

HcclResult NetworkManager::InitRDMA(
    u32 devicePhysicID, const HcclIpAddress& ipAddr, NetworkMode netMode, NotifyTypeT notifyType,
    RdmaHandle& rdmaHandle, bool disabledLiteThread, bool enable910ALite, HcclIpAddress ipAddrBackup)
{
    struct rdev nicRdevInfo = {};
    nicRdevInfo.phyId = devicePhysicID;
    nicRdevInfo.family = ipAddr.GetFamily();
    nicRdevInfo.localIp.addr = ipAddr.GetBinaryAddress().addr;
    nicRdevInfo.localIp.addr6 = ipAddr.GetBinaryAddress().addr6;

    struct RdevInitInfo init_info = {};
    init_info.mode = netMode;
    init_info.notifyType = notifyType;
    init_info.disabledLiteThread = disabledLiteThread;
    init_info.enabled910aLite = enable910ALite;
    init_info.enabled2mbLite = GetExternalInputRdmaFastPost();

    HcclResult ret = HCCL_SUCCESS;
    HCCL_DEBUG("isRaInitRepeated_[%d]", isRaInitRepeated_);
    if (isRaInitRepeated_) {
        // 重复RaInit时，调用此接口获取相同的rdmaHandle，防止重新生成
        ret = HrtRaRdmaGetHandle(devicePhysicID, rdmaHandle);
    } else {
        if (!ipAddrBackup.IsInvalid()) {
            struct rdev nicRdevInfoback;
            CHK_RET(hrtGetPairDevicePhyId(devicePhysicID, nicRdevInfoback.phyId));
            nicRdevInfoback.family = ipAddrBackup.GetFamily();
            nicRdevInfoback.localIp.addr = ipAddrBackup.GetBinaryAddress().addr;
            nicRdevInfoback.localIp.addr6 = ipAddrBackup.GetBinaryAddress().addr6;
            HCCL_DEBUG(
                "[%s]backup rdev info: ipAddr[%s], ipAddrBackup[%s]", __func__, ipAddr.GetReadableIP(),
                ipAddrBackup.GetReadableIP());
            ret = HrtRdmaInitWithBackupAttr(init_info, nicRdevInfo, nicRdevInfoback, rdmaHandle);
        } else {
            ret = HrtRaRdmaInitWithAttr(init_info, nicRdevInfo, rdmaHandle);
        }
    }

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][RDMA]errNo[0x%016llx] ra rdma init failed, devid[%u] ip[%s], notifyType[%d], return[%d]",
            HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhysicID, ipAddr.GetReadableAddress(), notifyType, ret),
        HCCL_E_TCP_CONNECT);
    HCCL_INFO(
        "devicePhyId[%u], ip[%s] disabledLiteThread[%u] enabled910aLite[%u] rdmaHandle[%p] rdma init OK",
        devicePhysicID, ipAddr.GetReadableAddress(), disabledLiteThread, enable910ALite, rdmaHandle);

    return HCCL_SUCCESS;
}

bool NetworkManager::GetRdmaLiteStatus() { return isRdmaLiteEn_; }

HcclResult NetworkManager::GetNotifyType(NotifyTypeT& notifyType) const
{
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType == DevType::DEV_TYPE_910 || deviceType == DevType::DEV_TYPE_910B
        || deviceType == DevType::DEV_TYPE_910_93) {
        notifyType = NOTIFY;
    } else if (deviceType == DevType::DEV_TYPE_310P3 || deviceType == DevType::DEV_TYPE_310P1) {
        notifyType = EVENTID;
    } else {
        HCCL_ERROR("[Init][DeviceRDMA]devType[%d] is invalid", deviceType);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

void NetworkManager::GetNetworkMode(NetworkMode& netMode) const
{
    if (Is310PDevice()) {
        netMode = NETWORK_PEER_ONLINE;
    } else {
        netMode = NETWORK_OFFLINE;
    }
}

HcclResult NetworkManager::InitDeviceSocket(u32 devicePhysicID, const HcclIpAddress& ipAddr, SocketHandle& socketHandle)
{
    struct rdev nicRdevInfo = {};
    nicRdevInfo.phyId = devicePhysicID;
    nicRdevInfo.family = ipAddr.GetFamily();
    nicRdevInfo.localIp.addr = ipAddr.GetBinaryAddress().addr;
    nicRdevInfo.localIp.addr6 = ipAddr.GetBinaryAddress().addr6;

    NetworkMode netMode;
    GetNetworkMode(netMode);

    HcclResult ret = hrtRaSocketInit(netMode, nicRdevInfo, socketHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][DeviceSocket]ra socket init failed, network mode[%d] devid[%u] ip[%s], return[%d]", netMode,
            devicePhysicID, ipAddr.GetReadableAddress(), ret),
        HCCL_E_TCP_CONNECT);
    HCCL_INFO("devicePhyId[%u], ip[%s] socket init OK", devicePhysicID, ipAddr.GetReadableAddress());

    return HCCL_SUCCESS;
}

HcclResult NetworkManager::StartHostNetAndListen(
    const HcclIpAddress& ipAddr, SocketHandle& socketHandle, u32& port, bool rdmaFlag, bool tagChkDis)
{
    CHK_PRT_RET(
        (hostNicInitRef_.Count() == 0),
        HCCL_ERROR("[Start][HostNetAndListen]cannot start nic socket before host nic inited!"), HCCL_E_INTERNAL);
    auto sockInfo = raResourceInfo_.hostNetSocketMap.find(ipAddr);
    if (sockInfo == raResourceInfo_.hostNetSocketMap.end()) {
        IpSocket tempSock;
        raResourceInfo_.hostNetSocketMap.insert(std::make_pair(ipAddr, tempSock)); // 本IP占位
        HCCL_INFO("device[%u] Start host nic insert Ip[%s]", devicePhyId_, ipAddr.GetReadableAddress());
    }

    HcclResult ret = HCCL_SUCCESS;
    IpSocket& sock = raResourceInfo_.hostNetSocketMap[ipAddr];
    if (sock.nicSocketHandle == nullptr) {
        ret = InitHostSocket(ipAddr, sock.nicSocketHandle);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Start][HostNetAndListen]start host socket failed, devid[%u], ip[%s] return[%d]", devicePhyId_,
                ipAddr.GetReadableAddress(), ret),
            ret);
    }
    if (sock.nicRdmaHandle == nullptr && rdmaFlag) {
        notifyType_ = EVENTID;
        ret = InitRDMA(devicePhyId_, ipAddr, NETWORK_PEER_ONLINE, notifyType_, sock.nicRdmaHandle);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Start][Nic]errNo[0x%016llx] ra nic init rdma failed, devid[%u], return[%d]",
                HCCL_ERROR_CODE(HCCL_E_NETWORK), devicePhyId_, ret),
            HCCL_E_NETWORK);
    }
    if (IPPortListenRefMapHost_[ipAddr][port].Count() == 0 && sock.listenedPort.find(port) == sock.listenedPort.end()) {
        bool isAutoPort = port == 0;
        HCCL_INFO("[Start][HostNetAndListen]trying to listen on ip[%s] port[%u].", ipAddr.GetReadableAddress(), port);
        ret = StartListenSocket(sock.nicSocketHandle, port, tagChkDis);
        CHK_PRT_RET(
            ret == HCCL_E_UNAVAIL,
            HCCL_INFO(
                "[Start][HostNetAndListen]Could not start listening socket for IP [%s] and port [%u].",
                ipAddr.GetReadableAddress(), port),
            ret);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Start][HostNetAndListen]start host socket failed, devid[%u], ip[%s], port[%u], return[%d]",
                devicePhyId_, ipAddr.GetReadableAddress(), port, ret),
            ret);
        sock.listenedPort.insert(port);
        HCCL_RUN_INFO(
            "[Start][HostNetAndListen]Listen on ip[%s], port[%u] success, "
            "devPhyId[%u], devLogicId[%u], isAutoPort[%d], tagChkDis[%u]",
            ipAddr.GetReadableAddress(), port, devicePhyId_, deviceLogicId_, isAutoPort, tagChkDis);
    }
    int refCount = IPPortListenRefMapHost_[ipAddr][port].Ref();
    HCCL_INFO("host ip[%s] port[%u] refcount is [%d]", ipAddr.GetReadableAddress(), port, refCount);

    socketHandle = sock.nicSocketHandle;
    raResourceInfo_.nicSocketMap.insert(std::make_pair(ipAddr, sock));
    hostNicSocketClientRef_[ipAddr].Ref();
    HCCL_INFO("HostNet, ip[%s] port[%u] tagChkDis[%u] socket init OK", ipAddr.GetReadableAddress(), port, tagChkDis);
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::CheckSocketInfo(const SocketHandle socketHandle, const HcclIpAddress& ipAddr, u32 port) const
{
    auto iterIP = raResourceInfo_.hostNetSocketMap.find(ipAddr);
    CHK_PRT_RET(
        (iterIP == raResourceInfo_.hostNetSocketMap.end()),
        HCCL_ERROR(
            "[Check][SocketInfo]ip[%s] port[%u] has not been started. ip is invalid.", ipAddr.GetReadableAddress(),
            port),
        HCCL_E_INTERNAL);

    CHK_PRT_RET(
        (socketHandle != iterIP->second.nicSocketHandle),
        HCCL_ERROR(
            "[Check][SocketInfo]ip[%s] port[%u] has not been started. socketHandle is invalid",
            ipAddr.GetReadableAddress(), port),
        HCCL_E_INTERNAL);

    if (port != NO_LISTEN_PORT) {
        CHK_PRT_RET(
            (iterIP->second.listenedPort.count(port) == 0),
            HCCL_ERROR(
                "[Check][SocketInfo]ip[%s] port[%u] has not been started. port is invalid", ipAddr.GetReadableAddress(),
                port),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::StopHostNetAndListen(SocketHandle socketHandle, const HcclIpAddress& ipAddr, u32 port)
{
    std::unique_lock<std::mutex> lock(raLock_);
    CHK_PRT_RET(
        (hostNicInitRef_.Count() == 0),
        HCCL_ERROR("[Stop][HostNetAndListen]cannot start nic socket before host nic inited!"), HCCL_E_INTERNAL);

    CHK_RET(CheckSocketInfo(socketHandle, ipAddr, port));

    if (IPPortListenRefMapHost_[ipAddr][port].Unref() == 0) {
        CHK_RET(StopListenSocket(socketHandle, port)); /* 当前只拉起一个server */

        HCCL_INFO("ip[%s] port[%u] stop success.", ipAddr.GetReadableAddress(), port);
        raResourceInfo_.hostNetSocketMap[ipAddr].listenedPort.erase(port);
        raResourceInfo_.nicSocketMap[ipAddr].listenedPort.erase(port);
    } else {
        HCCL_INFO(
            "ip[%s] port[%u] skip stop. ref[%d].", ipAddr.GetReadableAddress(), port,
            IPPortListenRefMapHost_[ipAddr][port].Count());
    }

    if (hostNicSocketClientRef_[ipAddr].Unref() == 0
        && raResourceInfo_.hostNetSocketMap[ipAddr].listenedPort.size() == 0) {
        // 在用(如心跳)时跳过deinit, 置位pendingStop交由最后一个用户释放时延迟拆除
        if (MarkNicUserPendingStop(ipAddr, true, false)) {
            return HCCL_SUCCESS;
        }
        CHK_RET(hrtRaSocketDeInit(socketHandle));
        raResourceInfo_.hostNetSocketMap.erase(ipAddr);
        raResourceInfo_.nicSocketMap.erase(ipAddr);
        HCCL_INFO("ip[%s] port[%u] deinit success.", ipAddr.GetReadableAddress(), port);
    }

    return HCCL_SUCCESS;
}
HcclResult NetworkManager::StopAllHostNicSockets()
{
    HcclResult ret = HCCL_SUCCESS;
    for (auto itSocket : raResourceInfo_.hostNetSocketMap) {
        // 在用条目跳过强拆, 统一延迟拆除; 值拷贝迭代无法回写置空, 强拆会残留悬空指针
        if (MarkNicUserPendingStop(itSocket.first, true, true)) {
            continue;
        }
        for (auto itPort : itSocket.second.listenedPort) {
            ret = (StopListenSocket(itSocket.second.nicSocketHandle, itPort));
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[Stop][StopAllHostNicSockets]errNo[0x%016llx] stop listen socket failed,"
                    "devid[%u], ip[%s], port[%u], return[%d].",
                    HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, itSocket.first.GetReadableAddress(), itPort,
                    ret);
            } else {
                HCCL_INFO("ip[%s] port[%u] stop success.", itSocket.first.GetReadableAddress(), itPort);
            }
        }
        ret = hrtRaSocketDeInit(itSocket.second.nicSocketHandle);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[Stop][StopAllHostNicSockets]errNo[0x%016llx] deinit socket failed, devid[%u], ip[%s], return[%d]",
                HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), devicePhyId_, itSocket.first.GetReadableAddress(), ret);
        } else {
            HCCL_INFO("ip[%s] deinit success.", itSocket.first.GetReadableAddress());
        }
        hostNicSocketClientRef_[itSocket.first].Clear();
    }

    // 在用(如心跳)条目保留, 由最后一个用户释放时延迟拆除, 无条件clear会泄漏handle
    for (auto it = raResourceInfo_.hostNetSocketMap.begin(); it != raResourceInfo_.hostNetSocketMap.end();) {
        if (MarkNicUserPendingStop(it->first, true, true)) {
            ++it;
            continue;
        }
        it = raResourceInfo_.hostNetSocketMap.erase(it);
    }
    return HCCL_SUCCESS;
}
/* destroy network resource */
HcclResult NetworkManager::Destroy()
{
    /* 停止nic ra的监听 */
    if (raResourceInfo_.nicSocketMap.size() != 0) {
        for (auto it : raResourceInfo_.nicSocketMap) {
            HCCL_WARNING("nicSocketMap[%s] is not stopped when NetworkManager Destroy", it.first.GetReadableAddress());
        }
        // StartHostNetAndListen等函数中，同一地址在nicSocketMap和hostNetSocketMap内同时插入
        // 此处StopAllDeviceNicSockets()销毁nicSocketMap后，StopAllHostNicSockets内会发生重复销毁导致core
        // 为了避免此种情况，同时尽量减少对既有函数的修改扩散影响，此处同步对hostNetSocketMap进行清理
        // 最终修改方案需要重构本类，解除两个MAP的耦合

        // 在用条目须保留host侧记录, 无条件erase会使host侧handle失去deinit入口而泄漏;
        // 同时置位socket与rdma的pending标志, 交由最后一个用户延迟拆除
        for (auto& it : raResourceInfo_.nicSocketMap) {
            if (MarkNicUserPendingStop(it.first, true, true)) {
                continue;
            }
            raResourceInfo_.hostNetSocketMap.erase(it.first); // key不存在则不会删除
        }
        StopAllDeviceNicSockets();
    }

    /* 停止vnic ra的监听 */
    if (raResourceInfo_.vnicSocketMap.size() != 0) {
        StopAllDeviceVnicSockets();
    }

    /* 停止host nic ra的监听 */
    if (raResourceInfo_.hostNetSocketMap.size() != 0) {
        for (auto it : raResourceInfo_.hostNetSocketMap) {
            HCCL_WARNING(
                "hostNicSocketMap[%s] is not stopped when NetworkManager Destroy", it.first.GetReadableAddress());
        }
        StopAllHostNicSockets();
    }

    /* 释放ra资源 重复RaInit时，不再调用内部DeInit */
    HCCL_DEBUG("Destroy call HrtRaDeInit.");
    if (deviceNicInitRef_.Count() != 0 && !isRaDeInit_) {
        HCCL_WARNING("device Nic is not deinit when NetworkManager Destroy. ref[%d]", deviceNicInitRef_.Count());
        struct RaInitConfig config = {DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE, isEnableHdcAsync_};
        GetDeviceRaInitConfig(config);
        config.nicPosition = static_cast<u32>(NICDeployment::NIC_DEPLOYMENT_DEVICE);
        if (HrtRaDeInit(&config) != HCCL_SUCCESS) {
            HCCL_ERROR("ra deinit failed. para: nicdeploy[%u], phyId[%u]", config.nicPosition, config.phyId);
        }

        isRaDeInit_ = true;
        deviceNicInitRef_.Clear();
    }
    if (hostNicInitRef_.Count() != 0) {
        HCCL_WARNING("host Nic is not deinit when NetworkManager Destroy. ref[%d]", hostNicInitRef_.Count());
        struct RaInitConfig config = {DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE, isEnableHdcAsync_};
        config.nicPosition = static_cast<u32>(NICDeployment::NIC_DEPLOYMENT_HOST);
        config.phyId = devicePhyId_;
        if (HrtRaDeInit(&config) != HCCL_SUCCESS) {
            HCCL_ERROR("ra deinit failed. para: nicdeploy[%u], phyId[%u]", config.nicPosition, config.phyId);
        }
        hostNicInitRef_.Clear();
    }
    HCCL_INFO("destroy all nic/vnic.");
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::StopNicsSocketListen(const HcclIpAddress& ipAddr, u32 port)
{
    auto it = raResourceInfo_.nicSocketMap.find(ipAddr);
    CHK_PRT_RET(
        it == raResourceInfo_.nicSocketMap.end(),
        HCCL_ERROR(
            "[Stop][NicsSocketPort]ip[%s] port[%u] is not found in nicSocketMap.", ipAddr.GetReadableAddress(), port),
        HCCL_E_INTERNAL);
    IpSocket& ipSock = it->second;
    bool portFound = false;
    for (auto itPort : ipSock.listenedPort) {
        if (itPort == port) {
            portFound = true;
        }
    }
    CHK_PRT_RET(
        !portFound,
        HCCL_ERROR("[Stop][NicsSocketPort]PORT(ip[%s] port[%u]) is not found.", ipAddr.GetReadableAddress(), port),
        HCCL_E_INTERNAL);
    if (ipSock.nicSocketHandle != nullptr && StopListenSocket(ipSock.nicSocketHandle, port)) {
        HCCL_ERROR("[Stop][NicsSocketPort]NIC socket listen is not stopped successfully");
        return HCCL_E_NETWORK;
    }
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::StopNicsSocket(const HcclIpAddress& ipAddr)
{
    auto it = raResourceInfo_.nicSocketMap.find(ipAddr);
    CHK_PRT_RET(
        it == raResourceInfo_.nicSocketMap.end(),
        HCCL_ERROR("[Stop][NicsSocket]ip[%s] is not found in nicSocketMap.", ipAddr.GetReadableAddress()),
        HCCL_E_INTERNAL);
    IpSocket& ipSock = it->second;
    // 在用时跳过拆除, 同时置位socket与rdma的pending标志, 交由最后一个用户延迟拆除
    if (MarkNicUserPendingStop(ipAddr, true, true)) {
        return HCCL_SUCCESS;
    }
    if (ipSock.nicRdmaHandle != nullptr && HrtRaRdmaDeInit(ipSock.nicRdmaHandle, notifyType_) != HCCL_SUCCESS) {
        HCCL_ERROR("[Stop][NicsSocket]NIC rdev deInit not successfully, notifyType_[%d]", notifyType_);
        return HCCL_E_NETWORK;
    }
    ipSock.nicRdmaHandle = nullptr;
    if (ipSock.nicSocketHandle != nullptr && hrtRaSocketDeInit(ipSock.nicSocketHandle) != HCCL_SUCCESS) {
        HCCL_ERROR("[Stop][NicsSocket]NIC socket deInit not successfully");
        return HCCL_E_NETWORK;
    }
    ipSock.nicSocketHandle = nullptr;
    return HCCL_SUCCESS;
}

HcclResult
NetworkManager::InitRdmaHandle(u32 devId, const HcclIpAddress& ipAddr, bool disabledLiteThread, bool enable910ALite)
{
    if (raResourceInfo_.nicSocketMap[ipAddr].nicRdmaHandle != nullptr
        && raResourceInfo_.hostNetSocketMap[ipAddr].nicRdmaHandle != nullptr) {
        HCCL_INFO("NetworkManager: RdmaInit already nic");
        return HCCL_SUCCESS;
    }

    u32 devicePhyId = ((static_cast<s32>(devId) == HOST_DEVICE_ID) ? 0 : devId);
    RdmaHandle rdmaHandle = nullptr;
    // 模式和notify类型按照是否为hdc模式进行赋值
    NetworkMode initRdmaMode = (isHostUseDevNic_) ? NETWORK_OFFLINE : NETWORK_PEER_ONLINE;
    NotifyTypeT notifyType = (isHostUseDevNic_) ? NotifyTypeT::NOTIFY : NotifyTypeT::NO_USE;
    HcclResult ret
        = InitRDMA(devicePhyId, ipAddr, initRdmaMode, notifyType, rdmaHandle, disabledLiteThread, enable910ALite);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][RDMA]errNo[0x%016llx] ra rdma init failed, ip[%s], return[%d]", HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT),
            ipAddr.GetReadableAddress(), ret),
        HCCL_E_TCP_CONNECT);
    HCCL_INFO("ip[%s] rdma init OK", ipAddr.GetReadableAddress());
    CHK_PTR_NULL(rdmaHandle);
    raResourceInfo_.nicSocketMap[ipAddr].nicRdmaHandle = rdmaHandle;
    raResourceInfo_.hostNetSocketMap[ipAddr].nicRdmaHandle = rdmaHandle;
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::GetRaResourceInfo(RaResourceInfo& raResourceInfo)
{
    raResourceInfo = raResourceInfo_;
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::PsWorkerRaInit(
    u32 devId, const HcclIpAddress& ipAddr, u32 port, bool isHostUseDevNic, [[maybe_unused]] bool remoteIsHdc,
    [[maybe_unused]] bool isBoardVersion)
{
    HCCL_INFO("PsWorkerRaInit, devicePhyId[%u], deviceLogicId_[%d]", devicePhyId_, deviceLogicId_);
    // 引用计数
    u32 devicePhyId = ((static_cast<s32>(devId) == HOST_DEVICE_ID) ? 0 : devId);
    std::string ipAddrStr(ipAddr.GetReadableAddress());
    if (ipAddrStr == "127.0.0.1") {
        hostNicInitRef_.Ref();
        HCCL_INFO("hostNicInitRef_[%d]", hostNicInitRef_.Count());
    } else {
        bool fistUsed{false};
        deviceNicInitRef_.Ref();
        CHK_RET(hrtRaIsFirstUsed(devicePhyId, fistUsed));
        HCCL_INFO(
            "deviceNicInitRef_[%d] firstUsed[%u] devicePhyId[%u]", deviceNicInitRef_.Count(), fistUsed, devicePhyId);
        if (deviceNicInitRef_.Count() == 1 && !fistUsed) {
            isRaInitRepeated_ = true;
        } else if (!fistUsed) {
            HCCL_INFO("[NetworkManager] PsWorkerRa is not firstUsed");
            return HCCL_SUCCESS;
        }
    }

    CHK_PRT_RET(
        (raResourceInfo_.nicSocketMap.count(ipAddr) != 0),
        HCCL_INFO("NetworkManager: PsWorkerRa already Init, ipAddr[%s]", ipAddr.GetReadableAddress()), HCCL_SUCCESS);
    isHostUseDevNic_ = isHostUseDevNic;

    // hdc模式下需要先拉起device上的hccp进程
    if (isHostUseDevNic_) {
        // 拉起device进程
        if (devId != INVALID_UINT) {
            devicePhyId_ = devId;
        } else {
            // 初始化ra资源(dev信息带入逻辑ID)
            CHK_RET(hrtGetDevice(&deviceLogicId_));
            CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId_), devicePhyId_));
        }

        // device 网卡初始化前需要拉起 hccp .
        rtNetServiceOpenArgs openArgs = {};
        rtProcExtParam extParam{};
        std::string extPam("--hdcType=" + std::to_string(PID_HDC_TYPE));

        extParam.paramInfo = extPam.c_str();
        extParam.paramLen = extPam.size();
        openArgs.extParamList = &extParam;
        openArgs.extParamCnt = 1UL;
        isTsdProcessOpen_ = true;
        CHK_RET(hrtOpenNetService(&openArgs));
        HCCL_INFO(
            "NetworkManager open tsd success, devicePhyId[%u], deviceLogicId_[%d], subPid[%lld]", devicePhyId_,
            deviceLogicId_, static_cast<s64>(subPid_));
    }

    RaInitConfig config = {DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE, false};
    config.phyId = devicePhyId;
    if (ipAddrStr == "127.0.0.1") {
        config.nicPosition = static_cast<u32>(NICDeployment::NIC_DEPLOYMENT_HOST);
    } else {
        config.nicPosition = static_cast<u32>(isHostUseDevNic_);
    }
    if (isHostUseDevNic_) {
        config.hdcType = PID_HDC_TYPE;
    }

    if (!isRaInitRepeated_) {
        HCCL_INFO(
            "PsWorkerRaInit call HrtRaInit. devicePhyId[%u] isRaInitRepeated_[%u]", devicePhyId_, isRaInitRepeated_);
        HcclResult hcclRet = HrtRaInit(&config);
        if (hcclRet != HCCL_SUCCESS) {
            HCCL_ERROR("[PsWorkerRaInit] ra init failed, ret[%d].", hcclRet);
            if (isTsdProcessOpen_) {
                (void)hrtCloseNetService();
                isTsdProcessOpen_ = false;
            }
            return hcclRet;
        }
    }

    struct rdev nicRdevInfo = {};
    nicRdevInfo.phyId = devicePhyId;
    nicRdevInfo.family = ipAddr.GetFamily();
    nicRdevInfo.localIp.addr = ipAddr.GetBinaryAddress().addr;
    nicRdevInfo.localIp.addr6 = ipAddr.GetBinaryAddress().addr6;
    SocketHandle socketHandle = nullptr;
    NetworkMode raSocketInitMode = (isHostUseDevNic_) ? NETWORK_OFFLINE : NETWORK_PEER_ONLINE;
    HcclResult ret = hrtRaSocketInit(raSocketInitMode, nicRdevInfo, socketHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[Init][HostSocket]errNo[0x%016llx] ra socket init failed, ip[%s], return[%d]",
            HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), ipAddr.GetReadableAddress(), ret);
        if (!isRaInitRepeated_) {
            (void)HrtRaDeInit(&config);
        }
        if (isTsdProcessOpen_) {
            (void)hrtCloseNetService();
            isTsdProcessOpen_ = false;
        }
        return HCCL_E_TCP_CONNECT;
    }
    HCCL_INFO(
        "ip[%s] socket init OK, devicePhyId_[%u], socketHandle[%llu]", ipAddr.GetReadableAddress(), devicePhyId_,
        hash<void*>{}(socketHandle));

    IpSocket ipSocketInfo;
    ipSocketInfo.nicSocketHandle = socketHandle;

    raResourceInfo_.nicSocketMap.insert(std::make_pair(ipAddr, ipSocketInfo));
    raResourceInfo_.hostNetSocketMap.insert(std::make_pair(ipAddr, ipSocketInfo));

    ret = HeterogStartListen(ipAddr, port);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[PsWorkerRaInit] HeterogStartListen failed, ret[%d]", ret);
        (void)hrtRaSocketDeInit(socketHandle);
        raResourceInfo_.nicSocketMap.erase(ipAddr);
        raResourceInfo_.hostNetSocketMap.erase(ipAddr);
        if (!isRaInitRepeated_) {
            (void)HrtRaDeInit(&config);
        }
        if (isTsdProcessOpen_) {
            (void)hrtCloseNetService();
            isTsdProcessOpen_ = false;
        }
        return ret;
    }

    return HCCL_SUCCESS;
}

// 最后一次ra_deinit时才关闭device的hccp进程。ES场景主要使用
HcclResult NetworkManager::CloseHccpSubProc()
{
    if (!isHostUseDevNic_ || subPid_ == 0) {
        HCCL_INFO("No need to close hccp sub proc, devicePhyId[%u], subPid[%lld]", devicePhyId_, subPid_);
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "NetworkManager ProcessCloseSubProcList HDC devicePhyId[%u], deviceLogicId_[%d], subPid[%lld]", devicePhyId_,
        deviceLogicId_, static_cast<s64>(subPid_));
    s32 locaLogDevid = 0;
    hrtGetDevice(&locaLogDevid);
    if (locaLogDevid != deviceLogicId_) {
        hrtSetDevice(deviceLogicId_);
    }
    HcclResult ret = hrtCloseNetService();
    if (locaLogDevid != deviceLogicId_) {
        hrtSetDevice(locaLogDevid);
    }
    CHK_RET(ret);
    subPid_ = 0;

    return HCCL_SUCCESS;
}

HcclResult NetworkManager::PingMeshRaPingInit(u32 devLogicId, u32 devPhyId, RaInitConfig* config)
{
    // 引用计数
    deviceLogicId_ = static_cast<s32>(devLogicId);
    devicePhyId_ = ((static_cast<s32>(devPhyId) == HOST_DEVICE_ID) ? 0 : devPhyId);
    isHostUseDevNic_ = true;
    isRaInitRepeated_ = false;

    // hccp侧初始化ping mesh资源
    CHK_RET(HrtRaInit(config));
    HCCL_INFO(
        "[HCCN][PingMeshRaPingInit]Device[%u] config.hdcType[%d], config.nicPosition[%u], config.phyId[%u].",
        deviceLogicId_, config->hdcType, config->nicPosition, config->phyId);
    deviceNicInitRef_.Ref();

    return HCCL_SUCCESS;
}

HcclResult NetworkManager::PingMeshRaPingDeinit()
{
    // 引用计数
    isRaInitRepeated_ = false;

    struct RaInitConfig config = {DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE, false};
    GetDeviceRaInitConfig(config);
    CHK_RET(HrtRaDeInit(&config));
    deviceNicInitRef_.Unref();

    return HCCL_SUCCESS;
}

void NetworkManager::GetDeviceRaInitConfig(RaInitConfig& config)
{
    u32 devicePhyId = ((static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) ? 0 : devicePhyId_);
    HCCL_INFO("RaDeinit devicePhyId_[%u] devicePhyId[%u]", devicePhyId_, devicePhyId);

    config.phyId = devicePhyId;
    config.nicPosition = static_cast<u32>(isHostUseDevNic_);

    if (isHostUseDevNic_) {
        config.hdcType = PID_HDC_TYPE;
    }
}

HcclResult NetworkManager::PsWorkerRaDeinit(u32 devId, const HcclIpAddress& ipAddr, u32 port)
{
    // 在用时跳过整个Deinit, 条目留给延迟拆除; 守卫须在Unref之前, 否则计数归零后
    // 进程退出时Destroy()的兜底失效, Ra资源泄漏
    if (MarkNicUserPendingStop(ipAddr, true, true)) {
        HCCL_INFO("[NetworkManager][PsWorkerRaDeinit]ip[%s] in use, defer teardown.", ipAddr.GetReadableAddress());
        return HCCL_SUCCESS;
    }
    string ipAddrStr(ipAddr.GetReadableAddress());
    u32 devicePhyId = ((static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) ? 0 : devicePhyId_);
    if (ipAddrStr == "127.0.0.1") {
        hostNicInitRef_.Unref();
        HCCL_INFO("hostNicInitRef_[%d]", hostNicInitRef_.Count());
    } else {
        bool lastUsed{false};
        deviceNicInitRef_.Unref();
        CHK_RET(hrtRaIsLastUsed(devicePhyId, lastUsed));
        HCCL_INFO(
            "deviceNicInitRef_[%d] lastUsed[%u] devicePhyId[%u]", deviceNicInitRef_.Count(), lastUsed, devicePhyId);
        if (deviceNicInitRef_.Count() == 0 && !lastUsed) {
            isRaInitRepeated_ = true;
        } else if (lastUsed) {
            isRaInitRepeated_ = false;
        } else if (deviceNicInitRef_.Count() > 0) {
            HCCL_INFO("[NetworkManager] PsWorkerRa is not lastUsed");
            return HCCL_SUCCESS;
        }
    }
    CHK_PRT_RET(
        (raResourceInfo_.nicSocketMap.count(ipAddr) == 0),
        HCCL_INFO("NetworkManager: PsWorkerRa already Deinit, ipAddr[%s]", ipAddr.GetReadableAddress()), HCCL_SUCCESS);

    HCCL_INFO("PsWorkerRaDeinit devId[%u], isRaInitRepeated[%d]", devId, isRaInitRepeated_);
    // 重复RaInit时，不再调用内部DeInit
    if (!isRaInitRepeated_) {
        CHK_RET(HeterogStopListen(ipAddr, port, true));
    }

    SocketHandle nicRdmaHandle = raResourceInfo_.nicSocketMap[ipAddr].nicRdmaHandle;
    if (!GetExternalInputHcclIsTcpMode() && nicRdmaHandle != nullptr && !isRaInitRepeated_) {
        // Helper的PS临时暂不调用 CHK_RET(HrtRaRdmaDeInit(nicRdmaHandle, (isHostUseDevNic_) ?
        // NotifyTypeT::NOTIFY : NotifyTypeT::NO_USE));
        HCCL_INFO("Not call RaRdmaDeInit devicePhyId[%u]", devicePhyId_);
    }

    struct RaInitConfig config = {DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE, false};
    GetDeviceRaInitConfig(config);

    if (ipAddrStr == "127.0.0.1") {
        if (hostNicInitRef_.Count() == 0) {
            HCCL_INFO("PsWorkerRaDeinit call hrtRaDeInit. devicePhyId[%u]", devicePhyId_);
            config.nicPosition = static_cast<u32>(NICDeployment::NIC_DEPLOYMENT_HOST);
            CHK_RET(HrtRaDeInit(&config));
        }
    } else {
        if (deviceNicInitRef_.Count() == 0) {
            HCCL_INFO("PsWorkerRaDeinit call HrtRaDeInit. devicePhyId[%u]", devicePhyId_);
            if (!isRaInitRepeated_) {
                CHK_RET(HrtRaDeInit(&config));
                CHK_RET(CloseHccpSubProc());
            }
        }
    }

    raResourceInfo_.nicSocketMap.erase(raResourceInfo_.nicSocketMap.find(ipAddr));
    raResourceInfo_.hostNetSocketMap.erase(raResourceInfo_.hostNetSocketMap.find(ipAddr));

    return HCCL_SUCCESS;
}

HcclResult NetworkManager::InitHostSocket(const HcclIpAddress& addr, SocketHandle& socketHandle) const
{
    struct SocketInitInfoT socketInitInfo = {};
    socketInitInfo.rdevInfo.family = addr.GetFamily();
    socketInitInfo.rdevInfo.phyId = devicePhyId_;
    socketInitInfo.rdevInfo.localIp.addr = addr.GetBinaryAddress().addr;
    socketInitInfo.rdevInfo.localIp.addr6 = addr.GetBinaryAddress().addr6;
    socketInitInfo.scopeId = addr.GetScopeID();
    HcclResult ret = hrtRaSocketInitV1(NETWORK_PEER_ONLINE, socketInitInfo, socketHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][HostSocket]errNo[0x%016llx] ra socket init v1 failed, ip[%s], return[%d]",
            HCCL_ERROR_CODE(HCCL_E_TCP_CONNECT), addr.GetReadableAddress(), ret),
        HCCL_E_TCP_CONNECT);
    HCCL_INFO("ip[%s] socket init OK", addr.GetReadableAddress());
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::StopHostNet(SocketHandle socketHandle, const HcclIpAddress& ipAddr)
{
    std::unique_lock<std::mutex> lock(raLock_);
    CHK_PRT_RET(
        (hostNicInitRef_.Count() == 0), HCCL_ERROR("[Stop][HostNet]cannot start nic socket before host nic inited!"),
        HCCL_E_INTERNAL);

    CHK_RET(CheckSocketInfo(socketHandle, ipAddr));

    if (hostNicSocketClientRef_[ipAddr].Unref() == 0
        && raResourceInfo_.hostNetSocketMap[ipAddr].listenedPort.size() == 0) {
        // 在用(如心跳)时跳过deinit, 由最后一个用户释放时延迟拆除
        if (MarkNicUserPendingStop(ipAddr, true, false)) {
            return HCCL_SUCCESS;
        }
        CHK_RET(hrtRaSocketDeInit(socketHandle));
        HCCL_INFO("ip[%s] deinit success.", ipAddr.GetReadableAddress());

        raResourceInfo_.hostNetSocketMap.erase(ipAddr);
        raResourceInfo_.nicSocketMap.erase(ipAddr);
    }
    return HCCL_SUCCESS;
}

// 从ip查handle 把ip下所有的listen全stop
HcclResult NetworkManager::StopHostSocketHandle(const HcclIpAddress& ipAddr)
{
    std::unique_lock<std::mutex> lock(raLock_);
    CHK_PRT_RET(
        (hostNicInitRef_.Count() == 0),
        HCCL_ERROR("[NetworkManager][StopHostSocketHandle]cannot start nic socket before host nic inited!"),
        HCCL_E_INTERNAL);

    auto sockInfo = raResourceInfo_.hostNetSocketMap.find(ipAddr);
    auto ipIt = IPPortListenRefMapHost_.find(ipAddr);
    CHK_PRT_RET(
        (sockInfo == raResourceInfo_.hostNetSocketMap.end()),
        HCCL_ERROR("[NetworkManager][StopHostSocketHandle]ipAddr is invalid"), HCCL_E_INTERNAL);
    IpSocket& sock = raResourceInfo_.hostNetSocketMap[ipAddr];
    // 在用(如心跳)时跳过拆除, 置位pendingStop交由最后一个用户释放时延迟拆除
    if (MarkNicUserPendingStop(ipAddr, true, false)) {
        return HCCL_SUCCESS;
    }
    CHK_RET(CheckSocketInfo(sock.nicSocketHandle, ipAddr));

    // 停止所有的listen
    if (ipIt != IPPortListenRefMapHost_.end()) {
        for (auto& portIt : ipIt->second) {
            u32 port = portIt.first;
            if (IPPortListenRefMapHost_[ipAddr][port].Count() > 0) {
                CHK_RET(StopListenSocket(sock.nicSocketHandle, port));
                HCCL_WARNING(
                    "[NetworkManager][StopHostSocketHandle] ip [%s] stop listen port [%u]", ipAddr.GetReadableAddress(),
                    port);
                raResourceInfo_.hostNetSocketMap[ipAddr].listenedPort.erase(port);
                IPPortListenRefMapHost_[ipAddr][port].Clear(); // 引用计数归0
            }
        }
    }
    CHK_PRT_RET(
        (raResourceInfo_.hostNetSocketMap[ipAddr].listenedPort.size() != 0),
        HCCL_ERROR(
            "[NetworkManager][StopHostSocketHandle]IPPortListenRefMapHost_[%s] is unequal to "
            "hostNetSocketMap[%s].listenedPort",
            ipAddr.GetReadableAddress(), ipAddr.GetReadableAddress()),
        HCCL_E_INTERNAL);
    // 删除socket 删除IP
    SocketHandle deinitSocketHandle = sock.nicSocketHandle;
    CHK_RET(hrtRaSocketDeInit(sock.nicSocketHandle));
    HCCL_INFO("[NetworkManager][StopHostSocketHandle] ip [%s] deinit success.", ipAddr.GetReadableAddress());
    sock.nicSocketHandle = nullptr;
    // 两个map记录同一handle(PsWorkerRaInit双写): 一侧deinit后同步置空另一侧, 防止二次释放
    if (deinitSocketHandle != nullptr) {
        auto nicIt = raResourceInfo_.nicSocketMap.find(ipAddr);
        if (nicIt != raResourceInfo_.nicSocketMap.end() && nicIt->second.nicSocketHandle == deinitSocketHandle) {
            nicIt->second.nicSocketHandle = nullptr;
        }
    }
    // 没有host和rdma时 删除ip
    if (sock.nicSocketHandle == nullptr && sock.nicRdmaHandle == nullptr) {
        raResourceInfo_.hostNetSocketMap.erase(ipAddr);
    }
    hostNicSocketClientRef_[ipAddr].Clear();
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::StartHostNet(const HcclIpAddress& ipAddr, SocketHandle& socketHandle)
{
    CHK_RET(CreateHostSocketHandle(ipAddr, socketHandle));
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::CreateHostSocketHandle(const HcclIpAddress& ipAddr, SocketHandle& socketHandle)
{
    CHK_PRT_RET(
        (hostNicInitRef_.Count() == 0),
        HCCL_ERROR("[CreateHostSocketHandle]cannot start nic socket before host nic inited!"), HCCL_E_INTERNAL);
    OccupyIp(ipAddr, raResourceInfo_.hostNetSocketMap);
    IpSocket& sock = raResourceInfo_.hostNetSocketMap[ipAddr];
    if (sock.nicSocketHandle == nullptr) {
        CHK_RET(InitHostSocket(ipAddr, sock.nicSocketHandle));
    }

    socketHandle = sock.nicSocketHandle;
    HCCL_INFO("ip[%s] socket start success socketHandle[%p]", ipAddr.GetReadableAddress(), socketHandle);
    hostNicSocketClientRef_[ipAddr].Ref(); // 引用计数
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::StartListenSocket(const SocketHandle socketHandle, u32& port, bool tagChkDis) const
{
    struct SocketListenInfoT serverInfo = {};
    serverInfo.socketHandle = const_cast<SocketHandle>(socketHandle);
    serverInfo.port = port;
    serverInfo.tagChkDis = tagChkDis;
    HCCL_INFO("StartListenSocket, port[%u], tagChkDis[%u]", port, tagChkDis);
    if (isRaInitRepeated_) {
        return HCCL_SUCCESS;
    }
    bool isAutoPort = port == AUTO_LISTEN_PORT;
    HcclResult ret = hrtRaSocketListenStart(&serverInfo, 1);
    CHK_PRT_RET(
        ret == HCCL_E_UNAVAIL,
        HCCL_INFO(
            "socket port[%u] has already been bound. Could not start listening host nic. please use an idle port.",
            port),
        ret);
    RPT_CALL_ERR(ret != HCCL_SUCCESS, "host nic listen start failed, port[%u], return[%d]", port, ret);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("host nic listen start failed, port[%u], return[%d]", port, ret), ret);
    if (isAutoPort) {
        port = serverInfo.port;
        CHK_PRT_RET(
            port == AUTO_LISTEN_PORT, HCCL_ERROR("start listen on a port selected by os automatically failed"),
            HCCL_E_NOT_SUPPORT);
        HCCL_RUN_INFO("start listen on port[%u] tagChkDis[%u] by auto success.", port, tagChkDis);
    }
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::StopListenSocket(const SocketHandle socketHandle, u32 port) const
{
    struct SocketListenInfoT serverInfo = {};
    serverInfo.socketHandle = const_cast<SocketHandle>(socketHandle);
    serverInfo.port = port;
    HcclResult ret = hrtRaSocketListenStop(&serverInfo, 1);
    RPT_CALL_ERR(ret != HCCL_SUCCESS, "socket listen stop failed, port[%u], return[%d]", port, ret);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("socket listen stop failed, port[%u], return[%d]", port, ret), ret);
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::GetRdmaHandleByIpAddr(const HcclIpAddress& ipAddr, RdmaHandle& rdmaHandle)
{
    auto it = raResourceInfo_.nicSocketMap.find(ipAddr);
    CHK_PRT_RET(
        it == raResourceInfo_.nicSocketMap.end(),
        HCCL_ERROR("GetRdmaHandleByIpAddr ip[%s] is not found in nicSocketMap.", ipAddr.GetReadableAddress()),
        HCCL_E_INTERNAL);
    rdmaHandle = raResourceInfo_.nicSocketMap[ipAddr].nicRdmaHandle;
    CHK_PTR_NULL(rdmaHandle);
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::GetNicHandleByIpAddr(const HcclIpAddress& ipAddr, SocketHandle& nicHandle)
{
    auto it = raResourceInfo_.nicSocketMap.find(ipAddr);
    CHK_PRT_RET(
        it == raResourceInfo_.nicSocketMap.end(),
        HCCL_ERROR("GetNicHandleByIpAddr ip[%s] is not found in nicSocketMap.", ipAddr.GetReadableAddress()),
        HCCL_E_INTERNAL);
    nicHandle = raResourceInfo_.nicSocketMap[ipAddr].nicSocketHandle;
    CHK_PTR_NULL(nicHandle);
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::CheckAutoListenVersion(bool isAutoPort)
{
    if (isAutoPort) {
        u32 listenStartVersion = 0;
        HcclResult vRet = hrtRaGetInterfaceVersion(devicePhyId_, SOCKET_LISTEN_AUTO_INTERFACE, &listenStartVersion);
        HCCL_INFO("[CheckAutoListenVersion] listen start version[%u].", listenStartVersion);
        CHK_PRT_RET(
            vRet != HCCL_SUCCESS || listenStartVersion < SOCKET_LISTEN_AUTO_INTERFACE_VERSION,
            HCCL_ERROR(
                "this package does not support hrtRaSocketNonBlockListenStart to "
                "listen automatically with port %u, please change new package.",
                AUTO_LISTEN_PORT),
            HCCL_E_NOT_SUPPORT);
    }
    return HCCL_SUCCESS;
}

// 新旧ip类型转换
HcclResult NetworkManager::HcclIpAddressConvertHcclAddr(HcclAddress* hccladdr, HcclIpAddress* hcclIP)
{
    CHK_PTR_NULL(hcclIP);
    CHK_PTR_NULL(hccladdr);
    if (hcclIP->GetFamily() == AF_INET) {
        hccladdr->type = HCCL_ADDR_TYPE_IP_V4;
        hccladdr->addr = hcclIP->GetBinaryAddress().addr;
    } else if (hcclIP->GetFamily() == AF_INET6) {
        hccladdr->type = HCCL_ADDR_TYPE_IP_V6;
        hccladdr->addr6 = hcclIP->GetBinaryAddress().addr6;
    } else {
        HCCL_ERROR("[HcclIpAddressConvertingHcclAddr]ERROR IP type!");
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult NetworkManager::OccupyIp(const HcclIpAddress& ipAddr, std::map<hccl::HcclIpAddress, IpSocket>& socketMap)
{
    auto sockInfo = socketMap.find(ipAddr);
    if (sockInfo == socketMap.end()) {
        IpSocket tempSock;
        socketMap.insert(std::make_pair(ipAddr, tempSock)); // 本IP占位
        HCCL_INFO("[NetworkManager][OccupyIp] device[%u] insert ip[%s]", devicePhyId_, ipAddr.GetReadableAddress());
    } else {
        HCCL_INFO(
            "[NetworkManager][OccupyIp] device[%u] ip[%s] has already occupied.", devicePhyId_,
            ipAddr.GetReadableAddress());
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
