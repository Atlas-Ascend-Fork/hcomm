/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "root_info_detect_bridge.h"

#include <memory>
#include <mutex>

#include "acl/acl_rt.h"
#include "hccl_common_v2.h"
#include "log.h"
#include "orion_adapter_rts.h"
#include "rank_info_detect.h"
#include "rank_info_detect_scalable.h"
#include "root_group_strategy.h"
#include "securec.h"

namespace Hccl {
namespace {

    HcclResult GetCurrentDeviceIds(s32& userDevId, s32& devPhyId)
    {
        CHK_RET(HrtGetDevice(userDevId));
        aclError aclRet = aclrtGetPhyDevIdByUserDevId(userDevId, &devPhyId);
        CHK_PRT_RET(
            aclRet != ACL_SUCCESS,
            HCCL_ERROR("[%s] aclrtGetPhyDevIdByUserDevId failed, userDevId[%d], ret[%d]", __func__, userDevId, aclRet),
            HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    // 公共实现：创建 RankInfoDetect、可选设置派生标识、SetupServer、序列化 rootHandle。
    // topParentId 为 nullptr 时走初始化路径；非 nullptr 时走派生路径。
    HcclResult GetRootInfoInternal(
        HcclRootInfo* rootInfo, const char* topParentId, uint32_t generationIndex,
        RootInfoDetectBridge::DetectContext& detectContext)
    {
        CHK_PTR_NULL(rootInfo);
        HcclUs startut = TIME_NOW();

        HcclRootHandleV2 rootHandle{};
        std::shared_ptr<RankInfoDetect> rankInfoDetectServer;
        EXCEPTION_CATCH(rankInfoDetectServer = std::make_shared<RankInfoDetect>(), return HCCL_E_MEMORY);

        // 在 SetupServer 前设置派生 identifier，确保 socket tag、whitelist tag 全部使用派生 identifier
        if (topParentId != nullptr) {
            CHK_RET(rankInfoDetectServer->SetDerivationIdentifier(std::string(topParentId), generationIndex));
            // 填充派生谱系字段，供新 rank 回填 topParentIdentifier_/generationIndex_，与 survivor 一致
            CHK_SAFETY_FUNC_RET(memcpy_s(
                rootHandle.topParentIdentifier, sizeof(rootHandle.topParentIdentifier), topParentId,
                std::string(topParentId).size() + 1));
            rootHandle.generationIndex = generationIndex + 1; // 派生代 = 父代 + 1
            rootHandle.valid = 1;
        }
        CHK_RET(rankInfoDetectServer->SetupServer(rootHandle));

        // detect 所有权经 detectContext 返回给调用方，由调用方存入 CommManager 单例或 CollComm。
        detectContext = std::move(rankInfoDetectServer);

        u32 rootHandleLen = sizeof(HcclRootHandleV2);
        CHK_PRT_RET(
            rootHandleLen > HCCL_ROOT_INFO_BYTES,
            HCCL_ERROR(
                "[%s] hccl root info overflow. max length: %u, actual:%zu, identifier[%s]", __func__,
                HCCL_ROOT_INFO_BYTES, rootHandleLen, rootHandle.identifier),
            HCCL_E_INTERNAL);

        // 对外 HcclRootInfo 仍使用固定大小的不透明缓冲区，避免迁移实现后改变既有接口布局。
        s32 sRet = memcpy_s(rootInfo->internal, HCCL_ROOT_INFO_BYTES, &rootHandle, rootHandleLen);
        CHK_PRT_RET(
            sRet != EOK,
            HCCL_ERROR(
                "[%s] memcpy root info fail. errno[%d] params:destMaxSize[%u], count[%u]", __func__, sRet,
                HCCL_ROOT_INFO_BYTES, rootHandleLen),
            HCCL_E_MEMORY);

        s32 userDevId = 0;
        s32 devPhyId = 0;
        CHK_RET(GetCurrentDeviceIds(userDevId, devPhyId));
        HCCL_RUN_INFO(
            "HcclGetRootInfoV2 success, take time [%lld]us, rootinfo: host ip[%s] port[%u] "
            "netMode[%s] identifier[%s], userDevId[%d], devPhyId[%d]",
            DURATION_US(TIME_NOW() - startut), rootHandle.ip, rootHandle.listenPort,
            rootHandle.netMode.Describe().c_str(), rootHandle.identifier, userDevId, devPhyId);
        return HCCL_SUCCESS;
    }

    // Provider 编译进 hcomm；hccl_v2 中保留的兼容入口只通过 RootInfoDetectBridge 转发到这里。
    HcclResult GetRootInfoImpl(HcclRootInfo* rootInfo, RootInfoDetectBridge::DetectContext& detectContext)
    {
        HCCL_RUN_INFO("Entry-HcclGetRootInfo V950");
        return GetRootInfoInternal(rootInfo, nullptr, 0, detectContext);
    }

    // root 进程复用入参 detect（已 SetupServer），非 root 进程新建 detect 走 SetupAgent。
    HcclResult DetectRankTableImpl(
        u32 nRanks, u32 rank, const HcclRootHandleV2& rootHandle, RankTableInfo& rankTable,
        RootInfoDetectBridge::DetectContext& detectContext, const char* netLayerExpression)
    {
        hccl::NetLayerSelection selection;
        if (netLayerExpression != nullptr) {
            CHK_RET(hccl::ReadNetLayerSelection(netLayerExpression, selection));
        }
        s32 userDevId = 0;
        s32 devPhyId = 0;
        CHK_RET(GetCurrentDeviceIds(userDevId, devPhyId));
        HCCL_RUN_INFO(
            "[%s] nRanks[%u], rank[%u] entry flat topo detect, rootinfo: host ip[%s] port[%u] "
            "netMode[%s] identifier[%s], userDevId[%d], devPhyId[%d]",
            __func__, nRanks, rank, rootHandle.ip, rootHandle.listenPort, rootHandle.netMode.Describe().c_str(),
            rootHandle.identifier, userDevId, devPhyId);

        // root 进程：复用入参 detectContext 中的 server detect（已 SetupServer，listen 线程在跑）。
        // 非 root 进程：入参 detectContext 为空或 devPhyId 不匹配，新建 detect 走 SetupAgent。
        std::shared_ptr<RankInfoDetect> rankInfoDetectAgent;
        auto inputDetect = std::static_pointer_cast<RankInfoDetect>(detectContext);
        if (inputDetect != nullptr && inputDetect->IsRoot()
            && inputDetect->GetDevPhyId() == static_cast<u32>(devPhyId)) {
            rankInfoDetectAgent = inputDetect;
        } else {
            if (inputDetect != nullptr) {
                EXCEPTION_CATCH(inputDetect->Shutdown(), return HCCL_E_INTERNAL);
                inputDetect.reset();
            }
            EXCEPTION_CATCH(rankInfoDetectAgent = std::make_shared<RankInfoDetect>(), return HCCL_E_MEMORY);
        }
        HCCL_INFO(
            "[RankInfoDetect::%s] detect rank[%u], ptr[%p], isRoot[%d].", __func__, rank,
            static_cast<void*>(rankInfoDetectAgent.get()), static_cast<int>(rankInfoDetectAgent->IsRoot()));

        rankInfoDetectAgent->SetNetLayerSelection(selection);
        CHK_RET(rankInfoDetectAgent->SetupAgent(nRanks, rank, rootHandle));
        CHK_RET(rankInfoDetectAgent->WaitComplete(rootHandle.listenPort, RANKINFO_DETECT_SERVER_STATUS_IDLE));

        rankInfoDetectAgent->GetRankTable(rankTable);
        detectContext = rankInfoDetectAgent;
        HCCL_RUN_INFO("[%s] end.", __func__);
        return HCCL_SUCCESS;
    }

    // 多 root 初始化使用
    HcclResult GetRootInfoScalableImpl(HcclRootInfo* rootInfo)
    {
        HCCL_RUN_INFO("Entry-HcclGetRootInfoScalable V950");
        CHK_PTR_NULL(rootInfo);
        HcclUs startut = TIME_NOW();

        s32 userDevId = 0;
        s32 devPhyId = 0;
        CHK_RET(GetCurrentDeviceIds(userDevId, devPhyId));
        s32 devLogicId = 0;
        aclError aclRet = aclrtGetLogicDevIdByUserDevId(userDevId, &devLogicId); // userDevId 转 logicDevId
        CHK_PRT_RET(
            aclRet != ACL_SUCCESS,
            HCCL_ERROR(
                "[%s] aclrtGetLogicDevIdByUserDevId failed, userDevId[%d], ret[%d]", __func__, userDevId, aclRet),
            HCCL_E_INTERNAL);

        HcclScalableRootHandleV2 scalableHandle{};
        std::shared_ptr<RankInfoDetectScalable> scalableServer;
        EXCEPTION_CATCH(
            (scalableServer
             = std::make_shared<RankInfoDetectScalable>(devLogicId, userDevId, static_cast<u32>(devPhyId))),
            return HCCL_E_MEMORY);
        CHK_RET(scalableServer->SetupScalableRoot(scalableHandle));

        const u32 handleLen = sizeof(HcclScalableRootHandleV2);
        CHK_PRT_RET(
            handleLen > HCCL_ROOT_INFO_BYTES,
            HCCL_ERROR(
                "[%s] hccl scalable root info overflow. max length: %u, actual:%u, identifier[%s]", __func__,
                HCCL_ROOT_INFO_BYTES, handleLen, scalableHandle.rootHandle.identifier),
            HCCL_E_INTERNAL);

        s32 sRet = memcpy_s(rootInfo->internal, HCCL_ROOT_INFO_BYTES, &scalableHandle, handleLen);
        CHK_PRT_RET(
            sRet != EOK,
            HCCL_ERROR(
                "[%s] memcpy scalable root info fail. errno[%d] params:destMaxSize[%u], count[%u]", __func__, sRet,
                HCCL_ROOT_INFO_BYTES, handleLen),
            HCCL_E_MEMORY);

        HCCL_RUN_INFO(
            "HcclGetRootInfoScalableV2 success, take time [%lld]us, rootinfo: host ip[%s] port[%u] meshPort[%u] "
            "netMode[%s], identifier[%s], userDevId[%d], devPhyId[%d]",
            DURATION_US(TIME_NOW() - startut), scalableHandle.rootHandle.ip, scalableHandle.rootHandle.listenPort,
            scalableHandle.meshPort, scalableHandle.rootHandle.netMode.Describe().c_str(),
            scalableHandle.rootHandle.identifier, userDevId, devPhyId);
        return HCCL_SUCCESS;
    }

    // root rank 信息设置：查注册表 → 组装 ScalableServerInfo（全量 meshInfos）→ 白名单下发 → 唤醒 server 线程。
    HcclResult InjectScalableServerInfo(
        u32 nRoot, u32 rootIndex, u32 rank, u32 groupSize, u32 groupFirstRank,
        const std::vector<HcclScalableRootHandleV2>& rootHandles)
    {
        HCCL_INFO(
            "[%s] start, nRoot[%u], rootIndex[%u], rank[%u], groupSize[%u], groupFirstRank[%u]", __func__, nRoot,
            rootIndex, rank, groupSize, groupFirstRank);
        const HcclScalableRootHandleV2& selfHandle = rootHandles[rootIndex];
        // 注入与 server 线程收尾以同一把注册表锁互斥：server 线程在释放资源前先摘除注册表条目，
        // 本处持锁完成 Find→白名单下发→注入，保证资源释放期间不再有注入方访问对象成员。
        std::lock_guard<std::mutex> lock(GetScalableDetectRegistryMutex());
        auto iter = GetScalableDetectRegistry().Find(selfHandle.rootHandle.identifier);
        CHK_PRT_RET(
            !iter.second,
            HCCL_ERROR(
                "[%s] scalable root server[%s] not found in registry, rank[%u], rootIndex[%u]. Possible causes: "
                "1) rootInfoList was not generated by HcclGetRootInfoScalable (e.g. handles from "
                "HcclGetRootInfo); 2) HcclGetRootInfoScalable was not called on this rank before init; "
                "3) rootInfoList content/order mismatch (entry[rootIndex] is not this rank's handle).",
                __func__, selfHandle.rootHandle.identifier, rank, rootIndex),
            HCCL_E_INTERNAL);
        std::shared_ptr<RankInfoDetectScalable> scalableServer = iter.first->second;
        ScalableServerInfo serverInfo{};
        serverInfo.nRoot = nRoot;
        serverInfo.rootIndex = rootIndex;
        serverInfo.groupSize = groupSize;
        serverInfo.groupFirstRank = groupFirstRank;
        serverInfo.identifier = selfHandle.rootHandle.identifier;
        serverInfo.meshInfos.resize(nRoot);
        for (u32 i = 0; i < nRoot; ++i) {
            s32 sRet = memcpy_s(
                serverInfo.meshInfos[i].ip, sizeof(serverInfo.meshInfos[i].ip), rootHandles[i].rootHandle.ip,
                sizeof(rootHandles[i].rootHandle.ip));
            CHK_PRT_RET(
                sRet != EOK, HCCL_ERROR("[%s] memcpy mesh ip[%u] fail. errno[%d]", __func__, i, sRet), HCCL_E_MEMORY);
            serverInfo.meshInfos[i].meshPort = rootHandles[i].meshPort;
        }
        CHK_RET(scalableServer->AddMeshSocketWhiteList(serverInfo));
        CHK_RET(scalableServer->SetScalableServerInfo(serverInfo));
        HCCL_INFO("[%s] end.", __func__);
        return HCCL_SUCCESS;
    }

    // 多 root 初始化使用
    HcclResult DetectRankTableScalableImpl(
        u32 nRanks, u32 nRoot, const HcclRootInfo* rootInfoList, u32 rank, RankTableInfo& rankTable,
        RootInfoDetectBridge::DetectContext& detectContext)
    {
        // 参数校验
        CHK_PTR_NULL(rootInfoList);
        CHK_PRT_RET(
            nRanks == 0 || nRoot == 0 || nRoot > nRanks || rank >= nRanks,
            HCCL_ERROR("[%s] invalid scalable params, nRanks[%u], nRoot[%u], rank[%u].", __func__, nRanks, nRoot, rank),
            HCCL_E_PARA);

        const RootGroupStrategy* strategy = GetRootGroupStrategy();
        CHK_PRT_RET(
            strategy == nullptr, HCCL_ERROR("[%s] root group strategy is not registered, rankId[%u].", __func__, rank),
            HCCL_E_INTERNAL);
        const u32 rootIndex = strategy->rootIdFromRank(nRanks, nRoot, rank);
        CHK_PRT_RET(
            rootIndex >= nRoot,
            HCCL_ERROR("[%s] invalid rootIndex[%u], nRoot[%u], rankId[%u].", __func__, rootIndex, nRoot, rank),
            HCCL_E_INTERNAL);
        const bool isRootRank = strategy->rankHasRoot(nRanks, nRoot, rank);
        const u32 groupSize = strategy->nRankFromRoot(nRanks, nRoot, rootIndex);
        const u32 groupFirstRank = strategy->firstRankFromRoot(nRanks, nRoot, rootIndex);

        // 解析 nRoot 个 HcclScalableRootHandleV2（下标 = rootIndex）。
        std::vector<HcclScalableRootHandleV2> rootHandles(nRoot);
        for (u32 i = 0; i < nRoot; ++i) {
            s32 sRet = memcpy_s(
                &rootHandles[i], sizeof(HcclScalableRootHandleV2), rootInfoList[i].internal,
                sizeof(HcclScalableRootHandleV2));
            CHK_PRT_RET(
                sRet != EOK,
                HCCL_ERROR(
                    "[%s] memcpy scalable root info[%u] fail, rankId[%u], rootIndex[%u]. errno[%d]", __func__, i, rank,
                    rootIndex, sRet),
                HCCL_E_MEMORY);
            rootHandles[i].rootHandle.identifier[ROOTINFO_IDENTIFIER_MAX_LENGTH - 1] = '\0';
            // 强制截断保证后续 %s/std::string/IpAddress 按 C 字符串读取安全
            rootHandles[i].rootHandle.ip[IP_ADDRESS_BUFFER_LEN - 1] = '\0';
            CHK_PRT_RET(
                rootHandles[i].meshPort == 0 || rootHandles[i].meshPort == HCCL_INVALID_PORT,
                HCCL_ERROR(
                    "[%s] invalid scalable root info[%u]: meshPort[%u], ip[%s], "
                    "rootInfo should be generated by HcclGetRootInfoScalable, rankId[%u], rootIndex[%u].",
                    __func__, i, rootHandles[i].meshPort, rootHandles[i].rootHandle.ip, rank, rootIndex),
                HCCL_E_PARA);
            HCCL_RUN_INFO(
                "[%s] scalable root[%u]: ip[%s], serverPort[%u], meshPort[%u], identifier[%s].", __func__, i,
                rootHandles[i].rootHandle.ip, rootHandles[i].rootHandle.listenPort, rootHandles[i].meshPort,
                rootHandles[i].rootHandle.identifier);
        }

        s32 userDevId = 0;
        s32 devPhyId = 0;
        CHK_RET(GetCurrentDeviceIds(userDevId, devPhyId));

        // root rank：查注册表注入 mesh 信息，唤醒 server 线程（server 线程随后做组内收集+mesh 合并+广播）。
        if (isRootRank) {
            CHK_RET(InjectScalableServerInfo(nRoot, rootIndex, rank, groupSize, groupFirstRank, rootHandles));
        }

        // 所有 rank（含 root 自己）：组内 client 路径，连本组 root 的 agent 服务收全局表（复用单root）
        const HcclScalableRootHandleV2& groupRootHandle = rootHandles[rootIndex];
        HCCL_RUN_INFO(
            "[%s] nRanks[%u], nRoot[%u], rank[%u], rootIndex[%u], isRootRank[%d], groupSize[%u], groupFirstRank[%u],"
            "groupRoot: ip[%s] port[%u] identifier[%s], userDevId[%d], devPhyId[%d]",
            __func__, nRanks, nRoot, rank, rootIndex, isRootRank, groupSize, groupFirstRank,
            groupRootHandle.rootHandle.ip, groupRootHandle.rootHandle.listenPort, groupRootHandle.rootHandle.identifier,
            userDevId, devPhyId);

        std::shared_ptr<RankInfoDetect> rankInfoDetectAgent;
        EXCEPTION_CATCH((rankInfoDetectAgent = std::make_shared<RankInfoDetect>()), return HCCL_E_MEMORY);
        CHK_RET(rankInfoDetectAgent->SetupAgent(nRanks, rank, groupRootHandle.rootHandle));
        CHK_RET(rankInfoDetectAgent->WaitComplete(
            groupRootHandle.rootHandle.listenPort, RANKINFO_DETECT_SERVER_STATUS_IDLE));
        rankInfoDetectAgent->GetRankTable(rankTable);
        // bridge 不暴露 RankInfoDetect 类型；调用方持有类型擦除后的 shared_ptr，
        // 使 agent 及其探测资源在后续通信域初始化完成前保持有效。
        detectContext = rankInfoDetectAgent;
        HCCL_RUN_INFO("[%s] end.", __func__);
        return HCCL_SUCCESS;
    }

    // root 重选：用父通信域的 topParentIdentifier + generationIndex+1 构造 identifier，
    HcclResult GetRootInfoWithDerivationImpl(
        HcclRootInfo* rootInfo, const char* topParentId, uint32_t generationIndex,
        RootInfoDetectBridge::DetectContext& detectContext)
    {
        HCCL_RUN_INFO("Entry-HcclGetRootInfo V950, derivation-aware");
        CHK_PTR_NULL(topParentId);
        return GetRootInfoInternal(rootInfo, topParentId, generationIndex, detectContext);
    }

    const RootInfoDetectBridge ROOT_INFO_DETECT_BRIDGE
        = {GetRootInfoImpl, DetectRankTableImpl, GetRootInfoScalableImpl, DetectRankTableScalableImpl,
           GetRootInfoWithDerivationImpl};

    // hcomm 链接 hccl_v2，并在自身装载阶段将 provider 回调注册到 hccl_v2 的槽位。
    // 这样兼容入口保持原符号不变，同时不再直接依赖 RankInfoDetect 实现类。
    struct RootInfoDetectBridgeRegistrar {
        RootInfoDetectBridgeRegistrar()
        {
            // 静态构造函数无法向加载方返回错误；重复注册由 bridge 侧日志记录。
            (void)RegisterRootInfoDetectBridge(
                ROOT_INFO_DETECT_BRIDGE, ROOT_INFO_DETECT_BRIDGE_VERSION, sizeof(RootInfoDetectBridge));
        }
    };

    RootInfoDetectBridgeRegistrar g_rootInfoDetectBridgeRegistrar;

} // namespace
} // namespace Hccl
