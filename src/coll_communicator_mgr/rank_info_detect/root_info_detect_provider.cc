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

#include "acl/acl_rt.h"
#include "hccl_common_v2.h"
#include "log.h"
#include "orion_adapter_rts.h"
#include "rank_info_detect.h"
#include "securec.h"

namespace Hccl {
namespace {

    HcclResult GetCurrentDeviceIds(s32& userDevId, s32& devPhyId)
    {
        userDevId = HrtGetDevice();
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
        RootInfoDetectBridge::DetectContext& detectContext)
    {
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

        CHK_RET(rankInfoDetectAgent->SetupAgent(nRanks, rank, rootHandle));
        CHK_RET(rankInfoDetectAgent->WaitComplete(rootHandle.listenPort, RANKINFO_DETECT_SERVER_STATUS_IDLE));

        rankInfoDetectAgent->GetRankTable(rankTable);
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
        = {GetRootInfoImpl, DetectRankTableImpl, GetRootInfoWithDerivationImpl};

    // hcomm 链接 hccl_v2，并在自身装载阶段将 provider 回调注册到 hccl_v2 的槽位。
    // 这样兼容入口保持原符号不变，同时不再直接依赖 RankInfoDetect 实现类。
    struct RootInfoDetectBridgeRegistrar {
        RootInfoDetectBridgeRegistrar()
        {
            // 静态构造函数无法向加载方返回错误；重复注册由 bridge 侧日志记录。
            (void)RegisterRootInfoDetectBridge(ROOT_INFO_DETECT_BRIDGE);
        }
    };

    RootInfoDetectBridgeRegistrar g_rootInfoDetectBridgeRegistrar;

} // namespace
} // namespace Hccl
