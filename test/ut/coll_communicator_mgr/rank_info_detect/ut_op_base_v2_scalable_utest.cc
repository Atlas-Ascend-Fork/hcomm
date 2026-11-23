/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include "securec.h"
#include "types.h"
#include "hccl_comm.h"
#include "op_base_v2.h"
#include "root_info_detect_bridge.h"
#include "root_handle_v2.h"

using namespace Hccl;

namespace {
// detectRankTableScalable 委派调用计数：成功路径用例据此断言 provider 回调确实被调用一次。
u32 g_detectRankTableScalableCalls = 0;

// 构造 nRoot 个合法 HcclScalableRootHandle 塞入 HcclRootInfo.internal（与 Get 产出格式一致）。
void FillScalableRootInfos(HcclRootInfo* rootInfos, u32 nRoot)
{
    for (u32 i = 0; i < nRoot; ++i) {
        (void)memset(&rootInfos[i], 0, sizeof(HcclRootInfo));
        HcclScalableRootHandleV2 handle{};
        (void)snprintf(handle.rootHandle.identifier, ROOTINFO_IDENTIFIER_MAX_LENGTH, "scalable_root_%u", i);
        handle.rootHandle.listenPort = 5000 + i;
        handle.meshPort = 6000 + i;
        (void)memcpy_s(rootInfos[i].internal, HCCL_ROOT_INFO_BYTES, &handle, sizeof(HcclScalableRootHandleV2));
    }
}
// 使用真实上下文，避免 mockcpp::invoke 按值保存含 mutex 的引用返回对象。
class ScopedEmptyCommInfo {
public:
    explicit ScopedEmptyCommInfo(HcclCommInfoV2& info) : info_(info), wasUsed_(info.isUsed)
    {
        std::lock_guard<std::mutex> lock(info_.groupParamsLock);
        savedGroups_.swap(info_.hcclGroupMap);
        info_.isUsed = true; // 固定使用本设备上下文，不回退到 backup context。
    }

    ~ScopedEmptyCommInfo()
    {
        std::lock_guard<std::mutex> lock(info_.groupParamsLock);
        savedGroups_.swap(info_.hcclGroupMap);
        info_.isUsed = wasUsed_;
    }

private:
    HcclCommInfoV2& info_;
    bool wasUsed_;
    std::map<std::string, HcclGroupParamsV2> savedGroups_;
};
} // namespace

class ScalableV2EntryTest : public testing::Test {
protected:
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(ScalableV2EntryTest, Ut_HcclGetRootInfoScalableV2_When_BridgeScalableSlotNull_Return_EInternal)
{
    MOCKER(GetRootInfoDetectBridgeScalable)
        .expects(once())
        .will(returnValue(static_cast<const RootInfoDetectBridge*>(nullptr)));
    HcclRootInfo rootInfo{};
    EXPECT_EQ(HcclGetRootInfoScalableV2(&rootInfo), HCCL_E_INTERNAL);
}

TEST_F(ScalableV2EntryTest, Ut_HcclGetRootInfoScalableV2_When_RootInfoNullptr_Return_EPtr)
{
    // The entry delegates pointer validation to the registered provider.
    RootInfoDetectBridge bridge{};
    bridge.getRootInfoScalable = [](HcclRootInfo* rootInfo) -> HcclResult {
        EXPECT_EQ(rootInfo, nullptr);
        return HCCL_E_PTR;
    };
    MOCKER(GetRootInfoDetectBridgeScalable)
        .expects(once())
        .will(returnValue(static_cast<const RootInfoDetectBridge*>(&bridge)));
    EXPECT_EQ(HcclGetRootInfoScalableV2(nullptr), HCCL_E_PTR);
}

// Ut_HcclCommInitRootInfoScalableV2_When_InvalidParams_Return_EPara：
// 参数校验各非法组合返回 HCCL_E_PARA（校验层先于 bridge 调用，无需注册 bridge）
TEST_F(ScalableV2EntryTest, Ut_HcclCommInitRootInfoScalableV2_When_InvalidParams_Return_EPara)
{
    HcclRootInfo rootInfoList[4];
    FillScalableRootInfos(rootInfoList, 4);
    HcclCommConfig config{};
    (void)memset(&config, 0, sizeof(config));
    HcclComm comm = nullptr;

    // nRanks=0
    EXPECT_EQ(HcclCommInitRootInfoScalableV2(0, 2, rootInfoList, 0, 0, &config, &comm), HCCL_E_PARA);
    // nRoot=0
    EXPECT_EQ(HcclCommInitRootInfoScalableV2(8, 0, rootInfoList, 0, 0, &config, &comm), HCCL_E_PARA);
    // nRoot > nRanks
    EXPECT_EQ(HcclCommInitRootInfoScalableV2(2, 4, rootInfoList, 0, 0, &config, &comm), HCCL_E_PARA);
    // rank >= nRanks
    EXPECT_EQ(HcclCommInitRootInfoScalableV2(8, 4, rootInfoList, 8, 0, &config, &comm), HCCL_E_PARA);
    // nExtRoot != 0（预留参数当前仅支持 0）
    EXPECT_EQ(HcclCommInitRootInfoScalableV2(8, 4, rootInfoList, 0, 1, &config, &comm), HCCL_E_PARA);
    // rootInfoList = nullptr
    EXPECT_EQ(HcclCommInitRootInfoScalableV2(8, 4, nullptr, 0, 0, &config, &comm), HCCL_E_PTR);
    // config = nullptr
    EXPECT_EQ(HcclCommInitRootInfoScalableV2(8, 4, rootInfoList, 0, 0, nullptr, &comm), HCCL_E_PTR);
    // comm = nullptr
    EXPECT_EQ(HcclCommInitRootInfoScalableV2(8, 4, rootInfoList, 0, 0, &config, nullptr), HCCL_E_PTR);

    // 非法入参未产生副作用
    EXPECT_EQ(comm, nullptr);
}

// Ut_HcclCommInitRootInfoScalableV2_When_ValidParamsButNoBridge_Return_EInternal：
// 参数合法但 bridge 未注册 scalable 槽时返回 HCCL_E_INTERNAL（不静默成功）
TEST_F(ScalableV2EntryTest, Ut_HcclCommInitRootInfoScalableV2_When_ValidParamsButNoBridge_Return_EInternal)
{
    HcclRootInfo rootInfoList[4];
    FillScalableRootInfos(rootInfoList, 4);
    HcclCommConfig config{};
    (void)memset(&config, 0, sizeof(config));
    HcclComm comm = nullptr;
    HcclCommConfigInit(&config);
    config.hcclBufferSize = 200;
    MOCKER(CallSingletons).stubs().will(returnValue(HCCL_SUCCESS));
    int32_t deviceId = 0;
    MOCKER(aclrtGetDevice).stubs().with(outBoundP(&deviceId, sizeof(deviceId))).will(returnValue(ACL_SUCCESS));
    ScopedEmptyCommInfo commInfoGuard(CommManager::GetInstance(deviceId).GetCommInfoV2());
    MOCKER(GetRootInfoDetectBridgeScalable)
        .expects(once())
        .will(returnValue(static_cast<const RootInfoDetectBridge*>(nullptr)));
    EXPECT_EQ(HcclCommInitRootInfoScalableV2(8, 4, rootInfoList, 0, 0, &config, &comm), HCCL_E_INTERNAL);
    EXPECT_EQ(comm, nullptr);
}

// op_base_v2.cc 内部接口（未在头文件中声明）：声明用于成功路径打桩。
HcclResult CreateCommConfigRootInfo(
    uint32_t rank, const HcclCommConfig* config, const std::string& identifier, const RankTableInfo& ranktable,
    HcclComm* comm);

// Ut_HcclGetRootInfoScalableV2_When_BridgeSlotReady_Return_Success：
// scalable 槽已注册时 Get 入口成功委派 provider，并透传 provider 写入的 handle。
TEST_F(ScalableV2EntryTest, Ut_HcclGetRootInfoScalableV2_When_BridgeSlotReady_Return_Success)
{
    RootInfoDetectBridge bridge{};
    bridge.getRootInfoScalable = [](HcclRootInfo* rootInfo) -> HcclResult {
        if (rootInfo == nullptr) {
            return HCCL_E_PTR;
        }
        HcclScalableRootHandleV2 handle{};
        (void)snprintf(handle.rootHandle.identifier, ROOTINFO_IDENTIFIER_MAX_LENGTH, "scalable_root_0");
        handle.rootHandle.listenPort = 6000;
        handle.meshPort = 6001;
        (void)memcpy_s(rootInfo->internal, HCCL_ROOT_INFO_BYTES, &handle, sizeof(handle));
        return HCCL_SUCCESS;
    };
    MOCKER(GetRootInfoDetectBridgeScalable)
        .expects(once())
        .will(returnValue(static_cast<const RootInfoDetectBridge*>(&bridge)));

    HcclRootInfo rootInfo{};
    EXPECT_EQ(HcclGetRootInfoScalableV2(&rootInfo), HCCL_SUCCESS);
    HcclScalableRootHandleV2 parsed{};
    EXPECT_EQ(memcpy_s(&parsed, sizeof(parsed), rootInfo.internal, sizeof(parsed)), EOK);
    EXPECT_EQ(parsed.rootHandle.listenPort, 6000U);
    EXPECT_EQ(parsed.meshPort, 6001U);
    EXPECT_STREQ(parsed.rootHandle.identifier, "scalable_root_0");
}

// Ut_HcclCommInitRootInfoScalableV2_When_BufferSizeZero_Return_EPara：
// hcclBufferSize=0 在查重通过后被拒绝，返回 HCCL_E_PARA。
TEST_F(ScalableV2EntryTest, Ut_HcclCommInitRootInfoScalableV2_When_BufferSizeZero_Return_EPara)
{
    HcclRootInfo rootInfoList[4];
    FillScalableRootInfos(rootInfoList, 4);
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    config.hcclBufferSize = 0;
    HcclComm comm = nullptr;
    MOCKER(CallSingletons).stubs().will(returnValue(HCCL_SUCCESS));
    int32_t deviceId = 0;
    MOCKER(aclrtGetDevice).stubs().with(outBoundP(&deviceId, sizeof(deviceId))).will(returnValue(ACL_SUCCESS));
    ScopedEmptyCommInfo commInfoGuard(CommManager::GetInstance(deviceId).GetCommInfoV2());

    EXPECT_EQ(HcclCommInitRootInfoScalableV2(8, 4, rootInfoList, 0, 0, &config, &comm), HCCL_E_PARA);
    EXPECT_EQ(comm, nullptr);
}

// Ut_HcclCommInitRootInfoScalableV2_When_IdentifierDuplicated_Return_EPara：
// 通信域名（缺省取 root[0] identifier）已存在时返回 HCCL_E_PARA。
TEST_F(ScalableV2EntryTest, Ut_HcclCommInitRootInfoScalableV2_When_IdentifierDuplicated_Return_EPara)
{
    HcclRootInfo rootInfoList[4];
    FillScalableRootInfos(rootInfoList, 4);
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    config.hcclBufferSize = 200;
    HcclComm comm = nullptr;
    MOCKER(CallSingletons).stubs().will(returnValue(HCCL_SUCCESS));
    int32_t deviceId = 0;
    MOCKER(aclrtGetDevice).stubs().with(outBoundP(&deviceId, sizeof(deviceId))).will(returnValue(ACL_SUCCESS));
    ScopedEmptyCommInfo commInfoGuard(CommManager::GetInstance(deviceId).GetCommInfoV2());
    {
        auto& commInfo = CommManager::GetInstance(deviceId).GetCommInfoV2();
        std::lock_guard<std::mutex> lock(commInfo.groupParamsLock);
        commInfo.hcclGroupMap["scalable_root_0"] = HcclGroupParamsV2{};
    }

    EXPECT_EQ(HcclCommInitRootInfoScalableV2(8, 4, rootInfoList, 0, 0, &config, &comm), HCCL_E_PARA);
    EXPECT_EQ(comm, nullptr);
}

// Ut_HcclCommInitRootInfoScalableV2_When_DetectSucceeds_Return_Success：
// scalable 槽已注册且参数合法时成功委派 detectRankTableScalable，并进入通信域创建。
TEST_F(ScalableV2EntryTest, Ut_HcclCommInitRootInfoScalableV2_When_DetectSucceeds_Return_Success)
{
    HcclRootInfo rootInfoList[4];
    FillScalableRootInfos(rootInfoList, 4);
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    config.hcclBufferSize = 200;
    HcclComm comm = nullptr;
    MOCKER(CallSingletons).stubs().will(returnValue(HCCL_SUCCESS));
    int32_t deviceId = 0;
    MOCKER(aclrtGetDevice).stubs().with(outBoundP(&deviceId, sizeof(deviceId))).will(returnValue(ACL_SUCCESS));
    ScopedEmptyCommInfo commInfoGuard(CommManager::GetInstance(deviceId).GetCommInfoV2());

    RootInfoDetectBridge bridge{};
    g_detectRankTableScalableCalls = 0;
    bridge.detectRankTableScalable = [](u32, u32, const HcclRootInfo*, u32, RankTableInfo& rankTable,
                                        RootInfoDetectBridge::DetectContext&) -> HcclResult {
        ++g_detectRankTableScalableCalls;
        rankTable = RankTableInfo{};
        return HCCL_SUCCESS;
    };
    MOCKER(GetRootInfoDetectBridgeScalable)
        .expects(once())
        .will(returnValue(static_cast<const RootInfoDetectBridge*>(&bridge)));
    MOCKER(CreateCommConfigRootInfo).expects(once()).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(HcclCommInitRootInfoScalableV2(8, 4, rootInfoList, 0, 0, &config, &comm), HCCL_SUCCESS);
    EXPECT_EQ(g_detectRankTableScalableCalls, 1U);
}

// Ut_HcclCommInitRootInfoScalableV2_When_DetectFailed_Return_ProviderError：
// detectRankTableScalable 返回错误时入口应原样透传错误码，且不进入通信域创建。
TEST_F(ScalableV2EntryTest, Ut_HcclCommInitRootInfoScalableV2_When_DetectFailed_Return_ProviderError)
{
    HcclRootInfo rootInfoList[4];
    FillScalableRootInfos(rootInfoList, 4);
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    config.hcclBufferSize = 200;
    HcclComm comm = nullptr;
    MOCKER(CallSingletons).stubs().will(returnValue(HCCL_SUCCESS));
    int32_t deviceId = 0;
    MOCKER(aclrtGetDevice).stubs().with(outBoundP(&deviceId, sizeof(deviceId))).will(returnValue(ACL_SUCCESS));
    ScopedEmptyCommInfo commInfoGuard(CommManager::GetInstance(deviceId).GetCommInfoV2());

    RootInfoDetectBridge bridge{};
    g_detectRankTableScalableCalls = 0;
    bridge.detectRankTableScalable
        = [](u32, u32, const HcclRootInfo*, u32, RankTableInfo&, RootInfoDetectBridge::DetectContext&) -> HcclResult {
        ++g_detectRankTableScalableCalls;
        return HCCL_E_INTERNAL;
    };
    MOCKER(GetRootInfoDetectBridgeScalable)
        .expects(once())
        .will(returnValue(static_cast<const RootInfoDetectBridge*>(&bridge)));
    MOCKER(CreateCommConfigRootInfo).expects(never());

    EXPECT_EQ(HcclCommInitRootInfoScalableV2(8, 4, rootInfoList, 0, 0, &config, &comm), HCCL_E_INTERNAL);
    EXPECT_EQ(g_detectRankTableScalableCalls, 1U);
    EXPECT_EQ(comm, nullptr);
}
