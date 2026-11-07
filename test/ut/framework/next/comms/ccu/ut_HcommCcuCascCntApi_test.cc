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
 * UT: HcommCcuCascCnt API — 3 public interfaces + internal components
 *
 *   HcommCcuCascCntAlloc / HcommCcuCascCntGetMem / HcommCcuGetRmtMemToken
 *   CcuResPack::RefreshCascCntBlocks / AcquireCascCntBlock / GetCascCntBlock / CcuReleaseRes
 */

#include <iostream>
#include "hccl_api_base_test.h"

#define private public
#define protected public

#include "log.h"
#include "hcom_common.h"
#include "op_base.h"
#include "comm_engine/ccu/ccu_device_pub.h"
#include "ccu_device.h"
#include "ccu_res.h"
#include "ccu_types.h"
#include "comm_engine/ccu/ccu_device_res_pub.h"
#include "ccu_res_desc_mgr.h"
#include "ccu_res_specs.h"
#include "ccu_res_batch_allocator.h"
#include "ccu_instance_mgr.h"
#include "base_comm_node_accessors.h"
#include "ccu_res_pack.h"
#include "ccu_res_repo.h"
#include "hcomm_res_defs.h"

#include "mocks/ccu_device_mock_utils.h"

#undef protected
#undef private

using namespace hcomm;

namespace {
constexpr int32_t TEST_DEVICE_LOGIC_ID = 0;
int32_t g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
HcclResult g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;
// 校验输出句柄时的非 0 初值：若初值就为 0，"失败置 0"/"失败不写"的断言恒真，验证不到实际行为。
constexpr HcommCcuCascCntHandle STALE_CASC_CNT_HANDLE = 0x1234;

HcclResult MockHrtGetDeviceRefresh(int32_t* deviceLogicId)
{
    if (deviceLogicId == nullptr) {
        return HcclResult::HCCL_E_PTR;
    }
    if (g_deviceRefreshResult != HcclResult::HCCL_SUCCESS) {
        return g_deviceRefreshResult;
    }
    *deviceLogicId = g_runtimeDeviceLogicId;
    return HcclResult::HCCL_SUCCESS;
}

int g_releaseCallCount = 0;

// 块申请已下沉到设备层分配器，实例层用例只需按块布局构造仓库快照内容
CntXnBlock MakeMockCntXnBlock(uint8_t dieId)
{
    CntXnBlock cntXnBlock{};
    cntXnBlock.wishCntXns = {0, 1021};
    cntXnBlock.wishCntXnsMem = {reinterpret_cast<void*>(0x1000ULL + dieId * 0x100ULL), 1022 * sizeof(uint64_t)};
    cntXnBlock.totalCntXn = dieId * 4;
    cntXnBlock.expectedCntXn = dieId * 4 + 1;
    cntXnBlock.blockIdx = 0;
    cntXnBlock.dieId = dieId;
    return cntXnBlock;
}

CcuResult MockReleaseTrack(const int32_t, const uint8_t, const ResInfo&)
{
    g_releaseCallCount++;
    return CcuResult::CCU_SUCCESS;
}

// 初始化路径的桩：申请成功并回填非空句柄（Reset 以句柄为空直接返回成功，不回填就验证不到快照刷新）
constexpr uintptr_t MOCK_RES_HANDLE = 0x1;

CcuResult MockAllocByResDescsSuccess(int32_t, const CcuResDesc*[], uint32_t, CcuResHandle& resHandle)
{
    resHandle = reinterpret_cast<CcuResHandle>(MOCK_RES_HANDLE);
    return CcuResult::CCU_SUCCESS;
}

CcuResult MockAllocByInsTypeSuccess(int32_t, CcuInstanceType, CcuResHandle& resHandle)
{
    resHandle = reinterpret_cast<CcuResHandle>(MOCK_RES_HANDLE);
    return CcuResult::CCU_SUCCESS;
}

CcuResult MockCheckResourceFillCascCnt(const int32_t, const CcuResHandle, CcuResRepository& resRepo)
{
    resRepo.cascCntBlock[0].emplace_back(MakeMockCntXnBlock(0));
    return CcuResult::CCU_SUCCESS;
}

// insType 路径不会申请 cascCnt 资源，仓库中不会有块
CcuResult MockCheckResourceNoCascCnt(const int32_t, const CcuResHandle, CcuResRepository&)
{
    return CcuResult::CCU_SUCCESS;
}

HcclResult MockReleaseResHandleSuccess(const int32_t, const CcuResHandle) { return HcclResult::HCCL_SUCCESS; }
} // namespace

class HcommCcuCascCntApiTest : public BaseInit {
public:
    void SetUp() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        BaseInit::SetUp();
        ResDescMgr(TEST_DEVICE_LOGIC_ID).Deinit();
        g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
        g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;
        g_releaseCallCount = 0;

        MOCKER(hrtGetDeviceRefresh).stubs().with(mockcpp::any()).will(invoke(MockHrtGetDeviceRefresh));
        int32_t seededDeviceLogicId = INVALID_INT;
        ASSERT_EQ(HcclDeviceRefresh(seededDeviceLogicId), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(seededDeviceLogicId, TEST_DEVICE_LOGIC_ID);

        MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));

        // 块随资源句柄在设备层统一释放, 此处跟踪设备层释放接口以校验实例层不再重复释放
        MOCKER(CcuReleaseCntXn)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(invoke(MockReleaseTrack));
    }

    void TearDown() override
    {
        g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
        g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;
        int32_t restoredDeviceLogicId = INVALID_INT;
        EXPECT_EQ(HcclDeviceRefresh(restoredDeviceLogicId), HcclResult::HCCL_SUCCESS);

        ResDescMgr(TEST_DEVICE_LOGIC_ID).Deinit();
        BaseInit::TearDown();
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }

protected:
    static CcuResDescMgr& ResDescMgr(int32_t deviceLogicId) { return GetCcuInstanceMgr(deviceLogicId).GetResDescMgr(); }

    static std::unique_ptr<CcuResPack> MakeResPackWithBlocks(uint8_t dieId, uint32_t blockCount)
    {
        auto resPack = std::make_unique<CcuResPack>();
        resPack->userDevId_ = TEST_DEVICE_LOGIC_ID;
        for (uint32_t i = 0; i < blockCount; i++) {
            resPack->cascCntBlocks_[dieId].emplace_back(MakeMockCntXnBlock(dieId), false);
        }
        return resPack;
    }
};

// ═══════════════════════════════════════════════════════════
// 1. HcommCcuCascCntAlloc — Parameter Validation
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuCascCntAlloc_When_HandleNull_Expect_CCU_E_PTR)
{
    CcuResult ret = HcommCcuCascCntAlloc(1, 0, nullptr);
    EXPECT_EQ(ret, CCU_E_PTR);
}

// 失败路径统一校验 handle 置 0（见 ccu_res.h 对 HcommCcuCascCntAlloc 的约定）：
// 初值刻意设为非 0，否则 handle 本就为 0 时断言恒真，无法验证清零行为。
TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuCascCntAlloc_When_CcuInsHandleZero_Expect_CCU_E_PARA)
{
    HcommCcuCascCntHandle handle = STALE_CASC_CNT_HANDLE;
    CcuResult ret = HcommCcuCascCntAlloc(0, 0, &handle);
    EXPECT_EQ(ret, CCU_E_PARA);
    EXPECT_EQ(handle, 0U);
}

TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuCascCntAlloc_When_DieIdInvalid_Expect_CCU_E_PARA)
{
    HcommCcuCascCntHandle handle = STALE_CASC_CNT_HANDLE;
    CcuResult ret = HcommCcuCascCntAlloc(1, CCU_MAX_IODIE_NUM, &handle);
    EXPECT_EQ(ret, CCU_E_PARA);
    EXPECT_EQ(handle, 0U);
}

TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuCascCntAlloc_When_CcuInsHandleNotFound_Expect_CCU_E_NOT_FOUND)
{
    HcommCcuCascCntHandle handle = STALE_CASC_CNT_HANDLE;
    CcuResult ret = HcommCcuCascCntAlloc(0xDEADBEEF, 0, &handle);
    EXPECT_EQ(ret, CCU_E_NOT_FOUND);
    EXPECT_EQ(handle, 0U);
}

TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuCascCntAlloc_When_DeviceRefreshFails_Expect_Error)
{
    g_deviceRefreshResult = HcclResult::HCCL_E_INTERNAL;
    HcommCcuCascCntHandle handle = STALE_CASC_CNT_HANDLE;
    CcuResult ret = HcommCcuCascCntAlloc(1, 0, &handle);
    EXPECT_NE(ret, CCU_SUCCESS);
    EXPECT_EQ(handle, 0U);
}

// ═══════════════════════════════════════════════════════════
// 2. HcommCcuCascCntGetMem — Parameter Validation
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuCascCntGetMem_When_HandleZero_Expect_CCU_E_PARA)
{
    CommMem commMem{};
    CcuResult ret = HcommCcuCascCntGetMem(0, &commMem);
    EXPECT_EQ(ret, CCU_E_PARA);
}

TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuCascCntGetMem_When_CommMemNull_Expect_CCU_E_PTR)
{
    CcuResult ret = HcommCcuCascCntGetMem(1, nullptr);
    EXPECT_EQ(ret, CCU_E_PTR);
}

// ═══════════════════════════════════════════════════════════
// 3. HcommCcuGetRmtMemToken — Parameter Validation
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuGetRmtMemToken_When_ChannelHandleZero_Expect_CCU_E_PARA)
{
    uint64_t tokenInfo = 0;
    CcuResult ret = HcommCcuGetRmtMemToken(0, 0x1000, &tokenInfo);
    EXPECT_EQ(ret, CCU_E_PARA);
}

TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuGetRmtMemToken_When_TokenInfoNull_Expect_CCU_E_PTR)
{
    CcuResult ret = HcommCcuGetRmtMemToken(1, 0x1000, nullptr);
    EXPECT_EQ(ret, CCU_E_PTR);
}

// ═══════════════════════════════════════════════════════════
// 4. CcuResPack — RefreshCascCntBlocks / CcuReleaseRes
//    块申请已下沉到设备层分配器，本类只维护仓库快照
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuCascCntApiTest, Ut_RefreshCascCntBlocks_When_RepoHasBlocks_Expect_SnapshotFilledAndIdle)
{
    CcuResPack resPack;
    resPack.userDevId_ = TEST_DEVICE_LOGIC_ID;
    for (uint32_t i = 0; i < 2; i++) {
        resPack.resRepo_.cascCntBlock[0].emplace_back(MakeMockCntXnBlock(0));
    }

    resPack.RefreshCascCntBlocks();
    ASSERT_EQ(resPack.cascCntBlocks_[0].size(), 2U);
    // 刷新出的快照初始均为空闲状态
    EXPECT_FALSE(resPack.cascCntBlocks_[0][0].second);
    EXPECT_FALSE(resPack.cascCntBlocks_[0][1].second);
}

TEST_F(HcommCcuCascCntApiTest, Ut_RefreshCascCntBlocks_When_RepoEmpty_Expect_SnapshotCleared)
{
    auto resPack = MakeResPackWithBlocks(0, 2);
    ASSERT_EQ(resPack->cascCntBlocks_[0].size(), 2U);

    resPack->RefreshCascCntBlocks();
    EXPECT_EQ(resPack->cascCntBlocks_[0].size(), 0U);
}

// insType 路径不申请级联计数器块，初始化只会清空上一次初始化可能残留的快照；
// 若残留旧快照，后续块申请会取到不属于本次资源的块
TEST_F(HcommCcuCascCntApiTest, Ut_InitByInsType_When_StaleSnapshotExists_Expect_SnapshotCleared)
{
    MOCKER(CcuAllocResHandleByInsType).stubs().will(invoke(MockAllocByInsTypeSuccess));
    MOCKER(CcuCheckResource).stubs().will(invoke(MockCheckResourceNoCascCnt));
    MOCKER(CcuReleaseResHandle).stubs().will(invoke(MockReleaseResHandleSuccess));

    auto resPack = MakeResPackWithBlocks(0, 2);
    resPack->userDevId_ = TEST_DEVICE_LOGIC_ID;
    ASSERT_EQ(resPack->cascCntBlocks_[0].size(), 2U);

    ASSERT_EQ(resPack->InitByInsType(CcuInstanceType::CCU_MS), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resPack->cascCntBlocks_[0].size(), 0U);
    EXPECT_EQ(resPack->AcquireCascCntBlock(0, 1), CcuResult::CCU_E_UNAVAIL);
}

TEST_F(HcommCcuCascCntApiTest, Ut_InitByResDescs_When_AllocSuccess_Expect_SnapshotRefreshed)
{
    MOCKER(CcuAllocResHandleByResDescs).stubs().will(invoke(MockAllocByResDescsSuccess));
    MOCKER(CcuCheckResource).stubs().will(invoke(MockCheckResourceFillCascCnt));
    MOCKER(CcuReleaseResHandle).stubs().will(invoke(MockReleaseResHandleSuccess));

    CcuResDesc desc{};
    desc.dieId = 0;
    const CcuResDesc* descs[] = {&desc};
    CcuResPack resPack;
    ASSERT_EQ(resPack.InitByResDescs(descs, 1), CcuResult::CCU_SUCCESS);
    ASSERT_EQ(resPack.cascCntBlocks_[0].size(), 1U);
    EXPECT_FALSE(resPack.cascCntBlocks_[0][0].second);
    EXPECT_EQ(resPack.cascCntBlocks_[0][0].first.dieId, 0);
    EXPECT_EQ(resPack.AcquireCascCntBlock(0, 1), CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuCascCntApiTest, Ut_CcuReleaseRes_When_HasBlocks_Expect_OnlyClearSnapshot)
{
    auto resPack = MakeResPackWithBlocks(1, 2);
    ASSERT_EQ(resPack->cascCntBlocks_[1].size(), 2U);

    resPack->CcuReleaseRes();
    // 块已随资源句柄在设备层统一释放，实例层不得再逐块释放
    EXPECT_EQ(g_releaseCallCount, 0);
    EXPECT_EQ(resPack->cascCntBlocks_[1].size(), 0U);
}

// ═══════════════════════════════════════════════════════════
// 5. CcuResPack — AcquireCascCntBlock
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuCascCntApiTest, Ut_AcquireCascCntBlock_When_BlocksAvailable_Expect_SuccessAndBlock)
{
    auto resPack = MakeResPackWithBlocks(0, 2);
    // handle 由上层 CcuInstanceMgr 全局唯一分配后按值传入, resPack 仅按其登记, 不再自增
    HcommCcuCascCntHandle handle = 1;
    EXPECT_EQ(resPack->AcquireCascCntBlock(0, handle), CcuResult::CCU_SUCCESS);
    // 传入句柄后应能按值取回块内容（块已 emplace 进 handleMap_）
    CntXnBlock block{};
    EXPECT_EQ(resPack->GetCascCntBlock(handle, block), CcuResult::CCU_SUCCESS);
    EXPECT_NE(block.wishCntXnsMem.first, nullptr);
    EXPECT_GT(block.wishCntXnsMem.second, 0U);
}

TEST_F(HcommCcuCascCntApiTest, Ut_AcquireCascCntBlock_When_AllBlocksUsed_Expect_CCU_E_UNAVAIL)
{
    auto resPack = MakeResPackWithBlocks(0, 1);
    // 取走唯一的块
    ASSERT_EQ(resPack->AcquireCascCntBlock(0, 1), CcuResult::CCU_SUCCESS);
    // 已无空闲块：失败返回 CCU_E_UNAVAIL（即便传入新的合法句柄）
    EXPECT_EQ(resPack->AcquireCascCntBlock(0, 2), CcuResult::CCU_E_UNAVAIL);
}

TEST_F(HcommCcuCascCntApiTest, Ut_AcquireCascCntBlock_When_NoBlocks_Expect_CCU_E_UNAVAIL)
{
    CcuResPack resPack;
    resPack.userDevId_ = TEST_DEVICE_LOGIC_ID;
    EXPECT_EQ(resPack.AcquireCascCntBlock(0, 1), CcuResult::CCU_E_UNAVAIL);
}

TEST_F(HcommCcuCascCntApiTest, Ut_AcquireCascCntBlock_When_DieIdOutOfRange_Expect_CCU_E_PARA)
{
    auto resPack = MakeResPackWithBlocks(0, 1);
    EXPECT_EQ(resPack->AcquireCascCntBlock(CCU_MAX_IODIE_NUM, 1), CcuResult::CCU_E_PARA);
}

// handle 恒由上层保证非 0(全局唯一从 1 起), resPack 对 0 句柄做防御性拒绝
TEST_F(HcommCcuCascCntApiTest, Ut_AcquireCascCntBlock_When_HandleZero_Expect_CCU_E_PARA)
{
    auto resPack = MakeResPackWithBlocks(0, 1);
    EXPECT_EQ(resPack->AcquireCascCntBlock(0, 0), CcuResult::CCU_E_PARA);
}

// ═══════════════════════════════════════════════════════════
// 6. CcuResPack — GetCascCntBlock
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuCascCntApiTest, Ut_GetCascCntBlock_When_ValidHandle_Expect_ReturnBlock)
{
    auto resPack = MakeResPackWithBlocks(0, 1);
    HcommCcuCascCntHandle handle = 1;
    ASSERT_EQ(resPack->AcquireCascCntBlock(0, handle), CcuResult::CCU_SUCCESS);

    CntXnBlock block{};
    EXPECT_EQ(resPack->GetCascCntBlock(handle, block), CcuResult::CCU_SUCCESS);
    EXPECT_NE(block.wishCntXnsMem.first, nullptr);
    EXPECT_GT(block.wishCntXnsMem.second, 0U);
}

TEST_F(HcommCcuCascCntApiTest, Ut_GetCascCntBlock_When_InvalidHandle_Expect_CCU_E_NOT_FOUND)
{
    auto resPack = MakeResPackWithBlocks(0, 1);
    CntXnBlock block{};
    EXPECT_EQ(resPack->GetCascCntBlock(9999, block), CcuResult::CCU_E_NOT_FOUND);
}

TEST_F(HcommCcuCascCntApiTest, Ut_GetCascCntBlock_When_HandleZero_Expect_CCU_E_NOT_FOUND)
{
    auto resPack = MakeResPackWithBlocks(0, 1);
    CntXnBlock block{};
    // handle 由上层全局唯一分配（从 1 起），resPack 不再自增，0 恒为无效
    EXPECT_EQ(resPack->GetCascCntBlock(0, block), CcuResult::CCU_E_NOT_FOUND);
}

// ═══════════════════════════════════════════════════════════
// 7. CcuResPack — 多个互异句柄各自独立登记
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuCascCntApiTest, Ut_AcquireCascCntBlock_When_MultipleDistinctHandles_Expect_AllRegistered)
{
    auto resPack = MakeResPackWithBlocks(0, 3);
    // 句柄由上层(Mgr)全局唯一分配后按值传入, resPack 按值登记;
    // 三个互异句柄应各自独立存入 handleMap_ 且可分别取回
    ASSERT_EQ(resPack->AcquireCascCntBlock(0, 1), CcuResult::CCU_SUCCESS);
    ASSERT_EQ(resPack->AcquireCascCntBlock(0, 2), CcuResult::CCU_SUCCESS);
    ASSERT_EQ(resPack->AcquireCascCntBlock(0, 3), CcuResult::CCU_SUCCESS);

    CntXnBlock block{};
    EXPECT_EQ(resPack->GetCascCntBlock(1, block), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resPack->GetCascCntBlock(2, block), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resPack->GetCascCntBlock(3, block), CcuResult::CCU_SUCCESS);
}
