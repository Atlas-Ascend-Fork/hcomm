/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#include <cstdint>

#include "acl/acl_rt.h"
#include "ccu_dev_mgr_imp.h"
#include "ccu_instance.h"
#include "hcomm_adapter_hccp.h"
#include "hcomm_res_ccu_entity.h"

using namespace hcomm;

namespace {

// 单条 CCU 微码指令宽度 32B；0x8000000 * 32B = 4GiB，超过 uint32 字节上限
constexpr uint32_t OVERFLOW_INST_NUM = 0x8000000U;
// startInstId 初始哨兵值，用于验证失败路径会被置 0
constexpr uint32_t START_INST_ID_SENTINEL = 0xFFFFU;

void MockAclrtMallocSuccess(void** devMem)
{
    MOCKER(aclrtMalloc)
        .stubs()
        .with(outBoundP(devMem, sizeof(void*)), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
}

void MockAclrtMemcpy(aclError ret)
{
    MOCKER(aclrtMemcpy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ret));
}

void MockCcuCustomChannel(HcclResult ret)
{
    MOCKER(HccpRaTlvCcuCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ret));
}

} // namespace

class HcommResCcuEntityTest : public testing::Test {
protected:
    void SetUp() override { GlobalMockObject::reset(); }

    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
};

TEST_F(HcommResCcuEntityTest, Ut_HcommCcuSubmitInsts_When_DataNull_Expect_HCCL_E_PARA)
{
    EXPECT_EQ(HcommCcuSubmitInsts(0, 0, 0, nullptr, 1), static_cast<HcommResult>(HCCL_E_PARA));
}

TEST_F(HcommResCcuEntityTest, Ut_HcommCcuSubmitInsts_When_InstNumZero_Expect_HCCL_E_PARA)
{
    uint8_t insts[32] = {};
    EXPECT_EQ(HcommCcuSubmitInsts(0, 0, 0, insts, 0), static_cast<HcommResult>(HCCL_E_PARA));
}

TEST_F(HcommResCcuEntityTest, Ut_HcommCcuSubmitInsts_When_DieIdInvalid_Expect_HCCL_E_PARA)
{
    uint8_t insts[32] = {};
    EXPECT_EQ(HcommCcuSubmitInsts(0, HCOMM_CCU_MAX_DIE_NUM, 0, insts, 1), static_cast<HcommResult>(HCCL_E_PARA));
}

TEST_F(HcommResCcuEntityTest, Ut_HcommCcuSubmitInsts_When_ByteSizeExceedsLimit_Expect_HCCL_E_PARA)
{
    uint8_t insts[32] = {};
    EXPECT_EQ(HcommCcuSubmitInsts(0, 0, 0, insts, OVERFLOW_INST_NUM), static_cast<HcommResult>(HCCL_E_PARA));
}

// 验证 aclrtMalloc 返回成功但 devMem 为空时按内部错误处理，不进入 H2D 拷贝。
TEST_F(HcommResCcuEntityTest, Ut_HcommCcuSubmitInsts_When_MallocOkButDevMemNull_Expect_HCCL_E_INTERNAL)
{
    uint8_t insts[64] = {};
    void* nullDevMem = nullptr;
    MockAclrtMallocSuccess(&nullDevMem);
    MOCKER(aclrtFree).expects(never());

    EXPECT_EQ(HcommCcuSubmitInsts(0, 0, 0, insts, 2), static_cast<HcommResult>(HCCL_E_INTERNAL));
}

TEST_F(HcommResCcuEntityTest, Ut_HcommCcuSubmitInsts_When_MallocFailed_Expect_ErrorPropagated)
{
    uint8_t insts[64] = {};
    MOCKER(aclrtMalloc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_ERROR_INVALID_PARAM));
    MOCKER(aclrtFree).expects(never());

    EXPECT_EQ(HcommCcuSubmitInsts(0, 0, 0, insts, 2), static_cast<HcommResult>(ACL_ERROR_INVALID_PARAM));
}

TEST_F(HcommResCcuEntityTest, Ut_HcommCcuSubmitInsts_When_MemcpyFailed_Expect_ErrorPropagatedAndDevMemFreed)
{
    uint8_t insts[64] = {};
    void* fakeDevMem = reinterpret_cast<void*>(0x10);
    MockAclrtMallocSuccess(&fakeDevMem);
    MockAclrtMemcpy(ACL_ERROR_INVALID_PARAM);
    MOCKER(aclrtFree).expects(once()).with(mockcpp::any()).will(returnValue(ACL_SUCCESS));

    EXPECT_EQ(HcommCcuSubmitInsts(0, 0, 0, insts, 2), static_cast<HcommResult>(ACL_ERROR_INVALID_PARAM));
}

TEST_F(HcommResCcuEntityTest, Ut_HcommCcuSubmitInsts_When_DriverFailed_Expect_ErrorPropagatedAndDevMemFreed)
{
    uint8_t insts[64] = {};
    void* fakeDevMem = reinterpret_cast<void*>(0x10);
    MockAclrtMallocSuccess(&fakeDevMem);
    MockAclrtMemcpy(ACL_SUCCESS);
    MockCcuCustomChannel(HCCL_E_INTERNAL);
    MOCKER(aclrtFree).expects(once()).with(mockcpp::any()).will(returnValue(ACL_SUCCESS));

    EXPECT_EQ(HcommCcuSubmitInsts(0, 0, 0, insts, 2), static_cast<HcommResult>(HCCL_E_INTERNAL));
}

TEST_F(HcommResCcuEntityTest, Ut_HcommCcuSubmitInsts_When_Normal_Expect_SuccessAndDevMemFreed)
{
    uint8_t insts[64] = {};
    void* fakeDevMem = reinterpret_cast<void*>(0x10);
    MockAclrtMallocSuccess(&fakeDevMem);
    MockAclrtMemcpy(ACL_SUCCESS);
    MockCcuCustomChannel(HCCL_SUCCESS);
    MOCKER(aclrtFree).expects(once()).with(mockcpp::any()).will(returnValue(ACL_SUCCESS));

    EXPECT_EQ(HcommCcuSubmitInsts(0, 0, 0, insts, 2), static_cast<HcommResult>(HCCL_SUCCESS));
}

// 验证 C ABI 边界兜底异常时，已申请的设备暂存内存仍会被释放。
TEST_F(HcommResCcuEntityTest, Ut_HcommCcuSubmitInsts_When_Exception_Expect_InternalAndDevMemFreed)
{
    uint8_t insts[64] = {};
    void* fakeDevMem = reinterpret_cast<void*>(0x10);
    MockAclrtMallocSuccess(&fakeDevMem);
    MockAclrtMemcpy(ACL_SUCCESS);
    MOCKER(HccpRaTlvCcuCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(throws(std::runtime_error("test exception")));
    MOCKER(aclrtFree).expects(once()).with(mockcpp::any()).will(returnValue(ACL_SUCCESS));

    EXPECT_EQ(HcommCcuSubmitInsts(0, 0, 0, insts, 2), static_cast<HcommResult>(HCCL_E_INTERNAL));
}

TEST_F(HcommResCcuEntityTest, Ut_HcommCcuAllocInstSpace_When_StartInstIdNull_Expect_HCCL_E_PTR)
{
    EXPECT_EQ(HcommCcuAllocInstSpace(0, 0, 8, nullptr), static_cast<HcommResult>(HCCL_E_PTR));
}

TEST_F(HcommResCcuEntityTest, Ut_HcommCcuAllocInstSpace_When_InstNumZero_Expect_HCCL_E_PARA)
{
    uint32_t startInstId = START_INST_ID_SENTINEL;
    EXPECT_EQ(HcommCcuAllocInstSpace(0, 0, 0, &startInstId), static_cast<HcommResult>(HCCL_E_PARA));
    EXPECT_EQ(startInstId, 0U);
}

TEST_F(HcommResCcuEntityTest, Ut_HcommCcuAllocInstSpace_When_DieIdInvalid_Expect_HCCL_E_PARA)
{
    uint32_t startInstId = START_INST_ID_SENTINEL;
    EXPECT_EQ(HcommCcuAllocInstSpace(0, HCOMM_CCU_MAX_DIE_NUM, 8, &startInstId), static_cast<HcommResult>(HCCL_E_PARA));
    EXPECT_EQ(startInstId, 0U);
}

TEST_F(HcommResCcuEntityTest, Ut_HcommCcuAllocInstSpace_When_AllocInsFailed_Expect_ErrorPropagatedAndStartInstIdZero)
{
    MOCKER(CcuDevMgrImp::AllocIns)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_UNAVAIL));

    uint32_t startInstId = START_INST_ID_SENTINEL;
    EXPECT_EQ(HcommCcuAllocInstSpace(0, 0, 8, &startInstId), static_cast<HcommResult>(HCCL_E_UNAVAIL));
    EXPECT_EQ(startInstId, 0U);
}

TEST_F(HcommResCcuEntityTest, Ut_HcommCcuAllocInstSpace_When_Normal_Expect_SuccessAndStartInstIdFilled)
{
    const ResInfo allocRange(64U, 8U);
    MOCKER(CcuDevMgrImp::AllocIns)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(allocRange))
        .will(returnValue(HCCL_SUCCESS));

    uint32_t startInstId = 0;
    EXPECT_EQ(HcommCcuAllocInstSpace(0, 0, 8, &startInstId), static_cast<HcommResult>(HCCL_SUCCESS));
    EXPECT_EQ(startInstId, 64U);
}
