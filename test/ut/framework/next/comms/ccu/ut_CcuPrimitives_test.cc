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
 * UT: CCU 批量 Load/Store 与 LoadAddImm/AddImmStore wrapper 的数组边界校验
 *
 * 四个批量重载须在访问 vArr[0] 之前完成 num 校验：合法区间为 1 <= num <= vArr.size()，
 * 越界时返回 CCU_E_PARA 且不得转发到底层 C API（否则空 Array 会先发生空指针解引用）。
 * LoadAddImm/AddImmStore 使用 array.size() 作为元素个数，空数组或 size > UINT16_MAX
 * 时同样返回 CCU_E_PARA，避免解空指针或向 uint16_t 形参静默截断。
 */

#include <cstdint>

#include <mockcpp/mockcpp.hpp>

#include "gtest/gtest.h"
#include "ccu_primitives.hpp"
#include "ccu_types.h"

namespace {
constexpr uint64_t IMM_ADDR = 0x1000;
constexpr CcuVariableHandle ADDR_VAR_HANDLE = 0x2000;
constexpr CcuVariableHandle DST_VAR_HANDLE = 0x2100;
constexpr CcuVariableHandle ARRAY_BASE_HANDLE = 0x3000;
constexpr uint16_t IMM_ADD_VALUE = 1;
uint32_t g_varAllocCount = 0;

using CcuVariableArray = hcomm::ccu::Array<hcomm::ccu::Variable>;
// 非法参数场景：四个底层C API 均不得被调用
void ExpectCapiNeverCalled()
{
    MOCKER(CcuLoadVar).expects(never());
    MOCKER(CcuLoadVarFromVarAddr).expects(never());
    MOCKER(CcuStoreVar).expects(never());
    MOCKER(CcuStoreVarToVarAddr).expects(never());
}

void ExpectCapiForwarded(uint32_t num)
{
    MOCKER(CcuLoadVar)
        .expects(exactly(1))
        .with(eq(IMM_ADDR), eq(ARRAY_BASE_HANDLE), eq(num))
        .will(returnValue(CcuResult::CCU_SUCCESS));
    MOCKER(CcuLoadVarFromVarAddr)
        .expects(exactly(1))
        .with(eq(ADDR_VAR_HANDLE), eq(ARRAY_BASE_HANDLE), eq(num))
        .will(returnValue(CcuResult::CCU_SUCCESS));
    MOCKER(CcuStoreVar)
        .expects(exactly(1))
        .with(eq(IMM_ADDR), eq(ARRAY_BASE_HANDLE), eq(num))
        .will(returnValue(CcuResult::CCU_SUCCESS));
    MOCKER(CcuStoreVarToVarAddr)
        .expects(exactly(1))
        .with(eq(ADDR_VAR_HANDLE), eq(ARRAY_BASE_HANDLE), eq(num))
        .will(returnValue(CcuResult::CCU_SUCCESS));
}

void ExpectParaError(CcuVariableArray& vArr, hcomm::ccu::Variable& addrVar, uint32_t num)
{
    SCOPED_TRACE("size=" + std::to_string(vArr.size()) + " num=" + std::to_string(num));
    EXPECT_EQ(hcomm::ccu::Load(IMM_ADDR, vArr, num), CcuResult::CCU_E_PARA);
    EXPECT_EQ(hcomm::ccu::Load(addrVar, vArr, num), CcuResult::CCU_E_PARA);
    EXPECT_EQ(hcomm::ccu::Store(IMM_ADDR, vArr, num), CcuResult::CCU_E_PARA);
    EXPECT_EQ(hcomm::ccu::Store(addrVar, vArr, num), CcuResult::CCU_E_PARA);
}

void ExpectSuccess(CcuVariableArray& vArr, hcomm::ccu::Variable& addrVar, uint32_t num)
{
    SCOPED_TRACE("size=" + std::to_string(vArr.size()) + " num=" + std::to_string(num));
    EXPECT_EQ(hcomm::ccu::Load(IMM_ADDR, vArr, num), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(hcomm::ccu::Load(addrVar, vArr, num), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(hcomm::ccu::Store(IMM_ADDR, vArr, num), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(hcomm::ccu::Store(addrVar, vArr, num), CcuResult::CCU_SUCCESS);
}

void ExpectAddImmCapiNeverCalled()
{
    MOCKER(CcuLoadAddImm).expects(never());
    MOCKER(CcuAddImmStore).expects(never());
}

void ExpectAddImmCapiForwarded(uint16_t num)
{
    MOCKER(CcuLoadAddImm)
        .expects(exactly(1))
        .with(eq(ARRAY_BASE_HANDLE), eq(num), eq(ADDR_VAR_HANDLE), eq(IMM_ADD_VALUE), eq(DST_VAR_HANDLE))
        .will(returnValue(CcuResult::CCU_SUCCESS));
    MOCKER(CcuAddImmStore)
        .expects(exactly(1))
        .with(eq(ARRAY_BASE_HANDLE), eq(num), eq(ADDR_VAR_HANDLE), eq(IMM_ADD_VALUE), eq(DST_VAR_HANDLE))
        .will(returnValue(CcuResult::CCU_SUCCESS));
}

// Variable 默认构造会申请资源；第一次给 offset/addr，第二次给 dst/src，便于检出参数错位
CcuResult MockVariableAlloc(CcuVariableHandle* handle)
{
    if (handle == nullptr) {
        return CcuResult::CCU_E_PARA;
    }
    *handle = (g_varAllocCount == 0) ? ADDR_VAR_HANDLE : DST_VAR_HANDLE;
    g_varAllocCount++;
    return CcuResult::CCU_SUCCESS;
}

// Array 构造走批量申请，句柄连续以便校验 wrapper 转发的是首元素
CcuResult MockBlockVariableAlloc(CcuVariableHandle* handles, uint32_t count)
{
    if (handles == nullptr || count == 0) {
        return CcuResult::CCU_E_PARA;
    }
    for (uint32_t i = 0; i < count; ++i) {
        handles[i] = ARRAY_BASE_HANDLE + i;
    }
    return CcuResult::CCU_SUCCESS;
}
} // namespace

class CcuPrimitivesTest : public testing::Test {
public:
    void SetUp() override
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
        g_varAllocCount = 0;
        MOCKER(CcuVariableAlloc).stubs().will(invoke(MockVariableAlloc));
        MOCKER(CcuBlockVariableAlloc).stubs().will(invoke(MockBlockVariableAlloc));
    }

    void TearDown() override
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
};

// num 为 0 时四个批量重载均返回参数错误，且不触达底层 C API。
TEST_F(CcuPrimitivesTest, Ut_BatchLoadStore_When_NumIsZero_Expect_ParaErrorWithoutCallingCapi)
{
    ExpectCapiNeverCalled();
    hcomm::ccu::Array<hcomm::ccu::Variable> vArr(1);
    hcomm::ccu::Variable addrVar;
    ExpectParaError(vArr, addrVar, 0);
}

// 空 Array 配非零 num：修复前会先解引用空的 vArr[0]，修复后应返回参数错误。
TEST_F(CcuPrimitivesTest, Ut_BatchLoadStore_When_ArrayIsEmpty_Expect_ParaErrorWithoutCallingCapi)
{
    ExpectCapiNeverCalled();
    hcomm::ccu::Array<hcomm::ccu::Variable> vArr(0);
    hcomm::ccu::Variable addrVar;
    ExpectParaError(vArr, addrVar, 1);
}

// num 超过数组长度时返回参数错误，不得把越界数量转发到底层 C API。
TEST_F(CcuPrimitivesTest, Ut_BatchLoadStore_When_NumExceedsArraySize_Expect_ParaErrorWithoutCallingCapi)
{
    ExpectCapiNeverCalled();
    hcomm::ccu::Array<hcomm::ccu::Variable> vArr(2);
    hcomm::ccu::Variable addrVar;
    ExpectParaError(vArr, addrVar, 3U);
}

// num 取合法下边界 1（数组长度大于 1，与 num == size 的上边界用例区分开）：底层各调用一次。
TEST_F(CcuPrimitivesTest, Ut_BatchLoadStore_When_NumIsOne_Expect_ForwardToCapi)
{
    ExpectCapiForwarded(1U);
    hcomm::ccu::Array<hcomm::ccu::Variable> vArr(2);
    hcomm::ccu::Variable addrVar;
    ExpectSuccess(vArr, addrVar, 1U);
}

// num 等于数组长度的合法上边界：底层各调用一次，参数原样转发。
TEST_F(CcuPrimitivesTest, Ut_BatchLoadStore_When_NumEqualsArraySize_Expect_ForwardToCapi)
{
    ExpectCapiForwarded(2U);
    hcomm::ccu::Array<hcomm::ccu::Variable> vArr(2);
    hcomm::ccu::Variable addrVar;
    ExpectSuccess(vArr, addrVar, 2U);
}

// num 小于数组长度仍属合法：不得被新增的上界校验拦截，且转发的 num 为调用方传入值。
TEST_F(CcuPrimitivesTest, Ut_BatchLoadStore_When_NumLessThanArraySize_Expect_ForwardCallerNum)
{
    ExpectCapiForwarded(2U);
    hcomm::ccu::Array<hcomm::ccu::Variable> vArr(3);
    hcomm::ccu::Variable addrVar;
    ExpectSuccess(vArr, addrVar, 2U);
}

// 空 Array：修复前 LoadAddImm/AddImmStore 会先解引用空的 array[0]，修复后应返回参数错误。
TEST_F(CcuPrimitivesTest, Ut_LoadAddImmStore_When_ArrayIsEmpty_Expect_ParaErrorWithoutCallingCapi)
{
    ExpectAddImmCapiNeverCalled();
    hcomm::ccu::Array<hcomm::ccu::Variable> vArr(0);
    hcomm::ccu::Variable offset;
    hcomm::ccu::Variable other;
    EXPECT_EQ(hcomm::ccu::LoadAddImm(vArr, offset, IMM_ADD_VALUE, other), CcuResult::CCU_E_PARA);
    EXPECT_EQ(hcomm::ccu::AddImmStore(vArr, offset, IMM_ADD_VALUE, other), CcuResult::CCU_E_PARA);
}

// 非空 Array：按 array.size() 转发到底层，且 num 窄化为 uint16_t 后与数组长度一致。
TEST_F(CcuPrimitivesTest, Ut_LoadAddImmStore_When_ArrayIsNonEmpty_Expect_ForwardToCapi)
{
    constexpr uint16_t kArraySize = 2;
    ExpectAddImmCapiForwarded(kArraySize);
    hcomm::ccu::Array<hcomm::ccu::Variable> vArr(kArraySize);
    hcomm::ccu::Variable offset;
    hcomm::ccu::Variable other;
    EXPECT_EQ(hcomm::ccu::LoadAddImm(vArr, offset, IMM_ADD_VALUE, other), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(hcomm::ccu::AddImmStore(vArr, offset, IMM_ADD_VALUE, other), CcuResult::CCU_SUCCESS);
}
