/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>

#include "hccl/hccl_types.h"

// 被测桩来自 src/proxy/level0/ccu_fallback_stub.cc，覆盖 hccl 的两个 extern "C" 弱符号。
// 声明与 hccl/src/ops/op_common/inc/ccu_fallback_c.h 保持一致；不透明指针以 void*/char* 接收，
// 桩内只打印指针值，绝不解引用。
extern "C" {
HcclResult CheckCcuResNegotiationC(HcclComm comm, const void* param, bool localResAvailable);
HcclResult
CheckCcuParamAndFallbackC(HcclComm comm, void* param, void** topoInfo, char* algNameBuf, uint32_t algNameBufLen);
}

class CcuFallbackStubLevel0Test : public testing::Test {};

TEST_F(CcuFallbackStubLevel0Test, ResNegotiation_ValidArgs_ReturnsSuccess)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    int dummyParam = 0;
    EXPECT_EQ(CheckCcuResNegotiationC(comm, &dummyParam, true), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuFallbackStubLevel0Test, ResNegotiation_NullParam_ReturnsSuccess)
{
    EXPECT_EQ(CheckCcuResNegotiationC(nullptr, nullptr, false), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuFallbackStubLevel0Test, ResNegotiation_BothLocalResValues_ReturnsSuccess)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x2000);
    int dummyParam = 0;
    EXPECT_EQ(CheckCcuResNegotiationC(comm, &dummyParam, true), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(CheckCcuResNegotiationC(comm, &dummyParam, false), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuFallbackStubLevel0Test, ResNegotiation_BogusPointer_NotDereferenced)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x3000);
    const void* bogus = reinterpret_cast<const void*>(0x1);
    EXPECT_EQ(CheckCcuResNegotiationC(comm, bogus, true), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuFallbackStubLevel0Test, ParamAndFallback_ValidArgs_ReturnsSuccess)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x4000);
    int dummyParam = 0;
    void* dummyTopo = nullptr;
    char dummyAlg[64] = {0};
    EXPECT_EQ(
        CheckCcuParamAndFallbackC(comm, &dummyParam, &dummyTopo, dummyAlg, sizeof(dummyAlg)), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuFallbackStubLevel0Test, ParamAndFallback_AllNull_ReturnsSuccess)
{
    EXPECT_EQ(CheckCcuParamAndFallbackC(nullptr, nullptr, nullptr, nullptr, 0), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuFallbackStubLevel0Test, ParamAndFallback_BogusPointers_NotDereferenced)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x5000);
    void* bogusParam = reinterpret_cast<void*>(0x1);
    void** bogusTopo = reinterpret_cast<void**>(0x2);
    char* bogusAlg = reinterpret_cast<char*>(0x3);
    EXPECT_EQ(CheckCcuParamAndFallbackC(comm, bogusParam, bogusTopo, bogusAlg, 64), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuFallbackStubLevel0Test, BothStubs_RepeatableCalls_ReturnSuccess)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x6000);
    int dummy = 0;
    void* dummyTopo = nullptr;
    char dummyAlg[64] = {0};
    for (int i = 0; i < 50; i++) {
        EXPECT_EQ(CheckCcuResNegotiationC(comm, &dummy, true), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(
            CheckCcuParamAndFallbackC(comm, &dummy, &dummyTopo, dummyAlg, sizeof(dummyAlg)), HcclResult::HCCL_SUCCESS);
    }
}
