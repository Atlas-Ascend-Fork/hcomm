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
#include <set>

#include "ccu_ins_generator_v2.h"
#include "ccu_rep_v1.h"
#include "ccu_rep_context_v1.h"
#include "ccu_api_exception.h"
#include "ccu_assist_v1.h"
#include "hcomm_c_adpt.h"
#include "src/base_comm/resources/ccu/ccu_kernel/ccu_kernel.h"
#include "src/base_comm/resources/endpoint_pairs/channels/ccu/ccu_urma_channel.h"

namespace hcomm {
namespace CcuRep {
    namespace {

        // 各远端信息取互不相同的值，登记或取用时若混用字段（如 signal 与 var 地址互换）可被识别
        constexpr ChannelHandle CHANNEL_HANDLE = 0xCCAA7701ULL;
        constexpr uint32_t CHANNEL_ID = 0x17;
        constexpr uint16_t SEM_INDEX = 3;
        constexpr uint16_t PARAM_INDEX = 5;
        constexpr uint16_t MASK = 0x0004;
        uint64_t g_rmtSignalAddr = 0x1000A000ULL;
        uint64_t g_rmtVarAddr = 0x2000B000ULL;
        uint32_t g_rmtTokenId = 0x31;
        uint32_t g_rmtTokenValue = 0x5A5A;

        CcuUrmaChannel* g_channel = nullptr;

        HcommResult StubChannelGet(ChannelHandle handle, void** channel)
        {
            if (channel == nullptr || handle != CHANNEL_HANDLE) {
                return HCCL_E_NOT_FOUND;
            }
            *channel = static_cast<Channel*>(g_channel);
            return HCCL_SUCCESS;
        }

        class CcuInsGeneratorV2RemPostTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV2 insGen{};
            CcuRepContext context{};
            CcuKernel kernel{};
            HcommChannelDesc channelDesc{};
            CcuUrmaChannel channel{nullptr, channelDesc, COMM_ENGINE_RESERVED};
            TransDep dep{};
            uint64_t rmtToken{GetToken(g_rmtTokenId, g_rmtTokenValue, 1)};

            void SetUp() override
            {
                g_channel = &channel;
                // 常量登记经 CreateVariable 按 dieId 记录资源，默认值 UINT32_MAX 会越界
                kernel.SetDieId(0);
                MOCKER(HcommChannelGet).stubs().will(invoke(StubChannelGet));
                MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(CHANNEL_ID));
                MOCKER_CPP(&CcuUrmaChannel::GetRmtCcuBufferTokenInfo)
                    .stubs()
                    .with(outBound(g_rmtTokenId), outBound(g_rmtTokenValue))
                    .will(returnValue(HCCL_SUCCESS));
            }

            void TearDown() override
            {
                GlobalMockObject::verify();
                g_channel = nullptr;
            }

            void MockSignalAddr(HcclResult ret = HCCL_SUCCESS)
            {
                MOCKER_CPP(&CcuUrmaChannel::GetRmtSignalAddrByIndex)
                    .stubs()
                    .with(eq(static_cast<uint32_t>(SEM_INDEX)), outBound(g_rmtSignalAddr))
                    .will(returnValue(ret));
            }

            void MockVarAddr(HcclResult ret = HCCL_SUCCESS)
            {
                MOCKER_CPP(&CcuUrmaChannel::GetRmtVarAddrByIndex)
                    .expects(once())
                    .with(eq(static_cast<uint32_t>(PARAM_INDEX)), outBound(g_rmtVarAddr))
                    .will(returnValue(ret));
            }

            std::set<uint64_t> ConstKeys()
            {
                std::set<uint64_t> keys;
                for (const auto& kv : kernel.GetConstValue2VarMap()) {
                    keys.insert(kv.first);
                }
                return keys;
            }

            uint16_t ConstId(uint64_t value) { return kernel.GetConstValue2VarMap().at(value).Id(); }
        };

        TEST_F(CcuInsGeneratorV2RemPostTest, RemPostSem_PrepareAndTranslate_Expect_SyncWtXWithSignalAndToken)
        {
            MockSignalAddr();
            MOCKER_CPP(&CcuUrmaChannel::GetRmtVarAddrByIndex).expects(never());
            CcuRepRemPostSem rep(&insGen, CHANNEL_HANDLE, SEM_INDEX, MASK);

            ASSERT_EQ(insGen.PrepareConstValue(&rep, dep, &kernel), HCCL_SUCCESS);
            EXPECT_EQ(ConstKeys(), (std::set<uint64_t>{CHANNEL_ID, g_rmtSignalAddr, rmtToken}));
            EXPECT_TRUE(kernel.IsSecConstValue(rmtToken));
            EXPECT_FALSE(kernel.IsSecConstValue(g_rmtSignalAddr));

            CcuInstr instr{};
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 0;
            ASSERT_TRUE(rep.Translate(&kernel, instrPtr, instrId, dep));
            EXPECT_EQ(instrPtr, &instr + 1);

            // 仅置位远端 CKE：xd/xdt 为远端 signal 地址与 token，xs 为 mask
            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0xD);
            EXPECT_EQ(instr.v2.syncWtX.xdId, ConstId(g_rmtSignalAddr));
            EXPECT_EQ(instr.v2.syncWtX.xdtId, ConstId(rmtToken));
            EXPECT_EQ(instr.v2.syncWtX.xsId, MASK);
            EXPECT_EQ(instr.v2.syncWtX.xcId, ConstId(CHANNEL_ID));
            EXPECT_EQ(instr.v2.syncWtX.notifyValid, 0);
            EXPECT_EQ(instr.v2.syncWtX.parMode, 0);
        }

        TEST_F(CcuInsGeneratorV2RemPostTest, RemPostVar_PrepareAndTranslate_Expect_SyncWtXToRmtVarAddr)
        {
            MockSignalAddr();
            MOCKER_CPP(&CcuUrmaChannel::GetRmtVarAddrByIndex)
                .expects(exactly(2))
                .with(eq(static_cast<uint32_t>(PARAM_INDEX)), outBound(g_rmtVarAddr))
                .will(returnValue(HCCL_SUCCESS));
            Variable param{&context};
            CcuRepRemPostVar rep(&insGen, param, CHANNEL_HANDLE, PARAM_INDEX, SEM_INDEX, MASK);

            ASSERT_EQ(insGen.PrepareConstValue(&rep, dep, &kernel), HCCL_SUCCESS);
            EXPECT_EQ(ConstKeys(), (std::set<uint64_t>{CHANNEL_ID, g_rmtSignalAddr, rmtToken, g_rmtVarAddr}));
            EXPECT_TRUE(kernel.IsSecConstValue(rmtToken));
            EXPECT_FALSE(kernel.IsSecConstValue(g_rmtVarAddr));

            CcuInstr instr{};
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 0;
            ASSERT_TRUE(rep.Translate(&kernel, instrPtr, instrId, dep));
            EXPECT_EQ(instrPtr, &instr + 1);

            // 本端 param 写入远端 var 地址，并以 notify 置位远端 signal
            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0xD);
            EXPECT_EQ(instr.v2.syncWtX.xdId, ConstId(g_rmtVarAddr));
            EXPECT_EQ(instr.v2.syncWtX.xdtId, ConstId(rmtToken));
            EXPECT_EQ(instr.v2.syncWtX.xsId, param.Id());
            EXPECT_EQ(instr.v2.syncWtX.xcId, ConstId(CHANNEL_ID));
            EXPECT_EQ(instr.v2.syncWtX.xnId, ConstId(g_rmtSignalAddr));
            EXPECT_EQ(instr.v2.syncWtX.xntId, ConstId(rmtToken));
            EXPECT_EQ(instr.v2.syncWtX.value, MASK);
            EXPECT_EQ(instr.v2.syncWtX.notifyValid, 1);
            EXPECT_EQ(instr.v2.syncWtX.parMode, 1);
        }

        TEST_F(CcuInsGeneratorV2RemPostTest, RemPostSem_Prepare_When_GetSignalAddrFail_Expect_Internal)
        {
            MockSignalAddr(HCCL_E_PARA);
            CcuRepRemPostSem rep(&insGen, CHANNEL_HANDLE, SEM_INDEX, MASK);
            EXPECT_EQ(insGen.PrepareConstValue(&rep, dep, &kernel), HCCL_E_INTERNAL);
            EXPECT_TRUE(kernel.GetConstValue2VarMap().empty());
        }

        TEST_F(CcuInsGeneratorV2RemPostTest, RemPostVar_Prepare_When_GetVarAddrFail_Expect_Internal)
        {
            MockSignalAddr();
            MockVarAddr(HCCL_E_PARA);
            Variable param{&context};
            CcuRepRemPostVar rep(&insGen, param, CHANNEL_HANDLE, PARAM_INDEX, SEM_INDEX, MASK);
            EXPECT_EQ(insGen.PrepareConstValue(&rep, dep, &kernel), HCCL_E_INTERNAL);
            EXPECT_TRUE(kernel.GetConstValue2VarMap().empty());
        }

        TEST_F(CcuInsGeneratorV2RemPostTest, RemPostSem_Prepare_When_ChannelNotFound_Expect_Fail)
        {
            CcuRepRemPostSem rep(&insGen, CHANNEL_HANDLE + 1, SEM_INDEX, MASK);
            EXPECT_NE(insGen.PrepareConstValue(&rep, dep, &kernel), HCCL_SUCCESS);
            EXPECT_TRUE(kernel.GetConstValue2VarMap().empty());
        }

    } // namespace
} // namespace CcuRep
} // namespace hcomm
