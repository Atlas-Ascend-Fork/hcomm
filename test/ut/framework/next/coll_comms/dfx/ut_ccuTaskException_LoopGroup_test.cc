/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 覆盖 CcuTaskException 的 LoopGroup 异常信息生成逻辑。用例走真实链路不打桩
// (桩库 RaTlvRequest 固定返回全 0), 读硬件 Xn 的字段断言为 0。

#include "gtest/gtest.h"
#include <vector>

#include "ccu_ins_generator_v1.h"
#include "ccu_rep_base_v1.h"
#include "ccu_rep_nop_v1.h"
#include "ccu_rep_loopblock_v1.h"
#include "ccu_datatype_v1.h"
#include "ccu_dfx_schema.h"
#include "ccu_assist_v1.h"

#define private public
#define protected public
#include "ccuTaskException.h"
#undef protected
#undef private

namespace hcomm {
namespace {

    constexpr int32_t TEST_DEVICE_ID = 0;
    constexpr uint8_t TEST_DIE_ID = 0;
    constexpr uint8_t TEST_MISSION_ID = 0;

    class CcuTaskExceptionLoopGroupTest : public testing::Test {
    protected:
        // 构造已翻译的 LoopEntry: repLoopBlock 含 2 条 NOP 指令, 引擎号/迭代数按布局由调用方补充。
        static CcuRep::CcuRepLoopGroupBundle::LoopEntry MakeLoopEntry(
            CcuRep::CcuInsGeneratorBase* insGen, uint16_t executorId, uint64_t iterNum,
            CcuRep::CcuRepLoopGroupBundle::Layout layout)
        {
            CcuRep::CcuRepLoopGroupBundle::LoopEntry entry;
            entry.config = CcuLoopCfg{};
            entry.config.iterNum = iterNum;
            entry.executor = CcuRep::Executor(nullptr);
            entry.executor.Reset(executorId);
            entry.repLoopBlock = std::make_shared<CcuRep::CcuRepLoopBlock>(insGen, "loop_block");
            entry.repLoopBlock->Append(std::make_shared<CcuRep::CcuRepNop>(insGen));
            entry.repLoopBlock->Append(std::make_shared<CcuRep::CcuRepNop>(insGen));
            entry.loopParamVar = CcuRep::Variable(nullptr);
            entry.iterNumVar = CcuRep::Variable(nullptr);
            entry.addrOffsetVar = CcuRep::Variable(nullptr);
            entry.ctxIdVar = CcuRep::Variable(nullptr);
            entry.layout = layout;

            CcuRep::CcuInstr blockInstr[2]{};
            CcuRep::CcuInstr* blockPtr = blockInstr;
            uint16_t blockInstrId = 0;
            CcuRep::TransDep blockDep{0, 0, 1, 1, 0, {0, 0}, {0}, 0, 0, {0, 0, 0}, {0, 0}, 0, false};
            (void)entry.repLoopBlock->Translate(nullptr, blockPtr, blockInstrId, blockDep);
            return entry;
        }

        static ErrorInfoBase MakeBaseInfo(uint16_t currentInsId)
        {
            return ErrorInfoBase{TEST_DEVICE_ID, TEST_DIE_ID, TEST_MISSION_ID, currentInsId, 0};
        }
    };

    // ----------------------------- GenErrorInfoLoop -----------------------------

    // Config 布局 + V1: start/end 取自 repLoopBlock, engineId/loopCnt 读 loopParamVar Xn 解包(桩环境恒 0)
    TEST_F(CcuTaskExceptionLoopGroupTest, Ut_GenErrorInfoLoop_WhenConfigLayout_ExpectStaticFieldsAndBlockRepsFilled)
    {
        CcuRep::CcuInsGeneratorV1 insGen;
        auto entry = MakeLoopEntry(&insGen, 7, 5, CcuRep::CcuRepLoopGroupBundle::Layout::Config);

        std::vector<CcuErrorInfo> errorInfo;
        ASSERT_EQ(
            CcuTaskException::GenErrorInfoLoop(MakeBaseInfo(17), entry, CcuSchemaVersion::CCU_SCHEMA_V1, errorInfo),
            HCCL_SUCCESS);

        ASSERT_EQ(errorInfo.size(), 3U); // 1 条 LOOP 消息 + 2 条块内 rep 消息
        const auto& loopMsg = errorInfo[0];
        EXPECT_EQ(loopMsg.type, CcuErrorType::LOOP);
        EXPECT_EQ(loopMsg.repType, CcuRep::CcuRepType::LOOP); // 打印路由键, 路由到 GetCcuErrorMsgLoop 解 msg.loop
        EXPECT_EQ(loopMsg.instrId, 17U);
        EXPECT_EQ(loopMsg.msg.loop.startInstrId, entry.repLoopBlock->StartInstrId());
        EXPECT_EQ(
            loopMsg.msg.loop.endInstrId,
            static_cast<uint16_t>(entry.repLoopBlock->StartInstrId() + entry.repLoopBlock->InstrCount() - 1));
        EXPECT_EQ(loopMsg.msg.loop.loopEngineId, 0U); // 桩环境 loopParamVar Xn 为 0(真机为编程的 executor.Id)
        EXPECT_EQ(loopMsg.msg.loop.loopCnt, 0U); // 桩环境 loopParamVar Xn 为 0(真机为编程的 config.iterNum)
        EXPECT_EQ(loopMsg.msg.loop.loopCurrentCnt, 0U); // 桩环境 Xn/loop ctx 全 0
        EXPECT_EQ(loopMsg.msg.loop.addrStride, 0U);
        // 块内每条指令均被分发, 填入 DEFAULT 消息
        EXPECT_EQ(errorInfo[1].type, CcuErrorType::DEFAULT);
        EXPECT_EQ(errorInfo[1].instrId, entry.repLoopBlock->StartInstrId());
        EXPECT_EQ(errorInfo[2].type, CcuErrorType::DEFAULT);
        EXPECT_EQ(errorInfo[2].instrId, static_cast<uint16_t>(entry.repLoopBlock->StartInstrId() + 1U));
    }

    // PackedVar 布局 + V1: 走 loopParamVar 的 Xn 读取路径(桩环境返回 0), 流程不中断。
    TEST_F(CcuTaskExceptionLoopGroupTest, Ut_GenErrorInfoLoop_WhenPackedVarLayout_ExpectXnReadPathCovered)
    {
        CcuRep::CcuInsGeneratorV1 insGen;
        auto entry = MakeLoopEntry(&insGen, 2, 0, CcuRep::CcuRepLoopGroupBundle::Layout::PackedVar);

        std::vector<CcuErrorInfo> errorInfo;
        ASSERT_EQ(
            CcuTaskException::GenErrorInfoLoop(MakeBaseInfo(1), entry, CcuSchemaVersion::CCU_SCHEMA_V1, errorInfo),
            HCCL_SUCCESS);
        ASSERT_EQ(errorInfo.size(), 3U);
        EXPECT_EQ(errorInfo[0].msg.loop.loopEngineId, 0U);
        EXPECT_EQ(errorInfo[0].msg.loop.loopCnt, 0U); // 桩环境 loopParamVar Xn 值为 0
    }

    // VersionV2 布局 + V2: 走 ctxIdVar/iterNumVar 的 Xn 直读路径(桩环境返回 0), 流程不中断。
    TEST_F(CcuTaskExceptionLoopGroupTest, Ut_GenErrorInfoLoop_WhenVersionV2Layout_ExpectXnReadPathCovered)
    {
        CcuRep::CcuInsGeneratorV1 insGen;
        auto entry = MakeLoopEntry(&insGen, 3, 0, CcuRep::CcuRepLoopGroupBundle::Layout::VersionV2);

        std::vector<CcuErrorInfo> errorInfo;
        ASSERT_EQ(
            CcuTaskException::GenErrorInfoLoop(MakeBaseInfo(1), entry, CcuSchemaVersion::CCU_SCHEMA_V2, errorInfo),
            HCCL_SUCCESS);
        ASSERT_EQ(errorInfo.size(), 3U);
        EXPECT_EQ(errorInfo[0].msg.loop.loopEngineId, 0U);
        EXPECT_EQ(errorInfo[0].msg.loop.loopCnt, 0U); // 桩环境 ctxIdVar/iterNumVar Xn 值为 0
    }

    // LoopEntry 无 loop block: 打日志跳过该 loop, 不产生消息, 不中断其余错误信息输出。
    TEST_F(CcuTaskExceptionLoopGroupTest, Ut_GenErrorInfoLoop_WhenLoopBlockNull_ExpectSkipped)
    {
        CcuRep::CcuRepLoopGroupBundle::LoopEntry entry{};

        std::vector<CcuErrorInfo> errorInfo;
        EXPECT_EQ(
            CcuTaskException::GenErrorInfoLoop(MakeBaseInfo(1), entry, CcuSchemaVersion::CCU_SCHEMA_V1, errorInfo),
            HCCL_SUCCESS);
        EXPECT_TRUE(errorInfo.empty());
    }

    // loop block 无指令(空块): 打日志跳过该 loop, 防止 endInstrId 回绕导致块内循环 65536 次。
    TEST_F(CcuTaskExceptionLoopGroupTest, Ut_GenErrorInfoLoop_WhenLoopBlockEmpty_ExpectSkipped)
    {
        CcuRep::CcuInsGeneratorV1 insGen;
        auto entry = MakeLoopEntry(&insGen, 6, 5, CcuRep::CcuRepLoopGroupBundle::Layout::Config);
        entry.repLoopBlock->GetReps().clear(); // 清空 rep 使 InstrCount() 为 0

        std::vector<CcuErrorInfo> errorInfo;
        EXPECT_EQ(
            CcuTaskException::GenErrorInfoLoop(MakeBaseInfo(1), entry, CcuSchemaVersion::CCU_SCHEMA_V1, errorInfo),
            HCCL_SUCCESS);
        EXPECT_TRUE(errorInfo.empty());
    }

    // loop block 内指令空洞: 打日志跳过该指令继续遍历, LOOP 消息与命中的指令消息正常产出。
    TEST_F(CcuTaskExceptionLoopGroupTest, Ut_GenErrorInfoLoop_WhenRepInBlockMissing_ExpectSkippedAndContinue)
    {
        CcuRep::CcuInsGeneratorV1 insGen;
        auto entry = MakeLoopEntry(&insGen, 6, 5, CcuRep::CcuRepLoopGroupBundle::Layout::Config);
        // 制造指令空洞: 首条 NOP 平移到 [1,1], instrId=0 无法命中任何 rep, instrId=1 仍命中
        entry.repLoopBlock->GetReps()[0]->instrId = 1;

        std::vector<CcuErrorInfo> errorInfo;
        EXPECT_EQ(
            CcuTaskException::GenErrorInfoLoop(MakeBaseInfo(0), entry, CcuSchemaVersion::CCU_SCHEMA_V1, errorInfo),
            HCCL_SUCCESS);
        ASSERT_EQ(errorInfo.size(), 2U); // LOOP 消息 + instrId=1 命中的 1 条 DEFAULT
        EXPECT_EQ(errorInfo[0].type, CcuErrorType::LOOP);
        EXPECT_EQ(errorInfo[1].type, CcuErrorType::DEFAULT);
        EXPECT_EQ(errorInfo[1].instrId, 1U);
    }

    // ----------------------------- GenErrorInfoLoopGroup -----------------------------

    // bundle 含 2 个 LoopEntry: 先产出 LOOP_GROUP 消息, 再按 startLoopInstrId+i 逐个分发。
    // UT 桩环境无设备类型信息(GetCcuSchemaVersion 失败走跳过路径), 仅在版本可查环境下验证分发次序。
    TEST_F(CcuTaskExceptionLoopGroupTest, Ut_GenErrorInfoLoopGroup_WhenTwoLoops_ExpectGroupMsgAndPerLoopDispatch)
    {
        CcuSchemaVersion schemaVersion = CcuSchemaVersion::CCU_SCHEMA_COUNT;
        if (GetCcuSchemaVersion(schemaVersion) != HCCL_SUCCESS) {
            GTEST_SKIP() << "UT stub env has no device type info, skip LoopGroup dispatch verification";
        }
        CcuRep::CcuInsGeneratorV1 insGen;
        CcuLoopGroupCfg grpCfg{};
        CcuRep::Variable parallelVar(nullptr);
        CcuRep::Variable offsetVar(nullptr);
        auto bundle = std::make_shared<CcuRep::CcuRepLoopGroupBundle>(&insGen, grpCfg, parallelVar, offsetVar);
        bundle->AddLoop(MakeLoopEntry(&insGen, 1, 2, CcuRep::CcuRepLoopGroupBundle::Layout::Config));
        bundle->AddLoop(MakeLoopEntry(&insGen, 2, 3, CcuRep::CcuRepLoopGroupBundle::Layout::Config));
        bundle->instrId = 10; // 2 个 Config loop: offset = 2 + 2, startLoopInstrId = 10 + 4 + 3 = 17

        std::vector<CcuErrorInfo> errorInfo;
        std::shared_ptr<CcuRep::CcuRepBase> repBase = bundle;
        ASSERT_EQ(CcuTaskException::GenErrorInfoLoopGroup(MakeBaseInfo(12), repBase, errorInfo), HCCL_SUCCESS);

        // 1 条 LOOP_GROUP + (1 条 LOOP + 2 条 DEFAULT) * 2 个 loop
        ASSERT_EQ(errorInfo.size(), 7U);
        const auto& groupMsg = errorInfo[0];
        EXPECT_EQ(groupMsg.type, CcuErrorType::LOOP_GROUP);
        EXPECT_EQ(groupMsg.repType, CcuRep::CcuRepType::LOOPGROUP);
        EXPECT_EQ(groupMsg.instrId, 10U);
        EXPECT_EQ(groupMsg.msg.loopGroup.startLoopInsId, 17U);
        EXPECT_EQ(groupMsg.msg.loopGroup.loopInsCnt, 0U); // 桩环境 parallelVar Xn 值为 0
        // 第 1 个 loop: instrId = startLoopInstrId + 0; engineId/loopCnt 读硬件 Xn(桩环境恒 0)
        EXPECT_EQ(errorInfo[1].type, CcuErrorType::LOOP);
        EXPECT_EQ(errorInfo[1].instrId, 17U);
        EXPECT_EQ(errorInfo[1].msg.loop.loopEngineId, 0U);
        EXPECT_EQ(errorInfo[1].msg.loop.loopCnt, 0U);
        EXPECT_EQ(errorInfo[2].type, CcuErrorType::DEFAULT);
        EXPECT_EQ(errorInfo[3].type, CcuErrorType::DEFAULT);
        // 第 2 个 loop: instrId = startLoopInstrId + 1
        EXPECT_EQ(errorInfo[4].type, CcuErrorType::LOOP);
        EXPECT_EQ(errorInfo[4].instrId, 18U);
        EXPECT_EQ(errorInfo[4].msg.loop.loopEngineId, 0U);
        EXPECT_EQ(errorInfo[4].msg.loop.loopCnt, 0U);
        EXPECT_EQ(errorInfo[5].type, CcuErrorType::DEFAULT);
        EXPECT_EQ(errorInfo[6].type, CcuErrorType::DEFAULT);
    }

    // bundle 无 loop: 仅产出 LOOP_GROUP 消息, 不分发任何 loop。
    // UT 桩环境无设备类型信息时版本查询失败走跳过路径, 仅在版本可查环境下验证。
    TEST_F(CcuTaskExceptionLoopGroupTest, Ut_GenErrorInfoLoopGroup_WhenNoLoops_ExpectGroupMsgOnly)
    {
        CcuSchemaVersion schemaVersion = CcuSchemaVersion::CCU_SCHEMA_COUNT;
        if (GetCcuSchemaVersion(schemaVersion) != HCCL_SUCCESS) {
            GTEST_SKIP() << "UT stub env has no device type info, skip LoopGroup dispatch verification";
        }
        CcuRep::CcuInsGeneratorV1 insGen;
        CcuLoopGroupCfg grpCfg{};
        CcuRep::Variable parallelVar(nullptr);
        CcuRep::Variable offsetVar(nullptr);
        auto bundle = std::make_shared<CcuRep::CcuRepLoopGroupBundle>(&insGen, grpCfg, parallelVar, offsetVar);
        bundle->instrId = 10; // 0 个 loop: offset = 0 + 2, startLoopInstrId = 10 + 2 + 3 = 15

        std::vector<CcuErrorInfo> errorInfo;
        std::shared_ptr<CcuRep::CcuRepBase> repBase = bundle;
        ASSERT_EQ(CcuTaskException::GenErrorInfoLoopGroup(MakeBaseInfo(12), repBase, errorInfo), HCCL_SUCCESS);

        ASSERT_EQ(errorInfo.size(), 1U);
        EXPECT_EQ(errorInfo[0].type, CcuErrorType::LOOP_GROUP);
        EXPECT_EQ(errorInfo[0].msg.loopGroup.startLoopInsId, 15U);
        EXPECT_EQ(errorInfo[0].msg.loopGroup.loopInsCnt, 0U);
    }

    // 首 个 LoopEntry 无 loop block: 打日志跳过该 loop, GenErrorInfoLoopGroup 正常返回。
    // UT 桩环境无设备类型信息时版本查询失败走跳过路径, 仅在版本可查环境下验证。
    TEST_F(CcuTaskExceptionLoopGroupTest, Ut_GenErrorInfoLoopGroup_WhenFirstLoopBlockNull_ExpectSkipped)
    {
        CcuSchemaVersion schemaVersion = CcuSchemaVersion::CCU_SCHEMA_COUNT;
        if (GetCcuSchemaVersion(schemaVersion) != HCCL_SUCCESS) {
            GTEST_SKIP() << "UT stub env has no device type info, skip LoopGroup dispatch verification";
        }
        CcuRep::CcuInsGeneratorV1 insGen;
        CcuLoopGroupCfg grpCfg{};
        CcuRep::Variable parallelVar(nullptr);
        CcuRep::Variable offsetVar(nullptr);
        auto bundle = std::make_shared<CcuRep::CcuRepLoopGroupBundle>(&insGen, grpCfg, parallelVar, offsetVar);
        auto entry = MakeLoopEntry(&insGen, 1, 2, CcuRep::CcuRepLoopGroupBundle::Layout::Config);
        entry.repLoopBlock = nullptr;
        bundle->AddLoop(entry);
        bundle->instrId = 10;

        std::vector<CcuErrorInfo> errorInfo;
        std::shared_ptr<CcuRep::CcuRepBase> repBase = bundle;
        EXPECT_EQ(CcuTaskException::GenErrorInfoLoopGroup(MakeBaseInfo(12), repBase, errorInfo), HCCL_SUCCESS);
        ASSERT_EQ(errorInfo.size(), 1U); // 仅 LOOP_GROUP 消息, 该 loop 被跳过
        EXPECT_EQ(errorInfo[0].type, CcuErrorType::LOOP_GROUP);
    }

    // ----------------------------- DecodeLoopGroupXn -----------------------------

    // V1(950) 位布局: 按 GetParallelParam 打包的值能被正确解出 (totalLoopNum@41/repeatLoopIndex@48/repeatNum@55)。
    TEST_F(CcuTaskExceptionLoopGroupTest, Ut_DecodeLoopGroupXn_WhenSchemaV1_ExpectV1LayoutDecoded)
    {
        // GetParallelParam(repeatNum=3, repeatLoopIndex=5, totalLoopNum=7)
        const uint64_t xnValue = CcuRep::GetParallelParam(3, 5, 7);
        const auto fields = CcuTaskException::DecodeLoopGroupXn(xnValue, CcuSchemaVersion::CCU_SCHEMA_V1);
        EXPECT_EQ(fields.loopInsCnt, 7U);
        EXPECT_EQ(fields.expandOffset, 5U);
        EXPECT_EQ(fields.expandCnt, 3U);
    }

    // V2 位布局: 按 GetParallelParamV2 打包的值能被正确解出(修复前按 V1 位域解恒为 0)
    TEST_F(CcuTaskExceptionLoopGroupTest, Ut_DecodeLoopGroupXn_WhenSchemaV2_ExpectV2LayoutDecoded)
    {
        // GetParallelParamV2(repeatNum=3, repeatLoopIndex=5, totalLoopNum=7)
        const uint64_t xnValue = CcuRep::GetParallelParamV2(3, 5, 7);
        const auto fields = CcuTaskException::DecodeLoopGroupXn(xnValue, CcuSchemaVersion::CCU_SCHEMA_V2);
        EXPECT_EQ(fields.loopInsCnt, 7U);
        EXPECT_EQ(fields.expandOffset, 5U);
        EXPECT_EQ(fields.expandCnt, 3U);
    }

} // namespace
} // namespace hcomm
