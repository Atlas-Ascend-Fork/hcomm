/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <sstream>

#include "ccu_resource_stats_dfx.h"
#include "ccu_resource_stats_reporter.h"

namespace HcclSim {
namespace CcuResourceStats {
    namespace {

        OpKey MakeOpKey(const std::string& name, uint32_t opIter)
        {
            OpKey key;
            key.commName = name;
            key.commHash = 0xABCDEF;
            key.opIter = opIter;
            return key;
        }

        class CcuResourceStatsDfxTest : public testing::Test {
        protected:
            void SetUp() override
            {
                Collector::GetInstance().Reset();
                Collector::GetInstance().SetChannelTableProvider(nullptr);
                Collector::GetInstance().SetEnabled(true);
            }

            void TearDown() override
            {
                Collector::GetInstance().SetEnabled(false);
                Collector::GetInstance().SetChannelTableProvider(nullptr);
                Collector::GetInstance().Reset();
            }
        };

        // 开关关闭：BeginOp 返回无效 guard，所有记录 no-op，不产生任何桶
        TEST_F(CcuResourceStatsDfxTest, DisabledNoCollection)
        {
            Collector::GetInstance().SetEnabled(false);
            {
                auto guard = Collector::GetInstance().BeginOp(MakeOpKey("all_reduce", 0));
                EXPECT_FALSE(guard.IsValid());
                Collector::GetInstance().RecordXn(0, 0, 5);
                Collector::GetInstance().RecordCke(0, 0, 1);
            }
            EXPECT_TRUE(Collector::GetInstance().GetData().empty());
        }

        // 基本分桶与 set 去重：同一 op 内同 id 多次记录只计一次；不同 op 各自成桶互不影响
        TEST_F(CcuResourceStatsDfxTest, BucketingAndDedup)
        {
            {
                auto guard = Collector::GetInstance().BeginOp(MakeOpKey("all_reduce", 0));
                Collector::GetInstance().RecordXn(0, 0, 5);
                Collector::GetInstance().RecordXn(0, 0, 5); // 重复写同 id，去重
                Collector::GetInstance().RecordXn(0, 0, 7);
                Collector::GetInstance().RecordCke(0, 0, 1);
                Collector::GetInstance().RecordGsa(0, 0, 3);
                Collector::GetInstance().RecordMs(0, 0, 2);
                Collector::GetInstance().RecordChannel(0, 0, 9);
                Collector::GetInstance().RecordMission(0, 0, 4);
            }
            {
                auto guard = Collector::GetInstance().BeginOp(MakeOpKey("all_reduce", 1));
                Collector::GetInstance().RecordXn(0, 0, 5); // 与 op0 相同 id，独立桶各自计数
                Collector::GetInstance().RecordXn(1, 0, 8); // 不同 device 分行
            }

            const OpStatsMap& data = Collector::GetInstance().GetData();
            ASSERT_EQ(data.size(), 2U);

            const CcuResStats& stats0 = data.at(MakeOpKey("all_reduce", 0)).at(0).at(0);
            EXPECT_EQ(stats0.xnIds.size(), 2U);
            EXPECT_EQ(stats0.xnMaxId, 7);
            EXPECT_TRUE(stats0.xnUsed);
            EXPECT_EQ(stats0.ckeIds.size(), 1U);
            EXPECT_EQ(stats0.gsaIds.size(), 1U);
            EXPECT_EQ(stats0.msIds.size(), 1U);
            EXPECT_EQ(stats0.channelIds.size(), 1U);
            EXPECT_EQ(stats0.missionIds.size(), 1U);

            const DevStatsMap& devMap1 = data.at(MakeOpKey("all_reduce", 1));
            EXPECT_EQ(devMap1.at(0).at(0).xnIds.size(), 1U);
            EXPECT_EQ(devMap1.at(1).at(0).xnIds.size(), 1U);
            EXPECT_EQ(devMap1.at(1).at(0).xnMaxId, 8);
        }

        // RAII：guard 作用域提前退出（模拟 CheckOneOp 失败 return）后上下文正确关闭，
        // 下一个 op 的数据不会串入上一个桶
        TEST_F(CcuResourceStatsDfxTest, RaiiGuardClosesOnEarlyExit)
        {
            {
                auto guard = Collector::GetInstance().BeginOp(MakeOpKey("all_gather", 0));
                Collector::GetInstance().RecordXn(0, 0, 1);
                // 模拟失败提前 return：guard 析构即 EndOp
            }
            EXPECT_FALSE(Collector::GetInstance().IsInOp());
            // 上下文关闭后，裸记录调用应被短路（不入任何桶）
            Collector::GetInstance().RecordXn(0, 0, 99);
            {
                auto guard = Collector::GetInstance().BeginOp(MakeOpKey("all_gather", 1));
                Collector::GetInstance().RecordXn(0, 0, 2);
            }

            const OpStatsMap& data = Collector::GetInstance().GetData();
            ASSERT_EQ(data.size(), 2U);
            const std::set<uint16_t>& xn0 = data.at(MakeOpKey("all_gather", 0)).at(0).at(0).xnIds;
            const std::set<uint16_t>& xn1 = data.at(MakeOpKey("all_gather", 1)).at(0).at(0).xnIds;
            EXPECT_EQ(xn0.size(), 1U);
            EXPECT_EQ(xn0.count(1), 1U);
            EXPECT_EQ(xn0.count(99), 0U); // 上下文关闭后的记录被丢弃
            EXPECT_EQ(xn1.size(), 1U);
            EXPECT_EQ(xn1.count(2), 1U);
        }

        // loop 幂等记账：同 key 同值重复记录只计一次；同 key 不同值保留首次值
        TEST_F(CcuResourceStatsDfxTest, LoopGroupIdempotent)
        {
            LoopGroupKey key;
            key.missionId = 1;
            key.instStartId = 100;
            key.loopGroupInstrId = 120;

            {
                auto guard = Collector::GetInstance().BeginOp(MakeOpKey("reduce_scatter", 0));
                // n=4, o=2, m=3 -> expanded = 4 + (4-2)*3 = 10
                Collector::GetInstance().RecordLoopGroup(0, 0, key, {4, 10});
                Collector::GetInstance().RecordLoopGroup(0, 0, key, {4, 10}); // 重复展开，幂等
                // 同 key 冲突值：保留首次
                Collector::GetInstance().RecordLoopGroup(0, 0, key, {9, 99});
                // 不同 key（不同 SQE）：分别计数
                LoopGroupKey key2 = key;
                key2.instStartId = 200;
                Collector::GetInstance().RecordLoopGroup(0, 0, key2, {2, 2}); // o=n-1 边界：2+(2-1)*0=2
            }

            const CcuResStats& stats
                = Collector::GetInstance().GetData().at(MakeOpKey("reduce_scatter", 0)).at(0).at(0);
            EXPECT_EQ(stats.LoopGroupNum(), 2U);
            EXPECT_EQ(stats.LoopTemplateNum(), 6U); // 4 + 2
            EXPECT_EQ(stats.LoopNum(), 12U);        // 10 + 2（冲突值 99 被丢弃）
        }

        // 指令区间：半开区间 [begin, end) 入集合并去重；非法区间忽略
        TEST_F(CcuResourceStatsDfxTest, InstrRange)
        {
            {
                auto guard = Collector::GetInstance().BeginOp(MakeOpKey("all_reduce", 0));
                Collector::GetInstance().RecordInstrRange(0, 0, 10, 15);
                Collector::GetInstance().RecordInstrRange(0, 0, 13, 18); // 重叠区间并集去重
                Collector::GetInstance().RecordInstrRange(0, 0, 20, 20); // 空区间忽略
                Collector::GetInstance().RecordInstrRange(0, 0, 30, 25); // 非法区间忽略
                Collector::GetInstance().SetInstrResidentNum(0, 0, 512);
            }

            const CcuResStats& stats = Collector::GetInstance().GetData().at(MakeOpKey("all_reduce", 0)).at(0).at(0);
            EXPECT_EQ(stats.opInstrIds.size(), 8U); // [10,18) = 8 条
            EXPECT_EQ(stats.instrResidentNum, 512U);
        }

        // channel 差分参考口径：EndOp 时与 BeginOp 快照差分得 newChannelIds
        TEST_F(CcuResourceStatsDfxTest, ChannelProviderDiff)
        {
            ChannelTableSnapshot table;
            table[0][0] = {1, 2}; // op 开始前已有 channel 1、2
            Collector::GetInstance().SetChannelTableProvider([&table]() {
                return table;
            });

            {
                auto guard = Collector::GetInstance().BeginOp(MakeOpKey("all_reduce", 0));
                table[0][0].insert(3);                           // op 执行期间新增 channel 3
                table[1][0] = {7};                               // 新增 device 1 的 channel 7
                Collector::GetInstance().RecordChannel(0, 0, 1); // 主口径：引用收集
                Collector::GetInstance().RecordChannel(0, 0, 3);
            }

            const OpStatsMap& data = Collector::GetInstance().GetData();
            const CcuResStats& stats00 = data.at(MakeOpKey("all_reduce", 0)).at(0).at(0);
            EXPECT_EQ(stats00.channelIds.size(), 2U);    // 引用口径：1、3
            EXPECT_EQ(stats00.newChannelIds.size(), 1U); // 差分口径：仅 3 是新引入
            EXPECT_EQ(stats00.newChannelIds.count(3), 1U);
            const CcuResStats& stats10 = data.at(MakeOpKey("all_reduce", 0)).at(1).at(0);
            EXPECT_EQ(stats10.newChannelIds.size(), 1U); // 新 device 的全部算新引入
            EXPECT_EQ(stats10.newChannelIds.count(7), 1U);
            EXPECT_TRUE(stats10.channelIds.empty()); // 未被引用
        }

        // MS 上下文记录：无翻译上下文 no-op；设置后归属正确 (dev, die)
        TEST_F(CcuResourceStatsDfxTest, MsByTranslateCtx)
        {
            auto guard = Collector::GetInstance().BeginOp(MakeOpKey("all_reduce", 0));
            Collector::GetInstance().RecordMsByCtx(1); // 无上下文，丢弃
            Collector::GetInstance().SetTranslateCtx(2, 1);
            Collector::GetInstance().RecordMsByCtx(1);
            Collector::GetInstance().RecordMsByCtx(1); // 去重
            Collector::GetInstance().RecordMsByCtx(2);
            Collector::GetInstance().SetTranslateCtx(2, 0); // 逐 queue 切换 die
            Collector::GetInstance().RecordMsByCtx(1);

            const DevStatsMap& devMap = Collector::GetInstance().GetData().at(MakeOpKey("all_reduce", 0));
            EXPECT_EQ(devMap.count(2), 1U);
            EXPECT_EQ(devMap.at(2).at(1).msIds.size(), 2U); // die1: ms 1、2
            EXPECT_EQ(devMap.at(2).at(0).msIds.size(), 1U); // die0: ms 1
            Collector::GetInstance().ClearTranslateCtx();
            Collector::GetInstance().RecordMsByCtx(9); // 上下文清除后丢弃
            EXPECT_EQ(devMap.at(2).at(1).msIds.count(9), 0U);
        }

        // xnMaxId 峰值：id=0 也是有效使用（xnUsed 区分未使用）
        TEST_F(CcuResourceStatsDfxTest, XnMaxIdWithZero)
        {
            {
                auto guard = Collector::GetInstance().BeginOp(MakeOpKey("all_reduce", 0));
                Collector::GetInstance().RecordXn(0, 0, 0);
            }
            const CcuResStats& stats = Collector::GetInstance().GetData().at(MakeOpKey("all_reduce", 0)).at(0).at(0);
            EXPECT_TRUE(stats.xnUsed);
            EXPECT_EQ(stats.xnMaxId, 0);
            EXPECT_EQ(stats.xnIds.size(), 1U);
        }

        // run 级 Reset：清空全部桶与上下文
        TEST_F(CcuResourceStatsDfxTest, RunLevelReset)
        {
            {
                auto guard = Collector::GetInstance().BeginOp(MakeOpKey("all_reduce", 0));
                Collector::GetInstance().RecordXn(0, 0, 1);
            }
            EXPECT_FALSE(Collector::GetInstance().GetData().empty());
            Collector::GetInstance().Reset();
            EXPECT_TRUE(Collector::GetInstance().GetData().empty());
            EXPECT_FALSE(Collector::GetInstance().IsInOp());
        }

        // CSV 报告：分节格式（节头注释 + 节内列头/数据行）、排序、NA、拓扑标签缺失兜底 -1
        TEST_F(CcuResourceStatsDfxTest, ReportCsvFormat)
        {
            // op0: dev0/die0 与 dev1/die0，附带 OpInfo；op1（opIter 更大，排在后一节，无 OpInfo）
            {
                auto guard = Collector::GetInstance().BeginOp(MakeOpKey("all_reduce", 0));
                Collector::GetInstance().RecordXn(0, 0, 1);
                Collector::GetInstance().RecordCke(0, 0, 2);
                Collector::GetInstance().RecordMs(0, 0, 3);
                Collector::GetInstance().RecordChannel(0, 0, 4);
                Collector::GetInstance().RecordMission(0, 0, 5);
                Collector::GetInstance().RecordInstrRange(0, 0, 0, 10);
                Collector::GetInstance().RecordXn(1, 0, 9);
                OpInfo info0;
                info0.collectiveType = "ALLREDUCE";
                info0.algName = "ring";
                info0.dataType = "FP16";
                info0.elementCount = 1024;
                info0.inputBytes = 2048;
                info0.outputBytes = 2048;
                info0.reduceType = "SUM";
                info0.rankSize = 4;
                Collector::GetInstance().SetCurrentOpInfo(info0);
            }
            {
                auto guard = Collector::GetInstance().BeginOp(MakeOpKey("all_reduce", 1));
                Collector::GetInstance().RecordXn(0, 0, 1);
                Collector::GetInstance().RecordGsa(0, 0, 6); // 无 OpInfo：节头字段全默认 NA
            }

            TopoLabelMap labels;
            labels[0] = TopoLabel{7, 8, 3}; // dev0 有拓扑标签(含boardId)；dev1 缺失 → -1

            const std::string csvA5 = BuildCcuResourceStatsCsv(
                Collector::GetInstance().GetData(), Collector::GetInstance().GetOpInfos(), labels, true);
            std::istringstream issA5(csvA5);
            std::string line;
            // op0 节：节头两行（单列注释，"; " 分隔）+ 列头 + 2 数据行（superPodId 升序 → dev1(-1) 在前）
            ASSERT_TRUE(std::getline(issA5, line));
            EXPECT_EQ(line, "# ===== op[0]: commName=all_reduce; commHash=11259375; opIter=0 =====");
            ASSERT_TRUE(std::getline(issA5, line));
            EXPECT_EQ(
                line, "# opInfo: type=ALLREDUCE; alg=ring; dataType=FP16; elements=1024; "
                      "inputBytes=2048; outputBytes=2048; reduceType=SUM; rankSize=4");
            ASSERT_TRUE(std::getline(issA5, line));
            EXPECT_EQ(
                line, "superPodId,serverId,boardId,phyDevId,dieId,channelNum,instructionNum,loopTemplateNum,loopNum,"
                      "xnNum,gsaNum,ckeNum,msNum,missionNum");
            ASSERT_TRUE(std::getline(issA5, line));
            EXPECT_EQ(line, "-1,-1,-1,1,0,0,0,0,0,1,0,0,0,0");
            ASSERT_TRUE(std::getline(issA5, line));
            EXPECT_EQ(line, "7,8,3,0,0,1,10,0,0,1,0,1,1,1");
            // 节间空行
            ASSERT_TRUE(std::getline(issA5, line));
            EXPECT_EQ(line, "");
            // op1 节：无 OpInfo → 全默认值
            ASSERT_TRUE(std::getline(issA5, line));
            EXPECT_EQ(line, "# ===== op[1]: commName=all_reduce; commHash=11259375; opIter=1 =====");
            ASSERT_TRUE(std::getline(issA5, line));
            EXPECT_EQ(
                line, "# opInfo: type=NA; alg=NA; dataType=NA; elements=0; "
                      "inputBytes=0; outputBytes=0; reduceType=NA; rankSize=0");
            ASSERT_TRUE(std::getline(issA5, line)); // 列头
            ASSERT_TRUE(std::getline(issA5, line));
            EXPECT_EQ(line, "7,8,3,0,0,0,0,0,0,1,1,0,0,0");

            // A6：gsaNum 列输出 NA
            const std::string csvA6 = BuildCcuResourceStatsCsv(
                Collector::GetInstance().GetData(), Collector::GetInstance().GetOpInfos(), labels, false);
            EXPECT_NE(csvA6.find(",NA,0,0,0"), std::string::npos);
        }

        // 报告写盘：自动建目录、文件内容与 BuildCcuResourceStatsCsv 一致
        TEST_F(CcuResourceStatsDfxTest, ReportWriteToFile)
        {
            {
                auto guard = Collector::GetInstance().BeginOp(MakeOpKey("all_reduce", 0));
                Collector::GetInstance().RecordXn(0, 0, 1);
            }
            const std::string csvPath = "/tmp/ccu_resource_stats_dfx_ut/nested/ccu_resource_stats.csv";
            const HcclResult ret = GenerateCcuResourceStatsReport(csvPath, true, TopoLabelMap{});
            EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

            std::ifstream ifs(csvPath);
            ASSERT_TRUE(ifs.is_open());
            std::stringstream buffer;
            buffer << ifs.rdbuf();
            const std::string content = buffer.str();
            EXPECT_EQ(
                content,
                BuildCcuResourceStatsCsv(
                    Collector::GetInstance().GetData(), Collector::GetInstance().GetOpInfos(), TopoLabelMap{}, true));
            EXPECT_NE(content.find("# ===== op[0]: commName=all_reduce"), std::string::npos);
            EXPECT_NE(content.find("-1,-1,0,0,0,0,0,0,1,0,0,0,0"), std::string::npos);
            std::remove(csvPath.c_str());
        }

    } // namespace
} // namespace CcuResourceStats
} // namespace HcclSim
