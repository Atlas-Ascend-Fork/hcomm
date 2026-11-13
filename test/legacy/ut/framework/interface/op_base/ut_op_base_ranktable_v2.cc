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
#include <cstdio>
#include <fstream>
#include <string>

#include "param_check_v2.h"
#include "op_base_v2.h"
#include "comm_manager.h"
#include "rank_table_crc_bridge.h"
#include "orion_adapter_rts.h"

using namespace Hccl;

namespace {
// 写入指定内容的 ranktable 文件(空内容即为空文件), 返回是否创建成功
bool Ut_RankTableFileWrite(const char* filename, const std::string& content)
{
    std::ofstream file(filename, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        return false;
    }
    file << content;
    return file.good();
}
} // namespace

class OpBaseRankTableV2Test : public testing::Test {
protected:
    void TearDown() override
    {
        GlobalMockObject::verify();
        // 清理本用例可能记录的 device 0 CRC, 避免影响其它用例
        (void)RankTableCrcBridge::GetInstance().ConsumeRankTableJsonCrc(0);
    }
};

// 空文件在读取阶段即被拒绝, 返回 HCCL_E_PARA(该校验先于 aclrtGetDevice 执行)
TEST_F(OpBaseRankTableV2Test, Ut_HcomLoadRankTableFileV2_When_FileEmpty_Expect_Para)
{
    const char* fileName = "ut_ranktable_v2_empty.json";
    ASSERT_TRUE(Ut_RankTableFileWrite(fileName, ""));
    std::string rankTableM;
    HcclResult ret = HcomLoadRankTableFileV2(fileName, rankTableM);
    EXPECT_EQ(ret, HCCL_E_PARA);
    (void)remove(fileName);
}

// 文件不存在时返回 HCCL_E_PARA
TEST_F(OpBaseRankTableV2Test, Ut_HcomLoadRankTableFileV2_When_FileMissing_Expect_Para)
{
    std::string rankTableM;
    HcclResult ret = HcomLoadRankTableFileV2("ut_ranktable_v2_no_such_file.json", rankTableM);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 装载器只负责读文件, 不做 JSON 语法校验: 截断 JSON 装载成功且内容原样保留,
// 语法合法性由入口的 rank_count 门禁与下层解析器负责
TEST_F(OpBaseRankTableV2Test, Ut_HcomLoadRankTableFileV2_When_TruncatedJson_Expect_SuccessAndContentKept)
{
    MOCKER(aclrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    const char* fileName = "ut_ranktable_v2_truncated.json";
    const std::string truncatedJson = "{\"version\": \"2.0\", \"rank_count\": 2, \"rank_list\": [";
    ASSERT_TRUE(Ut_RankTableFileWrite(fileName, truncatedJson));
    std::string rankTableM;
    HcclResult ret = HcomLoadRankTableFileV2(fileName, rankTableM);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankTableM, truncatedJson);
    (void)remove(fileName);
}

// 入口门禁: 顶层不存在 rank_count 键时返回 HCCL_E_PARA。
// 覆盖三类反例: 完全缺失 / 仅在嵌套对象中出现 / 仅作为字符串值出现(不做全文子串匹配)
TEST_F(OpBaseRankTableV2Test, Ut_HcclCommInitClusterInfoMemConfigV2_When_NoTopLevelRankCount_Expect_Para)
{
    MOCKER(CallSingletons).stubs().will(returnValue(HCCL_SUCCESS));
    const std::string variants[] = {
        "{\"version\": \"2.0\", \"rank_list\": []}",                    // 完全缺失 rank_count
        "{\"version\": \"2.0\", \"rank_list\": [{\"rank_count\": 1}]}", // 仅在嵌套对象中出现
        "{\"version\": \"rank_count\", \"rank_list\": []}",             // 仅作为字符串值出现
    };
    for (const auto& rankTableString : variants) {
        HcclCommConfig config = {};
        HcclComm comm = nullptr;
        HcclResult ret = HcclCommInitClusterInfoMemConfigV2(rankTableString.c_str(), 0, &config, &comm);
        EXPECT_EQ(ret, HCCL_E_PARA) << "variant: " << rankTableString;
        EXPECT_EQ(comm, nullptr);
    }
}
