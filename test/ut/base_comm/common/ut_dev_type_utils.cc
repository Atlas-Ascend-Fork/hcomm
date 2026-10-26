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
#include <optional>
#include <sstream>
#include <string>
#include "securec.h"

#include "dev_type_utils.h"
#include "plf_debug_config.h"
#include "hcomm_c_adpt_common.h"

// Log-level controls provided by test/ut/stub/llt_hccl_stub.cc.
s32 log_level_get_stub();
void log_level_set_stub(s32 logLevel);

using namespace hcomm;

namespace {
DevType g_stubDevType = DevType::DEV_TYPE_910;
HcclResult g_stubRet = HCCL_SUCCESS;

HcclResult StubHrtGetDeviceType(DevType& devType)
{
    devType = g_stubDevType;
    return g_stubRet;
}
} // namespace

class DevTypeUtilsTest : public testing::Test {
protected:
    void SetUp() override
    {
        g_stubDevType = DevType::DEV_TYPE_910;
        g_stubRet = HCCL_SUCCESS;
        MOCKER(hrtGetDeviceType).stubs().will(invoke(StubHrtGetDeviceType));
    }
    void TearDown() override { GlobalMockObject::verify(); }
    // 切换打桩返回的设备类型与返回码（stub 函数读取全局状态，可反复切换）
    static void MockAs(DevType devType, HcclResult ret = HCCL_SUCCESS)
    {
        g_stubDevType = devType;
        g_stubRet = ret;
    }
};

namespace {
struct TruthRow {
    DevType devType;
    bool eq950;
    bool eq960;
    bool ge950;
    bool ge960;
    bool lt950;
};
} // namespace

// 全真值表：10 种设备类型 × 5 谓词 = 50 断言
TEST_F(DevTypeUtilsTest, ut_FivePredicates_TruthTable_Expect_Match)
{
    static const TruthRow rows[] = {
        {DevType::DEV_TYPE_910, false, false, false, false, true},
        {DevType::DEV_TYPE_310P3, false, false, false, false, true},
        {DevType::DEV_TYPE_910B, false, false, false, false, true},
        {DevType::DEV_TYPE_310P1, false, false, false, false, true},
        {DevType::DEV_TYPE_910_93, false, false, false, false, true},
        {DevType::DEV_TYPE_NOSOC, false, false, false, false, false},
        {DevType::DEV_TYPE_950, true, false, true, false, false},
        {DevType::DEV_TYPE_MC62, false, false, false, false, false},
        {DevType::DEV_TYPE_960, false, true, true, true, false},
        {DevType::DEV_TYPE_COUNT, false, false, false, false, false},
    };
    for (const auto& row : rows) {
        MockAs(row.devType);
        auto eq950 = EqualTo950();
        auto eq960 = EqualTo960();
        auto ge950 = EqualToOrGreaterThan950();
        auto ge960 = EqualToOrGreaterThan960();
        auto lt950 = LessThan950();
        ASSERT_TRUE(eq950.has_value());
        ASSERT_TRUE(eq960.has_value());
        ASSERT_TRUE(ge950.has_value());
        ASSERT_TRUE(ge960.has_value());
        ASSERT_TRUE(lt950.has_value());
        EXPECT_EQ(*eq950, row.eq950);
        EXPECT_EQ(*eq960, row.eq960);
        EXPECT_EQ(*ge950, row.ge950);
        EXPECT_EQ(*ge960, row.ge960);
        EXPECT_EQ(*lt950, row.lt950);
    }
}

// 失败路径：hrtGetDeviceType 返回失败时五谓词全部返回 nullopt
TEST_F(DevTypeUtilsTest, ut_FivePredicates_GetDevTypeFailed_Expect_AllNullopt)
{
    MockAs(DevType::DEV_TYPE_950, HCCL_E_INTERNAL);
    EXPECT_FALSE(EqualTo950().has_value());
    EXPECT_FALSE(EqualTo960().has_value());
    EXPECT_FALSE(EqualToOrGreaterThan950().has_value());
    EXPECT_FALSE(EqualToOrGreaterThan960().has_value());
    EXPECT_FALSE(LessThan950().has_value());
}

// 否定形回归：NOSOC/MC62 下 !*EqualToOrGreaterThan950() 为 true 而 *LessThan950() 为 false，五谓词非穷举划分
TEST_F(DevTypeUtilsTest, ut_NegationForm_NosocMc62_Expect_NotLessThan950)
{
    for (DevType devType : {DevType::DEV_TYPE_NOSOC, DevType::DEV_TYPE_MC62}) {
        MockAs(devType);
        auto ge950 = EqualToOrGreaterThan950();
        auto lt950 = LessThan950();
        ASSERT_TRUE(ge950.has_value());
        ASSERT_TRUE(lt950.has_value());
        EXPECT_TRUE(!(*ge950)); // 否定形：非 950+
        EXPECT_FALSE(*lt950);   // 但也不是 950 之前
    }
}

/* ===================== 512B 地址对齐检查日志（hcomm_c_adpt_common.h） ===================== */
namespace {
using Hccl::GetPlfDebugConfigValue;
using Hccl::IsAddrAligned512B;
using Hccl::LogAddrAlignCheck512B;
using Hccl::LogDescsAddrAlignCheck512B;
using Hccl::PLF_DATA_OP;
using Hccl::PLF_TASK;
using Hccl::SetPlfDebugConfigValue;

/* 512B 对齐的测试缓冲区：g_alignBuf 与 +512 均对齐，+1/+511 等不对齐，仅靠指针算术取址 */
alignas(512) char g_alignBuf[1024];

/* 构造批量传输描述符：各数据类子结构 dst/src 偏移一致，借道 write 成员赋值 */
HcommBatchTransferDesc MakeDataDesc(HcommTransferType type, void* dst, void* src)
{
    HcommBatchTransferDesc desc{};
    desc.transType = type;
    desc.transferInfo.write.dst = dst;
    desc.transferInfo.write.src = src;
    desc.transferInfo.write.len = 512;
    return desc;
}

HcommBatchTransferDesc MakeNotifyRecordDesc(uint32_t notifyIdx)
{
    HcommBatchTransferDesc desc{};
    desc.transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
    desc.transferInfo.notifyRecord.notifyIdx = notifyIdx;
    return desc;
}

HcommBatchTransferDesc MakeInvalidDesc()
{
    HcommBatchTransferDesc desc{};
    desc.transType = HCOMM_TRANSFER_TYPE_INVALID;
    return desc;
}
} // namespace

class AddrAlignCheckLogTest : public testing::Test {
protected:
    void SetUp() override
    {
        savedCfg_ = GetPlfDebugConfigValue();
        savedLogLevel_ = log_level_get_stub();
        log_level_set_stub(HCCL_LOG_INFO);
        // Keep INFO enabled so filtering cannot hide a missing DATA_OP gate.
        MOCKER(::HcclCheckLogLevel).stubs().will(returnValue(true));
    }
    void TearDown() override
    {
        SetPlfDebugConfigValue(savedCfg_); // 恢复全局调试开关，避免污染其他用例
        log_level_set_stub(savedLogLevel_);
        GlobalMockObject::verify();
    }
    template <typename Callback>
    std::string CaptureLog(Callback callback)
    {
        testing::internal::CaptureStdout();
        callback();
        return testing::internal::GetCapturedStdout();
    }

    void ExpectAlignLog(const std::string& log, const void* dst, const void* src, int descIndex = -1)
    {
        std::ostringstream expected;
        expected << "[ut_func] ";
        if (descIndex >= 0) {
            expected << "desc[" << descIndex << "] ";
        }
        char addressText[128] = {};
        ASSERT_GT(
            snprintf_s(
                addressText, sizeof(addressText), sizeof(addressText) - 1, "dst[%p] or src[%p]", const_cast<void*>(dst),
                const_cast<void*>(src)),
            0);
        expected << addressText << " is not aligned to 512B.";
        EXPECT_NE(log.find(expected.str()), std::string::npos) << log;
        EXPECT_NE(log.find("[PLF_DATA_OP]"), std::string::npos) << log;
    }

    size_t CountAlignLogs(const std::string& log)
    {
        const std::string marker = "is not aligned to 512B.";
        size_t count = 0;
        for (size_t pos = log.find(marker); pos != std::string::npos; pos = log.find(marker, pos + marker.size())) {
            ++count;
        }
        return count;
    }

    u64 savedCfg_ = 0;
    s32 savedLogLevel_ = 0;
};

// 纯逻辑断言：512B 对齐边界真值表
TEST_F(AddrAlignCheckLogTest, ut_IsAddrAligned512B_BoundaryValues_Expect_Match)
{
    // 对齐：0 地址(nullptr)、缓冲区首址、+512、+1024
    EXPECT_TRUE(IsAddrAligned512B(nullptr));
    EXPECT_TRUE(IsAddrAligned512B(&g_alignBuf[0]));
    EXPECT_TRUE(IsAddrAligned512B(&g_alignBuf[512]));
    EXPECT_TRUE(IsAddrAligned512B(&g_alignBuf[1024]));
    // 不对齐：+1、+128、+256、+511
    EXPECT_FALSE(IsAddrAligned512B(&g_alignBuf[1]));
    EXPECT_FALSE(IsAddrAligned512B(&g_alignBuf[128]));
    EXPECT_FALSE(IsAddrAligned512B(&g_alignBuf[256]));
    EXPECT_FALSE(IsAddrAligned512B(&g_alignBuf[511]));
}

// Set/Get 配置值回环：后续用例依赖该读写接口控制门控
TEST_F(AddrAlignCheckLogTest, ut_SetGetPlfDebugConfigValue_RoundTrip_Expect_Match)
{
    const u64 testValue = PLF_TASK | PLF_DATA_OP;
    SetPlfDebugConfigValue(testValue);
    EXPECT_EQ(GetPlfDebugConfigValue(), testValue);
}

// Gate-off must suppress output even when ordinary INFO logging is enabled.
TEST_F(AddrAlignCheckLogTest, ut_LogAddrAlignCheck512B_GateOff_Expect_NoLog)
{
    for (u64 config : {static_cast<u64>(0), PLF_TASK}) {
        SetPlfDebugConfigValue(config);
        const auto log = CaptureLog([] {
            LogAddrAlignCheck512B("ut_func", &g_alignBuf[1], &g_alignBuf[3]);
        });
        EXPECT_TRUE(log.empty()) << log;
    }
}

TEST_F(AddrAlignCheckLogTest, ut_LogAddrAlignCheck512B_GateOn_Expect_MatchingLogs)
{
    SetPlfDebugConfigValue(PLF_DATA_OP);
    const auto alignedLog = CaptureLog([] {
        LogAddrAlignCheck512B("ut_func", &g_alignBuf[0], &g_alignBuf[512]);
        LogAddrAlignCheck512B("ut_func", nullptr, nullptr);
    });
    EXPECT_TRUE(alignedLog.empty()) << alignedLog;

    const struct {
        const void* dst;
        const void* src;
    } cases[] = {
        {&g_alignBuf[1], &g_alignBuf[512]},
        {&g_alignBuf[0], &g_alignBuf[511]},
        {&g_alignBuf[1], &g_alignBuf[3]},
    };
    for (const auto& item : cases) {
        const auto log = CaptureLog([&] {
            LogAddrAlignCheck512B("ut_func", item.dst, item.src);
        });
        EXPECT_EQ(CountAlignLogs(log), 1U) << log;
        ExpectAlignLog(log, item.dst, item.src);
    }
}

TEST_F(AddrAlignCheckLogTest, ut_LogDescsAddrAlignCheck512B_GateOff_Expect_NoLog)
{
    HcommBatchTransferDesc descs[] = {
        MakeDataDesc(HCOMM_TRANSFER_TYPE_WRITE, &g_alignBuf[1], &g_alignBuf[0]),
        MakeDataDesc(HCOMM_TRANSFER_TYPE_READ, &g_alignBuf[3], &g_alignBuf[2]),
    };
    for (u64 config : {static_cast<u64>(0), PLF_TASK}) {
        SetPlfDebugConfigValue(config);
        const auto log = CaptureLog([&] {
            LogDescsAddrAlignCheck512B("ut_func", descs, 2);
        });
        EXPECT_TRUE(log.empty()) << log;
    }
}

TEST_F(AddrAlignCheckLogTest, ut_LogDescsAddrAlignCheck512B_GateOn_NullDescs_Expect_NoLog)
{
    SetPlfDebugConfigValue(PLF_DATA_OP);
    const auto log = CaptureLog([] {
        LogDescsAddrAlignCheck512B("ut_func", nullptr, 8);
        LogDescsAddrAlignCheck512B("ut_func", nullptr, 0);
    });
    EXPECT_TRUE(log.empty()) << log;
}

TEST_F(AddrAlignCheckLogTest, ut_LogDescsAddrAlignCheck512B_GateOn_ZeroNum_Expect_NoLog)
{
    auto desc = MakeDataDesc(HCOMM_TRANSFER_TYPE_WRITE, &g_alignBuf[1], &g_alignBuf[0]);
    SetPlfDebugConfigValue(PLF_DATA_OP);
    const auto log = CaptureLog([&] {
        LogDescsAddrAlignCheck512B("ut_func", &desc, 0);
    });
    EXPECT_TRUE(log.empty()) << log;
}

TEST_F(AddrAlignCheckLogTest, ut_LogDescsAddrAlignCheck512B_GateOn_MixedTypes_Expect_MatchingLogs)
{
    HcommBatchTransferDesc descs[] = {
        MakeDataDesc(HCOMM_TRANSFER_TYPE_WRITE, &g_alignBuf[1], &g_alignBuf[0]),
        MakeDataDesc(HCOMM_TRANSFER_TYPE_WRITE_REDUCE, &g_alignBuf[0], &g_alignBuf[512]),
        MakeDataDesc(HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY, &g_alignBuf[0], &g_alignBuf[511]),
        MakeDataDesc(HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY, &g_alignBuf[3], &g_alignBuf[7]),
        MakeDataDesc(HCOMM_TRANSFER_TYPE_READ, &g_alignBuf[0], &g_alignBuf[512]),
        MakeDataDesc(HCOMM_TRANSFER_TYPE_READ_REDUCE, &g_alignBuf[1], &g_alignBuf[511]),
        MakeNotifyRecordDesc(1),
        MakeInvalidDesc(),
    };
    // Unaligned address slots ensure removing either type filter is observable.
    descs[6].transferInfo.write.dst = &g_alignBuf[1];
    descs[6].transferInfo.write.src = &g_alignBuf[3];
    descs[7].transferInfo.write.dst = &g_alignBuf[1];
    descs[7].transferInfo.write.src = &g_alignBuf[3];
    SetPlfDebugConfigValue(PLF_DATA_OP);
    const auto log = CaptureLog([&] {
        LogDescsAddrAlignCheck512B("ut_func", descs, 8);
    });
    EXPECT_EQ(CountAlignLogs(log), 4U) << log;
    for (int index : {0, 2, 3, 5}) {
        ExpectAlignLog(log, descs[index].transferInfo.write.dst, descs[index].transferInfo.write.src, index);
    }
    for (int index : {1, 4, 6, 7}) {
        EXPECT_EQ(log.find("desc[" + std::to_string(index) + "]"), std::string::npos) << log;
    }
}
