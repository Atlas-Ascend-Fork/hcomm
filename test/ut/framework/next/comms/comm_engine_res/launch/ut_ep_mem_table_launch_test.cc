/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ===================== ep_mem_table_launch 下发/回收链路 UT（检视整改） =====================
// 该模块此前在全仓测试中一律被 MOCKER 打桩、自身 0 用例覆盖，而 launch 失败后的尽力回收、
// D2H 读回失败透传、坏输出返回 HCCL_E_INTERNAL 恰是防 device 表泄漏的兜底分支。
// 本组用例保留真实编排（bin 加载 → 自建 stream → 参数 H2D → launch → sync → D2H 读回），
// 仅桩设备边界，桩集先例：ut_channel_process_test.cc 与 ut_my_rank.cc InstallCreateChannelsLaunchMocks。
// 场景④⑤补齐表同步正向链路：Update 正常下发（断言 kernel 名与 40 字节参数 ABI）与
// SyncUbLocMemToEpTable 未初始化守卫（tablePtr==0 零下发）。
// 场景⑥补齐 sync 失败路径：错误码透传、outTablePtr 保持 0、不盲删 destroy，并以 freeCount
// 锚定「参数块随 return 析构」的当前生命周期语义。

#include "../../../ut_hcomm_base.h"
#include "ep_mem_table_launch.h"
#include "endpoint_pair/channel/channel_param_pub.h"
#include "launch_aicpu.h"
#include "mem_device_pub.h"
#include "adapter_rts_common.h"
#include "base_comm_node_accessors.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static constexpr const char* EP_MEM_TABLE_INIT_KERNEL = "RunAicpuEpMemTableInit";
static constexpr const char* EP_MEM_TABLE_UPDATE_KERNEL = "RunAicpuEpMemTableUpdate";
static constexpr const char* EP_MEM_TABLE_DESTROY_KERNEL = "RunAicpuEpMemTableDestroy";

class TestEpMemTableLaunch : public TestHcommCAdptBase {
public:
    void SetUp() override
    {
        TestHcommCAdptBase::SetUp();
        activeTest_ = this;
        // bin 边界：进程级 AicpuCommEngineBinProc 成员直接打桩（先例：ut_channel_process_test.cc）
        MOCKER_CPP(&hcomm::AicpuCommEngineBinProc::EnsureKernelBinLoaded).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&hcomm::AicpuCommEngineBinProc::GetBinHandle).stubs().will(returnValue(fakeBinHandle_));
        // Stream(STREAM_TYPE_ONLINE) 构造边界：创建失败使 stream_ 保持空，规避 InitStream 与
        // 析构销毁连锁（先例：ut_hccl_comm_host.cc）；hrtGetHcclV2Support 出参不回写即 false
        MOCKER(hrtStreamCreateWithFlags).stubs().will(returnValue(static_cast<HcclResult>(HCCL_E_RUNTIME)));
        MOCKER(hrtGetHcclV2Support).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtStreamSetMode).stubs().will(returnValue(HCCL_SUCCESS));
        // 设备内存边界：真 malloc/free，保障 DeviceMem 生命周期安全
        // （free 计数供场景⑥生命周期锚点断言）
        MOCKER(hrtMalloc).stubs().will(invoke(+[](void** ptr, u64 size, bool) -> HcclResult {
            *ptr = std::calloc(1, size);
            return (*ptr != nullptr) ? HCCL_SUCCESS : HCCL_E_MEMORY;
        }));
        MOCKER(hrtFree).stubs().will(invoke(+[](void* ptr) -> HcclResult {
            if (activeTest_ != nullptr) {
                activeTest_->freeCount_++;
            }
            std::free(ptr);
            return HCCL_SUCCESS;
        }));
        MOCKER(hrtMemSyncCopy).stubs().will(invoke(StubHrtMemSyncCopy));
        MOCKER(hccl::AicpuAclKernelLaunch).stubs().will(invoke(StubAicpuKernelLaunch));
        // sync 桩改 fixture 注入式（场景⑥注入超时），默认 SUCCESS 保持既有 5 用例行为不变
        MOCKER(hcclStreamSynchronize).stubs().will(invoke(+[](HcclRtStream, s32) -> HcclResult {
            return (activeTest_ != nullptr) ? activeTest_->syncRet_ : HCCL_SUCCESS;
        }));
    }

    void TearDown() override
    {
        TestHcommCAdptBase::TearDown();
        activeTest_ = nullptr;
    }

    // D2H 读回是模块内唯一 DEVICE_TO_HOST 拷贝（Init 输出槽读回），为错误分支的关键注入点；
    // H2D 记录参数字节流供 destroy 入参回放断言（设备侧内容无人消费，无需真拷贝）
    static HcclResult
    StubHrtMemSyncCopy(void* dst, uint64_t /*destMax*/, const void* src, uint64_t count, HcclRtMemcpyKind kind)
    {
        auto* test = activeTest_;
        if (kind == HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST) {
            if (test->d2hRet_ == HCCL_SUCCESS && dst != nullptr && count >= sizeof(EpMemTableInitOutput)) {
                (void)std::memcpy(dst, &test->craftedOutput_, sizeof(EpMemTableInitOutput));
            }
            return test->d2hRet_;
        }
        if (src != nullptr && count > 0) {
            const auto* begin = static_cast<const char*>(src);
            test->h2dCopies_.emplace_back(begin, begin + count);
        }
        return HCCL_SUCCESS;
    }

    static HcclResult
    StubAicpuKernelLaunch(rtStream_t, void*, u32, aclrtBinHandle, const std::string& kernelName, bool, u16, void*, u32)
    {
        auto* test = activeTest_;
        test->launchedNames_.push_back(kernelName);
        const auto it = test->launchRetByKernel_.find(kernelName);
        return (it != test->launchRetByKernel_.end()) ? it->second : HCCL_SUCCESS;
    }

    // 统计内容等于 expected 的 size 字节参数 H2D 拷贝次数：destroy 参数为 tablePtr（8B），init 参数为
    // 输出槽设备地址（8B，真 malloc 值），update 参数为 40B 平铺字段，三者内容可区分
    uint32_t CountH2dParamBytes(const void* expected, size_t size) const
    {
        uint32_t hits = 0;
        for (const auto& copy : h2dCopies_) {
            if (copy.size() == size && std::memcmp(copy.data(), expected, size) == 0) {
                ++hits;
            }
        }
        return hits;
    }

    uint32_t CountH2dU64Param(uint64_t value) const { return CountH2dParamBytes(&value, sizeof(value)); }

    static TestEpMemTableLaunch* activeTest_;
    const aclrtBinHandle fakeBinHandle_{reinterpret_cast<aclrtBinHandle>(1)};
    EpMemTableInitOutput craftedOutput_{};
    HcclResult d2hRet_{HCCL_SUCCESS};
    std::map<std::string, HcclResult> launchRetByKernel_{};
    std::vector<std::string> launchedNames_{};
    std::vector<std::vector<char>> h2dCopies_{};
    HcclResult syncRet_{HCCL_SUCCESS}; // hcclStreamSynchronize 桩返回值（场景⑥注入超时）
    uint32_t freeCount_{0};            // hrtFree 桩计数（场景⑥参数块生命周期锚点断言）
};

TestEpMemTableLaunch* TestEpMemTableLaunch::activeTest_ = nullptr;

// 检视意见场景①：launch 失败但 kernel 已建表（读回三元组合法）→ 尽力 destroy 回收已建表后
// 返回原始错误码，保留编排层回滚语义
TEST_F(TestEpMemTableLaunch, Ut_Init_When_LaunchFailsButReadbackValid_Expect_ReclaimThenOriginalRet)
{
    constexpr uint64_t fakeTablePtr = 0x5A5A0000U;
    launchRetByKernel_[EP_MEM_TABLE_INIT_KERNEL] = HCCL_E_RUNTIME;
    craftedOutput_ = EpMemTableInitOutput{fakeTablePtr, DEV_EP_MEM_TABLE_MAGIC_WORD, DEV_EP_MEM_TABLE_VERSION};

    uint64_t outTablePtr = 0;
    EXPECT_EQ(hcomm::LaunchEpMemTableInit(outTablePtr), HCCL_E_RUNTIME);
    EXPECT_EQ(outTablePtr, 0U);

    // 回收时序：init 失败后紧跟一次 destroy kernel 下发
    ASSERT_EQ(launchedNames_.size(), 2U);
    EXPECT_EQ(launchedNames_[0], EP_MEM_TABLE_INIT_KERNEL);
    EXPECT_EQ(launchedNames_[1], EP_MEM_TABLE_DESTROY_KERNEL);
    // destroy 入参携带读回的表地址（经 H2D 参数拷贝回放断言）
    EXPECT_EQ(CountH2dU64Param(fakeTablePtr), 1U);
}

// 检视意见场景②：launch 成功但读回 ABI 校验不过（magic 错）→ 地址已读回则尽力 destroy，
// 返回 HCCL_E_INTERNAL
TEST_F(TestEpMemTableLaunch, Ut_Init_When_ReadbackBadMagic_Expect_DestroyThenInternal)
{
    constexpr uint64_t fakeTablePtr = 0x5A5A0000U;
    craftedOutput_ = EpMemTableInitOutput{fakeTablePtr, 0xDEADBEEFU, DEV_EP_MEM_TABLE_VERSION};

    uint64_t outTablePtr = 0;
    EXPECT_EQ(hcomm::LaunchEpMemTableInit(outTablePtr), HCCL_E_INTERNAL);
    EXPECT_EQ(outTablePtr, 0U);

    ASSERT_EQ(launchedNames_.size(), 2U);
    EXPECT_EQ(launchedNames_[0], EP_MEM_TABLE_INIT_KERNEL);
    EXPECT_EQ(launchedNames_[1], EP_MEM_TABLE_DESTROY_KERNEL);
    EXPECT_EQ(CountH2dU64Param(fakeTablePtr), 1U);
}

// 检视意见场景③：D2H 读回失败 → 表地址不可读，透传错误码且不发起 destroy（盲删半截值有风险）
TEST_F(TestEpMemTableLaunch, Ut_Init_When_D2HReadbackFails_Expect_PassthroughWithoutDestroy)
{
    constexpr uint64_t fakeTablePtr = 0x5A5A0000U;
    d2hRet_ = HCCL_E_MEMORY;

    uint64_t outTablePtr = 0;
    EXPECT_EQ(hcomm::LaunchEpMemTableInit(outTablePtr), HCCL_E_MEMORY);
    EXPECT_EQ(outTablePtr, 0U);

    // 仅 init 一次下发，无 destroy
    ASSERT_EQ(launchedNames_.size(), 1U);
    EXPECT_EQ(launchedNames_[0], EP_MEM_TABLE_INIT_KERNEL);
    EXPECT_EQ(CountH2dU64Param(fakeTablePtr), 0U);
}

// 检视意见场景④：表已就绪的 Reg 同步点（经 SyncUbLocMemToEpTable 守卫放行转发）→ Update kernel
// 正常下发，6 字段平铺为 40 字节 ABI 参数；各字段取非零互异值以捕获错位（op/rsv 均为 0，
// 偏移已由 channel_param_pub.h static_assert 锚定）
TEST_F(TestEpMemTableLaunch, Ut_Sync_When_TableReady_Expect_UpdateKernelWithFullParam)
{
    constexpr uint64_t tablePtr = 0x5A5A1000U;
    constexpr uint64_t addr = 0x12345000U;
    constexpr uint64_t size = 0x2000U;
    constexpr uint32_t tokenId = 0x1111U;
    constexpr uint32_t tokenValue = 0x2222U;
    EXPECT_EQ(
        hcomm::SyncUbLocMemToEpTable(tablePtr, EpMemTableOp::UPSERT, addr, size, tokenId, tokenValue), HCCL_SUCCESS);

    ASSERT_EQ(launchedNames_.size(), 1U);
    EXPECT_EQ(launchedNames_[0], EP_MEM_TABLE_UPDATE_KERNEL);
    ASSERT_EQ(h2dCopies_.size(), 1U);
    const HcclAicpuEpMemTableUpdateRes expected{
        tablePtr, static_cast<uint32_t>(EpMemTableOp::UPSERT), 0U, addr, size, tokenId, tokenValue};
    EXPECT_EQ(CountH2dParamBytes(&expected, sizeof(expected)), 1U);
}

// 检视意见场景⑤：表未初始化（tablePtr==0 守卫）→ 不下发任何 kernel 与参数 H2D，返回 HCCL_E_INTERNAL
TEST_F(TestEpMemTableLaunch, Ut_Sync_When_TablePtrZero_Expect_InternalWithoutLaunch)
{
    EXPECT_EQ(hcomm::SyncUbLocMemToEpTable(0U, EpMemTableOp::UPSERT, 0x12345000U, 0x2000U, 1U, 2U), HCCL_E_INTERNAL);

    EXPECT_TRUE(launchedNames_.empty());
    EXPECT_TRUE(h2dCopies_.empty());
}

// 检视意见场景⑥：sync 超时/失败 → 错误码原样透传、outTablePtr 保持 0；读回三元组全零
// （未证实建表）时不下发 destroy。freeCount==2（paramDev+outDev）显式钉住当前「参数块随
// return 析构」的生命周期语义（生产侧滞留整改意见未采纳）——后续若改为常驻/延迟回收，
// 本断言翻转报警，防止生命周期改动静默通过
TEST_F(TestEpMemTableLaunch, Ut_Init_When_SyncTimeout_Expect_PassthroughAndNoDestroy)
{
    syncRet_ = HCCL_E_TIMEOUT;

    uint64_t outTablePtr = 0;
    EXPECT_EQ(hcomm::LaunchEpMemTableInit(outTablePtr), HCCL_E_TIMEOUT);
    EXPECT_EQ(outTablePtr, 0U);

    // 读回三元组全零 → 未证实建表 → 不发起 destroy（盲删半截值有风险）
    ASSERT_EQ(launchedNames_.size(), 1U);
    EXPECT_EQ(launchedNames_[0], EP_MEM_TABLE_INIT_KERNEL);
    // 生命周期锚点：paramDev + outDev 均随 return 析构（当前语义，见用例头注释）
    EXPECT_EQ(freeCount_, 2U);
}
