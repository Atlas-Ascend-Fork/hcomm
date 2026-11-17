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
#include <cstring>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <vector>

#include "hccp_ctx.h"
#include "hccp_tlv.h"
#include "device_spec/device_spec_types.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"
#include "sqlite/simulation_storage_test_helper.h"
#include "storage/storage_runtime.h"
#include "storage/table_access.h"

// 被测实现编译进本目标（hccp_stub.cc 为 extern "C" 定义）
extern "C" {
int GetCcuMemInfo(struct TlvMsg* sendMsg, struct TlvMsg* recvMsg);
// 链接占位：hccp_stub.cc 引用 hccp_ccu_stub.cc 的 SimRaCustomChannel（本用例
// 不触达该路径；生产中由 hccl_proxy_level2 一并链接 ccu_stub）。
int SimRaCustomChannel(void* tlvHandle, struct TlvMsg* sendMsg, struct TlvMsg* recvMsg);
}

// 占位实现：仅满足链接，不参与任何用例路径。
int SimRaCustomChannel(void* tlvHandle, struct TlvMsg* sendMsg, struct TlvMsg* recvMsg)
{
    (void)tlvHandle;
    (void)sendMsg;
    (void)recvMsg;
    return -1;
}

// db_sim_runner_common.cc 的全局当前 server key（GetCurServerId 短路入口）
extern uint64_t g_cur_server_key;

namespace {

constexpr char kRunnerDb[] = "/tmp/hccp_ccu_mem_info_stub_runner.db";
constexpr char kOpDataDb[] = "/tmp/hccp_ccu_mem_info_stub_opdata.db";

// hcomm CcuMemTypeBitmap 位（orion_adapter_hccp.h），与 hccp_stub.cc 内常量一致
constexpr uint64_t kMemTypeBitGsa = 1ULL << 1;
constexpr uint64_t kMemTypeBitXn = 1ULL << 2;
constexpr uint64_t kMemTypeBitCke = 1ULL << 3;
// legacy 950 全量 bitmap（GetMemTypeVector 类型并集；本地 hcomm 18 类型）
constexpr uint64_t kLegacyFullMemTypeBitmap = 0xF0007FFFULL;
// 线上 hcomm 版本的全量 bitmap：GetMemTypeVector 去掉了 LOOP_CKE（bit4），17 类型。
// checker 实测故障（2026-11-16）：写死 18 位掩码判定把它误判成按位请求、
// 丢掉 72M 窗口条目 → legacy ccuRmaBufferMap 匹配失败 → CreateLoopChannels 失败。
constexpr uint64_t kLegacyFullMemTypeBitmapNoLoopCke = 0xF0007FEFULL;

constexpr uint64_t kDie0Base = 0x8e00000000ULL;
constexpr uint64_t kDie1Base = 0x8e40000000ULL;
// hcomm 权威值 CCU_V1_V2_RESOURCE_SIZE（72 MiB，spec.resource_space_size 口径）
constexpr uint32_t kResourceSpaceSize = 0x4800000U;

// 应答解码布局：与 hcomm orion_adapter_hccp.h（PR 6186 后）一致
struct ccu_mem_info_test {
    unsigned int long long mem_va;
    unsigned int mem_size;
    unsigned int resv[1];
};

struct ccu_mem_rsp_test {
    unsigned int die_id;
    unsigned int num;
    struct ccu_mem_info_test list[64U];
};

// A6/960 风格 region 表（offset 升序，与 asset 的 ascend960 yaml 一致）
struct RegionSeed {
    sim::devspec::CcuResType type;
    uint64_t offset;
    uint64_t size;
};

constexpr RegionSeed kA6Regions[] = {
    {sim::devspec::CcuResType::INSTR, 0x0, 0x100000},       {sim::devspec::CcuResType::XN, 0x100000, 0x40000},
    {sim::devspec::CcuResType::CKE, 0x140000, 0x8000},      {sim::devspec::CcuResType::PFE, 0x148000, 0x8000},
    {sim::devspec::CcuResType::CHANNEL, 0x150000, 0x20000}, {sim::devspec::CcuResType::JETTY, 0x170000, 0x40000},
    {sim::devspec::CcuResType::MISSION, 0x1b0000, 0x40000}, {sim::devspec::CcuResType::LOOP, 0x1f0000, 0x40000},
};

// 按 CcuResType 查 region 种子（UT 断言的期望值来源）
const RegionSeed* FindRegionSeed(const sim::devspec::CcuResType type)
{
    for (const auto& r : kA6Regions) {
        if (r.type == type) {
            return &r;
        }
    }
    return nullptr;
}

} // namespace

class HccpCcuMemInfoStubTest : public testing::Test {
protected:
    uint64_t serverId{0};
    uint64_t deviceId{0};
    uint64_t ctxId{0};

    /**
     * 每个用例以干净 DB 重建会话，并造齐 GetCcuMemInfo 依赖的四类数据。
     */
    void SetUp() override
    {
        ASSERT_TRUE(runnerdb_test::ResetTestSession(kRunnerDb, kOpDataDb));
        ASSERT_NO_FATAL_FAILURE(SeedTopology());
        ASSERT_NO_FATAL_FAILURE(SeedCcuBaseAddrs());
        ASSERT_NO_FATAL_FAILURE(SeedDeviceSpec());
        ASSERT_NO_FATAL_FAILURE(BindRunnerCtx());
    }

    /**
     * 清掉全局 server key 并删除测试 DB 文件（含 WAL/SHM）。
     */
    void TearDown() override
    {
        g_cur_server_key = 0;
        runnerdb_test::CleanUpDatabases(kRunnerDb, kOpDataDb);
    }

    /**
     * 建 server/host/device 拓扑行（GetCurrDeviceId 的 Device 解析路径）。
     */
    void SeedTopology()
    {
        sim::runtime::Server server{};
        server.pod_id = 100;
        serverId = runnerdb_test::InsertRecord<sim::runtime::Server>(server);
        ASSERT_NE(serverId, 0U);
        sim::runtime::Host host{};
        host.server_id = serverId;
        host.arch = 1;
        ASSERT_NE(runnerdb_test::InsertRecord<sim::runtime::Host>(host), 0U);
        sim::runtime::Device device{};
        device.server_id = serverId;
        device.logic_id = 0;
        device.physical_id = 0;
        deviceId = runnerdb_test::InsertRecord<sim::runtime::Device>(device);
        ASSERT_NE(deviceId, 0U);
    }

    /**
     * 写两个 die 的 Ccu.resource_addr（GetCcuResourceBaseAddr 的 DB 优先路径）。
     */
    void SeedCcuBaseAddrs()
    {
        for (uint8_t die = 0; die < 2; die++) {
            sim::runtime::Ccu ccu{};
            ccu.device_id = deviceId;
            ccu.die_id = die;
            ccu.resource_addr = (die == 0) ? kDie0Base : kDie1Base;
            ASSERT_NE(runnerdb_test::InsertRecord<sim::runtime::Ccu>(ccu), 0U);
        }
    }

    /**
     * 建 device 规格 + A6 regions + device 绑定（GetDeviceSpecByDeviceKey/Regions 路径）。
     */
    void SeedDeviceSpec()
    {
        sim::runtime::DeviceSpec spec{};
        std::strncpy(spec.name, "ut-a6", sizeof(spec.name) - 1);
        spec.ccu_version = static_cast<uint8_t>(sim::devspec::CcuVersion::CCU_V2);
        spec.die_num = 2;
        spec.resource_space_size = kResourceSpaceSize;
        const uint64_t specId = runnerdb_test::InsertRecord<sim::runtime::DeviceSpec>(spec);
        ASSERT_NE(specId, 0U);
        for (const auto& seed : kA6Regions) {
            sim::runtime::DeviceSpecRegion region{};
            region.spec_id = specId;
            region.type = static_cast<uint16_t>(seed.type);
            region.offset = seed.offset;
            region.size = seed.size;
            region.entry_size = 8;
            region.count = 1;
            ASSERT_NE(runnerdb_test::InsertRecord<sim::runtime::DeviceSpecRegion>(region), 0U);
        }
        sim::runtime::DeviceSpecBinding binding{};
        binding.device_id = deviceId;
        binding.spec_id = specId;
        ASSERT_NE(runnerdb_test::InsertRecord<sim::runtime::DeviceSpecBinding>(binding), 0U);
    }

    /**
     * 建 runner/ctx TLS，把当前上下文绑定到本设备（GetCurrDeviceId 记忆化路径）。
     */
    void BindRunnerCtx()
    {
        g_cur_server_key = serverId;
        sim::runtime::Runner runner{};
        ASSERT_TRUE(sim::runtime::GetCurrRunnerTls(serverId, runner));
        sim::runtime::Context context{};
        context.device_id = deviceId;
        context.is_default = 1;
        context.ref_cnt = 1;
        ctxId = runnerdb_test::InsertRecord<sim::runtime::Context>(context);
        ASSERT_NE(ctxId, 0U);
        ASSERT_TRUE(sim::runtime::SetCurrCtxTls(ctxId));
    }

    /**
     * 以指定 die/bitmap 调用被测桩并取回应答。
     *
     * @param dieId         请求 die
     * @param memTypeBitmap 请求类型位图（CcuMemReq.memTypeBitmap）
     * @param ret           输出：被测桩返回值
     * @return 应答结构（recv buffer 的拷贝）
     */
    ccu_mem_rsp_test CallGetCcuMemInfo(const uint32_t dieId, const uint64_t memTypeBitmap, int& ret)
    {
        CcuMemReq req{};
        req.udieIdx = dieId;
        req.memTypeBitmap = memTypeBitmap;
        struct TlvMsg sendMsg {};
        sendMsg.type = 0;
        sendMsg.length = sizeof(req);
        sendMsg.data = reinterpret_cast<char*>(&req);

        ccu_mem_rsp_test rsp{};
        struct TlvMsg recvMsg {};
        recvMsg.type = 0;
        recvMsg.length = sizeof(rsp);
        recvMsg.data = reinterpret_cast<char*>(&rsp);

        ret = GetCcuMemInfo(&sendMsg, &recvMsg);
        return rsp;
    }

    /**
     * 断言应答段等于 spec region 期望的 (mem_va, mem_size)。
     *
     * @param info  应答段
     * @param base  die 基址（mem_va = base + region.offset）
     * @param seed  region 期望
     */
    static void ExpectSegmentEq(const ccu_mem_info_test& info, const uint64_t base, const RegionSeed& seed)
    {
        EXPECT_EQ(info.mem_va, base + seed.offset);
        EXPECT_EQ(info.mem_size, static_cast<unsigned int>(seed.size));
    }
};

// 用例 1：960 开源流程主合同——XN|CKE 按位请求只回 2 段、位序 XN→[0]/CKE→[1]、无窗口条目
TEST_F(HccpCcuMemInfoStubTest, FilteredXnCke_When_A6Spec_Expect_TwoSegsInBitmapOrder)
{
    int ret = -1;
    const auto rsp = CallGetCcuMemInfo(0, kMemTypeBitXn | kMemTypeBitCke, ret);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(rsp.die_id, 0U);
    ASSERT_EQ(rsp.num, 2U);
    ExpectSegmentEq(rsp.list[0], kDie0Base, *FindRegionSeed(sim::devspec::CcuResType::XN));
    ExpectSegmentEq(rsp.list[1], kDie0Base, *FindRegionSeed(sim::devspec::CcuResType::CKE));
}

// 用例 2：单段请求（仅 XN）——num==1
TEST_F(HccpCcuMemInfoStubTest, FilteredXnOnly_When_A6Spec_Expect_SingleSeg)
{
    int ret = -1;
    const auto rsp = CallGetCcuMemInfo(0, kMemTypeBitXn, ret);

    EXPECT_EQ(ret, 0);
    ASSERT_EQ(rsp.num, 1U);
    ExpectSegmentEq(rsp.list[0], kDie0Base, *FindRegionSeed(sim::devspec::CcuResType::XN));
}

// 用例 3：请求 spec 未建模的类型（A6 无 GSA）——静默跳过、不伪造段
TEST_F(HccpCcuMemInfoStubTest, FilteredGsa_When_NoGsaRegion_Expect_ZeroSeg)
{
    int ret = -1;
    const auto rsp = CallGetCcuMemInfo(0, kMemTypeBitGsa, ret);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(rsp.num, 0U);
}

// 用例 4：legacy 950 全量 bitmap——保持"72M 窗口 + 全部 regions"旧应答
TEST_F(HccpCcuMemInfoStubTest, LegacyFullBitmap_When_A6Spec_Expect_WindowPlusAllRegions)
{
    int ret = -1;
    const auto rsp = CallGetCcuMemInfo(0, kLegacyFullMemTypeBitmap, ret);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(rsp.die_id, 0U);
    ASSERT_EQ(rsp.num, 1U + std::size(kA6Regions));
    // 窗口条目：legacy CreateCcuRmaBuffer 以 memVa==resourceAddr 等值匹配建 ccuRmaBufferMap
    EXPECT_EQ(rsp.list[0].mem_va, kDie0Base);
    EXPECT_EQ(rsp.list[0].mem_size, kResourceSpaceSize);
    for (size_t i = 0; i < std::size(kA6Regions); i++) {
        ExpectSegmentEq(rsp.list[i + 1], kDie0Base, kA6Regions[i]);
    }
}

// 用例 5：全量 bitmap 超集（附加未定义高位）——包含判定，仍走 legacy 分支
TEST_F(HccpCcuMemInfoStubTest, LegacyFullBitmapSuperset_When_ExtraHighBits_Expect_LegacyRsp)
{
    int ret = -1;
    const auto rsp = CallGetCcuMemInfo(0, kLegacyFullMemTypeBitmap | (1ULL << 40), ret);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(rsp.num, 1U + std::size(kA6Regions));
    EXPECT_EQ(rsp.list[0].mem_va, kDie0Base);
}

// 用例 5b：线上 hcomm 的全量 bitmap（LOOP_CKE 已删，=0xf00007fef）——
// 含 INS 位即全量枚举，不得因类型位集合变化被误判成按位请求（checker 故障回归）
TEST_F(HccpCcuMemInfoStubTest, LegacyFullBitmapNoLoopCke_When_InstalledHcommMask_Expect_LegacyRsp)
{
    int ret = -1;
    const auto rsp = CallGetCcuMemInfo(0, kLegacyFullMemTypeBitmapNoLoopCke, ret);

    EXPECT_EQ(ret, 0);
    ASSERT_EQ(rsp.num, 1U + std::size(kA6Regions));
    EXPECT_EQ(rsp.list[0].mem_va, kDie0Base);
    EXPECT_EQ(rsp.list[0].mem_size, kResourceSpaceSize);
}

// 用例 5c：bitmap 为 0——兼容旧调用方（旧实现不看 bitmap，恒返回全量）
TEST_F(HccpCcuMemInfoStubTest, ZeroBitmap_When_LegacyCaller_Expect_LegacyRsp)
{
    int ret = -1;
    const auto rsp = CallGetCcuMemInfo(0, 0, ret);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(rsp.num, 1U + std::size(kA6Regions));
    EXPECT_EQ(rsp.list[0].mem_va, kDie0Base);
}

// 用例 6：die1 请求——段基址取 die1 的 resource_addr，die_id 回显
TEST_F(HccpCcuMemInfoStubTest, FilteredXnCke_When_Die1_Expect_Die1BaseAddr)
{
    int ret = -1;
    const auto rsp = CallGetCcuMemInfo(1, kMemTypeBitXn | kMemTypeBitCke, ret);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(rsp.die_id, 1U);
    ASSERT_EQ(rsp.num, 2U);
    ExpectSegmentEq(rsp.list[0], kDie1Base, *FindRegionSeed(sim::devspec::CcuResType::XN));
    ExpectSegmentEq(rsp.list[1], kDie1Base, *FindRegionSeed(sim::devspec::CcuResType::CKE));
}

// 用例 7：无当前 ctx——返回 -1 且不应答（num 保持调用方初值，不写脏数据）
TEST_F(HccpCcuMemInfoStubTest, NoCurrentCtx_When_CtxZero_Expect_ReturnMinusOne)
{
    ASSERT_TRUE(sim::runtime::SetCurrCtxTls(0));

    int ret = 0;
    const auto rsp = CallGetCcuMemInfo(0, kMemTypeBitXn | kMemTypeBitCke, ret);

    EXPECT_EQ(ret, -1);
    EXPECT_EQ(rsp.num, 0U);
}
