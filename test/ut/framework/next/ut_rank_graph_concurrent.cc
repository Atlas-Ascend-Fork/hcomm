/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"
#include "hccl_comm_pub.h"
#include "llt_hccl_stub_rank_graph.h"
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>

/*
 * 950（A5）拓扑查询接口多线程并发 UT：验证同一通信域上多线程并发调用
 * 六个缓存型接口（GetLayers/GetRanksByLayer/GetInstSizeListByLayer/GetLinks/
 * GetTopoInstsByLayer/GetRanksByTopoInst）与天然安全接口时结果正确、无数据竞争。
 * 断言策略：工作线程内只计数不使用gtest宏（gtest断言宏非线程安全），
 * join后由主线程统一断言；每轮调用后立即比对内容，防撕裂/串数据。
 */
class HcclRankGraphConcurrentTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        const char* fakeA5SocName = "Ascend950PR_958b";
        MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeA5SocName));
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

protected:
    void SetUpCommAndGraph(
        std::shared_ptr<hccl::hcclComm>& hcclCommPtr, std::shared_ptr<Hccl::RankGraph>& rankGraphV2, void*& comm,
        HcclResult& ret)
    {
        MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

        bool isDeviceSide{false};
        MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
        MOCKER(IsSupportHCCLV2).stubs().will(returnValue(true));
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
        RankGraphStub rankGraphStub;
        rankGraphV2 = rankGraphStub.Create2PGraph();
        void* commV2 = (void*)0x2000;
        uint32_t rank = 1;
        HcclMem cclBuffer;
        cclBuffer.size = 1;
        cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
        cclBuffer.addr = (void*)0x1000;
        char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
        hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, commName);
        HcclCommConfig config;
        UtInitHcclCommConfig(config);
        config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
        config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
        config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
        unsetenv("HCCL_DFS_CONFIG");
        ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
        CollComm* collComm = hcclCommPtr->GetCollComm();
        comm = static_cast<HcclComm>(hcclCommPtr.get());
    }
};

// 同一通信域多线程并发调用HcclRankGraphGetLinks：结果正确且无数据竞争
TEST_F(HcclRankGraphConcurrentTest, Ut_GetLinks_MultiThread_SameComm_Expect_NoRaceAndResultCorrect)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm = nullptr;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    // 单线程预采集期望值（正确用法：调用后立即拷贝）
    uint32_t expectLinkNum = 0;
    CommLink* links = nullptr;
    ASSERT_EQ(HcclRankGraphGetLinks(comm, 0, 0, 1, &links, &expectLinkNum), HCCL_SUCCESS);
    ASSERT_EQ(expectLinkNum, 1u);
    const uint32_t expectHop = links[0].linkAttr.hop;

    constexpr uint32_t THREAD_N = 8;
    constexpr uint32_t LOOP = 2000;
    std::vector<uint32_t> failCnt(THREAD_N, 0);
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        threads.emplace_back([&, t]() {
            // 微秒级错峰扰动，放大竞态窗口
            std::this_thread::sleep_for(std::chrono::microseconds(t * 13U));
            for (uint32_t i = 0; i < LOOP; ++i) {
                CommLink* curLinks = nullptr;
                uint32_t curNum = 0;
                if (HcclRankGraphGetLinks(comm, 0, 0, 1, &curLinks, &curNum) != HCCL_SUCCESS || curNum != 1u
                    || curLinks[0].linkAttr.hop != expectHop) {
                    failCnt[t]++; // 内容撕裂/串数据即失败（线程内只计数，join后统一断言）
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        EXPECT_EQ(failCnt[t], 0u);
    }
}

// 同一通信域多线程并发调用HcclRankGraphGetLayers：结果正确且无数据竞争
TEST_F(HcclRankGraphConcurrentTest, Ut_GetLayers_MultiThread_SameComm_Expect_NoRaceAndResultCorrect)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm = nullptr;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    uint32_t expectNum = 0;
    uint32_t* netLayers = nullptr;
    ASSERT_EQ(HcclRankGraphGetLayers(comm, &netLayers, &expectNum), HCCL_SUCCESS);
    const std::vector<uint32_t> expectLayers(netLayers, netLayers + expectNum);

    constexpr uint32_t THREAD_N = 8;
    constexpr uint32_t LOOP = 2000;
    std::vector<uint32_t> failCnt(THREAD_N, 0);
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        threads.emplace_back([&, t]() {
            std::this_thread::sleep_for(std::chrono::microseconds(t * 13U));
            for (uint32_t i = 0; i < LOOP; ++i) {
                uint32_t* curLayers = nullptr;
                uint32_t curNum = 0;
                if (HcclRankGraphGetLayers(comm, &curLayers, &curNum) != HCCL_SUCCESS || curNum != expectNum
                    || !std::equal(expectLayers.begin(), expectLayers.end(), curLayers)) {
                    failCnt[t]++;
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        EXPECT_EQ(failCnt[t], 0u);
    }
}

// 同一通信域多线程并发调用HcclRankGraphGetRanksByLayer：结果正确且无数据竞争
TEST_F(HcclRankGraphConcurrentTest, Ut_GetRanksByLayer_MultiThread_SameComm_Expect_NoRaceAndResultCorrect)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm = nullptr;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    uint32_t expectNum = 0;
    uint32_t* ranks = nullptr;
    ASSERT_EQ(HcclRankGraphGetRanksByLayer(comm, 0, &ranks, &expectNum), HCCL_SUCCESS);
    const std::vector<uint32_t> expectRanks(ranks, ranks + expectNum);

    constexpr uint32_t THREAD_N = 8;
    constexpr uint32_t LOOP = 2000;
    std::vector<uint32_t> failCnt(THREAD_N, 0);
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        threads.emplace_back([&, t]() {
            std::this_thread::sleep_for(std::chrono::microseconds(t * 13U));
            for (uint32_t i = 0; i < LOOP; ++i) {
                uint32_t* curRanks = nullptr;
                uint32_t curNum = 0;
                if (HcclRankGraphGetRanksByLayer(comm, 0, &curRanks, &curNum) != HCCL_SUCCESS || curNum != expectNum
                    || !std::equal(expectRanks.begin(), expectRanks.end(), curRanks)) {
                    failCnt[t]++;
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        EXPECT_EQ(failCnt[t], 0u);
    }
}

// 同一通信域多线程并发调用HcclRankGraphGetInstSizeListByLayer：结果正确且无数据竞争
TEST_F(HcclRankGraphConcurrentTest, Ut_GetInstSizeListByLayer_MultiThread_SameComm_Expect_NoRaceAndResultCorrect)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm = nullptr;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    uint32_t expectNum = 0;
    uint32_t* instSizeList = nullptr;
    ASSERT_EQ(HcclRankGraphGetInstSizeListByLayer(comm, 0, &instSizeList, &expectNum), HCCL_SUCCESS);
    const std::vector<uint32_t> expectList(instSizeList, instSizeList + expectNum);

    constexpr uint32_t THREAD_N = 8;
    constexpr uint32_t LOOP = 2000;
    std::vector<uint32_t> failCnt(THREAD_N, 0);
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        threads.emplace_back([&, t]() {
            std::this_thread::sleep_for(std::chrono::microseconds(t * 13U));
            for (uint32_t i = 0; i < LOOP; ++i) {
                uint32_t* curList = nullptr;
                uint32_t curNum = 0;
                if (HcclRankGraphGetInstSizeListByLayer(comm, 0, &curList, &curNum) != HCCL_SUCCESS
                    || curNum != expectNum || !std::equal(expectList.begin(), expectList.end(), curList)) {
                    failCnt[t]++;
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        EXPECT_EQ(failCnt[t], 0u);
    }
}

// 同一通信域多线程并发调用HcclRankGraphGetTopoInstsByLayer：结果正确且无数据竞争
TEST_F(HcclRankGraphConcurrentTest, Ut_GetTopoInstsByLayer_MultiThread_SameComm_Expect_NoRaceAndResultCorrect)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm = nullptr;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    uint32_t expectNum = 0;
    uint32_t* topoInsts = nullptr;
    ASSERT_EQ(HcclRankGraphGetTopoInstsByLayer(comm, 0, &topoInsts, &expectNum), HCCL_SUCCESS);
    const std::vector<uint32_t> expectInsts(topoInsts, topoInsts + expectNum);

    constexpr uint32_t THREAD_N = 8;
    constexpr uint32_t LOOP = 2000;
    std::vector<uint32_t> failCnt(THREAD_N, 0);
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        threads.emplace_back([&, t]() {
            std::this_thread::sleep_for(std::chrono::microseconds(t * 13U));
            for (uint32_t i = 0; i < LOOP; ++i) {
                uint32_t* curInsts = nullptr;
                uint32_t curNum = 0;
                if (HcclRankGraphGetTopoInstsByLayer(comm, 0, &curInsts, &curNum) != HCCL_SUCCESS || curNum != expectNum
                    || !std::equal(expectInsts.begin(), expectInsts.end(), curInsts)) {
                    failCnt[t]++;
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        EXPECT_EQ(failCnt[t], 0u);
    }
}

// 同一通信域多线程并发调用HcclRankGraphGetRanksByTopoInst：结果正确且无数据竞争
TEST_F(HcclRankGraphConcurrentTest, Ut_GetRanksByTopoInst_MultiThread_SameComm_Expect_NoRaceAndResultCorrect)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm = nullptr;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    uint32_t expectNum = 0;
    uint32_t* ranks = nullptr;
    ASSERT_EQ(HcclRankGraphGetRanksByTopoInst(comm, 0, 0, &ranks, &expectNum), HCCL_SUCCESS);
    const std::vector<uint32_t> expectRanks(ranks, ranks + expectNum);

    constexpr uint32_t THREAD_N = 8;
    constexpr uint32_t LOOP = 2000;
    std::vector<uint32_t> failCnt(THREAD_N, 0);
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        threads.emplace_back([&, t]() {
            std::this_thread::sleep_for(std::chrono::microseconds(t * 13U));
            for (uint32_t i = 0; i < LOOP; ++i) {
                uint32_t* curRanks = nullptr;
                uint32_t curNum = 0;
                if (HcclRankGraphGetRanksByTopoInst(comm, 0, 0, &curRanks, &curNum) != HCCL_SUCCESS
                    || curNum != expectNum || !std::equal(expectRanks.begin(), expectRanks.end(), curRanks)) {
                    failCnt[t]++;
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        EXPECT_EQ(failCnt[t], 0u);
    }
}

// 六个缓存型接口在同一线程内交错混合并发：各接口输出缓冲相互独立，无串扰
TEST_F(HcclRankGraphConcurrentTest, Ut_MixedQuery_MultiThread_SameComm_Expect_NoRaceAndResultCorrect)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm = nullptr;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    uint32_t expectNum = 0;
    uint32_t* buf = nullptr;
    ASSERT_EQ(HcclRankGraphGetLayers(comm, &buf, &expectNum), HCCL_SUCCESS);
    const std::vector<uint32_t> expectLayers(buf, buf + expectNum);
    ASSERT_EQ(HcclRankGraphGetRanksByLayer(comm, 0, &buf, &expectNum), HCCL_SUCCESS);
    const std::vector<uint32_t> expectRanks(buf, buf + expectNum);
    ASSERT_EQ(HcclRankGraphGetInstSizeListByLayer(comm, 0, &buf, &expectNum), HCCL_SUCCESS);
    const std::vector<uint32_t> expectSizes(buf, buf + expectNum);
    uint32_t expectLinkNum = 0;
    CommLink* links = nullptr;
    ASSERT_EQ(HcclRankGraphGetLinks(comm, 0, 0, 1, &links, &expectLinkNum), HCCL_SUCCESS);
    const uint32_t expectHop = links[0].linkAttr.hop;
    ASSERT_EQ(HcclRankGraphGetTopoInstsByLayer(comm, 0, &buf, &expectNum), HCCL_SUCCESS);
    const std::vector<uint32_t> expectInsts(buf, buf + expectNum);
    ASSERT_EQ(HcclRankGraphGetRanksByTopoInst(comm, 0, 0, &buf, &expectNum), HCCL_SUCCESS);
    const std::vector<uint32_t> expectByInst(buf, buf + expectNum);

    constexpr uint32_t THREAD_N = 8;
    constexpr uint32_t LOOP = 1000;
    std::vector<uint32_t> failCnt(THREAD_N, 0);
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        threads.emplace_back([&, t]() {
            std::this_thread::sleep_for(std::chrono::microseconds(t * 13U));
            for (uint32_t i = 0; i < LOOP; ++i) {
                uint32_t curNum = 0;
                uint32_t* cur = nullptr;
                CommLink* curLinks = nullptr;
                bool ok = HcclRankGraphGetLayers(comm, &cur, &curNum) == HCCL_SUCCESS && curNum == expectLayers.size()
                          && std::equal(expectLayers.begin(), expectLayers.end(), cur);
                ok = ok && HcclRankGraphGetRanksByLayer(comm, 0, &cur, &curNum) == HCCL_SUCCESS
                     && curNum == expectRanks.size() && std::equal(expectRanks.begin(), expectRanks.end(), cur);
                ok = ok && HcclRankGraphGetInstSizeListByLayer(comm, 0, &cur, &curNum) == HCCL_SUCCESS
                     && curNum == expectSizes.size() && std::equal(expectSizes.begin(), expectSizes.end(), cur);
                ok = ok && HcclRankGraphGetLinks(comm, 0, 0, 1, &curLinks, &curNum) == HCCL_SUCCESS && curNum == 1u
                     && curLinks[0].linkAttr.hop == expectHop;
                ok = ok && HcclRankGraphGetTopoInstsByLayer(comm, 0, &cur, &curNum) == HCCL_SUCCESS
                     && curNum == expectInsts.size() && std::equal(expectInsts.begin(), expectInsts.end(), cur);
                ok = ok && HcclRankGraphGetRanksByTopoInst(comm, 0, 0, &cur, &curNum) == HCCL_SUCCESS
                     && curNum == expectByInst.size() && std::equal(expectByInst.begin(), expectByInst.end(), cur);
                if (!ok) {
                    failCnt[t]++;
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        EXPECT_EQ(failCnt[t], 0u);
    }
}

// 八个天然安全接口并发抽查：只读查询无共享可变状态
TEST_F(HcclRankGraphConcurrentTest, Ut_NaturallySafeQuery_MultiThread_SameComm_Expect_NoRaceAndResultCorrect)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm = nullptr;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    uint32_t expectRankId = 0;
    uint32_t expectRankSize = 0;
    uint32_t expectRankNum = 0;
    uint32_t expectEpNum = 0;
    CommTopo expectTopoTypeByLayer{};
    CommTopo expectTopoType{};
    ASSERT_EQ(HcclGetRankId(comm, &expectRankId), HCCL_SUCCESS);
    ASSERT_EQ(HcclGetRankSize(comm, &expectRankSize), HCCL_SUCCESS);
    ASSERT_EQ(HcclRankGraphGetRankSizeByLayer(comm, 0, &expectRankNum), HCCL_SUCCESS);
    ASSERT_EQ(HcclRankGraphGetTopoTypeByLayer(comm, 0, &expectTopoTypeByLayer), HCCL_SUCCESS);
    ASSERT_EQ(HcclRankGraphGetTopoType(comm, 0, 0, &expectTopoType), HCCL_SUCCESS);
    ASSERT_EQ(HcclRankGraphGetEndpointNum(comm, 0, 0, &expectEpNum), HCCL_SUCCESS);
    ASSERT_GE(expectEpNum, 1u);
    const uint32_t descNum = expectEpNum;
    std::vector<EndpointDesc> expectDescs(descNum);
    uint32_t outNum = descNum;
    ASSERT_EQ(HcclRankGraphGetEndpointDesc(comm, 0, 0, &outNum, expectDescs.data()), HCCL_SUCCESS);
    EndpointAttrBwCoeff expectCoeff{};
    ASSERT_EQ(
        HcclRankGraphGetEndpointInfo(
            comm, 0, &expectDescs[0], ENDPOINT_ATTR_BW_COEFF, sizeof(expectCoeff), &expectCoeff),
        HCCL_SUCCESS);

    constexpr uint32_t THREAD_N = 8;
    constexpr uint32_t LOOP = 2000;
    std::vector<uint32_t> failCnt(THREAD_N, 0);
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        threads.emplace_back([&, t]() {
            std::this_thread::sleep_for(std::chrono::microseconds(t * 13U));
            for (uint32_t i = 0; i < LOOP; ++i) {
                uint32_t rankId = 0;
                uint32_t rankSize = 0;
                uint32_t rankNum = 0;
                uint32_t epNum = 0;
                uint32_t curDescNum = descNum;
                std::vector<EndpointDesc> curDescs(descNum); // 每线程独占输出数组，避免与读侧线程共用
                CommTopo typeByLayer{};
                CommTopo type{};
                EndpointAttrBwCoeff coeff{};
                bool ok = HcclGetRankId(comm, &rankId) == HCCL_SUCCESS && rankId == expectRankId;
                ok = ok && HcclGetRankSize(comm, &rankSize) == HCCL_SUCCESS && rankSize == expectRankSize;
                ok = ok && HcclRankGraphGetRankSizeByLayer(comm, 0, &rankNum) == HCCL_SUCCESS
                     && rankNum == expectRankNum;
                ok = ok && HcclRankGraphGetTopoTypeByLayer(comm, 0, &typeByLayer) == HCCL_SUCCESS
                     && typeByLayer == expectTopoTypeByLayer;
                ok = ok && HcclRankGraphGetTopoType(comm, 0, 0, &type) == HCCL_SUCCESS && type == expectTopoType;
                ok = ok && HcclRankGraphGetEndpointNum(comm, 0, 0, &epNum) == HCCL_SUCCESS && epNum == expectEpNum;
                ok = ok && HcclRankGraphGetEndpointDesc(comm, 0, 0, &curDescNum, curDescs.data()) == HCCL_SUCCESS
                     && curDescNum == descNum;
                ok = ok
                     && HcclRankGraphGetEndpointInfo(
                            comm, 0, &curDescs[0], ENDPOINT_ATTR_BW_COEFF, sizeof(coeff), &coeff)
                            == HCCL_SUCCESS;
                if (!ok) {
                    failCnt[t]++;
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    for (uint32_t t = 0; t < THREAD_N; ++t) {
        EXPECT_EQ(failCnt[t], 0u);
    }
}

// 同线程重复调用契约：重复调用使前次返回指针失效，但每次结果独立完整（不累积）
TEST_F(HcclRankGraphConcurrentTest, Ut_GetTopoInstsByLayer_SameThread_RepeatedCall_Expect_ResultConsistent)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm = nullptr;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    uint32_t* first = nullptr;
    uint32_t firstNum = 0;
    ASSERT_EQ(HcclRankGraphGetTopoInstsByLayer(comm, 0, &first, &firstNum), HCCL_SUCCESS);
    ASSERT_GE(firstNum, 1u);
    // 正确用法：调用后立即拷贝
    const std::vector<uint32_t> firstCopy(first, first + firstNum);

    // 重复调用：前次返回指针可能已失效（输出缓冲复用），本次结果须与首次一致
    for (uint32_t i = 0; i < 100; ++i) {
        uint32_t* cur = nullptr;
        uint32_t curNum = 0;
        ASSERT_EQ(HcclRankGraphGetTopoInstsByLayer(comm, 0, &cur, &curNum), HCCL_SUCCESS);
        EXPECT_EQ(curNum, firstNum);
        const std::vector<uint32_t> curCopy(cur, cur + curNum);
        EXPECT_EQ(curCopy, firstCopy);
    }
}
