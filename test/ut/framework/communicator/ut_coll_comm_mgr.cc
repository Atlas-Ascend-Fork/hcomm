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
#include "coll_comm_mgr.h"
#include "hcomm_c_adpt.h"
#include "common/result_defs_pub.h"
#include "hcom_common.h"
#include "llt_dca_stub.h"

extern thread_local s32 g_hcclDeviceId;

// InitBaseCommRes 为 HcommResMgrInit 的透传封装，捕获传入 devId 以断言透传正确
static u32 g_ut_capturedHcommResMgrInitDevId = 0xFFFFFFFFu;
static HcommResult stub_CollCommMgrTest_HcommResMgrInit(uint32_t devPhyId)
{
    g_ut_capturedHcommResMgrInitDevId = devPhyId;
    return static_cast<HcommResult>(HCOMM_SUCCESS);
}

// 析构顺序问题：跟踪 GetInstance() 是否调用了 HcommResMgrInit
static bool g_ut_hcommResMgrInitCalledInGetInstance = false;
static HcommResult stub_DestructionOrder_HcommResMgrInit(uint32_t devPhyId)
{
    g_ut_hcommResMgrInitCalledInGetInstance = true;
    return static_cast<HcommResult>(HCOMM_SUCCESS);
}

class CollCommMgrTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        // Reset all opHcomInfos_ slots to clean state
        for (u32 i = 0; i <= MAX_MODULE_DEVICE_NUM; i++) {
            CollCommMgr::GetInstance().LegacyGetOpHcomInfo(i).isUsed = false;
        }
        g_hcclDeviceId = INVALID_INT;
    }

    void TearDown() override
    {
        // Clean up all slots
        for (u32 i = 0; i <= MAX_MODULE_DEVICE_NUM; i++) {
            CollCommMgr::GetInstance().LegacyGetOpHcomInfo(i).isUsed = false;
        }
        g_hcclDeviceId = INVALID_INT;
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

// ===== LegacyGetOpHcomInfo =====

TEST_F(CollCommMgrTest, Ut_LegacyGetOpHcomInfo_When_DevIdNormal_Expect_ReturnCorrespondingSlot)
{
    u32 devId = 3;
    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetOpHcomInfo(devId);
    HcclOpInfoCtx& expectedSlot = CollCommMgr::GetInstance().LegacyGetOpHcomInfo(devId);
    EXPECT_EQ(&info, &expectedSlot);
    EXPECT_EQ(info.isUsed, false);
}

TEST_F(CollCommMgrTest, Ut_LegacyGetOpHcomInfo_When_DevIdOutOfRange_Expect_ClampToBackupSlot)
{
    u32 outOfRangeDevId = MAX_MODULE_DEVICE_NUM + 10;
    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetOpHcomInfo(outOfRangeDevId);
    HcclOpInfoCtx& backupSlot = CollCommMgr::GetInstance().LegacyGetOpHcomInfo(MAX_MODULE_DEVICE_NUM);
    EXPECT_EQ(&info, &backupSlot);
}

// ===== LegacyGetHcclExistDeviceOpInfoCtx =====

TEST_F(CollCommMgrTest, Ut_LegacyGetHcclExistDeviceOpInfoCtx_When_SlotAlreadyUsed_Expect_DirectReturnDevIdUnchanged)
{
    s32 devId = 2;
    CollCommMgr::GetInstance().LegacyGetOpHcomInfo(devId).isUsed = true;
    s32 originalDevId = devId;
    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetHcclExistDeviceOpInfoCtx(devId);
    EXPECT_EQ(devId, originalDevId);
    EXPECT_EQ(info.isUsed, true);
}

TEST_F(CollCommMgrTest, Ut_LegacyGetHcclExistDeviceOpInfoCtx_When_SlotNotUsedAndBackupUsed_Expect_FallbackToBackup)
{
    s32 devId = 3;
    CollCommMgr::GetInstance().LegacyGetOpHcomInfo(MAX_MODULE_DEVICE_NUM).isUsed = true;
    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetHcclExistDeviceOpInfoCtx(devId);
    // 函数内通过 HcclSetThreadDeviceId 写回 backup 槽索引到 g_hcclDeviceId
    HcclOpInfoCtx& backupSlot = CollCommMgr::GetInstance().LegacyGetOpHcomInfo(MAX_MODULE_DEVICE_NUM);
    EXPECT_EQ(&info, &backupSlot);
    EXPECT_EQ(info.isUsed, true);
    EXPECT_EQ(g_hcclDeviceId, static_cast<s32>(MAX_MODULE_DEVICE_NUM));
}

TEST_F(CollCommMgrTest, Ut_LegacyGetHcclExistDeviceOpInfoCtx_When_SlotNotUsedAndBackupNotUsed_Expect_MarkAndReturn)
{
    s32 devId = 4;
    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetHcclExistDeviceOpInfoCtx(devId);
    EXPECT_EQ(devId, 4);
    EXPECT_EQ(info.isUsed, true);
}

// ===== LegacyGetHcclOpInfoCtx =====

TEST_F(CollCommMgrTest, Ut_LegacyGetHcclOpInfoCtx_When_HcclGetDeviceIdSuccess_Expect_DelegateToExistDevice)
{
    g_hcclDeviceId = 0;
    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    EXPECT_EQ(g_hcclDeviceId, 0);
    EXPECT_EQ(info.isUsed, true);
}

TEST_F(CollCommMgrTest, Ut_LegacyGetHcclOpInfoCtx_When_HcclGetDeviceIdFailAndHasUsedSlot_Expect_ScanFirstUsed)
{
    g_hcclDeviceId = INVALID_INT;
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));

    CollCommMgr::GetInstance().LegacyGetOpHcomInfo(5).isUsed = true;
    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    // 函数内通过 HcclSetThreadDeviceId 写回 scan 命中的槽索引到 g_hcclDeviceId
    HcclOpInfoCtx& slot5 = CollCommMgr::GetInstance().LegacyGetOpHcomInfo(5);
    EXPECT_EQ(&info, &slot5);
    EXPECT_EQ(info.isUsed, true);
    EXPECT_EQ(g_hcclDeviceId, 5);
}

TEST_F(CollCommMgrTest, Ut_LegacyGetHcclOpInfoCtx_When_HcclGetDeviceIdFailAndNoUsedSlot_Expect_FallbackToBackup)
{
    g_hcclDeviceId = INVALID_INT;
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));

    HcclOpInfoCtx& info = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    // 函数内通过 HcclSetThreadDeviceId 写回 backup 槽索引到 g_hcclDeviceId
    HcclOpInfoCtx& backupSlot = CollCommMgr::GetInstance().LegacyGetOpHcomInfo(MAX_MODULE_DEVICE_NUM);
    EXPECT_EQ(&info, &backupSlot);
    EXPECT_EQ(info.isUsed, true);
    EXPECT_EQ(g_hcclDeviceId, static_cast<s32>(MAX_MODULE_DEVICE_NUM));
}

// ===== HcclSetThreadDeviceId / HcclGetThreadDeviceId =====

TEST_F(CollCommMgrTest, Ut_HcclSetThreadDeviceId_When_SetValidValue_Expect_HcclGetThreadDeviceIdReturnsSameValue)
{
    HcclSetThreadDeviceId(3);
    EXPECT_EQ(g_hcclDeviceId, 3);
    EXPECT_EQ(HcclGetThreadDeviceId(), 3);
}

TEST_F(CollCommMgrTest, Ut_HcclSetThreadDeviceId_When_Overwrite_Expect_LatestValueWins)
{
    HcclSetThreadDeviceId(1);
    EXPECT_EQ(HcclGetThreadDeviceId(), 1);
    HcclSetThreadDeviceId(5);
    EXPECT_EQ(HcclGetThreadDeviceId(), 5);
}

// ===== InitBaseCommRes =====
// InitBaseCommRes 仅透传调用 HcommResMgrInit(devId)，验证其以传入 devId 调用底层入口。
// 使用 stubs + invoke 捕获参数，避免 expects(once) 受 TearDown 链路额外调用影响。

TEST_F(CollCommMgrTest, Ut_InitBaseCommRes_When_Called_Expect_InvokeHcommResMgrInitWithDevId)
{
    const u32 devId = 7;
    g_ut_capturedHcommResMgrInitDevId = 0xFFFFFFFFu;
    MOCKER(HcommResMgrInit).stubs().will(invoke(stub_CollCommMgrTest_HcommResMgrInit));
    CollCommMgr::GetInstance().InitBaseCommRes(devId);
    EXPECT_EQ(g_ut_capturedHcommResMgrInitDevId, devId);
}

// ===== RegisterCollComm / UnregisterCollComm =====
// taskAbortHandler_ 收编进 CollCommMgr 后，注册/注销为统一入口，同时维护 allCollComms_ 与 taskAbortHandler_。
// 使用 simpleMode 轻量桩验证 map 状态与往返安全：simpleMode 桩构造不解引用 null comm，析构短路返回；
// taskAbortHandler Register/UnRegister 全程仅存指针/指针比较；UnRegisterToClusterMonitor 在 initialized_==false 下
// no-op。

TEST_F(CollCommMgrTest, Ut_RegisteCollComm_When_NewComm_Expect_AddedToMap)
{
    const std::string commId = "ut_reg_stub";
    CollComm stub(nullptr, 0, commId, ManagerCallbacks{}, CollCommInitMode::simpleMode);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().count(commId), 0U);
    CollCommMgr::GetInstance().RegisterCollComm(&stub);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().count(commId), 1U);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().at(commId), &stub);
    CollCommMgr::GetInstance().UnregisterCollComm(&stub);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().count(commId), 0U);
}

TEST_F(CollCommMgrTest, Ut_RegisteCollComm_When_DuplicateRegister_Expect_OverwriteNotGrow)
{
    const std::string commId = "ut_reg_dup";
    CollComm stub(nullptr, 0, commId, ManagerCallbacks{}, CollCommInitMode::simpleMode);
    CollCommMgr::GetInstance().RegisterCollComm(&stub);
    CollCommMgr::GetInstance().RegisterCollComm(&stub);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().count(commId), 1U);
    CollCommMgr::GetInstance().UnregisterCollComm(&stub);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().count(commId), 0U);
}

TEST_F(CollCommMgrTest, Ut_UnRegisteCollComm_When_NotRegistered_Expect_NoCrashNoChange)
{
    const std::string commId = "ut_unreg_noreg";
    CollComm stub(nullptr, 0, commId, ManagerCallbacks{}, CollCommInitMode::simpleMode);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().count(commId), 0U);
    // 未注册直接注销：map erase no-op，taskAbortHandler UnRegister 仅告警，clusterMonitor initialized_==false no-op
    CollCommMgr::GetInstance().UnregisterCollComm(&stub);
    EXPECT_EQ(CollCommMgr::GetInstance().GetAllCollComms().count(commId), 0U);
}

// ===== GetDpuTaskExpMem / GetDpuDevId（commId 反查链，锁内完成反查与取值）=====

TEST_F(CollCommMgrTest, Ut_GetDpuTaskExpMem_When_CommNotRegistered_Expect_NotFound)
{
    DpuTaskExpMemRef ref;
    HcclResult ret = CollCommMgr::GetInstance().GetDpuTaskExpMem("ut_cpu_thread_absent", ref);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    EXPECT_EQ(ref.addr, nullptr);
    EXPECT_EQ(ref.keepAlive, nullptr);
}

// simpleMode 下 commEngineResMgr_ 为 null，GetThreadMgr() 返回 nullptr：验证反查链在解引用前的空指针拦截，
// 并确认注销后条目被摘除、不再反查到已销毁通信域
TEST_F(CollCommMgrTest, Ut_GetDpuTaskExpMem_When_ThreadMgrMissing_Expect_PtrError)
{
    const std::string commId = "ut_cpu_thread_stub";
    CollComm stub(nullptr, 0, commId, ManagerCallbacks{}, CollCommInitMode::simpleMode);
    CollCommMgr::GetInstance().RegisterCollComm(&stub);

    DpuTaskExpMemRef ref;
    EXPECT_EQ(CollCommMgr::GetInstance().GetDpuTaskExpMem(commId, ref), HCCL_E_PTR);
    EXPECT_EQ(ref.addr, nullptr);

    uint32_t devId = 0;
    EXPECT_EQ(CollCommMgr::GetInstance().GetDpuDevId(commId, devId), HCCL_E_PTR);

    CollCommMgr::GetInstance().UnregisterCollComm(&stub);
    EXPECT_EQ(CollCommMgr::GetInstance().GetDpuTaskExpMem(commId, ref), HCCL_E_NOT_FOUND);
    EXPECT_EQ(CollCommMgr::GetInstance().GetDpuDevId(commId, devId), HCCL_E_NOT_FOUND);
}

// ===== 析构顺序问题（static destruction order fiasco）=====
// CollCommMgr 是 Meyers 单例，~CollCommMgr() 间接依赖 base_comm 层单例。
// 若 base_comm 单例后于 CollCommMgr 构造，则先于其析构，导致 coredump。
// 修复：GetInstance() 中 magic static lambda 先获取 deviceId 再调 HcommResMgrInit，
// 确保 base_comm 单例先于 CollCommMgr 构造。本测试验证 magic static 不重复调用。
TEST_F(CollCommMgrTest, Ut_GetInstance_When_CalledAfterInit_Expect_HcommResMgrInitNotReInvoked)
{
    (void)CollCommMgr::GetInstance(); // 先触发 magic static 初始化

    g_ut_hcommResMgrInitCalledInGetInstance = false;
    MOCKER(HcommResMgrInit).stubs().will(invoke(stub_DestructionOrder_HcommResMgrInit));
    (void)CollCommMgr::GetInstance();
    EXPECT_FALSE(g_ut_hcommResMgrInitCalledInGetInstance)
        << "HcommResMgrInit should NOT be called again after magic static initialization";
}

// 验证 HcommResMgrInit 返回失败时 GetInstance() 仍返回有效实例，不因初始化失败而崩溃。
// magic static lambda 中即使 HcommResMgrInit 失败，也仅打印告警并返回 true，
// static CollCommMgr instance 照常构造。由于 magic static 在进程内仅执行一次，
// SetUp 中已触发初始化，此处 mock HcommResMgrInit 失败后调用 GetInstance()，
// 验证单例不崩溃且返回稳定引用。
TEST_F(CollCommMgrTest, Ut_GetInstance_When_HcommResMgrInitFails_Expect_ReturnsValidInstance)
{
    MOCKER(HcommResMgrInit).stubs().will(returnValue(static_cast<HcommResult>(HCOMM_E_INTERNAL)));
    CollCommMgr& inst = CollCommMgr::GetInstance();
    EXPECT_EQ(&inst, &CollCommMgr::GetInstance())
        << "GetInstance should return valid singleton even if HcommResMgrInit fails";
}

// ===== CollComm 与 DetectConnectionAnomalies 生命周期配对（dcaInited_）=====

TEST_F(CollCommMgrTest, Ut_CollComm_When_DcaInitedToggled_Expect_DeinitPairedWithInit)
{
    // 轮1：fullMode + dcaInited_=true：析构时 Deinit 恰好一次
    {
        CollComm comm(nullptr, 0, "ut_dca_pair_inited", ManagerCallbacks{}, CollCommInitMode::fullMode);
        comm.dcaInited_ = true;
        hcomm::DcaStubResetCounts();
    }
    EXPECT_EQ(hcomm::DcaStubDeinitCallCount(), 1u);

    // 轮2：未 Init（dcaInited_ 默认 false）→ 不触发 Deinit（零增量）
    {
        CollComm comm(nullptr, 0, "ut_dca_pair_notinited", ManagerCallbacks{}, CollCommInitMode::fullMode);
    }
    EXPECT_EQ(hcomm::DcaStubDeinitCallCount(), 1u);
}

TEST_F(CollCommMgrTest, Ut_GetDetectConnectionAnomalies_When_DeviceLogicIdOutOfRange_Expect_ReturnDefaultInstance)
{
    // 越界 deviceLogicId 回退 0 号实例
    s32 outOfRange = -1;
    hcomm::DetectConnectionAnomalies& fallback = CollCommMgr::GetInstance().GetDetectConnectionAnomalies(outOfRange);
    hcomm::DetectConnectionAnomalies& slot0 = CollCommMgr::GetInstance().GetDetectConnectionAnomalies(0);
    EXPECT_EQ(&fallback, &slot0);
}
