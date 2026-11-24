/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "ipc_remote_notify.h"
#undef private
#include "local_notify_v2.h"
#include "exchange_ipc_notify_dto.h"
using namespace Hccl;

class IpcRemoteNotifyTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "IpcRemoteNotifyTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "IpcRemoteNotifyTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType::DEV_TYPE_910A2));
        MOCKER(HrtIpcOpenNotify).stubs().with(mockcpp::any()).will(returnValue((void*)fakeNotifyHandleAddr));
        MOCKER(HrtIpcOpenNotifyWithFlag)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue((void*)fakeNotifyHandleAddr));
        MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(fakePid));
        MOCKER(HrtNotifyGetAddr).stubs().with(mockcpp::any()).will(returnValue(fakeAddress));
        std::cout << "A Test case in IpcRemoteNotifyTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RdmaRemoteNotifyTest TearDown" << std::endl;
    }
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[65] = "testRtsNotify";
};

// 跨进程(rmtPid != myPid): 句柄由 HrtIpcOpenNotify 打开, 归本对象所有, 析构应销毁一次
TEST_F(IpcRemoteNotifyTest, Ut_When_CrossProcess_ThenDestruct_Expect_DestroyCalledOnce)
{
    // given: rmtPid != myPid, 走 HrtIpcOpenNotify 跨进程分支
    MOCKER(HrtNotifyDestroy).expects(once());
    ExchangeIpcNotifyDto dto(fakeNotifyHandleAddr, fakeNotifyId, fakePid + 1, 0, false);

    // when: 构造后析构
    {
        IpcRemoteNotify remoteNotify(dto);
    }

    // then: 析构销毁自有句柄一次(由 TearDown 的 GlobalMockObject::verify 校验)
}

// 同进程(rmtPid == myPid): handle 借用本地 RtsNotify 句柄, 不归本对象所有, 析构不应销毁
TEST_F(IpcRemoteNotifyTest, Ut_When_SameProcess_ThenDestruct_Expect_DestroyNotCalled)
{
    // given: rmtPid == myPid, 同进程分支借用本地 RtsNotify 句柄
    MOCKER(HrtNotifyDestroy).expects(never());
    ExchangeIpcNotifyDto dto(fakeNotifyHandleAddr, fakeNotifyId, fakePid, 0, false);

    // when: 构造后析构
    {
        IpcRemoteNotify remoteNotify(dto);
    }

    // then: 析构不销毁借用句柄(由 TearDown 的 GlobalMockObject::verify 校验)
}

class RdmaRemoteNotifyTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RdmaRemoteNotifyTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RdmaRemoteNotifyTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType::DEV_TYPE_910A2));
        MOCKER(HrtIpcOpenNotify).stubs().with(mockcpp::any()).will(returnValue((void*)fakeNotifyHandleAddr));
        MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(fakePid));
        MOCKER(HrtNotifyGetAddr).stubs().with(mockcpp::any()).will(returnValue(fakeAddress));
        std::cout << "A Test case in RdmaRemoteNotifyTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RdmaRemoteNotifyTest TearDown" << std::endl;
    }
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[65] = "testRtsNotify";
};
