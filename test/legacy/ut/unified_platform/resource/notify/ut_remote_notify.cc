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
        MOCKER(HrtGetDeviceType)
            .stubs()
            .with(outBound(Hccl::DevType(DevType::DEV_TYPE_910A2)))
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtIpcOpenNotify).stubs().with(mockcpp::any()).will(returnValue((void*)fakeNotifyHandleAddr));
        MOCKER(HrtIpcOpenNotifyWithFlag)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue((void*)fakeNotifyHandleAddr));
        MOCKER(HrtDeviceGetBareTgid).stubs().with(outBound(fakePid)).will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtNotifyGetAddr)
            .stubs()
            .with(mockcpp::any(), outBound(fakeAddress))
            .will(returnValue(HcclResult::HCCL_SUCCESS));
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

class RdmaRemoteNotifyTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RdmaRemoteNotifyTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RdmaRemoteNotifyTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType)
            .stubs()
            .with(outBound(Hccl::DevType(DevType::DEV_TYPE_910A2)))
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtIpcOpenNotify).stubs().with(mockcpp::any()).will(returnValue((void*)fakeNotifyHandleAddr));
        MOCKER(HrtDeviceGetBareTgid).stubs().with(outBound(fakePid)).will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HrtNotifyGetAddr)
            .stubs()
            .with(mockcpp::any(), outBound(fakeAddress))
            .will(returnValue(HcclResult::HCCL_SUCCESS));
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
