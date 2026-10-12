/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include "hccp_async_request.h"

#include <new>
#include <mutex>
#include <unordered_map>

#include "sim_log.h"

namespace hccp_async_req {
namespace {

    struct RaAsyncRequest {
        int reqResult;
        unsigned int interfaceOpcode;
    };

    std::mutex g_raAsyncReqMutex;
    std::unordered_map<void*, RaAsyncRequest> g_raAsyncReqRegistry;

} // namespace

void* Register(int reqResult, unsigned int interfaceOpcode)
{
    auto* req = new (std::nothrow) RaAsyncRequest{reqResult, interfaceOpcode};
    if (req == nullptr) {
        HCCL_VM_ERROR("alloc hccp async request failed, opcode:{}", interfaceOpcode);
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_raAsyncReqMutex);
    g_raAsyncReqRegistry.emplace(req, *req);
    return req;
}

int Consume(void* reqHandle, int* reqResult, unsigned int* interfaceOpcode)
{
    if (reqHandle == nullptr || reqResult == nullptr || interfaceOpcode == nullptr) {
        HCCL_VM_ERROR("hccp async consume invalid param: reqHandle={:p}", reqHandle);
        return -1;
    }
    RaAsyncRequest finished{};
    {
        std::lock_guard<std::mutex> lock(g_raAsyncReqMutex);
        auto it = g_raAsyncReqRegistry.find(reqHandle);
        if (it == g_raAsyncReqRegistry.end()) {
            // 合同规定完成即释放；查无此句柄即重复查询或非法句柄，明确报错。
            HCCL_VM_ERROR("hccp async consume unknown or already-freed reqHandle:{:p}", reqHandle);
            return -1;
        }
        finished = it->second;
        g_raAsyncReqRegistry.erase(it);
    }
    delete static_cast<RaAsyncRequest*>(reqHandle);
    *reqResult = finished.reqResult;
    *interfaceOpcode = finished.interfaceOpcode;
    return 0;
}

} // namespace hccp_async_req
