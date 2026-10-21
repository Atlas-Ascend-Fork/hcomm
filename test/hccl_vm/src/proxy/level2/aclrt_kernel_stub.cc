/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "KERNEL_STUB"

#include <errno.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <sstream>
#include <set>
#include <string>
#include <unistd.h>
#include <sys/uio.h>
#include <vector>
#include <thread>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "runtime/base.h"
#include "aiv_kernel/aiv_mode_stub/aiv_mode_stub_base.h"
#include "hccl/hccl_types.h"
#include "sim_common_defs.h"
#include "sim_log.h"
#include "store_sim_comm_pool_policy.h"
#include "store_sim_device_memory_manager.h"
#include "store_sim_memory_manager.h"
#include "store_sim_run_mode.h"
#include "store_sim_store_pub.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "graph_capture.h"
#include "sim_common_api.h"
#include "hccl_proxy_common.h"
#include "sim_sub_process_manager.h"
#include "sim_dpu_kernel_lib_mgr.h"

namespace fs = std::filesystem;

// 首参类型随 CANN 版本变化(9.3 起由 aclrtFuncHandle 改为 const void*)，按头文件声明自身推导以兼容新旧
namespace {
template <typename F>
struct AclFuncHandleArg;
template <typename R, typename A0, typename... Rest>
struct AclFuncHandleArg<R (*)(A0, Rest...)> {
    using Type = A0;
};
using AclrtGetFunctionAddrArg = AclFuncHandleArg<decltype(&aclrtGetFunctionAddr)>::Type;
using AclrtGetFunctionNameArg = AclFuncHandleArg<decltype(&aclrtGetFunctionName)>::Type;
} // namespace

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// ===== DPU kernel launch support (hostdpu mode) =====
const std::string DPU_KERNEL_SO = "libccl_dpu.so";
struct ArgsBuffer {
    void* data;
    uint64_t size;
};

thread_local pid_t g_devicePid = 0;

static bool CheckDeviceProcStatus()
{
    if (g_devicePid == 0) {
        return false;
    }
    int status = 0;
    pid_t result = waitpid(g_devicePid, &status, WNOHANG);
    if (result == 0) {
        return true;
    }
    if (result == g_devicePid) {
        if (WIFSIGNALED(status)) {
            HCCL_VM_ERROR("device process[{}] killed by signal {}", g_devicePid, WTERMSIG(status));
        } else {
            HCCL_VM_ERROR("device process[{}] exited with status {}", g_devicePid, WEXITSTATUS(status));
        }
    } else {
        HCCL_VM_ERROR("waitpid failed for pid {}, errno: {} ({})", g_devicePid, errno, strerror(errno));
    }
    FlushLog();
    exit(EXIT_FAILURE);
    g_devicePid = 0;
    return false;
}

namespace sim {
    constexpr size_t MAX_ARGS_BUFF_SIZE = 64 * 1024U;

    struct FuncArgsDetail {
        uint8_t* argsData{nullptr};
        size_t argsDataSize{0};
        bool isHold{false};
    };

    struct FuncArgs {
        uint8_t* argsBuff{nullptr};
        size_t argsBufferSize{0};
        bool isSysMem{true};
        size_t useOffset{0};
        std::vector<FuncArgsDetail*> argDetail;

        FuncArgs()
        {
            argsBuff = new uint8_t[MAX_ARGS_BUFF_SIZE];
            argsBufferSize = MAX_ARGS_BUFF_SIZE;
            isSysMem = true;
        }
        ~FuncArgs()
        {
            if (isSysMem && argsBuff) {
                delete[] argsBuff;
            }

            for (auto& arg : argDetail) {
                delete arg;
            }
        }

        void ResetArgsBuff()
        {
            if (isSysMem && argsBuff) {
                delete[] argsBuff;
                argsBuff = nullptr;
                argsBufferSize = 0;
            }
        }
    };

    struct FuncHandle {
        std::string funcName{""};
        std::string kernelName{""};
        std::string soName{""};
        std::vector<FuncArgs*> funArgs;
        ~FuncHandle()
        {
            for (auto& funcArg : funArgs) {
                delete funcArg;
            }
        }
    };

    struct DevBinary;
    struct Program {
        DevBinary* bin{nullptr};
        std::map<std::string, FuncHandle*> funcs;
        ~Program()
        {
            for (auto& func : funcs) {
                delete func.second;
            }
        }
    };

    struct DevBinary {
        std::string binPath{""};
        std::map<std::string, std::string> funcSoMap;
        void* data{nullptr};
        size_t dataLen{0};
        Program prog;

        ~DevBinary()
        {
            if (data != nullptr) {
                delete[] (char*)data;
            }
        }
    };

    std::set<DevBinary*> g_kernelBinary;
} // namespace sim

aclrtBinary aclrtCreateBinary(const void* data, size_t dataLen)
{
    sim::DevBinary* binPtr = new sim::DevBinary();

    auto res = sim::g_kernelBinary.insert(binPtr);
    if (!res.second) {
        HCCL_VM_ERROR("failed");
        return 0;
    }

    binPtr->data = reinterpret_cast<void*>(new char[dataLen]);
    memcpy(binPtr->data, data, dataLen);
    binPtr->dataLen = dataLen;
    HCCL_VM_INFO("dataLen{:d} binary{:p}", dataLen, (aclrtBinary)(binPtr));
    return (aclrtBinary)(binPtr);
}

aclError aclrtDestroyBinary(aclrtBinary binary)
{
    sim::DevBinary* binPtr = (sim::DevBinary*)binary;
    if (auto search = sim::g_kernelBinary.find(binPtr); search == sim::g_kernelBinary.end()) {
        HCCL_VM_ERROR("can not find this binary");
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }
    sim::g_kernelBinary.erase(binPtr);

    HCCL_VM_INFO(" binPtr{:p}", binary);
    delete binPtr;
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoad(const aclrtBinary binary, aclrtBinHandle* binHandle)
{
    sim::DevBinary* binPtr = (sim::DevBinary*)binary;
    // 需要解析binary
    *binHandle = (aclrtBinHandle) & (binPtr->prog);
    HCCL_VM_INFO(" binHandle:{:p}", *binHandle);
    return ACL_SUCCESS;
}

aclError aclrtBinaryUnLoad(aclrtBinHandle binHandle)
{
    (void)binHandle;
    HCCL_VM_WARN("is empty.");
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions* options, aclrtBinHandle* binHandle)
{
    // 复用已有的
    for (auto* devBin : sim::g_kernelBinary) {
        if (devBin != nullptr && devBin->binPath == binPath) {
            *binHandle = (aclrtBinHandle) & (devBin->prog);
            HCCL_VM_INFO("binPath:{} reused binHandle:{:p}", binPath, *binHandle);
            return ACL_SUCCESS;
        }
    }

    sim::DevBinary* binPtr = new sim::DevBinary();
    binPtr->binPath = binPath;
    binPtr->prog.bin = binPtr;
    sim::ParseKernelJson(binPath, binPtr->funcSoMap);

    auto res = sim::g_kernelBinary.insert(binPtr);
    if (!res.second) {
        HCCL_VM_ERROR("file:{} insert failed", binPath);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    *binHandle = (aclrtBinHandle) & (binPtr->prog);
    HCCL_VM_INFO(" binPath:{} binHandle{:p}", binPath, *binHandle);
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromData(
    const void* data, size_t length, const aclrtBinaryLoadOptions* options, aclrtBinHandle* binHandle)
{
    (void)data;
    (void)length;
    (void)options;
    (void)binHandle;
    HCCL_VM_WARN("is empty.");
    return ACL_SUCCESS;
}

rtError_t rtRegisterFuncSymbol(void* binHandle, const void* symbol, const char* kernelName, void* reserve)
{
    (void)binHandle;
    (void)symbol;
    (void)kernelName;
    (void)reserve;
    HCCL_VM_INFO("is empty.");
    return 0;
}

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char* kernelName, aclrtFuncHandle* funcHandle)
{
    sim::Program* prog = (sim::Program*)(uintptr_t)binHandle;

    std::string kernelSoName{""};
    std::string funcName(kernelName);
    if (prog->bin != nullptr) {
        auto soIt = prog->bin->funcSoMap.find(funcName);
        if (soIt != prog->bin->funcSoMap.end()) {
            kernelSoName = soIt->second;
        }
    }

    auto funcIter = prog->funcs.find(funcName);
    if (funcIter == prog->funcs.end()) {
        HCCL_VM_WARN("kernelName:{} not register insert it", kernelName);

        sim::FuncHandle* func = new sim::FuncHandle;
        func->funcName = funcName;
        func->kernelName = kernelName;
        func->soName = kernelSoName;
        auto res = prog->funcs.insert(std::pair<std::string, sim::FuncHandle*>(func->funcName, func));
        if (!res.second) {
            HCCL_VM_ERROR("func:{} kernelName:{} insert failed", funcName, kernelName);
            return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
        }
        *funcHandle = reinterpret_cast<aclrtFuncHandle>(func);
    } else {
        *funcHandle = reinterpret_cast<aclrtFuncHandle>(funcIter->second);
    }

    HCCL_VM_INFO(" funcHandle:{:p}", *funcHandle);
    return ACL_SUCCESS;
}

aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry, aclrtFuncHandle* funcHandle)
{
    (void)binHandle;
    (void)funcEntry;
    (void)funcHandle;
    HCCL_VM_WARN("is empty.");
    return ACL_SUCCESS;
}

aclError aclrtGetFunctionAddr(AclrtGetFunctionAddrArg funcHandle, void** aicAddr, void** aivAddr)
{
    (void)funcHandle;
    (void)aicAddr;
    (void)aivAddr;
    HCCL_VM_WARN("is empty.");
    return ACL_SUCCESS;
}

aclError aclrtGetFunctionName(AclrtGetFunctionNameArg funcHandle, uint32_t maxLen, char* name)
{
    const sim::FuncHandle* funcHandlePtr = (const sim::FuncHandle*)(uintptr_t)funcHandle;

    size_t copyLen = std::min(static_cast<size_t>(maxLen - 1), funcHandlePtr->funcName.length());
    memcpy(name, funcHandlePtr->funcName.data(), copyLen);
    name[copyLen] = '\0';
    HCCL_VM_INFO(" funcName{}", funcHandlePtr->funcName.data());
    return ACL_SUCCESS;
}

aclError aclrtRegisterCpuFunc(
    const aclrtBinHandle handle, const char* funcName, const char* kernelName, aclrtFuncHandle* funcHandle)
{
    sim::Program* prog = (sim::Program*)(uintptr_t)handle;

    sim::FuncHandle* func = new sim::FuncHandle;
    func->funcName = funcName;
    func->kernelName = kernelName;
    auto res = prog->funcs.insert(std::pair<std::string, sim::FuncHandle*>(func->funcName, func));
    if (!res.second) {
        HCCL_VM_ERROR("func:{} kernelName:{} insert failed", funcName, kernelName);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }
    *funcHandle = reinterpret_cast<aclrtFuncHandle>(func);
    HCCL_VM_INFO(" funcHandle:{:p}", *funcHandle);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsInit(aclrtFuncHandle funcHandle, aclrtArgsHandle* argsHandle)
{
    sim::FuncHandle* func = (sim::FuncHandle*)(uintptr_t)funcHandle;
    sim::FuncArgs* args = new sim::FuncArgs;
    func->funArgs.push_back(args);

    auto iter = func->funArgs.rbegin();
    *argsHandle = (aclrtArgsHandle)args;
    HCCL_VM_INFO("FuncHandle:{:p} ArgsHandle:{:p} ", funcHandle, *argsHandle);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsInitByUserMem(
    aclrtFuncHandle funcHandle, aclrtArgsHandle argsHandle, void* userHostMem, size_t actualArgsSize)
{
    (void)funcHandle;
    HCCL_VM_INFO(" argsHandle:{:p} userHostMem:{:p},actualArgsSize:{:d}", argsHandle, userHostMem, actualArgsSize);
    sim::FuncArgs* args = (sim::FuncArgs*)argsHandle;
    args->ResetArgsBuff();
    args->argsBuff = (uint8_t*)userHostMem;
    args->argsBufferSize = actualArgsSize;
    args->isSysMem = false;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsGetMemSize(aclrtFuncHandle funcHandle, size_t userArgsSize, size_t* actualArgsSize)
{
    (void)funcHandle;
    HCCL_VM_INFO("userArgsSize {:d}.", userArgsSize);
    *actualArgsSize = userArgsSize;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsGetHandleMemSize(aclrtFuncHandle funcHandle, size_t* memSize)
{
    HCCL_VM_INFO("funcHandle:{:p} userArgsSize 64k", funcHandle);
    // 句柄 + 参数的内存大小
    *memSize = sim::MAX_ARGS_BUFF_SIZE;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void* param, size_t paramSize, aclrtParamHandle* paramHandle)
{
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;
    HCCL_VM_INFO(" argsHandle:{:p} paramSize:{:d}", argsHandle, paramSize);
    sim::FuncArgsDetail* argsDetail = new sim::FuncArgsDetail;
    if (paramSize > args->argsBufferSize || args->useOffset > args->argsBufferSize - paramSize) {
        HCCL_VM_ERROR(
            "args buffer overflow, useOffset:{}, paramSize:{}, bufferSize:{}", args->useOffset, paramSize,
            args->argsBufferSize);
        delete argsDetail;
        return ACL_ERROR_INVALID_PARAM;
    }
    argsDetail->argsData = args->argsBuff + args->useOffset;
    argsDetail->argsDataSize = paramSize;

    memcpy(argsDetail->argsData, param, paramSize);
    args->useOffset += paramSize;
    args->argDetail.push_back(argsDetail);
    *paramHandle = (aclrtParamHandle)argsDetail;
    HCCL_VM_INFO("argsHandle:{:p} paramHandle:{:p} ", argsHandle, *paramHandle);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppendPlaceHolder(aclrtArgsHandle argsHandle, aclrtParamHandle* paramHandle)
{
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;

    sim::FuncArgsDetail* argsDetail = new sim::FuncArgsDetail;
    argsDetail->isHold = true;

    args->argDetail.push_back(argsDetail);
    *paramHandle = (aclrtParamHandle)argsDetail;
    HCCL_VM_INFO("paramHandle:{:p} ", *paramHandle);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsGetPlaceHolderBuffer(
    aclrtArgsHandle argsHandle, aclrtParamHandle paramHandle, size_t dataSize, void** bufferAddr)
{
    HCCL_VM_INFO("argsHandle:{:p} ParamHandle:{:p} dataSize:{:d}", argsHandle, paramHandle, dataSize);
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;

    sim::FuncArgsDetail* detail = (sim::FuncArgsDetail*)(uintptr_t)paramHandle;
    if (dataSize > args->argsBufferSize || args->useOffset > args->argsBufferSize - dataSize) {
        HCCL_VM_ERROR(
            "args buffer overflow, useOffset:{}, dataSize:{}, bufferSize:{}", args->useOffset, dataSize,
            args->argsBufferSize);
        return ACL_ERROR_INVALID_PARAM;
    }
    detail->argsData = args->argsBuff + args->useOffset;
    detail->argsDataSize = dataSize;
    args->useOffset += dataSize;
    *bufferAddr = reinterpret_cast<void*>(detail->argsData);
    HCCL_VM_INFO("paramHandle:{:p} ", *bufferAddr);
    return ACL_SUCCESS;
}

aclError
aclrtKernelArgsParaUpdate(aclrtArgsHandle argsHandle, aclrtParamHandle paramHandle, void* param, size_t paramSize)
{
    HCCL_VM_INFO("argsHandle:{:p} ParamHandle:{:p} paramSize:{:d}", argsHandle, paramHandle, paramSize);
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;

    sim::FuncArgsDetail* detail = (sim::FuncArgsDetail*)(uintptr_t)paramHandle;
    if (detail->isHold || detail->argsDataSize != paramSize) {
        HCCL_VM_ERROR("invalid param handle type:{:d}", detail->isHold);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    memcpy(detail->argsData, param, paramSize);
    HCCL_VM_INFO("argsData:{:p} ", reinterpret_cast<void*>(detail->argsData));
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHandle)
{
    (void)argsHandle;
    HCCL_VM_WARN("is empty.");
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernel(
    aclrtFuncHandle funcHandle, uint32_t blockDim, const void* argsData, size_t argsSize, aclrtStream stream)
{
    (void)funcHandle;
    (void)blockDim;
    (void)argsData;
    (void)argsSize;
    (void)stream;
    HCCL_VM_WARN("is empty.");

    return ACL_SUCCESS;
}

// 检查展开模式退化至AICPU模式
void CheckExpansionModeDegradeToAICPU()
{
    // 避免同一轮次多次kernel下发重复更新展开模式
    int curMode = sim::operation::QueryLatestOpExpansionMode();
    if (curMode == static_cast<int>(sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU)) {
        return;
    }

    // CCU/AIV退化为AICPU模式时，更新模型中展开模式为AICPU
    const char* expanEnv = std::getenv("HCCL_OP_EXPANSION_MODE");
    std::string expanMode = expanEnv == nullptr ? "" : std::string(expanEnv);
    bool ccuEnabled = expanMode == "CCU_SCHED" || expanMode == "CCU_MS";
    bool aivEnabled = expanMode == "AIV";
    if (ccuEnabled || aivEnabled) {
        HCCL_VM_INFO("Switch the expansion mode[{} -> AICPU].", aivEnabled ? "AIV" : "CCU");
        constexpr uint8_t aicpuMode
            = static_cast<uint8_t>(sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU);
        sim::operation::UpdateOpExpansionMode(aicpuMode);
    }
}

void TryLaunchAicpuDevProcForRank(int32_t rankId, uint32_t deviceKey)
{
    auto& procMgr = sim::GetAicpuProcMgr();
    if (procMgr.IsAlive()) {
        return;
    }

    auto config = sim::CreateAicpuDeviceConfig(rankId, deviceKey);
    if (procMgr.CreateProcess(config) != 0) {
        HCCL_VM_ERROR("failed to create device process.");
        exit(EXIT_FAILURE);
    }

    g_devicePid = procMgr.GetPid();
    HCCL_VM_INFO("device process for rankId[{}] launched, pid={}", rankId, g_devicePid);
}

// 返回 0 成功；非 0 失败（kernel symbol 解析失败 / pipe 请求失败或 device 子进程
// 异常退出 / device 侧执行返回错误状态）。调用方必须检查返回值并向 host 业务层
// 传播，禁止失败后仍返回 ACL_SUCCESS。
static int LaunchAICPUKernelFunc(const std::string& kernelName, const std::string& soName, aclrtArgsHandle argsHandle)
{
    uint32_t deviceId = (uint32_t)sim::runtime::GetCurrDeviceId();
    uint32_t devKey = (uint32_t)sim::runtime::GetCurrDeviceKey();
    HCCL_VM_INFO("deviceId:{}, devKey:{}, kernelName:{}.", deviceId, devKey, kernelName);

    // 检查退化逻辑并在退化场景启动device进程
    CheckExpansionModeDegradeToAICPU();
    TryLaunchAicpuDevProcForRank(deviceId, devKey);

    sim::FuncArgs* args = (sim::FuncArgs*)argsHandle;

    // 零拷贝发送：ExecKernelPayload 头与 args 分段，一次 writev 直写管道，不再拼连续 msgBuf
    uint32_t argsSize = (uint32_t)args->useOffset;
    ExecKernelPayload payload{};
    strncpy(payload.kernelName, kernelName.c_str(), sizeof(payload.kernelName) - 1);
    strncpy(payload.soName, soName.c_str(), sizeof(payload.soName) - 1);

    struct iovec segs[2];
    uint32_t segCnt = 1;
    segs[0].iov_base = &payload;
    segs[0].iov_len = sizeof(payload);
    if (argsSize > 0 && args->argsBuff != nullptr) {
        segs[1].iov_base = args->argsBuff;
        segs[1].iov_len = argsSize;
        segCnt = 2;
    }
    // 只有当前仍处于 HCCL 算子调用期间，才把当前 opDetail 绑定到 device。
    payload.opDetailId = sim::runtime::g_cur_comm_key == 0 ? 0 : sim::operation::g_currOpDetailId;

    uint8_t rspCmd;
    RspExecKernelPayload rspPayload{};
    uint32_t rspLen = 0;
    if (sim::GetAicpuProcMgr().RequestV(
            PIPE_CMD_EXEC_KERNEL, segs, segCnt, rspCmd, &rspPayload, sizeof(rspPayload), rspLen)
        != 0) {
        HCCL_VM_ERROR("Request EXEC_KERNEL failed (pipe error or device process abnormal), kernel:{}.", kernelName);
        return -1;
    }

    if (rspCmd != PIPE_RSP_EXEC_KERNEL) {
        HCCL_VM_ERROR("unexpected response cmd: 0x{:02x}, kernel:{}", rspCmd, kernelName);
        return -1;
    }

    if (rspPayload.status != 0) {
        HCCL_VM_ERROR("kernel returned error status: {}, kernel:{}", rspPayload.status, kernelName);
        return static_cast<int>(rspPayload.status);
    }
    return 0;
}

// 图模式重放专用重发：只吃 GraphAction 固化的原料(kernelName/soName/args/opDetailId)。
// 与正常下发 LaunchAICPUKernelFunc 完全隔离、互不复用——正常下发怎么演进都不影响重放，
// 重放有 bug 也不影响正常下发。显式 extern "C"，与 graph_capture.h 声明一致。
extern "C" void LaunchAicpuKernelRaw(
    const char* kernelName, const char* soName, const uint8_t* argsBytes, uint32_t argsSize, uint32_t opDetailId)
{
    uint32_t deviceId = (uint32_t)sim::runtime::GetCurrDeviceId();
    uint32_t devKey = (uint32_t)sim::runtime::GetCurrDeviceKey();
    HCCL_VM_INFO("replay deviceId:{}, devKey:{}, kernelName:{}.", deviceId, devKey, kernelName);

    // 检查退化逻辑并在退化场景启动device进程
    sim::operation::g_currOpDetailId = opDetailId;
    sim::operation::UpdateOpExpansionMode(
        static_cast<uint8_t>(sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU));
    TryLaunchAicpuDevProcForRank(deviceId, devKey);

    ExecKernelPayload payload{};
    strncpy(payload.kernelName, kernelName, sizeof(payload.kernelName) - 1);
    strncpy(payload.soName, soName, sizeof(payload.soName) - 1);
    payload.opDetailId = opDetailId;

    struct iovec segs[2];
    uint32_t segCnt = 1;
    segs[0].iov_base = &payload;
    segs[0].iov_len = sizeof(payload);
    if (argsSize > 0 && argsBytes != nullptr) {
        segs[1].iov_base = const_cast<uint8_t*>(argsBytes);
        segs[1].iov_len = argsSize;
        segCnt = 2;
    }

    uint8_t rspCmd;
    RspExecKernelPayload rspPayload{};
    uint32_t rspLen = 0;
    if (sim::GetAicpuProcMgr().RequestV(
            PIPE_CMD_EXEC_KERNEL, segs, segCnt, rspCmd, &rspPayload, sizeof(rspPayload), rspLen)
        != 0) {
        HCCL_VM_ERROR("Request EXEC_KERNEL failed.");
        return;
    }

    if (rspCmd != PIPE_RSP_EXEC_KERNEL) {
        HCCL_VM_ERROR("unexpected response cmd: 0x{:02x}", rspCmd);
        return;
    }

    if (rspPayload.status != 0) {
        HCCL_VM_ERROR("kernel returned error status: {}", rspPayload.status);
    }
}

aclError aclrtLaunchKernelWithConfig(
    aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream, aclrtLaunchKernelCfg* cfg,
    aclrtArgsHandle argsHandle, void* reserve)
{
    (void)blockDim;
    (void)cfg;
    (void)reserve;
    sim::FuncHandle* func = (sim::FuncHandle*)(uintptr_t)funcHandle;
    if (funcHandle == nullptr || func == nullptr || argsHandle == nullptr) {
        HCCL_VM_ERROR("invalid input argsHandle or funcHandle");
        return ACL_ERROR_INVALID_PARAM;
    }

    HCCL_VM_INFO(
        "funcName:{}, kernelName:{}, func:{:p} stream:{:p} args:{:p}", func->funcName, func->kernelName,
        (void*)funcHandle, (void*)stream, (void*)argsHandle);

    if (func->kernelName.empty() || func->soName.empty()) {
        HCCL_VM_ERROR(
            "Launch kernel failed: invalid param, kernelName='{}', soName='{}'", func->kernelName, func->soName);
        return ACL_ERROR_INVALID_PARAM;
    }

    // ===== 图模式采集分支：独立取料并记录，与正常下发隔离 =====
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    if (IsCapturingStream(streamId)) {
        sim::FuncArgs* args = (sim::FuncArgs*)argsHandle;
        uint32_t argsSize = (uint32_t)args->useOffset;
        uint64_t commId = sim::runtime::g_cur_comm_key;
        uint32_t opDetailId = sim::runtime::g_cur_comm_key == 0 ? 0 : sim::operation::g_currOpDetailId;
        if (!RecordAicpuKernelAction(
                streamId, func->kernelName.c_str(), func->soName.c_str(), args->argsBuff, argsSize, commId,
                opDetailId)) {
            return ACL_ERROR_INTERNAL_ERROR;
        }
        HCCL_VM_INFO("kernel:{} captured to model(stream:{}), not executed", func->kernelName, streamId);
        return ACL_SUCCESS;
    }

    // 正常下发：走原始 LaunchAICPUKernelFunc（图模式不污染这条路径）。
    // AICPU或CCU退化为AICPU模式时调用；执行失败（symbol 缺失/地址转换失败/
    // device kernel 返回非 0/pipe 异常）必须向 host 传播，不得伪装成功。
    if (LaunchAICPUKernelFunc(func->kernelName, func->soName, argsHandle) != 0) {
        return ACL_ERROR_INTERNAL_ERROR;
    }
    HCCL_VM_INFO("kernel:{}[{}] execute finished.", func->kernelName, func->soName);

    return ACL_SUCCESS;
}

void LaunchDpuRpcKernel(sim::FuncHandle* func, void* hostArgs, size_t argsSize)
{
    DpuKernelFunc kernelFunc = sim::DpuKernelLibManager::GetInstance().GetOrLoadFunc(func->soName, func->kernelName);
    if (kernelFunc == nullptr) {
        HCCL_VM_ERROR("get DPU kernel {} from {} failed", func->kernelName, func->soName);
        return;
    }

    sim::DpuKernelLibManager::GetInstance().Launch(func->kernelName, kernelFunc, hostArgs, argsSize);
    return;
}

void LaunchAicpuDpuKernel(aclrtFuncHandle funcHandle, aclrtStream stream, void* hostArgs, size_t argsSize)
{
    uint32_t streamId = (uint32_t)(uintptr_t)stream;
    sim::FuncHandle* func = (sim::FuncHandle*)(uintptr_t)funcHandle;
    HCCL_VM_INFO(
        "kernelName:{}, soName:{} stream:{:p}, streamId is {}, argsSize is {}", func->kernelName, func->soName,
        (void*)stream, streamId, argsSize);

    // 1.DPU场景处理
    if (func->soName == DPU_KERNEL_SO) {
        LaunchDpuRpcKernel(func, hostArgs, argsSize);
        return;
    }

    // 2.非DPU场景，如HcclDpuTaskexpShmemRestore先不处理
    HCCL_VM_INFO(
        "AICPU kernel {} (so={}) skipped as no-op in simulation (consumer is defensive)", func->kernelName,
        func->soName);
    return;
}

#ifdef __cplusplus
}
#endif // __cplusplus

// ===== AIV virtual-kernel support scope begin =====
// AIV作用范围：这里保留HCCL AIV ExecuteKernelLaunch的C++ hook链路，并在真实
// aclrtLaunchKernelWithHostArgs launch点记录AIV_GRAPH任务、分配launchIndex、执行x86 AIV stub。
extern "C" bool GetPhyMemBlockByVirPtr(void* virPtr, uint64_t& offset, sim::runtime::PhyMemBlock& phyMem);

namespace {
std::atomic<uint32_t> g_aivLaunchIndex{0};
} // namespace

namespace ops_hccl {
constexpr uint32_t AIV_STUB_MAX_RANK_SIZE = 512;
constexpr uint32_t AIV_STUB_MAX_RANK_SIZE_V = 256;
constexpr uint32_t AIV_STUB_MAX_NUM_BLOCKS = 48;
constexpr int32_t AIV_STUB_TOPO_LEN = AIV_STUB_MAX_RANK_SIZE;
constexpr uint64_t AIV_STUB_GM_IN_TABLE_OFFSET = AivCommInfoLayout::GM_IN_TABLE_OFFSET;
constexpr uint64_t AIV_STUB_GM_OUT_TABLE_OFFSET = AivCommInfoLayout::GM_OUT_TABLE_OFFSET;
constexpr uint64_t AIV_STUB_TOPO_OFFSET = 32 * 1024;
constexpr uint64_t AIV_STUB_FLAG1_OFFSET = AivCommInfoLayout::FLAG1_OFFSET;
constexpr uint64_t AIV_STUB_FLAG2_OFFSET = AivCommInfoLayout::FLAG2_OFFSET;
constexpr uint64_t AIV_STUB_TAG_CLEAR_OFFSET = AivCommInfoLayout::TAG_OFFSET;
constexpr uint64_t AIV_STUB_BASE_FLAG_OFFSET = AivCommInfoLayout::BASE_FLAG_OFFSET;
constexpr uint64_t AIV_STUB_FLAG_EMPTY_OFFSET = AivCommInfoLayout::EMPTY_CLEAR_OFFSET;
constexpr uint64_t AIV_STUB_GM_OUT_PING_OFFSET = AivCommInfoLayout::PING_OFFSET;
constexpr uint64_t AIV_STUB_GM_OUT_PONG_OFFSET = AivCommInfoLayout::PONG_OFFSET;
constexpr uint64_t AIV_STUB_COMM_INFO_SIZE = AivCommInfoLayout::SIZE_BYTES;
constexpr uint64_t AIV_STUB_UB_ALIGN_SIZE = 32;
constexpr uint64_t AIV_STUB_FLAG_SLOT_SIZE = 128;
constexpr uint32_t AIV_STUB_FLAG_SLOT_PRINT_NUM = 16;
constexpr uint32_t AIV_STUB_TAG_PRINT_NUM = 16;

enum class KernelArgsType { ARGS_TYPE_SERVER = 0, ARGS_TYPE_TWO_SHOT = 1, ARGS_TYPE_DEFAULT };

struct ExtraArgs {
    uint64_t sendCounts[AIV_STUB_MAX_RANK_SIZE_V] = {};
    uint64_t sendDispls[AIV_STUB_MAX_RANK_SIZE_V] = {};
    uint64_t recvCounts[AIV_STUB_MAX_RANK_SIZE_V] = {};
    uint64_t recvDispls[AIV_STUB_MAX_RANK_SIZE_V] = {};
};

struct OpCounterInfo {
    uint64_t headCountMem = 0;
    uint64_t tailCountMem = 0;
    uint64_t addOneMem = 0;
    uint32_t memSize = 0;
    bool isEnableCounter = false;
};

struct AivOpArgs {
    HcclCMDType cmdType = HcclCMDType::HCCL_CMD_MAX;
    std::string comm = {};
    HcclComm hcclComm = nullptr;
    uint32_t numBlocks = AIV_STUB_MAX_NUM_BLOCKS;
    void* stream = nullptr;
    uint64_t beginTime = 0;
    OpCounterInfo counter = {};
    void* buffersIn = nullptr;
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t rank = 0;
    uint32_t sendRecvRemoteRank = 0;
    uint32_t rankSize = 0;
    uint64_t count = 0;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    HcclReduceOp op = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    uint32_t sliceId = 0;
    uint64_t inputSliceStride = 0;
    uint64_t outputSliceStride = 0;
    uint64_t repeatNum = 0;
    uint64_t inputRepeatStride = 0;
    uint64_t outputRepeatStride = 0;
    bool isOpBase = false;
    ExtraArgs extraArgs = {};
    uint64_t topo_[AIV_STUB_TOPO_LEN] = {0};
    AivOpArgs() {}
    KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER;
};

struct AivKernelArgs {
    const void* buffersIn = nullptr;
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t rank = 0;
    uint32_t sendRecvRemoteRank = 0;
    uint32_t rankSize = 0;
    uint64_t len = 0;
    uint32_t dataType = 0;
    uint32_t reduceOp = 0;
    uint32_t root = 0;
    uint32_t tag = 0;
    uint64_t inputSliceStride = 0;
    uint64_t outputSliceStride = 0;
    uint64_t repeatNum = 0;
    uint64_t inputRepeatStride = 0;
    uint64_t outputRepeatStride = 0;
    uint32_t numBlocks = 0;
    bool isOpBase = false;
    const void* headCountMem = nullptr;
    const void* tailCountMem = nullptr;
    const void* addOneMem = nullptr;
    uint32_t counterMemSize = 0;
    bool isEnableCounter = false;
};

struct AivExtraKernelArgs {
    const void* buffersIn = nullptr;
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t rank = 0;
    uint32_t sendRecvRemoteRank = 0;
    uint32_t rankSize = 0;
    uint64_t len = 0;
    uint32_t dataType = 0;
    uint32_t reduceOp = 0;
    uint32_t root = 0;
    uint32_t tag = 0;
    uint64_t inputSliceStride = 0;
    uint64_t outputSliceStride = 0;
    uint64_t repeatNum = 0;
    uint64_t inputRepeatStride = 0;
    uint64_t outputRepeatStride = 0;
    uint32_t numBlocks = 0;
    bool isOpBase = false;
    const void* headCountMem = nullptr;
    const void* tailCountMem = nullptr;
    const void* addOneMem = nullptr;
    uint32_t counterMemSize = 0;
    bool isEnableCounter = false;
    ExtraArgs extraArgs = {};
};

struct AivHostLaunchArgs {
    const void* buffersIn = nullptr;
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t rank = 0;
    uint32_t sendRecvRemoteRank = 0;
    uint32_t rankSize = 0;
    uint64_t len = 0;
    uint32_t dataType = 0;
    uint32_t reduceOp = 0;
    uint32_t root = 0;
    uint32_t tag = 0;
    uint64_t inputSliceStride = 0;
    uint64_t outputSliceStride = 0;
    uint64_t repeatNum = 0;
    uint64_t inputRepeatStride = 0;
    uint64_t outputRepeatStride = 0;
    uint32_t numBlocks = 0;
    bool isOpBase = false;
    const void* headCountMem = nullptr;
    const void* tailCountMem = nullptr;
    const void* addOneMem = nullptr;
    uint32_t counterMemSize = 0;
    bool isEnableCounter = false;
    ExtraArgs extraArgs = {};
    bool hasExtraArgs = false;
};

struct CheckerFuncHandleView {
    std::string funcName{};
    std::string kernelName{};
};

using AivOpKernelFunc = void (*)(
    uint8_t* buffIn, uint64_t input, uint64_t output, uint32_t rank, uint32_t sendRecvRemoteRank, uint32_t rankSize,
    uint64_t len, uint32_t dataType, uint32_t reduceOp, uint32_t root, uint32_t sliceId, uint64_t inputSliceStride,
    uint64_t outputSliceStride, uint64_t repeatNum, uint64_t inputRepeatStride, uint64_t outputRepeatStride,
    uint32_t numBlocks, bool isOpBase, uint8_t* headCountMem, uint8_t* tailCountMem, uint8_t* addOneMem,
    uint32_t counterMemSize, bool isEnableCounter);
using AivExtraOpKernelFunc = void (*)(
    uint8_t* buffIn, uint64_t input, uint64_t output, uint32_t rank, uint32_t sendRecvRemoteRank, uint32_t rankSize,
    uint64_t len, uint32_t dataType, uint32_t reduceOp, uint32_t root, uint32_t sliceId, uint64_t inputSliceStride,
    uint64_t outputSliceStride, uint64_t repeatNum, uint64_t inputRepeatStride, uint64_t outputRepeatStride,
    uint32_t numBlocks, bool isOpBase, uint8_t* headCountMem, uint8_t* tailCountMem, uint8_t* addOneMem,
    uint32_t counterMemSize, bool isEnableCounter, ExtraArgs extraArgs);
using AivEnvInitFunc = bool (*)(
    uint64_t commId, uint32_t rankId, size_t blockNum, const void* buffIn, uint32_t rankSize, uint64_t input,
    uint64_t inputSize, uint64_t output, uint64_t outputSize, uint64_t inputGlobalOffsetBase,
    uint64_t outputGlobalOffsetBase, uint64_t cclBufferSize, uint64_t aivCommInfoSize, AivSim::AivOpParam opParam);
using AivSetBlockIdxFunc = void (*)(int64_t blockIdx);
using AivDumpTasksFunc = void (*)(uint32_t launchIndex);

constexpr const char* AIV_STUB_ENV_INIT_SYMBOL = "aiv_env_init";
constexpr const char* AIV_STUB_SET_BLOCK_IDX_SYMBOL = "aiv_set_block_idx";
constexpr const char* AIV_STUB_DUMP_TASKS_SYMBOL = "aiv_dump_tasks";
constexpr const char* AIV_STUB_SO_NAME = "libhccl_aiv_kernel.so";
constexpr uint64_t INVALID_MEMORY_LAYOUT_SIZE = static_cast<uint64_t>(-1);

struct ResolvedHostPtrHandle {
    const uint8_t* hostPtr = nullptr;
    sim::runtime::PhyMemBlock phyMem = {};
    bool needRelease = false;
};

struct ResolvedKernelLaunchArgs {
    AivHostLaunchArgs args = {};
    ResolvedHostPtrHandle buffersInHandle = {};
};

struct OpMemInfoMatchInfo {
    uint64_t offsetInLayout = 0;
    uint64_t remainingSize = 0;
};

static ResolvedHostPtrHandle ResolveHostPtr(const void* devPtr)
{
    ResolvedHostPtrHandle handle{};
    if (devPtr == nullptr) {
        return handle;
    }

    sim::runtime::PhyMemBlock phyMem{};
    uint64_t offset = 0;
    if (!::GetPhyMemBlockByVirPtr(const_cast<void*>(devPtr), offset, phyMem)) {
        return handle;
    }

    handle.phyMem = phyMem;
    auto* devStartPtr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(devPtr) - offset);
    auto* hostBasePtr
        = static_cast<const uint8_t*>(sim::DeviceMemoryManager::GetInstance().GetHostPtrByDevPtr(devStartPtr));
    if (hostBasePtr == nullptr) {
        hostBasePtr = static_cast<const uint8_t*>(
            sim::DeviceMemoryManager::GetInstance().AcquirePhyMem(phyMem.name, phyMem.device_id, phyMem.size));
        if (hostBasePtr == nullptr) {
            return handle;
        }
        handle.phyMem = phyMem;
        handle.needRelease = true;
    }
    handle.hostPtr = hostBasePtr + offset;
    return handle;
}

static void ReleaseHostPtr(ResolvedHostPtrHandle& handle)
{
    if (!handle.needRelease) {
        return;
    }

    // needRelease 仅在走 AcquirePhyMem 时置位；按 size 判定是否走了复用区（与申请同一判据）
    if (sim::CommPoolPolicy::ShouldRedirect(handle.phyMem.size, sim::IsCheckOnlyMode())) {
        sim::MemoryManager::GetInstance().ReleaseMemByName(sim::CommPoolPolicy::kPoolName);
    } else {
        sim::DeviceMemoryManager::GetInstance().ReleasePhyMem(handle.phyMem.name, handle.phyMem.device_id);
    }
    handle.hostPtr = nullptr;
    handle.phyMem = {};
    handle.needRelease = false;
}

static void DumpAivExtraArgs(const ExtraArgs& extraArgs)
{
    std::ostringstream oss;
    oss << "[virtual-aiv-ExecuteKernelLaunch] extraArgs:\n";
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "  extraArgs.sendCounts[" << i << "] = " << static_cast<unsigned long long>(extraArgs.sendCounts[i])
            << '\n';
    }
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "  extraArgs.sendDispls[" << i << "] = " << static_cast<unsigned long long>(extraArgs.sendDispls[i])
            << '\n';
    }
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "  extraArgs.recvCounts[" << i << "] = " << static_cast<unsigned long long>(extraArgs.recvCounts[i])
            << '\n';
    }
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "  extraArgs.recvDispls[" << i << "] = " << static_cast<unsigned long long>(extraArgs.recvDispls[i])
            << '\n';
    }
    HCCL_VM_DEBUG("{}", oss.str());
}

static void DumpAivTopo(const uint64_t topo[AIV_STUB_TOPO_LEN])
{
    std::ostringstream oss;
    oss << "[virtual-aiv-ExecuteKernelLaunch] topo:\n";
    for (uint32_t i = 0; i < static_cast<uint32_t>(AIV_STUB_TOPO_LEN); ++i) {
        oss << "  opArgs.topo_[" << i << "] = " << static_cast<unsigned long long>(topo[i]) << '\n';
    }
    HCCL_VM_DEBUG("{}", oss.str());
}

static void DumpFlagSlots(
    std::ostringstream& oss, const uint8_t* flagBase, uint64_t byteOffset, uint32_t slotCount, const char* label)
{
    const uint64_t slotBase = byteOffset / AIV_STUB_UB_ALIGN_SIZE;
    const uint64_t slotStride = AIV_STUB_FLAG_SLOT_SIZE / AIV_STUB_UB_ALIGN_SIZE;
    oss << "    " << label << " byteOffset=0x" << std::hex << static_cast<unsigned long long>(byteOffset) << std::dec
        << " slotBase=" << static_cast<unsigned long long>(slotBase)
        << " slotStride=" << static_cast<unsigned long long>(slotStride) << " slotCount=" << slotCount << '\n';
    for (uint32_t i = 0; i < slotCount; ++i) {
        const uint64_t slotIndex = slotBase + static_cast<uint64_t>(i) * slotStride;
        const auto* slotHead = reinterpret_cast<const int32_t*>(
            flagBase + byteOffset + static_cast<uint64_t>(i) * AIV_STUB_FLAG_SLOT_SIZE);
        oss << "      slot[" << static_cast<unsigned long long>(slotIndex)
            << "] addr=" << static_cast<const void*>(slotHead) << " words=[" << slotHead[0] << ", " << slotHead[1]
            << ", " << slotHead[2] << ", " << slotHead[3] << "]\n";
    }
}

static void DumpBuffersInParsedDeviceView(const void* buffersInDev, uint32_t rankSize, uint32_t numBlocks)
{
    std::ostringstream oss;
    oss << "[virtual-aiv-ExecuteKernelLaunch] buffersIn-parse:\n";
    oss << "  [buffersIn-parse] aivCommInfoPtr base(dev) = " << buffersInDev << '\n';
    if (buffersInDev == nullptr) {
        oss << "  [buffersIn-parse] buffersIn is null, skip device-side parse dump.\n";
        HCCL_VM_DEBUG("{}", oss.str());
        return;
    }

    ResolvedHostPtrHandle buffersInHandle = ResolveHostPtr(buffersInDev);
    auto* buffersInHost = buffersInHandle.hostPtr;
    oss << "  [buffersIn-parse] aivCommInfoPtr base(host) = " << static_cast<const void*>(buffersInHost) << '\n';
    if (buffersInHost == nullptr) {
        oss << "  [buffersIn-parse] failed to translate device ptr to host ptr.\n";
        HCCL_VM_DEBUG("{}", oss.str());
        return;
    }

    const uint32_t parsedRankSize = (rankSize < AIV_STUB_MAX_RANK_SIZE) ? rankSize : AIV_STUB_MAX_RANK_SIZE;
    const auto* gmInTable = reinterpret_cast<const uint64_t*>(buffersInHost + AIV_STUB_GM_IN_TABLE_OFFSET);
    const auto* gmOutTable = reinterpret_cast<const uint64_t*>(buffersInHost + AIV_STUB_GM_OUT_TABLE_OFFSET);
    const auto* topoTable = reinterpret_cast<const uint64_t*>(buffersInHost + AIV_STUB_TOPO_OFFSET);
    const auto* flag1Base = buffersInHost + AIV_STUB_FLAG1_OFFSET;
    const auto* tagTable = reinterpret_cast<const int32_t*>(buffersInHost + AIV_STUB_TAG_CLEAR_OFFSET);
    const auto* emptyClearTable = reinterpret_cast<const int32_t*>(buffersInHost + AIV_STUB_FLAG_EMPTY_OFFSET);
    const uint64_t nonPingpongBaseFlagOffset = AIV_STUB_BASE_FLAG_OFFSET - AIV_STUB_FLAG1_OFFSET;

    oss << "  [buffersIn-parse] device-side parsed results from buffersIn (not a raw buffersIn pointer dump):\n";
    oss << "    AIV comm layout: commInfoSize=0x" << std::hex
        << static_cast<unsigned long long>(AIV_STUB_COMM_INFO_SIZE) << ", GM_OUT_TABLE=0x"
        << static_cast<unsigned long long>(AIV_STUB_GM_OUT_TABLE_OFFSET) << ", TOPO=0x"
        << static_cast<unsigned long long>(AIV_STUB_TOPO_OFFSET) << ", TAG/CLEAR=0x"
        << static_cast<unsigned long long>(AIV_STUB_TAG_CLEAR_OFFSET) << ", FLAG1=0x"
        << static_cast<unsigned long long>(AIV_STUB_FLAG1_OFFSET) << ", FLAG2=0x"
        << static_cast<unsigned long long>(AIV_STUB_FLAG2_OFFSET) << ", BASE_FLAG=0x"
        << static_cast<unsigned long long>(AIV_STUB_BASE_FLAG_OFFSET) << ", EMPTY_CLEAR=0x"
        << static_cast<unsigned long long>(AIV_STUB_FLAG_EMPTY_OFFSET) << ", PING=0x"
        << static_cast<unsigned long long>(AIV_STUB_GM_OUT_PING_OFFSET) << ", PONG=0x"
        << static_cast<unsigned long long>(AIV_STUB_GM_OUT_PONG_OFFSET) << std::dec << '\n';
    oss << "    Checker AIV mode models the complete per-rank aivCommInfo region.\n";
    oss << "    GM_IN parsed entries count=" << parsedRankSize << " from +0x" << std::hex
        << static_cast<unsigned long long>(AIV_STUB_GM_IN_TABLE_OFFSET) << std::dec << '\n';
    for (uint32_t i = 0; i < parsedRankSize; ++i) {
        oss << "      GM_IN[" << i << "] dev=0x" << std::hex << static_cast<unsigned long long>(gmInTable[i])
            << std::dec << '\n';
    }
    if (parsedRankSize == 0) {
        oss << "      rankSize is 0, device would not populate GM_IN[].\n";
    }

    oss << "    GM_OUT parsed entries count=" << parsedRankSize << " from (+0x" << std::hex
        << static_cast<unsigned long long>(AIV_STUB_GM_OUT_TABLE_OFFSET) << " table) + FLAG1_OFFSET(0x"
        << static_cast<unsigned long long>(AIV_STUB_FLAG1_OFFSET) << ')' << std::dec << '\n';
    for (uint32_t i = 0; i < parsedRankSize; ++i) {
        const uint64_t commInfoDev = gmOutTable[i];
        const uint64_t flagDev = (commInfoDev == 0) ? 0 : (commInfoDev + AIV_STUB_FLAG1_OFFSET);
        oss << "      GM_OUT[" << i << "] dev=0x" << std::hex << static_cast<unsigned long long>(flagDev)
            << " (src commInfoDev=0x" << std::hex << static_cast<unsigned long long>(commInfoDev) << std::dec << ")\n";
    }
    if (parsedRankSize == 0) {
        oss << "      rankSize is 0, device would not populate GM_OUT[].\n";
    }

    oss << "    TOPO_ parsed entries [0.." << static_cast<unsigned>(AIV_STUB_TOPO_LEN - 1) << "] from +0x" << std::hex
        << static_cast<unsigned long long>(AIV_STUB_TOPO_OFFSET) << std::dec << '\n';
    for (uint32_t i = 0; i < static_cast<uint32_t>(AIV_STUB_TOPO_LEN); ++i) {
        oss << "      TOPO_[" << i << "] = " << static_cast<unsigned long long>(topoTable[i]) << '\n';
    }

    const uint32_t barrierSlotPrintNum = (rankSize < AIV_STUB_MAX_RANK_SIZE) ? rankSize : AIV_STUB_MAX_RANK_SIZE;
    oss << "  [buffersIn-parse] follow-up work areas derived from the same base:\n";
    oss << "    FLAG1 non-pingpong GM_OUT @ +0x" << std::hex << static_cast<unsigned long long>(AIV_STUB_FLAG1_OFFSET)
        << ", flagSlotSize=" << std::dec << AIV_STUB_FLAG_SLOT_SIZE << ", baseFlagOffset relative to GM_OUT=0x"
        << std::hex << static_cast<unsigned long long>(nonPingpongBaseFlagOffset) << ", absoluteBaseFlag=0x"
        << static_cast<unsigned long long>(AIV_STUB_BASE_FLAG_OFFSET) << std::dec << '\n';
    oss << "    FLAG2 pingpong alt @ +0x" << std::hex << static_cast<unsigned long long>(AIV_STUB_FLAG2_OFFSET)
        << ", PING data @ +0x" << static_cast<unsigned long long>(AIV_STUB_GM_OUT_PING_OFFSET) << ", PONG data @ +0x"
        << static_cast<unsigned long long>(AIV_STUB_GM_OUT_PONG_OFFSET) << std::dec << '\n';
    oss << "    FLAG1 bytes=[0x0, 0x" << std::hex
        << static_cast<unsigned long long>(AIV_STUB_FLAG_EMPTY_OFFSET - AIV_STUB_FLAG1_OFFSET) << ')'
        << ", flagSlotSize=" << std::dec << AIV_STUB_FLAG_SLOT_SIZE << std::dec << '\n';
    DumpFlagSlots(oss, flag1Base, 0, AIV_STUB_FLAG_SLOT_PRINT_NUM, "FLAG1 operator slots[0..15]");
    DumpFlagSlots(
        oss, flag1Base, nonPingpongBaseFlagOffset, barrierSlotPrintNum,
        "BASE_FLAG_OFFSET - FLAG1_OFFSET barrier slots");

    oss << "    TAG/CLEAR ints[0.." << (AIV_STUB_TAG_PRINT_NUM - 1) << "] @ +0x" << std::hex
        << static_cast<unsigned long long>(AIV_STUB_TAG_CLEAR_OFFSET) << std::dec << '\n';
    oss << "      Checker AIV mode keeps tag_ fixed at 1; ping-pong kernels therefore select FLAG2 and PONG.\n";
    for (uint32_t i = 0; i < AIV_STUB_TAG_PRINT_NUM; ++i) {
        oss << "      TAG_CLEAR[" << i << "] = " << tagTable[i] << '\n';
    }
    oss << "    EMPTY_CLEAR ints[0.." << (AIV_STUB_TAG_PRINT_NUM - 1) << "] @ +0x" << std::hex
        << static_cast<unsigned long long>(AIV_STUB_FLAG_EMPTY_OFFSET) << std::dec << '\n';
    for (uint32_t i = 0; i < AIV_STUB_TAG_PRINT_NUM; ++i) {
        oss << "      EMPTY_CLEAR[" << i << "] = " << emptyClearTable[i] << '\n';
    }

    HCCL_VM_DEBUG("{}", oss.str());
    ReleaseHostPtr(buffersInHandle);
}

static void DumpLaunchKernelCfg(const aclrtLaunchKernelCfg* cfg)
{
    std::ostringstream oss;
    oss << "[virtual-aiv-aclrtLaunchKernelWithHostArgs] cfg:\n";
    oss << "  cfg = " << cfg << '\n';
    if (cfg == nullptr) {
        HCCL_VM_DEBUG("{}", oss.str());
        return;
    }

    oss << "  cfg->attrs = " << cfg->attrs << '\n';
    oss << "  cfg->numAttrs = " << cfg->numAttrs << '\n';
    for (size_t i = 0; i < cfg->numAttrs; ++i) {
        const auto& attr = cfg->attrs[i];
        oss << "    cfg->attrs[" << i << "].id = " << static_cast<int>(attr.id) << '\n';
        switch (attr.id) {
            case ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE:
                oss << "      schemMode = " << static_cast<unsigned>(attr.value.schemMode) << '\n';
                break;
            case ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE:
                oss << "      dynUBufSize = " << attr.value.dynUBufSize << '\n';
                break;
            case ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE:
                oss << "      engineType = " << static_cast<unsigned>(attr.value.engineType) << '\n';
                break;
            case ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET:
                oss << "      blockDimOffset = " << attr.value.blockDimOffset << '\n';
                break;
            case ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH:
                oss << "      isBlockTaskPrefetch = " << static_cast<unsigned>(attr.value.isBlockTaskPrefetch) << '\n';
                break;
            case ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP:
                oss << "      isDataDump = " << static_cast<unsigned>(attr.value.isDataDump) << '\n';
                break;
            case ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT:
                oss << "      timeout = " << static_cast<unsigned>(attr.value.timeout) << "(s)\n";
                break;
            case ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US:
                oss << "      timeoutUs.low = " << attr.value.timeoutUs.timeoutLow << '\n';
                oss << "      timeoutUs.high = " << attr.value.timeoutUs.timeoutHigh << '\n';
                break;
            default:
                oss << "      raw rsv = [" << attr.value.rsv[0] << ", " << attr.value.rsv[1] << ", "
                    << attr.value.rsv[2] << ", " << attr.value.rsv[3] << "]\n";
                break;
        }
    }
    HCCL_VM_DEBUG("{}", oss.str());
}

static void DumpPlaceHolderArray(const aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
{
    std::ostringstream oss;
    oss << "[virtual-aiv-aclrtLaunchKernelWithHostArgs] placeHolderArray:\n";
    oss << "  placeHolderArray = " << placeHolderArray << '\n';
    oss << "  placeHolderNum = " << placeHolderNum << '\n';
    if (placeHolderArray == nullptr) {
        HCCL_VM_DEBUG("{}", oss.str());
        return;
    }

    for (size_t i = 0; i < placeHolderNum; ++i) {
        oss << "    placeHolderArray[" << i << "].addrOffset = " << placeHolderArray[i].addrOffset << '\n';
        oss << "    placeHolderArray[" << i << "].dataOffset = " << placeHolderArray[i].dataOffset << '\n';
    }
    HCCL_VM_DEBUG("{}", oss.str());
}

static bool IsAivExtraArgsCmdType(HcclCMDType cmdType) { return cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV; }

static void CopyCommonKernelArgsToHostArgs(const AivKernelArgs& src, AivHostLaunchArgs& dst)
{
    dst.buffersIn = src.buffersIn;
    dst.input = src.input;
    dst.output = src.output;
    dst.rank = src.rank;
    dst.sendRecvRemoteRank = src.sendRecvRemoteRank;
    dst.rankSize = src.rankSize;
    dst.len = src.len;
    dst.dataType = src.dataType;
    dst.reduceOp = src.reduceOp;
    dst.root = src.root;
    dst.tag = src.tag;
    dst.inputSliceStride = src.inputSliceStride;
    dst.outputSliceStride = src.outputSliceStride;
    dst.repeatNum = src.repeatNum;
    dst.inputRepeatStride = src.inputRepeatStride;
    dst.outputRepeatStride = src.outputRepeatStride;
    dst.numBlocks = src.numBlocks;
    dst.isOpBase = src.isOpBase;
    dst.headCountMem = src.headCountMem;
    dst.tailCountMem = src.tailCountMem;
    dst.addOneMem = src.addOneMem;
    dst.counterMemSize = src.counterMemSize;
    dst.isEnableCounter = src.isEnableCounter;
}

static void CopyCommonKernelArgsToHostArgs(const AivExtraKernelArgs& src, AivHostLaunchArgs& dst)
{
    dst.buffersIn = src.buffersIn;
    dst.input = src.input;
    dst.output = src.output;
    dst.rank = src.rank;
    dst.sendRecvRemoteRank = src.sendRecvRemoteRank;
    dst.rankSize = src.rankSize;
    dst.len = src.len;
    dst.dataType = src.dataType;
    dst.reduceOp = src.reduceOp;
    dst.root = src.root;
    dst.tag = src.tag;
    dst.inputSliceStride = src.inputSliceStride;
    dst.outputSliceStride = src.outputSliceStride;
    dst.repeatNum = src.repeatNum;
    dst.inputRepeatStride = src.inputRepeatStride;
    dst.outputRepeatStride = src.outputRepeatStride;
    dst.numBlocks = src.numBlocks;
    dst.isOpBase = src.isOpBase;
    dst.headCountMem = src.headCountMem;
    dst.tailCountMem = src.tailCountMem;
    dst.addOneMem = src.addOneMem;
    dst.counterMemSize = src.counterMemSize;
    dst.isEnableCounter = src.isEnableCounter;
}

static bool ParseAivHostLaunchArgs(const void* hostArgs, size_t argsSize, AivHostLaunchArgs& parsedArgs)
{
    parsedArgs = {};
    if (hostArgs == nullptr) {
        HCCL_VM_ERROR("hostArgs is null, can not virtual execute kernel.");
        return false;
    }

    // The HCCL implementation passes the exact argument structure size.  The
    // extended structure is used only by ALLTOALLV and is self-identifying by
    // its trailing ExtraArgs payload, so no operation type context is needed
    // here.
    if (argsSize >= sizeof(AivExtraKernelArgs)) {
        HCCL_VM_WARN(
            "parse AIV host args as AivExtraKernelArgs, argsSize={}, "
            "sizeof(AivExtraKernelArgs)={}",
            argsSize, sizeof(AivExtraKernelArgs));
        const auto& rawArgs = *static_cast<const AivExtraKernelArgs*>(hostArgs);
        CopyCommonKernelArgsToHostArgs(rawArgs, parsedArgs);
        parsedArgs.extraArgs = rawArgs.extraArgs;
        parsedArgs.hasExtraArgs = true;
        return true;
    }

    if (argsSize < sizeof(AivKernelArgs)) {
        HCCL_VM_ERROR(
            "argsSize({}) < sizeof(AivKernelArgs)({}) "
            "stop virtual execution.",
            argsSize, sizeof(AivKernelArgs));
        return false;
    }
    const auto& rawArgs = *static_cast<const AivKernelArgs*>(hostArgs);
    CopyCommonKernelArgsToHostArgs(rawArgs, parsedArgs);
    parsedArgs.hasExtraArgs = false;
    return true;
}

static void DumpHostLaunchArgs(const AivHostLaunchArgs* hostArgs, size_t argsSize)
{
    std::ostringstream oss;
    oss << "[virtual-aiv-aclrtLaunchKernelWithHostArgs] hostArgs:\n";
    oss << "  hostArgs = " << hostArgs << '\n';
    oss << "  argsSize = " << argsSize << '\n';
    oss << "  sizeof(AivKernelArgs) = " << sizeof(AivKernelArgs) << '\n';
    oss << "  sizeof(AivExtraKernelArgs) = " << sizeof(AivExtraKernelArgs) << '\n';
    oss << "  sizeof(AivHostLaunchArgs normalized) = " << sizeof(AivHostLaunchArgs) << '\n';
    if (hostArgs == nullptr) {
        HCCL_VM_DEBUG("{}", oss.str());
        return;
    }

    oss << "  hostArgs.buffersIn = " << hostArgs->buffersIn << '\n';
    oss << "  hostArgs.input = 0x" << std::hex << static_cast<unsigned long long>(hostArgs->input) << std::dec << '\n';
    oss << "  hostArgs.output = 0x" << std::hex << static_cast<unsigned long long>(hostArgs->output) << std::dec
        << '\n';
    oss << "  hostArgs.rank = " << hostArgs->rank << '\n';
    oss << "  hostArgs.sendRecvRemoteRank = " << hostArgs->sendRecvRemoteRank << '\n';
    oss << "  hostArgs.rankSize = " << hostArgs->rankSize << '\n';
    oss << "  hostArgs.len = " << static_cast<unsigned long long>(hostArgs->len) << '\n';
    oss << "  hostArgs.dataType = " << hostArgs->dataType << '\n';
    oss << "  hostArgs.reduceOp = " << hostArgs->reduceOp << '\n';
    oss << "  hostArgs.root = " << hostArgs->root << '\n';
    oss << "  hostArgs.tag = " << hostArgs->tag << '\n';
    oss << "  hostArgs.inputSliceStride = " << static_cast<unsigned long long>(hostArgs->inputSliceStride) << '\n';
    oss << "  hostArgs.outputSliceStride = " << static_cast<unsigned long long>(hostArgs->outputSliceStride) << '\n';
    oss << "  hostArgs.repeatNum = " << static_cast<unsigned long long>(hostArgs->repeatNum) << '\n';
    oss << "  hostArgs.inputRepeatStride = " << static_cast<unsigned long long>(hostArgs->inputRepeatStride) << '\n';
    oss << "  hostArgs.outputRepeatStride = " << static_cast<unsigned long long>(hostArgs->outputRepeatStride) << '\n';
    oss << "  hostArgs.numBlocks = " << hostArgs->numBlocks << '\n';
    oss << "  hostArgs.isOpBase = " << static_cast<int>(hostArgs->isOpBase) << '\n';
    oss << "  hostArgs.headCountMem = " << hostArgs->headCountMem << '\n';
    oss << "  hostArgs.tailCountMem = " << hostArgs->tailCountMem << '\n';
    oss << "  hostArgs.addOneMem = " << hostArgs->addOneMem << '\n';
    oss << "  hostArgs.counterMemSize = " << hostArgs->counterMemSize << '\n';
    oss << "  hostArgs.isEnableCounter = " << static_cast<int>(hostArgs->isEnableCounter) << '\n';
    oss << "  hostArgs.hasExtraArgs = " << static_cast<int>(hostArgs->hasExtraArgs) << '\n';
    HCCL_VM_DEBUG("{}", oss.str());
    if (hostArgs->hasExtraArgs) {
        DumpAivExtraArgs(hostArgs->extraArgs);
    }
}

static std::string InferKernelNameFromFuncHandle(aclrtFuncHandle funcHandle)
{
    std::ostringstream oss;
    oss << "[virtual-aiv-aclrtLaunchKernelWithHostArgs] InferKernelNameFromFuncHandle:\n";
    oss << "  funcHandle = " << funcHandle << '\n';
    if (funcHandle == nullptr) {
        HCCL_VM_DEBUG("{}", oss.str());
        return {};
    }

    char funcNameBuffer[256] = {0};
    aclError funcNameRet = aclrtGetFunctionName(funcHandle, sizeof(funcNameBuffer), funcNameBuffer);
    oss << "  aclrtGetFunctionName(funcHandle) ret = " << static_cast<int>(funcNameRet) << '\n';
    oss << "  aclrtGetFunctionName(funcHandle) funcName = " << (funcNameBuffer[0] == '\0' ? "<empty>" : funcNameBuffer)
        << '\n';

    std::string resolvedKernelName;

    if (funcNameRet == ACL_SUCCESS) {
        const auto* funcHandleView = reinterpret_cast<const CheckerFuncHandleView*>(funcHandle);
        oss << "  [InferKernelNameFromFuncHandle] current checker aclrtBinaryGetFunction stores funcHandle as "
            << "sim::FuncHandle*.\n";
        oss << "  [InferKernelNameFromFuncHandle] funcHandle->funcName = "
            << (funcHandleView->funcName.empty() ? "<empty>" : funcHandleView->funcName.c_str()) << '\n';
        oss << "  [InferKernelNameFromFuncHandle] funcHandle->kernelName = "
            << (funcHandleView->kernelName.empty() ? "<empty>" : funcHandleView->kernelName.c_str()) << '\n';
        if (!funcHandleView->kernelName.empty()) {
            resolvedKernelName = funcHandleView->kernelName;
        } else if (!funcHandleView->funcName.empty()) {
            resolvedKernelName = funcHandleView->funcName;
        }
    }

    if (resolvedKernelName.empty()) {
        resolvedKernelName = std::string(funcNameBuffer);
    }
    oss << "  resolvedKernelName = " << (resolvedKernelName.empty() ? "<empty>" : resolvedKernelName.c_str()) << '\n';
    HCCL_VM_DEBUG("{}", oss.str());
    return resolvedKernelName;
}

static std::string GetAivLibraryPath(const std::string& soName, const std::string& kernelName)
{
    const char* kernelNameCStr = kernelName.empty() ? "<empty>" : kernelName.c_str();
    const std::string& installDir = InstallPath::GetHcclVmInstallAbsPath();
    if (installDir.empty()) {
        HCCL_VM_ERROR("install root is empty, can not locate {} for kernel {}", soName, kernelNameCStr);
        return {};
    }

    std::error_code ec;
    const fs::path soPath = fs::path(installDir) / "lib" / GetArchStr() / soName;
    if (!fs::exists(soPath, ec)) {
        if (ec) {
            HCCL_VM_ERROR(
                "failed to stat aiv library path, kernel={}, installDir={}, so={}, err={}", kernelNameCStr, installDir,
                soPath.string(), ec.message());
            return {};
        }
        HCCL_VM_ERROR(
            "missing aiv stub shared library, kernel={}, installDir={}, expectedSo={}", kernelNameCStr, installDir,
            soPath.string());
        return {};
    }

    return soPath.string();
}

static ResolvedKernelLaunchArgs PrepareResolvedKernelLaunchArgs(const AivHostLaunchArgs& rawArgs)
{
    ResolvedKernelLaunchArgs resolvedArgs{};
    resolvedArgs.args = rawArgs;
    // before launch, trans dev_map_addr to vir addr
    if (rawArgs.input != 0) {
        resolvedArgs.args.input = sim::GetVirPtrByDevPtr(rawArgs.input);
    }
    if (rawArgs.output != 0) {
        resolvedArgs.args.output = sim::GetVirPtrByDevPtr(rawArgs.output);
    }
    resolvedArgs.buffersInHandle = ResolveHostPtr(rawArgs.buffersIn);
    if (resolvedArgs.buffersInHandle.hostPtr == nullptr && rawArgs.buffersIn != nullptr) {
        HCCL_VM_ERROR(
            "buffersIn translation failed, translated host ptr is null, "
            "raw={:p}",
            rawArgs.buffersIn);
    }
    resolvedArgs.args.buffersIn = resolvedArgs.buffersInHandle.hostPtr;
    return resolvedArgs;
}

enum class OpMemInfoLookupStatus { RESOLVED, MISSING, INVALID };

static bool
TryMatchOpMemInfoRange(uint64_t queryVirtualAddr, uint64_t baseAddr, uint64_t size, OpMemInfoMatchInfo& matchInfo)
{
    matchInfo = {};
    if (size == 0 || queryVirtualAddr < baseAddr) {
        return false;
    }

    const uint64_t offsetInLayout = queryVirtualAddr - baseAddr;
    if (offsetInLayout >= size) {
        return false;
    }

    matchInfo.offsetInLayout = offsetInLayout;
    matchInfo.remainingSize = size - offsetInLayout;
    return true;
}

static bool StartsWith(const std::string& value, const char* prefix) { return value.rfind(prefix, 0) == 0; }

static bool IsAivBroadcastKernel(const std::string& kernelName) { return StartsWith(kernelName, "aiv_broadcast_"); }

static bool IsAivScatterKernel(const std::string& kernelName) { return StartsWith(kernelName, "aiv_scatter_"); }

static OpMemInfoLookupStatus LookupOpMemInfoByVirtualAddr(
    uint32_t rankId, uint64_t queryVirtualAddr, BufferType bufType, const sim::operation::OpMemInfoTab& opMemInfo,
    OpMemInfoMatchInfo& matchInfo)
{
    matchInfo = {};
    if (queryVirtualAddr == 0) {
        return OpMemInfoLookupStatus::RESOLVED;
    }

    uint64_t baseAddr = 0;
    uint64_t size = 0;
    if (bufType == BufferType::INPUT) {
        baseAddr = opMemInfo.inputAddr;
        size = opMemInfo.inputSize;
    } else if (bufType == BufferType::OUTPUT) {
        baseAddr = opMemInfo.outputAddr;
        size = opMemInfo.outputSize;
    } else {
        HCCL_VM_ERROR(
            "unsupported opMemInfo buffer type, rankId={}, baseAddr=0x{:x}, bufType={}", rankId, queryVirtualAddr,
            static_cast<uint32_t>(bufType));
        return OpMemInfoLookupStatus::INVALID;
    }

    if (!TryMatchOpMemInfoRange(queryVirtualAddr, baseAddr, size, matchInfo)) {
        HCCL_VM_DEBUG(
            "queryAddr=0x{:x} did not match opMemInfo range. "
            "rankId={}, bufType={}, opMemBase=0x{:x}, opMemSize={}",
            queryVirtualAddr, rankId, static_cast<uint32_t>(bufType), baseAddr, size);
        return OpMemInfoLookupStatus::MISSING;
    }

    return OpMemInfoLookupStatus::RESOLVED;
}

static bool ResolveVirtualAivBufferSizes(
    const std::string& kernelName, ResolvedKernelLaunchArgs& resolvedArgs, uint64_t commId, uint32_t deviceId,
    uint64_t& inputSize, uint64_t& outputSize, uint64_t& inputGlobalOffsetBase, uint64_t& outputGlobalOffsetBase,
    uint64_t& cclBufferSize, uint64_t& aivCommInfoSize, uint32_t replayOpDetailId)
{
    inputGlobalOffsetBase = 0;
    outputGlobalOffsetBase = 0;
    cclBufferSize = 0;
    aivCommInfoSize = AIV_STUB_COMM_INFO_SIZE;

    sim::operation::OpMemInfoTab opMemInfo{};
    const bool hasOpMemInfo
        = (replayOpDetailId != 0 ? sim::operation::QueryOpMemInfoByOpDetailId(replayOpDetailId, opMemInfo) :
                                   sim::operation::QueryCurrentOpMemInfo(commId, deviceId, opMemInfo))
          == 0;
    if (!hasOpMemInfo) {
        HCCL_VM_ERROR(
            "failed to query current opMemInfo for kernel {}, rank={}, commId={}, deviceId={}", kernelName,
            resolvedArgs.args.rank, commId, deviceId);
    }

    // 仅当数据面 buffer（input 与 output）同时为 0 时，才等同 count=0 的空算子：
    // 打警告后直接跳过 AIV 执行，不再走后续 baseAddr 匹配与 insert task。
    // 图模式（replayOpDetailId != 0）与普通模式（== 0）都在此处统一短路。
    //
    // 不可用"任一 buffer 为 0"作判据：部分算子的某个 buffer 语义上本就不使用，
    // size 为 0 属正常且该算子仍必须执行，否则数据面行为缺失：
    //   - Scatter 非 root rank：input 只由 root 使用（level0 桩对非 root 记 inputSize=0），
    //     但每个 rank 都要把 root 写入自己 CCL 区的数据搬进 output；
    //   - Send：output 不使用；Recv：input 不使用。
    // 这些场景分别由下方 IsAivScatterKernel / IsAivBroadcastKernel 的缺失分支处理。
    //
    // cclSize 不参与本判断：CCL buffer 非法属错误而非空算子，由下方 cclAddr/cclSize 校验显式报错。
    if (hasOpMemInfo && opMemInfo.inputSize == 0 && opMemInfo.outputSize == 0) {
        HCCL_VM_WARN(
            "skip AIV kernel execution: zero-size buffer recorded, kernel={}, rank={}, commId={}, deviceId={}, "
            "opDetailId={}, inputAddr=0x{:x}, inputSize={}, outputAddr=0x{:x}, outputSize={}, cclAddr=0x{:x}, "
            "cclSize={} (equivalent to count=0)",
            kernelName, resolvedArgs.args.rank, commId, deviceId, opMemInfo.opDetailId, opMemInfo.inputAddr,
            opMemInfo.inputSize, opMemInfo.outputAddr, opMemInfo.outputSize, opMemInfo.cclAddr, opMemInfo.cclSize);
        return true;
    }

    const uint64_t inputAddr = resolvedArgs.args.input;
    OpMemInfoMatchInfo inputMatchInfo{};
    const OpMemInfoLookupStatus inputStatus
        = hasOpMemInfo ? LookupOpMemInfoByVirtualAddr(
                             resolvedArgs.args.rank, inputAddr, BufferType::INPUT, opMemInfo, inputMatchInfo) :
                         OpMemInfoLookupStatus::MISSING;
    if (inputStatus == OpMemInfoLookupStatus::MISSING) {
        if (IsAivScatterKernel(kernelName) && resolvedArgs.args.rank != resolvedArgs.args.root) {
            HCCL_VM_INFO(
                "skip missing input opMemInfo for kernel {}, rank={}, root={}, baseAddr=0x{:x}", kernelName,
                resolvedArgs.args.rank, resolvedArgs.args.root, inputAddr);
            resolvedArgs.args.input = 0;
            inputSize = 0;
        } else {
            HCCL_VM_ERROR(
                "expected exactly one opMemInfo range, rankId={}, baseAddr=0x{:x}, bufType={}, matchedCount={}",
                resolvedArgs.args.rank, inputAddr, static_cast<uint32_t>(BufferType::INPUT), 0);
            inputSize = INVALID_MEMORY_LAYOUT_SIZE;
        }
    } else if (inputStatus == OpMemInfoLookupStatus::INVALID) {
        inputSize = INVALID_MEMORY_LAYOUT_SIZE;
    } else {
        inputSize = inputMatchInfo.remainingSize;
        inputGlobalOffsetBase = inputMatchInfo.offsetInLayout;
    }

    const uint64_t outputAddr = resolvedArgs.args.output;
    OpMemInfoMatchInfo outputMatchInfo{};
    const OpMemInfoLookupStatus outputStatus
        = hasOpMemInfo ? LookupOpMemInfoByVirtualAddr(
                             resolvedArgs.args.rank, outputAddr, BufferType::OUTPUT, opMemInfo, outputMatchInfo) :
                         OpMemInfoLookupStatus::MISSING;
    if (outputStatus == OpMemInfoLookupStatus::MISSING) {
        if (IsAivBroadcastKernel(kernelName)) {
            HCCL_VM_INFO(
                "skip missing output opMemInfo for kernel {}, rank={}, baseAddr=0x{:x}", kernelName,
                resolvedArgs.args.rank, outputAddr);
            resolvedArgs.args.output = 0;
            outputSize = 0;
        } else {
            HCCL_VM_ERROR(
                "expected exactly one opMemInfo range, rankId={}, baseAddr=0x{:x}, bufType={}, matchedCount={}",
                resolvedArgs.args.rank, outputAddr, static_cast<uint32_t>(BufferType::OUTPUT), 0);
            outputSize = INVALID_MEMORY_LAYOUT_SIZE;
        }
    } else if (outputStatus == OpMemInfoLookupStatus::INVALID) {
        outputSize = INVALID_MEMORY_LAYOUT_SIZE;
    } else {
        outputSize = outputMatchInfo.remainingSize;
        outputGlobalOffsetBase = outputMatchInfo.offsetInLayout;
    }

    if (!hasOpMemInfo) {
        cclBufferSize = INVALID_MEMORY_LAYOUT_SIZE;
        return false;
    }
    if (opMemInfo.cclAddr == 0 || opMemInfo.cclSize == 0) {
        HCCL_VM_ERROR(
            "current opMemInfo has invalid CCL buffer, kernel={}, rank={}, commId={}, deviceId={}, "
            "opMemId={}, opDetailId={}, cclAddr=0x{:x}, cclSize={}",
            kernelName, resolvedArgs.args.rank, commId, deviceId, opMemInfo.id, opMemInfo.opDetailId, opMemInfo.cclAddr,
            opMemInfo.cclSize);
        cclBufferSize = INVALID_MEMORY_LAYOUT_SIZE;
        return false;
    }
    cclBufferSize = opMemInfo.cclSize;
    return false;
}

static void DumpVirtualKernelExtraArgsWithSource(std::ostringstream& oss, const ExtraArgs& extraArgs)
{
    oss << "    kernelFunc.extraArgs <- aclrtLaunchKernelWithHostArgs(hostArgs->extraArgs)\n";
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "      kernelFunc.extraArgs.sendCounts[" << i
            << "] = " << static_cast<unsigned long long>(extraArgs.sendCounts[i])
            << " <- hostArgs->extraArgs.sendCounts[" << i << "]\n";
    }
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "      kernelFunc.extraArgs.sendDispls[" << i
            << "] = " << static_cast<unsigned long long>(extraArgs.sendDispls[i])
            << " <- hostArgs->extraArgs.sendDispls[" << i << "]\n";
    }
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "      kernelFunc.extraArgs.recvCounts[" << i
            << "] = " << static_cast<unsigned long long>(extraArgs.recvCounts[i])
            << " <- hostArgs->extraArgs.recvCounts[" << i << "]\n";
    }
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "      kernelFunc.extraArgs.recvDispls[" << i
            << "] = " << static_cast<unsigned long long>(extraArgs.recvDispls[i])
            << " <- hostArgs->extraArgs.recvDispls[" << i << "]\n";
    }
}

static void DumpVirtualKernelFuncArgs(
    const std::string& kernelName, uint32_t numBlocks, const AivHostLaunchArgs& rawArgs,
    const ResolvedKernelLaunchArgs& resolvedArgs)
{
    std::ostringstream oss;
    oss << "[virtual-aiv-VirtualExecuteAivKernel] kernelFunc shared args:\n";
    oss << "    launchContext.kernelName = " << kernelName << " <- aclrtLaunchKernelWithHostArgs(funcHandle)\n";
    oss << "    launchContext.numBlocks = " << numBlocks << " <- aclrtLaunchKernelWithHostArgs(numBlocks)\n";
    oss << "    kernelFunc.buffIn = " << resolvedArgs.args.buffersIn
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->buffersIn=" << rawArgs.buffersIn
        << "), checker translates only buffersIn to host address\n";
    oss << "    kernelFunc.input = 0x" << std::hex << static_cast<unsigned long long>(resolvedArgs.args.input)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->input=0x" << static_cast<unsigned long long>(rawArgs.input)
        << "), checker translates input to global virtual address\n"
        << std::dec;
    oss << "    kernelFunc.output = 0x" << std::hex << static_cast<unsigned long long>(resolvedArgs.args.output)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->output=0x" << static_cast<unsigned long long>(rawArgs.output)
        << "), checker translates output to global virtual address\n"
        << std::dec;
    oss << "    kernelFunc.rank = " << resolvedArgs.args.rank << " <- aclrtLaunchKernelWithHostArgs(hostArgs->rank)\n";
    oss << "    kernelFunc.sendRecvRemoteRank = " << resolvedArgs.args.sendRecvRemoteRank
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->sendRecvRemoteRank)\n";
    oss << "    kernelFunc.rankSize = " << resolvedArgs.args.rankSize
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->rankSize)\n";
    oss << "    kernelFunc.len = " << static_cast<unsigned long long>(resolvedArgs.args.len)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->len)\n";
    oss << "    kernelFunc.dataType = " << resolvedArgs.args.dataType
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->dataType)\n";
    oss << "    kernelFunc.reduceOp = " << resolvedArgs.args.reduceOp
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->reduceOp)\n";
    oss << "    kernelFunc.root = " << resolvedArgs.args.root << " <- aclrtLaunchKernelWithHostArgs(hostArgs->root)\n";
    oss << "    kernelFunc.sliceId = " << resolvedArgs.args.tag << " <- aclrtLaunchKernelWithHostArgs(hostArgs->tag)\n";
    oss << "    kernelFunc.inputSliceStride = " << static_cast<unsigned long long>(resolvedArgs.args.inputSliceStride)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->inputSliceStride)\n";
    oss << "    kernelFunc.outputSliceStride = " << static_cast<unsigned long long>(resolvedArgs.args.outputSliceStride)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->outputSliceStride)\n";
    oss << "    kernelFunc.repeatNum = " << static_cast<unsigned long long>(resolvedArgs.args.repeatNum)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->repeatNum)\n";
    oss << "    kernelFunc.inputRepeatStride = " << static_cast<unsigned long long>(resolvedArgs.args.inputRepeatStride)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->inputRepeatStride)\n";
    oss << "    kernelFunc.outputRepeatStride = "
        << static_cast<unsigned long long>(resolvedArgs.args.outputRepeatStride)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->outputRepeatStride)\n";
    oss << "    kernelFunc.numBlocks = " << resolvedArgs.args.numBlocks
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->numBlocks)\n";
    oss << "    kernelFunc.isOpBase = " << (resolvedArgs.args.isOpBase ? "true" : "false")
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->isOpBase)\n";
    oss << "    kernelFunc.headCountMem = " << resolvedArgs.args.headCountMem
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->headCountMem=" << rawArgs.headCountMem
        << "), checker keeps raw headCountMem pointer unchanged\n";
    oss << "    kernelFunc.tailCountMem = " << resolvedArgs.args.tailCountMem
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->tailCountMem=" << rawArgs.tailCountMem
        << "), checker keeps raw tailCountMem pointer unchanged\n";
    oss << "    kernelFunc.addOneMem = " << resolvedArgs.args.addOneMem
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->addOneMem=" << rawArgs.addOneMem
        << "), checker keeps raw addOneMem pointer unchanged\n";
    oss << "    kernelFunc.counterMemSize = " << resolvedArgs.args.counterMemSize
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->counterMemSize)\n";
    oss << "    kernelFunc.isEnableCounter = " << (resolvedArgs.args.isEnableCounter ? "true" : "false")
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->isEnableCounter)\n";
    if (resolvedArgs.args.hasExtraArgs) {
        DumpVirtualKernelExtraArgsWithSource(oss, resolvedArgs.args.extraArgs);
    } else {
        oss << "    kernelFunc.extraArgs <- not present in aclrtLaunchKernelWithHostArgs host args\n";
    }
    HCCL_VM_DEBUG("{}", oss.str());
}

struct VirtualAivLibrary {
    void* handle = nullptr;
    std::string soName{};
    std::string soPath{};
    AivEnvInitFunc envInit = nullptr;
    AivSetBlockIdxFunc setBlockIdx = nullptr;
    AivDumpTasksFunc dumpTasks = nullptr;
};

static void CloseVirtualAivLibrary(VirtualAivLibrary& lib)
{
    if (lib.handle != nullptr) {
        dlclose(lib.handle);
        lib.handle = nullptr;
    }

    lib.envInit = nullptr;
    lib.setBlockIdx = nullptr;
    lib.dumpTasks = nullptr;
}

static VirtualAivLibrary LoadVirtualAivLibrary(const std::string& soName, const std::string& kernelName)
{
    VirtualAivLibrary lib{};
    lib.soName = soName;
    if (lib.soName.empty()) {
        HCCL_VM_ERROR("empty soName for kernel {}", kernelName.empty() ? "<empty>" : kernelName.c_str());
        return lib;
    }

    lib.soPath = GetAivLibraryPath(lib.soName, kernelName);
    if (lib.soPath.empty()) {
        return lib;
    }

    dlerror();
    lib.handle = dlopen(lib.soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    const char* dlopenErr = dlerror();
    if (lib.handle == nullptr || dlopenErr != nullptr) {
        HCCL_VM_ERROR("dlopen {} failed, err = {}", lib.soPath, dlopenErr == nullptr ? "unknown" : dlopenErr);
        CloseVirtualAivLibrary(lib);
        return lib;
    }

    dlerror();
    lib.envInit = reinterpret_cast<AivEnvInitFunc>(dlsym(lib.handle, AIV_STUB_ENV_INIT_SYMBOL));
    const char* envInitErr = dlerror();
    if (lib.envInit == nullptr || envInitErr != nullptr) {
        HCCL_VM_ERROR(
            "dlsym {} from {} failed, err = {}", AIV_STUB_ENV_INIT_SYMBOL, lib.soPath,
            envInitErr == nullptr ? "unknown" : envInitErr);
        lib.envInit = nullptr;
    }

    dlerror();
    lib.setBlockIdx = reinterpret_cast<AivSetBlockIdxFunc>(dlsym(lib.handle, AIV_STUB_SET_BLOCK_IDX_SYMBOL));
    const char* setBlockIdxErr = dlerror();
    if (lib.setBlockIdx == nullptr || setBlockIdxErr != nullptr) {
        HCCL_VM_ERROR(
            "dlsym {} from {} failed, err = {}", AIV_STUB_SET_BLOCK_IDX_SYMBOL, lib.soPath,
            setBlockIdxErr == nullptr ? "unknown" : setBlockIdxErr);
        lib.setBlockIdx = nullptr;
    }

    dlerror();
    lib.dumpTasks = reinterpret_cast<AivDumpTasksFunc>(dlsym(lib.handle, AIV_STUB_DUMP_TASKS_SYMBOL));
    const char* dumpTasksErr = dlerror();
    if (lib.dumpTasks == nullptr || dumpTasksErr != nullptr) {
        HCCL_VM_ERROR(
            "dlsym {} from {} failed, err = {}", AIV_STUB_DUMP_TASKS_SYMBOL, lib.soPath,
            dumpTasksErr == nullptr ? "unknown" : dumpTasksErr);
        lib.dumpTasks = nullptr;
    }

    return lib;
}

class VirtualAivLibraryManager {
public:
    static VirtualAivLibraryManager& GetInstance()
    {
        static VirtualAivLibraryManager instance;
        return instance;
    }

    VirtualAivLibraryManager(const VirtualAivLibraryManager&) = delete;
    VirtualAivLibraryManager& operator=(const VirtualAivLibraryManager&) = delete;

    // WARNING: This manager intentionally has no internal synchronization. The
    // current simulator guarantees that AIV launches are time-ordered. Add
    // synchronization here before allowing concurrent AIV launch entry.
    VirtualAivLibrary* GetOrLoad(const std::string& soName, const std::string& kernelName)
    {
        if (library_.handle == nullptr) {
            library_ = LoadVirtualAivLibrary(soName, kernelName);
        } else if (library_.soName != soName) {
            HCCL_VM_ERROR(
                "AIV library {} is already loaded; refusing to replace it with {} during one simulator lifecycle",
                library_.soName, soName);
            return nullptr;
        }
        return &library_;
    }

    ~VirtualAivLibraryManager()
    {
        // For the preloaded level2 DSO this runs at normal process shutdown, or
        // earlier if level2 itself is explicitly unloaded.
        CloseVirtualAivLibrary(library_);
    }

private:
    VirtualAivLibraryManager() = default;

    VirtualAivLibrary library_{};
};

static aclError VirtualExecuteAivKernel(
    const std::string& kernelName, const std::string& soName, uint32_t numBlocks, const AivHostLaunchArgs& rawArgs,
    uint32_t launchIndex, uint64_t commId, uint32_t deviceId, uint32_t replayOpDetailId, bool& isNoop)
{
    isNoop = false;
    // The function-local singleton loads the AIV DSO on the first launch and
    // keeps it alive for all later launches in the same simulator lifecycle.
    auto* lib = VirtualAivLibraryManager::GetInstance().GetOrLoad(soName, kernelName);
    if (lib == nullptr || lib->handle == nullptr || lib->envInit == nullptr || lib->setBlockIdx == nullptr) {
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }
    if (kernelName.empty()) {
        HCCL_VM_ERROR("empty kernelName, can not virtual execute AIV kernel.");
        return ACL_ERROR_INVALID_PARAM;
    }

    dlerror();
    void* kernelSymbol = dlsym(lib->handle, kernelName.c_str());
    const char* kernelErr = dlerror();
    if (kernelSymbol == nullptr || kernelErr != nullptr) {
        HCCL_VM_ERROR(
            "dlsym {} from {} failed, err = {}", kernelName, lib->soPath, kernelErr == nullptr ? "unknown" : kernelErr);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    ResolvedKernelLaunchArgs resolvedArgs = PrepareResolvedKernelLaunchArgs(rawArgs);
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    uint64_t inputGlobalOffsetBase = 0;
    uint64_t outputGlobalOffsetBase = 0;
    uint64_t cclBufferSize = 0;
    uint64_t aivCommInfoSize = 0;
    if (ResolveVirtualAivBufferSizes(
            kernelName, resolvedArgs, commId, deviceId, inputSize, outputSize, inputGlobalOffsetBase,
            outputGlobalOffsetBase, cclBufferSize, aivCommInfoSize, replayOpDetailId)) {
        // 三种 buffer 任一 size=0（等同 count=0 空算子）：已在 ResolveVirtualAivBufferSizes 内打出警告；isNoop=true
        // 通知调用方跳过 insert task。
        ReleaseHostPtr(resolvedArgs.buffersInHandle);
        isNoop = true;
        return ACL_SUCCESS;
    }
    DumpVirtualKernelFuncArgs(kernelName, numBlocks, rawArgs, resolvedArgs);
    if (inputSize == INVALID_MEMORY_LAYOUT_SIZE || outputSize == INVALID_MEMORY_LAYOUT_SIZE
        || cclBufferSize == INVALID_MEMORY_LAYOUT_SIZE) {
        HCCL_VM_ERROR("failed to resolve opMemInfo size for kernel {}, rank={}", kernelName, resolvedArgs.args.rank);
        ReleaseHostPtr(resolvedArgs.buffersInHandle);
        return ACL_ERROR_INVALID_PARAM;
    }
    AivSim::AivOpParam curOpParam{};
    curOpParam.dataType = resolvedArgs.args.dataType;
    curOpParam.len = resolvedArgs.args.len;
    curOpParam.reduceOp = resolvedArgs.args.reduceOp;
    curOpParam.root = resolvedArgs.args.root;
    curOpParam.sliceId = resolvedArgs.args.tag;
    curOpParam.inputStride = resolvedArgs.args.inputSliceStride;
    curOpParam.outputStride = resolvedArgs.args.outputSliceStride;
    size_t kernelNameCopyLen = kernelName.size();
    if (kernelNameCopyLen >= AivSim::AIV_OP_KERNEL_NAME_MAX_LEN) {
        kernelNameCopyLen = AivSim::AIV_OP_KERNEL_NAME_MAX_LEN - 1;
    }
    std::memcpy(curOpParam.kernelName, kernelName.data(), kernelNameCopyLen);
    curOpParam.kernelName[kernelNameCopyLen] = '\0';
    HCCL_VM_DEBUG(
        "aiv_env_init and curOp:\n"
        "  rank = {}\n"
        "  blockNum = {}\n"
        "  buffIn = {:p}\n"
        "  rankSize = {}\n"
        "  input = 0x{:x}\n"
        "  inputSize = {}\n"
        "  output = 0x{:x}\n"
        "  outputSize = {}\n"
        "  inputGlobalOffsetBase = {}\n"
        "  outputGlobalOffsetBase = {}\n"
        "  cclBufferSize = {}\n"
        "  aivCommInfoSize = {}\n"
        "  curOp.dataType = {}\n"
        "  curOp.len = {}\n"
        "  curOp.reduceOp = {}\n"
        "  curOp.root = {}\n"
        "  curOp.sliceId = {}\n"
        "  curOp.inputStride = {}\n"
        "  curOp.outputStride = {}\n"
        "  curOp.kernelName = {}",
        resolvedArgs.args.rank, numBlocks, resolvedArgs.args.buffersIn, resolvedArgs.args.rankSize,
        static_cast<unsigned long long>(resolvedArgs.args.input), static_cast<unsigned long long>(inputSize),
        static_cast<unsigned long long>(resolvedArgs.args.output), static_cast<unsigned long long>(outputSize),
        static_cast<unsigned long long>(inputGlobalOffsetBase), static_cast<unsigned long long>(outputGlobalOffsetBase),
        static_cast<unsigned long long>(cclBufferSize), static_cast<unsigned long long>(aivCommInfoSize),
        curOpParam.dataType, static_cast<unsigned long long>(curOpParam.len), curOpParam.reduceOp, curOpParam.root,
        curOpParam.sliceId, static_cast<unsigned long long>(curOpParam.inputStride),
        static_cast<unsigned long long>(curOpParam.outputStride), curOpParam.kernelName);
    const bool envInitSuccess = lib->envInit(
        commId, resolvedArgs.args.rank, numBlocks, resolvedArgs.args.buffersIn, resolvedArgs.args.rankSize,
        resolvedArgs.args.input, inputSize, resolvedArgs.args.output, outputSize, inputGlobalOffsetBase,
        outputGlobalOffsetBase, cclBufferSize, aivCommInfoSize, curOpParam);
    if (!envInitSuccess) {
        HCCL_VM_ERROR(
            "AIV environment initialization failed, kernel={}, rank={}, blockNum={}", kernelName,
            resolvedArgs.args.rank, numBlocks);
        ReleaseHostPtr(resolvedArgs.buffersInHandle);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    if (numBlocks == 0) {
        HCCL_VM_DEBUG("numBlocks is 0, skip kernel invocation.");
        ReleaseHostPtr(resolvedArgs.buffersInHandle);
        return ACL_SUCCESS;
    }

    for (uint32_t blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
        HCCL_VM_DEBUG(
            "launch kernel {}, blockIdx={} <- "
            "aclrtLaunchKernelWithHostArgs(numBlocks) loop index; shared kernelFunc args are printed above",
            kernelName, blockIdx);
        lib->setBlockIdx(static_cast<int64_t>(blockIdx));
        if (resolvedArgs.args.hasExtraArgs) {
            auto kernelFunc = reinterpret_cast<AivExtraOpKernelFunc>(kernelSymbol);
            kernelFunc(
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.buffersIn)), resolvedArgs.args.input,
                resolvedArgs.args.output, resolvedArgs.args.rank, resolvedArgs.args.sendRecvRemoteRank,
                resolvedArgs.args.rankSize, resolvedArgs.args.len, resolvedArgs.args.dataType,
                resolvedArgs.args.reduceOp, resolvedArgs.args.root, resolvedArgs.args.tag,
                resolvedArgs.args.inputSliceStride, resolvedArgs.args.outputSliceStride, resolvedArgs.args.repeatNum,
                resolvedArgs.args.inputRepeatStride, resolvedArgs.args.outputRepeatStride, resolvedArgs.args.numBlocks,
                resolvedArgs.args.isOpBase,
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.headCountMem)),
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.tailCountMem)),
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.addOneMem)),
                resolvedArgs.args.counterMemSize, resolvedArgs.args.isEnableCounter, resolvedArgs.args.extraArgs);
        } else {
            auto kernelFunc = reinterpret_cast<AivOpKernelFunc>(kernelSymbol);
            kernelFunc(
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.buffersIn)), resolvedArgs.args.input,
                resolvedArgs.args.output, resolvedArgs.args.rank, resolvedArgs.args.sendRecvRemoteRank,
                resolvedArgs.args.rankSize, resolvedArgs.args.len, resolvedArgs.args.dataType,
                resolvedArgs.args.reduceOp, resolvedArgs.args.root, resolvedArgs.args.tag,
                resolvedArgs.args.inputSliceStride, resolvedArgs.args.outputSliceStride, resolvedArgs.args.repeatNum,
                resolvedArgs.args.inputRepeatStride, resolvedArgs.args.outputRepeatStride, resolvedArgs.args.numBlocks,
                resolvedArgs.args.isOpBase,
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.headCountMem)),
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.tailCountMem)),
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.addOneMem)),
                resolvedArgs.args.counterMemSize, resolvedArgs.args.isEnableCounter);
        }
    }
    if (lib->dumpTasks != nullptr) {
        lib->dumpTasks(launchIndex);
    }

    ReleaseHostPtr(resolvedArgs.buffersInHandle);
    return ACL_SUCCESS;
}

HcclResult ExecuteKernelLaunch(const AivOpArgs& opArgs)
{
    HCCL_VM_DEBUG(
        "called with parameters:\n"
        "  cmdType = {}\n"
        "  comm = {}\n"
        "  hcclComm = {:p}\n"
        "  numBlocks = {}\n"
        "  stream = {:p}\n"
        "  beginTime = {}\n"
        "  counter.headCountMem = 0x{:x}\n"
        "  counter.tailCountMem = 0x{:x}\n"
        "  counter.addOneMem = 0x{:x}\n"
        "  counter.memSize = {}\n"
        "  counter.isEnableCounter = {}\n"
        "  input = 0x{:x}\n"
        "  output = 0x{:x}\n"
        "  rank = {}\n"
        "  sendRecvRemoteRank = {}\n"
        "  rankSize = {}\n"
        "  count = {}\n"
        "  dataType = {}\n"
        "  op = {}\n"
        "  root = {}\n"
        "  sliceId = {}\n"
        "  inputSliceStride = {}\n"
        "  outputSliceStride = {}\n"
        "  repeatNum = {}\n"
        "  inputRepeatStride = {}\n"
        "  outputRepeatStride = {}\n"
        "  isOpBase = {}\n"
        "  argsType = {}\n"
        "  buffersIn(base aivCommInfoPtr, raw pointer only; parsed results below) = {:p}",
        static_cast<int>(opArgs.cmdType), opArgs.comm, opArgs.hcclComm, opArgs.numBlocks, opArgs.stream,
        static_cast<unsigned long long>(opArgs.beginTime), static_cast<unsigned long long>(opArgs.counter.headCountMem),
        static_cast<unsigned long long>(opArgs.counter.tailCountMem),
        static_cast<unsigned long long>(opArgs.counter.addOneMem), opArgs.counter.memSize,
        static_cast<int>(opArgs.counter.isEnableCounter), static_cast<unsigned long long>(opArgs.input),
        static_cast<unsigned long long>(opArgs.output), opArgs.rank, opArgs.sendRecvRemoteRank, opArgs.rankSize,
        static_cast<unsigned long long>(opArgs.count), static_cast<int>(opArgs.dataType), static_cast<int>(opArgs.op),
        opArgs.root, opArgs.sliceId, static_cast<unsigned long long>(opArgs.inputSliceStride),
        static_cast<unsigned long long>(opArgs.outputSliceStride), static_cast<unsigned long long>(opArgs.repeatNum),
        static_cast<unsigned long long>(opArgs.inputRepeatStride),
        static_cast<unsigned long long>(opArgs.outputRepeatStride), static_cast<int>(opArgs.isOpBase),
        static_cast<int>(opArgs.argsType), opArgs.buffersIn);
    if (IsAivExtraArgsCmdType(opArgs.cmdType)) {
        DumpAivExtraArgs(opArgs.extraArgs);
    } else {
        HCCL_VM_DEBUG("extraArgs are not used for cmdType={}.", static_cast<int>(opArgs.cmdType));
    }
    DumpAivTopo(opArgs.topo_);
    DumpBuffersInParsedDeviceView(opArgs.buffersIn, opArgs.rankSize, opArgs.numBlocks);

    using ExecuteKernelLaunchFunc = HcclResult (*)(const AivOpArgs&);
    constexpr const char* executeKernelLaunchSymbol = "_ZN8ops_hccl19ExecuteKernelLaunchERKNS_9AivOpArgsE";

    // RTLD_NEXT 无法穿透 torch_npu 等场景中以 RTLD_LOCAL 方式加载的真实 HCCL 库链,
    // 失败时按 SONAME 显式加载 libhccl.so (导出 ops_hccl::ExecuteKernelLaunch) 后重试.
    auto executeKernelLaunchFunc = reinterpret_cast<ExecuteKernelLaunchFunc>(
        sim::DlsymRealWithFallback(executeKernelLaunchSymbol, "libhccl.so"));
    if (executeKernelLaunchFunc == nullptr) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    if (sim::runtime::g_cur_comm_key == 0) {
        HCCL_VM_ERROR("ExecuteKernelLaunch has no current communicator ID");
        return HcclResult::HCCL_E_PARA;
    }

    HcclResult ret = executeKernelLaunchFunc(opArgs);
    HCCL_VM_INFO("returned {}", static_cast<int>(ret));
    return ret;
}
} // namespace ops_hccl

extern "C" aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle funcHandle, uint32_t numBlocks, aclrtStream stream, aclrtLaunchKernelCfg* cfg, void* hostArgs,
    size_t argsSize, aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
{
    (void)cfg;
    HCCL_VM_DEBUG(
        "called with parameters:\n"
        "  funcHandle = {:p}\n"
        "  numBlocks = {}\n"
        "  stream = {:p}\n"
        "  hostArgs = {:p}\n"
        "  argsSize = {}\n"
        "  placeHolderArray = {:p}\n"
        "  placeHolderNum = {}",
        funcHandle, numBlocks, stream, hostArgs, argsSize, static_cast<const void*>(placeHolderArray), placeHolderNum);

    if (funcHandle == nullptr) {
        HCCL_VM_WARN(
            "invalid parameters: funcHandle={:p}, stream={:p}, hostArgs={:p}, argsSize={}", funcHandle, stream,
            hostArgs, argsSize);
        return ACL_SUCCESS;
    }

    // PTA 未知内核仅录制占位；已识别的 AIV 内核继续走下方规范化参数后的采集分支。
    const uint64_t captureStreamId = (uint64_t)(uintptr_t)stream;
    if (IsCapturingStream(captureStreamId)) {
        // funcHandle 来自 aclrtBinaryGetFunction（HCCL AIV 内核）可解析出 kernelName；
        // 来自 aclrtBinaryLoadFromData 路径的 PTA 算子（桩未注册句柄，如 RNG fill）解析为空
        const std::string capturedKernelName = ops_hccl::InferKernelNameFromFuncHandle(funcHandle);
        if (capturedKernelName.empty()) {
            // UNKNOWN 占位：保证 capture 流程穿刺（数值正确性在一期豁免范围）
            RecordUnknownKernelAction(captureStreamId);
            HCCL_VM_INFO("kernel[<unknown>] captured to model(stream[{}]), not executed", captureStreamId);
            return ACL_SUCCESS;
        }
    }

    // AICPU-HostDPU模式下发流程。
    // DPU/AICPU内核经json注册（soName非空，如libccl_dpu.so的RunDpuRpcSrvLaunch、
    // libccl_kernel.so的HcclDpuTaskexpShmemRestore）不依赖展开模式：跨超节点通信域
    // 初始化(InitDpuKernel)在任何模式下都会下发；HCCL未设HCCL_OP_EXPANSION_MODE时
    // 内部默认AICPU_TS，IsAICPUExpMode()看不到该默认值。此类内核hostArgs布局与
    // AivKernelArgs不同，落入AIV解析分支会因argsSize(48)<sizeof(AivKernelArgs)(144)
    // 校验失败，导致全部rank通信域初始化失败(0x500000f)。
    // AIV内核从.o二进制加载、不经json注册（soName为空），不受此分流影响。
    sim::FuncHandle* funcView = reinterpret_cast<sim::FuncHandle*>(funcHandle);
    if (sim::IsAICPUExpMode() || (funcView != nullptr && !funcView->soName.empty())) {
        LaunchAicpuDpuKernel(funcHandle, stream, hostArgs, argsSize);
        return ACL_SUCCESS;
    }

    ops_hccl::DumpLaunchKernelCfg(cfg);
    ops_hccl::DumpPlaceHolderArray(placeHolderArray, placeHolderNum);

    const std::string kernelName = ops_hccl::InferKernelNameFromFuncHandle(funcHandle);
    if (kernelName.empty()) {
        // torch_npu 场景兜底：PTA 算子（如 aclGraph 捕获初始化的 RNG fill，funcHandle 来自
        // aclrtBinaryLoadFromData 路径，本桩未注册句柄）无法解析 kernelName。
        // 按评审定版的"流程穿刺"策略：不做虚拟执行、直接 no-op 放行，避免 207000 中断流程；
        // 数值正确性在一期豁免范围（x86 仿真不承载算子计算结果）。
        HCCL_VM_WARN(
            "unknown kernel (funcHandle={:p}) cannot resolve kernelName, "
            "treat as no-op virtual kernel, return success (flow-through).",
            static_cast<const void*>(funcHandle));
        return ACL_SUCCESS;
    }

    ops_hccl::AivHostLaunchArgs parsedHostArgs{};
    if (!ops_hccl::ParseAivHostLaunchArgs(hostArgs, argsSize, parsedHostArgs)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    ops_hccl::DumpHostLaunchArgs(&parsedHostArgs, argsSize);

    const std::string soName = ops_hccl::AIV_STUB_SO_NAME;
    HCCL_VM_INFO("resolved kernelName = {}, soName = {}", kernelName, soName);

    // ===== 图模式采集分支：独立取料并记录，与正常下发隔离 =====
    // 采集态只固化原料（kernelName/soName/numBlocks/规范化 hostArgs/commId/rankId/opDetailId），
    // 不虚拟执行也不插 AIV_GRAPH；重放期由 ReplayModelActions -> LaunchAivKernelRaw 重新虚拟执行。
    const uint64_t streamId = reinterpret_cast<uint64_t>(stream);
    if (IsCapturingStream(streamId)) {
        const uint64_t commId = sim::runtime::g_cur_comm_key;
        const uint32_t opDetailId = (commId == 0) ? 0 : sim::operation::g_currOpDetailId;
        const uint32_t deviceId = static_cast<uint32_t>(sim::runtime::GetCurrDeviceId());
        uint32_t rankId = UINT32_MAX;
        if (commId != 0) {
            (void)sim::runtime::GetCommRankByDeviceId(commId, deviceId, rankId);
        }
        RecordAivKernelAction(
            streamId, kernelName.c_str(), soName.c_str(), numBlocks, reinterpret_cast<const uint8_t*>(&parsedHostArgs),
            sizeof(parsedHostArgs), commId, rankId, opDetailId);
        HCCL_VM_INFO("AIV kernel:{} captured to model(stream:{}), not executed", kernelName, streamId);
        return ACL_SUCCESS;
    }

    // 模拟AIV核函数执行, 然后插入Task
    const uint32_t launchIndex = g_aivLaunchIndex.fetch_add(1, std::memory_order_relaxed);
    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::AIV_GRAPH;
    taskMetaData.profInfo = HcclTaskProfInfo(
        ProfTaskParamType::TASK_AIV, ProfLinkType::UNKNOWN, ProfTransportType::UNKNOWN); // AIV kernel 无链路维度
    taskMetaData.deviceId = static_cast<uint32_t>(sim::runtime::GetCurrDeviceId());
    taskMetaData.commId = sim::runtime::g_cur_comm_key;
    if (taskMetaData.commId == 0
        || !sim::runtime::GetCommRankByDeviceId(
            taskMetaData.commId, static_cast<uint32_t>(taskMetaData.deviceId), taskMetaData.rankId)) {
        HCCL_VM_ERROR("AIV task has no valid communicator, deviceId={}", taskMetaData.deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }
    taskMetaData.jettyId = 0;
    taskMetaData.streamId = reinterpret_cast<uint64_t>(stream);
    taskMetaData.taskData.aiv.launchIdx = static_cast<uint64_t>(launchIndex);

    bool isNoop = false; // 处理算子入口数据量为0的场景
    aclError ret = ops_hccl::VirtualExecuteAivKernel(
        kernelName, soName, numBlocks, parsedHostArgs, launchIndex, sim::runtime::g_cur_comm_key,
        static_cast<uint32_t>(sim::runtime::g_cur_device_key), 0, isNoop);
    if (ret != ACL_SUCCESS) {
        HCCL_VM_INFO("VirtualExecuteAivKernel failed, ret = {}", static_cast<int>(ret));
        return ACL_ERROR_INTERNAL_ERROR;
    }
    if (isNoop) {
        HCCL_VM_WARN(
            "AIV kernel {} is a no-op (zero-size buffer), skip task insert, deviceId={}, launchIndex={}", kernelName,
            taskMetaData.deviceId, launchIndex);
        return ACL_SUCCESS;
    }

    uint32_t unusedIndex = 0;
    auto insertRet = InsertTaskToCollection(&taskMetaData, &unusedIndex);
    if (insertRet != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR(
            "failed to insert AIV launch task, ret={}, deviceId={}", static_cast<uint32_t>(insertRet),
            taskMetaData.deviceId);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    HCCL_VM_INFO(
        "AIV launch task inserted, deviceId={}, launchIndex={}, streamId={}", taskMetaData.deviceId, launchIndex,
        taskMetaData.streamId);

    return ACL_SUCCESS;
}
// ===== AIV virtual-kernel support scope end =====

// 图模式重放专用重发：只吃 GraphAction 固化的原料(kernelName/soName/numBlocks/hostArgs/commId/rankId/streamId)。
// 与正常下发 aclrtLaunchKernelWithHostArgs 完全隔离、互不复用——正常下发怎么演进都不影响重放，
// 重放有 bug 也不影响正常下发。显式 extern "C"，与 graph_capture.h 声明一致。
extern "C" void LaunchAivKernelRaw(
    const char* kernelName, const char* soName, uint32_t numBlocks, const uint8_t* hostArgsBytes, uint32_t hostArgsSize,
    uint64_t commId, uint32_t rankId, uint64_t streamId, uint32_t opDetailId)
{
    // 反序列化采集期固化的规范化 hostArgs（POD 整块拷贝，size 必须与 AivHostLaunchArgs 一致）。
    ops_hccl::AivHostLaunchArgs rawArgs{};
    if (hostArgsBytes == nullptr || hostArgsSize != sizeof(ops_hccl::AivHostLaunchArgs)) {
        HCCL_VM_ERROR(
            "replay AIV invalid hostArgs bytes, size={}, expect={}", hostArgsSize, sizeof(ops_hccl::AivHostLaunchArgs));
        return;
    }
    (void)memcpy(&rawArgs, hostArgsBytes, sizeof(ops_hccl::AivHostLaunchArgs));

    // 重新分配 launchIdx（与采集期自增序列独立，重放期重新自增，checker 按 task 序列消费）。
    const uint32_t launchIndex = g_aivLaunchIndex.fetch_add(1, std::memory_order_relaxed);

    // 重新虚拟执行 x86 AIV stub，重新产数据面行为。
    const std::string replayKernelName = kernelName == nullptr ? "" : kernelName;
    const std::string replaySoName = soName == nullptr ? "" : soName;
    bool isNoop = false;
    aclError ret = ops_hccl::VirtualExecuteAivKernel(
        replayKernelName, replaySoName, numBlocks, rawArgs, launchIndex, commId,
        static_cast<uint32_t>(sim::runtime::GetCurrDeviceId()), opDetailId, isNoop);
    if (ret != ACL_SUCCESS) {
        HCCL_VM_ERROR(
            "replay VirtualExecuteAivKernel failed, kernel[{}], ret={}", replayKernelName, static_cast<int>(ret));
        return;
    }
    if (isNoop) {
        HCCL_VM_INFO(
            "replay AIV kernel {} is a no-op (zero-size buffer), skip task insert, launchIndex={}", replayKernelName,
            launchIndex);
        return;
    }

    // 重插 AIV_GRAPH task（opDetail 由调用方 ReplayModelActions 写入 g_currOpDetailId 后关联）。
    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::AIV_GRAPH;
    taskMetaData.deviceId = static_cast<uint32_t>(sim::runtime::GetCurrDeviceId());
    taskMetaData.commId = commId;
    taskMetaData.rankId = rankId;
    taskMetaData.jettyId = 0;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.aiv.launchIdx = launchIndex;
    uint32_t unusedIndex = 0;
    auto insertRet = InsertTaskToCollection(&taskMetaData, &unusedIndex);
    if (insertRet != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR(
            "replay insert AIV launch task failed, ret={}, deviceId={}", static_cast<uint32_t>(insertRet),
            taskMetaData.deviceId);
        return;
    }
    HCCL_VM_INFO(
        "replay AIV launch task inserted, deviceId={}, launchIndex={}, streamId={}", taskMetaData.deviceId, launchIndex,
        taskMetaData.streamId);
}
