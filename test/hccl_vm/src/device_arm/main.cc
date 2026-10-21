/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

#include "hccl_device_pub.h"
#include "operation_data/operation_data_ops.h"
#include "sim_log.h"
#include "sim_aicpu_pipe_msg.h"
#include "sim_pipe_io.h"
#include "sim_aicpu_pipe_handler.h"
#include "sim_kernel_lib_mgr.h"
#include "storage/storage_runtime.h"

// argv[2] 是 Device 表主键；各创建路径在启动时传入，直接用于 cann 日志目录。
int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::fprintf(stderr, "[device] Usage: %s <rankId> <deviceKey> <h2d_read_fd> <d2h_write_fd>\n", argv[0]);
        return -1;
    }
    uint32_t rankId = static_cast<uint32_t>(std::atoi(argv[1]));
    uint32_t devKey = static_cast<uint32_t>(std::atoi(argv[2]));
    int h2dReadFd = std::atoi(argv[3]);
    int d2hWriteFd = std::atoi(argv[4]);

    LogConfig config = LoadLogConfig("device_aarch64");
    InitLogger(config, devKey);

    RegisterSignalHandler();
    setvbuf(stdout, nullptr, _IOLBF, 0);
    HCCL_VM_INFO("[device] main process start.");

    // LD_PRELOAD rank/device 进程必须在首次 Table<T>() 前按自身 PID 安装独立会话，
    // 不复用 Host 父进程继承的 SQLite 连接。
    if (sim::operation::EnsureProcessStorageSession() != 0) {
        HCCL_VM_ERROR("[device] failed to initialize the process storage session");
        return -1;
    }

    HCCL_VM_INFO(
        "[device] parse input args: rankId={} devKey={} h2dReadFd={} d2hWriteFd={}", rankId, devKey, h2dReadFd,
        d2hWriteFd);

    SetCurRankId(rankId);
    SetCurDeviceKey(devKey);
    InitPipeFds(h2dReadFd, d2hWriteFd);
    DeviceSendMsg(PIPE_RSP_READY, nullptr, 0);

    uint8_t cmd = 0;
    static uint8_t payload[IPC_MSG_LEN_MAX] = {0}; // 定长接收缓冲，main 单线程安全
    uint32_t payloadLen = 0;

    while (true) {
        int rc = DeviceRecvMsg(cmd, payload, sizeof(payload), payloadLen);
        if (rc != 0) {
            HCCL_VM_INFO("[device] DeviceRecvMsg failed (EOF or error), exiting loop.");
            break;
        }

        switch (cmd) {
            case PIPE_CMD_SHUTDOWN:
                HCCL_VM_INFO("[device] Received PIPE_CMD_SHUTDOWN, exiting gracefully.");
                goto done;
            case PIPE_CMD_SET_DEV_ID:
                sim::HandlePipeCmdSetDevId(payload, payloadLen);
                break;
            case PIPE_CMD_GET_DEV_PTR:
                sim::HandlePipeCmdGetDevPtr(payload, payloadLen);
                break;
            case PIPE_CMD_EXEC_KERNEL:
                sim::HandlePipeCmdExecKernel(payload, payloadLen);
                break;
            case PIPE_CMD_FREE_DEV_PTR:
                sim::HandlePipeCmdFreeDevPtr(payload, payloadLen);
                break;
            case PIPE_CMD_GET_WQE_PTR:
                sim::HandlePipeCmdGetWqePtr(payload, payloadLen);
                break;
            case PIPE_CMD_FREE_WQE_PTR:
                sim::HandlePipeCmdFreeWqePtr(payload, payloadLen);
                break;
            default:
                HCCL_VM_ERROR("[device] Unknown command: 0x{:02x}", cmd);
                break;
        }
    }

done:
    uint32_t curRankId = 0;
    GetCurRankId(&curRankId);
    sim::KernelLibManager::GetInstance().Cleanup();
    HCCL_VM_INFO("[device] rankId[{}] exiting.", curRankId);

    sim::PipeClose(h2dReadFd);
    sim::PipeClose(d2hWriteFd);
    auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    if (!closed.ok()) {
        HCCL_VM_WARN("[device] storage session close failed: {}", closed.diagnostic);
    }
    // 显式关闭 logger 并刷盘（_exit 不会触发 atexit）。
    DeInitLogger();
    _exit(0);
}
