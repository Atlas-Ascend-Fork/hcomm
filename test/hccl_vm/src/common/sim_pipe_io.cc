/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_pipe_io.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/uio.h>

#include "sim_log.h"

namespace sim {

int PipeBlockRead(int fd, void* buf, size_t len)
{
    uint8_t* ptr = static_cast<uint8_t*>(buf);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = read(fd, ptr, remaining);
        if (n > 0) {
            ptr += n;
            remaining -= n;
        } else if (n == 0) {
            HCCL_VM_INFO("read EOF, peer closed.");
            return -1;
        } else if (errno == EINTR) {
            continue;
        } else {
            HCCL_VM_ERROR("read failed: {} (errno={}).", strerror(errno), errno);
            return -1;
        }
    }

    return 0;
}

int PipeBlockWrite(int fd, const void* buf, size_t len)
{
    const uint8_t* ptr = static_cast<const uint8_t*>(buf);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = write(fd, ptr, remaining);
        if (n > 0) {
            ptr += n;
            remaining -= n;
        } else if (n == 0) {
            HCCL_VM_INFO("write returned 0.");
            return -1;
        } else if (errno == EINTR) {
            continue;
        } else if (errno == EPIPE || errno == EAGAIN) {
            HCCL_VM_INFO("peer closed (errno={}).", errno);
            return -1;
        } else {
            HCCL_VM_ERROR("write failed: {} (errno={}).", strerror(errno), errno);
            return -1;
        }
    }

    return 0;
}

int PipeBlockWriteV(int fd, struct iovec* iov, int iovcnt)
{
    if (iov == nullptr || iovcnt <= 0) {
        HCCL_VM_ERROR("PipeBlockWriteV: invalid iov segments: {}", iovcnt);
        return -1;
    }

    struct iovec* cur = iov;
    int curCnt = iovcnt;
    while (curCnt > 0) {
        ssize_t n = writev(fd, cur, curCnt);
        if (n > 0) {
            size_t written = static_cast<size_t>(n);
            while (curCnt > 0 && written >= cur[0].iov_len) {
                written -= cur[0].iov_len;
                ++cur;
                --curCnt;
            }
            if (curCnt > 0 && written > 0) {
                cur[0].iov_base = static_cast<uint8_t*>(cur[0].iov_base) + written;
                cur[0].iov_len -= written;
            }
        } else if (n == 0) {
            HCCL_VM_INFO("writev returned 0.");
            return -1;
        } else if (errno == EINTR) {
            continue;
        } else if (errno == EPIPE || errno == EAGAIN) {
            HCCL_VM_INFO("peer closed (errno={}).", errno);
            return -1;
        } else {
            HCCL_VM_ERROR("writev failed: {} (errno={}).", strerror(errno), errno);
            return -1;
        }
    }

    return 0;
}

int PipeSendMsg(int fd, uint8_t cmd, const void* data, uint32_t len)
{
    // 单段是 writev 的特例，统一走 PipeSendMsgV：一次 writev、零拷贝、集中做长度校验
    if (len == 0 || data == nullptr) {
        return PipeSendMsgV(fd, cmd, nullptr, 0);
    }

    struct iovec seg {
        const_cast<void*>(data), len
    };
    return PipeSendMsgV(fd, cmd, &seg, 1);
}

int PipeSendMsgV(int fd, uint8_t cmd, const struct iovec* iov, uint32_t iovcnt)
{
    uint32_t totalLen = 0;
    for (uint32_t i = 0; i < iovcnt; ++i) {
        totalLen += iov[i].iov_len;
    }
    if (totalLen > IPC_MSG_LEN_MAX) {
        HCCL_VM_ERROR("PipeSendMsgV payload too large: {} > {}", totalLen, IPC_MSG_LEN_MAX);
        return -1;
    }

    constexpr uint32_t hdrSize = offsetof(PipeMessage, payload);
    constexpr int kMaxIov = 8;
    if (static_cast<int>(iovcnt) + 1 > kMaxIov) {
        HCCL_VM_ERROR("PipeSendMsgV too many segments: {}", iovcnt);
        return -1;
    }

    PipeMessage hdr{};
    hdr.cmd = cmd;
    hdr.len = totalLen;

    struct iovec all[kMaxIov];
    all[0].iov_base = &hdr;
    all[0].iov_len = hdrSize;
    for (uint32_t i = 0; i < iovcnt; ++i) {
        all[i + 1] = iov[i];
    }

    return PipeBlockWriteV(fd, all, static_cast<int>(iovcnt) + 1);
}

int PipeRecvMsg(int fd, uint8_t& outCmd, void* outData, uint32_t maxLen, uint32_t& outLen)
{
    constexpr uint32_t hdrSize = offsetof(PipeMessage, payload);
    PipeMessage hdr{};
    if (PipeBlockRead(fd, &hdr, hdrSize) != 0) {
        outCmd = 0;
        outLen = 0;
        return -1;
    }

    outCmd = hdr.cmd;
    outLen = hdr.len;

    if (hdr.len > maxLen) {
        HCCL_VM_ERROR("payload too large: {} > {}", hdr.len, maxLen);
        // 丢弃超出部分以保持流同步
        uint8_t tmp[4096];
        uint32_t remaining = hdr.len;
        while (remaining > 0) {
            uint32_t chunk = remaining > sizeof(tmp) ? sizeof(tmp) : remaining;
            if (PipeBlockRead(fd, tmp, chunk) != 0) {
                break;
            }
            remaining -= chunk;
        }
        outLen = 0;
        return -1;
    }

    if (hdr.len > 0 && outData != nullptr) {
        if (PipeBlockRead(fd, outData, hdr.len) != 0) {
            outCmd = 0;
            outLen = 0;
            return -1;
        }
    }

    return 0;
}

int PipeCreate(PipePair& pair)
{
    int fds[2] = {-1, -1};
    if (pipe2(fds, O_CLOEXEC) == -1) {
        HCCL_VM_ERROR("pipe2() failed: {}", strerror(errno));
        return -1;
    }

    pair.readFd = fds[0];
    pair.writeFd = fds[1];
    return 0;
}

void PipeClose(int& fd)
{
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
}

int PipeChildSetup(PipePair& h2d, PipePair& d2h, int targetReadFd, int targetWriteFd)
{
    // 子进程不写 h2d，不读 d2h
    close(h2d.writeFd);
    close(d2h.readFd);

    // 重定向到固定 fd
    dup2(h2d.readFd, targetReadFd);
    dup2(d2h.writeFd, targetWriteFd);

    // 清除 CLOEXEC 标记（execvp 后保留）
    fcntl(targetReadFd, F_SETFD, 0);
    fcntl(targetWriteFd, F_SETFD, 0);

    // 关闭原始 fd（如果与目标 fd 不同）
    if (h2d.readFd != targetReadFd) {
        close(h2d.readFd);
    }
    if (d2h.writeFd != targetWriteFd) {
        close(d2h.writeFd);
    }

    return 0;
}

} // namespace sim
