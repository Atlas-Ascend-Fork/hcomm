/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_sub_process_manager.h"
#include <sys/wait.h>
#include <sys/utsname.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <sstream>
#include <thread>
#include "sim_log.h"
#include "sim_common_defs.h"
#include "sim_common_api.h"
#include "sim_pipe_io.h"

namespace sim {

namespace {
    namespace fs = std::filesystem;

    // HCCL_VM_QEMU_TB_SIZE 未设置时的默认翻译代码缓存（MiB）
    constexpr uint32_t kDefaultQemuTbSizeMiB = 16;
    // 版本探测日志截断长度，避免发行版长版本串刷屏
    constexpr size_t kQemuVersionLogMax = 48;

    bool IsCompleteCannSysroot(const fs::path& sysroot, std::string& missingPath)
    {
        const std::vector<fs::path> requiredPaths{
            sysroot / "lib64/ld-linux-aarch64.so.1", sysroot / "lib64/libc.so.6", sysroot / "usr/lib64/libstdc++.so.6",
            sysroot / "usr/lib64/libgcc_s.so.1"};

        std::error_code ec;
        for (const auto& requiredPath : requiredPaths) {
            ec.clear();
            if (!fs::exists(requiredPath, ec)) {
                missingPath = requiredPath.string();
                if (ec) {
                    missingPath += ": " + ec.message();
                }
                return false;
            }
        }
        return true;
    }

    std::string ResolveCannSysroot(const char* ascendHomePath)
    {
        const char* configuredSysroot = std::getenv("CANN_HCC_SYSROOT");
        if (configuredSysroot != nullptr && configuredSysroot[0] != '\0') {
            std::string missingPath;
            if (IsCompleteCannSysroot(configuredSysroot, missingPath)) {
                return configuredSysroot;
            }
            HCCL_VM_ERROR(
                "invalid CANN_HCC_SYSROOT [{}]: required component [{}] was not found.", configuredSysroot,
                missingPath);
            return {};
        }

        if (ascendHomePath == nullptr || ascendHomePath[0] == '\0') {
            HCCL_VM_ERROR("CANN sysroot is not configured; set ASCEND_HOME_PATH or CANN_HCC_SYSROOT.");
            return {};
        }

        const std::vector<fs::path> candidates{
            fs::path(ascendHomePath) / "tools/hcc/sysroot", fs::path(ascendHomePath) / "toolkit/toolchain/hcc/sysroot"};
        std::ostringstream diagnostics;
        for (const auto& candidate : candidates) {
            std::string missingPath;
            if (IsCompleteCannSysroot(candidate, missingPath)) {
                return candidate.string();
            }
            diagnostics << "\n  " << candidate.string() << ": missing " << missingPath;
        }

        HCCL_VM_ERROR(
            "complete CANN sysroot was not found under ASCEND_HOME_PATH [{}]. Tried:{}", ascendHomePath,
            diagnostics.str());
        return {};
    }

    // HCCL_VM_QEMU_PATH 指定的 QEMU 程序校验：必须是非空绝对路径，目标为可执行的
    // 普通文件（允许最终指向普通文件的符号链接）。失败原因写入 reason 并返回空串，
    // 由调用方决定回退（PATH 查找）或硬失败（全机找不到 QEMU）。
    std::string ResolveConfiguredQemuProgram(const char* configuredPath, std::string& reason)
    {
        reason.clear();
        if (configuredPath == nullptr || configuredPath[0] == '\0') {
            reason = "is not set";
            return {};
        }
        const fs::path givenPath(configuredPath);
        if (!givenPath.is_absolute()) {
            reason = "must be an absolute path: " + std::string(configuredPath);
            return {};
        }
        std::error_code ec;
        const fs::file_status status = fs::status(givenPath, ec); // 跟随符号链接，断链在此暴露
        if (ec || !fs::exists(status)) {
            reason = "target does not exist: " + std::string(configuredPath);
            return {};
        }
        if (!fs::is_regular_file(status)) {
            reason = "target is not a regular file: " + std::string(configuredPath);
            return {};
        }
        if (access(configuredPath, X_OK) != 0) {
            reason = "target is not executable: " + std::string(configuredPath);
            return {};
        }
        return std::string(configuredPath);
    }

    // 按 PATH 顺序查找首个可执行的普通文件（用于 HCCL_VM_QEMU_PATH 未配置/无效时
    // 的回退，以及 timeout 工具的定位）。找不到返回空串。
    std::string ResolveExecutableInPath(const char* name)
    {
        if (name == nullptr || name[0] == '\0') {
            return {};
        }
        const char* pathEnv = std::getenv("PATH");
        if (pathEnv == nullptr || pathEnv[0] == '\0') {
            return {};
        }
        std::istringstream pathStream(pathEnv);
        std::string dir;
        while (std::getline(pathStream, dir, ':')) {
            if (dir.empty()) {
                continue;
            }
            const fs::path candidate = fs::path(dir) / name;
            std::error_code ec;
            if (!fs::is_regular_file(fs::status(candidate, ec)) || ec) {
                continue;
            }
            if (access(candidate.c_str(), X_OK) == 0) {
                return candidate.string();
            }
        }
        return {};
    }

    // 捕获命令输出头部（探测用途，最多 kQemuProbeOutputMax 字节）。返回值：-1 表示
    // popen/pclose 失败或子进程被信号终止；否则为子进程退出码。
    constexpr size_t kQemuProbeOutputMax = 2048;
    int RunCapture(const std::string& command, std::string& output)
    {
        output.clear();
        FILE* pipe = ::popen(command.c_str(), "r");
        if (pipe == nullptr) {
            return -1;
        }
        char buf[512];
        while (output.size() < kQemuProbeOutputMax) {
            size_t n = ::fread(buf, 1, sizeof(buf), pipe);
            if (n == 0) {
                break;
            }
            output.append(buf, n);
        }
        const int rawRc = ::pclose(pipe);
        if (rawRc == -1 || !WIFEXITED(rawRc)) {
            return -1;
        }
        return WEXITSTATUS(rawRc);
    }

    // 一次性探测 QEMU 版本与 -tb-size 能力：优先 `timeout 5 <program> -tb-size 16
    // -version`（rc==0 即支持，输出同时含版本行）；不支持时再取纯 -version 输出
    // 尽力解析版本。每进程仅在创建 AICPU 设备时调用一次。
    void ProbeQemuVersionAndTbSize(const std::string& program, bool& supportsTbSize, std::string& version)
    {
        supportsTbSize = false;
        version = "unknown";
        const std::string quotedProgram = "'" + program + "'";
        // timeout 定位为绝对路径：sh 经 PATH 查找，rank 环境的 PATH 可能不含系统目录
        const std::string timeoutTool = ResolveExecutableInPath("timeout");
        const std::string guard = timeoutTool.empty() ? "" : timeoutTool + " 5 ";
        std::string output;
        const int tbRc = RunCapture(guard + quotedProgram + " -tb-size 16 -version 2>&1", output);
        supportsTbSize = (tbRc == 0);
        if (!supportsTbSize) {
            std::string versionOutput;
            if (RunCapture(guard + quotedProgram + " -version 2>&1", versionOutput) == 0) {
                output = versionOutput;
            }
        }
        const std::string prefix = "qemu-aarch64 version ";
        std::istringstream in(output);
        std::string line;
        while (std::getline(in, line)) {
            const auto pos = line.find(prefix);
            if (pos == std::string::npos) {
                continue;
            }
            std::string ver = line.substr(pos + prefix.size());
            if (ver.size() > kQemuVersionLogMax) {
                ver.resize(kQemuVersionLogMax);
            }
            while (!ver.empty() && (ver.back() == ' ' || ver.back() == '\r' || ver.back() == '\n')) {
                ver.pop_back();
            }
            if (!ver.empty()) {
                version = ver;
            }
            break;
        }
    }

    // HCCL_VM_QEMU_TB_SIZE 解析：未设置时使用默认值；仅接受可完整解析到 uint32_t
    // 的十进制数字串（允许前导 0，如 0016）。空串、正负号、空白、单位、前缀或尾随
    // 字符、超出 uint32_t 范围均视为非法（由调用方告警并回退默认值）。解析成功后
    // 由调用方按十进制重新序列化进 argv，避免 QEMU 对前导 0 串的自动进制解析偏差。
    bool ParseQemuTbSizeMiB(const char* configuredValue, uint32_t& outValueMiB, std::string& reason)
    {
        outValueMiB = kDefaultQemuTbSizeMiB;
        reason.clear();
        if (configuredValue == nullptr) {
            return true;
        }
        const std::string text(configuredValue);
        if (text.empty()) {
            reason = "HCCL_VM_QEMU_TB_SIZE is empty";
            return false;
        }
        for (const char c : text) {
            if (c < '0' || c > '9') {
                reason = "HCCL_VM_QEMU_TB_SIZE must be decimal digits only: " + text;
                return false;
            }
        }
        errno = 0;
        char* endPtr = nullptr;
        const unsigned long long parsed = std::strtoull(text.c_str(), &endPtr, 10);
        if (errno == ERANGE || endPtr == nullptr || *endPtr != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
            reason = "HCCL_VM_QEMU_TB_SIZE is out of uint32_t range: " + text;
            return false;
        }
        outValueMiB = static_cast<uint32_t>(parsed);
        return true;
    }
} // namespace

static sim::SubProcessManager g_aiCpuProcMgr;

SubProcessManager& GetAicpuProcMgr() { return g_aiCpuProcMgr; }

SubProcessManager::~SubProcessManager()
{
    if (m_pid > 0) {
        DestroyProcess();
    }
}

int SubProcessManager::CreateProcess(const SubProcessConfig& config)
{
    std::lock_guard<std::mutex> lock(m_forkLock);

    if (config.executable.empty() || config.args.empty()) {
        HCCL_VM_ERROR("Invalid subprocess config: executable or argv is empty.");
        return -1;
    }

    if (m_pid > 0) {
        HCCL_VM_INFO("Process {} already exists, skip.", m_pid);
        return 0;
    }

    sim::PipePair h2dPipe{}, d2hPipe{};
    if (sim::PipeCreate(h2dPipe) != 0) {
        HCCL_VM_ERROR("Failed to create h2d pipe");
        return -1;
    }
    if (sim::PipeCreate(d2hPipe) != 0) {
        HCCL_VM_ERROR("Failed to create d2h pipe");
        sim::PipeClose(h2dPipe.readFd);
        sim::PipeClose(h2dPipe.writeFd);
        return -1;
    }

    pid_t pid = fork();
    if (pid == -1) {
        HCCL_VM_ERROR("fork() failed: {}", strerror(errno));
        sim::PipeClose(h2dPipe.readFd);
        sim::PipeClose(h2dPipe.writeFd);
        sim::PipeClose(d2hPipe.readFd);
        sim::PipeClose(d2hPipe.writeFd);
        return -1;
    }

    // device进程执行参数拼接
    std::vector<char*> argv;
    for (const auto& arg : config.args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }

    std::string h2dReadFdStr = std::to_string(kH2dReadFd);
    std::string d2hWriteFdStr = std::to_string(kD2hWriteFd);
    argv.push_back(const_cast<char*>(h2dReadFdStr.c_str()));
    argv.push_back(const_cast<char*>(d2hWriteFdStr.c_str()));
    argv.push_back(nullptr);

    if (pid == 0) {
        // fork 后只断开继承的日志入口，避免触碰父进程日志锁；exec 前不 flush/析构。
        hcclvm::g_logWriter = nullptr;
        g_logger = nullptr;
        sim::PipeChildSetup(h2dPipe, d2hPipe, kH2dReadFd, kD2hWriteFd);

        for (const auto& env : config.envVars) {
            setenv(env.first.c_str(), env.second.c_str(), 1);
        }

        execvp(config.executable.c_str(), argv.data());
        _exit(127);
    }

    m_pid = pid;
    sim::PipeClose(h2dPipe.readFd);  // 主进程不读 h2dPipe
    sim::PipeClose(d2hPipe.writeFd); // 主进程不写 d2hPipe
    m_h2dWriteFd = h2dPipe.writeFd;
    m_d2hReadFd = d2hPipe.readFd;

    HCCL_VM_INFO("Child process {} created (m_h2dWriteFd={}, m_d2hReadFd={}).", m_pid, m_h2dWriteFd, m_d2hReadFd);

    // 等待device进程管道准备就绪进入Ready状态
    if (WaitForReady() != 0) {
        HCCL_VM_ERROR("Child process {} failed to become READY, aborting.", m_pid);
        DestroyProcess();
        return -1;
    }

    return 0;
}

int SubProcessManager::WaitForReady()
{
    if (m_d2hReadFd < 0 || m_pid < 0) {
        HCCL_VM_ERROR("WaitForReady: child not running.");
        return -1;
    }

    uint8_t rspCmd = 0;
    uint8_t buf[16] = {0};
    uint32_t rspLen = 0;
    if (HostRecvMsg(rspCmd, buf, sizeof(buf), rspLen) != 0) {
        HCCL_VM_ERROR("WaitForReady: child pid={} exited or read failed.", m_pid);
        return -1;
    }
    if (rspCmd != PIPE_RSP_READY) {
        HCCL_VM_ERROR("WaitForReady: unexpected msg cmd=0x{:02x}.", rspCmd);
        return -1;
    }
    HCCL_VM_INFO("Child process {} is READY.", m_pid);
    return 0;
}

int SubProcessManager::DestroyProcess(int timeoutMs)
{
    if (m_pid < 0) {
        HCCL_VM_INFO("Process already stopped.");
        return 0;
    }

    HCCL_VM_INFO("Sending SHUTDOWN to process {}.", m_pid);
    {
        std::lock_guard<std::mutex> lock(m_rpcLock);
        HostSendMsg(PIPE_CMD_SHUTDOWN, nullptr, 0);
    }

    sim::PipeClose(m_h2dWriteFd);

    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        int status;
        pid_t wPid = waitpid(m_pid, &status, WNOHANG);
        if (wPid == m_pid) {
            HCCL_VM_INFO("Child process {} exited with status {}.", m_pid, status);
            break;
        } else if (wPid == -1 && errno == ECHILD) {
            break;
        }

        auto elapsed
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                  .count();
        if (elapsed >= timeoutMs) {
            HCCL_VM_WARN("Child process {} did not exit within {} ms, SIGKILL.", m_pid, timeoutMs);
            kill(m_pid, SIGKILL);
            waitpid(m_pid, &status, 0);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    sim::PipeClose(m_d2hReadFd);

    m_pid = -1;
    return 0;
}

bool SubProcessManager::IsAlive() const
{
    if (m_pid < 0) {
        return false;
    }

    int status = 0;
    pid_t wPid = waitpid(m_pid, &status, WNOHANG);
    return wPid == 0;
}

int SubProcessManager::Request(
    uint8_t reqCmd, const void* reqData, uint32_t reqLen, uint8_t& rspCmd, void* rspData, uint32_t rspMaxLen,
    uint32_t& rspLen)
{
    std::lock_guard<std::mutex> lock(m_rpcLock);
    if (HostSendMsg(reqCmd, reqData, reqLen) != 0) {
        return -1;
    }

    if (HostRecvMsg(rspCmd, rspData, rspMaxLen, rspLen) != 0) {
        return -1;
    }

    return 0;
}

int SubProcessManager::RequestV(
    uint8_t reqCmd, const struct iovec* reqIov, uint32_t reqSegCnt, uint8_t& rspCmd, void* rspData, uint32_t rspMaxLen,
    uint32_t& rspLen)
{
    std::lock_guard<std::mutex> lock(m_rpcLock);
    if (HostSendMsgV(reqCmd, reqIov, reqSegCnt) != 0) {
        return -1;
    }

    if (HostRecvMsg(rspCmd, rspData, rspMaxLen, rspLen) != 0) {
        return -1;
    }

    return 0;
}

int SubProcessManager::HostSendMsg(uint8_t cmd, const void* data, uint32_t len)
{
    if (m_h2dWriteFd < 0) {
        HCCL_VM_ERROR("HostSendMsg: h2d pipe not open.");
        return -1;
    }

    return PipeSendMsg(m_h2dWriteFd, cmd, data, len);
}

int SubProcessManager::HostSendMsgV(uint8_t cmd, const struct iovec* iov, uint32_t iovcnt)
{
    if (m_h2dWriteFd < 0) {
        HCCL_VM_ERROR("HostSendMsgV: h2d pipe not open.");
        return -1;
    }

    return PipeSendMsgV(m_h2dWriteFd, cmd, iov, iovcnt);
}

int SubProcessManager::HostRecvMsg(uint8_t& outCmd, void* outData, uint32_t maxLen, uint32_t& outLen)
{
    if (m_d2hReadFd < 0) {
        HCCL_VM_ERROR("HostRecvMsg: d2h pipe not open.");
        return -1;
    }

    return PipeRecvMsg(m_d2hReadFd, outCmd, outData, maxLen, outLen);
}

bool IsAarch64Host()
{
    utsname utsBuf;
    uname(&utsBuf);
    return (std::strstr(utsBuf.machine, "aarch64") != nullptr || std::strstr(utsBuf.machine, "arm") != nullptr);
}

SubProcessConfig CreateAicpuDeviceConfig(uint32_t rankId, uint32_t deviceKey)
{
    SubProcessConfig config;
    const char* ascendHomePath = std::getenv("ASCEND_HOME_PATH");
    const std::string cannSysroot = ResolveCannSysroot(ascendHomePath);
    if (cannSysroot.empty()) {
        return config;
    }

    const std::string devBinPath = InstallPath::ResolveToInstallRoot("bin/device");
    const std::string preloadPath = InstallPath::ResolveToInstallRoot("lib/aarch64/libhccl_device_proxy.so");
    const std::string preloadPathL1 = InstallPath::ResolveToInstallRoot("lib/aarch64/libhccl_device_proxy_level1.so");
    std::string libPath = InstallPath::ResolveToInstallRoot("lib/aarch64");
    if (ascendHomePath != nullptr) {
        libPath += ":" + std::string(ascendHomePath) + "/" + GetArchStr() + "-linux/devlib/device";
    }
    libPath += ":" + cannSysroot + "/lib64";
    libPath += ":" + cannSysroot + "/usr/lib64";

    if (IsAarch64Host()) {
        const std::string cannLoaderPath = (fs::path(cannSysroot) / "lib64/ld-linux-aarch64.so.1").string();
        config.executable = cannLoaderPath;
        config.args = {cannLoaderPath, "--library-path", libPath, devBinPath};
    } else {
        // x86_64 宿主：AICPU device 经外部 QEMU（linux-user 模拟 aarch64）启动。
        // 程序优先取 HCCL_VM_QEMU_PATH 指定的 10.2.4 部署（README §2.3 步骤产物）；
        // 未配置/配置无效/为不支持 -tb-size 的旧版时自动回退（PATH 查找/不传该
        // 选项）并打 WARN（含实际版本与内存提示），保证旧部署仍可运行；仅当全机
        // 找不到 QEMU 时硬失败（走现有空配置路径）。
        std::string reason;
        uint32_t qemuTbSizeMiB = kDefaultQemuTbSizeMiB;
        if (!ParseQemuTbSizeMiB(std::getenv("HCCL_VM_QEMU_TB_SIZE"), qemuTbSizeMiB, reason)) {
            qemuTbSizeMiB = kDefaultQemuTbSizeMiB;
            HCCL_VM_WARN(
                "AICPU QEMU tb-size is invalid ({}), fallback to default {} MiB.", reason, kDefaultQemuTbSizeMiB);
        }
        std::string qemuProgram = ResolveConfiguredQemuProgram(std::getenv("HCCL_VM_QEMU_PATH"), reason);
        bool pathFallback = false;
        if (qemuProgram.empty()) {
            qemuProgram = ResolveExecutableInPath("qemu-aarch64-static");
            if (qemuProgram.empty()) {
                HCCL_VM_ERROR(
                    "AICPU QEMU program not found: HCCL_VM_QEMU_PATH {} and no executable "
                    "qemu-aarch64-static in PATH. Deploy QEMU 10.2.4 per README section 2.3.",
                    reason);
                return config;
            }
            pathFallback = true;
        }
        bool supportsTbSize = false;
        std::string qemuVersion;
        ProbeQemuVersionAndTbSize(qemuProgram, supportsTbSize, qemuVersion);
        config.executable = qemuProgram;
        config.envVars["QEMU_LD_PREFIX"] = cannSysroot;
        if (pathFallback) {
            HCCL_VM_WARN(
                "AICPU QEMU fallback: HCCL_VM_QEMU_PATH is not set or invalid ({}), using "
                "PATH lookup [{}] ({}).",
                reason, qemuProgram, qemuVersion);
        }
        if (supportsTbSize) {
            // 显式 -tb-size 置于 device 路径之前。日志记录请求值；0 表示把缓存
            // 策略委托给 QEMU 自身默认（不是零缓存，也不虚报实际分配量）。
            config.args = {qemuProgram, "-tb-size", std::to_string(qemuTbSizeMiB), devBinPath};
            HCCL_VM_INFO(
                "AICPU QEMU: program=[{}], version=[{}], -tb-size request={} MiB{}.", qemuProgram, qemuVersion,
                qemuTbSizeMiB, qemuTbSizeMiB == 0 ? " (0 delegates cache sizing to QEMU default)" : "");
        } else {
            // 旧版 QEMU 不支持 -tb-size：回退为不限制翻译缓存并告警内存风险
            config.args = {qemuProgram, devBinPath};
            HCCL_VM_WARN(
                "AICPU QEMU fallback: [{}] ({}) does not support -tb-size, running without "
                "translation cache limit; QEMU memory may be large at scale. Deploy QEMU 10.2.4 per "
                "README section 2.3.",
                qemuProgram, qemuVersion);
        }
    }

    config.args.push_back(std::to_string(rankId));
    config.args.push_back(std::to_string(deviceKey));

    uint32_t vmLevel = 2;
    const char* levelEnv = std::getenv("HCCL_VM_LEVEL");
    if (levelEnv != nullptr && levelEnv[0] != '\0') {
        char* endptr = nullptr;
        unsigned long val = std::strtoul(levelEnv, &endptr, 10);
        if (endptr != levelEnv && *endptr == '\0') {
            vmLevel = static_cast<uint32_t>(val);
        }
    }
    if (vmLevel == 1) {
        config.envVars["LD_PRELOAD"] = preloadPathL1 + ":" + preloadPath;
    } else {
        config.envVars["LD_PRELOAD"] = preloadPath;
    }
    config.envVars["LD_LIBRARY_PATH"] = libPath;

    HCCL_VM_INFO("AICPU CANN runtime: sysroot=[{}], executable=[{}].", cannSysroot, config.executable);

    return config;
}

} // namespace sim
