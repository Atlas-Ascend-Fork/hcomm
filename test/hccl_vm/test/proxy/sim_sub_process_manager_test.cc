/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// AICPU device 子进程 QEMU 启动配置的定向测试（回退语义版）：
// 覆盖 HCCL_VM_QEMU_PATH / HCCL_VM_QEMU_TB_SIZE 运行时合同——
//   正常路径（配置有效且支持 -tb-size）：argv 带 -tb-size，版本探测一次；
//   配置无效/未配置：回退 PATH 查找 qemu-aarch64-static；
//   旧版 QEMU（不支持 -tb-size）：不传该选项继续运行；
//   TB 非法值：回退默认 16；
//   全机找不到 QEMU：空配置走现有失败路径。
// 替身说明：静态编译的 x86_64 测试替身扮演 QEMU（与真实 QEMU 同为静态 ELF），
// `-version` 时回报内置版本串；按 READY 协议（PIPE_RSP_READY=0xA5，5 字节头）
// 应答并把 argv/关键环境落盘供断言。
// 架构说明：本测试验证 x86_64 宿主的 QEMU 分支合同；原生 ARM 主机走 CANN
// loader 分支（不读取 QEMU 配置），无 ARM 运行环境时按静态/配置检查交付，
// 在 ARM 主机上这些用例跳过（见交付报告）。

#include <gtest/gtest.h>

#include <sys/utsname.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "sim_log.h"
#include "sim_sub_process_manager.h"

// 生产目标引用的日志全局指针：测试内不初始化日志，保持空指针即可（宏会判空跳过）
spdlog::logger* g_logger{nullptr};
namespace hcclvm {
LogWriter g_logWriter = nullptr;
spdlog::level::level_enum g_cannModuleLevels[kLogModuleCount]{};
} // namespace hcclvm

namespace {

const char* const kEnvQemuPath = "HCCL_VM_QEMU_PATH";
const char* const kEnvTbSize = "HCCL_VM_QEMU_TB_SIZE";

// 固定夹具根：InstallPath 在首次解析后缓存安装根，必须在任何用例前就绪（main 中初始化）
const std::string kFixtureRoot = "/tmp/hccl_vm_qemu_cfg_test";
const std::string kInstallRoot = kFixtureRoot + "/install";
const std::string kSysroot = kFixtureRoot + "/sysroot";

bool ArmHost()
{
    utsname utsBuf;
    return uname(&utsBuf) == 0
           && (std::strstr(utsBuf.machine, "aarch64") != nullptr || std::strstr(utsBuf.machine, "arm") != nullptr);
}

bool MakeDirs(const std::string& path) { return ::system(("mkdir -p '" + path + "'").c_str()) == 0; }

bool TouchFile(const std::string& path)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    return out.is_open();
}

// ── 静态替身：两种形态各编译一次，进程内缓存路径 ────────────────────────────
// new：支持 -tb-size（跳过值），-version 回报 "10.2.4-stub"
// old：遇 -tb-size 报错退出（rc=1），-version 回报 "8.2.2-stub"
// device 模式：argv/env 按 STUB_ARGV_DUMP/STUB_ENV_DUMP 落盘 → fd201 写 5 字节
// READY 帧 → 阻塞读 fd200 直到父进程关闭 → 退出 0。
std::string CompileStubProgram(const char* version, int rejectTb)
{
    std::ostringstream flag;
    flag << "-DQEMU_STUB_VERSION=\"\\\"" << version << "\\\"\" -DREJECT_TB=" << rejectTb;
    const std::string tag = std::string(rejectTb ? "old-" : "new-") + version;
    const std::string source = kFixtureRoot + "/stub-" + tag + ".c";
    const std::string program = kFixtureRoot + "/qemu-stub-" + tag;
    std::ofstream out(source, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return {};
    }
    out << "#include <stdio.h>\n"
           "#include <stdlib.h>\n"
           "#include <string.h>\n"
           "#include <unistd.h>\n"
           "int main(int argc, char **argv) {\n"
           "    for (int i = 1; i < argc; ++i) {\n"
           "        if (strcmp(argv[i], \"-tb-size\") == 0) {\n"
           "#if REJECT_TB\n"
           "            fprintf(stderr, \"qemu-stub: unknown option -tb-size\\n\");\n"
           "            return 1;\n"
           "#else\n"
           "            ++i; /* skip value */\n"
           "            continue;\n"
           "#endif\n"
           "        }\n"
           "        if (strcmp(argv[i], \"-version\") == 0) {\n"
           "            printf(\"qemu-aarch64 version \" QEMU_STUB_VERSION \"\\n\");\n"
           "            return 0;\n"
           "        }\n"
           "    }\n"
           "    const char *argvDump = getenv(\"STUB_ARGV_DUMP\");\n"
           "    const char *envDump = getenv(\"STUB_ENV_DUMP\");\n"
           "    if (argvDump != NULL) {\n"
           "        FILE *f = fopen(argvDump, \"w\");\n"
           "        if (f != NULL) { for (int i = 0; i < argc; ++i) fprintf(f, \"%s\\n\", argv[i]); fclose(f); }\n"
           "    }\n"
           "    if (envDump != NULL) {\n"
           "        FILE *f = fopen(envDump, \"w\");\n"
           "        if (f != NULL) {\n"
           "            const char *keys[] = {\"QEMU_LD_PREFIX\", \"LD_PRELOAD\", \"LD_LIBRARY_PATH\",\n"
           "                                  \"HCCL_VM_TEST_MARKER\", NULL};\n"
           "            for (int i = 0; keys[i] != NULL; ++i) {\n"
           "                const char *v = getenv(keys[i]);\n"
           "                fprintf(f, \"%s=%s\\n\", keys[i], v != NULL ? v : \"\");\n"
           "            }\n"
           "            fclose(f);\n"
           "        }\n"
           "    }\n"
           "    unsigned char frame[5] = {0xA5, 0, 0, 0, 0};  /* PipeMessage{PIPE_RSP_READY,len=0} */\n"
           "    (void)write(201, frame, 5);\n"
           "    char buf[256];\n"
           "    while (read(200, buf, sizeof(buf)) > 0) { }\n"
           "    return 0;\n"
           "}\n";
    out.flush();
    if (!out.good()) {
        return {};
    }
    const std::string cmd = "gcc -static -O2 " + flag.str() + " -o '" + program + "' '" + source + "' 2>/dev/null";
    if (::system(cmd.c_str()) != 0) {
        return {};
    }
    return program;
}

std::string NewQemuStub()
{
    static const std::string cached = CompileStubProgram("10.2.4-stub", 0);
    return cached;
}

std::string OldQemuStub()
{
    static const std::string cached = CompileStubProgram("8.2.2-stub", 1);
    return cached;
}

std::vector<std::string> ReadLines(const std::string& path)
{
    std::vector<std::string> lines;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    return lines;
}

class AicpuDeviceConfigTest : public testing::Test {
protected:
    std::string savedPath_;

    void SetUp() override
    {
        if (ArmHost()) {
            GTEST_SKIP() << "x86_64 QEMU-branch contract; native ARM host runs the CANN loader branch";
        }
        ::unsetenv(kEnvQemuPath);
        ::unsetenv(kEnvTbSize);
        ::unsetenv("HCCL_VM_LEVEL");
        ::unsetenv("HCCL_VM_TEST_MARKER");
        savedPath_ = ::getenv("PATH") != nullptr ? ::getenv("PATH") : "";
    }

    void TearDown() override
    {
        if (!savedPath_.empty()) {
            ::setenv("PATH", savedPath_.c_str(), 1);
        }
    }

    // 每个用例独立的替身与落盘目录（argv/env 互不串扰）
    std::string NewCaseDir(const char* name)
    {
        const std::string dir = kFixtureRoot + "/" + name;
        EXPECT_TRUE(MakeDirs(dir)) << "mkdir -p " << dir;
        return dir;
    }

    // 在 caseDir 放置 QEMU 替身并指定落盘文件（kind: "new"/"old"）
    std::string NewStub(const std::string& caseDir, const char* kind)
    {
        const std::string stubBin = (std::string(kind) == "old") ? OldQemuStub() : NewQemuStub();
        EXPECT_FALSE(stubBin.empty()) << "failed to compile static stub (gcc -static required)";
        if (stubBin.empty()) {
            return "";
        }
        const std::string program = caseDir + "/qemu-aarch64-static";
        EXPECT_EQ(::system(("cp '" + stubBin + "' '" + program + "'").c_str()), 0);
        EXPECT_EQ(::setenv("STUB_ARGV_DUMP", (caseDir + "/argv.txt").c_str(), 1), 0);
        EXPECT_EQ(::setenv("STUB_ENV_DUMP", (caseDir + "/env.txt").c_str(), 1), 0);
        return program;
    }

    // 让 PATH 首位是 caseDir（保留原 PATH 尾部，保证 timeout 等系统工具可解析）
    void PrependPath(const std::string& dir) { ASSERT_EQ(::setenv("PATH", (dir + ":" + savedPath_).c_str(), 1), 0); }
};

// ── 正常路径：配置有效且 QEMU 支持 -tb-size ─────────────────────────────────

TEST_F(AicpuDeviceConfigTest, TbUnsetDefaultsTo16)
{
    const std::string stub = NewStub(NewCaseDir("tb-unset"), "new");
    ASSERT_FALSE(stub.empty());
    ASSERT_EQ(::setenv(kEnvQemuPath, stub.c_str(), 1), 0);

    const auto config = sim::CreateAicpuDeviceConfig(3, 7);
    ASSERT_FALSE(config.executable.empty());
    EXPECT_EQ(config.executable, stub);
    EXPECT_EQ(config.args, (std::vector<std::string>{stub, "-tb-size", "16", kInstallRoot + "/bin/device", "3", "7"}));
}

TEST_F(AicpuDeviceConfigTest, TbLeadingZerosReserializedAsDecimal)
{
    const std::string stub = NewStub(NewCaseDir("tb-0016"), "new");
    ASSERT_FALSE(stub.empty());
    ASSERT_EQ(::setenv(kEnvQemuPath, stub.c_str(), 1), 0);
    ASSERT_EQ(::setenv(kEnvTbSize, "0016", 1), 0);

    const auto config = sim::CreateAicpuDeviceConfig(0, 0);
    ASSERT_FALSE(config.executable.empty());
    // 前导 0 必须按解析后的数值重新序列化，避免 QEMU 自动进制解析偏差
    EXPECT_EQ(config.args[1], "-tb-size");
    EXPECT_EQ(config.args[2], "16");
}

TEST_F(AicpuDeviceConfigTest, TbExplicitValues)
{
    struct Case {
        const char* value;
        const char* expectedArgv;
    };
    const Case cases[] = {{"16", "16"}, {"32", "32"}, {"0", "0"}, {"4294967295", "4294967295"}};
    for (const auto& item : cases) {
        const std::string stub = NewStub(NewCaseDir("tb-value"), "new");
        ASSERT_FALSE(stub.empty());
        ASSERT_EQ(::setenv(kEnvQemuPath, stub.c_str(), 1), 0);
        ASSERT_EQ(::setenv(kEnvTbSize, item.value, 1), 0);
        const auto config = sim::CreateAicpuDeviceConfig(1, 2);
        ASSERT_FALSE(config.executable.empty()) << "TB=" << item.value;
        EXPECT_EQ(config.args[2], item.expectedArgv) << "TB=" << item.value;
    }
}

TEST_F(AicpuDeviceConfigTest, TbIllegalValuesFallBackToDefault16)
{
    const char* illegalValues[]
        = {"",
           " 16",
           "16 ",
           "+16",
           "-1",
           "0x10",
           "0b101",
           "16MiB",
           "16m",
           "abc",
           "1.5",
           "4294967296",
           "999999999999999999999999",
           "16\t",
           "\t16"};
    for (const char* value : illegalValues) {
        const std::string stub = NewStub(NewCaseDir("tb-illegal"), "new");
        ASSERT_FALSE(stub.empty());
        ASSERT_EQ(::setenv(kEnvQemuPath, stub.c_str(), 1), 0);
        ASSERT_EQ(::setenv(kEnvTbSize, value, 1), 0);
        // 非法值不阻断启动：回退默认 16 继续运行
        const auto config = sim::CreateAicpuDeviceConfig(0, 0);
        ASSERT_FALSE(config.executable.empty()) << "TB=" << value;
        EXPECT_EQ(config.args[2], "16") << "TB=" << value;
    }
}

TEST_F(AicpuDeviceConfigTest, QemuPathValidSymlinkAccepted)
{
    const std::string caseDir = NewCaseDir("qemu-link");
    const std::string stub = NewStub(caseDir, "new");
    ASSERT_FALSE(stub.empty());
    const std::string link = caseDir + "/qemu-link";
    ASSERT_EQ(::symlink(stub.c_str(), link.c_str()), 0);
    ASSERT_EQ(::setenv(kEnvQemuPath, link.c_str(), 1), 0);

    const auto config = sim::CreateAicpuDeviceConfig(5, 9);
    ASSERT_FALSE(config.executable.empty());
    // 配置的绝对路径原样进入 executable/argv[0]，链接在 exec 时由内核解析
    EXPECT_EQ(config.executable, link);
    EXPECT_EQ(config.args[0], link);
}

// ── 旧版 QEMU：不支持 -tb-size 时不传该选项继续运行 ────────────────────────

TEST_F(AicpuDeviceConfigTest, QemuPathOldVersionRunsBareArgv)
{
    const std::string stub = NewStub(NewCaseDir("qemu-old"), "old");
    ASSERT_FALSE(stub.empty());
    ASSERT_EQ(::setenv(kEnvQemuPath, stub.c_str(), 1), 0);

    const auto config = sim::CreateAicpuDeviceConfig(3, 7);
    ASSERT_FALSE(config.executable.empty());
    // 旧版探测失败（-tb-size 16 -version rc!=0）：回退不带 -tb-size 的旧 argv
    EXPECT_EQ(config.args, (std::vector<std::string>{stub, kInstallRoot + "/bin/device", "3", "7"}));
}

// ── 未配置/配置无效：回退 PATH 查找 ────────────────────────────────────────

TEST_F(AicpuDeviceConfigTest, QemuPathUnsetFallsBackToPathLookup)
{
    const std::string caseDir = NewCaseDir("path-fallback");
    const std::string stub = NewStub(caseDir, "new");
    ASSERT_FALSE(stub.empty());
    PrependPath(caseDir); // HCCL_VM_QEMU_PATH 未设置 → PATH 首位命中替身

    const auto config = sim::CreateAicpuDeviceConfig(1, 1);
    ASSERT_FALSE(config.executable.empty());
    EXPECT_EQ(config.executable, stub);
    EXPECT_EQ(config.args[1], "-tb-size"); // 替身支持 -tb-size，正常路径
    EXPECT_EQ(config.args[2], "16");
}

TEST_F(AicpuDeviceConfigTest, QemuPathInvalidValuesFallBackToPathLookup)
{
    const std::string caseDir = NewCaseDir("path-fallback-invalid");
    const std::string stub = NewStub(caseDir, "old"); // PATH 里放旧版替身
    ASSERT_FALSE(stub.empty());
    PrependPath(caseDir);

    struct Case {
        const char* value;
    };
    const Case cases[] = {"", "bin/qemu-aarch64-static", "/nonexistent/qemu-aarch64-static"};
    for (const auto& item : cases) {
        ASSERT_EQ(::setenv(kEnvQemuPath, item.value, 1), 0);
        const auto config = sim::CreateAicpuDeviceConfig(0, 0);
        ASSERT_FALSE(config.executable.empty()) << "HCCL_VM_QEMU_PATH=" << item.value;
        // 回退到 PATH 中的旧版替身：不带 -tb-size
        EXPECT_EQ(config.executable, stub) << "HCCL_VM_QEMU_PATH=" << item.value;
        EXPECT_EQ(config.args, (std::vector<std::string>{stub, kInstallRoot + "/bin/device", "0", "0"}))
            << "HCCL_VM_QEMU_PATH=" << item.value;
    }

    // 目录 / 不可执行 / 断链同样回退
    const std::string dir = caseDir + "/a-dir";
    ASSERT_TRUE(MakeDirs(dir));
    const std::string noExec = caseDir + "/noexec";
    ASSERT_TRUE(TouchFile(noExec));
    ASSERT_EQ(::chmod(noExec.c_str(), 0644), 0);
    const std::string brokenLink = caseDir + "/broken-link";
    ASSERT_EQ(::symlink((caseDir + "/missing-target").c_str(), brokenLink.c_str()), 0);
    const char* moreCases[] = {"/tmp", noExec.c_str(), brokenLink.c_str()};
    for (const char* value : moreCases) {
        ASSERT_EQ(::setenv(kEnvQemuPath, value, 1), 0);
        const auto config = sim::CreateAicpuDeviceConfig(0, 0);
        ASSERT_FALSE(config.executable.empty()) << "HCCL_VM_QEMU_PATH=" << value;
        EXPECT_EQ(config.executable, stub) << "HCCL_VM_QEMU_PATH=" << value;
    }
}

TEST_F(AicpuDeviceConfigTest, QemuNotFoundAnywhereHardFails)
{
    const std::string emptyDir = kFixtureRoot + "/empty-path-dir";
    ASSERT_TRUE(MakeDirs(emptyDir));
    ASSERT_EQ(::setenv("PATH", emptyDir.c_str(), 1), 0); // PATH 中无 qemu，也未配置

    const auto config = sim::CreateAicpuDeviceConfig(0, 0);
    EXPECT_TRUE(config.executable.empty() && config.args.empty());
}

// ── 端到端：真实 CreateProcess 路径 ────────────────────────────────────────

TEST_F(AicpuDeviceConfigTest, EndToEndArgvOrderAndEnvInheritance)
{
    const std::string caseDir = NewCaseDir("e2e-argv");
    const std::string stub = NewStub(caseDir, "new");
    ASSERT_FALSE(stub.empty());
    ASSERT_EQ(::setenv(kEnvQemuPath, stub.c_str(), 1), 0);
    ASSERT_EQ(::setenv("HCCL_VM_TEST_MARKER", "keep", 1), 0);

    sim::SubProcessManager procMgr;
    const auto config = sim::CreateAicpuDeviceConfig(3, 7);
    ASSERT_FALSE(config.executable.empty());
    ASSERT_EQ(procMgr.CreateProcess(config), 0) << "child failed to become READY";

    // 既有环境继承：QEMU_LD_PREFIX/CANN sysroot、LD_PRELOAD、LD_LIBRARY_PATH 保持在子进程
    const auto envLines = ReadLines(caseDir + "/env.txt");
    ASSERT_EQ(envLines.size(), 4U);
    EXPECT_EQ(envLines[0], "QEMU_LD_PREFIX=" + kSysroot);
    EXPECT_TRUE(
        envLines[1].rfind("LD_PRELOAD=", 0) == 0
        && envLines[1].find("lib/aarch64/libhccl_device_proxy.so") != std::string::npos);
    EXPECT_TRUE(envLines[2].rfind("LD_LIBRARY_PATH=", 0) == 0 && envLines[2].find(kSysroot) != std::string::npos);
    EXPECT_EQ(envLines[3], "HCCL_VM_TEST_MARKER=keep");

    // argv 顺序：<QEMU> -tb-size 16 <device> <rankId> <deviceKey> <h2dReadFd> <d2hWriteFd>
    const auto argvLines = ReadLines(caseDir + "/argv.txt");
    ASSERT_EQ(argvLines.size(), 8U);
    EXPECT_EQ(argvLines[0], stub);
    EXPECT_EQ(argvLines[1], "-tb-size");
    EXPECT_EQ(argvLines[2], "16");
    EXPECT_EQ(argvLines[3], kInstallRoot + "/bin/device");
    EXPECT_EQ(argvLines[4], "3");
    EXPECT_EQ(argvLines[5], "7");
    EXPECT_EQ(argvLines[6], "200");
    EXPECT_EQ(argvLines[7], "201");

    ASSERT_EQ(procMgr.DestroyProcess(), 0);
}

TEST_F(AicpuDeviceConfigTest, EndToEndTbZeroPassedExplicitly)
{
    const std::string caseDir = NewCaseDir("e2e-tb0");
    const std::string stub = NewStub(caseDir, "new");
    ASSERT_FALSE(stub.empty());
    ASSERT_EQ(::setenv(kEnvQemuPath, stub.c_str(), 1), 0);
    ASSERT_EQ(::setenv(kEnvTbSize, "0", 1), 0);

    sim::SubProcessManager procMgr;
    const auto config = sim::CreateAicpuDeviceConfig(0, 0);
    ASSERT_FALSE(config.executable.empty());
    ASSERT_EQ(procMgr.CreateProcess(config), 0);

    const auto argvLines = ReadLines(caseDir + "/argv.txt");
    ASSERT_EQ(argvLines.size(), 8U);
    EXPECT_EQ(argvLines[1], "-tb-size");
    EXPECT_EQ(argvLines[2], "0"); // 0 显式传递，由 QEMU 自身默认策略决定缓存

    ASSERT_EQ(procMgr.DestroyProcess(), 0);
}

TEST_F(AicpuDeviceConfigTest, EndToEndOldQemuFallbackBareArgv)
{
    const std::string caseDir = NewCaseDir("e2e-old-fallback");
    const std::string stub = NewStub(caseDir, "old");
    ASSERT_FALSE(stub.empty());
    ASSERT_EQ(::setenv(kEnvQemuPath, stub.c_str(), 1), 0);

    sim::SubProcessManager procMgr;
    const auto config = sim::CreateAicpuDeviceConfig(2, 5);
    ASSERT_FALSE(config.executable.empty());
    ASSERT_EQ(procMgr.CreateProcess(config), 0);

    // 旧版回退：argv = <QEMU> <device> <rankId> <deviceKey> <h2dReadFd> <d2hWriteFd>
    const auto argvLines = ReadLines(caseDir + "/argv.txt");
    ASSERT_EQ(argvLines.size(), 6U);
    EXPECT_EQ(argvLines[0], stub);
    EXPECT_EQ(argvLines[1], kInstallRoot + "/bin/device");
    EXPECT_EQ(argvLines[2], "2");
    EXPECT_EQ(argvLines[3], "5");
    EXPECT_EQ(argvLines[4], "200");
    EXPECT_EQ(argvLines[5], "201");

    ASSERT_EQ(procMgr.DestroyProcess(), 0);
}

TEST_F(AicpuDeviceConfigTest, PathShadowDoesNotPreemptConfiguredAbsoluteProgram)
{
    const std::string caseDir = NewCaseDir("path-shadow");
    const std::string stub = NewStub(caseDir, "new");
    ASSERT_FALSE(stub.empty());

    // PATH 前置目录里放置同名旧版程序（同样能完成 READY 协议），验证不被抢先选中
    const std::string stubBin = OldQemuStub();
    ASSERT_FALSE(stubBin.empty());
    const std::string shadowDir = caseDir + "/shadow-bin";
    ASSERT_TRUE(MakeDirs(shadowDir));
    ASSERT_EQ(::system(("cp '" + stubBin + "' '" + shadowDir + "/qemu-aarch64-static'").c_str()), 0);
    PrependPath(shadowDir);
    ASSERT_EQ(::setenv(kEnvQemuPath, stub.c_str(), 1), 0);

    sim::SubProcessManager procMgr;
    const auto config = sim::CreateAicpuDeviceConfig(1, 1);
    ASSERT_FALSE(config.executable.empty());
    EXPECT_EQ(config.executable, stub);
    ASSERT_EQ(procMgr.CreateProcess(config), 0);

    // 实际执行的是配置的绝对路径：替身落盘的 argv[0] 即被 exec 的程序
    const auto argvLines = ReadLines(caseDir + "/argv.txt");
    ASSERT_EQ(argvLines.size(), 8U);
    EXPECT_EQ(argvLines[0], stub);

    ASSERT_EQ(procMgr.DestroyProcess(), 0);
}

} // namespace

// main：InstallPath 首次解析即缓存安装根，夹具目录与固定环境必须在用例前就绪
int main(int argc, char** argv)
{
    // 清掉上次崩溃/中断可能残留的夹具，防止旧落盘文件伪造断言结果
    ::system(("rm -rf '" + kFixtureRoot + "'").c_str());

    const std::string dirs[] = {
        kInstallRoot + "/lib/aarch64",
        kSysroot + "/lib64",
        kSysroot + "/usr/lib64",
    };
    for (const auto& dir : dirs) {
        if (!MakeDirs(dir)) {
            std::fprintf(stderr, "failed to create fixture dir: %s\n", dir.c_str());
            return 2;
        }
    }
    // CANN sysroot 完整性检查只认四个文件存在（IsCompleteCannSysroot）
    const std::string sysrootFiles[] = {
        kSysroot + "/lib64/ld-linux-aarch64.so.1",
        kSysroot + "/lib64/libc.so.6",
        kSysroot + "/usr/lib64/libstdc++.so.6",
        kSysroot + "/usr/lib64/libgcc_s.so.1",
    };
    for (const auto& file : sysrootFiles) {
        if (!TouchFile(file)) {
            std::fprintf(stderr, "failed to create fixture file: %s\n", file.c_str());
            return 2;
        }
    }
    // LD_PRELOAD 目标仅要求存在（真实场景为 aarch64 设备代理库；静态替身不加载它）
    if (!TouchFile(kInstallRoot + "/lib/aarch64/libhccl_device_proxy.so")) {
        std::fprintf(stderr, "failed to create fixture preload file\n");
        return 2;
    }

    if (::setenv("HCCL_VM_INSTALL_ROOT", kInstallRoot.c_str(), 1) != 0
        || ::setenv("CANN_HCC_SYSROOT", kSysroot.c_str(), 1) != 0) {
        std::fprintf(stderr, "failed to set fixture environment\n");
        return 2;
    }

    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    ::system(("rm -rf '" + kFixtureRoot + "'").c_str());
    return rc;
}
