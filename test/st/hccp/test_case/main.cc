/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <stdio.h>
#include <ctype.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/file.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <limits.h>
#include <libgen.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <atomic>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "hccp_shm_infra.h"
#include "hccp_vm_net.h"

struct IfAppendCtx {
    struct ifaddrs* tail;
};

static void AppendLoopbackNic(uint32_t phyId, uint32_t iface, void* opaque)
{
    struct IfAppendCtx* ctx = (struct IfAppendCtx*)opaque;
    struct ifaddrs* node = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
    struct sockaddr_in* sa = (struct sockaddr_in*)calloc(1, sizeof(struct sockaddr_in));
    struct sockaddr_in* mask = (struct sockaddr_in*)calloc(1, sizeof(struct sockaddr_in));
    char* name = (char*)calloc(1, 24);
    if (node == nullptr || sa == nullptr || mask == nullptr || name == nullptr) {
        free(node);
        free(sa);
        free(mask);
        free(name);
        return;
    }
    uint32_t virtIp = HccpVmNicIp(phyId, iface);
    struct in_addr loopback = HccpVmToLoopback((struct in_addr){.s_addr = htonl(virtIp)});
    sa->sin_family = AF_INET;
    sa->sin_addr = loopback;
    mask->sin_family = AF_INET;
    mask->sin_addr.s_addr = htonl(0xFF000000u);
    struct in_addr virt = {};
    virt.s_addr = htonl(virtIp);
    (void)snprintf(name, 24, "vmnic_%s", inet_ntoa(virt));
    node->ifa_name = name;
    node->ifa_addr = (struct sockaddr*)sa;
    node->ifa_netmask = (struct sockaddr*)mask;
    node->ifa_flags = (unsigned int)(IFF_UP | IFF_RUNNING);
    node->ifa_next = nullptr;
    ctx->tail->ifa_next = node;
    ctx->tail = node;
}
extern "C" int getifaddrs(struct ifaddrs** ifap)
{
    static int (*real_getifaddrs)(struct ifaddrs**) = nullptr;
    if (!real_getifaddrs) {
        real_getifaddrs = (int (*)(struct ifaddrs**))dlsym(RTLD_NEXT, "getifaddrs");
    }
    int ret = real_getifaddrs ? real_getifaddrs(ifap) : -1;
    if (ret != 0 || ifap == nullptr || *ifap == nullptr) {
        return ret;
    }
    struct ifaddrs* tail = *ifap;
    while (tail->ifa_next != nullptr) {
        tail = tail->ifa_next;
    }
    static const uint32_t vmPhyIds[2] = {1, 5};
    struct IfAppendCtx ctx = {};
    ctx.tail = tail;
    HccpVmForeachNic(vmPhyIds, 2, AppendLoopbackNic, &ctx);
    return ret;
}

class HccpShmEnvironment : public ::testing::Environment {
public:
    void SetUp() override
    {
        hccp_shm_purge_all();
        hccp_buf_pool_init();
        /* hccp_shm_ensure_zero(H2D) retired with the memfd migration: the readiness
         * table is a born-zero memfd now, so the no-op call was dropped here (the
         * infra-side symbol can go away per core-eng's retirement note). */
    }
    void TearDown() override { hccp_shm_destroy_all(); }
};

static const size_t VERDICT_MAX_REPORT = 20;

static int g_verdict_real_out = -1;
static int g_verdict_real_err = -1;
static int g_verdict_pipe_read = -1;
static pthread_t g_verdict_tee_thread;
static pthread_mutex_t g_verdict_lock = PTHREAD_MUTEX_INITIALIZER;
static std::vector<std::string> g_verdict_stdout_errors;
static bool g_case_load_failed = false;

static bool VerdictLineHasError(const std::string& line)
{
    static const char needle[] = "error";
    const size_t nlen = sizeof(needle) - 1;
    if (line.size() < nlen) {
        return false;
    }
    for (size_t i = 0; i + nlen <= line.size(); i++) {
        size_t j = 0;
        while (j < nlen && tolower((unsigned char)line[i + j]) == tolower((unsigned char)needle[j])) {
            j++;
        }
        if (j == nlen) {
            return true;
        }
    }
    return false;
}

static void VerdictRecord(const std::string& line)
{
    pthread_mutex_lock(&g_verdict_lock);
    if (g_verdict_stdout_errors.size() < VERDICT_MAX_REPORT) {
        g_verdict_stdout_errors.push_back(line);
    }
    pthread_mutex_unlock(&g_verdict_lock);
}

static void* VerdictTeeMain(void* arg)
{
    (void)arg;
    char buf[4096];
    std::string pending;
    ssize_t n = 0;
    while ((n = read(g_verdict_pipe_read, buf, sizeof(buf))) > 0) {
        ssize_t off = 0;
        while (off < n) {
            ssize_t w = write(g_verdict_real_out, buf + off, (size_t)(n - off));
            if (w <= 0) {
                break;
            }
            off += w;
        }
        pending.append(buf, (size_t)n);
        size_t nl = 0;
        while ((nl = pending.find('\n')) != std::string::npos) {
            std::string line = pending.substr(0, nl);
            pending.erase(0, nl + 1);
            if (VerdictLineHasError(line)) {
                VerdictRecord(line);
            }
        }
    }
    if (!pending.empty() && VerdictLineHasError(pending)) {
        VerdictRecord(pending);
    }
    return nullptr;
}

static void VerdictCaptureStart()
{
    int fds[2];
    if (pipe(fds) != 0) {
        return;
    }
    (void)fflush(stdout);
    (void)fflush(stderr);
    g_verdict_real_out = dup(STDOUT_FILENO);
    g_verdict_real_err = dup(STDERR_FILENO);
    if (g_verdict_real_out < 0 || g_verdict_real_err < 0) {
        if (g_verdict_real_out >= 0) {
            (void)close(g_verdict_real_out);
        }
        if (g_verdict_real_err >= 0) {
            (void)close(g_verdict_real_err);
        }
        (void)close(fds[0]);
        (void)close(fds[1]);
        g_verdict_real_out = g_verdict_real_err = -1;
        return;
    }
    (void)dup2(fds[1], STDOUT_FILENO);
    (void)dup2(fds[1], STDERR_FILENO);
    (void)close(fds[1]);
    (void)setvbuf(stdout, nullptr, _IONBF, 0);
    g_verdict_pipe_read = fds[0];
    if (pthread_create(&g_verdict_tee_thread, nullptr, VerdictTeeMain, nullptr) != 0) {
        dup2(g_verdict_real_out, STDOUT_FILENO);
        dup2(g_verdict_real_err, STDERR_FILENO);
        close(g_verdict_real_out);
        close(g_verdict_real_err);
        close(g_verdict_pipe_read);
        g_verdict_real_out = g_verdict_real_err = g_verdict_pipe_read = -1;
    }
}

static void VerdictCaptureEnd()
{
    if (g_verdict_pipe_read < 0) {
        return;
    }
    (void)fflush(stdout);
    (void)fflush(stderr);
    (void)close(STDOUT_FILENO);
    (void)close(STDERR_FILENO);
    (void)pthread_join(g_verdict_tee_thread, nullptr);
    (void)dup2(g_verdict_real_out, STDOUT_FILENO);
    (void)dup2(g_verdict_real_err, STDERR_FILENO);
    close(g_verdict_real_out);
    close(g_verdict_real_err);
    close(g_verdict_pipe_read);
    g_verdict_real_out = g_verdict_real_err = g_verdict_pipe_read = -1;
}

/* Run-slot identity (HCCP_ST_RUN_ID): 0 = legacy shared layout (default,
 * behavior byte-for-byte unchanged); N>0 = isolated slot -- verdict log tree
 * moves to <logRoot>/run_<N> and the guard lock file becomes per-slot
 * (see HccpStRunGuardAcquireOrDie). Parsed once at the very top of main(),
 * before the guard and before any verdict-log use. */
static unsigned long g_hccp_st_run_id = 0;

/* Single-source RUN_ID parsing: infra.h's hccp_st_run_id() is the one parser
 * (strtol full-string consumption, documented 0..2 bound, warn-once fallback
 * to 0). A second, looser parser here would split the runner identity -- the
 * verdict/log/lock side taking slot N while the port and dlog sides fall back
 * to 0 -- reopening the silent-port-ambush hazard the strict parser exists to
 * close. Keep only the startup-once caching structure. */
static unsigned long HccpStParseRunId(void) { return (unsigned long)hccp_st_run_id(); }

static void VerdictGetLogDir(char* dir, size_t sz)
{
    dir[0] = '\0';
    char exe_path[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (count <= 0) {
        return;
    }
    exe_path[count] = '\0';
    const char* dir_path = dirname(exe_path);
    /* Isolated run slot: HCCP_ST_RUN_ID=N (N>0) moves this runner's whole
     * verdict log tree (purge + scan) into <logRoot>/run_<N>, so concurrent
     * runners of the dual-run verification never delete or judge each other's
     * logs. N=0 (unset/invalid) keeps the legacy shared tree byte-for-byte.
     * The run_<N> format is agreed with the dlog-side GetLogRoot change
     * (t4 batch 0): decimal, no leading zeros. */
    int n;
    if (g_hccp_st_run_id > 0) {
        n = snprintf(dir, sz, "%s/../ascend/log/run_%lu", dir_path, g_hccp_st_run_id);
    } else {
        n = snprintf(dir, sz, "%s/../ascend/log", dir_path);
    }
    if (n >= (int)sz) {
        dir[0] = '\0';
    }
}

/* Foreign run slot: a run_<N> directory (N decimal) belonging to a concurrent
 * HCCP_ST_RUN_ID runner. The N=0 legacy purge/scan must never touch these:
 * deleting them would destroy a live runner's in-flight logs, scanning them
 * would judge another runner's logs (a false-FAIL vector subtler than the
 * purge one). Matched only while we ourselves run at N=0: with RUN_ID=N>0 our
 * whole log dir IS run_<N> and nothing foreign can be nested inside it.
 * Note: the legacy dlog's own "run"/"debug" case dirs do not match (they
 * carry no _<digits> suffix). */
static bool VerdictIsForeignRunDir(const char* name)
{
    if (g_hccp_st_run_id != 0 || strncmp(name, "run_", 4) != 0 || name[4] == '\0') {
        return false;
    }
    for (const char* p = name + 4; *p != '\0'; p++) {
        if (*p < '0' || *p > '9') {
            return false;
        }
    }
    return true;
}

static void VerdictPurgeTree(const char* dir)
{
    DIR* d = opendir(dir);
    if (d == nullptr) {
        return;
    }
    const struct dirent* ent = nullptr;
    while ((ent = readdir(d)) != nullptr) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }
        char path[PATH_MAX];
        if (snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name) >= (int)sizeof(path)) {
            continue;
        }
        struct stat st = {};
        if (lstat(path, &st) != 0) {
            continue;
        }
        if (S_ISDIR(st.st_mode)) {
            if (VerdictIsForeignRunDir(ent->d_name)) {
                continue; /* a concurrent runner's live run_<N> slot: not ours to delete */
            }
            VerdictPurgeTree(path);
            (void)rmdir(path);
        } else {
            (void)unlink(path);
        }
    }
    closedir(d);
}

static void VerdictPurgeLogDir()
{
    char log_dir[PATH_MAX];
    VerdictGetLogDir(log_dir, sizeof(log_dir));
    if (log_dir[0] == '\0') {
        return;
    }
    VerdictPurgeTree(log_dir);
}

static void VerdictScanTree(const char* dir, const std::string& rel, std::vector<std::string>& errors)
{
    DIR* d = opendir(dir);
    if (d == nullptr) {
        return;
    }
    const struct dirent* ent = nullptr;
    while ((ent = readdir(d)) != nullptr) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }
        char path[PATH_MAX];
        if (snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name) >= (int)sizeof(path)) {
            continue;
        }
        struct stat st = {};
        if (lstat(path, &st) != 0) {
            continue;
        }
        std::string name = rel.empty() ? ent->d_name : rel + "/" + ent->d_name;
        if (S_ISDIR(st.st_mode)) {
            if (VerdictIsForeignRunDir(ent->d_name)) {
                continue; /* a concurrent runner's logs: not ours to judge */
            }
            VerdictScanTree(path, name, errors);
            continue;
        }
        const size_t nlen = strlen(ent->d_name);
        if (nlen < 4 || strcmp(ent->d_name + nlen - 4, ".log") != 0) {
            continue;
        }
        FILE* f = fopen(path, "r");
        if (f == nullptr) {
            continue;
        }
        char* line = nullptr;
        size_t cap = 0;
        ssize_t len = 0;
        while ((len = getline(&line, &cap, f)) > 0) {
            if (strncmp(line, "[ERROR]", 7) == 0 && errors.size() < VERDICT_MAX_REPORT) {
                if (line[len - 1] == '\n') {
                    line[len - 1] = '\0';
                }
                errors.push_back(name + ": " + line);
            }
        }
        free(line);
        (void)fclose(f);
    }
    (void)closedir(d);
}

static void VerdictScanLogFiles(std::vector<std::string>& errors)
{
    char log_dir[PATH_MAX];
    VerdictGetLogDir(log_dir, sizeof(log_dir));
    if (log_dir[0] == '\0') {
        return;
    }
    VerdictScanTree(log_dir, "", errors);
}

static void LoadTestCase()
{
    const char* casePath = getenv("HCCP_VM_CASE");
    char defaultPath[PATH_MAX];
    if (casePath == nullptr || casePath[0] == '\0') {
        char exe_path[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
        if (count <= 0) {
            (void)fprintf(stderr, "[RUNNER] readlink /proc/self/exe failed, no case loaded\n");
            g_case_load_failed = true;
            return;
        }
        exe_path[count] = '\0';
        const char* dir = dirname(exe_path);
        (void)snprintf(defaultPath, sizeof(defaultPath), "%s/../lib/libhccp_case.so", dir);
        casePath = defaultPath;
    }
    void* handle = dlopen(casePath, RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        (void)fprintf(stderr, "[RUNNER] dlopen case %s failed: %s\n", casePath, dlerror());
        g_case_load_failed = true;
        return;
    }
    printf("[RUNNER] case loaded: %s\n", casePath);
}

static void VerdictReportAndExit(int gtest_ret)
{
    std::vector<std::string> log_errors;
    VerdictScanLogFiles(log_errors);

    if (g_case_load_failed) {
        printf("[VERDICT] FAIL: case .so not loaded, no test ran\n");
    }
    if (!g_verdict_stdout_errors.empty()) {
        printf(
            "[VERDICT] FAIL: stdout contains error lines (%zu shown, cap=%zu):\n", g_verdict_stdout_errors.size(),
            VERDICT_MAX_REPORT);
        for (const auto& stdoutErr : g_verdict_stdout_errors) {
            printf("  [stdout] %s\n", stdoutErr.c_str());
        }
    }
    if (!log_errors.empty()) {
        printf(
            "[VERDICT] FAIL: business log contains ERROR lines (%zu shown, cap=%zu):\n", log_errors.size(),
            VERDICT_MAX_REPORT);
        for (const auto& logErr : log_errors) {
            printf("  [log] %s\n", logErr.c_str());
        }
    }
    if (g_case_load_failed || !g_verdict_stdout_errors.empty() || !log_errors.empty()) {
        printf(
            "[VERDICT] FAILED (gtest=%d, stdout_err=%zu, log_err=%zu)\n", gtest_ret, g_verdict_stdout_errors.size(),
            log_errors.size());
        exit(EXIT_FAILURE);
    }
    printf("[VERDICT] PASS: no error in stdout or business logs (gtest=%d)\n", gtest_ret);
    if (gtest_ret != 0) {
        exit(EXIT_FAILURE);
    }
    exit(EXIT_SUCCESS);
}

/* Single-run guard for the hccp ST simulation: the framework serializes runs
 * because the current SysV segments carry machine-global state, so a second
 * overlapping run corrupts the first. The guard is a non-blocking flock on a
 * lock file derived from the running binary's own tree (<exe_dir>/../, i.e.
 * the hccp_test install dir, the same anchor VerdictGetLogDir and the service
 * children's VmHome() use); a contender fails fast instead of racing the
 * holder. A build-tree-local path (rather than a fixed /tmp name) follows the
 * team's build-tree exclusive-unit rule, survives /tmp cleaners deleting a
 * stale lock file (which would silently hand two runners two different lock
 * inodes), and matches the post-memfd end state where the tree is the only
 * shared resource; the residual pre-memfd window of two different trees
 * racing on the machine-global SysV keys is accepted for this stopgap.
 *
 * HCCP_ST_LOCK_BYPASS=1 skips the lock acquisition entirely (no open, no
 * flock, nothing held): a debug escape hatch for forcing a same-slot dual
 * run. Non-routine -- the memfd end-state gets its concurrency isolation
 * from HCCP_ST_RUN_ID namespaces (segments, ports, log trees and lock files
 * are all per-N), so dual runners use distinct RUN_IDs and each holds its
 * own lock without contention. Default (unset/any other value) keeps the
 * fail-fast guard.
 *
 * HCCP_ST_RUN_ID=N moves the lock file to the per-slot name
 * .hccp_st_run_<N>.lock, tree-anchored at <exe_dir>/../ (captain's final
 * caliber: slot naming for every N including 0; /tmp only as the
 * readlink-failure fallback). The lock SEMANTICS are classic fail-fast for
 * every N. The LOG tree stays split: N=0 (unset or explicit) keeps the
 * legacy log path byte-for-byte; N>0 derives <logRoot>/run_<N>.
 *
 * TRANSITION WINDOW: until the t4 memfd migration fully lands (batch 3),
 * the SysV segments are still on global keys -- RUN_ID>0 does NOT yet make
 * two concurrent runners safe (verified: 46 FAILED + residue across purge
 * cycles). N>0 concurrency verification is deferred to t5, after t4.
 *
 * The kernel releases the flock when the last fd holder exits, so a SIGKILLed
 * run cannot wedge the guard: it is held exactly as long as the run's process
 * tree lives (forked host children and exec'ed hccp_service.bin children keep
 * holding it, deliberately fail-fasting the next run while orphaned stragglers
 * still touch the shared segments). The lock fd is never closed and no signal
 * handler is installed on the success path. */
static void HccpStRunGuardAcquireOrDie(void)
{
    const char* bypass = getenv("HCCP_ST_LOCK_BYPASS");
    if (bypass != NULL && bypass[0] == '1' && bypass[1] == '\0') {
        fprintf(stderr, "[GUARD] HCCP_ST_LOCK_BYPASS=1 set, skipping run lock acquisition\n");
        return;
    }

    char exe[PATH_MAX] = {0};
    char lockPath[PATH_MAX] = {0};
    int n = 0;
    ssize_t cnt = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (cnt > 0) {
        exe[cnt] = '\0';
        /* dirname() may rewrite its buffer in place, which would corrupt the
         * exe copy the holder record below prints -- derive the lock path from
         * a private copy instead. */
        char exeDir[PATH_MAX] = {0};
        (void)snprintf(exeDir, sizeof(exeDir), "%s", exe);
        /* Uniform per-run slot naming for every N, captain's final caliber:
         * .hccp_st_run_<N>.lock, tree-anchored. An unset variable and an
         * explicit HCCP_ST_RUN_ID=0 both derive N=0 -- structurally the same
         * lock, so two logical id-0 runners always contend with each other. */
        n = snprintf(lockPath, sizeof(lockPath), "%s/../.hccp_st_run_%lu.lock", dirname(exeDir), g_hccp_st_run_id);
    }
    if (lockPath[0] == '\0' || n >= (int)sizeof(lockPath)) {
        (void)snprintf(lockPath, sizeof(lockPath), "/tmp/hccp_st_run.lock");
    }

    int fd = open(lockPath, O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        fprintf(stderr, "[GUARD] open %s failed: %s\n", lockPath, strerror(errno));
        exit(EXIT_FAILURE);
    }
    /* The creating umask may strip group/other bits; restore them so a runner
     * from another account still contends on the same guard. */
    if (fchmod(fd, 0666) != 0 && errno != EPERM) {
        fprintf(stderr, "[GUARD] fchmod %s failed: %s\n", lockPath, strerror(errno));
        (void)close(fd);
        exit(EXIT_FAILURE);
    }
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            char holder[256] = {0};
            (void)pread(fd, holder, sizeof(holder) - 1, 0);
            fprintf(
                stderr, "[GUARD] another hccp_test run holds %s (holder: %s), refusing to start\n", lockPath,
                holder[0] != '\0' ? holder : "unknown");
        } else {
            fprintf(stderr, "[GUARD] flock %s failed: %s\n", lockPath, strerror(errno));
        }
        (void)close(fd);
        exit(EXIT_FAILURE);
    }
    /* Publish the holder for contenders' diagnostics. Only the lock holder
     * ever writes here, so the record always describes the live run. */
    char rec[300];
    int len = snprintf(rec, sizeof(rec), "pid=%d exe=%s\n", (int)getpid(), exe[0] != '\0' ? exe : "?");
    if (len > 0 && (size_t)len < sizeof(rec)) {
        (void)ftruncate(fd, 0);
        int off = 0;
        while (off < len) {
            ssize_t w = write(fd, rec + off, (size_t)(len - off));
            if (w <= 0) {
                break;
            }
            off += (int)w;
        }
    }
    printf("[GUARD] run lock acquired (pid=%d, %s)\n", (int)getpid(), lockPath);
}

GTEST_API_ int main(int argc, char** argv)
{
    g_hccp_st_run_id = HccpStParseRunId();
    HccpStRunGuardAcquireOrDie();
    VerdictPurgeLogDir();
    VerdictCaptureStart();
    printf("Running hccp-vm runner\n");
    LoadTestCase();
    testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new HccpShmEnvironment);
    int gtest_ret = RUN_ALL_TESTS();
    VerdictCaptureEnd();
    VerdictReportAndExit(gtest_ret);
    return gtest_ret;
}
