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
#include "gtest/gtest.h"
#include "hccp_shm_infra.h"
#include "hccp_vm_net.h"
#include "hccp_verdict.h"

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
    char virtText[INET_ADDRSTRLEN] = {0};
    (void)inet_ntop(AF_INET, &virt, virtText, sizeof(virtText));
    (void)snprintf(name, 24, "vmnic_%s", virtText[0] != '\0' ? virtText : "?");
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
    /* Isolated run slot: HCCP_ST_RUN_ID=N (N>0) moves this runner's whole
     * verdict log tree (purge + scan) into <logRoot>/run_<N>, so concurrent
     * runners of the dual-run verification never delete or judge each other's
     * logs. N=0 (unset/invalid) keeps the legacy shared tree byte-for-byte.
     * The location decision lives once in hccp_shm_infra.h (hccp_st_log_root),
     * shared with the dlog writer: the scanned root and the written root
     * cannot drift apart. */
    if (hccp_st_log_root(dir, sz, (int)g_hccp_st_run_id) != 0) {
        dir[0] = '\0';
    }
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
            if (HccpSt::IsForeignRunDir(ent->d_name, g_hccp_st_run_id)) {
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

static HccpSt::CaseLoadResult LoadTestCase()
{
    const char* casePath = getenv("HCCP_VM_CASE");
    char defaultPath[PATH_MAX];
    if (casePath == nullptr || casePath[0] == '\0') {
        char exePath[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
        if (count <= 0) {
            (void)fprintf(stderr, "[RUNNER] readlink /proc/self/exe failed, no case loaded\n");
            return {};
        }
        exePath[count] = '\0';
        int length = snprintf(defaultPath, sizeof(defaultPath), "%s/../lib/libhccp_case.so", dirname(exePath));
        if (length < 0 || length >= static_cast<int>(sizeof(defaultPath))) {
            (void)fprintf(stderr, "[RUNNER] case path too long, no case loaded\n");
            return {};
        }
        casePath = defaultPath;
    }
    // Keep the library loaded: registered gtest objects require it through process exit.
    void* handle = dlopen(casePath, RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        (void)fprintf(stderr, "[RUNNER] dlopen case %s failed: %s\n", casePath, dlerror());
        return {false, HccpSt::SanitizeDetail(casePath)};
    }
    printf("[RUNNER] case loaded: %s\n", casePath);
    return {true, HccpSt::SanitizeDetail(casePath)};
}

// Execution events exclude selected-but-unrun tests (list/repeat=0/environment skip).
class HccpVerdictTestListener : public ::testing::EmptyTestEventListener {
public:
    explicit HccpVerdictTestListener(HccpSt::GtestRunResult& result) : result_(result) {}

    void OnTestStart(const ::testing::TestInfo&) override { result_.started++; }

    void OnTestEnd(const ::testing::TestInfo& testInfo) override
    {
        result_.finished++;
        if (testInfo.result()->Skipped()) {
            result_.skipped++;
        }
    }

private:
    HccpSt::GtestRunResult& result_;
};

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
    /* Lock before purge (a second runner would delete logs this one is writing);
     * purge before capture (startup noise must stay out of the error window). */
    g_hccp_st_run_id = HccpStParseRunId();
    HccpStRunGuardAcquireOrDie();
    VerdictPurgeLogDir();

    HccpSt::VerdictResults results;
    HccpSt::StdoutCapture capture;

    /* Capture opens before anything prints: every later byte flows through the tee. */
    capture.Start();
    printf("Running hccp-vm runner\n");
    results.caseLoad = LoadTestCase();

    testing::InitGoogleTest(&argc, argv);
    auto* unit = ::testing::UnitTest::GetInstance();
    /* Listener counts real executions while tests run: the count APIs report
     * selected-but-not-executed cases as passed. gtest owns Append'ed listeners;
     * Release below hands ownership back for explicit delete. */
    auto* listener = new HccpVerdictTestListener(results.gtest);
    unit->listeners().Append(listener);
    ::testing::AddGlobalTestEnvironment(new HccpShmEnvironment);

    results.gtest.returnCode = RUN_ALL_TESTS();

    /* Finish joins the tee thread; the log tree is quiescent only now — every
     * fixture TearDown has reaped its host/device children. */
    results.stdoutErrors = capture.Finish();
    delete unit->listeners().Release(listener);
    /* Display-only counts; the pass gate uses the listener's execution counts. */
    results.gtest.selected = unit->test_to_run_count();
    results.gtest.failed = unit->failed_test_count();

    char logDir[PATH_MAX];
    VerdictGetLogDir(logDir, sizeof(logDir));
    /* An absent log tree is a legal empty result (nothing was ever logged): the
     * scan diagnoses it on stderr instead of failing it. */
    results.logErrors = HccpSt::ScanLogFiles(logDir, results.gtest.started > 0, g_hccp_st_run_id);

    const auto decision = HccpSt::EvaluateVerdict(results);
    /* A failed restore closed that side's descriptor: printing on it would silently drop
     * the report, so the verdict goes to a stream that still sinks bytes. */
    HccpSt::PrintVerdict(results, decision, HccpSt::VerdictOutputStream(results.stdoutErrors));
    return decision.Passed() ? EXIT_SUCCESS : EXIT_FAILURE;
}
