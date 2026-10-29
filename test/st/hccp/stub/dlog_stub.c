/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * Test-side replacement of libunified_dlog.so for the hccp system tests.
 *
 * Log layout under <install root>/ascend/log:
 *   debug/<TestCase>/plog/host-phy<phyId>-pid<pid>.log       host child process, level controlled
 *   debug/<TestCase>/device-<phyId>/hccp_service-<pid>.log   device process, level controlled
 *   run/<TestCase>/plog/...                                  same files, always printed (RUN_LOG_MASK)
 *   run/<TestCase>/device-<phyId>/...
 *
 * Behavior contract:
 * - A moduleId carrying RUN_LOG_MASK is routed to the run tree and always passes CheckLogLevel.
 * - ASCEND_GLOBAL_LOG_LEVEL sets the global level (strictly validated, default 3 = ERROR). When it
 *   is set to a valid value it also wins over dlog_setlevel() so host and device processes of the
 *   whole test tree follow one knob (the runtime stub injects --logLevelInPid=3 by default).
 * - Otherwise dlog_setlevel() applies the level requested by the product parameter chain.
 * - The gtest case name is propagated through HCCP_VM_LOG_CASE (set by SpawnHostProcess and
 *   inherited by the exec'ed hccp_service.bin), sanitized before being used as a path segment.
 * - The runner main process keeps logging to stdout: the verdict scanner depends on stdout and the
 *   log purge in main() would orphan files opened before it runs.
 * - HCCP_ST_RUN_ID=N>0 moves the whole tree to <root>/ascend/log/run_<N> so concurrent runners
 *   keep disjoint log trees (run_<N> is decimal without leading zeros, format-identical to
 *   VerdictGetLogDir in main.cc; parsing rules live in hccp_shm_infra.h). N=0 keeps the original
 *   path byte-for-byte.
 */

#include <dlog_pub.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>
#include <stdarg.h>
#include <linux/limits.h>
#include <libgen.h>
#include <errno.h>
#include <fcntl.h>

#include "hccp_shm_infra.h"

/* Test-private module id for URMA stub logs; the product module enum in log_types.h is full. */
#define STUB_URMA_MODULE_ID 100
/* Upper bound of the sanitized case-name path segment. */
#define STUB_CASE_SEG_MAX 128

static FILE* g_log_file_debug = NULL; /* debug-tree handle, level controlled */
static FILE* g_log_file_run = NULL;   /* run-tree handle, always printed */
static int g_global_log_level = 3;    /* default level: ERROR */
static int g_env_level_fixed = 0;     /* ASCEND_GLOBAL_LOG_LEVEL wins over dlog_setlevel */
static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
static pid_t g_ctor_pid = -1;

static int mkdir_p(const char* path, mode_t mode)
{
    char tmp[PATH_MAX];
    char* p = NULL;
    struct stat st;

    if (snprintf(tmp, sizeof(tmp), "%s", path) >= (int)sizeof(tmp)) {
        errno = ENAMETOOLONG;
        return -1;
    }

    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (stat(tmp, &st) != 0) {
                if (mkdir(tmp, mode) != 0 && errno != EEXIST) {
                    return -1;
                }
            } else if (!S_ISDIR(st.st_mode)) {
                errno = ENOTDIR;
                return -1;
            }
            *p = '/';
        }
    }

    if (stat(tmp, &st) != 0) {
        if (mkdir(tmp, mode) != 0 && errno != EEXIST) {
            return -1;
        }
    } else if (!S_ISDIR(st.st_mode)) {
        errno = ENOTDIR;
        return -1;
    }

    return 0;
}

/* The tree location decision lives once in hccp_shm_infra.h (hccp_st_log_root),
 * shared with the runner's verdict scan: the written root and the scanned root
 * cannot drift apart. OpenLogFileAt mkdir_p's the full path, so the slot
 * subdirectory needs no explicit creation here. */
static int GetLogRoot(char* root, size_t sz) { return hccp_st_log_root(root, sz, hccp_st_run_id()); }

static int GetDevicePhyId(unsigned int* phyId)
{
    char exe_path[PATH_MAX] = {0};
    char cmdline[2048] = {0};
    const char* p = NULL;
    const char* end = NULL;
    const char* arg = NULL;
    ssize_t count;
    ssize_t n;
    int fd;

    count = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (count <= 0) {
        return -1;
    }
    exe_path[count] = '\0';
    if (strstr(exe_path, "hccp_service.bin") == NULL) {
        return -1;
    }

    fd = open("/proc/self/cmdline", O_RDONLY);
    if (fd < 0) {
        return -1;
    }
    n = read(fd, cmdline, sizeof(cmdline) - 1);
    (void)close(fd);
    if (n <= 0) {
        return -1;
    }
    p = cmdline;
    end = cmdline + n;
    while (p < end) {
        size_t len;
        arg = p;
        len = strnlen(p, (size_t)(end - p));
        p += len + 1;
        if (strncmp(arg, "--deviceId=", 11) == 0) {
            *phyId = (unsigned int)strtoul(arg + 11, NULL, 10);
            return 0;
        }
    }
    return -1;
}

/* A moduleId carrying RUN_LOG_MASK is always printed and lands in the run tree. */
static int IsRunLog(int32_t moduleId) { return (((uint32_t)moduleId) & RUN_LOG_MASK) != 0; }

/* Strictly parse ASCEND_GLOBAL_LOG_LEVEL: only a plain decimal 0..4 is accepted. Garbage does not
 * silently become 0 (which would flood the debug tree); it falls back to the default level. */
static void ParseGlobalLogLevel(void)
{
    const char* env = getenv("ASCEND_GLOBAL_LOG_LEVEL");
    char* end = NULL;
    long val;

    if (env == NULL || env[0] == '\0') {
        return;
    }
    errno = 0;
    val = strtol(env, &end, 10);
    if (end == env || *end != '\0' || errno != 0 || val < 0 || val > DLOG_NULL) {
        return;
    }
    g_global_log_level = (int)val;
    g_env_level_fixed = 1;
}

/* Sanitize the case-name segment taken from HCCP_VM_LOG_CASE: keep [A-Za-z0-9._-] and replace
 * anything else with '_' so the value can never escape the log root. */
static void GetCaseSegment(char* out, size_t sz)
{
    const char* envCase = getenv("HCCP_VM_LOG_CASE");
    size_t i = 0;

    out[0] = '\0';
    if (envCase == NULL || sz == 0) {
        return;
    }
    for (i = 0; envCase[i] != '\0' && i + 1 < sz; i++) {
        char c = envCase[i];
        int ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || (c == '.') || (c == '_')
                 || (c == '-');
        out[i] = ok ? c : '_';
    }
    out[i] = '\0';
    if (strcmp(out, ".") == 0 || strcmp(out, "..") == 0) {
        out[0] = '\0';
    }
}

/* Create the parent directory of path (best effort) and open the file for append. On failure fall
 * back to stdout so the log line is still visible somewhere. */
static void OpenLogFileAt(const char* path, FILE** handle)
{
    char dir[PATH_MAX];
    char* slash = NULL;
    FILE* file = NULL;

    (void)snprintf(dir, sizeof(dir), "%s", path);
    slash = strrchr(dir, '/');
    if (slash != NULL) {
        *slash = '\0';
        (void)mkdir_p(dir, 0755);
    }
    file = fopen(path, "a");
    if (file == NULL) {
        perror("dlog stub: fopen failed, falling back to stdout");
        *handle = stdout;
        return;
    }
    *handle = file;
}

/* Build <root>/<debug|run>[/<case>]/plog/host-*.log for host child processes. The pid in the file
 * name keeps processes spawned for the same phy within one case separated. */
static int BuildHostLogPath(char* path, size_t sz, const char* root, const char* caseSeg, int isRun)
{
    char fname[64];
    char phy[16];
    const char* phyEnv = getenv("HCCP_VM_PHYID");
    size_t i = 0;
    int digitsOnly = 1;

    for (i = 0; phyEnv != NULL && phyEnv[i] != '\0' && i + 1 < sizeof(phy); i++) {
        if (phyEnv[i] < '0' || phyEnv[i] > '9') {
            digitsOnly = 0;
        }
        phy[i] = phyEnv[i];
    }
    phy[i] = '\0';

    if (digitsOnly && phy[0] != '\0') {
        (void)snprintf(fname, sizeof(fname), "host-phy%s-pid%d.log", phy, (int)getpid());
    } else {
        (void)snprintf(fname, sizeof(fname), "host-pid%d.log", (int)getpid());
    }

    if (caseSeg[0] != '\0') {
        return snprintf(path, sz, "%s/%s/%s/plog/%s", root, isRun ? "run" : "debug", caseSeg, fname) < (int)sz ? 0 : -1;
    }
    return snprintf(path, sz, "%s/%s/plog/%s", root, isRun ? "run" : "debug", fname) < (int)sz ? 0 : -1;
}

/* Build <root>/<debug|run>[/<case>]/device-<phyId>/hccp_service-<pid>.log for device processes. */
static int
BuildDeviceLogPath(char* path, size_t sz, const char* root, const char* caseSeg, int isRun, unsigned int phyId)
{
    if (caseSeg[0] != '\0') {
        return snprintf(
                   path, sz, "%s/%s/%s/device-%u/hccp_service-%d.log", root, isRun ? "run" : "debug", caseSeg, phyId,
                   (int)getpid())
                       < (int)sz ?
                   0 :
                   -1;
    }
    return snprintf(
               path, sz, "%s/%s/device-%u/hccp_service-%d.log", root, isRun ? "run" : "debug", phyId, (int)getpid())
                   < (int)sz ?
               0 :
               -1;
}

static void OpenHostLogPairLocked(void)
{
    char root[PATH_MAX] = {0};
    char caseSeg[STUB_CASE_SEG_MAX] = {0};
    char path[PATH_MAX] = {0};

    if (GetLogRoot(root, sizeof(root)) != 0) {
        return;
    }
    GetCaseSegment(caseSeg, sizeof(caseSeg));
    if (BuildHostLogPath(path, sizeof(path), root, caseSeg, 0) == 0) {
        OpenLogFileAt(path, &g_log_file_debug);
    }
    if (BuildHostLogPath(path, sizeof(path), root, caseSeg, 1) == 0) {
        OpenLogFileAt(path, &g_log_file_run);
    }
}

static void OpenDeviceLogPairLocked(unsigned int phyId)
{
    char root[PATH_MAX] = {0};
    char caseSeg[STUB_CASE_SEG_MAX] = {0};
    char path[PATH_MAX] = {0};

    if (GetLogRoot(root, sizeof(root)) != 0) {
        return;
    }
    GetCaseSegment(caseSeg, sizeof(caseSeg));
    if (BuildDeviceLogPath(path, sizeof(path), root, caseSeg, 0, phyId) == 0) {
        OpenLogFileAt(path, &g_log_file_debug);
    }
    if (BuildDeviceLogPath(path, sizeof(path), root, caseSeg, 1, phyId) == 0) {
        OpenLogFileAt(path, &g_log_file_run);
    }
}

/* Switch a forked host child from inherited stdout to its per-case files. The runner main process
 * (ctor pid) keeps stdout: the verdict scanner depends on it and the log purge in main() runs after
 * this library's constructor, which would orphan files opened too early. */
static void LazyInitHostChildLocked(void)
{
    if (getpid() == g_ctor_pid) {
        return;
    }
    if (g_log_file_debug != stdout || g_log_file_run != stdout) {
        return;
    }
    OpenHostLogPairLocked();
}

static void __attribute__((constructor)) dlog_stub_init(void)
{
    unsigned int phyId = 0;

    ParseGlobalLogLevel();
    g_ctor_pid = getpid();

    if (GetDevicePhyId(&phyId) == 0) {
        /* Device service processes open both trees eagerly so the expected layout is always
         * visible, even when the current level filters every debug line out. */
        OpenDeviceLogPairLocked(phyId);
        return;
    }

    g_log_file_debug = stdout;
    g_log_file_run = stdout;
}

static void __attribute__((destructor)) dlog_stub_fini(void)
{
    if (g_log_file_debug != NULL && g_log_file_debug != stdout) {
        (void)fflush(g_log_file_debug);
        (void)fclose(g_log_file_debug);
        g_log_file_debug = NULL;
    }
    if (g_log_file_run != NULL && g_log_file_run != stdout) {
        (void)fflush(g_log_file_run);
        (void)fclose(g_log_file_run);
        g_log_file_run = NULL;
    }
}

/* Called by the test framework right after forking a host child: makes sure the plog files of the
 * current case exist even when the log level filters out every host log line. */
void dlog_stub_prepare_host_log(void)
{
    pthread_mutex_lock(&log_mutex);
    LazyInitHostChildLocked();
    pthread_mutex_unlock(&log_mutex);
}

int32_t dlog_setlevel(int32_t moduleId, int32_t level, int32_t enableEvent)
{
    (void)moduleId;
    (void)enableEvent;
    if (g_env_level_fixed) {
        /* An explicit ASCEND_GLOBAL_LOG_LEVEL pins the level so the whole test tree follows
         * one knob instead of the --logLevelInPid=3 injected by the runtime stub. */
        return 0;
    }
    if (level < DLOG_DEBUG) {
        level = DLOG_DEBUG;
    }
    if (level > DLOG_NULL) {
        level = DLOG_NULL;
    }
    g_global_log_level = level;
    return 0;
}

int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel)
{
    if (IsRunLog(moduleId)) {
        return 1; /* run-tree logs are printed regardless of the global level */
    }
    return logLevel >= g_global_log_level ? 1 : 0;
}

int32_t DlogSetAttr(LogAttr logAttrInfo)
{
    (void)logAttrInfo;
    return 0;
}

static const char* GetLogLevelStr(int32_t level)
{
    switch (level) {
        case 3:
            return "ERROR";
        case 2:
            return "WARNING";
        case 1:
            return "INFO";
        case 0:
            return "DEBUG";
        default:
            return "UNKNOWN";
    }
}

/* Render the module part of the line header. URMA has no product module id (the enum in
 * log_types.h is full), so the stubs use the test-private id STUB_URMA_MODULE_ID. */
static const char* GetModuleStr(int32_t moduleId, char* buf, size_t sz)
{
    int32_t mod = (int32_t)((uint32_t)moduleId
                            & ~(uint32_t)(RUN_LOG_MASK | DEBUG_LOG_MASK | SECURITY_LOG_MASK | STDOUT_LOG_MASK));

    switch (mod) {
        case DRV:
            return "DRV";
        case NET:
            return "NET";
        case HCCP:
            return "HCCP";
        case ROCE:
            return "ROCE";
        case STUB_URMA_MODULE_ID:
            return "URMA";
        default:
            (void)snprintf(buf, sz, "MOD%d", (int)mod);
            return buf;
    }
}

void DlogVaList(int32_t moduleId, int32_t level, const char* fmt, va_list list)
{
    struct timespec ts;
    struct tm tm_buf;
    char timestamp[64];
    char modBuf[16];
    size_t len;
    FILE* log_file = NULL;

    clock_gettime(CLOCK_REALTIME, &ts);
    localtime_r(&ts.tv_sec, &tm_buf);
    len = strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &tm_buf);
    (void)snprintf(timestamp + len, sizeof(timestamp) - len, ".%03ld", ts.tv_nsec / 1000000);

    pthread_mutex_lock(&log_mutex);

    LazyInitHostChildLocked();

    log_file = IsRunLog(moduleId) ? g_log_file_run : g_log_file_debug;
    if (log_file == NULL) {
        pthread_mutex_unlock(&log_mutex);
        return;
    }

    fprintf(
        log_file, "[%s] [%s] [%s] ", GetLogLevelStr(level), timestamp, GetModuleStr(moduleId, modBuf, sizeof(modBuf)));
    vfprintf(log_file, fmt, list);
    fprintf(log_file, "\n");
    fflush(log_file);

    pthread_mutex_unlock(&log_mutex);
}

void DlogRecord(int32_t moduleId, int32_t level, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    DlogVaList(moduleId, level, fmt, args);
    va_end(args);
}
