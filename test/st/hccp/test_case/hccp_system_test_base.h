/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCP_SYSTEM_TEST_BASE_H
#define HCCP_SYSTEM_TEST_BASE_H
#include "gtest/gtest.h"
#include <stdio.h>
#include <netinet/in.h>
#include <unistd.h>
#include <linux/limits.h>
#include <libgen.h>
#include <vector>
#include <map>
#include <functional>
#include <iostream>
#include <signal.h>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "hccp_sys_test_common.h"
#include "ascend_hal.h"
#include "ascend_hal_stub.h"
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif
/* Provided by the test dlog stub (libunified_dlog_stub.so): creates the per-case plog files of
 * the current host child, so the expected layout exists even when the log level filters every
 * host log line out. */
void dlog_stub_prepare_host_log(void);
#ifdef __cplusplus
}
#endif

static inline int HccpVmNormalizeExitCode(int logicRet) { return (logicRet == 0) ? 0 : 1; }

class HccpSystemTestBase : public ::testing::Test {
protected:
    using HostTestLogic = std::function<int(int phyid)>;

    std::map<int, pid_t> host_pids_;
    std::map<int, HostTestLogic> host_logic_map_;
    bool overall_test_success_ = true;

    void PrintExitStatus(int status)
    {
        if (WIFEXITED(status)) {
            printf("code=%d", WEXITSTATUS(status));
        } else if (WIFSIGNALED(status)) {
            int sig = WTERMSIG(status);
            printf("signal=%d", sig);
            if (sig == SIGSEGV)
                printf("\033[31m (SIGSEGV)\033[0m");
            else if (sig == SIGUSR1)
                printf("\033[1;33m (SIGUSR1)(killed by host)\033[0m");
            else if (sig == SIGABRT) {
                printf("\033[31m (SIGABRT sig=%d)\033[0m\n", sig);
            }
        }
        printf("\n");
    }

    void InitHalStubForHost(int phyId)
    {
        setHostPid(getpid(), HOST_ID_0, phyId);
        setDevPid(0, HOST_ID_0, phyId);
        InitHdcId();
        h2d_info_t(*info)[MAX_DEV_ID] = GetH2dInfo();
        if (info != nullptr) {
            info[HOST_ID_0][phyId].deviceReady = 0;
        }
    }

    bool SpawnHostProcess(int phy_id, HostTestLogic host_logic)
    {
        host_logic_map_[phy_id] = std::move(host_logic);

        /* Captured pre-fork: the PDEATHSIG race check below compares against
         * the runner we were forked from, not whatever getppid() returns after
         * a possible parent death in the window between fork and prctl. */
        pid_t parentPid = getpid();
        pid_t pid = fork();
        if (pid == -1) {
            printf(COLOR_MAIN "[MAIN]" COLOR_RESET " Fork host failed for phy_id=%d\n", phy_id);
            return false;
        }

        if (pid == 0) {
            /* Die with the runner: a SIGKILLed runner must not leave host
             * children (and their forked hccp_service.bin devices) probing a
             * dead world. PR_SET_PDEATHSIG only fires for parent death AFTER
             * the prctl, so close the fork-vs-prctl window explicitly: if the
             * parent already died, getppid() no longer matches parentPid. */
            if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parentPid) {
                _exit(1);
            }
            setvbuf(stdout, NULL, _IONBF, 0);
            char phyEnv[16] = {0};
            (void)snprintf(phyEnv, sizeof(phyEnv), "%d", phy_id);
            (void)setenv("HCCP_VM_PHYID", phyEnv, 1);
            /* Route this child's dlog output into ascend/log/<debug|run>/<case>/. The case
             * name is inherited by the hccp_service.bin children forked+exec'ed through
             * rtOpenNetService, so service and stub logs of one case share one directory. */
            const ::testing::TestInfo* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
            if (testInfo != nullptr) {
                char logCase[256] = {0};
                (void)snprintf(logCase, sizeof(logCase), "%s.%s", testInfo->test_suite_name(), testInfo->name());
                (void)setenv("HCCP_VM_LOG_CASE", logCase, 1);
            }
            dlog_stub_prepare_host_log();
            TEST_LOG_INFO(phy_id, "Start (pid=%d)", getpid());

            InitHalStubForHost(phy_id);

            int exit_code = EXIT_SUCCESS;
            try {
                int logic_ret = host_logic_map_[phy_id](phy_id);
                TEST_LOG_INFO(phy_id, "test finished, ret=%d", logic_ret);
                exit_code = HccpVmNormalizeExitCode(logic_ret);
            } catch (const std::exception& e) {
                TEST_LOG_ERROR(phy_id, "Exception: %s", e.what());
                exit_code = EXIT_FAILURE;
            } catch (...) {
                TEST_LOG_ERROR(phy_id, "Unknown exception");
                exit_code = EXIT_FAILURE;
            }
            _exit(exit_code);
        } else {
            host_pids_[phy_id] = pid;
            printf("[MAIN] Spawned host (phy_id=%d, pid=%d)\n", phy_id, pid);
            return true;
        }
    }

protected:
    void SetUp() override
    {
        overall_test_success_ = true;
        /* Per-case reset of the LITE QP rendezvous: the segment lives for
         * the whole run (memfd world) and every case's host pair must pair
         * from scratch. The old SysV world reset implicitly -- the previous
         * case's creator was dead by the next attach, so the kill-probe
         * recycled the segment. Case boundaries are the clean equivalent:
         * this runs in the runner, and the previous case's host children
         * have been reaped, so the reset is quiescent. */
        hccp_lite_rendezvous_reset();
    }

    void TearDown() override
    {
        int timeout_seconds = 5;
        if (const char* env = getenv("HCCP_VM_HOST_TIMEOUT_SEC")) {
            long v = atol(env);
            if (v > 0)
                timeout_seconds = (int)v;
        }
        const useconds_t poll_interval_us = 5000;

        for (auto& pair : host_pids_) {
            int phy_id = pair.first;
            pid_t pid = pair.second;
            if (pid == -1)
                continue;

            int status = 0;
            bool exited = false;
            bool timed_out = false;
            auto start_time = std::chrono::steady_clock::now();
            printf(COLOR_CLEANUP "[CLEANUP]" COLOR_RESET " Waiting for host (phy_id=%d, pid=%d)\n", phy_id, pid);

            while (true) {
                pid_t result = waitpid(pid, &status, WNOHANG);
                if (result > 0) {
                    exited = true;
                    break;
                } else if (result == 0) {
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                    if (elapsed >= timeout_seconds) {
                        timed_out = true;
                        break;
                    }
                    usleep(poll_interval_us);
                } else {
                    printf(COLOR_CLEANUP "[CLEANUP]" COLOR_RESET " waitpid(WNOHANG) error\n");
                    break;
                }
            }

            if (timed_out) {
                printf(COLOR_CLEANUP "[CLEANUP]" COLOR_RESET " Host-%d timed out. Sending SIGUSR1...\n", phy_id);
                kill(pid, SIGUSR1);
                (void)waitpid(pid, &status, 0);
                printf(COLOR_CLEANUP "[CLEANUP]" COLOR_RESET " Host (phy_id=%d) killed by TEST: ", phy_id);
                PrintExitStatus(status);
                overall_test_success_ = false;
            } else if (exited) {
                printf(COLOR_CLEANUP "[CLEANUP]" COLOR_RESET " Host-%d finished: ", phy_id);
                PrintExitStatus(status);
                if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                    printf(
                        COLOR_CLEANUP "[CLEANUP]" COLOR_RESET COLOR_ERROR " Error:" COLOR_RESET
                                      " Host-%d exited with non-success status\n",
                        phy_id);
                    overall_test_success_ = false;
                }
            }
        }

        host_pids_.clear();
        host_logic_map_.clear();
        EXPECT_TRUE(overall_test_success_);
    }
    static void SetUpTestCase() { printf("\033[36m--HCCP SYS Test SetUP--\033[0m\n"); }
    static void TearDownTestCase() { printf("\033[36m--HCCP SYS Test TearDown--\033[0m\n"); }

public:
    void RunTests(const std::vector<int>& phy_ids, HostTestLogic logic)
    {
        for (int phy_id : phy_ids) {
            bool ok = SpawnHostProcess(phy_id, logic);
            ASSERT_TRUE(ok) << "Spawn host failed for phy_id=" << phy_id;
        }
    }
};

#endif
