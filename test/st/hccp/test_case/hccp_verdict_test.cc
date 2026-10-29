/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "hccp_verdict.h"
#include "hccp_shm_infra.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include "gtest/gtest.h"

namespace {
enum class Fault {
    NONE,
    PIPE,
    DUP_OUT,
    DUP_ERR,
    REDIRECT_OUT,
    REDIRECT_ERR,
    REDIRECT_EINTR,
    RESTORE_OUT,
    RESTORE_ERR,
    RESTORE_OUT_EINTR,
    RESTORE_ERR_EINTR,
    THREAD,
    JOIN,
    TEE_READ,
    TEE_EINTR,
    WRITE_EINTR,
    WRITE_SHORT,
    WRITE_FAILURE,
    OPEN_DIR,
    READ_DIR,
    STAT,
    OPEN_FILE,
    READ_FILE,
    PARTIAL_READ,
    CLOSE_FILE,
    CLOSE_DIR
};

Fault g_fault = Fault::NONE;
std::atomic<bool> g_faultFired{false};
std::atomic<unsigned int> g_faultHits{0};
int g_dupCalls = 0;
int g_captureOut = -1;
int g_captureErr = -1;
std::string g_forwarded;
std::string g_logRoot;
DIR* g_scanDir = nullptr;
FILE* g_scanFile = nullptr;
size_t g_partialOffset = 0;

bool ProbeFile(const char* path)
{
    const char* name = strrchr(path, '/');
    return name != nullptr && strcmp(name, "/probe.log") == 0;
}

void Arm(Fault fault)
{
    g_fault = fault;
    g_faultFired = false;
    g_faultHits = 0;
    g_dupCalls = 0;
    g_captureOut = g_captureErr = -1;
    g_forwarded.clear();
    g_scanDir = nullptr;
    g_scanFile = nullptr;
    g_partialOffset = 0;
}

class ConsoleSilencer {
public:
    ConsoleSilencer(const ConsoleSilencer&) = delete;
    ConsoleSilencer& operator=(const ConsoleSilencer&) = delete;
    ConsoleSilencer()
    {
        (void)fflush(stdout);
        (void)fflush(stderr);
        savedOut_ = dup(STDOUT_FILENO);
        savedErr_ = dup(STDERR_FILENO);
        int sink = open("/dev/null", O_WRONLY);
        if (savedOut_ < 0 || savedErr_ < 0 || sink < 0 || dup2(sink, STDOUT_FILENO) < 0
            || dup2(sink, STDERR_FILENO) < 0) {
            std::abort();
        }
        (void)close(sink);
    }

    ~ConsoleSilencer()
    {
        Arm(Fault::NONE);
        (void)dup2(savedOut_, STDOUT_FILENO);
        (void)dup2(savedErr_, STDERR_FILENO);
        (void)close(savedOut_);
        (void)close(savedErr_);
    }

private:
    int savedOut_;
    int savedErr_;
};

int OpenFdCount()
{
    DIR* dir = opendir("/proc/self/fd");
    if (dir == nullptr) {
        return -1;
    }
    int count = 0;
    while (readdir(dir) != nullptr) {
        count++;
    }
    (void)closedir(dir);
    return count;
}

HccpSt::VerdictResults PassingResults()
{
    HccpSt::VerdictResults results;
    results.gtest.selected = results.gtest.started = results.gtest.finished = 1;
    results.caseLoad.loaded = true;
    results.caseLoad.path = "libhccp_case.so";
    return results;
}

std::string Report(const HccpSt::VerdictResults& results)
{
    FILE* file = tmpfile();
    if (file == nullptr) {
        std::abort();
    }
    HccpSt::PrintVerdict(results, HccpSt::EvaluateVerdict(results), file);
    (void)rewind(file);
    std::string output;
    char buffer[512];
    size_t size;
    while ((size = fread(buffer, 1, sizeof(buffer), file)) > 0) {
        output.append(buffer, size);
    }
    (void)fclose(file);
    return output;
}

class LogScanTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        Arm(Fault::NONE);
        char directory[] = "/tmp/hccp-verdict-XXXXXX";
        ASSERT_NE(mkdtemp(directory), nullptr);
        g_logRoot = directory;
    }

    void TearDown() override
    {
        Arm(Fault::NONE);
        // Every file created by the tests is known; avoid shell-based cleanup.
        for (const char* file :
             {"/probe.log", "/other.log", "/nested/probe.log", "/run_1/probe.log", "/run_/probe.log",
              "/run_x/probe.log", "/pipe.log", "/link.log", "/ignored.txt", "/root-link", "/file-link"}) {
            (void)unlink((g_logRoot + file).c_str());
        }
        (void)rmdir((g_logRoot + "/nested").c_str());
        (void)rmdir((g_logRoot + "/run_1").c_str());
        (void)rmdir((g_logRoot + "/run_").c_str());
        (void)rmdir((g_logRoot + "/run_x").c_str());
        (void)rmdir(g_logRoot.c_str());
    }

    void Write(const char* name, int count, const std::string& payload = "failure")
    {
        std::ofstream file(g_logRoot + name);
        ASSERT_TRUE(file.good());
        for (int index = 0; index < count; ++index) {
            file << "[ERROR] " << payload << '\n';
        }
        file.close();
        ASSERT_FALSE(file.fail());
    }
};

class ScanFaultTest : public LogScanTest, public ::testing::WithParamInterface<Fault> {};
class CaptureFaultTest : public ::testing::TestWithParam<Fault> {};
class CaptureRestoreTest : public ::testing::TestWithParam<Fault> {};
class ForwardWriteTest : public ::testing::TestWithParam<Fault> {};

TEST(VerdictTest, EveryComponentMustPass)
{
    for (int component = 0; component < 4; ++component) {
        auto results = PassingResults();
        ASSERT_TRUE(HccpSt::EvaluateVerdict(results).Passed());
        if (component == 0) {
            results.gtest.returnCode = 1;
        }
        if (component == 1) {
            results.stdoutErrors.errors.Record("error");
        }
        if (component == 2) {
            results.logErrors.errors.Record("[ERROR]");
        }
        if (component == 3) {
            results.caseLoad.loaded = false;
        }
        EXPECT_FALSE(HccpSt::EvaluateVerdict(results).Passed());
        EXPECT_NE(Report(results).find("[VERDICT] OVERALL:       FAILED\n"), std::string::npos);
    }
}

TEST(VerdictTest, DefaultDecisionFailsClosed)
{
    HccpSt::VerdictDecision decision;
    EXPECT_FALSE(decision.gtestOk);
    EXPECT_FALSE(decision.stdoutOk);
    EXPECT_FALSE(decision.logOk);
    EXPECT_FALSE(decision.loadOk);
    EXPECT_FALSE(decision.Passed()); // A default-constructed decision never passes.
}

TEST(VerdictTest, SelectedButUnrunAndAllSkippedFail)
{
    auto results = PassingResults();
    results.gtest.started = results.gtest.finished = 0;
    EXPECT_FALSE(HccpSt::EvaluateVerdict(results).Passed());
    EXPECT_NE(Report(results).find("NO TEST RAN"), std::string::npos);
    results.gtest.started = results.gtest.finished = results.gtest.skipped = 1;
    EXPECT_FALSE(HccpSt::EvaluateVerdict(results).Passed());
    EXPECT_NE(Report(results).find("ALL SKIPPED"), std::string::npos);
    results.gtest.selected = results.gtest.started = results.gtest.finished = 2;
    EXPECT_TRUE(HccpSt::EvaluateVerdict(results).Passed());
}

TEST(VerdictTest, FailedTestsAreCountedInTheReport)
{
    auto results = PassingResults();
    results.gtest.selected = 5;
    results.gtest.started = results.gtest.finished = 4;
    results.gtest.failed = 2;
    results.gtest.returnCode = 1;
    auto decision = HccpSt::EvaluateVerdict(results);
    EXPECT_FALSE(decision.gtestOk);
    EXPECT_TRUE(decision.stdoutOk);
    EXPECT_TRUE(decision.logOk);
    EXPECT_TRUE(decision.loadOk);
    EXPECT_FALSE(decision.Passed());
    EXPECT_EQ(
        Report(results), "[VERDICT] gtest:         FAIL (2 of 4 tests failed)\n"
                         "[VERDICT] stdout-errors: CLEAN (0 lines)\n"
                         "[VERDICT] log-errors:    CLEAN (0 [ERROR] lines)\n"
                         "[VERDICT] case-load:     OK (libhccp_case.so)\n"
                         "[VERDICT] OVERALL:       FAILED\n");
}

TEST(VerdictTest, RunSlotClassificationDependsOnNumericRunId)
{
    for (const char* name : {"run_", "run_x", "run_1x", "run_-1", "run_+1", "run_1.0", "nested"}) {
        SCOPED_TRACE(name);
        EXPECT_FALSE(HccpSt::IsForeignRunDir(name, 0));
        EXPECT_FALSE(HccpSt::IsForeignRunDir(name, 1));
    }
    /* Shared tree (runId 0): every slot-shaped name is a concurrent runner's. */
    for (const char* name : {"run_0", "run_1", "run_001", "run_999999999999999999999999"}) {
        SCOPED_TRACE(name);
        EXPECT_TRUE(HccpSt::IsForeignRunDir(name, 0));
    }
    /* Slotted scan (runId 1): only its own number -- even spelled with leading
     * zeros -- is non-foreign; other and unrepresentable numbers stay foreign. */
    EXPECT_FALSE(HccpSt::IsForeignRunDir("run_1", 1));
    EXPECT_FALSE(HccpSt::IsForeignRunDir("run_001", 1));
    EXPECT_TRUE(HccpSt::IsForeignRunDir("run_0", 1));
    EXPECT_TRUE(HccpSt::IsForeignRunDir("run_2", 1));
    EXPECT_TRUE(HccpSt::IsForeignRunDir("run_999999999999999999999999", 1));
    EXPECT_FALSE(HccpSt::IsForeignRunDir("run_2", 2));
}

TEST(VerdictTest, ScanAndWriteRootsShareOneBuilder)
{
    /* The scanner root (main.cc) and the writer root (dlog_stub.c) resolve through
     * the same hccp_st_log_root: this pins the one format both sides inherit. */
    char root[PATH_MAX] = {0};
    ASSERT_EQ(hccp_st_log_root(root, sizeof(root), 0), 0);
    const char* plain = "/ascend/log";
    EXPECT_EQ(std::string(root).compare(std::string(root).size() - strlen(plain), strlen(plain), plain), 0);
    ASSERT_EQ(hccp_st_log_root(root, sizeof(root), 2), 0);
    const char* slotted = "/ascend/log/run_2";
    EXPECT_EQ(std::string(root).compare(std::string(root).size() - strlen(slotted), strlen(slotted), slotted), 0);
    /* Not a constant size: keeps -Wformat-truncation from firing on the (here
     * intentional) truncating call while the runtime contract stays checked. */
    char tiny[8];
    volatile size_t tinySz = sizeof(tiny);
    EXPECT_EQ(hccp_st_log_root(tiny, tinySz, 1), -1);
}

TEST(VerdictTest, UnknownHidesPartialCountsAndDetails)
{
    auto results = PassingResults();
    for (int index = 0; index < 25; ++index) {
        results.logErrors.errors.Record("private partial detail");
        results.stdoutErrors.errors.Record("private partial detail");
    }
    results.logErrors.MarkUnknown("scan failed");
    results.stdoutErrors.MarkUnknown("capture failed");
    auto report = Report(results);
    EXPECT_FALSE(HccpSt::EvaluateVerdict(results).Passed());
    EXPECT_NE(report.find("UNKNOWN (scan failed)"), std::string::npos);
    EXPECT_NE(report.find("UNKNOWN (capture failed)"), std::string::npos);
    EXPECT_EQ(report.find("25 lines"), std::string::npos);
    EXPECT_EQ(report.find("private partial detail"), std::string::npos);
}

TEST(VerdictTest, DetailsAreBoundedWhenCollectedAndCannotForgeLines)
{
    HccpSt::ErrorSummary summary;
    std::string line = "error\n" + std::string(1024 * 1024, 'x');
    for (int index = 0; index < 37; ++index) {
        summary.Record(line);
    }
    ASSERT_EQ(summary.count, 37U);
    ASSERT_EQ(summary.details.size(), HccpSt::VERDICT_MAX_REPORT);
    for (const auto& detail : summary.details) {
        EXPECT_EQ(detail.size(), HccpSt::VERDICT_DETAIL_MAX);
        EXPECT_EQ(detail.find('\n'), std::string::npos);
    }
    auto results = PassingResults();
    results.caseLoad.path = "case\n[VERDICT] OVERALL: PASS";
    auto report = Report(results);
    EXPECT_EQ(report.find("\n[VERDICT] OVERALL: PASS"), std::string::npos);
    EXPECT_EQ(report.substr(report.rfind("[VERDICT]")), "[VERDICT] OVERALL:       PASS\n");
}

TEST_F(LogScanTest, CountsAllLinesAcrossBoundaries)
{
    for (int count : {0, 1, 20, 21, 37}) {
        Write("/probe.log", count);
        auto result = HccpSt::ScanLogFiles(g_logRoot.c_str(), true, 0);
        EXPECT_TRUE(result.complete);
        EXPECT_EQ(result.errors.count, static_cast<size_t>(count));
        EXPECT_EQ(result.errors.details.size(), std::min(static_cast<size_t>(count), HccpSt::VERDICT_MAX_REPORT));
        EXPECT_EQ(result.Passed(), count == 0);
    }
}

TEST_F(LogScanTest, AggregatesFilesAndSubdirectories)
{
    Write("/probe.log", 7);
    Write("/other.log", 10);
    ASSERT_EQ(mkdir((g_logRoot + "/nested").c_str(), 0700), 0);
    Write("/nested/probe.log", 20);
    auto result = HccpSt::ScanLogFiles(g_logRoot.c_str(), true, 0);
    EXPECT_TRUE(result.complete);
    EXPECT_EQ(result.errors.count, 37U);
    EXPECT_EQ(result.errors.details.size(), 20U);
}

TEST_F(LogScanTest, MissingRootIsACleanEmptyResult)
{
    std::string missing = g_logRoot + "/absent";
    EXPECT_TRUE(HccpSt::ScanLogFiles(missing.c_str(), false, 0).Passed());
    /* The tree is created lazily by the first written log line: a filtered run over
     * cases that never log must pass clean, not trip an absent-tree failure. */
    auto result = HccpSt::ScanLogFiles(missing.c_str(), true, 0);
    EXPECT_TRUE(result.complete);
    EXPECT_TRUE(result.Passed());
    EXPECT_FALSE(HccpSt::ScanLogFiles("", false, 0).complete);
    Write("/probe.log", 0);
    EXPECT_FALSE(HccpSt::ScanLogFiles((g_logRoot + "/probe.log").c_str(), true, 0).complete);
}

TEST_F(LogScanTest, SymlinkedRootFollowsTargetTree)
{
    Write("/probe.log", 2);
    const std::string rootLink = g_logRoot + "/root-link";
    const std::string fileLink = g_logRoot + "/file-link";
    ASSERT_EQ(symlink(g_logRoot.c_str(), rootLink.c_str()), 0);
    ASSERT_EQ(symlink("probe.log", fileLink.c_str()), 0);
    auto scanned = HccpSt::ScanLogFiles(rootLink.c_str(), true, 0);
    EXPECT_TRUE(scanned.complete); // The root check follows its symlink: the tree still scans.
    EXPECT_EQ(scanned.errors.count, 2U);
    auto rejected = HccpSt::ScanLogFiles(fileLink.c_str(), true, 0);
    EXPECT_FALSE(rejected.complete); // A root resolving to a file is still not a tree.
}

TEST_F(LogScanTest, ExcludesForeignSlotsAndNonregularFiles)
{
    Write("/probe.log", 1);
    Write("/ignored.txt", 10);
    ASSERT_EQ(mkdir((g_logRoot + "/run_1").c_str(), 0700), 0);
    Write("/run_1/probe.log", 5);
    ASSERT_EQ(mkfifo((g_logRoot + "/pipe.log").c_str(), 0600), 0);
    ASSERT_EQ(symlink("probe.log", (g_logRoot + "/link.log").c_str()), 0);
    EXPECT_EQ(HccpSt::ScanLogFiles(g_logRoot.c_str(), true, 0).errors.count, 1U);
    EXPECT_EQ(HccpSt::ScanLogFiles((g_logRoot + "/run_1").c_str(), true, 1).errors.count, 5U);
}

TEST_F(LogScanTest, MalformedRunDirectoriesAreNotExcluded)
{
    for (const char* name : {"/run_", "/run_x"}) {
        ASSERT_EQ(mkdir((g_logRoot + name).c_str(), 0700), 0);
        Write((std::string(name) + "/probe.log").c_str(), 1);
    }
    auto result = HccpSt::ScanLogFiles(g_logRoot.c_str(), true, 0);
    EXPECT_TRUE(result.complete);
    EXPECT_EQ(result.errors.count, 2U);
    EXPECT_FALSE(result.Passed());
}

TEST_F(LogScanTest, PathTruncationMakesScanUnknown)
{
    Write("/probe.log", 1);
    std::string directory = g_logRoot;
    // Each /. resolves to the same directory; only appending the entry exceeds PATH_MAX.
    while (directory.size() + 2 < PATH_MAX) {
        directory += "/.";
    }
    struct stat info = {};
    ASSERT_EQ(lstat(directory.c_str(), &info), 0);
    ASSERT_GE(directory.size() + strlen("/probe.log"), static_cast<size_t>(PATH_MAX));
    int before = OpenFdCount();
    ASSERT_GE(before, 0);
    ::testing::internal::CaptureStderr();
    auto result = HccpSt::ScanLogFiles(directory.c_str(), true, 0);
    auto diagnostic = ::testing::internal::GetCapturedStderr();
    EXPECT_NE(diagnostic.find("[VERDICT] scan failed: path "), std::string::npos);
    EXPECT_FALSE(result.complete);
    EXPECT_FALSE(result.Passed());
    EXPECT_EQ(result.reason, "scan failed");
    EXPECT_EQ(OpenFdCount(), before);
    auto results = PassingResults();
    results.logErrors = result;
    EXPECT_FALSE(HccpSt::EvaluateVerdict(results).Passed());
    EXPECT_NE(Report(results).find("log-errors:    UNKNOWN (scan failed)"), std::string::npos);
}

TEST_P(ScanFaultTest, IncompleteScanNeverPasses)
{
    Write("/probe.log", 37);
    HccpSt::ErrorCheckResult result;
    unsigned int hits;
    {
        ConsoleSilencer console;
        Arm(GetParam());
        result = HccpSt::ScanLogFiles(g_logRoot.c_str(), true, 0);
        hits = g_faultHits.load();
    }
    EXPECT_GT(hits, 0U);
    EXPECT_FALSE(result.complete);
    EXPECT_FALSE(result.Passed());
    if (GetParam() == Fault::PARTIAL_READ) {
        EXPECT_EQ(result.errors.count, 25U);
    }
    auto results = PassingResults();
    results.logErrors = result;
    EXPECT_FALSE(HccpSt::EvaluateVerdict(results).Passed());
    EXPECT_EQ(Report(results).find("  [log]"), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
    SystemCalls, ScanFaultTest,
    ::testing::Values(
        Fault::OPEN_DIR, Fault::READ_DIR, Fault::STAT, Fault::OPEN_FILE, Fault::READ_FILE, Fault::PARTIAL_READ,
        Fault::CLOSE_FILE, Fault::CLOSE_DIR));

TEST_P(CaptureFaultTest, InitializationRollsBackWithoutLeakingDescriptors)
{
    HccpSt::ErrorCheckResult result;
    int before;
    int after;
    bool streamsRestored;
    unsigned int hits;
    {
        ConsoleSilencer console;
        before = OpenFdCount();
        Arm(GetParam());
        HccpSt::StdoutCapture capture;
        capture.Start();
        result = capture.Finish();
        hits = g_faultHits.load();
        Arm(Fault::NONE);
        streamsRestored = fcntl(STDOUT_FILENO, F_GETFD) >= 0 && fcntl(STDERR_FILENO, F_GETFD) >= 0;
        after = OpenFdCount();
    }
    EXPECT_FALSE(result.complete);
    EXPECT_GT(hits, 0U);
    EXPECT_FALSE(result.Passed());
    EXPECT_TRUE(streamsRestored);
    EXPECT_EQ(after, before);
    auto results = PassingResults();
    results.stdoutErrors = result;
    EXPECT_FALSE(HccpSt::EvaluateVerdict(results).Passed());
    EXPECT_NE(Report(results).find("stdout-errors: UNKNOWN (capture failed)"), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
    SystemCalls, CaptureFaultTest,
    ::testing::Values(
        Fault::PIPE, Fault::DUP_OUT, Fault::DUP_ERR, Fault::REDIRECT_OUT, Fault::REDIRECT_ERR, Fault::THREAD));

TEST(CaptureDeathTest, JoinFailureAbortsInsteadOfReturningThreadOwnedResult)
{
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_EXIT(
        {
            struct rlimit coreLimit = {};
            if (setrlimit(RLIMIT_CORE, &coreLimit) != 0) {
                _exit(2);
            }
            Arm(Fault::JOIN);
            HccpSt::StdoutCapture capture;
            capture.Start();
            (void)capture.Finish();
            _exit(0); // A returned result must fail this test, without a second Finish in the destructor.
        },
        ::testing::KilledBySignal(SIGABRT), "\\[selftest\\] pthread_join failure injected");
}

TEST(CaptureTest, BothStreamsAreScannedIncludingPartialAndLongLines)
{
    std::string line(4093, 'x');
    line += "ErRoR\n";
    HccpSt::ErrorCheckResult result;
    ssize_t written;
    {
        ConsoleSilencer console;
        HccpSt::StdoutCapture capture;
        capture.Start();
        written = write(STDOUT_FILENO, line.data(), line.size());
        fprintf(stderr, "stderr error\ntrailing ERROR");
        result = capture.Finish();
    }
    ASSERT_EQ(written, static_cast<ssize_t>(line.size()));
    EXPECT_TRUE(result.complete);
    ASSERT_EQ(result.errors.count, 3U);
    ASSERT_EQ(result.errors.details.size(), 3U);
    EXPECT_EQ(result.errors.details.front().size(), HccpSt::VERDICT_DETAIL_MAX);
}

TEST(CaptureTest, InterruptedRedirectionAndReadAreRetried)
{
    for (Fault fault : {Fault::REDIRECT_EINTR, Fault::TEE_EINTR}) {
        HccpSt::ErrorCheckResult result;
        unsigned int hits;
        {
            ConsoleSilencer console;
            Arm(fault);
            HccpSt::StdoutCapture capture;
            capture.Start();
            fprintf(stderr, "error remains visible\n");
            result = capture.Finish();
            hits = g_faultHits.load();
        }
        EXPECT_GT(hits, 0U);
        EXPECT_TRUE(result.complete);
        EXPECT_EQ(result.errors.count, 1U);
    }
}

TEST(CaptureTest, CountsMatchesBeyondTheDetailCap)
{
    HccpSt::ErrorCheckResult result;
    {
        ConsoleSilencer console;
        HccpSt::StdoutCapture capture;
        capture.Start();
        for (int index = 0; index < 37; ++index) {
            fprintf(stderr, "error %d\n", index);
        }
        result = capture.Finish();
    }
    EXPECT_TRUE(result.complete);
    EXPECT_EQ(result.errors.count, 37U);
    EXPECT_EQ(result.errors.details.size(), HccpSt::VERDICT_MAX_REPORT);
}

/* Bounded streaming keeps the matched-lines count exact for arbitrarily long lines:
 * one 1.5 MiB line holding several needles still counts once, a clean 1.5 MiB line
 * counts zero, and the detail stays the line's bounded head. */
TEST(CaptureTest, StreamingCountsLogicalLinesNotSegments)
{
    const std::string head = "HEADMARK error early";
    std::string line1 = head;
    line1 += std::string(512 * 1024, 'x');
    line1 += "error late\n";
    std::string line2(1536 * 1024, 'y');
    line2 += "\n";
    const char* line3 = "short error tail\n";
    HccpSt::ErrorCheckResult result;
    {
        ConsoleSilencer console;
        HccpSt::StdoutCapture capture;
        capture.Start();
        (void)write(STDOUT_FILENO, line1.data(), line1.size());
        (void)write(STDOUT_FILENO, line2.data(), line2.size());
        (void)write(STDOUT_FILENO, line3, strlen(line3));
        result = capture.Finish();
    }
    EXPECT_TRUE(result.complete);
    EXPECT_EQ(result.errors.count, 2U); /* line1 once, line3 once; clean line2 zero */
    const std::string expected
        = (head + std::string(HccpSt::VERDICT_DETAIL_MAX, 'x')).substr(0, HccpSt::VERDICT_DETAIL_MAX);
    EXPECT_EQ(result.errors.details.front(), expected);
}

/* A needle split across read chunks must still match (seam), and a newline between the
 * halves must prevent a match; writes are separated by a pause so the tee thread has
 * consumed each write before the next one arrives. */
TEST(CaptureTest, StreamingMatchesAcrossReadChunksButNotAcrossLines)
{
    HccpSt::ErrorCheckResult split;
    {
        ConsoleSilencer console;
        HccpSt::StdoutCapture capture;
        capture.Start();
        (void)write(STDOUT_FILENO, "prefix er", 9);
        usleep(200 * 1000);
        (void)write(STDOUT_FILENO, "ror suffix\n", 11);
        split = capture.Finish();
    }
    EXPECT_TRUE(split.complete);
    EXPECT_EQ(split.errors.count, 1U);

    HccpSt::ErrorCheckResult across;
    {
        ConsoleSilencer console;
        HccpSt::StdoutCapture capture;
        capture.Start();
        (void)write(STDOUT_FILENO, "prefix er\n", 10);
        usleep(200 * 1000);
        (void)write(STDOUT_FILENO, "ror suffix\n", 11);
        across = capture.Finish();
    }
    EXPECT_TRUE(across.complete);
    EXPECT_EQ(across.errors.count, 0U);
}

/* An unterminated final line still settles at EOF with the same once-per-line rule. */
TEST(CaptureTest, StreamingRecordsUnterminatedLineAtEof)
{
    const char* payload = "no trailing newline but error here";
    HccpSt::ErrorCheckResult result;
    {
        ConsoleSilencer console;
        HccpSt::StdoutCapture capture;
        capture.Start();
        (void)write(STDOUT_FILENO, payload, strlen(payload));
        result = capture.Finish();
    }
    EXPECT_TRUE(result.complete);
    EXPECT_EQ(result.errors.count, 1U);
}

TEST(CaptureTest, ReadFailureMakesCaptureUnknown)
{
    HccpSt::ErrorCheckResult result;
    unsigned int hits;
    {
        ConsoleSilencer console;
        Arm(Fault::TEE_READ);
        HccpSt::StdoutCapture capture;
        capture.Start();
        result = capture.Finish();
        hits = g_faultHits.load();
    }
    EXPECT_EQ(hits, 1U);
    EXPECT_FALSE(result.complete);
    EXPECT_FALSE(result.Passed());
}

TEST_P(CaptureRestoreTest, FinishHandlesRestorationFailureAndInterruption)
{
    const Fault fault = GetParam();
    const bool interrupted = fault == Fault::RESTORE_OUT_EINTR || fault == Fault::RESTORE_ERR_EINTR;
    const bool restoreOut = fault == Fault::RESTORE_OUT || fault == Fault::RESTORE_OUT_EINTR;
    int before = OpenFdCount();
    ASSERT_GE(before, 0);
    HccpSt::ErrorCheckResult result;
    unsigned int startHits;
    unsigned int finishHits;
    bool targetOpen;
    bool verdictDelivered;
    {
        ConsoleSilencer console;
        Arm(fault);
        HccpSt::StdoutCapture capture;
        capture.Start();
        startHits = g_faultHits.load();
        fprintf(stderr, "error before stream restoration\n");
        result = capture.Finish();
        finishHits = g_faultHits.load();
        targetOpen = fcntl(restoreOut ? STDOUT_FILENO : STDERR_FILENO, F_GETFD) >= 0;
        // The verdict must reach a stream whose descriptor survived restoration.
        auto results = PassingResults();
        results.stdoutErrors = result;
        FILE* const chosen = HccpSt::VerdictOutputStream(result);
        clearerr(chosen);
        HccpSt::PrintVerdict(results, HccpSt::EvaluateVerdict(results), chosen);
        (void)fflush(chosen);
        verdictDelivered = !ferror(chosen);
        clearerr(chosen);
    }
    EXPECT_EQ(startHits, 0U); // Failure must target Finish, never Start's redirection.
    EXPECT_EQ(finishHits, 1U);
    EXPECT_EQ(result.complete, interrupted);
    EXPECT_EQ(targetOpen, interrupted);
    // The hard-failed side reports itself closed; EINTR retries leave both sides restored.
    EXPECT_EQ(result.restoreOutOk, fault != Fault::RESTORE_OUT);
    EXPECT_EQ(result.restoreErrOk, fault != Fault::RESTORE_ERR);
    if (!result.restoreOutOk) {
        EXPECT_EQ(HccpSt::VerdictOutputStream(result), stderr);
    }
    if (!result.restoreErrOk) {
        EXPECT_EQ(HccpSt::VerdictOutputStream(result), stdout);
    }
    EXPECT_TRUE(verdictDelivered); // Never select the closed side: the report must land.
    EXPECT_EQ(result.errors.count, 1U);
    EXPECT_EQ(OpenFdCount(), before);
    auto results = PassingResults();
    results.stdoutErrors = result;
    EXPECT_FALSE(HccpSt::EvaluateVerdict(results).Passed());
    auto report = Report(results); // An independent FILE stream survives broken stdout.
    EXPECT_NE(
        report.find(interrupted ? "stdout-errors: FAIL (1 lines" : "stdout-errors: UNKNOWN (capture failed)"),
        std::string::npos);
    if (!interrupted) {
        EXPECT_EQ(report.find("error before stream restoration"), std::string::npos);
    }
}

INSTANTIATE_TEST_SUITE_P(
    SystemCalls, CaptureRestoreTest,
    ::testing::Values(Fault::RESTORE_OUT, Fault::RESTORE_ERR, Fault::RESTORE_OUT_EINTR, Fault::RESTORE_ERR_EINTR));

TEST(VerdictOutputTest, SelectionPrefersARestoredStream)
{
    HccpSt::ErrorCheckResult result; // Both sides restored: the report goes to stdout.
    EXPECT_EQ(HccpSt::VerdictOutputStream(result), stdout);
    result.restoreOutOk = false;
    EXPECT_EQ(HccpSt::VerdictOutputStream(result), stderr);
    result.restoreErrOk = false;
    EXPECT_EQ(HccpSt::VerdictOutputStream(result), stdout); // Both sides closed: the loss is unavoidable.
}

TEST_P(ForwardWriteTest, TeeRetriesOrReportsUnknownWithoutLosingInput)
{
    const std::string payload = "forwarded error payload\n";
    const bool failed = GetParam() == Fault::WRITE_FAILURE;
    int before = OpenFdCount();
    ASSERT_GE(before, 0);
    HccpSt::ErrorCheckResult result;
    unsigned int hits;
    ssize_t written;
    std::string forwarded;
    {
        ConsoleSilencer console;
        Arm(GetParam());
        HccpSt::StdoutCapture capture;
        capture.Start();
        written = write(STDOUT_FILENO, payload.data(), payload.size());
        result = capture.Finish();
        hits = g_faultHits.load();
        forwarded = g_forwarded;
    }
    ASSERT_EQ(written, static_cast<ssize_t>(payload.size()));
    EXPECT_EQ(hits, 1U);
    EXPECT_EQ(result.complete, !failed);
    EXPECT_EQ(result.errors.count, 1U); // Forwarding failure must not skip input scanning.
    EXPECT_EQ(forwarded, failed ? "" : payload);
    EXPECT_EQ(OpenFdCount(), before);
    auto results = PassingResults();
    results.stdoutErrors = result;
    EXPECT_FALSE(HccpSt::EvaluateVerdict(results).Passed());
    if (failed) {
        EXPECT_NE(Report(results).find("stdout-errors: UNKNOWN (capture failed)"), std::string::npos);
        EXPECT_EQ(Report(results).find("forwarded error payload"), std::string::npos);
    }
}

INSTANTIATE_TEST_SUITE_P(
    SystemCalls, ForwardWriteTest, ::testing::Values(Fault::WRITE_EINTR, Fault::WRITE_SHORT, Fault::WRITE_FAILURE));

} // namespace

extern "C" {
int __real_pipe(int[2]);
int __real_dup(int);
int __real_dup2(int, int);
int __real_pthread_create(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
int __real_pthread_join(pthread_t, void**);
ssize_t __real_read(int, void*, size_t);
ssize_t __real_write(int, const void*, size_t);
DIR* __real_opendir(const char*);
struct dirent* __real_readdir(DIR*);
int __real_closedir(DIR*);
int __real_lstat(const char*, struct stat*);
FILE* __real_fopen(const char*, const char*);
int __real_fclose(FILE*);

int __wrap_pipe(int fds[2])
{
    if (g_fault == Fault::PIPE) {
        ++g_faultHits;
        errno = EMFILE;
        return -1;
    }
    return __real_pipe(fds);
}

int __wrap_dup(int fd)
{
    ++g_dupCalls;
    if ((g_fault == Fault::DUP_OUT && g_dupCalls == 1) || (g_fault == Fault::DUP_ERR && g_dupCalls == 2)) {
        ++g_faultHits;
        errno = EMFILE;
        return -1;
    }
    int result = __real_dup(fd);
    if (g_dupCalls == 1) {
        g_captureOut = result;
    } else if (g_dupCalls == 2) {
        g_captureErr = result;
    }
    return result;
}

int __wrap_dup2(int source, int target)
{
    bool shouldFail
        = (g_fault == Fault::REDIRECT_OUT && target == STDOUT_FILENO)
          || ((g_fault == Fault::REDIRECT_ERR || g_fault == Fault::REDIRECT_EINTR) && target == STDERR_FILENO);
    shouldFail = shouldFail
                 || ((g_fault == Fault::RESTORE_OUT || g_fault == Fault::RESTORE_OUT_EINTR) && source == g_captureOut
                     && target == STDOUT_FILENO)
                 || ((g_fault == Fault::RESTORE_ERR || g_fault == Fault::RESTORE_ERR_EINTR) && source == g_captureErr
                     && target == STDERR_FILENO);
    if (shouldFail && !g_faultFired.exchange(true)) {
        ++g_faultHits;
        bool interrupted = g_fault == Fault::REDIRECT_EINTR || g_fault == Fault::RESTORE_OUT_EINTR
                           || g_fault == Fault::RESTORE_ERR_EINTR;
        errno = interrupted ? EINTR : EBUSY;
        return -1;
    }
    return __real_dup2(source, target);
}

int __wrap_pthread_create(pthread_t* thread, const pthread_attr_t* attributes, void* (*entry)(void*), void* context)
{
    if (g_fault == Fault::THREAD) {
        ++g_faultHits;
        return EAGAIN;
    }
    return __real_pthread_create(thread, attributes, entry, context);
}

int __wrap_pthread_join(pthread_t thread, void** result)
{
    if (g_fault == Fault::JOIN && !g_faultFired.exchange(true)) {
        ++g_faultHits;
        static const char MESSAGE[] = "[selftest] pthread_join failure injected\n";
        (void)__real_write(STDERR_FILENO, MESSAGE, sizeof(MESSAGE) - 1);
        return EINVAL;
    }
    return __real_pthread_join(thread, result);
}

ssize_t __wrap_read(int fd, void* data, size_t size)
{
    if ((g_fault == Fault::TEE_READ || g_fault == Fault::TEE_EINTR) && !g_faultFired.exchange(true)) {
        ++g_faultHits;
        errno = g_fault == Fault::TEE_EINTR ? EINTR : EIO;
        return -1;
    }
    return __real_read(fd, data, size);
}

ssize_t __wrap_write(int fd, const void* data, size_t size)
{
    bool forwarding
        = fd == g_captureOut
          && (g_fault == Fault::WRITE_EINTR || g_fault == Fault::WRITE_SHORT || g_fault == Fault::WRITE_FAILURE);
    if (forwarding && !g_faultFired.exchange(true)) {
        ++g_faultHits;
        if (g_fault != Fault::WRITE_SHORT) {
            errno = g_fault == Fault::WRITE_EINTR ? EINTR : EIO;
            return -1;
        }
        size = std::min(size, size_t{3});
    }
    ssize_t result = __real_write(fd, data, size);
    if (forwarding && result > 0) {
        g_forwarded.append(static_cast<const char*>(data), static_cast<size_t>(result));
    }
    return result;
}

DIR* __wrap_opendir(const char* path)
{
    bool root = g_logRoot == path;
    if (root && g_fault == Fault::OPEN_DIR) {
        ++g_faultHits;
        errno = EACCES;
        return nullptr;
    }
    DIR* result = __real_opendir(path);
    if (root) {
        g_scanDir = result;
    }
    return result;
}

struct dirent* __wrap_readdir(DIR* dir)
{
    if (dir == g_scanDir && g_fault == Fault::READ_DIR) {
        ++g_faultHits;
        errno = EIO;
        return nullptr;
    }
    return __real_readdir(dir);
}

int __wrap_closedir(DIR* dir)
{
    bool fail = dir == g_scanDir && g_fault == Fault::CLOSE_DIR;
    int result = __real_closedir(dir);
    if (fail) {
        ++g_faultHits;
        errno = EIO;
        return -1;
    }
    return result;
}

int __wrap_lstat(const char* path, struct stat* info)
{
    if (ProbeFile(path) && g_fault == Fault::STAT) {
        ++g_faultHits;
        errno = EACCES;
        return -1;
    }
    return __real_lstat(path, info);
}

ssize_t BrokenRead(void*, char* buffer, size_t size)
{
    if (g_fault == Fault::PARTIAL_READ) {
        static const std::string PREFIX = [] {
            std::string lines;
            for (int index = 0; index < 25; ++index) {
                lines += "[ERROR] partial\n";
            }
            return lines;
        }();
        if (g_partialOffset < PREFIX.size()) {
            size_t copied = std::min(size, PREFIX.size() - g_partialOffset);
            std::copy_n(PREFIX.data() + g_partialOffset, copied, buffer);
            g_partialOffset += copied;
            return static_cast<ssize_t>(copied);
        }
    }
    ++g_faultHits;
    errno = EIO;
    return -1;
}

FILE* __wrap_fopen(const char* path, const char* mode)
{
    if (ProbeFile(path)) {
        if (g_fault == Fault::OPEN_FILE) {
            ++g_faultHits;
            errno = EACCES;
            return nullptr;
        }
        if (g_fault == Fault::READ_FILE || g_fault == Fault::PARTIAL_READ) {
            cookie_io_functions_t functions = {};
            functions.read = BrokenRead;
            return fopencookie(nullptr, "r", functions);
        }
    }
    FILE* result = __real_fopen(path, mode);
    if (ProbeFile(path)) {
        g_scanFile = result;
    }
    return result;
}

int __wrap_fclose(FILE* file)
{
    bool fail = file == g_scanFile && g_fault == Fault::CLOSE_FILE;
    int result = __real_fclose(file);
    if (fail) {
        ++g_faultHits;
        errno = EIO;
        return -1;
    }
    return result;
}
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
