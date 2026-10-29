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

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <memory>
#include <sys/stat.h>
#include <utility>
#include <unistd.h>

namespace HccpSt {
namespace {

    bool Redirect(int source, int target)
    {
        int result;
        do {
            result = dup2(source, target);
        } while (result < 0 && errno == EINTR);
        return result >= 0;
    }

    void CloseFd(int& fd)
    {
        if (fd >= 0) {
            (void)close(fd);
            fd = -1;
        }
    }

    bool LineHasError(const std::string& line)
    {
        static const char NEEDLE[] = "error";
        return std::search(
                   line.begin(), line.end(), NEEDLE, NEEDLE + sizeof(NEEDLE) - 1,
                   [](unsigned char left, unsigned char right) {
                       return std::tolower(left) == right;
                   })
               != line.end();
    }

    void MarkScanFailure(ErrorCheckResult& result, const char* operation, const char* path, int error)
    {
        result.MarkUnknown("scan failed");
        (void)fprintf(stderr, "[VERDICT] scan failed: %s %s: %s\n", operation, path, strerror(error));
    }

    void ScanLogFile(const char* path, const std::string& name, ErrorCheckResult& result)
    {
        std::unique_ptr<FILE, int (*)(FILE*)> file(fopen(path, "r"), fclose);
        if (!file) {
            MarkScanFailure(result, "fopen", path, errno);
            return;
        }
        char* line = nullptr;
        size_t capacity = 0;
        ssize_t length;
        while ((length = getline(&line, &capacity, file.get())) > 0) {
            if (strncmp(line, "[ERROR]", 7) == 0) {
                size_t detailLength = static_cast<size_t>(length) - (line[length - 1] == '\n' ? 1 : 0);
                result.errors.Record(line, detailLength, name + ": ");
            }
        }
        if (ferror(file.get()) || !feof(file.get())) {
            MarkScanFailure(result, "getline", path, errno);
        }
        free(line);
        if (fclose(file.release()) != 0) {
            MarkScanFailure(result, "fclose", path, errno);
        }
    }

    void ScanEntry(
        const char* directory, const char* entry, const std::string& relative, unsigned long runId,
        ErrorCheckResult& result);

    void ScanTree(const char* directory, const std::string& relative, unsigned long runId, ErrorCheckResult& result)
    {
        std::unique_ptr<DIR, int (*)(DIR*)> dir(opendir(directory), closedir);
        if (!dir) {
            MarkScanFailure(result, "opendir", directory, errno);
            return;
        }
        while (true) {
            errno = 0;
            const auto* entry = readdir(dir.get());
            if (entry == nullptr) {
                if (errno != 0) {
                    MarkScanFailure(result, "readdir", directory, errno);
                }
                break;
            }
            if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
                ScanEntry(directory, entry->d_name, relative, runId, result);
            }
        }
        if (closedir(dir.release()) != 0) {
            MarkScanFailure(result, "closedir", directory, errno);
        }
    }

    void ScanEntry(
        const char* directory, const char* entry, const std::string& relative, unsigned long runId,
        ErrorCheckResult& result)
    {
        char path[PATH_MAX];
        int length = snprintf(path, sizeof(path), "%s/%s", directory, entry);
        if (length < 0 || length >= static_cast<int>(sizeof(path))) {
            MarkScanFailure(result, "path", directory, length < 0 ? errno : ENAMETOOLONG);
            return;
        }
        struct stat info = {};
        if (lstat(path, &info) != 0) {
            MarkScanFailure(result, "lstat", path, errno);
            return;
        }
        std::string name = relative.empty() ? entry : relative + "/" + entry;
        if (S_ISDIR(info.st_mode)) {
            if (!IsForeignRunDir(entry, runId)) {
                ScanTree(path, name, runId, result);
            }
        } else if (S_ISREG(info.st_mode)) {
            size_t size = strlen(entry);
            if (size >= 4 && strcmp(entry + size - 4, ".log") == 0) {
                ScanLogFile(path, name, result);
            }
        }
    }

    void PrintGtest(const GtestRunResult& run, bool passed, FILE* output)
    {
        if (passed) {
            fprintf(output, "[VERDICT] gtest:         OK\n");
        } else if (run.started == 0) {
            fprintf(output, "[VERDICT] gtest:         FAIL (0 started of %d selected, NO TEST RAN)\n", run.selected);
        } else if (run.Executed() == 0) {
            fprintf(
                output, "[VERDICT] gtest:         FAIL (%d started, 0 completed non-skipped, ALL SKIPPED)\n",
                run.started);
        } else if (run.failed == 0) {
            fprintf(
                output, "[VERDICT] gtest:         FAIL (env/ad-hoc failure, 0 of %d tests failed)\n", run.Executed());
        } else {
            fprintf(output, "[VERDICT] gtest:         FAIL (%d of %d tests failed)\n", run.failed, run.started);
        }
    }

    void PrintErrorCheck(const ErrorCheckResult& result, bool isStdout, FILE* output)
    {
        const char* label = isStdout ? "stdout-errors:" : "log-errors:   ";
        if (!result.complete) {
            fprintf(output, "[VERDICT] %s UNKNOWN (%s)\n", label, SanitizeDetail(result.reason).c_str());
            return;
        }
        if (result.errors.count == 0) {
            fprintf(output, "[VERDICT] %s CLEAN (%s)\n", label, isStdout ? "0 lines" : "0 [ERROR] lines");
            return;
        }
        if (isStdout) {
            fprintf(
                output, "[VERDICT] %s FAIL (%zu lines, cap=%zu, first: %s)\n", label, result.errors.count,
                VERDICT_MAX_REPORT, result.errors.details.empty() ? "" : result.errors.details.front().c_str());
        } else {
            fprintf(
                output, "[VERDICT] %s FAIL (%zu lines, cap=%zu):\n", label, result.errors.count, VERDICT_MAX_REPORT);
        }
        for (const auto& detail : result.errors.details) {
            fprintf(output, "  [%s] %s\n", isStdout ? "stdout" : "log", detail.c_str());
        }
    }

} // namespace

std::string SanitizeDetail(const std::string& text)
{
    std::string detail = text.substr(0, VERDICT_DETAIL_MAX);
    for (char& ch : detail) {
        if (static_cast<unsigned char>(ch) < 0x20 || static_cast<unsigned char>(ch) == 0x7f) {
            ch = '?';
        }
    }
    return detail;
}

void ErrorSummary::Record(const std::string& line) { Record(line.data(), line.size()); }

void ErrorSummary::Record(const char* line, size_t length, const std::string& prefix)
{
    count++;
    if (details.size() < VERDICT_MAX_REPORT) {
        std::string detail = SanitizeDetail(prefix);
        size_t retained = std::min(length, VERDICT_DETAIL_MAX - detail.size());
        detail += SanitizeDetail(std::string(line, retained));
        details.push_back(std::move(detail));
    }
}

void ErrorCheckResult::MarkUnknown(const char* failureReason)
{
    complete = false;
    reason = failureReason;
}

bool ErrorCheckResult::Passed() const { return complete && errors.count == 0; }
int GtestRunResult::Executed() const { return finished - skipped; }
bool GtestRunResult::Passed() const { return returnCode == 0 && Executed() > 0; }
bool VerdictDecision::Passed() const { return gtestOk && stdoutOk && logOk && loadOk; }

VerdictDecision EvaluateVerdict(const VerdictResults& results)
{
    return {results.gtest.Passed(), results.stdoutErrors.Passed(), results.logErrors.Passed(), results.caseLoad.loaded};
}

FILE* VerdictOutputStream(const ErrorCheckResult& stdoutErrors)
{
    if (stdoutErrors.restoreOutOk) {
        return stdout;
    }
    return stdoutErrors.restoreErrOk ? stderr : stdout;
}

void PrintVerdict(const VerdictResults& results, const VerdictDecision& decision, FILE* output)
{
    PrintGtest(results.gtest, decision.gtestOk, output);
    PrintErrorCheck(results.stdoutErrors, true, output);
    PrintErrorCheck(results.logErrors, false, output);
    if (decision.loadOk) {
        fprintf(output, "[VERDICT] case-load:     OK (%s)\n", SanitizeDetail(results.caseLoad.path).c_str());
    } else {
        fprintf(output, "[VERDICT] case-load:     FAIL\n");
    }
    fprintf(output, "[VERDICT] OVERALL:       %s\n", decision.Passed() ? "PASS" : "FAILED");
}

bool IsForeignRunDir(const char* name, unsigned long runId)
{
    if (strncmp(name, "run_", 4) != 0 || name[4] == '\0') {
        return false;
    }
    unsigned long value = 0;
    for (const char* digit = name + 4; *digit != '\0'; digit++) {
        if (*digit < '0' || *digit > '9') {
            return false;
        }
        if (value > (ULONG_MAX - static_cast<unsigned long>(*digit - '0')) / 10UL) {
            return true; /* slot-shaped but unrepresentable: no legal slot number is this large */
        }
        value = value * 10UL + static_cast<unsigned long>(*digit - '0');
    }
    /* runId == 0 (shared tree): every run_* slot belongs to an isolated concurrent
     * runner -- foreign. runId > 0 (this runner scans its own run_<N> root): only
     * the matching slot number is ours. */
    return runId == 0 || value != runId;
}

ErrorCheckResult ScanLogFiles(const char* logDir, bool testsStarted, unsigned long runId)
{
    ErrorCheckResult result;
    if (logDir[0] == '\0') {
        result.MarkUnknown("scan failed");
        fprintf(stderr, "[VERDICT] scan failed: cannot resolve log dir\n");
        return result;
    }
    struct stat info = {};
    /* stat, not lstat: a log tree exposed through a symlinked root must still scan, while
     * entries below stay lstat so in-tree symlinks are never followed. */
    if (stat(logDir, &info) != 0) {
        int error = errno;
        if (error == ENOENT) {
            /* The tree is created lazily by the first written log line, so an absent
             * root means "nothing was ever logged" -- a legal empty result, not a
             * degradation: failing here would break filtered runs over cases that
             * never emit a log line. */
            if (testsStarted) {
                fprintf(stderr, "[VERDICT] log tree absent: no log line was ever written\n");
            }
            return result;
        }
        result.MarkUnknown("scan failed");
        fprintf(stderr, "[VERDICT] scan failed: stat %s: %s\n", logDir, strerror(error));
        return result;
    }
    if (!S_ISDIR(info.st_mode)) {
        result.MarkUnknown("scan failed");
        fprintf(stderr, "[VERDICT] scan failed: %s is not a directory\n", logDir);
        return result;
    }
    ScanTree(logDir, "", runId, result);
    return result;
}

StdoutCapture::~StdoutCapture() { (void)Finish(); }

void StdoutCapture::Start()
{
    int fds[2];
    if (pipe(fds) != 0) {
        result_.MarkUnknown("capture failed");
        return;
    }
    pipeRead_ = fds[0];
    pipeWrite_ = fds[1];
    (void)fflush(stdout);
    (void)fflush(stderr);
    savedOut_ = dup(STDOUT_FILENO);
    savedErr_ = dup(STDERR_FILENO);
    if (savedOut_ < 0 || savedErr_ < 0) {
        result_.MarkUnknown("capture failed");
        CloseDescriptors();
        return;
    }
    if (!Redirect(pipeWrite_, STDOUT_FILENO) || !Redirect(pipeWrite_, STDERR_FILENO)) {
        result_.MarkUnknown("capture failed");
        (void)RestoreStreams();
        CloseDescriptors();
        return;
    }
    CloseFd(pipeWrite_);
    (void)setvbuf(stdout, nullptr, _IONBF, 0);
    if (pthread_create(&thread_, nullptr, TeeEntry, this) != 0) {
        result_.MarkUnknown("capture failed");
        (void)RestoreStreams();
        CloseDescriptors();
        return;
    }
    threadStarted_ = true;
}

ErrorCheckResult StdoutCapture::Finish()
{
    if (threadStarted_) {
        (void)fflush(stdout);
        (void)fflush(stderr);
        bool restored = RestoreStreams();
        if (pthread_join(thread_, nullptr) != 0) {
            // A failed join cannot safely return a result still owned by the thread.
            std::abort();
        }
        threadStarted_ = false;
        if (!restored) {
            result_.MarkUnknown("capture failed");
        }
        CloseDescriptors();
    }
    return result_;
}

bool StdoutCapture::RestoreStreams()
{
    bool restored = true;
    if (!Redirect(savedOut_, STDOUT_FILENO)) {
        // Close the pipe writer even if restoring the console fails, so join can finish.
        (void)close(STDOUT_FILENO);
        restored = false;
        result_.restoreOutOk = false;
    }
    if (!Redirect(savedErr_, STDERR_FILENO)) {
        (void)close(STDERR_FILENO);
        restored = false;
        result_.restoreErrOk = false;
    }
    return restored;
}

void StdoutCapture::CloseDescriptors()
{
    CloseFd(savedOut_);
    CloseFd(savedErr_);
    CloseFd(pipeRead_);
    CloseFd(pipeWrite_);
}

void* StdoutCapture::TeeEntry(void* context)
{
    static_cast<StdoutCapture*>(context)->Tee();
    return nullptr;
}

void StdoutCapture::Tee()
{
    /* Streams the captured pipe with O(1) memory: the current logical line is summarized
     * by a bounded head kept for details, a needle-length seam so a match may span
     * read-chunk borders, and a matched-once flag; Record fires only at the line's
     * newline or at EOF, preserving the matched-lines count for arbitrarily long lines. */
    constexpr size_t SEAM = 4; /* strlen("error") - 1 */
    char buffer[4096];
    std::string seam;   /* tail of the current line received in earlier chunks */
    std::string prefix; /* bounded head of the current line, kept for details */
    bool matched = false;
    ssize_t length;
    while ((length = read(pipeRead_, buffer, sizeof(buffer))) != 0) {
        if (length < 0) {
            if (errno == EINTR) {
                continue;
            }
            result_.MarkUnknown("capture failed");
            break;
        }
        ssize_t offset = 0;
        while (offset < length) {
            ssize_t written = write(savedOut_, buffer + offset, static_cast<size_t>(length - offset));
            if (written < 0 && errno == EINTR) {
                continue;
            }
            if (written <= 0) {
                result_.MarkUnknown("capture failed");
                break;
            }
            offset += written;
        }
        size_t begin = 0;
        while (begin < static_cast<size_t>(length)) {
            const char* base = buffer + begin;
            size_t left = static_cast<size_t>(length) - begin;
            const void* found = memchr(base, '\n', left);
            size_t segment = found != nullptr ? static_cast<size_t>(static_cast<const char*>(found) - base) : left;
            std::string scan = seam + std::string(base, segment);
            if (LineHasError(scan)) {
                matched = true;
            }
            if (prefix.size() < VERDICT_DETAIL_MAX) {
                prefix.append(base, std::min(segment, VERDICT_DETAIL_MAX - prefix.size()));
            }
            begin += segment;
            if (found == nullptr) {
                size_t keep = std::min(scan.size(), SEAM);
                seam.assign(scan, scan.size() - keep, keep);
                break;
            }
            if (matched) {
                result_.errors.Record(prefix);
            }
            matched = false;
            prefix.clear();
            seam.clear();
            begin += 1;
        }
    }
    if (matched) {
        result_.errors.Record(prefix);
    }
}

} // namespace HccpSt
