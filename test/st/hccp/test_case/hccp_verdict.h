/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstddef>
#include <cstdio>
#include <pthread.h>
#include <string>
#include <vector>

namespace HccpSt {

constexpr size_t VERDICT_MAX_REPORT = 20;
constexpr size_t VERDICT_DETAIL_MAX = 200;

std::string SanitizeDetail(const std::string& text);

struct ErrorSummary {
    size_t count = 0;
    std::vector<std::string> details;

    /* Count every match, retaining only bounded, single-line diagnostic details. */
    void Record(const std::string& line);
    void Record(const char* line, size_t length, const std::string& prefix = "");
};

struct ErrorCheckResult {
    bool complete = true;
    /* Console restoration outcome, meaningful only on a StdoutCapture result: the failed
     * side was closed by RestoreStreams, so that stream can no longer sink bytes. */
    bool restoreOutOk = true;
    bool restoreErrOk = true;
    ErrorSummary errors;
    std::string reason;

    void MarkUnknown(const char* failureReason);
    bool Passed() const;
};

struct GtestRunResult {
    int returnCode = 0;
    int selected = 0;
    int failed = 0;
    int started = 0;
    int finished = 0;
    int skipped = 0;

    int Executed() const;
    bool Passed() const;
};

struct CaseLoadResult {
    bool loaded = false;
    std::string path;
};

struct VerdictResults {
    GtestRunResult gtest;
    ErrorCheckResult stdoutErrors;
    ErrorCheckResult logErrors;
    CaseLoadResult caseLoad;
};

struct VerdictDecision {
    /* Fail-closed defaults: a default-constructed decision never passes. */
    bool gtestOk = false;
    bool stdoutOk = false;
    bool logOk = false;
    bool loadOk = false;

    bool Passed() const;
};

VerdictDecision EvaluateVerdict(const VerdictResults& results);
void PrintVerdict(const VerdictResults& results, const VerdictDecision& decision, FILE* output = stdout);

/* Picks the console stream that still receives bytes: a failed restore closed that side's
 * descriptor, so the report goes to the surviving side; when both sides are closed the
 * report is unavoidably lost and stdout is returned. */
FILE* VerdictOutputStream(const ErrorCheckResult& stdoutErrors);

bool IsForeignRunDir(const char* name, unsigned long runId);
/* The root stat is the only ENOENT special case: the tree is created lazily by
 * the first written log line, so an absent root is a legal empty result (a
 * stderr diagnostic fires when tests did start), never a degradation verdict.
 * The root check follows symlinks (a symlinked log tree still scans) while
 * entries stay lstat, so in-tree symlinks are never followed. Any syscall
 * failure or path truncation during the traversal marks the scan UNKNOWN
 * instead of skipping or passing. runId classifies slot-shaped run_* names: a
 * shared-tree scan (runId 0) treats every slot as another concurrent runner's
 * and skips it; a slotted scan (runId > 0) enters only its own run_<runId> and
 * leaves any other -- or unrepresentable -- slot number untouched. */
ErrorCheckResult ScanLogFiles(const char* logDir, bool testsStarted, unsigned long runId);

/* The tee thread owns the result until Finish joins it. No concurrent readers. */
class StdoutCapture {
public:
    StdoutCapture() = default;
    ~StdoutCapture();
    StdoutCapture(const StdoutCapture&) = delete;
    StdoutCapture& operator=(const StdoutCapture&) = delete;

    void Start();
    ErrorCheckResult Finish();

private:
    static void* TeeEntry(void* context);
    void Tee();
    bool RestoreStreams();
    void CloseDescriptors();

    int savedOut_ = -1;
    int savedErr_ = -1;
    int pipeRead_ = -1;
    int pipeWrite_ = -1;
    pthread_t thread_ = {};
    bool threadStarted_ = false;
    ErrorCheckResult result_;
};

} // namespace HccpSt
