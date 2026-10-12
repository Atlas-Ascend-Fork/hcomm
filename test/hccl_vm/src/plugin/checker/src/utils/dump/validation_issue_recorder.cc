/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dump/dump_json_utils.h"
#include "dump/dump_manager.h"
#include "dump/dump_run_manifest.h"
#include "dump/validation_issue_recorder.h"

namespace HcclSim {
using Json = nlohmann::json;

static const std::string VALIDATION_ISSUE_DUMP_TYPE = "validation_issues";
static const std::string VALIDATION_ISSUE_DUMP_PATH = "validation/issues.msgpack";

void ValidationIssueRecorder::Reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    nextIssueId_ = 1;
    issues_.clear();
}

void ValidationIssueRecorder::RecordIssue(
    const std::string& severity, const std::string& stage, const std::string& code, const nlohmann::json& detail)
{
    if (!DumpManager::GetInstance().IsEnabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    Json issue = Json::object();
    issue["issue_id"] = nextIssueId_++;
    issue["severity"] = severity;
    issue["stage"] = stage;
    issue["code"] = code;
    issue["detail"] = detail;
    issues_.push_back(issue);
}

HcclResult ValidationIssueRecorder::Flush() const
{
    DumpManager& dumpManager = DumpManager::GetInstance();
    if (!dumpManager.IsEnabled()) {
        return HcclResult::HCCL_SUCCESS;
    }

    Json issuesSnapshot = Json::array();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        issuesSnapshot = issues_;
    }

    Json issueDumpJson = Json::object();
    issueDumpJson["type"] = VALIDATION_ISSUE_DUMP_TYPE;
    issueDumpJson["issue_count"] = issuesSnapshot.size();
    issueDumpJson["issues"] = issuesSnapshot;
    DumpRunManifest::GetInstance().SetErrorCount(issuesSnapshot.size());
    return dumpManager.Write(VALIDATION_ISSUE_DUMP_PATH, issueDumpJson);
}
} // namespace HcclSim
