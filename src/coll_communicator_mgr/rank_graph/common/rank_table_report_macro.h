/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANK_TABLE_REPORT_MACRO_H
#define RANK_TABLE_REPORT_MACRO_H

#include "../rank_table_info/rank_table_source.h" // RankTableSource 枚举
#include "nlohmann/json.hpp"                      // OBJ.contains / OBJ.begin / OBJ[CONFIG].dump()
#include "exception_util.h"                       // THROW / StringFormat
#include "adapter_error_manager_pub.h"            // RPT_INPUT_ERR
#include "common/hccl_log_keywords_pub.h"

namespace Hccl {

inline void LogMissingRootInfoConfig(const std::string& keys, const std::string& config, const std::string& expect)
{
    HCCL_ERROR(
        "[%s][%s] errNo[0x%016llx] required config[%s] is missing, available keys[%s], expected[%s]",
        LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA),
        config.c_str(), keys.c_str(), expect.c_str());
}

inline void LogInvalidRootInfoConfig(const std::string& value, const std::string& config, const std::string& expect)
{
    HCCL_ERROR(
        "[%s][%s] errNo[0x%016llx] config[%s] value[%s] is invalid, expected[%s]", LOG_KEYWORDS_INIT_GROUP.c_str(),
        LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), config.c_str(), value.c_str(),
        expect.c_str());
}

inline void LogInvalidRankTableConfig(const std::string& value, const std::string& config, const std::string& expect)
{
    HCCL_ERROR(
        "[%s][%s] errNo[0x%016llx] ranktable config[%s] value[%s] is invalid, expected[%s]",
        LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA),
        config.c_str(), value.c_str(), expect.c_str());
}

#define TRY_CATCH_THROW_REPORT_ROOTINFO(EXCEPTION, MSG, EXPR, OBJ, CONFIG, EXPECT)             \
    do {                                                                                       \
        try {                                                                                  \
            EXPR;                                                                              \
        } catch (HcclException & e) {                                                          \
            if (!OBJ.contains(CONFIG)) {                                                       \
                std::string keys;                                                              \
                if (OBJ.is_object()) {                                                         \
                    for (auto it = OBJ.begin(); it != OBJ.end(); ++it) {                       \
                        if (!keys.empty())                                                     \
                            keys += ", ";                                                      \
                        keys += it.key();                                                      \
                    }                                                                          \
                }                                                                              \
                RPT_INPUT_ERR(                                                                 \
                    true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}), \
                    std::vector<std::string>({keys, CONFIG, EXPECT}));                         \
                LogMissingRootInfoConfig(keys, CONFIG, EXPECT);                                \
                THROW<EXCEPTION>(StringFormat("%s, %s", e.what(), MSG.c_str()));               \
            } else {                                                                           \
                RPT_INPUT_ERR(                                                                 \
                    true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}), \
                    std::vector<std::string>({OBJ[CONFIG].dump(), CONFIG, EXPECT}));           \
                LogInvalidRootInfoConfig(OBJ[CONFIG].dump(), CONFIG, EXPECT);                  \
                THROW<EXCEPTION>(StringFormat("%s, %s", e.what(), MSG.c_str()));               \
            }                                                                                  \
        } catch (std::exception & e) {                                                         \
            THROW<EXCEPTION>(StringFormat("%s, %s", e.what(), MSG.c_str()));                   \
        } catch (...) {                                                                        \
            THROW<EXCEPTION>(StringFormat("Unknown error occurs, %s", MSG.c_str()));           \
        }                                                                                      \
    } while (0)

#define TRY_CATCH_THROW_REPORT_JSON(EXCEPTION, MSG, EXPR, OBJ, CONFIG, EXPECT)                                 \
    do {                                                                                                       \
        try {                                                                                                  \
            EXPR;                                                                                              \
        } catch (HcclException & e) {                                                                          \
            if (!OBJ.contains(CONFIG)) {                                                                       \
                RPT_INPUT_ERR(                                                                                 \
                    true, "EI0017", std::vector<std::string>({"config"}), std::vector<std::string>({CONFIG})); \
                HCCL_ERROR(                                                                                    \
                    "[%s][%s] errNo[0x%016llx] json object does not contain property[%s]",                     \
                    LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CONFIG.c_str(),                    \
                    HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), CONFIG);                                         \
                THROW<EXCEPTION>(StringFormat("%s, %s", e.what(), MSG.c_str()));                               \
            } else {                                                                                           \
                RPT_INPUT_ERR(                                                                                 \
                    true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),                 \
                    std::vector<std::string>({OBJ[CONFIG].dump(), CONFIG, EXPECT}));                           \
                LogInvalidRankTableConfig(OBJ[CONFIG].dump(), CONFIG, EXPECT);                                 \
                THROW<EXCEPTION>(StringFormat("%s, %s", e.what(), MSG.c_str()));                               \
            }                                                                                                  \
        } catch (std::exception & e) {                                                                         \
            THROW<EXCEPTION>(StringFormat("%s, %s", e.what(), MSG.c_str()));                                   \
        } catch (...) {                                                                                        \
            THROW<EXCEPTION>(StringFormat("Unknown error occurs, %s", MSG.c_str()));                           \
        }                                                                                                      \
    } while (0)

#define TRY_CATCH_THROW_REPORT(EXCEPTION, MSG, EXPR, OBJ, CONFIG, EXPECT, SOURCE)       \
    do {                                                                                \
        if (SOURCE == RankTableSource::ROOTINFO) {                                      \
            TRY_CATCH_THROW_REPORT_ROOTINFO(EXCEPTION, MSG, EXPR, OBJ, CONFIG, EXPECT); \
        } else {                                                                        \
            TRY_CATCH_THROW_REPORT_JSON(EXCEPTION, MSG, EXPR, OBJ, CONFIG, EXPECT);     \
        }                                                                               \
    } while (0)

#define TRY_CATCH_THROW_REPORT_TOPO(EXCEPTION, MSG, EXPR, OBJ, CONFIG, EXPECT) \
    TRY_CATCH_THROW_REPORT_JSON(EXCEPTION, MSG, EXPR, OBJ, CONFIG, EXPECT)

} // namespace Hccl

#endif // RANK_TABLE_REPORT_MACRO_H
