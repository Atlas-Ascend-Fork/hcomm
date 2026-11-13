/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_table_fast_parser.h"
#include "changed_rank_info.h"
#include "json_scan_simd.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>
#include <utility>
#include <thread>
#include <atomic>
#include <set>
#include "log.h"
#include "adapter_error_manager_pub.h"
#include "common/hccl_log_keywords_pub.h"
#include "ip_address.h"
#include "new_rank_info.h"
#include "rank_level_info.h"
#include "address_info.h"
#include "control_plane.h"
#include "topo_common_types.h"

namespace Hccl {
namespace {

    size_t SafeReserveCount(u32 declared, size_t totalBytes)
    {
        const size_t byBytes = totalBytes / kMinBytesPerRank;
        return std::min({static_cast<size_t>(declared), static_cast<size_t>(MAX_RANKCOUNT), byBytes});
    }

    // 并行 worker 内抑制字段级 RPT 上报: 并行失败会回退串行路径统一报错,
    // 提前抑制可避免并发上报与回退后的重复上报(thread_local 随 worker 线程销毁)
    thread_local bool gSuppressFastFieldReport = false;

    // 与 rank_table_report_macro.h 的 EI0017 上报保持一致: 必选字段缺失
    void ReportFastFieldMissing(const std::string& config)
    {
        if (!gSuppressFastFieldReport) {
            RPT_INPUT_ERR(true, "EI0017", std::vector<std::string>({"config"}), std::vector<std::string>({config}));
        }
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] json object does not contain property[%s]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CONFIG.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), config.c_str());
    }

    // 与 rank_table_report_macro.h 的 EI0014 上报保持一致: 字段取值非法
    void ReportFastFieldInvalid(const std::string& value, const std::string& config, const std::string& expect)
    {
        if (!gSuppressFastFieldReport) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({value, config, expect}));
        }
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] config[%s] value[%s] is invalid, expected[%s]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), config.c_str(),
            value.c_str(), expect.c_str());
    }

    // 轻量JSON扫描游标：只提供本schema需要的操作，不做DOM构建
    class JsonCursor {
    public:
        JsonCursor(const char* begin, const char* end) : p_(begin), end_(end) {}

        const char* Pos() const { return p_; }

        void SkipWs()
        {
            while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) {
                p_++;
            }
        }

        bool AtEnd() const { return p_ >= end_; }

        bool Is(char c)
        {
            SkipWs();
            return !AtEnd() && *p_ == c;
        }

        bool TryConsume(char c)
        {
            SkipWs();
            if (AtEnd() || *p_ != c) {
                return false;
            }
            p_++;
            return true;
        }

        // 容器内分隔符校验: 逗号后必须还有元素, 拒绝尾逗号
        bool CheckSeparator(char close)
        {
            if (TryConsume(',')) {
                return !Is(close);
            }
            return Is(close);
        }

        // 免分配键匹配：当前为 "key" 且与字面量一致（ranktable字段键均为普通ASCII）
        bool ParseKey(const char* expect)
        {
            SkipWs();
            if (AtEnd() || *p_ != '"') {
                return false;
            }
            const char* q = p_ + 1;
            const char* e = expect;
            while (*e != '\0') {
                if (q >= end_ || *q != *e) {
                    return false;
                }
                q++;
                e++;
            }
            if (q >= end_ || *q != '"') {
                return false;
            }
            p_ = q + 1;
            return true;
        }

        // 查表版: 字段名统一从 kRankTableKeys 取
        bool ParseKey(RankTableKey key) { return ParseKey(kRankTableKeys[key].name); }

        // 解析 "..." 字符串（支持标准转义）
        bool ParseString(std::string& out)
        {
            SkipWs();
            if (AtEnd() || *p_ != '"') {
                return false;
            }
            // 快速路径：ranktable绝大多数字符串无转义，NEON 批量定位 '"'/'\\'，一次分配
            const char* q = FindQuoteOrBackslash(p_ + 1, end_);
            if (q >= end_) {
                return false;
            }
            if (*q == '"') {
                for (const char* s = p_ + 1; s < q; ++s) {
                    if (static_cast<unsigned char>(*s) < 0x20) {
                        return false; // JSON 字符串不允许未转义控制字符
                    }
                }
                out.assign(p_ + 1, q - (p_ + 1));
                p_ = q + 1;
                return true;
            }
            // 含转义：回退到逐字符解析
            p_++;
            out.clear();
            return ParseEscapedString(out);
        }

        // 解析无符号整数（ranktable中字段均为普通十进制整数）。
        // 有意收紧为纯十进制: 与 nlohmann 对浮点/指数截断取整不同, -0/1e0/1.0/166.66 一律拒绝,
        // 属收紧非放宽, 拒绝行为由 UT 固化; 头文件一致性声明已同步标注该分歧。
        bool ParseU32(u32& out)
        {
            SkipWs();
            u64 value = 0;
            if (AtEnd() || *p_ < '0' || *p_ > '9') {
                return false;
            }
            if (*p_ == '0' && p_ + 1 < end_ && p_[1] >= '0' && p_[1] <= '9') {
                return false; // JSON 不允许前导零
            }
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') {
                value = value * 10 + static_cast<u64>(*p_ - '0');
                if (value > UINT32_MAX) {
                    return false;
                }
                p_++;
            }
            out = static_cast<u32>(value);
            return true;
        }

        bool ParseBool(bool& out)
        {
            SkipWs();
            if (end_ - p_ >= 4 && std::strncmp(p_, "true", 4) == 0) {
                p_ += 4;
                out = true;
                return true;
            }
            if (end_ - p_ >= 5 && std::strncmp(p_, "false", 5) == 0) {
                p_ += 5;
                out = false;
                return true;
            }
            return false;
        }

        // 递归跳过任意JSON值
        bool SkipValue(int depth = 0)
        {
            if (depth > kMaxJsonDepth) {
                return false;
            }
            SkipWs();
            if (AtEnd()) {
                return false;
            }
            const char ch = *p_;
            if (ch == '"') {
                return SkipJsonString();
            }
            if (ch == '{') {
                return SkipJsonObject(depth);
            }
            if (ch == '[') {
                return SkipJsonArray(depth);
            }
            if (ch == '-' || (ch >= '0' && ch <= '9')) {
                return SkipJsonNumber();
            }
            if (ch == 't') {
                return MatchLiteral("true", 4);
            }
            if (ch == 'f') {
                return MatchLiteral("false", 5);
            }
            if (ch == 'n') {
                return MatchLiteral("null", 4);
            }
            return false;
        }

    private:
        // 慢路径：逐字符处理转义，\uXXXX 原样保留
        bool ParseEscapedString(std::string& out)
        {
            while (p_ < end_) {
                const char ch = *p_++;
                if (ch == '"') {
                    return true;
                }
                if (ch == '\\') {
                    if (p_ >= end_) {
                        return false;
                    }
                    if (!AppendEscapedChar(out, *p_++)) {
                        return false;
                    }
                } else {
                    if (static_cast<unsigned char>(ch) < 0x20) {
                        return false;
                    }
                    out.push_back(ch);
                }
            }
            return false;
        }

        bool AppendEscapedChar(std::string& out, char esc)
        {
            switch (esc) {
                case '"':
                case '\\':
                case '/':
                    out.push_back(esc);
                    return true;
                case 'b':
                    out.push_back('\b');
                    return true;
                case 'f':
                    out.push_back('\f');
                    return true;
                case 'n':
                    out.push_back('\n');
                    return true;
                case 'r':
                    out.push_back('\r');
                    return true;
                case 't':
                    out.push_back('\t');
                    return true;
                case 'u':
                    return AppendUnicodeEscape(out);
                default:
                    return false; // JSON 只允许 "\\/bfnrtu 七种转义
            }
        }

        bool ReadHex4(u32& value)
        {
            if (end_ - p_ < 4) {
                return false;
            }
            value = 0;
            for (int i = 0; i < 4; ++i) {
                const char c = *p_++;
                u32 digit = 0;
                if (c >= '0' && c <= '9') {
                    digit = static_cast<u32>(c - '0');
                } else if (c >= 'a' && c <= 'f') {
                    digit = static_cast<u32>(c - 'a') + 10;
                } else if (c >= 'A' && c <= 'F') {
                    digit = static_cast<u32>(c - 'A') + 10;
                } else {
                    return false;
                }
                value = (value << 4) | digit;
            }
            return true;
        }

        bool AppendUnicodeEscape(std::string& out)
        {
            u32 code = 0;
            if (!ReadHex4(code)) {
                return false;
            }
            if (code >= 0xD800 && code <= 0xDBFF) {
                if (end_ - p_ < 2 || p_[0] != '\\' || p_[1] != 'u') {
                    return false;
                }
                p_ += 2;
                u32 low = 0;
                if (!ReadHex4(low) || low < 0xDC00 || low > 0xDFFF) {
                    return false;
                }
                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
            } else if (code >= 0xDC00 && code <= 0xDFFF) {
                return false;
            }
            AppendUtf8(out, code);
            return true;
        }

        static void AppendUtf8(std::string& out, u32 code)
        {
            // 按 RFC 3629 先确定 1~4 字节的序列长度，再统一写首字节和续字节，
            // 避免按码点范围展开四段重复的 push_back。
            static constexpr u32 kLeadByte[] = {0x00, 0x00, 0xC0, 0xE0, 0xF0};
            int length = 1;
            if (code > 0x7F) {
                length = (code <= 0x7FF) ? 2 : ((code <= 0xFFFF) ? 3 : 4);
            }
            int shift = 6 * (length - 1);
            out.push_back(static_cast<char>(kLeadByte[length] | (code >> shift)));
            while (shift > 0) {
                shift -= 6;
                out.push_back(static_cast<char>(0x80 | ((code >> shift) & 0x3F)));
            }
        }

        // 未知字段的字符串值：只扫描边界，不构建内容
        bool SkipJsonString()
        {
            p_++;
            while (p_ < end_) {
                const unsigned char ch = static_cast<unsigned char>(*p_);
                if (ch == '\\') {
                    if (end_ - p_ < 2) {
                        return false;
                    }
                    const char esc = p_[1];
                    if (esc == 'u') {
                        if (end_ - p_ < 6) {
                            return false;
                        }
                        for (int i = 2; i < 6; ++i) {
                            if (!IsHexDigit(p_[i])) {
                                return false;
                            }
                        }
                        p_ += 6;
                        continue;
                    }
                    if (esc != '"' && esc != '\\' && esc != '/' && esc != 'b' && esc != 'f' && esc != 'n' && esc != 'r'
                        && esc != 't') {
                        return false;
                    }
                    p_ += 2;
                    continue;
                }
                if (ch < 0x20) {
                    return false;
                }
                if (ch == '"') {
                    p_++;
                    return true;
                }
                p_++;
            }
            return false;
        }

        static bool IsHexDigit(char c)
        {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        }

        bool MatchLiteral(const char* literal, size_t len)
        {
            if (p_ > end_ || static_cast<size_t>(end_ - p_) < len || std::memcmp(p_, literal, len) != 0) {
                return false;
            }
            p_ += len;
            return true;
        }

        bool SkipJsonObject(int depth)
        {
            p_++;
            while (true) {
                SkipWs();
                if (AtEnd()) {
                    return false;
                }
                if (*p_ == '}') {
                    p_++;
                    return true;
                }
                std::string key;
                if (!ParseString(key) || !TryConsume(':') || !SkipValue(depth + 1)) {
                    return false;
                }
                if (!CheckSeparator('}')) {
                    return false;
                }
            }
        }

        bool SkipJsonArray(int depth)
        {
            p_++;
            while (true) {
                SkipWs();
                if (AtEnd()) {
                    return false;
                }
                if (*p_ == ']') {
                    p_++;
                    return true;
                }
                if (!SkipValue(depth + 1) || !CheckSeparator(']')) {
                    return false;
                }
            }
        }

        // 已知与 DOM 的有意分歧: 未知字段值在此只做词法扫描, 不做数值量级校验,
        // 如 1E6666 这类超 double 量级字面量 fast 接受而 nlohmann 对整文档 fail-fast;
        // 取舍: 未知字段不影响解析结果, 不为其引入整文档失败语义, 已知字段仍走严格 ParseU32。
        bool SkipJsonNumber()
        {
            const char* start = p_;
            if (*p_ == '-') {
                ++p_;
            }
            if (p_ >= end_) {
                return false;
            }
            if (*p_ == '0') {
                ++p_;
            } else if (*p_ >= '1' && *p_ <= '9') {
                while (p_ < end_ && *p_ >= '0' && *p_ <= '9') {
                    ++p_;
                }
            } else {
                return false;
            }
            if (p_ < end_ && *p_ == '.') {
                ++p_;
                if (p_ >= end_ || *p_ < '0' || *p_ > '9') {
                    return false;
                }
                while (p_ < end_ && *p_ >= '0' && *p_ <= '9') {
                    ++p_;
                }
            }
            if (p_ < end_ && (*p_ == 'e' || *p_ == 'E')) {
                ++p_;
                if (p_ < end_ && (*p_ == '+' || *p_ == '-')) {
                    ++p_;
                }
                if (p_ >= end_ || *p_ < '0' || *p_ > '9') {
                    return false;
                }
                while (p_ < end_ && *p_ >= '0' && *p_ <= '9') {
                    ++p_;
                }
            }
            return p_ > start;
        }

    private:
        const char* p_;
        const char* end_;
    };

    bool ParseAddrByTypeStr(const std::string& addrType, const std::string& address, IpAddress& ipAddress)
    {
        if (addrType == "IPV4") {
            if (!IpAddress::IsIPv4(address)) {
                HCCL_ERROR("[%s] invalid IPV4 addr[%s]", __func__, address.c_str());
                return false;
            }
            ipAddress = IpAddress(address, AF_INET);
            return true;
        }
        if (addrType == "IPV6") {
            if (!IpAddress::IsIPv6(address)) {
                HCCL_ERROR("[%s] invalid IPV6 addr[%s]", __func__, address.c_str());
                return false;
            }
            ipAddress = IpAddress(address, AF_INET6);
            return true;
        }
        if (addrType == "EID") {
            if (address.empty() || !IpAddress::IsEID(address)) {
                HCCL_ERROR("[%s] invalid EID addr[%s]", __func__, address.c_str());
                return false;
            }
            Eid eid = IpAddress::StrToEID(address);
            ipAddress = IpAddress(eid);
            return true;
        }
        HCCL_ERROR("[%s] invalid addrType[%s]", __func__, addrType.c_str());
        return false;
    }

    bool IsValidNetType(const std::string& str)
    {
        static const std::set<std::string> NET_TYPES
            = {"1DMESH", "2DMESH", "A3_SERVER", "A2_AX_SERVER", "TOPO_FILE_DESC", "CLOS"};
        return NET_TYPES.count(str) > 0;
    }

    NetType StrToNetType(const std::string& str)
    {
        if (str == "1DMESH") {
            return NetType::MESH_1D;
        }
        if (str == "2DMESH") {
            return NetType::MESH_2D;
        }
        if (str == "A3_SERVER") {
            return NetType::A3_SERVER;
        }
        if (str == "A2_AX_SERVER") {
            return NetType::A2_AX_SERVER;
        }
        if (str == "TOPO_FILE_DESC") {
            return NetType::TOPO_FILE_DESC;
        }
        return NetType::CLOS;
    }

    // 未知字段：解析键名后整体跳过值
    HcclResult SkipUnknownField(JsonCursor& cur)
    {
        std::string key;
        if (!cur.ParseString(key) || !cur.TryConsume(':')) {
            return HCCL_E_PARA;
        }
        return cur.SkipValue() ? HCCL_SUCCESS : HCCL_E_PARA;
    }

    bool SetAddrType(const std::string& addrType, AddrType& out)
    {
        if (addrType == "EID") {
            out = AddrType::EID;
            return true;
        }
        if (addrType == "IPV4") {
            out = AddrType::IPV4;
            return true;
        }
        if (addrType == "IPV6") {
            out = AddrType::IPV6;
            return true;
        }
        return false;
    }

    struct AddressFieldState {
        std::string addrType;
        std::string address;
        std::vector<std::string> backupAddrList;
        bool hasAddr{false};
        bool hasAddrType{false};
        bool hasPorts{false};
        bool hasBackupAddr{false};
    };

    bool ParsePortsField(JsonCursor& cur, AddressInfo& addrInfo)
    {
        if (!cur.TryConsume(':') || !cur.TryConsume('[')) {
            return false;
        }
        while (!cur.Is(']')) {
            if (cur.AtEnd()) {
                return false;
            }
            std::string port;
            if (!cur.ParseString(port)) {
                return false;
            }
            if (port.length() < MIN_VALUE_PORT_LENGTH || port.length() > MAX_VALUE_PORT_LENGTH) {
                ReportFastFieldInvalid(port, kRankTableKeys[RankTableKey::kPorts].name, "valid port length");
                return false;
            }
            addrInfo.ports.emplace(std::move(port));
            if (!cur.CheckSeparator(']')) {
                return false;
            }
        }
        if (!cur.TryConsume(']')) {
            return false;
        }
        if (addrInfo.ports.size() < MIN_VALUE_PORT || addrInfo.ports.size() > MAX_VALUE_PORT) {
            ReportFastFieldInvalid(
                std::to_string(addrInfo.ports.size()), kRankTableKeys[RankTableKey::kPorts].name, "valid ports count");
            return false;
        }
        return true;
    }

    bool ParseBackupAddrListField(JsonCursor& cur, std::vector<std::string>& backupAddrList)
    {
        if (!cur.TryConsume(':') || !cur.TryConsume('[')) {
            return false;
        }
        while (!cur.Is(']')) {
            if (cur.AtEnd()) {
                return false;
            }
            std::string backupAddr;
            if (!cur.ParseString(backupAddr)) {
                return false;
            }
            if (backupAddrList.size() >= MAX_VALUE_BACKUP_ADDR_SIZE) {
                ReportFastFieldInvalid(
                    backupAddr, kRankTableKeys[RankTableKey::kBackupAddr].name,
                    std::string("backup_addr count within ") + std::to_string(MAX_VALUE_BACKUP_ADDR_SIZE));
                return false;
            }
            backupAddrList.emplace_back(std::move(backupAddr));
            if (!cur.CheckSeparator(']')) {
                return false;
            }
        }
        return cur.TryConsume(']');
    }

    HcclResult ParseAddressScalarField(JsonCursor& cur, AddressInfo& addrInfo, AddressFieldState& state, bool& handled)
    {
        handled = true;
        if (cur.ParseKey(RankTableKey::kAddr)) {
            if (!cur.TryConsume(':') || !cur.ParseString(state.address)) {
                return HCCL_E_PARA;
            }
            state.hasAddr = true;
        } else if (cur.ParseKey(RankTableKey::kAddrType)) {
            if (!cur.TryConsume(':') || !cur.ParseString(state.addrType)) {
                return HCCL_E_PARA;
            }
            state.hasAddrType = true;
        } else if (cur.ParseKey(RankTableKey::kPlaneId)) {
            if (!cur.TryConsume(':') || !cur.ParseString(addrInfo.planeId)) {
                return HCCL_E_PARA;
            }
            if (addrInfo.planeId.length() > MAX_VALUE_PLANEID) {
                ReportFastFieldInvalid(
                    addrInfo.planeId, kRankTableKeys[RankTableKey::kPlaneId].name, "valid plane_id length");
                return HCCL_E_PARA;
            }
        } else {
            handled = false;
        }
        return HCCL_SUCCESS;
    }

    HcclResult ParseAddressObject(JsonCursor& cur, AddressInfo& addrInfo, AddressFieldState& state)
    {
        while (!cur.Is('}')) {
            if (cur.AtEnd()) {
                return HCCL_E_PARA;
            }
            bool handled = false;
            CHK_RET(ParseAddressScalarField(cur, addrInfo, state, handled));
            if (!handled && cur.ParseKey(RankTableKey::kPorts)) {
                if (state.hasPorts) {
                    addrInfo.ports.clear(); // 重复 ports 采用 last-wins, 与 nlohmann 一致
                }
                if (!ParsePortsField(cur, addrInfo)) {
                    return HCCL_E_PARA;
                }
                state.hasPorts = true;
                handled = true;
            }
            if (!handled && cur.ParseKey(RankTableKey::kBackupAddr)) {
                if (state.hasBackupAddr) {
                    state.backupAddrList.clear(); // 重复 backup_addr 采用 last-wins, 与 nlohmann 一致
                }
                if (!ParseBackupAddrListField(cur, state.backupAddrList)) {
                    return HCCL_E_PARA;
                }
                state.hasBackupAddr = true;
                handled = true;
            }
            if (!handled && cur.ParseKey(RankTableKey::kBackupAddrList)) {
                // 兼容旧数据：DOM 只识别 backup_addr，该键仅跳过
                if (!cur.TryConsume(':') || !cur.SkipValue()) {
                    return HCCL_E_PARA;
                }
                handled = true;
            }
            if (!handled) {
                CHK_RET(SkipUnknownField(cur));
            }
            if (!cur.CheckSeparator('}')) {
                return HCCL_E_PARA;
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult FinalizeAddressInfo(AddressInfo& addrInfo, AddressFieldState& state)
    {
        if (!state.hasAddr) {
            ReportFastFieldMissing(kRankTableKeys[RankTableKey::kAddr].name);
            return HCCL_E_PARA;
        }
        if (!state.hasAddrType) {
            ReportFastFieldMissing(kRankTableKeys[RankTableKey::kAddrType].name);
            return HCCL_E_PARA;
        }
        if (!state.hasPorts) {
            ReportFastFieldMissing(kRankTableKeys[RankTableKey::kPorts].name);
            return HCCL_E_PARA;
        }
        if (state.address.length() < MIN_VALUE_ADDR_LENGRH || state.address.length() > MAX_VALUE_ADDR_LENGRH) {
            ReportFastFieldInvalid(state.address, kRankTableKeys[RankTableKey::kAddr].name, "valid addr length");
            return HCCL_E_PARA;
        }
        if (!ParseAddrByTypeStr(state.addrType, state.address, addrInfo.addr)) {
            ReportFastFieldInvalid(state.address, kRankTableKeys[RankTableKey::kAddr].name, "valid addr for addr_type");
            return HCCL_E_PARA;
        }
        if (!SetAddrType(state.addrType, addrInfo.addrType)) {
            ReportFastFieldInvalid(state.addrType, kRankTableKeys[RankTableKey::kAddrType].name, "IPV4/IPV6/EID");
            return HCCL_E_PARA;
        }
        addrInfo.backupAddrs.reserve(state.backupAddrList.size());
        for (const auto& backupAddr : state.backupAddrList) {
            if (backupAddr.length() < MIN_VALUE_ADDR_LENGRH || backupAddr.length() > MAX_VALUE_ADDR_LENGRH) {
                ReportFastFieldInvalid(
                    backupAddr, kRankTableKeys[RankTableKey::kBackupAddr].name, "valid backup_addr length");
                return HCCL_E_PARA;
            }
            IpAddress backupIp;
            if (!ParseAddrByTypeStr(state.addrType, backupAddr, backupIp)) {
                ReportFastFieldInvalid(
                    backupAddr, kRankTableKeys[RankTableKey::kBackupAddr].name, "valid backup_addr for addr_type");
                return HCCL_E_PARA;
            }
            addrInfo.backupAddrs.emplace_back(std::move(backupIp));
        }
        return HCCL_SUCCESS;
    }

    // 解析rank_addr_list数组，填充RankLevelInfo::rankAddrs
    HcclResult ParseAddrList(JsonCursor& cur, RankLevelInfo& levelInfo)
    {
        if (!cur.TryConsume('[')) {
            return HCCL_E_PARA;
        }
        while (!cur.Is(']')) {
            if (cur.AtEnd() || !cur.TryConsume('{')) {
                return HCCL_E_PARA;
            }
            AddressInfo addrInfo;
            AddressFieldState state;
            CHK_RET(ParseAddressObject(cur, addrInfo, state));
            if (!cur.TryConsume('}')) {
                return HCCL_E_PARA;
            }
            CHK_RET(FinalizeAddressInfo(addrInfo, state));
            levelInfo.rankAddrs.emplace_back(std::move(addrInfo));
            if (!cur.CheckSeparator(']')) {
                return HCCL_E_PARA;
            }
        }
        if (!cur.TryConsume(']')) {
            return HCCL_E_PARA;
        }
        if (levelInfo.rankAddrs.size() > MAX_VALUE_RANKADDR_SIZE) {
            ReportFastFieldInvalid(
                std::to_string(levelInfo.rankAddrs.size()), kRankTableKeys[RankTableKey::kRankAddrList].name,
                "valid rank_addr_list count");
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    struct LevelFieldState {
        bool hasNetLayer{false};
        bool hasNetInstId{false};
        bool hasRankAddrList{false};
    };

    HcclResult ParseLevelScalarField(JsonCursor& cur, RankLevelInfo& levelInfo, LevelFieldState& state, bool& handled)
    {
        handled = true;
        if (cur.ParseKey(RankTableKey::kNetLayer)) {
            if (!cur.TryConsume(':') || !cur.ParseU32(levelInfo.netLayer)) {
                return HCCL_E_PARA;
            }
            if (levelInfo.netLayer > MAX_VALUE_NETLAYER) {
                ReportFastFieldInvalid(
                    std::to_string(levelInfo.netLayer), kRankTableKeys[RankTableKey::kNetLayer].name,
                    std::string("net_layer within ") + std::to_string(MAX_VALUE_NETLAYER));
                return HCCL_E_PARA;
            }
            state.hasNetLayer = true;
        } else if (cur.ParseKey(RankTableKey::kNetInstanceId)) {
            if (!cur.TryConsume(':') || !cur.ParseString(levelInfo.netInstId)) {
                return HCCL_E_PARA;
            }
            if (levelInfo.netInstId.length() < MIN_VALUE_NETID || levelInfo.netInstId.length() > MAX_VALUE_NETID) {
                ReportFastFieldInvalid(
                    levelInfo.netInstId, kRankTableKeys[RankTableKey::kNetInstanceId].name,
                    "valid net_instance_id length");
                return HCCL_E_PARA;
            }
            state.hasNetInstId = true;
        } else if (cur.ParseKey(RankTableKey::kNetType)) {
            if (!cur.TryConsume(':')) {
                return HCCL_E_PARA;
            }
            std::string netTypeStr;
            if (!cur.ParseString(netTypeStr)) {
                return HCCL_E_PARA;
            }
            if (!IsValidNetType(netTypeStr)) {
                ReportFastFieldInvalid(
                    netTypeStr, kRankTableKeys[RankTableKey::kNetType].name,
                    "1DMESH/2DMESH/A3_SERVER/A2_AX_SERVER/TOPO_FILE_DESC/CLOS");
                return HCCL_E_PARA;
            }
            levelInfo.netType = StrToNetType(netTypeStr);
        } else if (cur.ParseKey(RankTableKey::kNetAttr)) {
            if (!cur.TryConsume(':') || !cur.ParseString(levelInfo.netAttr)) {
                return HCCL_E_PARA;
            }
        } else if (cur.ParseKey(RankTableKey::kPcieFallback)) {
            if (!cur.TryConsume(':') || !cur.ParseBool(levelInfo.pcieFallback)) {
                return HCCL_E_PARA;
            }
        } else {
            handled = false;
        }
        return HCCL_SUCCESS;
    }

    HcclResult ParseLevelAddrField(JsonCursor& cur, RankLevelInfo& levelInfo, LevelFieldState& state, bool& handled)
    {
        handled = cur.ParseKey(RankTableKey::kRankAddrList);
        if (!handled) {
            return HCCL_SUCCESS;
        }
        if (!cur.TryConsume(':')) {
            return HCCL_E_PARA;
        }
        if (state.hasRankAddrList) {
            levelInfo.rankAddrs.clear(); // 重复 rank_addr_list 采用 last-wins, 与 nlohmann 一致
        }
        CHK_RET(ParseAddrList(cur, levelInfo));
        state.hasRankAddrList = true;
        return HCCL_SUCCESS;
    }

    HcclResult ParseLevelObject(JsonCursor& cur, RankLevelInfo& levelInfo, LevelFieldState& state)
    {
        while (!cur.Is('}')) {
            if (cur.AtEnd()) {
                return HCCL_E_PARA;
            }
            bool handled = false;
            CHK_RET(ParseLevelScalarField(cur, levelInfo, state, handled));
            if (!handled) {
                CHK_RET(ParseLevelAddrField(cur, levelInfo, state, handled));
            }
            if (!handled) {
                CHK_RET(SkipUnknownField(cur));
            }
            if (!cur.CheckSeparator('}')) {
                return HCCL_E_PARA;
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult FinalizeLevelInfo(RankLevelInfo& levelInfo, const LevelFieldState& state)
    {
        if (!state.hasNetInstId) {
            ReportFastFieldMissing(kRankTableKeys[RankTableKey::kNetInstanceId].name);
            return HCCL_E_PARA;
        }
        if (!state.hasNetLayer) {
            ReportFastFieldMissing(kRankTableKeys[RankTableKey::kNetLayer].name);
            return HCCL_E_PARA;
        }
        if (!state.hasRankAddrList) {
            ReportFastFieldMissing(kRankTableKeys[RankTableKey::kRankAddrList].name);
            return HCCL_E_PARA;
        }
        // 与RankLevelInfo::Deserialize一致：构建port->addr映射（level0链路构建依赖portAddrMap）
        for (const auto& rankAddr : levelInfo.rankAddrs) {
            const IpAddress& ipAddress = rankAddr.addr;
            for (const auto& port : rankAddr.ports) {
                levelInfo.portAddrMap[port].push_back(ipAddress);
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult ValidateLevelOrder(const NewRankInfo& rankInfo)
    {
        // 校验level按netLayer递增（与DeserializeLevelList语义一致）
        for (size_t i = 1; i < rankInfo.rankLevelInfos.size(); i++) {
            if (rankInfo.rankLevelInfos[i].netLayer <= rankInfo.rankLevelInfos[i - 1].netLayer) {
                HCCL_ERROR("[%s] level is not increased in sequence", __func__);
                return HCCL_E_PARA;
            }
        }
        return HCCL_SUCCESS;
    }

    // 解析level_list数组，填充NewRankInfo::rankLevelInfos
    HcclResult ParseLevelList(JsonCursor& cur, NewRankInfo& rankInfo)
    {
        if (!cur.TryConsume('[')) {
            return HCCL_E_PARA;
        }
        while (!cur.Is(']')) {
            if (cur.AtEnd() || !cur.TryConsume('{')) {
                return HCCL_E_PARA;
            }
            RankLevelInfo levelInfo;
            levelInfo.rankAddrs.reserve(1); // 常见每层1个地址，避免按8个过度预留
            LevelFieldState state;
            CHK_RET(ParseLevelObject(cur, levelInfo, state));
            if (!cur.TryConsume('}')) {
                return HCCL_E_PARA;
            }
            CHK_RET(FinalizeLevelInfo(levelInfo, state));
            rankInfo.rankLevelInfos.emplace_back(std::move(levelInfo));
            if (!cur.CheckSeparator(']')) {
                return HCCL_E_PARA;
            }
        }
        if (!cur.TryConsume(']')) {
            return HCCL_E_PARA;
        }
        if (rankInfo.rankLevelInfos.size() > MAX_LEVEL_lIST) {
            ReportFastFieldInvalid(
                std::to_string(rankInfo.rankLevelInfos.size()), kRankTableKeys[RankTableKey::kLevelList].name,
                "valid level_list count");
            return HCCL_E_PARA;
        }
        return ValidateLevelOrder(rankInfo);
    }

    struct ControlPlaneFieldState {
        std::string addrType;
        std::string address;
        bool hasAddrType{false};
        bool hasAddr{false};
        bool hasListenPort{false};
    };

    HcclResult
    ParseControlPlaneField(JsonCursor& cur, NewRankInfo& rankInfo, ControlPlaneFieldState& state, bool& handled)
    {
        handled = true;
        if (cur.ParseKey(RankTableKey::kAddrType)) {
            if (!cur.TryConsume(':') || !cur.ParseString(state.addrType)) {
                return HCCL_E_PARA;
            }
            state.hasAddrType = true;
        } else if (cur.ParseKey(RankTableKey::kAddr)) {
            if (!cur.TryConsume(':') || !cur.ParseString(state.address)) {
                return HCCL_E_PARA;
            }
            state.hasAddr = true;
        } else if (cur.ParseKey(RankTableKey::kListenPort)) {
            if (!cur.TryConsume(':') || !cur.ParseU32(rankInfo.controlPlane.listenPort)) {
                return HCCL_E_PARA;
            }
            state.hasListenPort = true;
        } else {
            handled = false;
        }
        return HCCL_SUCCESS;
    }

    HcclResult FinalizeControlPlane(NewRankInfo& rankInfo, const ControlPlaneFieldState& state)
    {
        if (!state.hasAddrType) {
            ReportFastFieldMissing(kRankTableKeys[RankTableKey::kAddrType].name);
            return HCCL_E_PARA;
        }
        if (!state.hasAddr) {
            ReportFastFieldMissing(kRankTableKeys[RankTableKey::kAddr].name);
            return HCCL_E_PARA;
        }
        if (!state.hasListenPort) {
            ReportFastFieldMissing(kRankTableKeys[RankTableKey::kListenPort].name);
            return HCCL_E_PARA;
        }
        if (state.address.length() < MIN_VALUE_ADDR || state.address.length() > MAX_VALUE_ADDR) {
            ReportFastFieldInvalid(
                state.address, kRankTableKeys[RankTableKey::kAddr].name, "valid control_plane addr length");
            return HCCL_E_PARA;
        }
        if (rankInfo.controlPlane.listenPort < MIN_VALUE_LISTENPORT
            || rankInfo.controlPlane.listenPort > MAX_VALUE_LISTENPORT) {
            ReportFastFieldInvalid(
                std::to_string(rankInfo.controlPlane.listenPort), kRankTableKeys[RankTableKey::kListenPort].name,
                "valid listen_port range");
            return HCCL_E_PARA;
        }
        if (!ParseAddrByTypeStr(state.addrType, state.address, rankInfo.controlPlane.addr)) {
            ReportFastFieldInvalid(
                state.address, kRankTableKeys[RankTableKey::kAddr].name, "valid control_plane addr for addr_type");
            return HCCL_E_PARA;
        }
        if (!SetAddrType(state.addrType, rankInfo.controlPlane.addrType)) {
            ReportFastFieldInvalid(state.addrType, kRankTableKeys[RankTableKey::kAddrType].name, "IPV4/IPV6/EID");
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    // 解析control_plane对象，填充NewRankInfo::controlPlane
    HcclResult ParseControlPlane(JsonCursor& cur, NewRankInfo& rankInfo)
    {
        if (!cur.TryConsume('{')) {
            return HCCL_E_PARA;
        }
        ControlPlaneFieldState state;
        while (!cur.Is('}')) {
            if (cur.AtEnd()) {
                return HCCL_E_PARA;
            }
            bool handled = false;
            CHK_RET(ParseControlPlaneField(cur, rankInfo, state, handled));
            if (!handled) {
                CHK_RET(SkipUnknownField(cur));
            }
            if (!cur.CheckSeparator('}')) {
                return HCCL_E_PARA;
            }
        }
        if (!cur.TryConsume('}')) {
            return HCCL_E_PARA;
        }
        return FinalizeControlPlane(rankInfo, state);
    }

    struct RankFieldState {
        bool hasRankId{false};
        bool hasLocalId{false};
        bool hasDeviceId{false};
        bool hasReplacedLocalId{false};
        bool hasLevelList{false};
    };

    HcclResult ParseRankIdentityField(JsonCursor& cur, NewRankInfo& rankInfo, RankFieldState& state, bool& handled)
    {
        handled = true;
        if (cur.ParseKey(RankTableKey::kRankId)) {
            if (!cur.TryConsume(':') || !cur.ParseU32(rankInfo.rankId)) {
                return HCCL_E_PARA;
            }
            state.hasRankId = true;
        } else if (cur.ParseKey(RankTableKey::kLocalId)) {
            if (!cur.TryConsume(':') || !cur.ParseU32(rankInfo.localId)) {
                return HCCL_E_PARA;
            }
            if (rankInfo.localId > BACKUP_LOCAL_ID) {
                ReportFastFieldInvalid(
                    std::to_string(rankInfo.localId), kRankTableKeys[RankTableKey::kLocalId].name,
                    std::string("local_id within ") + std::to_string(BACKUP_LOCAL_ID));
                return HCCL_E_PARA;
            }
            state.hasLocalId = true;
        } else if (cur.ParseKey(RankTableKey::kReplacedLocalId)) {
            if (!cur.TryConsume(':') || !cur.ParseU32(rankInfo.replacedLocalId)) {
                return HCCL_E_PARA;
            }
            if (rankInfo.replacedLocalId > BACKUP_LOCAL_ID - 1) {
                ReportFastFieldInvalid(
                    std::to_string(rankInfo.replacedLocalId), kRankTableKeys[RankTableKey::kReplacedLocalId].name,
                    std::string("replaced_local_id within ") + std::to_string(BACKUP_LOCAL_ID - 1));
                return HCCL_E_PARA;
            }
            state.hasReplacedLocalId = true;
        } else if (cur.ParseKey(RankTableKey::kDeviceId)) {
            if (!cur.TryConsume(':') || !cur.ParseU32(rankInfo.deviceId)) {
                return HCCL_E_PARA;
            }
            if (rankInfo.deviceId > MAX_VALUE_DEVICEID) {
                ReportFastFieldInvalid(
                    std::to_string(rankInfo.deviceId), kRankTableKeys[RankTableKey::kDeviceId].name,
                    std::string("device_id within ") + std::to_string(MAX_VALUE_DEVICEID));
                return HCCL_E_PARA;
            }
            state.hasDeviceId = true;
        } else {
            handled = false;
        }
        return HCCL_SUCCESS;
    }

    HcclResult ParseRankPortField(JsonCursor& cur, NewRankInfo& rankInfo, bool& handled)
    {
        handled = true;
        if (cur.ParseKey(RankTableKey::kDevicePort)) {
            if (!cur.TryConsume(':') || !cur.ParseU32(rankInfo.devicePort)) {
                return HCCL_E_PARA;
            }
        } else if (cur.ParseKey(RankTableKey::kHostPort)) {
            if (!cur.TryConsume(':') || !cur.ParseU32(rankInfo.hostPort)) {
                return HCCL_E_PARA;
            }
        } else {
            handled = false;
        }
        return HCCL_SUCCESS;
    }

    HcclResult ParseRankLevelField(JsonCursor& cur, NewRankInfo& rankInfo, RankFieldState& state, bool& handled)
    {
        handled = cur.ParseKey(RankTableKey::kLevelList);
        if (!handled) {
            return HCCL_SUCCESS;
        }
        if (!cur.TryConsume(':')) {
            return HCCL_E_PARA;
        }
        if (state.hasLevelList) {
            rankInfo.rankLevelInfos.clear(); // 重复 level_list 采用 last-wins, 与 nlohmann 一致
        }
        CHK_RET(ParseLevelList(cur, rankInfo));
        state.hasLevelList = true;
        return HCCL_SUCCESS;
    }

    HcclResult ParseRankControlPlaneField(JsonCursor& cur, NewRankInfo& rankInfo, bool& handled)
    {
        handled = cur.ParseKey(RankTableKey::kControlPlane);
        if (!handled) {
            return HCCL_SUCCESS;
        }
        if (!cur.TryConsume(':')) {
            return HCCL_E_PARA;
        }
        return ParseControlPlane(cur, rankInfo);
    }

    HcclResult ParseRankObject(JsonCursor& cur, NewRankInfo& rankInfo, RankFieldState& state)
    {
        while (!cur.Is('}')) {
            if (cur.AtEnd()) {
                return HCCL_E_PARA;
            }
            bool handled = false;
            CHK_RET(ParseRankIdentityField(cur, rankInfo, state, handled));
            if (!handled) {
                CHK_RET(ParseRankPortField(cur, rankInfo, handled));
            }
            if (!handled) {
                CHK_RET(ParseRankLevelField(cur, rankInfo, state, handled));
            }
            if (!handled) {
                CHK_RET(ParseRankControlPlaneField(cur, rankInfo, handled));
            }
            if (!handled) {
                CHK_RET(SkipUnknownField(cur));
            }
            if (!cur.CheckSeparator('}')) {
                return HCCL_E_PARA;
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult FinalizeRankInfo(NewRankInfo& rankInfo, const RankFieldState& state)
    {
        if (!state.hasRankId) {
            ReportFastFieldMissing(kRankTableKeys[RankTableKey::kRankId].name);
            return HCCL_E_PARA;
        }
        if (!state.hasLocalId) {
            ReportFastFieldMissing(kRankTableKeys[RankTableKey::kLocalId].name);
            return HCCL_E_PARA;
        }
        if (!state.hasDeviceId) {
            ReportFastFieldMissing(kRankTableKeys[RankTableKey::kDeviceId].name);
            return HCCL_E_PARA;
        }
        if (!state.hasLevelList) {
            ReportFastFieldMissing(kRankTableKeys[RankTableKey::kLevelList].name);
            return HCCL_E_PARA;
        }
        if (rankInfo.localId == BACKUP_LOCAL_ID) {
            // 备份rank必须有显式replaced_local_id（与DeserializeRankIdAndLocalId语义一致）
            if (!state.hasReplacedLocalId) {
                ReportFastFieldMissing(kRankTableKeys[RankTableKey::kReplacedLocalId].name);
                return HCCL_E_PARA;
            }
        } else {
            rankInfo.replacedLocalId = rankInfo.localId;
        }
        if (rankInfo.devicePort > MAX_VALUE_TCPPORT || rankInfo.devicePort < MIN_VALUE_TCPPORT) {
            ReportFastFieldInvalid(
                std::to_string(rankInfo.devicePort), kRankTableKeys[RankTableKey::kDevicePort].name,
                "valid device_port range");
            return HCCL_E_PARA;
        }
        if (rankInfo.hostPort > MAX_VALUE_TCPPORT || rankInfo.hostPort < MIN_VALUE_TCPPORT) {
            ReportFastFieldInvalid(
                std::to_string(rankInfo.hostPort), kRankTableKeys[RankTableKey::kHostPort].name,
                "valid host_port range");
            return HCCL_E_PARA;
        }
        // 与DeserializeLevelList语义一致：所有addr的socket监听端口统一取devicePort
        for (auto& levelInfo : rankInfo.rankLevelInfos) {
            for (auto& addrInfo : levelInfo.rankAddrs) {
                addrInfo.socketPort_ = rankInfo.devicePort;
            }
        }
        return HCCL_SUCCESS;
    }

    // 与 RankTableInfo::Deserialize 保持一致：ranks[rankId] 是下游建图的索引前提
    void SortRankTableRanks(RankTableInfo& info)
    {
        const auto rankIdLess = [](const NewRankInfo& lhs, const NewRankInfo& rhs) {
            return lhs.rankId < rhs.rankId;
        };
        if (!std::is_sorted(info.ranks.begin(), info.ranks.end(), rankIdLess)) {
            std::sort(info.ranks.begin(), info.ranks.end(), rankIdLess);
        }
    }

    // 解析rank_list中的单个rank对象
    HcclResult ParseOneRank(JsonCursor& cur, NewRankInfo& rankInfo)
    {
        if (!cur.TryConsume('{')) {
            return HCCL_E_PARA;
        }
        RankFieldState state;
        CHK_RET(ParseRankObject(cur, rankInfo, state));
        if (!cur.TryConsume('}')) {
            return HCCL_E_PARA;
        }
        return FinalizeRankInfo(rankInfo, state);
    }

    // 解析rank_list数组
    HcclResult ParseRankList(JsonCursor& cur, std::vector<NewRankInfo>& ranks, u32 reserveCount, size_t totalBytes)
    {
        if (!cur.TryConsume('[')) {
            return HCCL_E_PARA;
        }
        if (reserveCount > 0) {
            ranks.reserve(SafeReserveCount(reserveCount, totalBytes));
        }
        while (!cur.Is(']')) {
            if (cur.AtEnd()) {
                return HCCL_E_PARA;
            }
            NewRankInfo rankInfo;
            rankInfo.rankLevelInfos.reserve(2); // 常见为2层，避免过度预留
            CHK_RET(ParseOneRank(cur, rankInfo));
            ranks.emplace_back(std::move(rankInfo));
            if (!cur.CheckSeparator(']')) {
                return HCCL_E_PARA;
            }
        }
        if (!cur.TryConsume(']')) {
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    // ---- 并行解析辅助: 结构扫描定位 rank 元素区间 + 多线程分块 ----

    struct RankElemRange {
        size_t begin{0};
        size_t end{0};
    };

    struct RankListLayout {
        std::string version;
        u64 rankCount{0};
        bool detour{false};
        bool hasVersion{false};
        bool hasRankCount{false};
        bool hasRankList{false};
        u64 elementCount{0};
        std::vector<RankElemRange> chunks; // 连续 rank 元素组成的 chunk，起点/终点都落在对象边界
    };

    inline bool IsJsonWs(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

    // 字符串/转义感知的结构扫描: 只定位顶层字段与 rank_list 元素字节区间
    struct LayoutScanner {
        const std::string& text;
        const char* data;
        const char* end;
        const char* p;
        RankListLayout& layout;

        LayoutScanner(const std::string& input, RankListLayout& output)
            : text(input),
              data(input.data()),
              end(input.data() + input.size()),
              p(input.data()),
              layout(output)
        {}

        void SkipWs()
        {
            while (p < end && IsJsonWs(*p)) {
                ++p;
            }
        }

        bool Consume(char expected)
        {
            SkipWs();
            if (p < end && *p == expected) {
                ++p;
                return true;
            }
            return false;
        }

        static bool IsHexDigit(char c)
        {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        }

        bool ScanString(const char*& strBegin, size_t& strLen)
        {
            SkipWs();
            if (p >= end || *p != '"') {
                return false;
            }
            ++p;
            strBegin = p;
            while (p < end) {
                const char* quote = FindQuoteOrBackslash(p, end);
                if (quote == end) {
                    return false;
                }
                for (const char* s = p; s < quote; ++s) {
                    if (static_cast<unsigned char>(*s) < 0x20) {
                        return false;
                    }
                }
                if (*quote == '"') {
                    strLen = static_cast<size_t>(quote - strBegin);
                    p = quote + 1;
                    return true;
                }
                if (end - quote < 2) {
                    return false;
                }
                const char esc = quote[1];
                if (esc == 'u') {
                    if (end - quote < 6) {
                        return false;
                    }
                    for (int i = 2; i < 6; ++i) {
                        if (!IsHexDigit(quote[i])) {
                            return false;
                        }
                    }
                    p = quote + 6;
                } else {
                    if (esc != '"' && esc != '\\' && esc != '/' && esc != 'b' && esc != 'f' && esc != 'n' && esc != 'r'
                        && esc != 't') {
                        return false;
                    }
                    p = quote + 2;
                }
            }
            return false;
        }

        bool SkipString()
        {
            const char* begin = nullptr;
            size_t len = 0;
            return ScanString(begin, len);
        }

        bool KeyIs(const char* begin, size_t len, RankTableKey key)
        {
            return len == kRankTableKeys[key].len && std::memcmp(begin, kRankTableKeys[key].name, len) == 0;
        }

        bool SkipNumber()
        {
            const char* numberBegin = p;
            if (*p == '-') {
                ++p;
            }
            if (p >= end) {
                return false;
            }
            if (*p == '0') {
                ++p;
            } else if (*p >= '1' && *p <= '9') {
                while (p < end && *p >= '0' && *p <= '9') {
                    ++p;
                }
            } else {
                return false;
            }
            if (p < end && *p == '.') {
                ++p;
                if (p >= end || *p < '0' || *p > '9') {
                    return false;
                }
                while (p < end && *p >= '0' && *p <= '9') {
                    ++p;
                }
            }
            if (p < end && (*p == 'e' || *p == 'E')) {
                ++p;
                if (p < end && (*p == '+' || *p == '-')) {
                    ++p;
                }
                if (p >= end || *p < '0' || *p > '9') {
                    return false;
                }
                while (p < end && *p >= '0' && *p <= '9') {
                    ++p;
                }
            }
            return p > numberBegin;
        }

        bool SkipObject(int depth)
        {
            ++p;
            if (Consume('}')) {
                return true;
            }
            while (true) {
                if (!SkipString() || !Consume(':')) {
                    return false;
                }
                if (!SkipValue(depth + 1)) {
                    return false;
                }
                if (Consume(',')) {
                    continue;
                }
                return Consume('}');
            }
        }

        bool SkipArray(int depth)
        {
            ++p;
            if (Consume(']')) {
                return true;
            }
            while (true) {
                if (!SkipValue(depth + 1)) {
                    return false;
                }
                if (Consume(',')) {
                    continue;
                }
                return Consume(']');
            }
        }

        bool SkipValue(int depth)
        {
            if (depth > kMaxJsonDepth) {
                return false;
            }
            SkipWs();
            if (p >= end) {
                return false;
            }
            const char c = *p;
            if (c == '"') {
                return SkipString();
            }
            if (c == '{') {
                return SkipObject(depth);
            }
            if (c == '[') {
                return SkipArray(depth);
            }
            if (c == 't') {
                return MatchLiteral("true", 4);
            }
            if (c == 'f') {
                return MatchLiteral("false", 5);
            }
            if (c == 'n') {
                return MatchLiteral("null", 4);
            }
            return SkipNumber();
        }

        bool MatchLiteral(const char* literal, size_t len)
        {
            if (static_cast<size_t>(end - p) < len || std::memcmp(p, literal, len) != 0) {
                return false;
            }
            p += len;
            return true;
        }

        // 逐字节状态机: 非 SIMD 架构的回退路径, 亦是混合路径的语义基准
        bool ScanOneRankElementScalar(const char*& elemEnd)
        {
            if (p >= end || *p != '{') {
                return false;
            }
            size_t depth = 0;
            bool inStr = false;
            bool escaped = false;
            while (p < end) {
                const char c = *p;
                if (inStr) {
                    if (escaped) {
                        escaped = false;
                    } else if (c == '\\') {
                        escaped = true;
                    } else if (c == '"') {
                        inStr = false;
                    }
                    ++p;
                    continue;
                }
                if (c == '"') {
                    inStr = true;
                    ++p;
                    continue;
                }
                if (c == '{') {
                    if (++depth > kMaxJsonDepth) {
                        return false; // 与 JsonCursor 深度上限一致, 布局扫描阶段先行拒绝
                    }
                    ++p;
                    continue;
                }
                if (c == '}') {
                    --depth;
                    ++p;
                    if (depth == 0) {
                        elemEnd = p;
                        return true;
                    }
                    continue;
                }
                ++p;
            }
            return false;
        }

#if defined(__aarch64__) || defined(__x86_64__) || defined(_M_X64)
        // 混合 SIMD: 普通字节用 FindStructuralChar/FindQuoteOrBackslash 按 16B 批量跳过,
        // 命中 '{' '}' '"' '\\' 时做与标量一致的状态转移, 元素边界判定语义不变。
        // 字符串内 '\\' 后只跳 1 个被转义字节(与标量 escaped 标志等价), 不按 \uXXXX 特判 6 字节,
        // 避免恶意构造的 "\u12\"G 改变边界判定。
        bool ScanOneRankElementSimd(const char*& elemEnd)
        {
            if (p >= end || *p != '{') {
                return false;
            }
            size_t depth = 0;
            bool inStr = false;
            while (p < end) {
                if (inStr) {
                    const char* hit = FindQuoteOrBackslash(p, end);
                    if (hit == end) {
                        return false; // 字符串未闭合
                    }
                    p = hit;
                    if (*p == '\\') {
                        if (p + 1 >= end) {
                            return false; // 转义符后被截断
                        }
                        p += 2; // 被转义字节不再检查, 与 escaped 标志语义一致
                        continue;
                    }
                    inStr = false; // 收尾引号
                    ++p;
                    continue;
                }
                const char* hit = FindStructuralChar(p, end);
                if (hit == end) {
                    return false; // 对象未闭合
                }
                p = hit;
                const char c = *p;
                if (c == '"') {
                    inStr = true;
                    ++p;
                    continue;
                }
                if (c == '{') {
                    if (++depth > kMaxJsonDepth) {
                        return false; // 与 JsonCursor 深度上限一致, 布局扫描阶段先行拒绝
                    }
                    ++p;
                    continue;
                }
                if (c == '}') {
                    --depth;
                    ++p;
                    if (depth == 0) {
                        elemEnd = p;
                        return true;
                    }
                    continue;
                }
                ++p; // 字符串外的 '\\' 按普通字节跳过, 与标量路径一致
            }
            return false;
        }

        bool ScanOneRankElement(const char*& elemEnd) { return ScanOneRankElementSimd(elemEnd); }
#else
        bool ScanOneRankElement(const char*& elemEnd) { return ScanOneRankElementScalar(elemEnd); }
#endif

        bool ScanRankElements()
        {
            if (!Consume('[')) {
                return false;
            }
            SkipWs();
            if (p < end && *p == ']') {
                ++p;
                return true;
            }
            while (true) {
                SkipWs();
                const char* chunkBegin = p;
                size_t chunkBytes = 0;
                while (true) {
                    const char* elemEnd = nullptr;
                    if (!ScanOneRankElement(elemEnd)) {
                        return false;
                    }
                    ++layout.elementCount;
                    chunkBytes = static_cast<size_t>(elemEnd - chunkBegin);
                    SkipWs();
                    if (p < end && *p == ',') {
                        ++p;
                        SkipWs();
                        if (p < end && *p == ']') {
                            return false; // 拒绝尾逗号
                        }
                        if (chunkBytes >= kChunkTargetBytes) {
                            layout.chunks.push_back(
                                {static_cast<size_t>(chunkBegin - data), static_cast<size_t>(elemEnd - data)});
                            chunkBegin = p;
                            chunkBytes = 0;
                        }
                        continue;
                    }
                    if (p < end && *p == ']') {
                        ++p;
                        layout.chunks.push_back(
                            {static_cast<size_t>(chunkBegin - data), static_cast<size_t>(elemEnd - data)});
                        return true;
                    }
                    return false;
                }
            }
        }

        bool ScanRankCount()
        {
            SkipWs();
            if (p < end && *p == '0' && p + 1 < end && p[1] >= '0' && p[1] <= '9') {
                return false; // JSON 不允许前导零, 与串行路径保持一致
            }
            u64 value = 0;
            bool any = false;
            while (p < end && *p >= '0' && *p <= '9') {
                const u64 digit = static_cast<u64>(*p - '0');
                if (value > (static_cast<u64>(UINT32_MAX) - digit) / 10) {
                    return false; // rank_count 超出 u32 范围, 回退串行路径报错
                }
                value = value * 10 + digit;
                ++p;
                any = true;
            }
            if (!any) {
                return false;
            }
            layout.rankCount = value;
            layout.hasRankCount = true;
            return true;
        }

        bool ScanVersion()
        {
            const char* verBegin = nullptr;
            size_t verLen = 0;
            if (!ScanString(verBegin, verLen)) {
                return false;
            }
            // 布局扫描不做转义解码；含转义时回退串行路径，避免与 JsonCursor 结果不一致
            if (std::memchr(verBegin, '\\', verLen) != nullptr) {
                return false;
            }
            layout.version.assign(verBegin, verLen);
            layout.hasVersion = true;
            return true;
        }

        bool ScanDetour()
        {
            SkipWs();
            if (p >= end || *p != '"') {
                return false;
            }
            const char* detourBegin = nullptr;
            size_t detourLen = 0;
            if (!ScanString(detourBegin, detourLen)) {
                return false;
            }
            if (detourLen == 4 && std::memcmp(detourBegin, "true", 4) == 0) {
                layout.detour = true;
                return true;
            }
            if ((detourLen == 5 && std::memcmp(detourBegin, "false", 5) == 0) || detourLen == 0) {
                layout.detour = false;
                return true;
            }
            return false;
        }

        bool ScanField(const char* keyBegin, size_t keyLen)
        {
            if (KeyIs(keyBegin, keyLen, RankTableKey::kRankList)) {
                if (layout.hasRankList) {
                    // 重复 rank_list 采用 last-wins: 清空已扫元素与分块, 只统计最后一段
                    layout.elementCount = 0;
                    layout.chunks.clear();
                }
                layout.hasRankList = true;
                return ScanRankElements();
            }
            if (KeyIs(keyBegin, keyLen, RankTableKey::kRankCount)) {
                return ScanRankCount();
            }
            if (KeyIs(keyBegin, keyLen, RankTableKey::kVersion)) {
                return ScanVersion();
            }
            if (KeyIs(keyBegin, keyLen, RankTableKey::kDetour)) {
                return ScanDetour();
            }
            return SkipValue(0);
        }

        bool Run()
        {
            SkipWs();
            if (p >= end || *p != '{') {
                return false;
            }
            ++p;
            while (true) {
                SkipWs();
                if (p < end && *p == '}') {
                    ++p;
                    SkipWs();
                    return p == end;
                }
                const char* keyBegin = nullptr;
                size_t keyLen = 0;
                if (!ScanString(keyBegin, keyLen) || !Consume(':')) {
                    return false;
                }
                if (!ScanField(keyBegin, keyLen)) {
                    return false;
                }
                SkipWs();
                if (p < end && *p == ',') {
                    ++p;
                    SkipWs();
                    if (p < end && *p == '}') {
                        return false; // 拒绝顶层尾逗号
                    }
                    continue;
                }
                if (p < end && *p == '}') {
                    ++p;
                    SkipWs();
                    return p == end;
                }
                return false;
            }
        }
    };

    bool ScanRankListLayout(const std::string& text, RankListLayout& layout)
    {
        LayoutScanner scanner(text, layout);
        return scanner.Run();
    }

    // 线程数: 按输入字节数估算，再受硬件核数和上限约束。
    // 环境变量 HCCL_FAST_PARSER_THREADS 可强制指定线程数(1~kMaxParseThreads, 合法取值直接生效,
    // 不再受硬件核数约束, 单核 CI 也可确定性触发并行路径), 供 UT 断言并行路径; 非法取值报错后回退自动估算。
    size_t ResolveParseThreadNum(size_t inputBytes)
    {
        const char* forced = std::getenv("HCCL_FAST_PARSER_THREADS");
        if (forced != nullptr) {
            char* end = nullptr;
            const long parsed = std::strtol(forced, &end, 10);
            if (end != nullptr && end != forced && *end == '\0' && parsed >= 1
                && static_cast<size_t>(parsed) <= kMaxParseThreads) {
                return static_cast<size_t>(parsed);
            }
            HCCL_ERROR("[%s] invalid HCCL_FAST_PARSER_THREADS[%s], fallback to auto", __func__, forced);
        }
        const size_t byBytes = (inputBytes + kBytesPerParseThread - 1) / kBytesPerParseThread;
        const size_t threadNum = std::max<size_t>(1, std::min(byBytes, kMaxParseThreads));
        const size_t hardware = static_cast<size_t>(std::thread::hardware_concurrency());
        return hardware == 0 ? 1 : std::min(threadNum, hardware);
    }

    // 直接解析原字符串上的连续 chunk，不复制 [{...},{...}] 临时缓冲
    bool ParseRankChunk(const std::string& text, const RankElemRange& chunk, std::vector<NewRankInfo>& out)
    {
        JsonCursor cur(text.data() + chunk.begin, text.data() + chunk.end);
        while (true) {
            cur.SkipWs();
            if (cur.AtEnd()) {
                return true;
            }
            NewRankInfo rankInfo;
            rankInfo.rankLevelInfos.reserve(2);
            if (ParseOneRank(cur, rankInfo) != HCCL_SUCCESS) {
                return false;
            }
            out.emplace_back(std::move(rankInfo));
            cur.SkipWs();
            if (cur.AtEnd()) {
                return true;
            }
            if (!cur.TryConsume(',')) {
                return false;
            }
            cur.SkipWs();
            if (cur.AtEnd()) {
                return false; // 拒绝尾逗号
            }
        }
    }

    void JoinThreads(std::vector<std::thread>& pool)
    {
        for (auto& thread : pool) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    bool RunChunkWorkers(
        const std::string& text, const std::vector<RankElemRange>& chunks, size_t threadNum,
        std::vector<std::vector<NewRankInfo>>& partRanks)
    {
        std::atomic<size_t> nextChunk{0};
        std::atomic<bool> workerFailed{false};
        std::vector<char> chunkOk(chunks.size(), 0);
        auto worker = [&]() {
            // 并行失败会回退串行路径统一报错, worker 内抑制字段级 RPT, 避免并发上报与回退后的重复上报
            gSuppressFastFieldReport = true;
            try {
                for (;;) {
                    const size_t index = nextChunk.fetch_add(1, std::memory_order_relaxed);
                    if (index >= chunks.size()) {
                        return;
                    }
                    chunkOk[index] = ParseRankChunk(text, chunks[index], partRanks[index]) ? 1 : 0;
                }
            } catch (...) {
                workerFailed.store(true, std::memory_order_relaxed);
            }
        };
        std::vector<std::thread> pool;
        pool.reserve(threadNum);
        try {
            for (size_t i = 0; i < threadNum; ++i) {
                pool.emplace_back(worker);
            }
        } catch (const std::exception&) {
            JoinThreads(pool);
            return false;
        }
        JoinThreads(pool);
        if (workerFailed.load(std::memory_order_relaxed)) {
            return false;
        }
        for (const char ok : chunkOk) {
            if (ok == 0) {
                return false;
            }
        }
        return true;
    }

    void MergeChunks(std::vector<std::vector<NewRankInfo>>& partRanks, std::vector<NewRankInfo>& merged)
    {
        size_t total = 0;
        for (const auto& part : partRanks) {
            total += part.size();
        }
        merged.clear();
        merged.reserve(total);
        for (auto& part : partRanks) {
            for (auto& rankInfo : part) {
                merged.emplace_back(std::move(rankInfo));
            }
            // 搬空一个 part 立即释放其容量, 避免 partRanks 与 merged 同时存活带来约 2 倍内存峰值
            std::vector<NewRankInfo>().swap(part);
        }
    }

    bool TryParseRankListParallel(
        const std::string& text, const std::vector<RankElemRange>& chunks, size_t threadNum,
        std::vector<NewRankInfo>& merged)
    {
        if (chunks.size() < threadNum * 2) {
            return false;
        }
        std::vector<std::vector<NewRankInfo>> partRanks(chunks.size());
        if (!RunChunkWorkers(text, chunks, threadNum, partRanks)) {
            return false;
        }
        MergeChunks(partRanks, merged);
        return true;
    }

    // 测试观测: 并行解析路径累计成功次数, 仅供 UT 断言并行路径被触发, 不参与生产逻辑
    std::atomic<u64> gParallelParseUsed{0};

    struct ParseFlags {
        bool hasVersion{false};
        bool hasRankCount{false};
        bool hasRankList{false};
    };

    // 优先: 结构扫描定位 rank 元素区间 -> 多线程分块解析 -> 合并
    template <typename InfoT, bool kIsRankTable>
    bool TryParseRankTableParallel(const std::string& jsonString, InfoT& info, size_t threadNum)
    {
        RankListLayout layout;
        if (!ScanRankListLayout(jsonString, layout) || !layout.hasVersion || !layout.hasRankCount
            || (kIsRankTable && layout.rankCount > MAX_RANKCOUNT)
            || (kIsRankTable && layout.rankCount != layout.elementCount)) {
            return false;
        }
        std::vector<NewRankInfo> merged;
        if (!TryParseRankListParallel(jsonString, layout.chunks, threadNum, merged)
            || merged.size() != layout.elementCount) {
            return false;
        }
        info.version = layout.version;
        info.rankCount = static_cast<u32>(layout.rankCount);
        if constexpr (kIsRankTable) {
            info.detour = layout.detour;
        }
        info.ranks = std::move(merged);
        return true;
    }

    template <typename InfoT, bool kIsRankTable>
    HcclResult ParseRankCountField(JsonCursor& cur, InfoT& info, const std::string& jsonString, ParseFlags& flags)
    {
        if (!cur.TryConsume(':') || !cur.ParseU32(info.rankCount)) {
            HCCL_ERROR("[%s] parse rank_count failed", __func__);
            return HCCL_E_PARA;
        }
        if constexpr (kIsRankTable) { // 仅 RankTableInfo 校验 MAX_RANKCOUNT, 与 DOM 一致(ChangedRankInfo 无此上限)
            if (info.rankCount > MAX_RANKCOUNT) {
                HCCL_ERROR("[%s] rank_count[%u] exceeds maximum[%u]", __func__, info.rankCount, MAX_RANKCOUNT);
                return HCCL_E_PARA;
            }
        }
        // 按上限与输入字节数封顶，避免反复扩容，也避免伪造字段触发巨量分配
        info.ranks.reserve(SafeReserveCount(info.rankCount, jsonString.size()));
        flags.hasRankCount = true;
        return HCCL_SUCCESS;
    }

    template <typename InfoT>
    HcclResult ParseRankListField(JsonCursor& cur, InfoT& info, const std::string& jsonString, ParseFlags& flags)
    {
        if (!cur.TryConsume(':')) {
            HCCL_ERROR("[%s] parse rank_list failed", __func__);
            return HCCL_E_PARA;
        }
        if (flags.hasRankList) {
            info.ranks.clear(); // 重复 rank_list 采用 last-wins, 与 nlohmann 一致
        }
        CHK_RET(ParseRankList(cur, info.ranks, info.rankCount, jsonString.size()));
        flags.hasRankList = true;
        return HCCL_SUCCESS;
    }

    template <typename InfoT>
    HcclResult ParseVersionField(JsonCursor& cur, InfoT& info, ParseFlags& flags)
    {
        if (!cur.TryConsume(':') || !cur.ParseString(info.version)) {
            HCCL_ERROR("[%s] parse version failed", __func__);
            return HCCL_E_PARA;
        }
        flags.hasVersion = true;
        return HCCL_SUCCESS;
    }

    template <typename InfoT, bool kIsRankTable>
    HcclResult ParseDetourField(JsonCursor& cur, InfoT& info)
    {
        if constexpr (kIsRankTable) {
            if (!cur.TryConsume(':')) {
                HCCL_ERROR("[%s] parse detour failed", __func__);
                return HCCL_E_PARA;
            }
            std::string detourStr;
            if (!cur.ParseString(detourStr)) {
                HCCL_ERROR("[%s] parse detour failed", __func__);
                return HCCL_E_PARA;
            }
            if (detourStr == "true") {
                info.detour = true;
            } else if (detourStr == "false" || detourStr == "") {
                info.detour = false;
            } else {
                HCCL_ERROR("[%s] invalid detour value[%s]", __func__, detourStr.c_str());
                return HCCL_E_PARA;
            }
        }
        return HCCL_SUCCESS;
    }

    template <typename InfoT, bool kIsRankTable>
    HcclResult ParseSerialFields(JsonCursor& cur, InfoT& info, const std::string& jsonString, ParseFlags& flags)
    {
        while (!cur.Is('}')) {
            if (cur.AtEnd()) {
                return HCCL_E_PARA;
            }
            if (cur.ParseKey(RankTableKey::kRankCount)) {
                CHK_RET((ParseRankCountField<InfoT, kIsRankTable>(cur, info, jsonString, flags)));
            } else if (cur.ParseKey(RankTableKey::kRankList)) {
                CHK_RET(ParseRankListField(cur, info, jsonString, flags));
            } else if (cur.ParseKey(RankTableKey::kVersion)) {
                CHK_RET(ParseVersionField(cur, info, flags));
            } else if (kIsRankTable && cur.ParseKey(RankTableKey::kDetour)) {
                CHK_RET((ParseDetourField<InfoT, kIsRankTable>(cur, info)));
            } else {
                std::string key;
                if (!cur.ParseString(key) || !cur.TryConsume(':')) {
                    HCCL_ERROR("[%s] parse key failed at offset[%ld]", __func__, cur.Pos() - jsonString.data());
                    return HCCL_E_PARA;
                }
                if (!cur.SkipValue()) {
                    HCCL_ERROR("[%s] skip value failed at offset[%ld]", __func__, cur.Pos() - jsonString.data());
                    return HCCL_E_PARA;
                }
            }
            if (!cur.CheckSeparator('}')) {
                HCCL_ERROR("[%s] expect ',' or '}' at offset[%ld]", __func__, cur.Pos() - jsonString.data());
                return HCCL_E_PARA;
            }
        }
        return HCCL_SUCCESS;
    }

    template <typename InfoT, bool kIsRankTable>
    HcclResult FinalizeParsedInfo(InfoT& info, const ParseFlags& flags, [[maybe_unused]] bool check)
    {
        if (!flags.hasVersion || !flags.hasRankCount || !flags.hasRankList) {
            HCCL_ERROR(
                "[%s] %s missing version/rank_count/rank_list", __func__,
                kIsRankTable ? "rank_table" : "changed_rank_info");
            return HCCL_E_PARA;
        }
        if constexpr (kIsRankTable) {
            if (info.rankCount > 0 && info.rankCount != info.ranks.size()) {
                HCCL_ERROR("[%s] rank_count[%u] != ranks size[%zu]", __func__, info.rankCount, info.ranks.size());
                return HCCL_E_PARA;
            }
            // 是否执行完整领域校验由调用方决定；结构性的 rankCount 校验始终保留。
            if (check) {
                CHK_RET(info.Check());
            }
        }
        return HCCL_SUCCESS;
    }

    // ranktable 与变更 ranktable 的顶层结构一致(version/rank_count/rank_list)，共用同一套解析：
    // kIsRankTable=true 时额外解析 detour、要求 rankCount==ranks.size 并执行 RankTableInfo::Check。
    template <typename InfoT, bool kIsRankTable>
    HcclResult ParseRankTableLikeImpl(const std::string& jsonString, InfoT& info, bool check)
    {
        if (jsonString.empty()) {
            HCCL_ERROR("[%s] json string length is zero", __func__);
            return HCCL_E_PARA;
        }
        info.version.clear();
        info.rankCount = 0;
        info.ranks.clear();
        if constexpr (kIsRankTable) {
            info.detour = false;
        }
        const size_t threadNum = ResolveParseThreadNum(jsonString.size());
        // 只有分块数足够(>= 2*线程)才尝试并行, 避免小文件白付一次结构扫描
        const size_t minParallelBytes = threadNum * 2 * kChunkTargetBytes;
        if (threadNum > 1 && jsonString.size() >= minParallelBytes
            && TryParseRankTableParallel<InfoT, kIsRankTable>(jsonString, info, threadNum)) {
            gParallelParseUsed.fetch_add(1, std::memory_order_relaxed);
            if constexpr (kIsRankTable) {
                if (check) {
                    CHK_RET(info.Check());
                }
                SortRankTableRanks(info);
            }
            return HCCL_SUCCESS;
        }

        JsonCursor cur(jsonString.data(), jsonString.data() + jsonString.size());
        if (!cur.TryConsume('{')) {
            HCCL_ERROR("[%s] ranktable is not a json object", __func__);
            return HCCL_E_PARA;
        }
        ParseFlags flags;
        CHK_RET((ParseSerialFields<InfoT, kIsRankTable>(cur, info, jsonString, flags)));
        if (!cur.TryConsume('}')) {
            HCCL_ERROR("[%s] ranktable object not closed", __func__);
            return HCCL_E_PARA;
        }
        cur.SkipWs();
        if (!cur.AtEnd()) {
            HCCL_ERROR("[%s] trailing content after ranktable json object", __func__);
            return HCCL_E_PARA;
        }
        CHK_RET((FinalizeParsedInfo<InfoT, kIsRankTable>(info, flags, check)));
        if constexpr (kIsRankTable) {
            SortRankTableRanks(info);
        }
        return HCCL_SUCCESS;
    }

    // 与 DOM(nlohmann lexer skip_bom)一致: 仅跳过开头一个 UTF-8 BOM(EF BB BF),
    // 第二个 BOM 与其他前导非法字节仍按非法输入拒绝
    template <typename InfoT, bool kIsRankTable>
    HcclResult ParseRankTableLike(const std::string& jsonString, InfoT& info, bool check)
    {
        if (jsonString.size() >= 3 && static_cast<unsigned char>(jsonString[0]) == 0xEF
            && static_cast<unsigned char>(jsonString[1]) == 0xBB && static_cast<unsigned char>(jsonString[2]) == 0xBF) {
            const std::string stripped(jsonString.data() + 3, jsonString.size() - 3);
            return ParseRankTableLikeImpl<InfoT, kIsRankTable>(stripped, info, check);
        }
        return ParseRankTableLikeImpl<InfoT, kIsRankTable>(jsonString, info, check);
    }

} // namespace

// 仅供 UT 观测: 返回并行解析路径累计成功次数, 用于断言并行路径确实被触发
u64 GetFastParserParallelUsedForTest() { return gParallelParseUsed.load(std::memory_order_relaxed); }

HcclResult ParseRankTableStringFast(const std::string& jsonString, RankTableInfo& info)
{
    return ParseRankTableLike<RankTableInfo, true>(jsonString, info, true);
}

HcclResult ParseRankTableStringFastNoCheck(const std::string& jsonString, RankTableInfo& info)
{
    return ParseRankTableLike<RankTableInfo, true>(jsonString, info, false);
}

HcclResult ParseChangedRankInfoStringFast(const std::string& jsonString, ChangedRankInfo& info)
{
    return ParseRankTableLike<ChangedRankInfo, false>(jsonString, info, false);
}

} // namespace Hccl
