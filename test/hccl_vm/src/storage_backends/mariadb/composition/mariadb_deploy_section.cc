/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * See the License in the specific file packaged with this Software for details.
 */

// 文件职责：MariaDB 模块部署配置段解析实现（自迁移前公共 storage_deploy_config.cc
// 原样迁入，校验规则与错误诊断逐字保持）。
#include "storage_backends/mariadb/composition/mariadb_deploy_section.h"

#include <string>
#include <utility>

namespace sim::backends {
namespace {
    using HcclSim::Storage::DbCode;
    using HcclSim::Storage::DbResult;
    using HcclSim::Storage::MariadbConnectionProfile;

    template <typename T>
    DbResult<T> Invalid(const std::string& diagnostic)
    {
        return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, diagnostic};
    }

    /// uri 中不允许出现的字符（路径/查询/锚点分隔符与空白不属于 host 形态）
    bool HostCharacterValid(char value)
    {
        if (value == '/' || value == '?' || value == '#' || value == '@') {
            return false;
        }
        if (value == ' ' || value == '\t' || value == '\r' || value == '\n' || value == '\0') {
            return false;
        }
        return true;
    }

    /// ---- MariaDB 后端段（backend=mariadb 时必需；非法值解析期报错，不静默回退）。
    /// URI 形态校验与 mariadb_engine.cc 的 ParseMariadbUri 保持同一合同：
    /// mariadb://user:password@host:port（密码经最后一个 '@' 切分、可为空；用户名
    /// 非空；host 为 hostname/IPv4，本阶段不支持 IPv6/Unix socket）。

    /// 旧毫秒字段已删除（connection_establishment_timeout_s / packet_read_timeout_s /
    /// packet_write_timeout_s，单位秒）。旧字段出现即解析失败并提示新字段名——
    /// 不能忽略旧字段后静默使用缺省值；新旧字段同时出现同样在此拒绝（首个旧字段
    /// 即报错，不进新字段解析）。
    DbResult<void> RejectRemovedMariadbTimeoutFields(const YAML::Node& mariadbNode)
    {
        struct RemovedField {
            const char* removed;
            const char* replacement;
        };
        static constexpr RemovedField kRemovedFields[] = {
            {"connect_timeout_ms", "connection_establishment_timeout_s"},
            {"read_timeout_ms", "packet_read_timeout_s"},
            {"write_timeout_ms", "packet_write_timeout_s"},
        };
        for (const RemovedField& field : kRemovedFields) {
            if (mariadbNode[field.removed]) {
                return DbResult<void>{
                    DbCode::INVALID_ARGUMENT, 0,
                    std::string{"storage.mariadb."} + field.removed + " was removed: use " + field.replacement
                        + " (seconds) instead"};
            }
        }
        return {DbCode::OK, 0, {}};
    }

    DbResult<uint32_t> ReadMariadbTimeoutSeconds(const YAML::Node& mariadbNode, const char* field, uint32_t fallback)
    {
        if (!mariadbNode[field] || mariadbNode[field].IsNull()) {
            return {DbCode::OK, std::optional<uint32_t>{fallback}, 0, {}};
        }
        const YAML::Node& timeoutNode = mariadbNode[field];
        if (!timeoutNode.IsScalar()) {
            return Invalid<uint32_t>(std::string("storage.mariadb.") + field + " must be an integer");
        }
        uint32_t value = 0;
        try {
            value = timeoutNode.as<uint32_t>();
        } catch (const YAML::Exception&) {
            return Invalid<uint32_t>(std::string("storage.mariadb.") + field + " must be an integer");
        }
        if (value < HcclSim::Storage::kMariadbTimeoutMinS || value > HcclSim::Storage::kMariadbTimeoutMaxS) {
            return Invalid<uint32_t>(
                std::string("storage.mariadb.") + field + " is out of range ["
                + std::to_string(HcclSim::Storage::kMariadbTimeoutMinS) + ", "
                + std::to_string(HcclSim::Storage::kMariadbTimeoutMaxS) + "] seconds");
        }
        return {DbCode::OK, std::optional<uint32_t>{value}, 0, {}};
    }

    /// mariadb endpoint URI 形态校验（与引擎 Open 期的 ParseMariadbUri 同一规则：
    /// 解析期提前失败，装配期不再重复判形态以外的语义）
    DbResult<void> ValidateMariadbUri(const std::string& uri)
    {
        constexpr char kMariadbScheme[] = "mariadb://";
        const auto malformed = [&uri](const char* detail) {
            return DbResult<void>{DbCode::INVALID_ARGUMENT, 0, std::string{detail} + ": " + uri};
        };
        if (uri.find(kMariadbScheme) != 0) {
            return malformed("mariadb endpoint uri must be mariadb://user:password@host:port");
        }
        const std::string authority = uri.substr(sizeof(kMariadbScheme) - 1);
        const auto at = authority.rfind('@');
        if (at == std::string::npos || at == 0 || at + 1U == authority.size()) {
            return malformed("mariadb endpoint uri must be mariadb://user:password@host:port");
        }
        const std::string userinfo = authority.substr(0, at);
        const std::string hostPort = authority.substr(at + 1U);
        const auto colon = userinfo.find(':');
        if (colon == std::string::npos || colon == 0) {
            return malformed("mariadb endpoint uri requires user:password");
        }
        const auto separator = hostPort.find(':');
        if (separator == std::string::npos || separator == 0 || separator + 1U == hostPort.size()) {
            return malformed("mariadb endpoint uri must be mariadb://user:password@host:port");
        }
        if (hostPort.find(':', separator + 1U) != std::string::npos) {
            return malformed("mariadb endpoint host must be a hostname or IPv4 address (IPv6 is not supported)");
        }
        const std::string host = hostPort.substr(0, separator);
        const std::string port = hostPort.substr(separator + 1U);
        for (const char character : host) {
            if (!HostCharacterValid(character)) {
                return malformed("mariadb endpoint host contains invalid character");
            }
        }
        uint64_t portValue = 0;
        for (const char digit : port) {
            if (digit < '0' || digit > '9') {
                return malformed("mariadb endpoint port must be decimal digits");
            }
            portValue = portValue * 10U + static_cast<uint64_t>(digit - '0');
            if (portValue > 65535U) {
                return malformed("mariadb endpoint port is out of range");
            }
        }
        if (portValue == 0) {
            return malformed("mariadb endpoint port must not be zero");
        }
        return {DbCode::OK, 0, {}};
    }
} // namespace

DbResult<MariadbConnectionProfile> ParseMariadbDeploySection(const YAML::Node& mariadbNode)
{
    if (!mariadbNode.IsMap()) {
        return Invalid<MariadbConnectionProfile>("storage.mariadb must be a map");
    }
    if (!mariadbNode["uri"] || !mariadbNode["uri"].IsScalar()) {
        return Invalid<MariadbConnectionProfile>("storage.mariadb.uri is required");
    }
    MariadbConnectionProfile profile;
    try {
        profile.uri = mariadbNode["uri"].as<std::string>();
    } catch (const YAML::Exception&) {
        return Invalid<MariadbConnectionProfile>("storage.mariadb.uri must be a string");
    }
    auto uri = ValidateMariadbUri(profile.uri);
    if (!uri.ok()) {
        return {uri.code, std::nullopt, 0, uri.diagnostic};
    }
    auto removed = RejectRemovedMariadbTimeoutFields(mariadbNode);
    if (!removed.ok()) {
        return {removed.code, std::nullopt, 0, removed.diagnostic};
    }
    auto connectTimeout = ReadMariadbTimeoutSeconds(
        mariadbNode, "connection_establishment_timeout_s", profile.connectionEstablishmentTimeoutS);
    if (!connectTimeout.ok()) {
        return {connectTimeout.code, std::nullopt, 0, connectTimeout.diagnostic};
    }
    profile.connectionEstablishmentTimeoutS = *connectTimeout.value;
    auto readTimeout = ReadMariadbTimeoutSeconds(mariadbNode, "packet_read_timeout_s", profile.packetReadTimeoutS);
    if (!readTimeout.ok()) {
        return {readTimeout.code, std::nullopt, 0, readTimeout.diagnostic};
    }
    profile.packetReadTimeoutS = *readTimeout.value;
    auto writeTimeout = ReadMariadbTimeoutSeconds(mariadbNode, "packet_write_timeout_s", profile.packetWriteTimeoutS);
    if (!writeTimeout.ok()) {
        return {writeTimeout.code, std::nullopt, 0, writeTimeout.diagnostic};
    }
    profile.packetWriteTimeoutS = *writeTimeout.value;
    // 线路协议压缩（缺省 false，与 MariadbConnectionProfile 缺省一致；同机部署
    // 压缩只有 CPU 开销，远端链路可显式 true）。显式 null / 非标量 / 非布尔一律
    // 拒绝：行为字段，部署意图不明宁可装配失败，不静默回退缺省
    if (mariadbNode["wire_protocol_compression"]) {
        const YAML::Node& compressionNode = mariadbNode["wire_protocol_compression"];
        if (!compressionNode.IsScalar()) {
            return Invalid<MariadbConnectionProfile>("storage.mariadb.wire_protocol_compression must be a boolean");
        }
        try {
            profile.wireProtocolCompression = compressionNode.as<bool>();
        } catch (const YAML::Exception&) {
            return Invalid<MariadbConnectionProfile>("storage.mariadb.wire_protocol_compression must be a boolean");
        }
    }
    return {DbCode::OK, std::optional<MariadbConnectionProfile>{std::move(profile)}, 0, {}};
}
} // namespace sim::backends
