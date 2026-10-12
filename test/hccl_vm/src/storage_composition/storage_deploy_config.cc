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

// 文件职责：部署配置（HCCL_VM_STORAGE_CONFIG）的中立 YAML 解析实现。
// 只处理外层结构：root/storage 映射、storage.backend 模块名匹配、配置段键
// 与已编入模块的对应关系。后端段内字段校验全部经 BackendModule::Configure
// 分发到各模块（模块规则原样保留，公共层不出现任何后端专属字段）。
// 未编入后端的取值/配置段键不能被静默忽略：按当前模块名校验并明确报错，
// 这是裁剪版（单后端源码树）的明确配置规则。
#include "storage_composition/storage_deploy_config.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include "storage_composition/compiled_backends.h"

namespace HcclSim::Storage {
namespace {
    template <typename T>
    DbResult<T> Invalid(const std::string& diagnostic)
    {
        return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, diagnostic};
    }

    /// 按注册名查找已编入模块（未编入返回空）
    const sim::backends::BackendModule* FindCompiledBackend(const std::string& name)
    {
        const sim::backends::BackendModule* const* modules = sim::backends::CompiledBackendModules();
        for (std::size_t index = 0; index < sim::backends::CompiledBackendCount(); ++index) {
            if (modules[index]->name == name) {
                return modules[index];
            }
        }
        return nullptr;
    }
} // namespace

DbResult<StorageDeploySelection> ParseStorageDeployConfigText(const std::string& text)
{
    YAML::Node root;
    try {
        root = YAML::Load(text);
    } catch (const YAML::Exception& failure) {
        return Invalid<StorageDeploySelection>(std::string{"storage config YAML parse failure: "} + failure.what());
    }
    if (!root.IsMap()) {
        return Invalid<StorageDeploySelection>("storage config root must be a map");
    }
    const YAML::Node storageNode = root["storage"];
    if (!storageNode || !storageNode.IsMap()) {
        return Invalid<StorageDeploySelection>("storage config requires a storage map");
    }
    const YAML::Node backendNode = storageNode["backend"];
    if (!backendNode || !backendNode.IsScalar()) {
        return Invalid<StorageDeploySelection>("storage.backend is required");
    }
    std::string backend;
    try {
        backend = backendNode.as<std::string>();
    } catch (const YAML::Exception&) {
        return Invalid<StorageDeploySelection>("storage.backend must be a string");
    }
    // 选中后端必须已编入（未编入/已移除/历史后端取值在此统一拒绝，
    // 不静默回退到任何默认后端）。
    const sim::backends::BackendModule* selected = FindCompiledBackend(backend);
    if (selected == nullptr) {
        return Invalid<StorageDeploySelection>("unknown storage backend: " + backend);
    }

    // 配置段键逐一核对：除 backend 外只允许已编入模块的同名段；未知键
    // （含已移除后端的残留段）解析期明确报错。
    for (const auto& entry : storageNode) {
        std::string key;
        try {
            key = entry.first.as<std::string>();
        } catch (const YAML::Exception&) {
            return Invalid<StorageDeploySelection>("storage config section key must be a string");
        }
        if (key == "backend") {
            continue;
        }
        if (FindCompiledBackend(key) == nullptr) {
            return Invalid<StorageDeploySelection>(
                "unknown storage config section: " + key + " (no such compiled backend module)");
        }
    }

    // 每个已编入模块都建立部署描述（无配置段的模块经 Configure(section 缺省)
    // 建立；保证显式路由仍可选择已注册的其他后端）。
    // 注意：yaml-cpp 对缺失键返回僵尸节点，Node::operator= 赋值僵尸节点会抛
    // InvalidNode——只在段存在时赋值，缺省留给默认构造的未定义节点。
    StorageDeploySelection selection;
    selection.backend = backend;
    const sim::backends::BackendModule* const* modules = sim::backends::CompiledBackendModules();
    for (std::size_t index = 0; index < sim::backends::CompiledBackendCount(); ++index) {
        const sim::backends::BackendModule& module = *modules[index];
        sim::backends::BackendConfigureRequest request;
        request.sectionPresent = static_cast<bool>(storageNode[module.name]);
        if (request.sectionPresent) {
            request.section = storageNode[module.name];
        }
        request.selected = module.name == backend;
        auto deployment = module.configure(request);
        if (!deployment.ok()) {
            return {deployment.code, std::nullopt, 0, deployment.diagnostic};
        }
        selection.deployments[std::string{module.name}] = std::move(*deployment.value);
    }
    return {DbCode::OK, std::optional<StorageDeploySelection>{std::move(selection)}, 0, {}};
}

DbResult<StorageDeploySelection> ParseStorageDeployConfigFile(const std::string& path)
{
    std::ifstream input(path, std::ios::in | std::ios::binary);
    if (!input.good()) {
        return Invalid<StorageDeploySelection>("storage config file does not exist or is not readable: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (input.bad()) {
        return Invalid<StorageDeploySelection>("storage config file read failure: " + path);
    }
    return ParseStorageDeployConfigText(buffer.str());
}
} // namespace HcclSim::Storage
