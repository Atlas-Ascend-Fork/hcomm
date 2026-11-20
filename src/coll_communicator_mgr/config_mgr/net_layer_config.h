/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_LAYER_CONFIG_H
#define NET_LAYER_CONFIG_H

#include <cstdint>
#include <string>

#include "hccl/hccl_types.h"

namespace hccl {
constexpr uint32_t HCCL_COMM_CONFIG_NET_LAYER_VERSION = 13U;

enum class NetLayerSelectionMode { DEFAULT, INCLUDE, EXCLUDE };

// 域级选层策略；各通信域独立保存解析结果，不修改拓扑。
class NetLayerSelection {
public:
    // 仅用于独立域：非空域参数优先，否则读取HCCL_USE_NET_LAYER；子域/恢复不调用。
    // 非空commConfig须为安全读取后的完整本地配置快照；nullptr表示未提供参数。
    // 字段须在固定容量内终止，失败不修改selection；环境值在首次成功读取后由进程缓存。
    // RUN INFO分别记录通信域参数或环境变量，不支持与外部setenv/unsetenv并发。
    static HcclResult LoadForIndependentComm(const HcclCommConfig* commConfig, NetLayerSelection& selection);

    // 纯解析：输入须已安全复制，最多31字节且不含内嵌NUL，支持0~7列表和not(列表)。
    // 来源优先级由独立域入口处理；此处不读取环境。失败时selection保持不变。
    static HcclResult Parse(const std::string& expression, NetLayerSelection& selection);

    // 掩码的第n位表示真实layer n；候选层由调用方验证，不在此判断连通性。
    // 显式指定不存在层或结果为空时失败，effectiveLayers保持不变。
    HcclResult Resolve(uint32_t availableLayers, uint32_t& effectiveLayers) const;

    // 子域只继承或收窄，不重新读取环境；value为固定32字节字段，nullptr表示继承。
    // parentLayers为父域已固化有效层，availableLayers来自父图；同域成员候选层一致。
    // 先取父域允许层与子域候选层的交集，再按参数选层；失败不修改effectiveLayers。
    // 未配置时允许空交集，其合法性留给既有特殊场景处理；此处不验证连通性。
    static HcclResult
    ResolveForSubComm(const char* value, uint32_t parentLayers, uint32_t availableLayers, uint32_t& effectiveLayers);

    NetLayerSelectionMode GetMode() const { return mode_; }
    uint32_t GetRequestedLayers() const { return requestedLayers_; }
    const std::string& GetNormalizedExpression() const { return normalizedExpression_; }

    // 仅用于日志：按真实层号升序打印，空集合为 none；非法高位也保留，便于定位输入错误。
    static std::string FormatLayers(uint32_t layers)
    {
        std::string result;
        for (uint32_t layer = 0; layers != 0; ++layer, layers >>= 1) {
            if ((layers & 1U) == 0) {
                continue;
            }
            if (!result.empty()) {
                result += ',';
            }
            result += std::to_string(layer);
        }
        return result.empty() ? "none" : result;
    }

private:
    // 校验列表或 not(列表) 语法，生成模式、层掩码及规范化表达式。
    HcclResult ParseExpression(const std::string& expression);
    // 按层号升序输出表达式，合并重复层号，保留正选/排除语义。
    void Normalize();

    NetLayerSelectionMode mode_{NetLayerSelectionMode::DEFAULT};
    uint32_t requestedLayers_{0};
    std::string normalizedExpression_;
};

// 独立域受理时调用一次：保留原业务配置版本，仅将参数/环境选层冻结到完整内部副本。
// 仅适用于非空配置入口；异步任务应按值持有output。子域和恢复不得调用本函数。
// 失败不修改output；后续复用完整快照，禁止重新读取环境。
HcclResult PrepareIndependentCommConfig(const HcclCommConfig* input, HcclCommConfig& output);

// 只读取已冻结的固定32字节选层字段，不传递整份配置、不重新读取环境。
// value须非空且在字段容量内终止，失败不修改selection。
HcclResult ReadNetLayerSelection(const char* value, NetLayerSelection& selection);
} // namespace hccl

#endif // NET_LAYER_CONFIG_H
