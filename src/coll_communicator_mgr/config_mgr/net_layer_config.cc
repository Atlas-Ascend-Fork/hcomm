/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "net_layer_config.h"

#include <cstring>

#include "coll_comm_config.h"
#include "env_net_layer_config.h"
#include "log.h"
#include "securec.h"

namespace hccl {
namespace {
    constexpr uint32_t MAX_NET_LAYER_COUNT = 8;

    // 仅复制完整配置的固定字段，不接受任意未终止的短缓冲区。
    HcclResult CopyNetLayerExpression(const char* value, std::string& expression)
    {
        if (value == nullptr) {
            return HCCL_SUCCESS;
        }
        constexpr size_t READ_LIMIT = sizeof(HcclCommConfig::hcclUseNetLayer);
        const size_t length = strnlen(value, READ_LIMIT);
        if (length == READ_LIMIT) {
            HCCL_ERROR(
                "[NetLayerSelection][%s] expression has no terminator within readLimit[%zu], maxLength[%zu]", __func__,
                READ_LIMIT, READ_LIMIT - 1);
            return HCCL_E_PARA;
        }
        expression.assign(value, length);
        return HCCL_SUCCESS;
    }

    void SkipWhitespace(const std::string& expression, size_t& position)
    {
        // 固定ASCII空白规则，避免解析结果依赖进程locale。
        const size_t next = expression.find_first_not_of(" \t\n\r\f\v", position);
        position = next == std::string::npos ? expression.size() : next;
    }

    HcclResult ParseLayerList(const std::string& expression, size_t& position, uint32_t& layers)
    {
        while (position < expression.size()) {
            SkipWhitespace(expression, position);
            if (position == expression.size() || expression[position] < '0' || expression[position] > '7') {
                HCCL_ERROR(
                    "[NetLayerSelection][%s] expected layer[0-7] at position[%zu], expressionLength[%zu], "
                    "actualByte[0x%02x] (0 means end of input)",
                    __func__, position, expression.size(),
                    position < expression.size() ?
                        static_cast<uint32_t>(static_cast<unsigned char>(expression[position])) :
                        0U);
                return HCCL_E_PARA;
            }
            layers |= 1U << static_cast<uint32_t>(expression[position] - '0');
            ++position;
            SkipWhitespace(expression, position);
            if (position == expression.size() || expression[position] != ',') {
                return HCCL_SUCCESS;
            }
            ++position;
        }
        HCCL_ERROR(
            "[NetLayerSelection][%s] missing layer at position[%zu], expressionLength[%zu]", __func__, position,
            expression.size());
        return HCCL_E_PARA;
    }
} // namespace

// 独立域入口：读取参数或环境，解析后冻结选层配置。
HcclResult PrepareIndependentCommConfig(const HcclCommConfig* input, HcclCommConfig& output)
{
    HcclCommConfig preparedConfig{};
    CHK_RET(ReadHcclCommConfig(input, preparedConfig));
    NetLayerSelection selection;
    CHK_RET(NetLayerSelection::LoadForIndependentComm(&preparedConfig, selection));
    // 只冻结新增选层字段，其他参数及输入版本沿用原配置流程。
    auto ret = strcpy_s(
        preparedConfig.hcclUseNetLayer, sizeof(preparedConfig.hcclUseNetLayer),
        selection.GetNormalizedExpression().c_str());
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[PrepareIndependentCommConfig] copy selection failed[%d]", ret), HCCL_E_MEMORY);
    output = preparedConfig;
    return HCCL_SUCCESS;
}

HcclResult NetLayerSelection::LoadForIndependentComm(const HcclCommConfig* commConfig, NetLayerSelection& selection)
{
    std::string expression;
    CHK_RET(CopyNetLayerExpression(commConfig == nullptr ? nullptr : commConfig->hcclUseNetLayer, expression));
    if (expression.empty()) {
        // 仅参数未配置时读取环境；非法参数不能回退到环境默认值。
        return EnvNetLayerConfig::GetInstance().GetSelection(selection);
    }
    CHK_RET(Parse(expression, selection));
    HCCL_RUN_INFO(
        "[NetLayerSelection] hcclUseNetLayer set by comm config to [%s]", selection.GetNormalizedExpression().c_str());
    return HCCL_SUCCESS;
}

HcclResult NetLayerSelection::Parse(const std::string& expression, NetLayerSelection& selection)
{
    NetLayerSelection parsed;
    if (!expression.empty()) {
        CHK_RET(parsed.ParseExpression(expression));
    }
    selection = parsed;
    return HCCL_SUCCESS;
}

HcclResult NetLayerSelection::ParseExpression(const std::string& expression)
{
    if (expression.size() >= HCCL_COMM_NET_LAYER_MAX_LENGTH || expression.find('\0') != std::string::npos) {
        HCCL_ERROR(
            "[NetLayerSelection][%s] invalid expression: length[%zu], maxLength[%u], embeddedNull[%u]", __func__,
            expression.size(), HCCL_COMM_NET_LAYER_MAX_LENGTH - 1,
            static_cast<uint32_t>(expression.find('\0') != std::string::npos));
        return HCCL_E_PARA;
    }
    size_t position = 0;
    SkipWhitespace(expression, position);
    mode_ = NetLayerSelectionMode::INCLUDE;
    if (expression.compare(position, 3, "not") == 0) {
        mode_ = NetLayerSelectionMode::EXCLUDE;
        position += 3;
        SkipWhitespace(expression, position);
        if (position == expression.size() || expression[position] != '(') {
            HCCL_ERROR(
                "[NetLayerSelection][%s] expected '(' after 'not' at position[%zu], expressionLength[%zu]", __func__,
                position, expression.size());
            return HCCL_E_PARA;
        }
        ++position;
    }
    CHK_RET(ParseLayerList(expression, position, requestedLayers_));
    if (mode_ == NetLayerSelectionMode::EXCLUDE) {
        if (position == expression.size() || expression[position] != ')') {
            HCCL_ERROR(
                "[NetLayerSelection][%s] expected ')' at position[%zu], expressionLength[%zu]", __func__, position,
                expression.size());
            return HCCL_E_PARA;
        }
        ++position;
        SkipWhitespace(expression, position);
    }
    if (position != expression.size()) {
        HCCL_ERROR(
            "[NetLayerSelection][%s] unexpected trailing content at position[%zu], expressionLength[%zu], "
            "actualByte[0x%02x]; use ASCII ',' between layers",
            __func__, position, expression.size(),
            static_cast<uint32_t>(static_cast<unsigned char>(expression[position])));
        return HCCL_E_PARA;
    }
    Normalize();
    return HCCL_SUCCESS;
}

void NetLayerSelection::Normalize()
{
    normalizedExpression_ = mode_ == NetLayerSelectionMode::EXCLUDE ? "not(" : "";
    bool firstLayer = true;
    for (uint32_t layer = 0; layer < MAX_NET_LAYER_COUNT; ++layer) {
        if ((requestedLayers_ & (1U << layer)) == 0) {
            continue;
        }
        if (!firstLayer) {
            normalizedExpression_ += ',';
        }
        normalizedExpression_ += static_cast<char>('0' + layer);
        firstLayer = false;
    }
    if (mode_ == NetLayerSelectionMode::EXCLUDE) {
        normalizedExpression_ += ')';
    }
}

HcclResult NetLayerSelection::Resolve(uint32_t availableLayers, uint32_t& effectiveLayers) const
{
    if ((availableLayers >> MAX_NET_LAYER_COUNT) != 0) {
        HCCL_ERROR(
            "[NetLayerSelection][%s] invalid availableLayers[%s], supported layer count[%u]", __func__,
            FormatLayers(availableLayers).c_str(), MAX_NET_LAYER_COUNT);
        return HCCL_E_PARA;
    }
    uint32_t resolved = availableLayers;
    if (mode_ != NetLayerSelectionMode::DEFAULT) {
        // 正选、排除都要求所提层存在，不能用取交集静默忽略配置错误。
        if ((requestedLayers_ & ~availableLayers) != 0) {
            HCCL_ERROR(
                "[NetLayerSelection][%s] requested layers are unavailable: expression[%s], requestedLayers[%s], "
                "availableLayers[%s], missingLayers[%s]",
                __func__, normalizedExpression_.c_str(), FormatLayers(requestedLayers_).c_str(),
                FormatLayers(availableLayers).c_str(), FormatLayers(requestedLayers_ & ~availableLayers).c_str());
            return HCCL_E_PARA;
        }
        resolved = mode_ == NetLayerSelectionMode::INCLUDE ? requestedLayers_ : availableLayers & ~requestedLayers_;
        if (resolved == 0) {
            HCCL_ERROR(
                "[NetLayerSelection][%s] selection leaves no usable layer: expression[%s], requestedLayers[%s], "
                "availableLayers[%s]",
                __func__, normalizedExpression_.c_str(), FormatLayers(requestedLayers_).c_str(),
                FormatLayers(availableLayers).c_str());
            return HCCL_E_PARA;
        }
    }
    effectiveLayers = resolved;
    return HCCL_SUCCESS;
}

// 已冻结配置与子域复用公共解析，不重新读取环境。
HcclResult ReadNetLayerSelection(const char* value, NetLayerSelection& selection)
{
    CHK_PTR_NULL(value);
    std::string expression;
    CHK_RET(CopyNetLayerExpression(value, expression));
    return NetLayerSelection::Parse(expression, selection);
}

HcclResult NetLayerSelection::ResolveForSubComm(
    const char* value, uint32_t parentLayers, uint32_t availableLayers, uint32_t& effectiveLayers)
{
    if (((parentLayers | availableLayers) >> MAX_NET_LAYER_COUNT) != 0) {
        HCCL_ERROR(
            "[NetLayerSelection][%s] invalid layers: parentLayers[%s], availableLayers[%s], supported layer count[%u]",
            __func__, FormatLayers(parentLayers).c_str(), FormatLayers(availableLayers).c_str(), MAX_NET_LAYER_COUNT);
        return HCCL_E_PARA;
    }
    std::string expression;
    CHK_RET(CopyNetLayerExpression(value, expression));
    NetLayerSelection selection;
    // 子域参数为空表示继承，不能复用独立域的环境兜底逻辑。
    CHK_RET(Parse(expression, selection));
    // 先限制可用范围，显式提到父域禁用或子域不存在的层均由Resolve拒绝。
    return selection.Resolve(parentLayers & availableLayers, effectiveLayers);
}

} // namespace hccl
