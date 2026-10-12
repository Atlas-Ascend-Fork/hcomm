/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "TOPO_ADDR_STUB"

#include <cstddef>
#include <cstring>
#include <string>

#include "sim_common_api.h"
#include "sim_log.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 拦截 hcomm 的 topo_addr_info 模块, 直接返回模拟器安装目录下的 topo.json 路径。
 * @param phyId    NPU 物理 ID (模拟器统一拓扑, 忽略)
 * @param filePath [out] 拓扑文件路径输出缓冲
 * @param bufSize  filePath 的最大长度
 * @return 0 成功, 非 0 失败 (与 hcomm 约定一致)
 */
int TopoAddrInfoGetTopoFilePath(int phyId, char* filePath, size_t bufSize)
{
    (void)phyId;

    if (filePath == nullptr || bufSize == 0) {
        HCCL_VM_ERROR(
            "[TopoAddrInfoGetTopoFilePath] invalid args: filePath={:p}, bufSize={}", static_cast<void*>(filePath),
            bufSize);
        return -1;
    }

    const std::string topoPath = InstallPath::ResolveToInstallRoot("data/topo.json");
    if (topoPath.size() + 1 > bufSize) {
        HCCL_VM_ERROR(
            "[TopoAddrInfoGetTopoFilePath] topo path too long: path={}, len={}, bufSize={}", topoPath, topoPath.size(),
            bufSize);
        return -1;
    }

    (void)std::memcpy(filePath, topoPath.c_str(), topoPath.size() + 1);
    HCCL_VM_INFO("[TopoAddrInfoGetTopoFilePath] return topo path directly: {}", topoPath);
    return 0;
}

#ifdef __cplusplus
}
#endif
