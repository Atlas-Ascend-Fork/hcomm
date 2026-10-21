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
 * Description: MPI 接口劫持桩函数
 */

#define HCCL_VM_MODULE "MPI_STUB"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <dlfcn.h>

#include <nlohmann_json/json.hpp>
#include "sim_log.h"
#include "store_sim_run_mode.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * MPI_Comm 用 void* 声明, 不引入 mpi.h, 避免给 proxy 构建新增 MPI 依赖。
 * ABI 安全性:
 *   - OpenMPI: MPI_Comm 是指针, void* 完全匹配。
 *   - MPICH:   MPI_Comm 是 int, 寄存器零扩展后 void* 接收同值, 不解引用即无碍。
 */
constexpr int MPI_SUCCESS_RET = 0;

/**
 * @brief 从 RANK_TABLE_FILE 环境变量指定的 rank table 文件中解析 rank 数量。
 *
 * 优先读取 "rank_count" 字段; 若不存在则回退到 "rank_list" 数组长度。
 * 与 sim::RankTable::ParseFromJsonString 的解析规则保持一致, 但本函数自包含,
 * 不依赖 level1 的 sim::RankTable 符号, 因此在默认模式 (未加载 level1) 下也可工作。
 *
 * @param rankSize 输出: 解析成功时写入 rank 数量。
 * @return true 解析成功且 rankSize > 0; false 未设置 RANK_TABLE_FILE、文件读
 *         取失败、JSON 解析失败或 rank 数量为 0。
 */
static bool TryGetRankSizeFromRankTable(uint32_t& rankSize)
{
    const char* path = std::getenv("RANK_TABLE_FILE");
    if (path == nullptr || path[0] == '\0') {
        return false;
    }

    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        HCCL_VM_ERROR("{}: failed to open rank table file: {}", __func__, path);
        return false;
    }

    const auto j = nlohmann::json::parse(ifs, nullptr, false);
    if (!j.is_object()) {
        HCCL_VM_ERROR("{}: rank table root is not a JSON object: {}", __func__, path);
        return false;
    }

    if (j.contains("rank_count")) {
        rankSize = j["rank_count"].get<uint32_t>();
        return rankSize > 0;
    }

    if (j.contains("rank_list") && j["rank_list"].is_array()) {
        rankSize = static_cast<uint32_t>(j["rank_list"].size());
        return rankSize > 0;
    }

    HCCL_VM_ERROR("{}: rank table missing both 'rank_count' and 'rank_list': {}", __func__, path);
    return false;
}

/**
 * @brief 劫持 MPI_Comm_size, 仅估算模式下用 rank table 的 rank 数量覆盖返回值。
 *
 * 估算模式单进程模拟多 rank 集群通信, 从 RANK_TABLE_FILE 解析 rank_count 覆盖 *size
 * (不区分 communicator); 解析失败 (未设置/读失败/字段缺失) 时返回错误, 不回退真实
 * MPI_Comm_size —— 估算环境无 mpirun 拉起的多进程, 透传真实函数只会得到错误结果。
 * 其他模式 (normal / check-only) 不从 rank table 读卡数, 直接透传真实 MPI_Comm_size。
 * 模式由 sim::IsEstimateMode() 惰性读 DB 的 RunModeConfig 判定; 该符号 level0 链接期
 * 放行 (unresolved), 运行期由 LD_PRELOAD 链中靠后的 level2 提供。
 *
 * @param comm 输入: MPI 通信域句柄 (不解引用, 仅透传给真实函数)。
 * @param size 输出: rank 数量。
 * @return MPI_SUCCESS (0) 成功; 非 0 失败。
 */
int MPI_Comm_size(void* comm, int* size)
{
    if (size == nullptr) {
        HCCL_VM_ERROR("{}: size is nullptr", __func__);
        return 1; // 非 MPI_SUCCESS, 等价 MPI_ERR_ARG
    }

    // 估算模式: 只用 rank table 的卡数, 不回退真实 MPI_Comm_size
    if (sim::IsEstimateMode()) {
        uint32_t rankSize = 0;
        if (TryGetRankSizeFromRankTable(rankSize)) {
            *size = static_cast<int>(rankSize);
            HCCL_VM_INFO("{} hijacked: rankSize={:d}", __func__, rankSize);
            return MPI_SUCCESS_RET;
        }
        HCCL_VM_ERROR("{}: estimate mode requires RANK_TABLE_FILE with 'rank_count'/'rank_list'", __func__);
        return 1; // 非 MPI_SUCCESS, 等价 MPI_ERR_OTHER
    }

    // 其他模式 (normal / check-only) 不读 rank table, 走真实 MPI_Comm_size
    using RealFunc = int (*)(void*, int*);
    const auto realFn = reinterpret_cast<RealFunc>(dlsym(RTLD_NEXT, "MPI_Comm_size"));
    if (realFn != nullptr) {
        return realFn(comm, size);
    }

    // RTLD_NEXT 找不到 (libmpi.so 被外部 dlopen 进局部作用域): 重新打开并按 handle 查找
    static RealFunc mpiCommSizeFunc = nullptr;
    if (mpiCommSizeFunc == nullptr) {
        void* handle = dlopen("libmpi.so", RTLD_NOW);
        if (handle == nullptr) {
            HCCL_VM_WARN("dlopen libmpi.so failed: {}", dlerror());
            return 1;
        }

        // 必须用 handle 限定在 libmpi.so 内查找; RTLD_DEFAULT 会先命中本桩自身, 造成无限递归
        mpiCommSizeFunc = reinterpret_cast<RealFunc>(dlsym(handle, "MPI_Comm_size"));
        if (mpiCommSizeFunc == nullptr) {
            HCCL_VM_WARN("dlsym MPI_Comm_size failed: {}", dlerror());
            return 1;
        }
    }

    return mpiCommSizeFunc(comm, size);
}

#ifdef __cplusplus
}
#endif
