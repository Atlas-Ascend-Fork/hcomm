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
 * See the License in the root of the software repository for the full text of the License.
 */

// 文件职责：MariaDB 后端私有的强类型部署连接 profile（引擎与工厂的唯一配置输入）。
// profile 由 composition 按部署配置（HCCL_VM_STORAGE_CONFIG）构造；引擎与工厂只
// 消费本类型，不反向依赖 composition，也不识别业务库名。
#ifndef HCCL_VM_STORAGE_ENGINE_MARIADB_MARIADB_CONNECTION_PROFILE_H
#define HCCL_VM_STORAGE_ENGINE_MARIADB_MARIADB_CONNECTION_PROFILE_H

#include <cstdint>
#include <string>

namespace HcclSim::Storage {
/// MariaDB 客户端超时合法范围（秒；[1, 3600]）。YAML 非法值在配置解析期拒绝，
/// 程序化构造的越界 profile 在引擎 Open/Wipe 期拒绝（双层校验语义）。
inline constexpr uint32_t kMariadbTimeoutMinS = 1;
inline constexpr uint32_t kMariadbTimeoutMaxS = 3600;

/// MariaDB 后端部署连接 profile（backend=mariadb 时有效；bootstrap 一次解析，
/// 进程生命周期内不可变）。uri 形态 mariadb://user:password@host:port（本阶段
/// 不支持 IPv6 与 Unix socket）；v1 单实例（K=1），无分片/多实例概念。
/// 三类超时为 libmariadb 连接参数（单位秒，直接生效，不做毫秒换算）；
/// wireProtocolCompression 控制 MariaDB 线路协议压缩（MYSQL_OPT_COMPRESS；
/// 不是 InnoDB 页压缩或表压缩）——缺省 false（同机部署压缩只有 CPU 开销；
/// 远端/大 BLOB 传输链路可显式 true）。Open/Reconnect/Wipe 三条建连路径
/// 使用同一值。
struct MariadbConnectionProfile {
    std::string uri;
    uint32_t connectionEstablishmentTimeoutS{3};
    uint32_t packetReadTimeoutS{120};
    uint32_t packetWriteTimeoutS{120};
    bool wireProtocolCompression{false};
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_ENGINE_MARIADB_MARIADB_CONNECTION_PROFILE_H
