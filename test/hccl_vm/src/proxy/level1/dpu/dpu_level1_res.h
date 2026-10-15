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
 * Description: HostDPU 北向劫持（level1）进程内资源管理。
 *              DpuCallbackRegistry 维护 (commId, msgTag) -> DPU 回调函数指针；
 *              DpuMemRegistry 维护 commId -> DPU 共享内存视图（host 视角）。
 *              两者均为进程内单例（内容含函数指针/进程虚拟地址，不能也不需入库，
 *              跨进程协调由共享内存与 DpuDeviceInfo 表承载）。
 * Create: 2026-10-09
 */

#ifndef DPU_LEVEL1_RES_H
#define DPU_LEVEL1_RES_H

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "dpu_level1_common.h"

namespace HcclSim {
namespace DpuSim {

    /*==================== 1. DPU 回调注册表 ====================*/

    /**
     * @brief DPU 算法回调注册表（进程级单例，mutex 保护）。
     *
     * 对应真实系统的 TaskService::callbacks_（按 msgTag 分发 DPUKernelRun 回调）。
     * 与真实 g_taskServiceMap[commId][deviceId] 的组织方式对齐，key 取 (commId, msgTag)：
     * 共享内存按 comm 分配，执行器从消息来源内存块天然得知 commId。
     *
     * 内容为函数指针（本进程地址空间私有），生命周期与通信域一致：
     * HcclTaskRegister 注册 → 执行器 Lookup 分发 → HcclTaskUnRegister / 通信域销毁移除。
     */
    class DpuCallbackRegistry {
    public:
        /**
         * @brief 获取单例实例。
         * @return DpuCallbackRegistry 单例引用。
         */
        static DpuCallbackRegistry& Instance();

        DpuCallbackRegistry(const DpuCallbackRegistry&) = delete;
        DpuCallbackRegistry& operator=(const DpuCallbackRegistry&) = delete;

        /**
         * @brief 注册 DPU 回调（insert-if-absent，重复注册同 key 视为成功保持原值）。
         * @param commId 输入：通信域 id（Communicator 表主键）。
         * @param msgTag 输入：任务标识（hccl 算法实例 algTag）。
         * @param cb     输入：回调函数指针（HcclLaunchDPUKernel）。
         * @return 成功返回 true；参数非法（cb 为空/msgTag 为空）返回 false。
         */
        bool Register(uint64_t commId, const std::string& msgTag, DpuCallback cb);

        /**
         * @brief 注销指定 (commId, msgTag) 的回调。
         * @return 移除返回 true；不存在返回 false。
         */
        bool UnRegister(uint64_t commId, const std::string& msgTag);

        /**
         * @brief 按 (commId, msgTag) 查询回调。
         * @param out 输出：命中时回填回调指针。
         * @return 命中返回 true，否则 false。
         */
        bool Lookup(uint64_t commId, const std::string& msgTag, DpuCallback& out);

        /**
         * @brief 移除指定通信域的全部回调（通信域销毁路径）。
         * @return 实际移除的条目数。
         */
        uint32_t RemoveByCommId(uint64_t commId);

    private:
        DpuCallbackRegistry() = default;
        ~DpuCallbackRegistry() = default;

        std::mutex mutex_;
        std::map<std::pair<uint64_t, std::string>, DpuCallback> callbacks_; /**< (commId,msgTag) -> 回调 */
    };

    /*==================== 2. DPU 共享内存登记表 ====================*/

    /**
     * @brief 单个通信域的 DPU 共享内存视图（host 进程视角 + 设备地址）。
     *
     * 内存分配复刻真实 GetKFCWorkSpaceVA 的两步：aclrtMalloc（level2 跨进程
     * 共享内存，得设备地址 devAddr）+ halHostRegister（得 host 可解引用地址）。
     * 前半区为 NPU->DPU 请求区，后半区为 DPU->NPU 响应区。
     */
    struct DpuMemRegion {
        uint64_t commId{0};        /**< 所属通信域 id（Communicator 表主键） */
        uint8_t* npu2dpu{nullptr}; /**< 请求半区基址（host 可解引用地址） */
        uint8_t* dpu2npu{nullptr}; /**< 响应半区基址（host 可解引用地址） */
        uint64_t halfSize{0};      /**< 半区尺寸（字节） */
        void* devAddr{nullptr};    /**< 设备映射地址（aclrtMalloc 返回值，销毁时释放用） */
    };

    /**
     * @brief DPU 共享内存登记表（进程级单例，mutex 保护）。
     *
     * 执行器按快照轮询各 comm 的请求半区 flag；通信域销毁时按 commId 摘除。
     * 内容为 host 进程虚拟地址，仅本进程有效。
     */
    class DpuMemRegistry {
    public:
        /**
         * @brief 获取单例实例。
         * @return DpuMemRegistry 单例引用。
         */
        static DpuMemRegistry& Instance();

        DpuMemRegistry(const DpuMemRegistry&) = delete;
        DpuMemRegistry& operator=(const DpuMemRegistry&) = delete;

        /**
         * @brief 登记一个通信域的 DPU 共享内存区域（同 commId 重复登记返回 false）。
         * @param region 输入：内存区域描述（指针须非空、halfSize 非零）。
         * @return 成功返回 true。
         */
        bool Add(const DpuMemRegion& region);

        /**
         * @brief 摘除指定通信域的内存区域并取出其内容（供调用方释放内存）。
         * @param commId 输入：通信域 id。
         * @param out    输出：被摘除的区域描述；不存在时 out 不变。
         * @return 摘除成功返回 true；不存在返回 false。
         */
        bool Remove(uint64_t commId, DpuMemRegion& out);

        /**
         * @brief 获取当前登记区域的快照（副本），供执行器无锁轮询。
         * @return 全部登记区域的 vector 副本。
         */
        std::vector<DpuMemRegion> Snapshot();

    private:
        DpuMemRegistry() = default;
        ~DpuMemRegistry() = default;

        std::mutex mutex_;
        std::map<uint64_t, DpuMemRegion> regions_; /**< commId -> 内存区域 */
    };

} // namespace DpuSim
} // namespace HcclSim

#endif // DPU_LEVEL1_RES_H
