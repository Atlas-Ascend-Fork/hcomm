/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_INSTANCE_H
#define HCOMM_CCU_INSTANCE_H

#include <array>
#include <memory>
#include <vector>

#include "ccu_types.h"
#include "ccu_res_pack.h"
#include "ccu_drv_handle.h"
#include "ccu_device_res.h"
#include "ccu_res_desc.h"
#include "ccu_res_desc_mgr.h"
#include "hcomm/ccu/hcomm_res_ccu_entity_defs.h"

namespace hcomm {

/**
 * @brief CCU 控制操作槽位枚举。
 * @note HcommCcuInstanceEntity.ascCustom 数组中函数槽位的固定含义，仅 hcomm 与 asc-comm 内部约定。
 */
enum HcommCcuControlSlot {
    HCOMM_CCU_CONTROL_SLOT_CHANNEL_ENTITY = 0,   /**< 槽 0：获取 ChannelEntity 接口 */
    HCOMM_CCU_CONTROL_SLOT_ALLOC_INST_SPACE = 1, /**< 槽 1：按设备和 die 申请 CCU 指令空间 */
    HCOMM_CCU_CONTROL_SLOT_SUBMIT_INSTS = 2,     /**< 槽 2：下发 CCU 指令接口 */
};

/**
 * @brief CCU 批量资源类型枚举。
 * @note resTypes 数组按枚举值索引；block 与非 block 资源在 range 视图中分开表示。
 */
enum HcommCcuBatchResType {
    HCOMM_CCU_BATCH_RES_LOOP = 0,       /**< LoopEngine 资源 */
    HCOMM_CCU_BATCH_RES_BLOCK_LOOP = 1, /**< block LoopEngine 资源 */
    HCOMM_CCU_BATCH_RES_MS = 2,         /**< Memory Slice 资源 */
    HCOMM_CCU_BATCH_RES_BLOCK_MS = 3,   /**< block Memory Slice 资源 */
    HCOMM_CCU_BATCH_RES_CKE = 4,        /**< 完成事件/信号量资源 */
    HCOMM_CCU_BATCH_RES_BLOCK_CKE = 5,  /**< block 完成事件/信号量资源 */
    HCOMM_CCU_BATCH_RES_BLOCK_XN = 6,   /**< block 变量资源 */
    HCOMM_CCU_BATCH_RES_XN = 7,         /**< 变量资源 */
    HCOMM_CCU_BATCH_RES_GSA = 8,        /**< 通用存储地址资源 */
    HCOMM_CCU_BATCH_RES_BLOCK_GSA = 9,  /**< block 通用存储地址资源 */
    HCOMM_CCU_BATCH_RES_MISSION = 10,   /**< mission 资源 */
    HCOMM_CCU_BATCH_RES_TYPE_COUNT,
};

/**
 * @brief CCU 版本常量。
 * @note ABI 内透传的设备级 CCU 版本。
 */
enum HcommCcuVersion {
    HCOMM_CCU_VERSION_INVALID = -1, /**< 无效版本哨兵 */
    HCOMM_CCU_VERSION_V1 = 0,
    HCOMM_CCU_VERSION_V2 = 1,
};

constexpr uint32_t HCOMM_CCU_MAX_DIE_NUM = 2U; // 当前支持的 CCU die 数（现网 2 die；硬件扩展后调整）

/**
 * @brief 填充 CCU Instance 完整实体的 ABI 头。
 *
 * @param handle 输出：待填充的完整句柄。
 *
 * @return 成功返回 void。
 * @note 所有产出完整句柄的路径必须调用本函数，禁止手写 ABI 头。
 */
inline void HcommCcuInstanceEntityInit(HcommCcuInstanceEntity* handle)
{
    handle->header.version = HCOMM_CCU_INSTANCE_ENTITY_ABI_VERSION;
    handle->header.magicWord = HCOMM_CCU_INSTANCE_ENTITY_MAGIC_WORD;
    handle->header.size = sizeof(HcommCcuInstanceEntity);
    handle->header.reserved = 0;
}

/**
 * @note 职责：管理通信域持有的CCU资源
 */
class CcuInstance {
public:
    explicit CcuInstance() {};
    ~CcuInstance();
    CcuResult InitByInsType(const CcuInstanceType insType);
    // 基于资源描述符数组初始化（新接口路径，资源数量由 resDesc 驱动）
    CcuResult InitByResDescs(const CcuResDesc* descs[], uint32_t descNum);
    // 使用当前 Device 上所有已使能 ioDie 的全部资源初始化
    CcuResult InitByAllRes();
    CcuResult Reset();
    CcuResPack* GetResPack();
    // 获取 ccuIns 在指定 die 上持有的资源描述符，用于查询接口
    const CcuResDesc& GetTotalResDescs(uint8_t dieId) const;
    CcuResult SaveKernel(const CcuKernelHandle kernelHandle);
    const std::vector<CcuKernelHandle>& GetUntranslatedKernels();
    void SetHandle(CcuInsHandle insHandle);
    CcuResult BuildHcommCcuInstanceEntity();
    const HcommCcuInstanceEntity* GetHcommCcuInstanceEntity() const { return snapshotReady_ ? &snapshot_ : nullptr; }

    // 是否使用固定资源数量创建（InitByInsType 路径置 true，其余路径保持 false）
    bool IsFixedResNum() const { return isFixedResNum_; }

    CcuResult BeginRegister();
    CcuResult CheckRegistering() const;
    CcuResult EndRegister();
    void AbortRegister();

    CcuResult CascCntHandleAlloc(uint8_t dieId, HcommCcuCascCntHandle handle);
    CcuResult GetCascCntBlock(HcommCcuCascCntHandle handle, CntXnBlock& cascCntBlock);

private:
    // 从 resPack_ 取 CcuResRepository，把各 die 各资源类型的占用数量写入 totalResDescs_
    CcuResult FillTotalResDescs();
    // 查询设备 CCU 版本，并映射为跨 SO ABI 版本值
    CcuResult GetSnapshotCcuVersion(HcommCcuVersion& ccuVersion);
    // 清空上一次快照缓存的 range 数据，避免失败重试后残留旧视图
    void ResetSnapshotRanges();
    // 填充单个 die 的元数据与资源 range 视图
    CcuResult
    FillSnapshotDie(HcommCcuInstanceEntity& snapshot, const CcuResRepository& repo, uint32_t dieId, bool& enabled);
    // 把 repo 中单个 die 的各资源类型 range 拷贝到快照缓存，并挂到 die 视图上
    void FillSnapshotDieRanges(HcommCcuDieRes& dieRes, const CcuResRepository& repo, uint32_t dieId);
    // 填充全部启用 die 的快照视图，并返回启用 die 数
    CcuResult FillSnapshotDies(HcommCcuInstanceEntity& snapshot, const CcuResRepository& repo, uint32_t& enabledDieNum);
    // 填充跨 SO 控制操作槽位
    void FillSnapshotControlSlots(HcommCcuInstanceEntity& snapshot);

    enum class RegisterState { IDLE, REGISTERING, REGISTER_ABORTED };
    RegisterState registerState_{RegisterState::IDLE};
    int32_t userDevId_{INT32_MAX};
    CcuInsHandle insHandle_{0};
    std::shared_ptr<hcomm::CcuDrvHandle> ccuDrvHandle_{};
    std::unique_ptr<CcuResPack> resPack_{};
    std::vector<CcuKernelHandle> kernelHandles_{};
    std::vector<CcuKernelHandle> untranslatedKernelHandles_{};
    std::array<CcuResDesc, CCU_MAX_IODIE_NUM> totalResDescs_{}; // ccuIns持有的全部资源描述符
    bool isFixedResNum_{false}; // InitByInsType 路径置 true（使用固定资源数量创建），其余保持 false
    using SnapshotRanges = std::array<std::vector<HcommCcuDieResRange>, HCOMM_CCU_BATCH_RES_TYPE_COUNT>;
    std::array<SnapshotRanges, HCOMM_CCU_MAX_DIE_NUM> snapshotRanges_{};
    std::array<HcommCcuDieMetadata, HCOMM_CCU_MAX_DIE_NUM> snapshotMetadata_{};
    HcommCcuInstanceEntity snapshot_{};
    bool snapshotReady_{false};
};

} // namespace hcomm
#endif // HCOMM_CCU_INSTANCE_H
