/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCL_IBV_EXT_POST_SEND_ABI_H
#define HCCL_IBV_EXT_POST_SEND_ABI_H

#include <cstdint>
#include <type_traits>
#include <infiniband/verbs.h>

namespace hcomm {
namespace ibvExtPostSend {
    // 驱动侧确认结论：libhrn5.so 导出的 ibv_ext_post_send 使用独立的扩展 ABI，与 rdma-core
    // providers/hns/verbs_exp.h 中的版本不兼容，二者不可混用：
    // 1. 函数为 4 参数，无 ext_resp 出参；
    // 2. 扩展属性仅 1 字节，reduce_type 占低 4bit、reduce_op 占高 4bit。
    // 该 ABI 已完成与驱动同事的联调验证，扩展属性结构体与函数指针类型统一收敛于本头文件。
    struct ExtAttr {
        uint8_t reduceType : 4;
        uint8_t reduceOp : 4;
    };

    using PostSendFunc
        = int (*)(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** badWr, struct ExtAttr* extAttr);

    // 位域布局与驱动 ABI 强绑定：整体必须恰好 1 字节且无填充，否则驱动会按错误偏移解析
    static_assert(sizeof(ExtAttr) == 1, "ibv_ext_post_send ext_attr must be exactly 1 byte");
    static_assert(alignof(ExtAttr) == 1, "ibv_ext_post_send ext_attr must be 1-byte aligned");
    static_assert(std::is_trivially_copyable<ExtAttr>::value, "ibv_ext_post_send ext_attr must be trivially copyable");
} // namespace ibvExtPostSend
} // namespace hcomm

#endif // HCCL_IBV_EXT_POST_SEND_ABI_H
