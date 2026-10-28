/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_COMMON_DLSYM_PROC_H
#define HCOMM_COMMON_DLSYM_PROC_H

#include "hccl/hccl_types.h"
#include "hccl_common.h"
#include "common/dlurma_function.h"

namespace hcomm {

/**
 * @brief 动态库符号表节点：DlUrmaFunction（URMA 用户态库 dlopen 函数表）值成员，dl 句柄随节点析构。
 * hccl::DlRaFunction / DlHalFunction 为合法平台底座，维持直调。
 */
class DlsymProc {
public:
    void DeInit() {}

    DlUrmaFunction& GetDlUrmaFunction() { return dlUrmaFunction_; }

public:
    ~DlsymProc() = default;

public:
    DlsymProc() = default;

private:
    DlsymProc(const DlsymProc&) = delete;
    DlsymProc& operator=(const DlsymProc&) = delete;

    DlUrmaFunction dlUrmaFunction_;
};

} // namespace hcomm

#endif // HCOMM_COMMON_DLSYM_PROC_H
