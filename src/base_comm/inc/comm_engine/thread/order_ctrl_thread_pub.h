/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* order 控制流线程登记接口投影头：供 coll_communicator_mgr(L2) 使用，接口基于 ThreadHandle，不暴露 Thread 类。
 * 注：查询接口 IsOrderCtrlThread 仅 base_comm 内部（aicpu_launch_manager.cc）使用，声明保留在内部 thread.h。 */

#ifndef HCOMM_ORDER_CTRL_THREAD_PUB_H
#define HCOMM_ORDER_CTRL_THREAD_PUB_H

#include "hcomm_res_defs.h"

namespace hccl {

/**
 * @brief 登记线程为order控制流线程（进程级集合，key=host侧线程句柄）。幂等。
 * @param[in] hostHandle host侧线程句柄
 */
void MarkOrderCtrlThread(ThreadHandle hostHandle);

} // namespace hccl

#endif // HCOMM_ORDER_CTRL_THREAD_PUB_H
