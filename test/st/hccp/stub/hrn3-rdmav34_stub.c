/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stddef.h>
#include <infiniband/verbs.h>

int roce_set_qp_lb_value(struct ibv_qp* qp, int lbValue) { return 0; }

int roce_get_qp_lb_value(struct ibv_qp* qp, int* lbValue) { return 0; }

int roce_get_qp_num(struct ibv_context* context, int* qpNum) { return 0; }
