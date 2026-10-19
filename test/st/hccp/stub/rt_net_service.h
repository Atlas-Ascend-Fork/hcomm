/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RT_NET_SERVICE_H
#define RT_NET_SERVICE_H

/* The CANN runtime toolkit ships rtProcExtParam, rtNetServiceOpenArgs, RT_EXT_PARAM_CNT_MAX and the
 * rtOpenNetService/rtCloseNetService declarations in runtime/rt_external_device.h, which the current
 * toolchain also pulls in through runtime/runtime/config.h (the runtime/runtime/mem.h include
 * chain). Re-declaring the structs here would clash with the toolchain copy (struct tag
 * redefinition), so this header simply forwards to the toolchain definitions. */
#include <runtime/rt_external_device.h>

#endif /* RT_NET_SERVICE_H */
