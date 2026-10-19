/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdint.h>
#include <stddef.h>

static unsigned int g_net_adapt_id_counter = 0x7000;

int net_adapt_init(void) { return 0; }

void net_adapt_uninit(void) {}

int net_alloc_jfc_id(const char* udev_name, unsigned int jfc_mode, unsigned int* jfc_id)
{
    (void)udev_name;
    (void)jfc_mode;
    if (jfc_id == NULL) {
        return -1;
    }
    *jfc_id = ++g_net_adapt_id_counter;
    return 0;
}

int net_free_jfc_id(const char* udev_name, unsigned int jfc_mode, unsigned int jfc_id)
{
    (void)udev_name;
    (void)jfc_mode;
    (void)jfc_id;
    return 0;
}

int net_alloc_jetty_id(const char* udev_name, unsigned int jetty_mode, unsigned int* jetty_id)
{
    (void)udev_name;
    (void)jetty_mode;
    if (jetty_id == NULL) {
        return -1;
    }
    *jetty_id = ++g_net_adapt_id_counter;
    return 0;
}

int net_free_jetty_id(const char* udev_name, unsigned int jetty_mode, unsigned int jetty_id)
{
    (void)udev_name;
    (void)jetty_mode;
    (void)jetty_id;
    return 0;
}

unsigned long long net_get_cqe_base_addr(unsigned int die_id)
{
    (void)die_id;
    return 0x10000000ULL;
}
