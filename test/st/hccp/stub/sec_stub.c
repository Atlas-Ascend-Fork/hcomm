/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * The code snippet comes from libboundscheck project (https://atomgit.com/openeuler/libboundscheck)
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2014-2021. All rights reserved.
 *
 * Mulan Permissive Software License,Version 2 (Mulan PSL v2)
 *
 * Your use of this software is governed by the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *     http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "securec.h"

errno_t memcpy_s(void* dest, size_t destMax, const void* src, size_t count)
{
    memcpy(dest, src, count);
    return 0;
}

errno_t memset_s(void* dest, size_t destMax, int c, size_t count)
{
    memset(dest, c, count);
    return 0;
}

errno_t sscanf_s(const char* buffer, const char* format, ...)
{
    va_list args;
    int ret;
    va_start(args, format);
    ret = vsscanf(buffer, format, args);
    va_end(args);
    return ret;
}

int snprintf_s(char* strDest, size_t destMax, size_t count, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(strDest, count, format, args);
    va_end(args);
    return 1;
}

int sprintf_s(char* strDest, size_t destMax, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    vsprintf(strDest, format, args);
    va_end(args);
    return 1;
}

errno_t strcpy_s(char* strDest, size_t destMax, const char* strSrc)
{
    strcpy(strDest, strSrc);
    return 0;
}

errno_t strncpy_s(char* strDest, size_t destMax, const char* strSrc, size_t count)
{
    strncpy(strDest, strSrc, count);
    return 0;
}
errno_t strcat_s(char* strDest, size_t destMax, const char* strSrc)
{
    strcat(strDest, strSrc);
    return 0;
}

errno_t strncat_s(char* strDest, size_t destMax, const char* strSrc, size_t count)
{
    strncat(strDest, strSrc, count);
    return 0;
}

char* strtok_s(char* strToken, const char* strDelimit, char** context)
{
    return strtok_r(strToken, strDelimit, context);
}

errno_t memmove_s(void* dest, size_t destMax, const void* src, size_t count)
{
    memmove(dest, src, count);
    return 0;
}

int snprintf_truncated_s(char* strDest, size_t destMax, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    int ret = vsnprintf(strDest, destMax, format, args);
    va_end(args);
    return ret;
}

int vsnprintf_truncated_s(char* strDest, size_t destMax, const char* format, va_list argList)
{
    return vsnprintf(strDest, destMax, format, argList);
}
