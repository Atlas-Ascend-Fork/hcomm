/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef JSON_SCAN_SIMD_H
#define JSON_SCAN_SIMD_H

#include <cstddef>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace Hccl {

// 定位下一个 '"' 或 '\\'（aarch64 用 NEON、x86_64 用 SSE2，16B 批量比较），未命中返回 end
inline const char* FindQuoteOrBackslash(const char* p, const char* end)
{
#if defined(__aarch64__)
    while (p + 16 <= end) {
        const uint8x16_t v = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
        const uint8x16_t quoteMask = vceqq_u8(v, vdupq_n_u8('"'));
        const uint8x16_t slashMask = vceqq_u8(v, vdupq_n_u8('\\'));
        if (vmaxvq_u8(vorrq_u8(quoteMask, slashMask)) != 0) {
            for (int i = 0; i < 16; ++i) {
                if (p[i] == '"' || p[i] == '\\') {
                    return p + i;
                }
            }
        }
        p += 16;
    }
#elif defined(__x86_64__) || defined(_M_X64)
    // SSE2 是 x86_64 的基线指令集，不需要额外编译开关；movemask + ctz 直接定位首个命中字节。
    const __m128i quoteMask = _mm_set1_epi8('"');
    const __m128i slashMask = _mm_set1_epi8(static_cast<char>(0x5C));
    while (p + 16 <= end) {
        const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        const __m128i hit = _mm_or_si128(_mm_cmpeq_epi8(v, quoteMask), _mm_cmpeq_epi8(v, slashMask));
        const int bits = _mm_movemask_epi8(hit);
        if (bits != 0) {
            return p + __builtin_ctz(static_cast<unsigned int>(bits));
        }
        p += 16;
    }
#endif
    while (p < end) {
        if (*p == '"' || *p == '\\') {
            return p;
        }
        ++p;
    }
    return end;
}

// 定位下一个 '{' / '}' / '"' / '\\'（aarch64 用 NEON、x86_64 用 SSE2，16B 批量比较），未命中返回 end
inline const char* FindStructuralChar(const char* p, const char* end)
{
#if defined(__aarch64__)
    while (p + 16 <= end) {
        const uint8x16_t v = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
        const uint8x16_t hit = vorrq_u8(
            vorrq_u8(
                vorrq_u8(vceqq_u8(v, vdupq_n_u8('{')), vceqq_u8(v, vdupq_n_u8('}'))), vceqq_u8(v, vdupq_n_u8('"'))),
            vceqq_u8(v, vdupq_n_u8('\\')));
        if (vmaxvq_u8(hit) != 0) {
            for (int i = 0; i < 16; ++i) {
                if (p[i] == '{' || p[i] == '}' || p[i] == '"' || p[i] == '\\') {
                    return p + i;
                }
            }
        }
        p += 16;
    }
#elif defined(__x86_64__) || defined(_M_X64)
    const __m128i braceOpenMask = _mm_set1_epi8('{');
    const __m128i braceCloseMask = _mm_set1_epi8('}');
    const __m128i quoteMask = _mm_set1_epi8('"');
    const __m128i slashMask = _mm_set1_epi8(static_cast<char>(0x5C));
    while (p + 16 <= end) {
        const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        const __m128i hit = _mm_or_si128(
            _mm_or_si128(
                _mm_or_si128(_mm_cmpeq_epi8(v, braceOpenMask), _mm_cmpeq_epi8(v, braceCloseMask)),
                _mm_cmpeq_epi8(v, quoteMask)),
            _mm_cmpeq_epi8(v, slashMask));
        const int bits = _mm_movemask_epi8(hit);
        if (bits != 0) {
            return p + __builtin_ctz(static_cast<unsigned int>(bits));
        }
        p += 16;
    }
#endif
    while (p < end) {
        if (*p == '{' || *p == '}' || *p == '"' || *p == '\\') {
            return p;
        }
        ++p;
    }
    return end;
}

} // namespace Hccl

#endif // JSON_SCAN_SIMD_H
