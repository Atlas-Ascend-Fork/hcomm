# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(intf_llt_pub_base INTERFACE)

target_compile_definitions(intf_llt_pub_base INTERFACE
	CFG_BUILD_DEBUG
)

# aarch64 的 BL 跳转距离有限，超大 UT 二进制需用 lld 链接(GNU ld 不生成跳转垫片)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # mockcpp aarch64 打桩会在函数入口覆写跳转代码，需函数对齐避免越界破坏相邻函数
    set(LLT_AARCH64_COMPILE_OPTS -falign-functions=32)
    find_program(LLT_LLD_LINKER NAMES ld.lld)
    if(LLT_LLD_LINKER)
        set(LLT_AARCH64_LINK_OPTS -fuse-ld=lld)
        message(STATUS "[intf_llt_pub] aarch64 UT 链接使用 lld")
    else()
        message(WARNING "[intf_llt_pub] 未找到 ld.lld，aarch64 下超大 UT 二进制可能链接失败，建议安装")
    endif()
endif()

target_compile_options(intf_llt_pub_base INTERFACE
    -D_GLIBCXX_USE_CXX11_ABI=0
    -g
    --coverage
    -w
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
    $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fsanitize=leak -fsanitize-recover=address,all -fno-stack-protector -fno-omit-frame-pointer -g>
    $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>
    -fPIC
    -pipe
    ${LLT_AARCH64_COMPILE_OPTS}
)

target_link_options(intf_llt_pub_base INTERFACE
    -fprofile-arcs -ftest-coverage
    $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fsanitize=leak -fsanitize-recover=address>
    $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>
    ${LLT_AARCH64_LINK_OPTS}
)

target_link_libraries(intf_llt_pub_base INTERFACE
    $<$<BOOL:${ENABLE_GCOV}>:-lgcov>
)


add_library(intf_llt_pub INTERFACE)

target_link_libraries(intf_llt_pub INTERFACE
    $<BUILD_INTERFACE:intf_llt_pub_base>
    $<BUILD_INTERFACE:json>
    $<$<BOOL:${ENABLE_TEST}>:mockcpp>
    $<$<BOOL:${ENABLE_TEST}>:GTest::gtest>
    -Wl,-rpath,${CMAKE_INSTALL_PREFIX}/lib
)
