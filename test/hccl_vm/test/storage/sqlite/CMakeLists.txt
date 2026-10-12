# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# See the License in the specific file packaged with this Software for details.

# 文件职责：SQLite 引擎专属测试（test/storage/sqlite/，随 sqlite 后端模块一起
# 保留或裁剪；由收集器按模块存在性接入）。目标名与断言保持迁移前不变。

# 公共样板：include 面 + 链接面（storage 经 PUBLIC 传递 sqlite3 依赖）。
function(add_sqlite_storage_test target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/test/storage
    )
    target_link_libraries(${target} PRIVATE
        simulator_storage
        storage
        gtest_main
        gtest
        sqlite3
    )
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
    )
endfunction()

add_sqlite_storage_test(test_sqlite_engine sqlite_engine_test.cc)
add_sqlite_storage_test(test_sqlite_engine_fault sqlite_engine_fault_test.cc)
add_sqlite_storage_test(test_sqlite_engine_session sqlite_engine_session_test.cc)

# 批量插入 override 专项：入参合同/主键/约束失败/等价对照 + 链接期
# --wrap=sqlite3_prepare_v2 观察窗（只统计目标表 INSERT 的显式 prepare，
# 证明一批 N 行 override=1、基类默认=N）。包装只链接本测试目标。
add_sqlite_storage_test(test_sqlite_insert_many sqlite_insert_many_test.cc)
target_link_options(test_sqlite_insert_many PRIVATE "-Wl,--wrap=sqlite3_prepare_v2")

# 事务重叠合同专项：同线程立即拒绝（回归防线）/跨线程即时 CRUD 与第二笔
# Begin 有界等待/等待超时 RESOURCE_EXHAUSTED/关闭唤醒（等待上限经测试 peer
# 缩短；旧实现无等待成员时编译期退化为无操作——超时用例即 RED 证据）。
add_sqlite_storage_test(test_sqlite_transaction_overlap sqlite_transaction_overlap_test.cc)

# 公共 backend conformance suite 的 SQLite fixture（同一套公共 case 在 SQLite 实例化；
# 含投影清单——SQLite 实现字段投影）。
add_sqlite_storage_test(test_backend_conformance_sqlite backend_conformance_sqlite_test.cc)

# ---- SQLITE_BASELINE 业务装配与迁移语义测试 ----

# simulator composition（唯一 bootstrap）的会话装配与生命周期边界。
add_sqlite_storage_test(test_storage_composition storage_composition_test.cc)

# 冷进程会话 bootstrap 回归：首次 Storage 原子 API / 业务兼容 API 调用经进程
# Session Registry 惰性建立本进程会话（真实 fork 子进程冷状态 + 预置数据）。
add_sqlite_storage_test(test_storage_cold_process storage_cold_process_test.cc)

# composition 工厂退出生命周期回归：atexit 消费者经惰性路由首次访问已析构
# 部署快照之后的工厂（子进程 marker 模式需要自有 main，不链接 gtest_main）。
add_executable(test_storage_composition_exit storage_composition_exit_test.cc)
target_include_directories(test_storage_composition_exit PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/test/storage
)
target_link_libraries(test_storage_composition_exit PRIVATE
    simulator_storage
    storage
    gtest
    sqlite3
)
set_target_properties(test_storage_composition_exit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# 历史位置恢复回调合同回归：错误传播/成功不换库/成功换库/成功但缺外层载荷
#（链接期以真实 sqlite 模块 + 测试自有 fake 模块重定义 compiled_backends 清单，
# 需要生成头目录）。
add_sqlite_storage_test(test_storage_recorded_location storage_recorded_location_test.cc)
target_include_directories(test_storage_recorded_location PRIVATE
    ${CMAKE_BINARY_DIR}/generated
)

# SQLITE_BASELINE 物理清单（表/索引/外键/动态表）与迁移前 DDL 对照。
add_sqlite_storage_test(test_sqlite_baseline_inventory sqlite_baseline_inventory_test.cc)

# 会话表路由解析与动态 scope 边界。
add_sqlite_storage_test(test_storage_routing storage_routing_test.cc)

# RunnerDB BLOB 形态引擎语义（自增/BLOB 字节/读改写/序列）。
add_sqlite_storage_test(test_sqlite_runner_engine sqlite_runner_engine_test.cc)

# RunnerDB 会话与表契约故障路径。
add_sqlite_storage_test(test_sqlite_runner_engine_fault sqlite_runner_engine_fault_test.cc)

# OpDB 字段化记录引擎语义（列值/BLOB 规则/具名 JOIN/动态表）。
add_sqlite_storage_test(test_sqlite_opdb_engine sqlite_opdb_engine_test.cc)

# OpDB 原子组事务语义（提交/回滚/中毒/并发拒绝）。
add_sqlite_storage_test(test_sqlite_opdb_transaction sqlite_opdb_transaction_test.cc)

# OpDB 维护语义（备份/动态表清理/关闭边界）。
add_sqlite_storage_test(test_sqlite_opdb_maintenance sqlite_opdb_maintenance_test.cc)

# SQLite 后端模块专项静态门禁（部署知识/依赖配方/自描述完整性）
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(
    NAME sqlite_backend_static_gate
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/sqlite_static_gate.py
)
