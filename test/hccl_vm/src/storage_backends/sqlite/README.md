# HCCL-VM SQLite 后端

## 1. 默认行为

未设置 HCCL_VM_STORAGE_CONFIG 时默认即使用本模块（本模块是当前源码组合中的默认候选），无需额外配置。安装根下使用 data/hccl_sim.db 和 data/hccl_vm_data.db 两个默认文件。

若 HCCL_VM_STORAGE_CONFIG 仍指向其他配置文件，先清除该变量；只有变量未设置才是无配置默认路径。需要显式选择时只设置 storage.backend: sqlite。

## 2. 依赖与缓存

| 开源软件 | 版本 | 许可证 (SPDX) | 下载地址 |
| --- | --- | --- | --- |
| sqlite | 3.51.3 | Public domain | 随仓分发：third_party/sqlite-amalgamation-3510300.zip（[上游归档](https://www.sqlite.org/2026/sqlite-amalgamation-3510300.zip)） |

## 3. 回到公共流程

本模块没有需要填写的连接参数。确认 HCCL_VM_STORAGE_CONFIG 已清除后，回到[主指南环境配置](../../../README.md#321-环境配置)，按[主指南使用示例](../../../README.md#32-使用示例)完成运行。

[返回主指南](../../../README.md)
