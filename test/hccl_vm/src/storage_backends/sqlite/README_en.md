# HCCL-VM SQLite Backend

## 1. Default Behavior

This module is used by default when HCCL_VM_STORAGE_CONFIG is not set (this module is the default candidate in the current source code combination), so no additional configuration is required. Two default files, data/hccl_sim.db and data/hccl_vm_data.db, are used under the installation root directory.

If HCCL_VM_STORAGE_CONFIG still points to another configuration file, clear the variable first. The default path without configuration takes effect only when the variable is not set. To explicitly select this backend, set only storage.backend: sqlite.

## 2. Dependencies and Cache

| Open Source Software | Version | License (SPDX) | Download URL |
| --- | --- | --- | --- |
| sqlite | 3.51.3 | Public domain | [sqlite-amalgamation-3510300.zip](https://www.sqlite.org/2026/sqlite-amalgamation-3510300.zip) |

The source code cache is stored in third_party/ at the repository root (archives and unpacked artifacts, which are ignored by git). The first build downloads the source code online. In offline scenarios, copy the original archive named sqlite-amalgamation-3510300.zip to third_party/ at the repository root. During the build, the archive is directly unpacked for use, without requiring network access.

## 3. Returning to the Common Procedure

This module has no connection parameters to fill in. After confirming that HCCL_VM_STORAGE_CONFIG has been cleared, return to [environment configuration in the main guide](../../../README.md#321-环境配置) and complete the run as described in [usage examples in the main guide](../../../README.md#32-使用示例).

[Back to the main guide](../../../README.md)
