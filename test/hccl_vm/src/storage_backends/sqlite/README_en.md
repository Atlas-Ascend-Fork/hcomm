# HCCL-VM SQLite Backend

## 1. Default Behavior

The system uses this module by default when HCCL_VM_STORAGE_CONFIG is not set (this module is the default candidate in the current source combination). No additional configuration is required. Two default files data/hccl_sim.db and data/hccl_vm_data.db are used under the installation root.

If HCCL_VM_STORAGE_CONFIG still points to another configuration file, clear that variable first; only when the variable is unset does the default path apply. When explicit selection is needed, only set storage.backend: sqlite.

## 2. Dependencies and Caching

| Open Source Software | Version | License (SPDX) | Download Address |
| --- | --- | --- | --- |
| sqlite | 3.51.3 | Public domain | Distributed with repository: third_party/sqlite-amalgamation-3510300.zip ([upstream archive](https://www.sqlite.org/2026/sqlite-amalgamation-3510300.zip)) |

## 3. Return to Common Workflow

This module has no connection parameters to fill in. After confirming that HCCL_VM_STORAGE_CONFIG is cleared, return to the [Main Guide Environment Configuration](../../../README_en.md#321-environment-configuration) and complete the run according to the [Main Guide Usage Examples](../../../README_en.md#32-usage-examples).

[Return to Main Guide](../../../README_en.md)
