# hcomm4py Public API Boundary

English | [简体中文](./PUBLIC_API.md)

This file is the **hcomm4py external API boundary checklist** for post-merge governance.  
**Only symbols listed in `hcomm4py/__init__.py` `__all__` are stable public APIs.** Underscore-prefixed modules/symbols are internal. This repo does not ship a `hccl4py` compat package (avoids clashing with cann/hccl ops).

## Boundary File Mapping

| Role | File | Notes |
|---|---|---|
| Public export entry | `hcomm4py/__init__.py` | Sole stable import surface (`__all__`) |
| Types / enums / config layout | `hcomm4py/types.py` | Aligned with C headers |
| Domain APIs | `hcomm4py/comm.py` | Task §3.1.1, 5 domain APIs |
| Dynamic library loading | `hcomm4py/_lib.py` | **Internal**, not public |
| Packaging entry | `setup.py` | Wheel build entry (`python -m build` / `pip wheel .`) |

## Public Symbols ↔ C API

### Types

| Python | C / header concept |
|---|---|
| `HcclResult` / `HcclError` | `HcclResult` |
| `HcclDataType` | `HcclDataType` |
| `HcclReduceOp` | `HcclReduceOp` |
| `HcclSendRecvType` | `HcclSendRecvType` |
| `HcclRootInfo` | `HcclRootInfo` |
| `HcclCommConfig` | `HcclCommConfig` |
| `HcclSendRecvItem` | `HcclSendRecvItem` |
| `HCCL_ROOT_INFO_BYTES` | `HCCL_ROOT_INFO_BYTES` (4108) |
| `HCCL_COMM_CONFIG_BYTES` | Python raw buffer capacity (2304) |
| `HCCL_COMM_CONFIG_MAGIC_WORD` | Config magic `0xF0F0F0F0` |
| `HCCL_COMM_CONFIG_SIZE_V10` / `V11` / `V12` | Logical size gates (2240 / 2248 / 2248) |
| `HCCL_COMM_CONFIG_VERSION_DEFAULT` | Offline fallback version (11) |
| `HCCL_COMM_CONFIG_LOGICAL_SIZE_DEFAULT` | Offline fallback logical_size (= V11) |

### Domain

Domain APIs resolve the non-V2 symbols from runtime `libhcomm.so` via ctypes: `HcclGetRootInfo`, `HcclCommInitRootInfoConfig`, `HcclCommInitClusterInfoConfig`, `HcclCommDestroy`.  
These exports come from the legacy A910 / op_base path; Ascend 950PR / op_base_v2 builds that only ship `*V2` symbols are incompatible with this binding (a platform/export mismatch, not fixed by upgrading to a newer CANN release, e.g. beyond 9.1.0). Missing symbols raise a platform-aware `RuntimeError` from `_bind()`.

| Python | C API |
|---|---|
| `get_root_info` | `HcclGetRootInfo` |
| `get_comm_config_capability` | `HcclGetCommConfigCapability` (optional; `None` if unavailable) |
| `comm_config_init` | `HcclCommConfigInit` (header inline; Python reimplementation) |
| `comm_init_root_info_config` / `init_rank_config` | `HcclCommInitRootInfoConfig` |
| `comm_init_cluster_info_config` | `HcclCommInitClusterInfoConfig` |
| `comm_destroy` | `HcclCommDestroy` |
| `Comm` | Handle context manager (Python-side) |

Collective/P2P op boundaries live in the sister cann/hccl tree:
sister MR [cann/hccl#3798](https://gitcode.com/cann/hccl/merge_requests/3798)
(after merge: that repo's `experimental/eco_system/python/PUBLIC_API.md`).

## Change Control

1. Add/remove/rename public symbols: update this file and `__all__` together.
2. Internal-only refactors: this file may stay unchanged; do not leak into `__all__`.
3. Combined with hccl ops: `import hcomm4py` for domain / `import hccl4py` (hccl repo) for ops.
