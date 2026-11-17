# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Communication-domain Python bindings (task 3.1.1 - hcomm)."""

import ctypes
import threading
from typing import Optional

from ._lib import get_hcomm
from .types import (
    HCCL_COMM_CONFIG_MAGIC_WORD,
    HcclComm,
    HcclCommConfig,
    HcclRootInfo,
    _OFF,
    check,
    fill_comm_config,
    resolve_comm_config_layout,
)
from ._validate import (
    require_live_handle,
    require_positive_u32,
    require_u32,
)



def _require_root_info(root_info) -> HcclRootInfo:
    if not isinstance(root_info, HcclRootInfo):
        raise TypeError(
            f"root_info must be HcclRootInfo, got {type(root_info).__name__}"
        )
    return root_info


def _require_config(config, *, ready: bool = False):
    """Type-check config; when ready=True also require size/magic for C downcalls."""
    if config is None:
        return None
    if not isinstance(config, HcclCommConfig):
        raise TypeError(
            f"config must be HcclCommConfig or None, got {type(config).__name__}"
        )
    if ready:
        size = ctypes.c_size_t.from_buffer(config._raw, _OFF["size"]).value
        magic = ctypes.c_uint32.from_buffer(config._raw, _OFF["magic"]).value
        if size == 0 or magic != HCCL_COMM_CONFIG_MAGIC_WORD:
            raise ValueError(
                "config must be initialized via comm_config_init/fill_comm_config "
                f"(got size={size}, magic=0x{magic:x})"
            )
    return config


def _require_cluster_info(cluster_info):
    if isinstance(cluster_info, str):
        if not cluster_info:
            raise ValueError("cluster_info path must not be empty")
        return cluster_info
    if isinstance(cluster_info, (bytes, bytearray)):
        if not cluster_info:
            raise ValueError("cluster_info path must not be empty")
        return bytes(cluster_info)
    raise TypeError(
        f"cluster_info must be str or bytes, got {type(cluster_info).__name__}"
    )



_BOUND = False
_BIND_LOCK = threading.Lock()

# Non-V2 domain symbols (op_base path; Ascend950/op_base_v2 uses *V2 exports).
_DOMAIN_SYMBOLS = (
    "HcclGetRootInfo",
    "HcclCommInitRootInfoConfig",
    "HcclCommInitClusterInfoConfig",
    "HcclCommDestroy",
)

_CAPABILITY_BOUND = False
_CACHED_CAPABILITY: Optional[int] = None


def _require_domain_symbols(lib) -> None:
    missing = [name for name in _DOMAIN_SYMBOLS if not hasattr(lib, name)]
    if not missing:
        return
    raise RuntimeError(
        "current libhcomm.so does not export communication-domain symbol(s): "
        f"{', '.join(missing)}. hcomm4py binds the non-V2 APIs "
        "(HcclGetRootInfo / HcclCommInitRootInfoConfig / "
        "HcclCommInitClusterInfoConfig / HcclCommDestroy). These exports come "
        "from the legacy A910/op_base path — Ascend950/op_base_v2 builds that "
        "only ship *V2 symbols are not compatible with this binding. "
        "Load a libhcomm that exports the non-V2 domain APIs."
    )


def _bind():
    global _BOUND
    lib = get_hcomm()
    if _BOUND:
        return lib
    with _BIND_LOCK:
        if _BOUND:
            return lib
        _require_domain_symbols(lib)
        lib.HcclGetRootInfo.argtypes = [ctypes.POINTER(HcclRootInfo)]
        lib.HcclGetRootInfo.restype = ctypes.c_int

        lib.HcclCommInitRootInfoConfig.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(HcclRootInfo),
            ctypes.c_uint32,
            ctypes.POINTER(HcclCommConfig),
            ctypes.POINTER(HcclComm),
        ]
        lib.HcclCommInitRootInfoConfig.restype = ctypes.c_int

        lib.HcclCommInitClusterInfoConfig.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.POINTER(HcclCommConfig),
            ctypes.POINTER(HcclComm),
        ]
        lib.HcclCommInitClusterInfoConfig.restype = ctypes.c_int

        lib.HcclCommDestroy.argtypes = [HcclComm]
        lib.HcclCommDestroy.restype = ctypes.c_int
        if hasattr(lib, "HcclGetCommConfigCapability"):
            lib.HcclGetCommConfigCapability.argtypes = []
            lib.HcclGetCommConfigCapability.restype = ctypes.c_uint32
        _BOUND = True
        return lib


def get_comm_config_capability() -> Optional[int]:
    """Return ``HcclGetCommConfigCapability()`` or None if unavailable.

    Soft-fallback only for expected load/bind failures (``OSError`` /
    ``RuntimeError`` from missing lib or domain symbols). Those are *not*
    latched, so a later successful load can still probe. Missing capability
    symbol is latched as ``None``. Unexpected errors propagate.
    """
    global _CAPABILITY_BOUND, _CACHED_CAPABILITY
    if _CAPABILITY_BOUND:
        return _CACHED_CAPABILITY
    try:
        lib = _bind()
    except (OSError, RuntimeError):
        return None
    if hasattr(lib, "HcclGetCommConfigCapability"):
        _CACHED_CAPABILITY = int(lib.HcclGetCommConfigCapability())
    else:
        _CACHED_CAPABILITY = None
    _CAPABILITY_BOUND = True
    return _CACHED_CAPABILITY


def get_root_info():
    """HcclGetRootInfo -> HcclRootInfo."""
    lib = _bind()
    info = HcclRootInfo()
    check(lib.HcclGetRootInfo(ctypes.byref(info)), "HcclGetRootInfo")
    return info


def comm_config_init(
    config=None,
    logical_size=None,
    version=None,
    **field_overrides,
):
    """Python equivalent of inline HcclCommConfigInit (header-only in C).

    Layout selection and capability warnings: see
    ``resolve_comm_config_layout``. Optional field overrides (kwargs):
    buffer_size, deterministic, exec_timeout, qos, op_expansion_mode —
    see fill_comm_config.
    """
    if config is None:
        cfg = HcclCommConfig()
    else:
        cfg = _require_config(config)
    version, logical_size = resolve_comm_config_layout(
        version,
        logical_size,
        capability=get_comm_config_capability(),
        warn_stacklevel=3,
    )
    return fill_comm_config(
        cfg, logical_size=logical_size, version=version, **field_overrides
    )


def comm_init_root_info_config(nranks, root_info, rank, config=None):
    """HcclCommInitRootInfoConfig -> opaque hccl_comm handle (int)."""
    nranks = require_positive_u32("nranks", nranks)
    rank = require_u32("rank", rank)
    if rank >= nranks:
        raise ValueError(f"rank must be < nranks ({nranks}), got {rank}")
    root_info = _require_root_info(root_info)
    config = _require_config(config, ready=True)
    lib = _bind()
    cfg = config if config is not None else comm_config_init()
    comm = HcclComm()
    check(
        lib.HcclCommInitRootInfoConfig(
            ctypes.c_uint32(nranks),
            ctypes.byref(root_info),
            ctypes.c_uint32(rank),
            ctypes.byref(cfg),
            ctypes.byref(comm),
        ),
        "HcclCommInitRootInfoConfig",
    )
    if not comm.value:
        raise RuntimeError("HcclCommInitRootInfoConfig returned null handle")
    return comm.value


def comm_init_cluster_info_config(cluster_info, rank, config=None):
    """HcclCommInitClusterInfoConfig -> opaque hccl_comm handle (int)."""
    cluster_info = _require_cluster_info(cluster_info)
    rank = require_u32("rank", rank)
    config = _require_config(config, ready=True)
    lib = _bind()
    cfg = config if config is not None else comm_config_init()
    path = cluster_info.encode("utf-8") if isinstance(cluster_info, str) else cluster_info
    comm = HcclComm()
    check(
        lib.HcclCommInitClusterInfoConfig(
            path,
            ctypes.c_uint32(rank),
            ctypes.byref(cfg),
            ctypes.byref(comm),
        ),
        "HcclCommInitClusterInfoConfig",
    )
    if not comm.value:
        raise RuntimeError("HcclCommInitClusterInfoConfig returned null handle")
    return comm.value


def comm_destroy(hccl_comm):
    """HcclCommDestroy. Raises on null/negative handle or non-SUCCESS status.

    Returns 0 on success (HcclResult.SUCCESS) for callers that inspect the code.

    ``hccl_comm`` may be ``int`` or ``ctypes.c_void_p``; always normalize to a
    positive int first. Passing a raw ``c_void_p`` into ``HcclComm(...)`` raises
    TypeError, so the normalized address must be what we hand to the C API.
    """
    addr = require_live_handle(hccl_comm, what="comm_destroy")
    lib = _bind()
    code = lib.HcclCommDestroy(HcclComm(addr))
    check(code, "HcclCommDestroy")
    return int(code)


class Comm:
    """Context-manager wrapper around an opaque communicator handle."""

    def __init__(self, handle):
        # Store normalized int so destroy() never rebuilds from a raw c_void_p.
        self._handle = require_live_handle(handle, what="Comm")

    @property
    def handle(self):
        return self._handle

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.destroy()
        return False

    def destroy(self):
        if not self._handle:
            return
        # Clear first so a failed destroy cannot double-call HcclCommDestroy.
        handle = self._handle
        self._handle = None
        comm_destroy(handle)


init_rank_config = comm_init_root_info_config

__all__ = [
    "get_root_info",
    "get_comm_config_capability",
    "comm_config_init",
    "comm_init_root_info_config",
    "comm_init_cluster_info_config",
    "comm_destroy",
    "init_rank_config",
    "Comm",
]
