# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ctypes mirror of hccl_types.h / HcclCommConfigInit.

Public C layout (hccl_types.h)::

    char reserved[24];          # offsetof == 0  (opaque ABI blob)
    uint32_t hcclBufferSize;    # offsetof == 24
    ...

HcclCommConfigInit overlays the blob as (see hccl_comm.h)::

    size_t   size;       # @0
    uint32_t magicWord;  # @8
    uint32_t version;    # @12
    uint64_t reserved;   # @16  <- NOT the C member; trailing pad inside the blob
    # then public fields continue at hcclBufferSize @24

Python ``_OFF`` mirrors that *overlay* (Init writes), not ``offsetof(HcclCommConfig,
reserved)``. So ``_OFF["reserved"] == 16`` and C ``offsetof(..., reserved) == 0``
are both correct and must not be "unified".

Versioned logical sizes:
  v10: logical_size=2240 (through hcclSymWinMaxMemSizePerRank)
  v11: + hcclChannelSqDepth @2240 (padded sizeof commonly 2248) — CANN 9.2
  v12: + hcclChannelScqDepth @2244 (sizeof 2248) — current open-source headers

Offline / fallback defaults follow v11 / logical_size=2248.
Layout priority and capability warnings: see ``resolve_comm_config_layout``.
Library code gates sq/scq reads by the version field; we always write NOT_SET for
any field that the advertised version includes.
"""

import ctypes
import warnings
from enum import IntEnum
from typing import Optional, Tuple


class HcclResult(IntEnum):
    SUCCESS = 0
    E_PARA = 1
    E_PTR = 2
    E_MEMORY = 3
    E_INTERNAL = 4
    E_NOT_SUPPORT = 5
    E_NOT_FOUND = 6
    E_UNAVAIL = 7
    E_SYSCALL = 8
    E_TIMEOUT = 9
    E_OPEN_FILE_FAILURE = 10
    E_TCP_CONNECT = 11
    E_ROCE_CONNECT = 12
    E_TCP_TRANSFER = 13
    E_ROCE_TRANSFER = 14
    E_RUNTIME = 15
    E_DRV = 16
    E_PROFILING = 17
    E_CCE = 18
    E_NETWORK = 19
    E_AGAIN = 20
    E_REMOTE = 21
    E_SUSPENDING = 22
    E_OPRETRY_FAIL = 23
    E_OOM = 24
    E_IN_STATUS = 1041


class HcclReduceOp(IntEnum):
    SUM = 0
    PROD = 1
    MAX = 2
    MIN = 3
    RESERVED = 255


class HcclDataType(IntEnum):
    INT8 = 0
    INT16 = 1
    INT32 = 2
    FP16 = 3
    FP32 = 4
    INT64 = 5
    UINT64 = 6
    UINT8 = 7
    UINT16 = 8
    UINT32 = 9
    FP64 = 10
    BFP16 = 11
    INT128 = 12
    HIF8 = 14
    FP8E4M3 = 15
    FP8E5M2 = 16
    FP8E8M0 = 17
    RESERVED = 255


class HcclSendRecvType(IntEnum):
    SEND = 0
    RECV = 1
    RESERVED = 2


HCCL_ROOT_INFO_BYTES = 4108
HCCL_COMM_CONFIG_BYTES = 2304
HCCL_COMM_CONFIG_MAGIC_WORD = 0xF0F0F0F0
HCCL_COMM_CONFIG_SIZE_V10 = 2240
HCCL_COMM_CONFIG_SIZE_V11 = 2248  # padded; raw end of sqDepth is 2244
HCCL_COMM_CONFIG_SIZE_V12 = 2248
HCCL_COMM_CONFIG_VERSION_DEFAULT = 11
HCCL_COMM_CONFIG_LOGICAL_SIZE_DEFAULT = HCCL_COMM_CONFIG_SIZE_V11

HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET = 0xFFFFFFFF
HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET = 0xFFFFFFFF
HCCL_COMM_DEFAULT_OP_EXPANSION_MODE = 0
# Version domain supported by fill_comm_config / _OFF gates (v10..v12).
HCCL_COMM_CONFIG_VERSION_MIN = 10
HCCL_COMM_CONFIG_VERSION_MAX = 12
# ---------------------------------------------------------------------------
# WONTFIX: keep value 6 (do NOT "align" to hccl_comm.h enumerator 5).
# ---------------------------------------------------------------------------
# ``HCCL_OP_EXPANSION_MODE_CCU_SCHED`` here is the *toolkit-effective*
# ``HcclCommConfig.hcclOpExpansionMode`` field encoding for Ascend 950PR /
# Atlas 350, not the public C enum ordinal in ``include/hccl/hccl_comm.h``
# (``HCCL_OP_EXPANSION_MODE_CCU_SCHED = 5``). Lab (CANN 9.2.0-beta.2): writing
# 6 succeeds; writing header enum 5 or CCU_SCHED_ONLY=8 yields
# ``HcclCommInitRootInfoConfig`` ``E_NOT_SUPPORT``. Value 6 matches
# ``hccl_types.h`` field comment ("6:ccu_sched") and hccl_test accelerator
# table (``-a ccu_sched`` → 6). A5 needs 6; renaming kept optional — export
# name retained for existing call sites, with this warning as the contract.
HCCL_OP_EXPANSION_MODE_CCU_SCHED = 6


def logical_size_for_version(version: int) -> int:
    """Advertised ``HcclCommConfig.size`` for a given config version."""
    v = int(version)
    if v <= 10:
        return HCCL_COMM_CONFIG_SIZE_V10
    if v == 11:
        return HCCL_COMM_CONFIG_SIZE_V11
    return HCCL_COMM_CONFIG_SIZE_V12


HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET = 0xFFFFFFFF
HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET = 0xFFFFFFFF
HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET = 0xFFFFFFFF
HCCL_COMM_QOS_CONFIG_NOT_SET = 0xFFFFFFFF
HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET = 0xFFFFFFFF
HCCL_COMM_SCQ_DEPTH_CONFIG_NOT_SET = 0xFFFFFFFF
HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE = 16

# Overlay offsets into the leading char reserved[24] blob (Init view).
# Do NOT confuse with offsetof(HcclCommConfig, reserved) which is always 0.
_OFF = {
    "size": 0,  # size_t  (Init: info->size)
    "magic": 8,  # uint32 (Init: info->magicWord)
    "version": 12,  # uint32 (Init: info->version)
    "reserved": 16,  # uint64 (Init: info->reserved pad; NOT the C array member)
    "hcclBufferSize": 24,
    "hcclDeterministic": 28,
    "hcclCommName": 32,
    "hcclUdi": 160,
    "hcclOpExpansionMode": 288,
    "hcclRdmaTrafficClass": 292,
    "hcclRdmaServiceLevel": 296,
    "hcclWorldRankID": 300,
    "hcclJobID": 304,
    "aclGraphZeroCopyEnable": 312,
    "hcclExecTimeOut": 316,
    "hcclAlgo": 320,
    "hcclRetryEnable": 1920,
    "hcclRetryParams": 1970,
    "hcclBufferName": 2098,
    "hcclQos": 2228,
    "hcclSymWinMaxMemSizePerRank": 2232,
    "hcclChannelSqDepth": 2240,
    "hcclChannelScqDepth": 2244,
}


class HcclRootInfo(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * HCCL_ROOT_INFO_BYTES)]


class HcclCommConfig(ctypes.Structure):
    _fields_ = [("_raw", ctypes.c_ubyte * HCCL_COMM_CONFIG_BYTES)]


class HcclSendRecvItem(ctypes.Structure):
    # aarch64 layout: int + pad4 + ptr + u64 + int + u32 => 32 bytes
    _fields_ = [
        ("sendRecvType", ctypes.c_int),
        ("buf", ctypes.c_void_p),
        ("count", ctypes.c_uint64),
        ("dataType", ctypes.c_int),
        ("remoteRank", ctypes.c_uint32),
    ]


HcclComm = ctypes.c_void_p
aclrtStream = ctypes.c_void_p

# Highest field always written by fill_comm_config (independent of version gates).
_ALWAYS_WRITTEN_END = (
    _OFF["hcclSymWinMaxMemSizePerRank"] + ctypes.sizeof(ctypes.c_uint64)
)
_SQ_END = _OFF["hcclChannelSqDepth"] + ctypes.sizeof(ctypes.c_uint32)
_SCQ_END = _OFF["hcclChannelScqDepth"] + ctypes.sizeof(ctypes.c_uint32)


def _version_compatible_with_logical_size(logical_size: int) -> int:
    """Highest supported layout version that ``logical_size`` can cover."""
    if logical_size >= _SCQ_END:
        return HCCL_COMM_CONFIG_VERSION_MAX
    if logical_size >= _SQ_END:
        return 11
    return 10


def resolve_comm_config_layout(
    version: Optional[int] = None,
    logical_size: Optional[int] = None,
    *,
    capability: Optional[int] = None,
    warn_stacklevel: int = 2,
) -> Tuple[int, int]:
    """Pick (version, logical_size): explicit args > capability > DEFAULT (v11).

    Explicit ``version`` outside [10, 12] raises ``ValueError``. When
    ``version`` is omitted and ``logical_size`` is set, the highest layout
    version that size can cover is used (e.g. 2240–2243 → v10,
    2244–2247 → v11, 2248 → v12), so ``explicit logical_size > capability``
    even for cap in [10, 12]. Capability is used to *select* version only
    when size is omitted; when ``logical_size`` is omitted and capability is
    out of [10, 12], a ``UserWarning`` is emitted and version falls back to
    DEFAULT v11. With an explicit size, an out-of-range capability still
    emits a version-only ``UserWarning`` while the chosen version stays
    size-derived (not DEFAULT). Offline / bind-failure (``capability=None``)
    is silent. Explicit ``logical_size`` is range-checked before that
    warning. ``warn_stacklevel`` should be 3 when called from
    ``fill_comm_config`` / ``comm_config_init``.
    """
    explicit_version = version is not None
    # Validate explicit logical_size bounds before capability fallback warning.
    if logical_size is not None:
        logical_size = int(logical_size)
        if not (_ALWAYS_WRITTEN_END <= logical_size <= HCCL_COMM_CONFIG_SIZE_V12):
            raise ValueError(
                f"logical_size={logical_size} out of range "
                f"({_ALWAYS_WRITTEN_END}..{HCCL_COMM_CONFIG_SIZE_V12}); "
                "must cover fields fill_comm_config always writes and stay "
                "within ABI size"
            )

    cap_fallback_from: Optional[int] = None
    if version is None:
        if logical_size is not None:
            # Explicit size wins over capability (in-range or OOR).
            version = _version_compatible_with_logical_size(logical_size)
            if capability is not None:
                cap = int(capability)
                if (
                    cap < HCCL_COMM_CONFIG_VERSION_MIN
                    or cap > HCCL_COMM_CONFIG_VERSION_MAX
                ):
                    cap_fallback_from = cap
        elif capability is not None:
            version = int(capability)
        else:
            version = HCCL_COMM_CONFIG_VERSION_DEFAULT
    version = int(version)
    if version < HCCL_COMM_CONFIG_VERSION_MIN or version > HCCL_COMM_CONFIG_VERSION_MAX:
        if explicit_version:
            raise ValueError(
                f"version must be in [{HCCL_COMM_CONFIG_VERSION_MIN}, "
                f"{HCCL_COMM_CONFIG_VERSION_MAX}], got {version}"
            )
        # Capability ordinal is not a layout version (no explicit size above).
        cap_fallback_from = version
        version = HCCL_COMM_CONFIG_VERSION_DEFAULT

    if logical_size is None:
        logical_size = logical_size_for_version(version)
    else:
        if version >= 11 and logical_size < _SQ_END:
            src = "version" if explicit_version else "selected layout version"
            raise ValueError(
                f"{src}={version} needs logical_size>={_SQ_END} "
                f"(got {logical_size})"
            )
        if version >= 12 and logical_size < _SCQ_END:
            src = "version" if explicit_version else "selected layout version"
            raise ValueError(
                f"{src}={version} needs logical_size>={_SCQ_END} "
                f"(got {logical_size})"
            )

    if cap_fallback_from is not None:
        warnings.warn(
            "HcclGetCommConfigCapability()=%s outside supported layout versions "
            "[%s, %s]; falling back to version v%s"
            % (
                cap_fallback_from,
                HCCL_COMM_CONFIG_VERSION_MIN,
                HCCL_COMM_CONFIG_VERSION_MAX,
                version,
            ),
            UserWarning,
            stacklevel=warn_stacklevel,
        )
    return version, int(logical_size)


class HcclError(RuntimeError):
    def __init__(self, code: int, msg: str = ""):
        self.code = int(code)
        try:
            name = HcclResult(self.code).name
        except ValueError:
            name = "UNKNOWN"
        super().__init__(msg or f"HcclResult={self.code} ({name})")


def check(code: int, where: str = "") -> None:
    if int(code) != HcclResult.SUCCESS:
        try:
            name = HcclResult(int(code)).name
        except ValueError:
            name = "UNKNOWN"
        detail = f"HcclResult={int(code)} ({name})"
        raise HcclError(code, f"{where}: {detail}" if where else detail)


def _u32(cfg: HcclCommConfig, off: int, value: int) -> None:
    ctypes.c_uint32.from_buffer(cfg._raw, off).value = value & 0xFFFFFFFF


def _i32(cfg: HcclCommConfig, off: int, value: int) -> None:
    ctypes.c_int32.from_buffer(cfg._raw, off).value = int(value)


def _u64(cfg: HcclCommConfig, off: int, value: int) -> None:
    ctypes.c_uint64.from_buffer(cfg._raw, off).value = int(value)


def _size(cfg: HcclCommConfig, off: int, value: int) -> None:
    ctypes.c_size_t.from_buffer(cfg._raw, off).value = int(value)


def _u8(cfg: HcclCommConfig, off: int, value: int) -> None:
    cfg._raw[off] = value & 0xFF


# Optional field overrides accepted by fill_comm_config / comm_config_init.
# Grouped as kwargs (same pattern as comm_config_init) to keep the positional
# / keyword-only surface small when new config fields are added.
_FIELD_OVERRIDE_KEYS = frozenset(
    {
        "buffer_size",
        "deterministic",
        "exec_timeout",
        "qos",
        "op_expansion_mode",
    }
)


def fill_comm_config(
    cfg: HcclCommConfig,
    *,
    logical_size: Optional[int] = None,
    version: Optional[int] = None,
    capability: Optional[int] = None,
    **overrides,
) -> HcclCommConfig:
    """Fill ``HcclCommConfig`` like C ``HcclCommConfigInit``, with optional overrides.

    Layout selection: see ``resolve_comm_config_layout``. Field overrides
    (``buffer_size``, ``deterministic``, ``exec_timeout``, ``qos``,
    ``op_expansion_mode``) are keyword-only via ``**overrides``.
    """
    from ._validate import require_int, require_u32

    unknown = set(overrides) - _FIELD_OVERRIDE_KEYS
    if unknown:
        raise TypeError(
            "fill_comm_config() got unexpected keyword argument(s): "
            + ", ".join(sorted(unknown))
        )

    version, logical_size = resolve_comm_config_layout(
        version, logical_size, capability=capability, warn_stacklevel=3
    )

    buffer_size = overrides.get("buffer_size")
    deterministic = overrides.get("deterministic")
    exec_timeout = overrides.get("exec_timeout")
    qos = overrides.get("qos")
    op_expansion_mode = overrides.get("op_expansion_mode")

    if buffer_size is not None:
        buffer_size = require_u32("buffer_size", buffer_size)
    if deterministic is not None:
        deterministic = require_u32("deterministic", deterministic)
    if exec_timeout is not None:
        exec_timeout = require_int("exec_timeout", exec_timeout)
        if exec_timeout < -0x80000000 or exec_timeout > 0x7FFFFFFF:
            raise ValueError(
                f"exec_timeout must fit int32, got {exec_timeout}"
            )
    if qos is not None:
        qos = require_u32("qos", qos)
    if op_expansion_mode is not None:
        op_expansion_mode = require_u32("op_expansion_mode", op_expansion_mode)

    ctypes.memset(ctypes.addressof(cfg), 0, ctypes.sizeof(cfg))
    _size(cfg, _OFF["size"], logical_size)
    _u32(cfg, _OFF["magic"], HCCL_COMM_CONFIG_MAGIC_WORD)
    _u32(cfg, _OFF["version"], version)
    _u64(cfg, _OFF["reserved"], 0)
    _u32(cfg, _OFF["hcclBufferSize"], HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET)
    _u32(cfg, _OFF["hcclDeterministic"], HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET)
    _u32(cfg, _OFF["hcclOpExpansionMode"], HCCL_COMM_DEFAULT_OP_EXPANSION_MODE)
    _u32(cfg, _OFF["hcclRdmaTrafficClass"], HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET)
    _u32(cfg, _OFF["hcclRdmaServiceLevel"], HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET)
    _u32(cfg, _OFF["hcclWorldRankID"], 0)
    _u64(cfg, _OFF["hcclJobID"], 0)
    _u8(cfg, _OFF["aclGraphZeroCopyEnable"], 0)
    _i32(cfg, _OFF["hcclExecTimeOut"], HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET)
    _u32(cfg, _OFF["hcclQos"], HCCL_COMM_QOS_CONFIG_NOT_SET)
    _u64(cfg, _OFF["hcclSymWinMaxMemSizePerRank"], HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE)
    if int(version) >= 11:
        _u32(cfg, _OFF["hcclChannelSqDepth"], HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET)
    if int(version) >= 12:
        _u32(cfg, _OFF["hcclChannelScqDepth"], HCCL_COMM_SCQ_DEPTH_CONFIG_NOT_SET)

    if buffer_size is not None:
        _u32(cfg, _OFF["hcclBufferSize"], buffer_size)
    if deterministic is not None:
        _u32(cfg, _OFF["hcclDeterministic"], deterministic)
    if exec_timeout is not None:
        _i32(cfg, _OFF["hcclExecTimeOut"], exec_timeout)
    if qos is not None:
        _u32(cfg, _OFF["hcclQos"], qos)
    if op_expansion_mode is not None:
        _u32(cfg, _OFF["hcclOpExpansionMode"], op_expansion_mode)

    return cfg


__all__ = [
    "HcclResult",
    "HcclReduceOp",
    "HcclDataType",
    "HcclSendRecvType",
    "HcclRootInfo",
    "HcclCommConfig",
    "HcclSendRecvItem",
    "HcclError",
    "HcclComm",
    "aclrtStream",
    "check",
    "fill_comm_config",
    "logical_size_for_version",
    "resolve_comm_config_layout",
    "HCCL_ROOT_INFO_BYTES",
    "HCCL_COMM_CONFIG_BYTES",
    "HCCL_COMM_CONFIG_MAGIC_WORD",
    "HCCL_COMM_CONFIG_SIZE_V10",
    "HCCL_COMM_CONFIG_SIZE_V11",
    "HCCL_COMM_CONFIG_SIZE_V12",
    "HCCL_COMM_CONFIG_VERSION_DEFAULT",
    "HCCL_COMM_CONFIG_LOGICAL_SIZE_DEFAULT",
    "HCCL_COMM_CONFIG_VERSION_MIN",
    "HCCL_COMM_CONFIG_VERSION_MAX",
    "HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET",
    "HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET",
    "HCCL_COMM_DEFAULT_OP_EXPANSION_MODE",
    "HCCL_OP_EXPANSION_MODE_CCU_SCHED",
    "HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET",
    "HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET",
    "HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET",
    "HCCL_COMM_QOS_CONFIG_NOT_SET",
    "HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET",
    "HCCL_COMM_SCQ_DEPTH_CONFIG_NOT_SET",
    "HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE",
]
