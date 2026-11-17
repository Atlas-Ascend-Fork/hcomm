# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared argument validators for hcomm4py (HCOMM domain bindings)."""

from __future__ import annotations

import ctypes
from typing import Any, Optional

_U32_MAX = 0xFFFFFFFF
_U64_MAX = 0xFFFFFFFFFFFFFFFF


def require_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    return value


def require_nonneg(name: str, value: Any) -> int:
    value = require_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def require_positive(name: str, value: Any) -> int:
    value = require_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def require_u32(name: str, value: Any) -> int:
    value = require_nonneg(name, value)
    if value > _U32_MAX:
        raise ValueError(f"{name} must be <= {_U32_MAX} (uint32), got {value}")
    return value


def require_u64(name: str, value: Any) -> int:
    value = require_nonneg(name, value)
    if value > _U64_MAX:
        raise ValueError(f"{name} must be <= {_U64_MAX} (uint64), got {value}")
    return value


def require_positive_u32(name: str, value: Any) -> int:
    value = require_positive(name, value)
    if value > _U32_MAX:
        raise ValueError(f"{name} must be <= {_U32_MAX} (uint32), got {value}")
    return value


def handle_value(handle: Any) -> Optional[int]:
    """Normalize opaque handle / c_void_p / int to an integer address (or None).

    Rejects ``bool`` and ``float`` (including 0.0) — those are not handle types.
    """
    if handle is None:
        return None
    if isinstance(handle, bool):
        raise TypeError("handle must be int/c_void_p, got bool")
    if isinstance(handle, float):
        raise TypeError("handle must be int/c_void_p, got float")
    if isinstance(handle, ctypes.c_void_p):
        return handle.value
    if isinstance(handle, int):
        return handle
    value = getattr(handle, "value", None)
    if isinstance(value, bool):
        raise TypeError("handle.value must not be bool")
    if isinstance(value, int):
        return value
    raise TypeError(f"unsupported handle type: {type(handle).__name__}")


def is_null_handle(handle: Any) -> bool:
    value = handle_value(handle)
    return value is None or value == 0


def require_live_handle(handle: Any, *, what: str = "handle") -> int:
    """Return a positive integer address; reject null/zero/negative fake pointers."""
    value = handle_value(handle)
    if value is None or value == 0:
        raise ValueError(f"{what}: null/zero handle")
    if value < 0:
        raise ValueError(f"{what}: negative handle is not a valid pointer ({value})")
    return value
