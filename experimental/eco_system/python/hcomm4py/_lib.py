# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Load libhcomm.so and ACL runtime (hcomm / communication-domain side).

Library search order (no hardcoded /usr/local/Ascend paths):
  1. HCCL4PY_LIB_DIR / HCCL4PY_HCOMM_LIB / HCCL4PY_ASCENDCL_LIB (explicit override)
  2. ASCEND_LIBRARY_PATH / ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME (+ lib64)
  3. LD_LIBRARY_PATH entries
  4. ctypes.util.find_library / bare soname
"""

import ctypes
import ctypes.util
import os
import threading
from typing import Iterable, Optional

_hcomm: Optional[ctypes.CDLL] = None
_ascendcl: Optional[ctypes.CDLL] = None
_LOCK = threading.Lock()


def _split_paths(value: Optional[str]) -> Iterable[str]:
    if not value:
        return
    for part in value.split(os.pathsep):
        part = part.strip()
        if part:
            yield part


def _env_lib_roots() -> list:
    roots: list = []
    for key in ("HCCL4PY_LIB_DIR", "ASCEND_LIBRARY_PATH"):
        roots.extend(_split_paths(os.environ.get(key)))
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        base = os.environ.get(key)
        if not base:
            continue
        roots.append(os.path.join(base, "lib64"))
        roots.append(os.path.join(base, "lib"))
        # Some toolkits nest arch dirs under the home path.
        roots.append(os.path.join(base, "x86_64-linux", "lib64"))
        roots.append(os.path.join(base, "aarch64-linux", "lib64"))
    roots.extend(_split_paths(os.environ.get("LD_LIBRARY_PATH")))
    # Dedup while preserving order
    seen = set()
    out = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _explicit_override(name: str) -> Optional[str]:
    if "hcomm" in name:
        return os.environ.get("HCCL4PY_HCOMM_LIB") or None
    if name.startswith("libascendcl"):
        return os.environ.get("HCCL4PY_ASCENDCL_LIB") or None
    return None


def _candidates(name: str):
    for r in _env_lib_roots():
        yield os.path.join(r, name)
    short = name.replace(".so", "").replace("lib", "", 1)
    found = ctypes.util.find_library(short)
    if found:
        yield found
    yield name


def _load(name: str) -> ctypes.CDLL:
    explicit = _explicit_override(name)
    if explicit:
        try:
            return ctypes.CDLL(explicit, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            raise OSError(
                f"cannot load {name} from explicit override {explicit!r}: {e}"
            ) from e
    last = None
    tried = []
    for path in _candidates(name):
        tried.append(path)
        try:
            return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            last = e
            continue
    raise OSError(
        f"cannot load {name}: {last}. tried={tried}. "
        "Source CANN set_env.sh (sets ASCEND_HOME_PATH) or set HCCL4PY_LIB_DIR."
    )


def get_hcomm() -> ctypes.CDLL:
    """Load libhcomm.so only (hcomm4py must not depend on libhccl)."""
    global _hcomm
    if _hcomm is None:
        with _LOCK:
            if _hcomm is None:
                _hcomm = _load("libhcomm.so")
    return _hcomm


def get_ascendcl() -> ctypes.CDLL:
    global _ascendcl
    if _ascendcl is None:
        with _LOCK:
            if _ascendcl is None:
                _ascendcl = _load("libascendcl.so")
    return _ascendcl


__all__ = ["get_hcomm", "get_ascendcl"]

