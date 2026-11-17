#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Minimal communication-domain smoke: get_root_info + init + destroy.

Usage (after source set_env.sh):
  export PYTHONPATH=<this python dir>:$PYTHONPATH
  python3 01_comm_init.py --nranks 2
"""

import argparse
import ctypes
import multiprocessing as mp
import os
import sys


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError(msg)


def _worker(rank: int, nranks: int, root_bytes: bytes) -> None:
    import hcomm4py as h4
    from hcomm4py._lib import get_ascendcl

    acl = get_ascendcl()
    acl.aclInit.argtypes = [ctypes.c_char_p]
    acl.aclInit.restype = ctypes.c_int
    acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
    acl.aclrtSetDevice.restype = ctypes.c_int
    _check(acl.aclInit(None) == 0, "acl.aclInit(None) failed")
    _check(acl.aclrtSetDevice(rank) == 0, "acl.aclrtSetDevice(rank) failed")
    root = h4.HcclRootInfo()
    ctypes.memmove(ctypes.addressof(root), root_bytes, len(root_bytes))
    # Default layout follows runtime capability (legacy v12/2248, V2 v10/2240);
    # offline fallback is v11/2248. Override fields via kwargs when needed:
    #   cfg = h4.comm_config_init(buffer_size=200)
    cfg = h4.comm_config_init()
    with h4.Comm(h4.comm_init_root_info_config(nranks, root, rank, config=cfg)) as comm:
        print(f"rank={rank} handle={comm.handle}")


def main() -> int:
    p = argparse.ArgumentParser(description="hcomm4py communication-domain smoke")
    p.add_argument("--nranks", type=int, default=2)
    args = p.parse_args()
    if args.nranks <= 0:
        p.error("--nranks must be > 0")

    here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if here not in sys.path:
        sys.path.insert(0, here)

    import hcomm4py as h4
    from hcomm4py._lib import get_ascendcl

    acl = get_ascendcl()
    acl.aclInit.argtypes = [ctypes.c_char_p]
    acl.aclInit.restype = ctypes.c_int
    acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
    acl.aclrtSetDevice.restype = ctypes.c_int
    _check(acl.aclInit(None) == 0, "acl.aclInit(None) failed")
    _check(acl.aclrtSetDevice(0) == 0, "acl.aclrtSetDevice(0) failed")
    root = h4.get_root_info()
    root_bytes = ctypes.string_at(ctypes.addressof(root), ctypes.sizeof(root))

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(r, args.nranks, root_bytes)) for r in range(args.nranks)]
    for proc in procs:
        proc.start()
    rc = 0
    for proc in procs:
        proc.join(timeout=180)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive() and hasattr(proc, "kill"):
                proc.kill()
            rc = rc or 1
        elif proc.exitcode:
            rc = proc.exitcode or 1
    return rc


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
