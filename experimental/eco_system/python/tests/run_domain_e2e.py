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

"""Minimal 2-rank domain E2E for hcomm4py (GetRootInfo → ConfigInit → Init → Destroy).

Requires ≥2 NPU + CANN env. Offline CI continues to use unittest only.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import sys
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)



def _check_acl(rc, what: str) -> None:
    if int(rc) != 0:
        raise RuntimeError("%s failed rc=%s" % (what, rc))

def _acl():
    from hcomm4py._lib import get_ascendcl

    acl = get_ascendcl()
    acl.aclInit.argtypes = [ctypes.c_char_p]
    acl.aclInit.restype = ctypes.c_int
    acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
    acl.aclrtSetDevice.restype = ctypes.c_int
    acl.aclrtGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    acl.aclrtGetDeviceCount.restype = ctypes.c_int
    return acl


def _mp_collect(q, n, timeout, procs, join_timeout=30.0):
    import queue as _queue

    results = []
    try:
        for _ in range(n):
            try:
                results.append(q.get(timeout=timeout))
            except _queue.Empty:
                results.append(
                    {
                        "rank": "?",
                        "ok": False,
                        "error": "queue.Empty: worker hung or crashed before result",
                    }
                )
    finally:
        for p in procs:
            p.join(timeout=join_timeout)
            if p.is_alive():
                p.terminate()
                p.join(timeout=join_timeout)
            if p.is_alive() and hasattr(p, "kill"):
                p.kill()
                p.join(timeout=join_timeout)
    return results


def _rootinfo_worker(rank, nranks, root_bytes, q):
    try:
        import hcomm4py as h

        acl = _acl()
        _check_acl(acl.aclInit(None), "aclInit")
        _check_acl(acl.aclrtSetDevice(rank), "aclrtSetDevice")
        root = h.HcclRootInfo()
        ctypes.memmove(ctypes.addressof(root), root_bytes, len(root_bytes))
        cfg = h.comm_config_init()
        comm = h.comm_init_root_info_config(nranks, root, rank, config=cfg)
        _check_acl(0 if comm else 1, "comm_init returned null")
        rc = h.comm_destroy(comm)
        _check_acl(rc, "comm_destroy")
        q.put({"rank": rank, "ok": True, "destroy_rc": int(rc)})
    except Exception:
        q.put({"rank": rank, "ok": False, "error": traceback.format_exc()})


def main():
    import hcomm4py as h

    print("=== hcomm4py domain E2E ===")
    acl = _acl()
    _check_acl(acl.aclInit(None), "aclInit")
    n = ctypes.c_uint32()
    _check_acl(acl.aclrtGetDeviceCount(ctypes.byref(n)), "aclrtGetDeviceCount")
    print("device_count=%d" % int(n.value))
    if int(n.value) < 2:
        print("NEED >=2 devices")
        return 2

    _check_acl(acl.aclrtSetDevice(0), "aclrtSetDevice")
    root = h.get_root_info()
    assert ctypes.sizeof(root) > 0
    print("[PASS] get_root_info sizeof=%d" % ctypes.sizeof(root))
    cfg = h.comm_config_init()
    assert ctypes.sizeof(cfg) >= h.HCCL_COMM_CONFIG_SIZE_V10
    print("[PASS] comm_config_init")

    root_bytes = ctypes.string_at(ctypes.addressof(root), ctypes.sizeof(root))
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_rootinfo_worker, args=(r, 2, root_bytes, q)) for r in range(2)
    ]
    for p in procs:
        p.start()
    results = _mp_collect(q, 2, 180, procs)
    for r in results:
        print(r)
    ok = all(r.get("ok") for r in results)
    print("=== SUMMARY domain_e2e failed=%s ===" % ("none" if ok else "rootinfo_init_destroy"))
    return 0 if ok else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
