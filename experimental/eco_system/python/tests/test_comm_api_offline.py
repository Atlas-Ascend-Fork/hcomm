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

"""Device-free tests for Comm / comm_destroy exception paths."""

from __future__ import annotations

import ctypes
import os
import sys
import unittest
from unittest import mock

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from hcomm4py import comm as comm_mod  # noqa: E402
from hcomm4py.comm import (  # noqa: E402
    Comm,
    comm_config_init,
    comm_destroy,
    comm_init_cluster_info_config,
    comm_init_root_info_config,
)
from hcomm4py.types import HcclComm, HcclCommConfig, HcclRootInfo  # noqa: E402


class TestCommApiOffline(unittest.TestCase):
    def test_comm_rejects_null_handle(self):
        with self.assertRaises(ValueError):
            Comm(None)
        with self.assertRaises(ValueError):
            Comm(0)
        with self.assertRaises(ValueError):
            Comm(HcclComm(0))
        with self.assertRaises(ValueError):
            Comm(-1)
        with self.assertRaises(ValueError):
            comm_destroy(None)
        with self.assertRaises(ValueError):
            comm_destroy(0)
        with self.assertRaises(ValueError):
            comm_destroy(HcclComm(0))
        with self.assertRaises(ValueError):
            comm_destroy(-7)

    def test_comm_init_arg_validation(self):
        with self.assertRaises(ValueError):
            comm_init_root_info_config(0, HcclRootInfo(), 0)
        with self.assertRaises(ValueError):
            comm_init_root_info_config(4, HcclRootInfo(), 4)
        with self.assertRaises(TypeError):
            comm_init_root_info_config(2, "bad", 0)
        with self.assertRaises(TypeError):
            comm_init_root_info_config(2, HcclRootInfo(), 0, config="x")
        with self.assertRaises(ValueError):
            comm_init_cluster_info_config("", 0)
        with self.assertRaises(TypeError):
            comm_init_cluster_info_config(None, 0)
        with self.assertRaises(ValueError):
            comm_init_cluster_info_config(b"/tmp/ranktable.json", -1)
        with self.assertRaises(ValueError):
            comm_init_root_info_config(2, HcclRootInfo(), 0, config=HcclCommConfig())
        with self.assertRaises(ValueError):
            comm_init_cluster_info_config(
                "/tmp/ranktable.json", 0, config=HcclCommConfig()
            )
        with self.assertRaises(TypeError):
            comm_config_init(config="not-a-config")
        cfg = HcclCommConfig()
        out = comm_config_init(config=cfg)
        self.assertIs(out, cfg)

    def test_destroy_idempotent_and_context_manager(self):
        lib = mock.Mock()
        lib.HcclCommDestroy.return_value = 0
        with mock.patch("hcomm4py.comm._bind", return_value=lib):
            c = Comm(0xABC)
            c.destroy()
            c.destroy()
            self.assertEqual(lib.HcclCommDestroy.call_count, 1)

            lib.HcclCommDestroy.reset_mock()
            with self.assertRaises(RuntimeError):
                with Comm(0xDEF) as _:
                    raise RuntimeError("boom")
            self.assertEqual(lib.HcclCommDestroy.call_count, 1)

    def test_destroy_clears_handle_even_on_c_error(self):
        from hcomm4py.types import HcclError

        lib = mock.Mock()
        lib.HcclCommDestroy.return_value = 19  # E_NETWORK
        with mock.patch("hcomm4py.comm._bind", return_value=lib):
            c = Comm(0xABC)
            with self.assertRaises(HcclError):
                c.destroy()
            self.assertIsNone(c.handle)
            c.destroy()
            self.assertEqual(lib.HcclCommDestroy.call_count, 1)

    def test_destroy_accepts_c_void_p_via_normalized_addr(self):
        lib = mock.Mock()
        lib.HcclCommDestroy.return_value = 0
        with mock.patch("hcomm4py.comm._bind", return_value=lib):
            ptr = ctypes.c_void_p(0xABC)
            self.assertEqual(comm_destroy(ptr), 0)
            arg = lib.HcclCommDestroy.call_args[0][0]
            self.assertIsInstance(arg, HcclComm)
            self.assertEqual(int(arg.value), 0xABC)

            c = Comm(ctypes.c_void_p(0xDEF))
            self.assertEqual(c.handle, 0xDEF)
            c.destroy()
            self.assertIsNone(c.handle)
            self.assertEqual(lib.HcclCommDestroy.call_count, 2)

    def test_no_local_hccl4py_package(self):
        local = os.path.join(HERE, "hccl4py")
        self.assertFalse(
            os.path.isdir(local),
            "unexpected local hccl4py package at %s" % local,
        )
        import hcomm4py

        self.assertTrue(hasattr(hcomm4py, "comm_destroy"))
        self.assertTrue(hasattr(hcomm4py, "comm_config_init"))
        self.assertTrue(hasattr(hcomm4py, "get_comm_config_capability"))
        self.assertIn("get_comm_config_capability", hcomm4py.__all__)

    def test_bind_missing_domain_symbol_message(self):
        class _OldLib:
            pass

        with comm_mod._BIND_LOCK:
            was_bound = comm_mod._BOUND
            comm_mod._BOUND = False
        try:
            with mock.patch.object(comm_mod, "get_hcomm", return_value=_OldLib()):
                with self.assertRaises(RuntimeError) as ctx:
                    comm_mod._bind()
            msg = str(ctx.exception)
            self.assertIn("libhcomm.so", msg)
            self.assertIn("HcclGetRootInfo", msg)
            self.assertIn("non-V2", msg)
            self.assertIn("op_base_v2", msg)
            self.assertNotIn("up/9.1.0", msg)
        finally:
            with comm_mod._BIND_LOCK:
                comm_mod._BOUND = was_bound


if __name__ == "__main__":
    unittest.main()
