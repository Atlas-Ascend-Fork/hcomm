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

"""Device-free coverage for hcomm4py._lib search / load priority."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from hcomm4py import _lib as libmod  # noqa: E402


_ENV_KEYS = (
    "HCCL4PY_HCOMM_LIB",
    "HCCL4PY_ASCENDCL_LIB",
    "HCCL4PY_LIB_DIR",
    "ASCEND_LIBRARY_PATH",
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "LD_LIBRARY_PATH",
)


class TestLibOffline(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        libmod._hcomm = None
        libmod._ascendcl = None

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        libmod._hcomm = None
        libmod._ascendcl = None

    def test_explicit_override_priority(self):
        """Mirror hccl test_lib_candidates_explicit_override: env path wins."""
        os.environ["HCCL4PY_HCOMM_LIB"] = "/tmp/hcomm4py_explicit_libhcomm.so"
        self.assertEqual(
            libmod._explicit_override("libhcomm.so"),
            "/tmp/hcomm4py_explicit_libhcomm.so",
        )
        with self.assertRaises(OSError) as cm:
            libmod._load("libhcomm.so")
        self.assertIn("explicit override", str(cm.exception))
        self.assertIn("/tmp/hcomm4py_explicit_libhcomm.so", str(cm.exception))

        os.environ.pop("HCCL4PY_HCOMM_LIB", None)
        os.environ["HCCL4PY_ASCENDCL_LIB"] = "/tmp/hcomm4py_explicit_libascendcl.so"
        self.assertEqual(
            libmod._explicit_override("libascendcl.so"),
            "/tmp/hcomm4py_explicit_libascendcl.so",
        )
        with self.assertRaises(OSError) as cm2:
            libmod._load("libascendcl.so")
        self.assertIn("explicit override", str(cm2.exception))

    def test_env_lib_roots_order_and_dedup(self):
        os.environ["HCCL4PY_LIB_DIR"] = "/opt/hccl4py_lib"
        os.environ["ASCEND_LIBRARY_PATH"] = "/opt/ascend_lib"
        os.environ["ASCEND_HOME_PATH"] = "/opt/ascend_home"
        os.environ["ASCEND_TOOLKIT_HOME"] = "/opt/toolkit"
        # Duplicate first entry via LD_LIBRARY_PATH to exercise dedup.
        os.environ["LD_LIBRARY_PATH"] = "/opt/hccl4py_lib" + os.pathsep + "/opt/ld"
        roots = libmod._env_lib_roots()
        self.assertEqual(roots[0], "/opt/hccl4py_lib")
        self.assertEqual(roots[1], "/opt/ascend_lib")
        self.assertIn(os.path.join("/opt/ascend_home", "lib64"), roots)
        self.assertIn(os.path.join("/opt/toolkit", "lib64"), roots)
        self.assertIn("/opt/ld", roots)
        self.assertEqual(roots.count("/opt/hccl4py_lib"), 1)

        # Empty ASCEND_LIBRARY_PATH must not inject a blank root.
        os.environ["ASCEND_LIBRARY_PATH"] = ""
        roots2 = libmod._env_lib_roots()
        self.assertNotIn("", roots2)

    def test_candidates_lead_with_env_path(self):
        os.environ["HCCL4PY_LIB_DIR"] = "/fake/hcomm4py_root"
        cands = list(libmod._candidates("libhcomm.so"))
        self.assertTrue(cands)
        self.assertEqual(
            cands[0],
            os.path.join("/fake/hcomm4py_root", "libhcomm.so"),
        )
        self.assertIn("libhcomm.so", cands[-1])

    def test_load_failure_lists_tried(self):
        os.environ["HCCL4PY_LIB_DIR"] = "/no/such/hcomm4py_lib_dir"
        with mock.patch.object(libmod.ctypes, "CDLL", side_effect=OSError("boom")):
            with mock.patch.object(
                libmod.ctypes.util, "find_library", return_value=None
            ):
                with self.assertRaises(OSError) as cm:
                    libmod._load("libhcomm.so")
        msg = str(cm.exception)
        self.assertIn("cannot load libhcomm.so", msg)
        self.assertIn("tried=", msg)
        self.assertIn("hcomm4py_lib_dir", msg)
        self.assertIn("libhcomm.so", msg)
        self.assertIn("HCCL4PY_LIB_DIR", msg)


if __name__ == "__main__":
    unittest.main()
