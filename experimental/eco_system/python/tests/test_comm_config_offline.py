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

"""Device-free checks for HcclCommConfig mirror (offsets / version fields / enums)."""

from __future__ import annotations

import ctypes
import os
import sys
import unittest

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from hcomm4py.types import (  # noqa: E402
    HCCL_COMM_CONFIG_BYTES,
    HCCL_COMM_CONFIG_LOGICAL_SIZE_DEFAULT,
    HCCL_COMM_CONFIG_MAGIC_WORD,
    HCCL_COMM_CONFIG_SIZE_V10,
    HCCL_COMM_CONFIG_SIZE_V11,
    HCCL_COMM_CONFIG_SIZE_V12,
    HCCL_COMM_CONFIG_VERSION_DEFAULT,
    HCCL_COMM_SCQ_DEPTH_CONFIG_NOT_SET,
    HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET,
    HCCL_COMM_DEFAULT_OP_EXPANSION_MODE,
    HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE,
    HCCL_ROOT_INFO_BYTES,
    HcclCommConfig,
    HcclDataType,
    HcclReduceOp,
    HcclResult,
    HcclRootInfo,
    _OFF,
    check,
    fill_comm_config,
)

# Gold for Python overlay offsets (== HcclCommConfigInit layout).
# Key "reserved" here is Init's trailing uint64 @16 inside the blob —
# unrelated to C offsetof(HcclCommConfig, reserved)==0 (see probe below).
_GOLD_OFF = {
    "size": 0,
    "magic": 8,
    "version": 12,
    "reserved": 16,
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


def _u32(cfg: HcclCommConfig, off: int) -> int:
    return ctypes.c_uint32.from_buffer(cfg._raw, off).value


def _u64(cfg: HcclCommConfig, off: int) -> int:
    return ctypes.c_uint64.from_buffer(cfg._raw, off).value


def _size(cfg: HcclCommConfig, off: int) -> int:
    return ctypes.c_size_t.from_buffer(cfg._raw, off).value


class TestCommConfigOffline(unittest.TestCase):
    def test_offsets_gold_and_sizes(self):
        self.assertEqual(set(_OFF), set(_GOLD_OFF))
        for name, expect in _GOLD_OFF.items():
            self.assertEqual(_OFF[name], expect, name)
        prev = -1
        for name, off in sorted(_GOLD_OFF.items(), key=lambda kv: kv[1]):
            self.assertGreater(off, prev, name)
            prev = off
        self.assertEqual(ctypes.sizeof(HcclRootInfo), HCCL_ROOT_INFO_BYTES)
        self.assertEqual(ctypes.sizeof(HcclRootInfo), 4108)
        self.assertEqual(ctypes.sizeof(HcclCommConfig), HCCL_COMM_CONFIG_BYTES)
        self.assertEqual(ctypes.sizeof(HcclCommConfig), 2304)
        self.assertLessEqual(_OFF["hcclChannelScqDepth"] + 4, HCCL_COMM_CONFIG_BYTES)

    def test_default_v11_writes_sq_not_scq_gate(self):
        cfg = fill_comm_config(HcclCommConfig())
        self.assertEqual(_size(cfg, _OFF["size"]), HCCL_COMM_CONFIG_LOGICAL_SIZE_DEFAULT)
        self.assertEqual(_u32(cfg, _OFF["magic"]), HCCL_COMM_CONFIG_MAGIC_WORD)
        self.assertEqual(_u32(cfg, _OFF["version"]), HCCL_COMM_CONFIG_VERSION_DEFAULT)
        self.assertEqual(
            _u64(cfg, _OFF["hcclSymWinMaxMemSizePerRank"]),
            HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE,
        )
        self.assertEqual(
            _u32(cfg, _OFF["hcclChannelSqDepth"]), HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET
        )
        self.assertNotEqual(_u32(cfg, _OFF["hcclChannelSqDepth"]), 0)
        self.assertEqual(_u32(cfg, _OFF["hcclChannelScqDepth"]), 0)

    def test_check_includes_enum_name(self):
        with self.assertRaises(Exception) as ctx:
            check(HcclResult.E_NETWORK, "HcclCommDestroy")
        msg = str(ctx.exception)
        self.assertIn("HcclResult=19", msg)
        self.assertIn("E_NETWORK", msg)

    def test_v12_writes_scq(self):
        cfg = fill_comm_config(
            HcclCommConfig(),
            logical_size=HCCL_COMM_CONFIG_SIZE_V12,
            version=12,
        )
        self.assertEqual(_size(cfg, _OFF["size"]), HCCL_COMM_CONFIG_SIZE_V12)
        self.assertEqual(
            _u32(cfg, _OFF["hcclChannelSqDepth"]), HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET
        )
        self.assertEqual(
            _u32(cfg, _OFF["hcclChannelScqDepth"]), HCCL_COMM_SCQ_DEPTH_CONFIG_NOT_SET
        )

    def test_logical_size_bounds(self):
        def _sentinel_cfg():
            cfg = HcclCommConfig()
            cfg._raw[0] = 0xA5
            cfg._raw[8] = 0x5A
            return cfg

        for kwargs in (
            {"logical_size": 0},
            {"logical_size": 100, "version": 10},
            {"logical_size": HCCL_COMM_CONFIG_BYTES},
            {"logical_size": HCCL_COMM_CONFIG_SIZE_V12 + 1},
        ):
            cfg = _sentinel_cfg()
            with self.assertRaises(ValueError):
                fill_comm_config(cfg, **kwargs)
            self.assertEqual(cfg._raw[0], 0xA5)
            self.assertEqual(cfg._raw[8], 0x5A)

    def test_enums(self):
        self.assertEqual(HcclResult.E_IN_STATUS, 1041)
        self.assertEqual(HcclDataType.RESERVED, 255)
        self.assertEqual(HcclReduceOp.RESERVED, 255)
        self.assertEqual(HCCL_COMM_CONFIG_SIZE_V11, 2248)

    def test_field_overrides_and_abi_constants(self):
        self.assertGreaterEqual(HCCL_COMM_CONFIG_BYTES, HCCL_COMM_CONFIG_SIZE_V12)
        self.assertEqual(HCCL_COMM_CONFIG_SIZE_V11, HCCL_COMM_CONFIG_SIZE_V12)
        self.assertEqual(HCCL_COMM_CONFIG_VERSION_DEFAULT, 11)
        self.assertEqual(
            HCCL_COMM_CONFIG_LOGICAL_SIZE_DEFAULT, HCCL_COMM_CONFIG_SIZE_V11
        )
        cfg = fill_comm_config(
            HcclCommConfig(),
            buffer_size=200,
            deterministic=1,
            exec_timeout=30,
            qos=3,
            op_expansion_mode=2,
        )
        self.assertEqual(_u32(cfg, _OFF["hcclBufferSize"]), 200)
        self.assertEqual(_u32(cfg, _OFF["hcclDeterministic"]), 1)
        self.assertEqual(_u32(cfg, _OFF["hcclExecTimeOut"]), 30)
        self.assertEqual(_u32(cfg, _OFF["hcclQos"]), 3)
        self.assertEqual(_u32(cfg, _OFF["hcclOpExpansionMode"]), 2)
        # Defaults still applied for untouched fields.
        self.assertEqual(
            _u32(cfg, _OFF["hcclChannelSqDepth"]), HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET
        )
        cfg_default = fill_comm_config(HcclCommConfig())
        self.assertEqual(
            _u32(cfg_default, _OFF["hcclOpExpansionMode"]),
            HCCL_COMM_DEFAULT_OP_EXPANSION_MODE,
        )

    def test_version_requires_covering_logical_size(self):
        for kwargs in (
            {"version": 11, "logical_size": 2240},
            {"version": 12, "logical_size": 2244},
        ):
            cfg = HcclCommConfig()
            cfg._raw[0] = 0xA5
            cfg._raw[8] = 0x5A
            with self.assertRaises(ValueError):
                fill_comm_config(cfg, **kwargs)
            self.assertEqual(cfg._raw[0], 0xA5)
            self.assertEqual(cfg._raw[8], 0x5A)

    def test_resolve_layout_from_capability(self):
        import warnings

        from hcomm4py.types import (
            HCCL_COMM_CONFIG_LOGICAL_SIZE_DEFAULT,
            HCCL_COMM_CONFIG_SIZE_V10,
            HCCL_COMM_CONFIG_SIZE_V11,
            HCCL_COMM_CONFIG_VERSION_DEFAULT,
            logical_size_for_version,
            resolve_comm_config_layout,
        )

        self.assertEqual(logical_size_for_version(10), HCCL_COMM_CONFIG_SIZE_V10)
        self.assertEqual(logical_size_for_version(11), HCCL_COMM_CONFIG_SIZE_V11)
        self.assertEqual(
            resolve_comm_config_layout(capability=10),
            (10, HCCL_COMM_CONFIG_SIZE_V10),
        )
        self.assertEqual(
            resolve_comm_config_layout(version=12),
            (12, HCCL_COMM_CONFIG_SIZE_V11),  # v12 shares 2248 padded size
        )
        self.assertEqual(
            resolve_comm_config_layout(version=11, logical_size=2248, capability=10),
            (11, 2248),
        )
        # Out-of-range capability soft-falls back to DEFAULT version (not ValueError).
        for bad_cap in (9, 13, 100):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(
                    resolve_comm_config_layout(capability=bad_cap),
                    (
                        HCCL_COMM_CONFIG_VERSION_DEFAULT,
                        HCCL_COMM_CONFIG_LOGICAL_SIZE_DEFAULT,
                    ),
                )
            self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))
            msg = str(caught[0].message)
            self.assertIn("falling back to version v11", msg)
            self.assertNotIn("logical_size", msg)
            self.assertEqual(
                os.path.abspath(caught[0].filename), os.path.abspath(__file__)
            )
        # Explicit logical_size wins: size-compatible version + version-only warn.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(
                resolve_comm_config_layout(logical_size=2244, capability=13),
                (11, 2244),
            )
        self.assertTrue(caught)
        self.assertIn("falling back to version v11", str(caught[0].message))
        self.assertNotIn("logical_size", str(caught[0].message))
        # 2240–2243 is v10-legal; capability fallback must pick v10 (not fail v11 gate).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(
                resolve_comm_config_layout(logical_size=2240, capability=13),
                (10, 2240),
            )
        self.assertTrue(caught)
        self.assertIn("falling back to version v10", str(caught[0].message))
        # 2248 covers scq; OOR capability must pick highest version v12 (not DEFAULT v11).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(
                resolve_comm_config_layout(logical_size=2248, capability=13),
                (12, 2248),
            )
        self.assertTrue(caught)
        self.assertIn("falling back to version v12", str(caught[0].message))
        # Offline / capability=None: explicit size alone picks highest coverable version.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(
                resolve_comm_config_layout(logical_size=2240),
                (10, 2240),
            )
            self.assertEqual(
                resolve_comm_config_layout(logical_size=2244),
                (11, 2244),
            )
            self.assertEqual(
                resolve_comm_config_layout(logical_size=2248),
                (12, 2248),
            )
        self.assertEqual(caught, [])
        # In-range capability must not override explicit size (no ValueError).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(
                resolve_comm_config_layout(logical_size=2240, capability=12),
                (10, 2240),
            )
            self.assertEqual(
                resolve_comm_config_layout(logical_size=2244, capability=12),
                (11, 2244),
            )
        self.assertEqual(caught, [])
        # Bad explicit size fails before any capability-fallback warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(ValueError):
                resolve_comm_config_layout(logical_size=2200, capability=13)
        self.assertEqual(caught, [])
        # stacklevel=3 path via fill_comm_config points at this test file.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fill_comm_config(HcclCommConfig(), capability=13)
        self.assertTrue(caught)
        self.assertEqual(
            os.path.abspath(caught[0].filename), os.path.abspath(__file__)
        )
        cfg = fill_comm_config(HcclCommConfig(), capability=10)
        self.assertEqual(_u32(cfg, _OFF["version"]), 10)
        self.assertEqual(_size(cfg, _OFF["size"]), HCCL_COMM_CONFIG_SIZE_V10)

    def test_op_expansion_mode_rejects_float_bool(self):
        from hcomm4py.types import HCCL_OP_EXPANSION_MODE_CCU_SCHED

        self.assertEqual(HCCL_OP_EXPANSION_MODE_CCU_SCHED, 6)
        with self.assertRaises(TypeError):
            fill_comm_config(HcclCommConfig(), op_expansion_mode=6.5)
        with self.assertRaises(TypeError):
            fill_comm_config(HcclCommConfig(), op_expansion_mode=True)
        with self.assertRaises(TypeError):
            fill_comm_config(HcclCommConfig(), buffer_size=1.0)

    def test_field_overrides_reject_out_of_range(self):
        with self.assertRaises(ValueError):
            fill_comm_config(HcclCommConfig(), buffer_size=-1)
        with self.assertRaises(ValueError):
            fill_comm_config(HcclCommConfig(), deterministic=-1)
        with self.assertRaises(ValueError):
            fill_comm_config(HcclCommConfig(), qos=-1)
        with self.assertRaises(ValueError):
            fill_comm_config(HcclCommConfig(), qos=2**32)
        with self.assertRaises(ValueError):
            fill_comm_config(HcclCommConfig(), op_expansion_mode=2**32)
        with self.assertRaises(ValueError):
            fill_comm_config(HcclCommConfig(), exec_timeout=2**31)

    def test_version_range_rejected_before_mask(self):
        from hcomm4py.types import resolve_comm_config_layout

        for bad in (-1, 0, 9, 13, 2**32):
            with self.assertRaises(ValueError):
                resolve_comm_config_layout(version=bad)
            with self.assertRaises(ValueError):
                fill_comm_config(HcclCommConfig(), version=bad)
        # Public helper validates logical_size (not only version).
        with self.assertRaises(ValueError):
            resolve_comm_config_layout(version=11, logical_size=0)
        with self.assertRaises(ValueError):
            resolve_comm_config_layout(version=10, logical_size=5)
        with self.assertRaises(TypeError):
            fill_comm_config(HcclCommConfig(), not_a_field=1)

    def test_handle_value_branches(self):
        from hcomm4py._validate import handle_value, is_null_handle, require_live_handle

        self.assertIsNone(handle_value(None))
        self.assertEqual(handle_value(0), 0)
        self.assertEqual(handle_value(ctypes.c_void_p(0xABC)), 0xABC)
        self.assertTrue(is_null_handle(None))
        self.assertTrue(is_null_handle(0))
        self.assertFalse(is_null_handle(1))
        with self.assertRaises(TypeError):
            handle_value(True)
        with self.assertRaises(TypeError):
            handle_value(0.0)
        with self.assertRaises(TypeError):
            handle_value(1.5)
        with self.assertRaises(TypeError):
            handle_value(object())
        with self.assertRaises(ValueError):
            require_live_handle(-1, what="x")
        with self.assertRaises(ValueError):
            require_live_handle(0, what="x")
        self.assertEqual(require_live_handle(0x10), 0x10)

    def test_require_config_ready_positive(self):
        from hcomm4py.comm import _require_config

        cfg = fill_comm_config(HcclCommConfig())
        self.assertIs(cfg, _require_config(cfg, ready=True))
        blank = HcclCommConfig()
        with self.assertRaises(ValueError):
            _require_config(blank, ready=True)

    def test_get_comm_config_capability_soft_fallback(self):
        """Capability bind failure returns None without latching."""
        import hcomm4py.comm as comm
        from unittest import mock

        # Reset module-level cache between cases.
        def _reset():
            comm._CAPABILITY_BOUND = False
            comm._CACHED_CAPABILITY = None

        _reset()
        with mock.patch.object(comm, "_bind", side_effect=OSError("no lib")):
            self.assertIsNone(comm.get_comm_config_capability())
        self.assertFalse(comm._CAPABILITY_BOUND)

        _reset()
        with mock.patch.object(comm, "_bind", side_effect=RuntimeError("no domain")):
            self.assertIsNone(comm.get_comm_config_capability())
        self.assertFalse(comm._CAPABILITY_BOUND)

        _reset()
        with mock.patch.object(comm, "_bind", side_effect=AttributeError("bug")):
            with self.assertRaises(AttributeError):
                comm.get_comm_config_capability()
        self.assertFalse(comm._CAPABILITY_BOUND)

        class _Lib:
            pass

        _reset()
        with mock.patch.object(comm, "_bind", return_value=_Lib()):
            self.assertIsNone(comm.get_comm_config_capability())
        self.assertTrue(comm._CAPABILITY_BOUND)
        self.assertIsNone(comm._CACHED_CAPABILITY)
        with mock.patch.object(comm, "_bind", side_effect=AssertionError("rebind")):
            self.assertIsNone(comm.get_comm_config_capability())

        class _LibCap:
            def HcclGetCommConfigCapability(self):
                return 10

        _reset()
        with mock.patch.object(comm, "_bind", return_value=_LibCap()):
            self.assertEqual(comm.get_comm_config_capability(), 10)
        self.assertTrue(comm._CAPABILITY_BOUND)
        self.assertEqual(comm._CACHED_CAPABILITY, 10)
        _reset()


    def test_offsets_match_header_if_compilable(self):
        """Compile offsetof against include/hccl/hccl_types.h when available."""
        import shutil
        import subprocess
        import tempfile
        from pathlib import Path

        repo = Path(__file__).resolve().parents[4]  # hcomm/
        header = repo / "include" / "hccl" / "hccl_types.h"
        if not header.is_file():
            self.skipTest("hccl_types.h not in checkout")
        if not shutil.which("c++"):
            self.skipTest("no c++ compiler")

        named = [
            "hcclBufferSize",
            "hcclDeterministic",
            "hcclCommName",
            "hcclUdi",
            "hcclOpExpansionMode",
            "hcclRdmaTrafficClass",
            "hcclRdmaServiceLevel",
            "hcclWorldRankID",
            "hcclJobID",
            "aclGraphZeroCopyEnable",
            "hcclExecTimeOut",
            "hcclAlgo",
            "hcclRetryEnable",
            "hcclRetryParams",
            "hcclBufferName",
            "hcclQos",
            "hcclSymWinMaxMemSizePerRank",
            "hcclChannelSqDepth",
            "hcclChannelScqDepth",
        ]
        lines = [
            '#include <stdio.h>',
            '#include <stddef.h>',
            '#include "hccl/hccl_types.h"',
            "int main() {",
            '  printf("sizeof_config %zu\\n", sizeof(HcclCommConfig));',
            '  printf("sizeof_rootinfo %zu\\n", sizeof(HcclRootInfo));',
            # Distinct key from Python _OFF["reserved"] (overlay u64 @16).
            '  printf("c_reserved_array %zu\\n", offsetof(HcclCommConfig, reserved));',
            # Enum values from hccl_types.h (catch gold-table drift vs header).
            '  printf("HCCL_E_IN_STATUS %d\\n", (int)HCCL_E_IN_STATUS);',
            '  printf("HCCL_DATA_TYPE_RESERVED %d\\n", (int)HCCL_DATA_TYPE_RESERVED);',
            '  printf("HCCL_REDUCE_RESERVED %d\\n", (int)HCCL_REDUCE_RESERVED);',
        ]
        for k in named:
            lines.append(f'  printf("{k} %zu\\n", offsetof(HcclCommConfig, {k}));')
        lines.append("  return 0;")
        lines.append("}")
        src = "\n".join(lines) + "\n"

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfile = td / "probe.cc"
            cfile.write_text(src, encoding="utf-8")
            exe = td / "probe"
            proc = None
            for std in ("-std=c++17", "-std=c++14", "-std=c++11"):
                cmd = [
                    "c++",
                    "-x",
                    "c++",
                    std,
                    f"-I{repo / 'include'}",
                    str(cfile),
                    "-o",
                    str(exe),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode == 0:
                    break
            if proc is None or proc.returncode != 0:
                self.fail(
                    "header probe compile failed: %s"
                    % ((proc.stderr if proc else "") or (proc.stdout if proc else ""))[:400]
                )
            out = subprocess.check_output([str(exe)], text=True)
        vals = dict(line.split() for line in out.strip().splitlines())
        # Two different symbols, intentionally different values:
        #   c_reserved_array  = offsetof(HcclCommConfig, reserved)     -> 0
        #   _OFF["reserved"]  = Init overlay uint64 inside that blob   -> 16
        # Do not "fix" by asserting 16 here; that would disagree with the header.
        self.assertEqual(int(vals["c_reserved_array"]), 0)
        self.assertEqual(_OFF["reserved"], 16)
        self.assertEqual(int(vals["hcclBufferSize"]), _OFF["hcclBufferSize"])
        self.assertEqual(int(vals["hcclBufferSize"]), 24)
        for k in named:
            self.assertIn(k, vals)
            self.assertEqual(int(vals[k]), _OFF[k], msg=k)
        # Enum ABI: Python IntEnum mirrors must match header enumerators.
        self.assertEqual(int(vals["HCCL_E_IN_STATUS"]), HcclResult.E_IN_STATUS)
        self.assertEqual(int(vals["HCCL_DATA_TYPE_RESERVED"]), HcclDataType.RESERVED)
        self.assertEqual(int(vals["HCCL_REDUCE_RESERVED"]), HcclReduceOp.RESERVED)
        sizeof_cfg = int(vals["sizeof_config"])
        self.assertGreaterEqual(sizeof_cfg, HCCL_COMM_CONFIG_SIZE_V12)
        self.assertLessEqual(sizeof_cfg, HCCL_COMM_CONFIG_BYTES)
        self.assertEqual(int(vals["sizeof_rootinfo"]), HCCL_ROOT_INFO_BYTES)


if __name__ == "__main__":
    unittest.main()
