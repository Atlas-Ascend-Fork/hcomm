# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run the host memory tests with private storage and shared-memory roots."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    binary = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="hccl_vm_memory_tests_") as directory:
        root = Path(directory)
        (root / "install/data").mkdir(parents=True)
        (root / "shm").mkdir()
        env = dict(os.environ, HCCL_VM_INSTALL_ROOT=str(root / "install"),
                   HCCL_VM_RESOURCE_ROOT=str(root / "shm"))
        return subprocess.run([binary, *sys.argv[2:]], cwd=root, env=env, timeout=30).returncode


if __name__ == "__main__":
    sys.exit(main())
