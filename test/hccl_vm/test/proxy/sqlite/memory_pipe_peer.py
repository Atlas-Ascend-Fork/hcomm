# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Controlled pipe peer for host request/response tests; it does not simulate memory mapping."""

import os
import struct
import sys


def read_exact(fd, length):
    data = b""
    while len(data) < length:
        chunk = os.read(fd, length - len(data))
        if not chunk:
            raise EOFError("host closed the pipe")
        data += chunk
    return data


def send(fd, command, payload=b""):
    data = struct.pack("<BI", command, len(payload)) + payload
    while data:
        data = data[os.write(fd, data):]


def main():
    name, size, pooled, response = sys.argv[1:5]
    report = sys.argv[5]
    read_fd, write_fd = map(int, sys.argv[-2:])
    requested_names = set()
    free_failed = False
    send(write_fd, 0xA5)  # READY
    while True:
        command, length = struct.unpack("<BI", read_exact(read_fd, 5))
        payload = read_exact(read_fd, length)
        if command == 0:  # SHUTDOWN
            return
        if command not in (2, 4):
            raise ValueError(f"unexpected command: {command}")
        actual_name, actual_size, actual_pool = struct.unpack("<100sQB", payload)
        actual_name = actual_name.split(b"\0", 1)[0].decode()
        if command == 4:
            if actual_name not in requested_names or actual_size != 0 or actual_pool != 0:
                raise ValueError("unexpected memory release")
            with open(report, "a") as log:
                log.write(f"FREE {actual_name}\n")
            status = -1 if response == "free_fail_once" and not free_failed else 0
            free_failed = True
            send(write_fd, 0x84, struct.pack("<i", status))
            continue
        if ((name != "*" and actual_name != name)
                or actual_size != int(size) or actual_pool != int(pooled)):
            raise ValueError("unexpected memory name, size, or pool flag")
        requested_names.add(actual_name)
        with open(report, "a") as log:
            log.write(f"GET {actual_name}\n")
        if response == "eof":
            return
        result = struct.pack("<Q", 0 if response == "null" else 0x700000123000)
        if response == "short":
            result = result[:4]
        send(write_fd, 0xFF if response == "wrong_command" else 0x82, result)


if __name__ == "__main__":
    main()
