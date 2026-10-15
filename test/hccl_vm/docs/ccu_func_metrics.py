#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Level1 CCU 函数度量扫描（超长/超大/过深）。

口径（去注释/字符串 -> 花括号匹配取函数体）：
  sloc  = 函数体内非空、非注释代码行数；
  cc    = 1 + if + for + while + case + catch + && + || + ?（含 case）；
  depth = 函数体内相对函数起始的最大花括号嵌套深度；
  phys  = 物理行数。

用法：
  python3 docs/ccu_func_metrics.py [目标目录或文件...]
默认扫描 src/proxy/level1/ccu（相对本脚本所在仓库根）。
仅打印“违反指标（SLOC>50 或 CC>=20 或 depth>=5）”的函数；全部达标则无输出。
"""
import os
import re
import sys
import glob

SLOC_LIMIT = 50
CC_LIMIT = 20
DEPTH_LIMIT = 5

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DIR = os.path.join(REPO_ROOT, "src", "proxy", "level1", "ccu")

CTRL = {"if", "for", "while", "switch", "catch", "do", "else", "return",
        "sizeof", "throw", "typedef", "using"}


def strip_comments_strings(src):
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n and src[i + 1] == '/':
            j = i
            while j < n and src[j] != '\n':
                out[j] = ' '
                j += 1
            i = j
        elif c == '/' and i + 1 < n and src[i + 1] == '*':
            j = i
            while j < n and not (src[j] == '*' and j + 1 < n and src[j + 1] == '/'):
                if src[j] != '\n':
                    out[j] = ' '
                j += 1
            if j < n:
                out[j] = ' '
                if j + 1 < n:
                    out[j + 1] = ' '
            i = j + 2
        elif c == '"' or c == "'":
            q = c
            j = i + 1
            while j < n and src[j] != q:
                if src[j] == '\\':
                    j += 1
                if j < n and src[j] != '\n':
                    out[j] = ' '
                j += 1
            if j < n:
                out[j] = ' '
            i = j + 1
        else:
            i += 1
    return "".join(out)


def first_ident(s):
    m = re.match(r"\s*([A-Za-z_]\w*)", s)
    return m.group(1) if m else ""


def sig_tail_ok(s):
    t = s.strip()
    if t == "" or t.startswith("#"):
        return False
    if first_ident(t) in CTRL:
        return False
    if t.endswith(";") or t.endswith("{"):
        return False
    return t.endswith(")") or t.endswith("const")


def analyze(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        src = f.read()
    cl = strip_comments_strings(src)
    cl_lines = cl.split("\n")
    rows = []
    i = 0
    while i < len(cl_lines):
        if cl_lines[i].strip() == "{":
            j = i - 1
            while j >= 0 and cl_lines[j].strip() == "":
                j -= 1
            if j >= 0 and sig_tail_ok(cl_lines[j]):
                s = j
                k = j - 1
                while k >= 0:
                    t = cl_lines[k].strip()
                    if (t == "" or t.endswith(";") or t.endswith("{") or t.endswith("}")
                            or t.startswith("#") or t.endswith(":")
                            or t.startswith(("public", "private", "protected"))):
                        break
                    s = k
                    k -= 1
                depth = 0
                e = i
                while e < len(cl_lines):
                    depth += cl_lines[e].count("{") - cl_lines[e].count("}")
                    if depth <= 0:
                        break
                    e += 1
                body = cl_lines[s:e + 1]
                sloc = sum(1 for ln in body if ln.strip() != "")
                d = 0
                dep = 0
                for ln in body:
                    d += ln.count("{") - ln.count("}")
                    dep = max(dep, d)
                txt = "\n".join(body)
                cc = 1
                cc += len(re.findall(r"\bif\b", txt))
                cc += len(re.findall(r"\bfor\b", txt))
                cc += len(re.findall(r"\bwhile\b", txt))
                cc += len(re.findall(r"\bcase\b", txt))
                cc += len(re.findall(r"\bcatch\b", txt))
                cc += txt.count("&&") + txt.count("||") + txt.count("?")
                nm = first_ident(cl_lines[s])
                for ln in cl_lines[s:j + 1]:
                    m = re.search(r"([A-Za-z_]\w*)\s*\(", ln)
                    if m and m.group(1) not in CTRL:
                        nm = m.group(1)
                        break
                rows.append((os.path.basename(path), nm, s + 1, sloc, e - s + 1, cc, dep))
                i = e + 1
                continue
        i += 1
    return rows


def main():
    targets = sys.argv[1:] or [DEFAULT_DIR]
    files = []
    for t in targets:
        if os.path.isdir(t):
            files += sorted(glob.glob(os.path.join(t, "*.cc")) + glob.glob(os.path.join(t, "*.h")))
        else:
            files.append(t)
    total = 0
    bad = 0
    for f in files:
        rows = analyze(f)
        total += len(rows)
        for r in rows:
            if r[3] > SLOC_LIMIT or r[5] >= CC_LIMIT or r[6] >= DEPTH_LIMIT:
                bad += 1
                print("  %-22s %-40s L%-5d sloc=%-4d phys=%-4d cc=%-4d depth=%d"
                      % (r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
    print("总函数数：%d；违反（sloc>%d 或 cc>=%d 或 depth>=%d）：%d"
          % (total, SLOC_LIMIT, CC_LIMIT, DEPTH_LIMIT, bad))


if __name__ == "__main__":
    main()
