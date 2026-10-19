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

"""doc_lint.py — hcomm-doc-dev software-module 子规范产出文档的确定性校验（Step 5 自动化部分）。

四项检查（对齐 references 规则条文）：
  1. hw-codes   硬件代号扫描：正文出现 CKE/Xn/GSA/MS 即违规（代码块与反引号内除外）
                ——对齐 terminology.md「hcomm 硬件代号规避（强制）」
  2. bilingual  中英结构对齐：两版 H2/H3 标题序列、表格数、Mermaid 图数、代码块数一致
                ——对齐 SKILL.md Step 5 第二层「两版章节结构一致」
  3. line-refs  行号引用实证：`file.cc:123[-145]` 格式引用逐条验证文件存在且行号在界内
                ——对齐 SKILL.md「行号引用实证：grep -n 复核，禁止估算」
  4. mermaid    Mermaid 易错点：classDiagram 泛型未转义 <>、subgraph/alt 内 `end` 歧义、
                节点 ID 含中文标点、classDiagram 方法行含中文括号
                ——对齐 mermaid-guide.md「易错点清单」

用法：
  python3 doc_lint.py <中文文档.md> [--en <英文文档.md>] [--repo-root <仓根>] [--head-sha <sha>]
  python3 doc_lint.py --selftest   # 内置自测（无外部依赖）

说明：
  - --repo-root/--head-sha 提供时行号引用按该版本实证（git show <sha>:<path>）；
    未提供时行号引用仅做格式与本地文件存在性检查（当作仓根为文档所在目录）。
  - 退出码：0 全部通过；1 存在违规。输出为人读报告 + 机器可读 JSON（--json）。
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- 基础工具

HW_CODE_MAP = {
    "CKE": "Event",
    "Xn": "Variable",
    "GSA": "Address",
    "MS": "CcuBuffer",
}

LINE_REF_RE = re.compile(r"`([\w./-]+\.(?:cc|h|cpp|hpp|c)):(\d+)(?:-(\d+))?`")


def split_code_spans(text):
    """把 Markdown 文本切成 (自然语言段, 代码段) 序列。

    代码段 = 围栏代码块（``` / ~~~）与行内反引号 span，二者内的内容不受
    硬件代号与行号检查约束（对齐 terminology.md 的例外条款）。

    返回 (is_code, str, start_lineno)：start_lineno 为该 span 首行在全文中的
    行号（1 起），供违规报告换算文档绝对行号。
    """
    spans = []  # (is_code, str, start_lineno)
    lineno = 1
    # 先切围栏代码块
    fence_re = re.compile(r"(```[\s\S]*?```|~~~[\s\S]*?~~~)")
    for part in fence_re.split(text):
        if not part:
            continue
        if part.startswith(("```", "~~~")):
            spans.append((True, part, lineno))
            lineno += part.count("\n")
        else:
            # 再切行内反引号
            inline_re = re.compile(r"(`[^`\n]*`)")
            for sub in inline_re.split(part):
                if sub:
                    spans.append((sub.startswith("`"), sub, lineno))
                    lineno += sub.count("\n")
    return spans


def extract_mermaid_blocks(text):
    """提取全部 mermaid 围栏代码块内容。"""
    out = []
    for m in re.finditer(r"```mermaid\n([\s\S]*?)```", text):
        out.append(m.group(1))
    return out


def run_git_show(repo_root, sha, path):
    """git show <sha>:<path>，返回行列表；失败返回 None。"""
    cmd = ["git", "-C", str(repo_root), "show", "{}:{}".format(sha, path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.splitlines()


def resolve_bare_filename(repo_root, fname):
    """裸文件名（不带路径）时用 git ls-files 定位；唯一匹配返回仓相对路径，否则 None。

    存量文档实践是裸文件名引用（如 `hcclCommTaskExceptionLite.cc:429-435`），
    配合上下文说明定位。匹配语义为 basename 全等（排除 ut_ 前缀测试文件等同尾缀误配）；
    唯一匹配即可实证，多匹配/零匹配视为不可定位。
    """
    cmd = ["git", "-C", str(repo_root), "ls-files"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    matches = [l.strip() for l in r.stdout.splitlines()
               if l.strip() and l.strip().rsplit("/", 1)[-1] == fname]
    return matches[0] if len(matches) == 1 else None


# ---------------------------------------------------------------- 检查 1：硬件代号

def check_hw_codes(text):
    """正文（代码块与反引号外）出现硬件代号即违规。报告含文档绝对行号。"""
    findings = []
    for is_code, span, start_lineno in split_code_spans(text):
        if is_code:
            continue
        for lineno, line in enumerate(span.splitlines(), 1):
            for code in HW_CODE_MAP:
                # 词边界：CKE/Xn/GSA/MS 作为独立词出现（避免误伤 GsaXxx 变量名等）
                if re.search(r"\b{}\b".format(re.escape(code)), line):
                    findings.append({
                        "check": "hw-codes",
                        "severity": "ERROR",
                        "line": start_lineno + lineno - 1,
                        "line_hint": line.strip()[:60],
                        "message": "第 {} 行：正文出现内部硬件代号 {}，应替换为 API 类名 {}".format(
                            start_lineno + lineno - 1, code, HW_CODE_MAP[code]),
                    })
    return findings


# ---------------------------------------------------------------- 检查 2：中英结构对齐

def doc_skeleton(text):
    """抽取结构指纹：H2/H3 标题层级序列、表格张数、Mermaid 图数、代码块数。

    表格张数按分隔行（|---|---| 形态）计数——每张表恰有一行分隔行；
    逐行数管道符会把多张表的总行数混淆（2×3+4 行与 3×2+2+3 行同为 10 行）。
    标题取层级序列（[2,3,2,...]），可抓"数量一致但顺序不同"的结构漂移；
    中英标题文本不映射（terminology.md 对照是维护参考，不进确定性校验）。
    """
    headings = [len(h.group(1)) for h in re.finditer(r"^(#{2,3}) .+$", text, re.M)]
    tables = len(re.findall(r"^\|[-| :]+\|$", text, re.M))
    mermaid = len(extract_mermaid_blocks(text))
    codeblocks = len(re.findall(r"```", text)) // 2
    return {"headings": headings, "tables": tables, "mermaid": mermaid,
            "codeblocks": codeblocks}


def check_bilingual(zh_text, en_text):
    """两版结构指纹比对。中文版缺失英文版时跳过（单独跑其他检查）。"""
    if not en_text:
        return []
    findings = []
    zh, en = doc_skeleton(zh_text), doc_skeleton(en_text)
    if zh["tables"] != en["tables"]:
        findings.append({
            "check": "bilingual",
            "severity": "ERROR",
            "message": "表格数不一致：中文 {} 张 / 英文 {} 张".format(zh["tables"], en["tables"]),
        })
    if zh["mermaid"] != en["mermaid"]:
        findings.append({
            "check": "bilingual",
            "severity": "ERROR",
            "message": "Mermaid 图数不一致：中文 {} 张 / 英文 {} 张".format(zh["mermaid"], en["mermaid"]),
        })
    if zh["codeblocks"] != en["codeblocks"]:
        findings.append({
            "check": "bilingual",
            "severity": "WARN",
            "message": "代码块数不一致：中文 {} / 英文 {}（目录树/图代码块应原样复用）".format(
                zh["codeblocks"], en["codeblocks"]),
        })
    if zh["headings"] != en["headings"]:
        if len(zh["headings"]) != len(en["headings"]):
            findings.append({
                "check": "bilingual",
                "severity": "ERROR",
                "message": "H2/H3 标题数不一致：中文 {} / 英文 {}".format(
                    len(zh["headings"]), len(en["headings"])),
            })
        else:
            findings.append({
                "check": "bilingual",
                "severity": "ERROR",
                "message": "H2/H3 标题层级序列不一致：中文 {} / 英文 {}（数量一致但顺序不同）".format(
                    zh["headings"], en["headings"]),
            })
    return findings


# ---------------------------------------------------------------- 检查 3：行号引用实证

def check_line_refs(text, repo_root=None, head_sha=None):
    """提取 `file.cc:123[-145]` 引用，验证文件存在且行号在界内。"""
    findings = []
    # 行号引用约定写在反引号内，直接全文提取
    refs = LINE_REF_RE.findall(text)
    seen = set()
    for fname, start, end in refs:
        key = (fname, start, end or "")
        if key in seen:
            continue
        seen.add(key)
        if repo_root and head_sha:
            # 裸文件名（不含 /）先解析为仓相对路径；带路径的直接用
            path = fname
            if "/" not in fname:
                resolved = resolve_bare_filename(repo_root, fname)
                if resolved is None:
                    findings.append({
                        "check": "line-refs",
                        "severity": "ERROR",
                        "message": "行号引用裸文件名无法唯一定位：{}（零或多个同名文件）".format(fname),
                    })
                    continue
                path = resolved
            lines = run_git_show(repo_root, head_sha, path)
            if lines is None:
                findings.append({
                    "check": "line-refs",
                    "severity": "ERROR",
                    "message": "行号引用文件不存在：{}（sha {}）".format(path, head_sha[:12]),
                })
                continue
            total = len(lines)
            start_i, end_i = int(start), int(end) if end else int(start)
            if not (1 <= start_i <= total and 1 <= end_i <= total):
                findings.append({
                    "check": "line-refs",
                    "severity": "ERROR",
                    "message": "行号越界：{}:{}-{}（该文件共 {} 行）".format(
                        fname, start_i, end_i, total),
                })
        else:
            # 降级：仅格式检查（已在提取时保证），提示未实证
            findings.append({
                "check": "line-refs",
                "severity": "WARN",
                "message": "行号引用未实证（未提供 --repo-root/--head-sha）：{}:{}{}".format(
                    fname, start, "-" + end if end else ""),
            })
    return findings


# ---------------------------------------------------------------- 检查 4：Mermaid 易错点

def check_mermaid(text):
    """四类高发渲染失败点。"""
    findings = []
    for idx, block in enumerate(extract_mermaid_blocks(text), 1):
        is_class = "classDiagram" in block
        for lineno, line in enumerate(block.splitlines(), 1):
            # a) classDiagram 泛型未转义 <>
            if is_class and re.search(r"\w<\w", line):
                findings.append({
                    "check": "mermaid",
                    "severity": "ERROR",
                    "message": "图{} 第{}行：classDiagram 泛型 <> 应转义为 ~（{}）".format(
                        idx, lineno, line.strip()[:50]),
                })
            # b) `end` 关键字歧义——对齐 mermaid-guide.md 易错点第 3 条的三类真实形态：
            #    ① Note 文本以 end 开头（含 end 后接非词字符，如 "end。" / "end 结束"）；
            #       end 后紧跟字母（endX）时 Mermaid 按整词解析 label 文本，无歧义，不报。
            #    ② graph 节点 label 以 end 开头（A[end 阶段] / A["end 阶段"]）。
            #    ③ subgraph 名以 end 开头（subgraph endGroup）——其后闭合 end 产生解析歧义。
            if re.search(r"(Note|label)\b.*:\s*end(?=\W|$)", line):
                findings.append({
                    "check": "mermaid",
                    "severity": "ERROR",
                    "message": "图{} 第{}行：Note/label 文本以 end 开头，关键字歧义".format(idx, lineno),
                })
            if re.search(r"\[\s*[\"']?end(?=\W|$)", line):
                findings.append({
                    "check": "mermaid",
                    "severity": "ERROR",
                    "message": "图{} 第{}行：节点 label 以 end 开头，关键字歧义".format(idx, lineno),
                })
            if re.search(r"^\s*subgraph\s+end\w*", line, re.M):
                findings.append({
                    "check": "mermaid",
                    "severity": "ERROR",
                    "message": "图{} 第{}行：subgraph 名以 end 开头，关键字歧义".format(idx, lineno),
                })
            # c) classDiagram 方法行含中文括号
            if is_class and re.search(r"[（）（）]", line):
                findings.append({
                    "check": "mermaid",
                    "severity": "ERROR",
                    "message": "图{} 第{}行：classDiagram 方法行含中文括号（{}）".format(
                        idx, lineno, line.strip()[:50]),
                })
        # d) graph 节点 ID 含中文标点——对齐 mermaid-guide.md 易错点第 5 条。
        #    捕获 [ 前的完整非空白形态（\w+ 不含中文标点会恒不命中），
        #    再在 ID 内查中文标点；边标签（|--文字--|）内的中文标点合规，不查。
        for m in re.finditer(r"^\s*([^\s\[]+)\s*\[", block, re.M):
            if re.search(r"[，：；、]", m.group(1)):
                findings.append({
                    "check": "mermaid",
                    "severity": "ERROR",
                    "message": "图{}：节点 ID 含中文标点（{}）".format(idx, m.group(1)),
                })
    return findings


# ---------------------------------------------------------------- 主流程

def lint(zh_path, en_path=None, repo_root=None, head_sha=None):
    zh_text = Path(zh_path).read_text(encoding="utf-8")
    en_text = Path(en_path).read_text(encoding="utf-8") if en_path else None
    findings = []
    findings += check_hw_codes(zh_text)
    if en_text:
        findings += check_hw_codes(en_text)
        findings += check_bilingual(zh_text, en_text)
    findings += check_line_refs(zh_text, repo_root, head_sha)
    if en_text:
        findings += check_line_refs(en_text, repo_root, head_sha)
    findings += check_mermaid(zh_text)
    if en_text:
        findings += check_mermaid(en_text)
    return findings


def selftest():
    """内置自测：违规样本逐项命中 + 干净样本零误报。"""
    dirty = """# 模块代码解析

## 功能描述
检查 CKE 期望值，Xn 与 GSA 均需规避；`g_communicatorCallbackMapV2` 内的 MS 除外。
引用源码：`src/foo.cc:99999-100000`

## 数据描述
```mermaid
classDiagram
    class Foo {
        +Process(vector<CcuErrorInfo>) void
    }
```
"""
    clean = """# 模块代码解析

## 功能描述
检查 Event 期望值。引用源码格式见下。

## 数据描述
```mermaid
classDiagram
    class Foo {
        +Process(vector~CcuErrorInfo~) void
    }
```
"""
    errors = []
    f_dirty = check_hw_codes(dirty) + check_mermaid(dirty)
    # 期望：正文 4 个代号（CKE/Xn/GSA/MS——MS 在反引号 span 之外属正文）+ 泛型未转义 1 处
    hw = [f for f in f_dirty if f["check"] == "hw-codes"]
    mm = [f for f in f_dirty if f["check"] == "mermaid"]
    if len(hw) != 4:
        errors.append("hw-codes 期望 4 条命中，实际 {}: {}".format(len(hw), hw))
    # 每条命中须含文档绝对行号（第 4 行）
    if any(f.get("line") != 4 for f in hw):
        errors.append("hw-codes 行号期望均为 4，实际 {}".format([f.get("line") for f in hw]))
    if len(mm) != 1:
        errors.append("mermaid 期望 1 条命中，实际 {}: {}".format(len(mm), mm))
    # 行号引用：格式合法（无 repo_root 时 WARN）
    lr = check_line_refs(dirty)
    if not any(f["check"] == "line-refs" for f in lr):
        errors.append("line-refs 期望至少 1 条 WARN，实际 {}".format(lr))
    f_clean = check_hw_codes(clean) + check_mermaid(clean)
    if f_clean:
        errors.append("clean 样本期望零命中，实际 {}".format(f_clean))
    # Mermaid 易错点扩展形态：节点 ID 中文标点 / 节点 label 以 end 开头 / subgraph end 名
    mm_dirty = check_mermaid("```mermaid\ngraph TB\n  节点，A[x]\n  B[end 阶段]\n```\n"
                             "```mermaid\ngraph TB\n  subgraph endGroup\n    A[x]\n  end\n```\n")
    if len(mm_dirty) != 3:
        errors.append("mermaid 扩展形态期望 3 条命中（节点ID标点/label end/subgraph end），"
                      "实际 {}: {}".format(len(mm_dirty), mm_dirty))
    # 双语对齐：表格张数按分隔行计数（行数相同张数不同必须报）
    zh_tab = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
    en_tab = "| a | b |\n|---|---|\n| 1 | 2 |\n\n| c | d |\n|---|---|\n| 3 | 4 |\n"
    bi_tab = check_bilingual(zh_tab, en_tab)
    if not any("表格数不一致" in f["message"] for f in bi_tab):
        errors.append("bilingual 表格张数期望报不一致（1 张 vs 2 张，行数同为 4），实际 {}".format(bi_tab))
    # 双语对齐：标题层级序列（数量一致顺序不同必须报）
    bi_seq = check_bilingual("## A\n### B\n", "### A'\n## B'\n")
    if not any("层级序列不一致" in f["message"] for f in bi_seq):
        errors.append("bilingual 标题层级序列期望报不一致（[2,3] vs [3,2]），实际 {}".format(bi_seq))
    # 双语对齐（原有用例）
    zh_ok = "## A\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```mermaid\ngraph TB\n  A-->B\n```\n"
    en_bad = "## A\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    bi = check_bilingual(zh_ok, en_bad)
    # 少一张图 = 图数 ERROR + 代码块数 WARN，两条都应出现
    bi_err = [f for f in bi if f["severity"] == "ERROR"]
    bi_warn = [f for f in bi if f["severity"] == "WARN"]
    if len(bi_err) != 1 or len(bi_warn) != 1:
        errors.append("bilingual 期望 1 ERROR + 1 WARN，实际 {}".format(bi))
    if errors:
        print("SELFTEST FAILED:")
        for e in errors:
            print(" -", e)
        return 1
    print("SELFTEST PASSED: hw-codes/mermaid/line-refs/bilingual 全部符合预期")
    return 0


def main():
    ap = argparse.ArgumentParser(description="hcomm-doc-dev software-module 文档确定性校验")
    ap.add_argument("zh_doc", nargs="?", help="中文文档路径（模块名.md）")
    ap.add_argument("--en", help="英文文档路径（模块名_en.md）")
    ap.add_argument("--repo-root", help="源码仓根（行号实证用）")
    ap.add_argument("--head-sha", help="head 版本 sha（行号实证用）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--selftest", action="store_true", help="运行内置自测")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.zh_doc:
        ap.print_help()
        return 2

    findings = lint(args.zh_doc, args.en, args.repo_root, args.head_sha)
    errors = [f for f in findings if f["severity"] == "ERROR"]
    warns = [f for f in findings if f["severity"] == "WARN"]

    if args.json:
        print(json.dumps(findings, ensure_ascii=False, indent=2))
    else:
        for f in findings:
            print("[{}] {} | {}".format(f["severity"], f["check"], f["message"]))
        print("-" * 40)
        print("共 {} 条违规（ERROR {} / WARN {}）".format(len(findings), len(errors), len(warns)))

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
