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

五项检查（对齐 references 规则条文）：
  1. hw-codes   硬件代号扫描：正文出现 CKE/Xn/GSA/MS 即违规（代码块与反引号内除外）
                ——对齐 terminology.md「hcomm 硬件代号规避（强制）」
  2. bilingual  中英结构对齐：两版 H2/H3 标题序列、表格数、Mermaid 图数、代码块数一致
                ——对齐 SKILL.md Step 5 第二层「两版章节结构一致」
  3. line-refs  行号引用实证：`file.cc:123[-145]` 格式引用逐条验证文件存在且行号在界内
                ——对齐 SKILL.md「行号引用实证：grep -n 复核，禁止估算」
  4. mermaid    Mermaid 易错点：classDiagram 泛型未转义 <>、subgraph/alt 内 `end` 歧义、
                节点 ID 含中文标点、classDiagram 方法行含中文括号
                ——对齐 mermaid-guide.md「易错点清单」
  5. pos-dep    定位与依赖声明结构检查：目录描述段四标签（层级归属/编译归属/依赖头/
                被谁调用）齐全，或显式整段不适用声明——声明行以「定位与依赖声明」
                开头（容忍加粗与短前置语），不适用/不涉及 + 同行原因（EN 按标准译名
                检查，Not applicable + 原因）
                ——对齐 writing-rules.md §2「定位与依赖声明（必选）」

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


# ---------------------------------------------------------------- 检查 5：定位与依赖声明

POS_DEP_SECTION_ZH = re.compile(r"^##\s*目录描述\s*$", re.M)
POS_DEP_SECTION_EN = re.compile(r"^##\s*Directory\s+Description\s*$", re.M)
POS_DEP_LABELS_ZH = ("层级归属", "编译归属", "依赖头", "被谁调用")
POS_DEP_LABELS_EN = ("Layer Affiliation", "Build Affiliation",
                     "Dependency Headers", "Upstream Callers")
# 整段豁免声明（行级识别）：行首「定位与依赖声明」（容忍加粗）+ 冒号 +
# 可选短前置语（本目录/本模块/本组件/本文档；EN: this/the + directory/module/
# component/document/doc + 可选 is）+ 不适用/不涉及 + 同行原因。
# 短前置语之外的同行任意文本（如指导语「四要素缺一不可，确实不适用时……」）
# 不构成声明；分隔符一律 [ \t]（含换行的 \s 会把下一行文本吞成原因）。
POS_DEP_WHOLE_NA_LINE_ZH = re.compile(
    r"[ \t]*\*{0,2}定位与依赖声明\*{0,2}[ \t]*[:：][ \t]*"
    r"(?:本(?:目录|模块|组件|文档)?[ \t]*)?不(?:适用|涉及)(?P<rest>[^\n]*)")
POS_DEP_WHOLE_NA_LINE_EN = re.compile(
    r"[ \t]*\*{0,2}Location\s+and\s+Dependency\s+Statement\*{0,2}[ \t]*[:：][ \t]*"
    r"(?:(?:this|the)\s+(?:directory|module|component|document|doc)\s+(?:is\s+)?)?"
    r"not\s+(?:applicable|involved|n/a|na(?![A-Za-z]))(?P<rest>[^\n]*)", re.I)
_NA_REASON_STRIP_RE = re.compile(r"[\s。.;；，,、（）()：:！!？?\-—]")


def _na_reason(rest):
    """从「不适用/不涉及」之后的同行剩余文本提取原因；仅句读/空返回 ''。"""
    rest = (rest or "").strip()
    if not rest:
        return ""
    pm = re.match(r"^[（(]\s*([^（）()]+?)\s*[)）]", rest)
    if pm:
        return pm.group(1)
    tm = re.match(r"^[:：，,、]?\s*(.+)$", rest)
    return tm.group(1) if tm else rest


def _na_reason_ok(reason):
    """不适用声明的原因为非空：去掉句读/括号/空白后至少 2 个实质字符。"""
    return len(_NA_REASON_STRIP_RE.sub("", reason or "")) >= 2


def check_pos_dep(text, lang="zh"):
    """目录描述段必含「定位与依赖声明」：四标签齐全，或显式整段不适用声明。

    标签统一为 层级归属/编译归属/依赖头/被谁调用（EN 标准译名见
    terminology.md）。标签级「<标签>：不适用/不涉及」视为该标签已声明
    （标签字符串在场即算满足），但不豁免其他标签。

    整段豁免为强条件（对齐 writing-rules.md §2「确实不适用时须在同一行
    写明不适用及原因」）：声明行之外四标签全缺，且存在以「定位与依赖声明」
    开头、冒号后仅允许短前置语（本目录/本模块等）、明示不适用/不涉及且
    **原因同行**的显式声明——原因中提及标签名不计为标签在场；指导语
    （「四要素缺一不可，确实不适用时……」）、模板注释、段内其他位置的
    「不适用/不涉及」字样均不构成豁免。
    """
    if lang == "zh":
        sec_re, labels = POS_DEP_SECTION_ZH, POS_DEP_LABELS_ZH
        na_line_re = POS_DEP_WHOLE_NA_LINE_ZH
    else:
        sec_re, labels = POS_DEP_SECTION_EN, POS_DEP_LABELS_EN
        na_line_re = POS_DEP_WHOLE_NA_LINE_EN
    findings = []
    m = sec_re.search(text)
    if not m:
        findings.append({
            "check": "pos-dep",
            "severity": "ERROR",
            "message": "未找到目录描述章节，无法核对定位与依赖声明",
        })
        return findings
    nxt = re.search(r"^##\s", text[m.end():], re.M)
    section = text[m.end():m.end() + nxt.start()] if nxt else text[m.end():]
    # 行级识别整段不适用声明；标签在场统计剔除声明行（原因中的标签名不计）
    lines = section.splitlines()
    na_idx, na_reasons = set(), []
    for i, ln in enumerate(lines):
        mm = na_line_re.match(ln)
        if mm:
            na_idx.add(i)
            na_reasons.append(_na_reason(mm.group("rest")))
    body = "\n".join(ln for i, ln in enumerate(lines) if i not in na_idx)
    if lang == "zh":
        missing = [l for l in labels if l not in body]
    else:
        low = body.lower()
        missing = [l for l in labels if l.lower() not in low]
    if not missing:
        return findings
    if na_idx and len(missing) == len(labels):
        # 声明行之外四标签全缺：豁免与否取决于声明是否带同行原因
        if any(_na_reason_ok(r) for r in na_reasons):
            return findings
        findings.append({
            "check": "pos-dep",
            "severity": "ERROR",
            "message": "整段不适用声明缺原因或原因未与声明同行（须在同一行写明，如：定位与依赖声明：不适用（纯头文件发布目录，无独立编译目标））",
        })
        return findings
    findings.append({
        "check": "pos-dep",
        "severity": "ERROR",
        "message": "目录描述段缺「定位与依赖声明」要素：{}（四标签齐全；或四标签全缺时在同一行写明「定位与依赖声明：不适用/不涉及」及原因）".format(
            "、".join(missing)),
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
    findings += check_pos_dep(zh_text, "zh")
    if en_text:
        findings += check_pos_dep(en_text, "en")
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
    # 定位与依赖声明：四标签齐全 / 缺声明 / 整段豁免（含原因）/ 无原因不豁免 /
    # 模板注释不豁免（评审反例）/ 单要素不适用不连带豁免 / 章节缺失 / EN
    pd_full = ("## 目录描述\n\n**层级归属**：L2；**编译归属**：`hcomm_coll_comm_obj`；"
               "**依赖头**：`*_pub.h`；**被谁调用**：coll_comm。\n\n## 流程描述\n")
    pd_miss = "## 目录描述\n\n目录树与关系表。\n\n## 流程描述\n"
    pd_na = "## 目录描述\n\n定位与依赖声明：不适用（纯头文件发布目录，无独立编译目标）。\n\n## 流程描述\n"
    pd_na_no_reason = "## 目录描述\n\n定位与依赖声明：不适用。\n\n## 流程描述\n"
    pd_na_multiline = "## 目录描述\n\n定位与依赖声明：不适用\n\n目录树与关系表如下。\n\n## 流程描述\n"
    pd_bold_na = "## 目录描述\n\n**定位与依赖声明**：不适用（纯头文件目录）。\n\n## 流程描述\n"
    pd_prefix_na = "## 目录描述\n\n定位与依赖声明：本目录不适用（纯头文件目录）。\n\n## 流程描述\n"
    pd_guide = "## 目录描述\n\n定位与依赖声明：四要素缺一不可，确实不适用时写明不适用及原因\n\n## 流程描述\n"
    pd_reason_labels = "## 目录描述\n\n定位与依赖声明：不适用（本目录为纯头文件，无编译归属与依赖头）。\n\n## 流程描述\n"
    pd_tpl_comment = ("## 目录描述\n\n目录树。\n\n"
                       "<!-- 四要素缺一不可；确实不适用时写明不适用及原因 -->\n\n## 流程描述\n")
    pd_unrelated_na = ("## 目录描述\n\n**层级归属**：L2。\n\n"
                       "本模块不涉及动态加载。\n\n## 流程描述\n")
    pd_label_na = "## 目录描述\n\n**依赖头**：不涉及跨层依赖。\n\n## 流程描述\n"
    pd_en_ok = ("## Directory Description\n\n**Layer Affiliation**: L2; **Build Affiliation**: obj; "
                "**Dependency Headers**: `*_pub.h`; **Upstream Callers**: coll_comm.\n")
    pd_en_miss = "## Directory Description\n\nnothing\n"
    pd_en_na = "## Directory Description\n\nLocation and Dependency Statement: not applicable (pure header directory).\n"
    pd_en_na_multiline = ("## Directory Description\n\n"
                          "Location and Dependency Statement: not applicable.\n\nDirectory tree below.\n")
    pd_en_na_no_period = ("## Directory Description\n\n"
                          "Location and Dependency Statement: not applicable\n\nDirectory tree below.\n")
    pd_en_prefix = ("## Directory Description\n\n"
                    "Location and Dependency Statement: namespace layout owned by base_comm.\n")
    pd_en_guide = ("## Directory Description\n\n"
                   "Location and Dependency Statement: fill all four labels; if not applicable give a reason.\n")
    pd_en_reason_labels = ("## Directory Description\n\n"
                           "Location and Dependency Statement: not applicable (pure header files, "
                           "no build affiliation or dependency headers).\n")
    if check_pos_dep(pd_full, "zh"):
        errors.append("pos-dep 四标签齐全期望零命中，实际 {}".format(check_pos_dep(pd_full, "zh")))
    if not check_pos_dep(pd_miss, "zh"):
        errors.append("pos-dep 缺声明期望命中，实际零命中")
    if check_pos_dep(pd_na, "zh"):
        errors.append("pos-dep 整段不适用（含原因）期望零命中，实际 {}".format(check_pos_dep(pd_na, "zh")))
    if not check_pos_dep(pd_na_no_reason, "zh"):
        errors.append("pos-dep 整段不适用无原因期望命中，实际零命中")
    if not check_pos_dep(pd_na_multiline, "zh"):
        errors.append("pos-dep 原因跨行捕获（下行文本当原因）期望命中，实际零命中")
    if check_pos_dep(pd_bold_na, "zh"):
        errors.append("pos-dep 加粗声明+同行原因期望零命中，实际 {}".format(check_pos_dep(pd_bold_na, "zh")))
    if check_pos_dep(pd_prefix_na, "zh"):
        errors.append("pos-dep 句中前置说明+同行原因期望零命中，实际 {}".format(check_pos_dep(pd_prefix_na, "zh")))
    if not check_pos_dep(pd_guide, "zh"):
        errors.append("pos-dep 指导语（冒号后任意文本+不适用）不构成声明，期望命中，实际零命中")
    if check_pos_dep(pd_reason_labels, "zh"):
        errors.append("pos-dep 声明原因提及标签名不应计为在场（合规声明应豁免），实际 {}".format(check_pos_dep(pd_reason_labels, "zh")))
    if not check_pos_dep(pd_tpl_comment, "zh"):
        errors.append("pos-dep 模板注释不构成豁免（评审反例），期望命中，实际零命中")
    if not check_pos_dep(pd_unrelated_na, "zh"):
        errors.append("pos-dep 无关「不涉及」句不构成豁免（评审反例），期望命中，实际零命中")
    if not check_pos_dep(pd_label_na, "zh"):
        errors.append("pos-dep 单要素不适用不连带豁免其余标签（评审反例），期望命中，实际零命中")
    if not check_pos_dep("## 功能描述\n\n无目录描述章节。\n", "zh"):
        errors.append("pos-dep 章节缺失期望命中，实际零命中")
    if check_pos_dep(pd_en_ok, "en"):
        errors.append("pos-dep EN 四标签齐全期望零命中，实际 {}".format(check_pos_dep(pd_en_ok, "en")))
    if not check_pos_dep(pd_en_miss, "en"):
        errors.append("pos-dep EN 缺声明期望命中，实际零命中")
    if check_pos_dep(pd_en_na, "en"):
        errors.append("pos-dep EN 整段不适用（含原因）期望零命中，实际 {}".format(check_pos_dep(pd_en_na, "en")))
    if not check_pos_dep(pd_en_na_multiline, "en"):
        errors.append("pos-dep EN 原因跨行捕获期望命中，实际零命中")
    if not check_pos_dep(pd_en_prefix, "en"):
        errors.append("pos-dep EN na 前缀词（namespace）不得豁免，期望命中，实际零命中")
    if not check_pos_dep(pd_en_na_no_period, "en"):
        errors.append("pos-dep EN 无句号+下行内容（分隔符跨行）期望命中，实际零命中")
    if not check_pos_dep(pd_en_guide, "en"):
        errors.append("pos-dep EN 指导语不构成声明，期望命中，实际零命中")
    if check_pos_dep(pd_en_reason_labels, "en"):
        errors.append("pos-dep EN 声明原因提及标签名不应计为在场（合规声明应豁免），实际 {}".format(check_pos_dep(pd_en_reason_labels, "en")))
    if errors:
        print("SELFTEST FAILED:")
        for e in errors:
            print(" -", e)
        return 1
    print("SELFTEST PASSED: hw-codes/mermaid/line-refs/bilingual/pos-dep 全部符合预期")
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
