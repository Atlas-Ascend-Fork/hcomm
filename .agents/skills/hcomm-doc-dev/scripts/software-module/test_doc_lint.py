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

"""doc_lint.py 单元测试。

用本地 mock git 仓库（git init + 构造源码/同名尾缀干扰文件）覆盖四项检查的
机械逻辑，不发起任何真实网络请求。

运行: python3 -m unittest test_doc_lint -v
依赖: Python 3.7+，仅标准库；需要 PATH 里有 git。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doc_lint  # noqa: E402


def run_git(args, cwd):
    result = subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError("git {} failed: {}".format(args, result.stderr))
    return result.stdout


class MockRepoTestCase(unittest.TestCase):
    """构造 mock git 仓库基类。

    仓库结构:
      commit 1: src/demo.cc (30 行) + src/aicpu/demo.cc (同名干扰源) +
                test/ut/ut_demo.cc (UT 同尾缀干扰)
    """

    @classmethod
    def setUpClass(cls):
        cls.repo = Path(tempfile.mkdtemp(prefix="doc_lint_test_"))
        run_git(["init", "-q"], cls.repo)
        run_git(["config", "user.email", "t@t"], cls.repo)
        run_git(["config", "user.name", "t"], cls.repo)
        (cls.repo / "src" / "aicpu").mkdir(parents=True)
        (cls.repo / "test" / "ut").mkdir(parents=True)
        (cls.repo / "src" / "demo.cc").write_text(
            "line {}\n".join([""] * 0) + "".join("line {}\n".format(i) for i in range(1, 31)),
            encoding="utf-8")
        (cls.repo / "src" / "aicpu" / "demo.cc").write_text(
            "aicpu\n", encoding="utf-8")
        (cls.repo / "test" / "ut" / "ut_demo.cc").write_text(
            "ut\n", encoding="utf-8")
        run_git(["add", "-A"], cls.repo)
        run_git(["commit", "-qm", "init"], cls.repo)
        cls.head = run_git(["rev-parse", "HEAD"], cls.repo).strip()

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(str(cls.repo), ignore_errors=True)


class TestHwCodes(unittest.TestCase):
    """硬件代号扫描：豁免边界、词边界、绝对行号。"""

    def test_prose_code_hits(self):
        text = "检查 CKE 期望值与 GSA。"
        hits = doc_lint.check_hw_codes(text)
        self.assertEqual(len(hits), 2)

    def test_absolute_lineno_reported(self):
        # 违规报告须含文档绝对行号（span 起始行 + 段内偏移换算）
        text = "# 标题\n\n段落一。\n\n检查 CKE 期望值。\n"
        hits = doc_lint.check_hw_codes(text)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].get("line"), 5)
        self.assertIn("第 5 行", hits[0]["message"])

    def test_lineno_after_codeblock(self):
        # 代码块之后的自然语言段，行号须跳过代码块行数
        text = "开头。\n```cpp\nint a;\nint b;\n```\n正文出现 Xn。\n"
        hits = doc_lint.check_hw_codes(text)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].get("line"), 6)

    def test_backtick_exempt(self):
        text = "见 `g_msMapCKE` 与 `Xn` 标识符。"
        self.assertEqual(doc_lint.check_hw_codes(text), [])

    def test_codeblock_exempt(self):
        text = "正文无代号。\n```cpp\nint Xn = GetGSA();\n```\n"
        self.assertEqual(doc_lint.check_hw_codes(text), [])

    def test_word_boundary_no_false_positive(self):
        # MSCCKE 之类的长标识符内嵌代号不应命中（\b 词边界）
        text = "调用 MSCCKE_HELPER 与 XnController 接口。"
        hits = doc_lint.check_hw_codes(text)
        # XnController 中 Xn 后接字母，不构成词边界；MSCCKE 同理
        self.assertEqual(hits, [])


class TestBilingual(unittest.TestCase):
    """中英结构对齐。"""

    def test_identical_structure_passes(self):
        zh = "## A\n\n| a | b |\n|---|---|\n\n```mermaid\ngraph TB\n  A-->B\n```\n"
        en = "## A\n\n| a | b |\n|---|---|\n\n```mermaid\ngraph TB\n  A-->B\n```\n"
        self.assertEqual(doc_lint.check_bilingual(zh, en), [])

    def test_table_count_mismatch(self):
        # 中文 1 张表、英文 2 张表 → 张数不一致
        zh = "| a | b |\n|---|---|\n| 1 | 2 |\n"
        en = "| a | b |\n|---|---|\n| 1 | 2 |\n\n| c | d |\n|---|---|\n| 3 | 4 |\n"
        hits = doc_lint.check_bilingual(zh, en)
        self.assertTrue(any("表格数不一致" in h["message"] for h in hits))

    def test_same_table_count_different_rows_passes(self):
        # 张数一致（各 1 张）、仅数据行数不同：不算结构不一致（行数属内容层）
        zh = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
        en = "| a | b |\n|---|---|\n"
        self.assertEqual(doc_lint.check_bilingual(zh, en), [])

    def test_table_count_by_separator_row(self):
        # 行数相同但张数不同（1×4 行 vs 2×2 行）必须报：按分隔行计张数
        zh = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
        en = "| a | b |\n|---|---|\n| 1 | 2 |\n\n| c | d |\n|---|---|\n| 3 | 4 |\n"
        hits = doc_lint.check_bilingual(zh, en)
        self.assertTrue(any("表格数不一致" in h["message"] for h in hits))

    def test_heading_count_mismatch(self):
        zh = "## A\n## B\n"
        en = "## A\n"
        hits = doc_lint.check_bilingual(zh, en)
        self.assertTrue(any("标题数不一致" in h["message"] for h in hits))

    def test_heading_level_sequence_mismatch(self):
        # 标题数量一致但层级顺序不同（[2,3] vs [3,2]）必须报
        zh = "## A\n### B\n"
        en = "### A'\n## B'\n"
        hits = doc_lint.check_bilingual(zh, en)
        self.assertTrue(any("层级序列不一致" in h["message"] for h in hits))


class TestLineRefs(MockRepoTestCase):
    """行号引用实证：裸文件名定位与行号界内验证。"""

    def test_bare_name_unique_resolves(self):
        # demo.cc 在 src/ 与 src/aicpu/ 各一份 → 同名源文件，不可唯一定位
        text = "见 `demo.cc:10`。"
        hits = doc_lint.check_line_refs(text, self.repo, self.head)
        self.assertTrue(any("无法唯一定位" in h["message"] for h in hits))

    def test_ut_suffix_not_matched(self):
        # ut_demo.cc 与 src/demo.cc 同尾缀但 basename 不同：全等语义下唯一命中自身，
        # 定位成功且行号在界内 → 零 ERROR（同尾缀干扰被正确排除）
        text = "见 `ut_demo.cc:1`。"
        hits = doc_lint.check_line_refs(text, self.repo, self.head)
        self.assertEqual([h for h in hits if h["severity"] == "ERROR"], [])

    def test_pathed_ref_passes(self):
        # 带路径引用绕过裸名定位环节，直接 git show 验证
        text = "见 `src/aicpu/demo.cc:1`。"
        hits = doc_lint.check_line_refs(text, self.repo, self.head)
        self.assertEqual([h for h in hits if h["severity"] == "ERROR"], [])

    def test_line_out_of_range(self):
        text = "见 `src/demo.cc:999`。"
        hits = doc_lint.check_line_refs(text, self.repo, self.head)
        self.assertTrue(any("行号越界" in h["message"] for h in hits))

    def test_range_in_bounds_passes(self):
        text = "见 `src/demo.cc:5-10`。"
        hits = doc_lint.check_line_refs(text, self.repo, self.head)
        self.assertEqual([h for h in hits if h["severity"] == "ERROR"], [])

    def test_nonexistent_file(self):
        text = "见 `src/nope.cc:1`。"
        hits = doc_lint.check_line_refs(text, self.repo, self.head)
        self.assertTrue(any("文件不存在" in h["message"] for h in hits))

    def test_no_repo_warns_only(self):
        text = "见 `src/demo.cc:5`。"
        hits = doc_lint.check_line_refs(text)
        self.assertTrue(all(h["severity"] == "WARN" for h in hits))


class TestMermaid(unittest.TestCase):
    """Mermaid 易错点。"""

    def test_generic_angle_bracket_hits(self):
        text = "```mermaid\nclassDiagram\n  class A {\n    +P(vector<X>) void\n  }\n```\n"
        hits = doc_lint.check_mermaid(text)
        self.assertEqual(len(hits), 1)
        self.assertIn("转义", hits[0]["message"])

    def test_generic_tilde_passes(self):
        text = "```mermaid\nclassDiagram\n  class A {\n    +P(vector~X~) void\n  }\n```\n"
        self.assertEqual(doc_lint.check_mermaid(text), [])

    def test_end_keyword_ambiguity(self):
        text = "```mermaid\nsequenceDiagram\n  Note right of A: end 结束\n```\n"
        hits = doc_lint.check_mermaid(text)
        self.assertTrue(any("end" in h["message"] for h in hits))

    def test_end_keyword_note_with_punctuation(self):
        # end 后接非词字符（句号等）也应命中：:\s*end(?=\W|$)
        text = "```mermaid\nsequenceDiagram\n  Note right of A: end。\n```\n"
        hits = doc_lint.check_mermaid(text)
        self.assertTrue(any("Note/label" in h["message"] for h in hits))

    def test_end_keyword_label_prefix(self):
        # graph 节点 label 以 end 开头：A[end 阶段]
        text = "```mermaid\ngraph TB\n  B[end 阶段]\n```\n"
        hits = doc_lint.check_mermaid(text)
        self.assertTrue(any("节点 label" in h["message"] for h in hits))

    def test_end_keyword_subgraph_name(self):
        # subgraph 名以 end 开头：endGroup
        text = "```mermaid\ngraph TB\n  subgraph endGroup\n    A[x]\n  end\n```\n"
        hits = doc_lint.check_mermaid(text)
        self.assertTrue(any("subgraph 名" in h["message"] for h in hits))

    def test_end_word_suffix_not_flagged(self):
        # end 后紧跟字母（endX）按整词解析无歧义，不报
        text = "```mermaid\nsequenceDiagram\n  Note right of A: endX 结束\n```\n"
        self.assertEqual(doc_lint.check_mermaid(text), [])

    def test_node_id_chinese_punctuation_hits(self):
        # 节点 ID 含中文标点（检视 R4：原 \w+ 捕获恒不命中的死检查）
        text = "```mermaid\ngraph TB\n  节点，A[x]\n```\n"
        hits = doc_lint.check_mermaid(text)
        self.assertTrue(any("节点 ID 含中文标点" in h["message"] for h in hits))

    def test_normal_node_id_not_flagged(self):
        # 正常节点 ID（字母/汉字无标点）零命中
        text = "```mermaid\ngraph TB\n  HCD[HcclCommDfx]\n  AICPU侧[处理]\n```\n"
        self.assertEqual(doc_lint.check_mermaid(text), [])

    def test_chinese_paren_in_class_method(self):
        text = "```mermaid\nclassDiagram\n  class A {\n    +P（x） void\n  }\n```\n"
        hits = doc_lint.check_mermaid(text)
        self.assertTrue(any("中文括号" in h["message"] for h in hits))

    def test_clean_diagram_zero_hits(self):
        text = ("```mermaid\nsequenceDiagram\n"
                "  participant HCD as HcclCommDfx\n"
                "  HCD->>HCD: GetOpModeFlags\n"
                "  Note right of HCD: 结束处理\n"
                "```\n")
        self.assertEqual(doc_lint.check_mermaid(text), [])


class TestSelftestBridge(unittest.TestCase):
    """--selftest 与单测的桥接：保证内置自测本身可用。"""

    def test_selftest_passes(self):
        self.assertEqual(doc_lint.selftest(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
