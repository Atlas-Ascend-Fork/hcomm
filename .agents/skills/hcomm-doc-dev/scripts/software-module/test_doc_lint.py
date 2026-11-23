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

用本地 mock git 仓库（git init + 构造源码/同名尾缀干扰文件）覆盖五项检查的
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


class TestPosDep(unittest.TestCase):
    """定位与依赖声明结构检查：四标签、不适用豁免、EN 译名、章节缺失。"""

    def test_four_labels_pass(self):
        text = ("## 目录描述\n\n目录树。\n\n"
                "> **层级归属**：L2；**编译归属**：`hcomm_coll_comm_obj`；"
                "**依赖头**：`*_pub.h`；**被谁调用**：coll_comm。\n\n## 流程描述\n")
        self.assertEqual(doc_lint.check_pos_dep(text, "zh"), [])

    def test_missing_labels_hit(self):
        # 只有两个标签 → 报缺失的另两个
        text = ("## 目录描述\n\n"
                "**层级归属**：L2；**编译归属**：`hcomm_coll_comm_obj`。\n\n## 流程描述\n")
        hits = doc_lint.check_pos_dep(text, "zh")
        self.assertEqual(len(hits), 1)
        self.assertIn("依赖头", hits[0]["message"])
        self.assertIn("被谁调用", hits[0]["message"])

    def test_na_statement_exempt(self):
        # 显式「定位与依赖声明：不适用/不涉及」+ 原因 → 整段豁免
        text = ("## 目录描述\n\n"
                "定位与依赖声明：不涉及（本目录为纯工具宏头文件，无独立编译目标）。\n\n## 流程描述\n")
        self.assertEqual(doc_lint.check_pos_dep(text, "zh"), [])

    def test_na_without_reason_hits(self):
        # 整段不适用声明缺原因 → 不豁免，且文案提示原因须同行（作者可自纠）
        text = "## 目录描述\n\n定位与依赖声明：不适用。\n\n## 流程描述\n"
        hits = doc_lint.check_pos_dep(text, "zh")
        self.assertEqual(len(hits), 1)
        self.assertIn("同一行", hits[0]["message"])

    def test_guide_text_not_exempt(self):
        # 评审反例（指导语冒充豁免）：冒号后任意文本再接「不适用」不构成声明
        text = "## 目录描述\n\n定位与依赖声明：四要素缺一不可，确实不适用时写明不适用及原因\n\n## 流程描述\n"
        hits = doc_lint.check_pos_dep(text, "zh")
        self.assertEqual(len(hits), 1)

    def test_en_guide_text_not_exempt(self):
        text = ("## Directory Description\n\n"
                "Location and Dependency Statement: fill all four labels; if not applicable give a reason.\n")
        hits = doc_lint.check_pos_dep(text, "en")
        self.assertEqual(len(hits), 1)

    def test_en_no_period_multiline_hits(self):
        # 评审反例（EN 分隔符跨行）：无句号结尾 + 下一行内容 → 不得跨行捕获原因
        text = ("## Directory Description\n\n"
                "Location and Dependency Statement: not applicable\n\nDirectory tree below.\n")
        hits = doc_lint.check_pos_dep(text, "en")
        self.assertEqual(len(hits), 1)

    def test_en_prefix_with_reason_exempt(self):
        # EN 前置说明（this directory is ...）+ 同行原因 → 豁免
        text = ("## Directory Description\n\n"
                "Location and Dependency Statement: this directory is not applicable (headers).\n")
        self.assertEqual(doc_lint.check_pos_dep(text, "en"), [])

    def test_na_reason_mentioning_labels_exempt(self):
        # 评审反例（标签在场吞入声明原因）：原因提及标签名不计为标签在场，
        # 合规整段声明应豁免
        text = "## 目录描述\n\n定位与依赖声明：不适用（本目录为纯头文件，无编译归属与依赖头）。\n\n## 流程描述\n"
        self.assertEqual(doc_lint.check_pos_dep(text, "zh"), [])

    def test_en_na_reason_mentioning_labels_exempt(self):
        text = ("## Directory Description\n\n"
                "Location and Dependency Statement: not applicable (pure header files, "
                "no build affiliation or dependency headers).\n")
        self.assertEqual(doc_lint.check_pos_dep(text, "en"), [])

    def test_na_without_reason_and_trailing_content_hits(self):
        # 评审反例（原因跨行捕获）：声明行无原因、下一行为普通内容
        # （目录树/表格首行）→ 不得把下一行文本当原因，须 HIT
        text = "## 目录描述\n\n定位与依赖声明：不适用\n\n目录树与关系表如下。\n\n## 流程描述\n"
        hits = doc_lint.check_pos_dep(text, "zh")
        self.assertEqual(len(hits), 1)

    def test_bold_statement_with_reason_exempt(self):
        # 评审反例（锚点过窄）：加粗写法与句中前置说明 + 同行原因 → 豁免
        text = "## 目录描述\n\n**定位与依赖声明**：不适用（纯头文件目录）。\n\n## 流程描述\n"
        self.assertEqual(doc_lint.check_pos_dep(text, "zh"), [])
        text2 = "## 目录描述\n\n定位与依赖声明：本目录不适用（纯头文件目录）。\n\n## 流程描述\n"
        self.assertEqual(doc_lint.check_pos_dep(text2, "zh"), [])

    def test_na_reason_must_be_inline(self):
        # 加粗写法 + 无行内原因 + 下一行内容 → 仍 HIT（容忍加粗但不容忍跨行原因）
        text = "## 目录描述\n\n**定位与依赖声明**：不适用\n\n目录树。\n\n## 流程描述\n"
        hits = doc_lint.check_pos_dep(text, "zh")
        self.assertEqual(len(hits), 1)

    def test_template_comment_not_exempt(self):
        # 评审反例：模板自带注释「确实不适用时写明不适用及原因」不构成豁免
        text = ("## 目录描述\n\n目录树。\n\n"
                "<!-- 四要素缺一不可；确实不适用时写明不适用及原因 -->\n\n## 流程描述\n")
        hits = doc_lint.check_pos_dep(text, "zh")
        self.assertEqual(len(hits), 1)

    def test_unrelated_na_phrase_not_exempt(self):
        # 评审反例：无关句「本模块不涉及动态加载」不构成豁免，缺的三个标签逐个报出
        text = ("## 目录描述\n\n**层级归属**：L2。\n\n"
                "本模块不涉及动态加载。\n\n## 流程描述\n")
        hits = doc_lint.check_pos_dep(text, "zh")
        self.assertEqual(len(hits), 1)
        for label in ("编译归属", "依赖头", "被谁调用"):
            self.assertIn(label, hits[0]["message"])

    def test_per_label_na_does_not_exempt_others(self):
        # 评审反例：单要素「依赖头：不涉及跨层依赖」只满足该标签，其余仍须报
        text = "## 目录描述\n\n**依赖头**：不涉及跨层依赖。\n\n## 流程描述\n"
        hits = doc_lint.check_pos_dep(text, "zh")
        self.assertEqual(len(hits), 1)
        for label in ("层级归属", "编译归属", "被谁调用"):
            self.assertIn(label, hits[0]["message"])

    def test_whole_na_with_partial_labels_hits(self):
        # 四标签不全缺（写了 1 个）+ 整段不适用声明 → 整段豁免不适用，缺项照报
        text = ("## 目录描述\n\n**层级归属**：L2。\n\n"
                "定位与依赖声明：不适用（纯头文件目录）。\n\n## 流程描述\n")
        hits = doc_lint.check_pos_dep(text, "zh")
        self.assertEqual(len(hits), 1)

    def test_section_missing_hits(self):
        text = "## 功能描述\n\n无目录描述章节。\n"
        hits = doc_lint.check_pos_dep(text, "zh")
        self.assertTrue(any("未找到目录描述章节" in h["message"] for h in hits))

    def test_scope_limited_to_section(self):
        # 四标签写在其他章节不算数：目录描述段内缺 → 命中
        text = ("## 目录描述\n\n目录树。\n\n## 流程描述\n\n"
                "**层级归属**：L2；**编译归属**：obj；**依赖头**：h；**被谁调用**：x。\n")
        hits = doc_lint.check_pos_dep(text, "zh")
        self.assertEqual(len(hits), 1)

    def test_en_labels_pass(self):
        text = ("## Directory Description\n\n"
                "**Layer Affiliation**: L2; **Build Affiliation**: obj; "
                "**Dependency Headers**: `*_pub.h`; **Upstream Callers**: coll_comm.\n")
        self.assertEqual(doc_lint.check_pos_dep(text, "en"), [])

    def test_en_missing_hit(self):
        text = "## Directory Description\n\ntree only.\n"
        hits = doc_lint.check_pos_dep(text, "en")
        self.assertEqual(len(hits), 1)

    def test_en_na_case_insensitive_exempt(self):
        # EN 整段不适用（含原因）→ 豁免；大小写不敏感
        text = "## Directory Description\n\nLocation and Dependency Statement: Not Applicable (pure header directory).\n"
        self.assertEqual(doc_lint.check_pos_dep(text, "en"), [])

    def test_en_na_without_reason_hits(self):
        # EN 整段不适用无原因 → 不豁免（尾部接普通内容仍须 HIT，锁住"原因限同行"契约）
        text = ("## Directory Description\n\n"
                "Location and Dependency Statement: not applicable.\n\n"
                "Directory tree below.\n")
        hits = doc_lint.check_pos_dep(text, "en")
        self.assertEqual(len(hits), 1)

    def test_en_na_prefix_word_not_matched(self):
        # 评审反例（na 无词边界）：namespace 的 na 前缀不得命中整段豁免
        text = ("## Directory Description\n\n"
                "Location and Dependency Statement: namespace layout owned by base_comm.\n")
        hits = doc_lint.check_pos_dep(text, "en")
        self.assertEqual(len(hits), 1)

    def test_en_bold_statement_with_reason_exempt(self):
        # EN 加粗写法 + 同行原因 → 豁免
        text = ("## Directory Description\n\n"
                "**Location and Dependency Statement**: not applicable (pure header directory).\n")
        self.assertEqual(doc_lint.check_pos_dep(text, "en"), [])


class TestSelftestBridge(unittest.TestCase):
    """--selftest 与单测的桥接：保证内置自测本身可用。"""

    def test_selftest_passes(self):
        self.assertEqual(doc_lint.selftest(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
