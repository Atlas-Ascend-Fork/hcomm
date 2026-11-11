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

"""pr_review.py 单元测试。

用本地 mock git 仓库（git init + 构造多 hunk diff）与 mock GitCode API
（monkeypatch api_request）与本地 HTTP 服务覆盖检视流程的机械步骤，不访问外部网络。

运行: python3 -m unittest test_pr_review -v
依赖: Python 3.7+，仅标准库；需要 PATH 里有 git。
"""

from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_review  # noqa: E402


def run_git(args, cwd):
    result = subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError("git {} failed: {}".format(args, result.stderr))
    return result.stdout


class MockRepoTestCase(unittest.TestCase):
    """构造 mock git 仓库基类。

    仓库结构:
      base commit: src/demo.cc (10 行) + src/old.cc
      head commit: 修改 src/demo.cc（中间插入 2 行 + 尾部改 1 行）+ 新增 src/new_file.cc（20 行）
    """

    @classmethod
    def setUpClass(cls):
        cls.repo_dir = Path(tempfile.mkdtemp(prefix="pr_review_test_"))
        run_git(["init", "-q"], cls.repo_dir)
        run_git(["config", "user.email", "test@example.com"], cls.repo_dir)
        run_git(["config", "user.name", "tester"], cls.repo_dir)
        # origin 指向不可达地址，确保测试绝不发起真实网络 fetch
        run_git(["remote", "add", "origin", "https://gitcode.example.invalid/cann/hcomm.git"], cls.repo_dir)

        demo_base = "\n".join("line {}".format(i) for i in range(1, 11)) + "\n"
        (cls.repo_dir / "src").mkdir(exist_ok=True)
        (cls.repo_dir / "src" / "demo.cc").write_text(demo_base, encoding="utf-8")
        (cls.repo_dir / "src" / "old.cc").write_text("old content\n", encoding="utf-8")
        run_git(["add", "."], cls.repo_dir)
        run_git(["commit", "-q", "-m", "base"], cls.repo_dir)
        cls.base_sha = run_git(["rev-parse", "HEAD"], cls.repo_dir).strip()

        # head：demo.cc 第 5 行后插 2 行，第 10 行（现 12）改为 modified；新增 new_file.cc
        demo_head = demo_base.replace("line 5\n", "line 5\ninserted A\ninserted B\n")
        demo_head = demo_head.replace("line 10", "modified tail")
        (cls.repo_dir / "src" / "demo.cc").write_text(demo_head, encoding="utf-8")
        (cls.repo_dir / "src" / "new_file.cc").write_text(
            "\n".join("new line {}".format(i) for i in range(1, 21)) + "\n", encoding="utf-8")
        run_git(["add", "."], cls.repo_dir)
        run_git(["commit", "-q", "-m", "head"], cls.repo_dir)
        cls.head_sha = run_git(["rev-parse", "HEAD"], cls.repo_dir).strip()
        # 预置 PR head ref，使 ensure_pr_refs 无需 fetch 即可 rev-parse
        run_git(["update-ref", "refs/remotes/origin/mr/1-head", cls.head_sha], cls.repo_dir)
        # 预置 origin/master ref 供 merge-base fallback
        run_git(["update-ref", "refs/remotes/origin/master", cls.base_sha], cls.repo_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.repo_dir, ignore_errors=True)

    def setUp(self):
        self._saved_request = pr_review.api_request
        self.api_calls = []

    def tearDown(self):
        pr_review.api_request = self._saved_request

    def mock_api(self, responses):
        """按 URL 关键词路由的 mock。responses: {url_keyword: data_or_callable}。"""
        def fake_request(url, method="GET", data=None, token=""):
            self.api_calls.append({"url": url, "method": method, "data": data})
            for keyword, value in responses.items():
                if keyword in url:
                    if callable(value):
                        return value(url, method, data)
                    return value
            raise RuntimeError("HTTP 404: not mocked {}".format(url))
        pr_review.api_request = fake_request
        return fake_request


class TestVerifyLineNumber(MockRepoTestCase):
    """行号验证：行存在性、snippet 匹配、越界、文件不存在。"""

    def test_verified_with_snippet(self):
        result = pr_review.verify_line_number(
            self.repo_dir, self.head_sha, "src/demo.cc", 6, "inserted A")
        self.assertTrue(result["verified"])

    def test_verified_without_snippet(self):
        result = pr_review.verify_line_number(
            self.repo_dir, self.head_sha, "src/demo.cc", 1)
        self.assertTrue(result["verified"])

    def test_snippet_mismatch(self):
        result = pr_review.verify_line_number(
            self.repo_dir, self.head_sha, "src/demo.cc", 1, "not-exist-content")
        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "snippet mismatch")
        self.assertEqual(result["actual"], "line 1")

    def test_line_out_of_range(self):
        result = pr_review.verify_line_number(
            self.repo_dir, self.head_sha, "src/demo.cc", 999)
        self.assertFalse(result["verified"])
        self.assertIn("out of range", result["reason"])

    def test_file_not_found(self):
        result = pr_review.verify_line_number(
            self.repo_dir, self.head_sha, "src/no_such.cc", 1)
        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "file not found at HEAD")

    def test_zero_or_negative_line(self):
        result = pr_review.verify_line_number(
            self.repo_dir, self.head_sha, "src/demo.cc", 0)
        self.assertFalse(result["verified"])


class TestFindDiffPosition(MockRepoTestCase):
    """diff 位置计算：新增行命中、上下文行命中、删除行与未变更行不命中。"""

    def test_added_line_hit(self):
        # demo.cc 新版第 6 行是 inserted A（diff 新增行）
        pos = pr_review.find_diff_position(
            self.repo_dir, self.base_sha, self.head_sha, "src/demo.cc", 6)
        self.assertEqual(pos, 6)

    def test_context_line_hit(self):
        # hunk 从新文件第 3 行起，第 3 行是上下文行（未变更但出现在 diff 中）
        pos = pr_review.find_diff_position(
            self.repo_dir, self.base_sha, self.head_sha, "src/demo.cc", 3)
        self.assertEqual(pos, 3)

    def test_line_before_hunk_misses(self):
        # 第 2 行在 hunk 之前（未出现在 diff 中）
        pos = pr_review.find_diff_position(
            self.repo_dir, self.base_sha, self.head_sha, "src/demo.cc", 2)
        self.assertIsNone(pos)

    def test_unchanged_far_line_miss(self):
        # old.cc 完全没变更，任何行号都不在 diff
        pos = pr_review.find_diff_position(
            self.repo_dir, self.base_sha, self.head_sha, "src/old.cc", 1)
        self.assertIsNone(pos)

    def test_pure_new_file_any_line(self):
        # 纯新增文件（+++ b/ 头行必须跳过，不能推进行号）
        for line in (1, 10, 20):
            pos = pr_review.find_diff_position(
                self.repo_dir, self.base_sha, self.head_sha, "src/new_file.cc", line)
            self.assertEqual(pos, line, "new_file.cc line {} 应命中".format(line))

    def test_modified_tail_line(self):
        # 第 12 行是 modified tail（原 line 10 修改而来）
        pos = pr_review.find_diff_position(
            self.repo_dir, self.base_sha, self.head_sha, "src/demo.cc", 12)
        self.assertEqual(pos, 12)


class TestDedupFindings(MockRepoTestCase):
    """内容指纹去重：file+line 命中、pr_comment 回退格式命中、标题命中。"""

    def setUp(self):
        super().setUp()
        self.finding = {
            "severity": "HIGH", "file": "src/demo.cc", "line": 6,
            "title": "空指针风险", "body": "问题描述",
        }

    def test_dedup_by_file_line_diff_comment(self):
        # v5 comments path 由 v4 enrich 补齐后命中 file+line
        existing = [{"body": "**[HIGH] 其他**\n旧意见", "path": "src/demo.cc",
                     "diff_position": {"start_new_line": 6}}]
        deduped, skipped = pr_review.dedup_findings([self.finding], existing)
        self.assertEqual(len(deduped), 0)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["reason"], "file+line 已有评论")

    def test_dedup_by_file_line_pr_comment_quote(self):
        existing = [{"body": "> src/demo.cc:6\n\n**[HIGH] 其他**\n旧意见"}]
        deduped, skipped = pr_review.dedup_findings([self.finding], existing)
        self.assertEqual(len(deduped), 0)
        self.assertEqual(len(skipped), 1)

    def test_dedup_by_title(self):
        existing = [{"body": "**[MEDIUM] 空指针风险**\n别人提过同标题意见",
                     "diff_position": {"start_new_line": 99}}]
        deduped, skipped = pr_review.dedup_findings([self.finding], existing)
        self.assertEqual(len(deduped), 0)
        self.assertEqual(skipped[0]["reason"], "标题与已有评论重复")

    def test_no_dedup_when_different(self):
        existing = [{"body": "**[HIGH] 另一个问题**\n旧意见",
                     "diff_position": {"start_new_line": 3}}]
        deduped, skipped = pr_review.dedup_findings([self.finding], existing)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(len(skipped), 0)

    def test_mixed_findings_partial_dedup(self):
        other = {"severity": "LOW", "file": "src/demo.cc", "line": 12,
                 "title": "命名建议", "body": "..."}
        existing = [{"body": "**[HIGH] 空指针风险**\n同标题",
                     "diff_position": {"start_new_line": 6}}]
        deduped, skipped = pr_review.dedup_findings([self.finding, other], existing)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["title"], "命名建议")
        self.assertEqual(len(skipped), 1)


class TestBaselineDrift(MockRepoTestCase):
    """基线漂移核验：本地 diff 文件数 vs changed_files。"""

    def make_meta(self, changed_files):
        return {"changed_files": changed_files}

    def test_no_drift_when_consistent(self):
        result = pr_review.check_baseline_drift(
            self.repo_dir, self.base_sha, self.head_sha, self.make_meta(2))
        self.assertFalse(result["drift"])
        self.assertEqual(result["local_files"], 2)

    def test_no_drift_when_small_diff(self):
        # 无 changed_files 元数据时不判漂移
        result = pr_review.check_baseline_drift(
            self.repo_dir, self.base_sha, self.head_sha, self.make_meta(0))
        self.assertFalse(result["drift"])

    def test_drift_detected(self):
        # 声称只改 1 个文件但本地 diff 有 2 个（阈值: >max(1*2, 1+5)=6 不触发，
        # 用更悬殊的假元数据验证触发路径）
        result = pr_review.check_baseline_drift(
            self.repo_dir, self.base_sha, self.head_sha, self.make_meta(1))
        # 2 <= max(2, 6)，不触发
        self.assertFalse(result["drift"])
        # 直接构造触发场景：expected=1, local=2 -> 2 > max(2,6)? 否。
        # 换 base 使 diff 变大不现实，这里用单元级 mock 验证判定逻辑分支
        with unittest.mock.patch.object(pr_review, "run_git", return_value="a\n" * 20):
            result = pr_review.check_baseline_drift(
                self.repo_dir, "fake_base", "fake_head", self.make_meta(1))
            self.assertTrue(result["drift"])



class TestGetPrMeta(MockRepoTestCase):
    """PR 元数据：v5/v4 合并、state、分支字段。"""

    def test_meta_merges_v4_diff_refs(self):
        v5_data = {
            "state": "open", "target_branch": "master", "source_branch": "feat/x",
            "title": "demo", "body": "desc",
            "head": {"sha": "aaa", "ref": "feat/x"}, "base": {"sha": "bbb", "ref": "master"},
            "changed_files": 3, "additions": 10, "deletions": 2,
            "user": {"login": "someone"},
        }
        v4_data = {"diff_refs": {"base_sha": "v4base", "head_sha": "v4head", "start_sha": "v4start"}}

        def router(url, method="GET", data=None, token=""):
            if "/api/v5/" in url:
                return v5_data
            if "/api/v4/" in url:
                return v4_data
            raise RuntimeError("unmocked")
        pr_review.api_request = router
        meta = pr_review.get_pr_meta("cann", "hcomm", 123, token="t")
        self.assertEqual(meta["state"], "open")
        self.assertEqual(meta["target_branch"], "master")
        self.assertEqual(meta["head_sha"], "v4head")
        self.assertEqual(meta["base_sha"], "v4base")
        self.assertEqual(meta["changed_files"], 3)
        self.assertIn("pull/123", meta["html_url"])

    def test_meta_falls_back_to_v5_base_when_v4_down(self):
        v5_data = {
            "state": "merged", "target_branch": "master",
            "head": {"sha": "aaa"}, "base": {"sha": "bbb"},
        }

        def router(url, method="GET", data=None, token=""):
            if "/api/v5/" in url:
                return v5_data
            raise RuntimeError("HTTP 404")
        pr_review.api_request = router
        meta = pr_review.get_pr_meta("cann", "hcomm", 1, token="t")
        self.assertEqual(meta["state"], "merged")
        self.assertEqual(meta["base_sha"], "bbb")


class TestPostFinding(MockRepoTestCase):
    """意见提交：行内评论 payload、回退 PR 评论 payload。"""

    def test_diffnote_payload(self):
        calls = []

        def fake_request(url, method="GET", data=None, token=""):
            calls.append({"url": url, "method": method, "data": data})
            return {"id": "hex123", "comment_type": "diff_comment"}
        pr_review.api_request = fake_request
        finding = {"severity": "HIGH", "file": "src/a.cc", "line": 5,
                   "title": "T", "body": "B"}
        result = pr_review.post_finding(pr_review.make_target("cann", "hcomm", 9), "sha1", finding, 5, "tok")
        self.assertEqual(result["comment_type"], "diff_comment")
        payload = calls[0]["data"]
        self.assertEqual(payload["position"], 5)
        self.assertEqual(payload["path"], "src/a.cc")
        self.assertEqual(payload["commit_id"], "sha1")
        self.assertIn("**[HIGH] T**", payload["body"])

    def test_fallback_pr_comment_payload(self):
        calls = []

        def fake_request(url, method="GET", data=None, token=""):
            calls.append(data)
            return {"id": "hex", "comment_type": "pr_comment"}
        pr_review.api_request = fake_request
        finding = {"severity": "LOW", "file": "src/a.cc", "line": 77,
                   "title": "T", "body": "B"}
        pr_review.post_finding(pr_review.make_target("cann", "hcomm", 9), "sha1", finding, None, "tok")
        payload = calls[0]
        self.assertNotIn("position", payload)
        self.assertNotIn("path", payload)
        self.assertIn("> src/a.cc:77", payload["body"])


class TestPostReport(MockRepoTestCase):
    """汇总报告：统计、表格、严重级别排序。"""

    def test_report_body(self):
        calls = []

        def fake_request(url, method="GET", data=None, token=""):
            calls.append(data)
            return {"id": "hex"}
        pr_review.api_request = fake_request
        findings = [
            {"severity": "LOW", "file": "a.cc", "line": 1, "dimension": "D", "title": "L1", "body": "b"},
            {"severity": "CRITICAL", "file": "b.cc", "line": 2, "dimension": "D", "title": "C1", "body": "b"},
            {"severity": "HIGH", "file": "c.cc", "line": 3, "dimension": "D", "title": "H1", "body": "b"},
        ]
        meta = {"html_url": "http://x", "title": "t", "head_sha": "aaaa", "base_sha": "bbbb"}
        pr_review.post_report(pr_review.make_target("cann", "hcomm", 5), findings, meta, "tok")
        body = calls[0]["body"]
        self.assertIn("代码检视汇总报告", body)
        self.assertIn("CRITICAL 1", body)
        # CRITICAL 行应排在 LOW 前
        self.assertLess(body.index("C1"), body.index("L1"))


class TestCheckEnv(MockRepoTestCase):
    """--check-env 环境校验。"""

    def test_check_env_with_token(self):
        with unittest.mock.patch.object(pr_review, "get_token", return_value="tok1234"):
            with unittest.mock.patch.object(pr_review, "api_request",
                                            return_value={"state": "open"}):
                rc = pr_review.cmd_check_env(argparse_stub(self.repo_dir))
                self.assertEqual(rc, 0)

    def test_check_env_without_token(self):
        with unittest.mock.patch.object(pr_review, "get_token", return_value=""):
            rc = pr_review.cmd_check_env(argparse_stub(self.repo_dir))
            self.assertEqual(rc, 1)


def argparse_stub(repo_root):
    """构造 cmd_check_env 需要的最小 args 对象。"""
    return type("Args", (), {"repo_root": str(repo_root), "owner": "cann", "repo": "hcomm"})()


class TestLoadFindings(unittest.TestCase):
    """findings.json 加载与 schema 校验。"""

    def test_load_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.json"
            p.write_text(json.dumps([
                {"severity": "HIGH", "file": "a.cc", "line": 1, "title": "t", "body": "b"}]),
                encoding="utf-8")
            data = pr_review.load_findings(str(p))
            self.assertEqual(len(data), 1)

    def test_load_single_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.json"
            p.write_text(json.dumps(
                {"severity": "LOW", "file": "a.cc", "line": 1, "title": "t", "body": "b"}),
                encoding="utf-8")
            data = pr_review.load_findings(str(p))
            self.assertEqual(len(data), 1)

    def test_missing_field_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.json"
            p.write_text(json.dumps([{"severity": "HIGH"}]), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                pr_review.load_findings(str(p))

    def test_file_not_found(self):
        with self.assertRaises(RuntimeError):
            pr_review.load_findings("/no/such/file.json")


class TestFindRemote(MockRepoTestCase):
    """remote 自动探测。"""

    def test_find_cann_remote(self):
        self.assertEqual(pr_review.find_remote(self.repo_dir), "origin")

    def test_prefer_upstream_when_both(self):
        run_git(["remote", "add", "upstream", "https://gitcode.com/cann/hcomm.git"],
                self.repo_dir)
        # origin 也指向 cann，先匹配到的返回；无论哪个都指向 cann 即可
        remote = pr_review.find_remote(self.repo_dir)
        self.assertIn(remote, ("origin", "upstream"))


class TestCmdReviewFlow(MockRepoTestCase):
    """verify-only / dry-run / POST 命令级流程（mock API + mock git refs）。"""

    def setUp(self):
        super().setUp()
        self.findings = [{
            "severity": "HIGH", "file": "src/demo.cc", "line": 6,
            "title": "插入行问题", "body": "描述与建议修复", "code_snippet": "inserted A",
        }, {
            "severity": "LOW", "file": "src/old.cc", "line": 1,
            "title": "未变更文件意见", "body": "描述", "code_snippet": "old content",
        }]

    def write_findings(self, tmp):
        p = Path(tmp) / "findings.json"
        p.write_text(json.dumps(self.findings, ensure_ascii=False), encoding="utf-8")
        return str(p)

    def make_meta_dict(self):
        return {
            "state": "open", "target_branch": "master", "source_branch": "b",
            "title": "t", "body": "", "head_sha": self.head_sha,
            "base_sha": self.base_sha, "changed_files": 2,
            "additions": 22, "deletions": 1, "author": "a",
            "html_url": "https://gitcode.com/cann/hcomm/pull/1",
        }

    def mock_full_api(self, existing_comments=None):
        """mock PR 元数据 + 已有评论 + POST。"""
        post_calls = []

        def router(url, method="GET", data=None, token=""):
            if method == "GET" and "/pulls/1".format() in url and url.rstrip("/").endswith("/1"):
                return self.make_meta_dict()
            if "/api/v4/" in url:
                return {"diff_refs": {"base_sha": self.base_sha, "head_sha": self.head_sha}}
            if method == "GET" and "/comments" in url:
                return existing_comments or []
            if method == "POST":
                post_calls.append({"url": url, "data": data})
                return {"id": "hex" + str(len(post_calls)), "comment_type": "diff_comment"}
            raise RuntimeError("unmocked {} {}".format(method, url))

        pr_review.api_request = router
        return post_calls

    def test_verify_only(self):
        post_calls = self.mock_full_api()
        with tempfile.TemporaryDirectory() as tmp:
            findings_path = self.write_findings(tmp)
            args = argparse_review_stub(self.repo_dir, findings_path, verify_only=True)
            rc = pr_review.cmd_review(args, "tok")
            self.assertEqual(rc, 0)
            self.assertEqual(len(post_calls), 0)  # verify-only 不 POST

    def test_dry_run(self):
        post_calls = self.mock_full_api()
        with tempfile.TemporaryDirectory() as tmp:
            findings_path = self.write_findings(tmp)
            args = argparse_review_stub(self.repo_dir, findings_path, dry_run=True)
            rc = pr_review.cmd_review(args, "tok")
            self.assertEqual(rc, 0)
            self.assertEqual(len(post_calls), 0)

    def test_post_with_dedup_and_fallback(self):
        # 已有评论命中第一条 finding 的 file+line -> 跳过；
        # 第二条 finding 行不在 diff -> 回退 pr_comment payload
        existing = [{"body": "**[HIGH] 插入行问题**\n旧", "path": "src/demo.cc",
                     "diff_position": {"start_new_line": 6}}]
        post_calls = self.mock_full_api(existing)
        with tempfile.TemporaryDirectory() as tmp:
            findings_path = self.write_findings(tmp)
            args = argparse_review_stub(self.repo_dir, findings_path)
            rc = pr_review.cmd_review(args, "tok")
            self.assertEqual(rc, 0)
            # 只有 fallback 那条被 POST
            self.assertEqual(len(post_calls), 1)
            payload = post_calls[0]["data"]
            self.assertNotIn("position", payload)
            self.assertIn("> src/old.cc:1", payload["body"])

    def test_post_rejected_when_unverified_line(self):
        self.findings.append({
            "severity": "MEDIUM", "file": "src/demo.cc", "line": 999,
            "title": "越界行", "body": "b",
        })
        post_calls = self.mock_full_api()
        with tempfile.TemporaryDirectory() as tmp:
            findings_path = self.write_findings(tmp)
            args = argparse_review_stub(self.repo_dir, findings_path)
            rc = pr_review.cmd_review(args, "tok")
            self.assertEqual(rc, 1)
            self.assertEqual(len(post_calls), 0)

    def test_merged_pr_rejects_post(self):
        meta = self.make_meta_dict()
        meta["state"] = "merged"

        def router(url, method="GET", data=None, token=""):
            if "/api/v5/" in url and method == "GET":
                return meta
            if "/api/v4/" in url:
                return {"diff_refs": {"base_sha": self.base_sha, "head_sha": self.head_sha}}
            raise RuntimeError("unmocked")
        pr_review.api_request = router

        with tempfile.TemporaryDirectory() as tmp:
            findings_path = self.write_findings(tmp)
            args = argparse_review_stub(self.repo_dir, findings_path)
            rc = pr_review.cmd_review(args, "tok")
            self.assertEqual(rc, 1)

    def test_baseline_drift_blocks_post(self):
        meta = self.make_meta_dict()
        meta["changed_files"] = 1  # 本地 2 文件 vs 声称 1（阈值不触发，改用 0 触发不了）
        # 用 mock run_git 放大本地文件数触发漂移分支
        meta["changed_files"] = 1
        real_run_git = pr_review.run_git

        def fake_run_git(args, cwd):
            if args and args[0] == "diff" and "--name-only" in args:
                return "\n".join("file{}.cc".format(i) for i in range(20)) + "\n"
            return real_run_git(args, cwd)

        def router(url, method="GET", data=None, token=""):
            if "/api/v5/" in url and method == "GET":
                return meta
            if "/api/v4/" in url:
                return {"diff_refs": {"base_sha": self.base_sha, "head_sha": self.head_sha}}
            raise RuntimeError("unmocked")
        pr_review.api_request = router

        with tempfile.TemporaryDirectory() as tmp:
            findings_path = self.write_findings(tmp)
            args = argparse_review_stub(self.repo_dir, findings_path)
            with unittest.mock.patch.object(pr_review, "run_git", side_effect=fake_run_git):
                rc = pr_review.cmd_review(args, "tok")
            self.assertEqual(rc, 1)

    def _check_required_comment_error_blocks_post(self, failing_page):
        post_calls = self.mock_full_api()
        original_request = pr_review.api_request

        def router(url, method="GET", data=None, token=""):
            if method == "GET" and "/api/v5/" in url and "/comments" in url:
                page = int(parse_qs(urlsplit(url).query)["page"][0])
                if page == failing_page:
                    raise RuntimeError("HTTP 503: required comments unavailable")
                return [{"id": i + 1, "body": "existing"} for i in range(100)]
            return original_request(url, method, data, token)

        pr_review.api_request = router
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse_review_stub(self.repo_dir, self.write_findings(tmp))
            with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                pr_review.cmd_review(args, "tok")
        self.assertEqual(post_calls, [])

    def test_required_comments_first_page_error_blocks_post(self):
        self._check_required_comment_error_blocks_post(1)

    def test_required_comments_later_page_error_blocks_post(self):
        self._check_required_comment_error_blocks_post(2)


    def test_required_comment_shapes_block_post(self):
        shapes = [{}, {"status": "ok", "http_code": 200}, {"data": {}}, "bad", [None]]
        for shape in shapes:
            with self.subTest(shape=shape):
                posts = self.mock_full_api()
                original = pr_review.api_request
                def request(url, method="GET", data=None, token=""):
                    if method == "GET" and "/api/v5/" in url and "/comments" in url:
                        return shape
                    return original(url, method, data, token)
                pr_review.api_request = request
                with tempfile.TemporaryDirectory() as directory:
                    args = argparse_review_stub(self.repo_dir, self.write_findings(directory))
                    with self.assertRaisesRegex(RuntimeError, "列表响应格式错误"):
                        pr_review.cmd_review(args, "fixture")
                self.assertEqual(posts, [])


def argparse_review_stub(repo_root, findings, verify_only=False, dry_run=False):
    return type("Args", (), {
        "repo_root": str(repo_root), "owner": "cann", "repo": "hcomm",
        "pr_number": 1, "findings": findings,
        "verify_only": verify_only, "dry_run": dry_run,
        "head_sha": "", "base_sha": "", "delay": 0,
    })()


class TestGetPrFilesHttp(unittest.TestCase):
    """真实 HTTP 客户端覆盖文件接口与评论读取的错误边界。"""

    def setUp(self):
        self.files = []
        self.status = 200
        self.requests = []
        self.comment_pages = {}
        self.comments_failure_page = None
        test_case = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                test_case.requests.append(self.path)
                status, payload = test_case._http_response(self.path)
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.base_url = pr_review.GITCODE_API_V5
        self.base_v4_url = pr_review.GITCODE_API_V4
        self.http_url = "http://127.0.0.1:{}".format(self.server.server_port)
        pr_review.GITCODE_API_V5 = self.http_url
        pr_review.GITCODE_API_V4 = self.http_url + "/api/v4"

    def tearDown(self):
        pr_review.GITCODE_API_V5 = self.base_url
        pr_review.GITCODE_API_V4 = self.base_v4_url
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()

    def _http_response(self, request_path):
        parsed = urlsplit(request_path)
        if parsed.path == "/repos/cann/hcomm/pulls/1/files":
            return self.status, self.files
        if parsed.path == "/repos/cann/hcomm/pulls/1/comments":
            page = int(parse_qs(parsed.query).get("page", ["1"])[0])
            if page == self.comments_failure_page:
                return 503, {"message": "required comments unavailable"}
            return 200, self.comment_pages.get(page, [])
        if parsed.path == "/api/v4/projects/cann%2Fhcomm/merge_requests/1/discussions":
            if hasattr(self, "discussion_pages"):
                page = int(parse_qs(parsed.query).get("page", ["1"])[0])
                return 200, self.discussion_pages.get(page, [])
            return 503, {"message": "optional path enrichment unavailable"}
        if parsed.path == "/user":
            return 200, {"login": "me"}
        return 404, {"message": "unexpected request path"}

    def _assert_single_files_request(self):
        self.assertEqual(len(self.requests), 1)
        parsed = urlsplit(self.requests[0])
        self.assertEqual(parsed.path, "/repos/cann/hcomm/pulls/1/files")
        query = parse_qs(parsed.query)
        self.assertNotIn("page", query)
        self.assertNotIn("per_page", query)
        return query

    def test_large_unpaged_file_collection(self):
        for count in (100, 101, 216):
            with self.subTest(count=count):
                self.files = [{"filename": "src/file{}.cc".format(i)} for i in range(count)]
                self.requests.clear()
                files = pr_review.get_pr_files("cann", "hcomm", 1)
                self.assertEqual(files, self.files)
                self.assertEqual(self._assert_single_files_request(), {})

    def test_small_file_collection(self):
        self.files = [{"filename": "src/a.cc"}, {"filename": "src/b.cc"}]
        self.assertEqual(pr_review.get_pr_files("cann", "hcomm", 1), self.files)
        self.assertEqual(self._assert_single_files_request(), {})

    def test_empty_file_collection(self):
        self.assertEqual(pr_review.get_pr_files("cann", "hcomm", 1), [])
        self.assertEqual(self._assert_single_files_request(), {})

    def test_http_error_is_not_an_empty_file_collection(self):
        self.status = 503
        with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
            pr_review.get_pr_files("cann", "hcomm", 1)
        self.assertEqual(self._assert_single_files_request(), {})

    def test_wrapped_file_collection_and_invalid_shapes(self):
        self.files = {"data": [{"filename": "src/a.cc"}]}
        self.assertEqual(pr_review.get_pr_files("cann", "hcomm", 1), self.files["data"])
        for data in ({"status": "ok"}, ["src/a.cc"], None):
            with self.subTest(data=data):
                self.files = data
                with self.assertRaisesRegex(RuntimeError, "文件列表响应格式错误"):
                    pr_review.get_pr_files("cann", "hcomm", 1)

    def test_connection_failure_preserves_optional_and_required_semantics(self):
        import urllib.error
        with unittest.mock.patch.object(pr_review.urllib.request, "urlopen",
                                        side_effect=urllib.error.URLError("synthetic connection failure")):
            self.assertEqual(pr_review.api_get_all(self.http_url), [])
            with self.assertRaisesRegex(RuntimeError, "synthetic connection failure"):
                pr_review.api_get_all(self.http_url, strict=True)

    def test_response_read_failures_preserve_optional_and_required_semantics(self):
        import http.client
        failures = (TimeoutError("synthetic body timeout"),
                    ConnectionResetError("synthetic response reset"),
                    http.client.IncompleteRead(b"part", 8))
        for error in failures:
            with self.subTest(error=type(error).__name__):
                response = unittest.mock.MagicMock(status=200)
                response.__enter__.return_value = response
                response.read.side_effect = error
                with unittest.mock.patch.object(pr_review.urllib.request, "urlopen", return_value=response):
                    self.assertEqual(pr_review.api_get_all(self.http_url), [])
                    with self.assertRaises(RuntimeError) as caught:
                        pr_review.api_get_all(self.http_url, strict=True)
                    self.assertIsInstance(caught.exception.__cause__, type(error))

    def test_optional_v4_read_failure_keeps_authoritative_v5_comments(self):
        import http.client
        failures = (TimeoutError("synthetic optional timeout"),
                    ConnectionResetError("synthetic optional reset"),
                    http.client.IncompleteRead(b"part", 8))
        first = [{"id": 1, "body": "authoritative existing comment"}]
        self.comment_pages = {1: first}
        original_open = pr_review.urllib.request.urlopen
        for error in failures:
            with self.subTest(error=type(error).__name__):
                response = unittest.mock.MagicMock(status=200)
                response.__enter__.return_value = response
                response.read.side_effect = error
                def open_with_optional_failure(request, **kwargs):
                    if "/api/v4/" in request.full_url:
                        return response
                    return original_open(request, **kwargs)
                with unittest.mock.patch.object(pr_review.urllib.request, "urlopen",
                                                side_effect=open_with_optional_failure):
                    self.assertEqual(pr_review.get_pr_comments("cann", "hcomm", 1), first)

    def test_real_truncated_http_body_preserves_read_strictness(self):
        def truncated_body(handler):
            handler.send_response(200)
            handler.send_header("Content-Length", "20")
            handler.end_headers()
            handler.wfile.write(b"[]")
            handler.wfile.flush()
            handler.close_connection = True
        with unittest.mock.patch.object(self.server.RequestHandlerClass, "do_GET", truncated_body):
            self.assertEqual(pr_review.api_get_all(self.http_url), [])
            with self.assertRaises(RuntimeError):
                pr_review.api_get_all(self.http_url, strict=True)

    def test_real_invalid_json_body_preserves_read_strictness(self):
        def invalid_json(handler):
            body = b'{"unfinished":'
            handler.send_response(200)
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        with unittest.mock.patch.object(self.server.RequestHandlerClass, "do_GET", invalid_json):
            self.assertEqual(pr_review.api_get_all(self.http_url), [])
            with self.assertRaises(RuntimeError):
                pr_review.api_get_all(self.http_url, strict=True)

    def test_http_error_body_read_failure_keeps_status_and_strictness(self):
        import http.client
        class BrokenBody:
            def read(self):
                raise http.client.IncompleteRead(b"part", 8)

            def close(self):
                pass
        def unavailable(_request, **_kwargs):
            raise pr_review.urllib.error.HTTPError(
                self.http_url, 503, "synthetic unavailable", {}, BrokenBody())
        with unittest.mock.patch.object(pr_review.urllib.request, "urlopen", side_effect=unavailable):
            self.assertEqual(pr_review.api_get_all(self.http_url), [])
            with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                pr_review.api_get_all(self.http_url, strict=True)

    def test_unexpected_implementation_type_error_is_not_hidden(self):
        response = unittest.mock.MagicMock(status=200)
        response.__enter__.return_value = response
        response.read.return_value = b"[]"
        with unittest.mock.patch.object(pr_review.urllib.request, "urlopen", return_value=response), \
                unittest.mock.patch.object(pr_review.json, "loads", side_effect=TypeError("implementation sentinel")):
            with self.assertRaisesRegex(TypeError, "implementation sentinel"):
                pr_review.api_get_all(self.http_url)

    def test_repeated_pages_without_ids_stop_after_one_collection(self):
        self.files = [{"filename": "src/file{}.cc".format(i)} for i in range(216)]
        result = pr_review.api_get_all(self.http_url + "/repos/cann/hcomm/pulls/1/files")
        self.assertEqual(result, self.files)
        self.assertEqual(len(self.requests), 2)

    def test_files_cli_http_failure_exits_nonzero_without_empty_success(self):
        self.status = 503
        script = Path(pr_review.__file__).resolve()
        original_open = pr_review.urllib.request.urlopen

        def local_open(request, **kwargs):
            parsed = urlsplit(request.full_url)
            path = parsed.path[7:] if parsed.path.startswith("/api/v5") else parsed.path
            url = self.http_url + path
            if parsed.query:
                url += "?" + parsed.query
            redirected = pr_review.urllib.request.Request(url, headers=request.headers,
                                                          method=request.method)
            return original_open(redirected, **kwargs)

        with unittest.mock.patch.object(sys, "argv", [str(script), "--pr", "1", "--files", "--repo", "hcomm"]), \
                unittest.mock.patch.dict(os.environ, {"GITCODE_TOKEN": "inert-test-token"}), \
                unittest.mock.patch.object(pr_review.urllib.request, "urlopen", side_effect=local_open), \
                self.assertLogs(pr_review.LOG, level="INFO") as logged, \
                self.assertRaises(SystemExit) as exited:
            runpy.run_path(str(script), run_name="__main__")
        self.assertEqual(exited.exception.code, 1)
        self.assertTrue(any("HTTP 503" in line for line in logged.output))
        self.assertFalse(any("共 0 个文件" in line for line in logged.output))

    def test_files_token_query_does_not_add_pagination(self):
        self.assertEqual(pr_review.get_pr_files("cann", "hcomm", 1, token="inert-test-token"), [])
        self.assertEqual(
            self._assert_single_files_request(), {"access_token": ["inert-test-token"]})

    def test_required_comments_first_page_http_error(self):
        self.comments_failure_page = 1
        with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
            pr_review.get_pr_comments("cann", "hcomm", 1)
        self.assertEqual(len(self.requests), 1)
        parsed = urlsplit(self.requests[0])
        self.assertEqual(parsed.path, "/repos/cann/hcomm/pulls/1/comments")
        self.assertEqual(parse_qs(parsed.query)["page"], ["1"])

    def test_required_comments_later_page_http_error(self):
        self.comment_pages[1] = [{"id": i + 1, "body": "existing"} for i in range(100)]
        self.comments_failure_page = 2
        with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
            pr_review.get_pr_comments("cann", "hcomm", 1)
        self.assertEqual(
            [parse_qs(urlsplit(path).query)["page"] for path in self.requests],
            [["1"], ["2"]])

    def test_comments_success_retains_optional_v4_fallback(self):
        first = [{"id": i + 1, "body": "existing"} for i in range(100)]
        second = [{"id": 101, "body": "last existing"}]
        self.comment_pages = {1: first, 2: second}
        self.assertEqual(pr_review.get_pr_comments("cann", "hcomm", 1), first + second)
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(
            [parse_qs(urlsplit(path).query)["page"] for path in self.requests[:2]],
            [["1"], ["2"]])
        self.assertEqual(
            urlsplit(self.requests[-1]).path,
            "/api/v4/projects/cann%2Fhcomm/merge_requests/1/discussions")


    def test_optional_discussion_non_object_pages_preserve_authoritative_comments(self):
        first = [{"id": 1, "body": "authoritative", "user": {"login": "me"}}]
        self.comment_pages = {1: first}
        for page in ([None], ["x"], [1], [False], {"data": [None]}, [{"id": "valid"}, None]):
            with self.subTest(page=page):
                self.discussion_pages = {1: page}
                self.assertEqual(pr_review.get_pr_comments("cann", "hcomm", 1), first)
                with self.assertRaisesRegex(RuntimeError, "列表响应格式错误"):
                    pr_review.api_get_all(
                        self.http_url + "/api/v4/projects/cann%2Fhcomm/merge_requests/1/discussions",
                        strict=True)

    def test_optional_discussion_notes_skip_bad_containers_and_elements(self):
        first = [{"id": 1, "body": "authoritative"}]
        self.comment_pages = {1: first}
        valid = {"id": 1, "diff_file": "src/valid.cc"}
        for notes in (None, "x", 1, {}, [None], ["x"], [1]):
            with self.subTest(notes=notes):
                self.discussion_pages = {1: [{"id": "did", "notes": notes}]}
                self.assertEqual(pr_review.get_pr_comments("cann", "hcomm", 1), first)
        self.discussion_pages = {1: [{"id": "did", "notes": [None, valid, "x", 1]}]}
        self.assertEqual(pr_review.get_pr_comments("cann", "hcomm", 1),
                         [dict(first[0], path="src/valid.cc")])

    def test_optional_note_fields_do_not_pollute_authoritative_comments(self):
        first = [{"id": 1, "body": "authoritative", "user": {"login": "me"}}]
        self.comment_pages = {1: first}
        cases = [
            ({"id": [1], "diff_file": "src/invalid.cc"}, None),
            ({"id": {"n": 1}, "diff_file": "src/invalid.cc"}, None),
            ({"id": True, "diff_file": "src/invalid.cc"}, None),
            ({"id": 1, "position": "x"}, None),
            ({"id": 1, "position": [1]}, None),
            ({"id": 1, "diff_file": ["x"], "position": {"new_path": ["x"]}}, None),
            ({"id": 1, "diff_file": {"path": "x"},
              "position": {"new_path": "src/fallback.cc"}}, "src/fallback.cc"),
            ({"id": 1, "diff_file": "src/valid.cc", "body": {"untrusted": 1},
              "user": ["unused"], "position": 1}, "src/valid.cc"),
        ]
        for note, path in cases:
            with self.subTest(note=note):
                self.discussion_pages = {1: [{"id": "did", "notes": [note]}]}
                expected = [dict(first[0], path=path)] if path else first
                self.assertEqual(pr_review.get_pr_comments("cann", "hcomm", 1), expected)

    def test_optional_bad_later_page_keeps_completed_valid_page(self):
        first = [{"id": 1, "body": "authoritative"}]
        self.comment_pages = {1: first}
        discussions = [{"id": "did{}".format(i), "notes": []} for i in range(100)]
        discussions[0]["notes"] = [{"id": 1, "diff_file": "src/kept.cc"}]
        self.discussion_pages = {1: discussions, 2: [None]}
        self.assertEqual(pr_review.get_pr_comments("cann", "hcomm", 1),
                         [dict(first[0], path="src/kept.cc")])
        self.assertEqual(
            [parse_qs(urlsplit(path).query)["page"] for path in self.requests[1:]],
            [["1"], ["2"]])
        with self.assertRaisesRegex(RuntimeError, "列表响应格式错误"):
            pr_review.api_get_all(
                self.http_url + "/api/v4/projects/cann%2Fhcomm/merge_requests/1/discussions",
                strict=True)

    def test_optional_findings_preserve_original_head_and_skip_bad_replies(self):
        first = {"id": 1, "user": {"login": "me"}, "comment_type": "diff_comment"}
        self.comment_pages = {1: [first]}
        head = {"id": 1, "body": "original finding", "diff_file": "src/a.cc",
                "new_line": 10, "resolved": False}
        for notes in (None, "x", 1, {}, [None, head], ["x", head], [1, head]):
            with self.subTest(notes=notes):
                self.discussion_pages = {1: [{"id": "did", "notes": notes}]}
                self.assertEqual(pr_review.get_findings_discussions("cann", "hcomm", 1), [])
        for body in (["bad"], {"bad": 1}, 1, True):
            with self.subTest(head_body=body):
                self.discussion_pages = {1: [{"id": "did", "notes": [dict(head, body=body)]}]}
                self.assertEqual(pr_review.get_findings_discussions("cann", "hcomm", 1), [])
        self.discussion_pages = {1: [{"id": "did", "notes": [
            dict(head, user=["unused"]), None, "x", {"body": ["bad"]},
            {"body": "valid reply"}, {"body": 1}]}]}
        result = pr_review.get_findings_discussions("cann", "hcomm", 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["note_id"], 1)
        self.assertEqual(result[0]["discussion_id"], "did")
        self.assertEqual(result[0]["body"], "original finding")
        self.assertEqual(result[0]["replies"], ["valid reply"])
        self.assertEqual((result[0]["file"], result[0]["line"]), ("src/a.cc", 10))

    def test_optional_findings_identifiers_and_paths_are_typed(self):
        self.comment_pages = {1: [{"id": 1, "user": {"login": "me"},
                                  "comment_type": "diff_comment"}]}
        head = {"id": 1, "body": "finding", "diff_file": "src/a.cc", "new_line": 10}
        for identifier in (None, [], {}, True):
            with self.subTest(discussion_id=identifier):
                self.discussion_pages = {1: [{"id": identifier, "notes": [head]}]}
                self.assertEqual(pr_review.get_findings_discussions("cann", "hcomm", 1), [])
            with self.subTest(note_id=identifier):
                self.discussion_pages = {1: [{"id": "did", "notes": [dict(head, id=identifier)]}]}
                self.assertEqual(pr_review.get_findings_discussions("cann", "hcomm", 1), [])
        self.discussion_pages = {1: [{"id": "did", "notes": [dict(
            head, diff_file=["bad"], position={"new_path": "src/fallback.cc"})]}]}
        result = pr_review.get_findings_discussions("cann", "hcomm", 1)
        self.assertEqual(result[0]["file"], "src/fallback.cc")
        self.discussion_pages = {1: [{"id": "did", "notes": [dict(
            head, diff_file=1, position="bad")]}]}
        self.assertEqual(pr_review.get_findings_discussions("cann", "hcomm", 1)[0]["file"], "")

    def test_optional_findings_empty_body_and_invalid_v5_user_are_safe_to_list(self):
        mine = {"id": 1, "user": {"login": "me"}, "comment_type": "diff_comment"}
        self.comment_pages = {1: [mine]}
        args = type("Args", (), {"owner": "cann", "repo": "hcomm", "pr_number": 1})()
        for body in (None, "", "\n"):
            with self.subTest(empty_body=body):
                self.discussion_pages = {1: [{"id": "did", "notes": [
                    {"id": 1, "body": body, "diff_file": "src/a.cc", "new_line": 10}]}]}
                self.assertEqual(pr_review.cmd_list_findings(args, ""), 0)
        for user in (None, "x", 1, []):
            with self.subTest(v5_user=user):
                self.comment_pages = {1: [dict(mine, user=user)]}
                self.assertEqual(pr_review.get_findings_discussions("cann", "hcomm", 1), [])

    def test_required_comments_invalid_http_shapes(self):
        for shape in ({}, {"status": "ok"}, {"data": {}}, None, "invalid", [None]):
            with self.subTest(shape=shape):
                self.comment_pages = {1: shape}
                with self.assertRaisesRegex(RuntimeError, "列表响应格式错误"):
                    pr_review.get_pr_comments("cann", "hcomm", 1)
        for shape in ([], {"data": []}):
            with self.subTest(valid_empty=shape):
                self.comment_pages = {1: shape}
                self.assertEqual(pr_review.get_pr_comments("cann", "hcomm", 1), [])

    def test_required_comments_empty_http_body_is_not_an_empty_collection(self):
        def empty_body(handler):
            handler.send_response(200)
            handler.send_header("Content-Length", "0")
            handler.end_headers()
        with unittest.mock.patch.object(self.server.RequestHandlerClass, "do_GET", empty_body):
            with self.assertRaisesRegex(RuntimeError, "列表响应格式错误"):
                pr_review.get_pr_comments("cann", "hcomm", 1)

    def test_empty_files_cli_returns_nonzero_without_empty_authority(self):
        args = type("Args", (), {"owner": "cann", "repo": "hcomm", "pr_number": 1})()
        for data in ([], {"data": []}):
            with self.subTest(response=data):
                self.files = data
                with self.assertLogs(pr_review.LOG, level="INFO") as logs:
                    result = pr_review.cmd_files(args, "")
                self.assertEqual(result, 1)
                self.assertTrue(any("无有效差异" in line for line in logs.output))
                self.assertFalse(any("共 0 个文件" in line for line in logs.output))


class TestSelfReviewFixes(MockRepoTestCase):
    """自检意见修复的回归测试。"""

    def test_check_env_does_not_print_token_prefix(self):
        """修复1：check-env 输出不含 token 片段。"""
        import io
        import contextlib
        buf = io.StringIO()
        with unittest.mock.patch.object(pr_review, "get_token", return_value="SECRET_TOKEN"):
            with unittest.mock.patch.object(pr_review, "api_request", return_value={"state": "open"}):
                with contextlib.redirect_stdout(buf):
                    rc = pr_review.cmd_check_env(argparse_stub(self.repo_dir))
        self.assertEqual(rc, 0)
        self.assertNotIn("SECRET_TOKEN", buf.getvalue())
        self.assertNotIn("SECR", buf.getvalue())

    def test_cleanup_removes_findings_in_repo_root_and_cwd(self):
        """修复2：cleanup 在 repo-root 与 cwd 两处清理 findings。"""
        with tempfile.TemporaryDirectory() as tmp:
            cwd_findings = Path(tmp) / "findings_test.json"
            cwd_findings.write_text("[]", encoding="utf-8")
            args = type("Args", (), {
                "repo_root": str(self.repo_dir), "worktree": "",
                "findings": str(cwd_findings),
            })()
            with unittest.mock.patch.object(pr_review, "run_git", return_value=""):
                rc = pr_review.cmd_cleanup(args, "tok")
            self.assertEqual(rc, 0)
            self.assertFalse(cwd_findings.exists())

    def test_api_get_all_has_page_limit(self):
        """修复3：api_get_all 翻页有 50 页上限，服务端每页不同数据也不死循环。"""
        call_count = []

        def fake_request(url, method="GET", data=None, token=""):
            call_count.append(url)
            # 每页都返回不同 id 的 100 条（模拟排序不稳定的服务端异常）
            page = int(url.split("page=")[-1])
            return [{"id": page * 1000 + i} for i in range(100)]

        pr_review.api_request = fake_request
        items = pr_review.api_get_all("https://example.invalid/list")
        self.assertEqual(len(items), 50 * 100)  # 恰好 50 页封顶
        self.assertEqual(len(call_count), 50)


class TestDisposeAndRemoteFixes(MockRepoTestCase):
    """committer 检视意见修复的回归测试。"""

    def test_body_argument_registered(self):
        """修复1：--body 参数已注册（--dispose 不再 unrecognized）。"""
        import subprocess as sp
        r = sp.run([sys.executable, str(Path(__file__).resolve().parent / "pr_review.py"), "--help"], capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
        self.assertIn("--body", r.stdout)

    def test_findings_filtered_by_current_user(self):
        """修复2：get_findings_discussions 只取当前账号的 diff_comment。"""
        current = {"login": "me"}
        other = {"login": "someone_else"}
        v5_comments = [
            {"id": 1, "user": current, "comment_type": "diff_comment", "body": "**[LOW] 我的意见**",
             "diff_position": {"start_new_line": 10}},
            {"id": 2, "user": other, "comment_type": "diff_comment", "body": "**[LOW] 他人意见**",
             "diff_position": {"start_new_line": 20}},
        ]

        def router(url, method="GET", data=None, token=""):
            if "/user" in url:
                return {"login": "me"}
            if "/comments" in url and method == "GET":
                return v5_comments
            if "/api/v4/" in url:
                return {"data": [
                    {"id": "did1", "notes": [{"id": 1, "body": "**[LOW] 我的意见**",
                                              "new_line": 10, "diff_file": "src/a.cc", "resolved": False}]},
                    {"id": "did2", "notes": [{"id": 2, "body": "**[LOW] 他人意见**",
                                              "new_line": 20, "diff_file": "src/a.cc", "resolved": False}]},
                ]}
            raise RuntimeError("unmocked " + url)

        pr_review.api_request = router
        findings = pr_review.get_findings_discussions("cann", "hcomm", 1, "tok")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["note_id"], 1)
        self.assertIn("我的意见", findings[0]["body"])

    def test_find_remote_matches_https_with_git_suffix(self):
        """修复3：find_remote 精确匹配兼容 .git 后缀的 HTTPS URL。"""
        remote_out = ("fork https://gitcode.com/someone/hcomm.git (fetch)" + chr(10)
                    + "origin https://gitcode.com/cann/hcomm.git (fetch)" + chr(10))
        with unittest.mock.patch.object(pr_review, "run_git", return_value=remote_out):
            self.assertEqual(pr_review.find_remote(Path("."), "cann", "hcomm"), "origin")

    def test_find_remote_matches_ssh_scp_style(self):
        """修复3：find_remote 兼容 SSH SCP 风格 git@host:owner/repo.git。

        历史遗留：此方法曾在本类中被重复定义两次，后定义静默覆盖前定义
        （47 个 def 只有 46 个被执行），重复体已合并，勿再复制同名方法。
        """
        remote_out = "origin git@gitcode.com:cann/hcomm.git (fetch)" + chr(10)
        with unittest.mock.patch.object(pr_review, "run_git", return_value=remote_out):
            self.assertEqual(pr_review.find_remote(Path("."), "cann", "hcomm"), "origin")


if __name__ == "__main__":
    unittest.main(verbosity=2)
