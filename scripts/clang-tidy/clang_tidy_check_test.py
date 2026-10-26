#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Unit tests for the pure helpers of clang_tidy_check.py.

Run: python3 scripts/clang-tidy/clang_tidy_check_test.py

Covers changed-file discovery (cwd independence), -U0 added-line hunk
parsing, warning parsing (file/line/check) and compile database loading.
Requires git on PATH; all scenarios run inside throwaway repositories.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clang_tidy_check as ctc


def _git(args, cwd):
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)


class GitRepoTestCase(unittest.TestCase):
    """Base class providing a throwaway git repo with one committed file."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="clang_tidy_check_test_")
        os.makedirs(os.path.join(self.tmp, "src"))
        self._write("src/foo.cc", "int a;\nint b;\nint c;\nint d;\nint e;\n")
        _git(["init", "-q"], self.tmp)
        _git(["config", "user.email", "ut@example.com"], self.tmp)
        _git(["config", "user.name", "ut"], self.tmp)
        _git(["add", "."], self.tmp)
        _git(["commit", "-q", "-m", "base"], self.tmp)
        self.base_ref = _git(["rev-parse", "HEAD"], self.tmp).stdout.strip()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, rel, content):
        path = os.path.join(self.tmp, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(content)

    def _commit(self, rel):
        _git(["add", rel], self.tmp)
        _git(["commit", "-q", "-m", "change"], self.tmp)


class TestGetAddedLineRanges(GitRepoTestCase):
    def test_inserted_lines(self):
        self._write("src/foo.cc", "int a;\nint added1;\nint added2;\nint b;\nint c;\nint d;\nint e;\n")
        self._commit("src/foo.cc")
        added = ctc.get_added_line_ranges(self.base_ref, "src/foo.cc", self.tmp)
        self.assertEqual(added, {2, 3})

    def test_lines_added_at_top(self):
        self._write("src/foo.cc", "int z1;\nint z2;\nint a;\nint b;\nint c;\nint d;\nint e;\n")
        self._commit("src/foo.cc")
        added = ctc.get_added_line_ranges(self.base_ref, "src/foo.cc", self.tmp)
        self.assertEqual(added, {1, 2})

    def test_no_added_lines_on_pure_deletion(self):
        self._write("src/foo.cc", "int a;\nint b;\nint c;\nint e;\n")
        self._commit("src/foo.cc")
        added = ctc.get_added_line_ranges(self.base_ref, "src/foo.cc", self.tmp)
        self.assertEqual(added, set())


class TestGetChangedFiles(GitRepoTestCase):
    def test_cpp_only_and_cwd_independent(self):
        self._write("src/bar.cc", "int f;\n")
        self._write("doc/readme.md", "text\n")
        self._commit("src/bar.cc")
        self._commit("doc/readme.md")
        sub = os.path.join(self.tmp, "build")
        os.makedirs(sub)
        old = os.getcwd()
        os.chdir(sub)  # simulate build.sh invoking the check from the build dir
        try:
            files = ctc.get_changed_files(self.base_ref, self.tmp)
        finally:
            os.chdir(old)
        self.assertEqual(files, ["src/bar.cc"])

    def test_renamed_and_modified_file_included(self):
        # rename + modify: --diff-filter=ACMR must still list the new path
        _git(["mv", "src/foo.cc", "src/renamed.cc"], self.tmp)
        self._write("src/renamed.cc", "int a;\nint b;\nint c;\nint d;\nint e;\nint newline;\n")
        self._commit("src/renamed.cc")
        files = ctc.get_changed_files(self.base_ref, self.tmp)
        self.assertEqual(files, ["src/renamed.cc"])


class TestParseWarnings(unittest.TestCase):
    WARN_OWN = "src/foo.cc:42:7: warning: function has 7 parameters (threshold 6) [readability-function-size]"
    WARN_HEADER = "include/proj/bar.h:10:5: warning: function is too complex [readability-function-cognitive-complexity]"
    WARN_OTHER_CHECK = "src/foo.cc:99:1: warning: use a trailing return type [modernize-use-trailing-return-type]"
    WARN_ERROR_LINE = "error: unable to handle compilation, expected exactly one compiler job in ''"

    def test_yields_file_line_check_for_target_checks_only(self):
        out = list(ctc.parse_warnings(
            "\n".join([self.WARN_OWN, self.WARN_HEADER, self.WARN_OTHER_CHECK, self.WARN_ERROR_LINE])))
        self.assertEqual(out, [
            ("src/foo.cc", 42, "readability-function-size"),
            ("include/proj/bar.h", 10, "readability-function-cognitive-complexity"),
        ])

    def test_ignores_non_warning_lines(self):
        out = list(ctc.parse_warnings("1 warning generated.\nSuppressed 1 warnings.\n" + self.WARN_OTHER_CHECK))
        self.assertEqual(out, [])


class TestDefaultBaseRef(unittest.TestCase):
    def test_defaults_to_upstream_master(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ctc.default_base_ref(), "upstream/master")

    def test_target_branch_selects_upstream_remote(self):
        with unittest.mock.patch.dict(os.environ, {"TARGET_BRANCH": "dev"}, clear=True):
            self.assertEqual(ctc.default_base_ref(), "upstream/dev")


class TestResolveBaseRefExplicitLocalBranch(GitRepoTestCase):
    def test_local_branch_with_slash_is_used_as_is(self):
        # an explicitly passed local branch containing '/' must not be
        # silently replaced by the upstream/origin fallback
        _git(["branch", "feature/x"], self.tmp)
        old = os.getcwd()
        os.chdir(self.tmp)
        try:
            self.assertEqual(ctc.resolve_base_ref("feature/x"), "feature/x")
        finally:
            os.chdir(old)


class TestDetectRepoBasename(GitRepoTestCase):
    def _in_repo(self, fn):
        old = os.getcwd()
        os.chdir(self.tmp)
        try:
            return fn()
        finally:
            os.chdir(old)

    def test_canonical_https_url(self):
        _git(["remote", "add", "origin", "https://gitcode.com/cann/hccl.git"], self.tmp)
        self.assertEqual(self._in_repo(ctc.detect_repo_basename), "hccl")

    def test_fork_url_with_token(self):
        _git(["remote", "add", "origin", "https://user:token@gitcode.com/yanzhenghang/hccl.git"], self.tmp)
        self.assertEqual(self._in_repo(ctc.detect_repo_basename), "hccl")

    def test_ssh_url(self):
        _git(["remote", "add", "origin", "git@gitcode.com:cann/hccl.git"], self.tmp)
        self.assertEqual(self._in_repo(ctc.detect_repo_basename), "hccl")

    def test_no_gitcode_remote_returns_empty(self):
        _git(["remote", "add", "origin", "https://mirror.example.com/team/hccl.git"], self.tmp)
        self.assertEqual(self._in_repo(ctc.detect_repo_basename), "")


class TestLoadCompileDb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="clang_tidy_check_cdb_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_file_returns_none(self):
        self.assertIsNone(ctc.load_compile_db(self.tmp))

    def test_relative_entries_resolved_against_build_dir(self):
        # platform-neutral absolute path (Linux: /abs/..., Windows: C:\abs\...)
        abs_file = os.path.join(os.path.abspath(os.sep), "abs", "src", "bar.cc")
        with open(os.path.join(self.tmp, "compile_commands.json"), "w", encoding="utf-8") as fp:
            json.dump([{"file": "src/foo.cc"}, {"file": abs_file}], fp)
        db = ctc.load_compile_db(self.tmp)
        self.assertEqual(db, {
            ctc.norm_path(os.path.join(self.tmp, "src/foo.cc")),
            ctc.norm_path(abs_file),
        })


if __name__ == "__main__":
    unittest.main()
