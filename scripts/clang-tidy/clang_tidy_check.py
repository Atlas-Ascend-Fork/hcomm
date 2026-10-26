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
"""Clang-Tidy incremental check: block only complexity warnings on newly added lines.

Usage: python3 scripts/clang-tidy/clang_tidy_check.py [--build-dir build] [--base-ref upstream/master]
Exit 0: no new-code violation; Exit 1: new-code violation found (block).

Incremental base: the diff base defaults to the target branch of the gitcode
upstream remote (upstream/master, or upstream/<TARGET_BRANCH> in CI) and is
refreshed from the remote right before use, so "newly added" always means
"added by this branch relative to the latest upstream target branch". The
three-dot diff (base...HEAD) only counts lines added on this branch; lines
added upstream after the fork point are never attributed to this branch.

Exemptions (special cases):
  - // NOLINT(readability-function-size) / // NOLINTNEXTLINE(...) comments suppress
    warnings at the clang-tidy level, so suppressed warnings never reach this script.
  - Changed C/C++ sources without a compile_commands.json entry (e.g. generated or
    third-party sources) are skipped with a WARNING instead of being analyzed.

Limitations:
  - Warnings are anchored at the function definition line: growing the BODY of an
    existing function past a threshold (without touching its head line) is not
    flagged, matching the incremental (new-lines-only) interception model.
  - Standalone header analysis runs without include paths; a non-self-contained
    header that fails to parse is reported as a warning, never as a block.
"""
import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

WARN_RE = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):\d+:\s+warning:\s+.*\[(?P<check>[^\]]+)\]")
TARGET_CHECKS = ("readability-function-size", "readability-function-cognitive-complexity")
CPP_EXT = (".cc", ".cpp", ".c", ".cxx", ".h", ".hpp", ".hh", ".hxx")
HEADER_EXT = (".h", ".hpp", ".hh", ".hxx")


def run(cmd_list, cwd=None):
    """Run command from list (no shell=True) and return CompletedProcess."""
    return subprocess.run(cmd_list, cwd=cwd, capture_output=True, text=True)


def get_repo_root():
    """Return the git repository root, or '' if not inside one."""
    r = run(["git", "rev-parse", "--show-toplevel"])
    return r.stdout.strip() if r.returncode == 0 else ""


def get_changed_files(base_ref, root):
    """Return changed C/C++ files (paths relative to the repo root) vs base_ref.

    Both the git pathspec and the existence check are resolved against the
    repo root, so the result does not depend on the caller's cwd (build.sh
    invokes this script from the build directory). --diff-filter includes
    renames (R) so a renamed-and-modified file is still intercepted.
    """
    cmd = ["git", "diff", "--name-only", "--diff-filter=ACMR", base_ref + "...HEAD", "--"]
    cmd += ["*" + e for e in CPP_EXT]
    r = run(cmd, cwd=root)
    files = []
    for line in r.stdout.splitlines():
        rel = line.strip()
        if rel and os.path.isfile(os.path.join(root, rel)):
            files.append(rel)
    return files


def get_added_line_ranges(base_ref, f, root):
    """Return the set of added line numbers (1-indexed) via git diff -U0 hunk parsing."""
    r = run(["git", "diff", "-U0", base_ref + "...HEAD", "--", f], cwd=root)
    added = set()
    new_start = None
    for line in r.stdout.splitlines():
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if m:
            new_start = int(m.group(1))
            continue
        if line.startswith("+") and new_start is not None:
            added.add(new_start)
            new_start += 1
    return added


def norm_path(p):
    """Absolute normalized path, for reliable comparison of warning files."""
    return os.path.normpath(os.path.abspath(p))


def load_compile_db(build_dir):
    """Return the set of normalized absolute paths covered by compile_commands.json.

    Returns None if the file is missing or unreadable (caller skips the check);
    the empty set if it exists but lists no files.
    """
    path = os.path.join(build_dir, "compile_commands.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fp:
            entries = json.load(fp)
    except (OSError, ValueError):
        log.warning("[clang-tidy] WARNING: %s unreadable, skip", path)
        return None
    files = set()
    for entry in entries:
        f = entry.get("file", "")
        if f:
            if not os.path.isabs(f):
                f = os.path.join(build_dir, f)
            files.add(norm_path(f))
    return files


def run_clang_tidy(f, build_dir, config_file):
    """Run clang-tidy on f with the compile database; returns (output, returncode)."""
    r = run(["clang-tidy", "--config-file=" + config_file, "-p", build_dir, f])
    return r.stdout + r.stderr, r.returncode


def run_clang_tidy_standalone(f, config_file):
    """Run clang-tidy on a header that has no compile database entry."""
    r = run(["clang-tidy", "--config-file=" + config_file, f, "--", "-x", "c++"])
    return r.stdout + r.stderr, r.returncode


def parse_warnings(output):
    """Yield (file, line, check) for target-check warnings only.

    The file field is kept so callers can filter out warnings reported for
    headers included by the checked file.
    """
    for line in output.splitlines():
        m = WARN_RE.match(line)
        if m and m.group("check") in TARGET_CHECKS:
            yield m.group("file"), int(m.group("line")), m.group("check")


def default_base_ref():
    """Default diff base: the target branch of the gitcode upstream remote.

    TARGET_BRANCH (set by CI) only selects the branch name; the remote is
    always upstream first (the gitcode canonical repo, e.g. cann/hcomm), so a
    stale local branch can never become the base. resolve_base_ref refreshes
    the remote-tracking ref right before use.
    """
    tb = os.environ.get("TARGET_BRANCH", "")
    return ("upstream/" + tb) if tb else "upstream/master"


def detect_repo_basename():
    """Infer the repository basename from any gitcode remote URL.

    Works for the canonical repo (cann/hccl) and any fork (someone/hccl), for
    https and ssh URL forms. Returns '' if no gitcode remote is configured.
    """
    for remote in ("upstream", "origin"):
        r = run(["git", "remote", "get-url", remote])
        if r.returncode != 0:
            continue
        url = r.stdout.strip().rstrip("/")
        m = re.match(r".*(?:gitcode\.com[/:])[^/]+/([^/]+?)(?:\.git)?$", url)
        if m and m.group(1):
            return m.group(1)
    return ""


def fetch_canonical_target(tb):
    """Fetch the target branch straight from the canonical gitcode repository.

    Last-resort fallback for developers whose remotes are named differently
    or point elsewhere: derive the repo name from any gitcode remote and
    fetch cann/<repo> directly, so the diff base is always the true gitcode
    upstream target branch. Returns the ref name or ''.
    """
    basename = detect_repo_basename()
    if not basename:
        return ""
    url = "https://gitcode.com/cann/" + basename + ".git"
    ref = "refs/remotes/gitcode-upstream/" + tb
    if run(["git", "fetch", "--quiet", url, "+refs/heads/" + tb + ":" + ref]).returncode == 0:
        candidate = "gitcode-upstream/" + tb
        if run(["git", "rev-parse", "--verify", candidate]).returncode == 0:
            return candidate
    return ""


def resolve_base_ref(base_ref):
    """Resolve the diff base: refresh the ref if it exists, else fetch the target branch.

    A locally-present remote-tracking ref is refreshed before use so a stale
    upstream/master cannot make the diff base outdated (which would count
    upstream's lines as new and false-block). If the refresh fails (offline
    development), the local ref is still used with a warning. The fallback
    tries the remote named by the default base ref (upstream) first, then
    origin (the only remote in CI clones of cann/<repo>), and finally the
    canonical gitcode repository fetched directly (so the base is correct
    even when the developer's remotes are named differently). The branch
    name is taken from TARGET_BRANCH or derived from the default base ref.
    """
    if run(["git", "rev-parse", "--verify", base_ref]).returncode == 0:
        if "/" in base_ref:
            remote, tb = base_ref.split("/", 1)
            if run(["git", "remote", "get-url", remote]).returncode == 0:
                fr = run(["git", "fetch", "--quiet", remote, "+refs/heads/" + tb + ":refs/remotes/" + remote + "/" + tb])
                if fr.returncode != 0:
                    log.warning("[clang-tidy] WARNING: failed to refresh %s, using local (possibly stale) ref", base_ref)
                if run(["git", "rev-parse", "--verify", base_ref]).returncode == 0:
                    return base_ref
            else:
                # first segment is not a known remote (e.g. a local branch
                # named feature/x): honor the explicitly passed ref as-is
                return base_ref
        else:
            return base_ref
    tb = os.environ.get("TARGET_BRANCH", "")
    if not tb and "/" in base_ref:
        tb = base_ref.rsplit("/", 1)[1]
    if not tb:
        return ""
    for remote in ("upstream", "origin"):
        if run(["git", "remote", "get-url", remote]).returncode != 0:
            continue
        run(["git", "fetch", "--quiet", remote, "+refs/heads/" + tb + ":refs/remotes/" + remote + "/" + tb])
        candidate = remote + "/" + tb
        if run(["git", "rev-parse", "--verify", candidate]).returncode == 0:
            return candidate
    return fetch_canonical_target(tb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default=os.environ.get("CLANG_TIDY_BUILD_DIR", "build"))
    ap.add_argument("--base-ref", default=default_base_ref())
    ap.add_argument("--config-file", default=".clang-tidy")
    args = ap.parse_args()

    root = get_repo_root()
    if not root:
        log.warning("[clang-tidy] WARNING: not in a git repository, skip (not block)")
        return 0
    os.chdir(root)
    if not os.path.isabs(args.build_dir):
        args.build_dir = os.path.join(root, args.build_dir)
    if not os.path.isabs(args.config_file):
        args.config_file = os.path.join(root, args.config_file)

    if not shutil.which("clang-tidy"):
        log.warning("[clang-tidy] WARNING: clang-tidy not installed, skip (not block)")
        log.warning("[clang-tidy] hint: yum install -y clang-tools-extra")
        return 0
    if not os.path.isfile(os.path.join(args.build_dir, "compile_commands.json")):
        log.warning("[clang-tidy] WARNING: %s/compile_commands.json not found, skip (build first)", args.build_dir)
        return 0
    base = resolve_base_ref(args.base_ref)
    if not base:
        log.warning("[clang-tidy] WARNING: cannot resolve base ref %s, skip", args.base_ref)
        return 0

    files = get_changed_files(base, root)
    if not files:
        log.info("[clang-tidy] no changed C++ files, skip")
        return 0
    compile_db = load_compile_db(args.build_dir)
    if compile_db is None:
        return 0
    log.info("[clang-tidy] incremental check: base=%s, %d changed file(s)", base, len(files))

    violations = []
    source_failed = []
    header_failed = 0
    for f in files:
        # Skip files with no added lines before invoking clang-tidy: only new
        # code can be intercepted, and pure-deletion files need no analysis.
        added = get_added_line_ranges(base, f, root)
        if not added:
            continue
        is_header = f.endswith(HEADER_EXT)
        if norm_path(f) not in compile_db:
            if is_header:
                # Headers have no compile command of their own: analyze standalone
                # so new complexity in headers is still intercepted.
                output, rc = run_clang_tidy_standalone(f, args.config_file)
                if rc != 0:
                    # Standalone headers frequently cannot parse without -I flags:
                    # warn only, never block.
                    header_failed += 1
                    log.warning("[clang-tidy] WARNING: standalone analysis failed on header %s (exit %d), warn-only", f, rc)
            else:
                # Generated/third-party sources are out of scope: exempt them
                # explicitly instead of failing their analysis.
                log.warning("[clang-tidy] WARNING: %s has no entry in compile_commands.json (generated/third-party?), skip", f)
                continue
        else:
            output, rc = run_clang_tidy(f, args.build_dir, args.config_file)
            if rc != 0:
                # A changed source that cannot be analyzed would be a silent gap in
                # interception: treat it as a blocking failure.
                source_failed.append(f)
                log.warning("[clang-tidy] WARNING: clang-tidy exited %d on %s, analysis failure", rc, f)
        for wfile, lno, check in parse_warnings(output):
            if norm_path(wfile) == norm_path(f) and lno in added:
                violations.append((f, lno, check))

    if violations:
        log.warning("[clang-tidy] FAIL: %d new-code complexity violation(s):", len(violations))
        for f, lno, check in violations:
            log.warning("  %s:%d [%s]", f, lno, check)
        return 1
    if source_failed:
        log.warning("[clang-tidy] FAIL: analysis failed on %d changed source file(s), interception incomplete:", len(source_failed))
        for f in source_failed:
            log.warning("  %s", f)
        return 1
    if header_failed:
        log.warning("[clang-tidy] WARNING: %d header file(s) failed standalone analysis (warn-only)", header_failed)
    log.info("[clang-tidy] PASS: no new-code complexity violation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
