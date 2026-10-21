# CheckerL2 pre-commit Usage Guide

The hcomm repository has two pre-commit hooks configured. The CheckerL2 repository also needs these two hooks configured:
- **clang-format**: C/C++ code formatting (based on `.clang-format` rules, clang-format v18.1.8)
- **oat-check**: OAT open source compliance check (license header check and so on)

---

## I. Initial Configuration (One-time, Only Once Per Person)

### 1. Install pre-commit

```bash
pip3 install pre-commit
```

Verify installation:

```bash
pre-commit --version
# Expected output similar to: pre-commit 3.6.2
```

### 2. Install Hooks to Local Repository

Execute in the code root directory (the CheckerL2 directory containing `.pre-commit-config.yaml`):

```bash
cd <code root directory>
pre-commit install
```

Output:

```
pre-commit installed at .git/hooks/pre-commit
```

> After installation, every `git commit` automatically triggers hook checks.

### 3. First Run (Download Hook Tools)

```bash
pre-commit run --all-files
```

The first run automatically downloads tools such as clang-format, taking about 1-2 minutes. Subsequent runs use the cache.

---

## II. Daily Usage

### Method A: Automatic Trigger on Commit (Recommended)

```bash
git add <modified files>
git commit -m "[fix] your commit message"
```

If files do not meet format specifications, the commit is automatically blocked.

### Method B: Manually Execute Checks on Specified Files

```bash
# Check only clang-format
pre-commit run clang-format --files path/to/file1.cc path/to/file2.h

# Check only oat-check
pre-commit run oat-check --files path/to/file1.cc

# Check all hooks
pre-commit run --files path/to/file1.cc path/to/file2.h
```

### Method C: Use Helper Script (Recommended for Daily Use)

The repository provides the `pre-commit_check.sh` script, which runs clang-format + oat-check at once and outputs a concise result summary:

```bash
# Check specified files
bash ./pre-commit_check.sh path/to/file1.cc path/to/file2.h

# Check modified files in git (most common, --modified or -modified both work)
bash ./pre-commit_check.sh --modified

# Check all C/C++ files in repository (--all or -all both work)
bash ./pre-commit_check.sh --all
```

**Output Examples:**

✅ All Passed:
```
Found 3 C/C++ file(s) to check

Running clang-format on 3 file(s)...
Running oat-check on 3 file(s)...

======================================
PASS: all 3 file(s) passed clang-format and oat-check
======================================
```

❌ Failures Found:
```
======================================
ERROR: FAILED: 2 issue(s) found

  ✗ path/to/file.cc
    - clang-format: auto-formatted (run 'git add path/to/file.cc' and retry)
  ✗ path/to/file.cc
    - oat-check: License Header Invalid (NoLicenseHeader)

======================================
```

> **The script automatically fixes clang-format issues and precisely identifies files and reasons for oat-check failures. After fixing, rerun the script to confirm passage before committing.**

---

## III. Check Result Explanation

### ✅ Check Passed

```
clang-format.............................................................Passed
OAT Compliance Check (Python Edition)....................................Passed
```

Both hooks show `Passed`; you can commit normally.

---

### ❌ clang-format Failed

Output example:

```
clang-format.............................................................Failed
- hook id: clang-format
- files were modified by this hook

Formatting [1/1] path/to/file.cc
```

**Meaning**: Code format does not meet `.clang-format` specifications; the hook has **automatically modified the file** (formatted in place).

**Resolution Steps**:

1. View the modified content:
   ```bash
   git diff path/to/file.cc
   ```

2. The hook has automatically corrected the format for you; simply re-add and commit:
   ```bash
   git add path/to/file.cc
   git commit -m "[fix] your commit message"
   ```

> **Summary: clang-format failures do not require manual code changes; simply rerun `git add` + `git commit`.**

---

### ❌ oat-check Failed

Output example:

```
OAT Compliance Check (Python Edition)....................................Failed
- hook id: oat-check
- exit code: 1

[OAT] Found 1 files to scan.
====================================================================
  OAT: Compliance issues found. Commit blocked.
====================================================================

[OAT] License Header Invalid Total Count: 1
Name:  License Header Invalid   Content:  NoLicenseHeader   File: path/to/file.cc
```

**Meaning**: The file is missing the CANN-2.0 license header, or the license header format is incorrect.

**Resolution Steps**:

1. Add the standard license header at the top of the file (copy from any existing `.cc` file in the repository):

   ```c
   /**
    * Copyright (c) 2026 Huawei Technologies Co., Ltd.
    * This program is free software, you can redistribute it and/or modify it under
    * the terms and conditions of CANN Open Software License Agreement Version 2.0
    * (the "License"). For the License for details. You may not use
    * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
    * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
    * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
    * for the full text of the License.
    */
   ```

2. Re-add and commit:
   ```bash
   git add path/to/file.cc
   git commit -m "[fix] your commit message"
   ```

> **Note: OAT does not automatically fix files; you must manually add the license header.**

---

## IV. Common Issues

| Issue | Cause | Resolution |
|-------|-------|------------|
| `pre-commit: command not found` | Not installed | `pip3 install pre-commit` |
| `invalid choice: 'intsall'` | Spelling error | `pre-commit install` (check spelling) |
| `unrecognized arguments: -files` | Parameter format error | `--files` (double dash) |
| First run is slow / download fails | Need to download clang-format binary | Check network or configure proxy |
| clang-format repeatedly fails | File not re-added to staging area | After hook modifies file, you must rerun `git add` |
| Urgent commit needs to skip checks | — | `git commit --no-verify` (not recommended) |
