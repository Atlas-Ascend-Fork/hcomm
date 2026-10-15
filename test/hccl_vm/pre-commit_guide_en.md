# CheckerL2 pre-commit 使用指导

hcomm 仓配置了两个 pre-commit hook, 当前CheckerL2仓库也需要配置这两个 hook：
- **clang-format**：C/C++ 代码格式化（基于 `.clang-format` 规则，clang-format v18.1.8）
- **oat-check**：OAT 开源合规检查（许可证头检查等）

---

## 一、初始配置（一次性，每人只需做一次）

### 1. 安装 pre-commit

```bash
pip3 install pre-commit
```

验证安装：

```bash
pre-commit --version
# 预期输出类似：pre-commit 3.6.2
```

### 2. 安装 hook 到本地仓库

在代码根目录（即 CheckerL2 目录，包含 `.pre-commit-config.yaml` 的目录）执行：

```bash
cd <代码根目录>
pre-commit install
```

输出：

```
pre-commit installed at .git/hooks/pre-commit
```

> 安装后，每次 `git commit` 会自动触发 hook 检查。

### 3. 首次运行（下载 hook 工具）

```bash
pre-commit run --all-files
```

首次运行会自动下载 clang-format 等工具，耗时约 1-2 分钟。后续运行会使用缓存。

---

## 二、日常使用

### 方式 A：commit 时自动触发（推荐）

```bash
git add <修改的文件>
git commit -m "[fix] 你的提交信息"
```

如果文件不符合格式规范，commit 会被自动阻止。

### 方式 B：手动对指定文件执行检查

```bash
# 只检查 clang-format
pre-commit run clang-format --files path/to/file1.cc path/to/file2.h

# 只检查 oat-check
pre-commit run oat-check --files path/to/file1.cc

# 检查所有 hook
pre-commit run --files path/to/file1.cc path/to/file2.h
```

### 方式 C：使用辅助脚本（推荐日常使用）

仓库提供了 `pre-commit_check.sh` 脚本，一次性运行 clang-format + oat-check，并输出简洁的结果摘要：

```bash
# 检查指定文件
bash ./pre-commit_check.sh path/to/file1.cc path/to/file2.h

# 检查 git 中已修改的文件（最常用，--modified 或 -modified 均可）
bash ./pre-commit_check.sh --modified

# 检查仓库所有 C/C++ 文件（--all 或 -all 均可）
bash ./pre-commit_check.sh --all
```

**输出示例：**

✅ 全部通过：
```
Found 3 C/C++ file(s) to check

Running clang-format on 3 file(s)...
Running oat-check on 3 file(s)...

======================================
PASS: all 3 file(s) passed clang-format and oat-check
======================================
```

❌ 有失败：
```
======================================
ERROR: FAILED: 2 issue(s) found

  ✗ path/to/file.cc
    - clang-format: auto-formatted (run 'git add path/to/file.cc' and retry)
  ✗ path/to/file.cc
    - oat-check: License Header Invalid (NoLicenseHeader)

======================================
```

> **脚本会自动修复 clang-format 问题，并精确指出 oat-check 失败的文件和原因。修复后重新运行脚本确认通过即可提交。**

---

## 三、检查结果说明

### ✅ 检查通过

```
clang-format.............................................................Passed
OAT Compliance Check (Python Edition)....................................Passed
```

两个 hook 都显示 `Passed`，可以正常 commit。

---

### ❌ clang-format 失败

输出示例：

```
clang-format.............................................................Failed
- hook id: clang-format
- files were modified by this hook

Formatting [1/1] path/to/file.cc
```

**含义**：代码格式不符合 `.clang-format` 规范，hook 已**自动修改文件**（原地格式化）。

**处理步骤**：

1. 查看被修改的内容：
   ```bash
   git diff path/to/file.cc
   ```

2. hook 已自动帮你修正了格式，直接重新 add 并 commit：
   ```bash
   git add path/to/file.cc
   git commit -m "[fix] 你的提交信息"
   ```

> **总结：clang-format 失败不需要手动改代码，重新 `git add` + `git commit` 即可。**

---

### ❌ oat-check 失败

输出示例：

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

**含义**：文件缺少 CANN-2.0 许可证头，或许可证头格式不正确。

**处理步骤**：

1. 在文件顶部添加标准许可证头（从仓内任意已有 `.cc` 文件复制）：

   ```c
   /**
    * Copyright (c) 2026 Huawei Technologies Co., Ltd.
    * This program is free software, you can redistribute it and/or modify it under
    * the terms and conditions of CANN Open Software License Agreement Version 2.0
    * (the "License"). Please refer to the License for details. You may not use
    * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
    * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
    * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
    * for the full text of the License.
    */
   ```

2. 重新 add 并 commit：
   ```bash
   git add path/to/file.cc
   git commit -m "[fix] 你的提交信息"
   ```

> **注意：OAT 不会自动修复文件，需要你手动添加许可证头。**

---

## 四、常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `pre-commit: command not found` | 未安装 | `pip3 install pre-commit` |
| `invalid choice: 'intsall'` | 拼写错误 | `pre-commit install`（注意拼写） |
| `unrecognized arguments: -files` | 参数格式错误 | `--files`（双横线） |
| 首次运行很慢 / 下载失败 | 需要下载 clang-format 二进制 | 检查网络，或配置代理 |
| clang-format 反复 Failed | 文件未被 re-add 到暂存区 | hook 修改文件后必须重新 `git add` |
| 紧急提交想跳过检查 | — | `git commit --no-verify`（不推荐） |
