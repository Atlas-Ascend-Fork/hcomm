# HCCL_VM UT Test Execution Script

## Overview

`run_ut.sh` is the unit test execution script in the HCCL_VM project. It automates compilation and test case execution.

## Three-Step Workflow

The script automatically completes the following three steps:

| Step | Description | Log Output |
|------|-------------|------------|
| Step 1 | CMake configuration + make compilation | `build.log` |
| Step 2 | Generate executable files | List all binaries with size and time |
| Step 3 | Execute cases + display results | `run.log` + `summary.log` |
| Step 4 | Generate gcov/lcov coverage HTML report (requires `--cov`) | `coverage.log` |

## Usage

```bash
cd {HCCL_VM path}/test

# Basic usage
bash ./run_ut.sh                           # Full compilation + execute all tests
bash ./run_ut.sh --cov                     # Full compilation + execute + generate gcov/lcov coverage HTML report
bash ./run_ut.sh <directory>               # Compile + execute all tests in the specified directory (recursive)
bash ./run_ut.sh <binary name>             # Compile + execute the specified binary test
bash ./run_ut.sh <test file name>          # Compile + execute the binary corresponding to the specified test file
bash ./run_ut.sh <file> <test case name>   # Compile + execute a single test case in the specified file

# Other commands
bash ./run_ut.sh -l, --list                # List all available tests
bash ./run_ut.sh -h, --help                # Display help information
```

## Command Details

### 1. Full Execution

```bash
bash ./run_ut.sh
```

- Execute all tests under the `test` directory.
- Complete three-step workflow: compile → generate executables → execute all tests.
- Suitable for full regression testing.

### 2. Coverage Report

```bash
bash ./run_ut.sh --cov
```

- Full compilation + execute all tests + generate gcov/lcov coverage HTML report.
- The script automatically enables the `--coverage` compilation option and generates `.gcno`/`.gcda` files.
- Four-step workflow: compile (with coverage instrumentation) → execute → collect coverage data → generate HTML report.
- Report output path: `$CODE_DIR/coverage_report/html/index.html`.
- The script automatically filters system headers, third-party libraries, stub files, and other non-business code.

### 3. Directory Execution

```bash
bash ./run_ut.sh plugin/checker
bash ./run_ut.sh plugin/ccu_executor
bash ./run_ut.sh store
```

- Recursively find all `*_test.cc` files under the specified directory.
- Compile and execute all tests in that directory.
- Suitable for module-level testing.

### 4. Binary Execution

```bash
bash ./run_ut.sh test_checker
bash ./run_ut.sh test_allgather_semantics_checker
```

- Compile and execute the specified test binary.
- Binary names start with `test_`.

### 5. File Execution

```bash
bash ./run_ut.sh checker_test.cc
bash ./run_ut.sh allgather_semantics_checker_test.cc
```

- The script automatically matches the corresponding binary based on the test file name.
- Compile and execute.

### 6. Single Case Execution

```bash
bash ./run_ut.sh checker_test.cc CheckerTest.GenAndCheckGraph_EmptyQueues
bash ./run_ut.sh allgather_semantics_checker_test.cc AllgatherSemanticsCheckerTest.CheckBasic
```

- Execute a single test case in the specified test file.
- Case name format: `TestSuiteName.TestCaseName`.

### 7. List Tests

```bash
bash ./run_ut.sh -l
bash ./run_ut.sh --list
```

- List all available test files and their status.
- Display case count and corresponding binary name.

## Samples

```bash
# Sample 1: Full test
bash ./run_ut.sh

# Sample 2: Full test + coverage report
bash ./run_ut.sh --cov

# Sample 3: Execute all tests under plugin/checker
bash ./run_ut.sh plugin/checker

# Sample 4: Compile and execute test_checker
bash ./run_ut.sh test_checker

# Sample 5: Compile and execute the binary for checker_test.cc
bash ./run_ut.sh checker_test.cc

# Sample 6: Execute a single test case
bash ./run_ut.sh checker_test.cc CheckerTest.GenAndCheckGraph_EmptyQueues

# Sample 7: List all tests
bash ./run_ut.sh -l
```

## Log Directory

Each execution generates log files under `ut_logs/<timestamp>/`:

```text
{HCCL_VM path}/ut_logs/20260425_142048/
├── build.log      # Compilation detailed log (cmake + make output)
├── run.log        # Execution detailed log (full output for each test)
└── summary.log    # Summary log (execution result for each test)
```

### Log File Description

| File | Content |
|------|---------|
| `build.log` | CMake configuration output, make compilation output, generated executable list |
| `run.log` | Full output for each test (including gtest detailed information) |
| `summary.log` | Execution status for each test, pass/fail count, duration summary |

## Execution Results

The script displays summary information after completion:

```text
========================================
  Directory Test Result Summary: plugin/checker
========================================
  Execution count:   16
  PASSED:  185
  FAILED:  8
  CRASHED: 0
  TIMEOUT: 0
```

### Status Description

| Status | Description |
|--------|-------------|
| `PASS` | Test passed |
| `FAIL` | Test failed (assertion failure) |
| `CRASH` | Test crashed (core dump) |
| `TIMEOUT` | Test timed out (default 60 seconds) |

## Directory Structure

```text
test/
├── run_ut.sh              # This script
├── cmd/                   # Command-related tests
│   ├── base/
│   ├── subcmds/
│   └── utils/
├── device_arm/            # Device-related tests
├── device_vir/
├── ipc/                   # IPC-related tests
│   └── shm/
├── log/                   # Log-related tests
├── plugin/                # Plugin-related tests
│   ├── ccu_executor/
│   │   ├── control_type/
│   │   ├── load_type/
│   │   ├── reduce_type/
│   │   └── trans_type/
│   └── checker/
│       └── framework/
│           ├── mem_conflict_check/
│           ├── semantics_check/
│           └── singletask_check/
├── proxy/                 # Proxy-related tests
│   ├── level1/
│   └── level2/
├── runnerdb/              # Database-related tests
├── store/                 # Store-related tests
│   └── hccl_shm/
└── src_root/              # Source root directory tests
```

## Environment Requirements

**Before executing the script, modify the following environment variables in `run_ut.sh` to the actual paths:**

```bash
# The following are sample paths. Modify them according to the actual environment.
export HCOMM_CODE_HOME=/home/q30033976/checker/hcomm      # hcomm source path
export HCCL_CODE_HOME=/home/q30033976/checker/hccl        # hccl source path (required for AIV/AICPU mode)
source /home/q30033976/checker/Ascend/cann/set_env.sh     # Environment script under the CANN installation path
```

**Sample**: Assume your working directory is `/home/workspace`. Modify as follows:

```bash
export HCOMM_CODE_HOME=/home/workspace/hcomm
export HCCL_CODE_HOME=/home/workspace/hccl
source /home/workspace/Ascend/cann/set_env.sh
```

The script automatically loads these environment variables. Unmodified paths will cause compilation failure.

## Configuration Parameters

The script contains the following built-in configurations (modifiable at the top of the script):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Debug` | CMake build type |
| `MAKE_JOBS` | `8` | make parallel compilation count |
| `LOG_DIR` | `$CODE_DIR/ut_logs` | Log output directory |

## Precautions

1. **First execution**: The first execution performs a full CMake configuration, which takes a long time.
2. **Compilation failure**: If compilation fails, check `build.log` for detailed error information.
3. **Test failure**: If a test fails, check `run.log` for the specific failed cases.
4. **Log cleanup**: Log directories are named by timestamp. Clean old logs regularly to save space.

## Frequently Asked Questions

### Q: How to compile only without executing?

A: The script is designed as an integrated compile-and-execute workflow. To compile only, use the cmake and make commands directly:

```bash
cd {HCCL_VM path}/build
cmake .. && make -j8 test_checker
```

### Q: How to view detailed output for a test?

A: Check the `run.log` file, which contains the complete gtest output for each test.

### Q: What to do when a test times out?

A: The default timeout is 60 seconds. Modify the `timeout_sec` parameter in the script.

### Q: How to add new tests?

A: Create a `*_test.cc` file in the corresponding directory and add the compilation target in the corresponding CMakeLists.txt.

## Viewing Coverage Reports

After generating the HTML coverage report, start an HTTP server on Linux to view it:

```bash
cd <path where coverage_report/html resides>
python3 -m http.server 8080
```

- Local browser access: `http://localhost:8080`.
- Remote server: Access locally after SSH port forwarding.

```bash
ssh -L 8080:localhost:8080 <user>@<server_ip>
# Then open http://localhost:8080 in the local browser
```

Press `^C` to stop the server.

## Version History

- v1.0 - Initial version. Supports full/directory/file/case-level test execution.
- v2.0 - Optimized command-line arguments. Supports automatic path derivation and three-step workflow logging.
- v2.1 - Added `--cov` parameter. Supports gcov/lcov coverage HTML report generation.
