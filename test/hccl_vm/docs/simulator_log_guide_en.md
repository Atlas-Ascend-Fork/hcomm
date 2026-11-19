# Simulator Log Usage Guide (stub/cann Dual Backends)

This document is intended for users who use and troubleshoot HCCL-VM logs. It covers the configuration methods, on-disk locations, level semantics, and output formats of the two log backends (`stub` and `cann`), as well as the flag logs printed to the terminal after each command is complete.

## Table of Contents

- [1. Key Features](#1-key-features)
- [2. Quick Start](#2-quick-start)
- [3. On-Disk Log Locations of Each Process](#3-on-disk-log-locations-of-each-process)
- [4. stub Mode](#4-stub-mode)
  - [4.1 Configuration Sources](#41-configuration-sources)
  - [4.2 Configuration File Field Table](#42-configuration-file-field-table)
  - [4.3 On-Disk Location and Naming](#43-on-disk-location-and-naming)
  - [4.4 Output Format](#44-output-format)
- [5. cann Mode](#5-cann-mode)
  - [5.1 Directly Reusing the Genuine CANN Library](#51-directly-reusing-the-genuine-cann-library)
  - [5.2 Log Levels](#52-log-levels)
  - [5.3 Device Simulation Processes Started by qemu: Using spdlog to Emulate CANN Text](#53-device-simulation-processes-started-by-qemu-using-spdlog-to-emulate-cann-text)
- [6. When Configuration Takes Effect, Priorities, and Process Boundaries](#6-when-configuration-takes-effect-priorities-and-process-boundaries)
  - [6.1 Taking Effect Only at Startup](#61-taking-effect-only-at-startup)
  - [6.2 Priorities](#62-priorities)
  - [6.3 Where export HCCL_VM_LOG_MODE Is Written Affects the Log Attribution of Plugins](#63-where-export-hccl_vm_log_mode-is-written-affects-the-log-attribution-of-plugins)
  - [6.4 The Log Directory of stub Mode Is Cleared at Each Startup](#64-the-log-directory-of-stub-mode-is-cleared-at-each-startup)
- [7. Terminal Flag Logs](#7-terminal-flag-logs)
  - [7.1 Overview](#71-overview)
  - [7.2 Points Requiring Attention](#72-points-requiring-attention)

## 1. Key Features

Simulator logs have a dual-backend mechanism. All simulator logs use one of the two backends determined at process startup:

- **stub (default)**: The simulator has a built-in spdlog dual channel (console + file). The level, path, and rotation are all controlled by `config/log_config.yaml`, and the logs are written to `<installation root>/logs/`.
- **cann**: Reuses the CANN log capability. Host-side processes (the hccl-vm main process, checker, runner, and rank service processes) directly call the genuine CANN library `libunified_dlog.so`. Device processes do not load the genuine library. Instead, they use spdlog to emulate the CANN text format, and the logs are written to `run/device-<device primary key>/` under the CANN log root.

## 2. Quick Start

```bash
# Default (stub): no settings are required
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# cann (recommended to export before starting hccl-vm)
source /path/to/Ascend/cann/set_env.sh
export HCCL_VM_LOG_MODE=cann
export ASCEND_GLOBAL_LOG_LEVEL=1                     # 0-DEBUG 1-INFO 2-WARNING 3-ERROR 4-NULL
./hccl-vm start ascend950_cluster_32_server_normal.yaml
```

To enable cann only for service processes (rank/device) while keeping stub for the host and plugins, run export in the `(hvm)$>` terminal and then start `mpirun` (for the reason, see the [process boundary table in section 6.3](#63-where-export-hccl_vm_log_mode-is-written-affects-the-log-attribution-of-plugins)).

## 3. On-Disk Log Locations of Each Process

| Process | stub On-Disk Location | cann On-Disk Location |
| --- | --- | --- |
| hccl-vm main process (command interpreter and parent process of the interactive shell) | `logs/hccl_vm_<pid>.log` | Determined by the genuine CANN library (following its `ASCEND_*` configuration) |
| checker plugin | `logs/checker/checker_<pid>.log` | Same as above |
| runner plugin | `logs/runner/runner_<pid>.log` | Same as above |
| rank service process (LD_PRELOAD proxy library) | `logs/proxy/proxy_<pid>.log` | Same as above |
| device process | `logs/proxy/proxy_<parent pid>_<current pid>.log` | `<CANN log root>/run/device-<devKey>/device-<parent pid>_<initialization time in microseconds>.log`. The log root defaults to `~/ascend/log` and can be overridden by `ASCEND_PROCESS_LOG_PATH` (or `ASCEND_WORK_PATH/log` as the second choice) |

The relative paths in the preceding table are based on the installation root (`HCCL_VM_INSTALL_ROOT`, which is the parent directory of `bin/`). The cann path of a device process is assembled by the simulator itself and does not go through the genuine CANN library.

## 4. stub Mode

### 4.1 Configuration Sources

`<installation root>/config/log_config.yaml` configures the simulator log levels. The YAML file selects nodes by process name:

| Process Name (Passed in by Code) | YAML Node | Description |
| --- | --- | --- |
| `hccl_vm` | `hccl_vm` | hccl-vm main process |
| `proxy` | `proxy` | rank service process |
| `device_aarch64` | `proxy` | Reuses the proxy section. The file base name is changed to `proxy_<parent pid>`. |
| `runner` | `runner` | runner plugin |
| `checker` | `checker` | checker plugin |

### 4.2 Configuration File Field Table

| Field | Default Value in Code | Effective | Description |
| --- | --- | --- | --- |
| `console_level` | `2` (info) | Yes | Console log level, 0-6. Out-of-range values are clamped to 0/6. |
| `file_level` | `1` (debug) | Yes | Log file on-disk level, 0-6. |
| `max_file_size` | `52428800` (50 MB) | Yes | Size of a single file, in bytes. |
| `max_files` | `65535` | Yes | Maximum number of archived files. |
| `file_path` | `logs` | Yes | A relative path is assembled based on `<installation root>`. A path starting with `/` or `./` is used as is (`./` is relative to the **current working directory of the process**). |
| `file_suffix` | `.log` | Yes | Rotation renaming relies on suffix matching. Keep custom suffixes consistent. |
| `file_base_name` | — | **Not effective yet** | The current file base name is fixed to the process name (`proxy_<parent pid>` for a device process). |
| `enable_compress` | `false` | **Not effective yet** | Log compression is not supported yet. |

The level values use the unified spdlog semantics: `0-trace 1-debug 2-info 3-warn 4-error 5-critical 6-off`.

### 4.3 On-Disk Location and Naming

`<file_path>/<file_base_name>_<pid><file_suffix>`, where the base name is determined by the process name. Examples:

```
<installation root>/logs/hccl_vm_153313.log
<installation root>/logs/checker/checker_153364.log
<installation root>/logs/proxy/proxy_153778.log                    # rank service process
<installation root>/logs/proxy/proxy_153778_153777.log             # device process (parent pid_current pid)
```

If the directory of a file does not exist, spdlog creates it automatically.

### 4.4 Output Format

Console pattern:

```
[%l][PID:%P][TID:%t][%s][%!] %v
```

File pattern:

```
[%Y-%m-%d %H:%M:%S.%f][%l][PID:%P][TID:%t][%s:%#][%!] %v
```

Real example (in `logs/proxy/`):

```
[2026-11-16 20:44:28.514631][info][PID:153778][TID:153778][log_stub.cc:68][DlogPrintStub] [HCCL-VM-HCCL_LOG][71][trace_attr.c:120](tid:153778) doesn't set env ASCEND_LOG_DEVICE_FLUSH_TIMEOUT, use default timeout=0ms.
```

Meaning of each part:

| Part | Source | Description |
| --- | --- | --- |
| `2026-11-16 20:44:28.514631` | Recording time | Microsecond precision. Only the file channel has timestamps. |
| `info` | spdlog level name | `trace/debug/info/warning/error/critical` |
| `PID:%P` / `TID:%t` | Process/thread ID | The PID of a device process is the PID of the device itself. |
| `log_stub.cc:68` | `__FILE__` base name + `__LINE__` | Recording point, which is not the original line number of the service code. |
| `DlogPrintStub` | `SPDLOG_FUNCTION` | Recording function. |
| `[HCCL-VM-HCCL_LOG]` | Compile-time `HCCL_VM_MODULE` | Module coloring. It degrades to `[HCCL-VM]` when undefined. |
| `[71]` | Service `moduleId & 0xFF` | Added only by the `DlogPrintStub` bridge of proxy. Only the low 8 bits are retained, and the run/debug channel bits are discarded. |

## 5. cann Mode

### 5.1 Directly Reusing the Genuine CANN Library

The genuine CANN library is responsible for final filtering, synchronization, and file management. Therefore, in host-side cann mode, the simulator **does not create spdlog objects, does not read YAML, and does not write to `<installation root>/logs`**.

**Behavior when the library is missing**: If `dlopen` fails or a symbol is missing, only one line is printed to stderr:

```
[HCCL-VM][ERROR] CANN log library load failed: ...
```

Then the process **does not fall back to stub**, and all subsequent `HCCL_VM_*` logs of the process are silently discarded. When troubleshooting the issue that no log is output after cann is set, first check whether set_env.sh has been sourced and whether LD_LIBRARY_PATH contains lib64 of CANN.

### 5.2 Log Levels

The CANN log levels are fully reused (`ASCEND_GLOBAL_LOG_LEVEL`). For details, refer to the CANN log configuration. The following is for quick reference only:

| CANN Level | Meaning | Macro Threshold | Effect |
| --- | --- | --- | --- |
| 0 | DEBUG | trace | All logs at trace and higher are passed. |
| 1 | INFO | info | trace/debug are discarded. |
| 2 | WARNING | warn | trace/debug/info are discarded. |
| 3 | ERROR | err | Only error/critical remain (default when not set). |
| 4 | NULL | off | All logs are disabled. |

Note that DEBUG(0) of CANN maps to `trace` of spdlog, which is broader than debug. If `ASCEND_GLOBAL_LOG_LEVEL` is not set or is invalid, level 3 (ERROR) applies.

In cann mode, trace and debug are merged into DEBUG, and critical is downgraded to ERROR. Therefore, the level granularity is coarser.

**Module mapping (for simulator developers only)**: At compile time, simulator tags are grouped into existing CANN modules.

| CANN Module | moduleId | Tags Belonging to the Module |
| --- | --- | --- |
| HCCL | 3 | `CDM_STUB` `CDR_STUB` `COMM_OP` `COMM_STUB` `HCOMM_DEV_STUB` `HCOMM_LOCAL` `HCOMM_STUB` `OP_STUB` `OTHER_OP` `SCATTER_STUB` `TOPO` `TOPO_ADDR_STUB` `TOPO_QUERY_STUB` |
| HCCP | 28 | `BR_STUB` `HCCP_STUB` `IBV_DPU_STUB` `NEGOTIATION_STUB` `RASOCKET_STUB` |
| CCU | 5 | `CCU_CTRL_STUB` `CCU_DATA_STUB` `CCU_KERNEL_SIM` `CCU_L1_COMMON` `CCU_L1_RES` `CCU_RES_STUB` `CCU_STUB` `DPU_CTRL_STUB` `DPU_DATA_STUB` `DPU_EXEC` `DPU_KERNEL_MGR_STUB` `DPU_RES` `DPU_RPC_DEV` |
| RUNTIME | 7 | `ACLGRAPH_STUB` `CTX_STUB` `EVENT_STUB` `EXEC_STUB` `GRAPH_CAPTURE` `MEM_STUB` `MODEL_STUB` `NEW_STUB` `NOTIFY_STUB` `RTS_STUB` `STREAM_STUB` `TSD_STUB` |
| ACLRTC | 76 | `ACLRT_STUB` |
| DRV | 10 | `HAL_STUB` |
| KERNEL | 23 | `KERNEL_STUB` |
| PROFILING | 31 | `DFX_STUB` |
| AIVECTOR | 56 | `AIV` |
| OP | 63 | `OPERATION_DATA` |
| SLOG | 0 | No tag (empty tag) and all unmapped tags |

Tags that are currently unmapped and uniformly fall into SLOG(0) (26 in total):

```
CHECKER CMD CONFIG_STUB DEVICE DEVICE_STUB DEVSPEC DEV_L1_COMM DEV_L1_DATA_OP DEV_L1_DFX
DEV_L1_OTHER DLOPEN_STUB DLSYM_STUB DPU_RING EXCEPTION_STUB HCCL_LOG MPI_STUB OTHER_STUB
PROXY_COMMON PROXY_STUB RUNNER RUNTIME_STATE SIMULATOR_STORAGE STAT STORE STUB UTILS
```

This means that in cann mode, to separately adjust the levels of the logs of proxy itself (tag `HCCL_LOG`) or the plugin logs (`CHECKER`/`RUNNER`), you can only use `ASCEND_MODULE_LOG_LEVEL=SLOG=<level>` or the global level control. Writing `HCCL_LOG=0` by name does not take effect. When you add a module, remember to add the tag to `kCannLogModules`.

### 5.3 Device Simulation Processes Started by qemu: Using spdlog to Emulate CANN Text

Device processes **do not dlopen the genuine library**. Instead, a logger with only a file sink is created locally, and the format emulates the CANN device log:

- Directory priority: `ASCEND_PROCESS_LOG_PATH` (used as the log root as is) → `ASCEND_WORK_PATH` + `/log` → `~/ascend/log` (home of the current user). `/run/device-<devKey>/` is uniformly appended to all three. `devKey` is the primary key of the Device table in the startup parameters of the device process.
- File name: `device-<parent pid>_<initialization time in microseconds>.log`, aligned with the naming convention of CANN device logs.
- Level: The file sink is fixed to `trace` (module thresholds have been filtered in macros), and the console is always `off`.
- Capacity: A single file is 20 MB. The maximum number of files is taken from `ASCEND_HOST_LOG_FILE_NUM` (default 10, value range 0-1000; 0 or invalid values are handled as the default).
- Rotation: cann-device does not install the renaming hook of stub. It uses the native spdlog rule `<base>.<n>.log`, so archives can be properly reclaimed.
- Line format:

```
[level] HCCL-VM-<TAG>(<parent pid>,<parent process name>):<Y-m-d-H:M:S>.<3-digit millisecond>.<3-digit microsecond> [<file>:<line>] <body>
```

Example (derived from the formatter):

```
[INFO] HCCL-VM-COMM_STUB(153778,proxy):2026-11-17-10:46:20.123.456 [log_stub.cc:88] batch connect success
```

The module name is taken from the message prefix `[HCCL-VM-<TAG>]`, and the parent process name is read from `/proc/<parent pid>/comm` (note that this is the process name, not the host name).

## 6. When Configuration Takes Effect, Priorities, and Process Boundaries

### 6.1 Taking Effect Only at Startup

Backend selection, YAML parsing, and level thresholds are all completed once during initialization:

- stub: The logger level is always `trace`, and filtering occurs on the two sinks. Modifying the YAML file at runtime does not take effect.
- cann: Modifying `ASCEND_*` at runtime does not take effect.

Any log-related adjustment requires a process restart (changing `ASCEND_*` or `HCCL_VM_LOG_MODE` usually requires restarting hccl-vm).

### 6.2 Priorities

In cann mode, the YAML file is not read. Therefore:

```
Backend = HCCL_VM_LOG_MODE (process-level, the only switch)
stub channel: YAML/default values     cann channel: ASCEND_* environment variables (configuration on the genuine library side is fully delegated to CANN)
```

The two sets of configurations do not fall back to each other. In cann mode, the console/file levels, paths, and rotation parameters in the YAML file are not involved at all. In stub mode, `ASCEND_*` is not involved at all.

### 6.3 Where `export HCCL_VM_LOG_MODE` Is Written Affects the Log Attribution of Plugins

In the `(hvm)$>` terminal, `hccl-vm` is the **client** form of the executable file. It sends command lines to the host process through a POSIX message queue, and the host process executes commands and forks plugins. Therefore:

| export Location | Host Process | checker/runner Plugins | rank/device Service Processes |
| --- | --- | --- | --- |
| Terminal before hccl-vm is started | ✅ | ✅ (forked by the host process, inheriting the host environment) | ✅ (the `(hvm)` shell inherits the host environment) |
| export in the `(hvm)$>` terminal | ❌ Already started | ❌ Forked by the host process | ✅ `mpirun`/`hccl_test` started by this shell and their child processes |

This leads to two phenomena:

- To enable cann for plugin logs as well, you must run export before starting hccl-vm. Running export after entering the terminal takes effect only for service processes.
- After you change `HCCL_VM_LOG_MODE` in `(hvm)$>`, `hccl-vm plugin run @checker` still uses the old backend. This is by design, not because the configuration does not take effect.

### 6.4 The Log Directory of stub Mode Is Cleared at Each Startup

`envInit()` of the `hccl-vm` main process runs `remove_all(<installation root>/logs)` before command parsing, re-creates the directory, and then runs `setenv("HCCL_VM_INSTALL_ROOT", <installation root>)` (if the executable file is in `<installation root>/bin/`, the parent directory is used). Therefore:

- stub logs are brand new in each round and do not accumulate across rounds.
- cann logs are written to the CANN log root (default `~/ascend/log`) and are not affected by this cleanup. Historical files need to be managed by yourself according to CANN conventions.
- When multiple hccl-vm instances are started concurrently and share the same installation directory, the instance started later clears the stub logs of the instance started earlier. To retain logs, copy them first or use cann instead.

## 7. Terminal Flag Logs

After the simulator finishes a command, it prints a line with a fixed tag to the terminal, informing the caller that the command is complete. Such lines do not go through the log system. They are written to stdout directly by the host or a plugin and flushed. Therefore:

- They are independent of the log backend. The settings of `HCCL_VM_LOG_MODE`, `file_level`, and `ASCEND_GLOBAL_LOG_LEVEL` do not affect them, and they are visible even when all logs are disabled.
- They are not written to log files. These lines do not exist in `<installation root>/logs` or CANN logs. They appear only on the terminal (they follow the redirection when the stdout of hccl-vm is redirected).
- The naming rule is `[HCCL_VM_<action>_FINISHED]`. When information needs to be carried, `key=value` is appended to the same line.
- They are printed basically only on the success path. The completion flag of checker is an exception (it is printed on any exit path).

### 7.1 Overview

| Command | Flag Line | When Printed | Remarks |
| --- | --- | --- | --- |
| `hccl-vm start <model>` | `[HCCL_VM_START_FINISHED]` | When the rc script of the subshell is executed | Pure tag, indicating that the `(hvm)$>` interactive terminal has been entered. |
| `hccl-vm mode <m>` | `[HCCL_VM_MODE_FINISHED] mode=<normal\|check-only\|estimate>` | When the mode switching action is complete | Printed even if the target mode is the same as the current one; not printed on the failure path. |
| `hccl-vm mock-comm <cfg>` | `[HCCL_VM_MOCK_COMM_FINISHED] config=<cfg> level=<1\|2>` | After the communicator is set up and the number of ranks in this round is registered | Round monitoring flags can appear only after this flag. |
| `hccl-vm plugin install @x` | `[HCCL_VM_PLUGIN_INSTALL_FINISHED]` | When the plugin is installed successfully | Pure tag. `uninstall` has no corresponding flag. |
| `hccl-vm plugin run @checker` | `[HCCL_VM_PLUGIN_RUN_FINISHED] plugin=@checker` | When checker actually finishes | The command itself returns immediately, and the flag arrives asynchronously. It is printed on any exit path. |
| `hccl-vm plugin run @estimate` | `[HCCL_VM_PLUGIN_RUN_FINISHED] plugin=@estimate` | After the performance simulation ends and `system()` returns | Before this, two lines `[HVM] @estimate ...` (start/finish) are printed. |
| All service processes of a round exit | `[HCCL_VM_ROUND_PROCESSES_EXITED] workload=<N>` | When the host polling thread collects N exit slots | The line starts with `\r`. Only progress lines are printed before the count is reached. |

### 7.2 Points Requiring Attention

- **The flag of `start` comes from the subshell**: If the model file is missing or topology generation fails, `hccl-vm start` directly reports an error and returns without printing this line.
- **The completion flag of checker is asynchronous**: `plugin run @checker` delivers the message to the checker process and returns immediately. checker runs in its own worker thread. The flag is printed by checker, and `FlushLog()` is called before printing. Therefore, you can read the on-disk logs once you see the flag. The flag only indicates that the round is complete, not that the verification has passed. For the conclusion, check result lines such as `[CHECKER_RUN_SUMMARY]` in the logs.
- **`@estimate` is synchronous**: The host prints the completion flag only after the simulation command returns. Before that, it prints `[HVM] @estimate performance simulation started (results under <dir>/data/<timestamp>/, ...)`, and after the simulation ends, it prints `[HVM] @estimate performance simulation finished`.
- **Not all commands have flags**: Commands such as `reset`, `table`, `model`, `device`, `plugin uninstall`, and `plugin list` have no completion flags. Do not wait for them in scripts.
- **Round monitoring is asynchronous**: After `mock-comm` loads the exit slots based on the number of ranks in this round, the host polls the exit slots of service processes every 1 second. When enough exits are collected, the host first resets the slot files and then prints `[HCCL_VM_ROUND_PROCESSES_EXITED] workload=N`. Before the count is reached, the host prints a line `[HCCL_VM_ROUND_PROGRESS] seen=<x>/<N>` every 5 seconds. **This line does not indicate completion.** If processes are missing, the framework should use a timeout as the fallback.
