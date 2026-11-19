# 模拟器日志使用指导（stub / cann 双后端）

本文档面向使用和排查 HCCL-VM 日志的使用者，覆盖两种日志后端（`stub`、`cann`）的配置方式、落盘位置、等级语义、输出格式，以及每条命令结束后打到终端的标志日志。

## 目录

- [1. 主要特性](#1-主要特性)
- [2. 快速上手](#2-快速上手)
- [3. 各进程的日志落盘位置](#3-各进程的日志落盘位置)
- [4. stub 模式](#4-stub-模式)
  - [4.1 配置来源](#41-配置来源)
  - [4.2 配置文件字段表](#42-配置文件字段表)
  - [4.3 落盘位置与命名](#43-落盘位置与命名)
  - [4.4 输出格式](#44-输出格式)
- [5. cann 模式](#5-cann-模式)
  - [5.1 直接复用 CANN 真库](#51-直接复用-cann-真库)
  - [5.2 日志等级](#52-日志等级)
  - [5.3 qemu 启动的 device 模拟进程：用 spdlog 仿真 CANN 文本](#53-qemu-启动的-device-模拟进程用-spdlog-仿真-cann-文本)
- [6. 生效时机、优先级与进程边界](#6-生效时机优先级与进程边界)
  - [6.1 只在启动时生效](#61-只在启动时生效)
  - [6.2 优先级](#62-优先级)
  - [6.3 export HCCL_VM_LOG_MODE 写在哪里，影响插件日志归属](#63-export-hccl_vm_log_mode-写在哪里影响插件日志归属)
  - [6.4 每次启动会清空 stub 模式日志目录](#64-每次启动会清空-stub-模式日志目录)
- [7. 终端打屏标志日志](#7-终端打屏标志日志)
  - [7.1 一览](#71-一览)
  - [7.2 需要留意的点](#72-需要留意的点)

## 1. 主要特性

模拟器日志拥有双后端机制，所有模拟器日志按进程启动时确定的后端二选一：

- **stub（默认）**：模拟器自带 spdlog 双通道（控制台 + 文件），等级、路径、轮转全部由 `config/log_config.yaml` 控制，落 `<安装根>/logs/`。
- **cann**：复用 CANN 日志能力。主机侧进程（hccl-vm 主进程、checker、runner、rank 业务进程）直接调用 CANN 真库 `libunified_dlog.so`；device 进程不加载真库，改用 spdlog 仿真 CANN 的文本格式，落 CANN 日志根下的 `run/device-<设备主键>/`。

## 2. 快速上手

```bash
# 默认（stub）：什么都不用设
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# cann（推荐在启动 hccl-vm 之前 export）
source /path/to/Ascend/cann/set_env.sh
export HCCL_VM_LOG_MODE=cann
export ASCEND_GLOBAL_LOG_LEVEL=1                     # 0-DEBUG 1-INFO 2-WARNING 3-ERROR 4-NULL
./hccl-vm start ascend950_cluster_32_server_normal.yaml
```

只希望业务进程（rank / device）走 cann、主机与插件保持 stub 时，在 `(hvm)$>` 终端内 export，再启动 `mpirun`（原因见 6.3 节的[进程边界表](#63-export-hccl_vm_log_mode-写在哪里影响插件日志归属)）。

## 3. 各进程的日志落盘位置

| 进程 | stub 落盘 | cann 落盘 |
| --- | --- | --- |
| hccl-vm 主进程（命令解释器、交互 shell 的父进程） | `logs/hccl_vm_<pid>.log` | 由 CANN 真库决定（跟随其 `ASCEND_*` 配置） |
| checker 插件 | `logs/checker/checker_<pid>.log` | 同上 |
| runner 插件 | `logs/runner/runner_<pid>.log` | 同上 |
| rank 业务进程（LD_PRELOAD 代理库） | `logs/proxy/proxy_<pid>.log` | 同上 |
| device 进程 | `logs/proxy/proxy_<父pid>_<本pid>.log` | `<CANN 日志根>/run/device-<devKey>/device-<父pid>_<初始化微秒>.log`；日志根默认 `~/ascend/log`，可用 `ASCEND_PROCESS_LOG_PATH`（其次 `ASCEND_WORK_PATH/log`）改写 |

上表中的相对路径以安装根（`HCCL_VM_INSTALL_ROOT`，即 `bin/` 的上一级）为基准；device 的 cann 路径由本模拟器自行拼接，不经过 CANN 真库。

## 4. stub 模式

### 4.1 配置来源

 `<安装根>/config/log_config.yaml`配置模拟器日志等级。YAML 按“进程名”取节点：

| 进程名（代码传入） | YAML 节点 | 说明 |
| --- | --- | --- |
| `hccl_vm` | `hccl_vm` | hccl-vm 主进程 |
| `proxy` | `proxy` | rank 业务进程 |
| `device_aarch64` | `proxy` | 复用 proxy 段；文件名基名改为 `proxy_<父pid>` |
| `runner` | `runner` | runner 插件 |
| `checker` | `checker` | checker 插件 |

### 4.2 配置文件字段表

| 字段 | 代码默认值 | 是否生效 | 说明 |
| --- | --- | --- | --- |
| `console_level` | `2`（info） | 生效 | 控制台打屏日志等级，0-6，越界钳到 0/6 |
| `file_level` | `1`（debug） | 生效 | 日志文件落盘等级，0-6 |
| `max_file_size` | `52428800`（50MB） | 生效 | 单文件字节数 |
| `max_files` | `65535` | 生效 | 最大归档文件数量 |
| `file_path` | `logs` | 生效 | 相对路径按 `<安装根>` 拼接；以 `/` 或 `./` 开头时原样使用（`./` 相对**进程当前目录**） |
| `file_suffix` | `.log` | 生效 | 轮转重命名依赖后缀匹配，自定义后缀需保持一致 |
| `file_base_name` | — | **暂不生效** | 文件当前基名固定为进程名（device 为 `proxy_<父pid>`） |
| `enable_compress` | `false` | **暂不生效** | 当前暂不支持日志压缩 |

级别数值统一为 spdlog 语义：`0-trace 1-debug 2-info 3-warn 4-error 5-critical 6-off`。

### 4.3 落盘位置与命名

`<file_path>/<file_base_name>_<pid><file_suffix>`，其中基名由进程名决定。示例：

```
<安装根>/logs/hccl_vm_153313.log
<安装根>/logs/checker/checker_153364.log
<安装根>/logs/proxy/proxy_153778.log                    # rank 业务进程
<安装根>/logs/proxy/proxy_153778_153777.log             # device 进程（父 pid_本 pid）
```

文件所在目录不存在时由 spdlog 自动创建。

### 4.4 输出格式

控制台 pattern：

```
[%l][PID:%P][TID:%t][%s][%!] %v
```

文件 pattern：

```
[%Y-%m-%d %H:%M:%S.%f][%l][PID:%P][TID:%t][%s:%#][%!] %v
```

真实样例（`logs/proxy/`）：

```
[2026-11-16 20:44:28.514631][info][PID:153778][TID:153778][log_stub.cc:68][DlogPrintStub] [HCCL-VM-HCCL_LOG][71][trace_attr.c:120](tid:153778) doesn't set env ASCEND_LOG_DEVICE_FLUSH_TIMEOUT, use default timeout=0ms.
```

逐段含义：

| 片段 | 来源 | 说明 |
| --- | --- | --- |
| `2026-11-16 20:44:28.514631` | 记录时间 | 微秒精度，仅文件通道有时间戳 |
| `info` | spdlog 级别名 | `trace/debug/info/warning/error/critical` |
| `PID:%P` / `TID:%t` | 进程/线程号 | device 进程的 PID 是 device 自己的 |
| `log_stub.cc:68` | `__FILE__` 基名 + `__LINE__` | 记录点，不是业务原始行号 |
| `DlogPrintStub` | `SPDLOG_FUNCTION` | 记录函数 |
| `[HCCL-VM-HCCL_LOG]` | 编译期 `HCCL_VM_MODULE` | 模块染色，无定义时退化为 `[HCCL-VM]` |
| `[71]` | 业务 `moduleId & 0xFF` | 仅 proxy 的 `DlogPrintStub` 桥接会加，且只保留低 8 位，run/debug 通道位被丢弃 |

## 5. cann 模式

### 5.1 直接复用 CANN 真库

CANN真库负责最终过滤、同步和文件管理，所以主机侧 cann 模式**不创建 spdlog 对象、不读 YAML、不写 `<安装根>/logs`**。

**缺库行为**：`dlopen` 失败或符号缺失时只向 stderr 打一条

```
[HCCL-VM][ERROR] CANN log library load failed: ...
```

然后**不回退 stub**，该进程后续所有 `HCCL_VM_*` 日志静默丢弃。排查“设了 cann 却一条日志都没有”时先确认 `set_env.sh` 是否 source、`LD_LIBRARY_PATH` 是否含 CANN 的 `lib64`。

### 5.2 日志等级

CANN 日志等级完全复用（`ASCEND_GLOBAL_LOG_LEVEL` ），请参考cann日志配置，以下仅供快速参考；

| CANN 等级 | 含义 | 宏阈值 | 效果 |
| --- | --- | --- | --- |
| 0 | DEBUG | trace | trace 及以上全放行 |
| 1 | INFO | info | 丢 trace/debug |
| 2 | WARNING | warn | 丢 trace/debug/info |
| 3 | ERROR | err | 只剩 error/critical（未设置时的默认） |
| 4 | NULL | off | 全关 |

注意 CANN 的 DEBUG(0) 映射到 spdlog 的 `trace`，比“debug”更宽；`ASCEND_GLOBAL_LOG_LEVEL` 未设置或非法时按 3（ERROR）处理。

cann 模式下 trace/debug 合流为 DEBUG、critical 降为 ERROR，级别粒度是粗化的。

**模块映射(仅供模拟器开发人员参考)**：编译期把模拟器 tag 归到已有 CANN 模块。

| CANN 模块 | moduleId | 归属 tag |
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
| SLOG | 0 | 无 tag（空 tag）以及所有未映射 tag |

当前未被映射、统一落到 SLOG(0) 的 tag（共 26 个）：

```
CHECKER CMD CONFIG_STUB DEVICE DEVICE_STUB DEVSPEC DEV_L1_COMM DEV_L1_DATA_OP DEV_L1_DFX
DEV_L1_OTHER DLOPEN_STUB DLSYM_STUB DPU_RING EXCEPTION_STUB HCCL_LOG MPI_STUB OTHER_STUB
PROXY_COMMON PROXY_STUB RUNNER RUNTIME_STATE SIMULATOR_STORAGE STAT STORE STUB UTILS
```

这意味着：想在 cann 模式下单独调 proxy 自身日志（tag `HCCL_LOG`）或插件日志（`CHECKER`/`RUNNER`）的等级，只能通过 `ASCEND_MODULE_LOG_LEVEL=SLOG=<级别>` 或全局等级控制；按名字写 `HCCL_LOG=0` 不会命中。新增模块时记得把 tag 加进 `kCannLogModules`。

### 5.3 qemu 启动的 device 模拟进程：用 spdlog 仿真 CANN 文本

device 进程**不 dlopen 真库**，而是在本地建一个只有文件 sink 的 logger，把格式仿真成 CANN 设备日志：

- 目录优先级：`ASCEND_PROCESS_LOG_PATH`（原样作为日志根）→ `ASCEND_WORK_PATH` + `/log` → `~/ascend/log`（取当前用户 home）；三者之后统一追加 `/run/device-<devKey>/`，`devKey` 是 device 进程启动参数里的 Device 表主键。
- 文件名：`device-<父pid>_<初始化时刻微秒>.log`，与 CANN 设备日志命名习惯对齐。
- 级别：文件 sink 固定 `trace`（模块阈值已在宏里过滤），控制台恒 `off`。
- 容量：单文件 20MB；文件数上限取 `ASCEND_HOST_LOG_FILE_NUM`（默认 10，取值 0-1000，0 或非法按默认处理）。
- 轮转：cann-device 不安装 stub 的重命名钩子，用 spdlog 原生规则 `<base>.<n>.log`，归档能被正常回收。
- 行格式：

```
[级别] HCCL-VM-<TAG>(<父pid>,<父进程名>):<Y-m-d-H:M:S>.<毫秒3位>.<微秒3位> [<文件>:<行>] <正文>
```

示例（按格式器推导）：

```
[INFO] HCCL-VM-COMM_STUB(153778,proxy):2026-11-17-10:46:20.123.456 [log_stub.cc:88] batch connect success
```

其中模块名取自消息前缀 `[HCCL-VM-<TAG>]`，父进程名读 `/proc/<父pid>/comm`（注意是进程名，不是主机名）。

## 6. 生效时机、优先级与进程边界

### 6.1 只在启动时生效

后端选择、YAML 解析、等级阈值都在初始化时一次性完成：

- stub：logger 级别恒为 `trace`，过滤发生在两个 sink 上，运行期修改 YAML 不会生效。
- cann：运行期修改 `ASCEND_*` 不生效。

任何日志相关的调整都需要重启进程（改 `ASCEND_*` 或 `HCCL_VM_LOG_MODE` 时通常要重启 hccl-vm）。

### 6.2 优先级

cann 模式下并不读 YAML，所以：

```
后端 = HCCL_VM_LOG_MODE（进程级，唯一开关）
stub 通道：YAML/默认值     cann 通道：ASCEND_* 环境变量（真库侧配置完全交给 CANN）
```

两套配置互不回退：cann 下 YAML 里的 console/file 级别、路径、轮转参数一概不参与；stub 下 `ASCEND_*` 一概不参与。

### 6.3 export HCCL_VM_LOG_MODE 写在哪里，影响插件日志归属

`(hvm)$>` 终端里的 `hccl-vm` 是可执行文件的**客户端**形态：它把命令行通过 POSIX 消息队列发给主机进程，由主机进程执行命令、fork 插件。所以：

| export 位置 | 主机进程 | checker/runner 插件 | rank/device 业务进程 |
| --- | --- | --- | --- |
| 启动 hccl-vm 之前的终端 | ✅ | ✅（由主机进程 fork，继承主机环境） | ✅（`(hvm)` shell 继承主机环境） |
| `(hvm)$>` 终端内 export | ❌ 已启动 | ❌ 由主机进程 fork | ✅ 该 shell 拉起的 `mpirun`/`hccl_test` 及其子进程 |

由此得到两个现象：

- 想让插件日志也走 cann，必须在启动 hccl-vm 前 export；进入终端后再 export 只对业务进程有效。
- `(hvm)$>` 里改完 `HCCL_VM_LOG_MODE` 后 `hccl-vm plugin run @checker` 仍是旧后端，这是设计使然，不是配置没生效。

### 6.4 每次启动会清空 stub 模式日志目录

`hccl-vm` 主进程的 `envInit()` 在命令解析前执行 `remove_all(<安装根>/logs)` 并重建，然后 `setenv("HCCL_VM_INSTALL_ROOT", <安装根>)`（可执行文件位于 `<安装根>/bin/` 时取上一级）。因此：

- stub 日志每轮都是全新的，不会跨轮次累积；
- cann 日志落在 CANN 日志根（默认 `~/ascend/log`），不受这次清理影响，历史文件需要按 CANN 惯例自行管理；
- 并发启动多个 hccl-vm 实例且共用同一安装目录时，后启动的一方会清掉先启动一方的 stub 日志。需要留存时先拷走，或改用 cann。

## 7. 终端打屏标志日志

模拟器执行完一条命令后，会向终端打一行固定 tag，告诉调用方“这条命令走完了”。这类行不经过日志系统，由 host 或插件直接写 stdout 并 flush，因此：

- 与日志后端无关：`HCCL_VM_LOG_MODE`、`file_level`、`ASCEND_GLOBAL_LOG_LEVEL` 怎么设都不影响它，日志全关也能看到；
- 不进日志文件：`<安装根>/logs` 和 CANN 日志里都没有这些行，只出现在终端（hccl-vm 的 stdout 被重定向时跟随重定向）；
- 命名规则是 `[HCCL_VM_<动作>_FINISHED]`，需要携带信息时在同一行追加 `key=value`；
- 基本只在成功路径打印，checker 的完成标志是例外（任何退出路径都打）。

### 7.1 一览

| 命令 | 标志行 | 打印时机 | 备注 |
| --- | --- | --- | --- |
| `hccl-vm start <model>` | `[HCCL_VM_START_FINISHED]` | 子 shell 的 rc 脚本执行时 | 纯 tag；表示已进入 `(hvm)$>` 交互终端 |
| `hccl-vm mode <m>` | `[HCCL_VM_MODE_FINISHED] mode=<normal\|check-only\|estimate>` | 模式切换动作结束 | 目标模式与当前一致也打；失败路径不打 |
| `hccl-vm mock-comm <cfg>` | `[HCCL_VM_MOCK_COMM_FINISHED] config=<cfg> level=<1\|2>` | 通信域建好、本轮 rank 数登记后 | 这条之后才可能出现轮次监控标志 |
| `hccl-vm plugin install @x` | `[HCCL_VM_PLUGIN_INSTALL_FINISHED]` | 插件安装成功 | 纯 tag；`uninstall` 没有对应标志 |
| `hccl-vm plugin run @checker` | `[HCCL_VM_PLUGIN_RUN_FINISHED] plugin=@checker` | checker 真正跑完时 | 命令本身立即返回，标志异步到达；任何退出路径都打 |
| `hccl-vm plugin run @estimate` | `[HCCL_VM_PLUGIN_RUN_FINISHED] plugin=@estimate` | 性能仿真结束、`system()` 返回后 | 此前还有 start/finish 两行 `[HVM] @estimate ...` |
| 每轮业务进程全部退出 | `[HCCL_VM_ROUND_PROCESSES_EXITED] workload=<N>` | host 轮询线程数够 N 个退出槽位 | 行首带 `\r`；未凑齐时只打进度行 |

### 7.2 需要留意的点

- **`start` 的标志来自子 shell**：模型文件缺失、拓扑生成失败等情况下 `hccl-vm start` 直接报错返回，不会打这一行。
- **checker 的完成标志是异步的**：`plugin run @checker` 把消息投递给 checker 进程后立即返回，checker 在自己的 worker 线程里跑；标志由 checker 打印，打印前先 `FlushLog()`，所以看到标志即可去读落盘日志。标志只表示本轮结束，不表示校验通过，结论看日志里的 `[CHECKER_RUN_SUMMARY]` 等结果行。
- **`@estimate` 是同步的**：host 等仿真命令返回后才打完成标志；此前会先打 `[HVM] @estimate performance simulation started (results under <dir>/data/<timestamp>/, ...)`，结束后打 `[HVM] @estimate performance simulation finished`。
- **并非所有命令都有标志**：`reset`、`table`、`model`、`device`、`plugin uninstall`、`plugin list` 等命令没有完成标志，脚本不要等它们。
- **轮次监控是异步的**：`mock-comm` 按本轮 rank 数“上膛”后，host 每 1 秒轮询业务进程的退出槽位；数够了先复位槽位文件，再打 `[HCCL_VM_ROUND_PROCESSES_EXITED] workload=N`。没凑齐时每 5 秒打一行 `[HCCL_VM_ROUND_PROGRESS] seen=<x>/<N>`，**它不代表结束**，缺进程时框架应以超时兜底。
