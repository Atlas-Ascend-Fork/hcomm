# HCCL-VM User Guide

## 1. Overview

HCCL-VM is a high-performance collective communication virtual execution environment for Huawei Ascend NPU cards. This tool enables the development and functional verification of HCCL collective communication operators without actual Ascend hardware.

The tool provides three operating modes: `normal` (actual multi-card communication with full result correctness), `check-only` (verification only, large memory blocks reuse a shared pool to significantly reduce memory usage), and `estimate` (estimation, skips actual communication blocking points and quickly produces a task list). You can specify the mode at startup or switch it at runtime using the `hccl-vm mode` command. For details, see [Operating Modes](#49-operating-modes-normal--check-only--estimate).

The tool also provides two simulation levels: the default level 2 (southbound hijacking) and level 1 (northbound hijacking, specified with `--level 1`). For details, see [Simulation Levels and Level 1 Hijacking](#410-simulation-levels-and-level-1-hijacking-northbound-hijacking).

![hccl-vm GIF](docs/hccl-vm.gif)

## 2. Prerequisites

| Dependency | Version Requirement |
| -------- | ------------- |
| System Architecture | x86_64 Ubuntu 22.04 or later |
| Hardware Constraints | Ascend950; for other constraints, see [Specification Constraints](#45-tool-specification-constraints) |
| QEMU (x86_64 host machine) | Use the system-provided or apt-installed qemu-user-static (`qemu-aarch64-static` in PATH) as the first choice. When the version is below 10.2.4, the translation code cache cannot be limited, QEMU resident memory is large, and large-scale scenarios may cause paging and significantly affect performance. To compile version 10.2.4 or later from source, see [3.1.1 QEMU Source Compilation (Optional)](#311-qemu-source-compilation-optional-x86_64-host-machine). Native ARM host machines do not require QEMU. Compilation (`build.sh`/`build_pkg.sh`) does not depend on QEMU installed on the local machine. |
| Python3 (required only by the estimate mode performance simulation plugin) | >=3.8 (including the sqlite3 standard library). The simpy discrete event simulation package must also be pre-installed (`pip3 install simpy`). For details, see [4.9.5](#495-estimation-plugin-estimate-and-dependency-description). |

### 2.1 CANN Package Installation

Install the latest CANN Toolkit development kit and CANN ops operator package. [Download Link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)

```bash
# 确保安装包具有可执行权限
chmod +x Ascend-cann-toolkit_9.1.0_linux-x86_64.run
chmod +x Ascend-cann-950-ops_9.1.0_linux-x86_64.run
# 安装命令
./Ascend-cann-toolkit_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
./Ascend-cann-950-ops_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
```


### 2.2 hccl_test Compilation

hccl_test is the official Ascend HCCL performance testing tool. For details, see [HCCL Performance Testing Tool](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/devaids/hccltool/HCCLpertest_16_0001.html). HCCL-VM supports running hccl_test test cases in the virtual environment. First, follow the [hccl_test Test Case Build](#42-hccl-test-test-case-build) section to compile the test case binaries.

Note: hccl_test is an optional dependency. To run PyTorch test cases, see [PyTorch Test Case Samples](#33-pytorch-test-case-samples).

### 2.3 AICPU Runtime Environment Installation

HCCL-VM loads and simulates the execution of AICPU device-side symbols at runtime. An x86_64 host machine requires QEMU user-mode emulation (linux-user emulating aarch64):

- **Runtime**: Use the system-provided or apt-installed QEMU (`qemu-user-static` provides `qemu-aarch64-static`) as the first choice. You can also specify a local binary through `HCCL_VM_QEMU_PATH`.
- **Deployment**: `build.sh --full` and `build.sh --level1` automatically invoke `build_pkg.sh` after build and installation to extract `aicpu_hccl.tar.gz` and `aicpu_hcomm.tar.gz` from the CANN installation directory and deploy AICPU device-side symbols. No separate execution is required. If deployment fails, the build returns a failure. For compilation and installation steps, see [3.1 Manual Build and Installation](#31-manual-build-and-installation).

```bash
sudo apt install -y qemu-user-static binfmt-support
```

When the system-provided version is below **10.2.4**, the `-tb-size` option to limit the translation code cache is not supported. QEMU resident memory is large, and large-scale scenarios may significantly affect performance. To compile version 10.2.4 or later from source, see [3.1.1 QEMU Source Compilation (Optional)](#311-qemu-source-compilation-optional-x86_64-host-machine). Native ARM host machines do not require QEMU.

Runtime behavior and prompts:

- When a device process starts, the tool prints the actual QEMU binary path, version, and the requested `-tb-size` value.
- When `HCCL_VM_QEMU_PATH` is not set, is invalid, or points to an older QEMU version that does not support `-tb-size`, the tool **does not abort**: it automatically falls back (searches PATH for `qemu-aarch64-static` / omits `-tb-size`) and prints a WARN message that includes the actual fallback binary and version, along with a prompt that the translation cache is not limited and QEMU memory may be excessive. The AICPU runtime environment fails to start only when no `qemu-aarch64-static` binary is found on the entire machine.
- `HCCL_VM_QEMU_TB_SIZE` specifies the QEMU translation code cache size in MiB. When not set, the default is `16`. A value of `0` delegates the decision to the default QEMU policy (it does not disable the cache). An unlimited cache may significantly increase QEMU resident memory and cause paging.

---

## 3. Quick Start

### One-Click Installation

After pulling the complete CheckerL2 source code, run the following command from the repository root directory:

```bash
bash ./hccl_vm_installer
```

### 3.1 Manual Build and Installation

```bash
# 1. 创建工作目录
mkdir -p /home/workspace
cd /home/workspace

# 2. 下载依赖源码
git clone https://gitcode.com/cann/hccl.git
git clone https://gitcode.com/cann/hcomm.git

# 3. 安装第三方依赖
sudo apt-get update
sudo apt install build-essential cmake libboost-all-dev rdma-core libibverbs-dev pkg-config gcc-aarch64-linux-gnu g++-aarch64-linux-gnu qemu-user-static binfmt-support

# 4. 编译安装 HCCL-VM 工具并自动部署 AICPU device 侧符号
# 下载 hcomm 代码之后，工具源码所在路径：/home/workspace/hcomm/test/hccl_vm
cd /home/workspace/hcomm/test/hccl_vm
source /home/workspace/Ascend/cann/set_env.sh
export HCCL_CODE_HOME=/home/workspace/hccl
export HCOMM_CODE_HOME=/home/workspace/hcomm
bash ./build.sh --full
```

`--full` already includes the device-side build. You do not need to run `build.sh --aicpu` or `build_pkg.sh` separately. Automatic deployment uses only the AICPU packages already present in the current CANN installation and does not recompile HCCL/HCOMM. When `--pkg` is appended, the tool generates `hccl_vm_install.tar.gz` containing the AICPU libraries after deployment completes. A parameterless build still updates only the host side, and `--aicpu` still updates only the device side. To refresh the AICPU libraries in an existing CANN installation separately, run `bash build_pkg.sh`.

#### 3.1.1 QEMU Source Compilation (Optional, x86_64 Host Machine)

The tool uses the system-provided or apt-installed QEMU by default (see [2.3](#23-aicpu-runtime-environment-installation)). If you need version 10.2.4 or later to limit the translation code cache and reduce memory usage in large-scale scenarios, compile QEMU from source using the following steps:

```bash
# 1) 构建依赖（gcc/make/ninja 一般已随系统或本工具安装存在）
sudo apt-get install -y meson flex bison libglib2.0-dev libpcre2-dev libffi-dev

# 2) 官方源码归档（示例为 10.2.4，10.2.4 及以上版本均可）
curl -fsSL -O https://download.qemu.org/qemu-10.2.4.tar.xz
tar -xf qemu-10.2.4.tar.xz && cd qemu-10.2.4

# 3) 构建 aarch64 linux-user 静态版本
mkdir build && cd build
../configure --target-list=aarch64-linux-user --static --disable-system --disable-tools \
    --disable-docs --disable-guest-agent --disable-slirp
make -j$(nproc) qemu-aarch64

# 4) 部署并配置（部署路径可自定义，示例为 /opt/qemu-10.2.4）
sudo mkdir -p /opt/qemu-10.2.4/bin
sudo cp qemu-aarch64 /opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_TB_SIZE=16
```

### 3.2 Usage Samples

#### 3.2.1 Environment Configuration

If an `hccl_rootinfo.json` file already exists under the `/etc` path, delete it manually.

When the communicator scale exceeds 128 chips (129-1024), switch to a MariaDB database. See [MariaDB Database Usage](src/storage_backends/mariadb/README.md). After completing the deployment and setting `HCCL_VM_STORAGE_CONFIG`, return to this section. Running `unset` on that variable restores the default SQLite database.
<!-- internal-only:end -->

#### 3.2.2 CCU Mode

1. Configure the environment variables.

```bash
# 进入工具安装目录
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
```

2. Execute


```bash
# 需要进入到新的bin文件目录下执行hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# 选择昇腾集群拓扑配置文件，启动工具，初始化集群环境，进入工具命令行
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# 如需启用runner插件（可选）
(hvm)$> hccl-vm plugin install @runner

# 选择本次算子执行的通信域配置文件（在1个超节点1个Server1个NPU的集群环境运行hccl_test用例）
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# 执行checker校验
(hvm)$> hccl-vm plugin run @checker

# 退出工具终端
(hvm)$> exit
```

3. Verify the hccl_test test case results.
[Runner Results](#481-runner-plugin-results) 
[Checker Results](#482-checker-plugin-results)

#### 3.2.3 AICPU Mode

The AICPU expansion mode requires the algorithm expansion steps to run on the device side. Therefore, the hccl-vm tool must compile and simulate the execution of HCCL device-side symbols. Running on an x86_64 host machine requires QEMU to emulate device-side code execution. For installation and deployment, see [2.3 AICPU Runtime Environment Installation](#23-aicpu-runtime-environment-installation).

On an x86 host machine, device-side code runs through QEMU user-mode emulation. When `HCCL_VM_QEMU_PATH` is not configured, the tool automatically falls back to `qemu-aarch64-static` in PATH and prints a warning (including the actual version), which does not affect startup. When the system-provided version is below 10.2.4, the translation code cache cannot be limited and memory usage is high. Large-scale scenarios may cause paging and affect performance. Refer to [2.3 AICPU Runtime Environment Installation](#23-aicpu-runtime-environment-installation) to install version 10.2.4 or later and configure `HCCL_VM_QEMU_PATH`. Native ARM host machines do not require QEMU.

1. Configure the environment variables.

```bash
# 进入工具安装目录
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AI_CPU"

# x86_64 宿主：QEMU 程序与翻译代码缓存（MiB）配置，见 §2.3
export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_TB_SIZE=16
```

2. Execute

```bash
# 需要进入到新的bin文件目录下执行hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# 选择昇腾集群拓扑配置文件，启动工具，初始化集群环境，进入工具命令行
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# 如需启用runner插件（可选）
(hvm)$> hccl-vm plugin install @runner

# 选择本次算子执行的通信域配置文件（在1个超节点1个Server1个NPU的集群环境运行hccl_test用例）
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# 执行checker校验
(hvm)$> hccl-vm plugin run @checker

# 退出工具终端
(hvm)$> exit
```

3. Verify the hccl_test test case results. [Runner Results](#481-runner-plugin-results) [Checker Results](#482-checker-plugin-results)

#### 3.2.4 HostDPU Mode

The environment variable configuration and other runtime settings for the HostDPU expansion mode are the same as those for the AICPU mode. The only differences are in cluster scale and communicator scale:

1. Use `./hccl-vm start ascend950_cluster_32_server_normal_hostdpu.yaml` for the cluster specification configuration.
2. The number of servers must be >= 2. Single-server communicators are not supported (configurations such as 112.yaml, 114.yaml, and 118.yaml are not supported).

#### 3.2.5 AIV Mode

1. Configure the environment variables.

```bash
# 进入工具安装目录
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AIV"
```

2. Execute

```bash
# 需要进入到新的bin文件目录下执行hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# 选择昇腾集群拓扑配置文件，启动工具，初始化集群环境，进入工具命令行
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# 如需启用runner插件（可选）
(hvm)$> hccl-vm plugin install @runner

# 选择本次算子执行的通信域配置文件（在1个超节点1个Server1个NPU的集群环境运行hccl_test用例）
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# 执行checker校验
(hvm)$> hccl-vm plugin run @checker

# 退出工具终端
(hvm)$> exit
```

3. Verify the hccl_test test case results. [Runner Results](#481-runner-plugin-results) [Checker Results](#482-checker-plugin-results)

### 3.3 PyTorch Test Case Samples

HCCL-VM supports running a subset of PyTorch `aclgraph` graph-mode collective communication scripts. The scripts should use `torch.multiprocessing.spawn` to start each device process. Multi-server test cases require multiple Python main processes to launch concurrently on the same host machine, each simulating a separate server. For the complete description, see [PyTorch Mode Running Guide](docs/pytorch_guide.md).

#### 3.3.1 Environment Preparation

First complete the HCCL-VM build and installation and CANN environment configuration, and prepare a Python virtual environment with compatible versions of `torch` and `torch_npu` installed. PyTorch currently supports only the Ascend950 series. To use the Ascend960 series, install the A6 ops package and change `soc_version` in the topology cluster configuration to an actual model such as `Ascend960DT_961BY`.

Note: The simulator currently supports a limited number of AIV operators. A significant portion of torch computations cannot complete normally. For example, `print(tensor)` and `tensor.clone()` both cause runtime exceptions because they invoke unsupported computation operators.

```bash
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin
./hccl-vm start ascend950_cluster_32_server_normal.yaml
```

After entering the `(hvm)$>` terminal, run the following commands. Complete all subsequent communicator configuration, environment variable setup, and Python startup within this terminal so that the Python processes inherit the simulation environment.

```bash
# 启用 PyTorch 适配
export HCCL_VM_ENABLE_PYTORCH=1
# 激活虚拟环境（如有）
source /path/to/torch_test/bin/activate

# 需要模拟执行时安装 Runner
hccl-vm plugin install @runner
```

#### 3.3.2 Single-Server Running

The `118` configuration is used as an example. This configuration uses 1 server and 8 devices to execute an AllReduce SUM on 1024 int32 elements. The following command parameters correspond to the test script in the guide. When using a different script, adjust the parameters according to its parameter definitions.

```bash
hccl-vm mock-comm 118

python -u /path/to/torch/pytorch_test.py \
    --master_ip 127.0.0.1 \
    --master_port 50001 \
    --dev_num 8 \
    --server_num 1 \
    --server_index 0 \
    --data_list 1024 \
    --dtype int32 \
    --optype sum > server0.log 2>&1
```

#### 3.3.3 Multi-Server Running

The `128` configuration is used as an example. This configuration uses 2 servers, each with 8 devices, for a total of 16 ranks. Run the following commands in the HCCL-VM terminal where the environment has already been prepared:

```bash
hccl-vm mock-comm 128

# 并发启动两个 server，分别设置环境变量并保存日志
for server_index in 0 1; do
    (
        export HCCL_VM_SERVER_INDEX=${server_index}
        python -u /path/to/torch/pytorch_test.py \
            --master_ip 127.0.0.1 \
            --master_port 50001 \
            --dev_num 8 \
            --server_num 2 \
            --server_index ${server_index} \
            --data_list 1024 \
            --dtype int32 \
            --optype sum
    ) > server${server_index}.log 2>&1 &
done

# 等待两个 server 运行结束，并检查各自日志中的异常和执行结果
wait
```

`HCCL_VM_SERVER_INDEX` is the zero-based server index within the current communicator configuration. It takes effect only when `HCCL_VM_ENABLE_PYTORCH=1` and must match the `--server_index` of the corresponding script. Setting environment variables inside parentheses prevents server indexes from overwriting each other.

All servers must use the same `--master_ip` and `--master_port`. The `--dev_num` and `--server_num` values must match the `mock-comm` configuration. The local logical device IDs for each server are always `0-7`. In this example, server 0 corresponds to global ranks `0-7` and server 1 corresponds to global ranks `8-15`. The script calculates global ranks assuming the same number of devices per server. For other scales, select communicator configurations where each server has an equal number of devices.

> **Correspondence in POD Form (4-Layer Topology)**: The term "server" here refers to the **leaf node (host node)** in the communicator.
> In server form, the leaf node is the server. In **POD form, the leaf node is the board**. Therefore,
> `--server_num`/`HCCL_VM_SERVER_INDEX` should count and number by **board** (for example, the `11_64` configuration is 1 POD x 8 boards x 8 chips,
> which requires 8 concurrent Python main processes with `--server_num 8`, `--dev_num 8`, and `HCCL_VM_SERVER_INDEX` ranging from `0` to `7`).
> The local logical device IDs for each process are `0-7` within the board, consistent with the actual hardware.

#### 3.3.4 Result Viewing and Verification

After all processes finish, check the corresponding `server0.log` and `server1.log` (in multi-server scenarios). Pay attention to the exit status, exceptions, and relevant log output. Then run the Checker verification in the HCCL-VM terminal. For result descriptions, see [Checker Plugin Results](#482-checker-plugin-results).

```bash
hccl-vm plugin run @checker

# 全部用例和校验结束后退出工具终端
exit
```

### 3.4 HCCL Code Modification Verification Sample

If you have modified the CANN operator package code, such as adding a new algorithm type, follow these steps to ensure your changes take effect. The `build_pkg.sh` script helps you perform packaging, installation, and device-side dependency symbol copying. Set the environment variables before execution:

```bash
# 假设您的CANN安装目录为：/home/workspace/Ascend
source /home/workspace/Ascend/cann/set_env.sh
# 配置hccl代码仓路径
export HCCL_CODE_HOME=/home/workspace/hccl
# 配置hcomm代码仓路径
export HCOMM_CODE_HOME=/home/workspace/hcomm
```

1. If you updated or modified the CANN hccl repository code, run `bash build_pkg.sh --install hccl`.
2. If you updated or modified the CANN hcomm repository code, run `bash build_pkg.sh --install hcomm`.
3. If you updated or modified both the CANN hccl and hcomm repository code, run `bash build_pkg.sh --full`.
4. Refer to the [Usage Samples](#32-usage-samples) steps and re-run the test cases.

---

## 4. Detailed Guide

### 4.1 Tool Environment Variable Configuration

**HCCL-VM Environment Variable Description**:

| Environment Variable | Purpose | Sample |
| ------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------- |
| `HCCL_CODE_HOME` | Specifies the HCCL source code path for HCCL-VM compilation. Not configured by default. | `export HCCL_CODE_HOME=/home/workspace/hccl` |
| `HCOMM_CODE_HOME` | Specifies the HCOMM source code path for HCCL-VM compilation. Not configured by default. | `export HCOMM_CODE_HOME=/home/workspace/hcomm` |
| `HCCL_VM_ENABLE_PYTORCH` | When set to `1`, enables PyTorch adaptation. For the running method, see [PyTorch Test Case Samples](#33-pytorch-test-case-samples). | `export HCCL_VM_ENABLE_PYTORCH=1` |
| `HCCL_VM_SERVER_INDEX` | The **leaf node (host node)** index simulated by the PyTorch process, starting from `0`. Takes effect only when `HCCL_VM_ENABLE_PYTORCH=1` and must match the `--server_index` of the test script. In server form, the leaf node is the server. In **POD form (4-layer topology), the leaf node is the board**, so count and number by board. | `export HCCL_VM_SERVER_INDEX=0` |
| `HCCLVM_ENABLE_DUMP_DATA` | 1. Enables performance simulation file output to disk. When enabled, 3 `runner_hcclvm_*_data.bin` files and 1 `ccu_channel_jetty_config.xml` file are output during test case execution for performance simulation use. <br> 2. Enables the Runner plugin to dump input and output data. When enabled, the input and output data of each operator is dumped to the `all_rank_input_output.txt` file during test case execution. | `export HCCLVM_ENABLE_DUMP_DATA=1` to enable; `export HCCLVM_ENABLE_DUMP_DATA=0` or leave unset to disable. |
| `HCCL_VM_RES_STAT_DUMP` | Enables hijacking-layer resource statistics output to disk. When set, each device resource count (memory/Jetty/QP/Notify/AIV core and others) is exported as a CSV file during business process finalization, output to the installation directory `data/stat/<podId>_<serverId>_<boardId>_<device ID>_res_<timestamp>.csv` (`boardId` is the board index in POD form 4-layer topology; in server form it is always `0`). When not set (default), no export occurs. | `export HCCL_VM_RES_STAT_DUMP=1` |
| `HCCL_VM_QEMU_PATH` | Applies to x86_64 host machines (AICPU runtime): a non-empty absolute path to an external QEMU binary (the target must be an executable regular file; symbolic links pointing to regular files are allowed). When not set or invalid, the tool automatically falls back to searching PATH for `qemu-aarch64-static` and prints a WARN (including the actual binary and version). When no binary is found on the entire machine, the AICPU runtime environment fails to start. For deployment steps, see [2.3 AICPU Runtime Environment Installation](#23-aicpu-runtime-environment-installation). | `export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static` |
| `HCCL_VM_QEMU_TB_SIZE` | Applies to x86_64 host machines (AICPU runtime): QEMU translation code cache size in MiB. When not set, the default is `16`. A value of `0` explicitly passes `-tb-size 0` and delegates to the default QEMU policy (it does not disable the cache). Only decimal strings parseable to uint32_t are accepted. Invalid values trigger a warning and fall back to the default 16. When pointing to an older QEMU that does not support `-tb-size`, this option is not passed and a warning about potentially excessive memory is printed. | `export HCCL_VM_QEMU_TB_SIZE=16` |
| `HCCL_UB_MULTI_CHANNEL_NUM` | An HCCL-side environment variable read by the proxy stub (the `HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM` query item in `HcclConfigGetInfo` uses the same definition): the number of parallel jetties per channel. This value is stored with the operator as `opDetails.ubMultiChannelNum`. The `@estimate` lightweight performance simulation plugin uses this value to calculate port bandwidth utilization (`multi_port_utilization_low` tiers `+= (n-1) x jetty_utilization_step`, capped at 1.0; see `docs/light_performance/simulator_design.md` section 7.2). When not set, invalid, or `0`, the value falls back to `1` (multi-jetty is not enabled, and the original utilization table is used directly). **Note**: This value represents only the business configuration intent. The actual jetty count takes precedence from the **actual link establishment values in the proxy runtime database** (the `RaJetty` table in `hccl_sim.db` aggregates mode=3 jetty count by eid; in CCU mode, use `CcuChannelTab.jettyNum`). The environment variable value stored in the database serves as a fallback only when actual values are unavailable (for example, in old databases). When multi-jetty is configured but the actual link still uses single jetty, the tool calculates based on the actual value of 1 and prints `[JETTY][WARN]`. | `export HCCL_UB_MULTI_CHANNEL_NUM=8` |
`HCCL_VM_LOG_MODE` | Sets the simulator log type: (1) stub (default): uses the simulator spdlog to control log level and disk output; (2) cann: reuses the CANN log module, with log level and disk output fully controlled by the CANN logging capability. | `export HCCL_VM_LOG_MODE=cann`

### 4.2 HCCL-Test Test Case Build

The hccl_test test case source code is located in the CANN package installation directory. It supports compilation and execution in both OpenMPI and MPICH environments. For runtime differences, see [OpenMPI and MPICH Environment Test Case Execution Differences](#47-openmpi-and-mpich-environment-test-case-execution-differences). This guide uses the OpenMPI environment as an example.

#### 4.2.1 OpenMPI Environment Compilation

1. Install OpenMPI

```bash
sudo apt-get update
sudo apt install openmpi-bin libopenmpi-dev
```

2. Compile hccl_test

```bash
# 修改CANN安装目录权限
chmod -R 755 /home/workspace/Ascend

# 进入hccl_test用例源码目录
cd /home/workspace/Ascend/cann/tools/hccl_test

# 设置CANN环境变量
source /home/workspace/Ascend/cann/set_env.sh

# 临时修改Makefile脚本
if ! grep -q '\-lmpi_cxx' Makefile; then
    sed -i 's/-lmpi/-lmpi -lmpi_cxx/g' Makefile
fi

# 编译hccl_test用例
MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi make ASCEND_DIR=${ASCEND_HOME_PATH}
```

#### 4.2.2 MPICH Environment Compilation

Assume the mpich path is `/usr/lib/mpich`.

```bash
# 进入hccl_test用例源码目录
cd /home/workspace/Ascend/cann/tools/hccl_test

# 设置CANN环境变量
source /home/workspace/Ascend/cann/set_env.sh

# 配置环境变量
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH

# 编译hccl_test用例
make MPI_HOME=/usr/lib/mpich/ ASCEND_DIR=${ASCEND_HOME_PATH}
```

### 4.3 Ascend Cluster Topology Configuration File Description

#### 4.3.1 Server/Pod Topology Configuration File Description

An Ascend cluster topology is composed of one or more Server/Pod-form sub-topologies combined according to CLOS hierarchical network rules. Therefore, before generating a cluster topology, the user must confirm the topology type of each Server/Pod.
The user can either select a predefined topology type provided by the HCCL-VM tool or define a custom Server/Pod topology type according to the configuration file format requirements.

Describing the topology network relationships of a Server/Pod includes the following aspects:

 - **Port Configuration Table**: Describes the physical port configuration of an NPU card, such as NPU-to-NPU direct connection ports (P2P) and NPU out-of-chassis ports (P2NET).
 - **Link Configuration Table**: Describes the connection relationships among all NPU cards within a Server/Pod, such as full mesh connections.
 - **PortBound**: Describes the binding relationships of certain ports on an NPU card, where multiple ports are bound into a PortGroup.

```yaml
type: "server_intra_links"
name: "ascend950_links_topo_demo"
description: "ascend950芯片普通拓扑连接关系描述文件"

soc_version: "Ascend950"
device_num: 16

device_ports_allocate_map:
 # port分配表: 0: 不使用, 1: device直连, 2: device连交换机, 3: d2h端口
 #                 portId:  0  1  2  3  4  5  6  7  8
    - {die_id: 0, pin_map: [1, 1, 1, 0, 2, 2, 2, 2, 3]} # die0
    - {die_id: 1, pin_map: [0, 0, 0, 0, 0, 0, 0, 0, 0]} # die1

# port_group: 描述哪些port合并为一个portGroup，相同portGroup的port对于IP地址相同
port_group:
    - {layer: 0, ports: ["0/4", "0/5", "0/6", "0/7"]}

links:
  # ── 描述法：每行8个device组成全互联 ──
  - link_mode: "fullmesh"
    connections:
      # 如下示例表示：die0的device 0, 1, 2, 3都通过die0的port进行全连接
      - {die_id: 0, devices_range: [0, 3]}
      - {die_id: 0, devices_range: [4, 7]}
      - {die_id: 0, devices_range: [8, 11]}
      - {die_id: 0, devices_range: [12, 15]}
  
  #- link_mode: "enum"
  #  device_to_device_links:
  #    如下示例表示：device 0和1都通过die0的port，分别与device 1, 3, 5, 7的die1的port进行连接
  #            即：device0与device1, device3, device5, device7进行连接，device1与device3, device5, device7进行连接
  #    - {src_die_id: 1, src_device_id_range: [0, 2], dst_die_id: 1, dst_device_id_range: [1, 3, 5, 7]}

  - link_mode: "enum"
    device_to_switch_links:
      # 如下示例表示：device0到device15都通过die0的port连接到交换机。结合portGroup可知，device0到device15都通过portGroup[0/4, 0/5, 0/6, 0/7]连接到交换机。
      - {die_id: 0, devices_range: [0, 15]}

```

**Field Description**:

 - **soc_version**: The chip model, such as `Ascend950`.
 - **device_num**: The total number of devices, determined by the chip model and topology type. In server form, this is the number of cards in the server. In **POD form, this is the total number of cards in the POD** (for example, 64 for 1dpod64).
 - **board_num**: Optional. The number of boards (compute boards) within a POD. **Trigger field for POD form**: when absent or set to 1, the topology uses server form (3-layer topology). When greater than 1, the topology uses POD form (4-layer topology; see the description below). Constraints: `device_num % board_num == 0` and `board_num <= 255`.
 - **device_num_per_board**: Optional. The number of cards per board (typically 8, but not hardcoded). When absent, it is derived from `device_num / board_num`. When explicitly configured, it must match the derived value.
 - **form_type**: Optional. `"server"` or `"pod"` (case-insensitive). When absent, it is derived from `board_num`. When explicitly configured and contradictory to the derived result, an error is reported.
 - **net_instance_scope**: Optional. Specifies the scope of `net_instance_id` per layer. Values are `board`/`pod`/`superpod`/`cluster` (see section 4.3.4 for details). Default derivation rules: server form `{layer0: pod, layer1: superpod, layer>=2: cluster}`; POD form layer0 is derived based on whether PEER2PEER intra-board direct connections exist at that layer (direct connections present means `board`; pure switch network means `pod`), with remaining layers following server form.
 - **device_ports_allocate_map**: The port allocation table that describes the port configuration of each die. A value of 1 indicates a device direct connection port, 2 indicates a device-to-switch port, and 3 indicates a d2h port.
 - **port_group**: Describes which ports are merged into a single portGroup. Ports within the same portGroup share the same IP address. Ports not configured default to each port forming its own portGroup.
 - **links**: The link configuration table that describes the connection relationships among all NPU cards within a Server/Pod, as well as between NPUs and switches.
   - **NPU Direct Connection Relationships**: The tool provides two methods to configure NPU direct connection relationships:
    - **link_mode == "fullmesh"**: All devices within a die perform full-mesh connections through the ports of that die. Additional typical connection methods can be added as new link_mode types in the future, such as "ring".
    - **link_mode == "enum"**: Enumeration method. When the NPU connection method within a Server/Pod is complex, all link relationships can be described by enumeration.
   - **NPU-to-Switch Connection Relationships**: Users can configure NPU-to-switch connection relationships using the enumeration method.
  - **device_to_device_links**: Describes the connection relationships between NPUs. The two endpoint devices are declared using `src_device_id`/`src_device_id_range`
    and `dst_device_id`/`dst_device_id_range` (**physical IDs within the board**; the legacy names `src_local_id`/`dst_local_id`/`*_local_id_range` remain compatible).
    In POD form, **`src_board_id`/`dst_board_id`** can be used to declare the board each endpoint belongs to (both must be configured simultaneously), **supporting cross-board direct connections**.
  - **device_to_switch_links**: Describes the connection relationships between NPUs and switches. In POD form, a single **`board_id`** declares which board the device belongs to (the destination is a switch, so no dst_board_id is needed).

> **Note (POD Form)**: Device IDs that appear in `device_group`/`port_list`/`port_group`/`links`
> must be **physical IDs within the board** (such as `0-7`) in POD form. That is, the card structure is described per "single board". The `local_id` is calculated by the tool as
> `board_id x cards per board + physical ID within board`. The board ownership fields for link entries:
> **`fullmesh` `connections`, `device_to_switch_links`, and `matrix`** use a single **`board_id`** (both endpoints on the same board).
> **`device_to_device_links`** uses **`src_board_id`/`dst_board_id`** (endpoints can be on different boards, that is, cross-board direct connections).
> Entries without board fields apply to all boards (that is, a single-board description is replicated and expanded).
> If a POD-global ID (such as `8` or `63`) is mistakenly used, the tool reports an error directly (`device id exceeds value range [0, cards per board - 1]`).

**POD Form (4-Layer Topology) Description**

The topology has two forms with different layer counts:

| Form | Layers | Leaf Node (Host Node) | Card Physical ID | Output |
|------|------|---------------------|----------|------|
| Server Form | server -> supernode -> cluster (3 layers) | server | `[0, device_num-1]` | One `topo.json`/`hccl_rootinfo.json` per server |
| POD Form | **board -> pod -> supernode -> cluster (4 layers)** | **board** | **Within board `[0, cards per board - 1]`** (typically `[0,7]`) | **One `topo.json` per POD** (server directory) + **one `hccl_rootinfo.json` per board** |

In POD form, the cards within a POD are divided into several boards (compute boards). Each board is an independent host node (independent host IP, independent rootinfo).
**`topo.json` is a single file at the POD level** placed in the server directory, where `local_id` is the **POD-global ID**:
`local_id = board_id x cards per board + physical ID within board`.
For example, a `1dpod64` (64-card POD) configuration with `device_num: 64` + `board_num: 8` generates the following cluster directory:

```
config/network/cluster/ascend950_cluster_16_server_1dpod64/
├── servers_info.json                # 每叶子节点(每board)一条, 共 16 POD × 8 board = 128 条
└── superpod0/
    └── server0/                     # POD 0(不分配host IP)
        ├── topo.json                # POD级拓扑: 64卡, local_id = board_id*8 + board内id (0~63)
        ├── board0/
        │   └── hccl_rootinfo.json   # 8卡, device_id/local_id = board内物理id 0~7
        ├── board1/ ...
        └── board7/
```

Configuration sample (`asset/cluster_model/config/server_or_pod/ascend950_server_topo_1dpod64.yaml`, intra-board direct connections, links described per board):

```yaml
soc_version: "Ascend950"
device_num: 64                       # POD总卡数
board_num: 8                         # 8个board → 每board 8卡
hardware_type: "950D-1D-POD-64P-topo"

# 卡结构为"单个board"的描述, device id 为 board 内物理id(0~7)
device_group:
  - {group_id: 1, device_list: [0, 1, 2, 3]}
  - {group_id: 2, device_list: [4, 5, 6, 7]}
links:
  # links按board逐一描述: fullmesh条目携带board_id, d2d条目携带src_board_id/dst_board_id;
  # net_instance_id/topo_instance_id按board编号
  - link_mode: "fullmesh"
    connections:
      - {board_id: 0, die_id: 0, net_instance_id: 0, topo_instance_id: 0, device_id: [0, 1, 2, 3]}
      - {board_id: 0, die_id: 1, net_instance_id: 0, topo_instance_id: 0, device_id: [4, 5, 6, 7]}
      - {board_id: 1, die_id: 0, net_instance_id: 1, topo_instance_id: 1, device_id: [0, 1, 2, 3]}
      # ... board2~board7 同理
  - link_mode: "enum"
    device_to_device_links:
      # 板内直连: src_board_id == dst_board_id
      - {src_board_id: 0, net_instance_id: 0, topo_instance_id: 0, src_die_id: 0, src_device_id: [0, 1, 2, 3],
         dst_board_id: 0, dst_die_id: 1, dst_device_id: [4, 5, 6, 7]}
      # ... board1~board7 同理
```

Configuration sample (`ascend950_server_topo_1dpod64.yaml`, 8 boards x 8 chips, POD form, intra-board direct connections: 4 cards each in die0/die1 per board in full mesh + cross direct connections):

```yaml
device_num: 64
board_num: 8                          # 8个board → 每board 8卡
device_group:
  - {group_id: 1, device_list: [0, 1, 2, 3]}
  - {group_id: 2, device_list: [4, 5, 6, 7]}
links:
  - link_mode: "fullmesh"             # 每board内die0/die1各4卡全互联
    connections:
      - {board_id: 0, die_id: 0, net_instance_id: 0, topo_instance_id: 0, device_id: [0, 1, 2, 3]}
      - {board_id: 0, die_id: 1, net_instance_id: 0, topo_instance_id: 0, device_id: [4, 5, 6, 7]}
      # ... board1~board7 同理
  - link_mode: "enum"
    device_to_device_links:
      # 板内直连: 每board内die0的4卡与die1的4卡交叉直连
      - {src_board_id: 0, net_instance_id: 0, topo_instance_id: 0, src_die_id: 0, src_device_id: [0, 1, 2, 3],
         dst_board_id: 0, dst_die_id: 1, dst_device_id: [4, 5, 6, 7]}
      # ... board1~board7 同理
```

Configuration sample (`ascend960_pod_topo_clos.yaml`, full-switch within chassis with no intra-board direct connections, links without board_id are replicated and expanded to all boards):

```yaml
device_num: 72
board_num: 9                         # 9个board × 8卡
net_instance_scope:
  - {layer: 0, scope: "pod"}         # layer0为覆盖整POD 72卡的框内交换网, 不能按board切分
links:
  - link_mode: "enum"
    device_to_switch_links:
      - {die_id: 0, devices_range: [0, 7]}   # 未配置board_id → 对全部9个board生效
```

**Mandatory Validation for POD Form** (generation fails if any condition is not met):
1. `device_num % board_num == 0` and cards per board > 0.
2. All device IDs in the configuration must fall within `[0, cards per board - 1]`. The `board_id` in link entries and `src_board_id`/`dst_board_id` in `device_to_device_links`
   (if configured) must fall within `[0, board_num - 1]`, and `src_board_id`/`dst_board_id` must be configured simultaneously.
3. `board_num <= 255` (IP CC segment capacity) and endpoints per board `<= 255` (IP DD segment capacity).
4. When PEER2PEER direct connections exist at layer0: **intra-board direct connections** (each board forms an independent network) must use scope `board` (to ensure layer0
   `net_instance_id` is unique per board; otherwise, intra-board direct connection groups from multiple boards would be incorrectly merged by the business side). **Cross-board direct connections**
   (edges where `src_board_id != dst_board_id` exist, and the direct connection domain is a single POD-level instance) must use scope `pod`
   (otherwise, the `net_instance_id` at both endpoints of a cross-board direct connection edge would be inconsistent and would be filtered out by the business side).
5. When PEER2PEER direct connections exist, the number of direct connection edges per board must be consistent and every board must have direct connection entries (homogeneity check; cross-board edges count once for each endpoint board).
6. The layers declared in `net_instance_scope` must actually exist in the topology.

> `fullmesh`/`device_to_switch_links`/`matrix` entries use a single `board_id` (both endpoints on the same board). `device_to_device_links`
> uses `src_board_id`/`dst_board_id` to declare the ownership of both endpoints, **supporting cross-board PEER2PEER direct connections** (such as board0<->board1 full cross, requiring `src_board_id != dst_board_id`).
> The `device_id` in ranktable.json is the physical ID within the board, and `local_id` is the POD-global ID (consistent with the POD-level topo.json).

#### 4.3.2 Cluster Topology Configuration File Description

An Ascend cluster network is composed of one or more Server/Pod-form sub-topologies combined according to CLOS hierarchical network rules. Users can select different Server/Pod topology types based on cluster scale and requirements.

Users can define custom cluster topology configuration files according to the following configuration file format:

```yaml
name: "ascend950_cluster_32_server_normal"
description: "昇腾950 normal组网：32个超级节点，每个超级节点1个服务器"

# 超节点总数量
super_node_num: 4
# sever/pod总数量
server_num: 32
server_list:
  # 0-7 server: 均采用ascend950_server_topo_normal拓扑类型
  - {super_pod_id: 0, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 1, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 2, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 3, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}

```

The configuration file above describes a cluster topology containing 4 supernodes, 32 servers, and a total of 128 NPU chips. Each Server/Pod uses the ascend950_server_topo_normal topology type.

**Field Description**:

 - **super_node_num**: The total number of supernodes.
 - **server_num**: The total number of servers/pods.
 - **server_list**: The configuration information for each Server/Pod, including the supernode ID, device ID range, chip model, and Server/Pod topology configuration file path.

> **POD Form Scale Conversion**: `server_num` counts the number of servers/PODs. If the topology configuration pointed to by `server_topo` declares
> `board_num > 1` (POD form), the total number of **leaf nodes (host nodes) in the cluster = server_num x board_num**,
> and the total number of cards = number of leaf nodes x cards per board. For example, `ascend950_cluster_16_server_1dpod64` (`server_num: 16`,
> topology `board_num: 8`, 8 cards per board) has 128 leaf nodes and 1024 cards. `servers_info.json` will contain 128 records.
>
> Each record in `servers_info.json` corresponds to one leaf node and carries explicit `super_pod_id`/`server_id`/`board_id`
> triple IDs (the tool matches by ID, not by entry position). The host IP rules are:
> server form `AA.BB.0.0` (AA = supernode ID + 1, BB = server ID + 1); POD form `AA.BB.CC.0` (CC = board_id).

#### 4.3.3 Communicator Configuration File Description

In an Ascend cluster environment, users must select different communicator configuration files based on the communicator required by the operator to be executed.

The tool provides the `hccl-vm mock-comm` command to read and configure the operator communicator configuration file. The communicator configuration file uses YAML format and is located at `hccl_vm_install/config/topo_meta`. If the directory does not contain the corresponding communicator configuration file, the user must create one first.

The hccl-vm tool supports asymmetric topology communicator configurations. An example is shown below:

```yaml
# 1. 全局统计信息 podNum, serNum, rankNum 都小于 1024
meta:
  podNum: 1  # 总的超节点数
  serNum: 2  # 总的server数
  rankNum: 6 # 总的rank数

# 2. 详细拓扑结构 
topology:
  - podId: 0
    servers:
      - serId: 0
        # 每个server实际跑的rank的local id
        ranks: [0, 2]
      - serId: 1
        # 每个server实际跑的rank的local id
        ranks: [1, 3, 5, 7]
```

**Precautions**:

- When configuring a communicator, the tool regenerates the `topo.json` and `ranktable.json` files based on the specified communicator configuration ID.
- In the communicator configuration YAML file above, the `ranks` field represents the list of local IDs (that is, the physical IDs of the devices) of the ranks actually running on each server.

When the communicator scale is large, the server selection fields and rank selection fields in server entries can be freely combined. The following fields are supported:

- **Server Selection Fields** (choose one; they cannot be configured simultaneously):
  - `serId`: A single server ID or a list of server IDs (such as `3` or `[0, 1]`; a list represents each enumerated server).
  - `serId_range`: A server ID range with the value `[<start ID>, <end ID>]` (closed interval; only list form is supported), representing all servers from the start ID to the end ID under the supernode.
- **Rank Selection Fields** (choose one; they cannot be configured simultaneously):
  - `ranks`: The list of local IDs of the ranks actually running on each server.
  - `ranks_range`: A local ID range with the value `[<start ID>, <end ID>]` (closed interval; only list form is supported), representing all local IDs from the start ID to the end ID on each server.

The following example (2 supernodes, 4 servers in total, 32 ranks) demonstrates multiple combinations mixed within the same file:

```yaml
# 1. 全局统计信息 podNum, serNum, rankNum 都小于 1024
meta:
  podNum: 2  # 总的超节点数
  serNum: 4  # 总的server数
  rankNum: 32 # 总的rank数

# 2. 详细拓扑结构 
topology:
  - podId: 0
    servers:
      # serId为id列表，表示pod 0下server 0和server 1
      - serId: [0, 1]
        ranks: [0, 1, 2, 3, 4, 5, 6, 7]
  - podId: 1
    servers:
      - serId: 0
        ranks_range: [0, 7]
      - serId: 1
        ranks_range: [0, 7]
```

**Precautions**:

- `serId` and `serId_range` cannot be configured simultaneously. `ranks` and `ranks_range` cannot be configured simultaneously.
- `serId_range` and `ranks_range` support only the `[<start ID>, <end ID>]` list form (closed interval). The `<start ID>-<end ID>` format is not supported. The start ID must not be greater than the end ID, and the range width (end ID - start ID + 1) must not exceed 1024.
- A `serId_range` entry expands to all servers from the start ID to the end ID under the supernode. A `serId` list entry represents each enumerated server, and the ranks for each server are identical.
- If no rank selection field is configured in a server entry, the ranks for that server are an empty list. If no server selection field is configured, `serId` defaults to 0.
- The syntax above applies to **server form** (3-layer topology): `ranks` are physical IDs within the server, and one server is one leaf node.

**POD Form (4-Layer Topology) Boards Configuration**

In POD form, the leaf node is a board (not a POD). Therefore, the communicator must declare the participating cards per board,
and `ranks` must be **physical IDs within the board** (such as `[0, 7]`, not `[0, 63]` within the POD):

```yaml
# 1. 全局统计信息 podNum, serNum, boardNum, rankNum 都小于 1024
#    (serNum/boardNum/rankNum 均为全集群总数)
meta:
  podNum: 1        # 超节点总数
  serNum: 1        # server/POD 总数
  boardNum: 8      # board 总数(全集群)
  rankNum: 64      # rank 总数

# 2. 详细拓扑结构
topology:
  - podId: 0
    servers:
      - serId: 0
        boards:
          # 8个board, 每board取物理id 0~7 的卡 → 共64个rank
          - {boardId_range: [0, 7], ranks_range: [0, 7]}
```

- **Board Selection Fields** (choose one; they cannot be configured simultaneously):
  - `boardId`: A single board ID or an ID list (such as `3` or `[0, 1]`).
  - `boardId_range`: A board ID range, `[<start ID>, <end ID>]` (closed interval; width must not exceed 1024).
- **Rank Selection Fields**: `ranks` / `ranks_range` with the same semantics as above, but the value range is **physical IDs within the board**.
- `boards` is an optional field: **when not configured, the server is treated as having only one implicit `board0`**.
  In this case, `ranks`/`ranks_range` are written directly under the server entry (the existing server-form syntax, fully backward compatible).
- `boards` and `ranks`/`ranks_range` on the server entry are **mutually exclusive**. Configuring both simultaneously causes an error (rank ownership cannot be determined).
- When `boardId`/`boardId_range` is not configured, the default ownership is `board0`.
- The rank_id assignment order is: supernode -> server (POD) -> board -> device, incrementing from 0.

POD-form communicator configurations already provided by the tool (`asset/cluster_model/topo_meta/`):

| ID | Scale | Description |
|------|------|------|
| `118` | 1 supernode x 1 POD x 1 board x 8 chips = 8 ranks | Single board (server-form syntax; in POD form, this is board0) |
| `11b2` | 1 x 1 x 2 boards x 8 chips = 16 ranks | 2 boards within a single POD |
| `11b4` | 1 x 1 x 4 boards x 8 chips = 32 ranks | 4 boards within a single POD |
| `11b9` | 1 x 1 x 9 boards x 8 chips = 72 ranks | POD72 (ascend960_pod_topo_clos) entire POD |
| `11_64` | 1 x 1 x 8 boards x 8 chips = 64 ranks | 1dpod64 entire POD |
| `12_64` / `14_64` / `18_64` / `116_64` | 2 / 4 / 8 / 16 PODs, each POD with 8 boards x 8 chips | 128 / 256 / 512 / 1024 ranks |

#### 4.3.4 topo.json and ranktable.json File Description

The `topo.json` and `ranktable.json` files do not need to be created manually. The tool generates them automatically based on the following information:

- **Topology Configuration ID**: The ID specified by the user at startup (such as 112, 113, and others)
- **Chip Type**: The chip type automatically identified based on the runtime environment.

Although the configuration files are automatically generated by the tool, understanding their structure helps in comprehending the topology configuration.

**topo.json Structure**:

`topo.json` describes the connection relationships of all devices within a server:

```json
{
  "server": {
    "device_count": 8,
    "groups": [
      {
        "group_id": 0,
        "device_start": 0,
        "device_count": 8,
        "topo_layout": "1D"
      }
    ]
  },
  "ports": [
    {
      "ccu": "die0",
      "port_pattern": "0/{0-6}",
      "protocol": "HCCS",
      "func_id": 2,
      "usage": "peer2peer",
      "ip_binding": "independent"
    },
    {
      "ccu": "die0",
      "port_pattern": "0/7,0/8",
      "protocol": "ROCE",
      "func_id": 3,
      "usage": "peer2net",
      "ip_binding": "independent"
    }
  ],
  "links": [
    {
      "net_layer": 0,
      "link_type": "PEER2PEER",
      "topo_type": "1DMESH",
      "ccu": "die0",
      "port_pattern": "0/{0-6}",
      "connect_pattern": "full_mesh",
      "group_id": 0
    },
    {
      "net_layer": 1,
      "link_type": "PEER2NET",
      "topo_type": "CLOS",
      "ccu": "die0",
      "port_pattern": "0/7,0/8",
      "connect_pattern": "all_to_net",
      "group_id": 0
    }
  ]
}
```

**Field Description**:

- `server.device_count`: The total number of devices.
- `server.groups`: Device grouping information.
- `ports`: Port configuration.
  - `usage`: Port purpose (`peer2peer` indicates inter-device connections; `peer2net` indicates external connections)
- `links`: Link configuration.
  - `link_type`: Link type (`PEER2PEER` or `PEER2NET`)
  - `topo_type`: Topology type (`1DMESH`, `CLOS`, and others)

**ranktable.json Structure**:

`ranktable.json` describes the mapping between the devices used by the current communicator and the addresses (EIDs) at each network layer.
The tool generates this file automatically at `hccl_vm_install/data/ranktable.json` during `mock-comm` based on the communicator configuration:

```json
{
  "version": "2.0",
  "rank_count": 64,
  "rank_list": [
    {
      "rank_id": 0,
      "device_id": 0,
      "local_id": 0,
      "level_list": [
        {
          "net_layer": 0,
          "net_instance_id": "sp_0_srv_0_board_0_idx_0",
          "net_type": "TOPO_FILE_DESC",
          "net_attr": "",
          "position": "DEVICE",
          "protocols": ["UB_CTP", "UB_MEM"],
          "rank_addr_list": [
            {
              "addr_type": "EID",
              "addr": "00000000000000000000000001010001",
              "ports": ["0/0"],
              "plane_id": "plane_0"
            }
          ]
        },
        {
          "net_layer": 1,
          "net_instance_id": "sp_0_idx_0",
          "net_type": "CLOS",
          "rank_addr_list": [ { "addr_type": "EID", "addr": "00000000003f00000000000001010008", "ports": ["0/1", "0/2"], "plane_id": "plane_pg_0" } ]
        },
        {
          "net_layer": 3,
          "net_instance_id": "cluster",
          "net_type": "CLOS",
          "position": "HOST",
          "protocols": ["ROCE"],
          "rank_addr_list": [ { "addr_type": "EID", "addr": "0000000000480000000000000101000a", "ports": ["d2h"], "plane_id": "plane_pg_0" } ]
        }
      ]
    }
  ]
}
```

**Field Description**:

- `rank_count`: The total number of ranks in the current communicator.
- `rank_id`: The global rank number within the communicator, incrementing from 0 in the order of **supernode -> server (POD) -> board -> device**.
- `device_id`: The **physical ID within the leaf node** (physical ID, not logical ID). This is the physical device number selected by the business-side `aclrtSetDevice()`.
  The `CheckRankGraphAddrs` function in hcomm filters entries by `rank.deviceId == devPhyId` and queries hardware EIDs using this phyId.
  In server form, the value range is `[0, cards per server - 1]`. **In POD form, the value range is `[0, cards per board - 1]` (typically `[0,7]`)**.
- `local_id`: Uses the same definition as `local_id` in `topo.json`. The business side matches edges in topo.json using the `local_id` from ranktable.
  In server form, this value is the same as `device_id` (in this tool, `logic_id == physical_id`). **In POD form, `topo.json` is a single file at the POD level
  (in the server directory), and `local_id` is the POD-global ID = `board_id x cards per board + physical ID within board`** (for 1dpod64, the value range is `[0,63]`).
  `device_id` remains the physical ID within the board.
- `level_list[].net_layer`: The network layer number (0 is the intra-node direct connection layer, 1 is the inter-node/inter-chassis layer, 2 and above are outer layers, and the d2h/HOST layer is typically 3).
  **After introducing the board layer, the net_layer numbering rules remain unchanged**. The board dimension is expressed through `net_instance_id`.
- `level_list[].net_instance_id`: The network instance identifier for the layer, assembled according to the **scope**:

  | Scope | Format | Typical Application |
  |--------|------|----------|
  | `board` | `sp_{supernode}_srv_{POD}_board_{board}_idx_{k}` | POD form layer0 (intra-board direct connection, **unique per board**) |
  | `pod` | `sp_{supernode}_srv_{POD}_idx_{k}` | Server form layer0; POD form cross-board intra-chassis switch network |
  | `superpod` | `sp_{supernode}_idx_{k}` | layer1 (cross-server/POD but within the same supernode, inter-chassis interconnection) |
  | `cluster` | `cluster` | layer>=2 (cluster level, HOST/d2h layer) |

- `level_list[].net_type`: `TOPO_FILE_DESC` when `net_layer=0`; `CLOS` when `net_layer>=1`.
- `level_list[].rank_addr_list[].addr`: The EID bound to the port at this layer (with the `0x` prefix removed). For the UBOE protocol layer, it uses dotted IPv4 format (`addr_type: "IPV4"`).
- `level_list[].rank_addr_list[].plane_id`: Fixed as `plane_0` for the PEER2PEER layer. For the PEER2NET layer, it inherits `plane_pg_{N}` from the configuration.

> **Layer Trimming Rules**: For a layer to appear in ranktable, EIDs must be allocated to that layer in rootinfo.
> When the communicator involves only **1 leaf node** (1 server in server form, 1 board in POD form) and does not cross supernodes, `net_layer>=3` is not output.
> When the total number of net_layers for a single leaf node exceeds 3, only the 3 layers with the smallest numbers are retained. Cross-supernode communicators do not perform the trimming described above.

### 4.4 hccl\_config.sh File Description

The `hccl_config.sh` file contains the environment variable configuration required for running HCCL\_Test test cases. The environment variables in this file are consistent with those used in an actual hardware environment.
Users must modify the `hccl_config.sh` script according to their test cases and requirements to configure the HCCL test case runtime environment variables.

```bash
#!/bin/bash
# hccl_config.sh - HCCL 环境变量配置

remove_files_by_prefix() {
  if [ "$#" -ne 1 ]; then
    echo "Usage: remove_files_by_prefix <prefix>" >&2
    return 2
  fi

  local prefix="$1"
  if [ -z "$prefix" ]; then
    return 0
  fi

  shopt -s nullglob
  local any_deleted=0
  for f in "${prefix}"*; do
    if [ -f "$f" ]; then
      rm -f -- "$f" && any_deleted=1
    fi
  done
  shopt -u nullglob

  # 无论是否删除了文件，均返回 0，确保脚本继续执行
  return 0
}

# 清理 data/ 目录的冗余文件（CCU 模式下生成的临时文件）
cd "${HCCL_VM_INSTALL_DIR}/data" 2>/dev/null && {
  remove_files_by_prefix "sqe_info_rank_"
  remove_files_by_prefix "mc_instr_info_rank_"
  rm -f "all_rank_input_output.txt"
  cd "${HCCL_VM_INSTALL_DIR}"
}

# 设置CANN环境变量
source /home/workspace/Ascend/cann/set_env.sh

# 关闭hccl心跳功能
export HCCL_DFS_CONFIG=cluster_heartbeat:off

# 设置 HCCL-VM 安装路径，基于脚本自身位置推断（兼容 bin/ 与 script/ 子目录）
_INSTALL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$(basename "${_INSTALL_SCRIPT_DIR}")" in
    bin|script)
        export HCCL_VM_INSTALL_DIR="$(dirname "${_INSTALL_SCRIPT_DIR}")"
        ;;
    *)
        export HCCL_VM_INSTALL_DIR="${_INSTALL_SCRIPT_DIR}"
        ;;
esac
unset _INSTALL_SCRIPT_DIR

# 配置LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH

# 设置 ranktable.json文件路径 (与 mock-comm 生成路径保持一致)
export RANK_TABLE_FILE=${HCCL_VM_INSTALL_DIR}/data/ranktable.json

# 设置日志级别
export ASCEND_GLOBAL_LOG_LEVEL=1

# 设置日志打屏输出
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 设置 HCCL 运行模式（CCU、AI_CPU、AIV等展开模式）
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
# 或设置 HCCL 运行参数(AI_CPU展开模式) AI_CPU模式环境变量与其他模式不可同时设置
# export HCCL_OP_EXPANSION_MODE="AI_CPU"
# 或设置 HCCL 运行参数(AIV展开模式) AIV模式环境变量与其他模式不可同时设置
# export HCCL_OP_EXPANSION_MODE="AIV"

echo "HCCL-VM environment configured successfully!"

```

### 4.5 Tool Specification Constraints

**Supported Operator Types**:

The tool supports the following operator types: allgather/allreduce/alltoall/reduce/reduce\_scatter/scatter/alltoallv.

**Supported Data Types**:

The HCCL-VM tool supports the following data types: int8/int16/int32/fp16/fp32/uint8/uint16/uint32/bfp16/hif8/fp8e4m3/fp8e5m2/fp8e8m0.

The HCCL-VM Runner plugin supports the following data types:

| ReduceOp | DataType                           |
| -------- | ---------------------------------- |
| `ADD`    | `int8/int16/int32/uint8` |
| `MIN`    | `int8/int16/int32/uint8` |
| `MAX`    | `int8/int16/int32/uint8` |

**Hardware Specifications**:

This tool currently supports Ascend950 and Ascend960 chips. The built-in storage applies to communicators with 1 to 128 chips by default and does not require database configuration.

### 4.6 HCCL-VM Plugin Functions

#### 4.6.1 Runner Plugin

The Runner plugin simulates the execution of tasks generated by HCCL business orchestration and produces output data.
The runner plugin is **disabled** by default during hccl\_test test case execution. After an hccl\_test test case calls an operator interface, it waits for operator task execution to complete through the `aclrtSynchronizeStream` interface. The runner tool waits until all ranks are in a waiting state, then starts simulated execution of tasks for all ranks. After execution completes, it notifies each rank's test case to continue execution.
After test case execution completes, users can view the input buffer and output buffer data for each rank through the `all_rank_input_output.txt` file in the execution directory. This feature is not enabled by default. Users can enable it before test case execution using the corresponding command.

**Installation and Uninstallation**:

The Runner plugin supports installation and uninstallation through the `hccl-vm plugin install/uninstall` command. Install the runner plugin after entering the hccl-vm tool command line and before executing the test case. Only subsequent executions will run the runner.

```bash
# 安装runner插件
(hvm)$> hccl-vm plugin install @runner

# 卸载runner插件
(hvm)$> hccl-vm plugin uninstall @runner
```

#### 4.6.2 Checker Plugin

The Checker plugin is an algorithm analyzer plugin. Its function is to form a DAG graph from all tasks generated by HCCL, analyze the DAG graph to determine whether memory conflicts exist, and simulate DAG graph execution to determine whether semantic errors or other issues exist.

The algorithm analyzer plugin is started and executed by the user through commands.

The Checker plugin currently uses Checker V3 for single-operator verification. Checker V3 big graph verification is enabled by default: multiple operators within each sync window are merged into a single big graph for cross-operator synchronous resource conflict verification. Single-operator Checker V3 and big graph verification can be controlled independently through configuration parameters in `manifest.json`.

```json

# 配置文件位于 /pathto/hccl_vm_install/plugin/checker/manifest.json

{
  "name": "checker",        // Checker插件名
  "version": "1.0.0",       // Checker插件版本
  "entry": "./checker",     // Checker插件启动指令
  "dependency": {
      "min_core_version": "1.0.0"
  },
  "setting": {              // Checker插件配置项
      "enable_new_checker": true,           // 是否启用 Checker V3 单算子校验（默认开启）
      "enable_big_graph_checker": true,     // 是否启用 Checker V3 大图校验（默认开启，与单算子校验独立）
      "enable_insight_dump": false,         // 是否输出 Checker V3 Insight 数据（默认关闭）
      "enable_memory_snapshot_dump": false, // 是否输出内存快照数据（默认关闭，需要先开启 Insight 数据输出）
      "enable_dag_graphviz_dump": false,     // 是否输出大图 Graphviz DOT 文件（默认关闭）
      "enable_ccu_resource_stats_dump": false // 是否输出 CCU 资源统计数据（默认关闭）
  }
}
```

### 4.7 OpenMPI and MPICH Environment Test Case Execution Differences

Before running hccl_test test cases, users can determine which mpirun is used in the current environment by using the `which` command.

#### 4.7.1 Environment Variable Configuration Differences

The environment typically configures OpenMPI by default. If users run test cases with OpenMPI, additional environment variable configuration is generally not required.
If users run test cases in an MPICH environment, configure the environment variables as follows:

```bash
# 配置mpich环境变量
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH
export PATH=/usr/lib/mpich/bin:$PATH
```

#### 4.7.2 mpirun Command Parameter Differences

In an OpenMPI environment, users run hccl_test test cases with the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun --allow-run-as-root --oversubscribe -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter Description**:

 - --allow-run-as-root: An OpenMPI-specific parameter that allows running MPI processes as the root user, for use in environments without root privileges.
 - --oversubscribe: An OpenMPI-specific parameter that removes CPU slot limits, allowing a single node to start more processes than logical CPU cores (oversubscription).
 - -np 2: Specifies 2 processes, consistent with the number of nodes.

In an MPICH environment, users run hccl_test test cases with the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter Description**:

 - -np 2: Specifies 2 processes, consistent with the number of nodes.

### 4.8 Result Viewing

#### 4.8.1 Runner Plugin Results

If the runner plugin is installed by running `hccl-vm plugin install @runner` in the hccl-vm terminal, the operator pipeline execution automatically triggers the runner plugin after completion. The final result relies on hccl_test for verification. Users need to check whether `[error]`-level log entries exist in the redirected log file and the final verification result:

```bash
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
64                | 1000.00        | 0.00006              | success
```

#### 4.8.2 Checker Plugin Results

After running `hccl-vm plugin run @checker` in the hccl-vm terminal, the Checker verification process and results are printed in the terminal. Users need to check whether `[error]`-level log entries exist in the log file and the final verification result:

```bash
[HCCL_VL-CHECKER][CHECKER_RUN_SUMMARY] All Success (Total Op: 4, Big Grapth: 1)
```

---
### 4.9 Operating Modes (normal / check-only / estimate)

HCCL-VM has three operating modes. At any given time, one and only one mode is active. The default is `normal`:

| Mode | Purpose | Description |
| ---- | ---- | ---- |
| `normal` | Normal Mode (default) | Actual independent allocation and complete multi-rank communication with full result correctness |
| `check-only` | Verification-Only Mode | Large memory blocks reuse a 4 GB shared pool, significantly reducing memory usage. Only the Checker verification pipeline is guaranteed. Buffer data content correctness is not guaranteed. |
| `estimate` | Estimation Mode | Single chip (rank0) constructs phantom peers to skip actual communication blocking points. Quickly produces a task list without actually running data transfer. (Link heartbeat and consistency verification must be disabled first; see [4.9.4](#494-estimation-mode-limitations)) |

#### 4.9.1 Specifying the Mode at Startup

Both the `start` and `run` (one-shot) subcommands support the `--check-only` / `--estimate` flags (they are mutually exclusive). Without any flag, the mode defaults to `normal`:

```bash
# 以仅校验模式启动
./hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only

# 以估算模式启动
./hccl-vm start ascend950_cluster_32_server_normal.yaml --estimate
```

#### 4.9.2 Querying and Switching Modes at Runtime

Within an already-started VM command line, you can query and switch modes using `hccl-vm mode` without restarting the VM:

```bash
# 查询：列出全部支持模式与当前模式（不带参数等价于 show，无副作用）
(hvm)$> hccl-vm mode show
Supported modes:
        check-only : 仅校验模式，大块内存复用 4GB 共享池 HcclCommPool，大幅降低 /dev/shm 占用；只保证校验链路，缓冲区数据内容不保证正确
        estimate : 估算模式，单卡(rank0)构造幻影对端跳过真实通信阻塞点，快速产出任务清单，不真跑数据传输
        normal : 普通模式(默认)，真实独立分配与完整多 rank 通信，结果正确性无损
Current mode: normal

# 切换模式
(hvm)$> hccl-vm mode check-only
(chk-only)$> hccl-vm mode estimate
(estimate)$> hccl-vm mode normal
```

After a successful switch, the terminal prompt changes with the mode: normal -> `(hvm)$>`, check-only -> `(chk-only)$>`, estimate -> `(estimate)$>`.

**Switching Behavior and Constraints**:

- **Strict Spelling**: Mode names must use exact lowercase hyphenated spelling. Variants such as `CHECK-ONLY`, `check_only`, and other invalid strings are rejected, and the current mode remains unaffected.
- **Idempotent**: Switching to the current mode is a no-op, and the tool reports that the mode is already active.
- **Reuse Pool Creation and Teardown**: When switching to check-only, the tool automatically creates a 4 GB shared reuse pool (cleaning up residuals first). When switching away, the pool is automatically released. The pool does not exist in normal / estimate modes.
- **Runner Uninstallation Before Switch**: Before switching to a non-normal mode, if the runner plugin is installed, it is uninstalled first. If uninstallation fails, the switch is aborted. Switching to normal does not trigger uninstallation.
- **Switch Atomicity**: If any step during the switch fails (pool creation failure, pool teardown failure, and others), the switch is aborted with an error, and the mode and resources remain in their pre-switch state. Known limitation: the prompt may still change when a switch fails. In this case, use the `hccl-vm mode show` query result as the authoritative source.
- **Effect on New Test Cases**: The new mode takes effect for business test cases launched after the switch succeeds. Test cases already running before the switch are not affected and continue to run in the mode they were launched with.
- **Not Preserved Across Restarts**: The mode is not preserved across VM restarts. After a restart, the tool reverts to the initial mode determined by startup parameters (defaulting to normal if not specified).

#### 4.9.3 Check-Only Mode Large Memory Reuse Description

The check-only mode is used for large-scale cluster scenarios that only run Checker verification. When enabled, large memory block allocations reuse the same shared area `HcclCommPool`. All ranks share this pool and are allowed to overwrite each other, significantly reducing `/dev/shm` usage. In this mode, large block content correctness is not guaranteed. This mode applies only to Checker V3 verification pipelines that do not read buffer data. Do not enable this mode when numerically correct results are required. The reuse area default range is 200 MB to 4 GB. The 4 GB upper limit can be changed using the `--kpoolsize N` command.

Allocations smaller than 200 MB always use actual allocation. Single blocks larger than the upper limit of the reuse area are directly rejected with an error in check-only mode. The check-only mode and Runner are not mutually exclusive, but when the Runner is installed while check-only mode is enabled, large block reuse still takes effect and may overwrite Runner data. The tool prints a warning.

**Reuse Area Size Configuration Description**
`--kpoolsize N` supports three entry points. The following examples set the reuse pool size to 5 GB:

```bash
# 1) 启动时指定（分离式），
./bin/hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only --kpoolsize 5G

# 2) 运行期，在 (chk-only)$> 子 shell 里切换/重配
(chk-only)$> hccl-vm mode check-only --kpoolsize 5G

# 3) 运行期，在其他模式子 shell 里切换/重配
(hvm)$>  hccl-vm mode check-only --kpoolsize 5G
```
**Configuration Behavior and Constraints:**
- **Strict Spelling**: The option name must be exactly `--kpoolsize`. Uppercase or extra hyphens such as `--Kpoolsize` or `k_poolsize` are not allowed.
- **Idempotent**: The reuse area size can be changed even when already in check-only mode.
- **Syntax**: At startup, the `--check-only` mode must be selected before the size can be set. At runtime, the command must follow `mode check-only` before `--kpoolsize` can be configured. It cannot be used standalone.
- **Units**: Pure decimal numbers are supported, such as `--kpoolsize 2147483648`. K, M, and G units are also supported and are case-insensitive.
- **Upper and Lower Limits**: For `--kpoolsize N`, when the provided N is less than 4 GB, the value defaults to 4 GB. The size must not exceed the `/dev/shm` space limit.

#### 4.9.4 Estimation Mode Limitations

The estimate mode executes on a single chip (rank0) with constructed phantom peers. **The following two prerequisites must be met before using estimate mode**:

1. **Disable link heartbeat and consistency verification**: In estimate mode, no actual peers participate on the link. Link heartbeat and consistency verification cannot complete normally and may cause false positives or abnormal exits.

   ```bash
   export HCCL_DFS_CONFIG=cluster_heartbeat:off,inconsistent_check:off
   ```

   This environment variable must take effect before the business test case starts. Both the `start` and `run` (one-shot) methods apply. When modifying the `start.sh` example from the README, note that the default only disables heartbeat (`cluster_heartbeat:off`). For estimate mode, `inconsistent_check:off` must be added as shown above.

2. **Business test cases must start with a single process (`-np 1`)**: The estimate mode only constructs the rank0 execution path. Regardless of the cluster topology configuration scale, the mpirun command must specify `-np 1`. Other parameters and usage remain unchanged.

   ```bash
   mpirun --allow-run-as-root --oversubscribe -np 1 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt
   ```

Other limitations:

- Only constructs the single-chip (rank0) execution path and skips actual communication blocking points. Used to quickly produce a task list without actually running data transfer.
- When functional correctness verification or actual communication behavior is required, use `normal` mode instead.

#### 4.9.5 Estimation Plugin (@estimate) and Dependency Description

After estimate mode produces a task list, lightweight performance simulation can be performed through the estimation plugin (reads `data/hccl_vm_data.db`, builds a task graph, and performs discrete event simulation):

```bash
(estimate)$> hccl-vm plugin run @estimate
```

This plugin is implemented in Python (installation directory `plugin/light_performance/`, available only in estimate mode, mutually exclusive with @checker). Its runtime dependencies are **not distributed with the installation package** and must be pre-installed in the target environment:

- **python3**: >=3.8 (including the sqlite3 standard library)
- **simpy**: Discrete event simulation engine. Install by running `pip3 install simpy`. When missing, the plugin reports `ModuleNotFoundError: No module named 'simpy'` at startup.

Simulation results are output to the `plugin/light_performance/data/<timestamp>/` directory: `perf_profiling.json` (Chrome tracing format), per-operator trace files, and the runtime log `light_performance.log`.

### 4.10 Simulation Levels and Level 1 Hijacking (Northbound Hijacking)

#### 4.10.1 Northbound Hijacking Mechanism

HCCL-VM uses `LD_PRELOAD` multi-level hijacking to simulate HCCL communication operators: Level 0 hijacks the HCCL operator API; Level 1 hijacks the northbound interface that HCCL uses to call HComm, commonly known as northbound hijacking; Level 2 hijacks the southbound interfaces below HComm (HCCP/Runtime/Driver), commonly known as southbound hijacking. The default level is 2 (southbound hijacking), where operator execution still depends on the complete HComm runtime. When `--level 1` is specified to enable northbound hijacking, the stub function library completely replaces the actual HComm, and task metadata recording replaces actual communication. HCCL operators and the HComm runtime are fully decoupled. Northbound hijacking covers all four expansion modes: AIV, AICPU, CCU, and HostDPU. Checker verification capability also applies.

#### 4.10.2 Compilation Phase: Compiling Only Northbound (Level 1 Trimmed Build)

Run `bash ./build.sh --level1` to compile only northbound-related code (proxy level0/1/2 + support libraries + hccl-vm + device side + aiv_kernel). The hcomm source repository becomes optional (when provided, checker/runner plugins are also built). The `hccl-vm` output supports only `--level 1`. A full build with `--full` supports both `--level 1` and `--level 2`.

Consistent with `--full`, the build automatically deploys AICPU packages from the current CANN after build and installation. No additional `build_pkg.sh` execution is required. The hcomm source repository is optional, but the CANN installation must still contain `aicpu_hccl.tar.gz` and `aicpu_hcomm.tar.gz`.

#### 4.10.3 Runtime Phase: Running Northbound (--level 1)

Prerequisites: `libhccl_proxy_level1.so` must exist under the installation directory `lib/<arch>/` (AICPU mode also requires `lib/aarch64/libhccl_device_proxy_level1.so`). Otherwise, startup reports an error and exits.

The `start`, `run`, and `mock-comm` subcommands all support the `--level 1` parameter. Environment variable configuration, test case execution, and result viewing methods are identical to those for the default level 2.

***

## 5 Appendix

### Open-Source Third-Party Software Dependencies

The following open-source third-party software is required when compiling this project. For offline compilation scenarios, download and rename the packages and place them in the `third_party` directory within this project. For module-specific dependencies and offline placement methods, see the README in the corresponding module directory.

| Open-Source Software | Version | Download Address |
| ------------  | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| CLI11         | 2.2.0         | [cli11-2.2.0.tar.gz](https://raw.gitcode.com/src-openeuler/cli11/blobs/58c912141164a5c0f0139bfa91343fefe151d525/cli11-2.2.0.tar.gz) |
| json          | 3.11.3        | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| spdlog        | 1.11.0        | [spdlog-v1.11.0.tar.gz](https://raw.gitcode.com/src-openeuler/spdlog/blobs/c2dfb1aca26c607393665c836155613ff283de66/v1.11.0.tar.gz) |
| yaml-cpp      | 0.8.0         | [yaml-cpp-0.8.0.tar.gz](https://raw.gitcode.com/src-openeuler/yaml-cpp/blobs/d1ead4fff417073b9cdbf98b8b55eb0efc00b0ba/yaml-cpp-0.8.0.tar.gz) |
| googletest    | 1.14.0        | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) |
| cann-cmake    | master-044    | [cmake-master-044.tar.gz](https://raw.gitcode.com/cann/cmake/archive/refs/heads/master-044.tar.gz) |

### Glossary

| Term | Description |
| -------- | ------------------------------------------------------- |
| HCCL | Huawei Collective Communication Library |
| NPU | Neural Processing Unit |
| CANN | Compute Architecture for Neural Networks, the Huawei Ascend AI processor software stack |
| MPI | Message Passing Interface |
| CCU | Collective Communication Unit |
| Topology | The connection relationships among devices |
| Rank | The process identifier in MPI |

---

**Document Version**: v1.4.
**Last Updated**: 2026-11-12.
