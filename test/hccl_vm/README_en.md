# HCCL-VM User Guide

## 1. Overview

HCCL-VM is a virtual execution environment for high-performance collective communication on Huawei Ascend NPUs. This tool enables the development and functional verification of HCCL collective communication operators without real Ascend hardware.

The tool provides three running modes: `normal` (real multi-card communication, with no loss of result correctness), `check-only` (verification only, where large memory blocks reuse a shared pool to greatly reduce memory usage), and `estimate` (estimation, which skips real communication blocking points and quickly produces a task list). You can specify a mode at startup and switch modes at any time during runtime through the `hccl-vm mode` command. For details, see [Running Modes](#49-running-modes-normal--check-only--estimate).

The tool also provides two simulation levels: the default level 2 (southbound hijacking) and level 1 (northbound hijacking, specified with `--level 1`). For details, see [Simulation Levels and level1 Hijacking](#410-simulation-levels-and-level1-hijacking-northbound-hijacking).

![hccl-vm GIF](docs/hccl-vm.gif)

## 2. Prerequisites

Specification constraint: Ascend950. For other constraints, see [Constraint Details](#45-tool-specification-constraints).

**Compilation environment version requirements**:

| Software | Version Requirement |
| --- | --- |
| OS / architecture | Linux x86_64 / aarch64 |
| CMake | ≥3.16 and <4.0 |
| GCC / G++ | 7.3.0 – 14.2.x (C++17) |
| Python3 | ≥3.7 (pip3 ≥20.3, including PyYAML) |
| CANN Toolkit + 950 ops package | Matched with the same generation of hccl/hcomm source code |
| make / pkg-config / patch | Provided by the OS (pkg-config ≥0.29.1, patch ≥2.7.0) |

**Runtime environment version requirements**:

| Software | Version Requirement |
| --- | --- |
| QEMU (x86_64 host, AICPU/HostDPU modes) | ≥10.2.4 (recommended); not required for compilation. For installation and fallback behavior, see [2.3](#23-aicpu-runtime-environment-installation) |
| MPI (hccl_test) | OpenMPI 4.x or MPICH |
| MariaDB (optional, >128 cards) | 10.11 |
| Python3 (only for the @estimate estimation plugin) | ≥3.8 (including the sqlite3 standard library) + simpy (`pip3 install simpy`). For details, see [4.9.5](#495-estimate-plugin-estimate-and-dependencies) |

### 2.1 CANN Package Installation

Install the latest CANN Toolkit development suite package and CANN ops operator package. [Download link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)

```bash
# Ensure that the installation packages have executable permissions
chmod +x Ascend-cann-toolkit_9.1.0_linux-x86_64.run
chmod +x Ascend-cann-950-ops_9.1.0_linux-x86_64.run
# Installation commands
./Ascend-cann-toolkit_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
./Ascend-cann-950-ops_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
```


### 2.2 hccl_test Compilation

hccl_test is the HCCL performance testing tool officially provided for Ascend. For details, see [HCCL Performance Testing Tool](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/devaids/hccltool/HCCLpertest_16_0001.html). HCCL-VM supports running hccl_test cases in the virtual environment. Compile the case binaries first by following [hccl_test Case Compilation](#42-hccl-test-case-compilation).

NOTE: hccl_test is an optional dependency. To run PyTorch cases, see [PyTorch Example](#33-pytorch-example).

### 2.3 AICPU Runtime Environment Installation

During runtime, HCCL-VM loads and simulates the execution of AICPU device-side symbols. An x86_64 host requires QEMU user-mode emulation (linux-user mode emulating aarch64):

- **Runtime**: QEMU shipped with the OS or installed through apt is preferred (`qemu-user-static` provides `qemu-aarch64-static`). You can also use `HCCL_VM_QEMU_PATH` to specify an existing local binary.
- **Deployment**: After the build is installed, `build.sh --full` and `build.sh --level1` automatically invoke `build_pkg.sh`, which extracts `aicpu_hccl.tar.gz` and `aicpu_hcomm.tar.gz` from the CANN installation directory to deploy the AICPU device-side symbols. You do not need to run it separately, and the build fails if the deployment fails. For the compilation and installation steps, see [3.1 Manual Build and Installation](#31-manual-build-and-installation).

```bash
sudo apt install -y qemu-user-static binfmt-support
```

If the version shipped with the OS is earlier than **10.2.4**, the `-tb-size` option for limiting the translation code cache is unsupported, and the resident QEMU memory usage is high, which may severely degrade performance in large-scale scenarios. To compile version 10.2.4 or later by yourself, see [3.1.1 QEMU Source Code Compilation (Optional, x86_64 Host)](#311-qemu-source-code-compilation-optional-x86_64-host). A native ARM host does not require QEMU.

Runtime behavior and notes:

- When a device process starts, the tool prints the QEMU binary path, version, and `-tb-size` request value actually in use.
- If `HCCL_VM_QEMU_PATH` is unset or invalid, or points to an old QEMU version that does not support `-tb-size`, startup **does not abort**: the tool automatically falls back (searching PATH for `qemu-aarch64-static` / omitting `-tb-size`) and prints a WARN, including the binary and version it actually fell back to, and a note that "the translation cache is unlimited and QEMU memory usage may be excessive". The AICPU runtime environment fails to start only when no `qemu-aarch64-static` can be found on the machine.
- `HCCL_VM_QEMU_TB_SIZE` specifies the QEMU translation code cache (MiB): if unset, the default is `16`; `0` delegates to the QEMU default policy (it does not disable the cache). Leaving the cache unlimited may significantly increase resident QEMU memory and trigger paging.

---

## 3. Quick Start

### One-Click Installation

After pulling the complete CheckerL2 source code, run the following command in the repository root directory:

```bash
bash ./hccl_vm_installer
```

### 3.1 Manual Build and Installation

```bash
# 1. Create a working directory
mkdir -p /home/workspace
cd /home/workspace

# 2. Download the dependency source code
git clone https://gitcode.com/cann/hccl.git
git clone https://gitcode.com/cann/hcomm.git

# 3. Install third-party dependencies
sudo apt-get update
sudo apt install build-essential cmake libboost-all-dev rdma-core libibverbs-dev pkg-config gcc-aarch64-linux-gnu g++-aarch64-linux-gnu qemu-user-static binfmt-support

# 4. Build and install the HCCL-VM tool and automatically deploy the AICPU device-side symbols
# After the hcomm source code is downloaded, the tool source code is located at /home/workspace/hcomm/test/hccl_vm
cd /home/workspace/hcomm/test/hccl_vm
source /home/workspace/Ascend/cann/set_env.sh
export HCCL_CODE_HOME=/home/workspace/hccl
export HCOMM_CODE_HOME=/home/workspace/hcomm
bash ./build.sh --full
```

`--full` already includes the device-side build. You do not need to additionally run `build.sh --aicpu` or `build_pkg.sh`. Automatic deployment uses only the AICPU packages already available in the current CANN installation and does not recompile HCCL/HCOMM. With `--pkg` added, an `hccl_vm_install.tar.gz` containing the AICPU libraries is generated after the deployment. A build without arguments still updates only the host, and `--aicpu` still updates only the device. To refresh the AICPU libraries in the installed CANN separately, run `bash build_pkg.sh`.

#### 3.1.1 QEMU Source Code Compilation (Optional, x86_64 Host)

QEMU shipped with the OS or installed through apt is preferred by default (see [2.3](#23-aicpu-runtime-environment-installation)). To use version 10.2.4 or later to limit the translation code cache and reduce memory usage in large-scale scenarios, compile it as follows:

```bash
# 1) Build dependencies (gcc/make/ninja are usually already installed with the OS or by this tool)
sudo apt-get install -y meson flex bison libglib2.0-dev libpcre2-dev libffi-dev

# 2) Official source archive (10.2.4 is used as an example; version 10.2.4 or later is supported)
curl -fsSL -O https://download.qemu.org/qemu-10.2.4.tar.xz
tar -xf qemu-10.2.4.tar.xz && cd qemu-10.2.4

# 3) Build the static aarch64 linux-user version
mkdir build && cd build
../configure --target-list=aarch64-linux-user --static --disable-system --disable-tools \
    --disable-docs --disable-guest-agent --disable-slirp
make -j$(nproc) qemu-aarch64

# 4) Deploy and configure (the deployment path can be customized; /opt/qemu-10.2.4 is used as an example)
sudo mkdir -p /opt/qemu-10.2.4/bin
sudo cp qemu-aarch64 /opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_TB_SIZE=16
```

### 3.2 Usage Examples

#### 3.2.1 Environment Configuration

If an hccl_rootinfo.json file already exists in the /etc directory, delete it manually.

When the communicator scale exceeds 128 cards (129–1024), switch to the MariaDB database. See [MariaDB Database Usage](src/storage_backends/mariadb/README.md). After the deployment is complete and `HCCL_VM_STORAGE_CONFIG` is set, return to this section. Unsetting this variable restores the default SQLite database.
<!-- internal-only:end -->

#### 3.2.2 CCU Mode

1. Configure environment variables.

```bash
# Enter the tool installation directory
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
```

2. Run


```bash
# Run hccl-vm in the new bin directory
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select an Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# To enable the runner plugin (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for this operator execution (run the hccl_test case in a cluster with 1 supernode, 1 server, and 1 NPU)
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run checker verification
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal
(hvm)$> exit
```

3. Verify the hccl_test case run results
[Runner result viewing](#481-runner-plugin-results)
[Checker result viewing](#482-checker-plugin-results)

#### 3.2.3 AICPU Mode

The AICPU expansion mode moves the algorithm expansion steps to the device side. Therefore, the hccl-vm tool must compile the HCCL device-side symbols and simulate their execution. On an x86_64 host, QEMU is required to simulate the execution of the device-side code. For installation and deployment, see [2.3 AICPU Runtime Environment Installation](#23-aicpu-runtime-environment-installation).

On an x86_64 host, the device-side code is executed through QEMU user-mode emulation. If `HCCL_VM_QEMU_PATH` is not configured, the tool automatically falls back to `qemu-aarch64-static` in PATH and prints a warning (including the actual version), which does not affect startup. When the version shipped with the OS is earlier than 10.2.4, the translation code cache cannot be limited and memory usage is high, which may cause paging and affect performance in large-scale scenarios. Install version 10.2.4 or later and configure `HCCL_VM_QEMU_PATH` as described in [2.3 AICPU Runtime Environment Installation](#23-aicpu-runtime-environment-installation). A native ARM host does not require QEMU.

1. Configure environment variables.

```bash
# Enter the tool installation directory
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AI_CPU"

# x86_64 host: QEMU binary and translation code cache (MiB) configuration. See section 2.3.
export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_TB_SIZE=16
```

2. Run

```bash
# Run hccl-vm in the new bin directory
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select an Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# To enable the runner plugin (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for this operator execution (run the hccl_test case in a cluster with 1 supernode, 1 server, and 1 NPU)
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run checker verification
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal
(hvm)$> exit
```

3. Verify the hccl_test case run results [Runner result viewing](#481-runner-plugin-results) [Checker result viewing](#482-checker-plugin-results)

#### 3.2.4 HostDPU Mode

In HostDPU expansion mode, the runtime environment variable configuration is the same as that in AICPU mode. The only differences are in the cluster scale and communicator scale:

1. For the cluster specification configuration, use `./hccl-vm start ascend950_cluster_32_server_normal_hostdpu.yaml`.
2. The number of supernodes must be >= 2 (for example, 222.yaml). That is, a communicator within a single supernode is not supported (112.yaml, 122.yaml, and so on are not supported).

#### 3.2.5 AIV Mode

1. Configure environment variables.

```bash
# Enter the tool installation directory
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AIV"
```

2. Run

```bash
# Run hccl-vm in the new bin directory
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select an Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# To enable the runner plugin (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for this operator execution (run the hccl_test case in a cluster with 1 supernode, 1 server, and 1 NPU)
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run checker verification
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal
(hvm)$> exit
```

3. Verify the hccl_test case run results [Runner result viewing](#481-runner-plugin-results) [Checker result viewing](#482-checker-plugin-results)

### 3.3 PyTorch Example

HCCL-VM supports running some PyTorch `aclgraph` graph-mode collective communication scripts. Scripts must use `torch.multiprocessing.spawn` to start the device processes. For multi-server cases, you need to concurrently start multiple Python main processes on the same host machine to simulate each server. For the complete description, see the [PyTorch Mode Running Guide](docs/pytorch_guide.md).

#### 3.3.1 Environment Preparation

Complete the HCCL-VM build and installation and the CANN environment configuration first, and prepare a Python virtual environment with compatible versions of `torch` and `torch_npu` installed. PyTorch currently supports only the Ascend950 series. To use the Ascend960 series, install the A6 ops package and change `soc_version` in the topology cluster configuration to a real model such as `Ascend960DT_961BY`.

NOTE: The number of AIV operators supported by the current simulator is limited, and a considerable portion of torch computations cannot complete normally. For example, `print(tensor)` and `tensor.clone()` fail because they invoke unsupported compute operators.

```bash
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin
# Enable this before startup so that the VM main process generates the ranktable by local_rank of each server/board, supporting non-equal card count topologies.
export HCCL_VM_ENABLE_PYTORCH=1
./hccl-vm start ascend950_cluster_32_server_normal.yaml
```

After entering the `(hvm)$>` terminal, run the following commands. Subsequent communicator configuration, environment variable setup, and Python startup are all performed in this terminal so that the Python process inherits the simulated environment.

```bash
# Activate the virtual environment (if any)
source /path/to/torch_test/bin/activate

# Install Runner if simulated execution is required
hccl-vm plugin install @runner
```

#### 3.3.2 Single-Server Run

The following example uses the `118` configuration, with 1 server and 8 devices, to execute AllReduce SUM over 1024 int32 elements. The following command parameters correspond to the test script in the guide. When using other scripts, adjust the parameters according to their definitions.

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

#### 3.3.3 Multi-Server Run

The following example uses the `128` configuration, with 2 servers and 8 devices per server, totaling 16 ranks. Run the following commands in the HCCL-VM terminal where the environment has been prepared:

```bash
hccl-vm mock-comm 128

# Start two servers concurrently, setting environment variables and saving logs for each
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

# Wait until both servers finish running, and check each log for exceptions and execution results
wait
```

`HCCL_VM_SERVER_INDEX` is the zero-based server sequence number in the current communicator configuration. It takes effect only when `HCCL_VM_ENABLE_PYTORCH=1` and must be consistent with `--server_index` of the corresponding script. Setting environment variables inside the parentheses prevents the indexes of different servers from overwriting each other.

All servers must use the same `--master_ip` and `--master_port`; `--dev_num` and `--server_num` must be consistent with the `mock-comm` configuration. The local logical device number of each server is `0–7`. In this example, server 0 corresponds to global ranks `0–7`, and server 1 corresponds to global ranks `8–15`. The script calculates global ranks based on the same number of devices per server. For other scales, choose a communicator configuration where each server has the same number of devices.

> **Correspondence in POD form (4-layer topology)**: The "server" here refers to the **leaf node (host node)** in the communicator.
> In server form, the leaf node is the server; **in POD form, the leaf node is the board**. Therefore,
> `--server_num`/`HCCL_VM_SERVER_INDEX` must be counted and numbered by **board** (for example, the `11_64` configuration is 1 POD × 8 boards × 8 cards,
> which requires concurrently starting 8 Python main processes, with `--server_num 8`, `--dev_num 8`, and `HCCL_VM_SERVER_INDEX` set to `0~7`).
> The local logical device number of each process is `0–7` within the board, consistent with a real machine.

#### 3.3.4 Result Viewing and Verification

After all processes finish, check the corresponding `server0.log` and `server1.log` (multi-server scenario) for exit status, exceptions, and related log output. Then run Checker verification in the HCCL-VM terminal. For result descriptions, see [Checker Plugin Results](#482-checker-plugin-results).

```bash
hccl-vm plugin run @checker

# Exit the tool terminal after all cases and verification are complete
exit
```

### 3.4 Example: Verifying HCCL Code Modifications

If you have modified the CANN operator package code, for example, added an algorithm type, follow these steps to ensure that your modification takes effect. The build_pkg.sh script helps you build packages, install packages, and copy device-side dependency symbols. Set the environment variables before running it:

```bash
# Assume that your CANN installation directory is /home/workspace/Ascend
source /home/workspace/Ascend/cann/set_env.sh
# Configure the hccl repository path
export HCCL_CODE_HOME=/home/workspace/hccl
# Configure the hcomm repository path
export HCOMM_CODE_HOME=/home/workspace/hcomm
```

1. If you have updated or modified the code in the CANN hccl repository, run `bash build_pkg.sh --install hccl`.
2. If you have updated or modified the code in the CANN hcomm repository, run `bash build_pkg.sh --install hcomm`.
3. If you have updated or modified the code in both the CANN hccl and hcomm repositories, run `bash build_pkg.sh --full`.
4. Rerun the cases by following the steps in [Usage Examples](#32-usage-examples).

---

## 4. Detailed Guide

### 4.1 Tool Environment Variables

**HCCL-VM environment variable description**:

| Environment Variable | Purpose | Example |
| --- | --- | --- |
| `HCCL_CODE_HOME` | Specifies the HCCL source code path for HCCL-VM compilation. Not configured by default. | `export HCCL_CODE_HOME=/home/workspace/hccl` |
| `HCOMM_CODE_HOME` | Specifies the HCOMM source code path for HCCL-VM compilation. Not configured by default. | `export HCOMM_CODE_HOME=/home/workspace/hcomm` |
| `HCCL_VM_ENABLE_PYTORCH` | Set to `1` before `hccl-vm start` to enable PyTorch adaptation and generate the ranktable by local_rank of each server/board. For the running method, see [PyTorch Example](#33-pytorch-example). | `export HCCL_VM_ENABLE_PYTORCH=1` |
| `HCCL_VM_SERVER_INDEX` | Zero-based sequence number of the **leaf node (host node)** simulated by the PyTorch process. It takes effect only when `HCCL_VM_ENABLE_PYTORCH=1` and must be consistent with `--server_index` of the test script. In server form, the leaf node is the server; **in POD form (4-layer topology), the leaf node is the board**, so counting and numbering are performed by board. | `export HCCL_VM_SERVER_INDEX=0` |
| `HCCLVM_ENABLE_DUMP_DATA` | 1. Enables writing performance simulation files to disk. If enabled, three runner_hcclvm_*_data.bin files and one ccu_channel_jetty_config.xml file are output during test case execution for performance simulation. <br> 2. Enables the Runner plugin to dump input\&output data. If enabled, the input\&output data of each operator is dumped to the all\_rank\_input\_output.txt file during test case execution. | `export HCCLVM_ENABLE_DUMP_DATA=1 to enable; export HCCLVM_ENABLE_DUMP_DATA=0 or leave this environment variable unset to disable` |
| `HCCL_VM_RES_STAT_DUMP` | Enables writing hijack-layer resource statistics to disk. When set, the business process exports the per-device resource counts (memory, Jetty, QP, Notify, AIV core, and so on) as CSV files during finalize. The files are output to `data/stat/<podId>_<serverId>_<boardId>_<device ID>_res_<timestamp>.csv` in the installation directory (`boardId` is the board sequence number in the 4-layer POD topology and is always `0` in server form). Not exported when unset (default). | `export HCCL_VM_RES_STAT_DUMP=1` |
| `HCCL_VM_QEMU_PATH` | Effective on x86_64 hosts (AICPU runtime): non-empty absolute path to an external QEMU binary (the target must be an executable regular file; symlinks pointing to a regular file are allowed). If unset or invalid, the tool automatically falls back to searching PATH for `qemu-aarch64-static` and prints a WARN (including the actual binary and version); the AICPU runtime environment fails to start only if none can be found on the machine. For deployment steps, see [2.3 AICPU Runtime Environment Installation](#23-aicpu-runtime-environment-installation) | `export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static` |
| `HCCL_VM_QEMU_TB_SIZE` | Effective on x86_64 hosts (AICPU runtime): QEMU translation code cache (MiB). If unset, the default is `16`; `0` explicitly passes `-tb-size 0` and the QEMU default policy applies (it does not disable the cache). Only decimal digit strings parsable to uint32_t are accepted; invalid values trigger a warning and fall back to the default 16. When the value points to an old QEMU version that does not support `-tb-size`, the option is not passed and a warning about potentially excessive memory usage is printed. | `export HCCL_VM_QEMU_TB_SIZE=16` |
| `HCCL_UB_MULTI_CHANNEL_NUM` | HCCL-side environment variable read by the proxy stub (same semantics as the `HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM` query item of `HcclConfigGetInfo`): number of jetties running in parallel on one channel. It is stored to the database with each operator as `opDetails.ubMultiChannelNum`, and the lightweight performance simulation plugin of `@estimate` uses it to convert port bandwidth utilization (each bucket of `multi_port_utilization_low` does `+= (n-1)×jetty_utilization_step`, capped at 1.0; see section 7.2 of `docs/light_performance/simulator_design.md`). If unset, invalid, or `0`, it falls back to `1` (multi-jetty disabled; the original utilization table is used directly). **NOTE**: This value is only the business configuration intent. The jetty count used for conversion is determined first by the **actual link establishment value in the db after the proxy runs** (the `RaJetty` table of `hccl_sim.db` aggregates the number of mode=3 jetties by eid; in CCU mode it is `CcuChannelTab.jettyNum`). The value stored from this environment variable is only a fallback when the actual value is unavailable (for example, an old database). If multi-jetty is configured but links are still established with a single jetty, the actual value 1 is used for conversion and `[JETTY][WARN]` is printed. | `export HCCL_UB_MULTI_CHANNEL_NUM=8` |
`HCCL_VM_LOG_MODE` | Sets the simulator log type: (1) stub (default): the simulator spdlog controls the log level and disk output; (2) cann: reuses the CANN log module, and the log level and disk output are fully controlled by the CANN logging capability. | `export HCCL_VM_LOG_MODE=cann`

### 4.2 HCCL-Test Case Compilation

The hccl_test case source code is in the CANN package installation directory. Both OpenMPI and MPICH environments are supported for compilation and running. For runtime differences, see [Differences Between Running Cases in OpenMPI and MPICH Environments](#47-case-running-differences-between-openmpi-and-mpich-environments). This guide uses the OpenMPI environment as an example.

#### 4.2.1 Compilation in the OpenMPI Environment

1. Install OpenMPI.

```bash
sudo apt-get update
sudo apt install openmpi-bin libopenmpi-dev
```

2. Compile hccl_test.

```bash
# Modify the CANN installation directory permissions
chmod -R 755 /home/workspace/Ascend

# Enter the hccl_test case source code directory
cd /home/workspace/Ascend/cann/tools/hccl_test

# Set the CANN environment variables
source /home/workspace/Ascend/cann/set_env.sh

# Temporarily modify the Makefile script
if ! grep -q '\-lmpi_cxx' Makefile; then
    sed -i 's/-lmpi/-lmpi -lmpi_cxx/g' Makefile
fi

# Compile the hccl_test cases
MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi make ASCEND_DIR=${ASCEND_HOME_PATH}
```

#### 4.2.2 Compilation in the MPICH Environment

Assume that the mpich path is `/usr/lib/mpich`.

```bash
# Enter the hccl_test case source code directory
cd /home/workspace/Ascend/cann/tools/hccl_test

# Set the CANN environment variables
source /home/workspace/Ascend/cann/set_env.sh

# Configure environment variables
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH

# Compile the hccl_test cases
make MPI_HOME=/usr/lib/mpich/ ASCEND_DIR=${ASCEND_HOME_PATH}
```

### 4.3 Ascend Cluster Topology Configuration File Description

#### 4.3.1 Server/Pod Topology Configuration File Description

An Ascend cluster topology is formed by combining one or more Server/Pod sub-topologies according to CLOS layered network rules. Therefore, before generating a cluster topology, you need to determine the topology type of each Server/Pod.
You can select a predefined topology type provided by the HCCL-VM tool, or customize Server/Pod topology types according to the configuration file format requirements.

Describing the topology network relationships of a Server/Pod mainly involves the following aspects:

 - **Port configuration table**: Describes the physical port configuration of an NPU card, such as NPU-to-NPU direct-connect ports (P2P) and NPU out-of-frame ports (P2NET).
 - **Link configuration table**: Describes the connections between all NPU cards within a Server/Pod, such as full mesh.
 - **PortBound**: Describes the binding relationships of certain ports of an NPU card, that is, multiple ports bound into one PortGroup.

```yaml
type: "server_intra_links"
name: "ascend950_links_topo_demo"
description: "Description file of the normal topology connections of the ascend950 chip"

soc_version: "Ascend950"
device_num: 16

device_ports_allocate_map:
 # Port allocation table: 0: not used, 1: device direct connect, 2: device connected to a switch, 3: d2h port
 #                 portId:  0  1  2  3  4  5  6  7  8
    - {die_id: 0, pin_map: [1, 1, 1, 0, 2, 2, 2, 2, 3]} # die0
    - {die_id: 1, pin_map: [0, 0, 0, 0, 0, 0, 0, 0, 0]} # die1

# port_group: describes which ports are merged into one portGroup. Ports in the same portGroup share the same IP address.
port_group:
    - {layer: 0, ports: ["0/4", "0/5", "0/6", "0/7"]}

links:
  # ── Description method: 8 devices per group form a full interconnect ──
  - link_mode: "fullmesh"
    connections:
      # The following example indicates that devices 0, 1, 2, and 3 of die0 are fully connected through the ports of die0.
      - {die_id: 0, devices_range: [0, 3]}
      - {die_id: 0, devices_range: [4, 7]}
      - {die_id: 0, devices_range: [8, 11]}
      - {die_id: 0, devices_range: [12, 15]}
  
  #- link_mode: "enum"
  #  device_to_device_links:
  #    The following example indicates that devices 0 and 1, through the ports of die0, are connected to the die1 ports of devices 1, 3, 5, and 7 respectively.
  #    That is, device0 connects to device1, device3, device5, and device7, and device1 connects to device3, device5, and device7.
  #    - {src_die_id: 1, src_device_id_range: [0, 2], dst_die_id: 1, dst_device_id_range: [1, 3, 5, 7]}

  - link_mode: "enum"
    device_to_switch_links:
      # The following example indicates that devices 0 to 15 connect to the switch through the ports of die0. Combined with portGroup, devices 0 to 15 connect to the switch through portGroup [0/4, 0/5, 0/6, 0/7].
      - {die_id: 0, devices_range: [0, 15]}

```

**Field description**:

 - **soc_version**: Chip model, such as `Ascend950`.
 - **device_num**: Total number of devices, determined by the chip model and topology type. In server form, it is the number of cards in the server; **in POD form, it is the total number of cards in the POD** (for example, 64 for 1dpod64).
 - **board_num**: Optional. Number of boards (compute boards) in a POD. **Trigger field of POD form**: when omitted or set to 1, the topology is in server form (3-layer topology); when greater than 1, it is in POD form (4-layer topology, see the description below). Constraints: `device_num % board_num == 0` and `board_num <= 255`.
 - **device_num_per_board**: Optional. Number of cards per board (usually 8, but not hard-coded). If omitted, it is derived from `device_num / board_num`; when explicitly configured, it must equal the derived value.
 - **form_type**: Optional. Either `"server"` or `"pod"` (case-insensitive). If omitted, it is derived from `board_num`; if explicitly configured and contradicting the derived result, an error is reported.
 - **net_instance_scope**: Optional. Specifies the scope of `net_instance_id` layer by layer. Values are `board`/`pod`/`superpod`/`cluster` (see section 4.3.4 for details). Default derivation rule: in server form, `{layer0: pod, layer1: superpod, layer>=2: cluster}`; in POD form, layer0 is derived based on "whether PEER2PEER in-board direct connections exist at this layer" (with direct connections → `board`; pure switched network → `pod`), and the other layers are the same as in server form.
 - **device_ports_allocate_map**: Port allocation table, describing the port configuration of each die. 1 indicates a device direct-connect port, 2 indicates a device-to-switch port, and 3 indicates a d2h port.
 - **port_group**: Describes which ports are merged into one portGroup. Ports in the same portGroup share the same IP address. For ports that are not configured, each port is a portGroup by default.
 - **links**: Link configuration table, describing the connections between all NPU cards within a Server/Pod and between NPUs and switches.
   - **NPU direct connections**: The tool provides two ways to configure NPU direct connections:
     - **link_mode == "fullmesh"**: Indicates that all devices are fully connected based on the ports of one die. When new typical connection methods are added, new link_mode types can be introduced, such as "ring".
     - **link_mode == "enum"**: Enumeration method. When NPU connections within a Server/Pod are complex, you can describe them by enumerating all link relationships.
   - **NPU-to-switch connections**: You can configure the connections between NPUs and switches through the enumeration method.
   - **device_to_device_links**: Describes the connections between NPUs. The devices at both ends are declared with `src_device_id`/`src_device_id_range` and `dst_device_id`/`dst_device_id_range` (**in-board physical IDs**; the legacy names `src_local_id`/`dst_local_id`/`*_local_id_range` are still compatible).
     In POD form, **`src_board_id`/`dst_board_id`** can be used to declare the boards that both ends belong to (they must be configured together), supporting **cross-board direct connections**.
   - **device_to_switch_links**: Describes the connections between NPUs and switches. In POD form, a single **`board_id`** declares the board that the device belongs to (the dst is the switch, so dst_board_id is not needed).

> **NOTE (POD form)**: Device IDs in `device_group`/`port_list`/`port_group`/`links`
> must be **in-board physical IDs** in POD form (for example, `0~7`); that is, the card structure is described as "a single board". `local_id` is
> converted by the tool as `board_id × cards per board + in-board physical ID`. The board ownership fields of links entries:
> **`connections` of `fullmesh`, `device_to_switch_links`, and `matrix`** use a single **`board_id`** (both ends on the same board);
> **`device_to_device_links`** use **`src_board_id`/`dst_board_id`** (the two ends can be on different boards, that is, cross-board direct connections).
> Entries without board fields apply to all boards (that is, the single-board description is replicated and extended).
> If POD-global IDs are used by mistake (for example, `8` or `63`), the tool reports an error directly (`device id exceeds the value range [0, cards per board - 1]`).

**POD Form (4-Layer Topology) Description**

Topologies are divided into two forms with different numbers of layers:

| Form | Layers | Leaf Node (host node) | Card Physical ID | Artifacts |
|------|------|---------------------|----------|------|
| Server form | server → supernode → cluster (3 layers) | server | `[0, device_num-1]` | One `topo.json`/`hccl_rootinfo.json` per server |
| POD form | **board → pod → supernode → cluster (4 layers)** | **board** | **`[0, cards per board - 1]` within the board** (usually `[0,7]`) | **One `topo.json` per POD** (in the server directory) + **one `hccl_rootinfo.json` per board** |

In POD form, the cards in a POD are divided into several boards (compute boards). Each board is an independent host node (with an independent host IP and independent rootinfo).
**topo.json is a single POD-level file** stored in the server directory, where `local_id` is the **POD-global ID**:
`local_id = board_id × cards per board + in-board physical ID`.
For example, `1dpod64` (a 64-card POD) is configured with `device_num: 64` + `board_num: 8`, and the generated cluster directory is:

```
config/network/cluster/ascend950_cluster_16_server_1dpod64/
├── servers_info.json                # One entry per leaf node (per board): 16 PODs × 8 boards = 128 entries
└── superpod0/
    └── server0/                     # POD 0 (no host IP assigned)
        ├── topo.json                # POD-level topology: 64 cards; local_id = board_id*8 + in-board ID (0~63)
        ├── board0/
        │   └── hccl_rootinfo.json   # 8 cards; device_id/local_id = in-board physical IDs 0~7
        ├── board1/ ...
        └── board7/
```

Configuration example (`asset/cluster_model/config/server_or_pod/ascend950_server_topo_1dpod64.yaml`, in-board direct connections, links described board by board):

```yaml
soc_version: "Ascend950"
device_num: 64                       # Total number of cards in the POD
board_num: 8                         # 8 boards → 8 cards per board
hardware_type: "950D-1D-POD-64P-topo"

# The card structure is described as "a single board", and device IDs are in-board physical IDs (0~7)
device_group:
  - {group_id: 1, device_list: [0, 1, 2, 3]}
  - {group_id: 2, device_list: [4, 5, 6, 7]}
links:
  # links are described board by board: fullmesh entries carry board_id, and d2d entries carry src_board_id/dst_board_id;
  # net_instance_id/topo_instance_id are numbered by board
  - link_mode: "fullmesh"
    connections:
      - {board_id: 0, die_id: 0, net_instance_id: 0, topo_instance_id: 0, device_id: [0, 1, 2, 3]}
      - {board_id: 0, die_id: 1, net_instance_id: 0, topo_instance_id: 0, device_id: [4, 5, 6, 7]}
      - {board_id: 1, die_id: 0, net_instance_id: 1, topo_instance_id: 1, device_id: [0, 1, 2, 3]}
      # ... boards 2 to 7 are similar
  - link_mode: "enum"
    device_to_device_links:
      # In-board direct connection: src_board_id == dst_board_id
      - {src_board_id: 0, net_instance_id: 0, topo_instance_id: 0, src_die_id: 0, src_device_id: [0, 1, 2, 3],
         dst_board_id: 0, dst_die_id: 1, dst_device_id: [4, 5, 6, 7]}
      # ... boards 1 to 7 are similar
```

Configuration example (`ascend950_server_topo_1dpod64.yaml`, 8 boards × 8 cards, POD form, in-board direct connections: within each board, the 4 cards of die0 and the 4 cards of die1 are fully interconnected + cross direct-connected):

```yaml
device_num: 64
board_num: 8                          # 8 boards → 8 cards per board
device_group:
  - {group_id: 1, device_list: [0, 1, 2, 3]}
  - {group_id: 2, device_list: [4, 5, 6, 7]}
links:
  - link_mode: "fullmesh"             # Within each board, the 4 cards of die0 and the 4 cards of die1 are fully interconnected
    connections:
      - {board_id: 0, die_id: 0, net_instance_id: 0, topo_instance_id: 0, device_id: [0, 1, 2, 3]}
      - {board_id: 0, die_id: 1, net_instance_id: 0, topo_instance_id: 0, device_id: [4, 5, 6, 7]}
      # ... boards 1 to 7 are similar
  - link_mode: "enum"
    device_to_device_links:
      # In-board direct connection: within each board, the 4 cards of die0 and the 4 cards of die1 are cross direct-connected
      - {src_board_id: 0, net_instance_id: 0, topo_instance_id: 0, src_die_id: 0, src_device_id: [0, 1, 2, 3],
         dst_board_id: 0, dst_die_id: 1, dst_device_id: [4, 5, 6, 7]}
      # ... boards 1 to 7 are similar
```

Configuration example (`ascend960_pod_topo_clos.yaml`, full switching within the frame, no in-board direct connections; links entries without board_id are replicated and extended to all boards):

```yaml
device_num: 72
board_num: 9                         # 9 boards × 8 cards
net_instance_scope:
  - {layer: 0, scope: "pod"}         # layer0 is the in-frame switching network covering all 72 cards of the POD and cannot be split by board
links:
  - link_mode: "enum"
    device_to_switch_links:
      - {die_id: 0, devices_range: [0, 7]}   # board_id not configured → applies to all 9 boards
```

**Mandatory validation for POD form** (generation fails if any condition is not met):
1. `device_num % board_num == 0`, and the number of cards per board is > 0;
2. All device IDs in the configuration must fall within `[0, cards per board - 1]`; the `board_id` of links entries and the
   `src_board_id`/`dst_board_id` of `device_to_device_links` (if configured) must fall within `[0, board_num - 1]`, and `src_board_id`/`dst_board_id` must be configured together;
3. `board_num <= 255` (capacity of the CC segment of the IP address), and the number of endpoints per board is `<= 255` (capacity of the DD segment);
4. When PEER2PEER direct connections exist at layer0: the scope of **in-board direct connections** (each board forming an independent network) must be `board` (ensuring that the layer0
   `net_instance_id` is unique per board; otherwise, the in-board direct connection groups of multiple boards would be incorrectly merged on the business side); the scope of **cross-board direct connections**
   (edges where `src_board_id ≠ dst_board_id` exist; the direct connection domain is a single POD-level instance) must be `pod`
   (otherwise, the `net_instance_id` values at the two ends of a cross-board direct connection edge would be inconsistent, and the edge would be filtered out and become ineffective on the business side);
5. When PEER2PEER direct connections exist, the number of direct connection edges of each board must be identical and each board must have direct connection entries (isomorphism check; a cross-board edge is counted once for each of the boards at both ends);
6. The layers declared in `net_instance_scope` must actually exist in the topology.

> `fullmesh`/`device_to_switch_links`/`matrix` entries use a single `board_id` (both ends on the same board); `device_to_device_links`
> uses `src_board_id`/`dst_board_id` to declare the boards of both ends, supporting **cross-board PEER2PEER direct connections** (for example, a full cross connection between board0 and board1 requires `src_board_id ≠ dst_board_id`).
> In ranktable.json, `device_id` is the in-board physical ID and `local_id` is the POD-global ID (consistent with the POD-level topo.json).

#### 4.3.2 Cluster Topology Configuration File Description

An Ascend cluster network is formed by combining one or more Server/Pod sub-topologies according to CLOS layered network rules. You can select different Server/Pod topology types based on the cluster scale and requirements.

Customize a cluster topology configuration file in the following format:

```yaml
name: "ascend950_cluster_32_server_normal"
description: "Ascend 950 normal networking: 32 supernodes, one server per supernode"

# Total number of supernodes
super_node_num: 4
# Total number of servers/pods
server_num: 32
server_list:
  # Servers 0-7: all use the ascend950_server_topo_normal topology type
  - {super_pod_id: 0, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 1, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 2, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 3, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}

```

The configuration file above describes a cluster topology with 4 supernodes, 32 servers, and 128 NPU cards in total, where each Server/Pod uses the ascend950_server_topo_normal topology type.

**Field description**:

 - **super_node_num**: Total number of supernodes.
 - **server_num**: Total number of servers/pods.
 - **server_list**: Configuration information of each Server/Pod, including the supernode ID, device ID range, chip model, and Server/Pod topology configuration file path.

> **Scale conversion in POD form**: `server_num` counts the number of servers/PODs. If the topology specified by `server_topo` declares
> `board_num > 1` (POD form), then the total number of **leaf nodes (host nodes)** in the cluster = server_num × board_num,
> and the total number of cards = number of leaf nodes × cards per board. For example, `ascend950_cluster_16_server_1dpod64` (`server_num: 16`,
> topology `board_num: 8`, 8 cards per board) has 128 leaf nodes and 1024 cards in total, and `servers_info.json` has 128 entries.
>
> Each entry in `servers_info.json` corresponds to a leaf node and carries an explicit `super_pod_id`/`server_id`/`board_id`
> triple (the tool matches by ID, not by entry position). The host IP rules are as follows:
> in server form, `AA.BB.0.0` (AA = supernode ID + 1, BB = server ID + 1); in POD form, `AA.BB.CC.0` (CC = board_id).

#### 4.3.3 Communicator Configuration File Description

In an Ascend cluster environment, you need to select different communicator configuration files based on the communicator required by the operator to be executed.

The tool provides the hccl-vm mock-comm command to read and configure operator communicator configuration files. A communicator configuration file is in YAML format and is located in hccl_vm_install/config/topo_meta. If the directory does not contain the required communicator configuration file, you need to create one first.

The hccl-vm tool supports asymmetric topology communicator configurations, as shown below:

```yaml
# 1. Global statistics: podNum, serNum, and rankNum are all less than 1024
meta:
  podNum: 1  # Total number of supernodes
  serNum: 2  # Total number of servers
  rankNum: 6 # Total number of ranks

# 2. Detailed topology structure
topology:
  - podId: 0
    servers:
      - serId: 0
        # Local IDs of the ranks actually run on each server
        ranks: [0, 2]
      - serId: 1
        # Local IDs of the ranks actually run on each server
        ranks: [1, 3, 5, 7]
```

**Notes**:

- When you configure a communicator, the tool regenerates the topo.json and ranktable.json files based on the specified communicator configuration number.
- In the communicator configuration YAML file above, the ranks field lists the local IDs (that is, the physical IDs of the devices) of the ranks actually run on each server.

When the communicator scale is large, the server selection fields and rank selection fields in servers entries can be freely combined. The following fields are supported:

- **Server selection fields** (choose one; they cannot be configured together):
  - `serId`: ID of a single server or a list of server IDs (for example, `3` or `[0, 1]`; a list indicates each listed server);
  - `serId_range`: Server ID range, in the form of `[<start id>, <end id>]` (closed interval, list form only), indicating all servers with IDs from the start to the end within the supernode.
- **Rank selection fields** (choose one; they cannot be configured together):
  - `ranks`: List of local IDs of the ranks actually run on each server;
  - `ranks_range`: Local ID range, in the form of `[<start id>, <end id>]` (closed interval, list form only), indicating that each server selects all local IDs from the start to the end.

The following example (2 supernodes, 4 servers in total, 32 ranks) mixes multiple combinations in the same file:

```yaml
# 1. Global statistics: podNum, serNum, and rankNum are all less than 1024
meta:
  podNum: 2  # Total number of supernodes
  serNum: 4  # Total number of servers
  rankNum: 32 # Total number of ranks

# 2. Detailed topology structure
topology:
  - podId: 0
    servers:
      # serId is an ID list, indicating server 0 and server 1 under pod 0
      - serId: [0, 1]
        ranks: [0, 1, 2, 3, 4, 5, 6, 7]
  - podId: 1
    servers:
      - serId: 0
        ranks_range: [0, 7]
      - serId: 1
        ranks_range: [0, 7]
```

**Notes**:

- serId and serId_range cannot be configured together, and ranks and ranks_range cannot be configured together.
- serId_range and ranks_range support only the `[<start id>, <end id>]` list form (closed interval); the `<start id>-<end id>` format is not supported. The start ID cannot be greater than the end ID, and the range width (end ID - start ID + 1) cannot exceed 1024.
- A serId_range entry expands to all servers with IDs from the start to the end within the supernode. A serId list entry indicates each listed server, and each server has the same ranks.
- If no rank selection field is configured in a server entry, the ranks of the server are an empty list; if no server selection field is configured, serId defaults to 0.
- The above notation applies to **server form** (3-layer topology): `ranks` are in-server physical IDs, and one server is one leaf node.

**boards Configuration in POD Form (4-Layer Topology)**

In POD form, the leaf node is the board (not the POD). Therefore, the communicator must declare the participating cards by board,
and `ranks` uses **in-board physical IDs** (for example, `[0, 7]`, not `[0, 63]` within the POD):

```yaml
# 1. Global statistics: podNum, serNum, boardNum, and rankNum are all less than 1024
#    (serNum/boardNum/rankNum are cluster-wide totals)
meta:
  podNum: 1        # Total number of supernodes
  serNum: 1        # Total number of servers/PODs
  boardNum: 8      # Total number of boards (cluster-wide)
  rankNum: 64      # Total number of ranks

# 2. Detailed topology structure
topology:
  - podId: 0
    servers:
      - serId: 0
        boards:
          # 8 boards; each board takes cards with physical IDs 0~7 → 64 ranks in total
          - {boardId_range: [0, 7], ranks_range: [0, 7]}
```

- **Board selection fields** (choose one; they cannot be configured together):
  - `boardId`: ID of a single board or a list of board IDs (for example, `3` or `[0, 1]`);
  - `boardId_range`: Board ID range, `[<start id>, <end id>]` (closed interval, width not exceeding 1024).
- **Rank selection fields**: `ranks`/`ranks_range`, with the same semantics as above, except that the value range is **in-board physical IDs**.
- `boards` is an optional field: **when not configured, it is treated as a single implicit `board0` under the server**,
  and `ranks`/`ranks_range` can be written directly in the server entry (that is, the existing server-form notation, fully backward compatible).
- `boards` and `ranks`/`ranks_range` on the server entry are **mutually exclusive**; configuring both causes an error (rank ownership cannot be determined).
- When `boardId`/`boardId_range` is not configured, the entry belongs to `board0` by default.
- rank_id allocation order: supernode → server (POD) → board → device, incrementing from 0.

POD-form communicator configurations provided by the tool (`asset/cluster_model/topo_meta/`):

| No. | Scale | Description |
|------|------|------|
| `118` | 1 supernode × 1 POD × 1 board × 8 cards = 8 ranks | Single board (server-form notation; in POD form it is board0) |
| `11b2` | 1 × 1 × 2 boards × 8 cards = 16 ranks | 2 boards within a single POD |
| `11b4` | 1 × 1 × 4 boards × 8 cards = 32 ranks | 4 boards within a single POD |
| `11b9` | 1 × 1 × 9 boards × 8 cards = 72 ranks | Entire POD of POD72 (ascend960_pod_topo_clos) |
| `11_64` | 1 × 1 × 8 boards × 8 cards = 64 ranks | Entire 1dpod64 POD |
| `12_64` / `14_64` / `18_64` / `116_64` | 2 / 4 / 8 / 16 PODs, each POD with 8 boards × 8 cards | 128 / 256 / 512 / 1024 ranks |

#### 4.3.4 topo.json and ranktable.json File Description

You do not need to manually create the `topo.json` and `ranktable.json` files. The tool automatically generates them based on the following information:

- **Topology configuration number**: The number specified by the user at startup (for example, 112 and 113).
- **Chip type**: The chip type automatically identified based on the runtime environment.

Although the configuration files are automatically generated by the tool, understanding their structure helps you understand topology configuration.

**topo.json structure**:

`topo.json` describes the connections of all devices within a server:

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

**Field description**:

- `server.device_count`: Total number of devices.
- `server.groups`: Device grouping information.
- `ports`: Port configuration.
  - `usage`: Port usage (`peer2peer` indicates a connection between devices, and `peer2net` indicates a connection to the outside)
- `links`: Link configuration.
  - `link_type`: Link type (`PEER2PEER` or `PEER2NET`)
  - `topo_type`: Topology type (`1DMESH`, `CLOS`, and so on)

**ranktable.json structure**:

`ranktable.json` describes the mapping between the devices used by the current communicator and their addresses (EIDs) at each network layer. The tool automatically generates it in `hccl_vm_install/data/ranktable.json` based on the communicator configuration during `mock-comm`:

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

**Field description**:

- `rank_count`: Total number of ranks in the current communicator.
- `rank_id`: Global rank number within the communicator, incremented from 0 in the order of **supernode → server (POD) → board → device**.
- `device_id`: **Physical ID within the leaf node** (physical ID, not logical ID). That is, the physical device number selected by `aclrtSetDevice()` on the business side.
  hcomm's `CheckRankGraphAddrs` filters entries by `rank.deviceId == devPhyId` and queries the hardware EID with this phyId.
  In server form, the value range is `[0, number of cards in the server - 1]`; **in POD form, the value range is `[0, number of cards in the board - 1]` (usually `[0,7]`)**.
- `local_id`: Same definition as `local_id` in `topo.json`. The business side matches edges in topo.json by the `local_id` in the ranktable.
  In server form, it is the same value as `device_id` (`logic_id == physical_id` in this tool); **in POD form, topo.json is a single POD-level file
  (in the server directory), and `local_id` is the POD-global ID = `board_id × cards per board + in-board physical ID** (for example, the value range for 1dpod64 is `[0,63]`),
  while `device_id` remains the in-board physical ID.
- `level_list[].net_layer`: Network layer number (0 is the intra-node direct connection layer, 1 is the out-of-node/out-of-frame layer, 2 and above are outer layers, and the d2h/HOST layer is usually 3).
  **After the board layer is introduced, the net_layer numbering rule remains unchanged**; the board dimension is expressed through `net_instance_id`.
- `level_list[].net_instance_id`: Identifier of the network instance at this layer, assembled by **scope**:

  | Scope | Format | Typical Use |
  |--------|------|----------|
  | `board` | `sp_{supernode}_srv_{POD}_board_{board}_idx_{k}` | POD form, layer0 (in-board direct connections, **unique per board**) |
  | `pod` | `sp_{supernode}_srv_{POD}_idx_{k}` | Server form, layer0; in-frame switching network across boards in POD form |
  | `superpod` | `sp_{supernode}_idx_{k}` | layer1 (across servers/PODs within the same supernode, out-of-frame interworking) |
  | `cluster` | `cluster` | layer≥2 (cluster level, HOST/d2h layer) |

- `level_list[].net_type`: `TOPO_FILE_DESC` when `net_layer=0`, and `CLOS` when `net_layer>=1`.
- `level_list[].rank_addr_list[].addr`: EID bound to the port at this layer (with the `0x` prefix removed); the UBOE protocol layer is dotted IPv4 (`addr_type: "IPV4"`).
- `level_list[].rank_addr_list[].plane_id`: Fixed `plane_0` for the PEER2PEER layer; the PEER2NET layer inherits `plane_pg_{N}` from the configuration.

> **Layer pruning rule**: For a layer to appear in the ranktable, the layer must have EIDs assigned in rootinfo.
> When the communicator involves only **1 leaf node** (in server form = 1 server; in POD form = 1 board) and does not cross supernodes, `net_layer>=3` layers are not output.
> When the total number of net_layer layers of a single leaf node is > 3, only the 3 layers with the smallest numbers are retained. The preceding pruning is not performed for communicators that cross supernodes.

### 4.4 hccl_config.sh File Description

The hccl_config.sh file contains the environment variable configuration required for running HCCL_Test cases. The environment variables are the same as those used by Hccl_Test cases in a real machine environment.
Modify the hccl_config.sh script based on your own cases and requirements to configure the HCCL case running environment variables.

```bash
#!/bin/bash
# hccl_config.sh - HCCL environment variable configuration

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

  # Always return 0 whether or not files were deleted, so that the script continues
  return 0
}

# Clean up redundant files in the data/ directory (temporary files generated in CCU mode)
cd "${HCCL_VM_INSTALL_DIR}/data" 2>/dev/null && {
  remove_files_by_prefix "sqe_info_rank_"
  remove_files_by_prefix "mc_instr_info_rank_"
  rm -f "all_rank_input_output.txt"
  cd "${HCCL_VM_INSTALL_DIR}"
}

# Set CANN environment variables
source /home/workspace/Ascend/cann/set_env.sh

# Disable the HCCL heartbeat function
export HCCL_DFS_CONFIG=cluster_heartbeat:off

# Set the HCCL-VM installation path, inferred from the script location (compatible with bin/ and script/ subdirectories)
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

# Configure LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH

# Set the ranktable.json file path (consistent with the mock-comm generation path)
export RANK_TABLE_FILE=${HCCL_VM_INSTALL_DIR}/data/ranktable.json

# Set the log level
export ASCEND_GLOBAL_LOG_LEVEL=1

# Set log stdout output
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# Set the HCCL running mode (CCU, AI_CPU, AIV, and other expansion modes)
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
# Or set the HCCL running parameter (AI_CPU expansion mode). The AI_CPU mode environment variable cannot be set together with other modes.
# export HCCL_OP_EXPANSION_MODE="AI_CPU"
# Or set the HCCL running parameter (AIV expansion mode). The AIV mode environment variable cannot be set together with other modes.
# export HCCL_OP_EXPANSION_MODE="AIV"

echo "HCCL-VM environment configured successfully!"

```

### 4.5 Tool Specification Constraints

**Supported operator types**:

The supported operator types include: allgather/allreduce/alltoall/reduce/reduce_scatter/scatter/alltoallv.

**Supported data types**:

The data types supported by the HCCL-VM tool include: int8/int16/int32/fp16/fp32/uint8/uint16/uint32/bfp16/hif8/fp8e4m3/fp8e5m2/fp8e8m0.

The data types supported by the HCCL-VM Runner plugin are as follows:

| ReduceOp | DataType                           |
| -------- | ---------------------------------- |
| `ADD`    | `int8/int16/int32/uint8` |
| `MIN`    | `int8/int16/int32/uint8` |
| `MAX`    | `int8/int16/int32/uint8` |

**Hardware specifications**:

This tool currently supports the Ascend950 and Ascend960 chips. The built-in default storage applies to communicators of 1–128 cards, and no database configuration is required.

### 4.6 HCCL-VM Plugin Features

#### 4.6.1 Runner Plugin

The Runner plugin simulates the execution of the tasks generated by HCCL business orchestration and outputs the output data.
The simulated runner plugin is **disabled** by default during hccl_test case execution. After an hccl_test case calls the operator interface, it waits for the operator task to complete through the aclrtSynchronizeStream interface. The simulated runner waits until all ranks are in the waiting state, and then starts to simulate the execution of the tasks of all ranks. After the execution is complete, it notifies the cases of each rank to continue.
After the cases finish running, you can check the input buffer and output buffer data of each rank in the all\_rank\_input\_output.txt file in the execution directory. This function is disabled by default and can be enabled through the corresponding command before the cases run.

**Installation and uninstallation**:

The Runner plugin is installed and uninstalled through the `hccl-vm plugin install/uninstall` commands. Install the runner plugin after entering the hccl-vm tool command line and before running cases, so that subsequent executions run the runner.

```bash
# Install the runner plugin
(hvm)$> hccl-vm plugin install @runner

# Uninstall the runner plugin
(hvm)$> hccl-vm plugin uninstall @runner
```

#### 4.6.2 Checker Plugin

The Checker plugin, that is, the algorithm analyzer plugin, forms all tasks generated by hccl into a DAG and checks for memory conflicts by analyzing the DAG; it checks for semantic errors and other problems by simulating the execution of the DAG.

The algorithm analyzer plugin is started by the user through commands.

The Checker plugin currently uses Checker V3 for single-operator verification. Checker V3 big-graph verification is enabled by default: multiple operators within each sync window are merged into one big graph, and cross-operator synchronous resource conflict verification is performed. Single-operator Checker V3 and big-graph verification can be controlled separately through configuration parameters in `manifest.json`.

```json

# Configuration file location: /pathto/hccl_vm_install/plugin/checker/manifest.json

{
  "name": "checker",        // Checker plugin name
  "version": "1.0.0",       // Checker plugin version
  "entry": "./checker",     // Checker plugin startup command
  "dependency": {
      "min_core_version": "1.0.0"
  },
  "setting": {              // Checker plugin configuration items
      "enable_new_checker": true,           // Whether to enable Checker V3 single-operator verification (enabled by default)
      "enable_big_graph_checker": true,     // Whether to enable Checker V3 big-graph verification (enabled by default, independent of single-operator verification)
      "enable_insight_dump": false,         // Whether to output Checker V3 Insight data (disabled by default)
      "enable_memory_snapshot_dump": false, // Whether to output memory snapshot data (disabled by default; requires Insight data output to be enabled first)
      "enable_dag_graphviz_dump": false,     // Whether to output the big-graph Graphviz DOT file (disabled by default)
      "enable_ccu_resource_stats_dump": false // Whether to output CCU resource statistics (disabled by default)
  }
}
```

### 4.7 Case Running Differences Between OpenMPI and MPICH Environments

Before running hccl_test cases, you can use the which command to determine which mpirun the current environment uses.

#### 4.7.1 Environment Variable Configuration Differences

OpenMPI is generally configured by default in the environment. If you run cases with OpenMPI, you usually do not need to configure additional environment variables.
If you run cases in the MPICH environment, configure the environment variables as follows:

```bash
# Configure mpich environment variables
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH
export PATH=/usr/lib/mpich/bin:$PATH
```

#### 4.7.2 mpirun Command Parameter Differences

In the OpenMPI environment, run hccl_test cases with the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun --allow-run-as-root --oversubscribe -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter description**:

 - --allow-run-as-root: an OpenMPI-specific parameter that allows MPI processes to run as the root user, used in environments without root permissions.
 - --oversubscribe: an OpenMPI-specific parameter that removes the CPU slot limit, allowing a single node to start [number of processes > number of logical CPU cores], that is, running processes in an oversubscribed manner.
 - -np 2: specifies that the number of processes is 2, consistent with the number of nodes.

In the MPICH environment, run hccl_test cases with the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter description**:

 - -np 2: specifies that the number of processes is 2, consistent with the number of nodes.

### 4.8 Result Viewing

#### 4.8.1 Runner Plugin Results

If the plugin is installed by running hccl-vm plugin install @runner in the hccl-vm terminal, the runner plugin is automatically triggered after the operator process finishes. The final result is verified by hccl_test. Check the redirected log file for [error]-level logs and the final verification result:

```bash
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
64                | 1000.00        | 0.00006              | success
```

#### 4.8.2 Checker Plugin Results

After you run hccl-vm plugin run @checker in the hccl-vm terminal, the Checker verification process and results are printed in the terminal. Check the log file for [error]-level logs and the final verification result:

```bash
[HCCL_VL-CHECKER][CHECKER_RUN_SUMMARY] All Success (Total Op: 4, Big Grapth: 1)
```
---
### 4.9 Running Modes (normal / check-only / estimate)

HCCL-VM has three running modes. At any time, only one of them is in effect, and the default is `normal`:

| Mode | Positioning | Description |
| ---- | ---- | ---- |
| `normal` | Normal mode (default) | Real independent allocation and complete multi-rank communication, with no loss of result correctness |
| `check-only` | Verification-only mode | Large memory blocks reuse a 4GB shared pool, greatly reducing memory usage; only the Checker verification path is guaranteed, and the buffer data content is not guaranteed to be correct |
| `estimate` | Estimation mode | A single card (rank0) constructs phantom peers to skip real communication blocking points and quickly produces a task list, without actually running data transmission (link heartbeat and consistency verification must be disabled first; see [4.9.4](#494-estimate-mode-limitations)) |

#### 4.9.1 Specifying a Mode at Startup

The `start` and `run` (one-shot) subcommands both support the `--check-only` / `--estimate` flags (mutually exclusive). Without any flag, the mode is `normal`:

```bash
# Start in verification-only mode
./hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only

# Start in estimation mode
./hccl-vm start ascend950_cluster_32_server_normal.yaml --estimate
```

#### 4.9.2 Querying and Switching Modes at Runtime

In a started VM command line, you can use `hccl-vm mode` to query and switch modes at any time, without restarting the VM:

```bash
# Query: list all supported modes and the current mode (no argument is equivalent to show, with no side effects)
(hvm)$> hccl-vm mode show
Supported modes:
        check-only : verification-only mode; large memory blocks reuse the 4GB shared pool HcclCommPool, greatly reducing /dev/shm usage; only the verification path is guaranteed, and the buffer data content is not guaranteed to be correct
        estimate : estimation mode; a single card (rank0) constructs phantom peers to skip real communication blocking points and quickly produces a task list, without actually running data transmission
        normal : normal mode (default); real independent allocation and complete multi-rank communication, with no loss of result correctness
Current mode: normal

# Switch modes
(hvm)$> hccl-vm mode check-only
(chk-only)$> hccl-vm mode estimate
(estimate)$> hccl-vm mode normal
```

After a successful switch, the terminal prompt changes with the mode: normal → `(hvm)$>`, check-only → `(chk-only)$>`, estimate → `(estimate)$>`.

**Switching behavior and constraints**:

- **Strict spelling**: Mode names must be spelled strictly in lowercase hyphenated form. Variants such as `CHECK-ONLY` and `check_only` and invalid strings are rejected, and the current mode is not affected.
- **Idempotent**: Switching to the mode you are already in is a no-op, with a message indicating that you are already in that mode.
- **The reuse pool is created/destroyed with the mode**: Switching to check-only automatically creates the 4GB shared reuse pool (residual data is cleaned up before creation), and switching away automatically releases it; the pool does not exist in normal / estimate mode.
- **Uninstall runner before switching**: Before switching to a non-normal mode, if the runner plugin is resident, it is uninstalled first; if the uninstallation fails, the switch is aborted. Switching to normal does not trigger the uninstallation.
- **Switching atomicity**: If any step fails during the switch (pool creation failure, pool destruction failure, and so on), an error is reported and the switch is aborted, and the mode and resources remain as they were before the switch. Known limitation: the prompt may still change after a failed switch; in this case, use the query result of `hccl-vm mode show` as the reference.
- **Effective for new cases**: The new mode takes effect for business cases started after the switch succeeds; cases that were already running before the switch are not affected and continue to run in the mode that was active when they started.
- **Not retained across restarts**: The mode is not retained across VM restarts. After a restart, the initial mode determined by the startup parameters applies (normal if unspecified).

#### 4.9.3 Large Memory Block Reuse in check-only Mode

The check-only mode is used in scenarios where large-scale clusters run only Checker verification. When enabled, each large memory block request reuses the same shared region `HcclCommPool`; the region is shared by all ranks and mutual overwriting is allowed, which greatly reduces `/dev/shm` usage. In this case, the content of large blocks is not guaranteed to be correct. This applies only to the Checker V3 verification path, which does not read buffer data. Do not enable this mode when numerically correct results are required. The default reuse region range is 200 MB - 4 GB, and the upper limit of 4 GB can be changed with the command "--kpoolsize N".

Requests smaller than 200 MB always use real allocation; in check-only mode, a block larger than the upper limit of the reuse region is directly rejected with an error. The check-only mode is not mutually exclusive with Runner, but if Runner is installed while check-only mode is enabled, large block reuse still takes effect and may overwrite Runner data, and the tool prints a warning.

**Reuse region size configuration**
`--kpoolsize N` supports three entries. The following example sets the reuse pool size to 5G:

```bash
# 1) Specified at startup (separate style)
./bin/hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only --kpoolsize 5G

# 2) At runtime, switch/reconfigure in the (chk-only)$> subshell
(chk-only)$> hccl-vm mode check-only --kpoolsize 5G

# 3) At runtime, switch/reconfigure from a subshell of another mode
(hvm)$>  hccl-vm mode check-only --kpoolsize 5G
```
**Setting behavior and constraints:**
- **Strict spelling**: The option name must be spelled exactly as --kpoolsize. Uppercase letters or extra hyphens are not allowed, for example, "--Kpoolsize" and "k_poolsize".
- **Idempotent**: The reuse region size can also be changed while already in check-only mode.
- **Notation**: At startup, the "--check-only" mode must be selected before the size can be set; at runtime, the command must also follow "mode check-only" before --kpoolsize can be configured, and it cannot be used alone.
- **Units**: Pure decimal numbers are supported, for example, `--kpoolsize 2147483648`. The K, M, and G units are also supported and are case-insensitive.
- **Upper and lower limits**: For "--kpoolsize N", when the specified N is less than 4G, it defaults to 4G. The size cannot exceed the `/dev/shm` capacity.

#### 4.9.4 estimate Mode Limitations

In estimate mode, execution is performed by a single card (rank0), and the peer is a constructed phantom peer. **Before using estimate mode, you must meet the following two prerequisites**:

1. **Disable link heartbeat and consistency verification**: In estimate mode, no real peer participates on the link, so link heartbeat and consistency verification cannot complete normally, which would cause false alarms or abnormal exit.

   ```bash
   export HCCL_DFS_CONFIG=cluster_heartbeat:off,inconsistent_check:off
   ```

   This environment variable must take effect before the business cases start. It applies to both the `start` and `run` (one-shot) methods. If you adapt the `start.sh` example in the README, note that it disables only the heartbeat by default (`cluster_heartbeat:off`); when used for estimate mode, add `inconsistent_check:off` as shown above.

2. **Business cases must be started with a single process (`-np 1`)**: estimate mode constructs only the rank0 execution path. Regardless of the cluster topology scale, the mpirun command must specify `-np 1`, and the other parameters and usage remain unchanged.

   ```bash
   mpirun --allow-run-as-root --oversubscribe -np 1 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt
   ```

Other limitations:

- Only the single-card (rank0) execution path is constructed; real communication blocking points are skipped. This mode is used to quickly produce a task list and does not actually run data transmission.
- When functional correctness verification or real communication behavior is required, use the `normal` mode instead.

#### 4.9.5 estimate Plugin (@estimate) and Dependencies

After estimate mode produces the task list, you can run lightweight performance simulation with the estimation plugin (reading `data/hccl_vm_data.db`, building the task graph, and performing discrete event simulation):

```bash
(estimate)$> hccl-vm plugin run @estimate
```

This plugin is implemented in Python (installation directory `plugin/light_performance/`; available only in estimate mode and mutually exclusive with @checker). Its runtime dependencies are **not distributed with the installation package** and must be preinstalled in the target environment:

- **python3**: ≥3.8 (including the sqlite3 standard library);
- **simpy**: discrete event simulation engine, installed with `pip3 install simpy`; if missing, the plugin reports `ModuleNotFoundError: No module named 'simpy'` immediately at startup.

Simulation results are output to the `plugin/light_performance/data/<timestamp>/` directory: `perf_profiling.json` (Chrome tracing format), single-operator trace files, and the runtime log `light_performance.log`.

### 4.10 Simulation Levels and level1 Hijacking (Northbound Hijacking)

#### 4.10.1 Northbound Hijacking Mechanism

HCCL-VM simulates HCCL communication operators through multi-layer `LD_PRELOAD` hijacking: Level 0 hijacks the HCCL operator APIs; Level 1 hijacks the northbound interfaces through which HCCL calls HComm, commonly known as northbound hijacking; Level 2 hijacks the southbound interfaces below HComm (HCCP/Runtime/Driver), commonly known as southbound hijacking. The default level is 2 (southbound hijacking), where operator execution still depends on the complete HComm runtime; after `--level 1` is specified to enable northbound hijacking, the stub function library replaces real HComm entirely and records task metadata instead of performing real communication, fully decoupling HCCL operators from the HComm runtime. Northbound hijacking covers all four expansion modes: AIV, AICPU, CCU, and HostDPU, and the Checker verification capability also applies.

#### 4.10.2 Build Time: Northbound-Only Compilation (level1 Tailored Build)

Running `bash ./build.sh --level1` compiles only the northbound-related code (proxy level0/1/2 + supporting libraries + hccl-vm + device side + aiv_kernel), and the hcomm source repository becomes optional (when provided, the checker/runner plugins are built as well). The resulting `hccl-vm` supports only `--level 1`; full builds such as `--full` support `--level 1/2`.

Consistent with `--full`, the AICPU packages are automatically deployed from the current CANN after the build is installed, and you do not need to additionally run `build_pkg.sh`; the hcomm source repository is optional, but CANN must still contain `aicpu_hccl.tar.gz` and `aicpu_hcomm.tar.gz`.

#### 4.10.3 Runtime: Running Northbound (--level 1)

Prerequisite: `libhccl_proxy_level1.so` must exist in the `lib/<arch>/` directory of the installation directory (in AICPU mode, `lib/aarch64/libhccl_device_proxy_level1.so` is also required); otherwise, startup exits with an error.

The `start`, `run`, and `mock-comm` subcommands all support the `--level 1` parameter; the environment variable configuration, case execution, and result viewing methods are identical to those of the default level 2.

***

## 5 Appendix

### Open-Source Third-Party Software Dependencies

The following lists the third-party open-source software required to compile this project. In offline compilation scenarios, download the archives from the download addresses, and place or rename them in the third_party/ directory of the repository root according to the link text in the table (that is, the archive name recognized by the build). For the specific dependencies of each storage backend module and their offline placement, see the README in the corresponding module directory.

| Open-Source Software | Version | Download Address |
| ------------  | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| CLI11         | 2.2.0         | [cli11-2.2.0.tar.gz](https://raw.gitcode.com/src-openeuler/cli11/blobs/58c912141164a5c0f0139bfa91343fefe151d525/cli11-2.2.0.tar.gz) |
| json          | 3.11.3        | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| spdlog        | 1.11.0        | [spdlog-v1.11.0.tar.gz](https://raw.gitcode.com/src-openeuler/spdlog/blobs/c2dfb1aca26c607393665c836155613ff283de66/v1.11.0.tar.gz) |
| yaml-cpp      | 0.8.0         | [yaml-cpp-0.8.0.tar.gz](https://raw.gitcode.com/src-openeuler/yaml-cpp/blobs/d1ead4fff417073b9cdbf98b8b55eb0efc00b0ba/yaml-cpp-0.8.0.tar.gz) |
| googletest    | 1.14.0        | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) |
| cann-cmake    | master-044    | [cmake-master-044.tar.gz](https://raw.gitcode.com/cann/cmake/archive/refs/heads/master-044.tar.gz) |
| SQLite        | 3.51.3        | [sqlite-amalgamation-3510300.zip](https://www.sqlite.org/2026/sqlite-amalgamation-3510300.zip) |
| MariaDB Connector/C | 3.3.8  | [mariadb-connector-c-3.3.8-src.tar.gz](https://raw.gitcode.com/src-openeuler/mariadb-connector-c/blobs/9389ee06248685f68fe17b9d18a90c990e039020/mariadb-connector-c-3.3.8-src.tar.gz) |

### Glossary

| Term       | Description                                                      |
| -------- | ------------------------------------------------------- |
| HCCL     | Huawei Collective Communication Library         |
| NPU      | Neural Processing Unit                          |
| CANN     | Compute Architecture for Neural Networks, the software stack for Huawei Ascend AI processors |
| MPI      | Message Passing Interface                        |
| CCU      | Collective Communication Unit                    |
| Topology | Device connection relationships                                               |
| Rank     | Process number, the identifier in MPI                                         |

---

**Document version**: v1.4.
**Last updated**: 2026-11-12.
