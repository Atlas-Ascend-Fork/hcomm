# HCCL-VM User Guide

## 1. Overview

HCCL-VM is a high-performance collective communication virtual execution environment for Huawei Ascend NPU cards. This tool enables the development and functional verification of HCCL collective communication operators without real Ascend hardware.

The tool provides three operating modes: `normal` (real multi-card communication with lossless result correctness), `check-only` (verification only, large memory blocks reuse a shared pool to significantly reduce memory usage), and `estimate` (estimation, skips real communication blocking points to quickly produce a task list). Users can specify the mode at startup or switch modes at runtime through the `hccl-vm mode` command. For details, refer to [Operating Modes](#49-operating-modes-normal--check-only--estimate).

The tool also provides two simulation levels: the default Level 2 (southbound hijacking) and Level 1 (northbound hijacking, specified with `--level 1`). For details, refer to [Simulation Levels and Level 1 Hijacking](#410-simulation-levels-and-level1-hijacking-northbound-hijacking).

![hccl-vm GIF](docs/hccl-vm.gif)

## 2. Prerequisites

| Dependency | Version Requirement |
| -------- | ------------- |
| System Architecture | x86_64 Ubuntu 22.04 and above |
| Specification Constraints | Ascend 950 series products; for other specifications, refer to [Specification Constraints](#45-tool-specification-constraints) |
| QEMU (required only for AICPU mode on x86_64 host machines) | The system-provided qemu-user-static can be used directly. When the version is lower than 10.2.4, the translation code cache cannot be limited, QEMU resident memory is large, and large-scale scenarios may trigger paging and significantly affect performance. Install version 10.2.4 or above by following [2.3 QEMU Deployment](#23-qemu-deployment-aicpu-mode-x86_64-host-machine-only) for optimization. Native ARM host machines do not require QEMU. Compilation (`build.sh`/`build_pkg.sh`) does not depend on QEMU installed on the local machine. |
| Python3 (required only for the @estimate performance simulation plugin in estimate mode) | >=3.8 (including the sqlite3 standard library); the simpy discrete event simulation package must also be pre-installed (`pip3 install simpy`). For details, refer to [4.9.5](#495-estimation-plugin-estimate-and-dependency-description). |

### 2.1 CANN Package Installation

Install the latest version of the CANN Toolkit development kit package and the CANN ops operator package. [Download Link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)

```bash
# Ensure the installation packages have executable permissions
chmod +x Ascend-cann-toolkit_9.1.0_linux-x86_64.run
chmod +x Ascend-cann-950-ops_9.1.0_linux-x86_64.run
# Installation commands
./Ascend-cann-toolkit_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
./Ascend-cann-950-ops_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
```


### 2.2 hccl_test Compilation

hccl_test is the official HCCL performance testing tool provided by Ascend. For details, refer to [HCCL Performance Testing Tool](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/devaids/hccltool/HCCLpertest_16_0001.html). HCCL-VM supports running hccl_test cases in the virtual environment. Build the case binary programs by following the [hccl_test Case Build](#42-hccl-test-case-build) section first.

Remark: hccl_test is an optional dependency. For PyTorch use cases, refer to [PyTorch Use Case Examples](#33-pytorch-use-case-examples).

### 2.3 QEMU Deployment (AICPU Mode, x86_64 Host Machine Only)

The AICPU expansion mode requires QEMU user-mode emulation on x86_64 host machines (linux-user emulates aarch64). The system-provided qemu-user-static can be used directly. When the version is lower than **10.2.4**, the `-tb-size` option to limit the translation code cache is not supported, QEMU resident memory is large, and performance may be severely affected. Install version **10.2.4 or above** by following the steps below.

```bash
# 1) Install build dependencies (gcc/make/ninja are typically already installed with the system or this tool)
sudo apt-get install -y meson flex bison libglib2.0-dev libpcre2-dev libffi-dev

# 2) Download the official source code archive (the example uses 10.2.4; version 10.2.4 and above are all acceptable)
curl -fsSL -O https://download.qemu.org/qemu-10.2.4.tar.xz
tar -xf qemu-10.2.4.tar.xz && cd qemu-10.2.4

# 3) Build the aarch64 linux-user static version
mkdir build && cd build
../configure --target-list=aarch64-linux-user --static --disable-system --disable-tools \
    --disable-docs --disable-guest-agent --disable-slirp
make -j$(nproc) qemu-aarch64

# 4) Deploy and configure (the deployment path is customizable; the example uses /opt/qemu-10.2.4)
sudo mkdir -p /opt/qemu-10.2.4/bin
sudo cp qemu-aarch64 /opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_TB_SIZE=16
```

Runtime behavior and prompts:

- When a device process starts, the tool prints the actual QEMU program path, version, and the requested `-tb-size` value.
- When `HCCL_VM_QEMU_PATH` is not set, is invalid, or points to an older QEMU version that does not support `-tb-size`, the tool **does not abort**: it automatically falls back (searches PATH for `qemu-aarch64-static` / omits `-tb-size`) and prints a WARN message that includes the actual fallback program and version, along with a prompt that "translation cache is not limited and QEMU memory may be excessive." The AICPU mode startup fails only when no `qemu-aarch64-static` binary is found anywhere on the machine.
- `HCCL_VM_QEMU_TB_SIZE` specifies the QEMU translation code cache size (MiB). When not set, the default is `16`. A value of `0` delegates the decision to the QEMU default policy (the cache is not disabled). Failing to limit the cache can significantly increase QEMU resident memory and trigger paging.

---

## 3. Quick Start

### One-Click Installation

After pulling the complete CheckerL2 source code, run the following command in the repository root directory:

```bash
bash ./hccl_vm_installer
```

### 3.1 Manual Build and Installation

```bash
# 1. Create the working directory
mkdir -p /home/workspace
cd /home/workspace

# 2. Download dependency source code
git clone https://gitcode.com/cann/hccl.git
git clone https://gitcode.com/cann/hcomm.git

# 3. Install third-party dependencies
sudo apt-get update
sudo apt install build-essential cmake libboost-all-dev rdma-core libibverbs-dev pkg-config gcc-aarch64-linux-gnu g++-aarch64-linux-gnu qemu-user-static binfmt-support

# 4. Compile the HCCL-VM tool. After downloading the hcomm code, the tool source code path is: /home/workspace/hcomm/test/hccl_vm
cd /home/workspace/hcomm/test/hccl_vm
source /home/workspace/Ascend/cann/set_env.sh
export HCCL_CODE_HOME=/home/workspace/hccl
export HCOMM_CODE_HOME=/home/workspace/hcomm
bash ./build.sh --full

# 5. Copy and extract aicpu_hcxx.tar.gz from the CANN installation directory
bash build_pkg.sh
```

### 3.2 Usage Examples

#### 3.2.1 Environment Configuration

If the hccl_rootinfo.json file already exists in the /etc directory, delete it manually.

When the communicator scale exceeds 128 cards (129 to 1024), switch to the MariaDB database. For details, refer to [MariaDB Database Usage](src/storage_backends/mariadb/README.md). After completing the deployment and setting `HCCL_VM_STORAGE_CONFIG`, return to this section. Run `unset` on that variable to restore the default SQLite database.
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

2. Execute


```bash
# Enter the bin directory to run hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# Enable the runner plugin if needed (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for the current operator execution (run the hccl_test case in a cluster environment with 1 supernode, 1 server, and 1 NPU)
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run the checker verification
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal
(hvm)$> exit
```

3. Verify the hccl_test case execution results
[Runner Result Viewing](#481-runner-plugin-results) 
[Checker Result Viewing](#482-checker-plugin-results)

#### 3.2.3 AICPU Mode

The AICPU expansion mode requires the algorithm expansion steps to execute on the device side. Therefore, the hccl-vm tool must compile and simulate the device-side symbols of HCCL. Because the device-side symbols use the ARM architecture, cross-compilation is required on x86 environments, and QEMU is required at runtime to enable AICPU mode simulation.

Running AICPU mode on an x86 host machine uses QEMU user-mode emulation. When `HCCL_VM_QEMU_PATH` is not configured, the tool automatically falls back to `qemu-aarch64-static` in PATH and prints a warning (including the actual version). This does not affect startup. When the system-provided version is lower than 10.2.4, the translation code cache cannot be limited, memory usage is high, and large-scale scenarios may trigger paging and affect performance. For QEMU version 10.2.4 and above installation and `HCCL_VM_QEMU_PATH` configuration, refer to [2.3 QEMU Deployment](#23-qemu-deployment-aicpu-mode-x86_64-host-machine-only). Native ARM host machines do not require QEMU.

1. Configure environment variables.

```bash
# Enter the tool installation directory
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AI_CPU"

# x86_64 host machine: configure the QEMU program and translation code cache (MiB); refer to Section 2.3
export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_TB_SIZE=16
```

2. Execute

```bash
# Enter the bin directory to run hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# Enable the runner plugin if needed (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for the current operator execution (run the hccl_test case in a cluster environment with 1 supernode, 1 server, and 1 NPU)
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run the checker verification
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal
(hvm)$> exit
```

3. Verify the hccl_test case execution results [Runner Result Viewing](#481-runner-plugin-results) [Checker Result Viewing](#482-checker-plugin-results)

#### 3.2.4 HostDPU Mode

The environment variable configuration and other runtime settings for HostDPU expansion mode are the same as those for AICPU mode. The only differences are in cluster scale and communicator scale as follows:

1. The cluster specification configuration requires `./hccl-vm start ascend950_cluster_32_server_normal_hostdpu.yaml`
2. The number of servers must be >= 2; communicators on a single server are not supported (configurations such as 112.yaml, 114.yaml, and 118.yaml are not supported)

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

2. Execute

```bash
# Enter the bin directory to run hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# Enable the runner plugin if needed (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for the current operator execution (run the hccl_test case in a cluster environment with 1 supernode, 1 server, and 1 NPU)
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run the checker verification
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal
(hvm)$> exit
```

3. Verify the hccl_test case execution results [Runner Result Viewing](#481-runner-plugin-results) [Checker Result Viewing](#482-checker-plugin-results)

### 3.3 PyTorch Use Case Examples

HCCL-VM supports running selected PyTorch `aclgraph` graph-mode collective communication scripts. Scripts should use `torch.multiprocessing.spawn` to launch device processes. Multi-server use cases require launching multiple Python main processes concurrently on the same host machine, each simulating a separate server. For the complete description, refer to [PyTorch Mode Running Guide](docs/pytorch_guide.md).

#### 3.3.1 Environment Preparation

Complete the HCCL-VM build and installation and CANN environment configuration first, and prepare a Python virtual environment with compatible versions of `torch` and `torch_npu` installed. PyTorch currently supports only Ascend 950 series products. To use the Ascend 960 series, install the A6 ops package and change `soc_version` in the topology cluster configuration to a real model such as `Ascend960DT_961BY`.

```bash
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin
./hccl-vm start ascend950_cluster_32_server_normal.yaml
```

After entering the `(hvm)$>` terminal, run the following commands. Complete all subsequent communicator configuration, environment variable setup, and Python launches within this terminal so that the Python processes inherit the simulated environment.

```bash
# Enable PyTorch adaptation
export HCCL_VM_ENABLE_PYTORCH=1
# Activate the virtual environment (if applicable)
source /path/to/torch_test/bin/activate

# Install Runner when simulation execution is required
hccl-vm plugin install @runner
```

#### 3.3.2 Single-Server Running

This example uses the `118` configuration with 1 server and 8 devices to execute an AllReduce SUM on 1024 int32 elements. The following command parameters correspond to the test script described in the guide. When using other scripts, adjust the parameters according to their definitions.

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

This example uses the `128` configuration with 2 servers, 8 devices per server, and 16 ranks in total. Execute the following commands within the HCCL-VM terminal where environment preparation is already complete:

```bash
hccl-vm mock-comm 128

# Launch two servers concurrently; set environment variables separately and save logs
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

# Wait for both servers to finish, and check their logs for exceptions and execution results
wait
```

`HCCL_VM_SERVER_INDEX` is the 0-based server index in the current communicator configuration. It takes effect only when `HCCL_VM_ENABLE_PYTORCH=1` is set and must match the `--server_index` of the corresponding script. Setting environment variables within parentheses prevents different server indexes from overwriting each other.

All servers must use the same `--master_ip` and `--master_port`. The `--dev_num` and `--server_num` values must match the `mock-comm` configuration. The local logical device IDs for each server are always `0-7`. In this example, server 0 corresponds to global ranks `0-7` and server 1 corresponds to global ranks `8-15`. The script calculates global ranks assuming an equal number of devices per server. For other scales, select a communicator configuration where all servers have the same number of devices.

> **Correspondence in POD form (4-layer topology)**: "Server" here refers to the **leaf node (host node)** in the communicator.
> In server form, the leaf node is the server. **In POD form, the leaf node is the board**, so
> `--server_num`/`HCCL_VM_SERVER_INDEX` should count and number by **board** (for example, the `11_64` configuration is 1 POD x 8 boards x 8 cards;
> launch 8 Python main processes concurrently with `--server_num 8`, `--dev_num 8`, and `HCCL_VM_SERVER_INDEX` ranging from `0` to `7`).
> The local logical device IDs for each process are `0-7` within the board, consistent with real hardware.

#### 3.3.4 Result Viewing and Verification

After all processes finish, check the corresponding `server0.log` and `server1.log` files (for multi-server scenarios). Focus on the exit status, exceptions, and relevant log output. Then run the Checker verification within the HCCL-VM terminal. For result descriptions, refer to [Checker Plugin Results](#482-checker-plugin-results).

```bash
hccl-vm plugin run @checker

# Exit the tool terminal after all cases and verifications are complete
exit
```

### 3.4 HCCL Code Modification Verification Example

If you have modified the CANN operator package code, such as adding a new algorithm type, follow these steps to ensure your changes take effect. The build_pkg.sh script helps you perform packaging, installation, and device-side dependency symbol copying. Set the environment variables before running it:

```bash
# Assume your CANN installation directory is: /home/workspace/Ascend
source /home/workspace/Ascend/cann/set_env.sh
# Configure the hccl code repository path
export HCCL_CODE_HOME=/home/workspace/hccl
# Configure the hcomm code repository path
export HCOMM_CODE_HOME=/home/workspace/hcomm
```

1. If you updated or modified the CANN hccl repository code, run `bash build_pkg.sh --install hccl`.
2. If you updated or modified the CANN hcomm repository code, run `bash build_pkg.sh --install hcomm`.
3. If you updated or modified both the CANN hccl and hcomm repository code, run `bash build_pkg.sh --full`.
4. Refer to the [Usage Examples](#32-usage-examples) steps to re-run the use case.

---

## 4. Detailed Guide

### 4.1 Tool Environment Variable Configuration

**HCCL-VM Environment Variable Description**:

| Environment Variable | Purpose | Example |
| ------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------- |
| `HCCL_CODE_HOME` | Specifies the HCCL source code path for HCCL-VM compilation. Not configured by default. | `export HCCL_CODE_HOME=/home/workspace/hccl` |
| `HCOMM_CODE_HOME` | Specifies the HCOMM source code path for HCCL-VM compilation. Not configured by default. | `export HCOMM_CODE_HOME=/home/workspace/hcomm` |
| `HCCL_VM_ENABLE_PYTORCH` | When set to `1`, enables PyTorch adaptation. For the running method, refer to [PyTorch Use Case Examples](#33-pytorch-use-case-examples). | `export HCCL_VM_ENABLE_PYTORCH=1` |
| `HCCL_VM_SERVER_INDEX` | The **leaf node (host node)** index simulated by the PyTorch process, starting from `0`. Takes effect only when `HCCL_VM_ENABLE_PYTORCH=1` is set and must match the `--server_index` of the test script. In server form, the leaf node is the server. **In POD form (4-layer topology), the leaf node is the board**, so count and number by board. | `export HCCL_VM_SERVER_INDEX=0` |
| `HCCLVM_ENABLE_DUMP_DATA` | 1. Enables performance simulation file output to disk. When enabled, 3 runner_hcclvm_*_data.bin files and 1 ccu_channel_jetty_config.xml file are output during test case execution for performance simulation use. <br> 2. Enables the Runner plugin to dump input and output data. When enabled, the input and output data of each operator are dumped to the all_rank_input_output.txt file during test case execution. | `export HCCLVM_ENABLE_DUMP_DATA=1 to enable; export HCCLVM_ENABLE_DUMP_DATA=0 or leave the variable unset to disable` |
| `HCCL_VM_RES_STAT_DUMP` | Enables hijacking-layer resource statistics persistence. After this variable is set, each device resource count (memory/Jetty/QP/Notify/AIV core, and so on) is exported as CSV during business process finalization. The output goes to the installation directory `data/stat/<podId>_<serverId>_<boardId>_<deviceId>_res_<timestamp>.csv` (`boardId` is the board index in POD form 4-layer topology; it is always `0` in server form). When not set (default), no export occurs. | `export HCCL_VM_RES_STAT_DUMP=1` |
| `HCCL_VM_QEMU_PATH` | Applies only to AICPU mode on x86_64 host machines: a non-empty absolute path to an external QEMU program (the target must be an executable regular file; symlinks pointing to regular files are allowed). When not set or invalid, the tool automatically falls back to searching PATH for `qemu-aarch64-static` and prints a WARN message (including the actual program and version). When no binary is found anywhere on the machine, AICPU startup fails. For deployment steps, refer to [2.3 QEMU Deployment](#23-qemu-deployment-aicpu-mode-x86_64-host-machine-only). | `export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static` |
| `HCCL_VM_QEMU_TB_SIZE` | Applies only to AICPU mode on x86_64 host machines: the QEMU translation code cache size (MiB). When not set, the default is `16`. A value of `0` explicitly passes `-tb-size 0` and delegates the decision to the QEMU default policy (the cache is not disabled). Only decimal digit strings parseable to uint32_t are accepted; invalid values trigger a warning and fall back to the default of 16. When the target is an older QEMU version that does not support `-tb-size`, this option is omitted and a warning about potentially excessive memory is printed. | `export HCCL_VM_QEMU_TB_SIZE=16` |
| `HCCL_UB_MULTI_CHANNEL_NUM` | An HCCL-side environment variable read by the proxy stub (the `HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM` query item in `HcclConfigGetInfo` uses the same definition): the number of parallel jetties on one channel. This value is stored with the operator into `opDetails.ubMultiChannelNum`. The `@estimate` lightweight performance simulation plugin uses this value to calculate port bandwidth utilization (`multi_port_utilization_low` tiers: `+= (n-1) x jetty_utilization_step`, capped at 1.0; refer to `docs/light_performance/simulator_design.md` Section 7.2). When not set, invalid, or `0`, the value falls back to `1` (multi-jetty is not enabled; the original utilization table is used directly). | `export HCCL_UB_MULTI_CHANNEL_NUM=8` |

### 4.2 HCCL-Test Case Build

The hccl_test case source code is located in the CANN package installation directory. Both OpenMPI and MPICH environments are supported for compilation and running. For runtime differences, refer to [OpenMPI and MPICH Environment Case Execution Differences](#47-openmpi-and-mpich-environment-case-execution-differences). This guide uses the OpenMPI environment as an example.

#### 4.2.1 OpenMPI Environment Compilation

1. Install OpenMPI

```bash
sudo apt-get update
sudo apt install openmpi-bin libopenmpi-dev
```

2. Compile hccl_test

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

#### 4.2.2 MPICH Environment Compilation

Assume the MPICH path is: `/usr/lib/mpich`.

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

An Ascend cluster topology is composed of one or more Server/Pod sub-topologies combined according to CLOS layered network rules. Therefore, before generating a cluster topology, the user must confirm the topology type of each Server/Pod.
The user can either select a predefined topology type provided by the HCCL-VM tool or define a custom Server/Pod topology type according to the configuration file format requirements.

Describing the topology network relationships of a Server/Pod includes the following aspects:

 - **Port Configuration Table**: Describes the physical port configuration of an NPU card, such as NPU-to-NPU direct connection ports (P2P) and NPU out-of-chassis ports (P2NET).
 - **Link Configuration Table**: Describes the connection relationships among all NPU cards within a Server/Pod, such as full mesh connections.
 - **PortBound**: Describes the binding relationships of certain ports on an NPU card, where multiple ports are bound into a single PortGroup.

```yaml
type: "server_intra_links"
name: "ascend950_links_topo_demo"
description: "Ascend 950 chip normal topology connection relationship description file"

soc_version: "Ascend950"
device_num: 16

device_ports_allocate_map:
  # Port allocation table: 0: not used, 1: device direct connection, 2: device to switch, 3: d2h port
  #                 portId:  0  1  2  3  4  5  6  7  8
    - {die_id: 0, pin_map: [1, 1, 1, 0, 2, 2, 2, 2, 3]} # die0
    - {die_id: 1, pin_map: [0, 0, 0, 0, 0, 0, 0, 0, 0]} # die1

# port_group: describes which ports are merged into a portGroup; ports in the same portGroup share the same IP address
port_group:
    - {layer: 0, ports: ["0/4", "0/5", "0/6", "0/7"]}

links:
  # -- Notation: each row of 8 devices forms a full mesh --
  - link_mode: "fullmesh"
    connections:
      # The following example indicates: devices 0, 1, 2, 3 of die0 are all fully connected through die0 ports
      - {die_id: 0, devices_range: [0, 3]}
      - {die_id: 0, devices_range: [4, 7]}
      - {die_id: 0, devices_range: [8, 11]}
      - {die_id: 0, devices_range: [12, 15]}
  
  #- link_mode: "enum"
  #  device_to_device_links:
  #    The following example indicates: devices 0 and 1 connect through die0 ports to devices 1, 3, 5, 7 through die1 ports respectively
  #            That is: device0 connects to device1, device3, device5, device7; device1 connects to device3, device5, device7
  #    - {src_die_id: 1, src_device_id_range: [0, 2], dst_die_id: 1, dst_device_id_range: [1, 3, 5, 7]}

  - link_mode: "enum"
    device_to_switch_links:
      # The following example indicates: device0 through device15 all connect to the switch through die0 ports. Combined with portGroup, device0 through device15 all connect to the switch through portGroup[0/4, 0/5, 0/6, 0/7].
      - {die_id: 0, devices_range: [0, 15]}

```

**Field Description**:

 - **soc_version**: The chip model, such as `Ascend950`.
 - **device_num**: The total number of devices, determined by the chip model and topology type. In server form, this is the number of cards in the server. **In POD form, this is the total number of cards in the POD** (for example, 64 for 1dpod64).
 - **board_num**: Optional. The number of boards (compute boards) within a POD. **This is the trigger field for POD form**: when absent or set to 1, the topology is in server form (3-layer topology); when greater than 1, the topology is in POD form (4-layer topology; refer to the description below). Constraints: `device_num % board_num == 0`, `board_num <= 255`.
 - **device_num_per_board**: Optional. The number of cards per board (typically 8, but not hardcoded). When absent, the value is derived from `device_num / board_num`. When explicitly configured, the value must match the derived value.
 - **form_type**: Optional. `"server"` or `"pod"` (case-insensitive). When absent, the value is derived from `board_num`. When explicitly configured and contradictory to the derived result, an error is reported.
 - **net_instance_scope**: Optional. Specifies the scope of `net_instance_id` per layer, with values `board`/`pod`/`superpod`/`cluster` (for details, refer to Section 4.3.4). Default derivation rules: for server form `{layer0: pod, layer1: superpod, layer>=2: cluster}`; for POD form, layer0 is derived based on whether intra-board PEER2PEER direct connections exist at that layer (direct connections present -> `board`, pure switch network -> `pod`); other layers follow the server form rules.
 - **device_ports_allocate_map**: The port allocation table, describing the port configuration of each die. A value of 1 indicates device direct connection ports, 2 indicates device-to-switch ports, and 3 indicates d2h ports.
 - **port_group**: Describes which ports are merged into a single portGroup. Ports in the same portGroup share the same IP address. Ports not configured default to each port forming its own portGroup.
 - **links**: The link configuration table, describing the connection relationships among all NPU cards within a Server/Pod and between NPUs and switches.
   - **NPU Direct Connection Relationships**: The tool provides two methods to configure NPU direct connection relationships:
    - **link_mode == "fullmesh"**: Indicates that all devices are fully connected based on the ports of one die. When new typical connection methods are added in the future, new link_mode types can be added, such as "ring".
    - **link_mode == "enum"**: Enumeration method. When the NPU connection method within a Server/Pod is complex, all link relationships can be described by enumeration.
   - **NPU-to-Switch Connection Relationships**: The user can configure NPU-to-switch connection relationships by enumeration.
  - **device_to_device_links**: Describes NPU-to-NPU connection relationships. The two endpoint devices are declared with `src_device_id`/`src_device_id_range`
    and `dst_device_id`/`dst_device_id_range` (**physical IDs within the board**; the legacy names `src_local_id`/`dst_local_id`/`*_local_id_range` remain compatible).
    In POD form, **`src_board_id`/`dst_board_id`** can be used to declare the board to which each endpoint belongs (both must be configured together), **supporting cross-board direct connections**.
  - **device_to_switch_links**: Describes NPU-to-switch connection relationships. In POD form, a single **`board_id`** is used to declare the board to which the device belongs (the destination is a switch, so no dst_board_id is needed).

> **Precautions (POD form)**: Device IDs that appear in `device_group`/`port_list`/`port_group`/`links`
> must be **physical IDs within the board** (such as `0-7`) in POD form. The card structure is described per board. The `local_id` is calculated by the tool as
> `board_id x cards per board + physical ID within the board`. For link entry board attribution fields:
> **`fullmesh` `connections`, `device_to_switch_links`, and `matrix`** use a single **`board_id`** (both endpoints on the same board);
> **`device_to_device_links`** uses **`src_board_id`/`dst_board_id`** (endpoints can be on different boards, that is, cross-board direct connections).
> Entries without board fields apply to all boards (the single-board description is replicated and expanded).
> If POD-global IDs are used by mistake (such as `8` or `63`), the tool reports an error directly (`device id exceeds range [0, cards per board - 1]`).

**POD Form (4-Layer Topology) Description**

The topology has two forms with different layer counts:

| Form | Layers | Leaf Node (Host Node) | Card Physical ID | Output |
|------|------|---------------------|----------|------|
| Server form | server -> supernode -> cluster (3 layers) | server | `[0, device_num-1]` | One `topo.json`/`hccl_rootinfo.json` per server |
| POD form | **board -> pod -> supernode -> cluster (4 layers)** | **board** | **`[0, cards per board - 1]` within the board** (typically `[0,7]`) | **One `topo.json` per POD** (in the server directory) + **one `hccl_rootinfo.json` per board** |

In POD form, the cards within a POD are divided into boards (compute boards). Each board is an independent host node (independent host IP, independent rootinfo).
**topo.json is a single file at the POD level** placed in the server directory, where `local_id` is the **POD-global ID**:
`local_id = board_id x cards per board + physical ID within the board`.
For example, the `1dpod64` (64-card POD) configuration with `device_num: 64` + `board_num: 8` generates the following cluster directory:

```
config/network/cluster/ascend950_cluster_16_server_1dpod64/
├── servers_info.json                # One entry per leaf node (per board), totaling 16 PODs x 8 boards = 128 entries
└── superpod0/
    └── server0/                     # POD 0 (no host IP assigned)
        ├── topo.json                # POD-level topology: 64 cards, local_id = board_id*8 + ID within board (0-63)
        ├── board0/
        │   └── hccl_rootinfo.json   # 8 cards, device_id/local_id = physical IDs within board 0-7
        ├── board1/ ...
        └── board7/
```

Configuration example (`asset/cluster_model/config/server_or_pod/ascend950_server_topo_1dpod64.yaml`, intra-board direct connections, links described per board):

```yaml
soc_version: "Ascend950"
device_num: 64                       # Total cards in the POD
board_num: 8                         # 8 boards -> 8 cards per board
hardware_type: "950D-1D-POD-64P-topo"

# Card structure is described per board; device id is the physical ID within the board (0-7)
device_group:
  - {group_id: 1, device_list: [0, 1, 2, 3]}
  - {group_id: 2, device_list: [4, 5, 6, 7]}
links:
  # Links are described per board: fullmesh entries carry board_id; d2d entries carry src_board_id/dst_board_id;
  # net_instance_id/topo_instance_id are numbered by board
  - link_mode: "fullmesh"
    connections:
      - {board_id: 0, die_id: 0, net_instance_id: 0, topo_instance_id: 0, device_id: [0, 1, 2, 3]}
      - {board_id: 0, die_id: 1, net_instance_id: 0, topo_instance_id: 0, device_id: [4, 5, 6, 7]}
      - {board_id: 1, die_id: 0, net_instance_id: 1, topo_instance_id: 1, device_id: [0, 1, 2, 3]}
      # ... board2 through board7 follow the same pattern
  - link_mode: "enum"
    device_to_device_links:
      # Intra-board direct connection: src_board_id == dst_board_id
      - {src_board_id: 0, net_instance_id: 0, topo_instance_id: 0, src_die_id: 0, src_device_id: [0, 1, 2, 3],
         dst_board_id: 0, dst_die_id: 1, dst_device_id: [4, 5, 6, 7]}
      # ... board1 through board7 follow the same pattern
```

Configuration example (cross-board direct connection scenario, requires configuring `src_board_id` ≠ `dst_board_id`; when layer0 direct connection domain spans boards, the scope is automatically set to pod):

```yaml
soc_version: "Ascend950"
device_num: 16                        # 2 boards x 8 cards
board_num: 2
hardware_type: "950D-1D-POD-16P-topo"

device_group:
  - {group_id: 1, device_list: [0, 1, 2, 3]}
  - {group_id: 2, device_list: [4, 5, 6, 7]}

links:
  - link_mode: "fullmesh"             # 8 cards within each board are fully meshed through die1
    connections:
      - {board_id: 0, die_id: 1, net_instance_id: 0, topo_instance_id: 0, device_id: [0, 1, 2, 3, 4, 5, 6, 7]}
      - {board_id: 1, die_id: 1, net_instance_id: 1, topo_instance_id: 1, device_id: [0, 1, 2, 3, 4, 5, 6, 7]}
  - link_mode: "enum"
    device_to_device_links:
      # Cross-board direct connection: src_board_id(0) != dst_board_id(1), die0's 4 cards are cross-connected
      - {src_board_id: 0, net_instance_id: 0, topo_instance_id: 0, src_die_id: 0, src_device_id: [0, 1, 2, 3],
         dst_board_id: 1, dst_die_id: 0, dst_device_id: [4, 5, 6, 7]}
    device_to_switch_links:           # The destination is a switch; only a single board_id is needed to declare the board
      - {board_id: 0, die_id: 0, devices_range: [0, 7]}
      - {board_id: 1, die_id: 0, devices_range: [0, 7]}
      # ... board0 die1 / board1 die1 follow the same pattern
```

Configuration example (`ascend960_pod_topo_clos.yaml`, all-switch within chassis with no intra-board direct connections; links without board_id are replicated and expanded to all boards):

```yaml
device_num: 72
board_num: 9                         # 9 boards x 8 cards
net_instance_scope:
  - {layer: 0, scope: "pod"}         # layer0 is an intra-chassis switch network covering all 72 cards in the POD; cannot be split by board
links:
  - link_mode: "enum"
    device_to_switch_links:
      - {die_id: 0, devices_range: [0, 7]}   # No board_id configured -> applies to all 9 boards
```

**Mandatory validations for POD form** (generation fails if any condition is not met):
1. `device_num % board_num == 0`, and the number of cards per board > 0;
2. All device IDs in the configuration must fall within `[0, cards per board - 1]`; the `board_id` in link entries and the
   `src_board_id`/`dst_board_id` in `device_to_device_links` (if configured) must fall within `[0, board_num - 1]`, and `src_board_id`/`dst_board_id` must be configured together;
3. `board_num <= 255` (CC segment capacity of IP addresses), and the number of endpoints per board `<= 255` (DD segment capacity of IP addresses);
4. When PEER2PEER direct connections exist at layer0: **intra-board direct connections** (each board forms an independent network) must have scope set to `board` (to ensure that the
   `net_instance_id` at layer0 is unique per board; otherwise, intra-board direct connection groups from multiple boards may be incorrectly merged by the business side); **cross-board direct connections**
   (edges where `src_board_id != dst_board_id` exist, and the direct connection domain is a single POD-level instance) must have scope set to `pod`
   (otherwise, the `net_instance_id` at both endpoints of a cross-board direct connection edge would be inconsistent and would be filtered out by the business side);
5. When PEER2PEER direct connections exist, each board must have the same number of direct connection edges and every board must have direct connection entries (homogeneity validation; cross-board edges count once for each endpoint board);
6. Layers declared in `net_instance_scope` must actually exist in the topology.

> `fullmesh`/`device_to_switch_links`/`matrix` entries use a single `board_id` (both endpoints on the same board); `device_to_device_links`
> uses `src_board_id`/`dst_board_id` to declare endpoint attribution, **supporting cross-board PEER2PEER direct connections** (for example, hf: board0<->board1 full cross).
> In ranktable.json, `device_id` is the physical ID within the board, and `local_id` is the POD-global ID (consistent with the POD-level topo.json).

#### 4.3.2 Cluster Topology Configuration File Description

An Ascend cluster network is composed of one or more Server/Pod sub-topologies combined according to CLOS layered network rules. Users can select different Server/Pod topology types based on cluster scale and requirements.

Users can define custom cluster topology configuration files according to the following configuration file format:

```yaml
name: "ascend950_cluster_32_server_normal"
description: "Ascend 950 normal networking: 32 supernodes, 1 server per supernode"

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

The configuration file above describes a cluster topology containing 4 supernodes, 32 servers, and a total of 128 NPU cards. Each Server/Pod uses the ascend950_server_topo_normal topology type.

**Field Description**:

 - **super_node_num**: The total number of supernodes.
 - **server_num**: The total number of servers/pods.
 - **server_list**: The configuration information for each Server/Pod, including the supernode ID, device ID range, chip model, and Server/Pod topology configuration file path.

> **POD form scale conversion**: `server_num` counts the number of servers/PODs. If the topology configuration pointed to by `server_topo` declares
> `board_num > 1` (POD form), then the total number of **leaf nodes (host nodes) in the cluster = server_num x board_num**,
> and the total number of cards = number of leaf nodes x cards per board. For example, `ascend950_cluster_16_server_1dpod64` (`server_num: 16`,
> topology `board_num: 8`, 8 cards per board) has 128 leaf nodes and 1024 cards total, and `servers_info.json` contains 128 records.
>
> Each record in `servers_info.json` corresponds to one leaf node and carries explicit `super_pod_id`/`server_id`/`board_id`
> triple IDs (the tool matches by ID, not by entry position). Host IP rules:
> server form `AA.BB.0.0` (AA = supernode id + 1, BB = server id + 1); POD form `AA.BB.CC.0` (CC = board_id).

#### 4.3.3 Communicator Configuration File Description

In an Ascend cluster environment, users must select different communicator configuration files based on the communicator required by the operator to be executed.

The tool provides the `hccl-vm mock-comm` command to read and configure operator communicator configuration files. The communicator configuration file format is YAML, and the path is `hccl_vm_install/config/topo_meta`. If the directory does not contain the corresponding communicator configuration file, the user must create one first.

The hccl-vm tool supports asymmetric topology communicator configurations, as shown below:

```yaml
# 1. Global statistics: podNum, serNum, rankNum are all less than 1024
meta:
  podNum: 1  # Total number of supernodes
  serNum: 2  # Total number of servers
  rankNum: 6 # Total number of ranks

# 2. Detailed topology structure 
topology:
  - podId: 0
    servers:
      - serId: 0
        # The local IDs of the ranks actually run by each server
        ranks: [0, 2]
      - serId: 1
        # The local IDs of the ranks actually run by each server
        ranks: [1, 3, 5, 7]
```

**Precautions**:

- When configuring a communicator, the tool regenerates the topo.json and ranktable.json files based on the specified communicator configuration number.
- In the communicator configuration YAML file above, the ranks field represents the list of local IDs (that is, the physical IDs of the devices) of the ranks actually run by each server.

When the communicator scale is large, the server selection fields and rank selection fields in server entries can be freely combined. The following fields are supported:

- **Server selection fields** (choose one; cannot be configured simultaneously):
  - `serId`: A single server ID or a list of server IDs (such as `3` or `[0, 1]`; a list represents each enumerated server);
  - `serId_range`: A server ID range with the value `[<start id>, <end id>]` (closed interval; only list format is supported), representing all servers from the start ID to the end ID under the supernode.
- **Rank selection fields** (choose one; cannot be configured simultaneously):
  - `ranks`: The list of local IDs of the ranks actually run by each server;
  - `ranks_range`: A local ID range with the value `[<start id>, <end id>]` (closed interval; only list format is supported), representing all local IDs from the start ID to the end ID selected for each server.

The following example (2 supernodes, 4 servers total, 32 ranks) demonstrates multiple combinations mixed within the same file:

```yaml
# 1. Global statistics: podNum, serNum, rankNum are all less than 1024
meta:
  podNum: 2  # Total number of supernodes
  serNum: 4  # Total number of servers
  rankNum: 32 # Total number of ranks

# 2. Detailed topology structure 
topology:
  - podId: 0
    servers:
      # serId is an ID list, representing server 0 and server 1 under pod 0
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

- serId and serId_range cannot be configured simultaneously; ranks and ranks_range cannot be configured simultaneously.
- serId_range and ranks_range support only the `[<start id>, <end id>]` list format (closed interval). The `<start id>-<end id>` format is not supported. The start ID cannot be greater than the end ID, and the range width (end ID - start ID + 1) cannot exceed 1024.
- A serId_range entry expands to all servers from the start ID to the end ID under the supernode. A serId list entry represents each enumerated server, and the ranks are identical for each server.
- If no rank selection field is configured in a server entry, the ranks for that server are an empty list. If no server selection field is configured, serId defaults to 0.
- The syntax above applies to **server form** (3-layer topology): `ranks` are physical IDs within the server, and one server is one leaf node.

**POD Form (4-Layer Topology) Boards Configuration**

In POD form, the leaf node is the board (not the POD). Therefore, the communicator must declare participating cards per board,
and `ranks` are **physical IDs within the board** (such as `[0, 7]`, not `[0, 63]` within the POD):

```yaml
# 1. Global statistics: podNum, serNum, boardNum, rankNum are all less than 1024
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
          # 8 boards, each taking cards with physical IDs 0-7 -> 64 ranks total
          - {boardId_range: [0, 7], ranks_range: [0, 7]}
```

- **Board selection fields** (choose one; cannot be configured simultaneously):
  - `boardId`: A single board ID or an ID list (such as `3` or `[0, 1]`);
  - `boardId_range`: A board ID range, `[<start id>, <end id>]` (closed interval; width does not exceed 1024).
- **Rank selection fields**: `ranks`/`ranks_range` have the same semantics as above, but the value range is **physical IDs within the board**.
- `boards` is an optional field: **when not configured, the server is treated as having only one implicit `board0`**,
  and `ranks`/`ranks_range` are written directly under the server entry (the existing server-form syntax, fully backward compatible).
- `boards` and `ranks`/`ranks_range` on the server entry are **mutually exclusive**; configuring both simultaneously results in an error (rank attribution cannot be determined).
- When `boardId`/`boardId_range` is not configured, the default attribution is `board0`.
- The rank_id assignment order is: supernode -> server(POD) -> board -> device, incrementing from 0.

POD-form communicator configurations already provided by the tool (`asset/cluster_model/topo_meta/`):

| Number | Scale | Description |
|------|------|------|
| `118` | 1 supernode x 1 POD x 1 board x 8 cards = 8 ranks | Single board (server-form syntax; in POD form this is board0) |
| `11b2` | 1 x 1 x 2 boards x 8 cards = 16 ranks | 2 boards within a single POD |
| `11b4` | 1 x 1 x 4 boards x 8 cards = 32 ranks | 4 boards within a single POD |
| `11b9` | 1 x 1 x 9 boards x 8 cards = 72 ranks | POD72 (ascend960_pod_topo_clos) entire POD |
| `11_64` | 1 x 1 x 8 boards x 8 cards = 64 ranks | 1dpod64 entire POD |
| `12_64` / `14_64` / `18_64` / `116_64` | 2 / 4 / 8 / 16 PODs, 8 boards x 8 cards per POD | 128 / 256 / 512 / 1024 ranks |

#### 4.3.4 topo.json and ranktable.json File Description

The `topo.json` and `ranktable.json` files do not need to be created manually. The tool generates them automatically based on the following information:

- **Topology configuration number**: The number specified by the user at startup (such as 112, 113, and so on)
- **Chip type**: The chip type automatically identified from the running environment.

Although the configuration files are generated automatically by the tool, understanding their structure helps users comprehend the topology configuration.

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
  - `topo_type`: Topology type (`1DMESH`, `CLOS`, and so on)

**ranktable.json Structure**:

`ranktable.json` describes the mapping between the devices used by the current communicator and their addresses (EIDs) at each network layer.
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
- `rank_id`: The global rank number within the communicator, incrementing from 0 in the order **supernode -> server(POD) -> board -> device**.
- `device_id`: The **physical ID within the leaf node** (physical ID, not logical ID). This is the physical device number selected by `aclrtSetDevice()` on the business side;
  hcomm's `CheckRankGraphAddrs` filters entries by `rank.deviceId == devPhyId` and queries the hardware EID using this phyId.
  In server form, the value range is `[0, cards in server - 1]`. **In POD form, the value range is `[0, cards in board - 1]` (typically `[0,7]`)**.
- `local_id`: Has the same definition as `local_id` in `topo.json`. The business side matches edges in topo.json using the `local_id` from ranktable.
  In server form, the value is the same as `device_id` (in this tool, `logic_id == physical_id`). **In POD form, topo.json is a single file at the POD level
  (in the server directory), and `local_id` is the POD-global ID = `board_id x cards per board + physical ID within the board`** (for 1dpod64, the value range is `[0,63]`).
  `device_id` remains the physical ID within the board.
- `level_list[].net_layer`: The network layer number (0 is the intra-node direct connection layer, 1 is the out-of-node/out-of-chassis layer, 2 and above are higher-level layers, and the d2h/HOST layer is typically 3).
  **After the board layer is introduced, the net_layer numbering rules remain unchanged**; the board dimension is expressed through `net_instance_id`.
- `level_list[].net_instance_id`: The network instance identifier for the layer, assembled according to the **scope**:

  | Scope | Format | Typical Application |
  |--------|------|----------|
  | `board` | `sp_{supernode}_srv_{POD}_board_{board}_idx_{k}` | POD form layer0 (intra-board direct connection, **unique per board**) |
  | `pod` | `sp_{supernode}_srv_{POD}_idx_{k}` | Server form layer0; POD form intra-chassis switch network spanning boards |
  | `superpod` | `sp_{supernode}_idx_{k}` | layer1 (cross-server/POD but within the same supernode, out-of-chassis interconnection) |
  | `cluster` | `cluster` | layer>=2 (cluster-level, HOST/d2h layer) |

- `level_list[].net_type`: `TOPO_FILE_DESC` when `net_layer=0`; `CLOS` when `net_layer>=1`.
- `level_list[].rank_addr_list[].addr`: The EID bound to the port at this layer (with the `0x` prefix removed); for the UBOE protocol layer, this is a dotted IPv4 address (`addr_type: "IPV4"`).
- `level_list[].rank_addr_list[].plane_id`: Fixed as `plane_0` for the PEER2PEER layer; inherits `plane_pg_{N}` from the configuration for the PEER2NET layer.

> **Layer trimming rules**: For a layer to appear in the ranktable, it must have an EID allocated in rootinfo.
> When the communicator involves only **1 leaf node** (1 server in server form, 1 board in POD form) and does not span supernodes, `net_layer>=3` is not output.
> When the total number of layers for a single leaf node exceeds 3, only the 3 layers with the smallest numbers are retained. Cross-supernode communicators do not apply the trimming rules above.

### 4.4 hccl_config.sh File Description

The hccl_config.sh file contains the environment variable configuration required for running HCCL_Test cases. The environment variables in this file are consistent with those used by Hccl_Test cases in a real hardware environment.
Users must modify the hccl_config.sh script according to their use cases and requirements to configure the HCCL case runtime environment variables.

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

  # Return 0 regardless of whether files were deleted, to ensure the script continues execution
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

# Disable the HCCL heartbeat feature
export HCCL_DFS_CONFIG=cluster_heartbeat:off

# Set the HCCL-VM installation path, inferred from the script's own location (compatible with bin/ and script/ subdirectories)
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

# Enable log output to screen
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# Set the HCCL operating mode (CCU, AI_CPU, AIV, and other expansion modes)
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
# Or set the HCCL operating parameter (AI_CPU expansion mode). The AI_CPU mode environment variable cannot be set simultaneously with other modes
# export HCCL_OP_EXPANSION_MODE="AI_CPU"
# Or set the HCCL operating parameter (AIV expansion mode). The AIV mode environment variable cannot be set simultaneously with other modes
# export HCCL_OP_EXPANSION_MODE="AIV"

echo "HCCL-VM environment configured successfully!"

```

### 4.5 Tool Specification Constraints

**Supported Operator Types**:

The tool supports the following operator types: allgather/allreduce/alltoall/reduce/reduce_scatter/scatter/alltoallv.

**Supported Data Types**:

The HCCL-VM tool supports the following data types: int8/int16/int32/fp16/fp32/uint8/uint16/uint32/bfp16/hif8/fp8e4m3/fp8e5m2/fp8e8m0.

The HCCL-VM Runner plugin supports the following data types:

| ReduceOp | DataType                           |
| -------- | ---------------------------------- |
| `ADD`    | `int8/int16/int32/uint8` |
| `MIN`    | `int8/int16/int32/uint8` |
| `MAX`    | `int8/int16/int32/uint8` |

**Hardware Specifications**:

Currently, this tool supports only Ascend 950 series chips. A single server supports a maximum of 8 cards. For more than 8 cards, cross-server operation is required. The built-in default storage is suitable for communicators with 1 to 128 cards and does not require database configuration.

### 4.6 HCCL-VM Plugin Features

#### 4.6.1 Runner Plugin

The Runner plugin simulates the execution of tasks generated by HCCL business orchestration and outputs output data.
The simulated runner plugin is **disabled** by default during hccl_test case execution. After the hccl_test case calls the operator interface, it waits for the operator task to complete through the aclrtSynchronizeStream interface. The simulated runner tool waits until all ranks are in the waiting state, then starts simulated execution of all rank tasks. After execution completes, it notifies each rank's case to continue execution.
After the case execution completes, users can view the input buffer and output buffer data for each rank through the "all_rank_input_output.txt" file in the execution directory. This feature is not enabled by default. Users can enable it through the corresponding command before case execution.

**Installation and Uninstallation**:

The Runner plugin supports installation and uninstallation through the `hccl-vm plugin install/uninstall` command. Install the runner plugin after entering the hccl-vm tool command line and before executing the case. Subsequent execution will then run the runner.

```bash
# Install the runner plugin
(hvm)$> hccl-vm plugin install @runner

# Uninstall the runner plugin
(hvm)$> hccl-vm plugin uninstall @runner
```

#### 4.6.2 Checker Plugin

The Checker plugin is an algorithm analyzer plugin. Its function is to form a DAG graph from all tasks generated by HCCL, and to determine whether memory conflicts exist by analyzing the DAG graph. It also determines whether semantic errors exist by simulating the execution of the DAG graph.
The algorithm analyzer plugin is started and executed by the user through commands.

The Checker plugin currently uses Checker V3 for single-operator verification. Checker V3 big graph verification is enabled by default: multiple operators within each sync window are merged into one big graph to perform cross-operator synchronous resource conflict verification. The single-operator Checker V3 and big graph verification can be controlled independently through configuration parameters in `manifest.json`. The legacy Checker has been removed, and `enable_old_checker` is no longer a valid configuration item.

```json

# The configuration file is located at /pathto/hccl_vm_install/plugin/checker/manifest.json

{
  "name": "checker",        // Checker plugin name
  "version": "1.0.0",       // Checker plugin version
  "entry": "./checker",     // Checker plugin startup command
  "dependency": {
      "min_core_version": "1.0.0"
  },
  "setting": {              // Checker plugin configuration items
      "enable_new_checker": true,           // Whether to enable Checker V3 single-operator verification (enabled by default)
      "enable_big_graph_checker": true,     // Whether to enable Checker V3 big graph verification (enabled by default, independent of single-operator verification)
      "enable_insight_dump": false,         // Whether to output Checker V3 Insight data (disabled by default)
      "enable_memory_snapshot_dump": false, // Whether to output memory snapshot data (disabled by default; Insight data output must be enabled first)
      "enable_dag_graphviz_dump": false,     // Whether to output big graph Graphviz DOT files (disabled by default)
      "enable_ccu_resource_stats_dump": false // Whether to output CCU resource statistics data (disabled by default)
  }
}
```

### 4.7 OpenMPI and MPICH Environment Case Execution Differences

Before running hccl_test cases, users can determine which mpirun the current environment uses through the `which` command.

#### 4.7.1 Environment Variable Configuration Differences

The environment typically defaults to OpenMPI configuration. If users run cases with OpenMPI, no additional environment variable configuration is generally needed.
If users run cases with the MPICH environment, configure environment variables as follows:

```bash
# Configure MPICH environment variables
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH
export PATH=/usr/lib/mpich/bin:$PATH
```

#### 4.7.2 mpirun Command Parameter Differences

In the OpenMPI environment, users run hccl_test cases with the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun --allow-run-as-root --oversubscribe -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter Description**:

 - --allow-run-as-root: An OpenMPI-specific parameter that allows running MPI processes as the root user, used for running in environments without root permissions.
 - --oversubscribe: An OpenMPI-specific parameter that removes CPU slot limits, allowing a single node to launch [number of processes > number of CPU logical cores], that is, oversubscribing processes.
 - -np 2: Specifies the number of processes as 2, consistent with the number of nodes.

In the MPICH environment, users run hccl_test cases with the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter Description**:

 - -np 2: Specifies the number of processes as 2, consistent with the number of nodes.

### 4.8 Result Viewing

#### 4.8.1 Runner Plugin Results

If the runner plugin is installed by running `hccl-vm plugin install @runner` in the hccl-vm terminal, the runner plugin execution is automatically triggered after the operator flow completes. The final results depend on hccl_test for verification. Users should check the redirected log file for [error]-level logs and the final verification result:

```bash
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
64                | 1000.00        | 0.00006              | success
```

#### 4.8.2 Checker Plugin Results

After running `hccl-vm plugin run @checker` in the hccl-vm terminal, the Checker verification process and results are printed in the terminal. Users should check for [error]-level logs and the final verification result:

Big graph verification runs once per sync window. Big graph verification failure does not block other verification processes. Identify failures through `error`-level log messages such as `BigGraphCheckerV3 failed` or `Big graph sync-conflict check failed` in the logs.

```bash
[info][PID:144373][TID:144880][main.cc][RunChecker] [RunChecker] op[0] Checker Success.
```

---
### 4.9 Operating Modes (normal / check-only / estimate)

HCCL-VM has three operating modes. At any given time, exactly one mode is active. The default is `normal`:

| Mode | Purpose | Description |
| ---- | ---- | ---- |
| `normal` | Normal mode (default) | Real independent allocation and complete multi-rank communication with lossless result correctness |
| `check-only` | Verification-only mode | Large memory blocks reuse a 4 GB shared pool, significantly reducing memory usage. Only the Checker verification chain is guaranteed; buffer data content correctness is not guaranteed |
| `estimate` | Estimation mode | A single card (rank0) constructs phantom peers to skip real communication blocking points and quickly produce a task list without actual data transfer (link heartbeat and consistency verification must be disabled first; refer to [4.9.4](#494-estimation-mode-limitations)) |

#### 4.9.1 Specifying the Mode at Startup

Both the `start` and `run` (one-shot) subcommands support the `--check-only` / `--estimate` flags (mutually exclusive). Without any flag, the mode is `normal`:

```bash
# Start in verification-only mode
./hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only

# Start in estimation mode
./hccl-vm start ascend950_cluster_32_server_normal.yaml --estimate
```

#### 4.9.2 Querying and Switching Modes at Runtime

Within an already-started VM command line, users can query and switch modes in real time through `hccl-vm mode` without restarting the VM:

```bash
# Query: list all supported modes and the current mode (without arguments, equivalent to show; no side effects)
(hvm)$> hccl-vm mode show
Supported modes:
        check-only : Verification-only mode. Large memory blocks reuse the 4 GB shared pool HcclCommPool, significantly reducing /dev/shm usage. Only the verification chain is guaranteed; buffer data content correctness is not guaranteed.
        estimate : Estimation mode. A single card (rank0) constructs phantom peers to skip real communication blocking points and quickly produce a task list without actual data transfer.
        normal : Normal mode (default). Real independent allocation and complete multi-rank communication with lossless result correctness.
Current mode: normal

# Switch modes
(hvm)$> hccl-vm mode check-only
(chk-only)$> hccl-vm mode estimate
(estimate)$> hccl-vm mode normal
```

After a successful switch, the terminal prompt changes with the mode: normal -> `(hvm)$>`, check-only -> `(chk-only)$>`, estimate -> `(estimate)$>`.

**Switch Behavior and Constraints**:

- **Strict spelling**: Mode names must use exact lowercase-hyphen spelling. Variants such as `CHECK-ONLY`, `check_only`, and invalid strings are all rejected, and the current mode is not affected.
- **Idempotent**: Switching to the current mode is a no-op, and the tool indicates that the mode is already active.
- **Shared pool lifecycle**: When switching to check-only, the 4 GB shared reuse pool is automatically created (residual data is cleaned first). When switching away, the pool is automatically released. The pool does not exist in normal or estimate modes.
- **Unload runner before switching**: Before switching to a non-normal mode, if the runner plugin is loaded, it is unloaded first. If unloading fails, the switch is aborted. Switching to normal does not trigger unloading.
- **Switch atomicity**: If any step during the switch fails (pool creation failure, pool destruction failure, and so on), the switch is aborted with an error, and the mode and resources remain in their pre-switch state. Known limitation: the prompt may still change even when the switch fails. In that case, rely on the `hccl-vm mode show` query result.
- **Takes effect for new cases**: The new mode takes effect for business cases launched after a successful switch. Cases already running before the switch are not affected and continue to run in the mode they were started with.
- **Does not persist across restarts**: The mode does not persist across VM restarts. After a restart, the VM returns to the initial mode determined by the startup parameter (normal if not specified).

#### 4.9.3 Check-Only Mode Large Memory Reuse Description

The check-only mode is designed for large-scale cluster scenarios where only Checker verification is needed. When enabled, large memory block allocations reuse the same shared region `HcclCommPool`. All ranks share this region and are allowed to overwrite each other, significantly reducing `/dev/shm` usage. The content of large blocks is not guaranteed to be correct in this mode. This mode is suitable only for the Checker V3 verification chain that does not read buffer data. Do not enable this mode when numerically correct results are required. The reuse region default range is 200 MB to 4 GB. The 4 GB upper limit can be changed with the command "--kpoolsize N".

Allocations smaller than 200 MB always use real allocation. Single blocks larger than the reuse region upper limit are directly rejected with an error in check-only mode. The check-only mode and Runner are not mutually exclusive, but when check-only mode is enabled and Runner is installed, large block reuse still takes effect and may overwrite Runner data. The tool prints a warning.

**Reuse Region Size Configuration Description**
`--kpoolsize N` supports three entry points. The following examples set the reuse pool size to 5G:

```bash
# 1) Specify at startup (detached form)
./bin/hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only --kpoolsize 5G

# 2) At runtime, switch/reconfigure in the (chk-only)$> sub-shell
(chk-only)$> hccl-vm mode check-only --kpoolsize 5G

# 3) At runtime, switch/reconfigure from another mode sub-shell
(hvm)$>  hccl-vm mode check-only --kpoolsize 5G
```
**Configuration Behavior and Constraints:**
- **Strict spelling**: The option name must be exactly --kpoolsize. Uppercase letters or extra hyphens are not allowed, such as "--Kpoolsize" or "k_poolsize".
- **Idempotent**: The reuse region size can be changed while already in check-only mode.
- **Syntax**: The "--check-only" mode must be selected at startup before the size can be set. At runtime, the command must also follow "mode check-only" before "--kpoolsize" can be configured. It cannot be used alone.
- **Units**: Pure decimal numbers are supported, `--kpoolsize 2147483648`. K, M, and G units are also supported and are case-insensitive.
- **Upper and lower limits**: For "--kpoolsize N", when the provided N is less than 4G, the value defaults to 4G. The size cannot exceed the `/dev/shm` space limit.

#### 4.9.4 Estimation Mode Limitations

The estimate mode executes on a single card (rank0) with a constructed phantom peer. **The following two prerequisites must be met before using estimate mode**:

1. **Disable link heartbeat and consistency verification**: In estimate mode, no real peer participates on the link. Link heartbeat and consistency verification cannot complete normally and may cause false alarms or abnormal exits.

   ```bash
   export HCCL_DFS_CONFIG=cluster_heartbeat:off,inconsistent_check:off
   ```

   This environment variable must take effect before the business case starts. Both `start` and `run` (one-shot) methods apply. When modifying the `start.sh` example from the README, note that it disables only the heartbeat by default (`cluster_heartbeat:off`). For estimate mode, add `inconsistent_check:off` as shown above.

2. **The business case must start with a single process (`-np 1`)**: The estimate mode constructs only the rank0 execution path. Regardless of the cluster topology configuration scale, the mpirun command must specify `-np 1`. Other parameters and usage remain unchanged.

   ```bash
   mpirun --allow-run-as-root --oversubscribe -np 1 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt
   ```

Other limitations:

- Only the single-card (rank0) execution path is constructed, and real communication blocking points are skipped. This mode is used to quickly produce a task list without actual data transfer.
- When functional correctness verification or real communication behavior is needed, use `normal` mode instead.

#### 4.9.5 Estimation Plugin (@estimate) and Dependency Description

After estimate mode produces the task list, the estimation plugin can perform lightweight performance simulation (reading `data/hccl_vm_data.db`, building a task graph, and running discrete event simulation):

```bash
(estimate)$> hccl-vm plugin run @estimate
```

This plugin is implemented in Python (installation directory `plugin/light_performance/`; available only in estimate mode; mutually exclusive with @checker). Its runtime dependencies are **not distributed with the installation package** and must be pre-installed in the target environment:

- **python3**: >=3.8 (including the sqlite3 standard library);
- **simpy**: Discrete event simulation engine. Install by running `pip3 install simpy`. If missing, the plugin reports `ModuleNotFoundError: No module named 'simpy'` at startup.

Simulation results are output to the `plugin/light_performance/data/<timestamp>/` directory: `perf_profiling.json` (Chrome tracing format), single-operator trace files, and the runtime log `light_performance.log`.

### 4.10 Simulation Levels and Level1 Hijacking (Northbound Hijacking)

#### 4.10.1 Northbound Hijacking Mechanism

HCCL-VM simulates HCCL communication operators through `LD_PRELOAD` multi-layer hijacking: Level 0 hijacks the HCCL operator API; Level 1 hijacks the northbound interface that HCCL uses to call HComm, commonly known as northbound hijacking; Level 2 hijacks the southbound interfaces below HComm (HCCP/Runtime/Driver), commonly known as southbound hijacking. The default level is 2 (southbound hijacking), where operator execution still depends on the complete HComm runtime. When `--level 1` is specified to enable northbound hijacking, the stub function library entirely replaces the real HComm, records task metadata instead of performing real communication, and fully decouples HCCL operators from the HComm runtime. Northbound hijacking covers all four expansion modes: AIV, AICPU, CCU, and HostDPU. Checker verification capabilities also apply.

#### 4.10.2 Compilation: Compile Northbound Only (Level1 Trimmed Build)

Run `bash ./build.sh --level1` to compile only northbound-related code (proxy level0/1/2 + support libraries + hccl-vm + device side + aiv_kernel). The hcomm source repository becomes optional (when provided, checker/runner plugins are built together). The output `hccl-vm` supports only `--level 1`. A full build with `--full` supports `--level 1/2`.

#### 4.10.3 Runtime: Run Northbound (--level 1)

Prerequisites: `libhccl_proxy_level1.so` must exist in the installation directory `lib/<arch>/` (for AICPU mode, `lib/aarch64/libhccl_device_proxy_level1.so` is also required). Otherwise, startup fails with an error.

The `start`, `run`, and `mock-comm` subcommands all support the `--level 1` parameter. Environment variable configuration, case execution, and result viewing methods are identical to those of the default level 2.

***

## 5 Appendix

### Open-Source Third-Party Software Dependencies

When compiling this project, the required third-party open-source software dependencies are listed in the table below. For offline compilation scenarios, download and rename the software packages and place them in the third_party directory within this project. For module-specific dependencies and offline placement methods, refer to the README in the corresponding module directory.

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
| Topology | The device connection relationships |
| Rank | The process identifier in MPI |

---

**Document Version**: v1.3.
**Last Updated**: 2026-10-31.
