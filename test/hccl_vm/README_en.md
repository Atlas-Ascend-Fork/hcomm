# HCCL-VM Guide

## 1. Overview

HCCL-VM is a high-performance collective communication virtual execution environment for Huawei Ascend NPU cards. This tool enables HCCL collective communication operator development and functional verification without real Ascend hardware.

The tool provides three running modes: `normal` (real multi-card communication with lossless result correctness), `check-only` (verification only; large memory blocks reuse a shared pool to significantly reduce memory usage), and `estimate` (estimation; skips real communication blocking to quickly produce a task list). You can specify the mode at startup or switch it at runtime through the `hccl-vm mode` command. For details, see [Running Modes](#49-running-modes-normal--check-only--estimate).

The tool also provides two simulation levels: the default level 2 (southbound hijacking) and level 1 (northbound hijacking, specified with `--level 1`). For details, see [Simulation Levels and Level 1 Hijacking](#410-simulation-levels-and-level-1-hijacking-northbound-hijacking).

![hccl-vm GIF](docs/hccl-vm.gif)

## 2. Prerequisites

| Dependency | Version Requirement |
| -------- | ------------- |
| System architecture | x86_64 Ubuntu 22.04 or later |
| Specification constraints | Ascend 950; for other constraints, see [Tool Specification Constraints](#45-tool-specification-constraints) |
| QEMU (required only for AICPU mode on x86_64 hosts) | The system-provided qemu-user-static package is available directly. Versions earlier than 10.2.4 do not support limiting the translation code cache, so QEMU has a larger resident memory footprint. In large-scale scenarios, paging may occur and significantly affect performance. You can install version 10.2.4 or later following [2.3 QEMU Deployment](#23-qemu-deployment-required-only-for-aicpu-mode-on-x86_64-hosts) for optimization. Native ARM hosts do not require QEMU. Compilation (`build.sh`/`build_pkg.sh`) does not depend on a locally installed QEMU. |
| Python 3 (required only for the @estimate performance simulation plugin) | 3.8 or later (including the sqlite3 standard library); the simpy discrete event simulation package must also be preinstalled (`pip3 install simpy`). For details, see [4.9.5](#495-estimate-plugin-estimate-and-dependency-description). |

### 2.1 CANN Package Installation

Install the latest CANN Toolkit development kit and CANN ops operator package. [Download link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)

```bash
# Ensure that the installation packages have executable permissions.
chmod +x Ascend-cann-toolkit_9.1.0_linux-x86_64.run
chmod +x Ascend-cann-950-ops_9.1.0_linux-x86_64.run
# Installation commands
./Ascend-cann-toolkit_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
./Ascend-cann-950-ops_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
```


### 2.2 hccl_test Compilation

hccl_test is the official HCCL performance testing tool provided by Ascend. For details, see [HCCL Performance Test Tool](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/devaids/hccltool/HCCLpertest_16_0001.html). HCCL-VM supports running hccl_test samples in the virtual environment. First follow the [hccl_test Sample Build](#42-hccl-test-sample-build) section to compile the sample binary program.

Note: hccl_test is an optional dependency. For PyTorch samples, see [PyTorch Sample Examples](#33-pytorch-sample-examples).

### 2.3 QEMU Deployment (Required Only for AICPU Mode on x86_64 Hosts)

The AICPU expansion mode requires QEMU user-mode emulation (linux-user emulating aarch64) on x86_64 hosts. The system-provided qemu-user-static package is available directly. Versions earlier than **10.2.4** do not support the `-tb-size` option to limit the translation code cache, so QEMU has a larger resident memory footprint, which may seriously affect performance. Install version **10.2.4 or later** following these steps.

```bash
# 1) Install build dependencies (gcc/make/ninja are usually preinstalled with the system or this tool).
sudo apt-get install -y meson flex bison libglib2.0-dev libpcre2-dev libffi-dev

# 2) Obtain the official source code archive (version 10.2.4 shown as an example; 10.2.4 and later are all acceptable).
curl -fsSL -O https://download.qemu.org/qemu-10.2.4.tar.xz
tar -xf qemu-10.2.4.tar.xz && cd qemu-10.2.4

# 3) Build the aarch64 linux-user static version.
mkdir build && cd build
../configure --target-list=aarch64-linux-user --static --disable-system --disable-tools \
    --disable-docs --disable-guest-agent --disable-slirp
make -j$(nproc) qemu-aarch64

# 4) Deploy and configure (the deployment path is customizable; /opt/qemu-10.2.4 is used as an example).
sudo mkdir -p /opt/qemu-10.2.4/bin
sudo cp qemu-aarch64 /opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_TB_SIZE=16
```

Running behavior and prompts:

- When the device process starts, it prints the actual QEMU program path, version, and the requested `-tb-size` value.
- If `HCCL_VM_QEMU_PATH` is not set, is invalid, or points to an older QEMU version that does not support `-tb-size`, the tool **does not abort**. It automatically falls back (searches PATH for `qemu-aarch64-static` or omits `-tb-size`) and prints a WARN message containing the actual fallback program and version, along with a prompt that the translation cache is not limited and QEMU memory may be excessive. The AICPU mode fails to start only when no `qemu-aarch64-static` binary is found on the entire machine.
- `HCCL_VM_QEMU_TB_SIZE` specifies the QEMU translation code cache size in MiB. If not set, the default is `16`. A value of `0` delegates to the QEMU default policy (it does not disable the cache). Not limiting the cache may significantly increase the QEMU resident memory and trigger paging.

---

## 3. Quick Start

### One-Click Installation

After pulling the complete CheckerL2 source code, run the following command in the repository root directory:

```bash
bash ./hccl_vm_installer
```

### 3.1 Manual Build and Installation

```bash
# 1. Create a working directory.
mkdir -p /home/workspace
cd /home/workspace

# 2. Download the dependency source code.
git clone https://gitcode.com/cann/hccl.git
git clone https://gitcode.com/cann/hcomm.git

# 3. Install third-party dependencies.
sudo apt-get update
sudo apt install build-essential cmake libboost-all-dev rdma-core libibverbs-dev pkg-config gcc-aarch64-linux-gnu g++-aarch64-linux-gnu qemu-user-static binfmt-support

# 4. Compile the HCCL-VM tool. After downloading the hcomm code, the tool source code path is: /home/workspace/hcomm/test/hccl_vm
cd /home/workspace/hcomm/test/hccl_vm
source /home/workspace/Ascend/cann/set_env.sh
export HCCL_CODE_HOME=/home/workspace/hccl
export HCOMM_CODE_HOME=/home/workspace/hcomm
bash ./build.sh --full

# 5. Copy and extract aicpu_hcxx.tar.gz from the CANN installation directory.
bash build_pkg.sh
```

### 3.2 Usage Examples

#### 3.2.1 Environment Configuration

If the hccl_rootinfo.json file already exists in the /etc directory, manually delete it.

When the communicator scale exceeds 128 cards (129 to 1024), switch to MariaDB. See [MariaDB Database Usage](src/storage_backends/mariadb/README.md). After completing the deployment and setting `HCCL_VM_STORAGE_CONFIG`, return to this section. To restore the default SQLite database, run `unset` on that variable.
<!-- internal-only:end -->

#### 3.2.2 CCU Mode

1. Configure environment variables.

```bash
# Go to the tool installation directory.
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
```

2. Execute


```bash
# Go to the new bin directory before running hccl-vm.
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line.
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# To enable the runner plugin (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for the current operator execution (run the hccl_test sample in a cluster environment with one supernode, one server, and one NPU).
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run the checker verification.
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal.
(hvm)$> exit
```

3. Verify the hccl_test sample execution results
[Runner Results](#481-runner-plugin-results) 
[Checker Results](#482-checker-plugin-results)

#### 3.2.3 AICPU Mode

In AICPU expansion mode, the algorithm expansion steps run on the device side. Therefore, the hccl-vm tool must compile and emulate the HCCL device-side symbols. Because the device-side symbols are built for the ARM architecture, cross-compilation is required on x86 environments, and QEMU is required at runtime to emulate AICPU mode.

Running AICPU mode on an x86 host uses QEMU user-mode emulation. If `HCCL_VM_QEMU_PATH` is not configured, the tool automatically falls back to `qemu-aarch64-static` in PATH and prints a warning (including the actual version); this does not affect startup. If the system-provided version is earlier than 10.2.4, the translation code cache cannot be limited, memory usage is higher, and paging may occur in large-scale scenarios, affecting performance. Refer to [2.3 QEMU Deployment](#23-qemu-deployment-required-only-for-aicpu-mode-on-x86_64-hosts) to install version 10.2.4 or later and configure `HCCL_VM_QEMU_PATH`. Native ARM hosts do not require QEMU.

1. Configure environment variables.

```bash
# Go to the tool installation directory.
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AI_CPU"

# x86_64 host: QEMU program and translation code cache (MiB) configuration; see section 2.3.
export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_TB_SIZE=16
```

2. Execute

```bash
# Go to the new bin directory before running hccl-vm.
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line.
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# To enable the runner plugin (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for the current operator execution (run the hccl_test sample in a cluster environment with one supernode, one server, and one NPU).
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run the checker verification.
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal.
(hvm)$> exit
```

3. Verify the hccl_test sample execution results [Runner Results](#481-runner-plugin-results) [Checker Results](#482-checker-plugin-results)

#### 3.2.4 HostDPU Mode

Environment variable configuration and other runtime settings for HostDPU expansion mode are identical to those for AICPU mode. Only the following differences apply to cluster scale and communicator scale:

1. Use `./hccl-vm start ascend950_cluster_32_server_normal_hostdpu.yaml` for the cluster specification configuration.
2. The number of servers must be >= 2. Single-server communicators are not supported (configurations such as 112.yaml, 114.yaml, and 118.yaml are not supported).

#### 3.2.5 AIV Mode

1. Configure environment variables.

```bash
# Go to the tool installation directory.
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AIV"
```

2. Execute

```bash
# Go to the new bin directory before running hccl-vm.
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line.
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# To enable the runner plugin (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for the current operator execution (run the hccl_test sample in a cluster environment with one supernode, one server, and one NPU).
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run the checker verification.
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal.
(hvm)$> exit
```

3. Verify the hccl_test sample execution results [Runner Results](#481-runner-plugin-results) [Checker Results](#482-checker-plugin-results)

### 3.3 PyTorch Sample Examples

HCCL-VM supports running some PyTorch `aclgraph` graph-mode collective communication scripts. The script should use `torch.multiprocessing.spawn` to start each device process. For multi-server samples, you must start multiple Python main processes concurrently on the same host to simulate each server separately. For the complete description, see [PyTorch Mode Running Guide](docs/pytorch_guide.md).

#### 3.3.1 Environment Preparation

First complete the HCCL-VM build and installation and CANN environment configuration, and prepare a Python virtual environment with compatible versions of `torch` and `torch_npu` installed. PyTorch currently supports only the Ascend 950 series. To use the Ascend 960 series, install the A6 ops package and change `soc_version` in the topology cluster configuration to a real model such as `Ascend960DT_961BY`.

```bash
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin
./hccl-vm start ascend950_cluster_32_server_normal.yaml
```

After entering the `(hvm)$>` terminal, run the following commands. Subsequent communicator configuration, environment variable setup, and Python startup are all performed in this terminal so that the Python process inherits the simulation environment.

```bash
# Enable PyTorch adaptation.
export HCCL_VM_ENABLE_PYTORCH=1
# Activate the virtual environment (if any).
source /path/to/torch_test/bin/activate

# Install Runner when simulation execution is required.
hccl-vm plugin install @runner
```

#### 3.3.2 Single-Server Running

Take the `118` configuration as an example. It uses one server and eight devices to execute AllReduce SUM on 1024 int32 elements. The following command parameters correspond to the test script in the guide. When you use another script, adjust the parameters according to its definitions.

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

Take the `128` configuration as an example. It uses two servers, each with eight devices, for a total of 16 ranks. In the HCCL-VM terminal where environment preparation is complete, run:

```bash
hccl-vm mock-comm 128

# Start two servers concurrently. Set environment variables for each server and save the logs.
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

# Wait for both servers to finish, and check each log for exceptions and execution results.
wait
```

`HCCL_VM_SERVER_INDEX` is the 0-based server index in the current communicator configuration. It takes effect only when `HCCL_VM_ENABLE_PYTORCH=1` is set, and must match the `--server_index` of the corresponding script. Setting the environment variable inside parentheses prevents different server indices from overwriting each other.

All servers must use the same `--master_ip` and `--master_port`. The `--dev_num` and `--server_num` values must match the `mock-comm` configuration. The local logical device IDs of each server are always `0-7`. In this example, server 0 corresponds to global ranks `0-7`, and server 1 corresponds to global ranks `8-15`. The script computes global ranks assuming each server has the same number of devices. For other scales, select a communicator configuration where each server has the same number of devices.

#### 3.3.4 Result Viewing and Verification

After all processes finish, view the corresponding `server0.log` and `server1.log` (in multi-server scenarios). Pay attention to the exit status, exceptions, and related log output. Then run the Checker verification in the HCCL-VM terminal. For result descriptions, see [Checker Plugin Results](#482-checker-plugin-results).

```bash
hccl-vm plugin run @checker

# After all samples and verifications finish, exit the tool terminal.
exit
```

### 3.4 hccl Code Modification Verification Example

If you modify the CANN operator package code, such as adding a new algorithm type, follow these steps to ensure that your changes take effect. The build_pkg.sh script helps you package, install, and copy device-side dependency symbols. Before running it, set the environment variables:

```bash
# Assume that your CANN installation directory is /home/workspace/Ascend.
source /home/workspace/Ascend/cann/set_env.sh
# Configure the hccl code repository path.
export HCCL_CODE_HOME=/home/workspace/hccl
# Configure the hcomm code repository path.
export HCOMM_CODE_HOME=/home/workspace/hcomm
```

1. If you update or modify the CANN hccl repository code, run `bash build_pkg.sh --install hccl`.
2. If you update or modify the CANN hcomm repository code, run `bash build_pkg.sh --install hcomm`.
3. If you update or modify both the CANN hccl and hcomm repository code, run `bash build_pkg.sh --full`.
4. Follow the [Usage Examples](#32-usage-examples) steps and rerun the sample.

---

## 4. Detailed Guide

### 4.1 Tool Environment Variable Configuration

**HCCL-VM environment variable description**:

| Environment Variable | Purpose | Example |
| ------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------- |
| `HCCL_CODE_HOME` | Specifies the HCCL source code path for HCCL-VM compilation. Not configured by default. | `export HCCL_CODE_HOME=/home/workspace/hccl` |
| `HCOMM_CODE_HOME` | Specifies the HCOMM source code path for HCCL-VM compilation. Not configured by default. | `export HCOMM_CODE_HOME=/home/workspace/hcomm` |
| `HCCL_VM_ENABLE_PYTORCH` | When set to `1`, enables PyTorch adaptation. For the running method, see [PyTorch Sample Examples](#33-pytorch-sample-examples). | `export HCCL_VM_ENABLE_PYTORCH=1` |
| `HCCL_VM_SERVER_INDEX` | The server index simulated by the PyTorch process, starting from `0`. It takes effect only when `HCCL_VM_ENABLE_PYTORCH=1` is set, and must match the `--server_index` of the test script. | `export HCCL_VM_SERVER_INDEX=0` |
| `HCCLVM_ENABLE_DUMP_DATA` | 1. Enables performance simulation file output to disk. If enabled, three runner_hcclvm_*_data.bin files and one ccu_channel_jetty_config.xml file are output during test sample execution for performance simulation. <br> 2. Enables the Runner plugin to dump input and output data. If enabled, the input and output data of each operator is dumped to the all_rank_input_output.txt file during test sample execution. | `export HCCLVM_ENABLE_DUMP_DATA=1` to enable; `export HCCLVM_ENABLE_DUMP_DATA=0` or leave unset to disable. |
| `HCCL_VM_RES_STAT_DUMP` | Enables hijack-layer resource statistics to disk. After this variable is set, each device exports its resource counters (memory, Jetty, QP, Notify, AIV core, and others) to CSV when the business process finalizes. The output is written to the installation directory `data/stat/<podId>_<serverId>_<device ID>_res_<timestamp>.csv`. If not set (default), no export occurs. | `export HCCL_VM_RES_STAT_DUMP=1` |
| `HCCL_VM_QEMU_PATH` | Takes effect only in AICPU mode on x86_64 hosts. It is a non-empty absolute path to an external QEMU program. The target must be an executable regular file; a symbolic link that points to a regular file is allowed. If not set or invalid, the tool automatically falls back to searching PATH for `qemu-aarch64-static` and prints a WARN (including the actual program and version). If no binary is found on the entire machine, AICPU startup fails. For deployment steps, see [2.3 QEMU Deployment](#23-qemu-deployment-required-only-for-aicpu-mode-on-x86_64-hosts). | `export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static` |
| `HCCL_VM_QEMU_TB_SIZE` | Takes effect only in AICPU mode on x86_64 hosts. It specifies the QEMU translation code cache size in MiB. If not set, the default is `16`. A value of `0` explicitly passes `-tb-size 0` and delegates to the QEMU default policy (it does not disable the cache). Only decimal digit strings that parse to uint32_t are accepted; invalid values trigger a warning and fall back to the default 16. When pointing to an older QEMU that does not support `-tb-size`, the option is not passed and a warning about potentially excessive memory is printed. | `export HCCL_VM_QEMU_TB_SIZE=16` |
| `HCCL_UB_MULTI_CHANNEL_NUM` | An HCCL-side environment variable read by the proxy stub (the `HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM` query item of `HcclConfigGetInfo` uses the same convention). It specifies the number of parallel jetties on one channel. It is stored in `opDetails.ubMultiChannelNum` along with the operator. The `@estimate` lightweight performance simulation plugin uses it to compute port bandwidth utilization (for each tier of `multi_port_utilization_low`, `+= (n-1)*jetty_utilization_step`, capped at 1.0; see section 7.2 of `docs/light_performance/simulator_design.md`). If unset, invalid, or `0`, the value falls back to `1` (multi-jetty is not enabled; the original utilization table is used directly). | `export HCCL_UB_MULTI_CHANNEL_NUM=8` |

### 4.2 HCCL-Test Sample Build

The hccl_test sample source code resides in the CANN package installation directory. It supports compilation and running in both OpenMPI and MPICH environments. For runtime differences, see [OpenMPI and MPICH Environment Sample Running Differences](#47-openmpi-and-mpich-environment-sample-running-differences). This sample guide uses the OpenMPI environment as an example.

#### 4.2.1 OpenMPI Environment Compilation

1. Install OpenMPI

```bash
sudo apt-get update
sudo apt install openmpi-bin libopenmpi-dev
```

2. Compile hccl_test

```bash
# Change the CANN installation directory permissions.
chmod -R 755 /home/workspace/Ascend

# Go to the hccl_test sample source code directory.
cd /home/workspace/Ascend/cann/tools/hccl_test

# Set the CANN environment variables.
source /home/workspace/Ascend/cann/set_env.sh

# Temporarily modify the Makefile script.
if ! grep -q '\-lmpi_cxx' Makefile; then
    sed -i 's/-lmpi/-lmpi -lmpi_cxx/g' Makefile
fi

# Compile the hccl_test sample.
MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi make ASCEND_DIR=${ASCEND_HOME_PATH}
```

#### 4.2.2 MPICH Environment Compilation

Assume that the MPICH path is `/usr/lib/mpich`.

```bash
# Go to the hccl_test sample source code directory.
cd /home/workspace/Ascend/cann/tools/hccl_test

# Set the CANN environment variables.
source /home/workspace/Ascend/cann/set_env.sh

# Configure the environment variables.
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH

# Compile the hccl_test sample.
make MPI_HOME=/usr/lib/mpich/ ASCEND_DIR=${ASCEND_HOME_PATH}
```

### 4.3 Ascend Cluster Topology Configuration File Description

#### 4.3.1 Server/Pod Topology Configuration File Description

An Ascend cluster topology consists of one or more Server/Pod sub-topologies combined according to CLOS hierarchical network rules. Therefore, before generating a cluster topology, you must confirm the topology type of each Server/Pod.
You can select a predefined topology type provided by the HCCL-VM tool, or customize a Server/Pod topology type according to the configuration file format requirements.

To describe the topology network relationships of a Server/Pod, the following aspects are included:

 - **Port configuration table**: Describes the physical port configuration of an NPU card, such as NPU-to-NPU direct connection ports (P2P) and NPU out-of-box ports (P2NET).
 - **Link configuration table**: Describes the connection relationships among all NPU cards within a Server/Pod, such as full mesh.
 - **PortBound**: Describes the binding relationships of certain ports on an NPU card, where multiple ports are bound into one PortGroup.

```yaml
type: "server_intra_links"
name: "ascend950_links_topo_demo"
description: "ascend950 chip normal topology connection relationship description file"

soc_version: "Ascend950"
device_num: 16

device_ports_allocate_map:
 # Port allocation table: 0: not used, 1: device direct connection, 2: device to switch, 3: d2h port
 #                 portId:  0  1  2  3  4  5  6  7  8
    - {die_id: 0, pin_map: [1, 1, 1, 0, 2, 2, 2, 2, 3]} # die0
    - {die_id: 1, pin_map: [0, 0, 0, 0, 0, 0, 0, 0, 0]} # die1

# port_group: describes which ports are merged into one portGroup. Ports in the same portGroup share the same IP address.
port_group:
    - {layer: 0, ports: ["0/4", "0/5", "0/6", "0/7"]}

links:
  # -- Description method: each line forms a full mesh among 8 devices --
  - link_mode: "fullmesh"
    connections:
      # The following example indicates: devices 0, 1, 2, 3 on die0 are fully connected through die0 ports.
      - {die_id: 0, devices_range: [0, 3]}
      - {die_id: 0, devices_range: [4, 7]}
      - {die_id: 0, devices_range: [8, 11]}
      - {die_id: 0, devices_range: [12, 15]}
  
  #- link_mode: "enum"
  #  device_to_device_links:
  #    The following example indicates: devices 0 and 1 use die0 ports to connect to devices 1, 3, 5, 7 on die1 ports respectively.
  #            That is: device0 connects to device1, device3, device5, device7; device1 connects to device3, device5, device7.
  #    - {src_die_id: 1, src_local_id_range: [0, 2], dst_die_id: 1, dst_local_id_range: [1, 3, 5, 7]}

  - link_mode: "enum"
    device_to_switch_links:
      # The following example indicates: devices 0 through 15 connect to the switch through die0 ports. Combined with portGroup, devices 0 through 15 connect to the switch through portGroup[0/4, 0/5, 0/6, 0/7].
      - {die_id: 0, devices_range: [0, 15]}

```

**Field description**:

 - **soc_version**: The chip model, such as `Ascend950`.
 - **device_num**: The total number of devices, determined by the chip model and topology type.
 - **device_ports_allocate_map**: The port allocation table, describing the port configuration of each die. 1 indicates a device direct connection port, 2 indicates a device-to-switch port, and 3 indicates a d2h port.
 - **port_group**: Describes which ports are merged into one portGroup. Ports in the same portGroup share the same IP address. Ports not configured default to one portGroup per port.
 - **links**: The link configuration table, describing the connection relationships among all NPU cards within a Server/Pod, as well as between NPUs and switches.
   - **NPU direct connection relationships**: The tool provides two methods to configure NPU direct connections:
    - **link_mode == "fullmesh"**: All devices are fully connected based on the ports of one die. When new typical connection methods are added in the future, new link_mode types such as "ring" can be added.
    - **link_mode == "enum"**: Enumeration method. When the NPU connection method within a Server/Pod is complex, all link relationships can be enumerated for description.
   - **NPU-to-switch connection relationships**: You can use the enumeration method to configure the connection relationships between NPUs and switches.
 - **device_to_device_links**: Describes the connection relationships between NPUs.
 - **device_to_switch_links**: Describes the connection relationships between NPUs and switches.

#### 4.3.2 Cluster Topology Configuration File Description

The Ascend cluster network consists of one or more Server/Pod sub-topologies combined according to CLOS hierarchical network rules. You can select different Server/Pod topology types based on cluster scale and requirements.

You can customize a cluster topology configuration file following this format:

```yaml
name: "ascend950_cluster_32_server_normal"
description: "Ascend 950 normal network: 32 super nodes, each with one server"

# Total number of super nodes
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

The preceding configuration file describes a cluster topology containing 4 super nodes, 32 servers, and a total of 128 NPU cards. Each Server/Pod uses the ascend950_server_topo_normal topology type.

**Field description**:

 - **super_node_num**: The total number of super nodes.
 - **server_num**: The total number of servers/pods.
 - **server_list**: The configuration information of each Server/Pod, including the super node ID, device ID range, chip model, and Server/Pod topology configuration file path.

#### 4.3.3 Communicator Configuration File Description

In an Ascend cluster environment, you must select different communicator configuration files based on the communicator requirements of the operator to execute.

The tool provides the hccl-vm mock-comm command to read and configure the operator communicator configuration file. The communicator configuration file uses YAML format, and its path is hccl_vm_install/config/topo_meta. If the directory does not contain a corresponding communicator configuration file, you must create one first.

The hccl-vm tool supports asymmetric topology communicator configurations, as shown below:

```yaml
# 1. Global statistics: podNum, serNum, and rankNum are all less than 1024
meta:
  podNum: 1  # Total number of super nodes
  serNum: 2  # Total number of servers
  rankNum: 6 # Total number of ranks

# 2. Detailed topology structure
topology:
  - podId: 0
    servers:
      - serId: 0
        # Local IDs of ranks actually run by each server
        ranks: [0, 2]
      - serId: 1
        # Local IDs of ranks actually run by each server
        ranks: [1, 3, 5, 7]
```

**Precautions**:

- When you configure a communicator, the tool regenerates the topo.json and ranktable.json files based on the specified communicator configuration ID.
- In the preceding communicator configuration YAML file, the ranks field represents the list of local IDs (that is, physical IDs of devices) of ranks actually run by each server.

When the communicator scale is large, the server selection fields and rank selection fields in the servers entries can be freely combined. The following fields are supported:

- **Server selection fields** (choose one; cannot be configured simultaneously):
  - `serId`: A single server ID or a list of server IDs (such as `3` or `[0, 1]`; a list represents each enumerated server).
  - `serId_range`: A server ID range, with value `[<start id>, <end id>]` (closed interval; list form only), representing all servers from start to end under the current super node.
- **Rank selection fields** (choose one; cannot be configured simultaneously):
  - `ranks`: The list of local IDs of ranks actually run by each server.
  - `ranks_range`: A local ID range, with value `[<start id>, <end id>]` (closed interval; list form only), representing all local IDs from start to end selected by each server.

The following example (2 super nodes, 4 servers total, 32 ranks) shows multiple combinations used together in the same file:

```yaml
# 1. Global statistics: podNum, serNum, and rankNum are all less than 1024
meta:
  podNum: 2  # Total number of super nodes
  serNum: 4  # Total number of servers
  rankNum: 32 # Total number of ranks

# 2. Detailed topology structure
topology:
  - podId: 0
    servers:
      # serId as an ID list represents server 0 and server 1 under pod 0.
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

- You cannot configure serId and serId_range at the same time, and you cannot configure ranks and ranks_range at the same time.
- serId_range and ranks_range support only the `[<start id>, <end id>]` list form (closed interval). The `<start id>-<end id>` format is not supported. The start id cannot be greater than the end id, and the range width (end id - start id + 1) cannot exceed 1024.
- One serId_range entry expands to all servers with IDs from start to end under the super node. A serId list entry represents each enumerated server, and each server has the same ranks.
- If a server entry does not configure a rank selection field, the ranks of that server is an empty list. If no server selection field is configured, serId defaults to 0.

#### 4.3.4 topo and ranktable.json File Description

You do not need to create the `topo.json` and `ranktable.json` files manually. The tool generates them automatically based on the following information:

- **Topology configuration ID**: The ID specified by the user at startup (such as 112, 113, etc.)
- **Chip type**: The chip type automatically identified based on the running environment.

Although the tool automatically generates the configuration files, understanding their structure helps you understand the topology configuration.

**topo.json structure**:

`topo.json` describes the connection relationships of all devices within one server:

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

- `server.device_count`: The total number of devices.
- `server.groups`: Device grouping information.
- `ports`: Port configuration.
  - `usage`: Port purpose (`peer2peer` indicates device-to-device connection, `peer2net` indicates external connection)
- `links`: Link configuration.
  - `link_type`: Link type (`PEER2PEER` or `PEER2NET`)
  - `topo_type`: Topology type (`1DMESH`, `CLOS`, etc.)

**ranktable.json structure**:

`ranktable.json` describes the devices and IP mapping used for the current run:

```json
{
  "version": "1.0",
  "server_count": 1,
  "device_count": 8,
  "server_list": [
    {
      "server_id": 0,
      "device_id": 0,
      "device_ip": "192.168.1.10",
      "port": "2222"
    }
  ]
}
```

**Field description**:

- `server_count`: The number of servers.
- `device_count`: The total number of devices.
- `server_list`: The server and device list.
  - `device_ip`: The device IP address.
  - `port`: The device port number.

### 4.4 hccl_config.sh File Description

The hccl_config.sh file contains the environment variable configuration required for running HCCL_Test samples. The environment variables in this file are the same as those used for Hccl_Test samples on a real machine.
You must modify the hccl_config.sh script according to your samples and requirements to configure the HCCL sample running environment variables.

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

  # Always return 0 whether or not files were deleted, to ensure the script continues.
  return 0
}

# Clean redundant files in the data/ directory (temporary files generated in CCU mode).
cd "${HCCL_VM_INSTALL_DIR}/data" 2>/dev/null && {
  remove_files_by_prefix "sqe_info_rank_"
  remove_files_by_prefix "mc_instr_info_rank_"
  rm -f "all_rank_input_output.txt"
  cd "${HCCL_VM_INSTALL_DIR}"
}

# Set the CANN environment variables.
source /home/workspace/Ascend/cann/set_env.sh

# Disable the hccl heartbeat function.
export HCCL_DFS_CONFIG=cluster_heartbeat:off

# Set the HCCL-VM installation path, inferred from the script's own location (compatible with bin/ and script/ subdirectories).
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

# Configure LD_LIBRARY_PATH.
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH

# Set the ranktable.json file path (consistent with the mock-comm generation path).
export RANK_TABLE_FILE=${HCCL_VM_INSTALL_DIR}/data/ranktable.json

# Set the log level.
export ASCEND_GLOBAL_LOG_LEVEL=1

# Enable log output to the screen.
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# Set the HCCL running mode (CCU, AI_CPU, AIV and other expansion modes).
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
# Or set the HCCL running parameter (AI_CPU expansion mode). The AI_CPU mode environment variable cannot be set simultaneously with other modes.
# export HCCL_OP_EXPANSION_MODE="AI_CPU"
# Or set the HCCL running parameter (AIV expansion mode). The AIV mode environment variable cannot be set simultaneously with other modes.
# export HCCL_OP_EXPANSION_MODE="AIV"

echo "HCCL-VM environment configured successfully!"

```

### 4.5 Tool Specification Constraints

**Supported operator types**:

The tool supports the following operator types: allgather, allreduce, alltoall, reduce, reduce_scatter, scatter, and alltoallv.

**Supported data types**:

The HCCL-VM tool supports the following data types: int8, int16, int32, fp16, fp32, uint8, uint16, uint32, bfp16, hif8, fp8e4m3, fp8e5m2, and fp8e8m0.

The HCCL-VM Runner plugin supports the following data types:

| ReduceOp | DataType |
| -------- | ---------------------------------- |
| `ADD` | `int8/int16/int32/uint8` |
| `MIN` | `int8/int16/int32/uint8` |
| `MAX` | `int8/int16/int32/uint8` |

**Hardware specifications**:

Currently, this tool supports only the Ascend 950 chip. A single server supports up to 8 cards. For more than 8 cards, cross-server running is required. The built-in default storage applies to communicators with 1 to 128 cards and does not require database configuration.

### 4.6 HCCL-VM Plugin Functions

#### 4.6.1 Runner Plugin

The Runner plugin emulates the execution of tasks generated by Hccl business orchestration and outputs output data.
The Runner plugin is **disabled** by default during hccl_test sample execution. After the hccl_test sample calls the operator interface, it waits for the operator task to finish through the aclrtSynchronizeStream interface. The Runner tool waits until all ranks are in the waiting state, then starts simulated execution of all rank tasks. After execution completes, it notifies each rank's sample to continue.

After the sample finishes, you can view each rank's input buffer and output buffer data through the "all_rank_input_output.txt" file in the execution directory. This function is disabled by default. You can enable it with the corresponding command before running the sample.

**Installation and uninstallation**:

The Runner plugin supports installation and uninstallation through the `hccl-vm plugin install/uninstall` command. You must install the runner plugin after entering the hccl-vm tool command line and before executing the sample; only then will subsequent executions run the runner.

```bash
# Install the runner plugin.
(hvm)$> hccl-vm plugin install @runner

# Uninstall the runner plugin.
(hvm)$> hccl-vm plugin uninstall @runner
```

#### 4.6.2 Checker Plugin

The Checker plugin, or algorithm analyzer plugin, forms all tasks generated by hccl into a DAG, analyzes the DAG to detect memory conflicts, and simulates DAG execution to detect semantic errors and other problems.
The user starts the algorithm analyzer plugin manually through a command.

The Checker plugin currently uses Checker V3 to perform single-operator verification. Checker V3 big graph verification is enabled by default: multiple operators within each sync window are merged into one big graph to perform cross-operator synchronization resource conflict verification. You can control single-operator Checker V3 and big graph verification separately through configuration parameters in `manifest.json`. The old Checker version has been removed; `enable_old_checker` is no longer a valid configuration item.

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

### 4.7 OpenMPI and MPICH Environment Sample Running Differences

Before running hccl_test samples, you can use the which command to determine which mpirun the current environment uses.

#### 4.7.1 Environment Variable Configuration Differences

The environment usually configures OpenMPI by default. If you use OpenMPI to run samples, you generally do not need to configure additional environment variables.
If you use the MPICH environment to run samples, configure the environment variables as follows:

```bash
# Configure MPICH environment variables.
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH
export PATH=/usr/lib/mpich/bin:$PATH
```

#### 4.7.2 mpirun Command Parameter Differences

In the OpenMPI environment, run hccl_test samples with the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun --allow-run-as-root --oversubscribe -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter description**:

 - --allow-run-as-root: An OpenMPI-specific parameter that allows running MPI processes as the root user. It is used in environments without root permissions.
 - --oversubscribe: An OpenMPI-specific parameter that relaxes the CPU slot limit. It allows a single node to start [process count > CPU logical core count], that is, over-subscribing processes.
 - -np 2: Specifies the number of processes as 2, consistent with the number of nodes.

In the MPICH environment, run hccl_test samples with the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter description**:

 - -np 2: Specifies the number of processes as 2, consistent with the number of nodes.

### 4.8 Result Viewing

#### 4.8.1 Runner Plugin Results

If you install the plugin by running hccl-vm plugin install @runner in the hccl-vm terminal, the operator process automatically triggers the runner plugin after completion. The final result depends on hccl_test verification. You must check whether the redirected log file contains [error] level logs and the final verification result:

```bash
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
64                | 1000.00        | 0.00006              | success
```

#### 4.8.2 Checker Plugin Results

After you run hccl-vm plugin run @checker in the hccl-vm terminal, the Checker verification process and results are printed in the terminal. You must check whether [error] level logs exist and review the final verification result:

Big graph verification runs once per sync window. Big graph verification failure does not block other verification processes. You must identify it through `error` level logs such as `BigGraphCheckerV3 failed` or `Big graph sync-conflict check failed` in the log.

```bash
[info][PID:144373][TID:144880][main.cc][RunChecker] [RunChecker] op[0] Checker Success.
```

---
### 4.9 Running Modes (normal / check-only / estimate)

HCCL-VM has three running modes. At any time, one and only one mode is active. The default is `normal`:

| Mode | Purpose | Description |
| ---- | ---- | ---- |
| `normal` | Normal mode (default) | Real independent allocation and complete multi-rank communication; result correctness is lossless |
| `check-only` | Verification-only mode | Large memory blocks reuse a 4GB shared pool, significantly reducing memory usage. It guarantees only the Checker verification path; buffer data content correctness is not guaranteed |
| `estimate` | Estimation mode | Single card (rank0) constructs a phantom peer to skip real communication blocking points and quickly produces a task list. It does not actually run data transfer (link heartbeat and consistency verification must be disabled first; see [4.9.4](#494-estimate-mode-limitations)) |

#### 4.9.1 Specifying Mode at Startup

Both the `start` and `run` (one-shot) subcommands support the `--check-only` / `--estimate` flags (mutually exclusive). Without any flag, the mode is `normal`:

```bash
# Start in verification-only mode.
./hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only

# Start in estimation mode.
./hccl-vm start ascend950_cluster_32_server_normal.yaml --estimate
```

#### 4.9.2 Querying and Switching Mode at Runtime

In an already started VM command line, you can use `hccl-vm mode` to query and switch modes immediately without restarting the VM:

```bash
# Query: list all supported modes and the current mode (no argument is equivalent to show, with no side effects).
(hvm)$> hccl-vm mode show
Supported modes:
        check-only : Verification-only mode; large memory blocks reuse the 4GB shared pool HcclCommPool, significantly reducing /dev/shm usage. It guarantees only the verification path; buffer data content correctness is not guaranteed.
        estimate : Estimation mode; single card (rank0) constructs a phantom peer to skip real communication blocking points and quickly produce a task list. It does not actually run data transfer.
        normal : Normal mode (default); real independent allocation and complete multi-rank communication; result correctness is lossless.
Current mode: normal

# Switch modes.
(hvm)$> hccl-vm mode check-only
(chk-only)$> hccl-vm mode estimate
(estimate)$> hccl-vm mode normal
```

After a successful switch, the terminal prompt changes with the mode: normal -> `(hvm)$>`, check-only -> `(chk-only)$>`, estimate -> `(estimate)$>`.

**Switching behavior and constraints**:

- **Strict spelling**: Mode names must use lowercase hyphen spelling strictly. Variants such as `CHECK-ONLY`, `check_only`, and illegal strings are all rejected; the current mode is not affected.
- **Idempotent**: Switching to the current mode is a no-op, with a prompt that you are already in that mode.
- **Reuse pool created/torn down with mode**: When switching to check-only, the tool automatically creates the 4GB shared reuse pool (cleaning any residual data first); when switching away, the tool automatically releases it. The pool does not exist in normal or estimate mode.
- **Unload runner before switching**: Before switching to a non-normal mode, if the runner plugin is resident, it is unloaded first. If unloading fails, the switch is aborted. Switching to normal does not trigger unloading.
- **Switch atomicity**: If any step during the switch fails (pool creation failure, pool teardown failure, etc.), an error is reported and the switch is aborted; mode and resources remain in their pre-switch state. Known limitation: when a switch fails, the prompt may still change. In this case, rely on the query result of `hccl-vm mode show`.
- **Applies to new samples**: The new mode applies to business samples started after the switch succeeds. Samples already running before the switch are not affected and continue to run in the mode they started with.
- **Does not persist across restarts**: The mode does not persist across VM restarts. After restart, the VM returns to the initial mode determined by the startup parameters (normal if not specified).

#### 4.9.3 Verification-Only Mode Large Memory Reuse Description

The check-only mode applies to large-scale cluster scenarios that run only Checker verification. When enabled, large memory requests from 200MB to 4GB reuse the same 4GB shared area `HcclCommPool`. All ranks share it and may overwrite each other, significantly reducing `/dev/shm` usage. In this mode, large block content correctness is not guaranteed. It applies only to the Checker V3 verification path that does not read buffer data. Do not enable it when numerically correct results are required.

Requests smaller than 200MB always use real allocation. Single blocks larger than 4GB are directly rejected with an error in check-only mode. The check-only mode and Runner are not mutually exclusive. However, if you install Runner while check-only mode is enabled, large block reuse still takes effect and may overwrite Runner data; the tool prints a warning.

**Reuse pool size configuration description**
`--kpoolsize` supports three entry points:

```bash
# 1) Specify at startup (discrete)
./bin/hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only --kpoolsize 5G

# 2) At runtime, switch or reconfigure in the (chk-only)$> sub-shell
(chk-only)$> hccl-vm mode check-only --kpoolsize 5G

# 3) At runtime, switch or reconfigure from another mode sub-shell
(hvm)$>  hccl-vm mode check-only --kpoolsize 5G
```
**Setting behavior and constraints:**
- **Strict spelling**: The option name must use uppercase hyphen spelling strictly, such as "--Kpoolsize" or "k_poolsize".
- **Idempotent**: You can change the reuse pool size while already in check-only mode.
- **Usage**: At startup, you must select the "--check-only" mode before you can set the size. At runtime, the command must follow "mode check-only" before you can configure "--Kpoolsize". It cannot be used alone.
- **Units**: Supports pure decimal numbers, such as `--kpoolsize 2147483648`. It also supports K, M, G units (case insensitive).
- **Setting bounds**: --kpoolsize N. When the passed N is less than 4G, it is adjusted to 4G by default. The size cannot exceed the `/dev/shm` space limit.

#### 4.9.4 Estimate Mode Limitations

The estimate mode runs on a single card (rank0), with a constructed phantom peer. **Before using estimate mode, you must meet the following two prerequisites**:

1. **Disable link heartbeat and consistency verification**: In estimate mode, no real peer participates on the link, so link heartbeat and consistency verification cannot complete normally and may cause false positives or abnormal exits.

   ```bash
   export HCCL_DFS_CONFIG=cluster_heartbeat:off,inconsistent_check:off
   ```

   This environment variable must take effect before the business sample starts. It applies to both the `start` and `run` (one-shot) methods. When modifying the `start.sh` example in the README, note that by default it disables only heartbeat (`cluster_heartbeat:off`). For estimate mode, add `inconsistent_check:off` as shown above.

2. **The business sample must start as a single process (`-np 1`)**: The estimate mode constructs only the rank0 execution path. Regardless of the cluster topology configuration scale, the mpirun command must specify `-np 1`. Other parameters and usage remain unchanged.

   ```bash
   mpirun --allow-run-as-root --oversubscribe -np 1 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt
   ```

Other limitations:

- It constructs only the single-card (rank0) execution path, skipping real communication blocking points. It is used to quickly produce a task list and does not actually run data transfer.
- When you require functional correctness verification or real communication behavior, use `normal` mode instead.

#### 4.9.5 Estimate Plugin (@estimate) and Dependency Description

After estimate mode produces the task list, you can run the estimate plugin to perform lightweight performance simulation (reads `data/hccl_vm_data.db`, builds a task graph, and runs discrete event simulation):

```bash
(estimate)$> hccl-vm plugin run @estimate
```

This plugin is implemented in Python (installation directory `plugin/light_performance/`; available only in estimate mode and mutually exclusive with @checker). Its runtime dependencies are **not distributed with the installation package** and must be preinstalled in the target environment:

- **python3**: 3.8 or later (including the sqlite3 standard library);
- **simpy**: Discrete event simulation engine; install with `pip3 install simpy`. If missing, the plugin immediately reports `ModuleNotFoundError: No module named 'simpy'` at startup.

Simulation results are output to the `plugin/light_performance/data/<timestamp>/` directory: `perf_profiling.json` (Chrome tracing format), single-operator trace files, and the running log `light_performance.log`.

### 4.10 Simulation Levels and Level 1 Hijacking (Northbound Hijacking)

#### 4.10.1 Northbound Hijacking Mechanism

HCCL-VM uses `LD_PRELOAD` multi-level hijacking to simulate HCCL communication operators: Level 0 hijacks the HCCL operator API; Level 1 hijacks the northbound interface that HCCL calls into HComm, commonly called northbound hijacking; Level 2 hijacks the southbound interface below HComm (HCCP/Runtime/Driver), commonly called southbound hijacking. The default level is 2 (southbound hijacking), where operator execution still depends on the complete HComm runtime. After specifying `--level 1` to enable northbound hijacking, the stub function library completely replaces the real HComm, recording task metadata instead of performing real communication, fully decoupling HCCL operators from the HComm runtime. Northbound hijacking covers all four expansion modes: AIV, AICPU, CCU, and HostDPU. Checker verification capability also applies.

#### 4.10.2 Build Phase: Build Only Northbound (level1 Trimmed Build)

Running `bash ./build.sh --level1` compiles only northbound-related code (proxy level0/1/2 + support libraries + hccl-vm + device side + aiv_kernel). The hcomm source repository becomes optional (when provided, checker/runner plugins are also built). The resulting `hccl-vm` supports only `--level 1`. Full builds such as `--full` support `--level 1/2`.

#### 4.10.3 Runtime Phase: Run Northbound (--level 1)

Prerequisite: `libhccl_proxy_level1.so` must exist in the installation directory `lib/<arch>/` (AICPU mode also requires `lib/aarch64/libhccl_device_proxy_level1.so`); otherwise, startup reports an error and exits.

The `start` / `run` / `mock-comm` subcommands all support the `--level 1` parameter. Environment variable configuration, sample execution, and result viewing are identical to the default level 2.

***

## 5 Appendix

### Open-Source Third-Party Software Dependencies

When compiling this project, the following open-source third-party software is required. For offline compilation scenarios, you can download and rename the packages, then place them in the third_party directory within this project. For module-specific dependencies and offline placement methods, see the README in the corresponding module directory.

| Open-Source Software | Version | Download Address |
| ------------ | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| CLI11 | 2.2.0 | [cli11-2.2.0.tar.gz](https://raw.gitcode.com/src-openeuler/cli11/blobs/58c912141164a5c0f0139bfa91343fefe151d525/cli11-2.2.0.tar.gz) |
| json | 3.11.3 | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| spdlog | 1.11.0 | [spdlog-v1.11.0.tar.gz](https://raw.gitcode.com/src-openeuler/spdlog/blobs/c2dfb1aca26c607393665c836155613ff283de66/v1.11.0.tar.gz) |
| yaml-cpp | 0.8.0 | [yaml-cpp-0.8.0.tar.gz](https://raw.gitcode.com/src-openeuler/yaml-cpp/blobs/d1ead4fff417073b9cdbf98b8b55eb0efc00b0ba/yaml-cpp-0.8.0.tar.gz) |
| googletest | 1.14.0 | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) |
| cann-cmake | master-044 | [cmake-master-044.tar.gz](https://raw.gitcode.com/cann/cmake/archive/refs/heads/master-044.tar.gz) |

### Glossary

| Term | Description |
| -------- | ------------------------------------------------------- |
| HCCL | Huawei Collective Communication Library |
| NPU | Neural Processing Unit |
| CANN | Compute Architecture for Neural Networks, the Huawei Ascend AI processor software stack |
| MPI | Message Passing Interface |
| CCU | Collective Communication Unit |
| Topology | The device connection relationships |
| Rank | The process number, its identifier in MPI |

---

**Document version**: v1.3.
**Last updated**: 2026-10-31.
