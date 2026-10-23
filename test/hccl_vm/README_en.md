# HCCL-VM Guide

## 1. Overview

HCCL-VM is a high-performance collective communication virtual execution environment for Huawei Ascend NPUs. This tool enables the development and functional verification of HCCL collective communication operators without real Ascend hardware.

The tool provides three operating modes: `normal` (real multi-NPU communication with lossless result correctness), `check-only` (validation only, where large memory blocks reuse a shared pool to significantly reduce memory usage), and `estimate` (estimation, which skips real communication blocking to quickly produce a task list). You can specify a mode at startup or switch modes at runtime through the `hccl-vm mode` command. For details, see [Operating Modes](#49-operating-modes-normal--check-only--estimate).

![hccl-vm GIF](docs/hccl-vm.gif)

## 2. Prerequisites

| Dependency | Version Requirement |
| -------- | ------------- |
| System architecture | x86_64 Ubuntu 22.04 or later |
| Specification constraints | Ascend950; for other items, refer to [Specification Constraints](#45-tool-specification-constraints) |
| QEMU (required only for the AICPU mode on an x86_64 host) | The qemu-user-static package that comes with the operating system is directly usable. Versions earlier than 10.2.4 cannot limit the translated code cache, causing QEMU to consume a large amount of resident memory. In large-scale scenarios, paging may occur and performance may degrade noticeably. In such cases, install version 10.2.4 or later as described in [2.3 QEMU Deployment](#23-qemu-deployment-required-only-for-the-aicpu-mode-on-an-x86_64-host) to apply the optimization. Native ARM hosts do not require QEMU. Compilation (`build.sh`/`build_pkg.sh`) does not depend on QEMU installed on the local host |
| Python3 (required only for the @estimate performance simulation plugin in the estimate mode) | 3.8 or later (including the sqlite3 standard library); simpy, a discrete-event simulation package, must also be preinstalled (`pip3 install simpy`). For details, see [4.9.5](#495-estimate-plugin-estimate-and-dependency-notes) |

### 2.1 CANN Package Installation

Install the latest CANN Toolkit development kit and CANN ops package. [Download link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)

```bash
# Grant executable permissions to the installation packages.
chmod +x Ascend-cann-toolkit_9.1.0_linux-x86_64.run
chmod +x Ascend-cann-950-ops_9.1.0_linux-x86_64.run
# Installation commands
./Ascend-cann-toolkit_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
./Ascend-cann-950-ops_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
```


### 2.2 hccl_test Compilation

hccl_test is the official HCCL performance test tool provided by Ascend. For details, see [HCCL Performance Test Tool](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/devaids/hccltool/HCCLpertest_16_0001.html). HCCL-VM supports running hccl_test cases in the virtual environment. First compile the case binaries as described in [hccl_test Case Build](#42-hccl-test-case-build).

Note: hccl_test is an optional dependency. For PyTorch cases, see [PyTorch Case Examples](#33-pytorch-case-examples).

### 2.3 QEMU Deployment (Required Only for the AICPU Mode on an x86_64 Host)

The AICPU expansion mode requires QEMU user-mode emulation (linux-user emulating aarch64) on an x86_64 host. The qemu-user-static package that comes with the operating system is directly usable. Versions earlier than **10.2.4** do not support the `-tb-size` option that limits the translated code cache. QEMU then consumes a large amount of resident memory, which may severely degrade performance. Install **version 10.2.4 or later** as follows.

```bash
# 1) Install build dependencies (gcc, make, and ninja are typically already installed on the operating system or with this tool).
sudo apt-get install -y meson flex bison libglib2.0-dev libpcre2-dev libffi-dev

# 2) Download the official source archive (10.2.4 shown as an example; any version 10.2.4 or later is acceptable).
curl -fsSL -O https://download.qemu.org/qemu-10.2.4.tar.xz
tar -xf qemu-10.2.4.tar.xz && cd qemu-10.2.4

# 3) Build the static aarch64 linux-user version.
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

Runtime behavior and messages:

- When a device process starts, it prints the QEMU program path, version, and requested `-tb-size` value that are actually used.
- If `HCCL_VM_QEMU_PATH` is unset, invalid, or points to an older QEMU that does not support `-tb-size`, the tool **does not abort**. It automatically falls back (looks up `qemu-aarch64-static` through PATH / omits `-tb-size`) and prints a WARN message that includes the actual fallback program and version, along with a reminder that the translation cache is unrestricted and QEMU memory usage may be excessive. AICPU mode startup fails only when no `qemu-aarch64-static` binary is found on the host.
- `HCCL_VM_QEMU_TB_SIZE` specifies the QEMU translated code cache size in MiB. When unset, the default is `16`; `0` delegates to QEMU's own default policy (it does not disable the cache). Leaving the cache unrestricted may significantly increase QEMU resident memory usage and trigger paging.

---

## 3. Quick Start

### One-click Installation

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

# 4. Build the HCCL-VM tool. After downloading the hcomm source code, the tool source path is: /home/workspace/hcomm/test/hccl_vm.
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

If the file /etc/hccl_rootinfo.json already exists, delete it manually.

#### 3.2.2 CCU Mode

1. Configure the environment variables.

```bash
# Go to the tool installation directory.
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
```

2. Run the tool.


```bash
# Go to the new bin directory to run hccl-vm.
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select an Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line.
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# Optionally enable the runner plugin.
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for the current operator (run the hccl_test case in a cluster environment of 1 super node, 1 server, and 1 NPU).
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run the checker for validation.
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal.
(hvm)$> exit
```

3. Verify the hccl_test case results.
[Runner Results](#481-runner-plugin-results)
[Checker Results](#482-checker-plugin-results)

#### 3.2.3 AICPU Mode

The AICPU expansion mode moves the algorithm expansion step to the device side. The hccl-vm tool therefore needs to compile and emulate the device-side symbols of HCCL. Because the device-side symbols target the ARM architecture, cross-compilation is required on an x86 environment, and QEMU is used at runtime to emulate the AICPU mode.

Running the AICPU mode on an x86 host relies on QEMU user-mode emulation. If `HCCL_VM_QEMU_PATH` is not configured, the tool automatically falls back to `qemu-aarch64-static` found in PATH and prints a warning (including the actual version); startup is not affected. If the operating system ships a version earlier than 10.2.4, the translated code cache cannot be limited and memory usage is relatively high. In large-scale scenarios, paging may occur and performance may degrade. In this case, refer to [2.3 QEMU Deployment](#23-qemu-deployment-required-only-for-the-aicpu-mode-on-an-x86_64-host) to install version 10.2.4 or later and configure `HCCL_VM_QEMU_PATH`. Native ARM hosts do not require QEMU.

1. Configure the environment variables.

```bash
# Go to the tool installation directory.
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AI_CPU"

# x86_64 host: configure the QEMU program and translated code cache size (MiB); see §2.3.
export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_TB_SIZE=16
```

2. Run the tool.

```bash
# Go to the new bin directory to run hccl-vm.
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select an Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line.
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# Optionally enable the runner plugin.
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for the current operator (run the hccl_test case in a cluster environment of 1 super node, 1 server, and 1 NPU).
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run the checker for validation.
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal.
(hvm)$> exit
```

3. Verify the hccl_test case results. [Runner Results](#481-runner-plugin-results) [Checker Results](#482-checker-plugin-results)

#### 3.2.4 HostDPU Mode

The environment variable configuration and other runtime settings of the HostDPU expansion mode are the same as those of the AICPU mode. Only the following differences exist in cluster scale and communicator scale:

1. Use `./hccl-vm start ascend950_cluster_32_server_normal_hostdpu.yaml` for the cluster specification configuration.
2. The number of servers must be >= 2; single-server communicators are not supported (for example, 112.yaml, 114.yaml, and 118.yaml are not supported).

#### 3.2.5 AIV Mode

1. Configure the environment variables.

```bash
# Go to the tool installation directory.
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AIV"
```

2. Run the tool.

```bash
# Go to the new bin directory to run hccl-vm.
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select an Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line.
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# Optionally enable the runner plugin.
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for the current operator (run the hccl_test case in a cluster environment of 1 super node, 1 server, and 1 NPU).
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run the checker for validation.
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal.
(hvm)$> exit
```

3. Verify the hccl_test case results. [Runner Results](#481-runner-plugin-results) [Checker Results](#482-checker-plugin-results)

### 3.3 PyTorch Case Examples

HCCL-VM supports running a subset of PyTorch `aclgraph` graph-mode collective communication scripts. Scripts must use `torch.multiprocessing.spawn` to start one process per device. For multi-server cases, start multiple Python main processes concurrently on the same host to simulate each server. For the complete description, see [PyTorch Mode Guide](docs/pytorch_guide.md).

#### 3.3.1 Environment Preparation

First complete the HCCL-VM build and installation and the CANN environment configuration, and prepare a Python virtual environment with compatible versions of `torch` and `torch_npu` installed. PyTorch currently supports only the Ascend950 series. To use the Ascend960 series, install the A6 ops package and change `soc_version` in the cluster topology configuration to an actual model such as `Ascend960DT_961BY`.

```bash
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin
./hccl-vm start ascend950_cluster_32_server_normal.yaml
```

After entering the `(hvm)$>` terminal, run the following commands. The communicator configuration, environment variable setup, and Python startup are all performed inside this terminal so that the Python process inherits the emulated environment.

```bash
# Enable PyTorch adaptation.
export HCCL_VM_ENABLE_PYTORCH=1
# Activate the virtual environment (if applicable).
source /path/to/torch_test/bin/activate

# Install Runner when emulation-based execution is needed.
hccl-vm plugin install @runner
```

#### 3.3.2 Single-Server Execution

The `118` configuration is used as an example: 1 server, 8 devices, running an AllReduce SUM on 1024 int32 elements. The following parameters correspond to the test script described in the guide. When using a different script, adjust the parameters according to its definition.

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

#### 3.3.3 Multi-Server Execution

The `128` configuration is used as an example: 2 servers, 8 devices per server, 16 ranks in total. In the HCCL-VM terminal where the environment has been prepared, run:

```bash
hccl-vm mock-comm 128

# Start two servers concurrently. Set environment variables for each and save their logs.
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

`HCCL_VM_SERVER_INDEX` is the zero-based server index in the current communicator configuration. It takes effect only when `HCCL_VM_ENABLE_PYTORCH=1` is set and must match the `--server_index` value of the corresponding script. Setting the environment variable inside a subshell prevents the indices of different servers from overwriting each other.

All servers must use the same `--master_ip` and `--master_port`. `--dev_num` and `--server_num` must match the `mock-comm` configuration. The local logical device IDs of each server are always `0–7`. In this example, server 0 corresponds to global ranks `0–7`, and server 1 corresponds to global ranks `8–15`. The script computes global ranks under the assumption that each server has the same number of devices. For other scales, select a communicator configuration in which all servers have the same number of devices.

#### 3.3.4 Result Viewing and Validation

After all processes finish, examine the corresponding `server0.log` and `server1.log` (in multi-server scenarios). Pay attention to the exit status, exceptions, and related log output. Then run the Checker validation inside the HCCL-VM terminal. For result details, see [Checker Plugin Results](#482-checker-plugin-results).

```bash
hccl-vm plugin run @checker

# After all cases and validations finish, exit the tool terminal.
exit
```

### 3.4 HCCL Code Modification Verification Example

If you modify the CANN operator package code (for example, by adding a new algorithm type), follow these steps to ensure that your changes take effect. The `build_pkg.sh` script helps you build the package, install it, and copy the device-side dependency symbols. Before running it, set the following environment variables:

```bash
# Assume your CANN installation directory is /home/workspace/Ascend.
source /home/workspace/Ascend/cann/set_env.sh
# Set the HCCL repository path.
export HCCL_CODE_HOME=/home/workspace/hccl
# Set the hcomm repository path.
export HCOMM_CODE_HOME=/home/workspace/hcomm
```

1. If you updated or modified the CANN hccl repository, run `bash build_pkg.sh --install hccl`.
2. If you updated or modified the CANN hcomm repository, run `bash build_pkg.sh --install hcomm`.
3. If you updated or modified both the CANN hccl and hcomm repositories, run `bash build_pkg.sh --full`.
4. Refer to [Usage Examples](#32-usage-examples) and run the case again.

---

## 4. Detailed Guide

### 4.1 Tool Environment Variable Configuration

**HCCL-VM environment variable description**:

| Environment Variable | Purpose | Example |
| ------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------- |
| `HCCL_CODE_HOME` | HCCL source code path specified when HCCL-VM is compiled. Not configured by default. | `export HCCL_CODE_HOME=/home/workspace/hccl` |
| `HCOMM_CODE_HOME` | HCOMM source code path specified when HCCL-VM is compiled. Not configured by default. | `export HCOMM_CODE_HOME=/home/workspace/hcomm` |
| `HCCL_VM_ENABLE_PYTORCH` | When set to `1`, PyTorch adaptation is enabled. For the execution method, see [PyTorch Case Examples](#33-pytorch-case-examples). | `export HCCL_VM_ENABLE_PYTORCH=1` |
| `HCCL_VM_SERVER_INDEX` | The zero-based server index emulated by the PyTorch process. It takes effect only when `HCCL_VM_ENABLE_PYTORCH=1` is set and must match the `--server_index` of the test script. | `export HCCL_VM_SERVER_INDEX=0` |
| `HCCLVM_ENABLE_DUMP_DATA` | 1. Enables performance simulation file output to disk. When enabled, three `runner_hcclvm_*_data.bin` files and one `ccu_channel_jetty_config.xml` file are generated during case execution for performance simulation. <br> 2. Enables Runner plugin input and output data dumping. When enabled, the input and output data of each operator is dumped to the `all_rank_input_output.txt` file during case execution. | `export HCCLVM_ENABLE_DUMP_DATA=1` enables the feature; `export HCCLVM_ENABLE_DUMP_DATA=0` or leaving the variable unset disables it. |
| `HCCL_VM_QEMU_PATH` | Effective only in the AICPU mode on an x86_64 host: a non-empty absolute path to an external QEMU program (the target must be a regular executable file; a symbolic link pointing to a regular file is also allowed). When unset or invalid, the tool automatically falls back to searching `qemu-aarch64-static` in PATH and prints a WARN message (including the actual program and version). If no suitable binary is found on the host, AICPU startup fails. For deployment steps, see [2.3 QEMU Deployment](#23-qemu-deployment-required-only-for-the-aicpu-mode-on-an-x86_64-host). | `export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static` |
| `HCCL_VM_QEMU_TB_SIZE` | Effective only in the AICPU mode on an x86_64 host: the QEMU translated code cache size in MiB. When unset, the default is `16`. `0` explicitly passes `-tb-size 0` and lets QEMU apply its own default policy (it does not disable the cache). Only decimal strings that can be parsed to a uint32_t value are accepted; invalid values trigger a warning and fall back to the default of 16. When the configured QEMU does not support `-tb-size`, this option is not passed and a warning is printed about potentially excessive memory usage. | `export HCCL_VM_QEMU_TB_SIZE=16` |

### 4.2 HCCL-Test Case Build

The hccl_test source code is located in the CANN installation directory. It supports compilation and execution in both OpenMPI and MPICH environments. For runtime differences, see [Differences Between OpenMPI and MPICH Case Execution](#47-differences-between-openmpi-and-mpich-case-execution). This guide uses the OpenMPI environment as an example.

#### 4.2.1 OpenMPI Environment Compilation

1. Install OpenMPI.

```bash
sudo apt-get update
sudo apt install openmpi-bin libopenmpi-dev
```

2. Compile hccl_test.

```bash
# Change the permissions of the CANN installation directory.
chmod -R 755 /home/workspace/Ascend

# Go to the hccl_test source directory.
cd /home/workspace/Ascend/cann/tools/hccl_test

# Set the CANN environment variables.
source /home/workspace/Ascend/cann/set_env.sh

# Temporarily modify the Makefile.
if ! grep -q '\-lmpi_cxx' Makefile; then
    sed -i 's/-lmpi/-lmpi -lmpi_cxx/g' Makefile
fi

# Compile hccl_test cases.
MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi make ASCEND_DIR=${ASCEND_HOME_PATH}
```

#### 4.2.2 MPICH Environment Compilation

Assume the MPICH path is `/usr/lib/mpich`.

```bash
# Go to the hccl_test source directory.
cd /home/workspace/Ascend/cann/tools/hccl_test

# Set the CANN environment variables.
source /home/workspace/Ascend/cann/set_env.sh

# Configure environment variables.
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH

# Compile hccl_test cases.
make MPI_HOME=/usr/lib/mpich/ ASCEND_DIR=${ASCEND_HOME_PATH}
```

### 4.3 Ascend Cluster Topology Configuration File Description

#### 4.3.1 Server/Pod Topology Configuration File Description

An Ascend cluster topology is composed of one or more Server/Pod sub-topologies, combined according to the CLOS layered network rules. Before generating a cluster topology, you must confirm the topology type of each Server/Pod.
You can either select a predefined topology type provided by the HCCL-VM tool or define your own Server/Pod topology type according to the configuration file format requirements.

Describing the topology of a Server/Pod includes the following aspects:

 - **Port configuration table**: describes the physical port configuration of an NPU, such as NPU-to-NPU direct-connection ports (P2P) and NPU-to-external-network ports (P2NET).
 - **Link configuration table**: describes the connections among all NPUs within a Server/Pod, such as a full mesh.
 - **PortBound**: describes the binding relationship among certain ports of an NPU, where multiple ports are bound into one PortGroup.

```yaml
type: "server_intra_links"
name: "ascend950_links_topo_demo"
description: "ascend950 chip common topology connection description file"

soc_version: "Ascend950"
device_num: 16

device_ports_allocate_map:
 # Port allocation table: 0: not used, 1: device-to-device direct, 2: device-to-switch, 3: d2h port
 #                 portId:  0  1  2  3  4  5  6  7  8
    - {die_id: 0, pin_map: [1, 1, 1, 0, 2, 2, 2, 2, 3]} # die0
    - {die_id: 1, pin_map: [0, 0, 0, 0, 0, 0, 0, 0, 0]} # die1

# port_group: describes which ports are merged into one portGroup; ports in the same portGroup share the same IP address.
port_group:
    - {layer: 0, ports: ["0/4", "0/5", "0/6", "0/7"]}

links:
  # -- Description method: each row contains 8 devices that form a full mesh --
  - link_mode: "fullmesh"
    connections:
      # The following example indicates: devices 0, 1, 2, 3 on die0 are fully connected through the ports on die0.
      - {die_id: 0, devices_range: [0, 3]}
      - {die_id: 0, devices_range: [4, 7]}
      - {die_id: 0, devices_range: [8, 11]}
      - {die_id: 0, devices_range: [12, 15]}
  
  #- link_mode: "enum"
  #  device_to_device_links:
  #    The following example indicates: devices 0 and 1, through ports on die0, connect to devices 1, 3, 5, and 7 through ports on die1, respectively.
  #    That is: device 0 connects to device 1, device 3, device 5, and device 7; device 1 connects to device 3, device 5, and device 7.
  #    - {src_die_id: 1, src_local_id_range: [0, 2], dst_die_id: 1, dst_local_id_range: [1, 3, 5, 7]}

  - link_mode: "enum"
    device_to_switch_links:
      # The following example indicates: devices 0 through 15 connect to the switch through ports on die0. Combined with portGroup, devices 0 through 15 connect to the switch through portGroup[0/4, 0/5, 0/6, 0/7].
      - {die_id: 0, devices_range: [0, 15]}

```

**Field description**:

 - **soc_version**: the chip model, such as `Ascend950`.
 - **device_num**: the total number of devices, determined by the chip model and topology type.
 - **device_ports_allocate_map**: the port allocation table, which describes the port configuration of each die. 1 indicates a device-to-device direct-connection port; 2 indicates a device-to-switch port; 3 indicates a d2h port.
 - **port_group**: describes which ports are merged into one portGroup. Ports in the same portGroup share the same IP address. For ports not listed, each port forms its own portGroup by default.
 - **links**: the link configuration table, which describes the connections among all NPUs within a Server/Pod and between NPUs and switches.
   - **NPU direct-connection relationship**: the tool provides two methods to configure NPU direct connections:
    - **link_mode == "fullmesh"**: indicates that all devices are fully connected based on the ports of one die. New link_mode types, such as "ring", can be added for new typical connection patterns.
    - **link_mode == "enum"**: the enumeration method. When the NPU connections within a Server/Pod are complex, all link relationships can be enumerated to describe them.
   - **NPU-to-switch connection relationship**: you can use the enumeration method to configure the connections between NPUs and switches.
 - **device_to_device_links**: describes the connections between NPUs.
 - **device_to_switch_links**: describes the connections between NPUs and switches.

#### 4.3.2 Cluster Topology Configuration File Description

An Ascend cluster network is composed of one or more Server/Pod sub-topologies, combined according to the CLOS layered network rules. You can select different Server/Pod topology types based on the cluster scale and requirements.

You can customize the cluster topology configuration file in the following format:

```yaml
name: "ascend950_cluster_32_server_normal"
description: "Ascend 950 normal network: 32 super nodes, each super node contains 1 server"

# Total number of super nodes
super_node_num: 4
# Total number of servers/pods
server_num: 32
server_list:
  # Servers 0-7: all use the ascend950_server_topo_normal topology type.
  - {super_pod_id: 0, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 1, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 2, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 3, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}

```

The preceding configuration file describes a cluster topology that contains 4 super nodes and 32 servers, totaling 128 NPUs. Each Server/Pod uses the ascend950_server_topo_normal topology type.

**Field description**:

 - **super_node_num**: the total number of super nodes.
 - **server_num**: the total number of servers/pods.
 - **server_list**: the configuration of each Server/Pod, including the super node ID, device ID range, chip model, and Server/Pod topology configuration file path.

#### 4.3.3 Communicator Configuration File Description

In an Ascend cluster environment, you must select a communicator configuration file based on the communicator required by the operator to execute.

The tool provides the `hccl-vm mock-comm` command to read and configure the operator communicator configuration file. The communicator configuration file is in YAML format and located in `hccl_vm_install/config/topo_meta`. If no corresponding communicator configuration file exists in that directory, you must create one first.

The hccl-vm tool supports asymmetric topology communicator configurations, as shown below:

```yaml
# 1. Global statistics: podNum, serNum, and rankNum are all less than 1024.
meta:
  podNum: 1  # Total number of super nodes
  serNum: 2  # Total number of servers
  rankNum: 6 # Total number of ranks

# 2. Detailed topology structure
topology:
  - podId: 0
    servers:
      - serId: 0
        # The local IDs of the ranks that actually run on each server
        ranks: [0, 2]
      - serId: 1
        # The local IDs of the ranks that actually run on each server
        ranks: [1, 3, 5, 7]
```

**Precautions**:

- When configuring a communicator, the tool regenerates the `topo.json` and `ranktable.json` files based on the specified communicator configuration ID.
- In the preceding YAML communicator configuration file, the `ranks` field lists the local IDs (that is, the physical IDs of the devices) of the ranks that actually run on each server.

When the communicator scale is large, the server selection field and rank selection field within a servers entry can be freely combined. The following fields are supported:

- **Server selection field** (choose one; cannot be configured simultaneously):
  - `serId`: a single server ID or a list of server IDs (for example, `3` or `[0, 1]`, where a list enumerates each server);
  - `serId_range`: a server ID range, in the form `[<start_id>, <end_id>]` (closed interval, list format only), which selects all servers with IDs from start to end under the specified super node.
- **Rank selection field** (choose one; cannot be configured simultaneously):
  - `ranks`: the list of local IDs of the ranks that actually run on each server;
  - `ranks_range`: a local ID range, in the form `[<start_id>, <end_id>]` (closed interval, list format only), which selects all local IDs from start to end on each server.

The following example (2 super nodes, 4 servers in total, 32 ranks) shows several combinations used together within the same file:

```yaml
# 1. Global statistics: podNum, serNum, and rankNum are all less than 1024.
meta:
  podNum: 2  # Total number of super nodes
  serNum: 4  # Total number of servers
  rankNum: 32 # Total number of ranks

# 2. Detailed topology structure
topology:
  - podId: 0
    servers:
      # serId is an ID list, representing server 0 and server 1 under pod 0.
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

- `serId` and `serId_range` cannot be configured simultaneously; `ranks` and `ranks_range` cannot be configured simultaneously.
- `serId_range` and `ranks_range` support only the `[<start_id>, <end_id>]` list format (closed interval). The `<start_id>-<end_id>` format is not supported. The start ID cannot be greater than the end ID, and the range width (end_id - start_id + 1) cannot exceed 1024.
- One `serId_range` entry expands to all servers with IDs from start to end under the specified super node. A `serId` list entry enumerates each listed server, and the ranks of each server are identical.
- If no rank selection field is configured in a server entry, the ranks of that server are an empty list. If no server selection field is configured, `serId` defaults to 0.

#### 4.3.4 topo.json and ranktable.json File Description

You do not need to create `topo.json` and `ranktable.json` manually. The tool generates them automatically based on the following information:

- **Topology configuration ID**: the ID specified by the user at startup (for example, 112 or 113)
- **Chip type**: the chip type automatically identified based on the runtime environment.

Although the tool generates the configuration files automatically, understanding their structure helps in understanding the topology configuration.

**topo.json structure**:

`topo.json` describes the connections among all devices within a server:

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

- `server.device_count`: the total number of devices.
- `server.groups`: device grouping information.
- `ports`: port configuration.
  - `usage`: the port purpose (`peer2peer` indicates a connection between devices; `peer2net` indicates an external connection)
- `links`: link configuration.
  - `link_type`: the link type (`PEER2PEER` or `PEER2NET`)
  - `topo_type`: the topology type (`1DMESH`, `CLOS`, and so on)

**ranktable.json structure**:

`ranktable.json` describes the devices and IP mappings used in the current run:

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

- `server_count`: the number of servers.
- `device_count`: the total number of devices.
- `server_list`: the list of servers and devices.
  - `device_ip`: the device IP address.
  - `port`: the device port number.

### 4.4 hccl\_config.sh File Description

The `hccl_config.sh` file contains the environment variable configuration required to run HCCL\_Test cases. The environment variables are the same as those used by HCCL\_Test cases in a real-hardware environment.
Modify the `hccl_config.sh` script according to your cases and requirements to configure the environment variables for running HCCL cases.

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

  # Return 0 regardless of whether any files were deleted, so the script continues.
  return 0
}

# Clean up redundant files in the data/ directory (temporary files generated in CCU mode).
cd "${HCCL_VM_INSTALL_DIR}/data" 2>/dev/null && {
  remove_files_by_prefix "sqe_info_rank_"
  remove_files_by_prefix "mc_instr_info_rank_"
  rm -f "all_rank_input_output.txt"
  cd "${HCCL_VM_INSTALL_DIR}"
}

# Set the CANN environment variables.
source /home/workspace/Ascend/cann/set_env.sh

# Disable the HCCL heartbeat feature.
export HCCL_DFS_CONFIG=cluster_heartbeat:off

# Set the HCCL-VM installation path, inferred from the script's own location (compatible with the bin/ and script/ subdirectories).
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

# Set the ranktable.json file path (consistent with the path generated by mock-comm).
export RANK_TABLE_FILE=${HCCL_VM_INSTALL_DIR}/data/ranktable.json

# Set the log level.
export ASCEND_GLOBAL_LOG_LEVEL=1

# Enable log output to the terminal.
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# Set the HCCL running mode (CCU, AI_CPU, AIV, and other expansion modes).
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
# Or set the HCCL running parameter (AI_CPU expansion mode). The AI_CPU mode environment variable cannot be set simultaneously with other modes.
# export HCCL_OP_EXPANSION_MODE="AI_CPU"
# Or set the HCCL running parameter (AIV expansion mode). The AIV mode environment variable cannot be set simultaneously with other modes.
# export HCCL_OP_EXPANSION_MODE="AIV"

echo "HCCL-VM environment configured successfully!"

```

### 4.5 Tool Specification Constraints

**Supported operator types**:

The tool supports the following operator types: allgather, allreduce, alltoall, reduce, reduce\_scatter, scatter, and alltoallv.

**Supported data types**:

The HCCL-VM tool supports the following data types: int8, int16, int32, fp16, fp32, uint8, uint16, uint32, bfp16, hif8, fp8e4m3, fp8e5m2, and fp8e8m0.

The HCCL-VM Runner plugin supports the following data types:

| ReduceOp | DataType                           |
| -------- | ---------------------------------- |
| `ADD`    | `int8/int16/int32/uint8` |
| `MIN`    | `int8/int16/int32/uint8` |
| `MAX`    | `int8/int16/int32/uint8` |

**Hardware specifications**:

The tool currently supports only Ascend950 chips. A single server supports a maximum of 8 cards; more than 8 cards require cross-server execution. The built-in storage by default supports communicators with 1 to 128 cards, and no database configuration is required.

### 4.6 HCCL-VM Plugin Functions

#### 4.6.1 Runner Plugin

The Runner plugin emulates the execution of tasks generated by Hccl business orchestration and produces output data.
The Runner plugin is **disabled by default** during hccl\_test case execution. After an hccl\_test case invokes an operator API, it waits for operator task completion through the `aclrtSynchronizeStream` API. The Runner tool waits until all ranks are in the waiting state, then starts emulating the task execution for all ranks. After execution completes, it notifies the cases on each rank to continue.
After case execution completes, you can view the input buffer and output buffer data for each rank in the `all_rank_input_output.txt` file in the execution directory. This feature is disabled by default; you can enable it before case execution with the corresponding command.

**Installation and Uninstallation**:

The Runner plugin supports installation and uninstallation through the `hccl-vm plugin install/uninstall` command. Install the Runner plugin after entering the hccl-vm command line and before running a case; subsequent execution then runs the runner.

```bash
# Install the Runner plugin
(hvm)$> hccl-vm plugin install @runner

# Uninstall the Runner plugin
(hvm)$> hccl-vm plugin uninstall @runner
```

#### 4.6.2 Checker Plugin

The Checker plugin, also known as the algorithm analyzer plugin, forms a DAG from all tasks generated by hccl. By analyzing the DAG, it detects potential memory conflicts; by emulating DAG execution, it detects semantic errors and other issues.
The algorithm analyzer plugin is launched manually by the user through a command.

The Checker plugin currently uses Checker V3 for single-operator validation. Checker V3 big-graph validation is enabled by default: multiple operators within each sync window are merged into a single big graph to perform cross-operator synchronous resource conflict checking. Single-operator Checker V3 and big-graph validation can be controlled independently through configuration parameters in `manifest.json`. The legacy Checker has been removed, and `enable_old_checker` is no longer a valid configuration item.

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
      "enable_new_checker": true,           // Whether to enable Checker V3 single-operator validation (enabled by default)
      "enable_big_graph_checker": true,     // Whether to enable Checker V3 big-graph validation (enabled by default, independent of single-operator validation)
      "enable_insight_dump": false,         // Whether to output Checker V3 Insight data (disabled by default)
      "enable_memory_snapshot_dump": false, // Whether to output memory snapshot data (disabled by default; Insight data output must be enabled first)
      "enable_dag_graphviz_dump": false     // Whether to output big-graph Graphviz DOT files (disabled by default)
  }
}
```

### 4.7 Differences Between OpenMPI and MPICH Case Execution

Before running hccl_test cases, you can determine which mpirun the current environment uses through the `which` command.

#### 4.7.1 Environment Variable Configuration Differences

The environment typically defaults to OpenMPI. If you use OpenMPI to run cases, you generally do not need to configure additional environment variables.
If you use the MPICH environment to run cases, configure the environment variables as follows:

```bash
# Configure MPICH environment variables.
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH
export PATH=/usr/lib/mpich/bin:$PATH
```

#### 4.7.2 mpirun Command Parameter Differences

In the OpenMPI environment, run hccl_test cases as follows:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun --allow-run-as-root --oversubscribe -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter description**:

 - --allow-run-as-root: an OpenMPI-specific parameter that allows MPI processes to run as the root user, used when the environment does not grant root permissions.
 - --oversubscribe: an OpenMPI-specific parameter that relaxes the CPU slot limit and allows a single node to start more processes than logical CPU cores, that is, to oversubscribe processes.
 - -np 2: specifies that the number of processes is 2, consistent with the number of nodes.

In the MPICH environment, run hccl_test cases as follows:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter description**:

 - -np 2: specifies that the number of processes is 2, consistent with the number of nodes.

### 4.8 Result Viewing

#### 4.8.1 Runner Plugin Results

If you install the plugin by running `hccl-vm plugin install @runner` in the hccl-vm terminal, the runner plugin is automatically triggered after the operator flow completes. The final result is validated by hccl_test. Pay attention to any `[error]` level logs and the final validation result in the redirected log file:

```bash
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
64                | 1000.00        | 0.00006              | success
```

#### 4.8.2 Checker Plugin Results

After you run `hccl-vm plugin run @checker` in the hccl-vm terminal, the Checker validation flow and results are printed in the terminal. Pay attention to any `[error]` level logs and the final validation result:

Big-graph validation runs once per sync window. Big-graph validation failure does not block other validation flows. Identify failures through `error` level logs such as `BigGraphCheckerV3 failed` or `Big graph sync-conflict check failed`.

```bash
[info][PID:144373][TID:144880][main.cc][RunChecker] [RunChecker] op[0] Checker Success.
```

---
### 4.9 Operating Modes (normal / check-only / estimate)

HCCL-VM has three operating modes. At any given time, exactly one mode is active. The default is `normal`:

| Mode | Purpose | Description |
| ---- | ---- | ---- |
| `normal` | Normal mode (default) | Real independent allocation and full multi-rank communication; result correctness is lossless |
| `check-only` | Validation-only mode | Large memory blocks reuse a 4 GB shared pool, which significantly reduces memory usage; only the Checker validation path is guaranteed, and buffer data content is not guaranteed to be correct |
| `estimate` | Estimate mode | A single card (rank0) constructs a phantom peer to skip real communication blocking points, quickly producing a task list without performing actual data transfers (requires disabling link heartbeat and consistency check first; see [4.9.4](#494-estimate-mode-limitations)) |

#### 4.9.1 Specifying a Mode at Startup

Both the `start` and `run` (one-shot) subcommands support the `--check-only` / `--estimate` flags (they are mutually exclusive). Without any flag, the mode is `normal`:

```bash
# Start in validation-only mode.
./hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only

# Start in estimate mode.
./hccl-vm start ascend950_cluster_32_server_normal.yaml --estimate
```

#### 4.9.2 Querying and Switching Modes at Runtime

In an already-started VM command line, you can query and switch modes at any time through `hccl-vm mode` without restarting the VM:

```bash
# Query: list all supported modes and the current mode (with no arguments, this is equivalent to show, with no side effects).
(hvm)$> hccl-vm mode show
Supported modes:
        check-only : Validation-only mode. Large memory blocks reuse the 4 GB shared pool HcclCommPool, which greatly reduces /dev/shm usage. Only the validation path is guaranteed; buffer data content is not guaranteed to be correct.
        estimate : Estimate mode. A single card (rank0) constructs a phantom peer to skip real communication blocking points, quickly producing a task list without performing actual data transfers.
        normal : Normal mode (default). Real independent allocation and full multi-rank communication; result correctness is lossless.
Current mode: normal

# Switch modes.
(hvm)$> hccl-vm mode check-only
(chk-only)$> hccl-vm mode estimate
(estimate)$> hccl-vm mode normal
```

After a successful switch, the terminal prompt changes with the mode: normal → `(hvm)$>`, check-only → `(chk-only)$>`, estimate → `(estimate)$>`.

**Switch behavior and constraints**:

- **Strict spelling**: the mode name must use the exact lowercase hyphenated spelling. Variants such as `CHECK-ONLY`, `check_only`, and other invalid strings are rejected, and the current mode remains unchanged.
- **Idempotent**: switching to the current mode is a no-op, and the tool reports that the mode is already active.
- **Shared pool lifecycle**: when switching to check-only, a 4 GB shared reuse pool is automatically created (residual state is cleaned up first); when switching away, the pool is automatically released. The pool does not exist in normal or estimate mode.
- **Unload runner before switching**: when switching to a non-normal mode, the runner plugin is unloaded first if it is loaded; if unloading fails, the switch is aborted. Switching to normal does not trigger unloading.
- **Atomicity of switching**: if any step of the switch fails (pool creation failure, pool release failure, and so on), the switch aborts with an error, and the mode and resources remain in the pre-switch state. Known limitation: when a switch fails, the prompt may still change; in this case, rely on the query result of `hccl-vm mode show`.
- **Applies to new cases**: the new mode applies to business cases launched after the switch succeeds; cases already running before the switch are not affected and continue to run in the mode they started with.
- **Not preserved across restarts**: the mode is not preserved across VM restarts. After a restart, the VM returns to the initial mode determined by the startup parameter (or normal if none was specified).

#### 4.9.3 Validation-Only Mode Large Memory Reuse Description

The check-only mode is used for large-scale clusters that run only Checker validation. When enabled, memory allocations between 200 MB and 4 GB reuse the same 4 GB shared region `HcclCommPool`, which is shared across ranks and may be overwritten by one another. This significantly reduces `/dev/shm` usage. The content of large blocks is not guaranteed to be correct in this mode; it applies only to the Checker V3 validation path, which does not read buffer data. Do not enable it when numerically correct results are required.

Allocations smaller than 200 MB always use real allocation; allocations larger than 4 GB per block are directly rejected with an error in check-only mode. check-only mode is not mutually exclusive with Runner, but if Runner is installed while check-only mode is enabled, large-block reuse still takes effect and may overwrite Runner data; the tool prints a warning in that case.

#### 4.9.4 Estimate Mode Limitations

estimate mode runs on a single card (rank0), with a constructed phantom peer on the other side. **Before using estimate mode, you must satisfy the following two prerequisites**:

1. **Disable link heartbeat and consistency check**: in estimate mode, no real peer participates on the link, so link heartbeat and consistency check cannot complete normally and may produce false positives or abnormal exits.

   ```bash
   export HCCL_DFS_CONFIG=cluster_heartbeat:off,inconsistent_check:off
   ```

   This environment variable must take effect before the business case starts. It applies to both `start` and `run` (one-shot) invocation paths. When adapting the `start.sh` example from the README, note that it disables only the heartbeat by default (`cluster_heartbeat:off`); for estimate mode, add `inconsistent_check:off` as shown above.

2. **The business case must start as a single process (`-np 1`)**: estimate mode constructs only the rank0 execution path. Regardless of how large the cluster topology configuration is, the mpirun command must specify `-np 1`. Other parameters and usage remain unchanged.

   ```bash
   mpirun --allow-run-as-root --oversubscribe -np 1 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt
   ```

Other limitations:

- Only the execution path for a single card (rank0) is constructed, and real communication blocking points are skipped. It is used to quickly produce a task list and does not perform actual data transfers.
- When functional correctness validation or real communication behavior is required, use `normal` mode instead.

#### 4.9.5 Estimate Plugin (@estimate) and Dependency Notes

After estimate mode produces a task list, you can run the estimate plugin to perform lightweight performance simulation (reading `data/hccl_vm_data.db`, building a task graph, and performing discrete-event simulation):

```bash
(estimate)$> hccl-vm plugin run @estimate
```

This plugin is implemented in Python (installation directory `plugin/light_performance/`, available only in estimate mode, and mutually exclusive with @checker). Its runtime dependencies **are not shipped with the installation package**, so the target environment must have them preinstalled:

- **python3**: 3.8 or later (including the sqlite3 standard library);
- **simpy**: the discrete-event simulation engine; install it by running `pip3 install simpy`. If missing, the plugin exits at startup with `ModuleNotFoundError: No module named 'simpy'`.

Simulation results are output to the `plugin/light_performance/data/<timestamp>/` directory: `perf_profiling.json` (Chrome tracing format), per-operator trace files, and the runtime log `light_performance.log`.

***

## 5 Appendix

### Open-Source Third-Party Software Dependencies

When compiling this project, the required open-source third-party software is listed in the following table. For offline compilation scenarios, download and rename the packages and place them under the `third_party` directory of this project. The module-specific dependencies and offline placement instructions are described in the README inside each module directory.

| Open-Source Software | Version | Download Link |
| ------------  | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| CLI11         | 2.2.0         | [cli11-2.2.0.tar.gz](https://raw.gitcode.com/src-openeuler/cli11/blobs/58c912141164a5c0f0139bfa91343fefe151d525/cli11-2.2.0.tar.gz) |
| json          | 3.11.3        | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| spdlog        | 1.11.0        | [spdlog-v1.11.0.tar.gz](https://raw.gitcode.com/src-openeuler/spdlog/blobs/c2dfb1aca26c607393665c836155613ff283de66/v1.11.0.tar.gz) |
| yaml-cpp      | 0.8.0         | [yaml-cpp-0.8.0.tar.gz](https://raw.gitcode.com/src-openeuler/yaml-cpp/blobs/d1ead4fff417073b9cdbf98b8b55eb0efc00b0ba/yaml-cpp-0.8.0.tar.gz) |
| googletest    | 1.14.0        | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) |
| cann-cmake    | master-044    | [cmake-master-044.tar.gz](https://raw.gitcode.com/cann/cmake/archive/refs/heads/master-044.tar.gz) |

### Glossary

| Term | Description                                                      |
| -------- | ------------------------------------------------------- |
| HCCL     | Huawei Collective Communication Library         |
| NPU      | Neural Processing Unit                          |
| CANN     | Compute Architecture for Neural Networks, the Huawei Ascend AI processor software stack |
| MPI      | Message Passing Interface                        |
| CCU      | Collective Communication Unit                    |
| Topology | Topology, the device connection relationship                                               |
| Rank     | Process ID, the identifier in MPI                                         |

---

**Document version**: v1.2.
**Last updated**: 2026-10-20.
