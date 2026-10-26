# HCCL-VM Guide

## 1. Overview

HCCL-VM is a virtual execution environment for high-performance collective communication on Huawei Ascend NPU cards. This tool enables the development and functional verification of HCCL collective communication operators without real Ascend hardware.

The tool provides three running modes: `normal` (real multi-card communication with no loss of result correctness), `check-only` (verification only, large memory blocks reuse a shared pool to significantly reduce memory usage), and `estimate` (estimation, skips real communication blocking points to quickly produce a task list). You can specify the mode at startup or switch it in real time through the `hccl-vm mode` command. For details, see [Running Modes](#49-running-modes-normal--check-only--estimate).

![hccl-vm GIF](docs/hccl-vm.gif)

## 2. Prerequisites

| Dependency | Version Requirement |
| -------- | ------------- |
| System architecture | x86_64 Ubuntu 22.04 and later |
| Specification constraints | Ascend 950. For other constraints, see [Tool Specification Constraints](#45-tool-specification-constraints) |
| QEMU (required only for AICPU mode on x86_64 host machine) | The system-provided qemu-user-static package is directly usable. When the version is lower than 10.2.4, the translated code cache cannot be limited and QEMU resident memory is large. Large-scale scenarios may cause paging and significantly affect performance. You can install version 10.2.4 or later following [2.3 QEMU Deployment](#23-qemu-deployment-only-required-for-aicpu-mode-on-x86_64-host-machine) for optimization. Native ARM host machines do not require QEMU. Compilation (`build.sh`/`build_pkg.sh`) does not depend on QEMU installed on the local machine |
| Python3 (required only for the estimate mode @estimate performance simulation plugin) | Version 3.8 or later (including the sqlite3 standard library). The simpy discrete event simulation package must also be pre-installed (`pip3 install simpy`). For details, see [4.9.5](#495-estimation-plugin-estimate-and-dependency-description) |

### 2.1 CANN Package Installation

Install the latest CANN Toolkit development kit package and CANN ops operator package. [Download link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)

```bash
# Ensure the installation packages have executable permissions
chmod +x Ascend-cann-toolkit_9.1.0_linux-x86_64.run
chmod +x Ascend-cann-950-ops_9.1.0_linux-x86_64.run
# Installation commands
./Ascend-cann-toolkit_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
./Ascend-cann-950-ops_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
```


### 2.2 hccl_test Compilation

hccl_test is the official HCCL performance testing tool provided by Ascend. For details, see [HCCL Performance Testing Tool](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/devaids/hccltool/HCCLpertest_16_0001.html). HCCL-VM supports running hccl_test samples in the virtual environment. First refer to the [hccl_test Sample Build](#42-hccl-test-sample-build) section to compile the sample binary program.

Note: hccl_test is an optional dependency. To run PyTorch samples, see [PyTorch Sample Examples](#33-pytorch-sample-examples).

### 2.3 QEMU Deployment (Only Required for AICPU Mode on x86_64 Host Machine)

The AICPU expansion mode requires QEMU user-mode emulation on an x86_64 host machine (linux-user emulation for aarch64). The system-provided qemu-user-static package is directly usable. When the version is lower than **10.2.4**, the `-tb-size` option to limit the translated code cache is not supported, QEMU resident memory is large, and performance may be severely affected. Install version **10.2.4 or later** following the steps below.

```bash
# 1) Build dependencies (gcc/make/ninja are generally already installed with the system or this tool)
sudo apt-get install -y meson flex bison libglib2.0-dev libpcre2-dev libffi-dev

# 2) Official source code archive (version 10.2.4 is used as an example; version 10.2.4 and later are both acceptable)
curl -fsSL -O https://download.qemu.org/qemu-10.2.4.tar.xz
tar -xf qemu-10.2.4.tar.xz && cd qemu-10.2.4

# 3) Build the aarch64 linux-user static version
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

Running behavior and prompts:

- The device process prints the actual QEMU program path, version, and requested `-tb-size` value at startup.
- When `HCCL_VM_QEMU_PATH` is not set, is invalid, or points to an old QEMU version that does not support `-tb-size`, the tool **does not abort**: it automatically falls back (searches PATH for `qemu-aarch64-static` / does not pass `-tb-size`) and prints a WARN message containing the actual fallback program and version, along with a prompt that the translation cache is not limited and QEMU memory may be excessive. AICPU mode startup fails only when no `qemu-aarch64-static` is found on the entire machine.
- `HCCL_VM_QEMU_TB_SIZE` specifies the QEMU translated code cache size in MiB. The default value is `16` when not set. A value of `0` delegates the decision to the default QEMU policy (this does not disable the cache). Not limiting the cache may significantly increase QEMU resident memory and cause paging.

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

# 2. Download the dependency source code
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

### 3.2 Usage Samples

#### 3.2.1 Environment Configuration

If the hccl_rootinfo.json file already exists in the /etc path, manually delete the file.

When the communicator scale exceeds 128 cards (129 to 1024), you must switch to the MariaDB database. See [MariaDB Database Usage](src/storage_backends/mariadb/README.md). After completing the deployment and setting `HCCL_VM_STORAGE_CONFIG`, return to this section. To restore the default SQLite database, run `unset` on that variable.
<!-- internal-only:end -->

#### 3.2.2 CCU Mode

1. Configure the environment variables.

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
# Enter the new bin directory to run hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# To enable the runner plugin (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for the current operator execution (run the hccl_test sample in a cluster environment with 1 supernode, 1 server, and 1 NPU)
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run the checker verification
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal
(hvm)$> exit
```

3. Verify the hccl_test sample running result
[Runner Result Viewing](#481-runner-plugin-result)
[Checker Result Viewing](#482-checker-plugin-result)

#### 3.2.3 AICPU Mode

The AICPU expansion mode places the algorithm expansion steps on the device side for execution. Therefore, the hccl-vm tool must compile and simulate the device-side symbols of HCCL. Because the device-side symbols use the ARM architecture, cross-compilation is required when compiling in an x86 environment, and QEMU is required to simulate AICPU mode at runtime.

Running AICPU mode on an x86 host machine uses QEMU user-mode emulation. When `HCCL_VM_QEMU_PATH` is not configured, the tool automatically falls back to `qemu-aarch64-static` in PATH and prints a warning (including the actual version). This does not affect startup. When the system-provided version is lower than 10.2.4, the translated code cache cannot be limited, memory usage is high, and large-scale scenarios may cause paging and affect performance. Refer to [2.3 QEMU Deployment](#23-qemu-deployment-only-required-for-aicpu-mode-on-x86_64-host-machine) to install version 10.2.4 or later and configure `HCCL_VM_QEMU_PATH`. Native ARM host machines do not require QEMU.

1. Configure the environment variables.

```bash
# Enter the tool installation directory
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AI_CPU"

# x86_64 host machine: QEMU program and translated code cache (MiB) configuration, see Section 2.3
export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_TB_SIZE=16
```

2. Execute

```bash
# Enter the new bin directory to run hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# To enable the runner plugin (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for the current operator execution (run the hccl_test sample in a cluster environment with 1 supernode, 1 server, and 1 NPU)
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run the checker verification
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal
(hvm)$> exit
```

3. Verify the hccl_test sample running result [Runner Result Viewing](#481-runner-plugin-result) [Checker Result Viewing](#482-checker-plugin-result)

#### 3.2.4 HostDPU Mode

The HostDPU expansion mode uses the same environment variable configuration as the AICPU mode at runtime. The only differences are in the cluster scale and communicator scale as follows:

1. The cluster specification configuration must use `./hccl-vm start ascend950_cluster_32_server_normal_hostdpu.yaml`
2. The number of servers must be greater than or equal to 2. Single-server communicators are not supported (112.yaml, 114.yaml, 118.yaml, and so on are not supported)

#### 3.2.5 AIV Mode

1. Configure the environment variables.

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
# Enter the new bin directory to run hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# To enable the runner plugin (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for the current operator execution (run the hccl_test sample in a cluster environment with 1 supernode, 1 server, and 1 NPU)
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run the checker verification
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal
(hvm)$> exit
```

3. Verify the hccl_test sample running result [Runner Result Viewing](#481-runner-plugin-result) [Checker Result Viewing](#482-checker-plugin-result)

### 3.3 PyTorch Sample Examples

HCCL-VM supports running some PyTorch `aclgraph` graph-mode collective communication scripts. The script must use `torch.multiprocessing.spawn` to start each device process. Multi-server samples must concurrently start multiple Python main processes on the same host machine to simulate each server separately. For the complete description, see [PyTorch Mode Running Guide](docs/pytorch_guide.md).

#### 3.3.1 Environment Preparation

First complete the HCCL-VM build and installation and the CANN environment configuration, and prepare a Python virtual environment with compatible versions of `torch` and `torch_npu` installed. PyTorch currently supports only the Ascend 950 series. To use the Ascend 960 series, you must install the A6 ops package and change `soc_version` in the topology cluster configuration to a real model such as `Ascend960DT_961BY`.

```bash
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin
./hccl-vm start ascend950_cluster_32_server_normal.yaml
```

After entering the `(hvm)$>` terminal, run the following commands. Complete the communicator configuration, environment variable settings, and Python startup in this terminal so that the Python process inherits the simulation environment.

```bash
# Enable PyTorch adaptation
export HCCL_VM_ENABLE_PYTORCH=1
# Activate the virtual environment (if applicable)
source /path/to/torch_test/bin/activate

# Install Runner when simulation execution is required
hccl-vm plugin install @runner
```

#### 3.3.2 Single-Server Running

The `118` configuration is used as an example, with 1 server and 8 devices, to execute AllReduce SUM on 1024 int32 elements. The following command parameters correspond to the test script in the guide. When using other scripts, adjust according to their parameter definitions.

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

The `128` configuration is used as an example, with 2 servers and 8 devices per server, totaling 16 ranks. Run the following commands in the HCCL-VM terminal where the environment preparation is complete:

```bash
hccl-vm mock-comm 128

# Concurrently start two servers, set environment variables separately, and save logs
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

# Wait for both servers to finish running, and check for exceptions and execution results in their respective logs
wait
```

`HCCL_VM_SERVER_INDEX` is the zero-based server index in the current communicator configuration. It takes effect only when `HCCL_VM_ENABLE_PYTORCH=1` and must match the `--server_index` of the corresponding script. Setting the environment variable inside the subshell prevents different server indexes from overwriting each other.

All servers must use the same `--master_ip` and `--master_port`. The `--dev_num` and `--server_num` values must match the `mock-comm` configuration. The local logical device IDs for each server are all `0-7`. In this example, server 0 corresponds to global ranks `0-7`, and server 1 corresponds to global ranks `8-15`. The script calculates global ranks based on the same number of devices per server. For other scales, select a communicator configuration where each server has the same number of devices.

#### 3.3.4 Result Viewing and Verification

After all processes finish, view the corresponding `server0.log` and `server1.log` (for multi-server scenarios). Pay attention to the exit status, exceptions, and related log output. Then run the Checker verification in the HCCL-VM terminal. For result descriptions, see [Checker Plugin Result](#482-checker-plugin-result).

```bash
hccl-vm plugin run @checker

# Exit the tool terminal after all samples and verifications are complete
exit
```

### 3.4 HCCL Code Modification Verification Example

If you have modified the CANN operator package code, such as adding a new algorithm type, follow these steps to ensure your modification takes effect. The build_pkg.sh script helps you perform packaging, installation, and copying of device-side dependency symbols. Before running the script, set the environment variables:

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
4. Refer to the [Usage Samples](#32-usage-samples) steps and run the sample again.

---

## 4. Detailed Guide

### 4.1 Tool Environment Variable Configuration

**HCCL-VM environment variable description**:

| Environment Variable | Description | Example |
| ------------------------- | --------------------------------------------------------------------------------------------------- | --------------------- |
| `HCCL_CODE_HOME` | Specifies the HCCL source code path for HCCL-VM compilation. Not configured by default. | `export HCCL_CODE_HOME=/home/workspace/hccl` |
| `HCOMM_CODE_HOME` | Specifies the HCOMM source code path for HCCL-VM compilation. Not configured by default. | `export HCOMM_CODE_HOME=/home/workspace/hcomm` |
| `HCCL_VM_ENABLE_PYTORCH` | When set to `1`, enables PyTorch adaptation. For the running method, see [PyTorch Sample Examples](#33-pytorch-sample-examples). | `export HCCL_VM_ENABLE_PYTORCH=1` |
| `HCCL_VM_SERVER_INDEX` | The server index simulated by the PyTorch process, starting from `0`. Takes effect only when `HCCL_VM_ENABLE_PYTORCH=1` and must match the `--server_index` of the test script. | `export HCCL_VM_SERVER_INDEX=0` |
| `HCCLVM_ENABLE_DUMP_DATA` | 1. Enables performance simulation file output to disk. When enabled, the test sample execution outputs 3 runner_hcclvm_*_data.bin files and 1 ccu_channel_jetty_config.xml file for performance simulation. <br> 2. Enables the Runner plugin to dump input and output data. When enabled, the input and output data of each operator is dumped to the all_rank_input_output.txt file during test sample execution. | `export HCCLVM_ENABLE_DUMP_DATA=1` enables the feature. `export HCCLVM_ENABLE_DUMP_DATA=0` or not configuring this environment variable disables the feature. |
| `HCCL_VM_QEMU_PATH` | Applies only to AICPU mode on an x86_64 host machine: a non-empty absolute path to an external QEMU program (the target is an executable regular file; symbolic links pointing to regular files are allowed). When not set or invalid, the tool automatically falls back to searching PATH for `qemu-aarch64-static` and prints a WARN message (including the actual program and version). AICPU startup fails when no such program is found on the entire machine. For deployment steps, see [2.3 QEMU Deployment](#23-qemu-deployment-only-required-for-aicpu-mode-on-x86_64-host-machine) | `export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static` |
| `HCCL_VM_QEMU_TB_SIZE` | Applies only to AICPU mode on an x86_64 host machine: the QEMU translated code cache size in MiB. The default value is `16` when not set. A value of `0` explicitly passes `-tb-size 0`, leaving the decision to the default QEMU policy (this does not disable the cache). Only decimal digit strings that can be parsed to uint32_t are accepted. Invalid values trigger a warning and fall back to the default value 16. When pointing to an old QEMU version that does not support `-tb-size`, this option is not passed and a warning is printed that memory may be excessive. | `export HCCL_VM_QEMU_TB_SIZE=16` |

### 4.2 HCCL-Test Sample Build

The hccl_test sample source code is located in the CANN package installation directory. It supports compilation and running in both OpenMPI and MPICH environments. For running differences, see [OpenMPI and MPICH Environment Sample Running Differences](#47-openmpi-and-mpich-environment-sample-running-differences). This guide uses the OpenMPI environment as an example.

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

# Enter the hccl_test sample source code directory
cd /home/workspace/Ascend/cann/tools/hccl_test

# Set the CANN environment variables
source /home/workspace/Ascend/cann/set_env.sh

# Temporarily modify the Makefile script
if ! grep -q '\-lmpi_cxx' Makefile; then
    sed -i 's/-lmpi/-lmpi -lmpi_cxx/g' Makefile
fi

# Compile the hccl_test sample
MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi make ASCEND_DIR=${ASCEND_HOME_PATH}
```

#### 4.2.2 MPICH Environment Compilation

Assume the mpich path is: `/usr/lib/mpich`.

```bash
# Enter the hccl_test sample source code directory
cd /home/workspace/Ascend/cann/tools/hccl_test

# Set the CANN environment variables
source /home/workspace/Ascend/cann/set_env.sh

# Configure the environment variables
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH

# Compile the hccl_test sample
make MPI_HOME=/usr/lib/mpich/ ASCEND_DIR=${ASCEND_HOME_PATH}
```

### 4.3 Ascend Cluster Topology Configuration File Description

#### 4.3.1 Server/Pod Topology Configuration File Description

An Ascend cluster topology is composed of one or more Server/Pod sub-topologies combined according to CLOS hierarchical network rules. Therefore, before generating the cluster topology, you must confirm the topology type of each Server/Pod.
You can select a predefined topology type provided by the HCCL-VM tool, or you can customize a Server/Pod topology type according to the configuration file format requirements.

Describing the topology network relationship of a Server/Pod mainly includes the following aspects:

 - **Port configuration table**: Describes the physical port configuration information of an NPU card, such as NPU-to-NPU direct connection ports (P2P) and NPU out-of-chassis ports (P2NET).
 - **Link configuration table**: Describes the connection relationships among all NPU cards within a Server/Pod, such as full mesh connections.
 - **PortBound**: Describes the binding relationships of certain ports on an NPU card, that is, multiple ports bound into one PortGroup.

```yaml
type: "server_intra_links"
name: "ascend950_links_topo_demo"
description: "ascend950 chip normal topology connection relationship description file"

soc_version: "Ascend950"
device_num: 16

device_ports_allocate_map:
 # port allocation table: 0: not used, 1: device direct connection, 2: device to switch, 3: d2h port
 #                 portId:  0  1  2  3  4  5  6  7  8
    - {die_id: 0, pin_map: [1, 1, 1, 0, 2, 2, 2, 2, 3]} # die0
    - {die_id: 1, pin_map: [0, 0, 0, 0, 0, 0, 0, 0, 0]} # die1

# port_group: describes which ports are merged into one portGroup. Ports in the same portGroup share the same IP address
port_group:
    - {layer: 0, ports: ["0/4", "0/5", "0/6", "0/7"]}

links:
  # -- Description method: each row contains 8 devices forming a full mesh --
  - link_mode: "fullmesh"
    connections:
      # The following example indicates: devices 0, 1, 2, 3 of die0 are all fully connected through the ports of die0
      - {die_id: 0, devices_range: [0, 3]}
      - {die_id: 0, devices_range: [4, 7]}
      - {die_id: 0, devices_range: [8, 11]}
      - {die_id: 0, devices_range: [12, 15]}
  
  #- link_mode: "enum"
  #  device_to_device_links:
  #    The following example indicates: devices 0 and 1 both use the ports of die0 to connect to devices 1, 3, 5, 7 through the ports of die1 respectively
  #            That is: device0 connects to device1, device3, device5, device7, and device1 connects to device3, device5, device7
  #    - {src_die_id: 1, src_local_id_range: [0, 2], dst_die_id: 1, dst_local_id_range: [1, 3, 5, 7]}

  - link_mode: "enum"
    device_to_switch_links:
      # The following example indicates: device0 to device15 all connect to the switch through the ports of die0. Combined with portGroup, device0 to device15 all connect to the switch through portGroup[0/4, 0/5, 0/6, 0/7].
      - {die_id: 0, devices_range: [0, 15]}

```

**Field description**:

 - **soc_version**: The chip model, such as `Ascend950`.
 - **device_num**: The total number of devices, determined by the chip model and topology type.
 - **device_ports_allocate_map**: The port allocation table, describing the port configuration of each die. A value of 1 indicates device direct connection ports, 2 indicates device to switch ports, and 3 indicates d2h ports.
 - **port_group**: Describes which ports are merged into one portGroup. Ports in the same portGroup share the same IP address. Ports not configured default to each port being its own portGroup.
 - **links**: The link configuration table, describing the connection relationships among all NPU cards within a Server/Pod and between NPUs and switches.
   - **NPU direct connection relationship**: The tool provides two methods to configure NPU direct connection relationships:
    - **link_mode == "fullmesh"**: Indicates that all devices are fully connected based on the ports of one die. New link_mode types such as "ring" can be added when new typical connection methods are introduced.
    - **link_mode == "enum"**: Enumeration method. When the NPU connection method within a Server/Pod is complex, all link relationships can be described by enumeration.
   - **NPU to switch connection relationship**: You can configure the connection relationship between NPUs and switches using the enumeration method.
  - **device_to_device_links**: Describes the connection relationships between NPUs.
  - **device_to_switch_links**: Describes the connection relationships between NPUs and switches.

#### 4.3.2 Cluster Topology Configuration File Description

An Ascend cluster network is composed of one or more Server/Pod sub-topologies combined according to CLOS hierarchical network rules. You can select different Server/Pod topology types based on the cluster scale and requirements.

You can customize a cluster topology configuration file according to the following configuration file format:

```yaml
name: "ascend950_cluster_32_server_normal"
description: "Ascend 950 normal networking: 32 supernodes, each supernode has 1 server"

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

The preceding configuration file describes a cluster topology containing 4 supernodes, 32 servers, and a total of 128 NPU cards. Each Server/Pod uses the ascend950_server_topo_normal topology type.

**Field description**:

 - **super_node_num**: The total number of supernodes.
 - **server_num**: The total number of servers/pods.
 - **server_list**: The configuration information of each Server/Pod, including the supernode ID, device ID range, chip model, and Server/Pod topology configuration file path.

#### 4.3.3 Communicator Configuration File Description

In an Ascend cluster environment, you must select different communicator configuration files based on the communicator required by the operator to be executed.

The tool provides the hccl-vm mock-comm command to read and configure the operator communicator configuration file. The communicator configuration file format is yaml, and the path is hccl_vm_install/config/topo_meta. If the corresponding communicator configuration file does not exist in the directory, you must create one first.

The hccl-vm tool supports asymmetric topology communicator configuration, as shown below:

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
        # The local IDs of the ranks actually running on each server
        ranks: [0, 2]
      - serId: 1
        # The local IDs of the ranks actually running on each server
        ranks: [1, 3, 5, 7]
```

**Precautions**:

- When configuring the communicator, the tool regenerates the topo.json and ranktable.json files based on the specified communicator configuration ID.
- In the preceding communicator configuration yaml file, the ranks field represents the list of local IDs (that is, the physical IDs of the devices) of the ranks actually running on each server.

When the communicator scale is large, the server selection fields and rank selection fields in the servers entries can be freely combined. The following fields are supported:

- **Server selection fields** (choose one; they cannot be configured simultaneously):
  - `serId`: A single server ID or a list of server IDs (such as `3` or `[0, 1]`; a list represents each enumerated server).
  - `serId_range`: A server ID range, with the value `[<start_id>, <end_id>]` (closed interval, only the list format is supported), representing all servers in the supernode with IDs from start to end.
- **Rank selection fields** (choose one; they cannot be configured simultaneously):
  - `ranks`: A list of local IDs of the ranks actually running on each server.
  - `ranks_range`: A local ID range, with the value `[<start_id>, <end_id>]` (closed interval, only the list format is supported), representing all local IDs from start to end selected for each server.

The following example (2 supernodes, 4 servers total, 32 ranks) demonstrates mixing multiple combinations within the same file:

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

- serId and serId_range cannot be configured simultaneously. ranks and ranks_range cannot be configured simultaneously.
- serId_range and ranks_range only support the `[<start_id>, <end_id>]` list format (closed interval). The `<start_id>-<end_id>` format is not supported. The start_id cannot be greater than the end_id, and the range width (end_id - start_id + 1) cannot exceed 1024.
- A serId_range entry expands to all servers in the supernode with IDs from start to end. A serId list entry represents each enumerated server, and the ranks for each server are identical.
- If no rank selection field is configured in a server entry, the ranks for that server is an empty list. If no server selection field is configured, the default serId is 0.

#### 4.3.4 topo and ranktable.json File Description

The `topo.json` and `ranktable.json` files do not need to be created manually. The tool automatically generates them based on the following information:

- **Topology configuration ID**: The ID specified by the user at startup (such as 112, 113)
- **Chip type**: The chip type automatically identified based on the running environment.

Although the tool automatically generates the configuration files, understanding their structure helps you understand the topology configuration.

**topo.json structure**:

The `topo.json` file describes the connection relationships of all devices within a server:

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
- `server.groups`: The device grouping information.
- `ports`: The port configuration.
  - `usage`: The port purpose (`peer2peer` indicates device-to-device connections, `peer2net` indicates external connections)
- `links`: The link configuration.
  - `link_type`: The link type (`PEER2PEER` or `PEER2NET`)
  - `topo_type`: The topology type (`1DMESH`, `CLOS`, and so on)

**ranktable.json structure**:

The `ranktable.json` file describes the devices and IP mappings used for the current run:

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

The hccl_config.sh file contains the environment variable configuration required for running HCCL_Test samples. The environment variables in this file are the same as those used for Hccl_Test samples in a real machine environment.
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

  # Return 0 regardless of whether files were deleted, to ensure the script continues
  return 0
}

# Clean up redundant files in the data/ directory (temporary files generated in CCU mode)
cd "${HCCL_VM_INSTALL_DIR}/data" 2>/dev/null && {
  remove_files_by_prefix "sqe_info_rank_"
  remove_files_by_prefix "mc_instr_info_rank_"
  rm -f "all_rank_input_output.txt"
  cd "${HCCL_VM_INSTALL_DIR}"
}

# Set the CANN environment variables
source /home/workspace/Ascend/cann/set_env.sh

# Disable the hccl heartbeat feature
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

# Set log output to stdout
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# Set the HCCL running mode (CCU, AI_CPU, AIV expansion modes, and so on)
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
# Or set the HCCL running parameter (AI_CPU expansion mode). The AI_CPU mode environment variable cannot be set simultaneously with other modes
# export HCCL_OP_EXPANSION_MODE="AI_CPU"
# Or set the HCCL running parameter (AIV expansion mode). The AIV mode environment variable cannot be set simultaneously with other modes
# export HCCL_OP_EXPANSION_MODE="AIV"

echo "HCCL-VM environment configured successfully!"

```

### 4.5 Tool Specification Constraints

**Supported operator types**:

The tool supports the following operator types: allgather/allreduce/alltoall/reduce/reduce_scatter/scatter/alltoallv.

**Supported data types**:

The HCCL-VM tool supports the following data types: int8/int16/int32/fp16/fp32/uint8/uint16/uint32/bfp16/hif8/fp8e4m3/fp8e5m2/fp8e8m0.

The HCCL-VM Runner plugin supports the following data types:

| ReduceOp | DataType |
| -------- | ---------------------------------- |
| `ADD` | `int8/int16/int32/uint8` |
| `MIN` | `int8/int16/int32/uint8` |
| `MAX` | `int8/int16/int32/uint8` |

**Hardware specifications**:

Currently, this tool only supports Ascend 950 chips. A single server supports a maximum of 8 cards. More than 8 cards require cross-server running. The built-in default storage applies to communicators with 1 to 128 cards and does not require database configuration.

### 4.6 HCCL-VM Plugin Functions

#### 4.6.1 Runner Plugin

The Runner plugin simulates the execution of tasks generated by HCCL service orchestration and outputs the output data.
The simulation runner plugin is **disabled** by default during hccl_test sample execution. After the hccl_test sample calls the operator interface, it waits for the operator task execution to complete through the aclrtSynchronizeStream interface. The simulation runner tool waits until all ranks are in the waiting state, then starts simulating the execution of tasks for all ranks. After execution completes, it notifies the samples on each rank to continue execution.
After the sample execution completes, you can view the input buffer and output buffer data for each rank through the "all_rank_input_output.txt" file in the execution directory. This feature is not enabled by default. You can enable it through the corresponding command before sample execution.

**Installation and uninstallation**:

The Runner plugin supports installation and uninstallation through the `hccl-vm plugin install/uninstall` command. You must install the runner plugin after entering the hccl-vm tool command line and before executing the sample. Only subsequent execution will run the runner.

```bash
# Install the runner plugin
(hvm)$> hccl-vm plugin install @runner

# Uninstall the runner plugin
(hvm)$> hccl-vm plugin uninstall @runner
```

#### 4.6.2 Checker Plugin

The Checker plugin, that is, the algorithm analyzer plugin: this function forms a DAG graph from all tasks generated by hccl and analyzes the DAG graph to determine whether memory conflicts exist. It also simulates the execution of the DAG graph to determine whether semantic errors and other issues exist.
The user starts the algorithm analyzer plugin through a command.

The Checker plugin currently uses Checker V3 to perform single-operator verification. The Checker V3 big graph verification is enabled by default: multiple operators within each sync window are merged into one big graph to perform cross-operator synchronous resource conflict verification. The single-operator Checker V3 and big graph verification can be controlled separately through configuration parameters in `manifest.json`. The old Checker version has been removed, and `enable_old_checker` is no longer a valid configuration item.

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
      "enable_dag_graphviz_dump": false     // Whether to output big graph Graphviz DOT files (disabled by default)
  }
}
```

### 4.7 OpenMPI and MPICH Environment Sample Running Differences

Before running hccl_test samples, you can use the which command to determine which mpirun the current environment uses.

#### 4.7.1 Environment Variable Configuration Differences

The environment generally has OpenMPI configured by default. If you use OpenMPI to run samples, you generally do not need to configure additional environment variables.
If you use the MPICH environment to run samples, configure the environment variables as follows:

```bash
# Configure mpich environment variables
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH
export PATH=/usr/lib/mpich/bin:$PATH
```

#### 4.7.2 mpirun Command Parameter Differences

In the OpenMPI environment, run hccl_test samples using the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun --allow-run-as-root --oversubscribe -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter description**:

 - --allow-run-as-root: An OpenMPI-specific parameter that allows running MPI processes as the root user, used for running in environments without root permissions.
 - --oversubscribe: An OpenMPI-specific parameter that removes the CPU slot limit and allows starting [number of processes > number of CPU logical cores] on a single node, that is, running processes in an oversubscribed configuration.
 - -np 2: Specifies the number of processes as 2, consistent with the number of nodes.

In the MPICH environment, run hccl_test samples using the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter description**:

 - -np 2: Specifies the number of processes as 2, consistent with the number of nodes.

### 4.8 Result Viewing

#### 4.8.1 Runner Plugin Result

If you install the plugin by running hccl-vm plugin install @runner in the hccl-vm terminal, the runner plugin execution is automatically triggered after the operator process execution completes. The final result depends on hccl_test for verification. You must check whether [error] level logs exist and review the final verification result in the redirected log file:

```bash
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
64                | 1000.00        | 0.00006              | success
```

#### 4.8.2 Checker Plugin Result

After running hccl-vm plugin run @checker in the hccl-vm terminal, the Checker verification process and result are printed in the terminal. You must check whether [error] level logs exist and review the final verification result:

The big graph verification runs once per sync window. A big graph verification failure does not block other verification processes. You must identify failures through `error` level logs such as `BigGraphCheckerV3 failed` or `Big graph sync-conflict check failed` in the logs.

```bash
[info][PID:144373][TID:144880][main.cc][RunChecker] [RunChecker] op[0] Checker Success.
```

---
### 4.9 Running Modes (normal / check-only / estimate)

HCCL-VM has three running modes. At any time, exactly one mode is active, and the default is `normal`:

| Mode | Purpose | Description |
| ---- | ---- | ---- |
| `normal` | Normal mode (default) | Real independent allocation and complete multi-rank communication, with no loss of result correctness |
| `check-only` | Verification-only mode | Large memory blocks reuse a 4 GB shared pool, significantly reducing memory usage. Only the Checker verification chain is guaranteed. Buffer data content correctness is not guaranteed |
| `estimate` | Estimation mode | A single card (rank0) constructs a phantom peer to skip real communication blocking points, quickly producing a task list without actually running data transmission (link heartbeat and consistency verification must be disabled first; see [4.9.4](#494-estimation-mode-limitation-description)) |

#### 4.9.1 Specifying the Mode at Startup

Both the `start` and `run` (one-shot) subcommands support the `--check-only` / `--estimate` flags (they are mutually exclusive). Without any flag, the mode is `normal`:

```bash
# Start in verification-only mode
./hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only

# Start in estimation mode
./hccl-vm start ascend950_cluster_32_server_normal.yaml --estimate
```

#### 4.9.2 Querying and Switching Modes at Runtime

In an already started VM command line, you can query and switch modes in real time through `hccl-vm mode` without restarting the VM:

```bash
# Query: list all supported modes and the current mode (without arguments, equivalent to show, no side effects)
(hvm)$> hccl-vm mode show
Supported modes:
        check-only : Verification-only mode, large memory blocks reuse the 4 GB shared pool HcclCommPool, significantly reducing /dev/shm usage. Only the verification chain is guaranteed. Buffer data content correctness is not guaranteed
        estimate : Estimation mode, a single card (rank0) constructs a phantom peer to skip real communication blocking points, quickly producing a task list without actually running data transmission
        normal : Normal mode (default), real independent allocation and complete multi-rank communication, with no loss of result correctness
Current mode: normal

# Switch modes
(hvm)$> hccl-vm mode check-only
(chk-only)$> hccl-vm mode estimate
(estimate)$> hccl-vm mode normal
```

After a successful switch, the terminal prompt changes according to the mode: normal -> `(hvm)$>`, check-only -> `(chk-only)$>`, estimate -> `(estimate)$>`.

**Switching behavior and constraints**:

- **Strict spelling**: The mode name must strictly use lowercase hyphen spelling. Variants such as `CHECK-ONLY`, `check_only`, and illegal strings are all rejected. The current mode is not affected.
- **Idempotent**: Switching to the current mode is a no-op, and a prompt indicates that the mode is already active.
- **Shared pool creation and teardown with mode**: When switching to check-only, a 4 GB shared reuse pool is automatically created (residual data is cleaned first). When switching away, the pool is automatically released. The pool does not exist in normal or estimate mode.
- **Uninstall runner before switching**: Before switching to a non-normal mode, if the runner plugin is loaded, it is uninstalled first. If uninstallation fails, the switch is aborted. Switching to normal does not trigger uninstallation.
- **Switch atomicity**: If any step fails during the switching process (pool creation failure, pool teardown failure, and so on), an error is reported and the switch is aborted. The mode and resources remain in their pre-switch state. Known limitation: when a switch fails, the prompt may still change. In this case, refer to the `hccl-vm mode show` query result as the authoritative source.
- **Takes effect for new samples**: The new mode takes effect for service samples started after the switch succeeds. Samples already running before the switch are not affected and continue to run in their startup mode.
- **Does not persist across restarts**: The mode does not persist across VM restarts. After a restart, the initial mode is determined by the startup parameter (normal if not specified).

#### 4.9.3 Verification-Only Mode Large Memory Reuse Description

The check-only mode is used for large-scale cluster scenarios that only run Checker verification. When enabled, large memory requests from 200 MB to 4 GB reuse the same 4 GB shared area `HcclCommPool`. All ranks share the area and may overwrite each other, thereby significantly reducing `/dev/shm` usage. In this mode, the large block content correctness is not guaranteed. This mode is only applicable to the Checker V3 verification chain that does not read buffer data. Do not enable this mode when numerically correct results are required.

Requests smaller than 200 MB always use real allocation. A single block larger than 4 GB is directly rejected with an error in check-only mode. The check-only mode and Runner are not mutually exclusive, but when check-only mode is enabled and Runner is installed, the large block reuse still takes effect and may overwrite Runner data. The tool prints a warning.

#### 4.9.4 Estimation Mode Limitation Description

The estimate mode executes on a single card (rank0) with a constructed phantom peer. **Before using estimate mode, you must meet the following two prerequisites**:

1. **Disable link heartbeat and consistency verification**: In estimate mode, no real peer participates on the link, so link heartbeat and consistency verification cannot complete normally and will cause false alarms or abnormal exits.

   ```bash
   export HCCL_DFS_CONFIG=cluster_heartbeat:off,inconsistent_check:off
   ```

   This environment variable must take effect before the service sample starts. Both the `start` and `run` (one-shot) methods apply. When modifying the `start.sh` example in the README, note that it only disables heartbeat by default (`cluster_heartbeat:off`). For estimate mode, you must add `inconsistent_check:off` as shown above.

2. **The service sample must start with a single process (`-np 1`)**: The estimate mode only constructs the rank0 execution path. Regardless of the cluster topology configuration scale, the mpirun command must specify `-np 1`. Other parameters and usage methods remain unchanged.

   ```bash
   mpirun --allow-run-as-root --oversubscribe -np 1 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt
   ```

Other limitations:

- Only the single-card (rank0) execution path is constructed. Real communication blocking points are skipped. This mode is used to quickly produce a task list without actually running data transmission.
- When functional correctness verification or real communication behavior is required, use `normal` mode instead.

#### 4.9.5 Estimation Plugin (@estimate) and Dependency Description

After estimate mode produces the task list, you can run lightweight performance simulation through the estimation plugin (reads `data/hccl_vm_data.db`, builds a task graph, and performs discrete event simulation):

```bash
(estimate)$> hccl-vm plugin run @estimate
```

This plugin is implemented in Python (installation directory `plugin/light_performance/`, available only in estimate mode, mutually exclusive with @checker). Its runtime dependencies are **not distributed with the installation package** and must be pre-installed in the target environment:

- **python3**: Version 3.8 or later (including the sqlite3 standard library).
- **simpy**: Discrete event simulation engine. Run `pip3 install simpy` to install. When missing, the plugin startup reports `ModuleNotFoundError: No module named 'simpy'`.

Simulation results are output to the `plugin/light_performance/data/<timestamp>/` directory: `perf_profiling.json` (Chrome tracing format), single-operator trace files, and the running log `light_performance.log`.

***

## 5 Appendix

### Open-Source Third-Party Software Dependencies

When compiling this project, the dependent third-party open-source software list is as follows. For offline compilation scenarios, you can download and rename the software packages and place them in the third_party directory within this project. For module-specific dependencies and offline placement methods for each storage backend module, see the README in the corresponding module directory.

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
| Rank | The process identifier in MPI |

---

**Document version**: v1.2.
**Last updated**: 2026-10-20.