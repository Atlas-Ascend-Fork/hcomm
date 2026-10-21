# HCCL-VM Guide

## 1. Overview

HCCL-VM is a virtual execution environment for high-performance collective communication targeting Huawei Ascend NPU cards. This tool enables HCCL collective communication operator development and functional verification without real Ascend hardware.

The tool provides three run modes: `normal` (real multi-card communication with lossless result correctness), `check-only` (verification only, large memory blocks reuse a shared pool to reduce memory usage significantly), `estimate` (estimation, skips real communication blocking points to produce task lists quickly). Users specify the mode at startup or switch it at run time using the `hccl-vm mode` command. For details, see [Run Modes](#49-run-modes-normal--check-only--estimate).

![hccl-vm GIF](docs/hccl-vm.gif)

## 2. Prerequisites

|   Dependency   |     Version Requirement     |
| -------- | ------------- |
| System Architecture  | x86_64 Ubuntu 22.04 and above |
| Specification Constraints  | Ascend950; for others, see [Specification Constraints](#45-tool-specification-constraints) |
| QEMU (required only for AICPU mode on x86_64 host machines) | The system-provided qemu-user-static works directly; versions below 10.2.4 cannot limit translation code cache, QEMU resident memory is larger, and large-scale scenarios may cause paging and affect performance significantly. Install version 10.2.4 or above per [2.3 QEMU Deployment](#23-qemu-deployment-aicpu-mode-x86_64-host-machine-only) for optimization. Native ARM host machines do not require QEMU; compilation (`build.sh`/`build_pkg.sh`) does not depend on locally installed QEMU |

### 2.1 CANN Package Installation

Install the latest CANN Toolkit development kit and CANN ops operator packages from the [download link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)

```bash
# Ensure installation packages have executable permissions
chmod +x Ascend-cann-toolkit_9.1.0_linux-x86_64.run
chmod +x Ascend-cann-950-ops_9.1.0_linux-x86_64.run
# Installation commands
./Ascend-cann-toolkit_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
./Ascend-cann-950-ops_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
```


### 2.2 hccl_test Compilation

hccl_test is the official Ascend HCCL performance test tool. For details, see [HCCL Performance Test Tool](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/devaids/hccltool/HCCLpertest_16_0001.html). HCCL-VM supports running hccl_test samples in the virtual environment. Follow the [hccl_test Sample Build](#42-hccl-test-sample-build) section to compile the sample binary programs first.

Note: hccl_test is an optional dependency. For PyTorch samples, see [PyTorch Sample Examples](#33-pytorch-sample-examples).

### 2.3 QEMU Deployment (AICPU Mode, x86_64 Host Machine Only)

The AICPU expansion mode requires QEMU user-mode emulation on x86_64 host machines (linux-user emulating aarch64). The system-provided qemu-user-static works directly; versions below **10.2.4** do not support `-tb-size` to limit translation code cache, QEMU resident memory is larger, which may severely affect performance. Install version **10.2.4 or above** as follows.

```bash
# 1) Build dependencies (gcc/make/ninja are typically pre-installed)
sudo apt-get install -y meson flex bison libglib2.0-dev libpcre2-dev libffi-dev

# 2) Official source archive (example uses 10.2.4; version 10.2.4 and above are acceptable)
curl -fsSL -O https://download.qemu.org/qemu-10.2.4.tar.xz
tar -xf qemu-10.2.4.tar.xz && cd qemu-10.2.4

# 3) Build aarch64 linux-user static version
mkdir build && cd build
../configure --target-list=aarch64-linux-user --static --disable-system --disable-tools \
    --disable-docs --disable-guest-agent --disable-slirp
make -j$(nproc) qemu-aarch64

# 4) Deploy and configure (deployment path is customizable; example uses /opt/qemu-10.2.4)
sudo mkdir -p /opt/qemu-10.2.4/bin
sudo cp qemu-aarch64 /opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_TB_SIZE=16
```

Run behavior and prompts:

- Device processes print the actual QEMU program path, version, and requested `-tb-size` value at startup.
- When `HCCL_VM_QEMU_PATH` is unset, invalid, or points to an old QEMU that does not support `-tb-size`, the tool **does not abort**: it falls back automatically (searches PATH for `qemu-aarch64-static` / omits `-tb-size`) and prints a WARN with the actual fallback program and version, plus a "translation cache not limited, QEMU memory may be excessive" prompt. AICPU mode startup fails only when no `qemu-aarch64-static` is found on the entire machine.
- `HCCL_VM_QEMU_TB_SIZE` sets the QEMU translation code cache (MiB): defaults to `16` when unset; `0` delegates to QEMU's own default policy (it does not disable the cache). An unlimited cache may increase QEMU resident memory significantly and cause paging.

---

## 3. Quick Start

### One-Click Installation

After pulling the complete CheckerL2 source code, run the following command at the repository root:

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

# 4. Compile the HCCL-VM tool. After downloading the hcomm code, the tool source path is: /home/workspace/hcomm/test/hccl_vm
cd /home/workspace/hcomm/test/hccl_vm
source /home/workspace/Ascend/cann/set_env.sh
export HCCL_CODE_HOME=/home/workspace/hccl
export HCOMM_CODE_HOME=/home/workspace/hcomm
bash ./build.sh --full

# 5. Extract aicpu_hcxx.tar.gz from the CANN installation directory
bash build_pkg.sh
```

### 3.2 Usage Examples

#### 3.2.1 Environment Configuration

If an hccl_rootinfo.json file already exists under the /etc path, delete it manually.

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
# Enter the new bin directory to run hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# Enable the runner plugin if needed (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for this operator (run hccl_test samples in a cluster environment with 1 supernode, 1 server, and 1 NPU)
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run checker verification
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal
(hvm)$> exit
```

3. Verify the hccl_test sample run results
[Runner Results](#481-runner-plugin-results)
[Checker Results](#482-checker-plugin-results)

#### 3.2.3 AICPU Mode

The AICPU expansion mode places algorithm expansion steps on the device side. The hccl-vm tool must compile and simulate HCCL device-side symbols. Since device-side symbols target the ARM architecture, cross-compilation is required on x86 environments, and QEMU is required for AICPU mode simulation at run time.

Running AICPU mode on an x86 host machine uses QEMU user-mode emulation: when `HCCL_VM_QEMU_PATH` is not configured, the tool falls back automatically to `qemu-aarch64-static` in PATH and prints a warning with the actual version. This does not affect startup. System versions below 10.2.4 cannot limit translation code cache and have higher memory usage. Large-scale scenarios may cause paging and affect performance. Follow [2.3 QEMU Deployment](#23-qemu-deployment-aicpu-mode-x86_64-host-machine-only) to install version 10.2.4 or above and configure `HCCL_VM_QEMU_PATH`. Native ARM host machines do not require QEMU.

1. Configure environment variables.

```bash
# Enter the tool installation directory
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AI_CPU"

# x86_64 host machine: QEMU program and translation code cache (MiB) configuration, see §2.3
export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static
export HCCL_VM_QEMU_TB_SIZE=16
```

2. Execute

```bash
# Enter the new bin directory to run hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# Enable the runner plugin if needed (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for this operator (run hccl_test samples in a cluster environment with 1 supernode, 1 server, and 1 NPU)
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run checker verification
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal
(hvm)$> exit
```

3. Verify the hccl_test sample run results [Runner Results](#481-runner-plugin-results) [Checker Results](#482-checker-plugin-results)

#### 3.2.4 HostDPU Mode

The HostDPU expansion mode uses the same environment variable configuration as AICPU mode at run time. The only differences are in cluster scale and communicator scale:

1. The cluster specification configuration uses `./hccl-vm start ascend950_cluster_32_server_normal_hostdpu.yaml`
2. The server count must be >= 2; single-server communicators are not supported (configurations such as 112.yaml, 114.yaml, 118.yaml are not supported)

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
# Enter the new bin directory to run hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# Enable the runner plugin if needed (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communicator configuration file for this operator (run hccl_test samples in a cluster environment with 1 supernode, 1 server, and 1 NPU)
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Run checker verification
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal
(hvm)$> exit
```

3. Verify the hccl_test sample run results [Runner Results](#481-runner-plugin-results) [Checker Results](#482-checker-plugin-results)

### 3.3 PyTorch Sample Examples

HCCL-VM supports running certain PyTorch `aclgraph` graph-mode collective communication scripts. Scripts must use `torch.multiprocessing.spawn` to launch device processes. Multi-server samples require launching multiple Python main processes concurrently on the same host machine, each simulating a server. For the complete guide, see [PyTorch Mode Run Guide](docs/pytorch_guide_en.md).

#### 3.3.1 Environment Preparation

Complete the HCCL-VM build and installation and CANN environment configuration first, then prepare a Python virtual environment with compatible versions of `torch` and `torch_npu` installed. PyTorch currently supports only the Ascend950 series. To use the Ascend960 series, install the A6 ops package and change `soc_version` in the topology cluster configuration to a real model such as `Ascend960DT_961BY`.

```bash
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin
./hccl-vm start ascend950_cluster_32_server_normal.yaml
```

After entering the `(hvm)$>` terminal, run the following commands. Perform communicator configuration, environment variable setup, and Python launch within this terminal so that Python processes inherit the emulation environment.

```bash
# Enable PyTorch adaptation
export HCCL_VM_ENABLE_PYTORCH=1
# Activate the virtual environment (if applicable)
source /path/to/torch_test/bin/activate

# Install Runner when simulation execution is needed
hccl-vm plugin install @runner
```

#### 3.3.2 Single-Server Run

The `118` configuration uses 1 server and 8 devices to execute an AllReduce SUM of 1024 int32 elements. The following command parameters correspond to the test script in the guide. Adjust parameters when using other scripts.

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

The `128` configuration uses 2 servers with 8 devices each, totaling 16 ranks. Run the following commands in the HCCL-VM terminal after completing environment preparation:

```bash
hccl-vm mock-comm 128

# Launch two servers concurrently, set environment variables for each, and save logs
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

# Wait for both servers to finish and check logs for exceptions and execution results
wait
```

`HCCL_VM_SERVER_INDEX` is the zero-based server index in the current communicator configuration. It takes effect only when `HCCL_VM_ENABLE_PYTORCH=1` and must match the `--server_index` of the corresponding script. Setting environment variables inside parentheses prevents different server indexes from overwriting each other.

All servers must use the same `--master_ip` and `--master_port`; `--dev_num` and `--server_num` must match the `mock-comm` configuration. Local logical device IDs for each server are `0-7`. In this example, server 0 corresponds to global ranks `0-7` and server 1 corresponds to global ranks `8-15`. The script calculates global ranks assuming equal device counts per server. Other scales should use communicator configurations with equal device counts per server.

#### 3.3.4 Result Viewing and Verification

After all processes finish, check the corresponding `server0.log` and `server1.log` (for multi-server scenarios). Focus on exit status, exceptions, and relevant log output. Then run Checker verification in the HCCL-VM terminal. For result descriptions, see [Checker Plugin Results](#482-checker-plugin-results).

```bash
hccl-vm plugin run @checker

# Exit the tool terminal after all samples and verifications are complete
exit
```

### 3.4 HCCL Code Modification Verification Example

If you modified CANN operator package code, such as adding a new algorithm type, follow these steps to verify your changes. The build_pkg.sh script helps users perform packaging, installation, and copying of device-side dependency symbols. Set environment variables before running:

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
3. If you updated or modified both CANN hccl and hcomm repository code, run `bash build_pkg.sh --full`.
4. Follow the [Usage Examples](#32-usage-examples) steps to re-run samples.

---

## 4. Detailed Guide

### 4.1 Tool Environment Variable Configuration

**HCCL-VM Environment Variable Description**:

| Environment Variable                      | Purpose                                                                                                        | Example                                                                        |
| ------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------- |
| `HCCL_CODE_HOME`         | Specifies the HCCL source code path for HCCL-VM compilation. Not configured by default.     | `export HCCL_CODE_HOME=/home/workspace/hccl`                     |
| `HCOMM_CODE_HOME`         | Specifies the HCOMM source code path for HCCL-VM compilation. Not configured by default.     | `export HCOMM_CODE_HOME=/home/workspace/hcomm`                     |
| `HCCL_VM_ENABLE_PYTORCH` | Set to `1` to enable PyTorch adaptation. For the run method, see [PyTorch Sample Examples](#33-pytorch-sample-examples). | `export HCCL_VM_ENABLE_PYTORCH=1` |
| `HCCL_VM_SERVER_INDEX` | The server index simulated by the PyTorch process, starting from `0`. Takes effect only when `HCCL_VM_ENABLE_PYTORCH=1` and must match the `--server_index` of the test script. | `export HCCL_VM_SERVER_INDEX=0` |
| `HCCLVM_ENABLE_DUMP_DATA` | 1. Enables performance simulation file output to disk. When enabled, 3 runner_hcclvm_*_data.bin files and 1 ccu_channel_jetty_config.xml file are output during test case execution for performance simulation use. <br> 2. Enables the Runner plugin to dump input and output data. When enabled, each operator's input and output data is dumped to all_rank_input_output.txt during test case execution. | `export HCCLVM_ENABLE_DUMP_DATA=1 to enable, export HCCLVM_ENABLE_DUMP_DATA=0 or leave unset to disable` |
| `HCCL_VM_QEMU_PATH` | AICPU mode and x86_64 host machine only: non-empty absolute path to an external QEMU program (target is an executable regular file; symbolic links pointing to regular files are allowed). When unset or invalid, the tool falls back to searching PATH for `qemu-aarch64-static` and prints a WARN with the actual program and version. AICPU startup fails when no binary is found on the entire machine. For deployment steps, see [2.3 QEMU Deployment](#23-qemu-deployment-aicpu-mode-x86_64-host-machine-only) | `export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static` |
| `HCCL_VM_QEMU_TB_SIZE` | AICPU mode and x86_64 host machine only: QEMU translation code cache (MiB). Defaults to `16` when unset; `0` passes `-tb-size 0` explicitly and lets QEMU's own default policy decide (it does not disable the cache); accepts only decimal digit strings parseable as uint32_t; illegal values trigger a warning and fall back to the default 16. When the target is an old QEMU that does not support `-tb-size`, this option is not passed and a warning about potentially excessive memory is printed. | `export HCCL_VM_QEMU_TB_SIZE=16` |

### 4.2 HCCL-Test Sample Build

The hccl_test sample source code is located in the CANN package installation directory. It supports compilation and execution in both OpenMPI and MPICH environments. For run-time differences, see [OpenMPI and MPICH Environment Execution Differences](#47-openmpi-and-mpich-environment-execution-differences). This guide uses the OpenMPI environment.

#### 4.2.1 OpenMPI Environment Compilation

1. Install OpenMPI

```bash
sudo apt-get update
sudo apt install openmpi-bin libopenmpi-dev
```

2. Compile hccl_test

```bash
# Modify CANN installation directory permissions
chmod -R 755 /home/workspace/Ascend

# Enter the hccl_test sample source directory
cd /home/workspace/Ascend/cann/tools/hccl_test

# Set the CANN environment variables
source /home/workspace/Ascend/cann/set_env.sh

# Temporarily modify the Makefile script
if ! grep -q '\-lmpi_cxx' Makefile; then
    sed -i 's/-lmpi/-lmpi -lmpi_cxx/g' Makefile
fi

# Compile hccl_test samples
MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi make ASCEND_DIR=${ASCEND_HOME_PATH}
```

#### 4.2.2 MPICH Environment Compilation

Assume the mpich path is: `/usr/lib/mpich`.

```bash
# Enter the hccl_test sample source directory
cd /home/workspace/Ascend/cann/tools/hccl_test

# Set the CANN environment variables
source /home/workspace/Ascend/cann/set_env.sh

# Configure environment variables
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH

# Compile hccl_test samples
make MPI_HOME=/usr/lib/mpich/ ASCEND_DIR=${ASCEND_HOME_PATH}
```

### 4.3 Ascend Cluster Topology Configuration File Description

#### 4.3.1 Server/Pod Topology Configuration File Description

An Ascend cluster topology is composed of one or more Server/Pod sub-topologies combined according to CLOS layered network rules. Therefore, users must confirm the topology type of each Server/Pod before generating a cluster topology.
Users can choose predefined topology types provided by the HCCL-VM tool or define custom Server/Pod topology types according to the configuration file format requirements.

Describing the topology network relationships of a Server/Pod includes the following aspects:

 - **Port Configuration Table**: Describes the physical port configuration of an NPU card, such as NPU-to-NPU direct connection ports (P2P) and NPU out-of-chassis ports (P2NET).
 - **Link Configuration Table**: Describes the connection relationships among all NPU cards within a Server/Pod, such as full mesh connections.
 - **PortBound**: Describes port binding relationships of an NPU card, where multiple ports are bound into a PortGroup.

```yaml
type: "server_intra_links"
name: "ascend950_links_topo_demo"
description: "ascend950 chip normal topology connection description file"

soc_version: "Ascend950"
device_num: 16

device_ports_allocate_map:
 # port allocation table: 0: unused, 1: device direct connection, 2: device to switch, 3: d2h port
 #                 portId:  0  1  2  3  4  5  6  7  8
    - {die_id: 0, pin_map: [1, 1, 1, 0, 2, 2, 2, 2, 3]} # die0
    - {die_id: 1, pin_map: [0, 0, 0, 0, 0, 0, 0, 0, 0]} # die1

# port_group: describes which ports are merged into a portGroup; ports in the same portGroup share the same IP address
port_group:
    - {layer: 0, ports: ["0/4", "0/5", "0/6", "0/7"]}

links:
  # ── Description method: 8 devices per row form a full mesh ──
  - link_mode: "fullmesh"
    connections:
      # Example: devices 0, 1, 2, 3 of die0 are fully connected through die0 ports
      - {die_id: 0, devices_range: [0, 3]}
      - {die_id: 0, devices_range: [4, 7]}
      - {die_id: 0, devices_range: [8, 11]}
      - {die_id: 0, devices_range: [12, 15]}
  
  #- link_mode: "enum"
  #  device_to_device_links:
  #    Example: devices 0 and 1 connect through die0 ports to devices 1, 3, 5, 7 through die1 ports respectively
  #            That is: device0 connects to device1, device3, device5, device7; device1 connects to device3, device5, device7
  #    - {src_die_id: 1, src_local_id_range: [0, 2], dst_die_id: 1, dst_local_id_range: [1, 3, 5, 7]}

  - link_mode: "enum"
    device_to_switch_links:
      # Example: devices 0 through 15 connect to the switch through die0 ports. Combined with portGroup, devices 0 through 15 connect to the switch through portGroup[0/4, 0/5, 0/6, 0/7].
      - {die_id: 0, devices_range: [0, 15]}

```

**Field Description**:

 - **soc_version**: Chip model, such as `Ascend950`.
 - **device_num**: Total device count, determined by chip model and topology type.
 - **device_ports_allocate_map**: Port allocation table describing port configuration per die. 1 indicates device direct connection ports, 2 indicates device-to-switch ports, 3 indicates d2h ports.
 - **port_group**: Describes which ports merge into a portGroup. Ports in the same portGroup share the same IP address. Unconfigured ports default to one port per portGroup.
 - **links**: Link configuration table describing connection relationships among all NPU cards within a Server/Pod and between NPU cards and switches.
   - **NPU Direct Connection**: The tool provides two methods to configure NPU direct connections:
    - **link_mode == "fullmesh"**: All devices are fully connected based on ports of one die. New typical connection methods can be added as new link_mode types, such as "ring".
    - **link_mode == "enum"**: Enumeration method. When NPU connections within a Server/Pod are complex, all link relationships can be enumerated.
   - **NPU-to-Switch Connections**: Users can configure NPU-to-switch connections using the enumeration method.
  - **device_to_device_links**: Describes NPU-to-NPU connection relationships.
  - **device_to_switch_links**: Describes NPU-to-switch connection relationships.

#### 4.3.2 Cluster Topology Configuration File Description

An Ascend cluster network is composed of one or more Server/Pod sub-topologies combined according to CLOS layered network rules. Users can select different Server/Pod topology types based on cluster scale and requirements.

Users can define custom cluster topology configuration files using the following format:

```yaml
name: "ascend950_cluster_32_server_normal"
description: "Ascend 950 normal networking: 32 supernodes, 1 server per supernode"

# Total supernode count
super_node_num: 4
# Total server/pod count
server_num: 32
server_list:
  # Servers 0-7: all use ascend950_server_topo_normal topology type
  - {super_pod_id: 0, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 1, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 2, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 3, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}

```

The configuration file above describes a cluster topology with 4 supernodes, 32 servers, and 128 NPU cards total. Each Server/Pod uses the ascend950_server_topo_normal topology type.

**Field Description**:

 - **super_node_num**: Total supernode count.
 - **server_num**: Total server/pod count.
 - **server_list**: Configuration information for each Server/Pod, including supernode ID, device ID range, chip model, and Server/Pod topology configuration file path.

#### 4.3.3 Communicator Configuration File Description

In an Ascend cluster environment, users must select different communicator configuration files based on the communicator requirements of the operators to be executed.

The tool provides the hccl-vm mock-comm command to read and configure operator communicator configuration files. Communicator configuration files use the yaml format and are located at hccl_vm_install/config/topo_meta. If the directory does not contain the required communicator configuration file, users must create one first.

The hccl-vm tool supports asymmetric topology communicator configurations. An example is shown below:

```yaml
# 1. Global statistics: podNum, serNum, rankNum are all less than 1024
meta:
  podNum: 1  # Total supernode count
  serNum: 2  # Total server count
  rankNum: 6 # Total rank count

# 2. Detailed topology structure
topology:
  - podId: 0
    servers:
      - serId: 0
        # Local IDs of ranks actually running on each server
        ranks: [0, 2]
      - serId: 1
        # Local IDs of ranks actually running on each server
        ranks: [1, 3, 5, 7]
```

**Notes**:

- When configuring a communicator, the tool regenerates topo.json and ranktable.json files based on the specified communicator configuration number.
- In the communicator configuration yaml file above, the ranks field represents the list of local IDs (physical device IDs) of ranks actually running on each server.

When the communicator scale is large, server selection fields and rank selection fields in server entries can be combined freely. The following fields are supported:

- **Server selection fields** (choose one; do not configure both simultaneously):
  - `serId`: A single server ID or a list of server IDs (for example, `3` or `[0, 1]`; a list represents each enumerated server);
  - `serId_range`: A server ID range with value `[<start_id>, <end_id>]` (closed interval, supports list format only), representing all servers from start to end ID under that supernode.
- **Rank selection fields** (choose one; do not configure both simultaneously):
  - `ranks`: A list of local IDs of ranks actually running on each server;
  - `ranks_range`: A local ID range with value `[<start_id>, <end_id>]` (closed interval, supports list format only), representing all local IDs from start to end on each server.

The following example (2 supernodes, 4 servers total, 32 ranks) shows multiple combinations mixed within the same file:

```yaml
# 1. Global statistics: podNum, serNum, rankNum are all less than 1024
meta:
  podNum: 2  # Total supernode count
  serNum: 4  # Total server count
  rankNum: 32 # Total rank count

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

**Notes**:

- serId and serId_range cannot be configured simultaneously; ranks and ranks_range cannot be configured simultaneously.
- serId_range and ranks_range support only the `[<start_id>, <end_id>]` list format (closed interval); the `<start_id>-<end_id>` format is not supported. The start ID cannot be greater than the end ID, and the range width (end_id - start_id + 1) cannot exceed 1024.
- One serId_range entry expands to all servers from start to end ID under that supernode. A serId list entry represents each enumerated server, and each server has identical ranks.
- If no rank selection field is configured in a server entry, that server's ranks is an empty list. If no server selection field is configured, serId defaults to 0.

#### 4.3.4 topo.json and ranktable.json File Description

Users do not need to create `topo.json` and `ranktable.json` files manually. The tool generates them automatically based on the following information:

- **Topology Configuration Number**: The number specified by the user at startup (for example, 112, 113)
- **Chip Type**: The chip type automatically identified from the run environment.

Although the tool generates these configuration files automatically, understanding their structure helps users comprehend topology configurations.

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

- `server.device_count`: Total device count.
- `server.groups`: Device group information.
- `ports`: Port configuration.
  - `usage`: Port purpose (`peer2peer` indicates device-to-device connections, `peer2net` indicates external connections)
- `links`: Link configuration.
  - `link_type`: Link type (`PEER2PEER` or `PEER2NET`)
  - `topo_type`: Topology type (`1DMESH`, `CLOS`, and so on)

**ranktable.json Structure**:

`ranktable.json` describes the devices and IP mappings used for this run:

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

**Field Description**:

- `server_count`: Server count.
- `device_count`: Total device count.
- `server_list`: Server and device list.
  - `device_ip`: Device IP address.
  - `port`: Device port number.

### 4.4 hccl\_config.sh File Description

The hccl\_config.sh file contains environment variable configurations required for running HCCL\_Test samples. The environment variables in this file are consistent with those used for Hccl\_Test samples in real hardware environments.
Users must modify the hccl\_config.sh script according to their test cases and requirements to configure HCCL sample run environment variables.

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

  # Return 0 regardless of whether files were deleted to ensure script continues
  return 0
}

# Clean redundant files in data/ directory (temporary files generated in CCU mode)
cd "${HCCL_VM_INSTALL_DIR}/data" 2>/dev/null && {
  remove_files_by_prefix "sqe_info_rank_"
  remove_files_by_prefix "mc_instr_info_rank_"
  rm -f "all_rank_input_output.txt"
  cd "${HCCL_VM_INSTALL_DIR}"
}

# Set CANN environment variables
source /home/workspace/Ascend/cann/set_env.sh

# Disable the hccl heartbeat feature
export HCCL_DFS_CONFIG=cluster_heartbeat:off

# Set the HCCL-VM installation path, inferred from script location (compatible with bin/ and script/ subdirectories)
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

# Set the ranktable.json file path (consistent with mock-comm generation path)
export RANK_TABLE_FILE=${HCCL_VM_INSTALL_DIR}/data/ranktable.json

# Set log level
export ASCEND_GLOBAL_LOG_LEVEL=1

# Enable log stdout output
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# Set HCCL run mode (CCU, AI_CPU, AIV expansion modes)
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
# Or set HCCL run parameters (AI_CPU expansion mode) AI_CPU mode environment variables cannot be set simultaneously with other modes
# export HCCL_OP_EXPANSION_MODE="AI_CPU"
# Or set HCCL run parameters (AIV expansion mode) AIV mode environment variables cannot be set simultaneously with other modes
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

This tool currently supports only Ascend950 chips. A single server supports up to 8 cards; more than 8 cards require cross-server execution. The built-in storage covers communicators from 1 to 128 cards by default, requiring no database configuration.

### 4.6 HCCL-VM Plugin Features

#### 4.6.1 Runner Plugin

The Runner plugin simulates execution of tasks generated by Hccl business orchestration and outputs output data.
The simulator runner plugin is **disabled** by default during hccl\_test sample execution. After hccl\_test samples call the operator interface, they wait for operator task completion through the aclrtSynchronizeStream interface. The simulator runner tool waits until all ranks are in the waiting state, then starts simulating execution of all rank tasks. After execution completes, it notifies each rank's samples to continue execution.
After sample execution completes, users can view each rank's input buffer and output buffer data in the "all\_rank\_input\_output.txt" file under the execution directory. This feature is not enabled by default. Users can enable it before sample execution using the corresponding command.

**Installation and Uninstallation**:

The Runner plugin supports installation and uninstallation through the `hccl-vm plugin install/uninstall` commands. Install the runner plugin after entering the hccl-vm tool command line and before executing samples. Subsequent execution will then run the runner.

```bash
# Install the runner plugin
(hvm)$> hccl-vm plugin install @runner

# Uninstall the runner plugin
(hvm)$> hccl-vm plugin uninstall @runner
```

#### 4.6.2 Checker Plugin

The Checker plugin (algorithm analyzer plugin) forms a DAG graph from all tasks generated by hccl and analyzes the DAG graph to detect memory conflicts. It simulates DAG graph execution to detect semantic errors and other issues.
Users launch the algorithm analyzer plugin manually through commands.

The Checker plugin currently uses Checker V3 for single-operator verification. Checker V3 big graph verification is enabled by default: multiple operators within each sync window are merged into one big graph for cross-operator synchronous resource conflict verification. Single-operator Checker V3 and big graph verification can be controlled separately through configuration parameters in `manifest.json`. The old Checker version has been removed; `enable_old_checker` is no longer a valid configuration item.

```json

# Configuration file located at /pathto/hccl_vm_install/plugin/checker/manifest.json

{
  "name": "checker",        // Checker plugin name
  "version": "1.0.0",       // Checker plugin version
  "entry": "./checker",     // Checker plugin startup command
  "dependency": {
      "min_core_version": "1.0.0"
  },
  "setting": {              // Checker plugin configuration items
      "enable_new_checker": true,           // Enable Checker V3 single-operator verification (enabled by default)
      "enable_big_graph_checker": true,     // Enable Checker V3 big graph verification (enabled by default, independent from single-operator verification)
      "enable_insight_dump": false,         // Output Checker V3 Insight data (disabled by default)
      "enable_memory_snapshot_dump": false, // Output memory snapshot data (disabled by default; requires Insight data output to be enabled first)
      "enable_dag_graphviz_dump": false     // Output big graph Graphviz DOT file (disabled by default)
  }
}
```

### 4.7 OpenMPI and MPICH Environment Execution Differences

Before running hccl_test samples, users can determine which mpirun the current environment uses with the which command.

#### 4.7.1 Environment Variable Configuration Differences

The environment typically defaults to OpenMPI. If users run samples with OpenMPI, no additional environment variable configuration is generally needed.
If users run samples with the MPICH environment, configure environment variables as follows:

```bash
# Configure mpich environment variables
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH
export PATH=/usr/lib/mpich/bin:$PATH
```

#### 4.7.2 mpirun Command Parameter Differences

In the OpenMPI environment, users run hccl_test samples with the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun --allow-run-as-root --oversubscribe -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter Description**:

 - --allow-run-as-root: An OpenMPI-specific parameter that allows running MPI processes as root user, used in environments without root permissions.
 - --oversubscribe: An OpenMPI-specific parameter that removes CPU slot limits, allowing a single node to launch [process count > CPU logical core count], that is, over-subscribed process execution.
 - -np 2: Specifies 2 processes, consistent with the node count.

In the MPICH environment, users run hccl_test samples with the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter Description**:

 - -np 2: Specifies 2 processes, consistent with the node count.

### 4.8 Result Viewing

#### 4.8.1 Runner Plugin Results

If the runner plugin was installed by running hccl-vm plugin install @runner in the hccl-vm terminal, the runner plugin executes automatically after the operator process completes. The final result depends on hccl_test verification. Users must check the redirected log file for [error] level logs and the final verification result:

```bash
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
64                | 1000.00        | 0.00006              | success
```

#### 4.8.2 Checker Plugin Results

After running hccl-vm plugin run @checker in the hccl-vm terminal, the Checker verification process and results are printed in the terminal. Users must check for [error] level logs and the final verification result:

Big graph verification runs once per sync window. Big graph verification failure does not block other verification processes. Identify failures through `error` level log entries such as `BigGraphCheckerV3 failed` or `Big graph sync-conflict check failed`.

```bash
[info][PID:144373][TID:144880][main.cc][RunChecker] [RunChecker] op[0] Checker Success.
```

---
### 4.9 Run Modes (normal / check-only / estimate)

HCCL-VM has three run modes. Exactly one mode is active at any time. The default is `normal`:

| Mode | Purpose | Description |
| ---- | ---- | ---- |
| `normal` | Normal mode (default) | Real independent allocation with complete multi-rank communication; result correctness is lossless |
| `check-only` | Verification-only mode | Large memory blocks reuse a 4GB shared pool to reduce memory usage significantly; only guarantees the Checker verification path; buffer data content correctness is not guaranteed |
| `estimate` | Estimation mode | Single card (rank0) constructs phantom peers to skip real communication blocking points; produces task lists quickly without running actual data transfers (requires disabling link heartbeat and consistency checks first; see [4.9.4](#494-estimation-mode-limitations)) |

#### 4.9.1 Specifying the Mode at Startup

Both the `start` and `run` (one-shot) subcommands support `--check-only` / `--estimate` flags (mutually exclusive). Omitting both flags defaults to `normal`:

```bash
# Start in verification-only mode
./hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only

# Start in estimation mode
./hccl-vm start ascend950_cluster_32_server_normal.yaml --estimate
```

#### 4.9.2 Querying and Switching Modes at Run Time

In an already-started VM command line, users can query and switch modes instantly using `hccl-vm mode` without restarting the VM:

```bash
# Query: list all supported modes and current mode (no argument is equivalent to show, no side effects)
(hvm)$> hccl-vm mode show
Supported modes:
        check-only : Verification-only mode, large memory blocks reuse 4GB shared pool HcclCommPool to reduce /dev/shm usage significantly; only guarantees the verification path; buffer data content correctness is not guaranteed
        estimate : Estimation mode, single card (rank0) constructs phantom peers to skip real communication blocking points; produces task lists quickly without running actual data transfers
        normal : Normal mode (default), real independent allocation with complete multi-rank communication; result correctness is lossless
Current mode: normal

# Switch modes
(hvm)$> hccl-vm mode check-only
(chk-only)$> hccl-vm mode estimate
(estimate)$> hccl-vm mode normal
```

After a successful switch, the terminal prompt changes with the mode: normal → `(hvm)$>`, check-only → `(chk-only)$>`, estimate → `(estimate)$>`.

**Switch Behavior and Constraints**:

- **Strict spelling**: Mode names must use exact lowercase hyphenated spelling. Variants such as `CHECK-ONLY`, `check_only`, and illegal strings are all rejected. The current mode is unaffected.
- **Idempotent**: Switching to the current mode is a no-op with a prompt indicating the mode is already active.
- **Shared pool creation and teardown with mode**: When switching to check-only, a 4GB shared reuse pool is created automatically (residuals are cleaned first). When switching away, the pool is released automatically. The pool does not exist in normal / estimate modes.
- **Uninstall runner before switching**: Before switching to a non-normal mode, the runner plugin is uninstalled first if resident. Uninstallation failure aborts the switch. Switching to normal does not trigger uninstallation.
- **Switch atomicity**: If any step fails during switching (pool creation failure, pool teardown failure), the switch aborts with an error. The mode and resources remain in their pre-switch state. Known limitation: the prompt may still change on switch failure; in this case, trust the `hccl-vm mode show` query result.
- **Applies to new samples**: The new mode applies to business samples launched after the switch succeeds. Samples already running before the switch are unaffected and continue with their startup mode.
- **Not preserved across restarts**: The mode is not preserved across VM restarts. After restart, the initial mode is determined by startup parameters (defaults to normal if unspecified).

#### 4.9.3 Verification-Only Mode Large Memory Reuse Description

The check-only mode is for large-scale cluster scenarios that run only Checker verification. When enabled, large memory requests from 200MB to 4GB reuse the same 4GB shared region `HcclCommPool`. All ranks share this pool and may overwrite each other, reducing `/dev/shm` usage significantly. Large block content correctness is not guaranteed in this mode. This mode applies only to Checker V3 verification paths that do not read buffer data. Do not enable it when numerically correct results are needed.

Requests smaller than 200MB always use real allocation. Single blocks larger than 4GB are rejected directly with an error in check-only mode. The check-only mode and Runner are not mutually exclusive, but installing Runner while check-only mode is active still applies large block reuse, which may overwrite Runner data. The tool prints a warning in this case.

#### 4.9.4 Estimation Mode Limitations

The estimate mode runs on a single card with constructed phantom peers. No real peers participate on the link, so link heartbeat and consistency checks cannot complete normally, causing false reports or abnormal exits. Therefore, **disable link heartbeat and consistency checks before using estimate mode**:

```bash
export HCCL_DFS_CONFIG=cluster_heartbeat:off,inconsistent_check:off
```

This environment variable must take effect before the business sample starts. Both the `start` and `run` (one-shot) methods apply. When modifying the `start.sh` example from this README, note that it disables only heartbeat by default (`cluster_heartbeat:off`). Add `inconsistent_check:off` as shown above for estimate mode.

Other limitations:

- Only the single-card (rank0) execution path is constructed. Real communication blocking points are skipped to produce task lists quickly without running actual data transfers.
- Use `normal` mode when functional correctness verification or real communication behavior is needed.

***

## 5 Appendix

### Open-Source Third-Party Software Dependencies

When compiling this project, the required third-party open-source software list is as follows. For offline compilation scenarios, download and rename the packages, then place them in the third_party directory within this project. Module-specific dependencies and offline placement methods are described in the README within each module directory.

| Open-Source Software       | Version          |Download URL |
| ------------  | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| CLI11         | 2.2.0         | [cli11-2.2.0.tar.gz](https://raw.gitcode.com/src-openeuler/cli11/blobs/58c912141164a5c0f0139bfa91343fefe151d525/cli11-2.2.0.tar.gz) |
| json          | 3.11.3        | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| spdlog        | 1.11.0        | [spdlog-v1.11.0.tar.gz](https://raw.gitcode.com/src-openeuler/spdlog/blobs/c2dfb1aca26c607393665c836155613ff283de66/v1.11.0.tar.gz) |
| yaml-cpp      | 0.8.0         | [yaml-cpp-0.8.0.tar.gz](https://raw.gitcode.com/src-openeuler/yaml-cpp/blobs/d1ead4fff417073b9cdbf98b8b55eb0efc00b0ba/yaml-cpp-0.8.0.tar.gz) |
| googletest    | 1.14.0        | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) |
| cann-cmake    | master-044    | [cmake-master-044.tar.gz](https://raw.gitcode.com/cann/cmake/archive/refs/heads/master-044.tar.gz) |

### Glossary

| Term       | Description                                                      |
| -------- | ------------------------------------------------------- |
| HCCL     | Huawei Collective Communication Library         |
| NPU      | Neural Processing Unit                          |
| CANN     | Compute Architecture for Neural Networks, Huawei Ascend AI processor software stack |
| MPI      | Message Passing Interface                        |
| CCU      | Collective Communication Unit                    |
| Topology | Device connection relationships                                               |
| Rank     | Process identifier in MPI                                         |

---

**Document Version**: v1.2.
**Last Updated**: 2026-09-19.
