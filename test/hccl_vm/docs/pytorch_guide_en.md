# PyTorch Mode Run Guide

HCCL-VM can now run some PyTorch `aclgraph` graph mode related collective communication scripts. Scripts should use `torch.multiprocessing.spawn` to start device processes. Multi-server use cases require starting multiple Python main processes on the same host machine, each simulating a different server.

## 1. Prepare Environment

First complete HCCL-VM build and installation and CANN environment configuration according to [README](../README.md). Prepare a Python virtual environment with compatible versions of `torch` and `torch_npu` installed.

```bash
cd /path/to/CheckerL2/hccl_vm_install/bin

# Select communicator configuration file as needed. PyTorch currently only supports Ascend950 series.
# To use Ascend960 series, install A6 ops package and modify soc_version in topology cluster configuration to real models such as Ascend960DT_961BY.
./hccl-vm start ascend950_cluster_32_server_normal.yaml
```

After entering the `(hvm)$>` terminal, execute:

```bash
# Enable PyTorch mode
export HCCL_VM_ENABLE_PYTORCH=1
# Activate virtual environment (if applicable)
source /path/to/torch_test/bin/activate

# Install Runner when simulation execution is needed
hccl-vm plugin install @runner
```

All subsequent communicator configuration, environment variable settings, and Python startup are executed within this HCCL-VM terminal, so the Python process inherits the simulation environment.

## 2. Single Server Run

Take `118` configuration as an example: 1 server with 8 devices.

```bash
hccl-vm mock-comm 118

# Run the use case directly
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

## 3. Multi-Server Run

Take `128` configuration as an example: 2 servers with 8 devices each, totaling 16 ranks. Both servers must start concurrently using the same master address and port.

```bash
hccl-vm mock-comm 128
export HCCL_VM_ENABLE_PYTORCH=1
# Activate virtual environment (if applicable)
source /path/to/torch_test/bin/activate

# Start server 0
(
    export HCCL_VM_SERVER_INDEX=0
    python -u /path/to/torch/pytorch_test.py \
        --master_ip 127.0.0.1 \
        --master_port 50001 \
        --dev_num 8 \
        --server_num 2 \
        --server_index 0 \
        --data_list 1024 \
        --dtype int32 \
        --optype sum
) > server0.log 2>&1 &

# Start server 1
(
    export HCCL_VM_SERVER_INDEX=1
    python -u /path/to/torch/pytorch_test.py \
        --master_ip 127.0.0.1 \
        --master_port 50001 \
        --dev_num 8 \
        --server_num 2 \
        --server_index 1 \
        --data_list 1024 \
        --dtype int32 \
        --optype sum
) > server1.log 2>&1 &

# Wait for both servers to finish separately
```

| Configuration Item | Description |
| --- | --- |
| `HCCL_VM_ENABLE_PYTORCH=1` | Enable PyTorch adaptation; `HCCL_VM_SERVER_INDEX` only takes effect when set to `1` |
| `HCCL_VM_SERVER_INDEX` | Zero-based server index in current communicator configuration; set separately for each server subprocess |
| `--server_index` | Server index for test script; must match `HCCL_VM_SERVER_INDEX` for that process |
| `--dev_num` / `--server_num` | Device count per server and total server count; must match `mock-comm` configuration |
| `--master_ip` / `--master_port` | All servers use the same rendezvous address; processes in this example run on the same host machine using `127.0.0.1:50001`, which can be customized |
| `--data_list` / `--dtype` / `--optype` | This example executes AllReduce SUM on 1024 int32 elements |

Each server has local logical device numbers `0–7`; in this example, server 0 corresponds to global ranks `0–7` and server 1 corresponds to global ranks `8–15`. Setting environment variables within parentheses prevents the two server indexes from overwriting each other. The script calculates global ranks assuming equal device counts per server, so other scales should select communicator configurations with consistent device counts.

## 4. View Results

After all processes finish, check `server0.log` and `server1.log` for exit status, exceptions, and related log output.

Then use Checker for result verification:

```bash
hccl-vm plugin run @checker
```

After all use cases and verification complete, execute `exit` to leave the HCCL-VM terminal.
