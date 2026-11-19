# PyTorch Mode Running Guide

HCCL-VM can now run collective communication scripts related to the `aclgraph` graph mode of PyTorch. Scripts should use `torch.multiprocessing.spawn` to start the process of each device. Multi-server cases require multiple Python main processes on the same host machine, each simulating one server.

## 1. Preparing the Environment

First, complete the HCCL-VM build, installation, and CANN environment configuration as described in the [README](../README.md), and prepare a Python virtual environment where compatible versions of `torch` and `torch_npu` are installed.

```bash
cd /path/to/CheckerL2/hccl_vm_install/bin

# Select a communicator configuration file as required. PyTorch currently supports only the Ascend950 series.
# To use the Ascend960 series, install the A6 ops package and change soc_version in the topology cluster configuration to a real model such as Ascend960DT_961BY.
# Enable this environment variable before startup so that the VM main process generates the rank table based on the local_rank of each server/board. Topologies with different device counts are supported.
export HCCL_VM_ENABLE_PYTORCH=1
./hccl-vm start ascend950_cluster_32_server_normal.yaml
```

After the `(hvm)$>` terminal is displayed, run:

```bash
# Activate the virtual environment (if any)
source /path/to/torch_test/bin/activate

# Install Runner if simulated execution is required
hccl-vm plugin install @runner
```

Perform subsequent communicator configuration, environment variable settings, and Python startup in this HCCL-VM terminal so that the Python process inherits the simulation environment.

## 2. Single-Server Running

The following example uses the `118` configuration, which contains one server and eight devices:

```bash
hccl-vm mock-comm 118

# Run the case directly
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

## 3. Multi-Server Running

The following example uses the `128` configuration, which contains two servers with eight devices each, totaling 16 ranks. The two servers must be started concurrently, using the same master address and port.

```bash
hccl-vm mock-comm 128
export HCCL_VM_ENABLE_PYTORCH=1
# Activate the virtual environment (if any)
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

# Wait for both servers to finish
```

| Configuration Item | Description |
| --- | --- |
| `HCCL_VM_ENABLE_PYTORCH=1` | Enables the PyTorch adaptation. `HCCL_VM_SERVER_INDEX` takes effect only when this variable is set to `1`. |
| `HCCL_VM_SERVER_INDEX` | Zero-based server index in the current communicator configuration. Set this variable separately for the subprocesses of each server. |
| `--server_index` | Server index of the test script, which must be the same as `HCCL_VM_SERVER_INDEX` of the process. |
| `--dev_num` / `--server_num` | Number of devices on each server and total number of servers, which must be consistent with the `mock-comm` configuration. |
| `--master_ip` / `--master_port` | Rendezvous address shared by all servers. In this example, the processes run on the same host machine and use `127.0.0.1:50001`. You can set your own address. |
| `--data_list` / `--dtype` / `--optype` | In this example, an AllReduce SUM is performed on 1024 int32 elements. |

The local logical device ID of each server ranges from 0 to 7. In this example, server 0 corresponds to global ranks 0–7, and server 1 corresponds to global ranks 8–15. Setting environment variables inside parentheses prevents the indexes of the two servers from overwriting each other. The script calculates global ranks based on an equal number of devices on each server. Therefore, for other scales, select a communicator configuration with the same number of devices on each server.

## 4. Viewing Results

After all processes finish, check `server0.log` and `server1.log`, focusing on the exit status, exceptions, and related log output.

Then use Checker to verify the results:

```bash
hccl-vm plugin run @checker
```

After all cases and verifications are complete, run `exit` to exit the HCCL-VM terminal.
