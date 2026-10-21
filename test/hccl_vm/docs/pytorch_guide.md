# PyTorch 模式运行指南

HCCL-VM 现可运行部分 PyTorch 的 `aclgraph` 图模式相关的集合通信脚本。脚本应使用 `torch.multiprocessing.spawn` 启动各设备进程。多 server 用例需要在同一宿主机上启动多个 Python 主进程，分别模拟各 server。

## 1. 准备环境

先按 [README](../README.md) 完成 HCCL-VM 构建安装和 CANN 环境配置，准备已安装兼容版本 `torch`、`torch_npu` 的 Python 虚拟环境。

```bash
cd /path/to/CheckerL2/hccl_vm_install/bin

# 按需选择通信域配置文件，PyTorch暂时仅支持 Ascend950 系列。
# 如需使用 Ascend960 系列需要安装 A6 ops 包并修改拓扑集群配置中的 soc_version 为 Ascend960DT_961BY 等真实型号。
./hccl-vm start ascend950_cluster_32_server_normal.yaml
```

进入 `(hvm)$>` 终端后，执行：

```bash
# 启动 PyTorch 模式
export HCCL_VM_ENABLE_PYTORCH=1
# 激活虚拟环境（如有）
source /path/to/torch_test/bin/activate

# 需要模拟执行时安装 Runner
hccl-vm plugin install @runner
```

后续通信域配置、环境变量设置和 Python 启动均在该 HCCL-VM 终端内执行，使 Python 进程继承模拟环境。

## 2. 单 server 运行

以 `118` 配置为 1 个 server、8 个设备为例：

```bash
hccl-vm mock-comm 118

# 直接运行用例即可
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

## 3. 多 server 运行

以 `128` 配置为 2 个 server、每个 server 8 个设备，共 16 个 rank为例。两个 server 需要并发启动，使用相同的 master 地址和端口。

```bash
hccl-vm mock-comm 128
export HCCL_VM_ENABLE_PYTORCH=1
# 激活虚拟环境（如有）
source /path/to/torch_test/bin/activate

# 启动 server 0
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

# 启动 server 1
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

# 分别等待两个 server 运行结束
```

| 配置项 | 说明 |
| --- | --- |
| `HCCL_VM_ENABLE_PYTORCH=1` | 启用 PyTorch 适配；设置为 `1`时 `HCCL_VM_SERVER_INDEX` 才生效 |
| `HCCL_VM_SERVER_INDEX` | 当前通信域配置中从 0 开始的 server 序号；每个 server 的子进程分别设置 |
| `--server_index` | 测试脚本的 server 序号，须与该进程的 `HCCL_VM_SERVER_INDEX` 一致 |
| `--dev_num` / `--server_num` | 每个 server 的设备数和 server 总数，须与 `mock-comm` 配置一致 |
| `--master_ip` / `--master_port` | 所有 server 使用相同的 rendezvous 地址；本例进程在同一宿主机上，使用 `127.0.0.1:50001`，可自行设置 |
| `--data_list` / `--dtype` / `--optype` | 本例执行 1024 个 int32 元素的 AllReduce SUM |

每个 server 的本地逻辑设备号均为 `0–7`；本例 server 0 对应全局 rank `0–7`，server 1 对应全局 rank `8–15`。括号内设置环境变量可避免两个 server 的索引互相覆盖。脚本按每个 server 相同设备数计算全局 rank，因此其他规模应选择设备数一致的通信域配置。

## 4. 查看结果

等待全部进程结束后查看 `server0.log`、`server1.log`，关注退出状态、异常和相关日志输出。

随后可使用 Checker 进行结果校验：

```bash
hccl-vm plugin run @checker
```

全部用例和校验结束后，执行 `exit` 退出 HCCL-VM 终端。
