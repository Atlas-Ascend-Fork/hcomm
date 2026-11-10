# HCCL-VM 指导手册

## 1. 概述

HCCL-VM是面向华为昇腾NPU卡的高性能集合通信的虚拟执行环境，该工具旨在无真实昇腾硬件的条件下，实现HCCL集合通信算子的开发和功能验证。

工具提供三种运行模式：`normal`（真实多卡通信，结果正确性无损）、`check-only`（仅校验，大块内存复用共享池，大幅压低内存占用）、`estimate`（估算，跳过真实通信阻塞，快速产出任务清单），支持启动时指定与运行期通过 `hccl-vm mode` 命令即时切换，详见[运行模式](#49-运行模式normal--check-only--estimate)。

工具同时提供两种模拟等级：默认等级 2（南向劫持）与等级 1（北向劫持，`--level 1` 指定），详见[模拟等级与 level1 劫持](#410-模拟等级与-level1-劫持北向劫持)。

![hccl-vm GIF](docs/hccl-vm.gif)

## 2. 前置依赖

|   依赖项   |     版本要求     |
| -------- | ------------- |
| 系统架构  | x86_64 Ubuntu22.04及以上 |
| 规格约束  | Ascend950，其余参照[约束详情](#45-工具规格约束) |
| QEMU（仅 AICPU 模式运行需要，x86_64 宿主） | 系统自带 qemu-user-static 可直接使用；版本低于 10.2.4 时无法限制翻译代码缓存、QEMU 常驻内存较大，大规模场景可能引发换页、明显影响性能，可按 [2.3 QEMU 部署](#23-qemu-部署仅-aicpu-模式x86_64-宿主需要) 安装 10.2.4 及以上版本优化。原生 ARM 主机不需要 QEMU；编译（`build.sh`/`build_pkg.sh`）不依赖本机已安装 QEMU |
| Python3（仅估算模式 @estimate 性能仿真插件运行需要） | ≥3.8（含 sqlite3 标准库），另需预装 simpy 离散事件仿真包（`pip3 install simpy`），详见 [4.9.5](#495-估算插件estimate与依赖说明) |

### 2.1 CANN包安装

安装最新版本CANN Toolkit开发套件包和CANN ops算子包 [下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)

```bash
# 确保安装包具有可执行权限
chmod +x Ascend-cann-toolkit_9.1.0_linux-x86_64.run
chmod +x Ascend-cann-950-ops_9.1.0_linux-x86_64.run
# 安装命令
./Ascend-cann-toolkit_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
./Ascend-cann-950-ops_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
```


### 2.2 hccl_test编译

hccl_test是昇腾官方提供的HCCL性能测试工具，详见[HCCL性能测试工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/devaids/hccltool/HCCLpertest_16_0001.html)，HCCL-VM支持在虚拟环境中运行hccl_test用例。请先参照[hccl_test用例构建](#42-hccl-test用例构建)章节进行用例二进制程序的编译。

备注：hccl_test 为可选依赖。运行 PyTorch 用例请参见 [PyTorch 用例示例](#33-pytorch用例示例)。

### 2.3 QEMU 部署（仅 AICPU 模式、x86_64 宿主需要）

AICPU 展开模式在 x86_64 宿主上需要 QEMU 用户态模拟（linux-user 模拟 aarch64）。系统自带的 qemu-user-static 可直接使用；版本低于 **10.2.4** 时不支持 `-tb-size` 限制翻译代码缓存，QEMU 常驻内存较大，可能严重影响性能，按照如下操作安装 **10.2.4 及以上**版本。

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

运行行为与提示：

- 设备进程启动时打印实际使用的 QEMU 程序路径、版本与 `-tb-size` 请求值。
- `HCCL_VM_QEMU_PATH` 未设置/无效、或指向不支持 `-tb-size` 的旧版 QEMU 时**不会中断**：自动回退（按 PATH 查找 `qemu-aarch64-static` / 不传 `-tb-size`）并打 WARN，包含实际回退到的程序与版本、以及“翻译缓存未限制，QEMU 内存可能过大”提示。仅当全机找不到任何 `qemu-aarch64-static` 时 AICPU 模式启动失败。
- `HCCL_VM_QEMU_TB_SIZE` 为 QEMU 翻译代码缓存（MiB）：未设置默认 `16`；`0` 表示委托 QEMU 自身默认策略（不是禁用缓存）。不限制缓存可能会显著增大 QEMU 常驻内存并引发换页。

---

## 3. 快速上手

### 一键安装

拉取完整 CheckerL2 源码后，在仓库根目录执行：

```bash
bash ./hccl_vm_installer
```

### 3.1 手动构建&安装

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

# 4. 编译HCCL-VM工具，下载hcomm代码之后，工具源码所在路径：/home/workspace/hcomm/test/hccl_vm
cd /home/workspace/hcomm/test/hccl_vm
source /home/workspace/Ascend/cann/set_env.sh
export HCCL_CODE_HOME=/home/workspace/hccl
export HCOMM_CODE_HOME=/home/workspace/hcomm
bash ./build.sh --full

# 5.从CANN安装目录拷贝解压aicpu_hcxx.tar.gz
bash build_pkg.sh
```

### 3.2 使用示例

#### 3.2.1 环境配置

若/etc路径下已存在hccl_rootinfo.json文件，请手动删除该文件。

通信域规模超过 128 卡（129～1024）时，需切换为MariaDB数据库，见 [MariaDB 数据库使用](src/storage_backends/mariadb/README.md)；完成部署并设置 `HCCL_VM_STORAGE_CONFIG` 后回到本节，`unset` 该变量即恢复默认SQLite数据库。
<!-- internal-only:end -->

#### 3.2.2 CCU模式

1. 环境变量配置。

```bash
# 进入工具安装目录
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
```

2. 执行


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

3. 验证hccl_test用例运行结果
[Runner结果查看](#481-runner插件结果) 
[Checker结果查看](#482-checker插件结果)

#### 3.2.3 AICPU模式

AICPU展开模式需要将算法展开步骤放到设备侧执行，因此hccl-vm工具需要将HCCL的设备侧的符号编译并模拟执行。由于设备侧符号是ARM架构的，因此在X86环境上编译时需要借助交叉编译器，运行时需要借助QEMU实现AICPU模式的模拟运行。

X86宿主上运行AICPU模式借助QEMU用户态模拟：未配置 `HCCL_VM_QEMU_PATH` 时自动回退 PATH 中的 `qemu-aarch64-static` 并打告警（含实际版本），不影响启动；系统自带版本低于 10.2.4 时无法限制翻译代码缓存、内存占用较高，大规模场景可能引发换页、影响性能，可参考 [2.3 QEMU 部署](#23-qemu-部署仅-aicpu-模式x86_64-宿主需要) 安装 10.2.4 及以上版本并配置 `HCCL_VM_QEMU_PATH`。原生ARM主机不需要QEMU。

1. 环境变量配置。

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

2. 执行

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

3. 验证hccl_test用例运行结果 [Runner结果查看](#481-runner插件结果) [Checker结果查看](#482-checker插件结果)

#### 3.2.4 HostDPU模式

HostDPU展开模式运行时的环境变量配置等均与AICPU模式相同，仅在集群规模和通信域规模上有如下差异：

1. 集群规格配置需使用 `./hccl-vm start ascend950_cluster_32_server_normal_hostdpu.yaml`
2. 需要Server数量 >= 2，即不支持单Server的通信域（不支持 112.yaml、114.yaml、118.yaml等）

#### 3.2.5 AIV模式

1. 环境变量配置。

```bash
# 进入工具安装目录
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AIV"
```

2. 执行

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

3. 验证hccl_test用例运行结果 [Runner结果查看](#481-runner插件结果) [Checker结果查看](#482-checker插件结果)

### 3.3 PyTorch用例示例

HCCL-VM 支持运行部分 PyTorch 的 `aclgraph` 图模式集合通信脚本。脚本应使用 `torch.multiprocessing.spawn` 启动各设备进程；多 server 用例需要在同一宿主机上并发启动多个 Python 主进程，分别模拟各 server。完整说明见 [PyTorch 模式运行指南](docs/pytorch_guide.md)。

#### 3.3.1 环境准备

先完成 HCCL-VM 构建安装和 CANN 环境配置，并准备已安装兼容版本 `torch`、`torch_npu` 的 Python 虚拟环境。PyTorch 暂时仅支持 Ascend950 系列；如需使用 Ascend960 系列，需要安装 A6 ops 包，并将拓扑集群配置中的 `soc_version` 修改为 `Ascend960DT_961BY` 等真实型号。

```bash
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin
./hccl-vm start ascend950_cluster_32_server_normal.yaml
```

进入 `(hvm)$>` 终端后执行以下命令。后续通信域配置、环境变量设置和 Python 启动均在该终端内完成，使 Python 进程继承模拟环境。

```bash
# 启用 PyTorch 适配
export HCCL_VM_ENABLE_PYTORCH=1
# 激活虚拟环境（如有）
source /path/to/torch_test/bin/activate

# 需要模拟执行时安装 Runner
hccl-vm plugin install @runner
```

#### 3.3.2 单 server 运行

以 `118` 配置为例，使用 1 个 server、8 个设备，执行 1024 个 int32 元素的 AllReduce SUM。以下命令参数对应指南中的测试脚本，使用其他脚本时请按其参数定义调整。

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

#### 3.3.3 多 server 运行

以 `128` 配置为例，使用 2 个 server、每个 server 8 个设备，共 16 个 rank。在已完成环境准备的 HCCL-VM 终端内执行：

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

`HCCL_VM_SERVER_INDEX` 是当前通信域配置中从 0 开始的 server 序号，仅在 `HCCL_VM_ENABLE_PYTORCH=1` 时生效，须与对应脚本的 `--server_index` 一致。括号内设置环境变量可避免不同 server 的索引互相覆盖。

所有 server 必须使用相同的 `--master_ip` 和 `--master_port`；`--dev_num`、`--server_num` 必须与 `mock-comm` 配置一致。每个 server 的本地逻辑设备号均为 `0–7`，本例 server 0 对应全局 rank `0–7`，server 1 对应全局 rank `8–15`。脚本按每个 server 相同设备数计算全局 rank，其他规模应选择各 server 设备数一致的通信域配置。

> **POD形态（4层拓扑）下的对应关系**：此处的"server"指通信域中的**叶子节点（host节点）**。
> server形态下叶子节点就是 server；**POD形态下叶子节点是 board**，因此
> `--server_num`/`HCCL_VM_SERVER_INDEX` 应按 **board** 计数与编号（如 `11_64` 配置为 1个POD × 8个board × 8卡，
> 需并发启动 8 个 Python 主进程，`--server_num 8`、`--dev_num 8`、`HCCL_VM_SERVER_INDEX` 取 `0~7`），
> 每个进程的本地逻辑设备号为 board 内的 `0–7`，与真机一致。

#### 3.3.4 结果查看与校验

等待全部进程结束后，查看对应的 `server0.log`、`server1.log`（多 server 场景），关注退出状态、异常和相关日志输出。随后在 HCCL-VM 终端内执行 Checker 校验，结果说明见 [Checker 插件结果](#482-checker插件结果)。

```bash
hccl-vm plugin run @checker

# 全部用例和校验结束后退出工具终端
exit
```

### 3.4 hccl代码修改验证示例

若您修改了CANN的算子包代码，如新增了算法类型，为保证您的修改生效，需要按照如下步骤操作执行。build_pkg.sh脚本帮助用户执行编包、装包、拷贝Device侧依赖符号，执行前需设置环境变量:

```bash
# 假设您的CANN安装目录为：/home/workspace/Ascend
source /home/workspace/Ascend/cann/set_env.sh
# 配置hccl代码仓路径
export HCCL_CODE_HOME=/home/workspace/hccl
# 配置hcomm代码仓路径
export HCOMM_CODE_HOME=/home/workspace/hcomm
```

1. 若您更新/修改了CANN hccl仓代码，请执行`bash build_pkg.sh --install hccl`。
2. 若您更新/修改了CANN hcomm仓代码，请执行`bash build_pkg.sh --install hcomm`。
3. 若您同时更新/修改了CANN hccl、hcomm仓代码，请执行`bash build_pkg.sh --full`。
4. 参考[使用示例](#32-使用示例)步骤，重新运行用例。

---

## 4. 详细指导

### 4.1 工具环境变量配置

**HCCL-VM环境变量说明**：

| 环境变量                      | 用途                                                                                                        | 示例                                                                        |
| ------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------- |
| `HCCL_CODE_HOME`         | HCCL-VM编译指定HCCL源码路径。默认未配置。     | `export HCCL_CODE_HOME=/home/workspace/hccl`                     |
| `HCOMM_CODE_HOME`         | HCCL-VM编译指定HCOMM源码路径。默认未配置。     | `export HCOMM_CODE_HOME=/home/workspace/hcomm`                     |
| `HCCL_VM_ENABLE_PYTORCH` | 设置为 `1` 时启用 PyTorch 适配，运行方式见 [PyTorch 用例示例](#33-pytorch用例示例)。 | `export HCCL_VM_ENABLE_PYTORCH=1` |
| `HCCL_VM_SERVER_INDEX` | PyTorch 进程模拟的**叶子节点（host节点）**序号，从 `0` 开始；仅在 `HCCL_VM_ENABLE_PYTORCH=1` 时生效，须与测试脚本的 `--server_index` 一致。server形态下叶子节点即 server；**POD形态（4层拓扑）下叶子节点为 board**，故按 board 计数与编号。 | `export HCCL_VM_SERVER_INDEX=0` |
| `HCCLVM_ENABLE_DUMP_DATA` | 1. 使能性能仿真文件输出落盘。若使能，则在测试用例执行过程中输出3个runner_hcclvm_*_data.bin文件和1个ccu_channel_jetty_config.xml文件供性能仿真使用. <br> 2.使能Runner插件dump input\&output数据。若使能，则在测试用例执行过程中，会将每个算子的input\&output数据dump到all\_rank\_input\_output.txt文件中 | `export HCCLVM_ENABLE_DUMP_DATA=1 使能，export HCCLVM_ENABLE_DUMP_DATA=0 或不配置该环境变量 禁用` |
| `HCCL_VM_RES_STAT_DUMP` | 使能劫持层资源统计落盘：设置后，业务进程 finalize 时将各设备资源计数（内存/Jetty/QP/Notify/AIV core 等）导出为 CSV，输出到安装目录 `data/stat/<podId>_<serverId>_<boardId>_<设备ID>_res_<时间戳>.csv`（`boardId` 为 POD 形态 4 层拓扑下的 board 序号，server 形态恒为 `0`）；未设置（默认）不导出。 | `export HCCL_VM_RES_STAT_DUMP=1` |
| `HCCL_VM_QEMU_PATH` | 仅AICPU模式、x86_64宿主生效：外部QEMU程序的非空绝对路径（目标为可执行普通文件，允许指向普通文件的符号链接）。未设置/无效时自动回退 PATH 查找 `qemu-aarch64-static` 并打 WARN（含实际程序与版本）；全机找不到时 AICPU 启动失败。部署步骤见[2.3 QEMU 部署](#23-qemu-部署仅-aicpu-模式x86_64-宿主需要) | `export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static` |
| `HCCL_VM_QEMU_TB_SIZE` | 仅AICPU模式、x86_64宿主生效：QEMU翻译代码缓存（MiB）。未设置默认`16`；`0`表示显式传`-tb-size 0`、由QEMU自身默认策略决定（不是禁用缓存）；仅接受可解析到uint32_t的十进制数字串，非法值告警并回退默认16。指向不支持`-tb-size`的旧版QEMU时不传该选项并告警内存可能过大。 | `export HCCL_VM_QEMU_TB_SIZE=16` |
| `HCCL_UB_MULTI_CHANNEL_NUM` | HCCL 侧环境变量，由 proxy 桩读取（`HcclConfigGetInfo` 的 `HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM` 查询项同口径）：一个 channel 上并行的 jetty 数。随算子落库到 `opDetails.ubMultiChannelNum`，`@estimate` 轻量性能仿真插件据此折算端口带宽利用率（`multi_port_utilization_low` 各档 `+= (n-1)×jetty_utilization_step`，上限 1.0；见 `docs/light_performance/simulator_design.md` 7.2 节）。未设置/非法/`0` 回落 `1`（不使能多 jetty，直接用原利用率表）。 | `export HCCL_UB_MULTI_CHANNEL_NUM=8` |

### 4.2 HCCL-Test用例构建

hccl_test用例源码在CANN包安装目录下，支持OpenMPI和MPICH两种环境编译、运行，运行时差异详见[OpenMPI和MPICH环境用例执行差异](#47-openmpi和mpich环境用例运行差异)，本用例指导中以OpenMPI环境为例。

#### 4.2.1 OpenMPI环境编译

1. 安装OpenMPI

```bash
sudo apt-get update
sudo apt install openmpi-bin libopenmpi-dev
```

2. 编译hccl_test

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

#### 4.2.2 MPICH环境编译

假设mpich的路径为: `/usr/lib/mpich`。

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

### 4.3 昇腾集群拓扑配置文件说明

#### 4.3.1 Server/Pod拓扑配置文件说明

昇腾集群拓扑是由一个或多个Server/Pod形态的子拓扑，按照CLOS分层网络规则组合而成的。因此用户在生成集群拓扑前，需要确认每个Server/Pod的拓扑类型。
用户既可以选择HCCL-VM工具提供的预定义拓扑类型，也可以根据配置文件格式要求，自定义Server/Pod的拓扑类型。

描述一个Server/Pod的拓扑网络关系，主要包括以下方面：

 - **端口配置表**: 描述一个NPU卡的物理端口配置信息，如NPU与NPU直连端口（P2P），NPU出框端口（P2NET）等。
 - **链路配置表**: 描述一个Server/Pod内的所有NPU卡之间的连接关系，如全连接（Full Mesh）等。
 - **PortBound**: 描述一个NPU卡的某些端口的绑定关系，即多个端口绑定成一个PortGroup。

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

**字段说明**：

 - **soc_version**: 芯片型号，如 `Ascend950`。
 - **device_num**: 设备总数，根据芯片型号和拓扑类型确定。server形态为该server的卡数；**POD形态为该POD的总卡数**（如1dpod64为64）。
 - **board_num**: 可选，一个POD内的board（计算板）数量。**POD形态的触发字段**：缺省或为1时是server形态（3层拓扑），大于1时是POD形态（4层拓扑，见下方说明）。约束：`device_num % board_num == 0`、`board_num <= 255`。
 - **device_num_per_board**: 可选，每board卡数（一般为8，但不写死）。缺省由 `device_num / board_num` 推导；显式配置时必须与推导值一致。
 - **form_type**: 可选，`"server"` 或 `"pod"`（不区分大小写）。缺省按 `board_num` 推导；显式配置且与推导结果矛盾时报错。
 - **net_instance_scope**: 可选，逐层指定 `net_instance_id` 的作用域，取值 `board`/`pod`/`superpod`/`cluster`（详见4.3.4节）。缺省推导规则：server形态 `{layer0: pod, layer1: superpod, layer>=2: cluster}`；POD形态 layer0 按"该层是否存在PEER2PEER板内直连"推导（有直连→`board`，纯交换网→`pod`），其余同server形态。
 - **device_ports_allocate_map**: 端口分配表，描述每个die的端口配置。1表示device直连端口，2表示device连交换机端口，3表示d2h端口。
 - **port_group**: 描述哪些port合并为一个portGroup，相同portGroup的port对应IP地址相同。没有配置的port，则默认每个port为一个portGroup。
 - **links**: 链路配置表，描述Server/Pod内的所有NPU卡之间，以及NPU与交换机之间的连接关系。
   - **NPU直连关系**: 工具提供了两种方式配置NPU直连关系：
    - **link_mode == "fullmesh"**: 表示所有Device基于一个Die的Port进行全连接。后续有新增典型的连接方式，可以新增link_mode类型，如"ring"。
    - **link_mode == "enum"**: 枚举法。当Server/Pod内的NPU连接方式比较复杂时，可以通过枚举所有的链路关系来描述。
   - **NPU与交换机连接关系**: 用户可以通过枚举法配置NPU与交换机之间的连接关系。
  - **device_to_device_links**: 描述NPU与NPU之间的连接关系。两端device用 `src_device_id`/`src_device_id_range`
    与 `dst_device_id`/`dst_device_id_range` 声明（**board内物理id**；旧命名 `src_local_id`/`dst_local_id`/`*_local_id_range` 仍兼容）。
    POD形态下可用 **`src_board_id`/`dst_board_id`** 分别声明两端所属board（须同时配置），**支持跨board直连**。
  - **device_to_switch_links**: 描述NPU与交换机之间的连接关系。POD形态下用单个 **`board_id`** 声明device所属board（dst为交换机，无需dst_board_id）。

> **注意（POD形态）**：`device_group`/`port_list`/`port_group`/`links` 中出现的 device id，
> 在POD形态下必须是 **board内物理id**（如 `0~7`），即卡结构按"一个board"描述；`local_id` 由工具按
> `board_id × 每board卡数 + board内物理id` 换算得到。links条目的board归属字段：
> **`fullmesh` 的 `connections`、`device_to_switch_links`、`matrix`** 用单个 **`board_id`**（两端同board）；
> **`device_to_device_links`** 用 **`src_board_id`/`dst_board_id`**（两端可不同board，即跨board直连）。
> 未配置board字段的条目对所有board生效（即单board描述复制扩展）。
> 若误用POD全局id（如 `8`、`63`），工具会直接报错（`device id 超出值域[0, 每board卡数-1]`）。

**POD形态（4层拓扑）说明**

拓扑分两种形态，层级数不同：

| 形态 | 层级 | 叶子节点（host节点） | 卡物理id | 产物 |
|------|------|---------------------|----------|------|
| server形态 | server → 超节点 → cluster（3层） | server | `[0, device_num-1]` | 每server一份 `topo.json`/`hccl_rootinfo.json` |
| POD形态 | **board → pod → 超节点 → cluster（4层）** | **board** | **board内 `[0, 每board卡数-1]`**（一般`[0,7]`） | **每POD一份 `topo.json`**（server目录）+ **每board一份 `hccl_rootinfo.json`** |

POD形态下，一个POD内的卡分为若干 board（计算板），每个board是一个独立host节点（独立host IP、独立rootinfo），
**topo.json为POD级单份**放在server目录下，其中 `local_id` 为**POD全局id**：
`local_id = board_id × 每board卡数 + board内物理id`。
例如 `1dpod64`（64卡POD）配置 `device_num: 64` + `board_num: 8`，生成的集群目录为：

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

配置示例（`asset/cluster_model/config/server_or_pod/ascend950_server_topo_1dpod64.yaml`，板内直连，links按board逐一描述）：

```yaml
soc_version: "Ascend950"
device_num: 64                       # POD总卡数
board_num: 8                         # 8个board → 每board 8卡
hardware_type: "950D-1D-POD-64P-topo"

# 卡结构为“单个board”的描述, device id 为 board 内物理id(0~7)
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

配置示例（跨board直连场景，需配置 `src_board_id` ≠ `dst_board_id`，layer0 直连域跨 board 时作用域自动为 pod）：

```yaml
soc_version: "Ascend950"
device_num: 16                        # 2个board × 8卡
board_num: 2
hardware_type: "950D-1D-POD-16P-topo"

device_group:
  - {group_id: 1, device_list: [0, 1, 2, 3]}
  - {group_id: 2, device_list: [4, 5, 6, 7]}

links:
  - link_mode: "fullmesh"             # 每board内8卡经die1全互联
    connections:
      - {board_id: 0, die_id: 1, net_instance_id: 0, topo_instance_id: 0, device_id: [0, 1, 2, 3, 4, 5, 6, 7]}
      - {board_id: 1, die_id: 1, net_instance_id: 1, topo_instance_id: 1, device_id: [0, 1, 2, 3, 4, 5, 6, 7]}
  - link_mode: "enum"
    device_to_device_links:
      # 跨board直连: src_board_id(0) ≠ dst_board_id(1), die0的4卡交叉直连
      - {src_board_id: 0, net_instance_id: 0, topo_instance_id: 0, src_die_id: 0, src_device_id: [0, 1, 2, 3],
         dst_board_id: 1, dst_die_id: 0, dst_device_id: [4, 5, 6, 7]}
    device_to_switch_links:           # dst为交换机, 只需单个board_id声明device所属board
      - {board_id: 0, die_id: 0, devices_range: [0, 7]}
      - {board_id: 1, die_id: 0, devices_range: [0, 7]}
      # ... board0 die1 / board1 die1 同理
```

配置示例（`ascend960_pod_topo_clos.yaml`，框内全交换、无板内直连，links不带board_id即复制扩展到全部board）：

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

**POD形态的强制校验**（任一不满足即生成失败）：
1. `device_num % board_num == 0`，每board卡数 > 0；
2. 配置中所有 device id 必须落在 `[0, 每board卡数-1]`；links条目的 `board_id` 及 `device_to_device_links` 的
   `src_board_id`/`dst_board_id`（若配置）必须落在 `[0, board_num-1]`，且 `src_board_id`/`dst_board_id` 须同时配置；
3. `board_num <= 255`（IP的CC段容量）、每board endPoint数 `<= 255`（IP的DD段容量）；
4. layer0 存在 PEER2PEER 直连时：**板内直连**（各board独立成网）作用域必须为 `board`（保证 layer0 的
   `net_instance_id` 每board唯一，否则多个board的板内直连组会被业务侧误合并）；**跨board直连**
   （存在 `src_board_id ≠ dst_board_id` 的边，直连域为单一POD级实例）作用域必须为 `pod`
   （否则跨board直连边两端 `net_instance_id` 不一致，会被业务侧过滤而失效）；
5. 存在 PEER2PEER 直连时，各board直连边数必须一致且每个board都有直连条目（同构校验；跨board边对两端board各计1次）；
6. `net_instance_scope` 声明的 layer 必须在拓扑中真实存在。

> `fullmesh`/`device_to_switch_links`/`matrix` 条目用单个 `board_id`（两端同board）；`device_to_device_links`
> 用 `src_board_id`/`dst_board_id` 声明两端归属，**支持跨board PEER2PEER 直连**（如 hf：board0↔board1 全交叉）。
> ranktable.json 的 `device_id` 为board内物理id、`local_id` 为POD全局id（与POD级topo.json同口径）。

#### 4.3.2 集群拓扑配置文件说明

昇腾集群网络是由一个或多个Server/Pod形态的子拓扑，按照CLOS分层网络规则组合而成的。用户可根据集群规模和需求，选择不同的Server/Pod拓扑类型。

用户可按照如下配置文件格式，自定义集群拓扑配置文件：

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

上述配置文件描述了一个包含4个超节点，32个Server，共128个NPU卡的集群拓扑。其中，每个Server/Pod采用ascend950_server_topo_normal拓扑类型。

**字段说明**：

 - **super_node_num**: 超节点总数量。
 - **server_num**: sever/pod总数量。
 - **server_list**: 每个Server/Pod的配置信息，包括超节点ID、设备ID范围、芯片型号、Server/Pod拓扑配置文件路径。

> **POD形态的规模换算**：`server_num` 统计的是 server/POD 数量。若 `server_topo` 指向的拓扑配置声明了
> `board_num > 1`（POD形态），则集群的**叶子节点（host节点）总数 = server_num × board_num**，
> 总卡数 = 叶子节点数 × 每board卡数。例如 `ascend950_cluster_16_server_1dpod64`（`server_num: 16`，
> 拓扑 `board_num: 8`、每board 8卡）共 128 个叶子节点、1024 张卡，`servers_info.json` 会有 128 条记录。
>
> `servers_info.json` 的每条记录对应一个叶子节点，并带显式的 `super_pod_id`/`server_id`/`board_id`
> 三元id（工具按id匹配，而非按条目位置序匹配），host IP 规则为：
> server形态 `AA.BB.0.0`（AA=超节点id+1，BB=server id+1）；POD形态 `AA.BB.CC.0`（CC=board_id）。

#### 4.3.3 通信域配置文件说明

用户在昇腾集群环境中，需要根据待执行算子所需的通信域不同，选择不同的通信域配置文件。

工具提供了hccl-vm mock-comm命令读取和配置算子通信域配置文件。通信域配置文件格式为yaml，路径为hccl_vm_install/config/topo_meta。若目录中没有对应的通信域配置文件，则用户需要先创建一个。

hccl-vm工具支持非对称拓扑通信域配置。如下所示：

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

**注意事项**：

- 配置通信域时，工具会根据指定的通信域配置编号重新生成topo.json和ranktable.json文件。
- 上述通信域配置yaml文件中，ranks字段表示每个server实际跑的rank的local id（即device的物理ID）列表。

当通信域规模较大时，servers条目中的server选择字段与rank选择字段可自由组合使用，支持以下字段：

- **server选择字段**（二选一，不可同时配置）：
  - `serId`: 单个server的id或server的id列表（如`3`或`[0, 1]`，列表表示列举的每个server）；
  - `serId_range`: server的id范围，值为`[<起始id>, <结束id>]`（闭区间，仅支持列表形式），表示该超节点下id从起始至结束的所有server。
- **rank选择字段**（二选一，不可同时配置）：
  - `ranks`: 每个server实际跑的rank的local id列表；
  - `ranks_range`: local id范围，值为`[<起始id>, <结束id>]`（闭区间，仅支持列表形式），表示每个server选取id从起始至结束的所有local id。

如下示例（2个超节点、共4个server、32个rank）中，同一文件内多种组合混用：

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

**注意事项**：

- serId与serId_range不可同时配置，ranks与ranks_range不可同时配置。
- serId_range与ranks_range仅支持`[<起始id>, <结束id>]`列表形式（闭区间），不支持`<起始id>-<结束id>`格式；起始id不能大于结束id，且范围宽度（结束id-起始id+1）不能超过1024。
- 一个serId_range条目会展开为该超节点下id从起始至结束的所有server，一个serId列表条目表示列举的每个server，每个server的ranks均相同。
- 若server条目中未配置rank选择字段，则该server的ranks为空列表；若未配置server选择字段，则默认serId为0。
- 上述写法适用于 **server形态**（3层拓扑）：`ranks` 为 server 内物理id，一个 server 即一个叶子节点。

**POD形态（4层拓扑）的 boards 配置**

POD形态下叶子节点是 board（不是 POD），因此通信域需要按 board 声明参与的卡，
且 `ranks` 为 **board内物理id**（如 `[0, 7]`，而不是POD内的 `[0, 63]`）：

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

- **board选择字段**（二选一，不可同时配置）：
  - `boardId`: 单个board的id或id列表（如`3`或`[0, 1]`）；
  - `boardId_range`: board的id范围，`[<起始id>, <结束id>]`（闭区间，宽度不超过1024）。
- **rank选择字段**：`ranks` / `ranks_range`，语义同上，但值域为 **board内物理id**。
- `boards` 为可选字段：**未配置时视为该 server 下只有一个隐式 `board0`**，
  此时 `ranks`/`ranks_range` 直接写在 server 条目下（即server形态的既有写法，完全向后兼容）。
- `boards` 与 server 条目上的 `ranks`/`ranks_range` **互斥**，同时配置会报错（无法确定rank归属）。
- `boardId`/`boardId_range` 未配置时默认归属 `board0`。
- rank_id 的分配顺序为：超节点 → server(POD) → board → device，从0递增。

工具已提供的POD形态通信域配置（`asset/cluster_model/topo_meta/`）：

| 编号 | 规模 | 说明 |
|------|------|------|
| `118` | 1超节点 × 1POD × 1board × 8卡 = 8 rank | 单board（server形态写法，POD形态下即board0） |
| `11b2` | 1 × 1 × 2board × 8卡 = 16 rank | 单POD内2个board |
| `11b4` | 1 × 1 × 4board × 8卡 = 32 rank | 单POD内4个board |
| `11b9` | 1 × 1 × 9board × 8卡 = 72 rank | POD72（ascend960_pod_topo_clos）整POD |
| `11_64` | 1 × 1 × 8board × 8卡 = 64 rank | 1dpod64 整POD |
| `12_64` / `14_64` / `18_64` / `116_64` | 2 / 4 / 8 / 16 个POD，每POD 8board × 8卡 | 128 / 256 / 512 / 1024 rank |

#### 4.3.4 topo和ranktable.json文件说明

`topo.json` 和 `ranktable.json` 文件不需要手动创建，工具会根据以下信息自动生成：

- **拓扑配置编号**：用户在启动时指定的编号（如 112、113 等）
- **芯片类型**：根据运行环境自动识别的芯片类型。

虽然配置文件由工具自动生成，但了解其结构有助于理解拓扑配置。

**topo.json 结构**：

`topo.json` 描述了一个 server 内所有设备的连接关系：

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

**字段说明**：

- `server.device_count`：设备总数。
- `server.groups`：设备分组信息。
- `ports`：端口配置。
  - `usage`：端口用途（`peer2peer` 表示设备间连接，`peer2net` 表示与外部连接）
- `links`：链路配置。
  - `link_type`：链路类型（`PEER2PEER` 或 `PEER2NET`）
  - `topo_type`：拓扑类型（`1DMESH`、`CLOS` 等）

**ranktable.json 结构**：

`ranktable.json` 描述了本次通信域使用的设备与其各网络层级的地址（EID）映射，
由工具在 `mock-comm` 时按通信域配置自动生成到 `hccl_vm_install/data/ranktable.json`：

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

**字段说明**：

- `rank_count`：本次通信域的rank总数。
- `rank_id`：通信域内全局rank编号，按 **超节点 → server(POD) → board → device** 顺序从0递增。
- `device_id`：**叶子节点内的物理id**（物理id，不是逻辑id）。即业务侧 `aclrtSetDevice()` 选中的物理设备号；
  hcomm 的 `CheckRankGraphAddrs` 按 `rank.deviceId == devPhyId` 筛选条目，并以该 phyId 查询硬件EID。
  server形态下值域为 `[0, server内卡数-1]`；**POD形态下值域为 `[0, board内卡数-1]`（一般 `[0,7]`）**。
- `local_id`：与 `topo.json` 的 `local_id` 同口径，业务侧按 ranktable 的 `local_id` 匹配 topo.json 中的边。
  server形态下与 `device_id` 同值（本工具中 `logic_id == physical_id`）；**POD形态下 topo.json 为POD级单份
  （server目录），`local_id` 为POD全局id = `board_id × 每board卡数 + board内物理id`**（如1dpod64值域 `[0,63]`），
  `device_id` 仍为board内物理id。
- `level_list[].net_layer`：网络层级编号（0为节点内直连层，1为出节点/出框层，2及以上为更外层，d2h/HOST层通常为3）。
  **引入board层后 net_layer 编号规则不变**，board维度通过 `net_instance_id` 表达。
- `level_list[].net_instance_id`：该层网络实例标识，按**作用域(scope)**拼装：

  | 作用域 | 格式 | 典型适用 |
  |--------|------|----------|
  | `board` | `sp_{超节点}_srv_{POD}_board_{board}_idx_{k}` | POD形态 layer0（板内直连，**每board唯一**） |
  | `pod` | `sp_{超节点}_srv_{POD}_idx_{k}` | server形态 layer0；POD形态跨board的框内交换网 |
  | `superpod` | `sp_{超节点}_idx_{k}` | layer1（跨server/POD但同超节点，出框互通） |
  | `cluster` | `cluster` | layer≥2（集群级、HOST/d2h层） |

- `level_list[].net_type`：`net_layer=0` 时为 `TOPO_FILE_DESC`，`net_layer>=1` 时为 `CLOS`。
- `level_list[].rank_addr_list[].addr`：该层端口绑定的EID（去掉 `0x` 前缀）；UBOE协议层为点分IPv4（`addr_type: "IPV4"`）。
- `level_list[].rank_addr_list[].plane_id`：PEER2PEER层固定 `plane_0`；PEER2NET层继承配置中的 `plane_pg_{N}`。

> **层级裁剪规则**：某层要在 ranktable 中出现，要求该层在 rootinfo 中分配了EID；
> 通信域只涉及 **1个叶子节点**（server形态=1个server，POD形态=1个board）且不跨超节点时，`net_layer>=3` 不输出；
> 单叶子节点的 net_layer 总层数 > 3 时，只保留编号最小的3层；跨超节点通信域不执行上述裁剪。

### 4.4 hccl\_config.sh文件说明

hccl\_config.sh文件中包含了HCCL\_Test用例运行所需的环境变量配置。其中的环境变量与Hccl\_Test用例在真机环境中的环境变量一致。
用户需要根据自己的用例和需求，修改hccl\_config.sh脚本，配置HCCL用例运行环境变量。

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

### 4.5 工具规格约束

**支持算子类型**：

工具支持算子类型包含：allgather/allreduce/alltoall/reduce/reduce\_scatter/scatter/alltoallv.

**支持数据类型**：

HCCL-VM工具支持数据类型包含：int8/int16/int32/fp16/fp32/uint8/uint16/uint32/bfp16/hif8/fp8e4m3/fp8e5m2/fp8e8m0.

HCCL-VM Runner插件支持数据类如下：

| ReduceOp | DataType                           |
| -------- | ---------------------------------- |
| `ADD`    | `int8/int16/int32/uint8` |
| `MIN`    | `int8/int16/int32/uint8` |
| `MAX`    | `int8/int16/int32/uint8` |

**硬件规格**：

目前本工具仅适配支持Ascend950芯片。单server内最大支持8张卡；超过8张卡，需要跨server运行。默认内置存储适用 1～128 卡的通信域，无需配置数据库。

### 4.6 HCCL-VM插件功能

#### 4.6.1 Runner插件

Runner插件，即模拟执行Hccl业务编排生成的任务，输出output数据。
模拟运行器插件在hccl\_test用例执行过程中默认**关闭**。hccl\_test用例调用算子接口后，通过aclrtSynchronizeStream接口等待算子任务执行完成。模拟运行器工具会等待所有rank都处于等待状态时，开启模拟执行所有rank的task。执行完成后，通知各个rank的用例进行继续执行。
用例执行完成后，用户可以通过执行目录下的"all\_rank\_input\_output.txt"文件，查看每个rank的input buffer和output buffer数据。此功能默认不开启，用户可以在用例执行前通过对应命令开启。

**安装与卸载**：

Runner插件支持通过 `hccl-vm plugin install/uninstall` 命令进行安装和卸载。需在进入hccl-vm工具命令行后、执行算例前安装runner插件，后续的执行才会运行runner。

```bash
# 安装runner插件
(hvm)$> hccl-vm plugin install @runner

# 卸载runner插件
(hvm)$> hccl-vm plugin uninstall @runner
```

#### 4.6.2 Checker插件

Checker插件，即算法分析器插件：功能是将hccl生成的所有task形成一个DAG图，并且通过分析DAG图，判断是否存在内存冲突；通过模拟执行DAG图，判断是否存在语义错误等问题。
算法分析器插件，是由用户自行通过命令启动执行。

Checker 插件当前使用 Checker V3 执行单算子校验。Checker V3 大图校验默认开启：每个 sync window 内的多个算子会合并成一张大图，执行跨算子同步资源冲突校验。单算子 Checker V3 和大图校验可以通过 `manifest.json` 中的配置参数分别控制；老版本 Checker 已移除，`enable_old_checker` 不再是有效配置项。

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

### 4.7 OpenMPI和MPICH环境用例运行差异

运行hccl_test用例前，用户可以通过which命令判断当前环境使用的是哪个mpirun。

#### 4.7.1 环境变量配置差异

环境中一般默认配置的是OpenMPI，若用户使用OpenMPI运行用例，一般不需要额外配置环境变量。
若用户使用MPICH环境运行用例，需要按照如下方式配置环境变量：

```bash
# 配置mpich环境变量
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH
export PATH=/usr/lib/mpich/bin:$PATH
```

#### 4.7.2 mpirun命令参数差异

OpenMPI环境，用户按照如下命令运行hccl_test用例：

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun --allow-run-as-root --oversubscribe -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**参数说明**：

 - --allow-run-as-root：OpenMPI专属参数，允许以root用户运行MPI进程，用于在没有root权限的环境中运行。
 - --oversubscribe：OpenMPI专属参数，放开CPU槽位(slot)限制，允许单节点启动[进程数>CPU逻辑核数]，也就是超额超配跑进程。
 - -np 2：指定进程数为2，与节点数一致。

MPICH环境，用户按照如下命令运行hccl_test用例：

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**参数说明**：

 - -np 2：指定进程数为2，与节点数一致。

### 4.8 结果查看

#### 4.8.1 Runner插件结果

如果在hccl-vm终端执行hccl-vm plugin install @runner安装插件后，算子流程执行完成后会自动触发runner插件的执行，最终结果依赖hccl_test进行校验，用户需关注在重定向的日志文件中是否存在[error]级别日志和最终校验结果：

```bash
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
64                | 1000.00        | 0.00006              | success
```

#### 4.8.2 Checker插件结果

在hccl-vm终端内执行hccl-vm plugin run @checker后，Checker校验流程及结果会打印在终端内，用于需关注是否存在[error]级别日志和最终校验结果：

大图校验按每个sync window执行一次。大图校验失败不会阻断其他校验流程，需通过日志中的`BigGraphCheckerV3 failed`或`Big graph sync-conflict check failed`等`error`级别日志识别。

```bash
[info][PID:144373][TID:144880][main.cc][RunChecker] [RunChecker] op[0] Checker Success.
```

---
### 4.9 运行模式（normal / check-only / estimate）

HCCL-VM 有三种运行模式，任意时刻有且仅有一个生效，默认为 `normal`：

| 模式 | 定位 | 说明 |
| ---- | ---- | ---- |
| `normal` | 普通模式（默认） | 真实独立分配与完整多 rank 通信，结果正确性无损 |
| `check-only` | 仅校验模式 | 大块内存复用 4GB 共享池，大幅压低内存占用；只保证 Checker 校验链路，缓冲区数据内容不保证正确 |
| `estimate` | 估算模式 | 单卡（rank0）构造幻影对端跳过真实通信阻塞点，快速产出任务清单，不真跑数据传输（需先关闭链路心跳与一致性校验，见 [4.9.4](#494-估算模式限制说明)） |

#### 4.9.1 启动时指定模式

`start` 与 `run`（one-shot）子命令均支持 `--check-only` / `--estimate` 标志（二者互斥），不加任何标志即为 `normal`：

```bash
# 以仅校验模式启动
./hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only

# 以估算模式启动
./hccl-vm start ascend950_cluster_32_server_normal.yaml --estimate
```

#### 4.9.2 运行时查询与切换模式

在已启动的 VM 命令行内，可通过 `hccl-vm mode` 即时查询与切换模式，无需重启 VM：

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

切换成功后，终端提示符随模式变化：normal → `(hvm)$>`、check-only → `(chk-only)$>`、estimate → `(estimate)$>`。

**切换行为与约束**：

- **拼写严格**：模式名须严格使用小写连字符拼写，`CHECK-ONLY`、`check_only` 等变体及非法串均被拒绝，当前模式不受影响。
- **幂等**：切换到当前已处的模式为空操作，提示已处于该模式。
- **复用池随模式建/拆**：切入 check-only 时自动创建 4GB 共享复用池（创建前先清理残留），切走时自动释放；normal / estimate 模式下不存在该池。
- **切换前卸载 runner**：切向非 normal 模式前，若 runner 插件驻留则先卸载；卸载失败则中止切换。切向 normal 不触发卸载。
- **切换原子性**：切换过程中任一步骤失败（建池失败、拆池失败等）即报错中止，模式与资源保持切换前原状。已知局限：切换失败时提示符仍可能变更，此时以 `hccl-vm mode show` 的查询结果为准。
- **对新算例生效**：新模式对切换成功后新启动的业务算例生效；切换前已在运行的算例不受影响，仍按其启动时的模式运行。
- **不跨重启保留**：模式不跨 VM 重启保留，重启后回到由启动参数决定的初始模式（未指定即 normal）。

#### 4.9.3 仅校验模式大块内存复用说明

check-only 模式用于大规模集群仅运行 Checker 校验的场景。开启后，单块大内存申请复用同一块共享区 `HcclCommPool`，各 rank 共享、允许互相覆盖，以此大幅降低 `/dev/shm` 占用。此时大块内容不保证正确，仅适用于不读取缓冲区数据的 Checker V3 校验链路，需要数值正确的结果时请勿开启。复用区默认区间为200MB - 4GB，上限 4GB 大小通过命令"--kpoolsize N" 可更改。

小于 200MB 的申请始终走真实分配；单块大于复用区的上限在 check-only 模式下直接报错拒绝。check-only 模式与 Runner 不互斥，但在 check-only 模式开启时安装 Runner，大块复用仍会生效、可能覆盖 Runner 数据，工具会打印告警。

**复用区大小配置说明**
`--kpoolsize N` 支持三种入口，下面以设置复用池大小为5G为例：

```bash
# 1) 启动时指定（分离式），
./bin/hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only --kpoolsize 5G

# 2) 运行期，在 (chk-only)$> 子 shell 里切换/重配
(chk-only)$> hccl-vm mode check-only --kpoolsize 5G

# 3) 运行期，在其他模式子 shell 里切换/重配
(hvm)$>  hccl-vm mode check-only --kpoolsize 5G
```
**设置行为与约束：**
- **拼写严格**：模式名须严格使用--kpoolsize，不允许出现大写或额外连字符，例如"--Kpoolsize", "k_poolsize"等
- **幂等**：已处于 check-only 模式下，也可以改变复用区大小
- **写法**：启动时必须选择 "--check-only" 模式才可以设置大小，运行期内命令也必须在 "mode check-only" 后才能配置"--Kpoolsize"，不能单独使用。
- **单位**：支持纯十进制数字，`--kpoolsize 2147483648`。也支持K，M，G单位，大小写不敏感
- **设置上下限**："--kpoolsize N"，当传入的N小于 4G， 会默认为改为 4G。且大小不能超过`/dev/shm`空间上限

#### 4.9.4 估算模式限制说明

estimate 模式为单卡（rank0）执行、对端为构造出的幻影对端。**使用 estimate 模式前必须满足以下两个前提**：

1. **关闭链路心跳与一致性校验**：estimate 模式链路上没有真实对端参与，链路心跳与一致性校验无法正常完成，会带来误报或异常退出。

   ```bash
   export HCCL_DFS_CONFIG=cluster_heartbeat:off,inconsistent_check:off
   ```

   该环境变量需在业务算例启动前生效，`start` 与 `run`（one-shot）两种方式均适用；基于 README 中的 `start.sh` 示例修改时，注意其默认只关闭了心跳（`cluster_heartbeat:off`），用于 estimate 模式时需按上式补上 `inconsistent_check:off`。

2. **业务算例必须以单进程启动（`-np 1`）**：estimate 模式仅构造 rank0 执行路径，无论集群拓扑配置多大规模，mpirun 命令都必须指定 `-np 1`，其余参数与使用方式不变。

   ```bash
   mpirun --allow-run-as-root --oversubscribe -np 1 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt
   ```

其余限制：

- 仅构造单卡（rank0）执行路径，跳过真实通信阻塞点，用于快速产出任务清单，不真跑数据传输。
- 需要功能正确性校验或真实通信行为时，请改用 `normal` 模式。

#### 4.9.5 估算插件（@estimate）与依赖说明

estimate 模式产出任务清单后，可通过估算插件执行轻量性能仿真（读取 `data/hccl_vm_data.db`，构建任务图并做离散事件仿真）：

```bash
(estimate)$> hccl-vm plugin run @estimate
```

该插件为 Python 实现（安装目录 `plugin/light_performance/`，仅 estimate 模式可用，与 @checker 互斥），其运行依赖**不随安装包分发**，需目标环境预装：

- **python3**：≥3.8（含 sqlite3 标准库）；
- **simpy**：离散事件仿真引擎，执行 `pip3 install simpy` 安装；缺失时插件启动即报 `ModuleNotFoundError: No module named 'simpy'`。

仿真结果输出至 `plugin/light_performance/data/<时间戳>/` 目录：`perf_profiling.json`（Chrome tracing 格式）、单算子 trace 文件及运行日志 `light_performance.log`。

### 4.10 模拟等级与 level1 劫持（北向劫持）

#### 4.10.1 北向劫持机制

HCCL-VM 通过 `LD_PRELOAD` 多层劫持模拟 HCCL 通信算子：Level 0 劫持 HCCL 算子 API；Level 1 劫持 HCCL 调用 HComm 的北向接口，俗称北向劫持；Level 2 劫持 HComm 之下的南向接口（HCCP/Runtime/Driver），俗称南向劫持。默认等级为 2（南向劫持），算子执行仍依赖完整 HComm 运行时；指定 `--level 1` 开启北向劫持后，桩函数库整体替代真实 HComm、以记录任务元数据替代真实通信，HCCL 算子与 HComm 运行时完全解耦。北向劫持已覆盖 AIV、AICPU、CCU、HostDPU 四种展开模式，Checker 校验能力同样适用。

#### 4.10.2 编译期：只编译北向（level1 裁剪构建）

执行 `bash ./build.sh --level1` 只编译北向相关代码（proxy level0/1/2 + 支撑库 + hccl-vm + device 侧 + aiv_kernel），hcomm 源码仓变为可选（提供时连带构建 checker/runner 插件）。产物 `hccl-vm` 仅支持 `--level 1`；`--full` 等全量构建支持 `--level 1/2`。

#### 4.10.3 运行期：运行北向（--level 1）

前置条件：安装目录 `lib/<arch>/` 下存在 `libhccl_proxy_level1.so`（AICPU 模式还需 `lib/aarch64/libhccl_device_proxy_level1.so`），否则启动报错退出。

`start` / `run` / `mock-comm` 三个子命令均支持 `--level 1` 参数；环境变量配置、用例执行与结果查看方式与默认等级 2 完全一致。

***

## 5 附录

### 开源第三方软件依赖

编译本项目时，依赖的第三方开源软件列表如下，离线编译场景可下载并重命名软件包后放置在本项目内的third_party目录下。各存储后端模块的专属依赖及离线放置方式见对应模块目录内的 README。

| 开源软件       | 版本          |下载地址 |
| ------------  | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| CLI11         | 2.2.0         | [cli11-2.2.0.tar.gz](https://raw.gitcode.com/src-openeuler/cli11/blobs/58c912141164a5c0f0139bfa91343fefe151d525/cli11-2.2.0.tar.gz) |
| json          | 3.11.3        | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| spdlog        | 1.11.0        | [spdlog-v1.11.0.tar.gz](https://raw.gitcode.com/src-openeuler/spdlog/blobs/c2dfb1aca26c607393665c836155613ff283de66/v1.11.0.tar.gz) |
| yaml-cpp      | 0.8.0         | [yaml-cpp-0.8.0.tar.gz](https://raw.gitcode.com/src-openeuler/yaml-cpp/blobs/d1ead4fff417073b9cdbf98b8b55eb0efc00b0ba/yaml-cpp-0.8.0.tar.gz) |
| googletest    | 1.14.0        | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) |
| cann-cmake    | master-044    | [cmake-master-044.tar.gz](https://raw.gitcode.com/cann/cmake/archive/refs/heads/master-044.tar.gz) |

### 术语表

| 术语       | 说明                                                      |
| -------- | ------------------------------------------------------- |
| HCCL     | Huawei Collective Communication Library，华为集合通信库         |
| NPU      | Neural Processing Unit，神经网络处理器                          |
| CANN     | Compute Architecture for Neural Networks，华为昇腾 AI 处理器软件栈 |
| MPI      | Message Passing Interface，消息传递接口                        |
| CCU      | Collective Communication Unit，集合通信单元                    |
| Topology | 拓扑，设备连接关系                                               |
| Rank     | 进程编号，在 MPI 中的标识                                         |

---

**文档版本**：v1.3。
**最后更新**：2026-10-31。
