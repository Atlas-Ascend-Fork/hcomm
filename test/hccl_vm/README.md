# HCCL-VM 指导手册

## 1. 概述

HCCL-VM是面向华为昇腾NPU卡的高性能集合通信的虚拟执行环境，该工具旨在无真实昇腾硬件的条件下，实现HCCL集合通信算子的开发和功能验证。

工具提供三种运行模式：`normal`（真实多卡通信，结果正确性无损）、`check-only`（仅校验，大块内存复用共享池，大幅压低内存占用）、`estimate`（估算，跳过真实通信阻塞，快速产出任务清单），支持启动时指定与运行期通过 `hccl-vm mode` 命令即时切换，详见[运行模式](#49-运行模式normal--check-only--estimate)。

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
| `HCCL_VM_SERVER_INDEX` | PyTorch 进程模拟的 server 序号，从 `0` 开始；仅在 `HCCL_VM_ENABLE_PYTORCH=1` 时生效，须与测试脚本的 `--server_index` 一致。 | `export HCCL_VM_SERVER_INDEX=0` |
| `HCCLVM_ENABLE_DUMP_DATA` | 1. 使能性能仿真文件输出落盘。若使能，则在测试用例执行过程中输出3个runner_hcclvm_*_data.bin文件和1个ccu_channel_jetty_config.xml文件供性能仿真使用. <br> 2.使能Runner插件dump input\&output数据。若使能，则在测试用例执行过程中，会将每个算子的input\&output数据dump到all\_rank\_input\_output.txt文件中 | `export HCCLVM_ENABLE_DUMP_DATA=1 使能，export HCCLVM_ENABLE_DUMP_DATA=0 或不配置该环境变量 禁用` |
| `HCCL_VM_QEMU_PATH` | 仅AICPU模式、x86_64宿主生效：外部QEMU程序的非空绝对路径（目标为可执行普通文件，允许指向普通文件的符号链接）。未设置/无效时自动回退 PATH 查找 `qemu-aarch64-static` 并打 WARN（含实际程序与版本）；全机找不到时 AICPU 启动失败。部署步骤见[2.3 QEMU 部署](#23-qemu-部署仅-aicpu-模式x86_64-宿主需要) | `export HCCL_VM_QEMU_PATH=/opt/qemu-10.2.4/bin/qemu-aarch64-static` |
| `HCCL_VM_QEMU_TB_SIZE` | 仅AICPU模式、x86_64宿主生效：QEMU翻译代码缓存（MiB）。未设置默认`16`；`0`表示显式传`-tb-size 0`、由QEMU自身默认策略决定（不是禁用缓存）；仅接受可解析到uint32_t的十进制数字串，非法值告警并回退默认16。指向不支持`-tb-size`的旧版QEMU时不传该选项并告警内存可能过大。 | `export HCCL_VM_QEMU_TB_SIZE=16` |

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
  #    - {src_die_id: 1, src_local_id_range: [0, 2], dst_die_id: 1, dst_local_id_range: [1, 3, 5, 7]}

  - link_mode: "enum"
    device_to_switch_links:
      # 如下示例表示：device0到device15都通过die0的port连接到交换机。结合portGroup可知，device0到device15都通过portGroup[0/4, 0/5, 0/6, 0/7]连接到交换机。
      - {die_id: 0, devices_range: [0, 15]}

```

**字段说明**：

 - **soc_version**: 芯片型号，如 `Ascend950`。
 - **device_num**: 设备总数，根据芯片型号和拓扑类型确定。
 - **device_ports_allocate_map**: 端口分配表，描述每个die的端口配置。1表示device直连端口，2表示device连交换机端口，3表示d2h端口。
 - **port_group**: 描述哪些port合并为一个portGroup，相同portGroup的port对应IP地址相同。没有配置的port，则默认每个port为一个portGroup。
 - **links**: 链路配置表，描述Server/Pod内的所有NPU卡之间，以及NPU与交换机之间的连接关系。
   - **NPU直连关系**: 工具提供了两种方式配置NPU直连关系：
    - **link_mode == "fullmesh"**: 表示所有Device基于一个Die的Port进行全连接。后续有新增典型的连接方式，可以新增link_mode类型，如"ring"。
    - **link_mode == "enum"**: 枚举法。当Server/Pod内的NPU连接方式比较复杂时，可以通过枚举所有的链路关系来描述。
   - **NPU与交换机连接关系**: 用户可以通过枚举法配置NPU与交换机之间的连接关系。
 - **device_to_device_links**: 描述NPU与NPU之间的连接关系。
 - **device_to_switch_links**: 描述NPU与交换机之间的连接关系。

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

`ranktable.json` 描述了本次运行使用的设备和 IP 映射：

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

**字段说明**：

- `server_count`：服务器数量。
- `device_count`：设备总数。
- `server_list`：服务器和设备列表。
  - `device_ip`：设备 IP 地址。
  - `port`：设备端口号。

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
      "enable_dag_graphviz_dump": false     // 是否输出大图 Graphviz DOT 文件（默认关闭）
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

check-only 模式用于大规模集群仅运行 Checker 校验的场景。开启后，单块 200MB 到 4GB 的大内存申请复用同一块 4GB 共享区 `HcclCommPool`，各 rank 共享、允许互相覆盖，以此大幅降低 `/dev/shm` 占用。此时大块内容不保证正确，仅适用于不读取缓冲区数据的 Checker V3 校验链路，需要数值正确的结果时请勿开启。

小于 200MB 的申请始终走真实分配；单块大于 4GB 在 check-only 模式下直接报错拒绝。check-only 模式与 Runner 不互斥，但在 check-only 模式开启时安装 Runner，大块复用仍会生效、可能覆盖 Runner 数据，工具会打印告警。

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

**文档版本**：v1.2。
**最后更新**：2026-10-20。
