# CCU 建模与重构设计文档

**版本：** v0.1（分析 + 方案设计，未落地代码）
**日期：** 2026-10-28
**范围：** `asset/cluster_model/config/device/`（新增）、`src/proxy/level1/ccu`、`src/proxy/level2`、`src/plugin/runner`、`src/plugin/checker`、`src/store`、`src/topo`

---

## 1. 背景与目标

### 1.1 背景

hccl-vm 对 CCU 的建模目前是"三张皮"：

- **hccl-vm/proxy 侧**：对北向 hcomm 暴露一组**假基址**（die0 = `0x123123123`、die1 = `0x456456456`），这些地址不对应任何真实内存；
- **runner 侧**：用进程私有的 C++ 数组（`std::array<uint64_t, N>` / `std::vector<char>`）模拟 Variable/Address/Event/CcuBuffer，执行结果只存在于 runner 进程堆里；
- **checker 侧**：把 proxy 的假基址 + runner 的地址映射常量**复制了一份**，独立做地址↔id 反查。

三者靠"约定同一个魔数"维持一致，没有任何单一数据源，也没有任何真实的内存实体。

### 1.2 目标

| 编号 | 目标 | 验收口径 |
|---|---|---|
| G1 | 建立**单一 device 规格配置源**，hccl-vm / runner / checker / 第三方插件共用 | 代码中不再出现 `0x123123123`、`0x456456456` 及 A6 区域偏移字面量 |
| G2 | CCU 资源**内存实体化**：在 hccl-vm 阶段真实申请 shm，并绑定到规范化的设备虚拟地址 | runner 写入的 Variable/Event/CcuBuffer，可被 hccl-vm、proxy、checker 通过同一 VA 读到 |
| G3 | 地址**per-rank 唯一**且落在 rank 内存窗口内，符合真机编址语义 | `ccuBase(rank,die)` 由配置推导，各 rank 互不相同 |
| G4 | rtMalloc **永不落入 CCU 窗口**，且 user_mem 可用额度为 rank 窗口扣除 CCU 后的**全部空间** | **由 per-segment first-fit 分段映射结构性保证**（不依赖限额常量）；累计超出 1022 GiB 时返回失败并打 ERROR，不静默回绕、不侵入下一 rank 窗口（§4.5.7） |
| G5 | 消除地址↔id 转换与资源规格的**重复实现** | runner/checker/light_performance 三处转换代码收敛为一个库 |
| G6 | 第三方插件（性能仿真等）可直接使用**真实 CCU 基址** | 插件侧无需二次地址转换 |

### 1.3 非目标

- 不改变 CCU 指令语义、不新增指令支持；
- 不改变现有二进制落盘格式（`*_hcclvm_syn_data.bin` 等）的既有字段布局（见 §9）；
- 不重构 checker 的成图/校验算法，只替换其地址解析层。

---

## 2. 现状分析（代码事实）

### 2.1 组件与代码地图

```
                          ┌─────────────────────────────────────────────┐
                          │  hcomm (libhcomm.so / ccu_dataplane.so)     │
                          └───────┬─────────────────────┬───────────────┘
                    北向 HcommCcu*│                     │南向 TLV / rtCCULaunch
                 ┌────────────────▼──────┐   ┌──────────▼───────────────────┐
                 │ src/proxy/level1/ccu/ │   │ src/proxy/level2/            │
                 │  ccu_level1_res.cc    │   │  hccp_ccu_stub.cc            │
                 │  ★真实分配模型(bump)  │   │  hccp_stub.cc:GetCcuMemInfo  │
                 │  DIE_BASE_ADDR[2]     │   │  ★无资源模型，仅返回 caps     │
                 └───────────┬───────────┘   │  0x123123123/0x456456456     │
                             │               └──────────┬───────────────────┘
                             │  DB (Ccu / CcuInstrResTab / CcuChannelTab)
                             ▼                          │
                 ┌───────────────────────┐              ▼
                 │ src/store/            │   ┌─────────────────────────────┐
                 │  store_dump_shm_data  │   │ rtMalloc → AllocVirMem      │
                 │  ★ccu0/1_base_addr    │   │  0x8000000000 + (devPk-1)*1T│
                 │   (占位，无人读)      │   │  → /dev/shm/hvm_*/ (真实)   │
                 └───────────┬───────────┘   └─────────────────────────────┘
                             │
              ┌──────────────┴───────────────┐
              ▼                              ▼
  ┌───────────────────────────┐  ┌──────────────────────────────┐
  │ src/plugin/runner/        │  │ src/plugin/checker/          │
  │  ccu_resource_manager.cc  │  │  ccu_all_rank_param_         │
  │  ★XN/GSA/CKE/MS = 进程堆  │  │  recorder_v3.cc              │
  │  ★A6 地址表 (复制品)      │  │  ★A6 地址表 (复制品)         │
  │  外部不可查               │  │  curXn/curCKE/curHBM = map   │
  └───────────────────────────┘  └──────────────────────────────┘
```

### 2.2 事实一：CCU 基址在 8 处独立硬编码

| # | 位置 | 形式 | 说明 |
|---|---|---|---|
| 1 | `src/proxy/level2/hccp_ccu_stub.cc:73,75` | `baseinfo.resourceAddr` | `SetCcuV1ResourceBasicInfo`，带 `// todo: 后续根据dieId，设置不同的resourceAddr` |
| 2 | `src/proxy/level2/hccp_ccu_stub.cc:102,104` | 同上 | `SetCcuV2ResourceBasicInfo`（A6） |
| 3 | `src/proxy/level2/hccp_stub.cc:2098-2099` | `ccu_die0/1_base_addr` | `GetCcuMemInfo` 区域表基址 |
| 4 | `src/proxy/level1/ccu/ccu_level1_res.cc:37` | `DIE_BASE_ADDR[2]` | level1 用它**推导真实 VA** 返回给 hcomm |
| 5 | `src/store/store_dump_shm_data.cc:497-498` | `ccu0/1_resource_base_addr` | 注释「保留占位（与 checker 侧一致）」，**写入但无人读** |
| 6 | `src/plugin/runner/runner_worker/hccl_task_sequential_execute_v2.cc:39` | `CCU_RESOURCE_BASE_ADDR` | 注释「与 proxy 侧 SetCcuResourceBasicInfo 保持一致」——**靠注释维持一致** |
| 7 | `src/plugin/runner/runner_utils/storage_manager.cc:321-322` | `push_back(...)` | 该路径当前是死代码（`RunVirtualRuntime` 无调用者） |
| 8 | `src/plugin/checker/src/utils/storage_manager.cc:761-762` | `InitCcuInfo` | 灌入 `AllRankParamRecorder::ccu_resource_base_addr_` |

另有 2 处**结构已就绪但未启用**的载体：

- `sim::runtime::Ccu.resource_addr`（`include/runtime_state/sim_models.h:167`）：字段与注释齐全，`src/topo/topo_ascend_cluster_parser.cc:970` 的 `FillCcu()` **只填 device_id/die_id/status/id**，`resource_addr` 恒为 0；唯一读者是 `src/cmd/cmd_table_utils.cc:135` 的表打印。
- `ModelInfoCommInner.ccu0/ccu1_resource_base_addr`（`include/sim_binary_data_type_pub.h:151-152`）：随 syn 文件序列化落盘，runner/checker **都忽略它并重新硬编码**。

> **结论**：数据通路其实早已预留（DB 列 + 落盘字段），缺的是"写入方"和"读取方"。这是本次重构成本最低的切入点。

### 2.3 事实二：存在 4 套互相冲突的 CCU 地址布局

这是比"假地址"更严重的问题——**即使基址统一了，区域偏移仍然对不上**。

| 布局 | 来源 | INS | GSA | XN | CKE | MS |
|---|---|---|---|---|---|---|
| **L2 `GetCcuMemInfo`** | `hccp_stub.cc:2089-2209` | base+0（1MB） | base+0x100000 | base+0x108000 | base+0x110000 | 未上报 |
| **L1 `CcuResLayout` V1** | `ccu_level1_res.cc:44-46` | base+0x100000 | base+0x900000 | base+0x908000 | base+0x910000 | — |
| **L1 `CcuResLayout` V2(A6)** | `ccu_level1_res.cc:47-48` | base+0 | 0（A6 无 GSA） | base+0x100000 | base+0x140000 | — |
| **runner/checker A6 表** | `ccu_resource_manager.cc:378-396`<br>`ccu_all_rank_param_recorder_v3.cc:259-277` | base+0 | — | base+0x100000 | base+0x140000 | base+0x4000000 起，4×CCUA，2MB 间隔 |
| **level1 MS 窗口（第 5 套）** | `ccu_kernel_sim.cc:66` | — | — | — | — | `0x08000000`（128MB），per-die 32MB |

关键缺陷：

1. **`GetCcuMemInfo` 不区分 A5/A6**。`hccp_stub.cc:2089` 无 `soc_version` 分支，对 A6 芯片也返回 V1 风格的 `+0x8000` 步进表；而 A6 的 Variable 实际在 `+0x100000`。hcomm 拿到的偏移与 runner/checker 反查用的偏移**不是同一套**。
2. **`GetCcuMemInfo` 的步进与自身注释矛盾**：注释写"实际预留 32KByte"，Address→Variable 步进 `0x8000`(32KB) 但 `mem_size` 只报 `0x6000`(24KB)，中间留 8KB 空洞，而 PFE 报 `0x100` 却占 `0x8000` 步进。region 的 `size` 与 `stride` 语义混用。
3. **level1 的 CcuBuffer 基址 `0x08000000` 与 runner/checker 的 `0x4000000` 差 2 倍**，且 level1 用"die 均分 64MB 窗口"，runner/checker 用"4×CCUA×8 路交织"。同一 msId 在两侧算出的地址不同。
4. `hccp_ccu_stub.cc:400-410` 还把假基址**移位后**当远端 die 判别符用：`dstVa == 0x456000000 → die1`、`0x123000000 → die0`。该判定现已被 `devType == DEV_TYPE_950` 门控（**仅 A5 路径生效**，且经 `GetRemoteCcuVa()` 封装取值）。这是把魔数当协议字段，一旦基址变更，该判定必须同步重写。

### 2.4 事实三：地址↔id 转换代码存在 4 份实现

| 实现 | 位置 | 规模 | 差异 |
|---|---|---|---|
| runner | `ccu_resource_manager.cc:378-531` | ~154 行 | 返回 `bool` |
| checker | `ccu_all_rank_param_recorder_v3.cc:259-445` | ~187 行 | 返回 `HcclResult` + 错误码 |
| proxy level1 | `ccu_level1_res.cc:504-555`（正向 id→addr） | ~50 行 | 只有正向，无反查 |
| light_performance | `ccu_graph_builder.py:221-294`、`register_model.py` | Python 重写 | 注释「移植自 checker ...a6_v3.cc」 |

四份实现的**常量块、`findBaseAddr` / `findTypeByAddr` / `findBaseSize` 函数名与函数体完全一致**（连 `CcuComponerntType` 的拼写错误都一样：runner `ccu_resource_manager.h:41-50`，checker `ccu_all_rank_param_recorder_v3.h:37-46`）。

同时，`UpdateXnId/UpdateCkeId/UpdateMSId/GetXnId/UpdateAddress(WithoutStride)` 这组 loop 偏移辅助函数也在 runner（`ccu_executor_base.cc:90-140`）与 checker（`ccu_task_transform_instruct_a6_v3.cc:205-277`）各存一份，语义（bit15 = offset mode）相同。

> **实测风险**：`ccu_resource_manager.cc:512` 与 `:522` 两处均留有 `// 需要判断界限  todo`，即 `GetXnIdByAddr` / `GetAddrByXnId` **无上界校验**。checker 侧同函数有 `GRAPH_OUT_OF_RANGE` 校验。同一逻辑两份实现，健壮性不一致。

### 2.5 事实四：资源规格常量重复且互相矛盾

| 资源 | runner 数组上限 | proxy 上报容量 | checker 校验上界 | 全局默认 | 矛盾 |
|---|---|---|---|---|---|
| XN（A6/V2） | `SimCcuV2::CCU_RESOURCE_XN_MAX = 4096`<br>`ccu_microcode_common_v2.h:83` | `xnNum = 16384`<br>`hccp_ccu_stub.cc:114` | `MAX_LOADX_STOREX_ID_NUM = 16383`<br>`ccu_task_transform_instruct_a6_v3.cc:77` | `CCU_RESOURCE_XN_MAX = 3072`<br>`sim_common_defs.h:41` | **4096 vs 16384**：proxy 允许 hcomm 分配到 id 16383，runner 数组只有 4096 项 → 越界写 |
| Loop Engine（V2） | `CCU_RESOURCE_LOOP_ENGINE_MAX = 200`<br>`_v2.h:85` | `loopEngineNum = 512`<br>`hccp_ccu_stub.cc:110` | — | 200 | **200 vs 512** |
| Channel | `MAX_CCU_CHANNEL_NUM = 128`<br>`_v1.h:66`（runner 数组宽度） | V1=128 / **V2=1024**<br>`hccp_ccu_stub.cc:90,119` | `CCU_MAX_CHANNEL_NUM = 16`（profiling） | — | **V2 上报 1024，runner 只有 128 槽** |
| GSA | V2 数组仍有 4096 项 | V2 `gsaNum = 0` | `curGSA` 注释「A6 没有 GSA」 | 3072 | V2 是否保留 GSA 口径不一 |
| CKE | 1024 | 1024 | 1024 | 1024 | 一致 ✓ |
| MS | 1536×4KB = 6MB | 1536 | `msId*4096`，**无容量校验** | 1536 | checker 缺校验 |

另有 `DumpCcuCkeResouceInfo`（`ccu_resource_manager.cc:677`）用 `CCU_RESOURCE_MS_NUM`(1536) 去遍历 1024 项的 `cke_` 数组——一旦启用即越界读。

`GetXnAddr`（`ccu_resource_manager.cc:364-376`）按 `xnId * 64` 个 `uint64_t` 步进（= 512B/XN），与其它地方 8B 步进矛盾，且注释是从 `GetMsAddr` 复制的"一个MS容量为4KB"。对 3072/4096 项数组，`xnId ≥ 48/64` 即越界。调用方 `load_executor.cc:59` / `store_executor.cc:59` 传入的还是 XN 的**值**而非 id。

> 这些矛盾目前"没爆"，只是因为 hcomm 实际分配量远小于上限。**统一配置源必须同时统一容量口径**，否则 G1 只解决了一半问题。

### 2.6 事实五：runner 的 CCU 状态是进程私有堆内存

`CcuResourceV1/V2`（`ccu_resource_v1.h:42-46`）：

```cpp
std::array<uint64_t, SimCcuV1::CCU_RESOURCE_XN_MAX>  xn_[HcclSim::DIE_NUM];
std::array<uint64_t, SimCcuV1::CCU_RESOURCE_GSA_MAX> gsa_[HcclSim::DIE_NUM];
std::array<uint64_t, SimCcuV1::CCU_RESOURCE_CKE_MAX> cke_[HcclSim::DIE_NUM];
std::array<std::vector<char>, HcclSim::DIE_NUM>      ms_;   // 每 die 6MB
```

`runner` 是**独立可执行文件**（`src/plugin/runner/CMakeLists.txt:19`），这些数组位于 runner 进程堆。

当前 runner 对外可见的产物只有三种，且都不可编程查询：

| 产物 | 触发 | 局限 |
|---|---|---|
| JSON trace | `HCCLVM_ENABLE_TRACE=1` | **只在 legacy `SequentialExecutor::Execute()` 中初始化**（`hccl_task_sequential_execute.cc:95-111`）；生产路径 `SequentialExecutorV2::Execute()`（`_v2.cc:182-238`）从不调用 `SetEnabled/StartRun` → 采集器恒为 disabled，`RecordEntry` 全部短路 |
| input/output dump | `HCCLVM_ENABLE_DUMP_DATA` | 只能看 buffer，看不到 Variable/Event/CcuBuffer |
| 日志 | — | 不可编程查询 |

`enableDump_`（`ccu_resource_manager.h:121`）**从未被赋值**，`DumpCcuInstructions`（`:607`）/`DumpChannelId2RmtRank`（`:714`）恒为 no-op；`DumpCcuXn/Gsa/Cke/ChannelResouceInfo` 的唯一调用点（`ccu_simulator.cc:459`）已被注释掉。

> **这就是问题 1.1 与 2.2 的根因**：runner 的执行结果物理上无法被第二个进程看到。`hcomm` 的 CCU 接口用例（例如 launch 后回读 Variable/Event 判断同步是否完成）在模拟器上**原理性不可测**。

对照来看，**片上内存 buffer 路径早已是跨进程可查的**：`rtMalloc` → `AllocPhyMem`（`/dev/shm/hvm_*/`）+ 注册 `PhyMemBlock`/`VirtualMemBlock` → runner 用 `sim::AcquireDevPtrInNoHostProcess`（`store_sim_shm_memory_common.cc:22`）按 DB 范围查询拿到 host 指针。CCU 资源只要套用同一机制即可，**无需发明新 IPC**。

跨进程 shm 根目录也已具备：`HCCL_VM_RESOURCE_ROOT` 由主进程 `start` 阶段 setenv 并全链继承（`include/store/store_sim_resource_root.h:26`，`src/proxy/common/hccl_proxy_round_slot.h:57` 已有先例），默认 `/dev/shm/hvm_<pid>`。

### 2.7 事实六：rank 内存编址已具备 1TiB 窗口雏形

`src/store/store_sim_device_memory_manager.cc:47-62`：

```cpp
void* DeviceMemoryManager::AllocVirMem(uint64_t devPhyId, size_t size)
{
    size_t alignedSize = (size + 7) & ~7;
    constexpr uint64_t deviceBase     = 0x8000000000;    // = 2^39 = 512 GiB
    constexpr uint64_t deviceAddrStep = 0x10000000000;   // = 2^40 = 1 TiB
    virAddr = deviceBase + ((devPhyId - 1) * deviceAddrStep) + m_deviceMemSizes[devPhyId];
    m_deviceMemSizes[devPhyId] += alignedSize;
}
```

调用点 `aclrt_memory_stub.cc:277,293`：`curDeviceId = GetCurrDeviceId()`，而 `GetCurrDeviceId()`（`db_sim_runner_ops.cc:154-171`）返回的是 `Context.device_id`，即 **Device 表主键**（自增，1 起）。`virMem.rank_id = curDeviceId` 同样存的是主键。

**这与需求 3.1 的方案完全吻合**：

| 需求 | 现有代码 | 结论 |
|---|---|---|
| rank0 base = `0x8000000000` | `deviceBase = 0x8000000000`，devPk=1 → `(1-1)*stride` | ✅ 已一致 |
| rank 窗口 = 1T | `deviceAddrStep = 0x10000000000` | ✅ 已一致 |
| rank1 base = `0x18000000000` | devPk=2 → `0x8000000000 + 0x10000000000` | ✅ 已一致 |

checker 侧也已有对应常量（`src/plugin/checker/header/internal/sim_common.h:38-39`：`SIM_MEM_BLOCK_SIZE = 0x10000000000`、`SIM_MEM_MASKER = 0xFFFFFF0000000000`），**但目前无人使用**。

**现存缺口**：
- `AllocVirMem` **无任何上界检查**，`m_deviceMemSizes` 无上限 → 单 rank 分配超过 1TiB 会溢出到下一个 rank 窗口；一旦引入 CCU 窗口（56GiB 处），超过 56GiB 即静默踩坏 CCU。
- `m_deviceMemSizes` 是**进程内 map**（`DeviceMemoryManager` 是进程级单例）。多进程分 rank 场景下各进程按不同 devPk 索引，不冲突；但同进程内跨 rank 分配时水位是共享的，需确认语义。
- 窗口索引是 `devPk - 1`，而 hcomm 的 **rankId 来自 ranktable**，两者不保证相等（仓库已有 `docs/ranktable_rankid_device_conflict_analysis.md` 专门讨论此类冲突）。

---

## 3. 问题清单

| ID | 问题 | 根因 | 影响 | 证据 |
|---|---|---|---|---|
| P1 | runner 与 hccl-vm 割裂，CCU 执行结果不可查 | runner 用进程私有数组建模 CCU（§2.6） | hcomm CCU 接口用例（launch 后回读 Variable/Event）**原理性不可测**；proxy 无法回答"指令执行到哪了" | `ccu_resource_v1.h:42-46` |
| P2 | 假基址对第三方无意义，需二次转换 | 魔数硬编码 8 处，无配置源（§2.2） | 性能仿真等插件必须自带一份转换表；对外交付时地址不可解释 | `hccp_ccu_stub.cc:73` 等 8 处 |
| P3 | 各 rank 的 CCU 基址相同，与真机不符 | `DIE_BASE_ADDR[2]` 只按 die 索引，**不含 rank 维度** | 跨 rank 地址无法区分；远端 notify 目标地址在语义上歧义；`hccp_ccu_stub.cc:400-410`（A5-only）只能用魔数移位当 die 判别符 | `ccu_level1_res.cc:37,549` |
| P4 | 4 套地址布局互相冲突 | 无统一 layout 定义（§2.3） | `GetCcuMemInfo` 对 A6 返回 V1 偏移；level1 CcuBuffer 基址与 runner/checker 差 2 倍；同一 msId 两侧算出不同地址 | `hccp_stub.cc:2089`（无版本分支）、`ccu_kernel_sim.cc:66` vs `ccu_resource_manager.cc:385` |
| P5 | 地址↔id 转换 4 份重复实现 | 无共享库（§2.4） | 改一处漏三处；runner 侧缺边界校验而 checker 侧有 | `ccu_resource_manager.cc:512,522` 的 `todo` |
| P6 | 资源容量口径矛盾 | 无统一规格源（§2.5） | proxy 允许 XN id 到 16383，runner 数组仅 4096 → **潜在越界写**；Channel V2 上报 1024 而 runner 仅 128 槽 | `_v2.h:83` vs `hccp_ccu_stub.cc:114` |
| P7 | `Ccu.resource_addr` 字段空置 | `FillCcu()` 不填 | 已有的 DB 通路浪费，各方只能各自硬编码 | `topo_ascend_cluster_parser.cc:970-975` |
| P8 | syn 文件的 `ccu0/1_resource_base_addr` 写了没人读 | 消费方重新硬编码 | 落盘格式与运行时口径可能不一致而无告警 | `store_dump_shm_data.cc:497`（注释即"占位"） |
| P9 | rtMalloc 无上界保护 | `AllocVirMem` 无检查 | 引入 CCU 窗口后会静默踩坏 CCU 内存 | `store_sim_device_memory_manager.cc:57` |
| P10 | `CcuResourceManager` 全链路无边界检查 | 访问器只 `switch(version)` | `rankId/dieId/xnId/...` 越界即 UB；`GetCkeValue` 还把 `uint64_t` 截断成 `uint16_t` | `ccu_resource_manager.cc:134-237` |
| P11 | runner 的 trace 在生产路径失效 | `SequentialExecutorV2` 未初始化采集器 | CCU 调试能力在最常用路径上不可用 | `_v2.cc:182-238` |
| P12 | `StorageManager` 两个同名不同实现 | checker/runner 各一份 | ~150-200 行样板重复，`FindRootPath`/`IsDirExists`/`GetRankSize` 逐字重复 | 见 §2.1 与探索结论 |

---

## 4. 重构设计

### 4.1 总体架构

引入**单一数据源 + 单一解析库 + 单一内存实体**三层：

```
┌──────────────────────────────────────────────────────────────────────┐
│ ① 配置层  asset/cluster_model/config/device/<soc>.yaml               │
│    地址空间划分 / CCU 区域表 / 资源容量 / MS 布局                    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ 解析（yaml-cpp，带 HAVE_YAML_CPP 保护）
┌───────────────────────────▼──────────────────────────────────────────┐
│ ② 模型层  新增静态库 device_spec （src/device_spec/）                │
│    DeviceSpecRegistry : soc_version → DeviceSpec                     │
│    RankAddressSpace   : rank/devPk  → rankBase / ccuDieBase          │
│    CcuAddressSpace    : (rank,die)  ↔ (region,type,id) 双向转换       │
│    ★ 唯一持有区域偏移/步进球面量的地方                                │
└───────────┬─────────────────────────┬────────────────┬───────────────┘
            │                         │                │
┌───────────▼──────────┐  ┌───────────▼──────┐  ┌──────▼──────────────┐
│ ③ 实体层             │  │ proxy L1/L2      │  │ runner / checker /  │
│ CcuMemoryService     │  │ 返回真实基址     │  │ 第三方插件          │
│  shm + PhyMemBlock   │  │ 写 Ccu.          │  │ 直接按 VA 读写      │
│  + VirtualMemBlock   │  │   resource_addr  │  │ 同一份 CcuAddrSpace │
└──────────────────────┘  └──────────────────┘  └─────────────────────┘
```

设计要点：**② 模型层是纯计算、无状态、无 IO**（除加载配置），因此 runner/checker/proxy/Python 插件都能安全依赖；**③ 实体层只在 hccl-vm 侧创建**，其它进程通过既有的 DB + shm 机制映射，不引入新 IPC。

### 4.2 device 配置文件设计

#### 4.2.1 目录与命名

```
asset/cluster_model/config/device/
├── ascend950.yaml      # A5 / CCU V1
└── ascend960.yaml      # A6 / CCU V2
```

命名与 `cluster/`、`server_or_pod/` 保持同一风格。**参考实现已落地**：`asset/cluster_model/config/device/device_spec_yaml_demo_ascend950.yaml`（A5）与 `device_spec_yaml_demo_ascend960.yaml`（A6）；正式交付时重命名为 `ascend950.yaml` / `ascend960.yaml`。

**匹配规则（分两态）**：

| | 阶段 0（本期） | 目标态（阶段 1-3，见 §4.7） |
|---|---|---|
| spec 选择依据 | `name`（= 文件名）对 `cluster/*.yaml` 的 `soc_version` 做**大小写不敏感最长前缀**匹配 | `server_or_pod/*.yaml` 的 `device: "ascend950"` **显式引用** |
| 规则 | `lower(soc_version).startswith(lower(spec.name))`；多份命中取 name 最长者；最长仍并列 → fail-fast；零命中 → fail-fast 并列出候选 | 直接按名查表，未命中 fail-fast |
| 覆盖的现网值 | `ascend950` ↔ `Ascend950DT_95A1`、`Ascend950`；`ascend960` ↔ `Ascend960DT_961BY`、`Ascend960`；`ascend910b` ↔ `Ascend910B3` | 同左（值由 device yaml 提供） |

选用大小写不敏感前缀而非现有的 `strncmp(soc_version, "Ascend950", 9)`，是为了同时覆盖 SKU 级与 family 级两种粒度、并避开 `ParseDevType`（`topo_cluster_ir.cc:137-152`）只认 `ascend950/ASCEND950/950` 因而对真实值恒返回 `UNKNOWN` 的缺陷（该函数在 S2 删除）。

**收敛后统一删除的 5 处 `strncmp` 硬编码**：`hccp_ccu_stub.cc:134,258,513`、`ccu_level1_res.cc:494-498`、`aclrt_new_stub.cc:47-52`、`hccl_task_sequential_execute_v2.cc:103-105`（改查 `DeviceSpec.ccu_version`）。

> ⚠ device yaml 内虽然写有 `soc.soc_version` 字段，但**阶段 0 加载器完全忽略它**（决策 (i)），匹配只认 `name`。详见 §4.7.4。

#### 4.2.2 Schema（以 A6/Ascend960 为例）

> 权威实现见 `asset/cluster_model/config/device/device_spec_yaml_demo_ascend960.yaml`（含逐字段 hcomm 依据行号与实算校验）。此处为浓缩视图。

```yaml
type: "device_spec"
spec_version: 1                       # schema 版本，供消费方做兼容判定
name: "ascend960"                     # = 文件名 = spec 标识 = 阶段 0 的匹配键
description: "Ascend960(A6) 设备规格：地址空间划分与 CCU 资源布局"

soc:
  soc_version: "Ascend960"            # ⚠ 阶段 0 加载器【完全忽略】，见 §4.7.4 决策 (i)
  ccu_version: "CCU_V2"               # CCU_V1 | CCU_V2 | CCU_V3（V3 预留）
  die_num: 2

# ---------------- 全局地址空间划分（芯片无关，A5/A6 逐字段相同） ----------------
address_space:
  rank_base:        0x8000000000      # rank0 窗口起点（= 512 GiB）
  rank_stride:      0x10000000000     # 每 rank 1 TiB
  rank_window_size: 0x10000000000     # 须 ≤ rank_stride（校验 5）
  # 按 offset 升序、互不重叠、完整铺满 [0, rank_window_size)；**同名可多段**
  partitions:
    - {name: "user_mem",  offset: 0x0,          size: 0xE00000000}   # 段0:   0 ~   56 GiB
    - {name: "ccu",       offset: 0xE00000000,  size: 0x80000000}    #       56 ~   58 GiB（2 die × 1 GiB）
    - {name: "user_mem",  offset: 0xE80000000,  size: 0xF180000000}  # 段1:  58 ~ 1024 GiB
  ccu:
    die_offsets: [0x0, 0x40000000]    # 显式列出、不按步进推导（D10）；长度须 == soc.die_num
    die_size:    0x40000000           # 每 die 1 GiB 地址窗口
  # ★ 不设 alloc_guard / user_mem_limit：可用额度 = Σ user_mem 段 = 0xFF80000000（1022 GiB），
  #   由 partitions 派生。rtMalloc 不踩 CCU 由 per-segment first-fit **结构性保证**（§4.5.7）。

# ---------------- CCU 区域布局（偏移相对 ccuDieBase） --------------------
ccu:
  # 权威资源空间总大小 = backingSize（hcomm ccu_res_specs.h:24 CCU_V1_V2_RESOURCE_SIZE，
  # .cc:765 GetResourceSpaceSize()）。A5/A6 同为 72 MiB。**不用** max(region ∪ ms 末端) 推导。
  resource_space_size: 0x4800000

  # entry_size: 单个资源的字节步进；count: 硬件容量（caps 由此推导）
  # 不设 backed 字段——见 §4.5.2 与 §6.2 O3（已关闭）
  regions:
    - {name: "instruction", type: "INSTR",   offset: 0x0,       size: 0x100000, entry_size: 32, count: 32768}
    - {name: "xn",          type: "XN",      offset: 0x100000,  size: 0x40000,  entry_size: 8,  count: 16384}
    - {name: "cke",         type: "CKE",     offset: 0x140000,  size: 0x8000,   entry_size: 8,  count: 1024}
    - {name: "pfe",         type: "PFE",     offset: 0x148000,  size: 0x8000,   entry_size: 8,  count: 20}
    - {name: "channel",     type: "CHANNEL", offset: 0x150000,  size: 0x20000,  entry_size: 32, count: 1024}
    - {name: "jetty",       type: "JETTY",   offset: 0x170000,  size: 0x8000,   entry_size: 32, count: 128}
    - {name: "mission",     type: "MISSION", offset: 0x178000,  size: 0x8000,   entry_size: 64, count: 16}
    - {name: "loop",        type: "LOOP",    offset: 0x180000,  size: 0x8000,   entry_size: 32, count: 512}
  # A6 无 GSA 寄存器堆 → regions 中不含 GSA 条目，GetCaps().gsa_num 派生为 0。
  # ★ 不设独立 gsa: 键（避免与 regions 冗余，T6 已解决）。
  # ⚠ 但 A6 的 gsaOffset **位域仍在使用**（ccu_microcode_v1.h:630 CcuV2 段 xmId 的
  #   "gsaOffset[52:21]"），语义是循环展开的**地址步长**而非寄存器索引
  #   （ccu_simulator.cc:47-65 → ccu_executor_base.cc:127-136），不可随 count=0 一并清理。

  # MS 布局：A5/A6 **几何完全相同**（4 CCUA × 2 MiB × interleave 8），仅基址不同
  ms:
    ms_size:    4096                  # hcomm ccu_microcode_v1.h:28 CCU_MS_SIZE
    ms_num:     1536
    ccua_num:   4                     # ccu_block_res_strategy.cc:23 CCU_V1_V2_CCUA_NUM
    interleave: 8                     # ccu_microcode_v1.h:29 CCU_MS_INTERLEAVE
    ccua_windows:                     # A6 基址 0x4000000；**A5 基址 0x0**
      - {index: 0, offset: 0x4000000, size: 0x200000}
      - {index: 1, offset: 0x4200000, size: 0x200000}
      - {index: 2, offset: 0x4400000, size: 0x200000}
      - {index: 3, offset: 0x4600000, size: 0x200000}

  reserved_sync_slots: 4              # L1 SYNC_SLOT_NUM 语义（ccu_level1_res.h:250）
  count_xn_num: 4                     # A6 专有：ccu_res_specs.h:59
                                      # CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM（0.5RTT 4 组）；A5 为 0
  # ★ 不设 caps 段：10 项能力口径 100% 可由 regions[*].count / ms.ms_num 推导，
  #   由 DeviceSpec::GetCaps() 返回（§4.2.3）。
  # ★ 不设 WQE / SQ buffer region：其真实位置与偏移未经硬件规格确认，本期不建模（O12）。

compute:                              # T10/T11 占位，本期不消费（O7）
  ai_core_num: 0
  vector_core_num: 0
  ai_cpu_num: 0
bandwidth:
  hbm_gbps: 4000.0
  hbm_utilization: 0.7
  default_port_gbps: 53.125
```

**A5/`ascend950.yaml` 的差异项**（`address_space` 段与 A6 **逐字段相同**）：

| 项 | A5 | A6 | 依据 |
|---|---|---|---|
| `ccu_version` | `CCU_V1` | `CCU_V2` | — |
| **CCUM/CCUA 顺序** | CCUM 在 CCUA **之后**（`CCU_V1_CCUM_OFFSET = 0x800000`） | CCUM 在 CCUA **之前**（`CCU_V2_CCUM_OFFSET = 0x0`） | `ccu_res_specs.h:26-27` |
| `regions` | **9 条**：INSTR@`0x800000`、**GSA@`0x900000`**、XN@`0x908000`、CKE@`0x910000`、PFE/CHANNEL/JETTY/MISSION/LOOP@`0x918000` 起 0x8000 步进 | **8 条**（无 GSA） | `ccu_res_specs.cc:372-373,499-500`；与 proxy L1 `V1_*_OFFSET`（`ccu_level1_res.cc:44-46`）逐个吻合 |
| XN 预留 | `CCU_RESOURCE_XN_V1_RESERVE_SIZE` = 32K = `0x8000` | `..._V2_...` = 256K = `0x40000` | `ccu_res_specs.h:55-56` |
| `ms.ccua_windows` 基址 | **`0x0`**（4 bank 占 `[0, 0x800000)`） | `0x4000000` | 由 CCUM 位置反推 |
| `count_xn_num` | 0 | 4 | `ccu_res_specs.h:59` |
| caps（推导值） | instr 0x8000 / mission 16 / loop **200** / gsa **3072** / xn **3072** / cke 1024 / ms 1536 / channel **128** / jetty 128 / pfe **16** | loop 512 / gsa 0 / xn 16384 / channel 1024 / pfe 20 | 与 `hccp_ccu_stub.cc:69-96`(V1) / `:98-125`(V2) **各 10/10 命中** |
| `backingSize` | **72 MiB**（同 A6） | 72 MiB | `CCU_V1_V2_RESOURCE_SIZE` |

⚠ **A5 残留 TODO（T5）**：PFE/CHANNEL/JETTY/MISSION/LOOP 五项偏移在 hcomm `ccu_res_specs.{h,cc}` 中**查无定义**（全仓 grep `0x148000`/`0x150000`/`0x170000`/`0x178000`/`0x180000` 零命中），yaml 中取"延续 hcomm V1 链条 0x8000 预留惯例"的提案值。影响面有限：A5 指令用 **id 编址**，runner/checker 不做 `AddrToResId` 反查，故这五项当前只影响 `GetCcuMemInfo` 上报值，**不影响 backingSize**（已由 `resource_space_size` 定死）。

> **解析注意（更正早期版本的错误陈述）**：`0x` 前缀整数在两侧解析器下都能正确得到整数值 —— yaml-cpp 的 `convert<T>::decode` 内部 `unsetf(ios::dec)` 走 base-0 语义（`third_party/yaml-cpp/include/yaml-cpp/node/convert.h:180-190`），PyYAML 实现 YAML 1.1 原生支持，**无需 `strtoull` 特判**。真实陷阱是：所有地址/尺寸字段**必须以 `uint64_t` 读取** —— `rank_stride`(0x10000000000)、`user_mem 段1.size`(0xF180000000)、`rank_base`(0x8000000000) 均 ≥ 2^32，用 32 位类型取会因 `num_get` 置 failbit 而抛 `TypedBadConversion`。`device_spec` 库应封装统一的 `ReadU64(node, keyPath)` 并在失败时报出完整 key 路径。

#### 4.2.3 Schema 设计取舍

| 决策 | 选择 | 理由 |
|---|---|---|
| `regions` 用**显式列表**而非固定字段 | 列表 | A5/A6 区域集合不同（A5 有 GSA、A6 无），未来芯片可能增删区域；列表天然可扩展，避免 schema 反复改 |
| `offset` 与 `size` 分离，不用 `stride` 推算 | 显式给全 | 修掉 §2.3 缺陷 2（`GetCcuMemInfo` 中 size 与步进混用）；`entry_size × count ≤ size` 由加载器校验 |
| **不设 `caps` 段**，能力口径由 `DeviceSpec::GetCaps()` 从 `regions[*].count` / `ms.ms_num` 推导 | 在 yaml 中并列写一份 `caps`，再用加载器断言两者相等 | 逐项核验：`instruction/xn/cke/pfe/channel/jetty/mission/loop` 8 项 == 对应 `regions[*].count`，`ms_num` == `ms.ms_num`，`gsa_num` == `regions[GSA].count`（缺失即 0），**10/10 可推导**，且 A5/A6 各与 `hccp_ccu_stub.cc` 的 V1/V2 现值 10/10 命中。"上报口径 vs 数组口径"的解耦已由 region 的 `size`（地址空间槽数）与 `count`（容量）分离表达——例如 A6 XN `size=0x40000`（32768 槽）而 `count=16384`。再留一份 `caps` 只是把同一数字写两遍并额外增加一条防漂移校验，违背 D10 单一权威 |
| **不设独立 `gsa:` 键**，GSA 由 `regions[type=="GSA"]` 表达 | A5 写 region + 独立 `gsa: {count:3072}`，A6 写 `gsa: {count:0}` | 独立键与 `regions[GSA].count` 是同一数字的两份拷贝（A5 上实测冗余）；且 A6 无 GSA 时"独立键 count:0"与"regions 中无该条目"表达同一事实却用两种形态，schema 不一致。统一为「有则一条 region，无则不写」，`GetCaps().gsa_num` 按"条目缺失即 0"派生 |
| **user_mem 用多段 `partitions` 显式铺满**（甲），不用"补集派生"（乙） | 乙：删 `partitions`，`ccu.window_offset` 自描述，user_mem 由库算补集 | 甲的窗口划分在 yaml 里一眼可审计；未来新增保留区（AIV UB 等）就是**手动加一条 partition**，无需引入 `other_reserved` 这类扩展字段（Q1 决策）；甲的唯一弱点（可能没铺满/重叠）由校验 5 的铺满性断言完全兜住 |
| **不设 `alloc_guard` / `user_mem_limit`** | 保留 `user_mem_limit: 0xE00000000` 作硬上界 | 该值是**线性 bump 分配器无法跳过 ccu 空洞时的补偿上界**，per-segment first-fit 落地后失效；保留它会把可用额度从 1022 GiB 人为砍到 56 GiB（浪费 94.3%），且与两段 user_mem 之和自相矛盾。可用额度改由 `Σ user_mem 段` 派生 |
| **不设 `regions[*].backed`** | 布尔开关标记"不 backing"的区域 | 收益恒为 0（regions 仅占 backingSize 2.13%，tmpfs 稀疏按触碰页计费），且与方案 C 的前提 C-1/C-3 互斥。详见 §4.5.2 与 §6.2 O3（已关闭） |
| **不设 WQE / SQ buffer region** | 按 hcomm `CCU_V1_WQE_BASIC_BLOCK_OFFSET` 补一条 `offset 0x1000000, size 32 MiB` | 该偏移由 hcomm 代码反推（`ccu_res_specs.h:32` + `ccu_jetty_ctx_mgr.cc:149,172,222`），**未经硬件规格确认**，写进配置会被误当权威。删除是安全的：`backingSize` 取权威值 72 MiB 已完整覆盖 `[0x1000000, 0x3000000)`，hcomm 写入落在 backing 内、模拟器不解读。跟踪见 O12 |
| 地址用 `0x` 十六进制标量 | 是 | 与真机文档一致，便于人工比对。yaml-cpp（`convert<T>::decode` 内 `unsetf(ios::dec)` 走 base-0）与 PyYAML（YAML 1.1）**都能正确解析**，无需 `strtoull` 特判；但**必须以 `uint64_t` 读取**（见 §4.2.2 末尾解析注意） |
| `compute`/`bandwidth` 本期只留占位 | 是 | 收敛 `light_performance/sim_config.json` 与未落库的 `TaskSchedulerDevice` 是独立收益，不阻塞主线（O7） |
| 加 `spec_version` | 是 | 同仓 `topo.json` 有 `version` 先例（`topo_cluster_parser.cc:218-309` 会读）。device yaml 有 4 类 C++ 消费方 + Python 插件，无版本号则未来加字段无法做兼容判定 |

#### 4.2.4 device yaml 描述事项清单（TODO 跟踪）

**约定**：A5/A6 及未来所有芯片类型的设备规格——地址空间划分、device 基址与 die 偏移、CCU 区域布局、**MS 布局**、资源容量与上报口径——**一律以 device yaml 为唯一权威**（D10）。下表枚举每份 `asset/cluster_model/config/device/ascend*.yaml` 必须描述的事项与当前状态；新增芯片交付时逐项核对，**任何一项缺省即启动失败（加载器 fail-fast），禁止回退到代码内硬编码**。

> **状态说明**：A5 的 T5/T6/T7 原为 TODO，现已依据 hcomm 权威源码（`<hcomm>/src/base_comm/resources/ccu/ccu_device/ccu_res_specs.{h,cc}`、`ccu_block_res_strategy.cc:23`、`ccu_microcode/ccu_microcode_v1.h:28-29`）**核定完毕**，两份参考实现均已落地并通过全部加载器校验。

| ID | yaml 段落 | 必须描述的事项 | ascend960(A6) | ascend950(A5) | 状态 / 来源 |
|---|---|---|---|---|---|
| T1 | `soc` | `soc_version`（单值，**阶段 0 加载器忽略**）/ `ccu_version` / `die_num` | ✅ `"Ascend960"`、`CCU_V2`、2 | ✅ `"Ascend950DT_95A1"`、`CCU_V1`、2 | 阶段 0 匹配靠 `name` 最长前缀（§4.2.1）；`soc_version` 归一待 **O8**，字段语义见 §4.7.4 决策 (i) |
| T2 | `address_space` | `rank_base` / `rank_stride` / `rank_window_size` | ✅ 已定义 | ✅ **逐字段相同** | 与 `AllocVirMem` 现行 `deviceBase/deviceAddrStep` 同源（§2.7）；双文件一致性由 UT 断言 |
| T3 | `address_space.partitions` | 多段 `user_mem` + 单段 `ccu`，**完整铺满**窗口 | ✅ 3 段（56G + 2G + 966G） | ✅ 同一划分 | 校验 5：铺满性 + 唯一 ccu 段 + `userMemTotal = 0xFF80000000`(1022 GiB) > 0。**不再有 `alloc_guard`**（§4.2.3） |
| T4 | `address_space.ccu` | `die_offsets`（per-die 显式偏移）+ `die_size` | ✅ `[0x0, 0x40000000]` / `0x40000000` | ✅ 同 | D10：显式列出，不按步进推导；校验 4 补 die 窗口互不重叠 |
| T5 | `ccu.regions` | 区域表：name/type/offset/size/entry_size/count（**无 `backed`**，见 §4.5.2） | ✅ **8 条**（无 GSA） | ✅ **9 条**（含 GSA@`0x900000`）；⚠ **残留 TODO**：PFE/CHANNEL/JETTY/MISSION/LOOP 五项偏移 hcomm 无定义，取 0x8000 步进提案值 | 已按 `ccu_res_specs.cc:372-373,499-500` 核定；A5 五项残留仅影响 `GetCcuMemInfo` 上报，**不影响 backingSize**（A5 用 id 编址，不做 `AddrToResId` 反查） |
| T6 | ~~`ccu.gsa`~~ → `regions[GSA]` | GSA 容量**由 regions 表达**，不设独立键 | ✅ 无 GSA 条目 → `GetCaps().gsa_num` 派生 0 | ✅ 一条 GSA region（count 3072） | 已解决。⚠ A6 的 `gsaOffset` **位域**仍在 V2 微码中使用（`ccu_microcode_v1.h:630`），语义是地址步长而非寄存器索引，不可随 count=0 清理 |
| T7 | `ccu.ms` | **MS 布局**：`ms_size`/`ms_num`/`ccua_num`/`interleave`/`ccua_windows` | ✅ 基址 `0x4000000` | ✅ 基址 **`0x0`** | **已核定：A5/A6 几何完全相同**（4 CCUA × 2 MiB × interleave 8，`ccu_block_res_strategy.cc:23` + `ccu_microcode_v1.h:28-29`），仅基址不同。据此定论 level1 `CCU_MS_BASE = 0x08000000`（`ccu_kernel_sim.cc:65`）**错误**，随 §4.5.6 删除 |
| T8 | `ccu.reserved_sync_slots` | 跨 rank 同步预留槽数 | ✅ 4 | ✅ 4 | 沿用 L1 `SYNC_SLOT_NUM=4`（`ccu_level1_res.h:250`）。⚠ L1 实为 per-type 三水位（`:293-295` xn/gsa/cke 各 4），本标量语义是"统一作用于三类"；若未来需分别配置须改为映射形式 |
| T9 | ~~`caps`~~ → `DeviceSpec::GetCaps()` | 上报 hcomm 的能力口径**不入 yaml**，由库推导 | ✅ 推导值与 `hccp_ccu_stub.cc:98-125` **10/10 命中** | ✅ 与 `:69-96` **10/10 命中** | 消除"同一数字两处维护"；原校验 6 取消。推导值同时作为 T5/T7 的**交叉验证**：不命中即说明 regions 填错 |
| T10 | `compute` | `ai_core_num` / `vector_core_num` / `ai_cpu_num` | ⏸ 占位（0，不消费） | ⏸ 占位 | O7：第二期落 `TaskSchedulerDevice`/`ComputeDie` 表 |
| T11 | `bandwidth` | `hbm_gbps` / `hbm_utilization` / `default_port_gbps` | ⏸ 占位 | ⏸ 占位（0.0） | O7：第二期收敛 `light_performance/sim_config.json` |
| T12 | 未来芯片 | 新 soc 文件 / `die_num>2` / 新区域增删 / **CCU_V3** | — | — | O4：本期 `die_num` 仅支持 1/2。**V3 扩展位**：`CCU_V3_RESOURCE_SIZE=128 MiB`、CCUA 4→8、`XN_TOTAL_NUM_LIMIT=16K`、`CKE_TOTAL_NUM_LIMIT=2048`、私有/共享双区 + per-block 起始 id（`CcuResSpecInfo`，`ccu_res_specs.h:95-135`）。现有 `regions` 列表 + `ms.ccua_num` 可表达 CCUA 与容量变化，但分块语义需新增 `ccu.blocks`/`ccu.shared` 段（S1 预留、不填不消费）；`ccu_version` 枚举须含 `CCU_V3` |
| **T13** | ~~`ccu.regions[WQE]`~~ | **WQE / SQ buffer 区**：位置与偏移**未经硬件规格确认**，本期**不建模** | ➖ 不建模 | ➖ 不建模 | ⚠ **TODO（O12）**：hcomm 侧确有计算路径 —— `ccu_res_specs.h:32` `CCU_V1_WQE_BASIC_BLOCK_OFFSET = 0x1000000`；`ccu_jetty_ctx_mgr.cc:149` `GetSqeBuffVa` → `:172` `sqeBuffVa = ccuResBaseVa + 0x1000000 + jettyCtxId × CCU_SQ_BUFFER_SIZE(256K)`；`:222` 由 `jettyType == CCUM_CACHED_JETTY` 触发、**非版本门控**（V1/V2 共用）。确认后手动加一条 region（`offset 0x1000000, size 128×256K = 32 MiB, entry_size 0x40000, count 128`）并复核校验 8 |

**out-of-scope 清单（划清 D10「唯一权威」的边界）** —— 下列 CCU 相关的地址/id 空间**不由 device yaml 管辖**，避免"为什么这个常量没进 yaml"的反复质疑：

| 项 | 位置 | 性质 |
|---|---|---|
| `CCU_CHANNEL_NOTIFY_ID_BASE = 0xD000000000000000` | `ccu_kernel_sim.cc:61` | **id 编码**基址（与事件位 id、DB notifyId 空间隔离），非内存地址 |
| notify id 17-bit 空间 | `sim_capacity_limits.h:86-88` | id 空间划分 |
| AICPU jetty 硬件 id 分段（CCU `[1024,1151]` 等） | `sim_capacity_limits.h:133-147` | 硬件 id 分配，与 `CCU_START_TA_JETTY_ID=1024` 对应 |
| AIV comm buffer 布局（65 MiB 固定偏移） | `sim_common_defs.h:144-157` | AIV 侧，非 CCU；未来若需保留地址空间，手动加一条 `partitions`（Q1） |
| host 侧 shm 命名 / 根目录 | `store_sim_resource_root.cc:29-42` | host 资源，非设备地址空间 |
| **WQE / SQ buffer 区** | 见 T13 / O12 | 位置未确认，本期不建模（但已被 72 MiB backing 覆盖） |

### 4.3 配置加载与分发

#### 4.3.1 加载

- **路径解析**：`InstallPath::ResolveToInstallRoot("config/device/<name>.yaml")`（`include/sim_common_api.h:21`、`src/utils/sim_common_api.cc:41-78`），root 由 `HCCL_VM_INSTALL_ROOT` 或向上查找 `config/log_config.yaml` 确定。
- **新增库** `src/device_spec/`（静态库，`HCCL_VM_MODULE="DEVSPEC"`）：
  ```cpp
  namespace sim::devspec {

  enum class CcuResType : uint16_t {   // 取代 runner/checker 各自的 CcuComponerntType
      UNKNOWN = 0, INSTR, CCUM, GSA, XN, CKE, PFE, CHANNEL, JETTY, MISSION, LOOP, WQE, MS
  };                                   // ★ 去掉 _A6 后缀、修正 "Componernt" 拼写、补 INSTR/CCUM/GSA/WQE/MS
  enum class CcuVersion : uint8_t { CCU_V1 = 0, CCU_V2 = 1, CCU_V3 = 2 };

  struct RegionSpec { std::string name; CcuResType type; uint64_t offset, size; uint32_t entrySize, count; };
  struct MsWindow   { uint32_t index; uint64_t offset, size; };
  struct MsSpec     { uint32_t msSize, msNum, ccuaNum, interleave; std::vector<MsWindow> windows; };
  struct Partition  { std::string name; uint64_t offset, size; };
  struct AddrSpace  {
      uint64_t rankBase, rankStride, rankWindowSize;
      std::vector<Partition> partitions;          // 原样保留（甲 schema，多段 user_mem）
      std::vector<uint64_t>  ccuDieOffsets;  uint64_t ccuDieSize;
      // 派生（loader 内算好）
      uint64_t ccuWindowOffset, ccuWindowSize;
      std::vector<Partition> userMemSegs;         // partitions 中 name=="user_mem"，按 offset 升序
      uint64_t userMemTotal;                      // Σ userMemSegs.size = 0xFF80000000
  };
  struct CcuCaps { uint32_t instructionNum, missionNum, loopEngineNum, gsaNum, xnNum,
                              ckeNum, msNum, channelNum, jettyNum, pfeNum; };

  struct DeviceSpec {
      uint32_t    specVersion{0};
      std::string name;             // = 文件名 = 阶段 0 匹配键
      std::string socVersion;       // ★ 阶段 0 加载器不读该 key（决策 (i)），恒为空串
      CcuVersion  ccuVersion{CcuVersion::CCU_V1};
      uint32_t    dieNum{2};
      AddrSpace   addr;
      std::vector<RegionSpec> regions;
      MsSpec      ms;
      uint32_t    reservedSyncSlots{4};
      uint32_t    countXnNum{0};
      uint64_t    resourceSpaceSize{0};

      CcuCaps           GetCaps() const;       // 全部由 regions/ms 推导，yaml 无 caps 段
      uint64_t          BackingSize() const;   // == resourceSpaceSize（§4.5.2）
      const RegionSpec* FindRegion(CcuResType) const;
      bool              HasGsa() const;        // == FindRegion(GSA) != nullptr
      bool              SupportsAddrDecode() const;  // V2/V3 true；V1 为 id 编址，不做反查（§4.4.3）
  };

  class DeviceSpecRegistry {          // 进程级单例，加载后只读
  public:
      static DeviceSpecRegistry& GetInstance();
      bool LoadAll();                                   // 扫 config/device/*.yaml + 9 条校验
      bool LoadFromFile(const std::string& absPath);
      const DeviceSpec* FindByName(const std::string& specName) const;
      const DeviceSpec* FindBySocVersion(const std::string& socVersion) const;  // 阶段 0 唯一匹配入口
      const DeviceSpec* FindByDeviceKey(uint64_t deviceKey) const;   // 查 DB Device.soc_version 后转调
      std::vector<const DeviceSpec*> All() const;
      std::string LastError() const;
  };

  class RankAddressSpace {            // 纯计算，无 IO
  public:
      explicit RankAddressSpace(const AddrSpace& a);
      uint64_t RankBase(uint32_t rankIdx) const;
      uint64_t RankWindowSize() const;                  // 0x10000000000
      // ★ 无 MaxRankSlots()：rank 槽位不设规模上界（原 HCCL_VM_MAX_RANKS/GetMaxRanks 已删除），
      //   rankIdx 合法性由 RankBase() 的 uint64 溢出防护兜底；47-bit 位宽仅诊断（Va47BitRankLimit）。
      // CCU
      uint64_t CcuWindowOffset() const;                 // 0xE00000000
      uint64_t CcuWindowSize() const;                   // 0x80000000
      uint64_t CcuDieBase(uint32_t rankIdx, uint32_t dieId) const;
      bool     IsInCcuWindow(uint64_t va, uint32_t& rankIdx, uint32_t& dieId) const;
      // user_mem（多段；分配用 per-segment first-fit，见 §4.5.7）
      size_t   UserMemSegCount() const;                 // 本期 = 2
      uint64_t UserMemSegOffset(size_t i) const;        // 0x0 / 0xE80000000
      uint64_t UserMemSegSize(size_t i) const;          // 0xE00000000 / 0xF180000000
      uint64_t UserMemTotal() const;                    // 0xFF80000000，仅用于报告/日志
      bool     IsUserMemVa(uint64_t va, uint32_t rankIdx) const;
      // rankIdx ↔ deviceKey ↔ hcclRankId（口径见 §4.4.4）
      static uint32_t RankIdxOfDeviceKey(uint64_t deviceKey);   // == physical_id - 1
  };
  // ★ 不提供 UserMemVaFromLinear(rankIdx, linear)：单一全局线性水位会让分配块
  //   跨越 ccu 空洞（反例见 §4.5.7）。分配必须由调用方按段遍历。

  class CcuAddressSpace {             // 纯计算，替代 4 份重复实现
  public:
      explicit CcuAddressSpace(const DeviceSpec& spec);
      bool AddrToResId(uint64_t dieBase, uint64_t addr, CcuResType& type, uint64_t& id) const;
      bool ResIdToAddr(uint64_t dieBase, CcuResType type, uint64_t id, uint64_t& addr) const;
      bool MsAddrToId (uint64_t dieBase, uint64_t addr, uint32_t& msId) const;
      bool MsIdToAddr (uint64_t dieBase, uint32_t msId, uint64_t& addr) const;
      CcuResType TypeByOffset(uint64_t offset) const;   // 替代 findTypeByAddr
      uint32_t   EntrySize(CcuResType) const;           // 替代 findBaseSize
      uint64_t   RegionOffset(CcuResType) const;        // 替代 findBaseAddr
      // ★ 全部方法带完整边界检查，修掉 ccu_resource_manager.cc:512,522 的
      //   "需要判断界限 todo"（checker 侧同函数有校验、runner 侧没有 → 两份实现健壮性不一致）
  };
  }
  ```
- **加载器校验**（fail-fast，配置错不允许启动；两份参考实现均已实算通过）：
  1. 区间互不重叠，且 `[offset, offset+size)` ⊂ `[0, die_size)`。★ **作用域必须是 `regions[] ∪ ms.ccua_windows[]` 合并检查** —— MS 不在 `regions[]` 里，分开查会漏掉"某 region 与 MS 窗口重叠"；
  2. `entry_size × count ≤ size`（`entry_size == 0` 的不透明块跳过）；
  3. MS **per-window** 覆盖（★ **不用**求和式 `ms_num ≤ Σ窗口槽数` —— 异构窗口下求和式会漏判：窗口容量 `[256,512,512,512]` 时 `Σ=1792 ≥ 1536` 通过，但 `localIdx_max=383 > 255` 使窗口 0 溢出）：
     `localIdx_max = ((ms_num-1)/(ccua_num×interleave))×interleave + (interleave-1)`，要求 `∀i: localIdx_max × ms_size < ccua_windows[i].size`；
     另需 `ccua_num == len(ccua_windows)`、`ccua_windows[i].index == i`（有序无重复）、窗口互不重叠；
  4. `len(ccu.die_offsets) == soc.die_num`；die 窗口互不重叠（排序后 `die_offsets[i] + die_size ≤ die_offsets[i+1]`）；
  5. **partitions 铺满性与 user_mem 派生**：按 offset 升序、互不重叠、**完整铺满** `[0, rank_window_size)`（首段 `offset == 0`；末段 `offset+size == rank_window_size`；相邻段首尾相接）；恰好一个 `name=="ccu"` 段；`name` 只允许 `user_mem`/`ccu`（本期白名单）；`partitions[ccu].offset + max(die_offsets) + die_size ≤ partitions[ccu].offset + partitions[ccu].size`；`userMemTotal = Σ user_mem 段 > 0` 且段数 ≥ 1；`rank_window_size ≤ rank_stride`。
     ★ 不再有 `alloc_guard.user_mem_limit == partitions[user_mem].size`（该字段已删，§4.2.3）；
  6. ~~`caps.*` 与对应 `regions[*].count` 一致~~ —— **已取消**：`caps` 不入 yaml，改由 `DeviceSpec::GetCaps()` 推导（§4.2.3），不存在需要校验的第二份数据（编号保留以免交叉引用失效）；
  7. `rank_base + rank 数 × rank_stride` 的位宽检查 → 超 47-bit 打 **WARN**（见 §6.1 R4）。注：rank 数取**拓扑实际值**（集群 rank 总数），模拟器不设 rank 规模上界（原 `HCCL_VM_MAX_RANKS`/`GetMaxRanks()` 已删除），故本校验须在 `start` 阶段（拓扑已知后）执行，不能在 `LoadAll()` 内做；
  8. `max(regions ∪ ms 末端) ≤ ccu.resource_space_size ≤ ccu.die_size`（实算：A5 `0x940000 ≤ 0x4800000`，留白 62.75 MiB；A6 `0x4800000 ≤ 0x4800000`，取等）；
  9. 扫描 `config/cluster/*.yaml` + `config/server_or_pod/*.yaml` 出现的全部 `soc_version` 值，每个必须**唯一命中**一份 spec（大小写不敏感最长前缀）；反向：每份 spec 的 `name` 至少被一个 `soc_version` 命中（防 spec 腐化）。注：spec 内的 `soc.soc_version` 字段**不参与**此校验（决策 (i)）。
- **依赖**：`device_spec` 仅依赖 `log` + `yaml-cpp`。**checker 与 runner 都已链接 `yaml-cpp::yaml-cpp`**（`src/plugin/checker/CMakeLists.txt:58`、`src/plugin/runner/CMakeLists.txt:62`），无新增三方依赖。库骨架照抄 `src/topo/CMakeLists.txt`（`add_library(... STATIC "")` + `target_sources` + `hcclvm_apply_common_flags` + `HCCL_VM_MODULE`）。

#### 4.3.2 分发（三条通路，覆盖三类消费者）

| 通路 | 载体 | 消费者 | 说明 |
|---|---|---|---|
| **T1 直接读配置** | `config/device/*.yaml` | hccl-vm 主进程、runner、checker、C++ 插件、第三方 | 各方都能拿到 install root，最简单、无格式变更风险。**唯一权威源** |
| **T2 落 DB** | `Ccu.resource_addr`（已存在，`FillCcu()` 补写）+ **新增 4 张 `DeviceSpec*` 表**（详见 §4.8） | proxy stub、runner/checker（DB 可用时优先）、Python 插件、第三方 | `GetCcuFromDeviceByDieId`（`db_sim_runner_common.cc:768`）已在读 `Ccu` 行，**零新增查询代码**；`light_performance/db_reader.py:25,566` 已用 `sqlite3.connect(mode=ro)` 直连 `hccl_sim.db`，加读取即可 |
| **T3 落盘文件** | `ModelInfoCommInner.ccu0/1_resource_base_addr`（已存在） | 离线 checker、第三方工具 | 从"占位"改为"真实值"（`store_dump_shm_data.cc:497-498`），消费方改为读取而非硬编码 |

**规格/布局信息的唯一权威源是 yaml（T1）**：DB 与落盘文件都是**运行时产物**，不作为规格源。加载后必须做**交叉校验**：若 DB 的 `Ccu.resource_addr` 或 `DeviceSpec*` 表内容与本地 yaml 推导值不符 → 打 ERROR 并拒绝启动（防止"配置改了但拓扑没重生成"）。第三方若直接读 DB，必须处理"表为空（未跑过 `start`）"的情况并回退读 yaml。

**消费方的统一回退链**（硬性要求，因 §4.8.7 的 mock-comm 缺口）：
```
查 DB DeviceSpec → 命中则用；未命中（表空/未 start）→ DeviceSpecRegistry::FindBySocVersion() 读 yaml
```

#### 4.3.3 安装与打包

`CMakeLists.txt` 的 `config_hcclvm_install()`（`:154-204`）中，仿照 `server_or_pod`（`:182-192`）新增第四段：

```cmake
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/asset/cluster_model/config/device"
        DESTINATION "config"
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
        DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
```

配套：
- `test/topo/CMakeLists.txt:21-22`、`test/cmd/CMakeLists.txt:19-20` 已整体拷贝 `asset/cluster_model/config/`，**自动覆盖**；
- `test/topo/sqlite/topo_golden_test.cc:399-409` 是显式 symlink，若生成脚本需要读 `device/` 则补 `linkConfig("device")`；
- `asset/generate_cluster_topo.sh:100-113` 增加 `DEVICE_CONFIG_DIR`（仅当生成脚本需要校验 soc↔device 配置存在性时）。

### 4.4 地址空间划分方案

#### 4.4.1 规范化公式

```
rankBase(r)          = RANK_BASE + r × RANK_STRIDE
ccuDieBase(r, d)     = rankBase(r) + CCU_WINDOW_OFFSET + CCU_DIE_OFFSET[d]
resAddr(r, d, t, id) = ccuDieBase(r, d) + regionOffset(t) + id × entrySize(t)
msAddr(r, d, id)     = ccuDieBase(r, d) + ccuaOffset(id) + localIdx(id) × ms_size   // 见 4.4.3 交织
userMemVa(r, seg, off) = rankBase(r) + userMemSegs[seg].offset + off               // 见 §4.5.7 分段分配
```

其中 `RANK_BASE = 0x8000000000`、`RANK_STRIDE = 0x10000000000`、`CCU_WINDOW_OFFSET = 0xE00000000`、`CCU_DIE_OFFSET = [0x0, 0x40000000]`（即 yaml `ccu.die_offsets`，显式列出、不按步进推导，见 D10）；`userMemSegs = [(0x0, 0xE00000000), (0xE80000000, 0xF180000000)]`，由 `partitions` 中 `name=="user_mem"` 的段按 offset 升序派生。

#### 4.4.2 与需求 3.1/3.2 的一致性核算

| 项 | 需求值 | 公式推导 | 校验 |
|---|---|---|---|
| rank0 base | `0x8000000000` | `0x8000000000 + 0` | ✅ |
| rank1 base | `0x18000000000` | `0x8000000000 + 0x10000000000` | ✅ |
| rank0 die0 ccu | `0x8e00000000` | `0x8000000000 + 0xE00000000` | ✅ |
| rank0 die1 ccu | `0x8e40000000` | `0x8e00000000 + 0x40000000` | ✅ |
| rank1 die0 ccu | `0x18e00000000` | `0x18000000000 + 0xE00000000` | ✅ |
| ccu size | 1 GiB | `0x40000000` | ✅ |
| rank size | 1 TiB | `0x10000000000` | ✅ |
| 需求 3.3（rtMalloc 不踩 CCU） | — | **由分段 first-fit 构造性保证**，不依赖任何限额常量 | ✅ 见 §4.5.7 |

**结论：需求方案与现有 `AllocVirMem` 的 `deviceBase`/`deviceAddrStep` 完全同源，属于"把既有隐式约定显式化 + 在 56GiB 处开一个 CCU 窗口"，而非推倒重来。** 这是本方案最重要的可行性依据。

rank 窗口内布局：

```
rank r 窗口 (1 TiB)
0x000000000 ┌─────────────────────────────────────────────┐
            │  user_mem 段0 (rtMalloc 可分配)  56 GiB     │
0xE00000000 ├─────────────────────────────────────────────┤
            │  CCU die0                        1 GiB      │  ← rtMalloc 永不返回
0xE40000000 ├─────────────────────────────────────────────┤
            │  CCU die1                        1 GiB      │  ← rtMalloc 永不返回
0xE80000000 ├─────────────────────────────────────────────┤
            │  user_mem 段1 (rtMalloc 可分配) 966 GiB     │
0x10000000000 └───────────────────────────────────────────┘
              user_mem 合计 = 0xFF80000000 = 1022 GiB（1 TiB − 2 GiB CCU）
```

> ★ **CCU 窗口位置被锁定在 56 GiB**：第三方组件对 CCU 基址有硬性要求，不可挪至窗口末尾换取 user_mem 单段（挪到末尾则无需分段跳跃、分配器更简单，但会改变 `0x8e00000000`/`0x8e40000000` 的对外承诺）。若未来基址变更，需重新适配本段与 §4.5.7 的分配器。

die 内（A6）实际使用范围：

```
ccuDieBase + 0x0000000  instruction   1 MiB
           + 0x0100000  XN          256 KiB   (16384 × 8B = 128 KiB 有效)
           + 0x0140000  CKE          32 KiB   (1024 × 8B = 8 KiB 有效)
           + 0x0148000  PFE          32 KiB
           + 0x0150000  CHANNEL     128 KiB   (1024 × 32B = 32 KiB 有效)
           + 0x0170000  JETTY        32 KiB
           + 0x0178000  MISSION      32 KiB
           + 0x0180000  LOOP         32 KiB
           + 0x4000000  MS CCUA0      2 MiB ┐
           + 0x4200000  MS CCUA1      2 MiB │ 共 8 MiB
           + 0x4400000  MS CCUA2      2 MiB │
           + 0x4600000  MS CCUA3      2 MiB ┘
           + 0x4800000  ── 实际使用上界 ≈ 72 MiB ≪ 1 GiB ✅
```

#### 4.4.3 MS 交织的地址↔id 公式

沿用现有语义（`ccu_resource_manager.cc:398-423`，checker 同），但**参数全部来自配置**：

```
addr → id:
  off       = addr - ccuDieBase
  找 ccuaIndex 使 ccuaWindows[i].offset ≤ off < offset + size
  localIdx  = (off - ccuaWindows[i].offset) / ms_size
  groupIdx  = localIdx / interleave
  posInGrp  = localIdx % interleave
  msId      = groupIdx × (interleave × ccua_num) + ccuaIndex × interleave + posInGrp

id → addr:
  groupIdx  = msId / (interleave × ccua_num)
  ccuaIndex = (msId % (interleave × ccua_num)) / interleave
  posInGrp  = msId % interleave
  localIdx  = groupIdx × interleave + posInGrp
  addr      = ccuDieBase + ccuaWindows[ccuaIndex].offset + localIdx × ms_size
```

覆盖性（per-window，两份参考实现均已遍历 `msId ∈ [0,1536)` 验证互逆）：
`localIdx_max = ((1536-1)/(4×8))×8 + 7 = 383`，每窗口槽数 `0x200000/4096 = 512` → `383 ≤ 511` ✅

> ★ **A5/A6 的 MS 几何完全相同**（已按 hcomm 核定）：`ccua_num = 4`（`ccu_block_res_strategy.cc:23` `CCU_V1_V2_CCUA_NUM`；`ccu_res_specs.h:28` 注"V3 CCUA数量，V1/V2为4个"）、`ms_size = 4096` 与 `interleave = 8`（`ccu_microcode_v1.h:28-29`，**V1 头文件**），每 bank 2 MiB。**仅基址不同**：A5 = `0x0`（CCUM 在 CCUA 之后），A6 = `0x4000000`（CCUM 在 CCUA 之前）。
>
> ⇒ **由此定论**：proxy level1 的 `CCU_MS_BASE = 0x08000000`（128 MiB，`ccu_kernel_sim.cc:65`）**确定错误** —— 它既超出 V1 CCUA 区间 `[0, 0x800000)`，也超出整个 72 MiB 资源空间；形似 `CCU_V1_CCUM_OFFSET = 0x800000` 多写了一个 0。该"第 5 套 MS 地址空间"随 §4.5.6 删除。

> ⚠ **需修正的现存缺陷（A5/A6 皆然）**：`GetMSIdByAddr` 做了 4-bank × 8 路**去交织**，而 `GetMsAddr`（`ccu_resource_manager.cc:350-362`）与 checker 的 `GenSliceFromMs`（`ccu_task_common_v3.cc:1071-1080`）都用**平坦** `msId × 4096` —— 同一 msId 在"地址反查"与"直接编址"两条路径下语义**不一致**。另：`GetMSIdByAddr` 在 runner 侧**无调用者**（只有 checker 用），`addr → id` 与 `id → addr` 也只有单向实现。
>
> S2 统一后需连带修改：checker 的 MS slice offset、semantic check 的 `MsKey` 分桶（`task_graph_semantic_check_v3.cc:47-48` 按 `address/4096` 分桶）。此项与 `docs/ccu_resource_statistics_design.md` §3.4 的遗留确认项（"msId 是 die 内全局编号还是 bank 内编号"）**是同一个问题**，应合并到 S0 一次裁决（见 §4.7.6 的 A9 协同项）。
>
> 语义约定：`GetMsAddr` 返回"MS 池内的 **host 指针**"，`MsIdToAddr` 返回"**设备 VA**"，两者通过实体化后的 region base 关联（§4.5.4 的 `CcuDieView`）。

#### 4.4.4 rank 索引口径（必须显式定义）

现状 `AllocVirMem` 用 `devPhyId - 1` 作窗口索引，`devPhyId` = **Device 表主键**（1 起）。而 hcomm 的 rankId 来自 ranktable，二者不保证相等。

**设计决策**：引入 `RankIndexResolver`，明确定义

```
rankIdx := Device.physical_id - 1     // 与现有 AllocVirMem 完全一致，保证不破坏已分配地址
```

并要求：
- CCU 窗口索引、rtMalloc 窗口索引、`VirtualMemBlock.rank_id` **必须使用同一个 rankIdx**；
- 提供 `rankIdx ↔ hcclRankId ↔ deviceKey` 三向映射查询（数据来自 `Communicator` 表，`store_dump_shm_data.cc` 已在构造 `deviceToRank` 映射）；
- 在 `InitClusterStaticTopoData` 阶段做一次**唯一性断言**：`physical_id` 在全部 Device 中唯一且连续，否则 rank 窗口会重叠。

> 若未来要求"窗口按 hcclRankId 索引"，属破坏性变更（会改变所有已分配 VA），需单独立项。

### 4.5 CCU 内存实体化

这是解决 P1/P2/P3 的核心。

#### 4.5.1 方案：复用既有 shm + DB 注册机制

**不发明新 IPC**。让 CCU 内存走与片上内存 buffer 完全相同的通路：

```
CcuMemoryService::EnsureCcuDieMemory(deviceKey, dieId)
  1. 查 DB: Ccu(device_id, die_id).resource_addr != 0 ? → 已建，走 Acquire
  2. rankIdx = Device.physical_id - 1
     dieBase = RankAddressSpace.CcuDieBase(rankIdx, dieId)
  3. memName = "CCU_R<rankIdx>_D<dieId>"
     hostPtr = MemoryManager::AllocMemByName(memName, backingSize)   // → /dev/shm/$HCCL_VM_RESOURCE_ROOT/
  4. MapDevPtrHostPtr(dieBase, hostPtr)                              // 本进程 VA→host 缓存
  5. 插 PhyMemBlock{device_id, name=memName, size=backingSize, ref_count=1}
     插 VirtualMemBlock{start_ptr = dev_mapped_ptr = dieBase,
                        dev_mapped_end_ptr = dieBase + backingSize,
                        rank_id = rankIdx, device_id = deviceKey,
                        phy_mem_id, owner_pid, src_type = VIR_MEM_TYPE_CCU}
  6. UPDATE Ccu SET resource_addr = dieBase WHERE id = ccuId
```

**收益**（全部是既有能力，零新代码）：
- runner 侧 `sim::AcquireDevPtrInNoHostProcess(ccuVa)`（`store_sim_shm_memory_common.cc:22`）直接按 `VirtualMemBlock.start_ptr` 范围查询命中 → 返回 host 指针；
- proxy 侧 `GetRealPtrByAddr(ccuVa)`（`aclrt_memory_stub.cc:129`）直接按 `dev_mapped_ptr` 范围命中 → **`LoadMicrocodeInstructionStub` 不再需要任何特殊处理**；
- 跨进程共享由 `HCCL_VM_RESOURCE_ROOT` 保证（主进程 setenv，全链继承）。

> **⚠ 硬约束 1：CCU shm 必须走 `MemoryManager::AllocMemByName`，禁止改走 `DeviceMemoryManager::AllocPhyMem/AcquirePhyMem`。**
> 后者内嵌 `CommPoolPolicy`（`store_sim_device_memory_manager.cc:72-90`）：`kBigBlockThreshold = 200MB`、`kPoolSize = 4GB`（`include/store/store_sim_comm_pool_policy.h:19-21`），**check-only 模式下 size ≥ 200MB 会被 `ShouldRedirect` 引流到全 rank 共享的 `HcclCommPool`**。一旦 CCU 内存被引流，所有 rank 的 CCU 状态会 aliasing 到同一块池上 → **跨 rank 互相覆盖**，且只在 check-only 模式（恰是 checker 常用模式）复现，排查成本极高。
> 当前余量充足但**约束必须写死**：A5/A6 的 `backingSize` 同为 **72 MiB**（`resource_space_size`，§4.5.2），距 200MB 阈值有 128 MiB 余量。
> ⚠ 该余量**依赖两个前提**，任一被破坏则风险立即回升：① `backingSize` 取权威值而非 `max(regions ∪ ms)`（D10-补）；② O12 确认 WQE 后若把 `resource_space_size` 上调至 ≥ 200MB。
> （早期版本曾按 level1 的 `CCU_MS_BASE=0x08000000` 估出 A5 = 192 MiB、距阈值仅 8 MiB —— 该 MS 基址已在 §4.4.3 定论为**错误**，估算随之作废，R1b 由「高」降为「低」。）

> **⚠ 硬约束 2：`GetRealPtrByAddr` 带 `device_id` 过滤，只对"本 rank 的 CCU 地址"成立。**
> 其查询条件含 `Eq(&VirtualMemBlock::device_id, GetCurrDeviceKey())`（`aclrt_memory_stub.cc:133-138`），而 CCU 行的 `device_id` 是**属主 rank**。因此 proxy 侧解析**远端 rank** 的 CCU VA 必然返回 nullptr；runner 侧不受影响（`AcquireDevPtrInNoHostProcess` 的查询**无 device 过滤**，`store_sim_shm_memory_common.cc:28-31`，跨 rank 正常）。若后续需要 proxy 跨 rank 读 CCU（如在线对账），须为 `VIR_MEM_TYPE_CCU` 增加"跳过 device 过滤"的分支，或把 CCU 行的 `device_id` 记为通配值。

> **关键前置确认项**：当前 `LoadMicrocodeInstructionStub`（`hccp_ccu_stub.cc:238`）对 `insinfo.resourceAddr` 调用 `GetRealPtrByAddr`，而 `0x123123123` **不可能**命中任何 `VirtualMemBlock`。这说明今天 hcomm 传回的其实是它自己 `rtMalloc` 得到的地址（`resourceAddr` 只被用于计算 Variable/Event VA，指令空间另走 malloc）。落地前需实测确认该假设（见 §6 O1）；若成立，实体化后应改为**由模拟器提供指令空间**，hcomm 直接用 `ccuDieBase + 0`，从而消除一次 malloc + 一次地址转换。

#### 4.5.2 backing 策略：稀疏 vs 分区（关键决策）

| 方案 | 描述 | 内存开销（128 rank） | VA 开销 | 复杂度 | 评价 |
|---|---|---|---|---|---|
| **A. 全窗口 backing** | 每 die 一个 1 GiB shm | `ftruncate` 256 GiB（tmpfs 稀疏，实占≈触碰页） | 256 GiB/进程 | 低 | tmpfs 稀疏文件不占实存，但 `ShmCreate` 的 `refCount`/`size` 头字段与 DB `size` 都变成 1GiB，`ExceedsCeiling`（`CommPoolPolicy`）等按 size 判断的逻辑会被误触发 |
| **B. 区域粒度 backing** | 每个 region 一个 shm（9 个/die） | ≈ 9.6 MiB × 2 × 128 = 2.4 GiB | 同 | 高（9×128×2 个文件 + DB 行） | 文件数爆炸：1024 rank 时 ≈ 18k 个 shm 文件 |
| **C. 单 die 稀疏 backing（推荐）** | 每 die 一个 shm，`size = backingSize = ccu.resource_space_size`（A5/A6 **同为** `0x4800000` = 72 MiB），`mmap` 不加 `MAP_POPULATE` | 实占≈触碰页（典型 < 12 MiB/die）；`ftruncate` 72 MiB × 2 × 128 = 18 GiB 稀疏 | 18 GiB/进程 | **低** | ✅ 地址翻译统一为 `host = regionBase + (va - dieBase)`，任意偏移都可命中，不会因"未 backing 的洞"失败 |

**推荐 C**，并补充：

- **`backingSize` 取 hcomm 权威值 `ccu.resource_space_size`，★ 不用 `max(regions ∪ ms 末端)` 推导**：

  依据 `ccu_res_specs.h:24` `CCU_V1_V2_RESOURCE_SIZE = 72 * 1024 * 1024`、`ccu_res_specs.cc:765` `GetResourceSpaceSize()` 返回该值（V3 为 128 MiB）。实算对比：

  | 芯片 | `max(regions ∪ ms 末端)` | hcomm 权威值 | 差额（未建模但须 backing） |
  |---|---|---|---|
  | A6 | `0x4800000`（72 MiB） | `0x4800000`（72 MiB） | 0 —— **巧合相等** |
  | A5 | `0x940000`（**9.25 MiB**） | `0x4800000`（72 MiB） | **62.75 MiB** |

  A6 上两种推导恰好相同，属**巧合**；A5 上相差 62.75 MiB。若按 max 推导，A5 的 backing 只有 9.25 MiB，而 hcomm 会访问 `0x940000` 以上的空间（例如 SQ buffer 区 `[0x1000000, 0x3000000)`，见 O12）→ 落到 mmap 之外 → `GetRealPtrByAddr` / `AcquireDevPtrInNoHostProcess` 范围查询失败返回 nullptr，或 `CcuDieView` typed accessor 野指针 SIGSEGV。

  加载器校验 8：`max(regions ∪ ms 末端) ≤ resource_space_size ≤ die_size`。

  A6 的 72 MiB 构成：regions 1.53 MiB + 未建模空间 `0x188000..0x4000000` **62.47 MiB** + MS 8 MiB。这些空间与未触碰区域**不占实存**（tmpfs 稀疏、按触碰页计费），只占 `ftruncate` 名义长度与 VA 预留。

- ★ **`regions[]` 与 `resource_space_size` 的语义必须分清（D10 语义补充）**：`regions[]` 描述的是**模拟器需要建模/解码的区域**，`resource_space_size` 才是**资源空间完整大小**；两者之差是**有意留白**（A5 达 62.75 MiB）。**禁止**因为留白大就调小 `resource_space_size` —— 它必须等于 hcomm 的 `CCU_V1_V2_RESOURCE_SIZE`，否则 hcomm 对未建模区域的访问会越界。

- ⚠ **CCU shm 必须走 `MemoryManager::AllocMemByName`，禁止经 `DeviceMemoryManager::AllocPhyMem/AcquirePhyMem`**（§4.5.1 硬约束 1）。后者带 `CommPoolPolicy`，`size ≥ kBigBlockThreshold(200MB)` 且 check-only 模式下会引流到全 rank 共享的 `HcclCommPool` → 跨 rank CCU 状态互相覆盖。A5/A6 的 72 MiB 距该阈值余量充足，但约束必须写死（见 R1b）。

- **不设 `regions[*].backed` 字段**（原 O3 已关闭，详见 §6.2）。方案 C 依赖三个咬合前提：**C-1** 每 die 一块连续 mmap 覆盖 `[0, backingSize)`；**C-2** VA→host 是一次减法 `hostBase + (va - dieBase)`；**C-3** §4.5.4 的 `CcuDieView` typed accessor **无边界检查**。`backed:false` 若参与 `backingSize` 计算即打破 C-1/C-3（野指针 SIGSEGV）；若不参与则是死配置。且其收益恒为 0——regions 仅占 `backingSize` 的 **2.13%**，PFE/JETTY/MISSION/LOOP/CHANNEL 无人写、实存本就是 0 页。

- **懒创建**：仅在首次触碰该 (rank,die) 时创建（触发点：level2 `CCU_U_OP_GET_BASIC_INFO` / `SET_INSTRUCTION`、level1 `HcommCcuInsCreate`、runner `EnsureCcuResource`）。1024 rank 场景下实际只会创建被调度的 rank；
- 提供 `HCCL_VM_CCU_MEM_PREALLOC=1` 支持小拓扑下 `start` 阶段预建（便于提前暴露配置错误）；
- 提供 `HCCL_VM_CCU_MEM_DISABLE=1` 回退到旧的进程私有数组模式（灰度开关，见 §9）；
- **未来若实测出现 VA/页表/配额压力**，正确的收敛方向是 `backing_segments` 段模型（显式声明需映射的连续段，例如 A6 `[{0x0, 0x188000}, {0x4000000, 0x800000}]`，可跳过 62.47 MiB 未建模空间），配合 `CcuDieView` 持 N 段 + accessor 段查找与边界检查（顺带解决 P10）。**注意它省的是 `ftruncate` 名义长度与 VA 预留，不省实存**，故无实测证据前不做。

#### 4.5.3 DB 行的生命周期隔离（必须处理）

`aclrt_memory_stub.cc` 在每次 `aclrtMalloc` 前调用 `PurgeFreedRowsOfDevice<PhyMemBlock/VirtualMemBlock>(deviceId)`，`cmd_base_utils.cc:1540` 还有 `DeleteAll`。CCU 区域行若被误清，runner 会瞬间失去所有 CCU 内存映射。

**对策**：新增 `src_type = VIR_MEM_TYPE_CCU`（现有枚举只有 `VIR_MEM_TYPE_DEV` 等），并：
1. `AcquireDevPtrInNoHostProcess` / `GetRealPtrByAddr` 的 `src_type` 过滤条件放宽为 `IN (DEV, CCU)`；
2. `PurgeFreedRowsOfDevice` 显式排除 `VIR_MEM_TYPE_CCU`；
3. `DeleteAll` 路径按模式区分：拓扑重建时允许清（因为会重新 Ensure），单算子重跑时禁止清；
4. CCU 行的 `ref_count` 语义定义为"映射进程数"，由 `AcquireMemByName`/`ReleaseMemByName` 维护（既有机制）。

#### 4.5.4 runner 侧改造：后端替换而非重写

保持 `CcuResourceManager` 的**对外 API 不变**（35 个 executor 全部无需修改），只把存储后端从数组换成 shm 视图：

```cpp
// 改造前 (ccu_resource_v1.h:42-46)
std::array<uint64_t, XN_MAX> xn_[DIE_NUM];
std::vector<char>            ms_[DIE_NUM];

// 改造后：持有一个指向 shm 的视图
struct CcuDieView {
    uint64_t dieBaseVa{0};     // 设备 VA（ccuDieBase）
    char*    hostBase{nullptr};// shm host 指针
    uint64_t backingSize{0};
    // 按 CcuAddressSpace 的 region 表做 typed accessor
    uint64_t* Xn(uint16_t id)   { return reinterpret_cast<uint64_t*>(hostBase + xnOffset)   + id; }
    uint64_t* Cke(uint16_t id)  { return reinterpret_cast<uint64_t*>(hostBase + ckeOffset)  + id; }
    char*     Ms(uint16_t id)   { return hostBase + MsRegionOffset(id); }
    hcomm::CcuRep::CcuInstr* Instr(uint32_t id) { ... }
};
std::array<CcuDieView, DIE_NUM> views_;
```

配套修复（属于本次必须一并处理的技术债，否则实体化后会立刻暴露）：
- **P10**：所有 accessor 补 `rankId/dieId/id` 边界检查，越界返回失败而非 UB；
- `GetCkeValue` 的 `uint64_t → uint16_t` 截断（`ccu_resource_manager.cc:160-171`）：明确 CKE 位宽语义（真机 CKE 是 16-bit mask，但存储 8B/entry）；
- `GetXnAddr` 的 `xnId * 64` 错误步进（`:364-376`）：改为 `entry_size` 来自配置，并修正 `load_executor.cc:59` / `store_executor.cc:59` 传"值"当"id"的调用；
- `DumpCcuCkeResouceInfo` 用 `MS_NUM`(1536) 遍历 `cke_`(1024) 的越界（`:677`）；
- `Init` 每次覆盖 `ccuResourceBaseAddr_`（`:62`，last-writer-wins）：改为 per-(rank,die) 存储；
- `ccuExecutorMutex`（`ccu_resource_manager.h:123`）声明但从未加锁：实体化后 CCU 内存变成**跨进程共享**，必须明确并发模型（见 §6 R2）。

#### 4.5.5 checker 侧改造

- 删除 `ccu_all_rank_param_recorder_v3.cc:259-445` 的常量块与 4 个转换函数，改为持有 `const CcuAddressSpace*`；
- `StorageManager::InitCcuInfo`（`checker/.../storage_manager.cc:757-763`）改为从配置/DB 填充**按 (rankIdx,die) 索引**的基址表（当前 `std::vector<uint64_t>` 只有 die 维度，是 P3 的直接体现），`AllRankParamRecorder::ccu_resource_base_addr_` 类型同步升级；
- 新增能力（此前做不到）：checker 可选地**直接读 runner 落下的 shm**，做"静态成图校验 + 动态执行结果"对账；
- 补齐缺失校验：`msId < ms_num`、`ckeId < cke_num`、`xnId < xn_num`（当前只有 LoadX/StoreX 的 0..16383 检查，见 §2.5）；`GenSliceFromMs` 的 MS slice 当前**绕过** `CheckSingleSlice` 的边界检查（`task_graph_single_task_check_v3.cc:557-559`），实体化后应改为按配置容量校验。

#### 4.5.6 proxy 侧改造

| 位置 | 改造 |
|---|---|
| `hccp_ccu_stub.cc:69-125` | `SetCcuV1/V2ResourceBasicInfo` → caps 由 `DeviceSpec::GetCaps()` 推导填入（不再读 yaml `caps` 段），`resourceAddr` 取 `Ccu.resource_addr`（真实、per-rank） |
| `hccp_stub.cc:2089-2209` | `GetCcuMemInfo` → **按 ccu_version 分支**，区域表由 `DeviceSpec.regions` 生成（修掉 §2.3 缺陷 1、2）；`mem_va` 用真实 dieBase+offset |
| `hccp_ccu_stub.cc:400-410` | 删除 `0x456000000/0x123000000` 魔数判别，改用 `CcuAddressSpace::IsInCcuWindow(addr) → (rankIdx, dieId)`（这同时修掉 P3 的一个衍生 hack）。注：该判定现仅存在于 A5 分支（`devType == DEV_TYPE_950` 门控），改造范围相应限定于 A5 路径 |
| `ccu_level1_res.cc:33-48` | 删除 `CCU_V1_CAPACITY`/`CCU_V2_CAPACITY`/`DIE_BASE_ADDR`/`V1_*_OFFSET`/`V2_*_OFFSET` 全部字面量，改读 `DeviceSpec`；`GetLayout()` 由配置生成 |
| `ccu_level1_res.cc:386-481` | bump allocator 保持不变（`SYNC_SLOT_NUM=4` 改读 `reserved_sync_slots`）；`baseVa` 计算改用 `CcuAddressSpace::ResIdToAddr` |
| `ccu_kernel_sim.cc:66-72` | **删除第 5 套 MS 地址空间**（`CCU_MS_BASE=0x08000000`、`CCU_MS_WINDOW_CAP=64MB`、die 均分），改用 `CcuAddressSpace::MsIdToAddr`（修掉 §2.3 缺陷 3） |
| `topo_ascend_cluster_parser.cc:970` | `FillCcu()` 补写 `resource_addr = ccuDieBase(rankIdx, dieId)`（修 P7）；同时 `dieId < 2U` 硬编码（`:1129`）改读 `soc.die_num` |
| `store_dump_shm_data.cc:497-498` | 删掉"占位"注释，写真实值（修 P8） |
| **5 处 `strncmp(soc_version, "Ascend9xx", 9)`** | `hccp_ccu_stub.cc:134,258,513`、`ccu_level1_res.cc:494-498`、`aclrt_new_stub.cc:47-52`、`hccl_task_sequential_execute_v2.cc:103-105` → 全部改查 `DeviceSpec.ccu_version`（单点映射，消除 A5/A6 双分支重复） |
| **`ParseDevType`** | `topo_cluster_ir.cc:137-152` **整个删除** —— 它只认精确的 `ascend950`/`ASCEND950`/`950`，对真实值 `Ascend950DT_95A1`/`Ascend950` **恒返回 `UNKNOWN`**，且**无 960 分支**（只有 950/910B/910C/910D），已随 A6 引入而腐化。连带删除调用点 `topo_cluster_parser.cc:710`、`topo_cluster_dumper.cc:105` 的 `devType` 输出，并评估 `SimDevType` 枚举与 `Device.devType` 字段是否一并移除。`DevType` 改由 `DeviceSpec` 派生 |
| `ccu_resource_manager.h:41-50` + `ccu_all_rank_param_recorder_v3.h:37-46` | 两份重复的 `CcuComponerntType` 枚举（成员全带 `_A6` 后缀、且拼写错误 "Componernt"）→ 统一为 `devspec::CcuResType`。连带改 8+ 调用点：runner `sync_xn_wt_executor.cc:63,64,70,72,84`、`trans_mem_executor.cc:130`；checker `ccu_task_transform_instruct_a6_v3.cc:2312-2323,2366,2453-2463,2519` |

#### 4.5.7 rtMalloc 保护（P9 / 需求 3.3）

##### 4.5.7.1 三重保证

| # | 保证 | 机制 | 违反时 |
|---|---|---|---|
| ① | VA 不落入本 rank 的 CCU 2 GiB | **per-segment first-fit**（块必须完整落在单个 user_mem 段内）+ 首尾双端 `IsInCcuWindow` 断言 | 跳到下一段；断言触发＝内部错误 |
| ② | `VA + size` 不超出本 rank 的 1 TiB 上限 | 所有段遍历完仍放不下 → **ERROR 日志 + 返回 nullptr** | ✅ 显式报错，不静默回绕 |
| ③ | 不侵入下一个 rank 的窗口 | 由 ② 推出：`maxVA = rankBase + 1TiB − 1 < 下一 rankBase`（`0x8000000000 + 0x10000000000 = 0x18000000000` 恰为 rank1 起点，无缝无叠） | 同 ② |

★ 需求 3.3 由 ① **构造性保证**，不再依赖任何限额常量 —— 这是删除 `alloc_guard.user_mem_limit` 的前提。

##### 4.5.7.2 ⚠ 为什么不能用"单一全局线性水位"

一个自然的实现是把 user_mem 两段拼成一条线性空间、用单标量水位 + 一次映射：

```
UserMemVaFromLinear(rankIdx, linear):  逐段扣减 linear，命中即 rankBase + seg.offset + linear
```

**它有漏洞 —— 分配块会跨越 CCU 空洞**。精确反例：

```
seg0 = [0, 0xE00000000)                        # 56 GiB
cur  = 0xDFFFFFF00                               # seg0 剩余 0x100 字节
请求 alignedSize = 0x400

线性总量检查: 0xDFFFFFF00 + 0x400 = 0xE000000300 ≤ userMemTotal(0xFF80000000)   → ✅ 通过
VA = rankBase + 0xDFFFFFF00
块 = [rankBase+0xDFFFFFF00, rankBase+0xE000000300)
CCU 窗口 = [rankBase+0xE00000000, rankBase+0xE80000000)
→ 块尾侵入 CCU die0 达 0x300 字节 ❌
```

因此 `RankAddressSpace` **不提供** `UserMemVaFromLinear`，分配必须由调用方**按段遍历**（first-fit）。跳段浪费量 = `seg.size − cur`，而由于请求放不下必有 `seg.size − cur < alignedSize` → **浪费严格小于单次请求大小**（≤ `CommPoolPolicy::kPoolSize` 4 GB，实际典型 KB~MB 级）。按实测峰值 0.405 GiB / seg0 56 GiB，正常路径**永不跳段，浪费为 0**。

##### 4.5.7.3 `AllocVirMem` 改造（`store_sim_device_memory_manager.cc:47-62`）

> ⚠ **实现期发现的既有缺陷（O14）**：`AllocVirMem` 的调用点喂入的是**两个不同的 id 空间** ——
> `aclrtMalloc:277` 传 `GetCurrDeviceId()`（= `Context.device_id`，DB 主键，**1 起**），
> 而 `aclrtReserveMemAddress:972`、`:444`、`:1040` 传 `Device.physical_id`（服务器内 local id，**0 起**，
> 首卡合法为 0）。二者共用同一窗口序列 → `physical_id=0` 的设备与 `PK=1` 的设备会落到**同一 rank 窗口**。
> 这是改造前就存在的问题（`rankIdx = devPhyId - 1` 在 `devPhyId==0` 时 uint64 回绕，产出
> `0xFFFFFF8000000000` 这类垃圾 VA，但因非空而从未被发现）。
> **S4 刻意不修它**（属 S2-2/O14 范围），仅：① `devPhyId == 0` 时走降级路径、保持与改造前
> **逐位一致**的行为以免回归；② 打一次 WARN 指明问题。实测证据：`test_aclrt_memory_stub`
> 的 fixture 正是 `physical_id = 0`（`:199`），G0 若直接拒绝会让 4 个用例从 pass 变 fail。

```cpp
void* DeviceMemoryManager::AllocVirMem(uint64_t devPhyId, size_t size)
{
    // ── G2 size 合法性（★ 修 latent bug B3：size==0 原本返回非空指针并插入
    //    start_end_ptr == start_ptr 的零长度 VirtualMemBlock，范围查询 Le&&Gt 永不命中）──
    const uint64_t alignedSize = (static_cast<uint64_t>(size) + 7ULL) & ~7ULL;
    if (alignedSize == 0) { HCCL_VM_ERROR(...); return nullptr; }

    std::lock_guard<std::mutex> lock(m_virSizeLock);

    // ── G0 devPhyId == 0：既有 id 空间不一致（O14），保持旧行为 + 告警 ──────
    const bool devKeyIsZero = (devPhyId == 0);
    if (devKeyIsZero) { HCCL_VM_WARN("...physical_id(0-based) 与 Device PK(1-based) 共用窗口序列，见 O14"); }

    const devspec::AddrSpace* addrSpace = devKeyIsZero ? nullptr : ResolveAddrSpace(devPhyId);
    if (addrSpace == nullptr || addrSpace->userMemSegs.empty() || addrSpace->rankStride == 0) {
        // ── 降级路径：无 spec 可用，或 devPhyId == 0 ───────────────────────
        // 退回改造前的线性 bump（VA 逐位一致，含 (0-1) 的 uint64 回绕），
        // 但**新增** rank 窗口上界检查（严格优于改造前：改造前完全无上界）。
        // ⚠ 此路径不跳过 CCU 窗口；正常情况下 spec 总能从 config/device/*.yaml 加载
        //   （LoadAll 只读 yaml，不依赖 DB）。
        constexpr uint64_t kFallbackBase = 0x8000000000;      // = yaml address_space.rank_base
        constexpr uint64_t kFallbackStride = 0x10000000000;   // = yaml address_space.rank_stride
        auto& cursor = m_deviceMemCursors[devPhyId];
        if (cursor.empty()) { cursor.assign(1, 0); }
        const uint64_t used = cursor[0];
        if (used > kFallbackStride || alignedSize > kFallbackStride - used) { HCCL_VM_ERROR(...); return nullptr; }
        const uint64_t va = kFallbackBase + (devPhyId - 1) * kFallbackStride + used;
        cursor[0] = used + alignedSize;
        return reinterpret_cast<void*>(va);
    }

    const uint64_t rankIdx = devPhyId - 1;                    // 口径不变（D5）

    // ── G1（已删除）rank 槽位上界 ─────────────────────────────────────────
    // ★ 原实现为 `rankIdx >= MaxRankSlots()`（= GetMaxRanks()，默认 1024）直接失败。
    //   该上界把"集群 rank 数"与"Device 表主键"混为一谈：rankIdx 此处取的是 devPhyId-1，
    //   而 aclrtMalloc 传入的是全局 Device 主键（= (serverKey-1)×卡数 + logicId + 1），
    //   于是只要集群选到第 129 台及以后的服务器（跨 superpod），主键即 >1024，
    //   8 卡用例也会被判为"rank 越界"而分配失败（实测 32 卡跨 superpod 用例 rank16~31 全挂）。
    //   现改为：不设 rank 规模上界，唯一硬约束是 VA 计算不得溢出 uint64
    //   （RankBase() 溢出返回 0 → 分配失败）；47-bit 位宽按 R4 只告警（Va47BitRankLimit）。
    const devspec::RankAddressSpace ras(*addrSpace);

    // ── G2 size 合法性（★ 修 latent bug B3：size==0 会返回非空指针并插入
    //    start_end_ptr == start_ptr 的零长度块，范围查询 Le&&Gt 永不命中）──
    const uint64_t alignedSize = (static_cast<uint64_t>(size) + 7ULL) & ~7ULL;
    if (alignedSize == 0) {
        HCCL_VM_ERROR("alloc vir mem failed: devPhyId:{} invalid size {}", devPhyId, size);
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(m_virSizeLock);
    // ★ 修 latent bug B2：原成员为 std::map<int, uint64_t>，key 是 int 却用
    //   uint64_t devPhyId 索引 → 隐式窄化，devPhyId > INT_MAX 时键值截断/冲突。
    //   改为 per-segment 游标（语义从"窗口内 VA 偏移"变为"段内线性水位"）。
    auto& cursors = m_deviceMemCursors[devPhyId];
    if (cursors.size() != ras->UserMemSegCount()) { cursors.assign(ras->UserMemSegCount(), 0ULL); }

    // ── G3 分段 first-fit：块必须完整落在单个 user_mem 段内（保证 ①）──────
    const uint64_t rankBase  = ras->RankBase(rankIdx);
    const uint64_t windowTop = rankBase + ras->RankWindowSize();
    for (size_t i = 0; i < ras->UserMemSegCount(); ++i) {
        const uint64_t segOff = ras->UserMemSegOffset(i);
        const uint64_t segSize = ras->UserMemSegSize(i);
        const uint64_t cur = cursors[i];
        if (cur > segSize || alignedSize > segSize - cur) { continue; }   // 本段放不下 → 下一段
        const uint64_t va = rankBase + segOff + cur;

        // ── G4 不变量断言（防御式，正常路径不应触发；保证 ①③）───────────
        uint32_t hitRank = 0, hitDie = 0;
        if (va < rankBase || va + alignedSize > windowTop ||
            ras->IsInCcuWindow(va, hitRank, hitDie) ||
            ras->IsInCcuWindow(va + alignedSize - 1, hitRank, hitDie)) {
            HCCL_VM_ERROR("alloc vir mem INTERNAL ERROR: devPhyId:{} va={:#x} size={:#x} "
                          "violates window/ccu invariant", devPhyId, va, alignedSize);
            return nullptr;
        }
        cursors[i] = cur + alignedSize;
        HCCL_VM_INFO("dev:{} rankIdx:{} seg:{} alloc vir mem:{:#x} size:{} (seg used {:#x}/{:#x})",
                     devPhyId, rankIdx, i, va, alignedSize, cursors[i], segSize);
        return reinterpret_cast<void*>(va);
    }

    // ── G5 耗尽 = 保证 ②「VA 超出 device 1 TiB 上限必须报错」─────────────
    uint64_t used = 0, cap = 0;
    for (size_t i = 0; i < ras->UserMemSegCount(); ++i) { used += cursors[i]; cap += ras->UserMemSegSize(i); }
    HCCL_VM_ERROR(
        "alloc vir mem FAILED: devPhyId:{} rankIdx:{} req:{:#x} exceeds device address budget. "
        "rank window [{:#x},{:#x}) = {} GiB; ccu reserved {} GiB; "
        "user_mem total {:#x} ({} GiB), used {:#x} ({} GiB), all {} segment(s) full",
        devPhyId, rankIdx, alignedSize, rankBase, windowTop, ras->RankWindowSize() >> 30,
        ras->CcuWindowSize() >> 30, cap, cap >> 30, used, used >> 30, ras->UserMemSegCount());
    return nullptr;
}
```

配套：删除函数内的 `constexpr uint64_t deviceBase/deviceAddrStep` 局部常量（改由 spec 提供）；周期性输出 per-segment 水位日志（便于容量评估，替代原设想的 `HCCL_VM_STAT_USER_MEM_PEAK`）。

##### 4.5.7.4 零回归论证

| 论据 | 数据 |
|---|---|
| 现有 workload 峰值水位 | **0.405 GiB / rank**（实测 `log.txt`，112 次分配，dev 1/9/10） |
| 距段 0 边界（56 GiB） | **138 倍余量** → 现有全部落在段 0 |
| 段 0 内的 VA 公式 | `rankBase + 0 + cursors[0]`，与现行 `deviceBase + (devPhyId-1)*deviceAddrStep + m_deviceMemSizes[devPhyId]` **逐位相同** |
| 结论 | 已分配地址不变，19 个 golden `servers_info.json` 不受影响 |
| rank 归属提取 | `(va - rank_base) / rank_stride` 仍正确 —— 跳段只改窗口内偏移，不改窗口归属；checker 未使用的 `SIM_MEM_MASKER = 0xFFFFFF0000000000`（`sim_common.h:39`）语义亦不受影响 |
| 反算消费者核查 | 全仓仅 `store_sim_device_memory_manager.cc:57` 一处生产 `rankBase + linear`；**无任何** `(va - rank_base) % rank_stride` 式反算。7 处 VA→host 解析全部**按 DB 块**而非按窗口线性：`AcquireDevPtrInNoHostProcess`（`store_sim_shm_memory_common.cc:37` `offset = devPtr - virMem.start_ptr`）、`GetRealPtrByAddr`（`aclrt_memory_stub.cc:152`）、checker `GetSlice`（`storage_manager.cc:247,252`） |
| 单次分配上界 | `CommPoolPolicy::kPoolSize = 4GB` + `ExceedsCeiling` 仍在，无巨块风险 |

##### 4.5.7.5 `FreeVirMem` 有意不回收 VA（选项 a）

`FreeVirMem`（`:65-68`）保持空实现，但**必须补注释说明理由**，否则后人会当作遗漏：

> 释放路径只置 `is_freed=1` 而**保留 DB 行**，是 `aclrt_memory_stub.cc:474-482` 记载的、有生产故障取证背书的设计（FR1 128p / R3K4 256p：拆卸期 43/48 次远端地址解析未命中全部晚于目标行删除 16.7s 以上，保留行使慢 rank 的迟到解析仍命中）。其不变量是**「标记行与活跃行永不同址共存」**，而该不变量正是靠 bump 分配器**地址永不复用**成立的 —— 7 处范围查询（`store_sim_shm_memory_common.cc:28`、`aclrt_memory_stub.cc:97/139/1232/1321`、`device_arm/common/hccl_device_pub.cc:169/274`）都**故意不带 `is_freed` 过滤**。一旦回收并复用 VA，同一地址会同时被"已释放旧行"与"新活跃行"覆盖，`GetOneByPred` 返回不确定 → 解析到错误的 shm 块。

预算充分性：user_mem 1022 GiB vs 实测峰值 0.405 GiB（**2523 倍余量**）；真正的先失败点是 `CommPoolPolicy::kPoolSize`(4GB) 与 `/dev/shm` 容量。若未来确需回收，须走"隔离期 / epoch 屏障"方案并重新验证上述不变量 —— 四种候选方案的代价表见 **O13**。

#### 4.5.8 第三方插件接入（需求 2.3 / G6）

- **C++ 插件**：链接 `device_spec`，直接 `RankAddressSpace` / `CcuAddressSpace`；
- **Python 插件（light_performance）**：新增 `device_spec.py` 加载同一份 yaml（`pyyaml` 已是既有依赖，`generate_cluster_topo.sh` 就在用），删除 `ccu_graph_builder.py:221-294` / `register_model.py` 中移植自 checker 的常量与解码逻辑；
- 提供 CLI 导出能力（复用 `src/cmd/` 既有子命令框架：继承 `CommandBase` + 文件内放一个 `CommandAutoRegister<T>` 静态实例即自动进 `CommandRegistry::CreateAll()`，**无需改注册表**）：
  ```
  hccl-vm device spec show [<soc>]              # 打印解析后的规格 + 9 条校验结果
  hccl-vm device spec export <soc> -o x.json    # 导出 JSON，供非 C++/Python 第三方消费
  hccl-vm device addr decode 0x18e00140008      # VA → (rank, die, region=CKE, id=1)
  hccl-vm device addr encode --rank 1 --die 0 --type CKE --id 1   # 逆运算 → 0x18e00140008
  ```
  `decode` 对排查"地址从哪来"极有价值；`encode` 在 user_mem 改为分段映射后价值上升（人工算 VA 变难）。两者实现成本极低（复用 `CcuAddressSpace::AddrToResId` / `ResIdToAddr`）。
- **DB 通路**：第三方可直接读 `hccl_sim.db` 的 6 张 `DeviceSpec*` 表（§4.8），或用 `hccl-vm table show DeviceSpec` dump；须处理"表为空（未跑过 `start`）"并回退读 yaml。

### 4.6 统一后的调用链

```
hcomm 初始化
  → RaTlvRequest(MSG_TYPE_CCU_GET_MEM_INFO)
      → GetCcuMemInfo: DeviceSpec(soc).regions × ccuDieBase(rank,die)   ★真实地址
      → CcuMemoryService.EnsureCcuDieMemory(devKey, die)                ★真实申请 shm
      → UPDATE Ccu.resource_addr = ccuDieBase
  → SimRaCustomChannel(CCU_U_OP_GET_BASIC_INFO)
      → baseinfo.resourceAddr = Ccu.resource_addr; caps = DeviceSpec::GetCaps()
  → HcommCcuVariableAlloc / EventAlloc (level1)
      → baseVa = CcuAddressSpace.ResIdToAddr(dieBase, XN|CKE, id)       ★真实 VA
  → CCU_U_OP_SET_INSTRUCTION
      → GetRealPtrByAddr(dieBase + 0) → shm host ptr                    ★直接命中，无需特殊处理
  → rtCCULaunch → DB(task) → runner

runner (独立进程)
  → EnsureCcuResource: Ccu.device_id/die_id → dieBase
      → AcquireDevPtrInNoHostProcess(dieBase) → 同一块 shm              ★与 hccl-vm 共享
      → CcuDieView 直接读写 XN/CKE/MS
  → executor->Run(): 写 XN[id] = v  →  落到 shm

hcomm / proxy / checker (任意进程)
  → 读 XN[id]: AcquireDevPtrInNoHostProcess(dieBase + xnOff + id*8)     ★看到 runner 的写入
```

### 4.7 soc_version 配置收敛（目标态已定，分阶段实施）

#### 4.7.1 现状：5 处出现位置，仅 1 处活跃

| # | 位置 | 现值 | 实际下游 | 状态 |
|---|---|---|---|---|
| 1 | `cluster/*.yaml`<br>`server_list[].soc_version` | `Ascend950DT_95A1`<br>`Ascend960`<br>`Ascend960DT_961BY` | → `servers_info.json` → `topo_cluster_parser.cc:703-710` → `Device.socVersion` → DB `Device.soc_version` → 3 个对外 API（§4.7.6）+ 5 处 `strncmp` | ✅ **唯一活跃** |
| 2 | `server_or_pod/*.yaml`<br>顶层 `soc_version`（16 文件） | `Ascend950` / `Ascend960` | 只喂 `generate_server_topo.sh:300` 一行日志。`:1439` 的 `hardware_type = soc_version.upper()` 兜底**不可达**（16/16 文件都有显式 `hardware_type`）；且 `topo_json` 字典（`:1441-1448`）**根本不含 `soc_version` 键** | ❌ **死配置** |
| 3 | `generate_cluster_topo.sh:629-646`<br>兜底链 | — | `:637` 按 rootinfo 文件名 regex `(ascend\d+)`（`extract_soc_version`，`:584-588`）；`:643` 读 `topo.json` 的 `soc_version` → 承 #2，**该键不存在，死代码**。且 22 个 cluster 文件的每个 `server_list` 条目都显式写了值，兜底链**从未触发** | ❌ **半死** |
| 4 | `topo_meta/*.yaml` 顶层 `soc_version` | 仓库 33 个文件**全都没有** | C++ `ParseTopoMetaYaml`（`sim_common_api.cc:173-215`）只读 `meta`/`topology`，不读 `soc_version`。仅存在于 `ai_skills/.../Ascend_ranktable_file_generation.md:68,83` 的示例与 `generate_ranktable.sh:287`（默认 `'unknown'`） | ❌ **文档幽灵** |
| 5 | `device/*.yaml`（本方案新增） | 单值 | 目标态唯一权威；**阶段 0 加载器完全忽略**（§4.7.4） | 🆕 |

> 用户"搞不清区别"的困惑主要来自 #2/#3/#4 这三处**死配置** —— 它们看起来像权威，实际不产生任何效果。

**已核实的关键事实**：`server_topo ↔ soc_version` 是**严格 1:1**（16 组，无交叉），因此把权威下沉到 server_or_pod / device 层**无信息损失**：

| server_or_pod 文件 | cluster 实际用的 soc_version | 该文件的 hardware_type |
|---|---|---|
| `ascend960_server_topo_clos` / `pod_topo_clos` / `nc16_topo_clos` | `Ascend960` | `960D-clos-{8,72,16}P-topo` ✅ |
| `ascend960_server_topo_normal` / `_normal_hostdpu` / `x2_128p_..._normal` | `Ascend960DT_961BY` | **`950D-normal-...`** ❌ 见 R10 |
| `ascend950_server_topo_*`（10 个） | `Ascend950DT_95A1` | `950D-*` ✅ |

#### 4.7.2 目标态：单一权威 + 三层引用链

```
asset/cluster_model/config/
├── device/                     ★ 唯一 soc_version 权威
│   ├── ascend950.yaml            soc_version: "Ascend950DT_95A1" + 全量 CCU 规格
│   └── ascend960.yaml            soc_version: "Ascend960"（待 O8 归一）
├── server_or_pod/              删 soc_version，加 device 引用
│   └── ascend950_server_topo_normal.yaml
│         device: "ascend950"
├── cluster/                    删 server_list[].soc_version
│   └── ascend950_cluster_4_server_hf.yaml
│         server_list:
│           - {super_pod_id: 0, id_range: [0,3], server_topo: "ascend950_server_topo_hf.yaml"}
└── topo_meta/                  ★ mock-comm 路径需要（§4.8.7）
    └── 112.yaml
          device: "ascend950"   ← 取代 ai_skills 文档里的 soc_version 幽灵字段
```

解析链：`cluster.server_topo` → `server_or_pod/<x>.yaml` → `.device` → `device/<y>.yaml` → `soc_version` + 全量 CCU 规格。

引用解析有既有先例：`find_server_topo()`（`generate_cluster_topo.sh:345-361`）。且 `topo_bank/`、`config/super_pod/` 目录**实际不存在**（已核实），`find_server_topo` 的 `'bank'` 分支是死路 → 引用解析只需处理 yaml 一种情况。

**附带收益**：device spec 挂在 server_or_pod 层后，**异构集群（一个 cluster 混插不同 soc）变得可表达** —— 今天 `soc_version` 虽在 `server_list[]` 条目级，但 22 个 cluster 文件全部同构，该能力从未被使用。

#### 4.7.3 分阶段实施

| 阶段 | 内容 | 改动面 | 何时 |
|---|---|---|---|
| **阶段 0（本期）** | **三方并存**：cluster / server_or_pod / device 各自保留 `soc_version`；device yaml 写出该字段但**加载器完全忽略**（§4.7.4）；`server_or_pod` 不加 `device:` 引用；匹配仍靠 `name` 最长前缀（§4.2.1） | **拓扑配置零改动、19 golden 零变化、对外 API 零变化** | S1 |
| 阶段 1 | 删 #2/#3/#4 三处死配置（16 个 `server_or_pod` 删字段、删 `extract_soc_version` 与 `:637-646` 死兜底链、清 ai_skills 幽灵）；`server_or_pod` 加 `device:` 引用 | 16 yaml + 2 脚本 + 2 文档 | 主方案完成后 |
| 阶段 2 | `cluster` 删 `soc_version`；`generate_cluster_topo.sh` 经 `server_or_pod.device` → device yaml 取值写入 `servers_info.json`（并建议加 `device_spec` 键，O11）；19 golden 重生成 | 22 yaml + 1 脚本 + 19 golden | 阶段 1 后 |
| 阶段 3 | `topo_meta` 加 `device:` 引用打通 mock-comm（33 文件 + `ParseTopoMetaYaml` + `generate_ranktable.sh`）；**O8 归一裁决落地** | 33 yaml + 2 处代码 | 阶段 2 后 |

#### 4.7.4 阶段 0 的字段语义（决策 (i)）

device yaml 的 `soc.soc_version` 字段**存在但加载器完全忽略** —— 不解析、不参与匹配、不参与校验、不落库，仅供人工阅读；`DeviceSpec::socVersion` 成员恒为空串。

⚠ **这必须被显式记录**：它是一个已知的"存在但无效"字段，与 §4.7.1 的 #2（`server_or_pod` 的死 `soc_version`）**同类**。两份参考实现均在 `soc:` 段顶部写了显著注释说明此事并指向 O8；本节是设计侧的对应记录。若不在两处都写明，后来者会误以为它生效（这正是 #2 变成死配置的原因）。

阶段 0 的匹配权威仍是 `cluster/*.yaml` 的 `soc_version` + `name` 最长前缀（§4.2.1）；校验 9 扫描的也是拓扑配置里的值，**不读** spec 内的 `soc_version`。

#### 4.7.5 归一 TODO

见 **O8**：A6 的 `Ascend960` vs `Ascend960DT_961BY` 二选一（连带 R10 的 `hardware_type` 数据错误），主方案完成后处理。

#### 4.7.6 必须保持的对外契约

`Device.soc_version` **不是内部键**，它原样穿透到 3 个对外 API。收敛各阶段都不得破坏，除非 O8 明确裁决并接受 golden 变化：

| API | 位置 | 说明 |
|---|---|---|
| `aclrtGetSocName()` | `aclrt_device_stub.cc:477-497` | **公开 ACL API**，`memcpy(SocName, devRes->soc_version, ...)` 原样返回给应用程序 |
| `halGetSocVersion()` / `halGetChipInfo()` | `ascend_hal_stub.cc:237-247`、`device_arm/level2/proxy/ascend_hal_stub.cc:122-123` | **驱动 HAL API**，`chipInfo->name` 原样返回 |
| `fe::PlatformInfoManager::InitRuntimePlatformInfos(device.soc_version)` | `aclrt_device_stub.cc:90`（注释 `:77`「加载 platform_config ini」） | 传入 **CANN 前端**按 soc 名查平台配置；`HCCL_VM_ENABLE_PYTORCH=1` 路径（`docs/pytorch_guide.md`），**失败仅 WARN → 静默退化** |

另：`Device.soc_version` 是 **CLI 可写列**且被当作文档示例 —— `subcmd_table.cc:47` 的 help 文本就是 `"Column to update (e.g., soc_version)"`，`cmd_table_utils.cc:600-604` 为它写了 TEXT 列特判，`test/cmd/subcmd/subcmd_table_test.cc:123` 用它做用例。

⇒ **DB 列永久保留**，收敛的只是**配置来源**。

#### 4.7.7 与 `ccu_resource_statistics_design.md` 的协同（A9）

该文档枚举的资源分类（channel / instruction / Xn / GSA / CKE / MS / loop / mission）与本方案的 `regions[] + ms` 是**同一套 taxonomy**，但两份文档**零交叉引用**（已 grep 核实：其中不含 `device_spec` / `config/device` / `resource_addr` 任何字样），且它硬引用了 S2 要删除的 `recorder .cc:259-269` 区域常量与 `MAX_LOADX_STOREX_ID_NUM(16383)`。

**要求**：① 统计特性的"容量分母"（利用率 = 使用数 / 容量）直接取 `DeviceSpec::GetCaps()`；② `ccu_resource_statistics_design.md` §3.4 的遗留确认项（"`GetMSIdByAddr` 转出的 msId 是 die 内全局编号还是 bank 内编号"）与本文档 §4.4.3 是**同一个问题**，合并到 S0 一次裁决；③ 该文档补依赖声明。否则重构完成后会出现"地址表已统一、但统计侧又抄了一份常量"的复发（第 5 份实现）。

### 4.8 DeviceSpec 的 DB 落库

#### 4.8.1 落库时机与挂点

```
hccl-vm start <cluster.yaml>
  → subcmd_start.cc:70   GenerateClusterTopo（bash+python 生成 JSON）
  → subcmd_start.cc:94   InitHvmEnv
  → cmd_base_utils.cc:503 InitClusterTopo
  → topo_ascend_cluster_parser.cc:303-334 → InitClusterStaticTopoData(:1010-1167)
       已在批量插 Server / Host / Device / Ccu / Port / EndPoint / EndPointPortMapping
       ★ 新增：在 Device 批之后、Ccu 批之前插 DeviceSpec* 4 张表
```

顺带消掉 `FillDevice`（`:956-968`）里挂着的 `// todo: 从yaml文件中获取soc版本`。

#### 4.8.2 表结构（`Scope::Static` → RunnerDB `hccl_sim.db`）

```cpp
// include/runtime_state/sim_models.h，紧邻 Ccu(:156-162)
typedef struct {
    uint64_t id;                     // PK
    char     name[64];               // spec 标识 = yaml 文件名（"ascend950"），UNIQUE
    uint8_t  ccu_version;            // 0=CCU_V1 1=CCU_V2 2=CCU_V3
    uint8_t  die_num;
    uint32_t spec_version;
    uint64_t rank_base, rank_stride, rank_window_size;
    uint64_t ccu_window_offset, ccu_die_size;
    uint64_t resource_space_size;    // = backingSize
    uint32_t reserved_sync_slots, count_xn_num;
    // ★ 无 user_mem_limit 列（该字段已删，user_mem 由 partitions 派生）
    // ★ 无 soc_version 列（决策 (i)：阶段 0 加载器不读该 key）
    // ★ 无 gsa_count 列（由 DeviceSpecRegion 的 GSA 行表达）
} DeviceSpec;

typedef struct { uint64_t id, spec_id; char name[32]; uint16_t type;
                 uint64_t offset, size; uint32_t entry_size, count; } DeviceSpecRegion;
typedef struct { uint64_t id, spec_id; uint32_t ccua_index; uint64_t offset, size; } DeviceSpecMsWindow;
typedef struct { uint64_t id, spec_id; uint32_t die_id; uint64_t offset; } DeviceSpecDie;
typedef struct { uint64_t id, spec_id; uint64_t offset, size; } DeviceSpecPartition;  // 多段 user_mem + ccu
typedef struct { uint64_t id, device_id, spec_id; } DeviceSpecBinding;               // ★ 见 4.8.4
```

`db_sim_runner_ops.h` 的 `Table(...)` 清单（现 66 张）追加 6 条；DDL 宏已齐备（`Field/Version/PrimaryKey/Unique/Index/Text/Blob/Scope/AutoIncrement`，`:48-57`）。

**零后端成本**：sqlite（`sqlite_engine.cc:984`）与 mariadb（`mariadb_engine.cc:1063,1070`）都从同一元数据 `RenderCreateTable` 渲染 DDL，**不需要写两份建表语句**。

#### 4.8.3 查询接口

`db_sim_runner_common.{h,cc}` 新增（紧邻 `GetCcuFromDeviceByDieId`，`:768`）：

```cpp
HcclVmResult GetDeviceSpecByName(const std::string& name, sim::runtime::DeviceSpec& out);
HcclVmResult GetDeviceSpecByDeviceKey(uint64_t deviceKey, sim::runtime::DeviceSpec& out);  // 经 Binding
HcclVmResult GetDeviceSpecRegions(uint64_t specId, std::vector<DeviceSpecRegion>& out);
HcclVmResult GetDeviceSpecMsWindows(uint64_t specId, std::vector<DeviceSpecMsWindow>& out);
HcclVmResult GetDeviceSpecPartitions(uint64_t specId, std::vector<DeviceSpecPartition>& out);
bool LoadDeviceSpecFromDb(uint64_t deviceKey, sim::devspec::DeviceSpec& out);  // DB → 内存对象
```

#### 4.8.4 `Device` ⟷ `DeviceSpec` 关联选型

| 选项 | 做法 | 评价 |
|---|---|---|
| ~~soc_version join~~ | `Device.soc_version` ⟷ `DeviceSpec.soc_version`（UNIQUE） | ❌ **被决策 (i) 排除** —— 加载器不读 spec 的 `soc_version`，该列永远为空，无法 join |
| **Binding 表（推荐）** | 新增 `DeviceSpecBinding {device_id, spec_id}`，`InitClusterStaticTopoData` 落库时每 device 一行（≤1024 行） | ✅ 零歧义、**不动 `Device` 表**（规避 `Version(3)→(4)` 语义不明）、Python 侧可直接 join |
| FK 列 | `Device` 加 `device_spec_id`，`Version(3)→(4)` | 语义最规范，但依赖 O10 确认 `Table(Version(n))` 变更行为（仓库无 `ALTER TABLE` 迁移机制，只找到 `ALTER ... AUTO_INCREMENT` 用于 reset） |

本期采用 **Binding 表**；若 O10 确认 Version bump 安全，可在后续升级为 FK 列并废弃 Binding。

#### 4.8.5 消费者

| 消费者 | 方式 |
|---|---|
| proxy L1/L2 | 直接查 DB（本就在 DB 上下文，`GetCcuFromDeviceByDieId` 已是先例） |
| runner / checker | 优先查 DB（`LoadDeviceSpecFromDb`），未命中回退 `DeviceSpecRegistry` 读 yaml |
| light_performance | `db_reader.py` 加 `DeviceSpec*` 读取，复用 `:566` 的 `hccl_sim.db` 只读连接（该文件已用 `sqlite3.connect(f"file:{path}?mode=ro", uri=True)`）；删 `ccu_graph_builder.py:221-294` / `register_model.py` 中移植自 checker 的常量 |
| 第三方 | 读 DB（6 张表）**或** `config/device/*.yaml`；另有 `hccl-vm device spec show/export` CLI 与 `hccl-vm table show DeviceSpec` |

**★ 解析缓存（实现期补充，必读）**：`Db::` 薄层按合同**不缓存**（`db_sim_runner_ops.h`：每个接口只转调一次原子接口），每次 `GetById/GetOneByPred` 都是一条 SQL（SQLite 一次 prepare+step、MariaDB 一次往返），且同机全部进程共用 `data/hccl_sim.db`（WAL + busy_timeout 60s）。规格解析又恰好落在 CCU 热点路径上，若逐次直查会把读库次数随卡数/通道数/分配次数放大（实测口径：L1 单次资源分配 18 次、单次事件取址 6 次；checker 启动双 die 128 卡 385 次、1024 卡 3073 次；A5 每条通道 2 次且跨设备通道的结果必被丢弃）。因此约定：

| 层 | 做法 |
|---|---|
| `GetDeviceSpec` / `GetCcuResourceBaseAddr` | 进程级缓存（`Device.soc_version`、`(device,die) → 基址`；只缓存成功结果）。失效入口 `ResetDeviceSpecCache()`，调用点：`InitClusterStaticTopoData` 提交后、`hccl-vm table update Device.soc_version`、UT 重建库 |
| `GetCurrDeviceId` | 本线程记忆（`ctxId → device_id`，`Context.device_id` 建行后不改写）。失效点：`SetCurrCtxTls`、`InvalidateCurrDeviceTls`（绕过前者直接删 Context 行的调用方必须显式调，如 `aclrtDestroyContext`） |
| checker `ResolveAllCcuBaseAddrs` | **不逐 (device,die) 调用**上面的入口，改为 `Device` / `Ccu` 各一次全表扫 + 规格走常驻 `DeviceSpecRegistry`，共 2 次读库（与卡数无关） |
| runner / store / checker 的地址空间对象 | 各自按 rank/device 缓存 `CcuAddressSpace`（`CcuResourceManager::AddrSpace`、`GetCcuAddrSpaceByDeviceId`、`DeviceMemoryManager::ResolveAddrSpace`） |
| A5 `ConfigChannelInfo` | 先判 `locDevice.id == rmtEndPoint.device_id` 再解析 die 基址（跨设备通道不查）；目标态仍是 §4.5.6 的纯计算 `IsInCcuWindow` |

缓存合法性依据：`Device.soc_version`、`Ccu.resource_addr`、`DeviceSpec*` 均为 `Scope::Static`，由 `InitClusterStaticTopoData` 在 start 阶段一次性写入，运行期无改写路径（全仓仅 `ref_cnt` 自增与整行删除）。

#### 4.8.6 权威源约束

**yaml 是权威，DB 是运行时产物。** 两者不一致时以 yaml 为准并 fail-fast（§4.3.2 交叉校验）。第三方若读 DB，必须处理"表为空（未跑过 `start`）"的情况并回退读 yaml。

#### 4.8.7 ⚠ mock-comm 路径缺口

已核实：`hccl-vm mock-comm <topo_meta>` → `InitHvmCommEnv`（`cmd_base_utils.cc:523`）→ `ParseRanktableAndInitCommDomain`（`topo_ascend_cluster_parser.cc:346-370`）→ `ParseRanktable`（`:1265-1355`，只产出 `TopoMeta` = pod/server/device id 映射）→ `InitCommunicationDomain`（`:382-421`）。

**这条路径完全不经 `InitClusterStaticTopoData`** → mock-comm 模式下 `DeviceSpec*` 表无从落库（`Device`/`Ccu` 行的来源也需一并确认，见 O9）。而 mock-comm 是 checker 单算子校验的主入口。

解法与目标态自洽：`topo_meta/*.yaml` 加 `device:` 引用（§4.7.2），阶段 3 落地；同时顺带清掉 #4 的文档幽灵。**阶段 0/本期**：mock-comm 走 §4.3.2 的"DB 未命中 → 回退读 yaml"链路，功能不受阻。

---

## 5. 关键设计决策与权衡

| # | 决策 | 备选 | 选择理由 |
|---|---|---|---|
| D1 | 配置放 `asset/cluster_model/config/device/`，格式 YAML | JSON；放 `config/` 根目录 | 与 `cluster/`、`server_or_pod/`、`topo_meta/` 同构，安装规则/测试拷贝/golden 用例可全部复用；YAML 支持注释，规格文件需要大量注释 |
| D2 | 新增 `device_spec` 静态库，**纯计算无 IO 副作用** | 把逻辑塞进 `src/topo/` 或 `src/utils/` | runner/checker/proxy 都要链接；`topo` 依赖 DB 与 IR，插件不应为此拉入重依赖；纯计算库便于 UT 与 Python 对齐 |
| D3 | CCU 内存复用 `MemoryManager` + `PhyMemBlock`/`VirtualMemBlock` | 新建独立 shm 表 / 用 mmap 匿名共享 | 既有 `AcquireDevPtrInNoHostProcess`、`GetRealPtrByAddr`、跨进程 root、refcount 全部现成；**零新增 IPC 语义**，风险最低 |
| D4 | backing 方案 C（单 die 稀疏，`size = max(regions ∪ MS 窗口)`） | A（全 1GiB）/ B（每 region 一文件）/ `regions[*].backed` 布尔开关 | 见 §4.5.2 表格：A 会被按 size 判断的策略误伤，B 文件数爆炸，C 兼顾翻译简单与实存可控。`backed` 开关被否决——收益恒为 0（regions 仅占 backingSize 2.13%，稀疏文件按触碰页计费），且与方案 C 的前提 C-1（单块连续 mmap）/C-3（accessor 无边界检查）互斥，O3 已关闭 |
| D5 | rankIdx := `Device.physical_id - 1` | 用 hcclRankId | 与现有 `AllocVirMem` 完全一致，**不改变任何已分配 VA**；hcclRankId 方案是破坏性变更，收益不明确 |
| D6 | CCU 基址由配置**推导**，不写死在 yaml 里 | yaml 直接给每 rank 的绝对地址 | rank 数可达 1024，枚举不可行；推导公式只需 4 个常量，且天然保证 per-rank 唯一（解决 P3） |
| D7 | 三通路分发（yaml / DB / 落盘）+ 交叉校验 | 只用 yaml | proxy stub 在 DB 上下文里最自然（`GetCcuFromDeviceByDieId` 已在读该行）；离线 checker 只有落盘文件；交叉校验能第一时间发现"配置改了但拓扑没重生成" |
| D8 | `CcuResourceManager` 对外 API 不变，只换后端 | 重写 executor 层 | 35 个 executor + simulator 是本次重构**不需要动**的部分；API 冻结把改动面收敛到 1 个类 |
| D9 | 保留 `HCCL_VM_CCU_MEM_DISABLE` 灰度开关 | 直接切换 | CCU 内存实体化影响面大（proxy/runner/checker 三方），需要可回退的灰度路径 |
| D10 | **device yaml 为 device 规格的唯一权威**：地址空间划分、device 基址与 die 偏移、CCU 区域布局、MS 布局、容量口径全部由 yaml 推导，L1/L2/runner/checker 四份内部布局表全部删除 | 保留某一侧现网表（如"A5 复刻现行 `GetCcuMemInfo` 表"） | 大方案（device 基址 + die 偏移）与细致划分（CCU 内区域/MS 布局）两级同源、都出自同一份 yaml；A5 现存两套布局的裁决由此从"保哪一侧"变为"全部让位于 yaml"（初值按 hcomm `ccu_res_specs` 口径核定，已完成，见 §4.2.4/R5） |
| **D10-补** | ⚠ **D10 的语义边界**：`ccu.regions[]` 描述的是**模拟器需要建模/解码的区域**，`ccu.resource_space_size` 才是**资源空间完整大小**；两者之差是**有意留白**（A5 达 62.75 MiB，A6 为 0）。**禁止**因为留白大就调小 `resource_space_size` | 把 `resource_space_size` 也改成 `max(regions ∪ ms)` 推导 | 它必须等于 hcomm 的 `CCU_V1_V2_RESOURCE_SIZE`（`ccu_res_specs.h:24`），否则 hcomm 对未建模区域（如 SQ buffer `[0x1000000,0x3000000)`，O12）的访问会落到 mmap 之外。校验 8 只保证 `max ≤ resource_space_size`，不保证相等 |
| D11 | **soc_version 的单一权威在 device yaml**，引用链 `cluster → server_or_pod → device`（三层同构引用） | 保留 cluster 为权威（device yaml 不声明 soc_version）；或 device yaml 列 soc_version 列表 | 消除用户"3 处 soc_version 分不清区别"的困惑；`server_topo ↔ soc_version` 已核实为**严格 1:1**（16 组无交叉）→ 下沉无信息损失；device spec 挂在 server_or_pod 层还使**异构集群可表达**；引用解析有 `find_server_topo()`（`generate_cluster_topo.sh:345-361`）先例，且 `topo_bank`/`super_pod` 目录实际不存在，只需处理 yaml 一种情况 |
| D12 | **分阶段实施**：阶段 0（本期）三方并存，device yaml 的 `soc_version` 字段存在但**加载器完全忽略**；归一留待主方案完成后（阶段 1-3） | 一次性收敛（本期即改 22 cluster + 16 server_or_pod + 33 topo_meta + 19 golden） | 一次性收敛牵动 3 个对外 API（§4.7.6）、19 个 golden、2 个生成脚本、ai_skills 文档，且依赖 O8 的 A6 归一裁决（本身又依赖 CANN `InitRuntimePlatformInfos` 实测）——不应与 CCU 主方案耦合。阶段 0 做到**拓扑配置零改动、golden 零变化、对外 API 零变化** |
| D13 | `Device` ⟷ `DeviceSpec` 用 **Binding 表**关联 | `Device.soc_version` join（被决策 (i) 排除，该列恒空）；`Device` 加 FK 列（需 `Version(3)→(4)`） | 零歧义、不动 `Device` 表、规避 `Table(Version(n))` 变更语义不明（仓库无 `ALTER TABLE` 迁移机制）；Python 侧可直接 join。若 O10 确认 Version bump 安全，后续可升级为 FK 列 |
| D14 | **user_mem 用多段 `partitions` 显式铺满（甲）**，`alloc_guard` 删除，可用额度 = `Σ user_mem 段` = 1022 GiB；分配用 **per-segment first-fit** | 乙：删 `partitions`、`ccu.window_offset` 自描述、user_mem 由库算补集；或保留 `alloc_guard.user_mem_limit = 0xE00000000`（56 GiB） | 甲的窗口划分在 yaml 里一眼可审计；未来加保留区（AIV UB 等）就是手动加一条 partition，与"不引入 `other_reserved` 字段"（Q1）一致；甲的唯一弱点（可能没铺满/重叠）由校验 5 完全兜住。**56 GiB 上界是线性 bump 分配器无法跳过 ccu 空洞时的补偿值**，per-segment first-fit 后失效且会把可用额度人为砍掉 94.3% |
| D15 | **GSA 不设独立 `gsa:` 键**，由 `regions[type=="GSA"]` 表达（有则一条 region，无则不写），`GetCaps().gsa_num` 按"条目缺失即 0"派生 | A5 写 region + 独立 `gsa:{count:3072}`，A6 写 `gsa:{count:0}` | 独立键与 `regions[GSA].count` 是同一数字的两份拷贝（A5 上实测冗余）；且"A6 无 GSA"用两种形态表达（独立键 count:0 vs regions 无条目），schema 不一致 |
| D16 | **WQE / SQ buffer 区不建模**（`regions` 中不含该条目） | 按 hcomm `CCU_V1_WQE_BASIC_BLOCK_OFFSET=0x1000000` 补一条 `size 32 MiB` 的 region | 该偏移由 hcomm 代码反推、**未经硬件规格确认**，写进配置会被误当权威（宁缺勿错）。删除是安全的：其全范围 `[0x1000000,0x3000000)` 已被 `resource_space_size = 72 MiB` 完整覆盖，hcomm 写入落在 backing 内、模拟器不解读。⚠ 但该安全性**依赖 D10-补**（backingSize 取权威值）—— 若改回 max 推导，A5 backing 只剩 9.25 MiB，必然越界。跟踪见 O12/T13 |
| D17 | **不回收已释放的 device VA**（`FreeVirMem` 保持空实现），只保证 1022 GiB 预算 + 水位日志 | 实现 VA 回收（空闲列表 / 尾部回滚） | 回收会破坏 `aclrt_memory_stub.cc:474-482` 记载的、有生产故障取证背书的不变量「标记行与活跃行永不同址共存」（FR1 128p / R3K4 256p 修复），而 7 处范围查询**故意不带 `is_freed` 过滤**。预算充分性：1022 GiB vs 实测峰值 0.405 GiB = **2523 倍余量**；真正的先失败点是 `kPoolSize`(4GB) 与 `/dev/shm`。四种回收方案的代价表见 **O13** |

---

## 6. 风险与开放问题

### 6.1 风险

| ID | 风险 | 等级 | 缓解 |
|---|---|---|---|
| R1 | **shm 实存/名义长度超预期**。A5/A6 的 `backingSize` 同为 **72 MiB**（`CCU_V1_V2_RESOURCE_SIZE`），128 rank × 2 die = **18 GiB** 稀疏文件；MS 被大面积触碰时实占接近该值。`/dev/shm` 默认为内存的 50% | 中 | ① 懒创建（只建被调度的 rank）；② `backingSize` 取 `resource_space_size`（72 MiB），**不得**按 1 GiB die 窗口全量 backing；③ 提供 `HCCL_VM_CCU_MEM_DISABLE`；④ `start` 阶段检查 `/dev/shm` 可用空间并按 `rank 数 × die_num × backingSize` 预警；⑤ 若实测出现 VA/页表压力，评估 `backing_segments` 段模型（§4.5.2）。<br>★ 早期版本曾估"A5 = 192 MiB → 48 GiB（A6 的 2.7 倍）"，该估算基于 level1 **错误的** `CCU_MS_BASE=0x08000000`（§4.4.3 已定论其错误），现已修正 |
| R1b | **CCU shm 若误走 `AllocPhyMem` 会被 `CommPoolPolicy` 引流**。`kBigBlockThreshold = 200MB`（`store_sim_comm_pool_policy.h:19`），check-only 模式下 `ShouldRedirect` 会把大块引流到全 rank 共享的 `HcclCommPool` → **跨 rank CCU 状态互相覆盖**，且仅在 check-only（checker 常用模式）复现，排查极难 | 低（72 MiB 距阈值余量充足）→ 但约束必须写死 | CCU shm **必须**直接走 `MemoryManager::AllocMemByName`，禁止经 `DeviceMemoryManager::AllocPhyMem/AcquirePhyMem`（§4.5.1 硬约束 1）；补 UT 断言 check-only 模式下 CCU 行的 `phy_mem_id` 不指向 `HcclCommPool`；若 O12 确认 WQE 后 `resource_space_size` 上调至 ≥200MB，本风险立即回升为高 |
| R2 | **跨进程并发写 CCU 内存**。实体化后 runner 与 proxy/checker 可同时访问同一 shm；`ccuExecutorMutex`（`ccu_resource_manager.h:123`）声明但从未使用 | 高 | 明确并发模型：runner 是 CCU 状态的**唯一写者**，其它进程只读；若需要 checker 在线对账，走 `ShmHead.lock`（`store_sim_shm_ops.h:20` 已有字段）或读写快照。**必须在设计评审时定论** |
| R3 | **DB 行被误清**（`PurgeFreedRowsOfDevice` / `DeleteAll`） | 高 | 新增 `VIR_MEM_TYPE_CCU` 并在所有清理路径显式排除（§4.5.3）；补 UT 覆盖"malloc 后 CCU 行仍在" |
| R4 | **VA 位宽**。`rank_base + 1024 × 1TiB = 0x4008000000000`，需 51 bit，超出 47-bit 规范用户 VA（< 2^47）。精确边界：窗口**起点**超 47-bit 是 rank ≥ 128（rank127 起点 `0x7F8000000000` < 2^47；rank128 起点 `0x808000000000` > 2^47）；窗口**顶端**在 rank ≥ 127 时超出 | 中 | 现状 `AllocVirMem` **已有同样问题**，非本次引入。需确认：这些值是否曾被 host 侧解引用或用于 `mmap`（探索结论是仅作 token，通过 DB 映射）。措施：加载器校验 7 输出位宽告警；文档明确"模拟设备 VA 非 host 可解引用"；若需支持 >127 rank 的真实语义，考虑缩小 `rank_stride`。<br>★ **user_mem 扩至 1022 GiB 不改变该临界**：即使按旧的 56 GiB 上界，rank127 顶端 `0x7F8000000000 + 56GiB = 0x806000000000` 也已超 2^47；且 rank 归属提取 `(va - rank_base)/rank_stride` 不受分段跳跃影响（§4.5.7.4） |
| R5 | **A5/A6 区域布局统一后 hcomm 行为变化**。同一颗 A5 现存两套 CCU 内布局（L2 `GetCcuMemInfo` 表 vs L1 `CcuResLayout` V1，§2.3 缺陷 1），统一后必有一侧输出变化 | 中 | **布局以 device yaml 为唯一权威（D10）**：L1/L2/runner/checker 四份内部表全部删除、统一读 `DeviceSpec.regions`，不保留"A5 复刻现行表"分支；S0 实测 hcomm 对 `MSG_TYPE_CCU_GET_MEM_INFO`（`hccp_stub.cc:2220`）返回表的消费方式——若仅信息上报则 yaml 全权生效，若 hcomm 另有内建布局预期（`ccu_res_specs.{h,cc}`）则 yaml 值与之校准；golden 测试固化**统一后**的 `ccu_mem_rsp` 字节序列 |
| R6 | **容量口径统一后 hcomm 分配失败**。若把 proxy 上报的 `xnNum` 从 16384 降到 runner 的 4096，可能触发 hcomm 的 CCU→AICPU fallback（`ccu_northbound_call_sequence.md:224-343`） | 中 | 统一到**真机值 16384**，同时把 runner 数组扩到 16384（16384 × 8B × 2 die = 256 KiB，可接受）。方向是"抬高 runner"而非"压低 proxy" |
| R7 | **Channel 数组越界**。V2 上报 `channelNum=1024`，runner `MAX_CCU_CHANNEL_NUM=128` | 中 | 同 R6：数组宽度改由配置驱动；`RankChannelInfo` 从 `std::array<...,128>` 改为按配置的容器 |
| R8 | 落盘格式变更影响离线 checker 与历史数据 | 低 | `ccu0/1_resource_base_addr` 字段**位置和类型不变**，只是值从魔数变真值；旧文件仍可解析（见 §9） |
| R9 | **soc_version 收敛破坏对外契约**。`Device.soc_version` 原样穿透到 3 个对外 API（`aclrtGetSocName`、`halGetSocVersion`/`halGetChipInfo`、CANN `InitRuntimePlatformInfos`），其中第 3 个**失败仅 WARN → 静默退化**，且我无法在本机验证 CANN 是否接受 family 级名字 | 中 | 分阶段（D12）：阶段 0 拓扑配置零改动、值零变化；阶段 2 改值前**必须**先实测 `InitRuntimePlatformInfos("Ascend960")` 与 `("Ascend960DT_961BY")` 的返回值（S0-a）；阶段 2 完成后逐字节比对 19 个 golden 与 `aclrtGetSocName` 输出 |
| R10 | **A6 拓扑配置数据自相矛盾**。3 个 `ascend960_*_normal*` 文件的 `hardware_type` 写成 **`950D-*`**（A5 的），而 3 个 `*_clos` 文件是正确的 `960D-clos-*`；且 `soc_version` 的分歧（`Ascend960DT_961BY` vs `Ascend960`）与这个错误**完全对齐** → 高度怀疑 `Ascend960DT_961BY` 是同批从 A5 模板复制未改净的遗留。`hardware_type` 有真实消费者：`aclrt_device_stub.cc:793-796` 用 `ACL_DEV_ATTR_DEVICE_FORM_FACTOR` 在其中找 `"POD"/"pod"` 判形态（经 `db_sim_runner_common.cc:1024` → `Server.hardware_type`），`950D-normal-0plus8-8P-topo` 不含 POD → 判为 `ACL_DEVICE_FORM_FACTOR_A_K` | 中 | **可能是现网 bug**：若这些 A6 配置本该是 POD 形态，形态判断就是错的。S0-a 一并裁决；O8 归一时连带修正 `hardware_type` 的 `950D` → `960D`。注意此问题**独立于本重构**，即使不做 soc_version 收敛也存在 |

### 6.2 开放问题（需评审定论）

| ID | 问题 | 影响 | 建议 |
|---|---|---|---|
| O1 | `LoadMicrocodeInstructionStub` 今天传的 `insinfo.resourceAddr` 到底是 hcomm 自己 malloc 的地址，还是 `GET_BASIC_INFO` 返回的 `resourceAddr + offset`？若是后者，当前必然返回 `nullptr` 并失败——但用例是通过的 | 决定指令空间是否可由模拟器直接提供 | 加日志实测一次 A5/A6 的 `SET_INSTRUCTION` 入参；据结果决定 §4.5.1 末尾的优化是否可做 |
| O2 | 多进程分 rank 时，`DeviceMemoryManager` 的 per-segment 水位（`m_deviceMemCursors`，原 `m_deviceMemSizes`）是否需要跨进程共享？ | 影响 rank 内地址唯一性 | 确认部署形态：若一进程一 rank（各进程 devPk 不同）则无需；若单进程多 rank 则已是进程内共享，也无需 |
| ~~O3~~ | ~~是否需要支持"部分区域不 backing"~~ | — | **已关闭（不设 `backed` 字段）**。理由：① tmpfs 稀疏文件按**触碰页**计费而非 `ftruncate` 长度，模拟器从不读写的 PFE/JETTY/MISSION/LOOP/CHANNEL 实存天然为 0，标 false 省 0 字节；② A6 上 8 个 region 合计仅占 `backingSize` 的 **2.13%**（1.53 MiB / 72 MiB），且 `backingSize` 由 **MS 窗口末端**决定，与 region 无关；③ 若让 `backed` 参与 `backingSize` 计算，被排除区域的 VA 会落到 mmap 之外——走 DB 查询的路径优雅失败（返回 nullptr），而 §4.5.4 的 `CcuDieView` typed accessor **无边界检查** → 野指针 SIGSEGV，直接违反方案 C 的前提 C-1/C-3；④ 要让它真生效必须按区域分段 mmap，即退化为已被 D4 否决的方案 B（文件数爆炸 + "一次减法"翻译公式作废）。**后续若实测出现 VA/配额压力，改用 `backing_segments` 段模型（可顺带解决 P10 边界检查），而非布尔开关。** |
| O4 | `Ccu` 表当前每 device 固定 2 行（`topo_ascend_cluster_parser.cc:1125-1133` 硬编码 `dieId < 2U`）。若未来 `die_num` 可配，DB 行数需随之变化 | 影响 G1 完整性 | 本期即改读 `soc.die_num`，但仅支持 1/2，>2 报"暂不支持" |
| O5 | `CcuResourceManager` 在 `EnsureCcuResource` 路径下第一参数是 **deviceId**，legacy 路径下是 **rankId**（`_v2.cc:44-45,123` vs `storage_manager.cc:326`）。实体化后必须统一 | 决定 CCU 内存索引键 | 统一为 `deviceKey`（DB 主键），并在 API 上重命名参数消除歧义；legacy 路径（`RunVirtualRuntime`）已无调用者，建议同期删除 |
| O6 | checker 是否需要"在线读 runner shm 做动静态对账"这一新能力？ | 影响 R2 并发模型 | 本期**不做**（保持 checker 只读配置 + 落盘数据），先把地址模型统一；对账作为后续独立特性 |
| O7 | `compute` / `bandwidth` 段是否本期落地？ | 影响工作量 | 建议本期只写 schema 占位（值填 0 / 不消费），把 `light_performance/sim_config.json` 与 `TaskSchedulerDevice` 的收敛作为第二期 |
| **O8** | ⚠ **TODO：A6 的 soc_version 归一**。现网 A6 存在两个值 —— `Ascend960`（3 个 `*_clos` 配置）与 `Ascend960DT_961BY`（3 个 `*_normal*` 配置），而一份 device spec 只能声明一个。三条出路：**(a) 归一化**为一个值（推荐，但改变 `aclrtGetSocName()` 返回值 + 19 个 golden，且需先实测 CANN `InitRuntimePlatformInfos` 接受度，见 R9）；**(b) 建两份 A6 spec** + `inherits:` 继承机制（CCU 布局 100% 相同，需避免复制维护）；**(c) spec 内列变体、由 server_or_pod 指定**（❌ soc_version 又变两处，违背 D11）。**连带**：R10 的 `hardware_type` 写成 `950D-*` 的数据错误一并修正 | 阻塞 §4.7.3 阶段 2/3；不阻塞阶段 0（本期） | **主方案（S1-S6）完成后处理**。阶段 0 靠 `name` 最长前缀匹配，两个值都能命中 `ascend960`，功能不受阻 |
| **O9** | **mock-comm 路径**（`InitHvmCommEnv` → `ParseRanktableAndInitCommDomain` → `ParseRanktable` → `InitCommunicationDomain`）**不经 `InitClusterStaticTopoData`**，故该模式下 `DeviceSpec*` 表无从落库；且 `Device`/`Ccu` 行在此路径下的来源也需确认 | mock-comm 是 checker 单算子校验主入口 | 阶段 0：靠 §4.3.2 的"DB 未命中 → 回退读 yaml"链路兜住（**这是硬性要求，不是可选优化**）。阶段 3：`topo_meta/*.yaml` 加 `device:` 引用（§4.8.7），顺带清掉 §4.7.1 #4 的文档幽灵 |
| **O10** | `Table(..., Version(n))` 的 n 变更语义是什么？重建表还是迁移？（仓库无 `ALTER TABLE` 迁移机制，只找到 `mariadb_engine.cc:3691` 的 `ALTER ... AUTO_INCREMENT` 用于 reset） | 决定 D13 能否从 Binding 表升级为 `Device` 加 FK 列 | 读 `sqlite_engine.cc:984` / `mariadb_engine.cc:1063` 的 `RenderCreateTable` 调用链 + 试改一次 version 观察行为。**本期采用 Binding 表，不依赖此结论** |
| **O11** | `servers_info.json` 是否新增 `device_spec` 键？ | 加了则 C++ 无需做前缀匹配（更干净），但 19 个 golden 全变 | 建议阶段 2 加。注意 `ParseServersInfoJson`（`topo_cluster_parser.cc:670-724`）需同步读取 |
| **O12** | ⚠ **TODO：WQE / SQ buffer 区的位置与偏移未经硬件规格确认，本期不建模（D16）**。<br>**hcomm 侧证据**：`ccu_res_specs.h:32` `CCU_V1_WQE_BASIC_BLOCK_OFFSET = CCU_V1_CCUM_OFFSET + 0x800000 = 0x1000000`；`ccu_jetty_ctx_mgr.cc:149` `GetSqeBuffVa` → `:172` `sqeBuffVa = ccuResBaseVa + 0x1000000 + jettyCtxId × CCU_SQ_BUFFER_SIZE(256K)`；`:222` 由 `jettyType == CcuJettyType::CCUM_CACHED_JETTY` 触发，**非版本门控**（V1/V2 共用路径）。全范围 `[0x1000000, 0x3000000)` = 32 MiB（128 jetty × 256K）<br>**安全性**：该范围已被 `resource_space_size = 72 MiB` 完整覆盖 → hcomm 写入落在 backing 内、模拟器不解读，**无越界风险**。⚠ 但此结论**依赖 D10-补**（backingSize 取权威值而非 max 推导）—— A5 的 `max(regions ∪ ms)` 只有 9.25 MiB，若按 max 推导则必然越界<br>**后续动作**：① 确认偏移后手动在 `offset 0x1000000` 加一条 region（`size 0x2000000`、`entry_size 0x40000`、`count 128`）并复核校验 8；② 确认 CheckerL2 的 `ConfigJettyInfo`（`hccp_ccu_stub.cc:467`）是否会走到 `CCUM_CACHED_JETTY` 分支<br>**相关常量备查**：`CCU_ONE_WQE_SIZE=64`、`CCU_WQE_NUM_PER_SQE=4`（URMA 约束每 SQE 含 4 个 WQEBB）、`CCU_MIN/MAX_SQ_DEPTH=16/256`、`CCU_WQEBB_RESOURCE_NUM=4096`、`CCU_START_TA_JETTY_ID=1024`（与 `sim_capacity_limits.h:133-147` 的 "CCU [1024,1151]" 一致） | 不阻塞任何阶段（72 MiB backing 已覆盖） | 确认后单独适配（用户决策：暂不修改，因未确认具体位置和偏移） |
| **O13** | **是否实现 device VA 回收**（`FreeVirMem` 现为空实现，水位只增不减）。本期决策为**不做**（D17），此处记录四种方案的代价，避免被反复重提：<br>**c1** 给 7 处范围查询加 `Eq(is_freed, 0)` → ❌ **直接回退 FR1/R3K4 修复**（慢 rank 的迟到跨 rank 解析重新未命中）<br>**c2** 加 `generation`/`epoch` 列、查询取最新代 → ❌ 语义无法自洽（"迟到解析需要旧行"与"取最新代"直接矛盾），且需 7 处查询 + 加列 + Version bump<br>**c3** **隔离期回收**：释放区间进 quarantine，直到所有 rank 越过 round/epoch 屏障才可复用（可复用既有 `hccl_proxy_round_slot.h` 的 `<RESOURCE_ROOT>/round_slots`）→ ⚠ 可行但是**独立课题**，工作量与 S3 相当<br>**c4** 仅尾部回滚（释放块恰为最顶端时回退水位）→ ❌ 回滚后下次分配仍复用该区间，同样破坏不变量，除非同时硬删该行（又回退修复） | 长跑算例序列的 VA 累积 | **本期不做**。预算充分（1022 GiB vs 实测峰值 0.405 GiB = 2523× 余量），真正的先失败点是 `kPoolSize`(4GB) 与 `/dev/shm`。若未来实测耗尽，走 **c3** 并重新验证 §4.5.7.5 的不变量 |
| **O14** | ⚠ **`AllocVirMem` 的调用点喂入两个不同的 id 空间**（S4 实现期发现，改造前即存在）：`aclrtMalloc:277` 传 `GetCurrDeviceId()`（= `Context.device_id`，DB 主键，**1 起**），而 `aclrtReserveMemAddress:972`、`aclrt_memory_stub.cc:444`、`:1040` 传 `Device.physical_id`（服务器内 local id，**0 起**，首卡合法为 0，来源 `topo_ascend_cluster_parser.cc:959` `physical_id = deviceIn.localId`）。二者共用同一 rank 窗口序列 ⇒ `physical_id=0` 的设备与 `PK=1` 的设备落到**同一窗口**，地址可能互相覆盖；且 `rankIdx = devPhyId - 1` 在 `devPhyId==0` 时 uint64 回绕，产出 `0xFFFFFF8000000000` 这类垃圾 VA（因非空而长期未被发现）。<br>**证据**：`test/proxy/sqlite/aclrt_memory_stub_test.cc:199` 的 fixture 就是 `physical_id = 0`；S4 初版把 `devPhyId==0` 直接判失败，导致该套件 4 个用例从 pass 变 fail。<br>**同源问题**：`src/device_arm/level1/data_op_device_stub.cc:243,267` 的注释已记载过同类误传（"host 侧 TryLaunchAicpuDevProcForRank 误传 GetCurrDeviceId()，其与 GetCurrDeviceKey()…"）；`docs/ranktable_rankid_device_conflict_analysis.md` 亦讨论 rankId/deviceId 口径冲突 | VA 窗口归属可能错配；与 §4.4.4 的 `rankIdx := Device.physical_id - 1` 定义直接冲突 | **S4 刻意不修**，仅：① `devPhyId==0` 走降级路径、保持与改造前逐位一致的行为以免回归；② 打一次 WARN 指明问题。正式修复应在 **S2-2**：统一三个调用点传 `deviceKey`（DB 主键），并按 §4.4.4 断言 `physical_id` 全局唯一且连续；修复时需同步评估 `aclrtReserveMemAddress` 的地址是否被其它模块按 `physical_id` 窗口反查 |

---

## 7. 分阶段实施计划

| 阶段 | 内容 | 交付物 | 依赖 | 风险 |
|---|---|---|---|---|
| **S0 前置确认** | ① 实测 **O1**（`SET_INSTRUCTION` 的 `insinfo.resourceAddr` 来源）；② **O2**（进程/rank 部署形态 → 水位是否需跨进程共享）；③ **R5**（hcomm 对 `GET_MEM_INFO` 区域表的消费方式）；④ **R9/R10 前置**：实测 CANN `InitRuntimePlatformInfos("Ascend960")` vs `("Ascend960DT_961BY")` 的返回值，裁决 A6 `hardware_type` 的 `950D-*` 是否为 bug；⑤ **O9**（mock-comm 路径下 `Device`/`Ccu` 行来源）；⑥ **O10**（`Table(Version(n))` 变更语义）；⑦ **S0-6/O12**（WQE 真实偏移 + `ConfigJettyInfo` 是否触发 `CCUM_CACHED_JETTY`）；⑧ 更大算例集上复核单 rank user_mem 峰值水位（实测日志为 0.405 GiB）；⑨ **A5 残留 T5**：PFE/CHANNEL/JETTY/MISSION/LOOP 五项偏移向 CCU 驱动/硬件规格核定。<br>★ T6/T7 已由 hcomm `ccu_res_specs` 核定完毕，**不再是出口条件**；原"A5 backingSize < 200MB"硬条件已随 R1b 降级取消（A5/A6 同为 72 MiB） | 结论纪要 | — | 低 |
| **S1 配置与模型层** | ✅ **两份 device yaml 参考实现已落地**（`asset/cluster_model/config/device/device_spec_yaml_demo_ascend950.yaml` / `_ascend960.yaml`，全部 9 条加载器校验实算通过、`GetCaps()` 与 `hccp_ccu_stub` V1/V2 现值各 10/10 命中）。<br>待做：`src/device_spec/` 库（Registry + RankAddressSpace + CcuAddressSpace + **9 条校验**）；CMake 安装规则（§4.3.3）+ 4 处 link 接入；CLI `device spec show/export`、`device addr decode/encode`；`test/device_spec/` UT。<br>★ yaml 含 `soc.soc_version` 字段但**加载器完全忽略**（决策 (i)，§4.7.4） | 新库 + UT（yaml 已就绪） | — | 低（纯新增，不改现有行为） |
| **S2-1 DeviceSpec 落库** | 新增 6 张表（`DeviceSpec`/`Region`/`MsWindow`/`Die`/`Partition`/`Binding`，§4.8.2）+ 查询接口（§4.8.3）；`InitClusterStaticTopoData` 落库；`FillCcu` 写 `resource_addr`；`dieId < 2U` 改读 `soc.die_num`；`cmd_table_utils.cc` 表打印接入 | DB 可查规格 | S1 | 中 |
| **S2-2 消费方收敛（读）** | proxy L1/L2 删字面量改读 `DeviceSpec`；`GetCcuMemInfo` 加版本分支；runner/checker 删各自地址表改用 `CcuAddressSpace`；`store_dump_shm_data` 写真值；**删 5 处 `strncmp(soc_version,...)` + `ParseDevType`**；**`CcuComponerntType` → `devspec::CcuResType`**（含 8+ 调用点）；**删 level1 `CCU_MS_BASE=0x08000000` 第 5 套 MS 地址空间** | 8 处魔数清零；4 份转换实现收敛为 1 | S2-1 | 中（R5/R6/R7） |
| **S3 CCU 内存实体化** | `CcuMemoryService`；`VIR_MEM_TYPE_CCU` + 清理路径隔离；`CcuDieView` 替换数组（**连带把 `ccu_microcode_common_v{1,2}.h` 的容量常量从 `std::array` 模板参数改为 spec 驱动**，避免同一处改两遍）；懒创建 + 灰度开关 | runner 写入可被 proxy/checker 读到 | S2-2 | 高（R1/R2/R3） |
| **S4 rtMalloc 保护** | `AllocVirMem` 改为 **per-segment first-fit** + G0~G5 六道守卫（§4.5.7.3）；修 latent bug **B1**（`devPhyId==0` 下溢）/**B2**（`std::map<int,...>` key 窄化）/**B3**（`size==0` 返回非空指针）；`m_deviceMemSizes` → `m_deviceMemCursors`；per-segment 水位日志。`FreeVirMem` **保持空实现**并补不变量注释（D17/O13） | 需求 3.3 达成（结构性保证） | S1 | **低**（实测峰值 0.405 GiB，段 0 有 138× 余量 → 零回归，§4.5.7.4） |
| **S5 技术债与插件** | P10（accessor 边界检查）/P11（trace 在 V2 路径启用）；`GetXnAddr` 的 `xnId*64` 错误步进 + `load_executor.cc:59`/`store_executor.cc:59` 传"值"当"id"；`DumpCcuCkeResouceInfo` 用 `MS_NUM`(1536) 遍历 1024 项 `cke_` 的越界；`enableDump_` 从未赋值；删 legacy `RunVirtualRuntime` 与 runner `StorageManager` 死代码（⚠ 连带 `hccl_task_thread.cc:229-255` 的 trace channel 注册）；`light_performance` 新增 `device_spec.py`；MS 平坦编址与去交织的统一（§4.4.3，与 `ccu_resource_statistics_design.md` §3.4 合并裁决） | — | S3 | 低 |
| **S6 规格扩展（可选）** | `compute`/`bandwidth` 落库并驱动 `TaskSchedulerDevice`/`ComputeDie`（生产代码当前**从不插入**这两张表）；收敛 `sim_config.json`；CCU_V3 的 `ccu.blocks`/`ccu.shared` 段（T12） | — | S1 | 低 |
| **S7 soc_version 收敛** | §4.7.3 的**阶段 1-3**：删 3 处死配置（16 个 `server_or_pod` 字段、`extract_soc_version` 与 `:637-646` 死兜底链、ai_skills 幽灵）→ `server_or_pod` 加 `device:` 引用 → `cluster` 删字段 + `servers_info.json` 加 `device_spec` 键 + 19 golden 重生成 → `topo_meta` 加 `device:` 打通 mock-comm + **O8 归一裁决落地**（连带 R10 的 `hardware_type` 修正） | 全项目 soc_version 声明处数 3 → **1** | **主方案（S1-S6）完成后** | 中（R9：3 个对外 API） |

**建议 S1+S2 作为一个可独立发布的里程碑**：即使 S3 未落地，S2 已经消除了"假地址需二次转换"（P2）、"各 rank 基址相同"（P3）、"4 套布局冲突"（P4）、"4 份重复实现"（P5）、"容量矛盾"（P6）五个问题，收益/风险比最高。

**每阶段的"零回归"证明**：

| 阶段 | 证明方式 |
|---|---|
| S1 | `hccl-vm start` 行为逐字节不变（纯新增，未接消费方）；19 个 golden 不变 |
| S2-1 | `Ccu.resource_addr` 从 0 变为真值；`DeviceSpec*` 6 表可 dump；其余表不变 |
| S2-2 | golden 重生成后**人工 review diff**（`GetCcuMemInfo` 输出必然变化，这是修 §2.3 缺陷 1）；A5/A6 各跑一遍现有算例集 |
| S3 | 新增跨进程可见性测试通过；`HCCL_VM_CCU_MEM_DISABLE=1` 下行为回退到 S2 状态 |
| S4 | 现有算例全部落在 user_mem 段 0，已分配 VA 与旧公式逐位相同（§4.5.7.4）；19 golden 不变 |

---

## 8. 验收标准与测试计划

### 8.1 验收标准

| 编号 | 标准 | 验证方法 |
|---|---|---|
| A1 | 全仓不再有 `0x123123123` / `0x456456456` 字面量（测试 golden 除外） | `grep -rn` 断言 + CI 门禁脚本 |
| A2 | A6 区域偏移字面量（`0x100000`/`0x140000`/`0x4000000` 等）不再散落在 runner/checker/proxy | 同上 |
| A3 | `ccuBase(rank,die)` 对任意 (rank,die) 唯一，且落在 `[rankBase+56G, rankBase+58G)` | UT 遍历 rank 0..N-1 × die 0..1 断言 |
| A4 | runner 执行 `SetCKBit` 后，另一进程按同一 VA 能读到 CKE 位 | 新增跨进程集成测试（主进程 + runner） |
| A5 | **rtMalloc 分段映射**（需求 3.3）：① 段 0 内分配的 VA 与旧公式 `rankBase + used` **逐位相同**（零回归）；② 请求大小 > 段 0 余量时跳到段 1，**中间 2 GiB CCU 窗口永不返回**（rank0：`0x8DFFFFFFF8` → `0x8E80000000`）；③ 任意成功分配的 `[va, va+size)` 首尾双端经 `IsInCcuWindow()` 均为 false；④ 所有段耗尽（累计 > 1022 GiB）时返回 nullptr 并打 ERROR，日志含 rank window / ccu reserved / user_mem used-total；⑤ `devPhyId == 0`（B1）与 `size == 0`（B3）均返回 nullptr | UT（`test/store/`） |
| A6 | `GetCcuMemInfo` 对 A5/A6 返回不同且与各自配置一致的区域表 | golden 字节序列比对（★ 该函数当前**零测试覆盖**，需先补"改动前行为"的 characterization golden，见 M3） |
| A7 | 配置非法时启动失败并给出定位信息 | UT 覆盖 §4.3.1 的 **9 条**校验（含校验 3 的异构窗口反例、校验 5 的铺满性、校验 8 的 `resource_space_size` 上界） |
| A8 | `hccl-vm device addr decode <VA>` 能正确输出 (rank, die, region, id)；`device addr encode` 为其逆运算 | CLI 测试 |
| A9 | 现有 checker/runner 全量用例回归通过 | 既有 UT + `install.sh` 的 `run_test` |
| A10 | 128 rank 拓扑下 `/dev/shm` 实占可测且在预期内（18 GiB 稀疏） | 压测 + `du /dev/shm/hvm_*` |
| **A11** | **soc_version 声明处数**：阶段 0 = 3 处并存（cluster / server_or_pod / device）且 device 侧被加载器忽略；阶段 3 完成 = **1 处**（device yaml） | `grep -rn "soc_version" asset/cluster_model/config/` 计数 |
| **A12** | **阶段 0 结束时 19 个 golden `servers_info.json` 逐字节不变**（拓扑配置零改动的证明） | `test/topo` golden 全量比对 |
| **A13** | `DeviceSpec*` 6 张表可由 `hccl-vm table show` dump；runner / checker / light_performance 三方读到的规格**完全一致**（与 yaml 交叉校验通过） | 三方各跑一次并比对；`hccl-vm device spec export` 的 JSON 与 DB 内容一致 |
| **A14** | 两份 device yaml 的 `address_space` 段**逐字段相同**（芯片无关）；`resource_space_size` 相同；MS 几何参数（`ms_size`/`ms_num`/`ccua_num`/`interleave`）相同，仅 `ccua_windows[].offset` 不同 | UT（已用脚本验证通过，需固化为 C++ UT） |
| **A15** | `GetCaps()` 推导结果与 `hccp_ccu_stub.cc:69-96`(V1) / `:98-125`(V2) 的现值 **各 10/10 命中** | UT（作为 T5/T7 填写正确性的交叉验证） |

### 8.2 测试计划

- **UT（`test/device_spec/`，新增）**
  - yaml 解析：正常 / 缺字段 / 用 32 位类型取 ≥2^32 的值（应抛 `TypedBadConversion`，验证 §4.2.2 的解析注意）/ 区域重叠 / region 与 MS 窗口重叠（校验 1 合并检查的必要性）/ 交织参数不匹配；
  - **校验 3 的异构窗口反例**：窗口容量 `[256,512,512,512]` 槽时求和式通过但 per-window 失败 → 断言加载器 fail-fast；
  - **校验 5 铺满性**：缺一段 / 段重叠 / 首段 offset≠0 / 末段不达窗口末端 / 两个 ccu 段 → 各自 fail-fast；
  - **校验 8**：`max(regions ∪ ms) > resource_space_size` 与 `resource_space_size > die_size` → fail-fast；
  - `RankAddressSpace`：需求 3.1/3.2 的 6 组期望值精确断言（§4.4.2 表格）；`userMemTotal == 0xFF80000000`；段 0 末端 == CCU 起点、段 1 起点 == CCU 末端、段 1 末端 + 1 == rank1 起点；
  - `CcuAddressSpace`：`AddrToResId` 与 `ResIdToAddr` 互逆性（随机 id 全区域 fuzz）；MS 交织 `id↔addr` 互逆**遍历 0..1535**；边界（id=0、id=count-1、越界返回 false 而非 UB）；`SupportsAddrDecode()` 对 V1 返回 false；
  - **`GetCaps()` 交叉验证（A15）**：A5 → `hccp_ccu_stub.cc:69-96` V1 现值 10/10；A6 → `:98-125` V2 现值 10/10；含 `gsa_num` 的"条目缺失即 0"派生（A5=3072 / A6=0）；
  - **双文件一致性（A14）**：`address_space` 段逐字段相同、`resource_space_size` 相同、MS 几何参数相同（仅 `ccua_windows[].offset` 不同）；
  - **匹配规则**：4 个真实 soc_version 值（`Ascend950DT_95A1`/`Ascend950`/`Ascend960`/`Ascend960DT_961BY`）→ 正确 spec；未知值 fail-fast；并列 fail-fast；★ 断言 spec 内的 `soc.soc_version` **不参与**匹配（决策 (i)）；
  - **对齐测试**：用同一份 yaml 分别喂 C++ 与 Python 实现，断言 `BackingSize()`/`GetCaps()`/MS 映射输出一致（防第 5 份实现再度漂移）。
- **UT（既有目录扩展）**
  - `test/plugin/checker/storage_manager_checker_test.cc:841-861` 当前断言 `resourceBaseAddr[0] == 0x123123123`，需改为按配置期望值（且升级为 per-(rank,die) 断言）；
  - ★ `test/plugin/runner/ccu_resource_manager_test.cc` 当前所有 `Init(...)` 的第 4 参数都是 **`{}`**（空基址列表，`:59,66,73,80,92,104,116,131…`）→ `ccuResourceBaseAddr_` 为空，4 个地址转换函数一律在入口 `dieId >= size()` 处返回 false，**函数体从未被执行 = 零覆盖**（这正是 `ccu_resource_manager.cc:512,522` 的 `// 需要判断界限 todo` 长期未暴露的原因）。S2-2 对 runner 是**从零新建测试**，不是改现有测试；
  - `test/topo/sqlite/topo_golden_test.cc`：`Ccu.resource_addr` 与 `DeviceSpec*` 6 表落库值的 golden；`:399-409` 补 `linkConfig("device")`；
  - `test/store/`：`AllocVirMem` 分段映射（A5 的 5 条）+ **跨洞反例**（段 0 游标 `0xDFFFFFF00`、请求 `0x400` → 必须落到段 1 `rankBase+0xE80000000`，而非 `rankBase+0xDFFFFFF00`）+ 跳段浪费 < 请求大小 + B1/B2/B3 三个守卫 + `VIR_MEM_TYPE_CCU` 不被 `PurgeFreedRowsOfDevice` 清除 + check-only 模式下 CCU 行的 `phy_mem_id` 不指向 `HcclCommPool`（R1b）；
  - `test/proxy/`：★ `GetCcuMemInfo` / `SetCcuResourceBasicInfo` **当前零测试覆盖**（已核实 `test/` 下无相关文件），需**先在 S1 补 characterization golden**（记录改动前行为），再在 S2-2 补 V1/V2 分支的新 golden，使 diff 可审阅、证明是有意修正而非意外回归（M3）。
- **集成**
  - 跨进程可见性（A4）：主进程 Ensure → runner 写 → 主进程读；
  - hcomm CCU 接口用例（此前**原理性不可测**）：`HcommCcuVariableAlloc` → launch → 回读 XN；
  - A5 与 A6 各跑一遍现有算例集；
  - **mock-comm 路径**（O9）：`hccl-vm mock-comm 112` 下 DB 无 `DeviceSpec*` 行时，消费方必须回退读 yaml 且功能正常。
- **压测**：128 / 1024 rank 下的 shm 实占与 VA 占用（A10、R1、R4）。
- **门禁脚本（CI，防魔数复发）**
  ```bash
  # 期望全部 0 命中（test/topo/golden 与 docs 除外）
  grep -rn "0x123123123\|0x456456456" src/ include/
  grep -rn "XN_OFFSET_ADDR_A6\|XN_CKB_ADDR_A6\|XN_CCUA0_MS_ADDR_A6\|CCU_MS_BASE" src/ include/
  grep -rn 'strncmp(.*soc_version.*"Ascend9' src/
  grep -rn "findTypeByAddr\|findBaseSize\|findBaseAddr\|CcuComponerntType\|ParseDevType" src/ include/
  grep -rn "alloc_guard\|user_mem_limit\|soc_family\|soc_aliases" asset/cluster_model/config/device/
  grep -rni "wqe\|sq_buffer" asset/cluster_model/config/device/     # D16/O12：yaml 零痕迹
  ```

---

## 9. 兼容性说明

| 项 | 影响 | 处理 |
|---|---|---|
| 落盘二进制格式 | `ModelInfoCommInner.ccu0/1_resource_base_addr` 字段位置/类型不变，仅值从魔数变真值 | 旧文件仍可解析；新版 checker 读到魔数时 WARN 并按配置覆盖（过渡期兼容） |
| DB schema | **新增 6 张 `DeviceSpec*` 表**（§4.8.2，`Scope::Static` → `hccl_sim.db`）；`Ccu.resource_addr` 列已存在无需改；新增 `src_type` 枚举值 `VIR_MEM_TYPE_CCU`（**追加在末尾**，不改变既有值数值）；`Device` 表**不动**（D13 用 Binding 表关联） | 表由 `Table(...)` 宏元数据自动建（sqlite/mariadb 共用 `RenderCreateTable`），零后端 DDL；旧 DB 中 `resource_addr = 0` 视为"未初始化"，触发懒创建 |
| 已分配的设备 VA | `rankIdx = physical_id - 1` 与 `RANK_BASE`/`RANK_STRIDE` 保持不变；user_mem 从"56 GiB 上界"扩为"1022 GiB / 2 段" | **片上内存地址完全不变**（实测峰值 0.405 GiB 全落在段 0，段 0 内 VA 与旧公式逐位相同，§4.5.7.4）；19 个 golden 不受影响 |
| `m_deviceMemSizes` → `m_deviceMemCursors` | 语义从"窗口内 VA 偏移"变为"per-segment 线性水位"，且 key 类型从 `int` 修正为 `uint64_t`（B2） | **进程内状态、不持久化** → 无跨版本兼容问题；已核实无任何外部读者（仅 `AllocVirMem` 内 2 处） |
| device VA 回收 | `FreeVirMem` 保持空实现（D17） | 与现状一致，无行为变化；不变量与理由见 §4.5.7.5，四种回收方案代价见 O13 |
| hcomm 交互 | `resourceAddr` / `ccu_mem_rsp.mem_va` 数值变化 | hcomm 侧只做透传与偏移计算，不依赖具体数值；区域表统一由 `DeviceSpec.regions` 生成（D10/R5），A5 输出随 yaml 初值变化，S0 实测确认 hcomm 消费方式 |
| `soc_version` 对外契约 | **阶段 0 零变化**（拓扑配置不改、值不改）；阶段 2 才可能改值 | 3 个对外 API 见 §4.7.6；`Device.soc_version` DB 列**永久保留**（CLI 可写、被当文档示例）；改值前必须先过 R9 的 CANN 实测 |
| 灰度回退 | — | `HCCL_VM_CCU_MEM_DISABLE=1` 回到进程私有数组模式（S3 引入，S5 后可移除） |
| 第三方 | 假地址的二次转换逻辑作废 | 发布说明中给出 `旧魔数 → 新公式` 的映射说明，并提供 `device addr decode` CLI 与 `device spec export` JSON 辅助接入 |

---

## 10. 附：现状代码索引（便于实施时定位）

**魔数与地址表**
- `src/proxy/level2/hccp_ccu_stub.cc:69-143`（V1/V2 caps + resourceAddr）、`:201-305`（`LoadMicrocodeInstructionStub`）、`:400-410`（魔数移位判 die，A5-only）
- `src/proxy/level2/hccp_stub.cc:2077-2229`（`ccu_mem_info`/`GetCcuMemInfo`/`RaTlvRequest`）
- `src/proxy/level1/ccu/ccu_level1_res.cc:31-48`（常量）、`:386-481`（bump alloc）、`:504-578`（layout/addr/remaining）
- `src/proxy/level1/ccu/ccu_level1_res.h:38-135`（结构体）、`:250,293-295`（`SYNC_SLOT_NUM`/水位）
- `src/proxy/level1/ccu/ccu_kernel_sim.cc:60-75,1085-1113`（第 5 套 MS 地址空间）
- `src/plugin/runner/ccu_executor/ccu_resource_manager.cc:378-531`（A6 地址表 + 4 个转换函数）、`:45-63`（`Init`）、`:134-237`（accessor，无边界检查）、`:350-376`（`GetMsAddr`/`GetXnAddr`）
- `src/plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_all_rank_param_recorder_v3.cc:259-445`（复制品）
- `src/plugin/runner/runner_worker/hccl_task_sequential_execute_v2.cc:35-180`（`EnsureCcuResource`）
- `src/plugin/runner/runner_utils/storage_manager.cc:309-342`（死代码路径）
- `src/plugin/checker/src/utils/storage_manager.cc:757-763`（`InitCcuInfo`）
- `src/store/store_dump_shm_data.cc:454-540`（`CreateSimSynData`，`:497-498` 占位）

**内存与地址**
- `src/store/store_sim_device_memory_manager.cc:47-62`（`AllocVirMem`，1TiB 窗口）
- `src/proxy/level2/aclrt_memory_stub.cc:129-155`（`GetRealPtrByAddr`）、`:212-252`（`GetDevMappedMemPtr`）、`:253-350`（`aclrtMalloc`）
- `src/store/store_sim_shm_memory_common.cc:22-64`（`AcquireDevPtrInNoHostProcess`）
- `src/store/store_sim_shm_ops.cc:91-136`（`ShmCreate`）
- `src/store/store_sim_resource_root.cc:29-42`（`HCCL_VM_RESOURCE_ROOT`）

**规格常量**
- `include/sim_common_defs.h:34-47`（`DIE_NUM`/`CCU_*_MAX`/`MS_NUM`）
- `src/plugin/plugin_common/ccu_microcode_v1.h:20-50,847-853`（`CcuInstr` 32B）
- `src/plugin/runner/ccu_executor/ccu_microcode_common_v1.h:64-79` / `_v2.h:72-87`（`SimCcuV1`/`SimCcuV2` 容量）
- `include/sim_capacity_limits.h`（`HCCL_VM_MAX_CHANNEL_JETTIES`、notify id 位宽合同、AICPU Jetty 编号合同；★ 原 `HCCL_VM_MAX_RANKS=1024`/`GetMaxRanks()` 已删除，rank 规模不设上界）
- `include/runtime_state/sim_models.h:164-170`（`Ccu`）、`:429-443`（`SimModelData`）
- `include/sim_binary_data_type_pub.h:140-196`（`ModelInfoCommInner`/`HcclVmSynData`）

**配置与打包**
- `CMakeLists.txt:154-204`（`config_hcclvm_install`，新增第 4 段 install `config/device`）
- `asset/generate_cluster_topo.sh:100-113`（配置目录解析，需加 `DEVICE_CONFIG_DIR`）、`:345-361`（`find_server_topo`，引用解析先例）、`:442,502`（server_list 展开）、`:584-588`（`extract_soc_version`，**死代码待删**）、`:627-654`（`servers_info.json` 生成 + 死兜底链）
- `asset/generate_server_topo.sh:280,300`（读 `soc_version`，仅日志）、`:1425-1439`（`hardware_type` 派生链，`soc_version.upper()` 兜底**不可达**）、`:1441-1448`（`topo_json` 字典，**不含 `soc_version` 键**）
- `src/utils/sim_common_api.cc:41-78`（install root）、`:173-265`（`ParseTopoMetaYaml`，YAML 解析范例，**不读 `soc_version`**）
- `src/utils/sim_yaml_config.cc`（`LoadYamlStringMap`，`HAVE_YAML_CPP` 保护范例）
- `src/topo/topo_cluster_parser.cc:670-724`（`ParseServersInfoJson`，`:703-710` 读 `soc_version`）、`:218-309`（`ParseTopoJson`，读 `hardware_type`）
- `src/topo/topo_ascend_cluster_parser.cc:956-975`（`FillDevice`/`FillCcu`）、`:1010-1167`（`InitClusterStaticTopoData`，`:1125-1133` 硬编码 `dieId < 2U`）、`:346-370`（`ParseRanktableAndInitCommDomain`，mock-comm 路径）、`:382-421`（`InitCommunicationDomain`）、`:1265-1355`（`ParseRanktable`）
- `src/cmd/cmd_base_utils.cc:503`（`InitClusterTopo` 调用点）、`:523`（`InitHvmCommEnv`，mock-comm 入口）、`:1540`（`DeleteAll`）
- `src/cmd/subcmd/subcmd_start.cc:70,77,94`（生成 → 配置目录 → InitHvmEnv）
- `src/cmd/cmd_base.h:24-49`（`CommandBase` / `CommandRegistry` / `CommandAutoRegister`，新增 `subcmd_device` 的接入方式）、`src/cmd/cmd_base_utils.cc:1358-1367`（`ParseCommand`）
- `include/runtime_state/db_sim_runner_ops.h:48-57`（DDL 宏）、`:65-…`（66 张 `Table(...)` 清单，新增 6 条）
- `src/storage_backends/sqlite/engine/sqlite_engine.cc:984` / `mariadb/engine/mariadb_engine.cc:1063,1070`（`RenderCreateTable`，两后端共用元数据）
- `test/topo/CMakeLists.txt:17-24`、`test/cmd/CMakeLists.txt:15-22`、`test/topo/sqlite/topo_golden_test.cc:374-410`
- `test/topo/golden/cluster/*/servers_info.json`（19 个，含 `soc_version`）
- `ai_skills/skills/generate-ranktable-json/references/Ascend_ranktable_file_generation.md:68,83` + `scripts/generate_ranktable.sh:287,303,451,457`（topo_meta 的 `soc_version` **文档幽灵**）
- `ai_skills/skills/generate-cluster-json/references/ascend950_super_node_demo.yaml`、`generate-server-topo-json/references/server_topo_description.yaml`（schema 文档，阶段 1-2 需同步）

**soc_version 消费点（S7 收敛时的完整清单）**
- 对外 API（§4.7.6，**不可破坏**）：`src/proxy/level2/aclrt_device_stub.cc:477-497`（`aclrtGetSocName`）、`:78-96`（`InitRuntimePlatformInfoFromDevice` → `fe::PlatformInfoManager::InitRuntimePlatformInfos`）、`:785-800`（`ACL_DEV_ATTR_DEVICE_FORM_FACTOR` 用 `hardware_type` 判 POD/A_K）；`src/proxy/level2/ascend_hal_stub.cc:237-247`（`halGetSocVersion`）；`src/device_arm/level2/proxy/ascend_hal_stub.cc:122-123`（`halGetChipInfo`）
- 版本判定（S2-2 删除，改查 `DeviceSpec.ccu_version`）：`src/proxy/level2/hccp_ccu_stub.cc:134,258,513`、`src/proxy/level1/ccu/ccu_level1_res.cc:494-498`、`src/proxy/level2/aclrt_new_stub.cc:47-52`、`src/plugin/runner/runner_worker/hccl_task_sequential_execute_v2.cc:103-105`
- 待删除的字符串匹配表：`src/topo/topo_cluster_ir.cc:137-152`（`ParseDevType`，对真实值恒返回 `UNKNOWN` 且无 960 分支）、调用点 `topo_cluster_parser.cc:710`、`topo_cluster_dumper.cc:105`
- CLI 可写列（**保留**）：`src/cmd/subcmd/subcmd_table.cc:47`（help 示例）、`src/cmd/cmd_table_utils.cc:44-50,600-604`（TEXT 列特判）、`test/cmd/subcmd/subcmd_table_test.cc:123`
- DB 列：`include/runtime_state/sim_models.h` 的 `Device.soc_version[32]`、`Server.hardware_type[128]`；读取 `src/runtime_state/db_sim_runner_common.cc:1018-1026`（`GetHardwareTypeByDevice`）

**hcomm 权威来源（外部仓，S0/S2 核定依据）**
路径相对 `<hcomm>/src/base_comm/resources/ccu/`：
- `ccu_device/ccu_res_specs.h:24-25`（`CCU_V1_V2_RESOURCE_SIZE = 72 MiB`、`CCU_V3_RESOURCE_SIZE = 128 MiB`）、`:26-27`（`CCU_V1/V2_CCUM_OFFSET`，**CCUM 与 CCUA 顺序相反**）、`:28-31`（V3 CCUA 8 / Variable 16K / Event 2048）、`:32`（`CCU_V1_WQE_BASIC_BLOCK_OFFSET = 0x1000000`，O12）、`:34-46`（WQE/SQ/jetty/PFE 常量）、`:50-59`（INS/Address/Variable 预留大小、per-size、`CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM = 4`）、`:95-135`（`CcuResSpecInfo`，V3 私有/共享分块）、`:238`（`serveMode_` 默认 ARMX86「规避使用部分 ccua」）
- `ccu_device/ccu_res_specs.cc:353-395`（`GetXnBaseAddr`，V1/V2 分支）、`:397-440`（`GetCkeBaseAddr`）、`:765`（`GetResourceSpaceSize`）
- `ccu_device/ccu_block_res_strategy.cc:23`（`CCU_V1_V2_CCUA_NUM = 4`）、`:253`（`avoidCcu0Flag`）、`:436-438`（V1/V2 用 GlobalId、V3 用 RelativeId 策略）
- `ccu_device/ccu_comp/ccu_channel/ccu_jetty_ctx_mgr.cc:149-175`（`GetSqeBuffVa`）、`:222`（由 `jettyType == CCUM_CACHED_JETTY` 触发，**非版本门控**）
- `ccu_microcode/ccu_microcode_v1.h:28-29`（`CCU_MS_SIZE = 4096`、`CCU_MS_INTERLEAVE = 8`）、`:630`（CcuV2 段 `xmId` 的 `gsaOffset[52:21]` 位域，A6 用作地址步长）、`:857,859`（V1 专有的 GSA 指令编码器）
- `src/legacy/ascend950/unified_platform/ccu/ccu_device/ccu_res_specs_legacy.h:19-21`（A5 legacy 副本，与 base_comm 版一致）

**相关文档**
- `ccu_dataplane_architecture.md`（hcomm 控制面/数据面分层）
- `ccu_northbound_call_sequence.md`（北向调用与 CCU→AICPU fallback）
- `docs/hccl_vm_binary_file_format.md:88-105,229-256`（syn/instr 文件格式）
- `docs/checker_quick_intro.md`、`docs/faq/modules/checker_faq.md`（错误码 102-113 语义）
- `docs/light_performance/simulator_design.md:1061-1080,1534`（Python 侧移植说明与简化声明）
- `docs/ranktable_rankid_device_conflict_analysis.md`（rankId 与 deviceId 口径冲突，与 §4.4.4 相关）
- ★ `docs/ccu_resource_statistics_design.md`（CCU 资源统计特性；与本方案共用同一套资源 taxonomy，协同要求见 §4.7.7 / A9；其 §3.4 遗留问题与 §4.4.3 是同一个）
- ★ `asset/cluster_model/config/device/device_spec_yaml_demo_ascend950.yaml` / `device_spec_yaml_demo_ascend960.yaml`（**已落地的参考实现**，含逐字段 hcomm 依据行号与 9 条校验实算结果）
