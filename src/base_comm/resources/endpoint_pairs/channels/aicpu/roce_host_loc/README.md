# RoCE host-loc 通道模块说明

---

## 模块定位

本目录包含 **host-loc RoCE 链路**的 host 侧通道实现与 device 侧 lite transport 实现，用于A5 npu直驱host网卡的NDA特性。

- **host-loc**：表示通道的网卡位于host侧，QP/SQ/CQ 上下文由 host 侧建立、经序列化下发，device 侧直接按 VA 组 WQE、敲 doorbell，不经过 host 代理线程。
- 与之对应的是 **dev-loc** 链路（`AicpuTsRoceDevLocChannel`，见上级 `aicpu/` 目录），该通道的网卡位于device侧。

---

## 文件清单

```text
roce_host_loc/
├── CMakeLists.txt                          # host 侧 target 归属 + add_subdirectory(device)
├── aicpu_ts_roce_host_loc_channel.{h,cc}   # host：通道主控
├── roce_host_loc_connection.{h,cc}         # host：单 QP 连接
└── device/
    ├── CMakeLists.txt                      # device 侧 target 归属
    ├── roce_host_loc_transport_lite_impl.{h,cc}
    ├── roce_host_loc_conn_lite.{h,cc}
    ├── roce_vendor_base_ops.h
    └── roce_vendor_1825_ops.h
```

| 文件 | 主要类 / 内容 | 所属侧 |
|------|--------------|--------|
| `aicpu_ts_roce_host_loc_channel.{h,cc}` | `hcomm::AicpuTsRoceHostLocChannel`（继承 `hcomm::Channel`） | host |
| `roce_host_loc_connection.{h,cc}` | `hcomm::RoceHostLocConnection` | host |
| `device/roce_host_loc_transport_lite_impl.{h,cc}` | `Hccl::RoceHostLocTransportLiteImpl`（继承 `Hccl::BaseTransportLiteImpl`） | device(kernel) |
| `device/roce_host_loc_conn_lite.{h,cc}` | `Hccl::RoceHostLocConnLite` | device(kernel) |
| `device/roce_vendor_base_ops.h` | `Hccl::RoceBaseOps`、`RdmaSqContextLite` / `RdmaCqContextLite`、DbVendor 位域常量 | 纯头文件 |
| `device/roce_vendor_1825_ops.h` | `Hccl::Roce1825Ops`（UB/1825 厂商实现） | 纯头文件 |

职责细分：

- **AicpuTsRoceHostLocChannel**：通道生命周期主控。`ParseInputParam` 取本地 endpoint/rdmaHandle；`StartListen`/`BuildSocket` 建链；`CreateQp`/`ModifyQp`/`ExchangeData` 完成 QP 建链与参数交换；`BuildNotify`/`BuildBuffer`/`BuildConnection` 汇集资源，`H2DResPack`/`Serialize` 打包为 blob 下发 device；`GetStatus`/`ProcessStatus` 维护 `ChannelStatus` + 内部 `RdmaStatus` 状态机。
- **RoceHostLocConnection**：单个 QP 连接的 host 侧视图，负责与对端交换 QP 参数（`ExchangeRdmaConnDto`）并序列化进 blob。
- **RoceHostLocTransportLiteImpl**：device 侧数据面入口。`Init` 反序列化 blob（notify / loc buffer / rmt buffer / conn 四类），对外实现 `Read`/`Write`/`WriteReduce`/`WriteWithNotify`/`WriteReduceWithNotify`/`Fence`/`Post`/`WaitWithTimeout`，内部按需构建 `RdmaDbSend` 与 `NotifyWait` task 并做 profiling 上报。
- **RoceHostLocConnLite**：device 侧连接对象。解析 SQ/CQ context，按 `RDMA_DMA_MAX_SIZE`(2 GB) 分片，委托厂商 ops 组 WQE、构造 doorbell；`PollCq` 收 CQE 并构造 CQ doorbell。
- **RoceBaseOps**：厂商 ops 模板基类。通用流程（`WaitSqFree` → vendor 组装 WQE → `CommitWqe` 写 SQ → `UpdateSqPI` 更新软件 PI）在基类，`BuildReadWqe`/`BuildWriteWqe`/`BuildWriteReduceWqe`/`PollCqImpl`/`BuildDoorbell`/`BuildCqDoorbell` 为厂商扩展点（默认返回 `HCCL_E_NOT_SUPPORT`）。

---

## 类关系

```text
hcomm::Channel                          (channels/channel.h)
 └── AicpuTsRoceHostLocChannel          ← 本目录
          └─ has ──> hcomm::RoceHostLocConnection   ← 本目录

hcomm::Channel
 └── AivRoceHostLocChannel              (channels/aiv/)
          └─ has ──> hcomm::RoceHostLocConnection   ← 复用本目录实现

Hccl::BaseTransportLiteImpl              (legacy ascend950 unified_platform/resource/transport/aicpu/)
 └── Hccl::RoceHostLocTransportLiteImpl ← 本目录 device/；override 10+ 虚函数，由 unique_ptr<BaseTransportLiteImpl> 多态持有

Hccl::RoceHostLocConnLite               ← 本目录 device/（已不继承 RmaConnLite）
 └─ has ──> Hccl::RoceBaseOps
              └── Hccl::Roce1825Ops    ← 本目录
```

注意点：

- `BaseTransportLiteImpl` 基类**仍在 legacy 目录**，本目录是其一个实现，迁移不改变该继承关系。
- `RoceHostLocConnLite` 曾名义继承 `RmaConnLite`（仅 override `Describe()`，基类成员只写不读），迁移后已解除，仅在需要 `SqeConfigLite` 等类型定义时 `#include "rma_conn_lite.h"`。
- legacy `connection/aicpu/rdma_conn_lite.h` 中的 `RdmaConnLite` 是**打桩类**，仅供 legacy `RmaConnLite::Create` 工厂分发使用（该工厂为死代码），与本目录的 `RoceHostLocConnLite` 无继承关系。

---

## 数据流

```text
host 侧
  Channel 建链（socket / QP create+modify）→ 交换 QP 参数
      → 汇集 notify / buffer / conn → H2DResPack / Serialize → blob 下发

device 侧
  RoceHostLocTransportLiteImpl::Init(blob)  解出 notify / buffer / conn
      → RoceHostLocConnLite(vector<char>)   解析 dmaMode + SQ/CQ context
                        → RoceBaseOps（Roce1825Ops）组 WQE
                            → CommitWqe 写 SQ、UpdateSqPI 写软件 DB
      → 上层 Read/Write/... 生成 RdmaDbSend / NotifyWait task 并上报 profiling
```

---

## 编译归属

| 源文件 | target |
|--------|--------|
| `aicpu_ts_roce_host_loc_channel.cc`、`roce_host_loc_connection.cc` | `hcomm_base_obj`（host OBJECT，链接进 `hcomm`） |
| `device/roce_host_loc_transport_lite_impl.cc`、`device/roce_host_loc_conn_lite.cc` | `ccl_kernel`（仅 device 构建编译） |
| `device/roce_vendor_*.h` | header-only，无需登记 |

**约束**：device 侧源文件只登记给 device 内核 target（`ccl_kernel`），不得加入 host 侧 target
（`hcomm_base_obj`/`hcomm`/`hccl_v2`），否则污染 host 包与二进制拆分。

---

## 依赖说明

本目录对外部（legacy / 其他目录）的依赖**不会因迁移而减少**，新增文件时需同步维护 include 搜索路径（引号 include 先搜当前文件所在目录，再逐个 include dir 拼接）。device 子目录**由外部平名引用**（如 `coll_comm_aicpu.h`、`aicpu_channel_process.cc` 引用 `roce_host_loc_transport_lite_impl.h`），因此 host 目录与 device 目录需分别登记：

| 需维护的位置 | 影响 target |
|--------------|-------------|
| `src/base_comm/resources/endpoint_pairs/channels/aicpu/CMakeLists.txt` | `hcomm_base_obj`（host 目录） |
| `src/legacy/ascend950/hccl_v2.cmake` | `hccl_v2`（host + device 目录） |
| `src/legacy/ascend910/framework/ccl_kernel.cmake` | `ccl_kernel`（host + device 目录） |
| `src/legacy/ascend910/framework/hcomm.cmake` | `hcomm`（host + device 目录） |
| `test/ut/CMakeLists.txt`（`UT_COMMON_INCLUDE_LIST`） | UT 公共（host + device 目录） |
| `test/ut/stub/CMakeLists.txt` | `hccl_llt`（host + device 目录，需同时登记源文件） |

**原则**：只新增本目录路径，不得删除 legacy 路径项（本目录仍依赖 legacy 的基类与资源类型）。

主要外部依赖：

- host 侧：`Channel`/`AicpuTsChannelHelper`、`Endpoint`、`Hccl::Socket`、`Hccl::LocalRdmaRmaBuffer`、`Hccl::RdmaLocalNotify`、`ExchangeRdmaConnDto`（同级 `host/` 目录）、`SocketMgr`。
- device 侧：`BaseTransportLiteImpl`、`RmaBufferLite`/`RmtRmaBufferLite`、`StreamLite`/`RtsqBase`、`NotifyLite`、`BinaryStream`；并反向依赖 `base_comm` 的 `cast_utils.h`、`dfx_profiling_handler_lite.h`。

---

## 命名约定

- 目录与文件前缀统一 `roce_host_loc_`，类名统一 `RoceHostLoc*`。
- device 侧日志宏统一 `[RoceHostLocConnLite::%s]` / `[RoceHostLocTransportLiteImpl::%s]`，便于按模块过滤。
- 通道 `SetModuleDataName` 与 `HcommChannelKind` 取值分别为 `"AicpuTsRoceHostLocChannel"`、`AICPU_TS_ROCE_HOST_LOC`。