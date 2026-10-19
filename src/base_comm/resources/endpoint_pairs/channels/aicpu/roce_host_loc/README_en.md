# RoCE host-loc Channel Module Description

---

## Module Positioning

This directory contains the host-side channel implementation and the device-side lite transport implementation of the **host-loc RoCE link**, used by the NDA feature in which the A5 NPU directly drives the host NIC.

- **host-loc**: the NIC of the channel resides on the host side. QP/SQ/CQ contexts are created on the host and delivered through serialization; the device side builds WQEs by VA and rings the doorbell directly, without going through the host proxy thread.
- The counterpart is the **dev-loc** link (`AicpuTsRoceDevLocChannel`, in the parent `aicpu/` directory), whose NIC resides on the device side.

---

## File List

```text
roce_host_loc/
├── CMakeLists.txt                          # host-side target ownership + add_subdirectory(device)
├── aicpu_ts_roce_host_loc_channel.{h,cc}   # host: channel main controller
├── roce_host_loc_connection.{h,cc}         # host: single-QP connection
└── device/
    ├── CMakeLists.txt                      # device-side target ownership
    ├── roce_host_loc_transport_lite_impl.{h,cc}
    ├── roce_host_loc_conn_lite.{h,cc}
    ├── roce_vendor_base_ops.h
    └── roce_vendor_1825_ops.h
```

| File | Main class / content | Side |
|------|----------------------|------|
| `aicpu_ts_roce_host_loc_channel.{h,cc}` | `hcomm::AicpuTsRoceHostLocChannel` (derives from `hcomm::Channel`) | host |
| `roce_host_loc_connection.{h,cc}` | `hcomm::RoceHostLocConnection` | host |
| `device/roce_host_loc_transport_lite_impl.{h,cc}` | `Hccl::RoceHostLocTransportLiteImpl` (derives from `Hccl::BaseTransportLiteImpl`) | device(kernel) |
| `device/roce_host_loc_conn_lite.{h,cc}` | `Hccl::RoceHostLocConnLite` | device(kernel) |
| `device/roce_vendor_base_ops.h` | `Hccl::RoceBaseOps`, `RdmaSqContextLite` / `RdmaCqContextLite`, DbVendor bit-field constants | header-only |
| `device/roce_vendor_1825_ops.h` | `Hccl::Roce1825Ops` (UB/1825 vendor implementation) | header-only |

Responsibilities:

- **AicpuTsRoceHostLocChannel**: channel lifecycle controller. `ParseInputParam` obtains the local endpoint/rdmaHandle; `StartListen`/`BuildSocket` set up the link; `CreateQp`/`ModifyQp`/`ExchangeData` complete QP creation and parameter exchange; `BuildNotify`/`BuildBuffer`/`BuildConnection` gather resources, and `H2DResPack`/`Serialize` pack them into a blob for the device; `GetStatus`/`ProcessStatus` maintain the `ChannelStatus` plus the internal `RdmaStatus` state machine.
- **RoceHostLocConnection**: the host-side view of a single QP connection; exchanges QP parameters with the peer (`ExchangeRdmaConnDto`) and serializes them into the blob.
- **RoceHostLocTransportLiteImpl**: the device-side data-plane entry. `Init` deserializes the blob (four categories: notify / loc buffer / rmt buffer / conn) and implements `Read`/`Write`/`WriteReduce`/`WriteWithNotify`/`WriteReduceWithNotify`/`Fence`/`Post`/`WaitWithTimeout`; it builds `RdmaDbSend` and `NotifyWait` tasks as needed and reports profiling data.
- **RoceHostLocConnLite**: the device-side connection object. It parses the SQ/CQ context, splits transfers by `RDMA_DMA_MAX_SIZE` (2 GB), and delegates WQE construction and doorbell generation to vendor ops; `PollCq` collects CQEs and builds the CQ doorbell.
- **RoceBaseOps**: vendor ops template base class. The common flow (`WaitSqFree` → vendor WQE construction → `CommitWqe` writing SQ → `UpdateSqPI` updating the software PI) lives in the base class; `BuildReadWqe`/`BuildWriteWqe`/`BuildWriteReduceWqe`/`PollCqImpl`/`BuildDoorbell`/`BuildCqDoorbell` are vendor extension points (returning `HCCL_E_NOT_SUPPORT` by default).

---

## Class Relationships

```text
hcomm::Channel                          (channels/channel.h)
 └── AicpuTsRoceHostLocChannel          ← this directory
          └─ has ──> hcomm::RoceHostLocConnection   ← this directory

hcomm::Channel
 └── AivRoceHostLocChannel              (channels/aiv/)
          └─ has ──> hcomm::RoceHostLocConnection   ← reuses the implementation in this directory

Hccl::BaseTransportLiteImpl              (legacy ascend950 unified_platform/resource/transport/aicpu/)
 └── Hccl::RoceHostLocTransportLiteImpl ← device/ of this directory; overrides 10+ virtual functions,
                                           held polymorphically via unique_ptr<BaseTransportLiteImpl>

Hccl::RoceHostLocConnLite               ← device/ of this directory (no longer inherits from RmaConnLite)
 └─ has ──> Hccl::RoceBaseOps
              └── Hccl::Roce1825Ops    ← this directory
```

Notes:

- The `BaseTransportLiteImpl` base class **remains in the legacy directory**; this directory provides one of its implementations, and the migration does not change that inheritance relationship.
- `RoceHostLocConnLite` formerly inherited from `RmaConnLite` in name only (it overrode only `Describe()`, and base-class members were written but never read). That inheritance was removed after the migration; `rma_conn_lite.h` is included only when type definitions such as `SqeConfigLite` are needed.
- `RdmaConnLite` in legacy `connection/aicpu/rdma_conn_lite.h` is a **stub class**, used only by the legacy `RmaConnLite::Create` factory dispatch (which is dead code). It has no inheritance relationship with `RoceHostLocConnLite` in this directory.

---

## Data Flow

```text
Host side
  Channel link setup (socket / QP create+modify) → exchange QP parameters
      → gather notify / buffer / conn → H2DResPack / Serialize → blob delivery

Device side
  RoceHostLocTransportLiteImpl::Init(blob)  extracts notify / buffer / conn
      → RoceHostLocConnLite(vector<char>)   parses dmaMode + SQ/CQ context
                        → RoceBaseOps (Roce1825Ops) builds WQEs
                            → CommitWqe writes SQ, UpdateSqPI writes software DB
      → upper-layer Read/Write/... generate RdmaDbSend / NotifyWait tasks and report profiling
```

---

## Build Targets

| Source file | Target |
|-------------|--------|
| `aicpu_ts_roce_host_loc_channel.cc`, `roce_host_loc_connection.cc` | `hcomm_base_obj` (host OBJECT, linked into `hcomm`) |
| `device/roce_host_loc_transport_lite_impl.cc`, `device/roce_host_loc_conn_lite.cc` | `ccl_kernel` (device build only) |
| `device/roce_vendor_*.h` | header-only, no registration required |

**Constraint**: device-side source files must be registered only to the device kernel target (`ccl_kernel`) and must not be added to any host-side target (`hcomm_base_obj`/`hcomm`/`hccl_v2`); otherwise the host package and the binary split would be polluted.

---

## Dependency Notes

The dependencies of this directory on the outside (legacy / other directories) **do not decrease because of the migration**. When new files are added, the include search paths must be maintained accordingly (for quoted includes, the directory of the including file is searched first, then each include directory in turn). The device subdirectory is **referenced by other directories with plain file names** (e.g. `coll_comm_aicpu.h` and `aicpu_channel_process.cc` include `roce_host_loc_transport_lite_impl.h`), so the host directory and the device directory must be registered separately:

| Location to maintain | Affected target |
|----------------------|-----------------|
| `src/base_comm/resources/endpoint_pairs/channels/aicpu/CMakeLists.txt` | `hcomm_base_obj` (host directory) |
| `src/legacy/ascend950/hccl_v2.cmake` | `hccl_v2` (host + device directories) |
| `src/legacy/ascend910/framework/ccl_kernel.cmake` | `ccl_kernel` (host + device directories) |
| `src/legacy/ascend910/framework/hcomm.cmake` | `hcomm` (host + device directories) |
| `test/ut/CMakeLists.txt` (`UT_COMMON_INCLUDE_LIST`) | UT common (host + device directories) |
| `test/ut/stub/CMakeLists.txt` | `hccl_llt` (host + device directories; source files must be registered as well) |

**Principle**: only add paths for this directory; never delete legacy path entries (this directory still depends on legacy base classes and resource types).

Main external dependencies:

- Host side: `Channel`/`AicpuTsChannelHelper`, `Endpoint`, `Hccl::Socket`, `Hccl::LocalRdmaRmaBuffer`, `Hccl::RdmaLocalNotify`, `ExchangeRdmaConnDto` (in the sibling `host/` directory), `SocketMgr`.
- Device side: `BaseTransportLiteImpl`, `RmaBufferLite`/`RmtRmaBufferLite`, `StreamLite`/`RtsqBase`, `NotifyLite`, `BinaryStream`; it also depends back on `base_comm`'s `cast_utils.h` and `dfx_profiling_handler_lite.h`.

---

## Naming Conventions

- Directory and file prefixes use `roce_host_loc_` uniformly; class names use the `RoceHostLoc*` prefix.
- Device-side logging macros use `[RoceHostLocConnLite::%s]` / `[RoceHostLocTransportLiteImpl::%s]` uniformly, which makes per-module log filtering easy.
- The channel's `SetModuleDataName` and `HcommChannelKind` value are `"AicpuTsRoceHostLocChannel"` and `AICPU_TS_ROCE_HOST_LOC` respectively.