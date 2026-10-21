# CheckerL2 Third-Party Header Dependency List

> Based on the `CheckerL2` version, this document lists all third-party header dependency relationships.

## Description

This document lists all third-party headers actually referenced through `#include` in the project. It marks their source (CANN / HCCL / HCOMM) and actual path on disk.

**Path conventions**:
- CANN root directory: `/home/teamserver/workspace/Ascend/cann-9.1.0/` (abbreviated as `CANN/` below)
- HCOMM root directory: `/home/teamserver/workspace/hcomm/` (abbreviated as `HCOMM/` below)
- HCCL root directory: `/home/teamserver/workspace/hccl/` (abbreviated as `HCCL/` below)
- Project root directory: `CheckerL2/` (abbreviated as `Project/` below)

**Resolution rule**: The declaration order of `target_include_directories` in CMake determines search priority. The first match found is used.

---

## I. Headers from CANN

### 1.1 ACL Runtime Interfaces (`CANN/include/acl/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| `acl/acl_base.h` | `CANN/include/acl/acl_base.h` | All aclrt_*.cc under proxy, hccl_comm_stub.cc, hccp_stub.cc, hccp_ra_socket_stub.cc, ascend_hal_stub.cc, aclrt_exec_control.cc; include/runnerdb/db_sim_runner_common.h; test/proxy/ |
| `acl/acl_rt.h` | `CANN/include/acl/acl_rt.h` | All aclrt_*.cc under proxy, hccl_op_stub.cc; test/proxy/ |

### 1.2 HCCL Public Interfaces (`CANN/include/hccl/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| `hccl/hccl_types.h` | `CANN/include/hccl/hccl_types.h` | proxy/aclrt_kernel_stub.cc, hccl_op_stub.cc, hccl_proxy_common.h; plugin/runner/ccu_executor/ccu_fp16.h; common/sim_data_dump.cc; device_arm/level2/proxy/device_sqe_parse_stub.h; test/proxy/ |
| `hccl/hccl.h` | `CANN/include/hccl/hccl.h` | **proxy/hccl_op_stub.cc**; test/proxy/hccl_comm_stub_test.cc |
| `hccl/hcom.h` | `CANN/include/hccl/hcom.h` | proxy/hccl_inner_stub.cc; test/proxy/hccl_inner_stub_test.cc |
| `hccl/base.h` | `CANN/include/hccl/base.h` | proxy/aclrt_context_stub.cc; plugin/runner/ccu_executor/ccu_fp16.h; plugin/checker/header/external/task_param.h; include/sim_ip_address.h |
| `hccl/dtype_common.h` | `CANN/include/hccl/dtype_common.h` | proxy/scatter_stub.cc; test/proxy/scatter_stub_test.cc |
| `hccl/hccl_common.h` | `CANN/x86_64-linux/asc/include/adv_api/hccl/hccl_common.h` | test/proxy/hccl_proxy_common_test.cc |

### 1.3 HCCL Internal Interfaces (`CANN/pkg_inc/hccl/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| `"dtype_common.h"` | `CANN/pkg_inc/hccl/dtype_common.h` | proxy/aclrt_new_stub.cc, hccp_ccu_stub.cc, hccp_stub.cc, hccl_op_stub.cc, aclrt_device_stub.cc, hccl_proxy_common.cc; plugin/runner/runner_utils/storage_manager.h; plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_all_rank_param_recorder_v3.h; plugin/checker/src/framework/task_graph_generator/ccu_all_rank_param_recorder.h; plugin/checker/src/utils/storage_manager.h |

### 1.4 CANN Runtime Interfaces (`CANN/pkg_inc/runtime/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| `"runtime/base.h"` | `CANN/pkg_inc/runtime/runtime/base.h` | proxy/hccl_comm_stub.cc, aclrt_runtime_config.cc, aclrt_notify_stub.cc, hccp_stub.cc, aclrt_device_stub.cc, hccp_ra_socket_stub.cc, aclrt_stream_stub.cc, aclrt_stub.cc; test/proxy/ |
| `"runtime/event.h"` | `CANN/pkg_inc/runtime/runtime/event.h` | proxy/aclrt_runtime_config.cc, aclrt_notify_stub.cc |
| `"runtime/rt.h"` | `CANN/pkg_inc/runtime/runtime/rt.h` | device_arm/level2/proxy/device_sqe_parse_stub.h |

### 1.5 CANN Profiling Interfaces (`CANN/pkg_inc/profiling/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| `"aprof_pub.h"` | `CANN/pkg_inc/profiling/aprof_pub.h` | proxy/aprofiling_stub.cc; device_arm/level2/proxy/aprofiling_stub.cc; test/proxy/aprofiling_stub_test.cc |

### 1.6 CANN Trace Interfaces (`CANN/pkg_inc/trace/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| `"atrace_pub.h"` | `CANN/pkg_inc/trace/atrace_pub.h` | proxy/adapter_rts_stub.cc |
| `"atrace_types.h"` | `CANN/pkg_inc/trace/atrace_types.h` | proxy/aclrt_new_stub.cc, hccp_stub.cc |

### 1.7 CANN Driver Interfaces (`CANN/include/driver/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| `"ascend_hal.h"` | `CANN/include/driver/ascend_hal.h` | proxy/adapter_rts_stub.cc, aclrt_new_stub.cc, ascend_hal_stub.cc; device_arm/level2/proxy/device_sqe_parse_stub.h, ascend_hal_stub.cc; test/device_arm/ |

### 1.8 CANN Base Interfaces (`CANN/pkg_inc/base/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| `"dlog_pub.h"` | `CANN/pkg_inc/base/dlog_pub.h` | plugin/checker/header/external/log.h (as fallback) |
| `"log_types.h"` | `CANN/pkg_inc/base/log_types.h` | plugin/checker/header/external/dlog_pub.h |

### 1.9 CANN CCU Microcode Interfaces (`CANN/pkg_inc/hcomm/ccu/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| `"ccu_microcode_v1.h"` | `CANN/pkg_inc/hcomm/ccu/ccu_microcode_v1.h` | proxy/hccp_ccu_stub.cc; plugin/runner/ccu_executor/ccu_resource_manager.cc, ccu_resource_common.h; plugin/checker/header/external/ccu_rep_base.h, ccu_instr_info.h; plugin/checker/header/internal/binary_data_type_pub.h; plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/ multiple files; plugin/checker/src/framework/task_graph_generator/ multiple files; include/sim_binary_data_type_pub.h |
| `"ccu_common.h"` | `CANN/pkg_inc/hcomm/ccu/ccu_common.h` | proxy/hccp_ccu_stub.cc |

### 1.10 CANN RTS Device Interfaces (`CANN/pkg_inc/runtime/runtime/rts/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| `"rts_device.h"` | `CANN/pkg_inc/runtime/runtime/rts/rts_device.h` | proxy/hccp_stub.cc, adapter_rts_stub.cc, aclrt_new_stub.cc |

### 1.11 CANN Runtime Other Interfaces (`CANN/pkg_inc/runtime/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| `"mem.h"` | `CANN/pkg_inc/runtime/mem.h` | proxy/hccp_stub.cc, aclrt_new_stub.cc |
| `"rt_external_kernel.h"` | `CANN/pkg_inc/runtime/rt_external_kernel.h` | proxy/hccp_ccu_stub.cc |

---

## II. Headers from HCOMM

### 2.1 HCCP Network Interfaces (`HCOMM/src/base_comm/resources/hccp/inc/network/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| `"hccp.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp.h` | proxy/adapter_rts_stub.cc |
| `"hccp_async.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp_async.h` | proxy/adapter_rts_stub.cc |
| `"hccp_ping.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp_ping.h` | proxy/adapter_rts_stub.cc |
| `"hccp_tlv.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp_tlv.h` | proxy/adapter_rts_stub.cc, hccp_stub.cc, **hccp_ccu_stub.cc** |
| `"hccp_ctx.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp_ctx.h` | proxy/hccp_ccu_stub.cc, hccp_stub.cc |
| `"hccp_common.h"` | `HCOMM/src/base_comm/resources/hccp/inc/network/hccp_common.h` | proxy/hccl_comm_stub.cc, aclrt_new_stub.cc, hccp_ccu_stub.cc, hccp_stub.cc, hccp_ra_socket_stub.cc |

### 2.2 HCCP CCU External Dependencies (`HCOMM/src/base_comm/resources/hccp/external_depends/ccu/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| `"ccu_u_comm.h"` | `HCOMM/src/base_comm/resources/hccp/external_depends/ccu/ccu_u_comm.h` | proxy/hccp_ccu_stub.cc |

---

## III. Headers from HCCL (ARM Cross-Compilation Only)

### 3.1 Operator Common Interfaces (`HCCL/src/ops/op_common/inc/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| `"alg_param.h"` | `HCCL/src/ops/op_common/inc/alg_param.h` | device_arm/hccl_kernel_executor.cc; test/device_arm/ |

### 3.2 HCCL Common Code (`HCCL/src/common/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| (multiple headers exposed through include path) | `HCCL/src/common/*.h` | Only the device_arm build module references indirectly through include path |

### 3.3 Dynamic Symbol Loading (`HCCL/src/common/hcomm_dlsym/` and `ccu/`)

| #include | Actual File Path | Referenced In |
|----------|-----------------|---------------|
| (multiple headers exposed through include path) | `HCCL/src/common/hcomm_dlsym/*.h`, `HCCL/src/common/hcomm_dlsym/ccu/*.h`, `*.hpp` | Only the device_arm build module references indirectly through include path |

---

## IV. Project Local External Headers (Checker-Specific, HCCL Interface Definition Copies)

Path: `Project/src/plugin/checker/header/external/`

When a header with the same name does not exist in the CANN package, these local copies serve as fallback.

| File Name | Description | Referenced In |
|-----------|-------------|---------------|
| `hccl_types.h` | HcclResult enum (24 error codes including HCCL_SUCCESS) | 50+ files under plugin/checker/ through `#include "hccl_types.h"`; include/sim_binary_data_type_pub.h, sim_loader.h, sim_ccu_jetty_ctx.h, sim_ccu_channel_ctx.h, and so on |
| `base.h` | Basic type definitions | 19 references under plugin/checker/ (framework, semantics_check, checker, and so on); header/external/log.h |
| `data_type.h` | Data type definitions | 3 references under plugin/checker/src/framework/task_graph_generator_v3/, 4 references under task_graph_generator/ |
| `ccu_instr_info.h` | CCU instruction information | 8 references under plugin/checker/ (task_ccu.h, task_utils.cc, task_graph_generator, and so on) |
| `ccu_task_param.h` | CCU task parameters | plugin/checker/src/utils/task_ccu.h |
| `ccu_rep_base.h` | CCU Reporter base | File exists. No direct `#include` reference currently |
| `ccu_rep_type.h` | CCU Reporter types | File exists. No direct `#include` reference currently |
| `dlog_pub.h` | Log public interface (CANN fallback) | plugin/checker/header/external/log.h |
| `enum_factory.h` | Enum factory | 5 internal references under plugin/checker/ (sim_task.h, check_utils.h, data_slice.h, data_type.h, task_param.h) |
| `const_val.h` | Constant value definitions | plugin/checker/header/external/task_param.h |
| `log.h` | Log interface | Internal to plugin/checker/ |
| `log_types.h` | Log types (CANN fallback) | plugin/checker/header/external/dlog_pub.h |
| `string_util.h` | String utilities | plugin/checker/header/external/enum_factory.h; header/internal/task_def.h; src/utils/check_utils.h |
| `task_param.h` | Task parameters | File exists. No direct `#include` reference currently (included internally by other headers) |

---

## V. AIV Dependencies (AIV Operator Simulation Only)

This section describes dependencies related to AIV (AI Vector Core) operator simulation in the tool. These dependencies originate from the HCOMM repository and AscendC programming interfaces. To remove the strong dependency on external repositories, the tool maintains backup copies of key definitions internally.

### 5.1 AivOpArgs Data Structure (Project Internal Backup)

**Original source**: `hccl_aiv_utils.h` header from the HCOMM repository

**Backup location**: `src/proxy/level2/aclrt_kernel_stub.cc` lines 597-628

**Description**: `AivOpArgs` is the parameter structure for AIV collective communication operators. It passes AIV kernel execution parameters between the host side and the device side. The tool originally loaded the `ops_hccl::ExecuteKernelLaunch()` function from the HCOMM library dynamically through `dlsym` (symbol name: `_ZN8ops_hccl19ExecuteKernelLaunchERKNS_9AivOpArgsE`). That function accepts this structure as a parameter. To remove the dependency on the HCOMM runtime library, the tool maintains a complete backup of the structure internally.

**Structure definition**:
```cpp
struct AivOpArgs {
    HcclCMDType cmdType = HcclCMDType::HCCL_CMD_MAX;   // Collective communication command type
    std::string comm = {};                              // Communicator identifier
    HcclComm hcclComm = nullptr;                        // HCCL communicator handle
    uint32_t numBlocks = AIV_STUB_MAX_NUM_BLOCKS;       // AIV core count
    void *stream = nullptr;                             // Execution stream
    uint64_t beginTime = 0;                             // Start timestamp
    OpCounterInfo counter = {};                         // Counter information
    void *buffersIn = nullptr;                          // Input buffer address
    uint64_t input = 0;                                 // Input offset
    uint64_t output = 0;                                // Output offset
    uint32_t rank = 0;                                  // Current rank
    uint32_t sendRecvRemoteRank = 0;                    // SendRecv remote rank
    uint32_t rankSize = 0;                              // Total rank count
    uint64_t xRankSize = 0, yRankSize = 0, zRankSize = 0;  // Topology dimensions
    uint64_t count = 0;                                 // Element count
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // Data type
    HcclReduceOp op = HcclReduceOp::HCCL_REDUCE_SUM;   // Reduce operation type
    uint32_t root = 0;                                  // Root rank
    uint32_t sliceId = 0;                               // Slice ID
    uint64_t inputSliceStride = 0;                      // Input slice stride
    uint64_t outputSliceStride = 0;                     // Output slice stride
    uint64_t repeatNum = 0;                             // Repeat count
    uint64_t inputRepeatStride = 0;                     // Input repeat stride
    uint64_t outputRepeatStride = 0;                    // Output repeat stride
    bool isOpBase = false;                              // Whether this is a base operator
    ExtraArgs extraArgs = {};                           // Extra parameters (AllToAllV, and so on)
    uint64_t topo_[AIV_STUB_TOPO_LEN] = {0};            // Topology information
    KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER;  // Parameter type
};
```

**Dependency risk**: If the `AivOpArgs` structure on the HCOMM side changes (field additions/removals, order adjustments, type changes), the `dlsym` call may produce misaligned parameters, causing runtime errors.

### 5.2 aiv_communication_base_v2.h Header (Project Internal Backup)

**Original source**: Header with the same name from the HCOMM repository

**Backup location**: `src/proxy/level2/aiv_kernel/hccl_op_stub/aiv_communication_base_v2.h`

**Description**: This header defines the AIV collective communication operator base class `AivCommBase` and related kernel parameter macro definitions (`KERNEL_ARGS_DEF`, `KERNEL_ARGS_CALL`, and so on). The tool originally included this file from the HCOMM repository directly. To remove the dependency, the tool maintains a complete backup internally.

**Key definitions contained in the file**:

| Definition | Description |
|------------|-------------|
| `AivSuperKernelArgs` | SuperKernel parameter structure |
| `ExtraArgs` | Extra parameter structure (sendCounts/recvCounts, and so on, for AllToAllV) |
| `KERNEL_ARGS_DEF` / `KERNEL_ARGS_CALL` | AIV kernel parameter macro definitions (approximately 25 parameters) |
| `SUPERKERNEL_ARGS_DEF` / `SUPERKERNEL_ARGS_CALL` | SuperKernel parameter macro definitions |
| `AivCommBase` class | AIV communication base class. Encapsulates core methods such as Barrier, Record/WaitFlag, CpGM2GM |
| `EXPORT_AIV_META_INFO` | AIV kernel metadata export macro |
| `AIV_ATOMIC_DATA_TYPE_DEF` / `AIV_COPY_DATA_TYPE_DEF` | Supported data type macro definitions |

**Dependency risk**: If the AIV communication interface or kernel parameters on the HCOMM side change, this backup file must be updated accordingly.

### 5.3 AIV Operator Op Headers (HCOMM / HCCL Dependencies)

Each AIV kernel implementation file for collective communication operators in the tool (`src/proxy/level2/aiv_kernel/hccl_op_stub/*/aiv_communication_v2.cc`) references the corresponding operator op header through `#include`. These headers **do not exist within the project** and come from the HCOMM and HCCL repositories respectively:

| #include | Source | Actual File Path | Referenced In |
|----------|--------|-----------------|---------------|
| `"aiv_all_reduce_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/all_reduce/` | hccl_op_stub/all_reduce/aiv_communication_v2.cc |
| `"aiv_all_gather_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/all_gather/` | hccl_op_stub/all_gather/aiv_communication_v2.cc |
| `"aiv_reduce_scatter_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/reduce_scatter/` | hccl_op_stub/reduce_scatter/aiv_communication_v2.cc |
| `"aiv_broadcast_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/broadcast/` | hccl_op_stub/broadcast/aiv_communication_v2.cc |
| `"aiv_all_to_all_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/all_to_all/` | hccl_op_stub/all_to_all_v/aiv_communication_v2.cc |
| `"aiv_all_to_all_v_op.h"` | HCOMM | `hcomm/src/legacy/ascend910/algorithm/base/alg_aiv_template/all_to_all/` | hccl_op_stub/all_to_all_v/aiv_communication_v2.cc |
| `"aiv_scatter_op.h"` | HCCL | `hccl/src/ops/scatter/template/aiv/kernel/` | hccl_op_stub/scatter/aiv_communication_v2.cc |
| `"aiv_reduce_op.h"` | HCCL | `hccl/src/ops/reduce/template/aiv/kernel/` | hccl_op_stub/reduce/aiv_communication_v2.cc |

**Dependency risk**: If the operator op header interface signatures, template parameters, or member functions on the HCOMM or HCCL side change, the AIV kernel compilation in the tool will fail or produce inconsistent runtime behavior.

### 5.4 AscendC Interface Dependencies

The tool depends on AscendC programming interfaces provided by the CANN package to simulate AIV kernel execution. All AscendC interfaces in the project are implemented through local stub files under `src/proxy/level2/aiv_kernel/ascendc_stub/` rather than using the AscendC headers from the CANN SDK directly.

**AscendC interfaces used** (implemented through stubs):

| Stub Header | Wrapped AscendC Interfaces | Referenced In |
|-------------|---------------------------|---------------|
| `ascendc_base_stub.h` | `GlobalTensor`, `LocalTensor`, `TPipe`, `TBuf`, `TQueBind`, `pipe_barrier`, `GetBlockIdx`, and other base classes | Almost all files under aiv_kernel/ |
| `ascendc_copy_stub.h` | `DataCopy`, `DataCopyPad`, and other data transfer interfaces | ascendc_stub/kernel_operator.h |
| `ascendc_math_stub.h` | `Add`, `Sub`, `Mul`, `Reduce`, and other math operation interfaces | ascendc_stub/kernel_operator.h |
| `ascendc_memory_stub.h` | `SetAtomicAdd`, `SetAtomicMax`, `SetAtomicMin`, and other atomic operation interfaces | ascendc_stub/kernel_operator.h |
| `ascendc_sync_stub.h` | `send_flag`, `recv_flag`, `SyncAll`, and other synchronization interfaces | ascendc_stub/kernel_operator.h |
| `ascendc_utils_stub.h` | `PipeBarrier` and other utility functions | ascendc_stub/kernel_operator.h |

**Dependency risks**:
1. AscendC interface API changes (class member function signature changes, template parameter adjustments) may cause stub implementations to mismatch real interfaces
2. New AscendC data types (for example `fp8_e4m3fn_t`, `fp8_e5m2_t`, `hifloat8_t`) require corresponding stub updates
3. The `AivCommBase` class uses AscendC template classes extensively (`GlobalTensor<T>`, `LocalTensor<T>`, `TQueBind`, and so on). Semantic changes to these types affect AIV simulation correctness

---

## VI. AICPU Dependencies (AICPU Mode Only)

This section describes the dependency situation when the tool simulates host-device interaction in AICPU mode. This mode is implemented through the `ExecuteAicpuKernel` function in `src/device_arm/hccl_kernel_executor.cc`. It simulates the AICPU kernel dispatch to the device side, involving serialization and address conversion of multiple data structures.

### 6.1 AICPU Interaction Data Structures

The tool maintains local backups of the following data structures in `src/device_arm/common/aicpu_args_stub.h` to remove the strong dependency on the HCOMM runtime library:

| Data Structure | Purpose | Corresponding Kernel Function |
|---------------|---------|------------------------------|
| `CommAicpuParam` | Communicator initialization parameters (hcomId, device ID, H2D/D2H transfer parameters) | `RunAicpuIndOpCommInit` |
| `HDCommunicateParams` | H2D/D2H control transfer parameters (deviceAddr, readCacheAddr) | Referenced by `CommAicpuParam` |
| `ThreadMgrAicpuParam` | Thread management parameters (threadNum, serialized threadParam array, deviceHandle) | `RunAicpuIndOpThreadInit`, `RunAicpuThreadSupplementNotify` |
| `AicpuTsThread` | AICPU thread information (streamType, notifyLoadType, devId) | Referenced by `ThreadMgrAicpuParam` |
| `InitTask` | Channel initialization task (context pointer, isCustom flag) | `RunAicpuIndOpChannelInitV2` |
| `HcclChannelUrmaRes` | Channel URMA resource description (channelList, uniqueIdAddr, remoteRankList) | `RunAicpuIndOpChannelInitV2` |
| `UniqueIdV2Header` | UniqueId V2 header (type, notifyNum, bufferNum, connNum) | Channel link establishment serialization data parsing |
| `ConnUniqueBlock` | Connection unique block (flexible array, contains `ConnUniqueIds`) | Channel link establishment serialization data parsing |

### 6.2 OpParam Data Structure (External Dependency)

**Source**: `alg_param.h` header from the HCCL repository (`HCCL/src/ops/op_common/inc/`), in the `ops_hccl` namespace

**Description**: `OpParam` is the generic parameter structure for AICPU collective communication operators. It is used by the `HcclLaunchAicpuKernel` function. Unlike the locally backed-up structures above, the tool **uses the `OpParam` definition from HCCL source code directly** without a local backup.

**Dependency risk**: If the `OpParam` structure on the HCCL side changes (field additions/removals, type changes), the tool may parse parameters incorrectly or crash at runtime.

### 6.3 AICPU Kernel Function Dependencies

The tool dynamically loads the following kernel functions from the HCOMM runtime library through `dlopen` + `dlsym` to simulate actual AICPU-side execution:

| dlsym Symbol Name | Library | Function |
|-------------------|---------|----------|
| `RunAicpuIndOpCommInit` | `libccl_kernel.so` | AICPU communicator initialization |
| `RunAicpuIndOpThreadInit` | `libccl_kernel.so` | AICPU thread initialization |
| `RunAicpuIndOpChannelInitV2` | `libccl_kernel.so` | AICPU channel initialization V2 |
| `RunAicpuDfxOpInfoInitV2` | `libccl_kernel.so` | AICPU DFX operator information initialization V2 |
| `RunAicpuThreadSupplementNotify` | `libccl_kernel.so` | AICPU resource supplement notification |
| `HcclLaunchAicpuKernel` | `libscatter_aicpu_kernel.so` | AICPU collective communication kernel launch |

**Dependency risks**:
1. If the signatures or behavior of the above kernel functions on the HCOMM side change, the `dlsym` calls in the tool may fail or produce incorrect results
2. If `libccl_kernel.so` or `libscatter_aicpu_kernel.so` adds or removes kernel functions, the tool must update the function pointer list accordingly
3. If locally backed-up structures such as `CommAicpuParam`, `ThreadMgrAicpuParam`, `HcclChannelUrmaRes` become inconsistent with the actual definitions on the HCOMM side, address conversion errors and memory out-of-bounds access will occur

### 6.4 Kernel SO Library Name Dependencies

The tool hardcodes the following SO library names in the `InitKernelFuncHandle` function to load AICPU kernel execution entries:

| Hardcoded SO Name | Loaded dlsym Symbols | Description |
|-------------------|---------------------|-------------|
| `libccl_kernel.so` | `RunAicpuIndOpCommInit`, `RunAicpuIndOpThreadInit`, `RunAicpuIndOpChannelInitV2`, `RunAicpuDfxOpInfoInitV2`, `RunAicpuThreadSupplementNotify` | HCOMM communication framework core library |
| `libscatter_aicpu_kernel.so` | `HcclLaunchAicpuKernel` | AICPU collective communication operator kernel library |
| `libslog.so` | (log library dependency) | CANN security log library |
| `libc_sec.so` | (security library dependency) | CANN security function library |

**Dependency risks**:
1. **SO name change**: If the HCOMM side renames or splits the above SOs (for example renaming `libccl_kernel.so`), the `dlopen` call in the tool will fail, causing all AICPU kernels to be unable to execute
2. **Operator kernel SO name ambiguity**: Currently, regardless of the operator type (AllReduce, AllGather, ReduceScatter, and so on), the operator kernel SO name is fixed as `libscatter_aicpu_kernel.so`. This naming may cause ambiguity — the "scatter" in the name implies it applies only to Scatter operators, but it actually hosts kernels for all collective communication operators. If the HCOMM side later splits SOs by operator type (for example `liballreduce_aicpu_kernel.so`), the tool will not be able to adapt
3. **User-defined operator scenario**: The HCCL business layer does not strictly define SO names for user-defined operators. Users may compile private SOs (for example `libcustom_alltoall_kernel.so`). The tool currently only recognizes the fixed SO names above and will not be able to load or simulate user-defined operator private SOs

---

## VII. Dependency Overview

| Source | Header Count | Modules Used |
|--------|-------------|-------------|
| **CANN** | 23 distinct headers | Almost all modules (proxy, checker, runner, common, store, topo, utils, device_arm) |
| **HCOMM** | 13 distinct headers | proxy module (hccp network interfaces + AIV operator op) |
| **HCCL** | 3 direct references + indirect references | device_arm module (ARM cross-compilation) + proxy (AIV scatter/reduce op) |
| **Project local external** | 14 headers | checker module only (as fallback when CANN is unavailable) |
| **AIV backup** | 2 key definitions | proxy/aiv_kernel module only (AIV operator simulation) |
| **AIV operator Op** | 8 headers (6 HCOMM + 2 HCCL) | proxy/aiv_kernel module only (AIV operator simulation) |
| **AscendC interfaces** | 6 stub headers | proxy/aiv_kernel module only (AIV operator simulation) |
| **AICPU backup** | 8 local structures + 6 kernel functions + 4 SO library names | device_arm module only (AICPU mode host-device interaction) |
