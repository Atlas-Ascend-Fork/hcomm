# HcclCommSymWinRegister

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:39:57.035Z pushedAt=2026-10-12T06:12:35.575Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Registers service memory as a symmetric memory window, allowing HCCL to directly use this memory in supported collective communication operators.

Symmetric memory is a memory management model that allows parallel processing units (for example, each rank) to access each other's memory in a "globally visible" manner without explicit address exchange.

Currently, symmetric memory supports the following scenarios:

<!-- npu="950" id6 -->
- For Ascend 950PR&950DT products:

  - When the communication engine is AIV and the communication protocol is URMA: The user registers the allocated device memory as a symmetric memory window. This scenario must be used together with the [HcclTeamCreate](../comm_opdev/control_plane_api/comms_domain_resource_mgmt/HcclTeamCreate.md) API.

  - When the communication engine is AI CPU and the communication protocol is URMA: The user registers the allocated device memory as a symmetric memory window. In this scenario, [HcclCommSymWinRegister](HcclCommSymWinRegister.md) only completes the local symmetric memory window registration; cross-rank memory registration, **memHandle** exchange, and remote memory information update are completed when the related UB/URMA communication channel is created. When using collective communication APIs, this process is triggered internally by the collective communication operator. Before the remote memory information update is completed, [HcclSymWinGetRemoteAddr](HcclSymWinGetRemoteAddr.md) should not be called to obtain the remote address.

  - When the communication engine is AIV and the communication protocol is UB Memory: After allocating virtual memory and physical memory and completing the mapping, the user registers the virtual memory as symmetric memory. In this scenario, symmetric memory is implemented by reserving virtual addresses of the same size and the same layout in advance.
<!-- end id6 -->
<!-- npu="A3" id7 -->
- HCCS scenario for Atlas A3 products: After allocating virtual memory and physical memory and completing the mapping, the user registers the virtual memory as symmetric memory. In this scenario, symmetric memory is implemented by reserving virtual addresses of the same size and same layout in advance.
<!-- end id7 -->

<!-- npu="A3" id8 -->
The following figure shows the basic implementation model of symmetric memory in the HCCS scenario for Atlas A3 products.

![Symmetric memory implementation model](./figures/symmetric_memory.png)

- Allocate virtual memory for each rank. Assume that the virtual memory size corresponding to each rank is **heap_size** and the number of ranks in the communicator is **rank_size**. Then the total virtual memory size in the communicator is **heap_size**\***rank_size**.
- The virtual address layout of each rank is the same.
- The physical addresses of different ranks are mapped to the virtual addresses at the corresponding positions of each rank, enabling access to the memory of other ranks.
<!-- end id8 -->

The symmetric memory feature allows HCCL to directly operate on the memory passed in by the service without going through an intermediate buffer (HCCL buffer), thereby reducing memory copy overhead.

## Function Prototype

```c
HcclResult HcclCommSymWinRegister(HcclComm comm, void *addr, uint64_t size, HcclCommSymWindow *winHandle, uint32_t flag)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | HCCL communicator.<br>For the configuration description of this parameter under different product models, see [comm Description](#comm-description).|
| addr | Input | Start address of the symmetric memory window.<br>For the configuration description of this parameter under different product models, see [addr Description](#addr-description).|
| size | Input | Size of the symmetric memory window.<br>For the configuration description of this parameter under different product models, see [size Description](#size-description).|
| winHandle | Output | Pointer to the "symmetric memory window resource handle".<br>For the definition of the HcclCommSymWindow type, see [HcclCommSymWindow](./data_type_definition/HcclCommSymWindow.md). |
| flag | Input | Whether to enable symmetric memory. Currently, only **1** can be passed. |

### comm Description

<!-- npu="950" id13 -->
- In the URMA scenario for Ascend 950PR&950DT products, there is no need to configure the reserved symmetric memory size through **hcclSymWinMaxMemSizePerRank**.
<!-- end id13 -->
<!-- npu="A3" id14 -->
- In the HCCS scenario for Atlas A3 products, it is recommended to use the largest communicator within the SuperPoD, that is, the communicator covering the maximum number of devices. When initializing the communicator, you can set the symmetric memory size reserved for each rank through the **hcclSymWinMaxMemSizePerRank** parameter of [HcclCommConfig](./data_type_definition/HcclCommConfig.md). If **hcclSymWinMaxMemSizePerRank** is not set, the default value of 16 GB is used. The total virtual symmetric memory size reserved by the current communicator is: **rankSize** * **HcclCommConfig.hcclSymWinMaxMemSizePerRank**.
<!-- end id14 -->

### addr Description

<!-- npu="950" id15 -->
- In the URMA scenario for Ascend 950PR&950DT products, this address must be a device memory address that is a reserved virtual address with physical memory mapping completed. It is recommended to allocate it through the [HcommMemAlloc](HcommMemAlloc.md) API. The buffer must remain valid until it is deregistered by calling [HcclCommSymWinDeregister](HcclCommSymWinDeregister.md).
- In the UB Memory scenario for Ascend 950PR&950DT products, this address is a reserved virtual address with physical memory mapping completed.
<!-- end id15 -->
<!-- npu="A3" id16 -->
- In the HCCS scenario for Atlas A3 products, this address is a reserved virtual memory address. The virtual memory must be reserved by calling the **aclrtReserveMemAddress** API.
<!-- end id16 -->

### **size Description**

<!-- npu="950" id17 -->
- In the URMA scenario for Ascend 950PR&950DT products, **size** must be greater than 0, and the **size** input by all ranks when calling this API must be consistent.
- In the UB Memory scenario for Ascend 950PR&950DT products, **size** must be greater than 0, and `[addr, addr+size)` must be completely within the mapped physical memory range.
<!-- end id17 -->
<!-- npu="A3" id18 -->
- In the HCCS scenario of Atlas A3 products, 0 < size <= **HcclCommConfig.hcclSymWinMaxMemSizePerRank**, and **size** cannot exceed the size of the physical memory mapped to **addr** (that is, the device physical memory requested by calling the **aclrtMallocPhysical** API). Symmetric memory registration is aligned by the size of the physical memory, and the actually registered symmetric memory window size equals the size of the physical memory mapped to **addr**.
<!-- end id18 -->

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

<!-- npu="950" id11 -->
- For Ascend 950PR&950DT products:
  - When the communication engine is AIV, only the URMA scenario and the UB Memory scenario are supported.
  - When the communication engine is AI CPU, only the URMA scenario is supported.
  - In the URMA scenario where the communication engine is AI CPU, only the collective communication operators ReduceScatter, AllReduce, AllGather, Broadcast, AlltoAll, and AlltoAllVC are supported.
  - The URMA scenario does not require symmetric networking, and the **size** parameter input by all participating ranks must be consistent.
  - In the UB Memory scenario, you must ensure that all LSA WorldTeam members in the same communicator call this API simultaneously. The registration call order of each member and the physical memory size mapped by the input address of each registration must be consistent; otherwise, registration fails.
  - Registration in the UB Memory scenario involves collective operations among LSA WorldTeam members. If a member fails locally after the collective operation completes (for example, local mapping failure or insufficient resources), the UB Memory symmetric memory of this communicator enters an unavailable state, and subsequent registrations directly return an error. In this case, deregister the registered windows and destroy and recreate the communicator.
<!-- end id11 -->
<!-- npu="A3" id12 -->
- For Atlas A3 products:
  - Only the HCCS link communication scenario is supported.
  - Only symmetric networking is supported, that is, the scenario where each server has the same number of devices.
  - Only the scenario where AI servers within a SuperPoD use HCCS links for SDMA communication is supported. The scenario where RoCE is used for RDMA communication is not supported (that is, setting the environment variable **HCCL_INTER_HCCS_DISABLE** to **TRUE** is not supported; this environment variable is invalid in the single-server scenario).
  - Only collective communication operators AllGather, ReduceScatter, AllReduce, and AlltoAll are supported.
  - Only collective communication operators are supported.
  - Ensure that all ranks in the communicator call this registration API at the same time.
  - The physical memory sizes mapped by the input addresses of all ranks must be the same (symmetric memory registration is aligned by physical memory size).
<!-- end id12 -->
- When using the symmetric memory feature, the input and output memory of operators must be registered as symmetric memory by calling this API.
- The memory registered by calling this API must be deregistered by calling [HcclCommSymWinDeregister](HcclCommSymWinDeregister.md).
- This API does not support concurrent calls by multiple threads. The caller must ensure that calls to this API from different threads are executed serially.
- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, a wild pointer, or a destroyed handle.

## Example

<!-- npu="950" id9 -->
### URMA Scenario for Ascend 950PR&950DT Products

To query the window and offset based on the local address, see [HcclCommSymWinGet](HcclCommSymWinGet.md); to explicitly obtain the remote address in the URMA scenario, see [HcclSymWinGetRemoteAddr](HcclSymWinGetRemoteAddr.md).

```c
// Return value check macro.
#define HCCLCHECK(cmd) do { HcclResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define ACLCHECK(cmd) do { aclError ret = (cmd); if (ret != ACL_SUCCESS) { return (HcclResult)ret; } } while (0)

// Create and initialize the communicator configuration items.
HcclCommConfig config;
HcclCommConfigInit(&config);

// Obtain the communicator parameters.
uint32_t rankSize = 4;
uint32_t rankId = 0;
HcclRootInfo rootInfo;
HCCLCHECK(HcclGetRootInfo(&rootInfo));

// Initialize the collective communicator.
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, rankId, &config, &hcclComm));

// Create the task stream.
aclrtStream stream;
ACLCHECK(aclrtCreateStream(&stream));

size_t sendBytes = 1024;
size_t recvBytes = rankSize * sendBytes;
size_t memSize = sendBytes + recvBytes;

// Allocate the device memory.
void *devPtr = nullptr;
HCCLCHECK(static_cast<HcclResult>(HcommMemAlloc(&devPtr, memSize)));

HcclCommSymWindow symWin;
// Register the symmetric memory. The returned symWin identifies the window and is deregistered after use.
HCCLCHECK(HcclCommSymWinRegister(hcclComm, devPtr, memSize, &symWin, 1));

// Use the symmetric memory.
void *sendBuff = devPtr;
void *recvBuff = static_cast<char*>(sendBuff) + sendBytes;

// The collective communication operator automatically locates the corresponding symmetric memory window and the offset within the window through sendBuff and recvBuff,
// and uses the peer memory information internally, so users do not need to explicitly call HcclCommSymWinGet or HcclSymWinGetRemoteAddr.
HCCLCHECK(HcclAllGather(sendBuff, recvBuff, sendBytes, HCCL_DATA_TYPE_INT8, hcclComm, stream));

// Block and wait for the collective communication tasks in the task stream to complete.
ACLCHECK(aclrtSynchronizeStream(stream));

// Deregister the symmetric memory.
HCCLCHECK(HcclCommSymWinDeregister(symWin));

// Release the buffer.
HCCLCHECK(static_cast<HcclResult>(HcommMemFree(devPtr)));

// Destroy the task stream.
ACLCHECK(aclrtDestroyStream(stream));

// Destroy the communicator.
HCCLCHECK(HcclCommDestroy(hcclComm));
```
<!-- end id9 -->

<!-- npu="A3" id10 -->
### HCCS Scenario for Atlas A3 Products

```c
// Return value check macro.
#define HCCLCHECK(cmd) do { HcclResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define ACLCHECK(cmd) do { aclError ret = (cmd); if (ret != ACL_SUCCESS) { return (HcclResult)ret; } } while (0)

// Create and initialize the communicator configuration items.
HcclCommConfig config;
HcclCommConfigInit(&config);
// Modify the communicator configuration as needed.
config.hcclSymWinMaxMemSizePerRank = 10; // Unit: GB. Default value: 16. Set the total virtual symmetric memory size reserved for the current communicator = rankSize * config.hcclSymWinMaxMemSizePerRank;

// Obtain the communicator parameters.
uint32_t rankSize = 4;
uint32_t rankId = 0;
int32_t deviceId;
ACLCHECK(aclrtGetDevice(&deviceId));
HcclRootInfo rootInfo;
HCCLCHECK(HcclGetRootInfo(&rootInfo));

// Initialize the collective communicator.
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, rankId, &config, &hcclComm));

// Create the task stream.
aclrtStream stream;
ACLCHECK(aclrtCreateStream(&stream));

// Configure the physical memory attributes.
aclrtPhysicalMemProp prop;
prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
prop.memAttr = ACL_HBM_MEM_HUGE;
prop.location.id = deviceId;
prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
prop.reserve = 0;

// Obtain the alignment granularity, which is usually 2 MB.
size_t granularity;
ACLCHECK(aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity));

// Align size by granularity.
size_t size = 2 * 1024 * 1024;
size_t allocSize = (size + granularity - 1) / granularity * granularity;

// Reserve virtual memory.
void *virPtr;
ACLCHECK(aclrtReserveMemAddress(&virPtr, allocSize, 0, nullptr, 1));

// Allocate physical memory.
aclrtDrvMemHandle memHandle;
ACLCHECK(aclrtMallocPhysical(&memHandle, allocSize, &prop, 0));

// Establish the mapping from physical memory to virtual memory.
ACLCHECK(aclrtMapMem(virPtr, allocSize, 0, memHandle, 0));

size_t sendBytes = 1024;
size_t recvBytes = rankSize * sendBytes;
HcclCommSymWindow symWin;
// Register symmetric memory.
HCCLCHECK(HcclCommSymWinRegister(hcclComm, virPtr, sendBytes + recvBytes, &symWin, 1));

// Use symmetric memory.
void *sendBuff = virPtr;
void *recvBuff = static_cast<char*>(sendBuff) + sendBytes;

// Call the collective communication operator.
HCCLCHECK(HcclAllGather(sendBuff, recvBuff, sendBytes, HCCL_DATA_TYPE_INT8, hcclComm, stream));

// Block and wait for the collective communication tasks in the task stream to complete.
ACLCHECK(aclrtSynchronizeStream(stream));

// Deregister symmetric memory.
HCCLCHECK(HcclCommSymWinDeregister(symWin));

// Release the buffer.
ACLCHECK(aclrtUnmapMem(virPtr));
ACLCHECK(aclrtFreePhysical(memHandle));
ACLCHECK(aclrtReleaseMemAddress(virPtr));

// Destroy the task stream.
ACLCHECK(aclrtDestroyStream(stream));

// Destroy the communicator.
HCCLCHECK(HcclCommDestroy(hcclComm));
```
<!-- end id10 -->
