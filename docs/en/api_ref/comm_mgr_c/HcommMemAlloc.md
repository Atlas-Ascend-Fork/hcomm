# HcommMemAlloc

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:29:09.550Z pushedAt=2026-09-29T02:12:24.644Z -->

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

Allocates physical device memory and maps it to the virtual address space, completing virtual address reservation as well as physical memory allocation and mapping in a single step.

This API encapsulates the call flow of the following ACL runtime APIs to facilitate use by communication library developers:

- `aclrtReserveMemAddress`: Reserves a range in the process virtual address space.
- `aclrtMallocPhysical`: Allocates physical device memory.
- `aclrtMapMem`: Maps the physical memory to the reserved virtual address.

The allocated virtual address must be released using the [HcommMemFree](./HcommMemFree.md) API. Currently, this API mainly serves the symmetric memory registration scenario and is used together with [HcclCommSymWinRegister](./HcclCommSymWinRegister.md).

## Function Prototype

```c
HcommResult HcommMemAlloc(void **ptr, size_t size)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| ptr | Output | Address of the returned virtual address pointer. After the API is successfully called, `*ptr` points to the start of the virtual address to which the physical memory is mapped. |
| size | Input | Size of the memory to be allocated, in bytes. It must be greater than 0. Inside the API, the size is aligned upward to the device memory allocation granularity (usually 2 MB). |

## Return Value

[HcommResult](./data_type_definition/HcclResult.md): The API returns `HCCL_SUCCESS` on success, and other values on failure.

- `HCCL_E_PARA`: Invalid parameter (`ptr` is a null pointer or `size` is **0**).
- `HCCL_E_RUNTIME`: ACL runtime API call failure (failure to obtain the device, reserve the virtual address, allocate the physical memory, or perform the mapping).

## Constraints

- Before the call, set the current device through `aclrtSetDevice`. Inside the API, the current device ID is obtained through `aclrtGetDevice`.
- The attributes of the allocated physical memory are fixed as `ACL_HBM_MEM_HUGE`, `ACL_MEM_ALLOCATION_TYPE_PINNED`, and `ACL_MEM_HANDLE_TYPE_NONE`.
- `size` is aligned upward to the alignment granularity returned by `aclrtMemGetAllocationGranularity`. Therefore, the actual size of the allocated physical memory may be greater than `size`.
- The allocated virtual address must be released through [HcommMemFree](./HcommMemFree.md). It cannot be managed together with the `aclrtFree` or `aclrtMalloc` series APIs.
- Inside the API, when physical memory allocation or mapping fails, the reserved virtual address or the allocated physical memory is automatically rolled back.

## Example

The following example shows the process of using `HcommMemAlloc` to allocate memory and register it as a symmetric memory window. For more constraints on symmetric memory usage, see [HcclCommSymWinRegister](./HcclCommSymWinRegister.md).

```c
// Return value check macro.
#define HCCLCHECK(cmd) do { HcclResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define HCOMMCHECK(cmd) do { HcommResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define ACLCHECK(cmd) do { aclError ret = (cmd); if (ret != ACL_SUCCESS) { return (HcclResult)ret; } } while (0)

// Create and initialize the communicator configuration.
HcclCommConfig config;
HcclCommConfigInit(&config);
config.hcclSymWinMaxMemSizePerRank = 2; // Unit: GB. Reserve the VA size of symmetric memory for each rank.

// Obtain the communicator parameters.
uint32_t rankSize = 4;
uint32_t rankId = 0;
HcclRootInfo rootInfo;
HCCLCHECK(HcclGetRootInfo(&rootInfo));

// Initialize the collective communicator.
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, rankId, &config, &hcclComm));

// Create a task stream.
aclrtStream stream;
ACLCHECK(aclrtCreateStream(&stream));

// Allocate the device memory required for symmetric memory.
size_t sendBytes = 1024;
size_t recvBytes = rankSize * sendBytes;
size_t memSize = sendBytes + recvBytes;
void *devPtr = nullptr;
HCOMMCHECK(HcommMemAlloc(&devPtr, memSize));

// Register as a symmetric memory window.
HcclCommSymWindow symWin;
HCCLCHECK(HcclCommSymWinRegister(hcclComm, devPtr, memSize, &symWin, 1));

// Call the collective communication operator.
void *sendBuff = devPtr;
void *recvBuff = static_cast<char*>(devPtr) + sendBytes;
HCCLCHECK(HcclAllGather(sendBuff, recvBuff, sendBytes, HCCL_DATA_TYPE_INT8, hcclComm, stream));

// Block and wait for the collective communication tasks in the stream to complete.
ACLCHECK(aclrtSynchronizeStream(stream));

// Deregister the symmetric memory.
HCCLCHECK(HcclCommSymWinDeregister(symWin));

// Free the memory.
HCOMMCHECK(HcommMemFree(devPtr));

// Destroy the stream.
ACLCHECK(aclrtDestroyStream(stream));

// Destroy the communicator.
HCCLCHECK(HcclCommDestroy(hcclComm));
```
