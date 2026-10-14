# HcommMemFree

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:32:15.524Z pushedAt=2026-10-08T07:50:36.396Z -->

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

Releases the memory allocated through [HcommMemAlloc](./HcommMemAlloc.md), and sequentially completes unmapping, releasing physical memory, and releasing virtual address space.

This API encapsulates the call flow of the following ACL runtime APIs:

- `aclrtMemRetainAllocationHandle`: Looks up the physical memory handle based on the virtual address.
- `aclrtUnmapMem`: Unmaps the virtual address from the physical memory.
- `aclrtFreePhysical`: Releases the physical memory.
- `aclrtReleaseMemAddress`: Releases the reserved virtual address space.

## Function Prototype

```c
HcommResult HcommMemFree(void *ptr)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| ptr | Input | Virtual address pointer to be released, which must be the address returned by [HcommMemAlloc](./HcommMemAlloc.md). When `nullptr` is passed in, the API returns success directly without performing any operation. |

## Return Value

[HcommResult](./data_type_definition/HcclResult.md): The API returns `HCCL_SUCCESS` on success, and `HCCL_E_RUNTIME` if an ACL runtime API call fails (failure to look up the handle, perform unmapping, release the physical memory, or release the virtual address).

## Constraints

- `ptr` must be the virtual address returned by [HcommMemAlloc](./HcommMemAlloc.md). Memory addresses obtained through other means cannot be passed in.
- The API looks up the physical memory handle through `aclrtMemRetainAllocationHandle`. If `ptr` is an invalid value, a runtime error is triggered.
- After the memory is released, the memory pointed to by `ptr` must not be accessed again.
- Passing in `nullptr` is safe, and the API returns success directly.

## Example

The following example shows the process of allocating memory, registering it as a symmetric memory window, deregistering it, and releasing it. For more constraints on using symmetric memory, see [HcclCommSymWinRegister](./HcclCommSymWinRegister.md).

```c
// Return value check macro.
#define HCCLCHECK(cmd) do { HcclResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define HCOMMCHECK(cmd) do { HcommResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define ACLCHECK(cmd) do { aclError ret = (cmd); if (ret != ACL_SUCCESS) { return (HcclResult)ret; } } while (0)

// Create and initialize the communicator configuration item.
HcclCommConfig config;
HcclCommConfigInit(&config);
config.hcclSymWinMaxMemSizePerRank = 2; // Unit: GB.

// Obtain the communicator parameters.
uint32_t rankSize = 4;
uint32_t rankId = 0;
HcclRootInfo rootInfo;
HCCLCHECK(HcclGetRootInfo(&rootInfo));

// Initialize the collective communication communicator.
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, rankId, &config, &hcclComm));

// Create a task stream.
aclrtStream stream;
ACLCHECK(aclrtCreateStream(&stream));

// Allocate memory and register it as a symmetric memory window.
size_t sendBytes = 1024;
size_t recvBytes = rankSize * sendBytes;
size_t memSize = sendBytes + recvBytes;
void *devPtr = nullptr;
HCOMMCHECK(HcommMemAlloc(&devPtr, memSize));

HcclCommSymWindow symWin;
HCCLCHECK(HcclCommSymWinRegister(hcclComm, devPtr, memSize, &symWin, 1));

// Call the collective communication operator.
void *sendBuff = devPtr;
void *recvBuff = static_cast<char*>(devPtr) + sendBytes;
HCCLCHECK(HcclAllGather(sendBuff, recvBuff, sendBytes, HCCL_DATA_TYPE_INT8, hcclComm, stream));

// Block and wait until the collective communication tasks in the task stream are complete.
ACLCHECK(aclrtSynchronizeStream(stream));

// Deregister the symmetric memory.
HCCLCHECK(HcclCommSymWinDeregister(symWin));

// Release the memory allocated by HcommMemAlloc.
HCOMMCHECK(HcommMemFree(devPtr));

// Destroy the task stream.
ACLCHECK(aclrtDestroyStream(stream));

// Destroy the communicator.
HCCLCHECK(HcclCommDestroy(hcclComm));
```
