# HcclCommSymWinGet

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:39:49.907Z pushedAt=2026-10-12T06:12:35.575Z -->

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

Based on the address pointer of the registered symmetric memory, returns the corresponding window resource handle and its offset within the window.

<!-- npu="950" id6 -->
- For Ascend 950PR&950DT products, this API supports the URMA and UB Memory scenarios. When the query range does not hit a registered valid symmetric memory window, the API returns **HCCL_SUCCESS** and sets ***winHandle** to **NULL** and ***offset** to **0**.
<!-- end id6 -->
<!-- npu="A3" id7 -->
- For Atlas A3 products, this API supports the HCCS link communication scenario.
<!-- end id7 -->

## Function Prototype

```c
HcclResult HcclCommSymWinGet(HcclComm comm, void *ptr, size_t size, HcclCommSymWindow *winHandle, size_t *offset)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | HCCL communicator.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| ptr | Input | Address pointer of the registered symmetric memory. The memory must have been registered using the [HcclCommSymWinRegister](HcclCommSymWinRegister.md) API. This address is a reserved virtual address for which physical memory mapping has been completed, and it can be allocated using the [HcommMemAlloc](HcommMemAlloc.md) API. |
| size | Input | Size of the symmetric memory window.<br>Assume that the symmetric memory window size is **symSize** and the address pointer of the registered symmetric memory is **addr**. **size** must meet the following conditions:<br>  - size > 0<br>  - ptr+size <= addr + symSize |
| winHandle | Output | Pointer to the symmetric memory window resource handle. |
| offset | Output | Pointer to the offset.<br>Assume that the address pointer of the registered symmetric memory is **addr**. Then \*offset = ptr - addr. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

<!-- npu="950" id8 -->
- For Ascend 950PR&950DT products, the URMA and UB Memory scenarios are supported. When the query range hits a window, **ptr** and **ptr+size** must be entirely within the same valid symmetric memory window. On a miss, **HCCL_SUCCESS** is returned, and ***winHandle** is set to **NULL** and ***offset** is set to **0**.
<!-- end id8 -->
<!-- npu="A3" id9 -->
- For Atlas A3 products, only the HCCS link communication scenario is supported.
<!-- end id9 -->
- Only the scenario where the communication operator expansion mode is AI CPU is supported.
- This API does not support concurrent calls by multiple threads. The caller must ensure that calls to this API from different threads are executed serially.
- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, a wild pointer, or a destroyed handle.

## Example

<!-- npu="950" id10 -->
### URMA Scenario for Ascend 950PR&950DT Products

```c
// Return value check macro.
#define HCCLCHECK(cmd) do { HcclResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define ACLCHECK(cmd) do { aclError ret = (cmd); if (ret != ACL_SUCCESS) { return (HcclResult)ret; } } while (0)

// Create and initialize the communicator configuration item.
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

size_t memSize = 2 * 1024 * 1024;

// Allocate device memory.
void *devPtr = nullptr;
HCCLCHECK(static_cast<HcclResult>(HcommMemAlloc(&devPtr, memSize)));

HcclCommSymWindow symWin;
// Register the symmetric memory.
HCCLCHECK(HcclCommSymWinRegister(hcclComm, devPtr, memSize, &symWin, 1));

// Use HcclCommSymWinGet to obtain the symmetric memory resource.
HcclCommSymWindow tempWin;
size_t offset = 0;
HCCLCHECK(HcclCommSymWinGet(hcclComm, devPtr, memSize, &tempWin, &offset));

// Deregister the symmetric memory.
HCCLCHECK(HcclCommSymWinDeregister(symWin));

// Release the memory.
HCCLCHECK(static_cast<HcclResult>(HcommMemFree(devPtr)));

// Destroy the communicator.
HCCLCHECK(HcclCommDestroy(hcclComm));
```
<!-- end id10 -->

<!-- npu="A3" id11 -->
### HCCS Scenario for Atlas A3 Products

```c
// Return value check macro.
#define HCCLCHECK(cmd) do { HcclResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define ACLCHECK(cmd) do { aclError ret = (cmd); if (ret != ACL_SUCCESS) { return (HcclResult)ret; } } while (0)

// Create and initialize the communicator configuration item.
HcclCommConfig config;
HcclCommConfigInit(&config);
// Modify the communicator configuration as needed.
config.hcclSymWinMaxMemSizePerRank = 10; // Unit: GB. Default value: 16.

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

// Physical memory attribute configuration.
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

HcclCommSymWindow symWin;
// Register symmetric memory.
HCCLCHECK(HcclCommSymWinRegister(hcclComm, virPtr, allocSize, &symWin, 1));

// Use HcclCommSymWinGet to obtain symmetric memory resources.
HcclCommSymWindow tempWin;
size_t offset = 0;
HCCLCHECK(HcclCommSymWinGet(hcclComm, virPtr, allocSize, &tempWin, &offset));

// Deregister symmetric memory.
HCCLCHECK(HcclCommSymWinDeregister(symWin));

// Release memory.
ACLCHECK(aclrtUnmapMem(virPtr));
ACLCHECK(aclrtFreePhysical(memHandle));
ACLCHECK(aclrtReleaseMemAddress(virPtr));

// Destroy the task stream.
ACLCHECK(aclrtDestroyStream(stream));

// Destroy the communicator.
HCCLCHECK(HcclCommDestroy(hcclComm));
```

<!-- end id11 -->
