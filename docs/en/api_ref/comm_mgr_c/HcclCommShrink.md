# HcclCommShrink

<!-- md-trans-meta sourceCommit=544cff576cc281e781cbe3533ac51375d5de6aa9 translatedAt=2026-11-19T11:41:26.897Z pushedAt=2026-11-19T11:41:26.897Z -->

> [!NOTE] Note
> This API is an experimental feature identified by the `HCCL_EXPERIMENTAL` macro. It may be changed, renamed, or removed without notice in later versions, and backward compatibility is not guaranteed. No compile-time warning is emitted by default. Define `HCCL_ENABLE_EXPERIMENTAL_WARNING` to enable experimental feature warnings. If warnings are enabled and `-Werror` (or MSVC `/WX`) is also enabled, the warning must be disabled locally at the call site.

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
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

Removes specified ranks from an existing communicator, and surviving ranks create a smaller communicator. This API does not modify the input communicator in place. Instead, it returns a new communicator handle through `newComm`.

After scale-in, rank IDs are renumbered contiguously in ascending order of the original rank IDs. For example, if the original communicator contains ranks 0 to 7 and ranks 2 and 5 are excluded, original rank 3 becomes rank 2 in the new communicator.

## Function Prototype

```c
HcclResult HcclCommShrink(
    HcclComm comm,
    const uint32_t *excludeRanksList,
    const uint32_t excludeRanksCount,
    const HcclCommConfig *config,
    uint32_t shrinkFlags,
    HcclComm *newComm)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator to be scaled in. The communicator must be created by using rootInfo and must be the latest generation of the communicator.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| excludeRanksList | Input | Array of rank IDs to be excluded. The rank IDs are those in the current communicator.<br>The array cannot be empty and cannot contain duplicate rank IDs or the caller's own rank. All surviving ranks must pass the same content. Ascending rank ID order is recommended. |
| excludeRanksCount | Input | Number of ranks to be excluded. The value range is (0, number of ranks in the current communicator). |
| config | Input | Configuration of the new communicator.<br>If `NULL` is passed, the configuration of the parent communicator is inherited. If the value is not `NULL`, call [HcclCommConfigInit](HcclCommConfigInit.md) to initialize it first. Fields that are not explicitly set use default values. All surviving ranks must pass the same configuration.<br>For the definition of the HcclCommConfig type, see [HcclCommConfig](./data_type_definition/HcclCommConfig.md). |
| shrinkFlags | Input | Scale-in flags. This parameter is currently reserved and must be set to 0. |
| newComm | Output | Handle of the new communicator after scale-in. This parameter is valid when the API call is successful.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | Scale-in succeeds, and `newComm` returns the new communicator handle. |
| HCCL_E_PTR | `comm`, `excludeRanksList`, or `newComm` is a null pointer, or an internal object of the communicator is abnormal. |
| HCCL_E_PARA | The exclusion list is empty, the exclusion count is invalid, a rank ID is out of range, the exclusion list contains the caller's own rank, the configuration is invalid, or the derived communicator name is too long. |
| HCCL_E_NOT_SUPPORT | The communicator is created by using a rankTable, or the communicator is not the latest generation. |
| HCCL_E_INTERNAL | Creating the new communicator or initializing internal resources fails. |

## Constraints

- This API is supported only on Ascend 950PR&950DT products.
- Only communicators created by using rootInfo are supported. Communicators created by [HcclCommInitClusterInfo](HcclCommInitClusterInfo.md) or [HcclCommInitClusterInfoConfig](HcclCommInitClusterInfoConfig.md) do not support scale-in.
- Before scale-in, synchronize all related streams to ensure that no collective communication, point-to-point communication, or resource operation is in flight on `comm`.
- All surviving ranks must call this API and pass the same `excludeRanksList`, `excludeRanksCount`, and `config`.
- All participants must complete the call in the same scale-in round. If any surviving rank does not call the API or passes inconsistent parameters, the communicator membership view may differ among ranks.
- Excluded ranks must not call this API.
- Each scale-in must be based on the latest generation of the communicator. After scale-in succeeds, the old communicator no longer supports further scale-in or scale-out.
- If the root of the original communicator is excluded during scale-in, the new communicator has no valid root after scale-in. Before a subsequent scale-out, one surviving rank must call [HcclCommGetRootInfo](HcclCommGetRootInfo.md) to trigger root reselection and distribute the new `rootInfo` to all scale-out participants.
- This API does not support concurrent multi-thread calls. The caller must serialize calls from different threads.
- No compile-time warning is emitted by default. If `HCCL_ENABLE_EXPERIMENTAL_WARNING` is defined and `-Werror` (or MSVC `/WX`) is also enabled, the experimental feature warning causes compilation to fail. Disable the warning only locally around the call site and restore the warning settings afterward. For example, GCC/Clang can use `#pragma GCC diagnostic ignored "-Wattribute-warning"`, and MSVC can use `#pragma warning(disable: 4996)`.
- This API does not automatically destroy the old communicator. The service must call [HcclCommDestroy](HcclCommDestroy.md) after confirming that the old communicator is no longer used.
- After scale-in, symmetric memory windows, CCU instances, channels, threads, and other resources are not automatically migrated to the new communicator. Register or obtain them again on the new communicator.
- `comm` must be a valid communicator handle. Do not pass a null pointer, a wild pointer, or a destroyed handle.

## Example

```c
// Experimental feature: this API may be changed or removed without notice, and backward compatibility is not guaranteed.
#define HCCLCHECK(cmd) do { HcclResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define ACLCHECK(cmd) do { aclError ret = (cmd); if (ret != ACL_SUCCESS) { return (HcclResult)ret; } } while (0)

// Communicator and stream to be scaled in. Set them according to the actual scenario.
HcclComm comm = NULL;
aclrtStream stream = NULL;
HcclComm newComm = NULL;

// Ensure that no operations are in flight on the old communicator.
ACLCHECK(aclrtSynchronizeStream(stream));

// Exclude ranks 2 and 5 from the original communicator. All surviving ranks must pass the same list.
uint32_t excludeRanksList[] = {2, 5};
const uint32_t excludeRanksCount = 2;

// If config is NULL, the configuration of the parent communicator is inherited.
HCCLCHECK(HcclCommShrink(comm, excludeRanksList, excludeRanksCount, NULL, 0, &newComm));

// Query rank information by using the new communicator after scale-in.
uint32_t rankId = 0;
uint32_t rankSize = 0;
HCCLCHECK(HcclGetRankId(newComm, &rankId));
HCCLCHECK(HcclGetRankSize(newComm, &rankSize));
```
