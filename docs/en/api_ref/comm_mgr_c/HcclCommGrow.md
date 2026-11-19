# HcclCommGrow

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

Adds new ranks to an existing communicator to create a larger communicator. This API does not modify the input communicator in place. Instead, it returns a new communicator handle through `newComm`.

After scale-out, surviving ranks in the original communicator retain their original rank IDs. Newly added ranks are assigned contiguous IDs starting from the number of ranks in the original communicator.

## Function Prototype

```c
HcclResult HcclCommGrow(
    HcclComm comm,
    uint32_t nRanks,
    const HcclRootInfo *rootInfo,
    uint32_t rank,
    const HcclCommConfig *config,
    HcclComm *newComm)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Surviving ranks in the original communicator pass the current communicator handle, including the scenario where the original root was excluded during scale-in; new ranks pass `NULL`.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| nRanks | Input | Total number of ranks in the communicator after scale-out. The value must be greater than the number of ranks in the communicator before scale-out. All participants must pass the same value. |
| rootInfo | Input | Root information of the communicator after scale-out, obtained by calling [HcclCommGetRootInfo](HcclCommGetRootInfo.md). All participants must pass the same `rootInfo`. |
| rank | Input | Rank ID of the current rank in the communicator after scale-out. Surviving ranks retain their original IDs (0 to N-1), and new ranks are assigned IDs (N to N+M-1).<br>N is the number of ranks before scale-out, M is the number of new ranks, and `rank < nRanks`. |
| config | Input | Configuration of the new communicator.<br>If `NULL` is passed, the default configuration is used. If the value is not `NULL`, call [HcclCommConfigInit](HcclCommConfigInit.md) to initialize it first. Fields that are not explicitly set use default values. All participants must pass the same configuration.<br>For the definition of the HcclCommConfig type, see [HcclCommConfig](./data_type_definition/HcclCommConfig.md). |
| newComm | Output | Handle of the new communicator after scale-out. This parameter is valid when the API call is successful.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | Scale-out succeeds, and `newComm` returns the new communicator handle. |
| HCCL_E_PTR | `newComm` is a null pointer, or an internal object of the passed communicator is abnormal. |
| HCCL_E_PARA | `rootInfo` is null, `nRanks` is invalid, `rank` is out of range, the configuration is invalid, or the derived communicator name is too long. |
| HCCL_E_NOT_SUPPORT | The communicator is created by using a rankTable, or the communicator is not the latest generation. |
| HCCL_E_INTERNAL | Topology information collection is incomplete, link establishment fails, or internal resource initialization fails. |

## Constraints

- This API is supported only on Ascend 950PR&950DT products.
- Only communicators created by using rootInfo are supported. Communicators created by [HcclCommInitClusterInfo](HcclCommInitClusterInfo.md) or [HcclCommInitClusterInfoConfig](HcclCommInitClusterInfoConfig.md) do not support scale-out.
- Before scale-out, synchronize all related streams to ensure that no collective communication, point-to-point communication, or resource operation is in flight on the original communicator.
- All participants must call this API and pass the same `nRanks`, `rootInfo`, and `config`.
- All surviving ranks and new ranks must complete the call in the same scale-out round. If any participant does not start, does not call the API, or passes inconsistent parameters, topology information collection may be incomplete, link establishment may fail, or the operation may time out.
- Surviving ranks in the original communicator pass the current communicator handle, and new ranks pass `NULL`.
- Before scale-out, the service selects one surviving rank to call [HcclCommGetRootInfo](HcclCommGetRootInfo.md) to obtain `rootInfo` and distributes it to all participants. Multiple ranks must not trigger root reselection concurrently.
- If the root of the original communicator was excluded during scale-in, [HcclCommGetRootInfo](HcclCommGetRootInfo.md) triggers root reselection and creates a new root server. One surviving rank must perform this operation and distribute the result to all surviving ranks and new ranks.
- Each scale-out must be based on the latest generation of the communicator. After scale-out succeeds, the old communicator no longer supports further scale-out or scale-in.
- This API does not support concurrent multi-thread calls. The caller must serialize calls from different threads.
- No compile-time warning is emitted by default. If `HCCL_ENABLE_EXPERIMENTAL_WARNING` is defined and `-Werror` (or MSVC `/WX`) is also enabled, the experimental feature warning causes compilation to fail. Disable the warning only locally around the call site and restore the warning settings afterward. For example, GCC/Clang can use `#pragma GCC diagnostic ignored "-Wattribute-warning"`, and MSVC can use `#pragma warning(disable: 4996)`.
- This API does not automatically destroy the old communicator. The service must call [HcclCommDestroy](HcclCommDestroy.md) after confirming that the old communicator is no longer used.
- After scale-out, symmetric memory windows, CCU instances, channels, threads, and other resources are not automatically migrated to the new communicator. Register, bind, or obtain them again on the new communicator.
- The passed `comm` must be a valid communicator handle. Do not pass a wild pointer or a destroyed handle.

## Example

```c
#define HCCLCHECK(cmd) do { HcclResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)

// All participants use the same configuration.
HcclCommConfig config;
HcclCommConfigInit(&config);

// Surviving ranks pass the current communicator handle; new ranks pass NULL.
HcclComm oldComm = NULL;
int isNewRank = 0;
uint32_t newRankId = 0;
uint32_t newRankSize = 0;

// rootInfo is obtained by one surviving rank through HcclCommGetRootInfo and distributed through the service control plane.
HcclRootInfo rootInfo = {0};

HcclComm comm = isNewRank ? NULL : oldComm;
HcclComm newComm = NULL;
HCCLCHECK(HcclCommGrow(comm, newRankSize, &rootInfo, newRankId, &config, &newComm));

// Query rank information by using the new communicator after scale-out.
uint32_t rankId = 0;
uint32_t rankSize = 0;
HCCLCHECK(HcclGetRankId(newComm, &rankId));
HCCLCHECK(HcclGetRankSize(newComm, &rankSize));
```
