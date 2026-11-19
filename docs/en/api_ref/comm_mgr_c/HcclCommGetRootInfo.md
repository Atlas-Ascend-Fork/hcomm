# HcclCommGetRootInfo

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

Obtains the rootInfo required by [HcclCommGrow](HcclCommGrow.md) from an existing communicator.

If the root of the current communicator is valid, this API returns the root information. If the current root has become invalid, for example, because the root was excluded during scale-in, this API triggers root reselection and creates a new root server.

This API is different from [HcclGetRootInfo](HcclGetRootInfo.md). [HcclGetRootInfo](HcclGetRootInfo.md) generates root information when a new communicator is created, while this API obtains the root information required for scale-out from an existing communicator.

## Function Prototype

```c
HcclResult HcclCommGetRootInfo(HcclComm comm, HcclRootInfo *rootInfo)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Existing communicator handle. The communicator must be created by using rootInfo and must be the latest generation.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| rootInfo | Output | Root information for scale-out. After obtaining it, distribute it to all scale-out participants.<br>For the definition of the HcclRootInfo type, see [HcclRootInfo](./data_type_definition/HcclRootInfo.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | The rootInfo is obtained successfully. |
| HCCL_E_PTR | `comm` or `rootInfo` is a null pointer, or an internal object of the communicator is abnormal. |
| HCCL_E_PARA | The communicator derivation generation has overflowed, and a new rootInfo cannot be generated. |
| HCCL_E_INTERNAL | Root reselection, root server creation, or root information obtaining fails. |

## Constraints

- This API is supported only on Ascend 950PR&950DT products.
- Call this API before scale-out. The service selects one surviving rank to perform the call and distributes the obtained `rootInfo` to all surviving ranks and new ranks.
- If the current root is valid, the rank on which the root resides is advised to perform the call. If the root has become invalid, one surviving rank must perform the call to trigger root reselection.
- Multiple ranks must not trigger root reselection concurrently.
- This API cannot be executed concurrently with collective communication, point-to-point communication, scale-in or scale-out, communicator destruction, or other operations on the same communicator.
- This API does not support concurrent multi-thread calls. The caller must serialize calls from different threads.
- No compile-time warning is emitted by default. If `HCCL_ENABLE_EXPERIMENTAL_WARNING` is defined and `-Werror` (or MSVC `/WX`) is also enabled, the experimental feature warning causes compilation to fail. Disable the warning only locally around the call site and restore the warning settings afterward. For example, GCC/Clang can use `#pragma GCC diagnostic ignored "-Wattribute-warning"`, and MSVC can use `#pragma warning(disable: 4996)`.
- `comm` must be a valid communicator handle. Do not pass a null pointer, a wild pointer, or a destroyed handle.

## Example

```c
#define HCCLCHECK(cmd) do { HcclResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)

// Existing communicator. Set it according to the actual scenario.
HcclComm comm = NULL;
HcclRootInfo rootInfo = {0};

// Only one surviving rank calls this API.
HCCLCHECK(HcclCommGetRootInfo(comm, &rootInfo));

// The service distributes rootInfo to all scale-out participants through the control plane.
// All participants must pass the same rootInfo when calling HcclCommGrow.
```
