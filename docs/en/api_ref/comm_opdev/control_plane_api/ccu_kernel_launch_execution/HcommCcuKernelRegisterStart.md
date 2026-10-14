# HcommCcuKernelRegisterStart

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:07:51.710Z pushedAt=2026-10-08T09:16:14.797Z -->

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
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->

## Description

Marks the start of a kernel registration round and clears the set of pending kernels on the specified CCU instance, preparing for subsequent calls to [HcommCcuKernelRegister](HcommCcuKernelRegister.md).

A CCU instance supports multiple registration rounds. Each round starts with this API and ends with [HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md). Within each round, [HcommCcuKernelRegister](HcommCcuKernelRegister.md) can be called to register one or more kernels.

## Function Prototype

```c
CcuResult HcommCcuKernelRegisterStart(CcuInsHandle insHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| insHandle | Input | CCU instance handle, obtained from the HCCL communicator. Valid handles start from 1. Passing 0 or an unregistered handle returns `CCU_E_PTR`. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PTR` | `insHandle` is 0 or invalid, and no corresponding instance is found. |
| `CCU_E_INTERNAL` | Sequence error: This API is called again to start a new registration round before the previous registration round is ended by calling [HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md). |

## Constraints

- **CcuInsHandle** must be obtained in the HCCL communicator first, and must be obtained before [HcommCcuKernelRegister](HcommCcuKernelRegister.md) is called.
- This API and [HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md) must be called in pairs: before starting a new registration round, the previous round must have been ended by calling [HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md); otherwise, this API returns `CCU_E_INTERNAL`.
- This API can be called only on the host side, not inside a kernel function body.

## Example

```c
// Obtain insHandle from the HCCL communicator.
CcuInsHandle insHandle = 0;
// ... The HcommCcuInsCreate call is omitted here ...

// Start a round of kernel registration.
CcuResult ret = HcommCcuKernelRegisterStart(insHandle);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuKernelRegisterStart failed, ret = %d\n", ret);
    return ret;
}
// Subsequently call HcommCcuKernelRegister to register one or more kernels.
```
