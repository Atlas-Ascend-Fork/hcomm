# HcommCcuKernelRegisterEnd

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:07:41.009Z pushedAt=2026-09-29T07:37:27.447Z -->

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

Ends a round of kernel registration, translates all kernels generated during this registration round into CCU device instructions at once, and delivers them to the device memory. After the translation is complete, all kernels registered in this round can be launched for execution through [HcommCcuKernelLaunch](HcommCcuKernelLaunch.md).

## Function Prototype

```c
CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle insHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| insHandle | Input | CCU instance handle, obtained from the HCCL communicator. Valid handles start from 1. Passing 0 or an unregistered handle returns `CCU_E_PTR`. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | Operation succeeded. |
| `CCU_E_PTR` | `insHandle` is 0 or invalid, and the corresponding instance is not found. |
| `CCU_E_INTERNAL` | Internal error: translation failed or device memory copy failed; or a sequence error: this API is called without first calling [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md). |
| `CCU_E_UNAVAIL` | Hardware resources are insufficient to complete resource allocation and translation for this round of kernels. |

## Constraints

- This API should be called after [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md) and at least one [HcommCcuKernelRegister](HcommCcuKernelRegister.md) call, and before [HcommCcuKernelLaunch](HcommCcuKernelLaunch.md). If [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md) is not called first, this API returns `CCU_E_INTERNAL`.

> [!NOTE] Note
> The APIs must be called in the following order: [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md) → [HcommCcuKernelRegister](HcommCcuKernelRegister.md) → this API. Calling this API without first calling [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md) returns `CCU_E_INTERNAL`.

- After this API is successfully called, the kernels registered in this round can be launched independently. To register a new round of kernels, call [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md) again to start a new round.
- This API can be called only on the host side, not inside a kernel function body.

## Example

```c
// insHandle is obtained from the communicator, and RegisterStart and Register have been completed.
CcuInsHandle insHandle = 0;
// ... The calls to HcommCcuKernelRegisterStart and HcommCcuKernelRegister are omitted here ...

// End the registration, translate it into device instructions, and deliver them.
CcuResult ret = HcommCcuKernelRegisterEnd(insHandle);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuKernelRegisterEnd failed, ret = %d\n", ret);
    return ret;
}
// All kernels registered in this round are ready. HcommCcuKernelLaunch can be called to launch them.
```
