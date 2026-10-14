# HcommCcuGetTaskArgsNum

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:06:10.030Z pushedAt=2026-09-29T07:21:22.553Z -->

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

Queries the number of task arguments (`taskArgsNum`) of a registered kernel. This value equals the maximum `argId` used across all `CcuLoadArg` calls during the kernel registration period plus 1, that is, the minimum number of elements required by the `taskArgs` array. The operator layer can call this API after [HcommCcuKernelRegister](HcommCcuKernelRegister.md) to obtain this value, then construct the `taskArgs` array accordingly and pass it as `argNum` to [HcommCcuKernelLaunch](HcommCcuKernelLaunch.md), thereby supporting the `<<<>>>` direct call mode.

If no `CcuLoadArg` is called during the kernel registration period, this API returns `0`.

## Function Prototype

```c
CcuResult HcommCcuGetTaskArgsNum(CcuKernelHandle kernelHandle, uint32_t *taskArgsNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| kernelHandle | Input | Kernel handle, which must be a valid handle obtained through [HcommCcuKernelRegister](HcommCcuKernelRegister.md). The value cannot be 0. |
| taskArgsNum | Output | Output parameter, a pointer to `uint32_t`, into which the number of task arguments is written on success. It cannot be a null pointer. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PTR` | `taskArgsNum` is a null pointer. |
| `CCU_E_NOT_FOUND` | `kernelHandle` does not exist in the kernel management table, the handle is invalid, or it has been deregistered. |

## Constraints

- Before calling this API, ensure that **HcommCcuKernelRegister** has successfully returned a kernel handle. There is no mandatory requirement on the calling order between this API and **HcommCcuKernelRegisterEnd**. It can be called before or after **HcommCcuKernelRegisterEnd**, but it must be called before the kernel is deregistered.
- The return value is calculated as follows: the maximum `argId` among all `CcuLoadArg` calls during the kernel registration period plus 1. `argId` is numbered starting from 0, so this value equals the minimum array length required when indexing the `taskArgs` array by `argId`. If no `CcuLoadArg` is called, `0` is returned.
- The return value can be passed directly as the `argNum` parameter of [HcommCcuKernelLaunch](HcommCcuKernelLaunch.md).
- This API performs read-only query. It does not modify the kernel registration state and does not affect subsequent launches.
- This API is thread-safe. Internally, access to the kernel management table is protected by a lock.
- This API can be called only on the host side, not inside a kernel function body.

## Example

```c
// Obtain insHandle from the communicator, and obtain kernelHandle from HcommCcuKernelRegister.
// CcuLoadArg(0) and CcuLoadArg(1) are called inside the kernel function body.
CcuKernelHandle kernelHandle = 0;
// ... HcommCcuKernelRegisterStart / HcommCcuKernelRegister / HcommCcuKernelRegisterEnd are omitted here ...

// Query the number of elements required by the taskArgs array.
uint32_t taskArgsNum = 0;
CcuResult ret = HcommCcuGetTaskArgsNum(kernelHandle, &taskArgsNum);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuGetTaskArgsNum failed, ret = %d\n", ret);
    return ret;
}
// taskArgsNum is 2 (max(argId)=1, plus 1).

// Construct the taskArgs array based on taskArgsNum and start it.
uint64_t taskArgs[2] = { 100, 200 };
ret = HcommCcuKernelLaunch(threadHandle, kernelHandle, taskArgs, taskArgsNum);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuKernelLaunch failed, ret = %d\n", ret);
    return ret;
}
```
