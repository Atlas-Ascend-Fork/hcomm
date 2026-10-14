# HcclCommQueryAssignedCcuIns

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:47:52.589Z pushedAt=2026-10-12T06:12:35.596Z -->

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

Queries the CCU instance handle bound to a specified HCCL communicator through the new method, and returns the instance handle for subsequent registration and dispatch of CCU kernels.

The new method means that the caller first creates a CCU instance through `HcommCcuInsCreate` or `HcommCcuInsCreateDefault`, and then binds the instance to the communicator through [HcclCommAssignCcuIns](HcclCommAssignCcuIns.md). This API queries the instance handle (**assignedCcuInsHandle**) corresponding to this binding relationship. When the query succeeds, exactly one instance is returned: **insHandles[0]** is the bound instance handle, and ***insNum** is **1**.

This API does not create a CCU instance. If the communicator has not bound any CCU instance through `HcclCommAssignCcuIns`, this API returns **HCCL_E_UNAVAIL**.

> [!NOTE] Note
> - This API queries the CCU instance bound through `HcclCommAssignCcuIns`, which is independent of the communicator's own CCU instance (queried/created by [HcclCommQueryCcuIns](HcclCommQueryCcuIns.md)). To query the communicator's own instance, use `HcclCommQueryCcuIns`.
> - The query result is for borrowing only and does not transfer instance ownership. The ownership of the bound CCU instance belongs to the communicator, which is responsible for releasing it. The caller cannot destroy the instance; otherwise, the same instance will be released repeatedly, breaking the communicator's resource management.

## Function Prototype

```c
HcclResult HcclCommQueryAssignedCcuIns(HcclComm comm, CcuInsHandle *insHandles, uint32_t *insNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle, which cannot be **nullptr**.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| insHandles | Output | CCU instance handle array, which cannot be **nullptr**. If the query succeeds, **insHandles[0]** returns the bound CCU instance handle.<br>The caller must allocate space for at least one **CcuInsHandle** element.<br>The CcuInsHandle type is defined as follows:<br>typedef uint64_t CcuInsHandle; |
| insNum | Output | Number of CCU instances, which cannot be **nullptr**. If the query succeeds, the return value is fixed to **1**. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

- When **comm**, **insHandles**, or **insNum** is **nullptr**, **HCCL_E_PTR** is returned.
- When the communicator generation does not support CCU (earlier than the Ascend 950PR&950DT products), **HCCL_E_NOT_SUPPORT** is returned.
- When the communicator has not yet bound a CCU instance through `HcclCommAssignCcuIns`, **HCCL_E_UNAVAIL** is returned, and this API does not create an instance.

## Constraints

1. The CCU feature is supported only on the Ascend 950PR&950DT products and later generations. Calling this API on a communicator of an earlier generation returns **HCCL_E_NOT_SUPPORT**.
2. Before calling this API, a CCU instance must have been created through `HcommCcuInsCreate` or `HcommCcuInsCreateDefault` and bound to the communicator through `HcclCommAssignCcuIns`; otherwise, **HCCL_E_UNAVAIL** is returned.
3. The CCU instance queried by this API is independent of the communicator's own CCU instance (queried/created by `HcclCommQueryCcuIns`). The two do not overwrite or affect each other.
4. The returned CCU instance handle is borrowed only. Its ownership remains with the communicator, which is responsible for releasing it. The caller must not destroy the instance. Otherwise, the same instance will be released repeatedly, corrupting the resource management of the communicator.
5. This API does not guarantee safe concurrent execution with `HcclCommAssignCcuIns` or `HcclCommDestroy`. The caller must ensure that this API is not executed concurrently with `HcclCommAssignCcuIns` or `HcclCommDestroy`.
6. **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

```c
// comm must be initialized through APIs such as HcclCommInitClusterInfo. This is only a placeholder for the example.
HcclComm comm = /* Initialized communicator handle. */;

// 1. New method: Create a CCU instance first, and then bind it to the communicator.
CcuInsHandle insHandle = 0;
CcuResult ccuRet = HcommCcuInsCreateDefault(NULL, 0, &insHandle);
if (ccuRet != CCU_SUCCESS) {
    return ccuRet;
}

HcclResult hcclRet = HcclCommAssignCcuIns(comm, insHandle);
if (hcclRet != HCCL_SUCCESS) {
    // Binding fails, and the instance ownership remains with the caller.
    HcommCcuInsDestroy(insHandle);
    return hcclRet;
}
// Binding succeeds, and the instance ownership is transferred to the communicator. The caller no longer destroys insHandle.

// 2. Query the bound CCU instances (new method).
CcuInsHandle queriedInsHandle = 0;
uint32_t insNum = 0;
hcclRet = HcclCommQueryAssignedCcuIns(comm, &queriedInsHandle, &insNum);
// The current implementation always returns one CCU instance. insNum != 1 is treated as a failure.
if (hcclRet != HCCL_SUCCESS || insNum != 1) {
    // Error handling: If the API fails, the original error code is returned. If the API succeeds but the instance count is abnormal, an internal error code is returned.
    HcclCommDestroy(comm);
    return (hcclRet != HCCL_SUCCESS) ? hcclRet : HCCL_E_INTERNAL;
}

// Verify that the queried handle matches the handle used at binding time.
if (queriedInsHandle != insHandle) {
    HcclCommDestroy(comm);
    return HCCL_E_INTERNAL;
}

// Use queriedInsHandle to register and dispatch the CCU Kernel.
CcuResult regStartRet = HcommCcuKernelRegisterStart(queriedInsHandle);
// ...

// When the communicator is destroyed, the communicator destroys the bound CCU instances.
return HcclCommDestroy(comm);
```
