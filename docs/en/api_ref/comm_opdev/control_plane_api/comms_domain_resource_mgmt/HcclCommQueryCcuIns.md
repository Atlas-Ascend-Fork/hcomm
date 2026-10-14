# HcclCommQueryCcuIns

<!-- md-trans-meta sourceCommit=e169993072c8b4f8692fd6915a9e65337ecf402f translatedAt=2026-10-08T11:48:16.515Z pushedAt=2026-10-12T06:12:35.597Z -->

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

Queries the CCU instance handle owned by a specified HCCL communicator and returns the instance handle for subsequent registration and dispatch of CCU kernels.

The communicator no longer proactively creates a CCU instance during initialization. This API queries the CCU instance handle owned by the communicator (that is, the handle created with a fixed amount of resources and owned by the communicator). When the handle does not yet exist (its value is `0`), the API creates a CCU instance based on the communicator's operator expansion mode (**opExpansionMode**), saves it to the communicator, and then returns the handle. When the query is successful, exactly one instance is returned: **insHandles[0]** is the instance handle owned by the communicator, and ***insNum** is **1**.

Currently, a communicator holds at most one CCU instance of its own. Multiple queries return the same handle and do not create the instance repeatedly.

> [!NOTE] Note
> - This API queries the CCU instance owned by the communicator, which is independent of the CCU instance (new version) bound through [HcclCommAssignCcuIns](HcclCommAssignCcuIns.md). To query the instance bound through `HcclCommAssignCcuIns`, use [HcclCommQueryAssignedCcuIns](HcclCommQueryAssignedCcuIns.md).
> - The query result is for borrowing only and does not transfer instance ownership. The created CCU instance is owned by the communicator, which is responsible for releasing it. The caller must not destroy the instance; otherwise, the same instance will be released repeatedly, breaking the communicator's resource management.

## Function Prototype

```c
HcclResult HcclCommQueryCcuIns(HcclComm comm, CcuInsHandle *insHandles, uint32_t *insNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle, which must not be **nullptr**.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| insHandles | Output | CCU instance handle array, which must not be **nullptr**. After a successful query, **insHandles[0]** returns the communicator's own CCU instance handle (created if it does not exist before the call).<br>The caller must allocate space for at least one **CcuInsHandle** element.<br>The CcuInsHandle type is defined as follows:<br>typedef uint64_t CcuInsHandle; |
| insNum | Output | Number of CCU instances, which must not be **nullptr**. After a successful query, the return value is fixed to **1**. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

- When **comm**, **insHandles**, or **insNum** is **nullptr**, **HCCL_E_PTR** is returned.
- When the communicator generation does not support CCU (earlier than Ascend 950PR&950DT products), **HCCL_E_NOT_SUPPORT** is returned.
- When the communicator's own CCU instance handle is `0` and the operator expansion mode does not enable CCU, **HCCL_E_UNAVAIL** is returned and no instance is created.
- When the CCU instance fails to be created, the original error code of the creation API is returned (**CcuResult** is passed through, such as **CCU_E_DRV_BUSY** and **CCU_E_UNAVAIL**).

> [!NOTE] Note
> When resources are insufficient, the following handling is performed:
>
> - When the operator expansion mode is CCU_MS and CCU resources are insufficient to create a CCU_MS instance, this API automatically degrades to CCU_SCHED and retries creation once.
> - When the CCU_SCHED instance also cannot be created (resources are still insufficient), **HCCL_E_UNAVAIL** is returned. **This API does not automatically degrade to AICPU_TS**, and the caller must decide whether to degrade after receiving **HCCL_E_UNAVAIL**.

## Constraints

1. The CCU feature is supported only on Ascend 950PR&950DT products and later generations. Calling this API on a communicator of an earlier generation returns **HCCL_E_NOT_SUPPORT**.
2. This API must be called after the communicator is initialized. When the communicator's own CCU instance has not been created, this API creates it; if the communicator expansion mode does not enable CCU, **HCCL_E_UNAVAIL** is returned.
3. The CCU instance queried by this API is independent of the CCU instance bound through `HcclCommAssignCcuIns`; the two do not overwrite or affect each other.
4. The returned CCU instance handle is borrowed only. Its ownership remains with the communicator, which is responsible for releasing it. The caller must not destroy the instance. Otherwise, the same instance will be released repeatedly, corrupting the resource management of the communicator.
5. This API only guarantees idempotency across multiple `HcclCommQueryCcuIns` calls (if already created, it is returned directly without being created again). The caller must ensure that this API is not executed concurrently with `HcclCommAssignCcuIns` or `HcclCommDestroy`.
6. When CCU resources are insufficient, this API automatically degrades only once between CCU_MS and CCU_SCHED, and does not automatically degrade to AICPU_TS. After receiving **HCCL_E_UNAVAIL**, the caller must handle whether to degrade to AICPU_TS mode.
7. **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

```c
// comm must be initialized through APIs such as HcclCommInitClusterInfo. This is only a placeholder for the example.
HcclComm comm = /* Initialized communicator handle. */;
CcuInsHandle insHandle = 0;
uint32_t insNum = 0;

// Query the CCU instance owned by the communicator. If it has not been created, create it based on the communicator expansion mode.
HcclResult ret = HcclCommQueryCcuIns(comm, &insHandle, &insNum);
if (ret == HCCL_E_UNAVAIL) {
    // CCU resources are insufficient to create a CCU instance. The caller must handle whether to degrade to AICPU_TS mode.
    return fallbackToAicpuTs(comm, ...);
}
// The current implementation always returns one CCU instance. insNum != 1 is treated as a failure.
if (ret != HCCL_SUCCESS || insNum != 1) {
    // Error handling: If the API fails, return the original error code. If the API succeeds but the instance count is abnormal, return an internal error code.
    return (ret != HCCL_SUCCESS) ? ret : HCCL_E_INTERNAL;
}

// Use insHandle to register and dispatch the CCU Kernel.
CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
// ...
```
