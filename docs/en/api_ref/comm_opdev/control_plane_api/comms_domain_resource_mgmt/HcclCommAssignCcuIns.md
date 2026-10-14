# HcclCommAssignCcuIns

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:47:27.170Z pushedAt=2026-10-12T06:12:35.594Z -->

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

Binds a CCU instance created by the caller to the specified HCCL communicator. After binding is successful, the communicator stores the instance handle and takes over the ownership and destruction responsibility of the CCU instance. The instance can be queried later through `HcclCommQueryAssignedCcuIns`.

Each communicator can bind only one CCU instance. If the communicator already has a bound instance, this API does not overwrite the original instance.

## Function Prototype

```c
HcclResult HcclCommAssignCcuIns(HcclComm comm, CcuInsHandle insHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | HCCL communicator handle, which cannot be a null pointer. For the definition of the HcclComm type, see [HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md). |
| insHandle | Input | Handle of the CCU instance to be bound, which cannot be `0`. The instance must be created on the current device by `HcommCcuInsCreate` or `HcommCcuInsCreateDefault`. For the definition of the CcuInsHandle type, see [CcuInsHandle](../../datatype_definition/CcuInsHandle.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns `HCCL_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `HCCL_SUCCESS` | Binding successful. The ownership and destruction responsibility of `insHandle` are transferred to the communicator. |
| `HCCL_E_PTR` | `comm` is a null pointer, or the internal object of the communicator is null. |
| `HCCL_E_PARA` | `insHandle` is `0`, or the communicator is already bound to a CCU instance. |
| `HCCL_E_NOT_SUPPORT` | The communicator generation does not support CCU (earlier than the Ascend 950PR&950DT products). |
| `HCCL_E_NOT_FOUND` | No CCU instance corresponding to `insHandle` exists on the current device. |
| `HCCL_E_INTERNAL` | Another internal error occurs. |

## Constraints

- After binding succeeds, the caller must not call `HcommCcuInsDestroy` to destroy `insHandle`. When the communicator is destroyed, the bound CCU instance is destroyed as well.
- When binding fails, the ownership of `insHandle` remains with the caller, who is responsible for continuing to use or destroying the instance.
- The same `insHandle` can be bound to only one communicator and must not be bound repeatedly to other communicators.
- The same communicator does not support repeated binding. Regardless of whether the new and old handles are the same, calling this API again returns `HCCL_E_PARA`, and the original binding relationship remains unchanged.
- This API only guarantees concurrency safety among multiple `HcclCommAssignCcuIns` calls. The caller must ensure that this API is not executed concurrently with `HcclCommQueryCcuIns`, `HcclCommQueryAssignedCcuIns`, or `HcclCommDestroy`.
- `insHandle` must be created on the current device where `comm` resides.
- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

```c
CcuInsHandle insHandle = 0;
// dieIds and dieNum are reserved parameters. In the current version, pass NULL and 0 respectively.
CcuResult ccuRet = HcommCcuInsCreateDefault(NULL, 0, &insHandle);
if (ccuRet != CCU_SUCCESS) {
    return ccuRet;
}

HcclResult hcclRet = HcclCommAssignCcuIns(comm, insHandle);
if (hcclRet != HCCL_SUCCESS) {
    // Binding failed. The ownership of the instance remains with the caller.
    HcommCcuInsDestroy(insHandle);
    return hcclRet;
}

// Binding successful. The ownership of the instance has been transferred to the communicator, and the caller no longer destroys insHandle.
CcuInsHandle queriedInsHandle = 0;
uint32_t insNum = 0;
hcclRet = HcclCommQueryAssignedCcuIns(comm, &queriedInsHandle, &insNum);
if (hcclRet != HCCL_SUCCESS || insNum != 1 ||
    queriedInsHandle != insHandle) {
    HcclCommDestroy(comm);
    return HCCL_E_INTERNAL;
}

// When the communicator is destroyed, the bound CCU instance is destroyed by the communicator.
return HcclCommDestroy(comm);
```
