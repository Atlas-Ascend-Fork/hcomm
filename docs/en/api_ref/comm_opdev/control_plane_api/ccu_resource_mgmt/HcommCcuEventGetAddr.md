# HcommCcuEventGetAddr

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:10:41.247Z pushedAt=2026-09-29T08:22:10.786Z -->

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

Queries the cached virtual address of the `index`-th event (completion event unit) under a specified reservation handle. The reservation handle is returned by [HcommCcuEventAlloc](HcommCcuEventAlloc.md).

This address is mapped and cached during the [HcommCcuEventAlloc](HcommCcuEventAlloc.md) phase, for use by target modules outside the CCU. This API is called on the host side and returns the cached address by `index`. It can be called repeatedly, and multiple queries with the same `index` return the same address. The address must be passed to the target module as-is.

If the same reservation handle is used again in the kernel to bind the [event](../../data_plane_api/ccu/resource_allocation_operation/Event.md) of the same `index`, it can point to the same physical completion event unit as the module above. For address mapping details, see [HcommCcuEventAlloc](HcommCcuEventAlloc.md).

`index` selects a completion event unit in this reservation, not a specific bit within the completion event unit. On the kernel side, the `mask` of [EventRecord](../../data_plane_api/ccu/synchronization/EventRecord.md) / [EventWait](../../data_plane_api/ccu/synchronization/EventWait.md) selects the bits to be set or waited on within that completion event unit. `index` and `mask` are independent of each other: the former selects a completion event unit, and the latter selects bits within that completion event unit.

## Function Prototype

```c
CcuResult HcommCcuEventGetAddr(CcuEventHandle eventHandle, uint32_t index, uint64_t *va)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| eventHandle | Input | Reservation handle returned by [HcommCcuEventAlloc](HcommCcuEventAlloc.md). It must be an event reservation handle. |
| index | Input | Index within the reservation segment, in the range `[0, num)`, where `num` is the number specified when calling [HcommCcuEventAlloc](HcommCcuEventAlloc.md). |
| va | Output | Receives the cached virtual address of the event (completion event unit). It cannot be a null pointer. It must be passed to the target module as is. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | Query succeeded, and `*va` is the cached virtual address of the event (completion event unit). |
| `CCU_E_PTR` | `va` is a null pointer. |
| `CCU_E_NOT_FOUND` | No reservation record corresponding to `eventHandle` exists on the current device, or the reservation has been destroyed along with the CCU instance. |
| `CCU_E_PARA` | The handle exists on the current device but its type is not event (for example, a handle from [HcommCcuVariableAlloc](HcommCcuVariableAlloc.md) was passed by mistake), or `index` is out of `[0, num)`. |

## Constraints

- This API can be called only on the host side, not inside a kernel function body.
- The thread calling this API must be bound to the same NPU device used when the reservation was initiated, via the `aclrtSetDevice(int32_t deviceId)` API.
- The validity period of the returned address is the same as that of the reservation handle: after [HcommCcuInsDestroy](HcommCcuInsDestroy.md) destroys the instance, the mapping is released, and the previously obtained address must no longer be used.
- Registering a kernel is not required when only querying the address. To share the same completion event unit with a kernel, the same reservation handle must be used inside the kernel to bind the corresponding `index`. For the complete process, see [HcommCcuEventAlloc](HcommCcuEventAlloc.md).

## Example

```c
// insHandle is the created CCU instance handle. See HcommCcuInsCreateDefault/HcommCcuInsCreate.
CcuEventHandle acqEventHandle = 0;
const uint32_t eventNum = 4;
CcuResult ret = HcommCcuEventAlloc(insHandle, 0, eventNum, &acqEventHandle);
if (ret != CCU_SUCCESS) {
    return ret;
}

// Obtain the cached virtual address of each event (completion event unit) in this reservation one by one.
for (uint32_t i = 0; i < eventNum; i++) {
    uint64_t va = 0;
    ret = HcommCcuEventGetAddr(acqEventHandle, i, &va);
    if (ret != CCU_SUCCESS) {
        printf("HcommCcuEventGetAddr failed, index = %u, ret = %d\n", i, ret);
        return ret;
    }
    // Pass va to the target module as is, for example:
    //   taskParam.eventVa[i] = va;
}
```
