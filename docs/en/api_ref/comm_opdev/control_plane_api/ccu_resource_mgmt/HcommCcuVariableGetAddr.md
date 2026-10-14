# HcommCcuVariableGetAddr

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:18:37.659Z pushedAt=2026-09-29T10:54:31.196Z -->

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

Queries the cached virtual address of the `index`-th variable (scalar register) under the specified reservation handle. The reservation handle is returned by [HcommCcuVariableAlloc](HcommCcuVariableAlloc.md).

This address is mapped and cached during the [HcommCcuVariableAlloc](HcommCcuVariableAlloc.md) phase for use by target modules other than the CCU. This API is called on the host side and returns the cached address by `index`. It can be called repeatedly, and multiple queries with the same `index` return the same address. The address must be passed to the target module as the original value.

If the same reservation handle is used again in the kernel to bind the [Variable](../../data_plane_api/ccu/resource_allocation_operation/Variable.md) with the same `index`, it can point to the same physical scalar register as the module described above. For address mapping details, see [HcommCcuVariableAlloc](HcommCcuVariableAlloc.md).

## Function Prototype

```c
CcuResult HcommCcuVariableGetAddr(CcuVariableHandle varHandle, uint32_t index, uint64_t *va)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| varHandle | Input | Reservation handle returned by [HcommCcuVariableAlloc](HcommCcuVariableAlloc.md), which must be a variable reservation handle. |
| index | Input | Sequence number within the reservation segment, with a value range of `[0, num)`, where `num` is the number specified when calling [HcommCcuVariableAlloc](HcommCcuVariableAlloc.md). |
| va | Output | Receives the cached virtual address of the variable (scalar register). It cannot be a null pointer and must be passed to the target module as the original value. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The query succeeds, and `*va` is the cached virtual address of this variable (scalar register). |
| `CCU_E_PTR` | `va` is a null pointer. |
| `CCU_E_NOT_FOUND` | No reservation record corresponding to `varHandle` exists on the current device, or the reservation has been destroyed along with the CCU instance. |
| `CCU_E_PARA` | The handle exists on the current device but its type is not variable (for example, a handle from [HcommCcuEventAlloc](HcommCcuEventAlloc.md) is passed by mistake), or `index` is out of the range `[0, num)`. |

## Constraints

- This API can be called only on the host side, not inside a kernel function body.
- The thread calling this API must be bound to the same NPU device used when the reservation was initiated through the `aclrtSetDevice(int32_t deviceId)` API. This API searches only the reservation records of the current device and does not verify handle ownership. Using a reservation handle of this device on another device yields meaningless query results.
- The validity period of the returned address is the same as that of the reservation handle: after [HcommCcuInsDestroy](HcommCcuInsDestroy.md) destroys the instance, the mapping is released, and the previously obtained address must not be used any further.
- Registering a kernel is not required when only querying the address. To share the same scalar register with a kernel, the same reservation handle must be used inside the kernel to bind the corresponding `index`. For the complete process, see [HcommCcuVariableAlloc](HcommCcuVariableAlloc.md).

## Example

```c
// insHandle is the created CCU instance handle. See HcommCcuInsCreateDefault/HcommCcuInsCreate.
CcuVariableHandle acqHandle = 0;
const uint32_t varNum = 8;
CcuResult ret = HcommCcuVariableAlloc(insHandle, 0, varNum, &acqHandle);
if (ret != CCU_SUCCESS) {
    return ret;
}

// Obtain the cached virtual address of each variable (scalar register) in this reservation one by one.
for (uint32_t i = 0; i < varNum; i++) {
    uint64_t va = 0;
    ret = HcommCcuVariableGetAddr(acqHandle, i, &va);
    if (ret != CCU_SUCCESS) {
        printf("HcommCcuVariableGetAddr failed, index = %u, ret = %d\n", i, ret);
        return ret;
    }
    // Pass va to the target module as its original value, for example:
    //   taskParam.varVa[i] = va;
}
```
