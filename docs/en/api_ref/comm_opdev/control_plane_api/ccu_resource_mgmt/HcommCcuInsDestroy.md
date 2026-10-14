# HcommCcuInsDestroy

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:12:30.416Z pushedAt=2026-09-29T08:45:42.141Z -->

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

Destroys the CCU instance held by the caller, deregisters the kernel associated with the instance, and releases the CCU resources occupied by the instance. If resources reserved through [HcommCcuVariableAlloc](HcommCcuVariableAlloc.md)/[HcommCcuEventAlloc](HcommCcuEventAlloc.md) exist on the instance, the reserved handles become invalid when the instance is destroyed. After the instance is destroyed successfully, the instance handle becomes invalid.

## Function Prototype

```c
CcuResult HcommCcuInsDestroy(CcuInsHandle ccuInsHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| ccuInsHandle | Input | CCU instance handle to be destroyed, which must be created on the current device by [HcommCcuInsCreate](HcommCcuInsCreate.md) or [HcommCcuInsCreateDefault](HcommCcuInsCreateDefault.md). |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The instance is destroyed successfully, and the CCU instance handle becomes invalid. |
| `CCU_E_NOT_FOUND` | No CCU instance corresponding to `ccuInsHandle` exists on the current device, or the instance has already been destroyed. |

## Constraints

- The thread that calls this API must be bound to the same NPU device used when the instance was created.
- This API can be called only when the instance ownership still belongs to the caller.
- After the instance is successfully bound to a communicator through [HcclCommAssignCcuIns](../comms_domain_resource_mgmt/HcclCommAssignCcuIns.md), the instance ownership and destruction responsibility are transferred to the communicator, and the caller must not call this API again.
- After the instance is destroyed successfully, `ccuInsHandle` must not be used again. Repeated destruction returns `CCU_E_NOT_FOUND`.
- The caller must ensure that no other thread is using the instance.
- This API can be called only on the host side.

## Example

```c
CcuInsHandle insHandle = 0;
CcuResult ret = HcommCcuInsCreateDefault(NULL, 0, &insHandle);
if (ret != CCU_SUCCESS) {
    return ret;
}

// Register and execute the kernel using insHandle.
// ...

ret = HcommCcuInsDestroy(insHandle);
return ret;
```
