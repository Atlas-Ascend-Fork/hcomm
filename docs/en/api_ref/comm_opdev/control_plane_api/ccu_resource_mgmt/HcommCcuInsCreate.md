# HcommCcuInsCreate

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:10:25.454Z pushedAt=2026-09-29T08:36:48.839Z -->

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

Creates a CCU instance based on one or more CCU resource descriptors and returns the instance handle. Each resource descriptor corresponds to one IO die and specifies the number of each type of CCU resource to be allocated on that die.

The underlying layer allocates resources continuously according to the resource alignment granularity. When the allocated quantity is not an integer multiple of the alignment granularity, the actually allocated quantity may be rounded up. You can query the number of resources actually occupied by the instance through [HcommCcuInsQueryResDesc](HcommCcuInsQueryResDesc.md).

This API does not read the resource quantity corresponding to `HCOMM_CCU_RES_TYPE_INSTRUCTION` in the resource descriptor, nor does it allocate instruction resources. Instruction resources are allocated based on the actual number of instructions in the kernel during CCU kernel registration.

The mission resources corresponding to `HCOMM_CCU_RES_TYPE_CCU_THREAD` are allocated in fused multi-die mode: the maximum number of mission resources allocated across all resource descriptors is taken, and mission resources with the same quantity and the same ID range are allocated on all enabled IO dies of the current device. When only one IO die is enabled, resources are requested only on that die.

## Function Prototype

```c
CcuResult HcommCcuInsCreate(const HcommCcuResDescHandle *resDescs, uint32_t resDescNum, CcuInsHandle *ccuInsHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| resDescs | Input | Resource descriptor handle array, which must not be a null pointer. Each handle in the array must be created on the current device by [HcommCcuInsResDescCreate](HcommCcuInsResDescCreate.md), and each handle corresponds to a different IO die. |
| resDescNum | Input | Number of handles in `resDescs`, with a value range of `(0, CCU_MAX_IODIE_NUM]`. The current maximum value is 2. |
| ccuInsHandle | Output | CCU instance handle returned after successful creation, which must not be a null pointer. See [CcuInsHandle](../../datatype_definition/CcuInsHandle.md) for the type definition. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | Creation success. `*ccuInsHandle` is a valid CCU instance handle. |
| `CCU_E_PTR` | `resDescs` or `ccuInsHandle` is a null pointer, or a resource descriptor handle in the array does not belong to the current device. |
| `CCU_E_PARA` | `resDescNum` is out of the valid range, or multiple resource descriptors correspond to the same IO die. |
| `CCU_E_UNAVAIL` | Continuous resources on the current device are insufficient to satisfy the requested quantity in the resource descriptors. |
| `CCU_E_DRV_BUSY` | In a single-device multi-process scenario, the CCU driver has already been started by another process. |
| `CCU_E_INTERNAL` | CCU instance creation, resource allocation, or internal registration failed. |
| Other error codes | Other errors generated when initializing the CCU driver or allocating resources. |

## Constraints

- Currently, only collective communication scenarios are supported, and resource management depends on the communicator.
- The thread that calls this API must be bound to the target NPU device through the AscendCL API `aclrtSetDevice(int32_t deviceId)`. All descriptors in `resDescs` must be created on the same device.
- The IO die numbers in `resDescs` must not be duplicated.
- During creation, the caller must not concurrently modify or destroy the resource descriptors in `resDescs`. After successful creation, the CCU instance independently holds the allocated resources, and the original resource descriptors can continue to be used or destroyed.
- After successful creation, the instance ownership belongs to the caller. The caller should destroy the instance through [HcommCcuInsDestroy](HcommCcuInsDestroy.md). If the instance is successfully bound to a communicator through [HcclCommAssignCcuIns](../comms_domain_resource_mgmt/HcclCommAssignCcuIns.md), the ownership and destruction responsibility are transferred to the communicator.
- This API can only be called on the host side.

## Example

```c
HcommCcuResDescHandle resDesc = 0;
CcuInsHandle insHandle = 0;

CcuResult ret = HcommCcuInsResDescCreate(0, &resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

ret = HcommCcuInsResDescSetNum(
    resDesc, HCOMM_CCU_RES_TYPE_LOOP, 8);
if (ret == CCU_SUCCESS) {
    ret = HcommCcuInsCreate(&resDesc, 1, &insHandle);
}

HcommCcuInsResDescDestroy(resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

// Register the kernel with insHandle, or bind it to the communicator.
// ...

return HcommCcuInsDestroy(insHandle);
```
