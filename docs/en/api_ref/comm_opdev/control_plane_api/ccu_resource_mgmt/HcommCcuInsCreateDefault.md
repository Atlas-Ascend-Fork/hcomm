# HcommCcuInsCreateDefault

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:11:45.095Z pushedAt=2026-09-29T08:43:05.863Z -->

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

Creates a CCU instance using all CCU resources of all enabled IO dies on the current device, and returns the instance handle.

This API does not allocate instruction resources. Instruction resources are allocated based on the actual number of Kernel instructions during CCU kernel registration.

The current version does not read `dieIds` and `dieNum`. The caller should pass `NULL` and `0` respectively. To request resources on demand, use [HcommCcuInsCreate](HcommCcuInsCreate.md).

## Function Prototype

```c
CcuResult HcommCcuInsCreateDefault(const uint32_t *dieIds, uint32_t dieNum, CcuInsHandle *ccuInsHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| dieIds | Input | Reserved parameter, not read in the current version. Pass `NULL`. |
| dieNum | Input | Reserved parameter, not read in the current version. Pass `0`. |
| ccuInsHandle | Output | CCU instance handle returned after the instance is created successfully. It cannot be a null pointer. See [CcuInsHandle](../../datatype_definition/CcuInsHandle.md) for the type definition. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The instance is created successfully, and `*ccuInsHandle` is a valid CCU instance handle. |
| `CCU_E_PTR` | `ccuInsHandle` is a null pointer. |
| `CCU_E_UNAVAIL` | The resources on the current device are insufficient to allocate all resources. |
| `CCU_E_DRV_BUSY` | In a single-device multi-process scenario, the CCU driver has already been started by another process. |
| `CCU_E_INTERNAL` | CCU instance creation, resource query, resource allocation, or internal registration failed. |
| Other error codes | Other errors that occur when initializing the CCU driver, querying the IO die status, or allocating resources. |

## Constraints

- Currently, only collective communication scenarios are supported, and resource management relies on the communicator.
- The thread that calls this API must be bound to the target NPU device through the AscendCL API `aclrtSetDevice(int32_t deviceId)`.
- This API allocates all CCU resources of all enabled IO dies on the current device. To share resources with other CCU instances, use [HcommCcuInsCreate](HcommCcuInsCreate.md) to allocate resources on demand instead.
- After the instance is created successfully, its ownership belongs to the caller. The caller should destroy the instance through [HcommCcuInsDestroy](HcommCcuInsDestroy.md). If the instance is successfully bound to a communicator through [HcclCommAssignCcuIns](../comms_domain_resource_mgmt/HcclCommAssignCcuIns.md), the ownership and destruction responsibility are transferred to the communicator.
- This API can only be called on the host side.

## Example

```c
CcuInsHandle insHandle = 0;
CcuResult ret = HcommCcuInsCreateDefault(NULL, 0, &insHandle);
if (ret != CCU_SUCCESS) {
    return ret;
}

HcclResult hcclRet = HcclCommAssignCcuIns(comm, insHandle);
if (hcclRet != HCCL_SUCCESS) {
    // If binding fails, the instance ownership remains with the caller.
    HcommCcuInsDestroy(insHandle);
    return CCU_E_INTERNAL;
}

// After successful binding, the communicator manages the instance lifecycle.
return CCU_SUCCESS;
```
