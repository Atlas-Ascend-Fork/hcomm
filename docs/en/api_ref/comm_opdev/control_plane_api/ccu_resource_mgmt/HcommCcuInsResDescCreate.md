# HcommCcuInsResDescCreate

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:13:48.572Z pushedAt=2026-09-29T09:13:46.659Z -->

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

Creates an empty CCU resource descriptor. A resource descriptor describes the quantity specifications of various resources (loop, buffer, variable, address, event, thread, instruction) required by a CCU instance, and does not contain actual resources.

One resource descriptor corresponds to one IO die. After creation, the quantity of each internal resource type is initialized to 0, and can be set type by type later through `HcommCcuInsResDescSetNum`.

## Function Prototype

```c
CcuResult HcommCcuInsResDescCreate(uint32_t dieId, HcommCcuResDescHandle *resDesc)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| dieId | Input | IO die number to which the resource descriptor belongs. Value range: `[0, CCU_MAX_IODIE_NUM)`. |
| resDesc | Output | Resource descriptor handle returned after successful creation. It cannot be a null pointer. The caller is responsible for destroying it through `HcommCcuInsResDescDestroy`. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | Created successfully, and `*resDesc` is a valid resource descriptor handle. |
| `CCU_E_PARA` | `dieId` is out of the valid range (`>= CCU_MAX_IODIE_NUM`). |
| `CCU_E_PTR` | `resDesc` is a null pointer. |
| `CCU_E_INTERNAL` | Memory allocation or internal registration failed. |

## Constraints

- Before calling this API, bind the current thread to the target NPU device by using the AscendCL API `aclrtSetDevice(int32_t deviceId)`. This API obtains the device context from the current thread.
- After the descriptor is created successfully, the caller must destroy it by using `HcommCcuInsResDescDestroy` when it is no longer in use to avoid resource leakage.
- A resource descriptor cannot be shared across threads. The caller must ensure that access to the same descriptor is serialized.
- This API can be called only on the host side.

## Dependencies

- Depends on `HcommCcuInsResDescDestroy` to destroy the descriptor.
- Depends on `HcommCcuInsResDescSetNum` to set the number of resources.
- Depends on `HcommCcuInsResDescQueryNum`, `HcommCcuInsResDescQueryDieId`, and `HcommCcuQueryRemainResDesc` to query descriptor information.

## Example

```c
uint32_t dieId = 0;
HcommCcuResDescHandle resDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(dieId, &resDesc);
if (ret != CCU_SUCCESS) {
    // Error handling.
    return ret;
}

// Use resDesc to set resource specifications, query remaining resources, and so on ...
// ...

HcommCcuInsResDescDestroy(resDesc);
return CCU_SUCCESS;
```
