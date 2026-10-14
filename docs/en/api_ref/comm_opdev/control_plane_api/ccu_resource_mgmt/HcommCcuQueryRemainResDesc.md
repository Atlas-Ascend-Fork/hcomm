# HcommCcuQueryRemainResDesc

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:17:10.567Z pushedAt=2026-10-08T09:22:35.531Z -->

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

Queries the maximum number of remaining contiguous resources on the IO die corresponding to the CCU resource descriptor, and writes the query results of each resource type into the descriptor.

After the call, you can query the remaining number of each type of resources through `HcommCcuInsResDescQueryNum`, or adjust the allocated number of resources based on the query results and create a CCU instance.

## Function Prototype

```c
CcuResult HcommCcuQueryRemainResDesc(HcommCcuResDescHandle resDesc)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| resDesc | Input/Output | Resource descriptor handle, created by `HcommCcuInsResDescCreate`. The `dieId` specified at creation is used to select the IO die to be queried, and the query results overwrite the original resource quantity in the descriptor. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The query succeeds, and the maximum numbers of various types of remaining contiguous resources have been written to `resDesc`. |
| `CCU_E_PARA` | The IO die number in the descriptor corresponding to `resDesc` is out of the valid range. |
| `CCU_E_NOT_FOUND` | `resDesc` is not registered. |
| `CCU_E_UNAVAIL` | The specified IO die is not enabled. |
| `CCU_E_INTERNAL` | An internal error occurs during the query (for example, a resource specification query failure). |

## Constraints

- Before calling this API, you must create a resource descriptor by calling `HcommCcuInsResDescCreate`.
- When `resDesc` is created and this API is called, the current thread must be bound to the same NPU device.
- The caller must ensure that `resDesc` is not concurrently modified or destroyed during the query.
- The query result overwrites the existing numbers of resources in the descriptor. To retain the original configured values, the caller must back them up before the call.
- This API can be called only on the host side, not inside a kernel function body.

## Dependencies

- Depends on the resource descriptor handle created by `HcommCcuInsResDescCreate`.
- Works with `HcommCcuInsResDescQueryNum`: after the query returns the result, read the numbers of various types of remaining resources through **QueryNum**.

## Example

```c
uint32_t dieId = 0;
HcommCcuResDescHandle resDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(dieId, &resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

// Query the remaining resources on the IO die corresponding to dieId.
ret = HcommCcuQueryRemainResDesc(resDesc);
if (ret != CCU_SUCCESS) {
    HcommCcuInsResDescDestroy(resDesc);
    return ret;
}

// Read the remaining number of each resource type.
uint32_t loopRemain = 0;
HcommCcuInsResDescQueryNum(resDesc, HCOMM_CCU_RES_TYPE_LOOP, &loopRemain);

// To set a specific resource specification, adjust it based on the remaining quantity.
HcommCcuInsResDescSetNum(resDesc, HCOMM_CCU_RES_TYPE_LOOP, loopRemain / 2);

// Create a CCU instance based on the adjusted descriptor (omitted).

HcommCcuInsResDescDestroy(resDesc);
return ret;
```
