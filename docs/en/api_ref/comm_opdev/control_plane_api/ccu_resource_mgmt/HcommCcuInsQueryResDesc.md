# HcommCcuInsQueryResDesc

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:13:16.474Z pushedAt=2026-09-29T09:09:57.549Z -->

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

Queries the actual number of various resources occupied by a CCU instance on a specified IO die, and writes the results into the resource descriptor provided by the caller.

The IO die to be queried is specified by the `dieId` passed in when creating `resDesc`. After the API returns success, the query results can be read type by type through [HcommCcuInsResDescQueryNum](HcommCcuInsResDescQueryNum.md).

## Function Prototype

```c
CcuResult HcommCcuInsQueryResDesc(CcuInsHandle ccuInsHandle, HcommCcuResDescHandle resDesc)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| ccuInsHandle | Input | Handle of the CCU instance to be queried, which cannot be `0`. |
| resDesc | Input/Output | Resource descriptor handle, which cannot be `0`. It must be created on the current device through [HcommCcuInsResDescCreate](HcommCcuInsResDescCreate.md). The `dieId` specified at creation is used to select the IO die to be queried, and the query results overwrite the original resource counts in the descriptor. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The query succeeds, and the resource usage of the instance on the specified IO die has been written to `resDesc`. |
| `CCU_E_PARA` | `ccuInsHandle` or `resDesc` is `0`, or the IO die number in `resDesc` is invalid. |
| `CCU_E_NOT_FOUND` | The corresponding CCU instance or resource descriptor does not exist on the current device. |
| `CCU_E_INTERNAL` | An internal error occurs while querying or writing the resource counts. |

## Constraints

- Before calling this API, create `resDesc` by calling [HcommCcuInsResDescCreate](HcommCcuInsResDescCreate.md), and use its `dieId` to specify the IO die to be queried.
- The thread that calls this API must be bound to the same NPU device used when creating the CCU instance and `resDesc`.
- The query result overwrites the original resource counts of each type in `resDesc`, but does not modify its IO die number.
- For resources allocated by alignment granularity, this API returns the number of resources actually occupied by the instance, which may be greater than the number of resources allocated when the instance was created.
- The caller must ensure that `ccuInsHandle` and `resDesc` are not concurrently destroyed or modified during the query.
- This API can be called only on the host side.

## Example

```c
HcommCcuResDescHandle queryDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(0, &queryDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

ret = HcommCcuInsQueryResDesc(insHandle, queryDesc);
if (ret != CCU_SUCCESS) {
    HcommCcuInsResDescDestroy(queryDesc);
    return ret;
}

uint32_t loopNum = 0;
ret = HcommCcuInsResDescQueryNum(
    queryDesc, HCOMM_CCU_RES_TYPE_LOOP, &loopNum);

HcommCcuInsResDescDestroy(queryDesc);
return ret;
```
