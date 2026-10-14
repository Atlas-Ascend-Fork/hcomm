# HcommCcuInsResDescQueryNum

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:15:45.441Z pushedAt=2026-09-29T09:35:14.055Z -->

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

Queries the number of resources by resource type that have been set in a CCU resource descriptor.

## Function Prototype

```c
CcuResult HcommCcuInsResDescQueryNum(HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t *resNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| resDesc | Input | Resource descriptor handle, created by `HcommCcuInsResDescCreate`. |
| resType | Input | Resource type, which takes an [HcommCcuResType](../../datatype_definition/HcommCcuResType.md) enumeration value. |
| resNum | Output | Returns the number of resources of this resource type that have been set after a successful query. It cannot be a null pointer. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The query succeeds, and `*resNum` is the number of resources that have been set. |
| `CCU_E_PARA` | `resType` is out of the valid range. |
| `CCU_E_PTR` | `resNum` is a null pointer. |
| `CCU_E_NOT_FOUND` | `resDesc` is not registered. |

## Constraints

- Before calling this API, you must create a resource descriptor by calling `HcommCcuInsResDescCreate`.
- If `HcommCcuInsResDescSetNum` has never been called to set this resource type, the query result is 0.
- This API can be called only on the host side.

## Dependencies

- Depends on the resource descriptor handle created by `HcommCcuInsResDescCreate`.
- Depends on `HcommCcuInsResDescSetNum` to set the number of resources.

## Example

```c
HcommCcuResDescHandle resDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(0, &resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

HcommCcuInsResDescSetNum(resDesc, HCOMM_CCU_RES_TYPE_LOOP, 8);

uint32_t loopNum = 0;
ret = HcommCcuInsResDescQueryNum(resDesc, HCOMM_CCU_RES_TYPE_LOOP, &loopNum);
// loopNum == 8

HcommCcuInsResDescDestroy(resDesc);
return ret;
```
