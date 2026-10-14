# HcommCcuInsResDescQueryDieId

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:14:51.390Z pushedAt=2026-09-29T09:31:51.941Z -->

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

Queries the IO die number to which a CCU resource descriptor belongs.

## Function Prototype

```c
CcuResult HcommCcuInsResDescQueryDieId(HcommCcuResDescHandle resDesc, uint32_t *dieId)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| resDesc | Input | Resource descriptor handle, created by `HcommCcuInsResDescCreate`. |
| dieId | Output | Returns the IO die number specified when the descriptor was created after a successful query. It cannot be a null pointer. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The query succeeds, and `*dieId` is the IO die number to which the descriptor belongs. |
| `CCU_E_PTR` | `dieId` is a null pointer. |
| `CCU_E_NOT_FOUND` | `resDesc` is not registered. |

## Constraints

- Before calling this API, you must create a resource descriptor by calling `HcommCcuInsResDescCreate`.
- This API can be called only on the host side.

## Dependencies

- Depends on the resource descriptor handle created by `HcommCcuInsResDescCreate`.
- Used by `HcommCcuQueryRemainResDesc` to obtain the die ID internally for querying the remaining resources of the corresponding die.

## Example

```c
HcommCcuResDescHandle resDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(1, &resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

uint32_t dieId = 0;
ret = HcommCcuInsResDescQueryDieId(resDesc, &dieId);
// dieId == 1

HcommCcuInsResDescDestroy(resDesc);
return ret;
```
