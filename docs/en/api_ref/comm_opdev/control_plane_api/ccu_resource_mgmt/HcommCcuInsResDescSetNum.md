# HcommCcuInsResDescSetNum

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:16:20.265Z pushedAt=2026-10-08T09:20:44.588Z -->

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

Sets the number of resources in the CCU resource descriptor by resource type. You can call this API multiple times to set the expected value for each type. Setting the value to **0** means that no resources of this type are allocated.

## Function Prototype

```c
CcuResult HcommCcuInsResDescSetNum(HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t resNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| resDesc | Input/Output | Resource descriptor handle, created by `HcommCcuInsResDescCreate`. After the setting succeeds, the quantity of the specified resource type is updated to `resNum`. |
| resType | Input | Resource type, which takes an enumeration value of [HcommCcuResType](../../datatype_definition/HcommCcuResType.md). |
| resNum | Input | Expected resource quantity. Setting the value to **0** means that no resources of this type are allocated. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The setting succeeds. |
| `CCU_E_PARA` | `resType` is out of the valid range. |
| `CCU_E_NOT_FOUND` | `resDesc` is not registered. |

## Constraints

- Before calling this API, create the resource descriptor by calling `HcommCcuInsResDescCreate`.
- If this API is called multiple times to set the same resource type, the value set by the later call overwrites the value set by the earlier call.
- This API performs a pure data write operation and does not verify whether `resNum` exceeds the hardware capacity limit. Capacity verification is performed when the descriptor is used to create a CCU instance or when remaining resources are queried.
- This API can be called only on the host side.

## Dependencies

- Depends on the resource descriptor handle created by `HcommCcuInsResDescCreate`.
- Depends on `HcommCcuInsResDescQueryNum` to query the configured number of resources.
- Works with `HcommCcuQueryRemainResDesc`: first query the remaining resources, then adjust the number of resources in the descriptor as needed.

## Example

```c
HcommCcuResDescHandle resDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(0, &resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

ret = HcommCcuInsResDescSetNum(resDesc, HCOMM_CCU_RES_TYPE_LOOP, 8);
if (ret != CCU_SUCCESS) {
    HcommCcuInsResDescDestroy(resDesc);
    return ret;
}
ret = HcommCcuInsResDescSetNum(resDesc, HCOMM_CCU_RES_TYPE_CCU_BUF, 16);

HcommCcuInsResDescDestroy(resDesc);
return ret;
```
