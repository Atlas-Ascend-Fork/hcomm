# HcommCcuInsResDescDestroy

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:14:34.930Z pushedAt=2026-09-29T09:29:52.811Z -->

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

Destroys the CCU resource descriptor created by `HcommCcuInsResDescCreate` and releases the memory it occupies. After destruction, the handle becomes invalid and must not be used again.

## Function Prototype

```c
CcuResult HcommCcuInsResDescDestroy(HcommCcuResDescHandle resDesc)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| resDesc | Input | Resource descriptor handle, created by `HcommCcuInsResDescCreate`. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | Destruction succeeded. |
| `CCU_E_NOT_FOUND` | `resDesc` is not registered, or has already been destroyed. |

## Constraints

- After destruction, the handle becomes permanently invalid and cannot be used with any API. Destroying it again returns `CCU_E_NOT_FOUND`.
- This API can only be called on the host side.

## Dependencies

- Depends on the resource descriptor handle created by `HcommCcuInsResDescCreate`.

## Example

```c
HcommCcuResDescHandle resDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(0, &resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

// Use resDesc ...

ret = HcommCcuInsResDescDestroy(resDesc);
return ret;
```
