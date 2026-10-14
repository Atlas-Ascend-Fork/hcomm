# HcclGroupStatusGet

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:25:49.194Z pushedAt=2026-09-29T01:43:58.517Z -->

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
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Obtains the group feature status and determines whether this feature is enabled.

## Function Prototype

```c
HcclResult HcclGroupStatusGet(bool *isGroupEnabled)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| isGroupEnabled | Output | Group status. **TRUE** indicates enabled, and **FALSE** indicates not enabled. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

None

## Example

```c
bool isGroupEnabled = false;
HCCLCHECK(HcclGroupStatusGet(&isGroupEnabled));
```
