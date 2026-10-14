# HcclTeamCreateDescInit

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:33:42.754Z pushedAt=2026-09-30T02:05:53.512Z -->

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

Initializes a team creation descriptor. This API first fills the entire structure with 0xFF, and then sets the ABI header (**version**/**magicWord**/**size**) and the default values of each service field.

## Function Prototype

```c
HcclResult HcclTeamCreateDescInit(HcclTeamCreateDesc *desc)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| desc | Output | Team creation descriptor to be initialized, which cannot be **NULL**. For the definition of the **HcclTeamCreateDesc** type, see [HcclTeamCreateDesc](../../datatype_definition/HcclTeamCreateDesc.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and **HCCL_E_PTR** when **desc** is **NULL**.

## Constraints

Before calling [HcclTeamCreate](HcclTeamCreate.md), you must initialize the **HcclTeamCreateDesc** structure through this API. After initialization, you need to fill in the service fields such as **rankIds**, **rankNum**, **selfRankId**, **netLayer**, **protocol**, **requirement**, **engine**, **notifyNum**, and **channelCnt**. If a shared queue is required, you also need to fill in the **isSharedQueue** and **sharedQueueTag** fields.

## Example

```c
HcclTeamCreateDesc desc;
HcclResult ret = HcclTeamCreateDescInit(&desc);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// Fill in the service fields.
uint32_t rankIds[2] = {0, 1};
desc.rankIds = rankIds;
desc.rankNum = 2;
desc.selfRankId = 1;
desc.netLayer = 0;
desc.protocol = COMM_PROTOCOL_UB_CTP;
desc.requirement.signalCount = 0;
desc.requirement.counterCount = 0;
desc.requirement.barrierCount = 1;
desc.engine = COMM_ENGINE_AIV;
desc.notifyNum = 8;
desc.channelCnt = 1;
```
