# HcclTeamDestroy

<!-- md-trans-meta sourceCommit=b0b5276bf70c24e578edb2b23d612d0b28163287 translatedAt=2026-10-08T11:51:10.896Z pushedAt=2026-10-12T06:12:35.602Z -->

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

Destroys a team and releases its resources. During destruction, the local memory of the team's **syncMem** is released and the memory handle of **syncMem** is deregistered.

## Function Prototype

```c
HcclResult HcclTeamDestroy(HcommTeamHandle team)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| team | Input | Handle of the team to be destroyed, which cannot be **NULL**. It can be created by [HcclTeamCreate](HcclTeamCreate.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

1. During destruction, the local memory of the team's **syncMem** is released, the **CommRegMem** memory handle of **syncMem** is deregistered, and the registration entries are cleared.

2. The team handle cannot be used after it is destroyed.

3. This API does not support concurrent calls by multiple threads. The caller must ensure that calls to this API from different threads are executed serially.

## Example

```c
HcclResult ret = HcclTeamDestroy(team);
if (ret != HCCL_SUCCESS) {
    printf("HcclTeamDestroy failed, ret = %d\n", ret);
}
```
