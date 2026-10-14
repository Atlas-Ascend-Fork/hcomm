# get_local_rank_size

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:44:24.169Z pushedAt=2026-09-29T02:50:50.303Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
<!-- end id5 -->

## Description

Obtains the number of local ranks on the server where the device in the group resides.

## Function Prototype

```python
def get_local_rank_size(group="hccl_world_group")
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| group | Input | String type, with a maximum length of 128 bytes including the terminator.<br>Group name. If this parameter is not configured, the default value **hccl_world_group** is used. |

## Return Value

Int type. The API returns the number of local ranks on the server where the device resides.

## Constraints

- This API must be called after collective communication initialization is complete.
- The rank that calls this API must be within the range defined by the group parameter of the current API. If a rank outside this range calls this API, the call fails.
- After [create_group](create_group.md) is complete, call this API to obtain the number of local ranks in this group.
- If **hccl_world_group** is passed in, the number of local ranks in **world_group** is returned.

## Example

```python
from hccl.manage.api import create_group
from hccl.manage.api import get_local_rank_size
create_group("myGroup", 4, [0, 1, 2, 3])  
localRankSize = get_local_rank_size("myGroup")
```
