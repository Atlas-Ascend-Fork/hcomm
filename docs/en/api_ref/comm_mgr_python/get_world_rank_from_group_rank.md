# get_world_rank_from_group_rank

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:46:19.902Z pushedAt=2026-09-29T02:55:49.892Z -->

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

Obtains the corresponding world rank ID based on the rank ID of a process in a group.

## Function Prototype

```python
def get_world_rank_from_group_rank(group, group_rank_id)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| group | Input | String type, with a maximum length of 128 bytes including the terminator.<br>Group name, which can be a custom group or **hccl_world_group**. |
| group_rank_id | Input | Int type.<br>The rank ID of the process in the group. |

## Return Value

Int type. The API returns the rank ID of the process in the global group (**hccl_world_group**).

## Constraints

- This API must be called after collective communication initialization is complete.
- The rank that calls this API must be within the range defined by the group parameter of the current API. If a rank outside this range calls this API, the call fails.
- After [create_group](create_group.md) is complete, call this API to convert a group rank ID to a world rank ID.
- This API and [get_group_rank_from_world_rank](get_group_rank_from_world_rank.md) are inverse operations of each other. Note the difference in parameter order: this API uses the parameter order (**group**, **group_rank_id**), while its inverse operation uses (**world_rank_id**, **group**).

## Example

```python
from hccl.manage.api import create_group
from hccl.manage.api import get_world_rank_from_group_rank
create_group("myGroup", 4, [0, 1, 2, 3])
worldRankId = get_world_rank_from_group_rank("myGroup", 1)
```
