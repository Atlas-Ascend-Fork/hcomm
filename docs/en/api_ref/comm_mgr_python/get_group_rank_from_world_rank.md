# get_group_rank_from_world_rank

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:44:34.482Z pushedAt=2026-09-29T02:47:55.205Z -->

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

Obtains the group rank ID of a process in a group based on its world rank ID.

## Function Prototype

```python
def get_group_rank_from_world_rank(world_rank_id, group)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| world_rank_id | Input | Int type.<br>Rank ID of the process in **hccl_world_group**. |
| group | Input | String type, with a maximum length of 128 bytes including the terminator.<br>Group name, which can be a custom group or **hccl_world_group**. |

## Return Value

Int type. The rank ID of the process in the group is returned on success.

## Constraints

- This API must be called after collective communication initialization is complete.
- The rank that calls this API must be within the range defined by the group parameter of the current API. If a rank outside this range calls this API, the call fails.
- After [create_group](create_group.md) is complete, call this API to convert a world rank ID to a group rank ID.
- This API and [get_world_rank_from_group_rank](get_world_rank_from_group_rank.md) are inverse operations of each other. Note the difference in parameter order: this API uses the parameter order (**world_rank_id**, **group**), while its inverse operation uses (**group**, **group_rank_id**).

## Example

```python
from hccl.manage.api import create_group
from hccl.manage.api import get_group_rank_from_world_rank
create_group("myGroup", 4, [0, 1, 2, 3])
groupRankId = get_group_rank_from_world_rank(1, "myGroup")
```
