# destroy_group

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:43:09.112Z pushedAt=2026-09-29T02:45:28.936Z -->

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

Destroys a custom collective communication group.

## Function Prototype

```python
def destroy_group(group)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| group | Input | String type, with a maximum length of 128 bytes including the terminator.<br>Group name, which is the identifier of the collective communication group. |

## Return Value

None

## Constraints

- This API must be called after collective communication initialization is complete.
- The rank that calls this API must be within the range defined by the group parameter of the current API. If a rank outside this range calls this API, the call fails.
- For groups with the same name, [destroy_group](destroy_group.md) and [create_group](create_group.md) must be used together, and **destroy_group** must be called after **create_group** is complete.
- If the group passed by the user is **hccl_world_group** (the default group), destroying the group fails.

## Example

```python
from hccl.manage.api import create_group
from hccl.manage.api import destroy_group
create_group("myGroup", 4, [0, 1, 2, 3])
destroy_group("myGroup")
```
