# set_split_strategy_by_size

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:48:59.728Z pushedAt=2026-09-29T03:18:56.822Z -->

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

Sets the reverse gradient splitting strategy in a collective communication group based on the gradient data size percentage to implement allreduce fusion for collective communication performance tuning.

## Function Prototype

```python
def set_split_strategy_by_size(dataSizeList, group="hccl_world_group")
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| dataSizeList | Input | List type.<br>List of gradient parameter data size percentages.<br>  - The gradient data size percentage list must be non-negative, and the sum of all percentages in the gradient data size sequence must be 100.<br>  - Gradient splitting supports a maximum of 8 segments.<br>  - For example, if a model has 150 MB of gradient data in total and needs to be split into three segments of 90 MB, 30 MB, and 30 MB, set **dataSizeList = [60,20,20]**. |
| group | Input | String type, with a maximum length of 128 bytes including the terminator.<br>Group name. It can be **hccl_world_group** or a custom group. The default value is **hccl_world_group**. |

## Return Value

None

## Constraints

- This API must be called after collective communication initialization is complete.
- The rank that calls this API must be within the range defined by the group parameter of the current API. If a rank outside this range calls this API, the call fails.
- When the reverse gradient splitting strategy is set based on both the gradient data size percentage and the gradient index ID, the setting based on the gradient data size percentage takes precedence.
- If you do not call the gradient splitting API to set a splitting strategy, the default reverse gradient splitting strategy is used.

  Default splitting strategy: the optimal splitting position of ResNet50, that is, the gradient data is split into two segments by data size, with the first segment accounting for 96.54% and the second segment accounting for 3.46%.

## Example

```python
from hccl.split.api import *
set_split_strategy_by_size([60, 20, 20], "group")
```
