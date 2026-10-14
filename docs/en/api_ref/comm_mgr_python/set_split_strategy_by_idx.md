# set_split_strategy_by_idx

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:48:44.825Z pushedAt=2026-09-29T03:15:14.526Z -->

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

Sets the reverse gradient splitting strategy in a collective communication group based on gradient index IDs to implement allreduce fusion, for performance tuning of collective communication.

## Function Prototype

```python
def set_split_strategy_by_idx(idxList, group="hccl_world_group")
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| idxList | Input | List type.<br>List of gradient index IDs.<br>  - The list of gradient index IDs must be non-negative, in ascending sequence.<br>  - The gradient index IDs must be set based on the total number of gradient parameters of the model. Index IDs start from 0, and the maximum value can be obtained as follows: perform training without calling the gradient splitting API to set a gradient splitting strategy. In this case, the script uses the default gradient splitting method in [set_split_strategy_by_size](set_split_strategy_by_size.md) for training. After training, search for the keyword "segment result" in the host training log at the INFO level to obtain the gradient splitting segments, for example: segment index list: [0,107] [108,159]. The largest number in this segment sequence (for example, 159) is the maximum value of the total gradient parameter index. Note: During a complete training process, logs may be overwritten. In this case, you can modify the **LogAgentMaxFileNum** configuration item in **/var/log/npu/conf/slog/slog.conf** to increase the number of log files retained on the host side. Alternatively, you can perform only one iteration of training.<br>  - Gradient splitting supports a maximum of 8 segments.<br>  - For example, if the model has 160 parameters that generate gradients in total and needs to be split into three segments [0,20], [21,100], and [101,159], you can set **idxList=[20,100,159]**. |
| group | Input | String type, with a maximum length of 128 bytes including the terminator.<br>Group name. It can be **hccl_world_group** or a custom group. The default value is **hccl_world_group**. |

## Return Value

None

## Constraints

- This API must be called after collective communication initialization is complete.
- The rank that calls this API must be within the range defined by the group parameter of the current API. If a rank outside this range calls this API, the call fails.
- If you do not call the gradient splitting API to set a splitting strategy, the default reverse gradient splitting strategy is used.

  Default splitting strategy: The gradient is split into two segments by data size. The first segment accounts for 96.54% of the data, and the second segment accounts for 3.46% (in some cases, the gradient may be split into a single segment).

## Example

```python
from hccl.split.api import *
set_split_strategy_by_idx([20, 100, 159], "group")
```
