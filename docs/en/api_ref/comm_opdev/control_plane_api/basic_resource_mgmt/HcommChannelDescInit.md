# HcommChannelDescInit

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:53:59.310Z pushedAt=2026-09-29T03:52:51.183Z -->

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
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->

## Description

Initializes a communication channel description list.

## Function Prototype

```c
HcommResult HcommChannelDescInit(HcommChannelDesc *channelDesc, uint32_t descNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| channelDesc | Input/Output | Communication channel description list. The list length is **descNum**. The function initializes this structure.<br>For details about the HcommChannelDesc type, see [HcommChannelDesc](../../datatype_definition/HcommChannelDesc.md). |
| descNum | Input | Number of communication channel descriptions. |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

- The **HcommChannelDesc** structure must be initialized by calling this API.

## Example

The following example initializes a communication channel description list with two communication channel descriptions:

```c
uint32_t channelNum = 2;
std::vector<HcommChannelDesc> channelDesc(channelNum);
HcommChannelDescInit(channelDesc.data(), channelNum);
```
