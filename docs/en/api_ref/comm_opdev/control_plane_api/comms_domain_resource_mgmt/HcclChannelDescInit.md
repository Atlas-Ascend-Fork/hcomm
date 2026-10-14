# HcclChannelDescInit

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:23:59.131Z pushedAt=2026-09-29T11:31:39.164Z -->

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

Initializes a communication channel descriptor list.

## Function Prototype

```c
HcclResult HcclChannelDescInit(HcclChannelDesc *channelDesc, uint32_t descNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| channelDesc | Output | Communication channel descriptor list. The list length is **descNum**. This API initializes the structure.<br>For details about the HcclChannelDesc type, see [HcclChannelDesc](../../datatype_definition/HcclChannelDesc.md). |
| descNum | Input | Number of communication channel descriptors. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

The **HcclChannelDesc** structure must be initialized by calling this API.

## Example

The following example initializes a communication channel descriptor list with two communication channel descriptors:

```c
uint32_t channelNum = 2;
std::vector<HcclChannelDesc> channelDesc(channelNum);
HcclChannelDescInit(channelDesc.data(), channelNum);
```
