# HcclChannelConfigSetStr

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:23:20.821Z pushedAt=2026-09-29T11:30:18.872Z -->

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

Sets a string-type attribute of the channel configuration object. For the currently supported attribute types, see [HcclChannelConfigType](../../datatype_definition/HcclChannelConfigType.md).

## Function Prototype

```c
HcclResult HcclChannelConfigSetStr(HcclChannelConfig config, HcclChannelConfigType type, const char *value)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| config | Input | Channel configuration object handle, which must be created by [HcclChannelConfigCreate](HcclChannelConfigCreate.md). |
| type | Input | Attribute type enumeration value.<br>For the definition of the HcclChannelConfigType type, see [HcclChannelConfigType](../../datatype_definition/HcclChannelConfigType.md). |
| value | Input | Attribute value string, ending with '\0'. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- Neither **config** nor **value** can be **nullptr**; otherwise, a parameter error is returned.
- **type** must be a valid string-type attribute enumeration value; otherwise, a parameter error is returned.
- Currently supported string-type attributes:
  - **HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG**: tag identifier of the shared queue. It takes effect only when **IS_SHARED_QUEUE** is set to **true**. The tag cannot be empty, and its maximum byte length cannot exceed 255 (including '\0'); otherwise, a parameter error is returned.

## Example

```c
HcclChannelConfig config = nullptr;
HcclChannelConfigCreate(&config);
HcclChannelConfigSetInt(config, HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);

// Set the shared queue tag.
HcclChannelConfigSetStr(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, "group_a_tag");

HcclChannelAcquireWithConfig(comm, engine, channelDescs, channelNum, config, channels);
HcclChannelConfigDestroy(config);
```
