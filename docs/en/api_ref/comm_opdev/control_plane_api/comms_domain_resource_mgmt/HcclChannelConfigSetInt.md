# HcclChannelConfigSetInt

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:22:50.393Z pushedAt=2026-09-29T11:27:23.435Z -->

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

Sets an integer attribute of a channel configuration object. For the currently supported attribute types, see [HcclChannelConfigType](../../datatype_definition/HcclChannelConfigType.md).

## Function Prototype

```c
HcclResult HcclChannelConfigSetInt(HcclChannelConfig config, HcclChannelConfigType type, uint32_t value)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| config | Input | Channel configuration object handle, which must be created by [HcclChannelConfigCreate](HcclChannelConfigCreate.md). |
| type | Input | Attribute type enumeration value.<br>For the definition of the HcclChannelConfigType type, see [HcclChannelConfigType](../../datatype_definition/HcclChannelConfigType.md). |
| value | Input | Attribute value. For a bool type attribute, **0** indicates false and a non-zero value indicates true. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- **config** cannot be **nullptr**; otherwise, a parameter error is returned.
- **type** must be a valid integer attribute enumeration value; otherwise, a parameter error is returned.
- Currently supported integer attributes:
  - **HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE**: Whether to enable the shared queue mode.

## Example

```c
HcclChannelConfig config = nullptr;
HcclChannelConfigCreate(&config);

// Enable the shared queue mode.
HcclChannelConfigSetInt(config, HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);
HcclChannelConfigSetStr(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, "my_tag");

HcclChannelAcquireWithConfig(comm, engine, channelDescs, channelNum, config, channels);
HcclChannelConfigDestroy(config);
```
