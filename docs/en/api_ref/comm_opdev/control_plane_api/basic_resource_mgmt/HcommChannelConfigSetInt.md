# HcommChannelConfigSetInt

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:52:01.660Z pushedAt=2026-09-29T03:30:11.551Z -->

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

Sets an integer attribute of a channel configuration object. For the currently supported attribute types, see [HcommChannelConfigType](../../datatype_definition/HcommChannelConfigType.md).

## Function Prototype

```c
HcommResult HcommChannelConfigSetInt(HcommChannelConfig config, HcommChannelConfigType type, uint32_t value)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| config | Input | Handle to the channel configuration object, which must be created through [HcommChannelConfigCreate](HcommChannelConfigCreate.md). |
| type | Input | Attribute type enumeration value.<br>For the definition of the HcommChannelConfigType type, see [HcommChannelConfigType](../../datatype_definition/HcommChannelConfigType.md). |
| value | Input | Attribute value. For a bool attribute, **0** indicates false and a non-zero value indicates true. |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

- **config** cannot be **nullptr**; otherwise, a parameter error is returned.
- **type** must be a valid integer attribute enumeration value; otherwise, a parameter error is returned.
- Currently supported integer attributes:
  - **HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE**: Whether to enable shared queue mode.

## Example

```c
HcommChannelConfig config = nullptr;
HcommChannelConfigCreate(&config);

// Enable shared queue mode.
HcommChannelConfigSetInt(config, HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);

HcommChannelCreateWithConfig(endpointHandle, COMM_ENGINE_AIV,
    channelDescs, channelNum, config, channels);
HcommChannelConfigDestroy(config);
```
