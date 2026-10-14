# HcommChannelConfigCreate

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:51:29.603Z pushedAt=2026-09-29T03:28:56.998Z -->

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

Creates a channel configuration object, which is used to pass advanced configurations such as shared Jetty when [HcommChannelCreateWithConfig](HcommChannelCreateWithConfig.md) creates a communication channel.

After the configuration object is created, you can set its properties through [HcommChannelConfigSetInt](HcommChannelConfigSetInt.md), and you must destroy it through [HcommChannelConfigDestroy](HcommChannelConfigDestroy.md) after use.

## Function Prototype

```c
HcommResult HcommChannelConfigCreate(HcommChannelConfig *config)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| config | Output | Opaque handle to the channel configuration object.<br>For the definition of the HcommChannelConfig type, see [HcommChannelConfig](../../datatype_definition/HcommChannelConfig.md).<br>The caller only needs to pass in a pointer; the API allocates and returns the handle internally. |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

- The caller must ensure that the passed-in **config** pointer is valid.
- After use, the created configuration object must be destroyed by calling [HcommChannelConfigDestroy](HcommChannelConfigDestroy.md); otherwise, a memory leak occurs.
- The configuration object can be destroyed immediately after [HcommChannelCreateWithConfig](HcommChannelCreateWithConfig.md) completes, without affecting the created channel.

## Example

```c
HcommChannelConfig config = nullptr;
HcommResult ret = HcommChannelConfigCreate(&config);
if (ret != 0) {
    printf("Failed to create channel config, ret = %d\n", ret);
    return ret;
}

// Set the shared queue attributes.
HcommChannelConfigSetInt(config, HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);

// Create a Channel using the configuration.
HcommChannelCreateWithConfig(endpointHandle, engine, channelDescs, channelNum, config, channels);

// Destroy the configuration object.
HcommChannelConfigDestroy(config);
```
