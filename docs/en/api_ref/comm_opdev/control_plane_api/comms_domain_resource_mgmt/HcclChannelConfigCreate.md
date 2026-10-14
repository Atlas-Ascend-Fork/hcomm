# HcclChannelConfigCreate

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:21:17.709Z pushedAt=2026-09-29T11:20:39.363Z -->

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

Creates a channel configuration object, which is used to pass advanced configurations such as shared Jetty when the [HcclChannelAcquireWithConfig](HcclChannelAcquireWithConfig.md) API creates a communication channel.

After the configuration object is created, you can set its properties through [HcclChannelConfigSetInt](HcclChannelConfigSetInt.md) and [HcclChannelConfigSetStr](HcclChannelConfigSetStr.md). After use, you must destroy it through [HcclChannelConfigDestroy](HcclChannelConfigDestroy.md).

## Function Prototype

```c
HcclResult HcclChannelConfigCreate(HcclChannelConfig *config)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| config | Output | Opaque handle to the channel configuration object.<br>For the definition of the HcclChannelConfig type, see [HcclChannelConfig](../../datatype_definition/HcclChannelConfig.md).<br>The caller only needs to pass in a pointer; the API allocates and returns the handle internally. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- The caller must ensure that the passed-in **config** pointer is valid.
- After use, the created configuration object must be destroyed by calling [HcclChannelConfigDestroy](HcclChannelConfigDestroy.md); otherwise, a memory leak occurs.
- The configuration object can be destroyed immediately after [HcclChannelAcquireWithConfig](HcclChannelAcquireWithConfig.md) completes, without affecting the created channels.

## Example

```c
HcclChannelConfig config = nullptr;
HcclResult ret = HcclChannelConfigCreate(&config);
if (ret != HCCL_SUCCESS) {
    printf("Failed to create channel config, ret = %d\n", ret);
    return ret;
}

// Set the shared queue attributes.
HcclChannelConfigSetInt(config, HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);
HcclChannelConfigSetStr(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, "my_shared_tag");

// Create a channel using the configuration.
HcclChannelAcquireWithConfig(comm, engine, channelDescs, channelNum, config, channels);

// Destroy the configuration object.
HcclChannelConfigDestroy(config);
```
