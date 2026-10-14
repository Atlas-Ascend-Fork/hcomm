# HcclChannelAcquireWithConfig

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:45:55.351Z pushedAt=2026-10-12T06:12:35.590Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->

## Description

Creates a communication channel based on a communicator and a configuration object. It is an enhanced version of [HcclChannelAcquire](HcclChannelAcquire.md) that supports advanced configurations such as shared Jetty passed through the configuration object.

When **config** is **nullptr** or **IS_SHARED_QUEUE** is not set, the behavior is completely equivalent to [HcclChannelAcquire](HcclChannelAcquire.md).

When **IS_SHARED_QUEUE** is set to **true** in **config**, the shared Jetty mode is enabled:
- For channels created by calling this API multiple times with the same **SHARED_QUEUE_TAG**, if the source and destination **endpointPair** objects are the same, they share the same underlying Jetty resource, enabling reuse of communication resources.
- It is suitable for scenarios that require frequent creation/destruction of communication channels, significantly reducing link establishment overhead.

## Function Prototype

```c
HcclResult HcclChannelAcquireWithConfig(HcclComm comm, CommEngine engine,
    const HcclChannelDesc *channelDescs, uint32_t channelNum, HcclChannelConfig config, ChannelHandle *channels)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| engine | Input | Communication engine type.<br>For the definition of the CommEngine type, see [CommEngine](../../datatype_definition/CommEngine.md). |
| channelDescs | Input | Communication channel description list, with a list length of **channelNum**.<br>For the definition of the HcclChannelDesc type, see [HcclChannelDesc](../../datatype_definition/HcclChannelDesc.md).<br>**[HcclChannelDescInit](HcclChannelDescInit.md) must be used for initialization.** |
| channelNum | Input | Number of communication channels, with a value range of (0, 1024 * 1024]. |
| config | Input | Pointer to the channel configuration object, which can be **nullptr** (equivalent to [HcclChannelAcquire](HcclChannelAcquire.md)).<br>For the definition of the HcclChannelConfig type, see [HcclChannelConfig](../../datatype_definition/HcclChannelConfig.md).<br>It is created with [HcclChannelConfigCreate](HcclChannelConfigCreate.md). |
| channels | Output | Communication channel handle list, with a list length of **channelNum**. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

1. **channelDescs** must be initialized with [HcclChannelDescInit](HcclChannelDescInit.md).

2. When **config** is **nullptr**, the behavior is completely equivalent to [HcclChannelAcquire](HcclChannelAcquire.md).

3. When **IS_SHARED_QUEUE** is set to **true**, the following additional constraints apply:
   - Only the AIV engine is supported.
   - Only the UB network semantic protocols (COMM_PROTOCOL_UB_CTP/COMM_PROTOCOL_UB_RTP) are supported; UBMem/RoCE/UBOE are not supported.
   - **SHARED_QUEUE_TAG** must be set (a non-empty string).
   - The **localEndpoint** of all elements in **channelDescs** must be the same (a reused Jetty can be associated with only one endpoint).
   - Only V2 communicators are supported (the communicator must be created through V2 APIs such as **HcclCommInitClusterInfoConfig**).
   - Different channels sharing a Jetty do not support concurrent use. The caller must call them serially in business order.

4. In shared Jetty mode, the reuse rules for channels with the same **tag**+**endpointPair** are as follows:
   - On repeated call, if the number of existing channels is insufficient, new ones are created to supplement them; if sufficient, the existing channels are returned in order.
   - Channel destruction is managed uniformly by the communicator, and the caller does not need to destroy them separately.

5. After this API call completes, the **config** object can be destroyed through [HcclChannelConfigDestroy](HcclChannelConfigDestroy.md) without affecting the created channels.
6. **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

The following example creates an AIV communication channel in shared Jetty mode:

```c
// 1. Create and configure the channel configuration object.
HcclChannelConfig config = nullptr;
HcclChannelConfigCreate(&config);
HcclChannelConfigSetInt(config, HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);
HcclChannelConfigSetStr(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, "layer1_shared");

// 2. Prepare the channel descriptor.
const uint32_t CHANNEL_NUM = 4;
HcclChannelDesc channelDescs[CHANNEL_NUM];
ChannelHandle channels[CHANNEL_NUM] = {0};
for (uint32_t i = 0; i < CHANNEL_NUM; i++) {
    HcclChannelDescInit(&channelDescs[i], 1);
    channelDescs[i].remoteRank = remoteRanks[i];
    channelDescs[i].channelProtocol = COMM_PROTOCOL_UB_CTP;
    // Fill in localEndpoint/remoteEndpoint ...
}

// 3. Create the communication channel using the configuration.
HcclResult ret = HcclChannelAcquireWithConfig(comm, COMM_ENGINE_AIV,
    channelDescs, CHANNEL_NUM, config, channels);
if (ret != HCCL_SUCCESS) {
    printf("Failed to acquire channels with config, ret = %d\n", ret);
    HcclChannelConfigDestroy(config);
    return ret;
}

// 4. Destroy the configuration object (the channel has been created and the configuration object is no longer needed).
HcclChannelConfigDestroy(config);

// 5. It can be called repeatedly later; channels with the same tag reuse the underlying Jetty.
// HcclChannelAcquireWithConfig(comm, COMM_ENGINE_AIV,
//     channelDescs2, CHANNEL_NUM2, config2, channels2);
```
