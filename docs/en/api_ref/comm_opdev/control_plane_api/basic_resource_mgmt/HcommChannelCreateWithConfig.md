# HcommChannelCreateWithConfig

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:52:58.864Z pushedAt=2026-10-08T08:27:54.796Z -->

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

Creates a communication channel through a configuration object. It is an enhanced version of [HcommChannelCreate](HcommChannelCreate.md) that supports passing advanced configurations such as shared Jetty through the configuration object.

When **config** is **nullptr** or **IS_SHARED_QUEUE** is not set, the behavior is exactly equivalent to [HcommChannelCreate](HcommChannelCreate.md).

When **IS_SHARED_QUEUE** is set to **true** in **config**, the shared Jetty mode is enabled:
- For channels created by calling this API multiple times with the same **endpointHandle**, if the source and destination **endpointPair**s are the same, the same underlying Jetty resource is shared, enabling reuse of communication resources.
- It is suitable for scenarios that require frequent creation/destruction of communication channels, significantly reducing the link establishment overhead.

After this API is executed, only the channel objects are created, and **link connections are not established immediately**. The caller needs to subsequently drive the link establishment state machine through the [HcommChannelGetStatus](HcommChannelGetStatus.md) API and perform communication operations after the channel state becomes ready.

## Function Prototype

```c
HcommResult HcommChannelCreateWithConfig(EndpointHandle endpointHandle, CommEngine engine,
    HcommChannelDesc *channelDescs, uint32_t channelNum, HcommChannelConfig config, ChannelHandle *channels)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| endpointHandle | Input | Handle of the network device endpoint, which identifies a created local network device endpoint.<br>For the definition of the EndpointHandle type, see [EndpointHandle](../../datatype_definition/EndpointHandle.md). This handle must be successfully created through [HcommEndpointCreate](HcommEndpointCreate.md) and must not be destroyed. |
| engine | Input | Communication engine type, which specifies the execution location of the channel.<br>For the definition of the CommEngine type, see [CommEngine](../../datatype_definition/CommEngine.md). |
| channelDescs | Input | Array of channel descriptors, where each element describes the attribute information of a channel to be created.<br>For the definition of the HcommChannelDesc type, see [HcommChannelDesc](../../datatype_definition/HcommChannelDesc.md). |
| channelNum | Input | Number of channels to be created. Value range: [1, 1048576]. |
| config | Input | Pointer to the channel configuration object, which can be **nullptr** (equivalent to [HcommChannelCreate](HcommChannelCreate.md)).<br>For the definition of the HcommChannelConfig type, see [HcommChannelConfig](../../datatype_definition/HcommChannelConfig.md).<br>Created through [HcommChannelConfigCreate](HcommChannelConfigCreate.md). |
| channels | Output | Array of channel handles, used to return the list of handles of successfully created channels.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../datatype_definition/ChannelHandle.md).<br>An array allocated by the caller, which must contain space for at least **channelNum** elements. |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

1. The length of the **channelDescs** array must be consistent with the **channelNum** parameter.

2. When **config** is **nullptr**, the behavior is completely equivalent to [HcommChannelCreate](HcommChannelCreate.md).

3. When **IS_SHARED_QUEUE** is set to **true**, the following additional constraints apply:
   - Only the UB network semantic protocols (COMM_PROTOCOL_UB_CTP/COMM_PROTOCOL_UB_RTP) are supported. UBMem/RoCE/UBOE are not supported.
   - Channels created by calling this API multiple times with the same **endpointHandle** share the same Jetty.
   - Channels created by calling this API with different **endpointHandle** values do not share a Jetty.
   - Different channels sharing a Jetty do not support concurrent use. The caller must call them serially in business order.
   - Before destroying **endpointHandle**, ensure that all channels sharing the Jetty have been destroyed.

4. The **config** object can be destroyed through [HcommChannelConfigDestroy](HcommChannelConfigDestroy.md) immediately after this API call completes, without affecting the created channels.

5. The communication protocols supported by each CommEngine depend on the chip model, as described below:

   <!-- npu="950" id6 -->
   For Ascend 950PR&950DT products, only the UB network semantic protocols (UB_CTP/UB_RTP) of the AIV engine are supported:
   - COMM_ENGINE_AIV
     - COMM_PROTOCOL_UB_CTP
     - COMM_PROTOCOL_UB_RTP
   <!-- end id6 -->

## Example

The following example creates an AIV communication channel in shared Jetty mode:

```c
EndpointHandle endpointHandle = nullptr;
// ... Code for creating the endpoint (omitted)

// 1. Create and configure the channel configuration object.
HcommChannelConfig config = nullptr;
HcommChannelConfigCreate(&config);
HcommChannelConfigSetInt(config, HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);

// 2. Prepare the channel descriptor.
const uint32_t CHANNEL_NUM = 4;
HcommChannelDesc channelDescs[CHANNEL_NUM] = {0};
ChannelHandle channels[CHANNEL_NUM] = {0};
for (uint32_t i = 0; i < CHANNEL_NUM; i++) {
    HcommChannelDescInit(&channelDescs[i], 1);
    channelDescs[i].remoteEndpoint.protocol = COMM_PROTOCOL_UB_CTP;
    // Populate localEndpoint / remoteEndpoint ...
}

// 3. Create a communication channel using the configuration.
HcommResult ret = HcommChannelCreateWithConfig(endpointHandle, COMM_ENGINE_AIV,
    channelDescs, CHANNEL_NUM, config, channels);
if (ret != 0) {
    printf("Failed to create channels with config, ret = %d\n", ret);
    HcommChannelConfigDestroy(config);
    HcommEndpointDestroy(endpointHandle);
    return ret;
}

// 4. Destroy the configuration object (the channel has been created, so the configuration object is no longer needed).
HcommChannelConfigDestroy(config);

// 5. Drive the link establishment state machine through HcommChannelGetStatus subsequently.
```
