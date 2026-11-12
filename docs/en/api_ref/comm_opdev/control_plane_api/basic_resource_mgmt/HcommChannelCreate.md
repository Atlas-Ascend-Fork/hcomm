# HcommChannelCreate

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:52:45.590Z pushedAt=2026-09-29T03:39:23.628Z -->

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

Creates communication channels. This API is a resource management API. Based on the created network endpoints, it creates communication channels in batches according to the given channel description information, providing the data transmission infrastructure for point-to-point communication or collective communication.

After this API is executed, only the channel objects are created, and **link connections are not established immediately**. The caller needs to subsequently drive the link establishment state machine through the [HcommChannelGetStatus](HcommChannelGetStatus.md) API and perform communication operations after the channel state becomes ready.

## Function Prototype

```c
HcommResult HcommChannelCreate(EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc *channelDescs, uint32_t channelNum, ChannelHandle *channels);
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| endpointHandle | Input | Handle of the network device endpoint, which identifies a created local network device endpoint.<br>For the definition of the EndpointHandle type, see [EndpointHandle](../../datatype_definition/EndpointHandle.md). This handle must be successfully created through [HcommEndpointCreate](HcommEndpointCreate.md) and must not be destroyed. |
| engine | Input | Communication engine type, which specifies the execution location of the channel.<br>For the definition of the CommEngine type, see [CommEngine](../../datatype_definition/CommEngine.md).<br>Note: It must be a valid engine type. |
| channelDescs | Input | Array of channel descriptors, where each element describes the attribute information of a channel to be created.<br>For the definition of the HcommChannelDesc type, see [HcommChannelDesc](../../datatype_definition/HcommChannelDesc.md).<br>The number of array elements must be equal to **channelNum**, and each element must be correctly filled with the required fields. |
| channelNum | Input | Number of channels to be created.<br>Unit: count, value range: [1, 1048576].<br>This parameter must be greater than 0. |
| channels | Output | Array of channel handles, used to return the list of handles of successfully created channels.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../datatype_definition/ChannelHandle.md).<br>An array allocated by the caller, which must contain space for at least **channelNum** elements. |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

- The length of the **channelDescs** array must be consistent with the **channelNum** parameter.
- The **remoteEndpoint** in **HcommChannelDesc** must be correctly filled with the remote endpoint information.
- When **exchangeAllMems** in **HcommChannelDesc** is **false**, **memHandles** and **memHandleNum** must be configured.
- In the AIV direct URMA scenario, the number of memories participating in the exchange on the local and remote ends must be consistent; otherwise, channel creation fails.
- When the current **CommEngine** is configured as CCU, only one **memHandle** can be exchanged.
- When the current **CommEngine** is configured as CCU, external configuration of **NotifyNum** is not supported, and the default is 4 CCU Notify.
- The communication protocols supported by each **CommEngine** depend on the chip model, as described below:

  <!-- npu="950" id6 -->
  For Ascend 950PR&950DT products, the communication protocols supported by each communication engine are as follows:

  - COMM_ENGINE_CPU
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_UB_CTP
  - COMM_ENGINE_AICPU_TS
    - COMM_PROTOCOL_UBOE
    - COMM_PROTOCOL_UB_CTP
    - COMM_PROTOCOL_ROCE
  - COMM_ENGINE_AIV
    - COMM_PROTOCOL_UB_CTP
    - COMM_PROTOCOL_ROCE
  <!-- end id6 -->

  <!-- npu="A3" id7 -->
  For Atlas A3 products, the communication protocols supported by each communication engine are as follows:

  - COMM_ENGINE_AICPU_TS
    - COMM_PROTOCOL_HCCS
    - COMM_PROTOCOL_ROCE

  - Only DEVICE-side endpoints can be used to create channels. For the protocols supported by endpoints, see [HcommEndpointCreate](HcommEndpointCreate.md).
  - The HCCS protocol is used for link establishment between chips within the same host, and the RoCE protocol is used for cross-host link establishment. The listening port for link establishment is specified by the port field in HcommChannelDesc; when port is set to 0, the default port 16666 is used. Cross-host link establishment requires the peer end to complete listening first and the network to be reachable.
  - The COMM_ENGINE_CPU, COMM_ENGINE_CPU_TS, COMM_ENGINE_AIV, and COMM_ENGINE_CCU engines are not supported for channel creation.
  - When channels are created based on endpoints of the NIC plugin type, the supported engine and protocol combinations depend on the plugin implementation.
  <!-- end id7 -->

  <!-- npu="910b" id8 -->
  For Atlas A2 products, the communication protocols supported by each communication engine are as follows:

  - COMM_ENGINE_AICPU_TS
    - COMM_PROTOCOL_HCCS
    - COMM_PROTOCOL_ROCE

  - Only DEVICE-side endpoints can be used to create channels. For the protocols supported by endpoints, see [HcommEndpointCreate](HcommEndpointCreate.md).
  - The HCCS protocol is used for link establishment between chips within the same host, and the RoCE protocol is used for cross-host link establishment. The listening port for link establishment is specified by the port field in HcommChannelDesc; when port is set to 0, the default port 16666 is used. Cross-host link establishment requires the peer end to complete listening first and the network to be reachable.
  - The COMM_ENGINE_CPU, COMM_ENGINE_CPU_TS, COMM_ENGINE_AIV, and COMM_ENGINE_CCU engines are not supported for channel creation.
  - When channels are created based on endpoints of the NIC plugin type, the supported engine and protocol combinations depend on the plugin implementation.
  <!-- end id8 -->

  **Note:**
    - The Atlas 350 accelerator card does not support the **COMM_ENGINE_CPU** communication engine and its corresponding communication protocols.
    - The Atlas 350 accelerator card does not support the **COMM_PROTOCOL_ROCE** communication protocol of the **COMM_ENGINE_AICPU_TS** communication engine.

## Example

```c
EndpointHandle endpointHandle = nullptr;
 // ... Code for creating endpoints (omitted).

 // Create multiple channels.
 const uint32_t CHANNEL_NUM = 4;
 HcommChannelDesc channelDescs[CHANNEL_NUM] = {0};
 ChannelHandle channels[CHANNEL_NUM] = {0};

 // Prepare the channel descriptors and create the channels.
 for (uint32_t i = 0; i < CHANNEL_NUM; i++) {
     // ... Fill in channelDescs[i].
 }

 HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_CPU,
                                      channelDescs, CHANNEL_NUM, channels);
 if (ret != 0) {
     printf("Failed to create channels, ret = %d\n", ret);
     HcommEndpointDestroy(endpointHandle);
     return ret;
 }

 printf("%u channels created successfully\n", CHANNEL_NUM);
```
