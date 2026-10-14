# HcommChannelDestroy

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:54:03.118Z pushedAt=2026-09-29T03:58:05.657Z -->

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

Destroys communication channels. This API is a resource management API. It releases the communication channels created by [HcommChannelCreate](HcommChannelCreate.md) and all system resources occupied by them, including network connections, synchronization signals, and communication queues.

This API supports batch destruction and can release multiple channels in a single call, improving resource release efficiency.

## Function Prototype

```c
HcommResult HcommChannelDestroy(const ChannelHandle *channels, uint32_t channelNum);
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| channels | Input | Array of channel handles to be destroyed. Each element identifies a created communication channel.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../datatype_definition/ChannelHandle.md).<br>This parameter cannot be a null pointer. Each channel handle in the array must be a valid handle created by [HcommChannelCreate](HcommChannelCreate.md) (a handle that has not been destroyed). |
| channelNum | Input | Number of channels to be destroyed.<br>Unit: number. Value range: [1, 1048576].<br>This parameter must be greater than 0 and equal to the number of valid handles in the channels array. |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

- The length of the channels array must be consistent with the **channelNum** parameter.
- Channels created by different engines can be destroyed in a batch.
- The destruction operation destroys channels one by one. If a channel fails to be destroyed, the operation exits immediately. Destroyed channels are not rolled back, and the operation is non-atomic.

## Example

```c
EndpointHandle endpointHandle = nullptr;
 // ... Code for creating the endpoint (omitted)

 // Create multiple channels.
 const uint32_t CHANNEL_NUM = 4;
 HcommChannelDesc channelDescs[CHANNEL_NUM] = {0};
 ChannelHandle channels[CHANNEL_NUM] = {0};

 // Prepare the channel descriptor and create the channel.
 for (uint32_t i = 0; i < CHANNEL_NUM; i++) {
     // ... Fill channelDescs[i].
 }

 HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_CPU,
                                      channelDescs, CHANNEL_NUM, channels);
 if (ret != 0) {
     printf("Failed to create channels, ret = %d\n", ret);
     HcommEndpointDestroy(endpointHandle);
     return ret;
 }

 printf("%u channels created successfully\n", CHANNEL_NUM);

 // Use the channels for communication.
 // ...

 // Destroy all channels in a batch.
 ret = HcommChannelDestroy(channels, CHANNEL_NUM);
 if (ret != 0) {
     printf("Failed to destroy channels, ret = %d\n", ret);
 } else {
     printf("All channels destroyed successfully\n");
 }

 // Destroy the endpoint.
 HcommEndpointDestroy(endpointHandle);
```
