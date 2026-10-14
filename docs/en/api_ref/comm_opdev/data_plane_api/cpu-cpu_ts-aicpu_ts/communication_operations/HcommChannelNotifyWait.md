# HcommChannelNotifyWait

<!-- md-trans-meta sourceCommit=01a4daa48785c34db54992320c7441d255cce5c2 translatedAt=2026-10-08T12:09:10.743Z pushedAt=2026-10-12T06:12:35.641Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
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

Waits for a synchronization signal and blocks until the Notify on the specified channel is complete.

## Function Prototype

```c
int32_t HcommChannelNotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| channel | Input | Communication channel handle, which is the channel obtained through [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md). For constraints on the channel, see the constraints.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| localNotifyIdx | Input | Local Notify index.<br>Value range: [0, notifyNum).<br>**notifyNum** is the **notifyNum** in the **channelDesc** parameter passed to the [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API. |
| timeOut | Input | Timeout duration, in seconds. For the CPU engine RoCE scenario of Ascend 950PR&950DT products, a timeout greater than 0 must be configured. |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Constraints

<!-- npu="950" id6 -->
- For Ascend 950PR/Ascend 950DT products, this API can be called only on the host CPU, and cannot be called on the AI CPU side.
- When calling [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) to allocate the input parameter channel, set `engine = COMM_ENGINE_CPU` and `channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE`. This API does not support protocol channels such as URMA/UBC.
- `localNotifyIdx` must be smaller than the number of Notify resources on the local communication channel, `notifyNum` must be greater than 0 when the communication channel is created, and `timeOut` must be greater than 0.
<!-- end id6 -->
- This API must be used together with [HcommChannelNotifyRecord](HcommChannelNotifyRecord.md).
- The same `ChannelHandle` does not support concurrent access from multiple threads.

## Example

```c
CommEngine engine = CommEngine::COMM_ENGINE_CPU;
HcclComm comm;

// Allocate communication channel resources.
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
channelDesc.channelProtocol = COMM_PROTOCOL_ROCE;
channelDesc.localEndpoint.protocol = COMM_PROTOCOL_ROCE;
channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
channelDesc.notifyNum = 1;
// Omitted: Fill in other information in channelDesc.
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// The following APIs are called on the host CPU side.

// Notify the remote end.
HcommChannelNotifyRecord(channel, 0);

// Data plane operation.
// ...

// Wait for the remote end to notify the local end.
uint32_t notifyTimeout = 1800;
HcommChannelNotifyWait(channel, 0, notifyTimeout);
```
