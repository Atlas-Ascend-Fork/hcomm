# HcommChannelNotifyWaitOnThread

<!-- md-trans-meta sourceCommit=01a4daa48785c34db54992320c7441d255cce5c2 translatedAt=2026-10-08T12:09:47.576Z pushedAt=2026-10-12T06:12:35.642Z -->

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

Waits for a synchronization signal and blocks until the Notify on the specified channel is complete.

## Function Prototype

```c
int32_t HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| channel | Input | Communication channel handle, which is the channel obtained through [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md). For constraints on the channel, see the constraints.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| localNotifyIdx | Input | Local Notify index.<br>Value range: [0, notifyNum).<br>**notifyNum** is the **notifyNum** in the **channelDescs** parameter passed to the [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API. |
| timeOut | Input | Timeout duration, in seconds.<br> - **0**: Indicates waiting forever.<br> - > 0: The configured specific timeout duration.|

<!-- npu="950" id7 -->
**Supplementary description**:

- **thread** parameter:

  For Ascend 950PR&950DT products, in the CPU engine RoCE scenario, the **thread** parameter has no effect, and 0 can be passed in. In the CPU_TS/AICPU_TS scenario, the **thread** parameter indicates the thread obtained through [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md).

- **timeOut** parameter:

  For the CPU engine RoCE scenario on Ascend 950PR&950DT products, a timeout duration greater than 0 must be configured.
<!-- end id7 -->

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Constraints

- This API must be used together with [HcommChannelNotifyRecordOnThread](HcommChannelNotifyRecordOnThread.md).
<!-- npu="950" id6 -->
- For Ascend 950PR&950DT products, this API can be called on the device side in the AICPU_TS scenario, and can also be called on the host CPU side in the CPU engine RoCE scenario.
- For the CPU engine RoCE scenario on Ascend 950PR&950DT products, when calling [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) to allocate the input parameter **channel**, set `engine = COMM_ENGINE_CPU` and `channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE`. This API is not supported for channels using protocols such as URMA or UBC.
- For the CPU engine RoCE scenario on Ascend 950PR&950DT products, `localNotifyIdx` must be less than the number of Notify resources on the local communication channel, and `notifyNum` must be greater than 0 when the communication channel is created. `timeOut` must be greater than 0.
<!-- end id6 -->
- When this API is called on the host CPU, the `thread` parameter has no effect and can be set to **0**.
- The same `ThreadHandle` does not support concurrent access from multiple threads.
- The same `ChannelHandle` does not support concurrent access from multiple threads.

## Example

```c
// Allocate communication thread resources.
CommEngine engine = CommEngine::COMM_ENGINE_CPU_TS;
// Configure for Ascend 950PR&950DT products.
// CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS;
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
ThreadHandle thread;
HcclComm comm;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// Allocate communication channel resources.
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// For Ascend 950PR&950DT products, call the following APIs on the device side.

// Notify the remote end.
HcommChannelNotifyRecordOnThread(thread, channel, 0);

// Perform data plane operations.
// ...

// Wait for the remote end to notify the local end.
uint32_t notifyTimeout = 1800;
HcommChannelNotifyWaitOnThread(thread, channel, 0, notifyTimeout);
```
