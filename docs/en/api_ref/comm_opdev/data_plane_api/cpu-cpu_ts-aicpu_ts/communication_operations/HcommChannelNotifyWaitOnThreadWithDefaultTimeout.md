# HcommChannelNotifyWaitOnThreadWithDefaultTimeout

<!-- md-trans-meta sourceCommit=01a4daa48785c34db54992320c7441d255cce5c2 translatedAt=2026-10-08T12:09:48.193Z pushedAt=2026-10-12T06:12:35.643Z -->

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

Waits for a synchronization signal and blocks until the Notify on the specified channel is complete. **Uses the default timeout duration set through HcommSetNotifyWaitTimeOut**, so you do not need to manually pass the timeout parameter.

## Function Prototype

```c
int32_t HcommChannelNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| channel | Input | Communication channel handle, which is the channel obtained through [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md).<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| localNotifyIdx | Input | Local Notify index.<br>Value range: [0, notifyNum).<br>**notifyNum** is the **notifyNum** in the **channelDescs** parameter passed to the [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API. |

<!-- npu="950" id7 -->
**Supplementary description**:

**thread** parameter: For Ascend 950PR&950DT products, in the CPU engine RoCE scenario, the **thread** parameter has no effect and can be set to **0**. In the CPU_TS/AICPU_TS scenario, the **thread** parameter is the threads obtained through [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md).
<!-- end id7 -->

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Default Timeout Description

- The default timeout duration is set through the [HcommSetNotifyWaitTimeOut](./HcommSetNotifyWaitTimeOut.md) API.
- If it is not set, in AICPU_TS mode, the default timeout duration on the device side is 1836 seconds.
- Setting it to **0** indicates waiting indefinitely.
- Setting it to a value greater than 0 indicates a specific timeout duration (unit: second).

## Constraints

- This API must be used together with [HcommChannelNotifyRecordOnThread](HcommChannelNotifyRecordOnThread.md).
<!-- npu="950" id6 -->
- On Ascend 950PR&950DT products, this API can be called only on the device side in AICPU_TS mode.
- In AICPU_TS mode, before calling this API on the device side, if you need to set the timeout duration, call [HcommSetNotifyWaitTimeOut](./HcommSetNotifyWaitTimeOut.md). If the setting API is not called, the default timeout duration is 1836 seconds.
- For the AICPU_TS mode of Ascend 950PR&950DT products, `localNotifyIdx` must be smaller than the number of Notify resources of the local communication channel, and `notifyNum` must be greater than 0 when the communication channel is created.
<!-- end id6 -->
- The same `ThreadHandle` does not support concurrent access from multiple threads.
- The same `ChannelHandle` does not support concurrent access from multiple threads.

## Example

```c
// 1. Set the default timeout duration (optional; the default value of 1836 seconds is used if not set).
uint32_t defaultTimeout = 1800;  // 30 minutes.
HcommSetNotifyWaitTimeOut(defaultTimeout);

// 2. Allocate communication thread resources.
CommEngine engine = COMM_ENGINE_AICPU_TS;  // Configuration for Ascend 950PR&950DT products.
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
ThreadHandle thread;
HcclComm comm;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// 3. Allocate communication channel resources.
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// 4. Record the notification on the device side.
HcommChannelNotifyRecordOnThread(thread, channel, 0);

// 5. Perform data plane operations.
// ...

// 6. Wait for the remote end notification (using the default timeout duration, no need to pass it manually).
HcommChannelNotifyWaitOnThreadWithDefaultTimeout(thread, channel, 0);
```
