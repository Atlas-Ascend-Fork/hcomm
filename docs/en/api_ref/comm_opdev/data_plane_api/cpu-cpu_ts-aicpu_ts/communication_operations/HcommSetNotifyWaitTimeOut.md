# HcommSetNotifyWaitTimeOut

<!-- md-trans-meta sourceCommit=01a4daa48785c34db54992320c7441d255cce5c2 translatedAt=2026-10-08T12:11:40.900Z pushedAt=2026-10-12T06:12:35.648Z -->

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

Sets the default timeout duration for notification waiting. This timeout duration applies to the subsequent calls to [HcommChannelNotifyWaitOnThreadWithDefaultTimeout](./HcommChannelNotifyWaitOnThreadWithDefaultTimeout.md) and [HcommThreadNotifyWaitOnThreadWithDefaultTimeout](../local_operations/HcommThreadNotifyWaitOnThreadWithDefaultTimeout.md).

## Function Prototype

```c
int32_t HcommSetNotifyWaitTimeOut(float timeOut)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | -------------- |
| timeOut | Input | Notification waiting timeout duration.<br>Unit: second.<br>Value description:<br>  - **0**: waits indefinitely without timeout duration.<br>  - >0: specific timeout duration (in seconds).<br>Default value: 1836 seconds (about 30 minutes).<br>Currently, only integers are supported. |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Timeout Mechanism

1. **Default timeout duration**
   - If this API is not called, the default timeout duration is 1836 seconds (about 30 minutes).

2. **Conditions for timeout duration to take effect**
   - The timeout duration set by this API is applied as-is to subsequent `*WithDefaultTimeout` APIs, without appending the default offset.
   - The timeout duration set by calling `HcommChannelNotifyWaitOnThread` or `HcommThreadNotifyWaitOnThread` is not affected by this API.

3. **Special handling in non-AI CPU mode**
   - In non-AI CPU mode, if this API is not called to set the default timeout duration, the system automatically adds a 27-second offset to the default value as a safety buffer.

## Constraints

- The timeout duration set by this API takes effect only for the `*WithDefaultTimeout` series of APIs.
- `HcommChannelNotifyWaitOnThread` and `HcommThreadNotifyWaitOnThread`, which accept a manually specified timeout duration, are not affected by this API.
- It is recommended that you set the default timeout duration before starting data plane operations.
- Currently, only integers are supported for `timeOut`. Decimal values are not supported.
- Concurrency note: This API is a global configuration API and must be called serially.

## Example

```c
// 1. Set the default timeout duration (for example, 10 minutes).
float defaultTimeOut = 600;  // 600 seconds = 10 minutes.
HcommSetNotifyWaitTimeOut(defaultTimeOut);

// 2. Allocate communication thread resources.
CommEngine engine = COMM_ENGINE_AICPU_TS;
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

// 6. Wait for the notification using the default timeout duration (no need to manually pass the timeout duration parameter).
HcommChannelNotifyWaitOnThreadWithDefaultTimeout(thread, channel, 0);

// Alternatively, use thread notification waiting.
HcommThreadNotifyWaitOnThreadWithDefaultTimeout(thread, 0);
```

## Related APIs

- [HcommChannelNotifyWaitOnThreadWithDefaultTimeout](./HcommChannelNotifyWaitOnThreadWithDefaultTimeout.md): Waits for a channel notification using the default timeout duration.
- [HcommThreadNotifyWaitOnThreadWithDefaultTimeout](../local_operations/HcommThreadNotifyWaitOnThreadWithDefaultTimeout.md): Waits for a thread notification using the default timeout duration.
- [HcommChannelNotifyWaitOnThread](./HcommChannelNotifyWaitOnThread.md): Waits for a channel notification with a manually specified timeout duration.
- [HcommThreadNotifyWaitOnThread](../local_operations/HcommThreadNotifyWaitOnThread.md): Waits for a thread notification with a manually specified timeout duration.
