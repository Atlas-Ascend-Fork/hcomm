# HcommThreadResAcquireTimeOut

<!-- md-trans-meta sourceCommit=01a4daa48785c34db54992320c7441d255cce5c2 translatedAt=2026-10-08T12:11:52.247Z pushedAt=2026-10-12T06:12:35.649Z -->

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

Sets the timeout duration for thread resource acquisition. This timeout duration applies to the queue-full waiting scenario of Remote Transport Sequence Queue (RTSQ). When the RTSQ queue is full, the system waits until space becomes available or a timeout occurs.

## Function Prototype

```c
int32_t HcommThreadResAcquireTimeOut(float timeOut)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| timeOut | Input | Timeout duration for waiting when the RTSQ queue is full.<br>Unit: second.<br>Value description:<br>  - **0**: Indicates that the timeout never occurs. The system waits until space becomes available in RTSQ.<br>  - >0: Specific timeout duration (in seconds).<br>Default value: 1856 seconds.<br>Currently, only integers are supported. |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Timeout Mechanism

1. **Timeout value semantics**
   - When the RTSQ queue is full, the thread blocks and waits until space becomes available.
   - If a timeout duration is set, an error is reported when no space becomes available after the timeout duration expires.
   - Setting the value to **0** means no timeout, and the thread waits indefinitely.

2. **Default value**
   - If this API is not called to set a timeout duration, the default timeout duration is 1856 seconds (about 30 minutes).

3. **Setting scope**
   - It applies to all RTSQ resources acquired through threads.
   - After the setting takes effect, it affects all subsequent resource acquisition operations.

## Scenarios

This API is mainly used in the following scenarios:

- **RTSQ queue-full waiting**: When the send buffer is full, wait for available space.
- **Flow control mechanism**: Work with send operations to implement the backpressure mechanism.
- **Resource acquisition**: Ensure that resources are available before performing subsequent operations.

## Constraints

- The timeout duration set by this API affects the resource acquisition operations of all threads.
- It is recommended that you set an appropriate timeout duration during initialization or before allocating resources.
- If the timeout duration is set to **0**, the thread waits indefinitely, which may cause deadlock. Use this value with caution.
- Currently, only integers are supported for `timeOut`. Decimal values are not supported.
- Concurrency note: This API is a global configuration API and must be called serially.

## Example

```c
// 1. Set the timeout duration for waiting when the RTSQ queue is full (for example, set it to 10 minutes).
float timeout = 600;  // 600 seconds = 10 minutes
HcommThreadResAcquireTimeOut(timeout);

// Or set it to never time out (0 indicates that the timeout never occurs).
float timeoutNever = 0;
HcommThreadResAcquireTimeOut(timeoutNever);

// 2. Allocate communication thread resources.
CommEngine engine = COMM_ENGINE_AICPU_TS;
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
ThreadHandle thread;
HcclComm comm;
HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, &thread);

// 3. Perform data plane operations (when RTSQ is full, it waits for available space or times out).
// ... Perform the data sending operation ...
```

## Related APIs

[HcommSetNotifyWaitTimeOut](./HcommSetNotifyWaitTimeOut.md): Sets the notification wait timeout duration.
