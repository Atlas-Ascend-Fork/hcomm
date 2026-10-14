# HcommThreadNotifyWaitOnThreadWithDefaultTimeout

<!-- md-trans-meta sourceCommit=01a4daa48785c34db54992320c7441d255cce5c2 translatedAt=2026-10-08T12:16:39.446Z pushedAt=2026-10-12T06:12:35.661Z -->

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

Waits for the synchronization signal sent by other threads. It is mainly used in multi-thread synchronization waiting scenarios. **It uses the default timeout duration set through [HcommSetNotifyWaitTimeOut](../communication_operations/HcommSetNotifyWaitTimeOut.md)**, so you do not need to manually pass a timeout parameter.

## Function Prototype

```c
int32_t HcommThreadNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle thread, uint32_t notifyIdx)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Thread handle, which is the thread obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| notifyIdx | Input | Index of the Notify notification to wait for.<br>Value range: [0, the value of the notifyNumPerThread parameter passed to [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md)). |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Timeout Mechanism

1. **Default timeout duration**
   - The default timeout duration is set through the [HcommSetNotifyWaitTimeOut](../communication_operations/HcommSetNotifyWaitTimeOut.md) API.
   - If no timeout duration is set, in AICPU_TS mode, the default timeout duration on the device side is 1836 seconds, and that on the host side is 1836+27 seconds.

2. **Conditions for timeout to take effect**
   - The configured timeout duration is applied to the wait operation of this API without appending the default offset.
   - Setting it to **0** indicates waiting indefinitely without timeout.
   - Setting it to a value greater than 0 indicates a specific timeout duration (unit: second).

3. **Special handling in non-AICPU_TS mode**
   - In non-AICPU_TS mode, if the default timeout duration is not manually set, the system automatically adds a 27-second offset to the default value as a safety buffer.

## Constraints

- This API must be used together with [HcommThreadNotifyRecordOnThread](./HcommThreadNotifyRecordOnThread.md).
<!-- npu="950" id6 -->
- On Ascend 950PR&950DT products, this API can be called on both the device side and the host CPU side in AICPU_TS mode.
<!-- end id6 -->
- In AICPU_TS mode, before calling this API on the host side or device side, if you need to set the timeout duration, call [HcommSetNotifyWaitTimeOut](../communication_operations/HcommSetNotifyWaitTimeOut.md) on the corresponding side. If the setting API is not called, the default timeout duration is 1836 seconds (1836+27 seconds by default on the host side).
<!-- npu="950" id7 -->
- For the AICPU_TS mode of Ascend 950PR&950DT products, `notifyIdx` must be smaller than the number of Notify resources on the local thread, and `notifyNumPerThread` must be greater than 0 when the thread is created.
<!-- end id7 -->
- The same `ThreadHandle` does not support concurrent access from multiple threads.

## Example

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_CPU_TS;
aclrtStream streams[2];
ThreadHandle threads[2];
// Allocate two streams, each with two Notify resources.
aclrtCreateStream(&streams[0]);
aclrtCreateStream(&streams[1]);
HcclResult result = HcclThreadAcquireWithStream(comm, engine, streams[0], 2, &threads[0]);
result = HcclThreadAcquireWithStream(comm, engine, streams[1], 2, &threads[1]);

// 1. Set the default timeout duration (optional; if not set, the default value of 1836 seconds is used).
uint32_t defaultTimeout = 600;  // 10 minutes.
HcommSetNotifyWaitTimeOut(defaultTimeout);

uint32_t notifyIdx = 0;
// Send the synchronization signal.
HcommThreadNotifyRecordOnThread(threads[0], threads[1], notifyIdx);
// Wait for the synchronization signal using the default timeout duration (no need to manually pass the timeout parameter).
HcommThreadNotifyWaitOnThreadWithDefaultTimeout(threads[1], notifyIdx);
```
<!-- npu="950" id8 -->
On Ascend 950PR&950DT products, this function must be compiled for use on the device side:

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;
ThreadHandle threads[2];
uint32_t notifyNumPerThread = 2;
HcclThreadAcquire(comm, engine, 1, notifyNumPerThread, &threads[0]);
HcclThreadAcquire(comm, engine, 1, notifyNumPerThread, &threads[1]);

// Allocate other resources.
// Copy the parameters and launch the kernel.

// 1. Set the default timeout duration (optional; if not set, the default value of 1836 seconds is used).
uint32_t defaultTimeout = 1800;  // 30 minutes.
HcommSetNotifyWaitTimeOut(defaultTimeout);

// Device-side algorithm orchestration.
uint32_t notifyIdx = 0;
// Send the synchronization signal.
HcommThreadNotifyRecordOnThread(threads[0], threads[1], notifyIdx);
// Wait for the synchronization signal using the default timeout duration (no need to manually pass the timeout parameter).
HcommThreadNotifyWaitOnThreadWithDefaultTimeout(threads[1], notifyIdx);
```
<!-- end id8 -->

## Related APIs

- [HcommSetNotifyWaitTimeOut](../communication_operations/HcommSetNotifyWaitTimeOut.md): Sets the default timeout duration.
- [HcommThreadNotifyRecordOnThread](./HcommThreadNotifyRecordOnThread.md): Sends a thread synchronization signal.
- [HcommThreadNotifyWaitOnThread](./HcommThreadNotifyWaitOnThread.md): Waits for a thread synchronization signal (manually specifies the timeout duration).
