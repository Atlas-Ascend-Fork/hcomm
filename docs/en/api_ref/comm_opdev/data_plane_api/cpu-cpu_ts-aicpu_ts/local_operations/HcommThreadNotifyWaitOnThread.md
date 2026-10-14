# HcommThreadNotifyWaitOnThread

<!-- md-trans-meta sourceCommit=01a4daa48785c34db54992320c7441d255cce5c2 translatedAt=2026-10-08T12:16:15.910Z pushedAt=2026-10-12T06:12:35.661Z -->

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

Waits for a synchronization signal. This API blocks and waits for the thread to run until the specified Notify is recorded.

## Function Prototype

```c
int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Thread handle, which is the thread obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| notifyIdx | Input | Index of the Notify notification to wait for.<br>Value range: [0, the value of the notifyNumPerThread parameter passed to [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md)). |
| timeOut | Input | Timeout duration, in seconds.<br>  - **0**: waits indefinitely.<br>  - > 0: the configured timeout duration.<br> |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Constraints

- This API must be used together with [HcommThreadNotifyRecordOnThread](HcommThreadNotifyRecordOnThread.md).

<!-- npu="950" id6 -->
- On the Ascend 950PR&950DT products, this API can be called only in AICPU_TS mode on the device side.
<!-- end id6 -->

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
uint32_t notifyIdx = 0;
// Send the synchronization signal.
HcommThreadNotifyRecordOnThread(threads[0], threads[1], notifyIdx);
uint32_t timeout = 1;
// Wait for the synchronization signal.
HcommThreadNotifyWaitOnThread(threads[1], notifyIdx, timeout);
```

<!-- npu="950" id7 -->
On Ascend 950PR&950DT products, this function must be compiled for use on the device side:

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;
ThreadHandle threads[2];
uint32_t notifyNumPerThread = 2;
HcclThreadAcquire(comm, engine, 1, notifyNumPerThread, &threads[0]);
HcclThreadAcquire(comm, engine, 1, notifyNumPerThread, &threads[1]);

// Allocate the remaining resources.
// Copy the parameters and launch the kernel.

// Orchestrate the algorithm on the device side.
uint32_t notifyIdx = 0;
// Send the synchronization signal.
HcommThreadNotifyRecordOnThread(threads[0], threads[1], notifyIdx);
uint32_t timeout = 1;
// Wait for the synchronization signal.
HcommThreadNotifyWaitOnThread(threads[1], notifyIdx, timeout);
```
<!-- end id7 -->
