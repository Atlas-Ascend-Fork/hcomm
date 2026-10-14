# HcommWriteWithNotifyOnThread

<!-- md-trans-meta sourceCommit=01a4daa48785c34db54992320c7441d255cce5c2 translatedAt=2026-10-08T12:14:50.101Z pushedAt=2026-10-12T06:12:35.656Z -->

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

Writes data to the specified memory on the channel, copies the memory data of length **len** from **src** to the memory area of the same length pointed to by **dst**, and sends a synchronization signal to the node where **dst** resides. The API caller is the node where **src** resides. This API is asynchronous.

## Function Prototype

```c
int32_t HcommWriteWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len, uint32_t remoteNotifyIdx)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle, which is the thread obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| channel | Input | Communication channel handle, which is the channel obtained through the [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) API.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| dst | Output | Destination memory address, which is the memory obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) or [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| src | Input | Source memory address, which is the memory obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) or [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| len | Input | Data length (in bytes). |
| remoteNotifyIdx | Input | Notify index of the other end of the communication channel.<br>Value range: [0, notifyNum in the channelDescs parameter passed to [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md)). |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Constraints

- This API must be used together with [HcommChannelNotifyWaitOnThread](HcommChannelNotifyWaitOnThread.md).

<!-- npu="950" id6 -->
- On the Ascend 950PR&950DT products, this API can be called only in AICPU_TS mode on the device side.
<!-- end id6 -->

- The same `ThreadHandle` does not support concurrent access from multiple threads.
- The same `ChannelHandle` does not support concurrent access from multiple threads.

## Example

```c
// Allocate communication thread resources.
CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS;
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

// Obtain the local communication memory information.
void * localBuffer;
uint64_t localBufferSize;
HcclGetHcclBuffer(comm, &localBuffer, &localBufferSize);

// Obtain the remote communication memory information.
void * remoteBuffer;
uint64_t remoteBufferSize;
HcclChannelGetHcclBuffer(comm, channel, &remoteBuffer, &remoteBufferSize);
uint64_t len = std::min(localBufferSize, remoteBufferSize);

// For the Ascend 950PR&950DT products, call the following APIs on the device side.

// Write the local memory content to the remote memory and notify the remote end.
uint32_t rmtNotifyIdx = 0;
HcommWriteWithNotifyOnThread(thread, channel, remoteBuffer, localBuffer, len, rmtNotifyIdx);

// Perform data plane operations.
// ...

// Wait for the remote end to notify the local end.
uint32_t lclNotifyIdx = 0;
uint32_t notifyTimeout = 0;
HcommChannelNotifyWaitOnThread(thread, channel, lclNotifyIdx, notifyTimeout);
```
