# HcommFenceOnThread

<!-- md-trans-meta sourceCommit=01a4daa48785c34db54992320c7441d255cce5c2 translatedAt=2026-10-08T12:10:14.720Z pushedAt=2026-10-12T06:12:35.644Z -->

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

Triggers a global data-plane fence operation on the host CPU side, flushes the currently initialized internal flush resources, and waits for completion.

This API is not bound to a specific communication channel. To wait for the completion of read and write operations that have been submitted on a specified communication channel, call [HcommChannelFenceOnThread](HcommChannelFenceOnThread.md) first.

## Function Prototype

```c
int32_t HcommFenceOnThread(ThreadHandle thread)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle. When called on the host CPU side, this parameter has no effect and can be set to **0**.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Constraints

<!-- npu="950" id6 -->
- For Ascend 950PR&950DT products, this API can be called only on the host CPU side.
<!-- end id6 -->
- When this API is called on the host CPU side, the communication engine is the CPU, the supported communication protocol is RoCE, and the `thread` parameter has no effect and can be set to **0**.
- This API is used to flush the data plane on the host CPU side and does not wait for the read/write operations on the specified communication channel to complete. To wait for the read/write operations submitted on the channel to complete, call [HcommChannelFenceOnThread](HcommChannelFenceOnThread.md) first.
- If no internal resources need to be flushed, the API returns success.
- Concurrency note: This API is thread-safe and supports concurrent calls from multiple threads (internally protected by a lock and executed serially). It performs a global flush and is not isolated by handle.

## Example

### Collective Communication Example

```c
// Allocate communication channel resources.
CommEngine engine = CommEngine::COMM_ENGINE_CPU;
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclChannelDescInit(&channelDesc, channelNum);
HcclComm comm;
ChannelHandle channel;
HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// Obtain the local communication memory information.
void * localBuffer;
uint64_t localBufferSize;
HcclGetHcclBuffer(comm, &localBuffer, &localBufferSize);

// Obtain the remote end communication memory information.
void * remoteBuffer;
uint64_t remoteBufferSize;
HcclChannelGetHcclBuffer(comm, channel, &remoteBuffer, &remoteBufferSize);
uint64_t len = std::min(localBufferSize, remoteBufferSize);

// Write the local memory content to the remote memory.
int32_t ret = HcommWriteNbiOnThread(0, channel, remoteBuffer, localBuffer, len);
ret = HcommChannelFenceOnThread(0, channel);

// Execute the host CPU-side data plane flush and wait for completion.
ret = HcommFenceOnThread(0);
```
