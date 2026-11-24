# HcommChannelDrainOnThread

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:22:44.570Z pushedAt=2026-09-30T09:33:26.581Z -->

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

Blocks and waits for the communication operations already submitted on the specified channel to complete naturally, until the pending task queue is empty.

This API is used to wait for the tasks submitted before the call to complete, and does not add order preservation constraints for tasks submitted after the call. If you need to guarantee the read/write operation order on the channel before and after the barrier, call [HcommChannelFenceOnThread](HcommChannelFenceOnThread.md).

## Function Prototype

```c
int32_t HcommChannelDrainOnThread(ThreadHandle thread, ChannelHandle channel)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle. When called on the AI CPU side, this is the thread obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) or [HcclThreadAcquireWithConfig](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquireWithConfig.md) API. When called on the host CPU side, this parameter can be set to **0**. The host CPU thread (`THREAD_TYPE_CPU`) can only be obtained through the [HcclThreadAcquireWithConfig](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquireWithConfig.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| channel | Input | Communication channel handle, which is the channel obtained through [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md).<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |

## Return Value

**int32_t**: The API returns **0** on success and a non-zero value on failure.

## Constraints

<!-- npu="A3,910b" id6 -->
- For the following products, when this API is called on the AI CPU side, the communication engine is AICPU_TS, and only the RoCE communication protocol is supported.
    <!-- npu="A3" id7 -->
    - Atlas A3 products: Supported
    <!-- end id7 -->
    <!-- npu="910b" id8 -->
    - Atlas A2 products: Supported
    <!-- end id8 -->
<!-- end id6 -->
<!-- npu="950" id9 -->
For Ascend 950PR&950DT products, this API can be called on the AI CPU side and the host CPU side, and can also be called on the device side using a host CPU thread (the `THREAD_TYPE_CPU` thread applied for through the [HcclThreadAcquireWithConfig](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquireWithConfig.md) API).

- When the API is called on the AI CPU side, the communication engine is AICPU_TS, and the communication protocols UB_CTP, UBoE, and UB_RTP are supported.
- When the API is called on the host CPU side, if [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) is called to allocate the input parameter `channel`, `engine = COMM_ENGINE_CPU` must be passed in, and `channelDesc.remoteEndpoint.protocol` must be `COMM_PROTOCOL_ROCE` or `COMM_PROTOCOL_UB_CTP`.
- In host CPU thread mode, this API submits the drain task to the task queue of the host CPU thread and returns immediately. The task is executed by the host in submission order, and a success return only indicates that the submission is successful. The input parameter `channel` must meet the channel constraints for the host CPU side described above.
<!-- end id9 -->
- The same `ChannelHandle` does not support concurrent access from multiple threads.

## Example

### Call Example for AI CPU

```c
CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS;
uint32_t threadNum = 1;
uint32_t notifyNumPerThread = 1;
HcclComm comm;
ThreadHandle thread;
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

// Read the remote memory content to the local memory.
HcommReadOnThread(thread, channel, localBuffer, remoteBuffer, len);

HcommChannelDrainOnThread(thread, channel);
```

### Call Example for Host CPU

```c
// endpointHandle is the created endpoint, and channelDesc has been configured based on the remote endpoint information.
HcommResult DrainHostCpuChannel(EndpointHandle endpointHandle, HcommChannelDesc *channelDesc)
{
    ChannelHandle channel = 0;
    HcommResult result = HcommChannelCreate(endpointHandle, COMM_ENGINE_CPU, channelDesc, 1, &channel);
    if (result != 0) {
        printf("Failed to create channel, result = %d\n", result);
        return result;
    }

    // Submit communication tasks through the channel (omitted).

    // Wait for the submitted communication tasks on the channel to complete. Pass 0 to the thread parameter on the host CPU side.
    result = HcommChannelDrainOnThread(0, channel);
    if (result != 0) {
        printf("Failed to drain channel, result = %d\n", result);
        return result;
    }
    return 0;
}
```
