# HcommWriteWithNotifyNbi

<!-- md-trans-meta sourceCommit=01a4daa48785c34db54992320c7441d255cce5c2 translatedAt=2026-10-08T12:14:03.684Z pushedAt=2026-10-12T06:12:35.654Z -->

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

Writes data to the specified memory on the channel, writes the memory data of length `len` in `src` to the memory area of the same length pointed to by `dst`, and sends a synchronization signal to the node where `dst` resides. The API caller is the node where `src` resides. This API is non-blocking.

## Function Prototype

```c
int32_t HcommWriteWithNotifyNbi(ChannelHandle channel, void *dst, const void *src, uint64_t len,
    uint32_t remoteNotifyIdx)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| channel | Input | Communication channel handle, which is the channel obtained through [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md). For constraints on the channel, see the constraints.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| dst | Output | Destination memory address, which is the remote memory address obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) or [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| src | Input | Source memory address, which is the local memory address obtained through [HcclGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclGetHcclBuffer.md) or [HcclChannelGetHcclBuffer](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelGetHcclBuffer.md). |
| len | Input | Data length in bytes, which must be greater than 0. |
| remoteNotifyIdx | Input | Notify index at the other end of the communication channel.<br>Value range: [0, notifyNum).<br>**notifyNum** is the **notifyNum** in the **channelDesc** parameter passed to [HcommChannelCreate](../../../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md). |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Constraints

<!-- npu="950" id6 -->
- For Ascend 950PR&950DT products, this API can be called only on the host CPU, and cannot be called on the AI CPU side.
- When calling [HcclChannelAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md) to allocate the input parameter `channel`, pass `engine = COMM_ENGINE_CPU` and set `channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE`. Protocol channels such as URMA and UBC do not support this API currently.
<!-- end id6 -->
- `[src, src + len)` must fall within the memory range registered or obtained on the local end, and `[dst, dst + len)` must fall within the memory range imported or obtained on the remote end.
- `remoteNotifyIdx` must be smaller than the number of Notify resources on the other end of the communication channel, and `notifyNum` must be greater than 0 when the communication channel is created.
- The write target end should call [HcommChannelNotifyWait](HcommChannelNotifyWait.md) to wait for the synchronization signal.
- A successful return from this API only indicates that the write-with-notification request has been submitted successfully. To confirm that the submitted write operations on the local channel have completed, call [HcommChannelFence](HcommChannelFence.md).
- The same `ChannelHandle` does not support concurrent access from multiple threads.

## Example

```c
// Omitted: Create the communicator handle comm.

// Allocate communication channel resources.
CommEngine engine = CommEngine::COMM_ENGINE_CPU;
uint32_t channelNum = 1;
HcclChannelDesc channelDesc;
HcclResult result = HcclChannelDescInit(&channelDesc, channelNum);
channelDesc.channelProtocol = COMM_PROTOCOL_ROCE;
channelDesc.localEndpoint.protocol = COMM_PROTOCOL_ROCE;
channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
channelDesc.notifyNum = 1;
// Omitted: Fill other information into channelDesc.
ChannelHandle channel;
result = HcclChannelAcquire(comm, engine, &channelDesc, channelNum, &channel);

// Obtain the local communication memory information.
void * localBuffer;
uint64_t localBufferSize;
result = HcclGetHcclBuffer(comm, &localBuffer, &localBufferSize);

// Obtain the remote communication memory information.
void * remoteBuffer;
uint64_t remoteBufferSize;
result = HcclChannelGetHcclBuffer(comm, channel, &remoteBuffer, &remoteBufferSize);
uint64_t len = std::min(localBufferSize, remoteBufferSize);

// Write the local memory content to the remote memory and notify the remote end.
uint32_t rmtNotifyIdx = 0;
int32_t ret = HcommWriteWithNotifyNbi(channel, remoteBuffer, localBuffer, len, rmtNotifyIdx);

// Wait for the submitted write operations on the communication channel to complete.
ret = HcommChannelFence(channel);
```

The following code is called on the write target end to wait for the synchronization signal sent by the write initiator.

```c
// Omitted: Resource creation.

// Use the local Notify index to wait for the synchronization signal sent by the write initiator.
uint32_t lclNotifyIdx = 0;
uint32_t notifyTimeout = 1800;
int32_t ret = HcommChannelNotifyWait(channel, lclNotifyIdx, notifyTimeout);
```
