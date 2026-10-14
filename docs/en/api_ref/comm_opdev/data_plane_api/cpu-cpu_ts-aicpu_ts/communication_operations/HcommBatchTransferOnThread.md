# HcommBatchTransferOnThread

<!-- md-trans-meta sourceCommit=01a4daa48785c34db54992320c7441d255cce5c2 translatedAt=2026-10-08T12:07:24.646Z pushedAt=2026-10-12T06:12:35.637Z -->

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

Asynchronously submits a group of transfer tasks on a specified thread and channel. Each transfer task is described by [HcommBatchTransferDesc](../../../datatype_definition/HcommBatchTransferDesc.md), and supports operation types such as one-sided write, one-sided read, write reduction, write with notification, and notification record/wait.

## Function Prototype

```c
int32_t HcommBatchTransferOnThread(ThreadHandle thread, ChannelHandle channel,
    const HcommBatchTransferDesc *transferDescs, uint32_t transferDescNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Communication thread handle.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| channel | Input | Communication channel handle.<br>For the definition of the ChannelHandle type, see [ChannelHandle](../../../datatype_definition/ChannelHandle.md). |
| transferDescs | Input | Array of batch transfer descriptors. Each element in the array describes one transfer task, and the tasks are submitted sequentially in array order. For the descriptor definition, see [HcommBatchTransferDesc](../../../datatype_definition/HcommBatchTransferDesc.md). |
| transferDescNum | Input | Number of descriptors, which must be greater than 0. |

## Return Value

**int32_t**: The API returns **0** on success and a non-zero value on failure.

## Constraints

<!-- npu="A3,910b" id6 -->
- For the following products, this API supports only the RoCE communication protocol, and **transType** supports only **HCOMM_TRANSFER_TYPE_WRITE** and **HCOMM_TRANSFER_TYPE_READ**. For other types, **HCCL_E_NOT_SUPPORT** is returned.
    <!-- npu="A3" id7 -->
    - Atlas A3 products
    <!-- end id7 -->
    <!-- npu="910b" id8 -->
    - Atlas A2 products
    <!-- end id8 -->
<!-- end id6 -->

- The same `ThreadHandle` does not support concurrent access from multiple threads.
- The same `ChannelHandle` does not support concurrent access from multiple threads.

## Example

```c
// Allocate communication thread resources and channel resources (omitted).
// Obtain the local and remote communication memory information (omitted).

// Construct the batch transfer descriptors.
HcommBatchTransferDesc descs[2];

// Descriptor 1: one-sided write.
descs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;
descs[0].transferInfo.write.len = 4096;
descs[0].transferInfo.write.dst = remoteBuffer;    // Remote address.
descs[0].transferInfo.write.src = localBuffer;     // Local address.

// Descriptor 2: one-sided write (with notification).
descs[1].transType = HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY;
descs[1].transferInfo.writeWithNotify.len = 8192;
descs[1].transferInfo.writeWithNotify.dst = remoteBuffer2;    // Remote address.
descs[1].transferInfo.writeWithNotify.src = localBuffer2;     // Local address.
descs[1].transferInfo.writeWithNotify.notifyIdx = 0;          // Notification index.

// Submit the batch transfer.
HcommBatchTransferOnThread(thread, channel, descs, 2);
```
