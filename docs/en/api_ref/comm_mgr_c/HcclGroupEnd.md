# HcclGroupEnd

<!-- md-trans-meta sourceCommit=1435fccbb795a1ecf7fa4675cb37d8bf3e83f779 translatedAt=2026-10-08T11:42:30.533Z pushedAt=2026-10-12T06:12:35.582Z -->

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
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Ends a group call.

Multiple functions called between **HcclGroupStart** and **HcclGroupEnd** are executed as a whole. A group call supports the following three scenarios:

- Managing NPUs in a single process with multiple threads: supports calling the communicator management APIs [HcclCommInitClusterInfo](HcclCommInitClusterInfo.md), [HcclCommInitClusterInfoConfig](HcclCommInitClusterInfoConfig.md), [HcclCommInitRootInfo](HcclCommInitRootInfo.md), [HcclCommInitRootInfoConfig](HcclCommInitRootInfoConfig.md), and [HcclCommDestroy](HcclCommDestroy.md).
- Merging multiple collective communication operations (not supported yet on Ascend 950PR&950DT products).
- Merging multiple point-to-point communication operations.

## Function Prototype

```c
HcclResult HcclGroupEnd()
```

## Parameters

None

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- The group call APIs can be used for communicator management only in a single-server environment.
- Within a single group call, APIs for communicator management, collective communication, and point-to-point communication cannot be mixed.
- When merging multiple point-to-point communication operations, the [HcclBatchSendRecv](https://gitcode.com/cann/hccl/blob/9.2.0/docs/en/api_ref/comm_op_interface/HcclBatchSendRecv.md) API is not supported.
- When merging multiple collective communication operations, ensure that there are no dependencies between the collective communication operations.
- **HcclGroupStart** must be used together with **HcclGroupEnd**, with **HcclGroupStart** called first and **HcclGroupEnd** called last.

## Example

- Example 1: Managing NPUs in a single process with multiple threads

    ```c
    HcclComm hccl_comms[devCount];
    HcclGroupStart();
    for(int i = 0; i &lt; ndev; i++){
        //
            aclrtSetDevice(i);
            HcclCommInitRootInfo(devCount, &rootInfo, global_rank, &(hccl_comms[i]));
        }
    HcclGroupEnd();
    ```

- Example 2: Merging multiple collective communication operations

    ```c
    HcclGroupStart();
        HCCLCHECK(HcclReduceScatter(sendBuf, recvBuf, 1, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream));
        HCCLCHECK(HcclAllGather(recvBuf, sendBuf, 1, HCCL_DATA_TYPE_FP32, hcclComm, stream));
    HcclGroupEnd();
    ```

- Example 3: Merging multiple point-to-point communication operations

    ```c
    HcclGroupStart();
    for(int i = 0; i &lt; devCount; i++){
        HCCLCHECK(HcclSend(sendBuf[i], count, HCCL_DATA_TYPE_FP32, i, hcclComm, stream));
        }
    for(int i = 0; i &lt; devCount; i++){
        HCCLCHECK(HcclRecv(recvBuf[i], count, HCCL_DATA_TYPE_FP32, i, hcclComm, stream));
        }
    HcclGroupEnd();
    ```
