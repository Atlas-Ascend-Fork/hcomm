# Scheduling Tasks

<!-- md-trans-meta sourceCommit=53fdf106ef330fac6b558d13103b73026d3055b6 translatedAt=2026-09-24T07:04:57.678Z pushedAt=2026-09-28T05:59:29.454Z -->

## Scheduling Procedure

The process in which all ranks participating in a collective communication operation coordinate in an orderly manner to perform synchronization and data movement, thereby completing the operation, is called task scheduling. The primary goal of task scheduling is to execute tasks on different communication threads in parallel, maximizing resource utilization and improving overall performance.

The main task scheduling procedure is as follows:

1. Obtain the local communication memory, referred to as the HCCL Buffer in HCCL.
2. Copy the operator input data to the HCCL Buffer. A commonly used data plane API is [HcommLocalCopyOnThread](../../api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts/local_operations/HcommLocalCopyOnThread.md).

   > [!NOTE] Note
   > The HCCL Buffer is a block of pinned memory on the device managed by each HCCL communicator. Since communication tasks are executed asynchronously, the input data must first be copied to the HCCL Buffer, which has a fixed memory address, to ensure that the user input data remains valid when the communication task is actually executed.

3. Split the input data and compute the offset.

    The default size of the HCCL Buffer is 200 MB. If the input data exceeds this size, it must be split into multiple data blocks and processed separately.

4. Perform pre-synchronization: The primary thread notifies the secondary thread to start the task, and the secondary thread waits for the notification from the primary thread. Commonly used data plane APIs include [HcommThreadNotifyRecordOnThread](../../api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts/local_operations/HcommThreadNotifyRecordOnThread.md) and [HcommThreadNotifyWaitOnThread](../../api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts/local_operations/HcommThreadNotifyWaitOnThread.md).
5. Perform data movement to copy remote data to the local HCCL Buffer. Commonly used data plane APIs include [HcommReadOnThread](../../api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts/communication_operations/HcommReadOnThread.md) and [HcommReadReduceOnThread](../../api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts/communication_operations/HcommReadReduceOnThread.md).
6. Perform post-synchronization: The secondary stream notifies the primary stream that the task is complete, and the primary stream waits for the notification from the secondary stream.
7. Copy the result data in the HCCL Buffer to the operator output memory. A commonly used data plane API is [HcommLocalCopyOnThread](../../api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts/local_operations/HcommLocalCopyOnThread.md).

## Sample Code

- Taking the custom Send operator as an example, the task scheduling code snippet on the AI CPU is as follows:

    ```c
    uint64_t size = count * dataTypeSize;  // Data size = data quantity × data type size
    // 1. Copy data to the intermediate memory.
    HcommLocalCopyOnThread(threadHandle, localAddr, inputPtr, size);
    // 2. Notify the receiver that the local end has prepared the data.
    HcommChannelNotifyRecordOnThread(threadHandle, channelHandle, 0);
    // 3. Wait for the receiver to confirm that it has finished reading the local data.
    HcommChannelNotifyWaitOnThread(threadHandle, channelHandle, 1, 1800);
    ```

- Taking the custom Receive operator as an example, the task scheduling code snippet on the AI CPU is as follows:

    ```c
    // 1. Wait for the sender to notify the local end that it can start reading data.
    HcommChannelNotifyWaitOnThread(threadHandle, channelHandle, 0, 1800);
    // 2. Read data from the sender.
    HcommReadOnThread(threadHandle, channelHandle, outputPtr, remoteAddr, size);
    // 3. Notify the sender that the local end has finished reading data.
    HcommChannelNotifyRecordOnThread(threadHandle, channelHandle, 1);
    ```
