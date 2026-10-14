# Scheduling Tasks

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-11T07:12:16.382Z pushedAt=2026-08-20T11:39:14.569Z -->

## Scheduling Procedure

The process in which ranks participating in a collective communication coordinate in an orderly manner to perform synchronization and data movement, thereby completing a collective communication operation, is called task scheduling. The primary purpose of task scheduling is to execute tasks on different communication threads in parallel, maximizing resource utilization and improving overall performance.

For AIV communication operators, algorithm orchestration is primarily based on the data copy and synchronization APIs provided by the Ascend C programming language. Users can also further encapsulate high-level APIs for data movement and synchronization operations in communication scenarios to facilitate repeated calls.

In the Ascend C programming framework, the external and internal storage accessible to the AI Core are referred to as Global Memory and Local Memory, respectively. The GlobalTensor data structure stores global data in Global Memory, while the LocalTensor data structure stores data in the AI Core's Local Memory. Whether for soft synchronization setting, flag checking, or data copy, the underlying operations all use the DataCopy-related APIs of Ascend C. These APIs support bidirectional data copy between GlobalTensor and LocalTensor, but the data flow direction varies depending on the scenario:

- Soft synchronization record: After the flag value is set in LocalTensor, it is copied to GlobalTensor through DataCopy.
- Soft synchronization wait: The flag value is set in GlobalTensor. After copying the flag value to LocalTensor, perform a scalar comparison. If the value matches the expected value, proceed to the next step; otherwise, repeat the copy and comparison operations.
- Data movement: Both the source and destination addresses reside in GlobalTensor. The data must be transferred through the internal storage of the Vector Core. First, copy the source GlobalTensor to LocalTensor via DataCopy, and then copy it to the destination GlobalTensor.

For Reduce operators, you can also enable the in-transit Reduce feature for data transfer from LocalTensor to GlobalTensor by using the atomic operation APIs of Ascend C. For example, when the Reduce type is SUM, call the SetAtomicAdd API; when it is MAX, call the SetAtomicMax API.

Specifically, the task scheduling of an AIV operator involves the following steps:

1. Obtain the local communication memory, which is referred to as the HCCL Buffer in HCCL.

   > [!NOTE] Note
   > The HCCL Buffer is a piece of pinned memory on the device managed by each HCCL communicator. Since communication tasks are executed asynchronously, the input data must first be copied to the HCCL Buffer, which has a fixed memory address, to ensure that the user-provided data remains valid when the communication task is actually executed.

2. Split the input data and compute the offsets.

    The default size of the HCCL Buffer is 200 MB. If the input data exceeds this size, it must be split into multiple data blocks for separate processing, which requires multiple operator kernel launches on the host. Each task scheduling handles only one data block.

3. Copy the operator input data to the HCCL Buffer. Commonly used Ascend C APIs include **DataCopy** and **DataCopyPad**.
4. Perform pre-synchronization to confirm with other ranks whether the communication preparation is ready. Commonly used Ascend C APIs include **DataCopy**, **PipeBarrier**, **SetFlag**, and **WaitFlag**.
5. Move data by copying remote data to the local HCCL Buffer. Commonly used Ascend C APIs include **DataCopy** and **DataCopyPad**.
6. Perform post-synchronization to confirm with other ranks that data transfer is complete.
7. Copy the result data from the HCCL Buffer to the operator output memory.

## Sample Code

Taking the AllGather operator as an example, the code snippet for task scheduling on the Vector Core is as follows:

Here, CpGM2GM encapsulates the on-chip memory copy API that supports unaligned data movement based on **DataCopy** and **DataCopyPad**, while Record1vN and WaitNv1 encapsulate the inter-rank synchronization APIs based on **DataCopy**, **SetFlag**, and **WaitFlag**.

```c
if (GetBlockIdx() != rank) {
    // Synchronize between ranks. Each core waits for data from other ranks to be ready.
    WaitNv1(tag);
    PipeBarrier<PIPE_ALL>();
    // Read data from the peer rank to the output memory of the current rank.
    CpGM2GM(outputGM + GetBlockIdx() * count, cclGMOther, count);
    // Perform post-synchronization with the peer whose rank number is GetBlockIdx().
    Record(GetBlockIdx(), tag);
    Wait(GetBlockIdx(), tag);
} else {
    // Use one core to copy data from the input memory of the current rank to the intermediate memory for other ranks to read.
    CpGM2GM(cclGMSelf, inputGM, count);
    PipeBarrier<PIPE_ALL>();
    // Synchronize between ranks and notify other ranks to read the data.
    Record1vN(tag);
    // Use one core to copy data from the intermediate memory of the current rank to the user output memory.
    CpGM2GM(outputGM + count * rank, cclGMSelf, count);
}
```
