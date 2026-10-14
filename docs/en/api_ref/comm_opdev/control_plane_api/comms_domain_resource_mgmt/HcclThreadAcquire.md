# HcclThreadAcquire

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:52:20.092Z pushedAt=2026-10-12T06:12:35.606Z -->

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

Obtains communication threads based on a communicator and allocates a specified number of synchronization resources (Notify) to each communication thread. For related concepts, see the [Communication Operator Development Guide - Concurrency Model](../../../../comm_op_dev_guide/prog_models_concepts/concurrency_model.md) section.

## Function Prototype

```c
HcclResult HcclThreadAcquire(HcclComm comm, CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread, ThreadHandle *threads)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>For the definition of the HcclComm type, see [HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md). |
| engine | Input | Communication engine type.<br>For the definition of the CommEngine type, see [CommEngine](../../datatype_definition/CommEngine.md). |
| threadNum | Input | Number of communication threads. |
| notifyNumPerThread | Input | Number of synchronization resources (Notify) in each communication thread. Value range: \[0, 65535\], with the specific limit determined by the specific product. It is recommended to configure them reasonably based on the service scenario to avoid resource shortage or waste. |
| threads | Output | Returned communication thread handles. Pass in a **ThreadHandle** array of size **threadNum**.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../datatype_definition/ThreadHandle.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

1. The returned communication thread and synchronization resources are managed by the library. Callers must not release them.

2. The CommEngine ranges supported by each product form are as follows:

  <!-- npu="950" id6 -->
  - Ascend 950PR&950DT products:
    - COMM_ENGINE_CPU_TS
    - COMM_ENGINE_AICPU_TS
  <!-- end id6 -->

  <!-- npu="A3" id7 -->
  - Atlas A3 products:
    - COMM_ENGINE_CPU_TS
    - COMM_ENGINE_AICPU_TS
  <!-- end id7 -->

  <!-- npu="910b" id8 -->
  - Atlas A2 products:
    - COMM_ENGINE_CPU_TS
    - COMM_ENGINE_AICPU_TS
  <!-- end id8 -->

3. This API does not support the **COMM_ENGINE_AIV** and **COMM_ENGINE_CCU** communication engines.
4. **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

The following shows an example of creating thread resources:

```c
// Communicator handle.
HcclComm comm;
// Allocate 5 communication threads of the AICPU_TS type, each with 2 Notify resources.
CommEngine engine = COMM_ENGINE_AICPU_TS;
ThreadHandle threads[5];
HcclThreadAcquire(comm, engine, 5, 2, threads);
```

The following shows an example of synchronizing host thread resources:

```c
// Allocate one host stream.
aclrtStream stream;
aclrtCreateStream(&stream);
// Create a CPU_TS thread based on the allocated stream.
ThreadHandle cpuThread;
HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, stream, 1, &cpuThread);
// Task orchestration.
// ...
// Stream synchronization.
aclrtSynchronizeStream(stream);
```

The following shows an example of synchronizing AI CPU thread resources:

```c
// --Host-side call flow--
// Allocate one host stream.
aclrtStream stream;
aclrtCreateStream(&stream);
// Create a CPU_TS thread based on the allocated stream.
ThreadHandle cpuThread;
HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, stream, 1, &cpuThread);

// Create an AICPU_TS thread.
ThreadHandle aicpuThread;
HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 1, &aicpuThread);

// Export the created AICPU_TS thread as a thread available on the CPU.
ThreadHandle exportedCpuThread;
HcclThreadExportToCommEngine(comm, 1, &aicpuThread, COMM_ENGINE_CPU_TS, &exportedCpuThread);
// Export the created CPU thread as a thread available on the AI CPU.
ThreadHandle exportedAicpuThread;
HcclThreadExportToCommEngine(comm, 1, &cpuThread, COMM_ENGINE_AICPU_TS, &exportedAicpuThread);

// Send a synchronization signal.
HcommThreadNotifyRecordOnThread(cpuThread, exportedCpuThread, 0);
// Deliver the kernel to bring exportedAicpuThread and aicpuThread to the AI CPU side.
// ...
uint32_t timeout = 1;
// Wait for the synchronization signal.
HcommThreadNotifyWaitOnThread(cpuThread, 0, timeout);

// --Device-side call flow--
// Wait for the synchronization signal.
uint32_t timeout = 1;
HcommThreadNotifyWaitOnThread(aicpuThread, 0, timeout);
// Orchestrate tasks and dispatch them to aicpuThread.
// ...
// Send the synchronization signal.
HcommThreadNotifyRecordOnThread(aicpuThread, exportedAicpuThread, 0);

// --Host-side call flow--
// Synchronize the stream.
aclrtSynchronizeStream(stream);
```
