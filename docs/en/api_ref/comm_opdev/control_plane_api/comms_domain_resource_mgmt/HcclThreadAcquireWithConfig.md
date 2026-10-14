# HcclThreadAcquireWithConfig

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:52:48.710Z pushedAt=2026-10-12T06:12:35.607Z -->

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

Obtains communication threads based on the communicator and thread configuration, and allocates a specified number of synchronization resources (Notify) to each communication thread. For related concepts, see [Communication Operator Development Guide - Concurrency Model](../../../../comm_op_dev_guide/prog_models_concepts/concurrency_model.md).

> [!NOTE] Note
> Compared with [HcclThreadAcquire](./HcclThreadAcquire.md), this API supports configuring the number of synchronization resources (Notify) for each thread through the ThreadConfig structure, which is suitable for scenarios where different threads require different numbers of Notify resources.

## Function Prototype

```c
HcclResult HcclThreadAcquireWithConfig(HcclComm comm, CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig *config, ThreadHandle *threads)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>For the definition of the HcclComm type, see [HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md). |
| engine | Input | Communication engine type.<br>For the definition of the CommEngine type, see [CommEngine](../../datatype_definition/CommEngine.md). |
| threadNum | Input | Number of communication threads. |
| type | Input | Thread type.<br>For the definition of the ThreadType type, see [ThreadType](../../datatype_definition/ThreadType.md). |
| config | Input | Array of thread configuration information. The array length must be the same as **threadNum**. Before calling this API, call [ThreadConfigInit](./ThreadConfigInit.md) to initialize the parameters.<br>For the definition of the ThreadConfig type, see [ThreadConfig](../../datatype_definition/ThreadConfig.md). |
| threads | Output | Returned communication thread handles. Pass in a **ThreadHandle** array of size **threadNum**.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../datatype_definition/ThreadHandle.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API return values are described as follows:

## Constraints

1. The returned communication thread and synchronization resources are managed by the library. Callers must not release them.

2. The CommEngine ranges supported by each product form are as follows:

  <!-- npu="950" id6 -->
  - Ascend 950PR&950DT products:
    - COMM_ENGINE_CPU
    - COMM_ENGINE_AICPU
  <!-- end id6 -->

  <!-- npu="A3" id7 -->
  - Atlas A3 products:
    - COMM_ENGINE_CPU
    - COMM_ENGINE_AICPU
  <!-- end id7 -->

  <!-- npu="910b" id8 -->
  - Atlas A2 products:
    - COMM_ENGINE_CPU
    - COMM_ENGINE_AICPU
  <!-- end id8 -->

3. This API does not support the **COMM_ENGINE_AIV**, **COMM_ENGINE_CCU**, **COMM_ENGINE_CPU_TS**, and **COMM_ENGINE_AICPU_TS** communication engines.

  > To create a TS-type thread, pass the **COMM_ENGINE_CPU** or **COMM_ENGINE_AICPU** communication engine and set `type` to **THREAD_TYPE_TS**.

4. **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

The following shows an example of creating thread resources:

```c
// Communicator handle.
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU;
uint32_t threadNum = 5;
ThreadConfig configs[5];
ThreadConfigInit(configs, threadNum);
for (uint32_t i = 0; i < threadNum; i++) {
    configs[i].notifyNumPerThread = 2;
}
ThreadHandle threads[5];
HcclThreadAcquireWithConfig(comm, engine, threadNum, THREAD_TYPE_TS, configs, threads);
```

The following shows an example of configuring different numbers of Notify resources for different threads:

```c
// Communicator handle.
HcclComm comm;
CommEngine engine = COMM_ENGINE_CPU;
uint32_t threadNum = 3;
ThreadConfig configs[3];
ThreadConfigInit(configs, threadNum);
configs[0].notifyNumPerThread = 2;
configs[1].notifyNumPerThread = 4;
configs[2].notifyNumPerThread = 1;
ThreadHandle threads[3];
HcclThreadAcquireWithConfig(comm, engine, threadNum, THREAD_TYPE_TS, configs, threads);
```
