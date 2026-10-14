# HcclDedicatedThreadAcquire

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:48:23.476Z pushedAt=2026-10-12T06:12:35.596Z -->

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

Allocates a dedicated communication thread based on the communicator and allocates a specified number of synchronization resources (Notify) for the thread. Depending on **useType**, thread management is divided into two types:

- **Non-order-preservation type** (**HCCL_DED_THREAD_TYPE_AICPU_LAUNCH**, **HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE**): The dedicated thread is cached within the communicator by **useType**. Repeated calls with the same **useType** directly return the cached thread handle without creating a new thread.

- **Order-preservation type** (**HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_*** series): The dedicated thread is managed by **OrderLaunchThreadMgr** at the process level and is not bound to a single communicator. This type applies to order-preservation task dispatch scenarios.

This API is mainly used in dedicated scenarios such as AI CPU task dispatch. For related concepts, see the [Communication Operator Development Guide - Concurrency Model](../../../../comm_op_dev_guide/prog_models_concepts/concurrency_model.md) section.

> [!NOTE] Note
> Compared with [HcclThreadAcquire](./HcclThreadAcquire.md), this API caches dedicated threads by **useType**, and repeated calls with the same **useType** reuse the same thread. It supports only dedicated dispatch scenarios and does not support specifying a communication engine through **CommEngine**.

## Function Prototype

```c
HcclResult HcclDedicatedThreadAcquire(HcclComm comm, HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle *thread)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>For the definition of the HcclComm type, see [HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md). |
| useType | Input | Dedicated thread usage type. For the definition of the HcclDedicatedThreadType type, see [HcclDedicatedThreadType](../../datatype_definition/HcclDedicatedThreadType.md). |
| notifyNumPerThread | Input | Number of synchronization resources (Notify) in the communication thread. Value range: \[0, 64\]. Configure it properly based on the service scenario to avoid resource shortage or waste. |
| thread | Output | Returned dedicated communication thread handle.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../datatype_definition/ThreadHandle.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

1. The returned communication thread and synchronization resources are managed by the library. Callers must not release them.

2. For the non-order-preservation types (**HCCL_DED_THREAD_TYPE_AICPU_LAUNCH** and **HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE**), a dedicated thread is created only once within the same communicator, and repeated calls return the cached thread handle. If the **notifyNumPerThread** allocated this time is greater than the number of Notify resources of the cached thread, the library supplements the synchronization resources by the difference.

3. The order-preservation types (the **HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_*** series) are managed by the process-level **OrderLaunchThreadMgr**. The threads are not bound to a single communicator. The behavior of each type is as follows:

   - **HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE**: Caches the CPU_TS thread by **aclrtContext**. Repeated calls under the same context reuse the same thread.
   - **HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH**: Caches the CPU_TS thread by **aclrtContext**, with a cache independent of OPBASE.
   - **HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE**: Obtains the thread from the attached stream that has been set. Before calling, set the attached stream through the **HcomSetAttachedStream** API. If it is not set, the returned thread handle is 0.
   - **HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE**: Caches the AICPU_TS thread by communicator identifier.

4. For the order-preservation types, the returned thread handle is 0 in the following scenarios (the API still returns **HCCL_SUCCESS**), indicating that no dedicated order-preservation thread needs to be created:

   - When the number of communicators registered under the current context does not exceed the number of device AI CPU blocks, the OPBASE/ACLGRAPH/DEVICE/GE types do not create a dedicated thread.
   - The GE type does not set an attached stream through **HcomSetAttachedStream**.

5. In graph mode (when **useType** is **HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE**), if the dedicated thread has not been cached in the communicator, the API does not actually create a thread, and the returned thread handle is 0.

6. Before calling the **HcclDedicatedThreadAcquire** API to allocate a dedicated thread, you must first call the **aclrtSetDevice** API on the same thread to specify **deviceId**.

7. **HCCL_DED_THREAD_TYPE_INVALID** is an invalid value. Passing it returns **HCCL_E_PARA**.
8. **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

The following is an example of allocating a dedicated thread for AI CPU task dispatch:

```c
// Communicator handle.
HcclComm comm;
// Allocate dedicated threads of the AICPU_LAUNCH type, including two Notify resources.
ThreadHandle dedThread;
HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, &dedThread);
if (ret != HCCL_SUCCESS) {
    // Error handling.
}
```

The following is an example of allocating a dedicated thread in graph mode:

```c
// Communicator handle.
HcclComm comm;
// Allocate a dedicated thread of the AICPU_LAUNCH_GE type in graph mode.
// If the thread is not cached in the communicator, the returned thread is 0 and the thread is not actually created.
ThreadHandle dedThread;
HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 2, &dedThread);
if (ret != HCCL_SUCCESS) {
    // Error handling.
}
```

The following is an example of allocating a dedicated thread for order-preservation task dispatch of a single operator:

```c
// Communicator handle.
HcclComm comm;
// Allocate dedicated threads of the OPBASE order-preservation type, including one Notify resource.
ThreadHandle orderThread;
HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE, 1, &orderThread);
if (ret != HCCL_SUCCESS) {
    // Error handling.
}
// If orderThread is 0, no dedicated order-preservation thread is required, and the default dispatch path can be used directly.
```

The following is an example of allocating a dedicated thread for order-preservation task dispatch in graph mode:

```c
// Communicator handle.
HcclComm comm;
// In graph mode, set the attached stream through HcomSetAttachedStream first.
aclrtStream stream;
aclrtCreateStream(&stream);
HcomSetAttachedStream(nullptr, graphId, &stream, 1);
// Allocate dedicated threads of the GE order-preservation type.
ThreadHandle geOrderThread;
HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE, 1, &geOrderThread);
if (ret != HCCL_SUCCESS) {
    // Error handling.
}
// If geOrderThread is 0, it indicates that no attached stream is set or no dedicated order-preservation thread is currently required.
```

The following shows an example of allocating a dedicated thread for device-side order-preservation task dispatch:

```c
// Communicator handle.
HcclComm comm;
// Allocate dedicated threads of the device order-preservation type, including one Notify resource.
ThreadHandle deviceOrderThread;
HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, &deviceOrderThread);
if (ret != HCCL_SUCCESS) {
    // Error handling.
}
```
