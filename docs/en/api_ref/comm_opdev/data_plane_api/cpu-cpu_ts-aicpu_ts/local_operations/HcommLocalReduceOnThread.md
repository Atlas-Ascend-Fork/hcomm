# HcommLocalReduceOnThread

<!-- md-trans-meta sourceCommit=01a4daa48785c34db54992320c7441d255cce5c2 translatedAt=2026-10-08T12:15:49.106Z pushedAt=2026-10-12T06:12:35.659Z -->

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

Provides a local reduce operation that performs the reduceOp operation on the memory data of length **count\*sizeof\(dataType\)** pointed to by **src** and the memory data of the same length pointed to by **dst**, and outputs the result to **dst**.

## Function Prototype

```c
int32_t HcommLocalReduceOnThread(ThreadHandle thread, void *dst, const void *src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| **thread** | Input | Thread handle, which is the thread obtained through the [HcclThreadAcquire](../../../control_plane_api/comms_domain_resource_mgmt/HcclThreadAcquire.md) API.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../../datatype_definition/ThreadHandle.md). |
| **dst** | Output | Destination address in device memory. |
| **src** | Input | Source address in device memory. |
| **count** | Input | Number of elements. |
| **dataType** | Input | Data type. Supported types: int8, int16, int32, float16, float32, bfp16.<br>For the definition of the HcommDataType type, see [HcommDataType](../../../datatype_definition/HcommDataType.md). |
| **reduceOp** | Input | Reduce operation type. Supported types: sum, max, min.<br>For the definition of the HcommReduceOp type, see [HcommReduceOp](../../../datatype_definition/HcommReduceOp.md). |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Constraints

- The **dst** and **src** memory must be device memory.

<!-- npu="950" id6 -->
- On the Ascend 950PR&950DT products, this API can be called only in AICPU_TS mode on the device side.
<!-- end id6 -->

- The same `ThreadHandle` does not support concurrent access from multiple threads.

## Example

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_CPU_TS;
aclrtStream stream;
aclrtCreateStream(&stream);
ThreadHandle thread;
HcclResult result = HcclThreadAcquireWithStream(comm, engine, stream, 2, &thread);

// Allocate device memory.
uint64_t memSize = 256;
s32 policy = static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH) | static_cast<int>(ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMallocAttrValue moduleIdValue;
moduleIdValue.moduleId = HCCL;
aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};

void* inputMem;
void* outputMem;
aclrtMallocWithCfg(&inputMem, memSize, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
aclrtMallocWithCfg(&outputMem, memSize, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
// Perform the Reduce operation on the device side.
uint64_t count = memSize / SIZE_TABLE[HCOMM_DATA_TYPE_FP32];
HcommLocalReduceOnThread(thread, outputMem, inputMem, count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM);
```

<!-- npu="950" id7 -->
On the Ascend 950PR&950DT products, this function must be compiled for use on the device side:

```c
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;

void *cclBufferAddr = nullptr;
uint64_t cclBufferSize = 0;
HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize);
ThreadHandle thread;
HcclThreadAcquire(comm, engine, 1, 1, &thread);

// Allocate other resources.
// Copy parameters and launch the kernel.

// Orchestrate the algorithm on the device side.
uint64_t len = 256;
void *src = param.userIn;
void *dst = param.cclBuf;
uint64_t sizeOfFP32 = 4;
uint64_t count = len / sizeOfFP32;
HcommLocalReduceOnThread(thread, dst, src, count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM);
```

<!-- end id7 -->
