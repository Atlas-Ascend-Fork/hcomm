# HcommAcquireComm

<!-- md-trans-meta sourceCommit=01a4daa48785c34db54992320c7441d255cce5c2 translatedAt=2026-10-08T12:16:50.894Z pushedAt=2026-10-12T06:12:35.662Z -->

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

Obtains the corresponding communicator based on the passed **commId** and locks the communicator to prevent it from being obtained repeatedly.

## Function Prototype

```c
int32_t HcommAcquireComm(const char* commId)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| commId | Input | Communicator ID. |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Constraints

1. This API can be called only on the device side in AI CPU mode.
2. **HcommAcquireComm** and **HcommReleaseComm** correspond to the lock and unlock actions respectively, and must be called in pairs. The API internally intercepts repeated lock scenarios to prevent the same communicator from being occupied by multiple threads simultaneously.
3. Concurrency note: This API must be called serially, and the communicator is protected by a global lock inside the API. When the target communicator is already occupied by another thread, this API keeps waiting until the thread occupying the communicator calls [HcommReleaseComm](HcommReleaseComm.md) to release it. Release it in a timely manner to avoid blocking other threads.

## Example

This function must be compiled for use on the device side:

```c
// Kernel function executed on the AI CPU.
extern "C" unsigned int HcclAicpuKernel(const char* commId)
{
    // Lock the communicator to prevent it from being used concurrently.
    if (HcommAcquireComm(commId) != HCCL_SUCCESS) {
        return 1;
    }

    // Perform task orchestration.
    // ...

    // Release the communicator.
    if (HcommReleaseComm(commId) != HCCL_SUCCESS) {
        return 1;
    }
    return 0;
}
```
