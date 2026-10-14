# HcommCcuKernelQueryResReq

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:17:03.419Z pushedAt=2026-09-29T10:22:25.713Z -->

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

Queries the resource requirements of a specified CCU kernel before creating a CCU instance. The API performs a dry-run of `kernelFunc` on the host side and writes the counted resource quantities of each type into the resource descriptor `resDesc` pre-created by the caller.

The query process executes the kernel function and its existing channel check and die selection logic, but does not register the kernel, allocate CCU instance resources, generate a kernel handle, translate instructions, or dispatch tasks.

## Function Prototype

```c
CcuResult HcommCcuKernelQueryResReq(const void *kernelFunc,
    const void **kernelArgs, uint32_t argNum, HcommCcuResDescHandle resDesc)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| kernelFunc | Input | CCU kernel function pointer, which cannot be a null pointer. When `argNum` is `0`, the function signature must be consistent with a kernel that has no input parameters; when `argNum` is `1`, the function signature must be consistent with a kernel that has a single input parameter. |
| kernelArgs | Input | Pointer array of kernel function input parameters. When `argNum` is `0`, this parameter is ignored and a null pointer can be passed; when `argNum` is `1`, it cannot be a null pointer, and `kernelArgs[0]` cannot be a null pointer either. |
| argNum | Input | Number of kernel function input parameters. Currently, only `0` or `1` is supported. |
| resDesc | Input/Output | CCU resource descriptor handle. It must be pre-created through the `HcommCcuInsResDescCreate` API based on **dieId** on which the kernel runs, and cannot be `0`. After a successful query, the API writes the counted resource quantity into this descriptor and retains the die ID set when the descriptor was created. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | Query successful. The kernel resource requirements have been written into `resDesc`. |
| `CCU_E_PTR` | `kernelFunc` is a null pointer; or when `argNum` is `1`, `kernelArgs` or `kernelArgs[0]` is a null pointer. |
| `CCU_E_PARA` | `argNum` is greater than `1`, `resDesc` is `0`, the die ID in the descriptor is invalid, or the die selected by the kernel based on the channel is inconsistent with the die ID in the descriptor. |
| `CCU_E_NOT_FOUND` | The resource descriptor corresponding to `resDesc` is not found, or the channel used by the kernel does not exist. |
| `CCU_E_INTERNAL` | An internal error occurred during the kernel dry-run or while writing the resource quantity. |
| Other error codes | Other errors generated during kernel execution, channel check, or die selection. |

## Constraints

- Before calling this API, create `resDesc` by calling `HcommCcuInsResDescCreate`. After the call, the caller is responsible for destroying the descriptor by calling `HcommCcuInsResDescDestroy`.
- If the kernel uses a channel, obtain a channel of the `COMM_ENGINE_CCU` type in advance through [HcclChannelAcquire](../comms_domain_resource_mgmt/HcclChannelAcquire.md), and ensure that the channel remains valid during the query.
- When creating `resDesc` by calling `HcommCcuInsResDescCreate` and when calling this API, the current thread must be bound to the same NPU device. That is, the `deviceLogicId` obtained from the two calls must be consistent. The caller can call the AscendCL API `aclrtSetDevice(int32_t deviceId)` in the corresponding thread to specify the device used by that thread. This API obtains `deviceLogicId` from the current thread, does not derive device information from `resDesc`, and does not support using a resource descriptor across devices. The channel used by the kernel must be located on the same die, and the selected die must be consistent with the die ID specified when creating `resDesc`.
- This API does not support concurrent call by multiple threads. The caller must ensure that calls to this API from different threads are executed serially.
- When the query is successful, the statistics obtained this time overwrite the existing resource quantity in `resDesc`, without modifying its die ID.
- When the kernel dry-run fails, no resource quantity is written. If an error occurs while writing the resource quantity, the API exits immediately, and the resource types that have already been written are not rolled back.
- This API can be called only on the host side, not inside a kernel function body.

## Example

```c
typedef struct {
    ChannelHandle channel;
    uint32_t loopCount;
} MyKernelArg;

CcuResult MyKernel(CcuKernelArg arg)
{
    MyKernelArg *kernelArg = (MyKernelArg *)arg;
    // Use kernelArg->channel and the CCU data-plane APIs to describe the kernel operation sequence.
    // ...
    return CCU_SUCCESS;
}

uint32_t dieId = 0;
HcommCcuResDescHandle resDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(dieId, &resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

// Obtain the channel in advance through HcclChannelAcquire, and associate it with dieId.
ChannelHandle channel = 0;
// ... The HcclChannelAcquire call is omitted here ...

MyKernelArg kernelArg = { .channel = channel, .loopCount = 10 };
const void *kernelArgs[] = { &kernelArg };
ret = HcommCcuKernelQueryResReq(
    (const void *)MyKernel, kernelArgs, 1, resDesc);
if (ret != CCU_SUCCESS) {
    HcommCcuInsResDescDestroy(resDesc);
    return ret;
}

uint32_t instructionNum = 0;
ret = HcommCcuInsResDescQueryNum(
    resDesc, HCOMM_CCU_RES_TYPE_INSTRUCTION, &instructionNum);

HcommCcuInsResDescDestroy(resDesc);
return ret;
```
