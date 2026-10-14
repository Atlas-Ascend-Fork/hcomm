# HcommCcuVariableAlloc

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:18:25.853Z pushedAt=2026-10-08T09:23:29.163Z -->

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

On the host side, reserves `num` scalar registers with consecutive physical numbers from the resource pool of a specified CCU instance (corresponding to [Variable](../../data_plane_api/ccu/resource_allocation_operation/Variable.md) on the kernel side), and returns a reservation handle. After the reservation is successful, these scalar registers are carved out of the instance resource pool and no longer participate in resource allocation during the subsequent kernel registration phase. However, they can still be bound and used within the kernel through the reservation handle.

After the reservation is complete, there are two usage methods, which are independent of each other. You can use one of them or both:

- On the host side, call [HcommCcuVariableGetAddr](HcommCcuVariableGetAddr.md) to obtain the cached virtual address of each scalar register, and then pass it as-is to the target module outside the CCU.
- Pass the reservation handle to the kernel through `kernelArgs`, and use the reservation handle within the kernel to construct [Variable](../../data_plane_api/ccu/resource_allocation_operation/Variable.md) (for example, `Variable v(acqHandle, index)`) or [Array\<Variable\>](../../data_plane_api/ccu/resource_allocation_operation/Array.md) (for example, `Array<Variable> vars(acqHandle, count)`), binding them to these reserved scalar registers.

If both paths use the same reservation handle and the same intra-segment sequence number (that is, the queried address and the kernel-side construction pass in the same `index`), the kernel and the aforementioned modules point to the same group of physical scalar registers.

Difference from the default construction of `Variable v;` in the kernel: during the registration phase, the default construction only obtains a virtual handle, and the physical scalar registers are determined only in the [HcommCcuKernelRegister](../ccu_kernel_launch_execution/HcommCcuKernelRegister.md) phase (after the kernel function finishes execution). This API fixes the physical numbers and the die to which they belong at the time of reservation on the host side.

On the successful reservation path, this API establishes and caches an address mapping for each scalar register. If any one of these mappings fails, the API releases the mappings already established this time, returns the entire segment of resources to the resource pool, and does not write out a valid reservation handle.

> [!NOTE] Note
> A reservation occupies the same number of available scalar registers. If a subsequent kernel constructs `Variable` by default or newly allocates `Array<Variable>` and the remaining quantity is insufficient, `HcommCcuKernelRegister` returns `CCU_E_UNAVAIL`.

## Function Prototype

```c
CcuResult HcommCcuVariableAlloc(CcuInsHandle insHandle, uint8_t dieId, uint32_t num, CcuVariableHandle *varHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| insHandle | Input | CCU instance handle. Resources are allocated from the resource pool of this instance. It must be a valid instance that has been created and not destroyed on the current device. Passing 0 or a handle that does not belong to the current device returns `CCU_E_PTR`. For the type definition, see [CcuInsHandle](../../datatype_definition/CcuInsHandle.md). |
| dieId | Input | IO die number to which the resources belong. The value range is `[0, CCU_MAX_IODIE_NUM)`, and the current `CCU_MAX_IODIE_NUM` is 2, meaning the value is `0` or `1`. When a contiguous free block of length `num` cannot be carved out from the contiguous scalar register resource pool of this die, this API returns `CCU_E_UNAVAIL` instead of `CCU_E_PARA`. |
| num | Input | Number of contiguous scalar registers to reserve, which must be greater than 0. |
| varHandle | Output | Reservation handle, which cannot be a null pointer. On successful reservation, a non-zero handle is written; on failure, it is set to 0. If `varHandle` itself is a null pointer, `CCU_E_PTR` is returned directly without writing. For the type definition, see [CcuVariableHandle](../../datatype_definition/CcuVariableHandle.md). |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | Reservation successful, and `*varHandle` is a valid reservation handle. |
| `CCU_E_PTR` | `varHandle` is a null pointer, or `insHandle` is not a valid CCU instance handle that has been created and not destroyed on the current device. |
| `CCU_E_PARA` | `dieId` is out of the range `[0, CCU_MAX_IODIE_NUM)`, or `num` is 0. |
| `CCU_E_UNAVAIL` | No contiguous free block with a length not less than `num` exists in the contiguous scalar register resource pool of this die. This error code is also returned when the total amount is sufficient but is fragmented into multiple pieces. |
| `CCU_E_RUNTIME` | Address mapping failed. On failure, the mappings established this time are released, and the entire segment of resources is returned to the resource pool. |

## Constraints

- This API can be called only on the host side, not inside a kernel function body.
- The thread that calls this API must be bound to the same NPU device as the one used when the instance was created, via the `aclrtSetDevice(int32_t deviceId)` API.
- The reservation handle has no separate release API, and its lifecycle is bound to `insHandle`: when [HcommCcuInsDestroy](HcommCcuInsDestroy.md) is called to destroy the instance, the framework releases the address mappings and the reservation handle becomes invalid accordingly.
- To use the reserved resources in the kernel, you must first complete the reservation with this API, and then pass the reservation handle to [HcommCcuKernelRegister](../ccu_kernel_launch_execution/HcommCcuKernelRegister.md) via `kernelArgs`. When only querying the mapped address, it is not necessary to register the kernel.

## Example

Host-side code (C++):

```cpp
CcuInsHandle insHandle = 0;
CcuResult ret = HcommCcuInsCreateDefault(nullptr, 0, &insHandle);
if (ret != CCU_SUCCESS) {
    return ret;
}

// Reserve 8 physically contiguous scalar registers on die 0.
const uint8_t dieId = 0;
const uint32_t varNum = 8;
CcuVariableHandle acqHandle = 0;
ret = HcommCcuVariableAlloc(insHandle, dieId, varNum, &acqHandle);
if (ret != CCU_SUCCESS) {
    (void)HcommCcuInsDestroy(insHandle);
    return ret;
}

// Retrieve the cached virtual address of each scalar register (the host only passes it through).
for (uint32_t i = 0; i < varNum; i++) {
    uint64_t va = 0;
    ret = HcommCcuVariableGetAddr(acqHandle, i, &va);
    if (ret != CCU_SUCCESS) {
        (void)HcommCcuInsDestroy(insHandle);
        return ret;
    }
    // Write va as-is into the task parameters delivered to the target module, for example:
    //   taskParam.varVa[i] = va;
}

// Pass the reservation handle to the kernel, which binds the same segment of scalar registers through Array<Variable>(acqHandle, varNum).
// For the definitions of MyKernelArg and MyKernel, see the kernel-side code below.
MyKernelArg arg = {};
arg.acqHandle = acqHandle;
arg.varNum = varNum;
const void *kernelArgs[] = { &arg };
CcuKernelHandle kernelHandle = 0;

ret = HcommCcuKernelRegisterStart(insHandle);
if (ret == CCU_SUCCESS) {
    // The die to which the reserved scalar registers belong is fixed when HcommCcuVariableAlloc is called.
    // The dieId of HcommCcuKernelRegister is a reserved parameter and is not used in the current implementation; pass 0.
    ret = HcommCcuKernelRegister(insHandle, 0, "MyKernel",
        (const void *)MyKernel, kernelArgs, 1, &kernelHandle);
    // After start succeeds, end must be called in pairs, even if this register fails.
    CcuResult endRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ret == CCU_SUCCESS) {
        ret = endRet;
    }
}

// When the instance is destroyed, the reservation handle becomes invalid and the mapping is released; there is no separate release API.
(void)HcommCcuInsDestroy(insHandle);
return ret;
```

Kernel-side code (C++):

```cpp
// Kernel input parameter struct: passes the reservation handle and count to the kernel.
struct MyKernelArg {
    CcuVariableHandle acqHandle;
    uint32_t varNum;
};

// Kernel function: binds the varNum scalar registers reserved on the host side using Array<Variable>(acqHandle, varNum),
// The bound variable is used in exactly the same way as a normal variable; see the example in the array documentation for a more complete usage.
CcuResult MyKernel(CcuKernelArg arg)
{
    auto *myArg = static_cast<MyKernelArg *>(arg);
    AscendC::ccu::Array<AscendC::ccu::Variable> vars(myArg->acqHandle, myArg->varNum);
    for (uint32_t i = 0; i < vars.size(); i++) {
        vars[i] = i;
    }
    return CCU_SUCCESS;
}
```
