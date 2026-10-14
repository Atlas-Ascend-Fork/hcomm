# HcommCcuEventAlloc

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:10:11.597Z pushedAt=2026-09-29T08:14:52.315Z -->

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

On the host side, reserves `num` completion event units with consecutive physical numbers from the resource pool of a specified CCU instance (corresponding to [Event](../../data_plane_api/ccu/resource_allocation_operation/Event.md) on the kernel side), and returns a reservation handle. After a successful reservation, these completion event units are carved out of the instance resource pool and no longer participate in resource allocation during the subsequent kernel registration phase. However, they can still be bound and used within the kernel through the reservation handle.

After the reservation is complete, there are two usage modes, which are independent of each other. You can use one of them or both:

- On the host side, call [HcommCcuEventGetAddr](HcommCcuEventGetAddr.md) to retrieve the cached virtual address of each completion event unit, and then pass it as-is to the target module outside the CCU.
- Pass the reservation handle to the kernel through `kernelArgs`, and use the reservation handle within the kernel to construct [Event](../../data_plane_api/ccu/resource_allocation_operation/Event.md) (for example, `Event e(acqHandle, index)`) or [Array\<Event\>](../../data_plane_api/ccu/resource_allocation_operation/Array.md) (for example, `Array<Event> events(acqHandle, count)`), binding them to this reserved completion event unit.

If both paths use the same reservation handle and the same intra-segment index (that is, the queried address and the kernel-side construction pass in the same `index`), the kernel and the aforementioned module point to the same group of physical completion event units.

Difference from the default construction `Event e;` in the kernel: the default construction only obtains a virtual handle during the registration phase, and the physical completion event units are determined only in the [HcommCcuKernelRegister](../ccu_kernel_launch_execution/HcommCcuKernelRegister.md) phase (after the kernel function finishes execution). This API fixes the physical numbers and the die to which they belong at the time of reservation on the host side.

One event corresponds to one completion event unit. `num` indicates the number of reserved completion event units. The `index` of the subsequent [HcommCcuEventGetAddr](HcommCcuEventGetAddr.md) is the intra-segment index, with a value range of `[0, num)`, and is not the bit number within a completion event unit. Different bits within the same completion event unit are distinguished by the `mask` of the kernel-side [EventRecord](../../data_plane_api/ccu/synchronization/EventRecord.md) / [EventWait](../../data_plane_api/ccu/synchronization/EventWait.md). `index` and `mask` are independent of each other.

On the reservation success path, this API establishes and caches an address mapping for each completion event unit. If any one of the mappings fails, the API releases the mappings already established this time, returns the entire segment of resources to the resource pool, and does not write out a valid reservation handle.

> [!NOTE] Note
> A reservation occupies the same number of available completion event units. If the kernel subsequently constructs `Event` or newly allocates `Array<Event>` by default and the remaining number is insufficient, `HcommCcuKernelRegister` returns `CCU_E_UNAVAIL`.

## Function Prototype

```c
CcuResult HcommCcuEventAlloc(CcuInsHandle insHandle, uint8_t dieId, uint32_t num, CcuEventHandle *eventHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| insHandle | Input | CCU instance handle. Resources are allocated from the resource pool of this instance. It must be a valid instance that has been created and not destroyed on the current device. Passing 0 or a handle that does not belong to the current device returns `CCU_E_PTR`. For the type definition, see [CcuInsHandle](../../datatype_definition/CcuInsHandle.md). |
| dieId | Input | Number of the IO die to which the resources belong, with a value range of `[0, CCU_MAX_IODIE_NUM)`. Currently `CCU_MAX_IODIE_NUM` is 2, that is, the value is `0` or `1`. When a contiguous free block of length `num` cannot be carved out from the contiguous completion event unit resource pool of this die, this API returns `CCU_E_UNAVAIL` instead of `CCU_E_PARA`. |
| num | Input | Number of contiguous completion event units to reserve, which must be greater than 0. |
| eventHandle | Output | Reservation handle, which cannot be a null pointer. On reservation success, a non-zero handle is written; on failure, it is set to 0. If `eventHandle` itself is a null pointer, `CCU_E_PTR` is returned directly without writing. For the type definition, see [CcuEventHandle](../../datatype_definition/CcuEventHandle.md). |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | Reservation success. `*eventHandle` is a valid reservation handle. |
| `CCU_E_PTR` | `eventHandle` is a null pointer, or `insHandle` is not a valid CCU instance handle that has been created and not destroyed on the current device. |
| `CCU_E_PARA` | `dieId` is out of the range `[0, CCU_MAX_IODIE_NUM)`, or `num` is 0. |
| `CCU_E_UNAVAIL` | No contiguous free block of length not less than `num` exists in the contiguous completion event unit resource pool of this die. This error code is also returned when the total amount is sufficient but is fragmented into multiple pieces. |
| `CCU_E_RUNTIME` | Address mapping failed. On failure, the mappings established this time are released, and the entire segment of resources is returned to the resource pool. |

## Constraints

- This API can be called only on the host side, not inside a kernel function body.
- The thread that calls this API must be bound to the same NPU device as the one used when the instance was created, through the `aclrtSetDevice(int32_t deviceId)` API.
- The reservation handle has no separate release API, and its lifecycle is bound to `insHandle`: when [HcommCcuInsDestroy](HcommCcuInsDestroy.md) is called to destroy the instance, the framework releases the address mappings, and the reservation handle becomes invalid accordingly.
- To use the reserved resources in the kernel, you must first complete the reservation through this API, and then pass the reservation handle to [HcommCcuKernelRegister](../ccu_kernel_launch_execution/HcommCcuKernelRegister.md) through `kernelArgs`. When only querying the mapped address, it is not necessary to register the kernel.

## Example

Host-side code (C++):

```cpp
CcuInsHandle insHandle = 0;
CcuResult ret = HcommCcuInsCreateDefault(nullptr, 0, &insHandle);
if (ret != CCU_SUCCESS) {
    return ret;
}

// Reserve four physically consecutive completion event units on die 0.
const uint8_t dieId = 0;
const uint32_t eventNum = 4;
CcuEventHandle acqEventHandle = 0;
ret = HcommCcuEventAlloc(insHandle, dieId, eventNum, &acqEventHandle);
if (ret != CCU_SUCCESS) {
    (void)HcommCcuInsDestroy(insHandle);
    return ret;
}

// Retrieve the cached virtual address of each completion event unit (the host only passes it through).
for (uint32_t i = 0; i < eventNum; i++) {
    uint64_t va = 0;
    ret = HcommCcuEventGetAddr(acqEventHandle, i, &va);
    if (ret != CCU_SUCCESS) {
        (void)HcommCcuInsDestroy(insHandle);
        return ret;
    }
    // Write va as-is into the task parameters delivered to the target module, for example:
    //   taskParam.eventVa[i] = va;
}

// Pass the reservation handle to the kernel, where Array<Event>(acqEventHandle, eventNum) binds the same completion event units.
// For the definitions of MyKernelArg and MyKernel, see the kernel-side code below.
MyKernelArg arg = {};
arg.acqEventHandle = acqEventHandle;
arg.eventNum = eventNum;
const void *kernelArgs[] = { &arg };
CcuKernelHandle kernelHandle = 0;

ret = HcommCcuKernelRegisterStart(insHandle);
if (ret == CCU_SUCCESS) {
    // The die to which the reserved completion event units belong is fixed at the time of HcommCcuEventAlloc.
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
    CcuEventHandle acqEventHandle;
    uint32_t eventNum;
};

// Kernel function: binds the eventNum completion event units reserved on the host side using Array<Event>(acqEventHandle, eventNum).
// The bound event is used exactly the same as a normal event; see the array documentation for a more complete example.
// In this example, EventRecord is performed on the CCU side (mask defaults to 0x1, setting bit0); the corresponding EventWait is executed by the target module
// on the same completion event unit, not within this kernel.
CcuResult MyKernel(CcuKernelArg arg)
{
    auto *myArg = static_cast<MyKernelArg *>(arg);
    AscendC::ccu::Array<AscendC::ccu::Event> events(myArg->acqEventHandle, myArg->eventNum);
    for (uint32_t i = 0; i < events.size(); i++) {
        AscendC::ccu::EventRecord(events[i]);
    }
    return CCU_SUCCESS;
}
```
