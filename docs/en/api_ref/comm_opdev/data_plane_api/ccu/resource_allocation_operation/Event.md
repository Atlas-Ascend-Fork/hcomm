# Event

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:07:51.790Z pushedAt=2026-09-30T07:56:05.494Z -->

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

`ccu::Event` is a C++ wrapper class for completion event units within a CCU kernel.

- Allocation on construction: The default constructor automatically allocates one virtual handle of a completion event unit.
- No release on destruction: The destructor does not release hardware resources; the virtual handle becomes invalid after translation completes, and the physical resource is managed and reclaimed uniformly over the lifecycle of the CCU instance.

An event is used to mark the completion status of an asynchronous operation (such as data movement). The last parameter `(event, mask)` of the data movement API is automatically set by the hardware to `event[mask]` when the movement completes, and the downstream waits for that bit to be set through `EventWait(event, mask)`. An event itself does not hold a `mask` field; mask is passed independently on each `EventRecord`/`EventWait` call.

## Class Declaration

```cpp
namespace AscendC {
namespace ccu {
class Event final {
public:
    Event();                    // Allocation on construction.
    // Bind the index-th completion event unit reserved on the host side without allocating a new completion event unit.
    explicit Event(CcuEventHandle acqHandle, uint32_t index = 0);
    CcuEventHandle handle{0};  // Virtual handle.
};
} // namespace ccu
} // namespace AscendC
```

## Constructor Description

| Construction Form | Description |
| --- | --- |
| `Event e;` | Allocates one virtual handle of a completion event unit. It can be called only in the kernel registration phase (inside the kernel function body executed by `HcommCcuKernelRegister`). |
| `Event e(acqHandle, index);` | Binds the `index`-th completion event unit reserved by the host through [HcommCcuEventAlloc](../../../control_plane_api/ccu_resource_mgmt/HcommCcuEventAlloc.md), without allocating a new completion event unit. The default value of `index` is `0`. |

### Constructor for Binding Host-Side Reserved Resources

In `explicit Event(CcuEventHandle acqHandle, uint32_t index = 0)`, `acqHandle` is the **reserved handle** returned by the host-side [HcommCcuEventAlloc](../../../control_plane_api/ccu_resource_mgmt/HcommCcuEventAlloc.md), not the `handle` of another `Event`. The reserved handle of [Variable](Variable.md) cannot be passed in. A typical usage is to pass the reserved handle into the kernel through `kernelArgs`, and then bind one of the completion event units inside the kernel.

After binding, the event is used with `EventRecord`/`EventWait` like an ordinary event. Calling this constructor multiple times with the same `index` of the same reserved handle yields a new in-kernel `handle` each time, all pointing to the same physical completion event unit. This is not the same path as copy/move: `Event e2 = e1;` only copies the existing `handle` and does not go through this constructor.

When construction fails, an exception is thrown, which is caught uniformly by [HcommCcuKernelRegister](../../../control_plane_api/ccu_kernel_launch_execution/HcommCcuKernelRegister.md). `CCU_E_INTERNAL` is returned externally and the current registration is aborted.

> [!CAUTION] Caution
> The copy/move constructors only copy the `handle` field and do not allocate a new completion event unit. After `Event e2 = e1;`, the two objects are the same in-kernel handle, and `EventRecord`/`EventWait` on either object operates on the same hardware status bit. If an independent event is required, use the default construction, bind a different `index`, or use [`Array<Event>`](Array.md).

## Return Value

After successful construction, the `e.handle` field holds the allocated virtual handle. The handle value is allocated by the framework during the registration phase (the handle of the first event is `0`, and subsequent handles increment). Do not use "`handle != 0`" to determine validity.

## Constraints

- An event can only be constructed during the kernel registration phase.
- The destructor does not release hardware resources. Do not save or compare the `handle` value outside the kernel; the handle becomes invalid once translation is complete.
- An event has no `mask` field. The mask is passed as a parameter in `EventRecord`/`EventWait` and data movement APIs (the C++ default value is `1`). Different bits (masks) of the same event are independent of each other and can carry multiple pairs.
- The default construction only allocates a virtual handle, always succeeds, and does not throw exceptions. When the physical resources of the completion event unit are insufficient, `CCU_E_UNAVAIL` is returned during the `HcommCcuKernelRegister` phase instead of being thrown at construction time. If multiple events are required, use [`Array<Event>`](Array.md) to allocate them in batches.
- The reserved-handle construction `Event e(acqHandle, index);` can likewise only be called during the kernel registration phase. It validates the reserved handle and `index`, and throws an exception when the parameters are invalid, so it no longer guarantees that it always succeeds.

## Example

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    Event evt;    // Allocate one completion event unit.

    // Use with the data movement API: the hardware automatically sets evt[0x1] when the movement completes.
    LocalAddr src, dst;
    Variable len;
    LocalCopy(dst, src, len, evt);    // The mask defaults to 0x1.
    EventWait(evt);                    // Wait for bit0 to be set.

    return CCU_SUCCESS;
}
```
