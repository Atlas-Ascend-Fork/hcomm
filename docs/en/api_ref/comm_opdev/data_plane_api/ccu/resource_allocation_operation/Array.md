# Array

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T12:03:29.004Z pushedAt=2026-10-12T06:12:35.631Z -->

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

`ccu::Array<T>` is a C++ template class for batch holding of physically contiguous resources within a CCU kernel.

- `Array(count)` construction means batch allocation: it allocates `count` physically contiguous virtual handles at once.
- `Array(acqHandle, count)` binds the first `count` resources reserved on the host side and does not allocate new physical resources.
- No release on destruction: The destructor does not release hardware resources; the virtual handle becomes invalid after translation completes, and the physical resource is managed and reclaimed uniformly over the lifecycle of the CCU instance.
- Move-only: copying is prohibited, while moving is allowed.

Physical contiguity is a prerequisite for certain APIs: `Load(addr, vArr, num)`/`Store` batch load/storage and `LocalReduce(buffers*, count, ...)` (2 ≤ count ≤ 8, for details, see [LocalReduce](../data_movement/LocalReduce.md)) all require the parameters to be physically contiguous, which must be allocated through `Array<T>`. Declaring multiple objects separately does not guarantee physical contiguity.

Currently, only the following three specialized types are supported:

| Specialized Type | Resource Description | Supported Construction Form |
| --- | --- | --- |
| `Array<Variable>` | N physically contiguous scalar registers | `Array(count)`, `Array(acqHandle, count)` |
| `Array<Event>` | N physically contiguous completion event units | `Array(count)`, `Array(acqHandle, count)` |
| `Array<CcuBuffer>` | N physically contiguous CcuBuffers | Only `Array(count)` |

Instantiating `Array<T>` with other types will cause a compilation error (only the three specialized types above are supported).

## Class Declaration

```cpp
namespace AscendC {
namespace ccu {
template <typename T>  // T supports only Variable / Event / CcuBuffer.
class Array final {
public:
    explicit Array(uint32_t count);       // Construction performs batch virtual allocation; count can be 0.
    // Bind the first count resources reserved on the host side; available only for the variable/event specializations.
    Array(typename CcuArrayTraits<T>::Handle acqHandle, uint32_t count);
    T& operator[](uint32_t i);            // Subscript access (without bounds checking).
    const T& operator[](uint32_t i) const;
    T* data();                             // Obtain the pointer to the first element (used to pass to batch APIs that require a pointer parameter).
    const T* data() const;
    uint32_t size() const;                 // Returns the number of elements.
    // Copying is prohibited; moving is allowed.
    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    Array(Array&& other) noexcept;
    Array& operator=(Array&& other) noexcept;
};
} // namespace ccu
} // namespace AscendC
```

## Constructor Description

| Construction Form | Description |
| --- | --- |
| `Array<Variable> vars(n);` | Allocates `n` physically contiguous scalar register handles. |
| `Array<Event> evts(n);` | Allocates `n` physically contiguous completion event unit handles. |
| `Array<CcuBuffer> bufs(n);` | Allocates `n` physically contiguous CcuBuffer handles. |
| `Array<Variable> vars(acqHandle, n);` | Binds the first `n` scalar registers reserved on the host side and does not allocate new scalar registers. |
| `Array<Event> evts(acqHandle, n);` | Binds the first `n` completion event units reserved on the host side and does not allocate new completion event units. |

`count` can be 0, and both construction forms are supported (see the note below). An exception is thrown when construction fails.

### Constructor for Binding Host-Side Reserved Resources

The `acqHandle` in `Array(acqHandle, count)` must correspond to `T`: `Array<Variable>` uses the reserved handle returned by [HcommCcuVariableAlloc](../../../control_plane_api/ccu_resource_mgmt/HcommCcuVariableAlloc.md), and `Array<Event>` uses the reserved handle returned by [HcommCcuEventAlloc](../../../control_plane_api/ccu_resource_mgmt/HcommCcuEventAlloc.md). During construction, the resources with sequence numbers `0` through `count - 1` in the reserved segment are bound in order. Element `arr[i]` corresponds to the `i`-th resource in the reserved segment, matching the `index` in [HcommCcuVariableGetAddr](../../../control_plane_api/ccu_resource_mgmt/HcommCcuVariableGetAddr.md) and [HcommCcuEventGetAddr](../../../control_plane_api/ccu_resource_mgmt/HcommCcuEventGetAddr.md) one by one.

This construction is available only for `Array<Variable>` and `Array<Event>`. `Array<CcuBuffer>` has no corresponding implementation; writing `Array<CcuBuffer> bufs(acqHandle, n);` fails at compilation time.

If construction fails, an exception is thrown, which is uniformly caught by [HcommCcuKernelRegister](../../../control_plane_api/ccu_kernel_launch_execution/HcommCcuKernelRegister.md). `CCU_E_INTERNAL` is returned externally and the current registration round is aborted.

> [!CAUTION] Caution
> `count` must not exceed the number of resources under the reserved handle. When `count` is 0, an empty array is constructed directly (`size() == 0`), with no resources bound and no validation of `acqHandle` or whether the current phase is the kernel registration phase.

## Member Function Description

| Function | Description |
| --- | --- |
| `arr[i]` | Returns the reference to the `i`-th element (starting from 0, without bounds checking). |
| `arr.data()` | Returns the pointer to the first element, used to pass to batch APIs that require a `T*` parameter (for example, `LocalReduce(bufs.data(), count, ...)`). |
| `arr.size()` | Returns the `count` value at the time of allocation. |

## Constraints

- `Array(count)` and `Array(acqHandle, count)` with `count > 0` can be constructed only during the kernel registration phase.
- The destructor does not release hardware resources. Do not save the `handle` value of an element outside the kernel, because the handle becomes invalid after translation is complete.
- `Array<T>` specializes only the three types: `Variable/Event/CcuBuffer`. Instantiating it with other types fails at compilation time.
- Multiple separately declared `Variable`/`Event`/`CcuBuffer` objects are not guaranteed to be physically contiguous, and cannot be used with APIs that require physically contiguous resources (such as batch `Load`/`Store` and multi-CcuBuffer `LocalReduce`).

> [!NOTE] Note
> `Load`/`Store` performs a contiguity check on the variable array (returning `CCU_E_PARA` if it is not contiguous), whereas the multi-CcuBuffer overload of `LocalReduce` does not check whether the CcuBuffers are physically contiguous, so the caller must guarantee this. When multiple CcuBuffers are allocated without using `Array`, no error is reported immediately, but the runtime behavior is undefined.

- `Array(count)` only allocates virtual handles and always succeeds. If the resource pool cannot provide *N* contiguous physical resources, `CCU_E_UNAVAIL` is returned during the `HcommCcuKernelRegister` phase, rather than being thrown at construction time.
- `Array(acqHandle, count)` validates the reserved handle and `count` when `count > 0`. If the parameters are invalid, an exception is thrown at construction time, and success is no longer guaranteed.

## Example

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    // Allocate 4 physically contiguous variables in batch for batch loading by Load.
    Array<Variable> vArr(4);
    Load(0x80000000ULL, vArr, 4);    // Load four uint64_t values at a time.

    // Allocate four physically contiguous CcuBuffers in batch for multi-CcuBuffer reduction.
    Array<CcuBuffer> bufs(4);
    Variable len;
    Event evt;
    len = 4096;
    LocalReduce(bufs.data(), 4,
                HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP16,
                HCCL_REDUCE_SUM, len, evt);
    EventWait(evt);

    // Access a single element by subscript.
    vArr[0] = 1024;    // Assign an immediate value to the 0th variable.

    return CCU_SUCCESS;
}
```

Usage of binding resources reserved on the host side:

```cpp
using namespace AscendC::ccu;

// After the host side reserves resources through HcommCcuVariableAlloc / HcommCcuEventAlloc,
// put the reserved handles and count into kernelArgs and pass them to the kernel.
struct MyKernelArg {
    CcuVariableHandle acqHandle;
    uint32_t varNum;
    CcuEventHandle acqEventHandle;
    uint32_t eventNum;
};

CcuResult MyKernel(CcuKernelArg arg) {
    auto* args = static_cast<MyKernelArg*>(arg);

    // Bind varNum scalar registers reserved on the host side, and do not allocate new scalar registers; the bound variable is used in the same way as
    // an ordinary variable. Here, subscript access is demonstrated by assigning values one by one.
    Array<Variable> vars(args->acqHandle, args->varNum);
    for (uint32_t i = 0; i < vars.size(); i++) {
        vars[i] = i;
    }

    // Bind eventNum completion event units reserved on the host side; in this example, EventRecord is performed on the CCU side.
    Array<Event> events(args->acqEventHandle, args->eventNum);
    for (uint32_t i = 0; i < events.size(); i++) {
        EventRecord(events[i]);
    }

    return CCU_SUCCESS;
}
```
