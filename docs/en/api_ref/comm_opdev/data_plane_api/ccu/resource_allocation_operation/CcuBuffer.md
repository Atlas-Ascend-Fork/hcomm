# CcuBuffer

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:07:45.999Z pushedAt=2026-09-30T07:43:31.372Z -->

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

`ccu::CcuBuffer` is a C++ wrapper class for the on-chip Memory Slice (MS) in a CCU kernel. Each MS has a fixed size of 4096 bytes.

- Allocation on construction: The default constructor automatically allocates one MS virtual handle.
- No release on destruction: The destructor does not release hardware resources; the virtual handle becomes invalid after translation completes, and the physical resource is managed and reclaimed uniformly over the lifecycle of the CCU instance.

An MS is an on-chip high-speed scratchpad area within the CCU die, used to relay data between HBM and the remote end, or as an operand for multi-buffer reduction (up to 8 MSs at a time).

> [!NOTE] Note
> The C++ class name retains the `Ccu` prefix (the class name is `CcuBuffer` instead of `Buffer`), and is written as `ccu::CcuBuffer` together with the namespace.

## Class Declaration

```cpp
namespace hcomm {
namespace ccu {
class CcuBuffer final {
public:
    CcuBuffer();                      // Allocation on construction.
    CcuBufferHandle handle{0};       // Virtual handle.
};
} // namespace ccu
} // namespace hcomm
```

## Constructor Description

| Construction Form | Description |
| --- | --- |
| `CcuBuffer buf;` | Allocates a virtual handle for one 4 KB MS. It can only be called during the kernel registration phase (inside the kernel function body executed by `HcommCcuKernelRegister`). |

> [!CAUTION] Caution
> The `CcuBuffer` class has copy/move constructors, which only copy the `handle` field and do not allocate a new MS. After `CcuBuffer b2 = b1;`, `b1` and `b2` point to the same MS. To obtain an independent CcuBuffer, you must explicitly use the default constructor `CcuBuffer b2;` (or use `Array<CcuBuffer>` to allocate multiple physically contiguous slices).

If construction fails, an exception is thrown (carrying the [CcuResult](../../../datatype_definition/CcuResult.md) error code).

## Constraints

- **CcuBuffer** can be constructed only during the kernel registration phase.
- The destructor does not release hardware resources. Do not save or compare the `handle` value outside the kernel; the handle becomes invalid once translation is complete.
- Each **CcuBuffer** always represents a 4096-byte on-chip slice, and its size is not configurable.
- For APIs that operate on **CcuBuffer** (such as `LocalCopy`, `LocalReduce`, `Read`, and `Write`), the `len` passed in must not exceed 4096 bytes. The caller is responsible for ensuring this upper limit; exceeding it leads to undefined hardware behavior at runtime (such as data truncation or out-of-bounds access).
- The C++ construction only allocates a virtual handle and always succeeds without throwing an exception. When `MS` physical resources are insufficient, `HcommCcuKernelRegister` returns `CCU_E_UNAVAIL` during the registration phase instead of throwing an exception at construction time. To perform reduction with multiple buffers, you can use [`Array<CcuBuffer>`](Array.md) to allocate multiple physically contiguous buffers in a batch.

## Example

```cpp
using namespace hcomm::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    CcuBuffer buf;    // Allocate a 4 KB MS.
    LocalAddr src;
    Variable len;
    Event evt;

    // Copy HBM data to the MS buffer.
    LocalCopy(buf, src, len, evt);
    EventWait(evt);

    return CCU_SUCCESS;
}
```
