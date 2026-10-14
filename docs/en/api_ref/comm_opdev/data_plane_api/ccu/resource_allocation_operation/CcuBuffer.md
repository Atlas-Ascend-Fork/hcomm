# CcuBuffer

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T12:04:58.596Z pushedAt=2026-10-12T06:12:35.632Z -->

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

`ccu::CcuBuffer` is a C++ wrapper class for the on-chip buffer in a CCU kernel. Each CcuBuffer has a fixed size of 4096 bytes.

- Allocation on construction: The default constructor automatically allocates one CcuBuffer virtual handle.
- No release on destruction: The destructor does not release hardware resources; the virtual handle becomes invalid after translation completes, and the physical resource is managed and reclaimed uniformly over the lifecycle of the CCU instance.

CcuBuffer is an on-chip high-speed scratch area within the CCU die, used to transfer data between the on-chip memory and the peer, or as an operand for multi-CcuBuffer reduction (up to 8 CcuBuffers at a time).

> [!NOTE] Note
> The C++ class name retains the `Ccu` prefix (the class name is `CcuBuffer` instead of `Buffer`), and is written as `ccu::CcuBuffer` together with the namespace.

## Class Declaration

```cpp
namespace AscendC {
namespace ccu {
class CcuBuffer final {
public:
    CcuBuffer();                      // Allocation on construction.
    CcuBufferHandle handle{0};       // Virtual handle.
};
} // namespace ccu
} // namespace AscendC
```

## Constructor Description

| Construction Form | Description |
| --- | --- |
| `CcuBuffer buf;` | Allocates a virtual handle for one 4 KB CcuBuffer. It can only be called during the kernel registration phase (inside the kernel function body executed by `HcommCcuKernelRegister`). |

> [!CAUTION] Caution
> The `CcuBuffer` class has copy/move constructors, which only copy the `handle` field and do not allocate a new CcuBuffer. After `CcuBuffer b2 = b1;`, `b1` and `b2` point to the same CcuBuffer. To obtain an independent CcuBuffer, you must explicitly use the default constructor `CcuBuffer b2;` (or use `Array<CcuBuffer>` to allocate multiple physically contiguous ones).

If construction fails, an exception is thrown (carrying the [CcuResult](../../../datatype_definition/CcuResult.md) error code).

## Constraints

- **CcuBuffer** can be constructed only during the kernel registration phase.
- The destructor does not release hardware resources. Do not save or compare the `handle` value outside the kernel; the handle becomes invalid once translation is complete.
- Each **CcuBuffer** always represents a 4096-byte on-chip slice, and its size is not configurable.
- For APIs that operate on **CcuBuffer** (such as `LocalCopy`, `LocalReduce`, `Read`, and `Write`), the `len` passed in must not exceed 4096 bytes. The caller is responsible for ensuring this upper limit; exceeding it leads to undefined hardware behavior at runtime (such as data truncation or out-of-bounds access).
- The C++ construction only allocates a virtual handle and always succeeds without throwing an exception. When CcuBuffer physical resources are insufficient, `HcommCcuKernelRegister` returns `CCU_E_UNAVAIL` during the registration phase instead of throwing an exception at construction time. To perform multi-CcuBuffer reduction, you can use [`Array<CcuBuffer>`](Array.md) to allocate multiple physically contiguous CcuBuffers in a batch.

## Example

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    CcuBuffer buf;    // Allocate one 4KB CcuBuffer.
    LocalAddr src;
    Variable len;
    Event evt;

    // Copy the on-chip memory data to CcuBuffer.
    LocalCopy(buf, src, len, evt);
    EventWait(evt);

    return CCU_SUCCESS;
}
```
