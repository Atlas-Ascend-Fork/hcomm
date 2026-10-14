# LocalCopy

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T11:58:22.537Z pushedAt=2026-10-12T06:12:35.623Z -->

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

Initiates a local memory copy operation (asynchronous) within a CCU kernel. When the hardware completes data movement, bit `mask` of `event` is automatically set to 1. The following three data paths are supported:

| Overload | Source | Target |
| --- | --- | --- |
| Overload 1 | Local on-chip memory (`LocalAddr`) | Local on-chip memory (`LocalAddr`) |
| Overload 2 | Local on-chip memory (`LocalAddr`) | Local CcuBuffer (`CcuBuffer`) |
| Overload 3 | Local CcuBuffer (`CcuBuffer`) | Local on-chip memory (`LocalAddr`) |

> [!NOTE] Note
> This API is asynchronous. After calling it, you must call `EventWait(event, mask)` to wait for the movement to complete; otherwise, the destination memory data is indeterminate. Unlike `EventRecord`, `event[mask]` is automatically set when the hardware completes data movement, so there is no need to explicitly call `EventRecord`.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
// Overload 1: local on-chip memory → local on-chip memory.
CcuResult LocalCopy(LocalAddr dst, LocalAddr src, Variable len, Event event, uint16_t mask = 1);
// Overload 2: local on-chip memory → local CcuBuffer.
CcuResult LocalCopy(CcuBuffer dst, LocalAddr src, Variable len, Event event, uint16_t mask = 1);
// Overload 3: local CcuBuffer → local on-chip memory.
CcuResult LocalCopy(LocalAddr dst, CcuBuffer src, Variable len, Event event, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| dst | Input/Output | Target address (where the hardware copy result is written). For overloads 1/3, it is `LocalAddr` (a composite object of the local on-chip memory address and token); for overload 2, it is `CcuBuffer` (a local CcuBuffer slice object, with a maximum of 4096 bytes per slice). |
| src | Input | Source address. For overloads 1/2, it is `LocalAddr`; for overload 3, it is `CcuBuffer`. |
| len | Input | Number of bytes to copy, of type `Variable` (variable length at runtime). When `CcuBuffer` is involved, it must not exceed 4096 bytes. |
| event | Input | Completion event object. When the hardware completes data movement, `event[mask]` is automatically set, and the downstream calls `EventWait(event, mask)` to wait. |
| mask | Input | 16-bit event mask that specifies the bit to set. The default value is `1` (that is, bit0). Different bits of the same `Event` object are independent of each other and can carry multiple pairs. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation succeeded. |
| `CCU_E_PTR` | No kernel is currently in the registration state (the interface is called outside the kernel registration phase). |
| `CCU_E_NOT_FOUND` | The passed `dst`/`src`/`len`/`event` handle is not registered in the current kernel. |

> [!NOTE] Note
> This API does not return `CCU_E_PARA`. When `len` exceeds 4096 bytes per `CcuBuffer` slice, no error is reported (for details, see "Constraints"), but the runtime behavior is undefined.

## Constraints

- The memory ranges of `dst` and `src` must not overlap. If they overlap, the behavior is undefined.
- When `CcuBuffer` is involved (overloads 2/3), `len` must not exceed the size of a single `CcuBuffer` slice (4096 bytes). The caller must guarantee this upper limit; exceeding it results in undefined runtime hardware behavior.
- This API is asynchronous. You must wait for the movement to complete by calling `EventWait(event, mask)` before accessing the destination memory.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario 1: local on-chip memory → local on-chip memory copy.
CcuResult MyKernel(CcuKernelArg arg) {
    LocalAddr src, dst;
    Variable len;
    Event evt;

    LocalCopy(dst, src, len, evt);    // The default mask is 0x1.
    EventWait(evt);
    return CCU_SUCCESS;
}

// Scenario 2: local on-chip memory → local CcuBuffer copy.
CcuResult MyKernel2(CcuKernelArg arg) {
    CcuBuffer buf;
    LocalAddr src;
    Variable len;
    Event evt;

    LocalCopy(buf, src, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}

// Scenario 3: local CcuBuffer → local on-chip memory copy.
CcuResult MyKernel3(CcuKernelArg arg) {
    LocalAddr dst;
    CcuBuffer buf;
    Variable len;
    Event evt;

    LocalCopy(dst, buf, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```
