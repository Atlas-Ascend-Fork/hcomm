# EventWait

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T12:07:05.322Z pushedAt=2026-10-12T06:12:35.636Z -->

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

Blocks and waits within the CCU kernel until the specified mask bit of the local event is set. Hardware execution blocks at this point until the corresponding event bit becomes 1, after which execution continues.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
CcuResult EventWait(Event e, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| e | Input | Local event object, which refers to the same object as the `(event, mask)` parameter of the producer-side `EventRecord` or data movement API. |
| mask | Input | 16-bit event mask that specifies the bit to wait for. The default value is `1`. Supports waiting for multiple bits simultaneously, for example, `mask = 0x3` waits for both bit0 and bit1 to be set. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PTR` | No kernel is currently in the registration phase (the API is called outside the kernel registration phase). |
| `CCU_E_NOT_FOUND` | The passed `e` handle is not registered in the current kernel. |

> [!NOTE] Note
> This API does not return `CCU_E_PARA` and can be called inside a hardware loop body.

## Constraints

- `EventWait` can be called inside a hardware loop body.
- Before the call, a corresponding `EventRecord` or a data movement call with `(event, mask)` as the last parameter must already exist. Otherwise, the call is blocked permanently, causing a hardware-level deadlock.
- The `mask` to wait for must be consistent with the `mask` passed to the producer-side `EventRecord` or movement API.
- The same `Event` object can carry multiple independent production-consumption pairings through different `mask` values, and each pairing does not affect the others.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario: wait for the local on-chip memory to on-chip memory copy to complete before continuing.
// Inside the CCU kernel function body.
CcuResult MyKernel(CcuKernelArg arg) {
    LocalAddr src, dst;
    Variable len;
    Event evt;

    // Initiate the asynchronous copy. The hardware automatically sets evt[0x1] when the copy completes.
    LocalCopy(dst, src, len, evt, 0x1);

    // Block until the copy completes.
    EventWait(evt, 0x1);

    // The data pointed to by dst can then be safely used.
    return CCU_SUCCESS;
}
```
