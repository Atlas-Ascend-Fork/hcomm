# EventRecord

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T12:06:36.086Z pushedAt=2026-10-12T06:12:35.635Z -->

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

Explicitly marks the specified mask bit of a local event as complete within a CCU kernel.

> [!NOTE] Note
> In most scenarios, you do not need to explicitly call this API. The `(event, mask)` parameter at the end of the [data movement API](../data_movement/README.md) (such as `LocalCopy`, `Read`, and `Write`) is automatically set by the hardware when the data movement is complete, which is equivalent to an implicit `EventRecord`. You need to explicitly call this API only when you need to use the end of a control flow branch or a position without data movement operations as a synchronization point.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
CcuResult EventRecord(Event e, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| **e** | Input | Local event object. When the `Event` class is constructed, an event virtual handle is allocated (physical resources are allocated in the `HcommCcuKernelRegister` phase), and no manual allocation is required. |
| mask | Input | 16-bit event mask that specifies the bit to be set. The default value is `1` (that is, bit0). Different bits of the same `Event` object are independent of each other and can carry multiple pairs. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | Operation succeeded. |
| `CCU_E_PTR` | No kernel is currently being registered (the API is called outside the kernel registration phase). |
| `CCU_E_NOT_FOUND` | The passed `e` handle is not registered in the current kernel. |
| `CCU_E_NOT_SUPPORT` | Called inside a hardware loop body, which is not supported. |

> [!NOTE] Note
> This API does not return `CCU_E_PARA`.

## Constraints

- This API cannot be called inside a hardware loop body; otherwise, `CCU_E_NOT_SUPPORT` is returned. The loop body is expanded into N parallel copies by the hardware, so the semantics of `EventRecord` are not unique in a parallel environment.
- Each `EventRecord` must be followed by a corresponding `EventWait`; otherwise, the waiting side will be blocked permanently, causing a hardware-level deadlock.
- The bit specified by `mask` must be consistent with the `mask` of `EventWait`; otherwise, `EventWait` will never receive the signal.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario: Manually insert a synchronization point where there is no data movement.
// Inside the CCU kernel function body.
CcuResult MyKernel(CcuKernelArg arg) {
    Event evt;

    // After performing some operations, manually mark bit0 as complete.
    EventRecord(evt, 0x1);

    // The downstream waits for bit0 to be set.
    EventWait(evt, 0x1);
    return CCU_SUCCESS;
}
```
