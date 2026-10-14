# LocalNotifyWait

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:18:12.195Z pushedAt=2026-09-30T09:02:10.474Z -->

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

Blocks and waits for a cross-kernel synchronization signal in the same die within the CCU kernel, with kernels paired by a string tag. It is a consumer-side API. When hardware execution reaches this point, it blocks until the corresponding bit is set.

> [!NOTE] Note
> This API is implemented at the C++ layer as an overload of `AscendC::ccu::EventWait(const char* notifyTag, uint16_t mask)`. It shares the same function name as `EventWait(Event, mask)` and is distinguished by the input parameter type. It applies to sequential synchronization between different kernels within the same die, but not to cross-die or cross-rank scenarios. For cross-die scenarios, use [NotifyWait](NotifyWait.md).

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
// LocalNotifyWait corresponds to this overload.
CcuResult EventWait(const char *notifyTag, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| notifyTag | Input | String tag that must exactly match the `notifyTag` of the producer-side `LocalNotifyRecord` to complete pairing. It cannot be a null pointer. The constraints are the same as those on the producer side: it must be a string literal or `static` storage within the kernel. |
| mask | Input | 16-bit event mask that specifies the bit to wait for. The default value is `1`. It must match the `mask` of the producer-side `LocalNotifyRecord`. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PTR` | `notifyTag` is a null pointer, or no kernel is currently being registered. |

## Constraints

- `LocalNotifyWait` can be called inside a hardware loop body.
- `notifyTag` must remain valid during kernel registration, with the same constraints as `LocalNotifyRecord`.
- Before the call, a corresponding `LocalNotifyRecord` must already exist at a preceding position; otherwise, the call is blocked permanently and causes a hardware-level deadlock.
- The `mask` to wait for must be consistent with the `mask` passed to `LocalNotifyRecord` on the producer side.

## Example

> In the following example, `EventWait("phase1_done", 0x1)` is the `LocalNotifyWait` described in this document. At the C++ layer, it shares the same function name as [EventWait](EventWait.md) (the `Event` object overload), and the two are distinguished by the input parameter type (`const char *` vs `Event`).

```cpp
using namespace AscendC::ccu;

// Scenario: Within the same die, PhaseConsumerKernel waits for PhaseProducerKernel to complete phase 1 before starting phase 2.
// Inside the CCU kernel function body registered to core 1.
CcuResult PhaseConsumerKernel(CcuKernelArg arg) {
    // Wait for bit0 of the "phase1_done" signal sent by PhaseProducerKernel (that is, LocalNotifyWait semantics).
    EventWait("phase1_done", 0x1);

    // After this, the results of PhaseProducerKernel phase 1 can be safely used.
    // ... Perform phase 2 operations ...
    return CCU_SUCCESS;
}
```
