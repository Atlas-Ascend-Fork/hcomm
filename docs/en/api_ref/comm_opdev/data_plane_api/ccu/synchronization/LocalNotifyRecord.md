# LocalNotifyRecord

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:17:55.546Z pushedAt=2026-10-08T10:20:16.025Z -->

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

Sends a cross-kernel synchronization signal within the same die in a CCU kernel. It is a producer-side API. Pairing is identified by a string tag: the producer and consumer sides with the same tag are automatically bound to the same synchronization resource.

> [!NOTE] Note
> This API is implemented in C++ as the `AscendC::ccu::EventRecord(const char* notifyTag, uint16_t mask)` overload, which shares the same function name as `EventRecord(Event, mask)` and is distinguished by the input parameter type. It applies to sequential synchronization between different kernels within the same die (cross-kernel on the same device), and does not apply to cross-die or cross-rank scenarios. For cross-die scenarios, use [NotifyRecord](NotifyRecord.md).

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
// LocalNotifyRecord corresponds to this overload.
CcuResult EventRecord(const char *notifyTag, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| notifyTag | Input | String tag. It is paired with the `notifyTag` of the consumer-side `LocalNotifyWait` as long as they are consistent, without the need to allocate a number in advance. It cannot be a null pointer. It must remain valid during kernel registration (it must be a string literal or `static` storage within the kernel), and a temporary array on the stack cannot be used. |
| mask | Input | 16-bit event mask that specifies the bit to be set. The default value is `1`. Different bits of the same tag are independent of each other. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PTR` | `notifyTag` is a null pointer, or no kernel is currently being registered. |
| `CCU_E_NOT_SUPPORT` | Called inside a hardware loop body, which is not supported. |

## Constraints

- The API cannot be called inside a hardware loop body; otherwise, `CCU_E_NOT_SUPPORT` is returned. The loop body is expanded into N copies in parallel by the hardware, and the semantics of the notify record are not unique in a parallel environment.
- `notifyTag` must remain valid during kernel registration. Use a string literal (such as `"phase1_done"`) or a `static char[]` inside the kernel. Do not use a temporary `char[]` on the stack.
- The producer-side `LocalNotifyRecord` and the consumer-side `LocalNotifyWait` must use exactly the same `notifyTag` string content for pairing.
- The same tag can carry multiple pairs, distinguished by different bits of `mask`. The `mask` on the waiting side must be consistent with the one here.
- `LocalNotifyRecord` and the corresponding `LocalNotifyWait` must appear in pairs, and `LocalNotifyWait` must be located after `LocalNotifyRecord`. Any unpaired `LocalNotifyWait` will be blocked permanently, causing a hardware-level deadlock.

## Example

> In the following example, `EventRecord("phase1_done", 0x1)` is the `LocalNotifyRecord` described in this document. At the C++ layer, it shares the same function name as [EventRecord](EventRecord.md) (the `Event` object overload), and the two are distinguished by the input parameter type (`const char *` vs `Event`).

```cpp
using namespace AscendC::ccu;

// Scenario: On the same die, PhaseProducerKernel notifies PhaseConsumerKernel to continue after completing the phase 1 operation.
// Inside the CCU kernel function body registered to core 0.
CcuResult PhaseProducerKernel(CcuKernelArg arg) {
    // ... Execute phase 1 operations ...

    // Notify PhaseConsumerKernel on the same die that phase 1 is complete (i.e., LocalNotifyRecord semantics).
    EventRecord("phase1_done", 0x1);
    return CCU_SUCCESS;
}

// Inside the CCU kernel function body registered to core 1.
CcuResult PhaseConsumerKernel(CcuKernelArg arg) {
    // Wait for the phase 1 completion signal sent by PhaseProducerKernel.
    EventWait("phase1_done", 0x1);

    // The phase 1 results of PhaseProducerKernel can now be safely used.
    return CCU_SUCCESS;
}
```
