# GetResByChannel

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T12:04:46.131Z pushedAt=2026-10-12T06:12:35.633Z -->

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

Obtains the handle to a channel-shared variable slot within a CCU kernel.

This slot is a shared variable that has already been reserved on the local end when the channel establishes a link and is mirrored one-to-one with the slot number of the remote end. This API does not consume additional variables; it only wraps the existing slot into an operable variable object.

Typical scenario: the remote rank writes a value into shared variable slot N of the local channel through [WriteVariableWithNotify](../synchronization/WriteVariableWithNotify.md), and the local kernel uses `GetResByChannel<Variable>(ch, N)` to obtain the handle to the same slot and read the value sent by the remote end.

> [!NOTE] Note
> Currently, only the `Variable` specialization (`GetResByChannel<Variable>`) is supported. Template instantiation for other types causes a compile-time error.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
template <typename T>
T GetResByChannel(ChannelHandle channel, uint32_t index);

// Currently, only the variable specialization is supported:
template <>
Variable GetResByChannel<Variable>(ChannelHandle channel, uint32_t varIndex);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| channel | Input | Cross-rank channel handle (`ChannelHandle`). The channel must have completed link establishment, and its pre-allocated variable pool must contain the slot corresponding to `varIndex`. |
| varIndex | Input | Variable index within the channel (starting from 0), corresponding to a slot in the variable array pre-allocated when the channel is established. |

## Return Value

Returns a `Variable` object whose `handle` points to the `varIndex`-th shared variable slot of the channel. This object does not allocate a new variable; its release right belongs to the channel and is not released upon destruction of the returned variable.

If the call fails, an exception is thrown (carrying an error code). Common error codes:

| Scenario | Error Code |
| --- | --- |
| `channel == nullptr`, invalid channel type, or no kernel currently in the registration state | `CCU_E_PTR` |
| `varIndex` out of range (exceeding the size of the pre-allocated variable pool of the channel) | `CCU_E_PARA` |

## Constraints

- This API can only be called during the kernel registration phase, and must be within the kernel function body executed by `HcommCcuKernelRegister`.
- No new variable resources are consumed. Its allocation behavior is completely different from that of `Variable v;` (ordinary construction), and the two are not interchangeable.
- The lifetime of the returned variable is managed by the channel and remains valid until the channel is destroyed.
- Currently, only `T = Variable` is supported. Calling with other types results in a compile-time failure.

## Example

```cpp
using namespace AscendC::ccu;

// End-to-end scenario: The remote end writes a value to shared variable slot 0 of the local channel through WriteVariableWithNotify.
// The local kernel reads the value through GetResByChannel.

CcuResult MyKernel(CcuKernelArg arg) {
    auto* params = static_cast<MyKernelArg*>(arg);
    ChannelHandle ch = params->channelHandle;

    // Obtain the preallocated Variable[0] of the channel (without allocating a new Variable).
    Variable syncVar = GetResByChannel<Variable>(ch, 0);

    // Wait for the remote end to write (see WriteVariableWithNotify for details).
    NotifyWait(ch, /*localNotifyIdx=*/0);

    // At this point, syncVar holds the value written by the remote end and can be read for subsequent computation.
    Variable result;
    result = syncVar;

    return CCU_SUCCESS;
}
```
